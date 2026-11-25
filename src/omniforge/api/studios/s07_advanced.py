"""Studio 07 (Responsible AI, Explainability & Fairness) router."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Header, HTTPException
from sklearn.base import is_regressor

from omniforge.api.studios.schemas import ExplainReq, FairnessReq
from omniforge.core.exceptions import MLOpsError
from omniforge.core.types import ModelMetadata, TaskType
from omniforge.core.utils import to_json_object
from omniforge.evaluation.fairness import FairnessAuditor
from omniforge.evaluation.interpretability import InterpretabilityEngine
from omniforge.evaluation.model_cards import ModelCardGenerator
from omniforge.mlops.fairness_evidence import FairnessEvidence, holdout_fingerprint

router = APIRouter(tags=["workbench-s07"])


def _wb() -> Any:
    from omniforge.api import workbench as wb

    return wb


@router.post("/s07/explain")
def s07_explain(req: ExplainReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = _wb().get_session_state(_wb()._sid(x_session_id))
    model = state.get("uap_active_model")
    X_te = state.get("uap_active_X_test")
    y_te = state.get("uap_active_y_test")
    if model is None:
        raise HTTPException(status_code=400, detail="Train a model in Studio 06 first to inspect explanations.")
    if not isinstance(X_te, pd.DataFrame) or X_te.empty:
        raise HTTPException(status_code=400, detail="Explanations require held-out data. Enable Split Lock and retrain.")
    eng = InterpretabilityEngine()
    perm = eng.permutation_importance(model, X_te, y_te, n_repeats=5)
    local_idx = max(0, min(req.row_index, len(X_te) - 1))
    local_shap = eng.explain_shap_local(model, X_te, X_te.iloc[[local_idx]], n_samples=48)
    pdp_col = req.feature_for_pdp if (req.feature_for_pdp and req.feature_for_pdp in X_te.columns) else str(X_te.columns[0])
    pdp = eng.partial_dependence(model, X_te, feature=pdp_col, grid_resolution=15)
    state["uap_shap_result"] = {"global": perm, "local": local_shap, "pdp": pdp}
    payload = to_json_object({"global_importance": perm, "local_explanation": local_shap, "pdp": pdp, "pdp_feature": pdp_col})
    _wb().save_studio_result(state, "s07_explain", payload)
    return payload


@router.post("/s07/fairness")
def s07_fairness(req: FairnessReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = _wb().get_session_state(_wb()._sid(x_session_id))
    full_df = _wb()._require_df(state)
    df = _wb().get_analysis_dataframe(state, full_df)
    model = state.get("uap_active_model")
    X_te = state.get("uap_active_X_test")
    y_te = state.get("uap_active_y_test")
    sens_source = df if req.sensitive_col in df.columns else full_df
    if model is None or req.sensitive_col not in sens_source.columns:
        raise HTTPException(status_code=400, detail="Train a model and select a valid sensitive attribute column.")
    if not isinstance(X_te, pd.DataFrame) or X_te.empty:
        raise HTTPException(status_code=400, detail="Fairness auditing requires held-out data. Enable Split Lock and retrain.")
    if (
        not isinstance(y_te, pd.Series) or y_te.isna().any()
        or not X_te.index.equals(y_te.index) or not X_te.index.is_unique
        or not sens_source.index.is_unique or not X_te.index.isin(sens_source.index).all()
    ):
        raise HTTPException(status_code=400, detail="Fairness auditing requires aligned held-out data with observed labels.")
    raw_sensitive = sens_source[req.sensitive_col].reindex(X_te.index)
    lock = state.get("uap_split_lock") or {}
    cluster_col = req.cluster_col or (lock.get("group_col") if lock.get("active") else None)
    if cluster_col is not None and (cluster_col == req.sensitive_col or cluster_col not in full_df.columns):
        raise HTTPException(status_code=400, detail="Select a valid cluster ID column distinct from the protected column.")
    clusters = full_df[cluster_col].reindex(X_te.index) if cluster_col is not None else None
    preds = np.asarray(model.predict(X_te))
    if preds.shape != (len(X_te),):
        raise HTTPException(status_code=400, detail="Fairness auditing requires one prediction per held-out row.")
    y_arr = np.asarray(y_te)
    unique_labels = np.unique(y_arr)
    task = getattr(model, "task_type", None)
    classes = getattr(model, "classes_", None)
    regression = task == TaskType.REGRESSION or (task is None and is_regressor(model))
    fixed_binary_outcome = (
        len(unique_labels) == 2 and not regression and task in (None, TaskType.BINARY_CLASSIFICATION)
        and (classes is None or len(classes) == 2)
    )
    observed = ~pd.isna(preds)
    p_bin = np.full(len(preds), np.nan)
    if regression:
        med = float(np.median(y_arr))
        y_bin = (y_arr >= med).astype(int)
        p_bin[observed] = preds[observed].astype(float) >= med
        outcome_definition = {"method": "at_or_above_holdout_median", "threshold": med}
    else:
        pos_val = unique_labels[-1]
        y_bin = (y_arr == pos_val).astype(int)
        p_bin[observed] = preds[observed] == pos_val
        outcome_definition = {"method": "label_equals", "positive_label": pos_val}
    report = FairnessAuditor().audit(y_bin, p_bin, raw_sensitive, attribute_name=req.sensitive_col, cluster_ids=clusters)
    if not fixed_binary_outcome:
        report = report.model_copy(update={
            "passed": False,
            "assessment_status": "insufficient_evidence",
            "violations": [*report.violations, "Promotion requires a binary classification outcome with both labels observed."],
            "recommendations": ["Use a fairness policy designed for this model's outcome before requesting promotion."],
        })
    recommended_thresholds: dict[str, dict[str, float]] = {}
    if classes is not None and hasattr(model, "predict_proba") and report.assessment_status != "insufficient_evidence":
        try:
            proba = np.asarray(model.predict_proba(X_te), dtype=float)
            positive_index = int(np.flatnonzero(np.asarray(classes) == unique_labels[-1])[0])
            positive_proba = proba[:, positive_index]
            recommended_thresholds["demographic_parity"] = FairnessAuditor.optimize_group_thresholds(
                positive_proba, raw_sensitive, criterion="demographic_parity"
            )
            recommended_thresholds["equalized_odds"] = FairnessAuditor.optimize_group_thresholds(
                positive_proba, raw_sensitive, y_true=y_bin, criterion="equalized_odds"
            )
        except Exception:
            recommended_thresholds = {}
    evidence = None
    if state.get("uap_active_holdout_fingerprint") == holdout_fingerprint(X_te, y_te):
        try:
            evidence = FairnessEvidence.capture(model, X_te, y_te, raw_sensitive, report, clusters=clusters)
        except MLOpsError as exc:
            _wb().logger.warning("fairness_binding_unavailable", error_type=type(exc).__name__)
    state["uap_fairness_evidence"] = evidence.model_dump(mode="json") if evidence is not None else None
    state["uap_fairness_result"] = report.model_dump()
    meta = getattr(model, "metadata_", None)
    if not isinstance(meta, ModelMetadata):
        meta = ModelMetadata(
            name=str(state.get("uap_active_model_name") or "OmniForge Model"),
            model_family="automl",
            framework="omniforge",
            task_type=_wb().TaskType.BINARY_CLASSIFICATION,
        )
    card = ModelCardGenerator.build_model_card(
        meta,
        evaluation_metrics=state.get("uap_active_model_metrics") or {},
        fairness_report=report,
    )
    state["uap_model_card"] = card.to_markdown()
    payload = to_json_object(
        {
            "fairness": report.model_dump(),
            "outcome_definition": outcome_definition,
            "promotion_evidence": {"holdout_fingerprint": evidence.holdout_fingerprint,
                                   "n_holdout_rows": evidence.n_holdout_rows} if evidence is not None else None,
            "recommended_thresholds": recommended_thresholds,
            "model_card_markdown": card.to_markdown(),
        }
    )
    _wb().save_studio_result(state, "s07_fairness", payload)
    return payload
