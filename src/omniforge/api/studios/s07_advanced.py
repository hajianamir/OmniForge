# mypy: ignore-errors
"""s07_advanced router."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Header, HTTPException

from omniforge.core.types import ModelMetadata
from omniforge.evaluation.fairness import FairnessAuditor
from omniforge.evaluation.interpretability import InterpretabilityEngine
from omniforge.evaluation.model_cards import ModelCardGenerator

router = APIRouter(tags=["workbench-s07"])


def _wb() -> Any:
    from omniforge.api import workbench as wb
    return wb


@router.post("/s07/explain")
def s07_explain(req: _wb().ExplainReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = _wb().get_session_state(_wb()._sid(x_session_id))
    model = state.get("uap_active_model")
    X_te = state.get("uap_active_X_test")
    y_te = state.get("uap_active_y_test")
    if model is None or not isinstance(X_te, pd.DataFrame) or X_te.empty:
        raise HTTPException(status_code=400, detail="Train a model in Studio 06 first to inspect explanations.")
    eng = InterpretabilityEngine()
    perm = eng.permutation_importance(model, X_te, y_te, n_repeats=5)
    local_idx = max(0, min(req.row_index, len(X_te) - 1))
    local_shap = eng.explain_shap_local(model, X_te, X_te.iloc[[local_idx]], n_samples=48)
    pdp_col = req.feature_for_pdp if (req.feature_for_pdp and req.feature_for_pdp in X_te.columns) else str(X_te.columns[0])
    pdp = eng.partial_dependence(model, X_te, feature=pdp_col, grid_resolution=15)
    state["uap_shap_result"] = {"global": perm, "local": local_shap, "pdp": pdp}
    payload = _wb().to_json_object({"global_importance": perm, "local_explanation": local_shap, "pdp": pdp, "pdp_feature": pdp_col})
    _wb().save_studio_result(state, "s07_explain", payload)
    return payload


@router.post("/s07/fairness")
def s07_fairness(req: _wb().FairnessReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = _wb().get_session_state(_wb()._sid(x_session_id))
    full_df = _wb()._require_df(state)
    df = _wb().get_analysis_dataframe(state, full_df)
    model = state.get("uap_active_model")
    X_te = state.get("uap_active_X_test")
    y_te = state.get("uap_active_y_test")
    sens_source = df if req.sensitive_col in df.columns else full_df
    if model is None or X_te is None or req.sensitive_col not in sens_source.columns:
        raise HTTPException(status_code=400, detail="Train a model and select a valid sensitive attribute column.")
    _sens = sens_source[req.sensitive_col].reindex(X_te.index)
    if isinstance(_sens.dtype, pd.CategoricalDtype) and "Missing" not in _sens.cat.categories:
        _sens = _sens.cat.add_categories(["Missing"])
    sens = _sens.fillna("Missing").astype(str).to_numpy()
    preds = np.asarray(model.predict(X_te))
    y_arr = np.asarray(y_te)
    if np.issubdtype(y_arr.dtype, np.number) and len(np.unique(y_arr)) > 2:
        med = float(np.median(y_arr))
        y_bin = (y_arr >= med).astype(int)
        p_bin = (preds.astype(float) >= med).astype(int)
    else:
        pos_val = np.unique(y_arr)[-1]
        y_bin = (y_arr == pos_val).astype(int)
        p_bin = (preds == pos_val).astype(int)
    report = FairnessAuditor().audit(y_bin, p_bin, sens, attribute_name=req.sensitive_col)
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
    payload = _wb().to_json_object({"fairness": report.model_dump(), "model_card_markdown": card.to_markdown()})
    _wb().save_studio_result(state, "s07_fairness", payload)
    return payload


# ---------------------------------------------------------------------------
# Studio 08 — Forecasting (N-BEATS / PatchTST / TFT), Causal, Optimization & Simulation
# ---------------------------------------------------------------------------
