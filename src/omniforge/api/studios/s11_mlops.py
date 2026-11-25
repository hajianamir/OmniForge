"""Studio 11 (MLOps Registry, Experiment Tracking & Drift Monitoring) router."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException

from omniforge.api.auth import authorize_principal, require_permission
from omniforge.api.studios.schemas import DriftReq, PromoteModelReq, RegisterModelReq, RollbackModelReq, RootCauseReq
from omniforge.core.exceptions import MLOpsError
from omniforge.core.types import ModelStage, Principal
from omniforge.core.utils import to_json_object
from omniforge.mlops.drift import DriftDetector
from omniforge.mlops.fairness_evidence import FairnessEvidence, holdout_fingerprint
from omniforge.mlops.root_cause import DriftRootCauseAnalyzer

router = APIRouter(tags=["workbench-s11"])


def _wb() -> Any:
    from omniforge.api import workbench as wb

    return wb


@router.get("/s11/overview")
def s11_overview() -> dict[str, Any]:
    reg = _wb().get_tenant_registry()
    trk = _wb().get_tenant_tracker()
    models = [v.model_dump() for v in reg.list_versions()]
    runs = trk.list_runs()
    return to_json_object({"registered_models": models, "experiment_runs": runs})


@router.post("/s11/register")
def s11_register(
    req: RegisterModelReq,
    x_session_id: str | None = Header(default="default"),
    principal: Principal = Depends(require_permission("mlops:write")),
) -> dict[str, Any]:
    stage = _wb().parse_model_stage(req.stage, error_status=400)
    if stage == ModelStage.PRODUCTION:
        authorize_principal(principal, "model:promote")
    state = _wb().get_session_state(_wb()._sid(x_session_id))
    model = state.get("uap_active_model")
    if model is None:
        raise HTTPException(status_code=400, detail="No trained model in active workspace. Run Studio 06 first.")
    reg = _wb().get_tenant_registry()
    evidence = None
    if state.get("uap_fairness_evidence") is not None:
        evidence = FairnessEvidence.model_validate(state["uap_fairness_evidence"])
        X_test, y_test = state.get("uap_active_X_test"), state.get("uap_active_y_test")
        full_df = state.get("uap_df")
        group = evidence.report.protected_attribute
        cluster = evidence.report.cluster_attribute
        if not (
            isinstance(X_test, pd.DataFrame) and isinstance(y_test, pd.Series)
            and isinstance(full_df, pd.DataFrame) and group in full_df.columns and full_df.index.is_unique
            and X_test.index.isin(full_df.index).all()
            and (cluster is None or cluster in full_df.columns)
            and holdout_fingerprint(
                X_test, y_test, full_df[group].reindex(X_test.index),
                clusters=full_df[cluster].reindex(X_test.index) if cluster is not None else None,
            ) == evidence.holdout_fingerprint
        ):
            raise HTTPException(status_code=409, detail="The held-out data changed. Run the fairness audit again.")
    try:
        ver = reg.register_model(
            model=model,
            model_name=req.model_name,
            metrics={
                k: float(v)
                for k, v in (state.get("uap_active_model_metrics") or {}).items()
                if isinstance(v, (int, float))
            },
            owner=req.owner,
            intended_use=req.intended_use,
            approved_domains=req.approved_domains,
            risk_tier=req.risk_tier,
            fairness_evidence=evidence,
            override_reason=req.override_reason,
            stage=stage,
        )
    except MLOpsError as exc:
        if "fairness" in str(exc).lower():
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise
    payload = to_json_object(
        {"registered": ver.model_dump(), "registered_models": [v.model_dump() for v in reg.list_versions()]}
    )
    _wb().save_studio_result(state, "s11_registered", payload)
    return payload


@router.post("/s11/promote", dependencies=[Depends(require_permission("model:promote"))])
def s11_promote(req: PromoteModelReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    reg = _wb().get_tenant_registry()
    stage = _wb().parse_model_stage(req.stage, error_status=400)
    try:
        updated = reg.transition_stage(
            req.model_name,
            req.version,
            stage,
            override_reason=req.override_reason,
        )
    except MLOpsError as exc:
        if "fairness" in str(exc).lower():
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise
    return to_json_object(
        {"updated": updated.model_dump(), "registered_models": [v.model_dump() for v in reg.list_versions()]}
    )


@router.post("/s11/rollback", dependencies=[Depends(require_permission("model:promote"))])
def s11_rollback(req: RollbackModelReq) -> dict[str, Any]:
    reg = _wb().get_tenant_registry()
    try:
        restored = reg.rollback_production(req.model_name, reason=req.reason)
    except MLOpsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return to_json_object(
        {"restored": restored.model_dump(), "registered_models": [v.model_dump() for v in reg.list_versions()]}
    )


@router.post("/s11/drift")
def s11_drift(req: DriftReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = _wb().get_session_state(_wb()._sid(x_session_id))
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    num_df = df.select_dtypes(include=[np.number]).dropna()
    if len(num_df) < 20 or num_df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Drift detection requires at least 20 numeric rows.")
    cut = max(5, min(len(num_df) - 5, int(len(num_df) * req.split_ratio)))
    report = DriftDetector().detect(num_df.iloc[:cut], num_df.iloc[cut:])
    payload = to_json_object({"drift_report": report.model_dump()})
    _wb().save_studio_result(state, "s11_drift", payload)
    return payload


@router.post("/s11/root-cause")
def s11_root_cause(
    req: RootCauseReq, x_session_id: str | None = Header(default="default")
) -> dict[str, Any]:
    """C5 causal drift root-cause attribution: drift -> Delta-attribution -> DAG ancestors."""
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    num_df = df.select_dtypes(include=[np.number]).dropna()
    if len(num_df) < 20:
        raise HTTPException(status_code=400, detail="Root-cause analysis requires at least 20 numeric rows.")
    cut = max(5, min(len(num_df) - 5, int(len(num_df) * 0.5)))
    ref_df = num_df.iloc[:cut]
    cur_df = num_df.iloc[cut:]
    model = state.get("uap_active_model")
    feat_cols = list(state.get("uap_active_feature_cols") or num_df.columns)
    analyzer = DriftRootCauseAnalyzer(
        psi_warning=req.psi_warning,
        psi_critical=req.psi_critical,
        max_cond_set_size=req.max_cond_set_size,
    )
    report = analyzer.analyze(
        ref_df, cur_df, model=model, feature_cols=feat_cols,
        target_col=req.target_col, time_col=req.time_col,
    )
    payload = to_json_object({"root_cause_report": report.model_dump()})
    _wb().save_studio_result(state, "s11_root_cause", payload)
    return payload
