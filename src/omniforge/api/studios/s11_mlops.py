"""Studio 11 (MLOps Registry, Experiment Tracking & Drift Monitoring) router."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Header, HTTPException

from omniforge.api.studios.schemas import DriftReq, PromoteModelReq, RegisterModelReq, RollbackModelReq
from omniforge.core.exceptions import MLOpsError
from omniforge.core.utils import to_json_object
from omniforge.mlops.drift import DriftDetector

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
def s11_register(req: RegisterModelReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = _wb().get_session_state(_wb()._sid(x_session_id))
    model = state.get("uap_active_model")
    if model is None:
        raise HTTPException(status_code=400, detail="No trained model in active workspace. Run Studio 06 first.")
    reg = _wb().get_tenant_registry()
    stage = _wb().parse_model_stage(req.stage, error_status=400)
    fairness_res = state.get("uap_fairness_result")
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
            fairness_report=fairness_res,
            override_reason=req.override_reason,
            stage=stage,
        )
    except MLOpsError as exc:
        if "fairness gate" in str(exc):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise
    payload = to_json_object(
        {"registered": ver.model_dump(), "registered_models": [v.model_dump() for v in reg.list_versions()]}
    )
    _wb().save_studio_result(state, "s11_registered", payload)
    return payload


@router.post("/s11/promote")
def s11_promote(req: PromoteModelReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = _wb().get_session_state(_wb()._sid(x_session_id))
    reg = _wb().get_tenant_registry()
    stage = _wb().parse_model_stage(req.stage, error_status=400)
    fairness_res = state.get("uap_fairness_result")
    try:
        updated = reg.transition_stage(
            req.model_name,
            req.version,
            stage,
            fairness_report=fairness_res,
            override_reason=req.override_reason,
        )
    except MLOpsError as exc:
        if "fairness gate" in str(exc):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise
    return to_json_object(
        {"updated": updated.model_dump(), "registered_models": [v.model_dump() for v in reg.list_versions()]}
    )


@router.post("/s11/rollback")
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
