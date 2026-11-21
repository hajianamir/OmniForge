# mypy: ignore-errors
"""s11_mlops router."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Header, HTTPException

from omniforge.core.types import ModelStage
from omniforge.mlops.drift import DriftDetector

router = APIRouter(tags=["workbench-s11"])


def _wb() -> Any:
    from omniforge.api import workbench as wb
    return wb


@router.get("/s11/overview")
def s11_overview() -> dict[str, Any]:
    models = [v.model_dump() for v in _wb()._SHARED_REGISTRY.list_versions()]
    runs = _wb()._SHARED_TRACKER.list_runs()
    return _wb().to_json_object({"registered_models": models, "experiment_runs": runs})


@router.post("/s11/register")
def s11_register(req: _wb().RegisterModelReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = _wb().get_session_state(_wb()._sid(x_session_id))
    model = state.get("uap_active_model")
    if model is None:
        raise HTTPException(status_code=400, detail="No trained model in active workspace. Run Studio 06 first.")
    ver = _wb()._SHARED_REGISTRY.register_model(
        model=model,
        model_name=req.model_name,
        metrics={k: float(v) for k, v in (state.get("uap_active_model_metrics") or {}).items() if isinstance(v, (int, float))},
        stage=ModelStage(req.stage) if req.stage in [s.value for s in ModelStage] else ModelStage.STAGING,
    )
    payload = _wb().to_json_object({"registered": ver.model_dump(), "registered_models": [v.model_dump() for v in _wb()._SHARED_REGISTRY.list_versions()]})
    _wb().save_studio_result(state, "s11_registered", payload)
    return payload


@router.post("/s11/promote")
def s11_promote(req: _wb().PromoteModelReq) -> dict[str, Any]:
    updated = _wb()._SHARED_REGISTRY.transition_stage(req.model_name, req.version, ModelStage(req.stage))
    return _wb().to_json_object({"updated": updated.model_dump(), "registered_models": [v.model_dump() for v in _wb()._SHARED_REGISTRY.list_versions()]})


@router.post("/s11/drift")
def s11_drift(req: _wb().DriftReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = _wb().get_session_state(_wb()._sid(x_session_id))
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    num_df = df.select_dtypes(include=[np.number]).dropna()
    if len(num_df) < 20 or num_df.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Drift detection requires at least 20 numeric rows.")
    cut = int(len(num_df) * req.split_ratio)
    report = DriftDetector().detect(num_df.iloc[:cut], num_df.iloc[cut:])
    payload = _wb().to_json_object({"drift_report": report.model_dump()})
    _wb().save_studio_result(state, "s11_drift", payload)
    return payload


# ---------------------------------------------------------------------------
# Studio 12 — Customizable Executive Reporting & Export Center (ALL 9 Formats!)
# ---------------------------------------------------------------------------
