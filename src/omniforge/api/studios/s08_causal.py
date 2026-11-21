# mypy: ignore-errors
"""s08_causal router."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Header, HTTPException

from omniforge.simulation import MonteCarloSimulator

from omniforge.causal import CausalInferenceEngine
from omniforge.forecasting.engine import ForecastingEngine
from omniforge.models.time_series.deep_forecast import DeepForecaster
from omniforge.optimization import OptimizationEngine

router = APIRouter(tags=["workbench-s08"])


def _wb() -> Any:
    from omniforge.api import workbench as wb
    return wb


@router.post("/s08/forecast")
def s08_forecast(req: _wb().ForecastReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    if req.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.target_col}' not found.")
    work = df.sort_values(req.time_col).reset_index(drop=True) if (req.time_col and req.time_col in df.columns) else df.reset_index(drop=True)
    y_series = pd.to_numeric(work[req.target_col], errors="coerce").dropna().reset_index(drop=True)
    if len(y_series) < 20:
        raise HTTPException(status_code=400, detail="Forecasting requires at least 20 numeric observations.")

    classical_res = ForecastingEngine().forecast(
        work.iloc[: min(len(work), 2000)],
        target_col=req.target_col,
        time_col=req.time_col if (req.time_col and req.time_col in work.columns) else None,
        horizon=req.horizon,
    )
    deep_f = DeepForecaster(architecture=req.deep_arch, epochs=req.epochs)
    deep_f.fit(y_series)
    deep_df = deep_f.predict(req.horizon)

    hist_tail = y_series.tail(min(80, len(y_series))).to_list()
    cl_dump = classical_res.model_dump() if hasattr(classical_res, "model_dump") else classical_res
    state["uap_forecast_report"] = {"classical": cl_dump, "deep_arch": req.deep_arch, "deep_forecast": deep_df.to_dict(orient="records")}
    _wb().record_lineage(state, "Forecasting", f"Ran {req.deep_arch.upper()} + Classical Tournament", f"Horizon={req.horizon}")
    return _wb()._cache_and_respond(
        state,
        sid,
        "s08_forecast",
        {
            "history": [{"step": i - len(hist_tail) + 1, "actual": round(float(v), 5)} for i, v in enumerate(hist_tail)],
            "deep_arch": req.deep_arch,
            "deep_forecast": deep_df.to_dict(orient="records"),
            "classical_summary": cl_dump,
        },
    )


@router.post("/s08/causal")
def s08_causal(req: _wb().CausalReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    confs = [c for c in req.confounders if c in df.columns and c not in (req.treatment_col, req.outcome_col)]
    if not confs:
        confs = [c for c in df.select_dtypes(include=[np.number]).columns if c not in (req.treatment_col, req.outcome_col)][:5]
    res = CausalInferenceEngine().estimate_ate(
        df,
        treatment_col=req.treatment_col,
        outcome_col=req.outcome_col,
        confounders=confs,
        method=req.method,
    )
    out = res.model_dump() if hasattr(res, "model_dump") else res
    state["uap_causal_result"] = out
    return _wb()._cache_and_respond(state, sid, "s08_causal", {"causal": out})


@router.post("/s08/optimize")
def s08_optimize(req: _wb().OptimizeReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = _wb().get_session_state(_wb()._sid(x_session_id))
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    cols = [c for c in req.asset_cols if c in df.columns] or list(df.select_dtypes(include=[np.number]).columns)[:5]
    if len(cols) < 2:
        raise HTTPException(status_code=400, detail="Portfolio optimization needs at least 2 numeric columns.")
    sub = df[cols].dropna()
    returns = sub.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 5:
        returns = (sub - sub.mean()) / sub.std().replace(0, 1.0)
    mu = returns.mean().to_numpy(dtype=float)
    cov = returns.cov().to_numpy(dtype=float) + np.eye(len(cols)) * 1e-6
    res = OptimizationEngine().optimize_portfolio(mu, cov, risk_aversion=req.risk_aversion, asset_names=cols)
    out = res.model_dump() if hasattr(res, "model_dump") else res
    state["uap_opt_result"] = out
    payload = _wb().to_json_object({"optimization": out})
    _wb().save_studio_result(state, "s08_optimize", payload)
    return payload


@router.post("/s08/simulate")
def s08_simulate(req: _wb().SimulateReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = _wb().get_session_state(_wb()._sid(x_session_id))
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    if req.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.target_col}' not found.")
    vals = pd.to_numeric(df[req.target_col], errors="coerce").dropna().to_numpy(dtype=float)
    mu, sigma = float(np.mean(vals)), max(float(np.std(vals)), 1e-6)
    sim_res, _samples = MonteCarloSimulator(n_simulations=req.n_simulations).run(
        input_specs={
            "base": {"dist": "normal", "mean": mu, "std": sigma},
            "drift": {"dist": "normal", "mean": 0.0, "std": sigma * 0.25},
        },
        model_fn=lambda s: s["base"] + np.sqrt(req.horizon_steps) * s["drift"],
    )
    out = sim_res.model_dump()
    state["uap_sim_result"] = out
    payload = _wb().to_json_object({"simulation": out})
    _wb().save_studio_result(state, "s08_simulate", payload)
    return payload


# ---------------------------------------------------------------------------
# Studio 09 — Specialized AI (Extended NLP, Vision Foundation Hub, Full Graph Link Prediction, RecSys, Survival)
# ---------------------------------------------------------------------------
