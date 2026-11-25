"""Studio 08 (Forecasting, Causal Inference, Optimization & Simulation) router."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, cast

import numpy as np
import pandas as pd
from fastapi import APIRouter, Header, HTTPException

from omniforge.api.studios.schemas import CausalReq, ForecastReq, OptimizeReq, SimulateReq, wb_cache_and_respond
from omniforge.causal import CausalInferenceEngine
from omniforge.core.utils import to_json_object
from omniforge.forecasting.engine import ForecastingEngine, _causal_impute_split
from omniforge.models.time_series.deep_forecast import DeepForecaster
from omniforge.optimization import OptimizationEngine
from omniforge.simulation import MonteCarloSimulator

router = APIRouter(tags=["workbench-s08"])


def _wb() -> Any:
    from omniforge.api import workbench as wb

    return wb


@router.post("/s08/forecast")
def s08_forecast(req: ForecastReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    if req.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.target_col}' not found.")
    work = df.sort_values(req.time_col, kind="mergesort").reset_index(drop=True) if (req.time_col and req.time_col in df.columns) else df.reset_index(drop=True)
    work_capped = work.tail(2000).reset_index(drop=True)
    raw_capped = pd.to_numeric(work_capped[req.target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if int(raw_capped.notna().sum()) < 20:
        raise HTTPException(status_code=400, detail="Forecasting requires at least 20 finite numeric observations in the analysis window.")

    classical_res = ForecastingEngine().forecast(
        work_capped,
        target_col=req.target_col,
        time_col=req.time_col if (req.time_col and req.time_col in work_capped.columns) else None,
        horizon=req.horizon,
        interval_calibration_origins=req.interval_calibration_origins,
        interval_evaluation_origins=req.interval_evaluation_origins,
    )
    causal_history, _ = _causal_impute_split(raw_capped)
    deep_f = DeepForecaster(architecture=cast(Any, req.deep_arch), epochs=req.epochs)
    deep_f.fit(causal_history)
    deep_df = deep_f.predict(req.horizon)

    hist_tail = raw_capped.tail(80).to_numpy(dtype=float)
    cl_dump = to_json_object(asdict(classical_res))
    fc_record = {"classical": cl_dump, "deep_arch": req.deep_arch, "deep_forecast": deep_df.to_dict(orient="records")}
    state["uap_forecast_report"] = fc_record
    state["uap_forecast_result"] = fc_record
    _wb().record_lineage(state, "Forecasting", f"Ran {req.deep_arch.upper()} + Classical Tournament", f"Horizon={req.horizon}")
    return wb_cache_and_respond(
        state,
        sid,
        "s08_forecast",
        {
            "history": [{"step": i - len(hist_tail) + 1, "actual": round(float(v), 5) if np.isfinite(v) else None} for i, v in enumerate(hist_tail)],
            "deep_arch": req.deep_arch,
            "deep_forecast": deep_df.to_dict(orient="records"),
            "classical_summary": cl_dump,
        },
    )


@router.post("/s08/causal")
def s08_causal(req: CausalReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    if req.method == "clustered_did":
        if not req.post_col or not req.cluster_col:
            raise HTTPException(status_code=400, detail="Clustered difference-in-differences requires a period and cluster column.")
        out = CausalInferenceEngine().difference_in_differences(
            df, group_col=req.treatment_col, post_col=req.post_col, outcome_col=req.outcome_col,
            cluster_col=req.cluster_col,
        )
        state["uap_causal_result"] = out
        return wb_cache_and_respond(state, sid, "s08_causal", {"causal": out})
    confs = [c for c in req.confounders if c in df.columns and c not in (req.treatment_col, req.outcome_col)]
    if not confs:
        confs = [c for c in df.select_dtypes(include=[np.number]).columns if c not in (req.treatment_col, req.outcome_col)][:5]
    res = CausalInferenceEngine().estimate_ate(
        df,
        treatment_col=req.treatment_col,
        outcome_col=req.outcome_col,
        confounders=confs,
        method=req.method,
        instrument_col=req.instrument_col if (req.instrument_col and req.instrument_col in df.columns) else None,
    )
    out = res.model_dump()
    state["uap_causal_result"] = out
    return wb_cache_and_respond(state, sid, "s08_causal", {"causal": out})


@router.post("/s08/optimize")
def s08_optimize(req: OptimizeReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    cols = [c for c in req.asset_cols if c in df.columns] or list(df.select_dtypes(include=[np.number]).columns)[:5]
    if len(cols) < 2:
        raise HTTPException(status_code=400, detail="Portfolio optimization needs at least 2 numeric columns.")
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 3:
        raise HTTPException(status_code=400, detail="Portfolio optimization needs at least 3 complete numeric rows.")
    returns = sub.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 5:
        returns = (sub - sub.mean()) / sub.std().replace(0, 1.0)
    mu = returns.mean().to_numpy(dtype=float)
    cov = returns.cov().to_numpy(dtype=float) + np.eye(len(cols)) * 1e-6
    res = OptimizationEngine().optimize_portfolio(mu, cov, risk_aversion=req.risk_aversion, asset_names=cols)
    out = res.model_dump() if hasattr(res, "model_dump") else res
    state["uap_opt_result"] = out
    return wb_cache_and_respond(state, sid, "s08_optimize", {"optimization": out})


@router.post("/s08/simulate")
def s08_simulate(req: SimulateReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    if req.target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{req.target_col}' not found.")
    vals = pd.to_numeric(df[req.target_col], errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        raise HTTPException(status_code=400, detail=f"Column '{req.target_col}' contains no numeric values.")
    mu, sigma = float(np.mean(vals)), max(float(np.std(vals)), 1e-6)
    sim_res, samples = MonteCarloSimulator(n_simulations=req.n_simulations).run(
        input_specs={
            "base": {"dist": "normal", "mean": mu, "std": sigma},
            "drift": {"dist": "normal", "mean": 0.0, "std": sigma * 0.25},
        },
        model_fn=lambda s: s["base"] + np.sqrt(req.horizon_steps) * s["drift"],
    )
    out = sim_res.model_dump()
    finite_samples = np.asarray(samples, dtype=float)
    finite_samples = finite_samples[np.isfinite(finite_samples)]
    if len(finite_samples) >= 5:
        counts, edges = np.histogram(finite_samples, bins=18)
        out["histogram"] = [{"label": f"{edges[i]:.3g}", "count": int(counts[i])} for i in range(len(counts))]
    state["uap_sim_result"] = out
    return wb_cache_and_respond(state, sid, "s08_simulate", {"simulation": out})
