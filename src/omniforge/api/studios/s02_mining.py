"""Studio 02 (Data Mining & Advanced Diagnostics) router."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Header, HTTPException

from omniforge.api.studios.schemas import wb_cache_and_respond
from omniforge.eda.statistics import distance_correlation
from omniforge.models.clustering.clusterer import ClusteringEngine
from omniforge.models.mining.miner import subgroup_beam_search
from omniforge.ui.state import get_analysis_dataframe, get_selected_targets, get_session_state


def _wb() -> Any:
    from omniforge.api import workbench as wb

    return wb

router = APIRouter(tags=["workbench-s02"])


def resolve_mining_target(targets: list[str], requested: str | None) -> str | None:
    if requested is not None:
        if requested not in targets:
            raise HTTPException(status_code=400, detail="Choose a mining target from the selected study targets.")
        return requested
    if len(targets) > 1:
        raise HTTPException(status_code=400, detail="Choose one target for this mining run.")
    return targets[0] if targets else None


@router.post("/s02/advanced_mining")
def s02_advanced_mining(
    target_col: str | None = None,
    n_clusters: int = 3,
    x_session_id: str | None = Header(default="default"),
) -> dict[str, Any]:
    """Run Subgroup Beam Search, Bootstrap ARI Cluster Stability, and Distance Correlation."""
    sid = _wb()._sid(x_session_id)
    state = get_session_state(sid)
    raw = state.get("uap_df")
    if raw is None or len(raw) == 0:
        raise HTTPException(status_code=400, detail="No active dataset in session. Ingest a dataset first.")
    df = get_analysis_dataframe(state, raw)
    targets = get_selected_targets(state, df)
    tgt = resolve_mining_target(targets, target_col)
    predictors = df.drop(columns=targets)
    if predictors.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Advanced mining requires an eligible predictor.")

    subgroups = subgroup_beam_search(df.drop(columns=[c for c in targets if c != tgt]), tgt, beam_width=5, max_depth=2, min_support=10) if tgt is not None else []
    stability = ClusteringEngine.bootstrap_stability_ari(predictors, n_clusters=max(2, n_clusters), n_bootstraps=15)
    hopkins = ClusteringEngine.hopkins_statistic(predictors)

    dcor_results: dict[str, Any] = {}
    if tgt and tgt in df.columns and pd.api.types.is_numeric_dtype(df[tgt]):
        num_feats = list(predictors.select_dtypes(include=[np.number]).columns[:6])
        for col in num_feats:
            dcor_results[col] = distance_correlation(df[col], df[tgt])

    return wb_cache_and_respond(
        state,
        sid,
        "s02_advanced_mining",
        {
            "target": tgt,
            "beam_subgroups": subgroups,
            "cluster_stability_ari": stability,
            "hopkins_clustering_tendency": hopkins,
            "distance_correlation": dcor_results,
        }
    )
