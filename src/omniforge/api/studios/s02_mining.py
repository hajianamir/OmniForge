"""Studio 02 (Data Mining & Advanced Diagnostics) router."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Header, HTTPException

from omniforge.core.utils import to_json_object
from omniforge.eda.statistics import distance_correlation
from omniforge.models.clustering.clusterer import ClusteringEngine
from omniforge.models.mining.miner import subgroup_beam_search
from omniforge.ui.state import get_analysis_dataframe, get_selected_targets, get_session_state


def _wb() -> Any:
    from omniforge.api import workbench as wb

    return wb

router = APIRouter(tags=["workbench-s02"])


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
    tgt = target_col or (targets[0] if targets else None)

    subgroups = subgroup_beam_search(df, tgt, beam_width=5, max_depth=2, min_support=10) if tgt and tgt in df.columns else []
    stability = ClusteringEngine.bootstrap_stability_ari(df, n_clusters=max(2, n_clusters), n_bootstraps=15)
    hopkins = ClusteringEngine.hopkins_statistic(df)

    dcor_results: dict[str, Any] = {}
    if tgt and tgt in df.columns and pd.api.types.is_numeric_dtype(df[tgt]):
        num_feats = [c for c in df.select_dtypes(include=[np.number]).columns if c != tgt][:6]
        for col in num_feats:
            dcor_results[col] = distance_correlation(df[col], df[tgt])

    return to_json_object(
        {
            "target": tgt,
            "beam_subgroups": subgroups,
            "cluster_stability_ari": stability,
            "hopkins_clustering_tendency": hopkins,
            "distance_correlation": dcor_results,
        }
    )
