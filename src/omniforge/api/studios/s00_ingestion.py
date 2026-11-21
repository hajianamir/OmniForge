"""Studio 00 (Data Ingestion & Relationships) router."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Header, HTTPException

from omniforge.eda.statistics import distance_correlation
from omniforge.ui.state import get_session_state

router = APIRouter(tags=["workbench-s00"])


def _safe_float(val: Any) -> float:
    try:
        f = float(val)
        return 0.0 if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return 0.0


def _to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    codes, _ = pd.factorize(s.astype(str), sort=True)
    return pd.Series(codes, index=s.index, dtype=float)


def _classify_rel(pear: float, spear: float, dcor: float, both_numeric: bool) -> str:
    abs_p = abs(pear)
    if not both_numeric:
        if dcor >= 0.45 or abs(spear) >= 0.5:
            return "Strong Categorical/Mixed Association"
        if dcor >= 0.25 or abs(spear) >= 0.25:
            return "Moderate Categorical/Mixed Association"
        return "Weak / Independent"
    if abs_p >= 0.7:
        return "Strong Linear"
    if dcor >= 0.35 and abs_p < 0.3:
        return "Nonlinear Dependence"
    if abs(spear) >= 0.5 and abs_p < 0.45:
        return "Monotonic Nonlinear"
    if abs_p >= 0.35 or dcor >= 0.3:
        return "Moderate Association"
    return "Weak / Independent"


@router.get("/s00/data_relationships")
def s00_data_relationships(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    """Compute pairwise relationships across all active (non-excluded) variables and vs. target."""
    sid = (x_session_id or "default").strip() or "default"
    state = get_session_state(sid)
    df = state.get("uap_df")
    if df is None or len(df) == 0:
        raise HTTPException(status_code=400, detail="No active dataset in session. Ingest a dataset first.")

    excluded = set(state.get("uap_excluded_columns") or [])
    raw_targets = list(state.get("uap_targets") or [])
    single_target = state.get("uap_target")
    if not raw_targets and single_target:
        raw_targets = [str(single_target)]
    targets = [str(t) for t in raw_targets if str(t) in df.columns]
    target = targets[0] if targets else None
    active_cols = [c for c in df.columns if c not in excluded]
    for t in targets:
        if t not in active_cols:
            active_cols.append(t)

    work_cols = active_cols[:30]
    numeric_set = set(df[work_cols].select_dtypes(include=[np.number]).columns)
    encoded_df = pd.DataFrame(
        {c: _to_numeric_series(df[c]) for c in work_cols},
        index=df.index,
    ).replace([np.inf, -np.inf], np.nan)

    pairs: list[dict[str, Any]] = []
    for i in range(len(work_cols)):
        for j in range(i + 1, len(work_cols)):
            c1, c2 = work_cols[i], work_cols[j]
            pair_df = encoded_df[[c1, c2]].dropna()
            if len(pair_df) < 5:
                continue
            x = pair_df[c1].to_numpy(dtype=float)
            y = pair_df[c2].to_numpy(dtype=float)
            pear = _safe_float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0
            spear = _safe_float(pair_df[c1].corr(pair_df[c2], method="spearman"))
            dcor = _safe_float(distance_correlation(x[:400], y[:400]).get("dcor", 0.0))
            both_num = (c1 in numeric_set) and (c2 in numeric_set)
            rel_type = _classify_rel(pear, spear, dcor, both_num)
            pairs.append(
                {
                    "feature_a": c1,
                    "feature_b": c2,
                    "pearson": round(pear, 4),
                    "spearman": round(spear, 4),
                    "dcor": round(dcor, 4),
                    "relationship_type": rel_type,
                }
            )
    pairs.sort(key=lambda r: max(abs(r["pearson"]), r["dcor"]), reverse=True)

    target_relationships: list[dict[str, Any]] = []
    for t_col in targets:
        if t_col not in encoded_df.columns:
            continue
        for col in work_cols:
            if col == t_col:
                continue
            pair_df = encoded_df[[col, t_col]].dropna()
            if len(pair_df) < 5:
                continue
            x = pair_df[col].to_numpy(dtype=float)
            y = pair_df[t_col].to_numpy(dtype=float)
            pear = _safe_float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0
            spear = _safe_float(pair_df[col].corr(pair_df[t_col], method="spearman"))
            dcor = _safe_float(distance_correlation(x[:400], y[:400]).get("dcor", 0.0))
            both_num = (col in numeric_set) and (t_col in numeric_set)
            target_relationships.append(
                {
                    "feature": col,
                    "target": t_col,
                    "pearson": round(pear, 4),
                    "spearman": round(spear, 4),
                    "dcor": round(dcor, 4),
                    "relationship_type": _classify_rel(pear, spear, dcor, both_num),
                }
            )
    target_relationships.sort(key=lambda r: max(abs(r["pearson"]), r["dcor"]), reverse=True)

    return {
        "n_rows": int(len(df)),
        "n_cols": int(len(active_cols)),
        "target": target,
        "targets": targets,
        "excluded_columns": sorted(excluded),
        "numeric_cols": [c for c in work_cols if c in numeric_set],
        "active_cols": work_cols,
        "target_relationships": target_relationships,
        "pairs": pairs[:200],
    }
