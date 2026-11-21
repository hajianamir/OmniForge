"""Studio 01 (Data Preparation & Cleaning Audit) router."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Header, HTTPException

from omniforge.data.cleaning import benford_digit_test, cleaning_distortion_audit, robust_mahalanobis_outliers
from omniforge.ui.state import get_analysis_dataframe, get_session_state

router = APIRouter(tags=["workbench-s01"])


@router.post("/s01/advanced_audit")
def s01_advanced_audit(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    """Run Benford's Law, MCD Robust Mahalanobis outliers, and Cleaning Distortion Audit."""
    sid = (x_session_id or "default").strip() or "default"
    state = get_session_state(sid)
    raw = state.get("uap_df")
    if raw is None or len(raw) == 0:
        raise HTTPException(status_code=400, detail="No active dataset in session. Ingest a dataset first.")
    df = get_analysis_dataframe(state, raw)
    raw_df = state.get("uap_raw_df")
    audit_df = raw_df if isinstance(raw_df, pd.DataFrame) and not raw_df.empty else df

    num_cols = list(audit_df.select_dtypes(include=[np.number]).columns)[:6]
    benford_results = {col: benford_digit_test(audit_df[col]) for col in num_cols}
    mcd_res = robust_mahalanobis_outliers(audit_df)
    distortion = (
        cleaning_distortion_audit(raw_df, df)
        if isinstance(raw_df, pd.DataFrame) and not raw_df.empty
        else {"columns_audited": 0, "per_column": {}, "high_distortion_columns": []}
    )
    return {
        "benford_tests": benford_results,
        "robust_mahalanobis": mcd_res,
        "cleaning_distortion": distortion,
    }
