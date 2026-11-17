"""Comprehensive statistical dataset profiling engine for OmniForge."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from omniforge.core.logging import get_logger
from omniforge.core.utils import infer_semantic_column_type

logger = get_logger(__name__)


class ColumnProfile(BaseModel):
    """Detailed statistical profile of an individual dataset column."""

    name: str
    dtype: str
    semantic_type: str
    count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    q05: float | None = None
    q25: float | None = None
    median: float | None = None
    q75: float | None = None
    q95: float | None = None
    max: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    zeros_count: int | None = None
    negatives_count: int | None = None
    top_values: dict[str, int] = Field(default_factory=dict)
    alerts: list[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    """Complete dataset-level profile combining column statistics, correlations, and alerts."""

    n_rows: int
    n_columns: int
    memory_mb: float
    missing_cells: int
    missing_pct: float
    duplicate_rows: int
    duplicate_pct: float
    numeric_columns: list[str] = Field(default_factory=list)
    categorical_columns: list[str] = Field(default_factory=list)
    datetime_columns: list[str] = Field(default_factory=list)
    boolean_columns: list[str] = Field(default_factory=list)
    columns: list[ColumnProfile] = Field(default_factory=list)
    correlations_pearson: dict[str, dict[str, float]] = Field(default_factory=dict)
    high_correlations: list[dict[str, Any]] = Field(default_factory=list)
    target_correlations: dict[str, float] = Field(default_factory=dict)
    alerts: list[str] = Field(default_factory=list)
    profiled_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DataProfiler:
    """Computes univariate, bivariate, and structural profiles for tabular and time-series data."""

    def __init__(self, high_corr_threshold: float = 0.85, top_k_categories: int = 10) -> None:
        self.high_corr_threshold = high_corr_threshold
        self.top_k_categories = top_k_categories

    def profile(
        self, df: pd.DataFrame, *, target_column: str | None = None
    ) -> DatasetProfile:
        """Profile every column and global relationship in ``df``."""
        n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
        total_cells = max(n_rows * n_cols, 1)
        missing_cells = int(df.isna().sum().sum())
        try:
            dup_rows = int(df.duplicated().sum())
        except TypeError:
            dup_rows = int(df.astype(str).duplicated().sum())

        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        datetime_cols: list[str] = []
        boolean_cols: list[str] = []
        col_profiles: list[ColumnProfile] = []
        global_alerts: list[str] = []

        for col in df.columns:
            series = df[col]
            col_name = str(col)
            semantic = infer_semantic_column_type(series)
            missing_cnt = int(series.isna().sum())
            missing_pct = round(missing_cnt / max(n_rows, 1) * 100.0, 2)
            non_null = series.dropna()
            try:
                uniq_cnt = int(non_null.nunique())
            except TypeError:
                uniq_cnt = 0
            uniq_pct = round(uniq_cnt / max(len(non_null), 1) * 100.0, 2) if len(non_null) > 0 else 0.0
            alerts: list[str] = []

            if uniq_cnt <= 1 and n_rows > 0:
                alerts.append("constant_column")
                global_alerts.append(f"Column '{col_name}' is constant (<=1 unique value).")
            if missing_pct >= 30.0:
                alerts.append("high_missingness")
                global_alerts.append(f"Column '{col_name}' has {missing_pct:.1f}% missing values.")

            cp_kwargs: dict[str, Any] = {
                "name": col_name,
                "dtype": str(series.dtype),
                "semantic_type": semantic,
                "count": int(len(non_null)),
                "missing_count": missing_cnt,
                "missing_pct": missing_pct,
                "unique_count": uniq_cnt,
                "unique_pct": uniq_pct,
                "alerts": alerts,
            }

            if pd.api.types.is_bool_dtype(series):
                boolean_cols.append(col_name)
                vc = non_null.value_counts().head(self.top_k_categories)
                cp_kwargs["top_values"] = {str(k): int(v) for k, v in vc.items()}
            elif pd.api.types.is_datetime64_any_dtype(series):
                datetime_cols.append(col_name)
            elif pd.api.types.is_numeric_dtype(series):
                numeric_cols.append(col_name)
                vals = non_null.astype(float)
                if len(vals) > 0:
                    skew_val = float(vals.skew()) if len(vals) >= 3 else 0.0
                    kurt_val = float(vals.kurt()) if len(vals) >= 4 else 0.0
                    if abs(skew_val) > 2.0:
                        alerts.append("highly_skewed")
                        global_alerts.append(f"Column '{col_name}' is highly skewed (skew={skew_val:.2f}).")
                    cp_kwargs.update(
                        {
                            "mean": round(float(vals.mean()), 6),
                            "std": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 6),
                            "min": round(float(vals.min()), 6),
                            "q05": round(float(vals.quantile(0.05)), 6),
                            "q25": round(float(vals.quantile(0.25)), 6),
                            "median": round(float(vals.median()), 6),
                            "q75": round(float(vals.quantile(0.75)), 6),
                            "q95": round(float(vals.quantile(0.95)), 6),
                            "max": round(float(vals.max()), 6),
                            "skewness": round(0.0 if np.isnan(skew_val) else skew_val, 4),
                            "kurtosis": round(0.0 if np.isnan(kurt_val) else kurt_val, 4),
                            "zeros_count": int((vals == 0).sum()),
                            "negatives_count": int((vals < 0).sum()),
                        }
                    )
            else:
                categorical_cols.append(col_name)
                if uniq_cnt > 50 and uniq_pct > 50.0:
                    alerts.append("high_cardinality")
                try:
                    vc = non_null.astype(str).value_counts().head(self.top_k_categories)
                    cp_kwargs["top_values"] = {str(k): int(v) for k, v in vc.items()}
                except Exception as exc:
                    logger.debug("categorical_value_counts_skipped", column=col_name, error=str(exc))

            col_profiles.append(ColumnProfile(**cp_kwargs))

        # Correlation analysis on numeric columns
        corr_dict: dict[str, dict[str, float]] = {}
        high_corrs: list[dict[str, Any]] = []
        target_corrs: dict[str, float] = {}

        if len(numeric_cols) >= 2 and n_rows >= 3:
            corr_df = df[numeric_cols].corr(method="pearson").fillna(0.0)
            corr_dict = {
                str(r): {str(c): round(float(corr_df.loc[r, c]), 4) for c in corr_df.columns}
                for r in corr_df.index
            }
            for i, col_a in enumerate(numeric_cols):
                for col_b in numeric_cols[i + 1 :]:
                    val = float(corr_df.loc[col_a, col_b])
                    if abs(val) >= self.high_corr_threshold:
                        high_corrs.append({"col_a": col_a, "col_b": col_b, "pearson_r": round(val, 4)})
                        global_alerts.append(
                            f"High collinearity between '{col_a}' and '{col_b}' (r={val:.2f})."
                        )
            if target_column and target_column in numeric_cols:
                t_series = corr_df[target_column].drop(labels=[target_column], errors="ignore")
                target_corrs = {
                    str(k): round(float(v), 4)
                    for k, v in t_series.reindex(t_series.abs().sort_values(ascending=False).index).items()
                }

        if target_column and target_column in df.columns:
            vc = df[target_column].dropna().value_counts(normalize=True)
            if 1 < len(vc) <= 20 and float(vc.min()) < 0.15:
                global_alerts.append(
                    f"Target '{target_column}' exhibits class imbalance (minority share={float(vc.min())*100:.1f}%)."
                )

        profile = DatasetProfile(
            n_rows=n_rows,
            n_columns=n_cols,
            memory_mb=round(float(df.memory_usage(deep=True).sum()) / (1024**2), 4),
            missing_cells=missing_cells,
            missing_pct=round(missing_cells / total_cells * 100.0, 2),
            duplicate_rows=dup_rows,
            duplicate_pct=round(dup_rows / max(n_rows, 1) * 100.0, 2),
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            datetime_columns=datetime_cols,
            boolean_columns=boolean_cols,
            columns=col_profiles,
            correlations_pearson=corr_dict,
            high_correlations=high_corrs,
            target_correlations=target_corrs,
            alerts=global_alerts,
        )
        logger.info("dataset_profiled", rows=n_rows, cols=n_cols, alerts=len(global_alerts))
        return profile
