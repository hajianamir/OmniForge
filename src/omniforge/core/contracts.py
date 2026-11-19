"""Core methodological contracts: AnalysisSpec, LeakageGuard, and ResultBundle."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class AnalysisSpec(BaseModel):
    """Shared specification (s00_plan) governing splitter, tests, metrics, and reporting."""

    task_kind: Literal[
        "descriptive",
        "hypothesis_test",
        "prediction",
        "time_series_forecast",
        "clustering",
        "causal_inference",
        "survival",
        "optimization",
    ] = "prediction"
    analysis_mode: Literal["exploratory", "confirmatory"] = "exploratory"
    independent_unit: str = "row"
    target_column: str | None = None
    target_timestamp_column: str | None = None
    time_column: str | None = None
    group_column: str | None = None
    id_column: str | None = None
    forecast_horizon: int | None = Field(default=None, ge=1)
    treatment_column: str | None = None
    outcome_column: str | None = None
    estimand: Literal["ATE", "ATT", "ATC", "CATE"] | None = None
    primary_metric: str = "auto"
    decision_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    sensitive_attributes: list[str] = Field(default_factory=list)
    random_seed: int = 42


class LeakageGuardReport(BaseModel):
    """Audit report returned by LeakageGuard."""

    passed: bool
    row_overlap_count: int = 0
    group_overlap_count: int = 0
    temporal_order_valid: bool = True
    train_size: int = 0
    test_size: int = 0
    warnings: list[str] = Field(default_factory=list)
    seed: int = 42
    checked_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class LeakageGuard:
    """Validates train/test splits against row, group, and temporal leakage."""

    @staticmethod
    def validate_split(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        group_col: str | None = None,
        time_col: str | None = None,
        seed: int = 42,
    ) -> LeakageGuardReport:
        warnings: list[str] = []
        row_overlap = len(train_df.index.intersection(test_df.index))
        if row_overlap > 0:
            warnings.append(f"Row index overlap detected: {row_overlap} shared indices.")

        group_overlap = 0
        if group_col and group_col in train_df.columns and group_col in test_df.columns:
            tr_g = set(train_df[group_col].dropna().tolist())
            te_g = set(test_df[group_col].dropna().tolist())
            group_overlap = len(tr_g & te_g)
            if group_overlap > 0:
                warnings.append(f"Group leakage detected on '{group_col}': {group_overlap} shared groups.")

        temporal_valid = True
        if time_col and time_col in train_df.columns and time_col in test_df.columns:
            tr_t = pd.to_datetime(train_df[time_col], errors="coerce").dropna()
            te_t = pd.to_datetime(test_df[time_col], errors="coerce").dropna()
            if not tr_t.empty and not te_t.empty:
                if tr_t.max() > te_t.min():
                    temporal_valid = False
                    warnings.append(
                        f"Temporal leakage on '{time_col}': max(train)={tr_t.max()} > min(test)={te_t.min()}."
                    )

        passed = (row_overlap == 0) and (group_overlap == 0) and temporal_valid
        return LeakageGuardReport(
            passed=passed,
            row_overlap_count=int(row_overlap),
            group_overlap_count=int(group_overlap),
            temporal_order_valid=temporal_valid,
            train_size=int(len(train_df)),
            test_size=int(len(test_df)),
            warnings=warnings,
            seed=seed,
        )


class ResultBundle(BaseModel):
    """Standardized statistical/modeling result container with effect size, CI, and provenance."""

    analysis_name: str
    test_or_model: str
    effect_estimate: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    statistic: float | None = None
    p_value_raw: float | None = None
    p_value_adjusted: float | None = None
    multiple_testing_method: str | None = None
    n_samples: int = 0
    n_dropped: int = 0
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def bootstrap_ci(
        values: np.ndarray | list[float],
        *,
        n_boot: int = 500,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> tuple[float, float, float]:
        """Compute mean estimate and percentile bootstrap CI."""
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            raise ValueError("bootstrap_ci requires at least one finite value.")
        rng = np.random.default_rng(seed)
        means = np.empty(n_boot, dtype=float)
        n = len(arr)
        for i in range(n_boot):
            sample = rng.choice(arr, size=n, replace=True)
            means[i] = float(np.mean(sample))
        lo = float(np.quantile(means, alpha / 2.0))
        hi = float(np.quantile(means, 1.0 - alpha / 2.0))
        return float(np.mean(arr)), lo, hi
