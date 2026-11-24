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
    # Target-leakage checks (new in Fix #8)
    target_leakage_detected: bool = False
    target_leakage_columns: list[str] = Field(default_factory=list)
    target_leakage_details: list[str] = Field(default_factory=list)


class LeakageGuard:
    """Validates train/test splits against row, group, and temporal leakage."""

    @staticmethod
    def validate_split(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        group_col: str | None = None,
        time_col: str | None = None,
        target_col: str | None = None,
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

        # Target-leakage detection: flag features that are copies of, or near-perfect
        # predictors of, the target column. A feature identical to the target gives
        # a trivial score of 1.0 and must be caught before modeling.
        target_leakage_detected = False
        target_leakage_columns: list[str] = []
        target_leakage_details: list[str] = []

        if target_col and target_col in train_df.columns and target_col in test_df.columns:
            y_train = train_df[target_col]
            y_test = test_df[target_col]
            feature_cols = [
                c
                for c in train_df.columns
                if c != target_col and c in test_df.columns
            ]
            for col in feature_cols:
                x_train = train_df[col]
                x_test = test_df[col]

                # 1. Exact copy: feature values equal target values (dtype-agnostic compare)
                if x_train.dtype == y_train.dtype:
                    train_copy = (x_train == y_train).all()
                    test_copy = (x_test == y_test).all()
                else:
                    train_copy = (x_train.astype(str) == y_train.astype(str)).all()
                    test_copy = (x_test.astype(str) == y_test.astype(str)).all()
                if train_copy and test_copy:
                    target_leakage_detected = True
                    target_leakage_columns.append(col)
                    target_leakage_details.append(
                        f"Column '{col}' is an exact copy of target '{target_col}' in both train and test."
                    )
                    continue

                # 2. Near-perfect numeric correlation (|r| > 0.999)
                if (
                    pd.api.types.is_numeric_dtype(x_train)
                    and pd.api.types.is_numeric_dtype(y_train)
                    and len(x_train.dropna()) > 2
                    and len(x_test.dropna()) > 2
                ):
                    try:
                        corr_tr = float(x_train.corr(y_train))
                        corr_te = float(x_test.corr(y_test))
                        if abs(corr_tr) > 0.999 and abs(corr_te) > 0.999:
                            target_leakage_detected = True
                            target_leakage_columns.append(col)
                            target_leakage_details.append(
                                f"Column '{col}' has near-perfect correlation with target '{target_col}' "
                                f"(train r={corr_tr:.6f}, test r={corr_te:.6f})."
                            )
                            continue
                    except Exception:
                        pass

                # 3. Near-perfect categorical prediction: each feature value maps to
                #    exactly one target value (one-to-one) in both splits
                if (
                    not pd.api.types.is_numeric_dtype(x_train)
                    or not pd.api.types.is_numeric_dtype(y_train)
                ):
                    try:
                        tr_groups = train_df.groupby(x_train.astype(str))[y_train.name].nunique()
                        te_groups = test_df.groupby(x_test.astype(str))[y_test.name].nunique()
                        if (tr_groups == 1).all() and (te_groups == 1).all():
                            target_leakage_detected = True
                            target_leakage_columns.append(col)
                            target_leakage_details.append(
                                f"Column '{col}' perfectly predicts target '{target_col}' "
                                f"(one-to-one mapping in both train and test)."
                            )
                            continue
                    except Exception:
                        pass

        # Any target leakage makes the split fail
        if target_leakage_detected:
            warnings.extend(target_leakage_details)

        passed = (
            (row_overlap == 0)
            and (group_overlap == 0)
            and temporal_valid
            and not target_leakage_detected
        )
        return LeakageGuardReport(
            passed=passed,
            row_overlap_count=int(row_overlap),
            group_overlap_count=int(group_overlap),
            temporal_order_valid=temporal_valid,
            train_size=int(len(train_df)),
            test_size=int(len(test_df)),
            target_leakage_detected=target_leakage_detected,
            target_leakage_columns=target_leakage_columns,
            target_leakage_details=target_leakage_details,
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
