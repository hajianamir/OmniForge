"""Multi-dimensional enterprise Data Quality Profiler (6 canonical quality dimensions)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from omniforge.core.logging import get_logger
from omniforge.core.performance import GLOBAL_DF_CACHE

logger = get_logger(__name__)


class ColumnQualityScore(BaseModel):
    """Granular quality metrics for an individual column."""

    column: str
    dtype: str
    completeness: float
    uniqueness: float
    validity: float
    consistency: float
    outlier_ratio: float
    overall_score: float
    issues: list[str] = Field(default_factory=list)


class DataQualityReport(BaseModel):
    """Comprehensive six-dimensional Data Quality scorecard and remediation plan."""

    overall_score: float = Field(ge=0.0, le=100.0)
    grade: str
    dimensions: dict[str, float]
    row_count: int
    column_count: int
    duplicate_row_ratio: float
    missing_cell_ratio: float
    column_scores: list[ColumnQualityScore] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    profiled_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DataQualityProfiler:
    """Computes completeness, uniqueness, validity, consistency, accuracy, and timeliness scores."""

    DEFAULT_WEIGHTS: dict[str, float] = {
        "completeness": 0.25,
        "uniqueness": 0.15,
        "validity": 0.20,
        "consistency": 0.15,
        "accuracy": 0.15,
        "timeliness": 0.10,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        raw = dict(weights or self.DEFAULT_WEIGHTS)
        total = sum(raw.values()) or 1.0
        self.weights = {k: v / total for k, v in raw.items()}

    def profile(
        self,
        df: pd.DataFrame,
        *,
        timestamp_column: str | None = None,
        max_freshness_days: float = 365.0,
        reference_rules: dict[str, tuple[float, float]] | None = None,
        key_columns: list[str] | None = None,
    ) -> DataQualityReport:
        """Profile a DataFrame across all six enterprise data quality dimensions.

        Args:
            df: Input DataFrame.
            timestamp_column: Optional datetime column used to score timeliness.
            max_freshness_days: Age threshold beyond which timeliness degrades.
            reference_rules: Optional ``{column: (min_valid, max_valid)}`` domain accuracy ranges.
            key_columns: Optional primary key columns whose uniqueness is scored strictly.

        Returns:
            :class:`DataQualityReport` with scores on a 0–100 scale and recommendations.
        """
        cache_params = f"{timestamp_column}|{max_freshness_days}|{sorted(reference_rules.keys()) if reference_rules else ''}|{key_columns}"
        hit, cached = GLOBAL_DF_CACHE.get("data_quality", df, cache_params)
        if hit and isinstance(cached, DataQualityReport):
            return cached

        n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
        if n_rows == 0 or n_cols == 0:
            return DataQualityReport(
                overall_score=0.0,
                grade="F",
                dimensions={k: 0.0 for k in self.weights},
                row_count=n_rows,
                column_count=n_cols,
                duplicate_row_ratio=0.0,
                missing_cell_ratio=1.0,
                recommendations=["Dataset is empty; verify upstream ingestion pipeline."],
            )

        missing_cell_ratio = float(df.isna().sum().sum()) / float(n_rows * n_cols)
        completeness = max(0.0, (1.0 - missing_cell_ratio) * 100.0)

        try:
            dup_ratio = float(df.duplicated(subset=key_columns).mean())
        except TypeError:
            dup_ratio = float(df.astype(str).duplicated(subset=key_columns).mean())
        uniqueness = max(0.0, (1.0 - dup_ratio) * 100.0)

        col_scores: list[ColumnQualityScore] = []
        validity_parts: list[float] = []
        consistency_parts: list[float] = []
        accuracy_parts: list[float] = []
        recommendations: list[str] = []

        rules = reference_rules or {}

        for col in df.columns:
            series = df[col]
            col_name = str(col)
            col_issues: list[str] = []

            col_comp = float(series.notna().mean()) * 100.0
            if col_comp < 90.0:
                col_issues.append(f"High missing rate ({100.0 - col_comp:.1f}%).")

            try:
                col_uniq = float(series.nunique(dropna=True)) / max(int(series.notna().sum()), 1) * 100.0
            except TypeError:
                col_uniq = 100.0

            # Validity: finite numerics, no placeholder tokens in strings, no NaT
            col_valid = 100.0
            outlier_ratio = 0.0
            col_consistency = 100.0
            col_accuracy = 100.0

            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                vals = series.dropna().astype(float)
                if len(vals) > 0:
                    inf_ratio = float(np.isinf(vals).mean())
                    col_valid = max(0.0, (1.0 - inf_ratio) * 100.0)
                    if inf_ratio > 0:
                        col_issues.append(f"Contains infinite values ({inf_ratio*100:.1f}%).")

                    # Outlier density informs statistical accuracy
                    q1, q3 = float(vals.quantile(0.25)), float(vals.quantile(0.75))
                    iqr = q3 - q1
                    if iqr > 0:
                        out_mask = (vals < q1 - 3.0 * iqr) | (vals > q3 + 3.0 * iqr)
                        outlier_ratio = float(out_mask.mean())
                        col_accuracy = max(0.0, (1.0 - min(outlier_ratio * 2.0, 1.0)) * 100.0)
                        if outlier_ratio > 0.05:
                            col_issues.append(f"Extreme outlier density ({outlier_ratio*100:.1f}%).")

                    if col_name in rules:
                        low, high = rules[col_name]
                        in_range = float(((vals >= low) & (vals <= high)).mean()) * 100.0
                        col_accuracy = min(col_accuracy, in_range)
                        if in_range < 100.0:
                            col_issues.append(f"Domain range [{low}, {high}] violated by {100-in_range:.1f}% of values.")
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 25_000:
                    non_null = non_null.sample(n=25_000, random_state=42)
                non_null = non_null.astype(str)
                if len(non_null) > 0:
                    # Consistency checks case/whitespace fragmentation in categoricals
                    raw_unique = non_null.nunique()
                    norm_unique = non_null.str.strip().str.lower().nunique()
                    if raw_unique > 0 and norm_unique < raw_unique:
                        frag = (raw_unique - norm_unique) / raw_unique
                        col_consistency = max(0.0, (1.0 - frag) * 100.0)
                        col_issues.append("Inconsistent casing or trailing whitespace across categories.")
                    bad_tokens = non_null.str.strip().str.lower().isin({"na", "n/a", "null", "none", "?", ""}).mean()
                    col_valid = max(0.0, (1.0 - float(bad_tokens)) * 100.0)
                    if bad_tokens > 0:
                        col_issues.append(f"Contains unparsed null placeholder strings ({bad_tokens*100:.1f}%).")

            validity_parts.append(col_valid)
            consistency_parts.append(col_consistency)
            accuracy_parts.append(col_accuracy)

            col_overall = round(
                0.35 * col_comp + 0.25 * col_valid + 0.20 * col_consistency + 0.20 * col_accuracy, 2
            )
            col_scores.append(
                ColumnQualityScore(
                    column=col_name,
                    dtype=str(series.dtype),
                    completeness=round(col_comp, 2),
                    uniqueness=round(col_uniq, 2),
                    validity=round(col_valid, 2),
                    consistency=round(col_consistency, 2),
                    outlier_ratio=round(outlier_ratio, 4),
                    overall_score=col_overall,
                    issues=col_issues,
                )
            )

        validity = float(np.mean(validity_parts)) if validity_parts else 100.0
        consistency = float(np.mean(consistency_parts)) if consistency_parts else 100.0
        accuracy = float(np.mean(accuracy_parts)) if accuracy_parts else 100.0
        timeliness = self._compute_timeliness(df, timestamp_column, max_freshness_days)

        dimensions = {
            "completeness": round(completeness, 2),
            "uniqueness": round(uniqueness, 2),
            "validity": round(validity, 2),
            "consistency": round(consistency, 2),
            "accuracy": round(accuracy, 2),
            "timeliness": round(timeliness, 2),
        }
        overall = round(sum(dimensions[k] * w for k, w in self.weights.items()), 2)
        grade = self._score_to_grade(overall)

        if completeness < 95.0:
            recommendations.append(
                f"Completeness is {completeness:.1f}%. Configure DataCleaner imputation and missing indicators."
            )
        if dup_ratio > 0.01:
            recommendations.append(
                f"Detected {dup_ratio*100:.2f}% duplicate rows. Enable exact or fuzzy deduplication."
            )
        if consistency < 95.0:
            recommendations.append(
                "String casing/whitespace inconsistencies found. Enable string cleaning and normalization."
            )
        if accuracy < 92.0:
            recommendations.append(
                "Elevated extreme outlier density or range violations detected. Apply IQR/IsolationForest handling."
            )
        if timeliness < 85.0:
            recommendations.append(
                "Dataset freshness is below threshold. Schedule incremental ingestion or check pipeline SLA."
            )
        if not recommendations:
            recommendations.append("Dataset passes all enterprise quality gates cleanly.")

        report = DataQualityReport(
            overall_score=overall,
            grade=grade,
            dimensions=dimensions,
            row_count=n_rows,
            column_count=n_cols,
            duplicate_row_ratio=round(dup_ratio, 4),
            missing_cell_ratio=round(missing_cell_ratio, 4),
            column_scores=col_scores,
            recommendations=recommendations,
        )
        logger.info("data_quality_profiled", overall_score=overall, grade=grade)
        GLOBAL_DF_CACHE.set("data_quality", df, report, cache_params)
        return report

    @staticmethod
    def _compute_timeliness(
        df: pd.DataFrame, timestamp_column: str | None, max_freshness_days: float
    ) -> float:
        ts_col = timestamp_column
        if ts_col is None:
            dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
            ts_col = str(dt_cols[0]) if dt_cols else None
        if ts_col is None or ts_col not in df.columns:
            return 100.0
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dropna()
        if len(ts) == 0:
            return 50.0
        latest = ts.max().to_pydatetime()
        now = datetime.now(UTC)
        age_days = max(0.0, (now - latest).total_seconds() / 86400.0)
        if age_days <= max_freshness_days * 0.25:
            return 100.0
        decay = max(0.0, 1.0 - age_days / max(max_freshness_days * 2.0, 1.0))
        return round(decay * 100.0, 2)

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 95.0:
            return "A+"
        if score >= 90.0:
            return "A"
        if score >= 80.0:
            return "B"
        if score >= 70.0:
            return "C"
        if score >= 60.0:
            return "D"
        return "F"
