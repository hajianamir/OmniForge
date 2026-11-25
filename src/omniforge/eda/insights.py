"""Automated analytical insight discovery and executive narrative synthesis."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from omniforge.eda.profiler import DatasetProfile
from omniforge.eda.statistics import StatisticalAnalyzer


class AnalyticalInsight(BaseModel):
    """Single actionable analytical finding surfaced from a dataset."""

    category: str  # quality, distribution, relationship, target, multicollinearity, temporal
    severity: str = "info"  # info, warning, critical
    title: str
    description: str
    recommendation: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class InsightGenerator:
    """Extracts ranked analytical insights and produces natural-language executive narratives."""

    def __init__(self, llm_client: Any | None = None) -> None:
        self._llm = llm_client
        self._stats = StatisticalAnalyzer()

    def generate(
        self,
        df: pd.DataFrame,
        profile: DatasetProfile,
        *,
        target_column: str | None = None,
    ) -> tuple[list[AnalyticalInsight], str]:
        """Discover analytical insights and synthesize a cohesive executive narrative."""
        insights: list[AnalyticalInsight] = []

        # 1. Scale & completeness overview
        insights.append(
            AnalyticalInsight(
                category="quality",
                severity="warning" if profile.missing_pct > 5.0 else "info",
                title="Dataset Footprint & Completeness",
                description=(
                    f"Dataset spans {profile.n_rows:,} rows and {profile.n_columns} columns "
                    f"({profile.memory_mb:.2f} MB) with {profile.missing_pct:.2f}% missing cells "
                    f"and {profile.duplicate_pct:.2f}% duplicate rows."
                ),
                recommendation=(
                    "Run DataCleaner with imputation and deduplication before model training."
                    if (profile.missing_pct > 1.0 or profile.duplicate_rows > 0)
                    else "Data completeness is strong; standard preprocessing suffices."
                ),
                evidence={
                    "rows": profile.n_rows,
                    "columns": profile.n_columns,
                    "missing_pct": profile.missing_pct,
                    "duplicate_rows": profile.duplicate_rows,
                },
            )
        )

        # 2. Highly skewed numeric features
        skewed = [
            c for c in profile.columns if c.skewness is not None and abs(c.skewness) >= 2.0
        ]
        if skewed:
            names = [f"{c.name} ({c.skewness:+.2f})" for c in skewed[:6]]
            insights.append(
                AnalyticalInsight(
                    category="distribution",
                    severity="warning",
                    title="Strong Distributional Skew Detected",
                    description=f"{len(skewed)} numeric columns exhibit strong skewness: {', '.join(names)}.",
                    recommendation="Apply Yeo-Johnson / log1p power transformation for linear or neural models.",
                    evidence={"skewed_columns": {c.name: c.skewness for c in skewed}},
                )
            )

        # 3. Multicollinearity
        if profile.high_correlations:
            pairs = [
                f"{item['col_a']} <-> {item['col_b']} (r={item['pearson_r']})"
                for item in profile.high_correlations[:5]
            ]
            insights.append(
                AnalyticalInsight(
                    category="multicollinearity",
                    severity="warning",
                    title="High Feature Collinearity",
                    description=f"Identified {len(profile.high_correlations)} highly correlated feature pairs: {'; '.join(pairs)}.",
                    recommendation="Apply correlation pruning, PCA, or regularized models (Ridge/Lasso/ElasticNet).",
                    evidence={"pairs": profile.high_correlations[:10]},
                )
            )

        # 4. Target signal & class balance analysis
        if target_column and target_column in df.columns:
            target_series = df[target_column].dropna()
            if profile.target_correlations:
                top_drivers = list(profile.target_correlations.items())[:5]
                driver_str = ", ".join(f"{k} (r={v:+.3f})" for k, v in top_drivers)
                insights.append(
                    AnalyticalInsight(
                        category="target",
                        severity="info",
                        title=f"Primary Linear Drivers of '{target_column}'",
                        description=f"Strongest numeric linear correlations with '{target_column}': {driver_str}.",
                        recommendation="Prioritize interaction and polynomial features among top drivers.",
                        evidence={"top_drivers": dict(top_drivers)},
                    )
                )
            elif 1 < target_series.nunique() <= 20:
                vc = target_series.value_counts(normalize=True)
                minority = float(vc.min())
                insights.append(
                    AnalyticalInsight(
                        category="target",
                        severity="warning" if minority < 0.15 else "info",
                        title=f"Target Class Distribution ('{target_column}')",
                        description=(
                            f"Target contains {len(vc)} classes; largest share is {float(vc.max())*100:.1f}% "
                            f"and smallest share is {minority*100:.1f}%."
                        ),
                        recommendation=(
                            "Use stratified CV, class weighting, and AUC-PR / F1 metrics due to imbalance."
                            if minority < 0.15
                            else "Class balance is healthy; standard stratified CV and F1/ROC-AUC apply."
                        ),
                        evidence={"class_distribution": {str(k): round(float(v), 4) for k, v in vc.items()}},
                    )
                )

        # 5. Constant or zero-variance columns
        const_cols = [c.name for c in profile.columns if "constant_column" in c.alerts]
        if const_cols:
            insights.append(
                AnalyticalInsight(
                    category="quality",
                    severity="warning",
                    title="Zero-Variance Columns Detected",
                    description=f"Columns with <=1 unique value carry no predictive signal: {', '.join(const_cols)}.",
                    recommendation="Drop constant columns during feature selection.",
                    evidence={"constant_columns": const_cols},
                )
            )

        # 6. Temporal stationarity insight if datetime + numeric target exist
        if profile.datetime_columns and target_column and target_column in profile.numeric_columns:
            sorted_df = df.sort_values(profile.datetime_columns[0])
            adf = self._stats.stationarity_test(sorted_df[target_column])
            insights.append(
                AnalyticalInsight(
                    category="temporal",
                    severity="info" if adf.reject_null else "warning",
                    title="Time-Series Stationarity Assessment",
                    description=adf.interpretation,
                    recommendation=(
                        "Use time-series cross-validation and lag/rolling/seasonal features."
                        if not adf.reject_null
                        else "Series is stationary; ARMA/ML lag features can be modeled directly."
                    ),
                    evidence={"adf_stat": adf.statistic, "p_value": adf.p_value},
                )
            )

        narrative = self._synthesize_narrative(insights, profile, target_column)
        return insights, narrative

    def _synthesize_narrative(
        self,
        insights: list[AnalyticalInsight],
        profile: DatasetProfile,
        target_column: str | None,
    ) -> str:
        """Build a concise executive summary paragraph (with optional LLM enrichment)."""
        bullet_lines = [f"- **{ins.title}**: {ins.description} Recommended action: {ins.recommendation}" for ins in insights]
        base_narrative = (
            f"### Executive Analytical Summary\n\n"
            f"The analyzed dataset comprises **{profile.n_rows:,} records** across **{profile.n_columns} attributes** "
            f"({len(profile.numeric_columns)} numeric, {len(profile.categorical_columns)} categorical, "
            f"{len(profile.datetime_columns)} datetime). "
            + (f"Target variable: **`{target_column}`**. " if target_column else "")
            + f"Overall missingness is **{profile.missing_pct:.2f}%** with **{len(profile.alerts)}** structural alerts.\n\n"
            + "#### Key Findings\n"
            + "\n".join(bullet_lines)
        )
        if self._llm is not None and callable(getattr(self._llm, "complete", None)):
            try:
                prompt = (
                    "Rewrite and enrich the following analytical summary into a crisp executive briefing "
                    "without altering any numeric statistics:\n\n" + base_narrative
                )
                return str(self._llm.complete(prompt))
            except Exception:
                return base_narrative
        return base_narrative
