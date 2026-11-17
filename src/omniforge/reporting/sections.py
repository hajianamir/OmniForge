"""Structured report section catalog, audit models, and user selection schemas for OmniForge Reporting."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ReportSectionSpec(BaseModel):
    """Metadata describing one user-selectable report section."""

    section_id: str
    title: str
    description: str
    category: str = "Core"
    default_enabled: bool = True


class AnalysisAuditRecord(BaseModel):
    """Structured record of one analysis executed during the analytical workflow."""

    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"))
    studio: str
    analysis_name: str
    dataset_mode: str = "RAW"
    criteria_or_config: str = ""
    inputs: str = ""
    outcome_summary: str = ""
    status: str = "Completed"


class ExclusionRecord(BaseModel):
    """Structured record documenting what data was excluded or altered from calculations and why."""

    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"))
    stage: str
    exclusion_type: str
    affected_scope: str
    count: int = 0
    pct_of_dataset: float = 0.0
    criterion_rule: str = ""
    reason: str = ""
    sample_identifiers: list[str] = Field(default_factory=list)


class FormulaRuleRecord(BaseModel):
    """Structured mathematical formula, model equation, computational rule, and plain-language explanation."""

    category: str
    name: str
    mathematical_expression: str
    variables_and_parameters: str
    computational_rule: str
    detailed_explanation: str
    scope: str = "Global Methodology"


class ReportChartArtifact(BaseModel):
    """Rendered visual chart artifact ready for embedding into DOCX, PDF, XLSX, HTML, and PPTX."""

    chart_id: str
    section_id: str
    title: str
    caption: str
    png_base64: str
    width_inches: float = 6.2
    height_inches: float = 3.6


class ReportSelectionConfig(BaseModel):
    """User-configurable selection of sections, charts, and appendices to include in a generated report."""

    selected_sections: list[str] = Field(
        default_factory=lambda: [spec.section_id for spec in REPORT_SECTIONS_CATALOG if spec.default_enabled]
    )
    include_charts: bool = True
    include_tables: bool = True
    include_dataset_appendix: bool = True
    max_table_rows: int = 100
    author: str = "OmniForge Enterprise"

    def is_enabled(self, section_id: str) -> bool:
        """Return True if ``section_id`` is selected by the user."""
        return section_id in self.selected_sections


REPORT_SECTIONS_CATALOG: tuple[ReportSectionSpec, ...] = (
    ReportSectionSpec(
        section_id="executive_summary",
        title="Executive Summary, Dataset Profile & KPIs",
        description="High-level study overview, dataset dimensions, target structure, RAW vs. PROCESSED provenance, and KPI scorecard.",
        category="Executive",
        default_enabled=True,
    ),
    ReportSectionSpec(
        section_id="analyses_performed",
        title="Audit Log of All Analyses Performed",
        description="Chronological record of every studio, statistical test, cleaning action, clustering run, and modeling pipeline executed.",
        category="Audit & Governance",
        default_enabled=True,
    ),
    ReportSectionSpec(
        section_id="data_exclusions",
        title="Excluded Data, Filtering & Removals Log (What & Why)",
        description="Exact rows, columns, outliers, duplicates, noise clusters, and dropped features excluded from calculations with exact rules and justifications.",
        category="Audit & Governance",
        default_enabled=True,
    ),
    ReportSectionSpec(
        section_id="intake_audit",
        title="Automatic Post-Ingestion Defect Audit & Statistical Pre-Tests",
        description="Immediate intake health scan: format defects, non-finite values, missingness, normality tests, skewness, multicollinearity & leakage alerts.",
        category="Data & Statistics",
        default_enabled=True,
    ),
    ReportSectionSpec(
        section_id="quality_diagnostics",
        title="6D Data Quality Scorecard & Statistical Diagnostics",
        description="Completeness, uniqueness, validity, consistency, column profiles, normality tests, VIF multicollinearity, and data validation rules.",
        category="Data & Statistics",
        default_enabled=True,
    ),
    ReportSectionSpec(
        section_id="clustering_anomalies",
        title="Clustering, Data Mining (Rules & Subgroups) & Anomaly Screening",
        description="Unsupervised segmentation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz), cluster profiles, association rules, subgroups, and anomaly screening.",
        category="Data & Statistics",
        default_enabled=True,
    ),
    ReportSectionSpec(
        section_id="eda_hypothesis",
        title="Automated EDA, Correlations & Hypothesis Tests",
        description="Distributional statistics, correlation structure, hypothesis test statistics, p-values, effect sizes, and significant relationships.",
        category="Data & Statistics",
        default_enabled=True,
    ),
    ReportSectionSpec(
        section_id="feature_engineering",
        title="Feature Engineering, Selection & Transformations",
        description="Engineered features, selection rankings, retained vs. dropped predictors, and dimensionality reduction summary.",
        category="Modeling",
        default_enabled=True,
    ),
    ReportSectionSpec(
        section_id="modeling_leaderboard",
        title="Modeling, AutoML & Deep Learning vs. Baselines",
        description="Champion model selection, cross-validated leaderboard, baseline benchmarks, verified lift over baseline, and multi-target comparison.",
        category="Modeling",
        default_enabled=True,
    ),
    ReportSectionSpec(
        section_id="mathematical_equations",
        title="Mathematical Relationships, Model Equations, Computational Rules & Explanations",
        description="All analytical formulas, statistical rules, outlier fences, and extracted fitted model equations (Baselines, Classical, Neural, Ensembles) with detailed explanations.",
        category="Methodology & Equations",
        default_enabled=True,
    ),
    ReportSectionSpec(
        section_id="future_predictions",
        title="Future Target Predictions, Actual-vs-Baseline Supervisory Monitor & Tolerance Alarms",
        description="Live operational predictions from Excel/SQL/Access, Actual vs. Model vs. Baseline accuracy, configurable error tolerance compliance, SPC bounds, and breach alarms.",
        category="Modeling",
        default_enabled=True,
    ),
    ReportSectionSpec(
        section_id="responsible_ai_governance",
        title="Responsible AI: Explainability, Fairness, Robustness & Green AI",
        description="SHAP/LIME attributions, Disparate Impact & Equalized Odds fairness gates, adversarial robustness, and carbon footprint.",
        category="Audit & Governance",
        default_enabled=True,
    ),
    ReportSectionSpec(
        section_id="decision_intelligence",
        title="Forecasting, Causal Inference, Risk, Optimization & Simulation",
        description="Structural Risk Index (VaR/CVaR), time-series champion forecasts, causal ATE estimates, constrained optimization, and Monte Carlo simulations.",
        category="Decision Science",
        default_enabled=True,
    ),
    ReportSectionSpec(
        section_id="recommendations",
        title="Prioritized Action Plan & Recommendations",
        description="Concrete engineering, statistical, governance, and deployment next steps derived from workspace evidence.",
        category="Executive",
        default_enabled=True,
    ),
)

ALL_SECTION_IDS: tuple[str, ...] = tuple(spec.section_id for spec in REPORT_SECTIONS_CATALOG)

REPORT_PRESETS: dict[str, list[str]] = {
    "Full Comprehensive Audit Report (All Sections)": list(ALL_SECTION_IDS),
    "Executive Briefing": [
        "executive_summary",
        "analyses_performed",
        "data_exclusions",
        "modeling_leaderboard",
        "recommendations",
    ],
    "Data Quality & Exclusions Forensic Audit": [
        "executive_summary",
        "analyses_performed",
        "data_exclusions",
        "quality_diagnostics",
        "clustering_anomalies",
        "eda_hypothesis",
        "mathematical_equations",
        "recommendations",
    ],
    "Modeling, Equations & Predictions Technical Report": [
        "executive_summary",
        "data_exclusions",
        "feature_engineering",
        "modeling_leaderboard",
        "mathematical_equations",
        "future_predictions",
        "responsible_ai_governance",
        "recommendations",
    ],
}


def compute_raw_vs_processed_diff(
    raw_df: Any | None,
    active_df: Any | None,
) -> dict[str, Any]:
    """Compute an automatic forensic diff between raw ingested data and processed active data."""
    if raw_df is None or active_df is None:
        return {}

    import numpy as np
    import pandas as pd

    if not isinstance(raw_df, pd.DataFrame) or not isinstance(active_df, pd.DataFrame):
        return {}

    raw_rows, raw_cols = int(len(raw_df)), int(raw_df.shape[1])
    cur_rows, cur_cols = int(len(active_df)), int(active_df.shape[1])
    rows_delta = cur_rows - raw_rows
    removed_cols = [str(c) for c in raw_df.columns if c not in active_df.columns]
    added_cols = [str(c) for c in active_df.columns if c not in raw_df.columns]
    raw_missing = int(raw_df.isna().sum().sum())
    cur_missing = int(active_df.isna().sum().sum())
    raw_dups = int(raw_df.duplicated().sum())
    cur_dups = int(active_df.duplicated().sum())

    # Identify numeric columns whose min/max range changed (evidence of outlier clipping or filtering)
    range_shifts: list[dict[str, Any]] = []
    shared_num = [
        c
        for c in raw_df.select_dtypes(include=[np.number]).columns
        if c in active_df.columns and pd.api.types.is_numeric_dtype(active_df[c])
    ]
    for col in shared_num[:25]:
        r_ser = pd.to_numeric(raw_df[col], errors="coerce").dropna()
        c_ser = pd.to_numeric(active_df[col], errors="coerce").dropna()
        if r_ser.empty or c_ser.empty:
            continue
        r_min, r_max = float(r_ser.min()), float(r_ser.max())
        c_min, c_max = float(c_ser.min()), float(c_ser.max())
        if abs(r_min - c_min) > 1e-6 or abs(r_max - c_max) > 1e-6:
            range_shifts.append(
                {
                    "column": str(col),
                    "raw_min": round(r_min, 4),
                    "raw_max": round(r_max, 4),
                    "processed_min": round(c_min, 4),
                    "processed_max": round(c_max, 4),
                    "raw_std": round(float(r_ser.std(ddof=1)) if len(r_ser) > 1 else 0.0, 4),
                    "processed_std": round(float(c_ser.std(ddof=1)) if len(c_ser) > 1 else 0.0, 4),
                }
            )

    return {
        "raw_rows": raw_rows,
        "processed_rows": cur_rows,
        "rows_removed": max(0, -rows_delta),
        "rows_added": max(0, rows_delta),
        "rows_removed_pct": round((max(0, -rows_delta) / max(raw_rows, 1)) * 100.0, 3),
        "raw_columns": raw_cols,
        "processed_columns": cur_cols,
        "removed_columns": removed_cols,
        "added_columns": added_cols,
        "raw_missing_cells": raw_missing,
        "processed_missing_cells": cur_missing,
        "missing_cells_resolved": max(0, raw_missing - cur_missing),
        "raw_duplicate_rows": raw_dups,
        "processed_duplicate_rows": cur_dups,
        "numeric_range_shifts": range_shifts,
    }
