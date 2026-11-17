"""Automated executive & technical reporting, visual charts, formula catalog, and multi-format exporters."""

from omniforge.reporting.charts import ReportChartBuilder
from omniforge.reporting.exporters import ReportExporters
from omniforge.reporting.formulas import FormulaRulesCatalog
from omniforge.reporting.generator import ReportGenerator
from omniforge.reporting.narrative import NarrativeGenerator
from omniforge.reporting.sections import (
    ALL_SECTION_IDS,
    REPORT_PRESETS,
    REPORT_SECTIONS_CATALOG,
    AnalysisAuditRecord,
    ExclusionRecord,
    FormulaRuleRecord,
    ReportChartArtifact,
    ReportSectionSpec,
    ReportSelectionConfig,
    compute_raw_vs_processed_diff,
)

__all__ = [
    "ALL_SECTION_IDS",
    "REPORT_PRESETS",
    "REPORT_SECTIONS_CATALOG",
    "AnalysisAuditRecord",
    "ExclusionRecord",
    "FormulaRuleRecord",
    "FormulaRulesCatalog",
    "NarrativeGenerator",
    "ReportChartArtifact",
    "ReportChartBuilder",
    "ReportExporters",
    "ReportGenerator",
    "ReportSectionSpec",
    "ReportSelectionConfig",
    "compute_raw_vs_processed_diff",
]
