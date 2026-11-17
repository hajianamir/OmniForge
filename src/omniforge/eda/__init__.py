"""Exploratory Data Analysis, statistical testing, visualization, and insight generation."""

from omniforge.eda.automated_eda import AutomatedEDA, EDAReport
from omniforge.eda.diagnostics import (
    DIAGNOSTIC_TEST_CATALOG,
    DOWNSTREAM_MODULE_CATALOG,
    StatisticalCriteriaConfig,
    UniversalDiagnosticsReport,
    UniversalDiagnosticsSuite,
)
from omniforge.eda.insights import AnalyticalInsight, InsightGenerator
from omniforge.eda.profiler import ColumnProfile, DataProfiler, DatasetProfile
from omniforge.eda.statistics import HypothesisTestResult, StatisticalAnalyzer
from omniforge.eda.visualization import VisualizationEngine

__all__ = [
    "DIAGNOSTIC_TEST_CATALOG",
    "DOWNSTREAM_MODULE_CATALOG",
    "AnalyticalInsight",
    "AutomatedEDA",
    "ColumnProfile",
    "DataProfiler",
    "DatasetProfile",
    "EDAReport",
    "HypothesisTestResult",
    "InsightGenerator",
    "StatisticalAnalyzer",
    "StatisticalCriteriaConfig",
    "UniversalDiagnosticsReport",
    "UniversalDiagnosticsSuite",
    "VisualizationEngine",
]
