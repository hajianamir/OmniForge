"""OmniForge v4.0.0 — Enterprise Analytics & AI Workbench (OmniForge SDK).

Unified end-to-end Python SDK for multi-source data ingestion, automatic format auditing,
cleaning with deleted-row provenance, quality profiling, automated EDA, feature engineering,
baseline-first modeling, deep learning, forecasting, causal inference, mathematical optimization,
simulation, responsible AI, AutoML, DAG pipelines, MLOps, AI Copilot, 18 industry domain plugins,
and automated executive reporting.

Subpackages are loaded lazily via PEP 562 ``__getattr__`` so stdlib-only entrypoints (such as
``omniforge.capabilities`` used by ``installer/provision.py`` and ``scripts/build_locks.py``
before wheels are provisioned) import with zero third-party dependencies and minimal startup cost.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from omniforge.version import __edition__, __version__, get_version_info

if TYPE_CHECKING:
    from omniforge.automl import AutoMLPipelineSearch, HyperparameterOptimizer, NeuralArchitectureSearcher
    from omniforge.causal import CausalInferenceEngine
    from omniforge.config import OmniForgeSettings, get_settings, load_config, load_settings
    from omniforge.copilot import OmniForgeCopilotAgent
    from omniforge.core import (
        AccessController,
        AuditLogger,
        CarbonTracker,
        ComplianceValidator,
        EncryptionManager,
        JWTManager,
        ModelStage,
        OmniForgeError,
        Principal,
        TaskType,
        WorkspaceManifest,
        WorkspaceSessionManager,
        WorkspaceSnapshotSummary,
        get_logger,
        set_global_seed,
    )
    from omniforge.data import (
        DataCleaner,
        DataFormatAuditor,
        DataIngestionEngine,
        DataQualityProfiler,
        DatasetVersionManager,
        DataValidator,
        LocalFeatureStore,
        SyntheticDataGenerator,
    )
    from omniforge.domains import BaseDomainPlugin, default_domain_registry
    from omniforge.eda import AutomatedEDA
    from omniforge.evaluation import (
        BacktestingEngine,
        ConformalPredictor,
        CrossValidator,
        FairnessAuditor,
        InterpretabilityEngine,
        MetricsEvaluator,
        ModelCardGenerator,
        RobustnessTester,
    )
    from omniforge.features import FeatureEngineeringPipeline, FeatureSelector
    from omniforge.forecasting import ForecastingEngine
    from omniforge.mlops import (
        ContinuousTrainingOrchestrator,
        DriftDetector,
        ExperimentTracker,
        ModelMonitor,
        ModelRegistry,
        ModelServingEngine,
    )
    from omniforge.models import (
        BaselineBenchmarkSuite,
        BaseOmniForgeModel,
        ClassicalMLModel,
        LassoLinearBaseline,
        ModelCatalog,
        RidgeLinearBaseline,
        create_model,
    )
    from omniforge.optimization import OptimizationEngine
    from omniforge.pipelines import PipelineBuilder, PipelineDAG, PipelineExecutor
    from omniforge.reporting import ReportExporters, ReportGenerator
    from omniforge.simulation import (
        AgentBasedSimulator,
        DigitalTwinEngine,
        DiscreteEventSimulator,
        MonteCarloSimulator,
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AutoMLPipelineSearch": ("omniforge.automl", "AutoMLPipelineSearch"),
    "HyperparameterOptimizer": ("omniforge.automl", "HyperparameterOptimizer"),
    "NeuralArchitectureSearcher": ("omniforge.automl", "NeuralArchitectureSearcher"),
    "CausalInferenceEngine": ("omniforge.causal", "CausalInferenceEngine"),
    "OmniForgeSettings": ("omniforge.config", "OmniForgeSettings"),
    "get_settings": ("omniforge.config", "get_settings"),
    "load_config": ("omniforge.config", "load_config"),
    "load_settings": ("omniforge.config", "load_settings"),
    "OmniForgeCopilotAgent": ("omniforge.copilot", "OmniForgeCopilotAgent"),
    "AccessController": ("omniforge.core", "AccessController"),
    "AuditLogger": ("omniforge.core", "AuditLogger"),
    "CarbonTracker": ("omniforge.core", "CarbonTracker"),
    "ComplianceValidator": ("omniforge.core", "ComplianceValidator"),
    "EncryptionManager": ("omniforge.core", "EncryptionManager"),
    "JWTManager": ("omniforge.core", "JWTManager"),
    "ModelStage": ("omniforge.core", "ModelStage"),
    "OmniForgeError": ("omniforge.core", "OmniForgeError"),
    "Principal": ("omniforge.core", "Principal"),
    "TaskType": ("omniforge.core", "TaskType"),
    "WorkspaceManifest": ("omniforge.core", "WorkspaceManifest"),
    "WorkspaceSessionManager": ("omniforge.core", "WorkspaceSessionManager"),
    "WorkspaceSnapshotSummary": ("omniforge.core", "WorkspaceSnapshotSummary"),
    "get_logger": ("omniforge.core", "get_logger"),
    "set_global_seed": ("omniforge.core", "set_global_seed"),
    "DataCleaner": ("omniforge.data", "DataCleaner"),
    "DataFormatAuditor": ("omniforge.data", "DataFormatAuditor"),
    "DataIngestionEngine": ("omniforge.data", "DataIngestionEngine"),
    "DataQualityProfiler": ("omniforge.data", "DataQualityProfiler"),
    "DatasetVersionManager": ("omniforge.data", "DatasetVersionManager"),
    "DataValidator": ("omniforge.data", "DataValidator"),
    "LocalFeatureStore": ("omniforge.data", "LocalFeatureStore"),
    "SyntheticDataGenerator": ("omniforge.data", "SyntheticDataGenerator"),
    "BaseDomainPlugin": ("omniforge.domains", "BaseDomainPlugin"),
    "default_domain_registry": ("omniforge.domains", "default_domain_registry"),
    "AutomatedEDA": ("omniforge.eda", "AutomatedEDA"),
    "BacktestingEngine": ("omniforge.evaluation", "BacktestingEngine"),
    "ConformalPredictor": ("omniforge.evaluation", "ConformalPredictor"),
    "CrossValidator": ("omniforge.evaluation", "CrossValidator"),
    "FairnessAuditor": ("omniforge.evaluation", "FairnessAuditor"),
    "InterpretabilityEngine": ("omniforge.evaluation", "InterpretabilityEngine"),
    "MetricsEvaluator": ("omniforge.evaluation", "MetricsEvaluator"),
    "ModelCardGenerator": ("omniforge.evaluation", "ModelCardGenerator"),
    "RobustnessTester": ("omniforge.evaluation", "RobustnessTester"),
    "FeatureEngineeringPipeline": ("omniforge.features", "FeatureEngineeringPipeline"),
    "FeatureSelector": ("omniforge.features", "FeatureSelector"),
    "ForecastingEngine": ("omniforge.forecasting", "ForecastingEngine"),
    "ContinuousTrainingOrchestrator": ("omniforge.mlops", "ContinuousTrainingOrchestrator"),
    "DriftDetector": ("omniforge.mlops", "DriftDetector"),
    "ExperimentTracker": ("omniforge.mlops", "ExperimentTracker"),
    "ModelMonitor": ("omniforge.mlops", "ModelMonitor"),
    "ModelRegistry": ("omniforge.mlops", "ModelRegistry"),
    "ModelServingEngine": ("omniforge.mlops", "ModelServingEngine"),
    "BaselineBenchmarkSuite": ("omniforge.models", "BaselineBenchmarkSuite"),
    "BaseOmniForgeModel": ("omniforge.models", "BaseOmniForgeModel"),
    "ClassicalMLModel": ("omniforge.models", "ClassicalMLModel"),
    "LassoLinearBaseline": ("omniforge.models", "LassoLinearBaseline"),
    "ModelCatalog": ("omniforge.models", "ModelCatalog"),
    "RidgeLinearBaseline": ("omniforge.models", "RidgeLinearBaseline"),
    "create_model": ("omniforge.models", "create_model"),
    "OptimizationEngine": ("omniforge.optimization", "OptimizationEngine"),
    "PipelineBuilder": ("omniforge.pipelines", "PipelineBuilder"),
    "PipelineDAG": ("omniforge.pipelines", "PipelineDAG"),
    "PipelineExecutor": ("omniforge.pipelines", "PipelineExecutor"),
    "ReportExporters": ("omniforge.reporting", "ReportExporters"),
    "ReportGenerator": ("omniforge.reporting", "ReportGenerator"),
    "AgentBasedSimulator": ("omniforge.simulation", "AgentBasedSimulator"),
    "DigitalTwinEngine": ("omniforge.simulation", "DigitalTwinEngine"),
    "DiscreteEventSimulator": ("omniforge.simulation", "DiscreteEventSimulator"),
    "MonteCarloSimulator": ("omniforge.simulation", "MonteCarloSimulator"),
}

__all__ = [
    "AccessController",
    "AgentBasedSimulator",
    "AuditLogger",
    "AutoMLPipelineSearch",
    "AutomatedEDA",
    "BacktestingEngine",
    "BaseDomainPlugin",
    "BaseOmniForgeModel",
    "BaselineBenchmarkSuite",
    "CarbonTracker",
    "CausalInferenceEngine",
    "ClassicalMLModel",
    "ComplianceValidator",
    "ConformalPredictor",
    "ContinuousTrainingOrchestrator",
    "CrossValidator",
    "DataCleaner",
    "DataFormatAuditor",
    "DataIngestionEngine",
    "DataQualityProfiler",
    "DataValidator",
    "DatasetVersionManager",
    "DigitalTwinEngine",
    "DiscreteEventSimulator",
    "DriftDetector",
    "EncryptionManager",
    "ExperimentTracker",
    "FairnessAuditor",
    "FeatureEngineeringPipeline",
    "FeatureSelector",
    "ForecastingEngine",
    "HyperparameterOptimizer",
    "InterpretabilityEngine",
    "JWTManager",
    "LassoLinearBaseline",
    "LocalFeatureStore",
    "MetricsEvaluator",
    "ModelCardGenerator",
    "ModelCatalog",
    "ModelMonitor",
    "ModelRegistry",
    "ModelServingEngine",
    "ModelStage",
    "MonteCarloSimulator",
    "NeuralArchitectureSearcher",
    "OptimizationEngine",
    "PipelineBuilder",
    "PipelineDAG",
    "PipelineExecutor",
    "Principal",
    "ReportExporters",
    "ReportGenerator",
    "RidgeLinearBaseline",
    "RobustnessTester",
    "SyntheticDataGenerator",
    "TaskType",
    "OmniForgeCopilotAgent",
    "OmniForgeError",
    "OmniForgeSettings",
    "WorkspaceManifest",
    "WorkspaceSessionManager",
    "WorkspaceSnapshotSummary",
    "__edition__",
    "__version__",
    "create_model",
    "default_domain_registry",
    "get_logger",
    "get_settings",
    "get_version_info",
    "load_config",
    "load_settings",
    "set_global_seed",
]


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
