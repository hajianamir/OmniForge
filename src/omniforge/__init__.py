"""OmniForge v2.0.0 — Enterprise Analytics & AI Workbench (OmniForge SDK).

Unified end-to-end Python SDK for multi-source data ingestion, automatic format auditing,
cleaning with deleted-row provenance, quality profiling, automated EDA, feature engineering,
baseline-first modeling, deep learning, forecasting, causal inference, mathematical optimization,
simulation, responsible AI, AutoML, DAG pipelines, MLOps, AI Copilot, 18 industry domain plugins,
and automated executive reporting.
"""

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
from omniforge.models import BaselineBenchmarkSuite, BaseOmniForgeModel, ClassicalMLModel, ModelCatalog, create_model
from omniforge.optimization import OptimizationEngine
from omniforge.pipelines import PipelineBuilder, PipelineDAG, PipelineExecutor
from omniforge.reporting import ReportExporters, ReportGenerator
from omniforge.simulation import (
    AgentBasedSimulator,
    DigitalTwinEngine,
    DiscreteEventSimulator,
    MonteCarloSimulator,
)
from omniforge.version import __edition__, __version__, get_version_info

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
