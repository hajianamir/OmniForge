"""Unified modeling package covering baselines, classical ML, ensembles, deep learning, forecasting, and specialized ML."""

from omniforge.models.anomaly import AnomalyDetector
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.baselines import (
    BaselineBenchmarkSuite,
    DecisionStumpClassifierBaseline,
    DecisionStumpRegressorBaseline,
    DummyClassifierBaseline,
    DummyRegressorBaseline,
    ForecastingBaseline,
    LassoLinearBaseline,
    LogisticBaseline,
    RidgeLinearBaseline,
)
from omniforge.models.causal import CausalEffectResult, CausalInferenceEngine
from omniforge.models.classical import ClassicalMLModel
from omniforge.models.clustering import ClusteringEngine, ClusteringResult
from omniforge.models.computer_vision import ComputerVisionEngine, VisionFoundationHub, probe_vision_backends
from omniforge.models.deep import (
    AutoencoderModel,
    CNN1DModel,
    FTTransformerModel,
    GraphAutoencoderLinkModel,
    GraphNeuralNetworkModel,
    MLPModel,
    MultimodalFusionModel,
    RecurrentSequenceModel,
    ResNetTabularModel,
    TabNetModel,
    TabularTransformerModel,
)
from omniforge.models.ensemble import StackingEnsembleModel, VotingEnsembleModel
from omniforge.models.graphs import GraphAnalyticsEngine, GraphLinkPredictionSuite
from omniforge.models.mining import AssociationRuleMiner, ManifoldProjector, SubgroupDiscoveryMiner
from omniforge.models.nlp import AdvancedNLPSuite, NLPEngine, probe_nlp_backends
from omniforge.models.nonparametric import (
    DistanceWeightedKNNModel,
    IsotonicMonotonicModel,
    NadarayaWatsonKernelModel,
    NonParametricAnalyticalSuite,
    PenalizedSplineGAMModel,
    RBFKernelRidgeModel,
)
from omniforge.models.optimization import OptimizationEngine, OptimizationResult
from omniforge.models.recommendation import HybridRecommender
from omniforge.models.registry import ModelCatalog, create_model, default_model_catalog
from omniforge.models.simulation import (
    AgentBasedSimulator,
    DigitalTwinEngine,
    DiscreteEventSimulator,
    MonteCarloResult,
    MonteCarloSimulator,
)
from omniforge.models.survival import CoxPHModel, KaplanMeierEstimator
from omniforge.models.time_series import (
    DeepForecaster,
    FoundationForecaster,
    HierarchicalReconciler,
    MLForecaster,
    StatisticalForecaster,
)

__all__ = [
    "AdvancedNLPSuite",
    "AgentBasedSimulator",
    "AnomalyDetector",
    "AssociationRuleMiner",
    "AutoencoderModel",
    "BaseOmniForgeModel",
    "BaselineBenchmarkSuite",
    "CNN1DModel",
    "CausalEffectResult",
    "CausalInferenceEngine",
    "ClassicalMLModel",
    "ClusteringEngine",
    "ClusteringResult",
    "ComputerVisionEngine",
    "CoxPHModel",
    "DecisionStumpClassifierBaseline",
    "DecisionStumpRegressorBaseline",
    "DeepForecaster",
    "DigitalTwinEngine",
    "DiscreteEventSimulator",
    "DistanceWeightedKNNModel",
    "DummyClassifierBaseline",
    "DummyRegressorBaseline",
    "FTTransformerModel",
    "ForecastingBaseline",
    "FoundationForecaster",
    "GraphAnalyticsEngine",
    "GraphAutoencoderLinkModel",
    "GraphLinkPredictionSuite",
    "GraphNeuralNetworkModel",
    "HierarchicalReconciler",
    "HybridRecommender",
    "IsotonicMonotonicModel",
    "KaplanMeierEstimator",
    "LassoLinearBaseline",
    "LogisticBaseline",
    "MLForecaster",
    "MLPModel",
    "ManifoldProjector",
    "ModelCatalog",
    "MonteCarloResult",
    "MonteCarloSimulator",
    "MultimodalFusionModel",
    "NLPEngine",
    "NadarayaWatsonKernelModel",
    "NonParametricAnalyticalSuite",
    "OptimizationEngine",
    "OptimizationResult",
    "PenalizedSplineGAMModel",
    "RBFKernelRidgeModel",
    "RecurrentSequenceModel",
    "ResNetTabularModel",
    "RidgeLinearBaseline",
    "StackingEnsembleModel",
    "StatisticalForecaster",
    "SubgroupDiscoveryMiner",
    "TabNetModel",
    "TabularTransformerModel",
    "VisionFoundationHub",
    "VotingEnsembleModel",
    "create_model",
    "default_model_catalog",
    "probe_nlp_backends",
    "probe_vision_backends",
]
