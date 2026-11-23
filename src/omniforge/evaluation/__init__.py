"""Evaluation & Responsible AI package: metrics, CV, interpretability, fairness, robustness, conformal, backtesting, cards."""

from omniforge.evaluation.backtesting import BacktestingEngine
from omniforge.evaluation.equations import ModelEquationExtractor, ModelEquationReport
from omniforge.evaluation.fairness import FairnessAuditor, FairnessReport
from omniforge.evaluation.interpretability import InterpretabilityEngine
from omniforge.evaluation.metrics import MetricsEvaluator, ProbabilityCalibrator
from omniforge.evaluation.model_cards import DataCard, ModelCard, ModelCardGenerator
from omniforge.evaluation.robustness import ConformalPredictor, RobustnessTester
from omniforge.evaluation.validation import CombinatorialPurgedKFold, CrossValidator, PurgedTimeSeriesSplit

__all__ = [
    "BacktestingEngine",
    "CombinatorialPurgedKFold",
    "ConformalPredictor",
    "CrossValidator",
    "DataCard",
    "FairnessAuditor",
    "FairnessReport",
    "InterpretabilityEngine",
    "MetricsEvaluator",
    "ModelCard",
    "ModelCardGenerator",
    "ModelEquationExtractor",
    "ModelEquationReport",
    "ProbabilityCalibrator",
    "PurgedTimeSeriesSplit",
    "RobustnessTester",
]
