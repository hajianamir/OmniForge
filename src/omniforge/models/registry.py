"""Central Model Catalog and factory for instantiating any OmniForge model by canonical name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from omniforge.core.exceptions import ModelError
from omniforge.core.types import TaskType
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.baselines import (
    DecisionStumpClassifierBaseline,
    DecisionStumpRegressorBaseline,
    DummyClassifierBaseline,
    DummyRegressorBaseline,
    LogisticBaseline,
    RidgeLinearBaseline,
)
from omniforge.models.classical import ClassicalMLModel
from omniforge.models.ensemble import StackingEnsembleModel, VotingEnsembleModel


class ModelCatalog:
    """Extensible factory mapping model identifiers to constructors across all OmniForge model families."""

    def __init__(self) -> None:
        self._builders: dict[str, Callable[..., Any]] = {}
        self._register_defaults()

    def register(self, key: str, builder: Callable[..., Any]) -> None:
        """Register a model constructor under ``key``."""
        self._builders[key.lower()] = builder

    def list_models(self) -> list[str]:
        """Return sorted names of all registered model identifiers."""
        return sorted(self._builders.keys())

    def create(self, model_name: str, **kwargs: Any) -> Any:
        """Instantiate a registered model by name."""
        key = model_name.lower()
        if key not in self._builders:
            raise ModelError(
                f"Unknown model identifier '{model_name}'.",
                suggestions=[f"Available models: {', '.join(self.list_models()[:20])}"],
            )
        return self._builders[key](**kwargs)

    def _register_defaults(self) -> None:
        # Baselines
        self.register("baseline_most_frequent", lambda **kw: DummyClassifierBaseline(strategy="most_frequent", **kw))
        self.register("baseline_stratified", lambda **kw: DummyClassifierBaseline(strategy="stratified", **kw))
        self.register("baseline_logistic", lambda **kw: LogisticBaseline(**kw))
        self.register("baseline_stump_clf", lambda **kw: DecisionStumpClassifierBaseline(**kw))
        self.register("baseline_mean", lambda **kw: DummyRegressorBaseline(strategy="mean", **kw))
        self.register("baseline_median", lambda **kw: DummyRegressorBaseline(strategy="median", **kw))
        self.register("baseline_ridge", lambda **kw: RidgeLinearBaseline(**kw))
        self.register("baseline_stump_reg", lambda **kw: DecisionStumpRegressorBaseline(**kw))

        # Classical ML
        for alg in (
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "xgboost",
            "lightgbm",
            "catboost",
            "logistic_regression",
            "ridge",
            "lasso",
            "elasticnet",
            "svm",
            "knn",
            "naive_bayes",
        ):
            self.register(alg, lambda _a=alg, **kw: ClassicalMLModel(algorithm=_a, **kw))

        # Ensembles
        self.register("voting_ensemble", lambda **kw: VotingEnsembleModel(**kw))
        self.register("stacking_ensemble", lambda **kw: StackingEnsembleModel(**kw))

        # Lazy bindings for deep learning models
        def _lazy_deep(mod_name: str, cls_name: str) -> Callable[..., Any]:
            def _factory(**kw: Any) -> Any:
                import importlib

                mod = importlib.import_module(f"omniforge.models.deep.{mod_name}")
                return getattr(mod, cls_name)(**kw)

            return _factory

        self.register("mlp", _lazy_deep("mlp", "MLPModel"))
        self.register("cnn", _lazy_deep("cnn", "CNN1DModel"))
        self.register("rnn", _lazy_deep("rnn_lstm", "RecurrentSequenceModel"))
        self.register("lstm", lambda **kw: _lazy_deep("rnn_lstm", "RecurrentSequenceModel")(cell_type="lstm", **kw))
        self.register("gru", lambda **kw: _lazy_deep("rnn_lstm", "RecurrentSequenceModel")(cell_type="gru", **kw))
        self.register("transformer", _lazy_deep("transformers", "TabularTransformerModel"))
        self.register("gnn", _lazy_deep("gnn", "GraphNeuralNetworkModel"))
        self.register("autoencoder", _lazy_deep("autoencoders", "AutoencoderModel"))
        self.register("multimodal_net", _lazy_deep("multimodal", "MultimodalFusionModel"))


default_model_catalog = ModelCatalog()


def create_model(
    model_name: str,
    task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
    **kwargs: Any,
) -> BaseOmniForgeModel:
    """Convenience factory returning a configured OmniForge model."""
    return default_model_catalog.create(model_name, task_type=task_type, **kwargs)
