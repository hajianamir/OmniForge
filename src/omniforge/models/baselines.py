"""Strong baseline models for classification, regression, and time-series forecasting."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from omniforge.core.types import TaskType
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.nonparametric import (
    DistanceWeightedKNNModel,
    NadarayaWatsonKernelModel,
    PenalizedSplineGAMModel,
)


class DummyClassifierBaseline(BaseOmniForgeModel):
    """Classification baseline supporting most_frequent, stratified, prior, and uniform strategies."""

    model_family = "baseline_classification"

    def __init__(self, strategy: str = "most_frequent", **kwargs: Any) -> None:
        super().__init__(
            name=f"Baseline_{strategy}",
            task_type=TaskType.BINARY_CLASSIFICATION,
            strategy=strategy,
            **kwargs,
        )
        self.strategy = strategy
        self._model = DummyClassifier(strategy=strategy, random_state=self.random_seed)

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        self._model.fit(X, y)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.asarray(self._model.predict(X))

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.asarray(self._model.predict_proba(X), dtype=float)


class LogisticBaseline(BaseOmniForgeModel):
    """Regularized standardized logistic regression strong baseline."""

    model_family = "baseline_classification"

    def __init__(self, C: float = 1.0, **kwargs: Any) -> None:
        super().__init__(
            name="Baseline_LogisticRegression",
            task_type=TaskType.BINARY_CLASSIFICATION,
            C=C,
            **kwargs,
        )
        self._scaler = StandardScaler()
        try:
            self._model = LogisticRegression(
                C=C,
                l1_ratio=1.0,
                solver="saga",
                max_iter=2000,
                random_state=self.random_seed,
            )
        except TypeError:
            self._model = LogisticRegression(
                C=C,
                penalty="l1",
                solver="liblinear",
                max_iter=2000,
                random_state=self.random_seed,
            )

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        self._model.fit(self._scaler.fit_transform(X), y)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.asarray(self._model.predict(self._scaler.transform(X)))

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.asarray(self._model.predict_proba(self._scaler.transform(X)), dtype=float)


class DecisionStumpClassifierBaseline(BaseOmniForgeModel):
    """Single-split decision stump classification baseline."""

    model_family = "baseline_classification"

    def __init__(self, max_depth: int = 2, **kwargs: Any) -> None:
        super().__init__(
            name="Baseline_DecisionStumpClassifier",
            task_type=TaskType.BINARY_CLASSIFICATION,
            max_depth=max_depth,
            **kwargs,
        )
        self._model = DecisionTreeClassifier(max_depth=max_depth, random_state=self.random_seed)

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        self._model.fit(X, y)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.asarray(self._model.predict(X))

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.asarray(self._model.predict_proba(X), dtype=float)


class DummyRegressorBaseline(BaseOmniForgeModel):
    """Regression baseline predicting constant mean or median target value."""

    model_family = "baseline_regression"

    def __init__(self, strategy: str = "mean", **kwargs: Any) -> None:
        super().__init__(
            name=f"Baseline_{strategy}",
            task_type=TaskType.REGRESSION,
            strategy=strategy,
            **kwargs,
        )
        self.strategy = strategy
        self._model = DummyRegressor(strategy=strategy)

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        self._model.fit(X, y)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.asarray(self._model.predict(X), dtype=float)


class RidgeLinearBaseline(BaseOmniForgeModel):
    """Standardized L2 Tikhonov / Hoerl-Kennard Ridge linear regression strong baseline.

    Standardizes features via :class:`~sklearn.preprocessing.StandardScaler` and minimizes the
    scikit-learn L2-penalized residual sum of squares (Hoerl & Kennard, 1970, *Technometrics*
    12(1):55-67):

    .. math::
        \\min_{w, w_0} \\|\\tilde{X} w + w_0 \\mathbf{1} - y\\|_2^2 + \\alpha \\|w\\|_2^2, \\quad \\alpha \\ge 0
    """

    model_family = "baseline_regression"

    def __init__(self, alpha: float = 1.0, **kwargs: Any) -> None:
        if isinstance(alpha, bool) or not math.isfinite(float(alpha)) or float(alpha) < 0.0:
            raise ValueError("RidgeLinearBaseline requires a finite regularization parameter alpha >= 0.0.")
        self.alpha = float(alpha)
        super().__init__(
            name="Baseline_RidgeLinear",
            task_type=TaskType.REGRESSION,
            alpha=self.alpha,
            **kwargs,
        )
        self._scaler = StandardScaler()
        self._model = Ridge(alpha=self.alpha, random_state=self.random_seed)

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        self._model.fit(self._scaler.fit_transform(X), y)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.asarray(self._model.predict(self._scaler.transform(X)), dtype=float)


class LassoLinearBaseline(BaseOmniForgeModel):
    """Standardized L1 sparse Lasso linear regression strong baseline.

    Standardizes features via :class:`~sklearn.preprocessing.StandardScaler` and minimizes the
    L1-penalized least-squares objective (Tibshirani, 1996, *JRSS-B* 58(1):267-288):

    .. math::
        \\min_{w, w_0} \\frac{1}{2 n_{\\text{samples}}} \\|\\tilde{X} w + w_0 \\mathbf{1} - y\\|_2^2 + \\alpha \\|w\\|_1

    Uses cross-validated :class:`~sklearn.linear_model.LassoCV` when ``alpha is None`` (with safe
    small-sample fallback) or fixed-regularization :class:`~sklearn.linear_model.Lasso` when
    ``alpha`` is provided explicitly.
    """

    model_family = "baseline_regression"

    def __init__(
        self,
        alpha: float | None = None,
        cv: int = 3,
        max_iter: int = 2000,
        **kwargs: Any,
    ) -> None:
        if alpha is not None and (isinstance(alpha, bool) or not math.isfinite(float(alpha)) or float(alpha) <= 0.0):
            raise ValueError("LassoLinearBaseline requires alpha > 0.0 when explicitly specified.")
        if isinstance(cv, bool) or int(cv) < 2:
            raise ValueError("LassoLinearBaseline requires cv >= 2.")
        self.alpha = float(alpha) if alpha is not None else None
        self.cv = int(cv)
        self.max_iter = max(100, int(max_iter))
        super().__init__(
            name="Baseline_LassoLinear",
            task_type=TaskType.REGRESSION,
            alpha=self.alpha,
            cv=self.cv,
            max_iter=self.max_iter,
            **kwargs,
        )
        self._scaler = StandardScaler()
        self._model: Lasso | LassoCV = (
            Lasso(alpha=self.alpha, max_iter=self.max_iter, random_state=self.random_seed)
            if self.alpha is not None
            else LassoCV(cv=self.cv, max_iter=self.max_iter, random_state=self.random_seed)
        )

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        X_scaled = self._scaler.fit_transform(X)
        if self.alpha is not None or len(X_scaled) < 3:
            eff_alpha = self.alpha if self.alpha is not None else 0.1
            self._model = Lasso(alpha=eff_alpha, max_iter=self.max_iter, random_state=self.random_seed)
        else:
            eff_cv = max(2, min(self.cv, len(X_scaled)))
            self._model = LassoCV(cv=eff_cv, max_iter=self.max_iter, random_state=self.random_seed)
        self._model.fit(X_scaled, y)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.asarray(self._model.predict(self._scaler.transform(X)), dtype=float)


class DecisionStumpRegressorBaseline(BaseOmniForgeModel):
    """Shallow decision stump regression baseline."""

    model_family = "baseline_regression"

    def __init__(self, max_depth: int = 2, **kwargs: Any) -> None:
        super().__init__(
            name="Baseline_DecisionStumpRegressor",
            task_type=TaskType.REGRESSION,
            max_depth=max_depth,
            **kwargs,
        )
        self._model = DecisionTreeRegressor(max_depth=max_depth, random_state=self.random_seed)

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        self._model.fit(X, y)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.asarray(self._model.predict(X), dtype=float)


class ForecastingBaseline:
    """Classical time-series forecasting baselines: naive, seasonal_naive, drift, moving_average."""

    def __init__(self, method: str = "seasonal_naive", seasonal_period: int = 7, window: int = 7) -> None:
        self.method = method
        self.seasonal_period = max(1, seasonal_period)
        self.window = max(1, window)
        self._history: np.ndarray = np.array([], dtype=float)
        self._residual_std: float = 0.0

    def fit(self, y: pd.Series | np.ndarray) -> ForecastingBaseline:
        """Fit baseline on historical 1D series."""
        self._history = np.asarray(y, dtype=float).ravel()
        if len(self._history) > 1:
            diffs = np.diff(self._history)
            self._residual_std = float(np.std(diffs))
        return self

    def predict(self, horizon: int, return_intervals: bool = False, z: float = 1.96) -> np.ndarray | dict[str, np.ndarray]:
        """Forecast ``horizon`` steps ahead with optional conformal-like Gaussian intervals."""
        if len(self._history) == 0:
            fc = np.zeros(horizon, dtype=float)
        elif self.method == "naive":
            fc = np.full(horizon, self._history[-1], dtype=float)
        elif self.method == "seasonal_naive":
            p = min(self.seasonal_period, len(self._history))
            season = self._history[-p:]
            fc = np.array([season[i % p] for i in range(horizon)], dtype=float)
        elif self.method == "drift":
            if len(self._history) == 1:
                slope = 0.0
            else:
                slope = (self._history[-1] - self._history[0]) / max(len(self._history) - 1, 1)
            fc = np.array([self._history[-1] + slope * (h + 1) for h in range(horizon)], dtype=float)
        elif self.method == "moving_average":
            w = min(self.window, len(self._history))
            fc = np.full(horizon, float(np.mean(self._history[-w:])), dtype=float)
        else:
            fc = np.full(horizon, float(np.mean(self._history)), dtype=float)

        if not return_intervals:
            return fc
        steps = np.sqrt(np.arange(1, horizon + 1, dtype=float))
        half_width = z * self._residual_std * steps
        return {"yhat": fc, "yhat_lower": fc - half_width, "yhat_upper": fc + half_width}


class BaselineBenchmarkSuite:
    """Automatically evaluates strong baseline models so candidate ML models always prove lift."""

    @staticmethod
    def run_classification_detailed(
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        random_seed: int = 42,
    ) -> tuple[dict[str, dict[str, float]], dict[str, BaseOmniForgeModel]]:
        """Run all classification baselines and return both comparative metrics and fitted baseline models."""
        baselines: list[BaseOmniForgeModel] = [
            DummyClassifierBaseline(strategy="most_frequent", random_seed=random_seed),
            DummyClassifierBaseline(strategy="stratified", random_seed=random_seed),
            DecisionStumpClassifierBaseline(max_depth=2, random_seed=random_seed),
            LogisticBaseline(random_seed=random_seed),
            PenalizedSplineGAMModel(task_type=TaskType.BINARY_CLASSIFICATION, random_seed=random_seed),
            NadarayaWatsonKernelModel(task_type=TaskType.BINARY_CLASSIFICATION, random_seed=random_seed),
            DistanceWeightedKNNModel(task_type=TaskType.BINARY_CLASSIFICATION, random_seed=random_seed),
        ]
        results: dict[str, dict[str, float]] = {}
        fitted_models: dict[str, BaseOmniForgeModel] = {}
        y_true = np.asarray(y_val)
        for model in baselines:
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                results[model.name] = {
                    "accuracy": round(float(accuracy_score(y_true, preds)), 6),
                    "f1_weighted": round(float(f1_score(y_true, preds, average="weighted", zero_division=0)), 6),
                }
                fitted_models[model.name] = model
            except Exception:
                continue
        return results, fitted_models

    @staticmethod
    def run_classification(
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        random_seed: int = 42,
    ) -> dict[str, dict[str, float]]:
        """Run all classification baselines and return comparative metrics."""
        results, _ = BaselineBenchmarkSuite.run_classification_detailed(
            X_train, y_train, X_val, y_val, random_seed=random_seed
        )
        return results

    @staticmethod
    def run_regression_detailed(
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        random_seed: int = 42,
    ) -> tuple[dict[str, dict[str, float]], dict[str, BaseOmniForgeModel]]:
        """Run all regression baselines and return both comparative metrics and fitted baseline models."""
        baselines: list[BaseOmniForgeModel] = [
            DummyRegressorBaseline(strategy="mean", random_seed=random_seed),
            DummyRegressorBaseline(strategy="median", random_seed=random_seed),
            DecisionStumpRegressorBaseline(max_depth=2, random_seed=random_seed),
            RidgeLinearBaseline(random_seed=random_seed),
            LassoLinearBaseline(random_seed=random_seed),
            PenalizedSplineGAMModel(task_type=TaskType.REGRESSION, random_seed=random_seed),
            NadarayaWatsonKernelModel(task_type=TaskType.REGRESSION, random_seed=random_seed),
            DistanceWeightedKNNModel(task_type=TaskType.REGRESSION, random_seed=random_seed),
        ]
        results: dict[str, dict[str, float]] = {}
        fitted_models: dict[str, BaseOmniForgeModel] = {}
        y_true = np.asarray(y_val, dtype=float)
        for model in baselines:
            try:
                model.fit(X_train, y_train)
                preds = np.asarray(model.predict(X_val), dtype=float)
                ss_res = float(np.sum((y_true - preds) ** 2))
                ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
                r2_val = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
                results[model.name] = {
                    "r2": round(float(r2_val), 6),
                    "rmse": round(float(root_mean_squared_error(y_true, preds)), 6),
                    "mae": round(float(mean_absolute_error(y_true, preds)), 6),
                }
                fitted_models[model.name] = model
            except Exception:
                continue
        return results, fitted_models

    @staticmethod
    def run_regression(
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: pd.Series | np.ndarray,
        random_seed: int = 42,
    ) -> dict[str, dict[str, float]]:
        """Run all regression baselines and return comparative metrics."""
        results, _ = BaselineBenchmarkSuite.run_regression_detailed(
            X_train, y_train, X_val, y_val, random_seed=random_seed
        )
        return results

    @staticmethod
    def run_forecasting_detailed(
        y_train: np.ndarray | pd.Series,
        y_val: np.ndarray | pd.Series,
        seasonal_period: int = 7,
        *,
        gap: int = 0,
    ) -> tuple[dict[str, dict[str, float]], dict[str, ForecastingBaseline]]:
        """Run all forecasting baselines and return both metrics and fitted baseline models."""
        y_true = np.asarray(y_val, dtype=float).ravel()
        horizon = len(y_true)
        observed = np.isfinite(y_true)
        if not observed.any():
            return {}, {}
        y_observed = y_true[observed]
        gap = max(0, gap)
        results: dict[str, dict[str, float]] = {}
        fitted_models: dict[str, ForecastingBaseline] = {}
        for method in ("naive", "seasonal_naive", "drift", "moving_average"):
            bl = ForecastingBaseline(method=method, seasonal_period=seasonal_period).fit(y_train)
            preds = np.asarray(bl.predict(horizon + gap), dtype=float)[gap:][observed]
            denom = (np.abs(y_observed) + np.abs(preds)) / 2.0 + 1e-9
            smape = float(np.mean(np.abs(y_observed - preds) / denom) * 100.0)
            key = f"Baseline_{method}"
            results[key] = {
                "rmse": round(float(root_mean_squared_error(y_observed, preds)), 6),
                "mae": round(float(mean_absolute_error(y_observed, preds)), 6),
                "smape": round(smape, 4),
            }
            fitted_models[key] = bl
        return results, fitted_models

    @staticmethod
    def run_forecasting(
        y_train: np.ndarray | pd.Series,
        y_val: np.ndarray | pd.Series,
        seasonal_period: int = 7,
        *,
        gap: int = 0,
    ) -> dict[str, dict[str, float]]:
        """Run all forecasting baselines and return RMSE/MAE/sMAPE scores."""
        results, _ = BaselineBenchmarkSuite.run_forecasting_detailed(
            y_train, y_val, seasonal_period=seasonal_period, gap=gap
        )
        return results
