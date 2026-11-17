"""Adversarial Robustness Stress Testing and Distribution-Free Split Conformal Prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from omniforge.core.exceptions import EvaluationError
from omniforge.core.types import TaskType
from omniforge.models.base import BaseOmniForgeModel


class ConformalPredictor:
    """Split Conformal Prediction providing finite-sample coverage guarantees for regression & classification."""

    def __init__(self, model: BaseOmniForgeModel, alpha: float = 0.1) -> None:
        self.model = model
        self.alpha = float(np.clip(alpha, 0.001, 0.5))
        self.quantile_: float = 0.0
        self.is_calibrated_: bool = False

    def calibrate(self, X_cal: pd.DataFrame | np.ndarray, y_cal: pd.Series | np.ndarray) -> ConformalPredictor:
        """Compute finite-sample conformal nonconformity quantile on a calibration split."""
        y_arr = np.asarray(y_cal)
        n = len(y_arr)
        if n < 5:
            raise EvaluationError("ConformalPredictor requires at least 5 calibration samples.")

        if self.model.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            proba = self.model.predict_proba(X_cal)
            classes = list(self.model.classes_) if self.model.classes_ is not None else list(range(proba.shape[1]))
            cls_to_idx = {c: i for i, c in enumerate(classes)}
            scores = np.array([1.0 - proba[i, cls_to_idx[y_arr[i]]] for i in range(n)], dtype=float)
        else:
            preds = self.model.predict(X_cal).astype(float)
            scores = np.abs(y_arr.astype(float) - preds)

        q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - self.alpha)) / n))
        self.quantile_ = float(np.quantile(scores, q_level, method="higher"))
        self.is_calibrated_ = True
        return self

    def predict_intervals(self, X: pd.DataFrame | np.ndarray) -> dict[str, np.ndarray]:
        """Return distribution-free prediction intervals ``[lower, upper]`` for regression."""
        if not self.is_calibrated_:
            raise EvaluationError("Call calibrate() before predict_intervals().")
        preds = self.model.predict(X).astype(float)
        return {
            "prediction": preds,
            "lower": preds - self.quantile_,
            "upper": preds + self.quantile_,
            "interval_width": np.full(len(preds), 2.0 * self.quantile_),
        }

    def predict_sets(self, X: pd.DataFrame | np.ndarray) -> list[list[Any]]:
        """Return conformal prediction label sets for classification."""
        if not self.is_calibrated_:
            raise EvaluationError("Call calibrate() before predict_sets().")
        proba = self.model.predict_proba(X)
        classes = list(self.model.classes_) if self.model.classes_ is not None else list(range(proba.shape[1]))
        sets: list[list[Any]] = []
        for row in proba:
            included = [classes[i] for i, p in enumerate(row) if (1.0 - p) <= self.quantile_]
            if not included:
                included = [classes[int(np.argmax(row))]]
            sets.append(included)
        return sets


class RobustnessTester:
    """Stress-tests model stability under Gaussian noise, feature dropout, and distribution shifts."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def evaluate(
        self,
        model: BaseOmniForgeModel,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        noise_levels: tuple[float, ...] = (0.05, 0.15, 0.30),
        dropout_rates: tuple[float, ...] = (0.05, 0.15),
    ) -> dict[str, Any]:
        """Measure score degradation under adversarial perturbations, missingness, and OOD shifts."""
        rng = np.random.default_rng(self.random_seed)
        base_score = model.score(X, y)
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        stds = X[num_cols].std().replace(0.0, 1.0).fillna(1.0)

        noise_results: dict[str, float] = {}
        for sigma in noise_levels:
            X_noisy = X.copy()
            for col in num_cols:
                X_noisy[col] = X_noisy[col].astype(float) + rng.normal(0.0, sigma * float(stds[col]), size=len(X))
            noise_results[f"gaussian_sigma_{sigma}"] = round(model.score(X_noisy, y), 6)

        dropout_results: dict[str, float] = {}
        for rate in dropout_rates:
            X_drop = X.copy()
            for col in num_cols:
                mask = rng.uniform(0.0, 1.0, size=len(X)) < rate
                X_drop.loc[mask, col] = 0.0
            dropout_results[f"feature_zeroout_{rate}"] = round(model.score(X_drop, y), 6)

        # Covariate shift (+1.0 std drift across top 3 numeric features)
        X_shift = X.copy()
        for col in num_cols[:3]:
            X_shift[col] = X_shift[col].astype(float) + float(stds[col])
        shift_score = round(model.score(X_shift, y), 6)

        all_stressed = [*noise_results.values(), *dropout_results.values(), shift_score]
        worst_drop = max(0.0, base_score - min(all_stressed)) if all_stressed else 0.0
        robustness_index = round(max(0.0, 100.0 * (1.0 - worst_drop / max(abs(base_score), 1e-6))), 2)

        return {
            "clean_score": round(base_score, 6),
            "noise_stress": noise_results,
            "missingness_stress": dropout_results,
            "ood_shift_score": shift_score,
            "worst_score_drop": round(worst_drop, 6),
            "robustness_index_0_100": robustness_index,
        }
