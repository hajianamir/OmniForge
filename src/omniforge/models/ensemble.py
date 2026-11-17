"""Enterprise ensemble models: Stacking, Soft/Hard Voting, Blending, and Weighted Averaging."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from omniforge.core.types import TaskType
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.classical import ClassicalMLModel


class VotingEnsembleModel(BaseOmniForgeModel):
    """Weighted soft-voting classification or weighted-average regression ensemble."""

    model_family = "ensemble"

    def __init__(
        self,
        estimators: list[BaseOmniForgeModel] | None = None,
        *,
        weights: list[float] | None = None,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="VotingEnsemble", task_type=task_type, **kwargs)
        self.estimators = estimators or [
            ClassicalMLModel("random_forest", task_type=task_type, random_seed=self.random_seed),
            ClassicalMLModel("extra_trees", task_type=task_type, random_seed=self.random_seed + 1),
            ClassicalMLModel("gradient_boosting", task_type=task_type, random_seed=self.random_seed + 2),
        ]
        self.weights = weights

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        for est in self.estimators:
            est.task_type = self.task_type
            est.fit(X, y)
        if self.weights is None:
            self._norm_weights = np.full(len(self.estimators), 1.0 / len(self.estimators), dtype=float)
        else:
            w = np.asarray(self.weights, dtype=float)
            self._norm_weights = w / max(float(w.sum()), 1e-12)

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        probas = [est.predict_proba(X) for est in self.estimators]
        stacked = np.zeros_like(probas[0], dtype=float)
        for p, w in zip(probas, self._norm_weights, strict=False):
            stacked += w * p
        return stacked

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            proba = self._predict_proba(X)
            idx = np.argmax(proba, axis=1)
            classes = self.classes_ if self.classes_ is not None else np.arange(proba.shape[1])
            return np.asarray([classes[i] for i in idx])
        preds = [est.predict(X).astype(float) for est in self.estimators]
        return np.sum([w * p for w, p in zip(self._norm_weights, preds, strict=False)], axis=0)


class StackingEnsembleModel(BaseOmniForgeModel):
    """Out-of-fold Stacking ensemble with regularized linear/logistic meta-learner."""

    model_family = "ensemble"

    def __init__(
        self,
        estimators: list[BaseOmniForgeModel] | None = None,
        *,
        cv_folds: int = 3,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="StackingEnsemble", task_type=task_type, cv_folds=cv_folds, **kwargs)
        self.cv_folds = max(2, cv_folds)
        self.estimators = estimators or [
            ClassicalMLModel("random_forest", task_type=task_type, random_seed=self.random_seed),
            ClassicalMLModel("gradient_boosting", task_type=task_type, random_seed=self.random_seed + 1),
            ClassicalMLModel(
                "logistic_regression"
                if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
                else "ridge",
                task_type=task_type,
                random_seed=self.random_seed + 2,
            ),
        ]
        self._meta_learner: Any = None
        self._fitted_estimators: list[BaseOmniForgeModel] = []

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        is_clf = self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        n_samples = X.shape[0]
        n_folds = min(self.cv_folds, max(2, n_samples // 4))
        splitter = (
            StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
            if is_clf
            else KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        )

        oof_blocks: list[np.ndarray] = []
        self._fitted_estimators = []

        for base_est in self.estimators:
            base_est.task_type = self.task_type
            if is_clf:
                n_cls = len(np.unique(y))
                oof = np.zeros((n_samples, n_cls), dtype=float)
            else:
                oof = np.zeros((n_samples, 1), dtype=float)

            for tr_idx, va_idx in splitter.split(X, y):
                fold_model = ClassicalMLModel(
                    algorithm=getattr(base_est, "algorithm", "random_forest"),
                    task_type=self.task_type,
                    random_seed=base_est.random_seed,
                    track_carbon=False,
                )
                fold_model.fit(X[tr_idx], y[tr_idx])
                if is_clf:
                    p = fold_model.predict_proba(X[va_idx])
                    if p.shape[1] == oof.shape[1]:
                        oof[va_idx] = p
                else:
                    oof[va_idx, 0] = fold_model.predict(X[va_idx])
            oof_blocks.append(oof)

            # Fit full model on all training rows
            base_est.fit(X, y)
            self._fitted_estimators.append(base_est)

        meta_X = np.hstack(oof_blocks)
        if is_clf:
            self._meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_seed)
            self._meta_learner.fit(meta_X, y)
        else:
            self._meta_learner = Ridge(alpha=1.0, random_state=self.random_seed)
            self._meta_learner.fit(meta_X, y.astype(float))

    def _build_meta_features(self, X: np.ndarray) -> np.ndarray:
        is_clf = self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        blocks = [
            est.predict_proba(X) if is_clf else est.predict(X).reshape(-1, 1)
            for est in self._fitted_estimators
        ]
        return np.hstack(blocks)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.asarray(self._meta_learner.predict(self._build_meta_features(X)))

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if hasattr(self._meta_learner, "predict_proba"):
            return np.asarray(self._meta_learner.predict_proba(self._build_meta_features(X)), dtype=float)
        return super()._predict_proba(X, **kwargs)
