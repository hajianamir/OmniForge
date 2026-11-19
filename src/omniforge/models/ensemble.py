"""Enterprise ensemble models: Stacking, Soft/Hard Voting, Blending, and Weighted Averaging."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold

from omniforge.core.exceptions import ModelTrainingError
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
            if len(w) != len(self.estimators) or np.any(w < 0.0) or float(w.sum()) <= 0.0:
                raise ModelTrainingError("Ensemble weights must be non-negative, match estimator count, and sum > 0.")
            self._norm_weights = w / float(w.sum())

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        global_classes = self.classes_ if self.classes_ is not None else None
        probas = [np.asarray(est.predict_proba(X), dtype=float) for est in self.estimators]
        n_samples = probas[0].shape[0]
        n_cols = len(global_classes) if global_classes is not None else probas[0].shape[1]
        stacked = np.zeros((n_samples, n_cols), dtype=float)
        for est, p, w in zip(self.estimators, probas, self._norm_weights, strict=False):
            if global_classes is not None and est.classes_ is not None and len(est.classes_) == p.shape[1]:
                cls_to_idx = {c: i for i, c in enumerate(global_classes)}
                aligned = np.zeros((n_samples, n_cols), dtype=float)
                for j, c in enumerate(est.classes_):
                    if c in cls_to_idx:
                        aligned[:, cls_to_idx[c]] = p[:, j]
                stacked += w * aligned
            else:
                stacked += w * p
        row_sums = stacked.sum(axis=1, keepdims=True)
        return np.where(row_sums > 0.0, stacked / np.where(row_sums > 0.0, row_sums, 1.0), 1.0 / max(n_cols, 1))

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            proba = self._predict_proba(X)
            idx = np.argmax(proba, axis=1)
            classes = self.classes_ if self.classes_ is not None else np.arange(proba.shape[1])
            return np.asarray([classes[i] for i in idx])
        preds = [est.predict(X).astype(float) for est in self.estimators]
        return np.asarray(np.sum([w * p for w, p in zip(self._norm_weights, preds, strict=False)], axis=0), dtype=float)


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

    def _clone_estimator(self, base_est: BaseOmniForgeModel) -> BaseOmniForgeModel:
        """Create an unfitted clone preserving exact estimator class and hyperparameters."""
        cloned = copy.deepcopy(base_est)
        cloned.task_type = self.task_type
        cloned.track_carbon = False
        cloned.is_fitted_ = False
        return cloned

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        is_clf = self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        n_samples = X.shape[0]
        n_folds = min(self.cv_folds, max(2, n_samples // 4))
        if is_clf:
            classes, counts = np.unique(y, return_counts=True)
            min_cls_count = int(counts.min()) if len(counts) > 0 else 2
            n_folds = max(2, min(n_folds, min_cls_count)) if min_cls_count >= 2 else 2
            splitter: Any = (
                StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
                if min_cls_count >= 2
                else KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
            )
            cls_to_col = {c: i for i, c in enumerate(classes)}
            n_cls = len(classes)
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
            cls_to_col = {}
            n_cls = 1

        oof_blocks: list[np.ndarray] = []
        self._fitted_estimators = []

        for base_est in self.estimators:
            base_est.task_type = self.task_type
            oof = np.zeros((n_samples, n_cls), dtype=float)

            for tr_idx, va_idx in splitter.split(X, y):
                fold_model = self._clone_estimator(base_est)
                fold_model.fit(X[tr_idx], y[tr_idx])
                if is_clf:
                    p = np.asarray(fold_model.predict_proba(X[va_idx]), dtype=float)
                    if fold_model.classes_ is not None and len(fold_model.classes_) == p.shape[1]:
                        for j, c in enumerate(fold_model.classes_):
                            if c in cls_to_col:
                                oof[va_idx, cls_to_col[c]] = p[:, j]
                    elif p.shape[1] == n_cls:
                        oof[va_idx] = p
                else:
                    oof[va_idx, 0] = np.asarray(fold_model.predict(X[va_idx]), dtype=float).ravel()
            oof_blocks.append(oof)

            # Fit full model on all training rows
            full_est = self._clone_estimator(base_est)
            full_est.fit(X, y)
            self._fitted_estimators.append(full_est)

        meta_X = np.hstack(oof_blocks)
        if is_clf:
            self._meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_seed)
            self._meta_learner.fit(meta_X, y)
        else:
            self._meta_learner = Ridge(alpha=1.0, random_state=self.random_seed)
            self._meta_learner.fit(meta_X, y.astype(float))

    def _build_meta_features(self, X: np.ndarray) -> np.ndarray:
        is_clf = self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        global_classes = self.classes_
        blocks: list[np.ndarray] = []
        for est in self._fitted_estimators:
            if is_clf:
                p = np.asarray(est.predict_proba(X), dtype=float)
                if global_classes is not None and est.classes_ is not None and len(est.classes_) == p.shape[1]:
                    cls_to_col = {c: i for i, c in enumerate(global_classes)}
                    aligned = np.zeros((X.shape[0], len(global_classes)), dtype=float)
                    for j, c in enumerate(est.classes_):
                        if c in cls_to_col:
                            aligned[:, cls_to_col[c]] = p[:, j]
                    blocks.append(aligned)
                else:
                    blocks.append(p)
            else:
                blocks.append(np.asarray(est.predict(X), dtype=float).reshape(-1, 1))
        return np.hstack(blocks)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return np.asarray(self._meta_learner.predict(self._build_meta_features(X)))

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if hasattr(self._meta_learner, "predict_proba"):
            return np.asarray(self._meta_learner.predict_proba(self._build_meta_features(X)), dtype=float)
        return super()._predict_proba(X, **kwargs)
