"""Classical ML models covering scikit-learn, XGBoost, LightGBM, and CatBoost with unified API."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from omniforge.core.types import TaskType
from omniforge.core.utils import is_package_available
from omniforge.models.base import BaseOmniForgeModel


class ClassicalMLModel(BaseOmniForgeModel):
    """Unified wrapper over scikit-learn, XGBoost, LightGBM, and CatBoost algorithms."""

    model_family = "classical_ml"

    def __init__(
        self,
        algorithm: str = "random_forest",
        *,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        scale_features: bool | None = None,
        **hyperparameters: Any,
    ) -> None:
        super().__init__(
            name=f"Classical_{algorithm}",
            task_type=task_type,
            algorithm=algorithm,
            **hyperparameters,
        )
        self.algorithm = algorithm.lower()
        needs_scaling = self.algorithm in {
            "logistic_regression",
            "ridge",
            "lasso",
            "elasticnet",
            "svm",
            "knn",
        }
        self.scale_features = needs_scaling if scale_features is None else scale_features
        self._scaler: StandardScaler | None = StandardScaler() if self.scale_features else None
        self._estimator: Any = None
        self._label_to_idx: dict[Any, int] | None = None
        self._idx_to_label: dict[int, Any] | None = None

    def _build_estimator(self, n_classes: int = 2) -> Any:
        is_clf = self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        params = {k: v for k, v in self.hyperparameters.items() if k != "algorithm"}
        seed = self.random_seed

        if self.algorithm == "xgboost":
            if is_package_available("xgboost"):
                import xgboost as xgb  # type: ignore[import-not-found]

                self.framework = "xgboost"
                if is_clf:
                    return xgb.XGBClassifier(
                        n_estimators=params.get("n_estimators", 150),
                        max_depth=params.get("max_depth", 6),
                        learning_rate=params.get("learning_rate", 0.08),
                        random_state=seed,
                        eval_metric="logloss",
                    )
                return xgb.XGBRegressor(
                    n_estimators=params.get("n_estimators", 150),
                    max_depth=params.get("max_depth", 6),
                    learning_rate=params.get("learning_rate", 0.08),
                    random_state=seed,
                )
            self.framework = "scikit-learn(hgb_fallback)"
            return (
                HistGradientBoostingClassifier(
                    max_iter=params.get("n_estimators", 150),
                    max_depth=params.get("max_depth", 6),
                    learning_rate=params.get("learning_rate", 0.08),
                    random_state=seed,
                )
                if is_clf
                else HistGradientBoostingRegressor(
                    max_iter=params.get("n_estimators", 150),
                    max_depth=params.get("max_depth", 6),
                    learning_rate=params.get("learning_rate", 0.08),
                    random_state=seed,
                )
            )

        if self.algorithm == "lightgbm":
            if is_package_available("lightgbm"):
                import lightgbm as lgb  # type: ignore[import-not-found]

                self.framework = "lightgbm"
                if is_clf:
                    return lgb.LGBMClassifier(
                        n_estimators=params.get("n_estimators", 150),
                        learning_rate=params.get("learning_rate", 0.08),
                        num_leaves=params.get("num_leaves", 31),
                        random_state=seed,
                        verbose=-1,
                    )
                return lgb.LGBMRegressor(
                    n_estimators=params.get("n_estimators", 150),
                    learning_rate=params.get("learning_rate", 0.08),
                    num_leaves=params.get("num_leaves", 31),
                    random_state=seed,
                    verbose=-1,
                )
            self.framework = "scikit-learn(hgb_fallback)"
            return (
                HistGradientBoostingClassifier(
                    max_iter=params.get("n_estimators", 150),
                    learning_rate=params.get("learning_rate", 0.08),
                    max_leaf_nodes=params.get("num_leaves", 31),
                    random_state=seed,
                )
                if is_clf
                else HistGradientBoostingRegressor(
                    max_iter=params.get("n_estimators", 150),
                    learning_rate=params.get("learning_rate", 0.08),
                    max_leaf_nodes=params.get("num_leaves", 31),
                    random_state=seed,
                )
            )

        if self.algorithm == "catboost":
            if is_package_available("catboost"):
                from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore[import-not-found]

                self.framework = "catboost"
                if is_clf:
                    return CatBoostClassifier(
                        iterations=params.get("n_estimators", 150),
                        depth=params.get("max_depth", 6),
                        learning_rate=params.get("learning_rate", 0.08),
                        random_seed=seed,
                        verbose=False,
                    )
                return CatBoostRegressor(
                    iterations=params.get("n_estimators", 150),
                    depth=params.get("max_depth", 6),
                    learning_rate=params.get("learning_rate", 0.08),
                    random_seed=seed,
                    verbose=False,
                )
            self.framework = "scikit-learn(gb_fallback)"
            return (
                GradientBoostingClassifier(
                    n_estimators=params.get("n_estimators", 120),
                    max_depth=params.get("max_depth", 4),
                    learning_rate=params.get("learning_rate", 0.08),
                    random_state=seed,
                )
                if is_clf
                else GradientBoostingRegressor(
                    n_estimators=params.get("n_estimators", 120),
                    max_depth=params.get("max_depth", 4),
                    learning_rate=params.get("learning_rate", 0.08),
                    random_state=seed,
                )
            )

        self.framework = "scikit-learn"
        if self.algorithm == "random_forest":
            return (
                RandomForestClassifier(
                    n_estimators=params.get("n_estimators", 120),
                    max_depth=params.get("max_depth", None),
                    min_samples_leaf=params.get("min_samples_leaf", 1),
                    random_state=seed,
                    n_jobs=-1,
                )
                if is_clf
                else RandomForestRegressor(
                    n_estimators=params.get("n_estimators", 120),
                    max_depth=params.get("max_depth", None),
                    min_samples_leaf=params.get("min_samples_leaf", 1),
                    random_state=seed,
                    n_jobs=-1,
                )
            )
        if self.algorithm == "extra_trees":
            return (
                ExtraTreesClassifier(
                    n_estimators=params.get("n_estimators", 120),
                    max_depth=params.get("max_depth", None),
                    random_state=seed,
                    n_jobs=-1,
                )
                if is_clf
                else ExtraTreesRegressor(
                    n_estimators=params.get("n_estimators", 120),
                    max_depth=params.get("max_depth", None),
                    random_state=seed,
                    n_jobs=-1,
                )
            )
        if self.algorithm in ("gradient_boosting", "gbdt", "hist_gradient_boosting"):
            return (
                HistGradientBoostingClassifier(
                    max_iter=params.get("n_estimators", 150),
                    learning_rate=params.get("learning_rate", 0.08),
                    max_depth=params.get("max_depth", 6),
                    random_state=seed,
                )
                if is_clf
                else HistGradientBoostingRegressor(
                    max_iter=params.get("n_estimators", 150),
                    learning_rate=params.get("learning_rate", 0.08),
                    max_depth=params.get("max_depth", 6),
                    random_state=seed,
                )
            )
        if self.algorithm == "logistic_regression":
            return LogisticRegression(C=params.get("C", 1.0), max_iter=1000, random_state=seed)
        if self.algorithm == "ridge":
            return Ridge(alpha=params.get("alpha", 1.0), random_state=seed)
        if self.algorithm == "lasso":
            return Lasso(alpha=params.get("alpha", 0.1), random_state=seed, max_iter=2000)
        if self.algorithm == "elasticnet":
            return ElasticNet(
                alpha=params.get("alpha", 0.1), l1_ratio=params.get("l1_ratio", 0.5), random_state=seed
            )
        if self.algorithm == "svm":
            return (
                SVC(C=params.get("C", 1.0), kernel=params.get("kernel", "rbf"), probability=True, random_state=seed)
                if is_clf
                else SVR(C=params.get("C", 1.0), kernel=params.get("kernel", "rbf"))
            )
        if self.algorithm == "knn":
            k = params.get("n_neighbors", 5)
            return KNeighborsClassifier(n_neighbors=k) if is_clf else KNeighborsRegressor(n_neighbors=k)
        if self.algorithm == "naive_bayes":
            return GaussianNB()

        # Default robust tree ensemble
        return (
            RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            if is_clf
            else RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
        )

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        X_in = self._scaler.fit_transform(X) if self._scaler is not None else X
        n_classes = len(np.unique(y))
        self._estimator = self._build_estimator(n_classes=n_classes)
        self.metadata_.framework = self.framework

        if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            classes = np.unique(y)
            self._label_to_idx = {c: idx for idx, c in enumerate(classes)}
            self._idx_to_label = {idx: c for idx, c in enumerate(classes)}
            y_encoded = np.array([self._label_to_idx[val] for val in y], dtype=int)
            self._estimator.fit(X_in, y_encoded)
        else:
            self._estimator.fit(X_in, y.astype(float))

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X_in = self._scaler.transform(X) if self._scaler is not None else X
        raw = np.asarray(self._estimator.predict(X_in))
        if self._idx_to_label is not None:
            return np.array([self._idx_to_label[int(v)] for v in raw])
        return raw.astype(float)

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X_in = self._scaler.transform(X) if self._scaler is not None else X
        if hasattr(self._estimator, "predict_proba"):
            return np.asarray(self._estimator.predict_proba(X_in), dtype=float)
        return super()._predict_proba(X, **kwargs)

    def get_feature_importance(self) -> dict[str, float]:
        """Extract normalized feature importances or absolute linear coefficients."""
        self._check_is_fitted()
        raw_imp: np.ndarray | None = None
        if hasattr(self._estimator, "feature_importances_"):
            raw_imp = np.asarray(self._estimator.feature_importances_, dtype=float)
        elif hasattr(self._estimator, "coef_"):
            coef = np.asarray(self._estimator.coef_, dtype=float)
            raw_imp = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        if raw_imp is None or len(raw_imp) != len(self.feature_names_):
            return {}
        total = float(np.sum(np.abs(raw_imp))) or 1.0
        return {
            name: round(float(val / total), 6)
            for name, val in sorted(
                zip(self.feature_names_, raw_imp, strict=False), key=lambda kv: kv[1], reverse=True
            )
        }
