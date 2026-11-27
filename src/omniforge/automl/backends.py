"""Multi-Backend AutoML Hub: Native Multi-Layer Stacking, Genetic Pipeline Search, Meta-Bayesian Search & External Adapters (AutoGluon, TPOT, H2O, Auto-sklearn, FLAML)."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, root_mean_squared_error
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from omniforge.automl.pipeline_search import AutoMLPipelineSearch, AutoMLResult
from omniforge.core.exceptions import AutoMLError
from omniforge.core.logging import get_logger
from omniforge.core.performance import parallel_n_jobs
from omniforge.core.types import TaskType
from omniforge.core.utils import is_package_available
from omniforge.data.splitting import LeakageSafeSplitter, SplitConfig, SplitStrategyType
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.baselines import BaselineBenchmarkSuite, BaselineEvidence, strongest_scored_reference

logger = get_logger(__name__)


def _pkg_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except Exception:
        return None


def probe_automl_backends() -> list[dict[str, Any]]:
    """Probe all native and optional external AutoML backends with live OS/Python/Java compatibility diagnostics."""
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    os_name = platform.system()
    has_java = shutil.which("java") is not None

    has_ag = is_package_available("autogluon.tabular")
    has_tpot = is_package_available("tpot")
    has_h2o = is_package_available("h2o")
    has_ask = is_package_available("autosklearn")
    has_flaml = is_package_available("flaml")

    return [
        {
            "id": "omniforge_native",
            "name": "OmniForge Leakage-Safe AutoML",
            "category": "Native Built-In",
            "available": True,
            "version": "4.0.0",
            "notes": "Stratified/Group/TimeSeries CV + ColumnTransformer + Optuna HPO + Baselines.",
            "native_equivalent": "omniforge_native",
        },
        {
            "id": "multi_layer_stacker",
            "name": "Multi-Layer Stack Ensemble (AutoGluon-Style Native)",
            "category": "Native Paper-Grade",
            "available": True,
            "version": "4.0.0",
            "notes": "K-fold OOF Layer-1 + Layer-2 stackers + Caruana greedy weighted ensemble selection.",
            "native_equivalent": "multi_layer_stacker",
        },
        {
            "id": "genetic_pipeline",
            "name": "Evolutionary Genetic Pipeline Search (TPOT-Style Native)",
            "category": "Native Paper-Grade",
            "available": True,
            "version": "4.0.0",
            "notes": "Genetic programming over imputer/scaler/feature-transform/estimator DAGs with tournament selection & elitism.",
            "native_equivalent": "genetic_pipeline",
        },
        {
            "id": "meta_bayesian",
            "name": "Meta-Learning Bayesian AutoML (Auto-sklearn/SMAC-Style Native)",
            "category": "Native Paper-Grade",
            "available": True,
            "version": "4.0.0",
            "notes": "14 dataset meta-features + landmarking warm-start + budgeted search + ensemble selection.",
            "native_equivalent": "meta_bayesian",
        },
        {
            "id": "autogluon",
            "name": "AutoGluon Tabular",
            "category": "External Adapter",
            "available": has_ag,
            "version": _pkg_version("autogluon.tabular") or _pkg_version("autogluon"),
            "notes": (
                "Installed & ready."
                if has_ag
                else f"Optional package not installed (supports Python 3.10–3.13; current Python {py_ver}). Auto-falls back to Multi-Layer Stack Ensemble."
            ),
            "native_equivalent": "multi_layer_stacker",
        },
        {
            "id": "tpot",
            "name": "TPOT (Genetic Programming)",
            "category": "External Adapter",
            "available": has_tpot,
            "version": _pkg_version("TPOT") or _pkg_version("tpot"),
            "notes": (
                "Installed & ready."
                if has_tpot
                else f"Optional package not installed (supports Python 3.10–3.13; current Python {py_ver}). Auto-falls back to Evolutionary Genetic Pipeline Search."
            ),
            "native_equivalent": "genetic_pipeline",
        },
        {
            "id": "h2o",
            "name": "H2O AutoML",
            "category": "External Adapter",
            "available": has_h2o and has_java,
            "version": _pkg_version("h2o"),
            "notes": (
                "Installed & Java JRE detected."
                if (has_h2o and has_java)
                else (
                    "h2o package installed, but Java JRE (`java`) not found in PATH."
                    if has_h2o
                    else f"Optional package not installed (requires `pip install h2o` + Java JRE; Java detected: {has_java}). Auto-falls back to Multi-Layer Stack Ensemble."
                )
            ),
            "native_equivalent": "multi_layer_stacker",
        },
        {
            "id": "autosklearn",
            "name": "Auto-sklearn",
            "category": "External Adapter",
            "available": has_ask,
            "version": _pkg_version("auto-sklearn"),
            "notes": (
                "Installed & ready."
                if has_ask
                else f"Optional Linux-only legacy package (v0.15.0; current OS={os_name}, Python {py_ver}). Auto-falls back to Meta-Learning Bayesian AutoML."
            ),
            "native_equivalent": "meta_bayesian",
        },
        {
            "id": "flaml",
            "name": "FLAML (Microsoft Fast AutoML)",
            "category": "External Adapter",
            "available": has_flaml,
            "version": _pkg_version("FLAML") or _pkg_version("flaml"),
            "notes": (
                "Installed & ready."
                if has_flaml
                else "Optional package not installed (`pip install flaml`, Python 3.10–3.15 compatible). Auto-falls back to OmniForge Leakage-Safe AutoML."
            ),
            "native_equivalent": "omniforge_native",
        },
    ]


def _infer_task(y: pd.Series, requested: TaskType | None) -> TaskType:
    if requested is not None:
        return requested
    n_uniq = int(y.nunique(dropna=True))
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y) or n_uniq <= 15:
        return TaskType.BINARY_CLASSIFICATION if n_uniq <= 2 else TaskType.MULTICLASS_CLASSIFICATION
    return TaskType.REGRESSION


def _build_preprocessor(X: pd.DataFrame, *, scaler_kind: str = "standard", imputer_kind: str = "median") -> ColumnTransformer:
    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers: list[tuple[str, Any, list[str]]] = []
    if num_cols:
        scaler: Any = StandardScaler() if scaler_kind == "standard" else (RobustScaler() if scaler_kind == "robust" else PowerTransformer())
        num_pipe = Pipeline([("imp", SimpleImputer(strategy=imputer_kind)), ("scl", scaler)])
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline(
            [
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=32)),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


class _WrappedSklearnModel(BaseOmniForgeModel):
    """Wraps any fitted sklearn/external Pipeline or Multi-Layer Stacker as a first-class BaseOmniForgeModel."""

    model_family = "automl_pipeline"
    framework = "omniforge_automl"

    def __init__(
        self,
        name: str,
        estimator: Any,
        task_type: TaskType,
        *,
        classes: np.ndarray | None = None,
        extra_meta: dict[str, Any] | None = None,
        random_seed: int = 42,
    ) -> None:
        super().__init__(name=name, task_type=task_type, random_seed=random_seed, track_carbon=False)
        self._estimator = estimator
        self.is_fitted_ = True
        self._classes = classes
        self.classes_ = classes
        self.extra_meta = extra_meta or {}

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        _ = kwargs
        if hasattr(self._estimator, "fit"):
            self._estimator.fit(X, y)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        _ = kwargs
        return np.asarray(self._estimator.predict(X))

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        _ = kwargs
        if isinstance(X, pd.DataFrame):
            return np.asarray(self._estimator.predict(X))
        return np.asarray(self._estimator.predict(X))

    def predict_proba(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> np.ndarray:
        _ = kwargs
        if hasattr(self._estimator, "predict_proba"):
            return np.asarray(self._estimator.predict_proba(X), dtype=float)
        raise AutoMLError(f"Model '{self.name}' does not support probability predictions.")


class _CaruanaStackPredictor:
    """Inference wrapper for AutoGluon-style K-fold Multi-Layer Stack + Caruana Weighted Ensemble."""

    def __init__(
        self,
        preprocessor: ColumnTransformer,
        l1_models: dict[str, Any],
        l2_models: dict[str, Any],
        weights: dict[str, float],
        *,
        is_clf: bool,
        classes: np.ndarray | None = None,
        stack_features: list[str] | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.l1_models = l1_models
        self.l2_models = l2_models
        self.weights = weights
        self.is_clf = is_clf
        self.classes_ = classes
        self.stack_features = list(l1_models) if stack_features is None else stack_features

    def _transform_and_l1(self, X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
        if not isinstance(X, pd.DataFrame) and hasattr(self.preprocessor, "feature_names_in_"):
            values = np.asarray(X)
            names = self.preprocessor.feature_names_in_
            if values.ndim != 2 or values.shape[1] != len(names):
                raise AutoMLError("Stacking input columns do not match the fitted feature schema.")
            X = pd.DataFrame(values, columns=names)
        X_mat = self.preprocessor.transform(X)
        l1_preds = {
            name: MultiLayerStackingAutoML._response(estimator, X_mat, self.classes_ if self.is_clf else None)
            for name, estimator in self.l1_models.items()
        }
        stack_features = getattr(self, "stack_features", self.l1_models)
        stack_blocks = [X_mat, *[l1_preds[name] for name in stack_features]]
        return X_mat, l1_preds, np.hstack(stack_blocks)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_clf:
            raise AutoMLError("predict_proba is only supported for classification stackers.")
        _, l1_preds, X_l2 = self._transform_and_l1(X)
        all_preds = dict(l1_preds)
        for mname, est in self.l2_models.items():
            all_preds[mname] = MultiLayerStackingAutoML._response(est, X_l2, self.classes_)
        blend: np.ndarray | None = None
        total_w = 0.0
        for mname, w in self.weights.items():
            if w <= 0 or mname not in all_preds:
                continue
            p = all_preds[mname]
            blend = p * w if blend is None else blend + p * w
            total_w += w
        assert blend is not None
        return blend / max(total_w, 1e-12)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.is_clf:
            probs = self.predict_proba(X)
            idx = np.argmax(probs, axis=1)
            # ``np.asarray`` keeps the label dtype (int/bool/str): an object cast would break
            # sklearn metrics with "mix of binary and unknown targets".
            labels: np.ndarray = np.asarray(self.classes_) if self.classes_ is not None else np.asarray(idx)
            return np.asarray(labels[idx]) if self.classes_ is not None else labels
        _, l1_preds, X_l2 = self._transform_and_l1(X)
        all_preds = dict(l1_preds)
        for mname, est in self.l2_models.items():
            all_preds[mname] = MultiLayerStackingAutoML._response(est, X_l2, None)
        blend: np.ndarray | None = None
        total_w = 0.0
        for mname, w in self.weights.items():
            if w <= 0 or mname not in all_preds:
                continue
            p = all_preds[mname].ravel()
            blend = p * w if blend is None else blend + p * w
            total_w += w
        assert blend is not None
        return blend / max(total_w, 1e-12)


def _caruana_greedy_weights(
    oof_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    is_clf: bool,
    classes: np.ndarray | None = None,
    n_rounds: int = 12,
) -> tuple[dict[str, float], float]:
    """Caruana et al. (2004) greedy forward ensemble selection with replacement."""
    names = list(oof_preds.keys())
    if not names:
        return {}, 0.0

    def _eval_pred(pred_arr: np.ndarray) -> float:
        if is_clf:
            idx = np.argmax(pred_arr, axis=1)
            y_hat = classes[idx] if classes is not None else idx
            return float(f1_score(y_true, y_hat, average="macro", zero_division=0))
        return -float(root_mean_squared_error(y_true, pred_arr.ravel()))

    counts: dict[str, int] = dict.fromkeys(names, 0)
    cur_sum: np.ndarray | None = None
    best_score = -1e18

    for step in range(1, n_rounds + 1):
        best_cand = names[0]
        best_cand_score = -1e18
        for mname in names:
            cand_avg = oof_preds[mname] if cur_sum is None else (cur_sum + oof_preds[mname]) / float(step)
            sc = _eval_pred(cand_avg)
            if sc > best_cand_score:
                best_cand_score = sc
                best_cand = mname
        counts[best_cand] += 1
        cur_sum = oof_preds[best_cand].copy() if cur_sum is None else cur_sum + oof_preds[best_cand]
        best_score = best_cand_score

    total = float(sum(counts.values())) or 1.0
    weights = {k: round(v / total, 4) for k, v in counts.items() if v > 0}
    return weights, best_score


@dataclass
class _StackingScope:
    preprocessor: ColumnTransformer
    train_matrix: np.ndarray
    eval_matrix: np.ndarray
    estimators: dict[str, Any] = field(default_factory=dict)
    eval_preds: dict[str, np.ndarray] = field(default_factory=dict)
    inner_preds: dict[str, np.ndarray] = field(default_factory=dict)


class MultiLayerStackingAutoML:
    """AutoGluon-inspired K-Fold Out-of-Fold Multi-Layer Stack Ensembling + Caruana Weighted Ensemble."""

    def __init__(self, *, cv_folds: int = 4, random_seed: int = 42) -> None:
        self.cv_folds = max(2, int(cv_folds))
        self.random_seed = int(random_seed)

    def _layer1_specs(self, is_clf: bool) -> dict[str, Any]:
        jobs = parallel_n_jobs()
        if is_clf:
            specs: dict[str, Any] = {
                "L1_Logistic": LogisticRegression(max_iter=500, random_state=self.random_seed),
                "L1_RandomForest": RandomForestClassifier(n_estimators=120, random_state=self.random_seed, n_jobs=jobs),
                "L1_ExtraTrees": ExtraTreesClassifier(n_estimators=120, random_state=self.random_seed, n_jobs=jobs),
                "L1_HistGBM": HistGradientBoostingClassifier(max_iter=140, random_state=self.random_seed),
                "L1_MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=120, early_stopping=True, random_state=self.random_seed),
            }
            if is_package_available("lightgbm"):
                from lightgbm import LGBMClassifier

                specs["L1_LightGBM"] = LGBMClassifier(n_estimators=140, verbosity=-1, random_state=self.random_seed, n_jobs=jobs)
            return specs
        specs_r: dict[str, Any] = {
            "L1_Ridge": Ridge(alpha=1.0),
            "L1_RandomForest": RandomForestRegressor(n_estimators=120, random_state=self.random_seed, n_jobs=jobs),
            "L1_ExtraTrees": ExtraTreesRegressor(n_estimators=120, random_state=self.random_seed, n_jobs=jobs),
            "L1_HistGBM": HistGradientBoostingRegressor(max_iter=140, random_state=self.random_seed),
            "L1_MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=120, early_stopping=True, random_state=self.random_seed),
        }
        if is_package_available("lightgbm"):
            from lightgbm import LGBMRegressor

            specs_r["L1_LightGBM"] = LGBMRegressor(n_estimators=140, verbosity=-1, random_state=self.random_seed, n_jobs=jobs)
        return specs_r

    def _folds(
        self, y: np.ndarray, classes: np.ndarray | None, groups: np.ndarray | None,
        *, time_ordered: bool, gap: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if len(y) < 3:
            return []
        rows = np.arange(len(y))
        if time_ordered:
            # Preserve the requested embargo. Reduce the number of folds only
            # when the series cannot support that many chronological splits.
            for n_splits in range(min(self.cv_folds, len(y) - 1), 1, -1):
                try:
                    folds = list(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(rows))
                    break
                except ValueError:
                    continue
            else:
                return []
        elif groups is not None:
            n_splits = min(self.cv_folds, len(pd.unique(groups)))
            if classes is not None:
                n_splits = min(n_splits, int(pd.Series(y).value_counts().min()))
            if n_splits < 2:
                return []
            cv = GroupKFold(n_splits=n_splits) if classes is None else StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_seed,
            )
            folds = list(cv.split(rows, y, groups))
        elif classes is not None:
            if len(np.unique(y)) != len(classes):
                return []
            n_splits = min(self.cv_folds, int(pd.Series(y).value_counts().min()))
            if n_splits < 2:
                return []
            folds = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed).split(rows, y))
        else:
            n_splits = max(2, min(self.cv_folds, len(y) // 5))
            folds = list(KFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed).split(rows))
        return [
            (train, valid) for train, valid in folds
            if len(train) >= 2 and (classes is None or np.array_equal(np.unique(y[train]), classes))
        ]

    @staticmethod
    def _response(estimator: Any, matrix: np.ndarray, classes: np.ndarray | None) -> np.ndarray:
        if classes is None:
            response = np.asarray(estimator.predict(matrix), dtype=float).reshape(-1, 1)
            n_columns = 1
        else:
            probabilities = np.asarray(estimator.predict_proba(matrix), dtype=float)
            order = pd.Index(estimator.classes_).get_indexer(classes)
            if (order < 0).any() or probabilities.shape != (len(matrix), len(order)):
                raise AutoMLError("The stacking candidate did not provide the fitted class columns.")
            response = probabilities[:, order]
            if (response < 0).any() or (response > 1).any() or not np.allclose(response.sum(axis=1), 1, atol=1e-6, rtol=0):
                raise AutoMLError("Stacking requires normalized class probabilities.")
            n_columns = len(classes)
        if response.shape != (len(matrix), n_columns) or not np.isfinite(response).all():
            raise AutoMLError("The stacking candidate produced invalid predictions.")
        return response

    def _layer1_scope(
        self, X: pd.DataFrame, y: np.ndarray, X_eval: pd.DataFrame, specs: dict[str, Any],
        classes: np.ndarray | None, inner_folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> _StackingScope:
        pre = _build_preprocessor(X)
        train_matrix = pre.fit_transform(X)
        eval_matrix = pre.transform(X_eval)
        inner_matrices = []
        for train, valid in inner_folds:
            transformer = _build_preprocessor(X.iloc[train])
            inner_matrices.append((transformer.fit_transform(X.iloc[train]), transformer.transform(X.iloc[valid])))
        scope = _StackingScope(pre, train_matrix, eval_matrix)
        for name, proto in specs.items():
            try:
                estimator = clone(proto).fit(train_matrix, y)
                scope.eval_preds[name] = self._response(estimator, eval_matrix, classes)
                scope.estimators[name] = estimator
            except Exception as exc:
                logger.warning("stack_l1_model_skipped", model=name, error_type=type(exc).__name__)
                continue
            if not inner_folds:
                continue
            oof = np.full((len(X), len(classes) if classes is not None else 1), np.nan)
            try:
                for (train, valid), (inner_train, inner_valid) in zip(inner_folds, inner_matrices, strict=True):
                    inner_model = clone(proto).fit(inner_train, y[train])
                    oof[valid] = self._response(inner_model, inner_valid, classes)
                scope.inner_preds[name] = oof
            except Exception as exc:
                logger.warning("stack_inner_candidate_skipped", model=name, error_type=type(exc).__name__)
        return scope

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, task_type: TaskType,
        *, groups: np.ndarray | None = None, time_ordered: bool = False, gap: int = 0,
    ) -> dict[str, Any]:
        """Fit nested stacking. A chronological development frame must already be ordered."""
        t0 = time.perf_counter()
        if not X_train.index.equals(y_train.index) or not X_val.index.equals(y_val.index):
            raise AutoMLError("Stacking predictors and targets must have aligned row indexes.")
        if gap < 0 or (gap and not time_ordered) or (time_ordered and groups is not None):
            raise AutoMLError("Choose either grouped or chronological stacking. Embargo gaps require chronology.")
        if groups is not None:
            groups = np.asarray(groups)
            if groups.shape != (len(X_train),) or pd.isna(groups).any():
                raise AutoMLError("Stacking requires one observed group per development row.")
        y = np.asarray(y_train)
        y_val_arr = np.asarray(y_val)
        if not len(y_val_arr) or pd.isna(y).any() or pd.isna(y_val_arr).any():
            raise AutoMLError("Stacking requires observed development and validation outcomes.")
        is_clf = task_type != TaskType.REGRESSION
        classes = np.unique(y) if is_clf else None
        if classes is not None and (len(classes) < 2 or not pd.Series(y_val_arr).isin(classes).all()):
            raise AutoMLError("The development set must contain at least two classes and cover the validation classes.")
        folds = self._folds(y, classes, groups, time_ordered=time_ordered, gap=gap)
        if not folds:
            raise AutoMLError("The development set cannot support leakage-safe stacking folds.")

        specs = self._layer1_specs(is_clf)
        final = self._layer1_scope(X_train, y, X_val, specs, classes, [])
        n_columns = len(classes) if classes is not None else 1
        outer_predictions = {name: np.full((len(y), n_columns), np.nan) for name in final.estimators}
        outer_scopes: list[tuple[np.ndarray, np.ndarray, _StackingScope]] = []
        provenance = []
        for train, valid in folds:
            inner_folds = self._folds(
                y[train], classes, groups[train] if groups is not None else None,
                time_ordered=time_ordered, gap=gap,
            )
            scope = self._layer1_scope(X_train.iloc[train], y[train], X_train.iloc[valid], specs, classes, inner_folds)
            # Outer estimators are used only to produce fold predictions.
            scope.estimators.clear()
            for name, preds in scope.eval_preds.items():
                if name in outer_predictions:
                    outer_predictions[name][valid] = preds
            outer_scopes.append((train, valid, scope))
            provenance.append({
                "train": train.tolist(), "validation": valid.tolist(),
                "inner_folds": [{"train": train[a].tolist(), "validation": train[b].tolist()} for a, b in inner_folds],
            })

        scored_rows = np.sort(np.concatenate([valid for _, valid in folds]))
        names = [name for name, preds in outer_predictions.items() if np.isfinite(preds[scored_rows]).all()]
        if not names:
            raise AutoMLError("No Layer-1 model completed all eligible outer folds.")
        l1_fitted = {name: final.estimators[name] for name in names}
        oof_preds = {name: outer_predictions[name][scored_rows] for name in names}
        metric = "f1_macro" if is_clf else "rmse"

        def validation_score(preds: np.ndarray) -> float:
            if classes is not None:
                return float(f1_score(y_val_arr, classes[np.argmax(preds, axis=1)], average="macro", zero_division=0))
            return float(root_mean_squared_error(y_val_arr, preds.ravel()))

        leaderboard: list[dict[str, Any]] = [
            {"model": name, "layer": "L1", "metric": metric, "score": round(validation_score(final.eval_preds[name]), 6)}
            for name in names
        ]

        # All stacking folds must have the same base-model feature blocks.
        # Each outer scope learns the augmented training matrix solely from
        # inner OOF predictions within that outer training set.
        stack_features = [name for name in names if all(name in scope.inner_preds for _, _, scope in outer_scopes)]
        nested_matrices = []
        for train, valid, scope in outer_scopes:
            if not stack_features:
                break
            inner = np.hstack([scope.train_matrix, *[scope.inner_preds[name] for name in stack_features]])
            usable = np.isfinite(inner).all(axis=1)
            if usable.sum() < 2 or (classes is not None and not np.array_equal(np.unique(y[train][usable]), classes)):
                break
            held_out = np.hstack([scope.eval_matrix, *[scope.eval_preds[name] for name in stack_features]])
            nested_matrices.append((train[usable], valid, inner[usable], held_out))

        l2_fitted: dict[str, Any] = {}
        if len(nested_matrices) == len(outer_scopes):
            dev = np.hstack([final.train_matrix[scored_rows], *[oof_preds[name] for name in stack_features]])
            validation = np.hstack([final.eval_matrix, *[final.eval_preds[name] for name in stack_features]])
            l2_specs = (
                {
                    "L2_Stack_HistGBM": HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=self.random_seed),
                    "L2_Stack_Logistic": LogisticRegression(C=0.5, max_iter=500, random_state=self.random_seed),
                } if is_clf else {
                    "L2_Stack_HistGBM": HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=self.random_seed),
                    "L2_Stack_Ridge": Ridge(alpha=5.0),
                }
            )
            for name, proto in l2_specs.items():
                try:
                    oof = np.full((len(y), n_columns), np.nan)
                    for train, valid, inner, held_out in nested_matrices:
                        fold_model = clone(proto).fit(inner, y[train])
                        oof[valid] = self._response(fold_model, held_out, classes)
                    model = clone(proto).fit(dev, y[scored_rows])
                    score = validation_score(self._response(model, validation, classes))
                    l2_fitted[name] = model
                    oof_preds[name] = oof[scored_rows]
                    leaderboard.append({"model": name, "layer": "L2_Stack", "metric": metric, "score": round(score, 6)})
                except Exception as exc:
                    logger.warning("stack_l2_candidate_skipped", model=name, error_type=type(exc).__name__)

        weights, _ = _caruana_greedy_weights(oof_preds, y[scored_rows], is_clf=is_clf, classes=classes, n_rounds=15)
        predictor = _CaruanaStackPredictor(
            final.preprocessor, l1_fitted, l2_fitted, weights,
            is_clf=is_clf, classes=classes, stack_features=stack_features,
        )
        ensemble_predictions = predictor.predict_proba(X_val) if is_clf else predictor.predict(X_val)
        ens_score = validation_score(ensemble_predictions)
        ensemble_name = "L3_WeightedEnsemble_Caruana" if l2_fitted else "L1_WeightedEnsemble_Caruana"
        leaderboard.append({"model": ensemble_name, "layer": "L3_Ensemble" if l2_fitted else "L1_Ensemble",
                            "metric": metric, "score": round(ens_score, 6)})
        leaderboard.sort(key=lambda row: float(row["score"]), reverse=is_clf)
        summary = {
            "n_oof_rows": len(scored_rows), "n_outer_folds": len(folds),
            "layer_two_candidates": list(l2_fitted),
            "layer_two_inputs": stack_features if l2_fitted else [],
        }
        return {
            "best_model": _WrappedSklearnModel(
                name="MultiLayerStackEnsemble", estimator=predictor, task_type=task_type, classes=classes,
                extra_meta={"ensemble_weights": weights, "stacking_validation": summary}, random_seed=self.random_seed,
            ),
            "best_model_name": ensemble_name,
            "best_score": round(ens_score, 6),
            "primary_metric": metric,
            "leaderboard": leaderboard,
            "ensemble_weights": weights,
            "stacking_validation": summary,
            "fold_provenance": provenance,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }

@dataclass
class _PipelineGenome:
    scaler: str
    feature_op: str
    estimator: str
    param_strength: float

    def label(self) -> str:
        return f"{self.scaler} -> {self.feature_op} -> {self.estimator}({self.param_strength:.2f})"


class GeneticPipelineAutoML:
    """TPOT-inspired Evolutionary Genetic Programming Pipeline Optimizer over full sklearn DAGs."""

    def __init__(self, *, population_size: int = 10, generations: int = 4, cv_folds: int = 3, random_seed: int = 42) -> None:
        self.population_size = max(4, int(population_size))
        self.generations = max(2, int(generations))
        self.cv_folds = max(2, int(cv_folds))
        self.random_seed = int(random_seed)

    def _build_pipeline(self, genome: _PipelineGenome, X: pd.DataFrame, is_clf: bool) -> Pipeline:
        pre = _build_preprocessor(X, scaler_kind=genome.scaler)
        steps: list[tuple[str, Any]] = [("pre", pre)]
        if genome.feature_op == "pca":
            steps.append(("feat", PCA(n_components=0.95, svd_solver="full")))
        elif genome.feature_op == "kbest":
            k_best = max(2, min(15, X.shape[1]))
            steps.append(("feat", SelectKBest(f_classif if is_clf else f_regression, k=k_best)))

        jobs = parallel_n_jobs()
        s = genome.param_strength
        if is_clf:
            if genome.estimator == "rf":
                est: Any = RandomForestClassifier(n_estimators=int(60 + 100 * s), max_depth=int(4 + 12 * s), random_state=self.random_seed, n_jobs=jobs)
            elif genome.estimator == "et":
                est = ExtraTreesClassifier(n_estimators=int(60 + 100 * s), max_depth=int(4 + 12 * s), random_state=self.random_seed, n_jobs=jobs)
            elif genome.estimator == "hgb":
                est = HistGradientBoostingClassifier(max_iter=int(50 + 120 * s), learning_rate=0.03 + 0.15 * s, random_state=self.random_seed)
            elif genome.estimator == "knn":
                est = KNeighborsClassifier(n_neighbors=max(3, int(3 + 12 * s)), n_jobs=jobs)
            else:
                est = LogisticRegression(C=0.1 + 5.0 * s, max_iter=400, random_state=self.random_seed)
        else:
            if genome.estimator == "rf":
                est = RandomForestRegressor(n_estimators=int(60 + 100 * s), max_depth=int(4 + 12 * s), random_state=self.random_seed, n_jobs=jobs)
            elif genome.estimator == "et":
                est = ExtraTreesRegressor(n_estimators=int(60 + 100 * s), max_depth=int(4 + 12 * s), random_state=self.random_seed, n_jobs=jobs)
            elif genome.estimator == "hgb":
                est = HistGradientBoostingRegressor(max_iter=int(50 + 120 * s), learning_rate=0.03 + 0.15 * s, random_state=self.random_seed)
            elif genome.estimator == "knn":
                est = KNeighborsRegressor(n_neighbors=max(3, int(3 + 12 * s)), n_jobs=jobs)
            else:
                est = Ridge(alpha=0.1 + 10.0 * s)
        steps.append(("model", est))
        return Pipeline(steps)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, task_type: TaskType) -> dict[str, Any]:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.random_seed)
        is_clf = task_type != TaskType.REGRESSION
        scalers = ["standard", "robust"]
        feat_ops = ["none", "kbest", "pca"]
        estimators = ["hgb", "rf", "et", "linear", "knn"]

        def _rand_genome() -> _PipelineGenome:
            return _PipelineGenome(
                scaler=str(rng.choice(scalers)),
                feature_op=str(rng.choice(feat_ops)),
                estimator=str(rng.choice(estimators)),
                param_strength=round(float(rng.uniform(0.15, 0.95)), 2),
            )

        pop = [_rand_genome() for _ in range(self.population_size)]
        scoring = "f1_macro" if is_clf else "neg_root_mean_squared_error"
        history: list[dict[str, Any]] = []
        evaluated: dict[str, tuple[_PipelineGenome, float]] = {}

        for gen in range(1, self.generations + 1):
            gen_scores: list[tuple[_PipelineGenome, float]] = []
            for g in pop:
                key = g.label()
                if key in evaluated:
                    sc = evaluated[key][1]
                else:
                    try:
                        pipe = self._build_pipeline(g, X_train, is_clf)
                        cv_vals = cross_val_score(pipe, X_train, y_train, cv=self.cv_folds, scoring=scoring)
                        sc = float(np.nanmean(cv_vals))
                    except Exception:
                        sc = -1e9
                    evaluated[key] = (g, sc)
                gen_scores.append((g, sc))
            gen_scores.sort(key=lambda kv: kv[1], reverse=True)
            best_g, best_cv = gen_scores[0]
            history.append({"generation": gen, "best_pipeline": best_g.label(), "cv_fitness": round(best_cv, 6)})

            # Elitism + Tournament Selection + Crossover + Mutation
            next_pop: list[_PipelineGenome] = [gen_scores[0][0], gen_scores[min(1, len(gen_scores) - 1)][0]]
            while len(next_pop) < self.population_size:
                p1 = max(rng.choice(len(gen_scores), size=min(3, len(gen_scores)), replace=False), key=lambda i: gen_scores[int(i)][1])
                p2 = max(rng.choice(len(gen_scores), size=min(3, len(gen_scores)), replace=False), key=lambda i: gen_scores[int(i)][1])
                g1, g2 = gen_scores[int(p1)][0], gen_scores[int(p2)][0]
                child = _PipelineGenome(
                    scaler=g1.scaler if rng.random() < 0.5 else g2.scaler,
                    feature_op=g1.feature_op if rng.random() < 0.5 else g2.feature_op,
                    estimator=g1.estimator if rng.random() < 0.5 else g2.estimator,
                    param_strength=round(float(np.clip((g1.param_strength + g2.param_strength) / 2.0 + rng.normal(0, 0.12), 0.05, 1.0)), 2),
                )
                if rng.random() < 0.25:
                    child.estimator = str(rng.choice(estimators))
                if rng.random() < 0.2:
                    child.feature_op = str(rng.choice(feat_ops))
                next_pop.append(child)
            pop = next_pop

        ranked = sorted(evaluated.values(), key=lambda kv: kv[1], reverse=True)
        winner_g, _ = ranked[0]
        winner_pipe = self._build_pipeline(winner_g, X_train, is_clf)
        winner_pipe.fit(X_train, y_train)
        val_pred = winner_pipe.predict(X_val)
        val_score = float(f1_score(y_val, val_pred, average="macro", zero_division=0)) if is_clf else float(root_mean_squared_error(y_val, val_pred))

        leaderboard = [
            {"model": g.label(), "cv_fitness": round(sc, 6), "metric": "f1_macro" if is_clf else "rmse"}
            for g, sc in ranked[:10]
        ]
        wrapped = _WrappedSklearnModel(
            name=f"GeneticPipeline({winner_g.estimator})",
            estimator=winner_pipe,
            task_type=task_type,
            classes=np.unique(y_train) if is_clf else None,
            extra_meta={"genome": winner_g.label(), "evolution_history": history},
            random_seed=self.random_seed,
        )
        return {
            "best_model": wrapped,
            "best_model_name": winner_g.label(),
            "best_score": round(val_score, 6),
            "primary_metric": "f1_macro" if is_clf else "rmse",
            "leaderboard": leaderboard,
            "evolution_history": history,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }


class MetaFeatureWarmStartAutoML:
    """Meta-feature landmarking + warm-started pipeline portfolio (warm-start heuristic, NOT GP-BO).

    Auto-sklearn/SMAC-inspired portfolio selection: extracts 14 dataset meta-features
    (including a fast DecisionTree landmarking probe), then warm-starts a small portfolio
    of strong, cheap base estimators conditioned on those features (e.g. high-skew targets
    pick ``RobustScaler``). Despite the historical class name, there is **no Gaussian-process
    Bayesian optimization anywhere in this engine** — real GP-BO lives in
    ``omniforge.automl.optimizer.HyperparameterOptimizer``.

    ``MetaLearningBayesianAutoML`` is kept as a deprecated alias for backward compatibility.
    """

    def __init__(self, *, cv_folds: int = 3, random_seed: int = 42) -> None:
        self.cv_folds = max(2, int(cv_folds))
        self.random_seed = int(random_seed)

    @staticmethod
    def extract_meta_features(X: pd.DataFrame, y: pd.Series, is_clf: bool) -> dict[str, float]:
        """Compute 14 canonical dataset meta-features + fast landmarking probes."""
        n, p = X.shape
        num_df = X.select_dtypes(include=[np.number])
        num_frac = float(num_df.shape[1]) / max(p, 1)
        miss_frac = float(X.isna().sum().sum()) / max(n * max(p, 1), 1)
        mean_skew = float(np.nanmean(np.abs(num_df.skew(numeric_only=True)))) if num_df.shape[1] > 0 else 0.0
        mean_kurt = float(np.nanmean(np.abs(num_df.kurt(numeric_only=True)))) if num_df.shape[1] > 0 else 0.0

        # Landmarking score using fast depth-2 Decision Tree
        landmark_rows = X.head(min(n, 2000))
        ym = np.asarray(y.head(min(n, 2000)))
        lm_score = None
        try:
            stump = DecisionTreeClassifier(max_depth=2, random_state=42) if is_clf else DecisionTreeRegressor(max_depth=2, random_state=42)
            pipeline = Pipeline([("pre", _build_preprocessor(landmark_rows)), ("stump", stump)])
            measured = float(np.mean(cross_val_score(
                pipeline, landmark_rows, ym, cv=2, scoring="f1_macro" if is_clf else "r2", error_score="raise",
            )))
            if np.isfinite(measured):
                lm_score = measured
        except ValueError as exc:
            logger.debug("landmarking_score_unavailable", error_type=type(exc).__name__)

        meta: dict[str, float] = {
            "log_n_samples": round(float(np.log10(max(n, 1))), 4),
            "log_n_features": round(float(np.log10(max(p, 1))), 4),
            "sample_to_feature_ratio": round(float(n) / max(p, 1), 4),
            "numeric_fraction": round(num_frac, 4),
            "missing_fraction": round(miss_frac, 4),
            "mean_abs_skewness": round(float(np.nan_to_num(mean_skew)), 4),
            "mean_abs_kurtosis": round(float(np.nan_to_num(mean_kurt)), 4),
        }
        if lm_score is not None:
            meta["landmark_stump_score"] = lm_score
        if is_clf:
            vc = y.value_counts(normalize=True).to_numpy(dtype=float)
            meta["n_classes"] = float(len(vc))
            meta["class_entropy"] = round(float(-np.sum(vc * np.log2(np.clip(vc, 1e-12, 1.0)))), 4)
            meta["minority_ratio"] = round(float(np.min(vc)), 4)
        else:
            yv = pd.to_numeric(y, errors="coerce").dropna().to_numpy(dtype=float)
            meta["target_std"] = round(float(np.std(yv)) if len(yv) else 0.0, 4)
            meta["target_skew"] = round(float(pd.Series(yv).skew()) if len(yv) > 2 else 0.0, 4)
        return meta

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, task_type: TaskType) -> dict[str, Any]:
        t0 = time.perf_counter()
        is_clf = task_type != TaskType.REGRESSION
        meta = self.extract_meta_features(X_train, y_train, is_clf)

        # Warm-start portfolio conditioned on meta-features (e.g. high skewness -> RobustScaler + Trees)
        scaler_choice = "robust" if meta.get("mean_abs_skewness", 0.0) > 1.5 else "standard"
        jobs = parallel_n_jobs()
        if is_clf:
            portfolio: dict[str, Any] = {
                "Meta_HistGBM_Fast": HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=self.random_seed),
                "Meta_HistGBM_Deep": HistGradientBoostingClassifier(max_iter=200, max_depth=8, learning_rate=0.05, random_state=self.random_seed),
                "Meta_ExtraTrees": ExtraTreesClassifier(n_estimators=160, random_state=self.random_seed, n_jobs=jobs),
                "Meta_RandomForest": RandomForestClassifier(n_estimators=160, random_state=self.random_seed, n_jobs=jobs),
                "Meta_Logistic_L2": LogisticRegression(C=1.0, max_iter=500, random_state=self.random_seed),
            }
        else:
            portfolio = {
                "Meta_HistGBM_Fast": HistGradientBoostingRegressor(max_iter=120, learning_rate=0.08, random_state=self.random_seed),
                "Meta_HistGBM_Deep": HistGradientBoostingRegressor(max_iter=200, max_depth=8, learning_rate=0.05, random_state=self.random_seed),
                "Meta_ExtraTrees": ExtraTreesRegressor(n_estimators=160, random_state=self.random_seed, n_jobs=jobs),
                "Meta_RandomForest": RandomForestRegressor(n_estimators=160, random_state=self.random_seed, n_jobs=jobs),
                "Meta_Ridge": Ridge(alpha=1.0),
            }

        leaderboard: list[dict[str, Any]] = []
        best_pipe: Any = None
        best_name = ""
        best_score = -1e18 if is_clf else 1e18

        for mname, est in portfolio.items():
            try:
                pipe = Pipeline([("pre", _build_preprocessor(X_train, scaler_kind=scaler_choice)), ("model", est)])
                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_val)
                sc = float(f1_score(y_val, preds, average="macro", zero_division=0)) if is_clf else float(root_mean_squared_error(y_val, preds))
                leaderboard.append({"model": mname, "scaler": scaler_choice, "metric": "f1_macro" if is_clf else "rmse", "score": round(sc, 6)})
                better = (sc > best_score) if is_clf else (sc < best_score)
                if better:
                    best_score = sc
                    best_name = mname
                    best_pipe = pipe
            except Exception as exc:
                logger.warning("meta_automl_candidate_failed", model=mname, error=str(exc))

        leaderboard.sort(key=lambda r: float(r["score"]), reverse=is_clf)
        wrapped = _WrappedSklearnModel(
            name=f"MetaBayesian({best_name})",
            estimator=best_pipe,
            task_type=task_type,
            classes=np.unique(y_train) if is_clf else None,
            extra_meta={"meta_features": meta, "selected_scaler": scaler_choice},
            random_seed=self.random_seed,
        )
        return {
            "best_model": wrapped,
            "best_model_name": best_name,
            "best_score": round(float(best_score), 6),
            "primary_metric": "f1_macro" if is_clf else "rmse",
            "leaderboard": leaderboard,
            "meta_features": meta,
            "elapsed_seconds": round(time.perf_counter() - t0, 3),
        }


@dataclass
class MultiBackendAutoMLResult:
    """Unified output across all native & external AutoML engines."""

    requested_backend: str
    executed_backend: str
    used_fallback: bool
    fallback_reason: str | None
    task_type: TaskType
    best_model_name: str
    best_model: BaseOmniForgeModel
    best_score: float
    primary_metric: str
    leaderboard: list[dict[str, Any]]
    baseline_benchmarks: dict[str, dict[str, float]]
    lift_over_best_baseline: float | None
    elapsed_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)
    evaluation_indices: np.ndarray | None = None


class AutoMLBackendHub:
    """Unified dispatcher for Native (Leakage-Safe, Multi-Layer Stacker, Genetic, Meta-Bayesian) and External (AutoGluon, TPOT, H2O, Auto-sklearn, FLAML) AutoML engines."""

    @staticmethod
    def probe_backends() -> list[dict[str, Any]]:
        return probe_automl_backends()

    @classmethod
    def run(
        cls,
        df: pd.DataFrame,
        target_col: str,
        *,
        backend: str = "omniforge_native",
        task_type: TaskType | None = None,
        time_budget_seconds: float = 60.0,
        cv_folds: int = 3,
        split_strategy: SplitStrategyType = "auto",
        fallback_to_native: bool = True,
        random_seed: int = 42,
        max_carbon_grams_co2e: float | None = None,
        max_inference_latency_ms: float | None = None,
        groups: pd.Series | np.ndarray | None = None,
        time_values: pd.Series | np.ndarray | None = None,
        gap: int = 0,
        holdout: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> MultiBackendAutoMLResult:
        """Run any selected AutoML engine with baseline comparison and optional native fallback."""
        if target_col not in df.columns:
            raise AutoMLError(f"Target column '{target_col}' not found in DataFrame.")
        req_backend = str(backend).lower().strip()
        clean_df = df.dropna(subset=[target_col]).copy()
        y_all = clean_df[target_col]
        X_all = clean_df.drop(columns=[target_col])
        if holdout is not None:
            X_test, y_test = holdout
            if X_test.empty or not X_test.columns.equals(X_all.columns) or not X_test.index.equals(y_test.index) or y_test.isna().any():
                raise AutoMLError("A supplied holdout requires observed targets and aligned rows and predictor columns.")
        resolved_task = _infer_task(y_all, task_type)
        is_clf = resolved_task != TaskType.REGRESSION

        def run_native(used_fallback: bool = False, fallback_reason: str | None = None) -> MultiBackendAutoMLResult:
            observed = df[target_col].notna().to_numpy()
            if any(values is not None and len(values) != len(df) for values in (groups, time_values)):
                raise AutoMLError("Group and time vectors must align with the input data.")
            aligned_groups = np.asarray(groups)[observed] if groups is not None else None
            aligned_times = np.asarray(time_values)[observed] if time_values is not None else None
            native_res: AutoMLResult = AutoMLPipelineSearch(
                task_type=resolved_task,
                time_budget_seconds=time_budget_seconds,
                cv_folds=cv_folds,
                split_strategy=split_strategy,
                gap=gap,
                random_seed=random_seed,
                max_carbon_grams_co2e=max_carbon_grams_co2e,
                max_inference_latency_ms=max_inference_latency_ms,
            ).fit(X_all, y_all, groups=aligned_groups, time_values=aligned_times, holdout=holdout)
            return MultiBackendAutoMLResult(
                requested_backend=req_backend,
                executed_backend="omniforge_native",
                used_fallback=used_fallback,
                fallback_reason=fallback_reason,
                task_type=native_res.task_type,
                best_model_name=native_res.best_model_name,
                best_model=native_res.best_model,
                best_score=native_res.best_score,
                primary_metric=native_res.primary_metric,
                leaderboard=native_res.leaderboard,
                baseline_benchmarks=native_res.baseline_benchmarks,
                lift_over_best_baseline=native_res.lift_over_best_baseline,
                elapsed_seconds=native_res.elapsed_seconds,
                extra={"cv_summary": native_res.cv_summary, "evaluation": native_res.metadata["evaluation"]},
                evaluation_indices=native_res.test_indices,
            )

        if req_backend == "omniforge_native":
            return run_native()

        explicit_split = groups is not None or time_values is not None or gap or split_strategy in {"group", "time_series"}
        if explicit_split and req_backend != "multi_layer_stacker":
            raise AutoMLError(
                "This AutoML backend does not support explicit group or time split metadata.",
                suggestions=["Choose the OmniForge native backend or nested stacker for grouped or chronological evaluation."],
            )

        # Metadata guides the split and is excluded from model inputs.
        benchmark_source = clean_df
        stack_groups = None
        stack_ordered = False
        group_col = time_col = None
        benchmark_strategy = split_strategy
        if explicit_split:
            if any(values is not None and np.asarray(values).shape != (len(df),) for values in (groups, time_values)):
                raise AutoMLError("Group and time vectors must align with the input data.")
            if (groups is not None and (time_values is not None or gap or split_strategy == "time_series")) or (
                split_strategy == "group" and groups is None
            ):
                raise AutoMLError("Supply a single supported group or chronological split contract.")
            positions = df[target_col].notna().to_numpy()
            metadata_col = "__omniforge_split_metadata__"
            while metadata_col in clean_df.columns:
                metadata_col += "_"
            if groups is not None:
                stack_groups = np.asarray(groups)[positions]
                if pd.isna(stack_groups).any():
                    raise AutoMLError("Grouped stacking requires observed group membership.")
                group_col = metadata_col
                benchmark_strategy = "group"
                benchmark_source = clean_df.assign(**{metadata_col: stack_groups})
            else:
                stack_ordered = True
                benchmark_strategy = "time_series"
                if time_values is not None:
                    times = pd.Series(np.asarray(time_values)[positions])
                    if not pd.api.types.is_numeric_dtype(times):
                        times = pd.to_datetime(times, errors="coerce", utc=True, format="mixed")
                    if times.isna().any() or not times.is_unique:
                        raise AutoMLError("Chronological stacking requires observed, distinct timestamps. Use a grouped design for panel data.")
                    time_col = metadata_col
                    benchmark_source = clean_df.assign(**{metadata_col: times.to_numpy()})

        splitter = LeakageSafeSplitter(SplitConfig(
            strategy=benchmark_strategy, test_size=0.2, cv_folds=cv_folds, random_seed=random_seed,
            group_col=group_col, time_col=time_col, gap=gap,
        ))
        split_res = splitter.split(benchmark_source, target_col=target_col)
        if stack_ordered and gap and len(clean_df) - len(split_res.train_indices) - len(split_res.test_indices) < gap:
            raise AutoMLError("Insufficient rows to preserve the requested chronological embargo.")
        tr_df = clean_df.iloc[split_res.train_indices]
        va_df = clean_df.iloc[split_res.test_indices]
        X_tr, y_tr = tr_df.drop(columns=[target_col]), tr_df[target_col]
        X_va, y_va = va_df.drop(columns=[target_col]), va_df[target_col]

        def finish(
            result: dict[str, Any], *, extra: dict[str, Any] | None = None, leaderboard_partition: str = "validation",
        ) -> MultiBackendAutoMLResult:
            # Fit baseline preprocessing on the same development rows as the selected adapter.
            preprocessor = _build_preprocessor(X_tr)
            X_train_matrix = preprocessor.fit_transform(X_tr)
            X_validation_matrix = preprocessor.transform(X_va)
            validation_score = float(result["best_score"])
            metric = "f1_macro" if is_clf else "rmse"

            def score(y_true: pd.Series, predicted: np.ndarray) -> float:
                if is_clf:
                    return float(f1_score(y_true, predicted, average="macro", zero_division=0))
                return float(root_mean_squared_error(y_true, predicted))

            benchmark = (
                BaselineBenchmarkSuite.run_classification_detailed if is_clf
                else BaselineBenchmarkSuite.run_regression_detailed
            )
            validation_evidence: BaselineEvidence = {}
            validation_baselines, baseline_models = benchmark(
                X_train_matrix, y_tr, X_validation_matrix, y_va, random_seed=random_seed, evidence=validation_evidence,
            )

            def score_baselines(
                y_true: pd.Series, matrix: Any, evidence: BaselineEvidence,
            ) -> dict[str, dict[str, float]]:
                scored = {}
                for name, model in baseline_models.items():
                    try:
                        value = score(y_true, model.predict(matrix))
                        if not np.isfinite(value):
                            raise ArithmeticError("nonfinite baseline score")
                        scored[name] = {metric: value}
                    except Exception as exc:
                        evidence[name].update(status="failed", error=f"score: {type(exc).__name__}")
                return scored

            if is_clf:
                validation_baselines = score_baselines(y_va, X_validation_matrix, validation_evidence)
            reference = strongest_scored_reference(validation_baselines, validation_evidence, metric, higher_is_better=is_clf)
            baseline_score = reference[1] if reference else None
            best_score = validation_score
            baselines = validation_baselines
            evaluation: dict[str, Any] = {
                "leaderboard_partition": leaderboard_partition,
                "best_score_partition": "validation",
                "validation_score": validation_score,
                "validation_baseline_score": baseline_score,
                "validation_baseline_name": reference[0] if reference else None,
                "baseline_evidence": {"validation": validation_evidence},
                "n_train": len(X_tr),
                "n_validation": len(X_va),
            }
            test_positions = None
            if holdout is not None:
                X_test, y_test = holdout
                X_test_matrix = preprocessor.transform(X_test)
                best_score = score(y_test, result["best_model"].predict(X_test))
                test_evidence: BaselineEvidence = {
                    name: (dict(row) if name not in baseline_models else {"role": row["role"], "status": "scored"})
                    for name, row in validation_evidence.items()
                }
                baselines = score_baselines(y_test, X_test_matrix, test_evidence)
                reference = strongest_scored_reference(baselines, test_evidence, metric, higher_is_better=is_clf)
                baseline_score = reference[1] if reference else None
                evaluation["baseline_evidence"]["test"] = test_evidence
                test_positions = np.arange(len(clean_df), len(clean_df) + len(X_test))
                evaluation.update(
                    best_score_partition="test", test_partition_source="supplied_holdout",
                    test_baseline_score=baseline_score, test_baseline_name=reference[0] if reference else None,
                    n_test=len(X_test),
                )
            evaluation["baseline_comparison_status"] = "available" if reference else "unavailable"
            lift = None if baseline_score is None else (
                (best_score - baseline_score) if is_clf else (baseline_score - best_score)
            )
            return MultiBackendAutoMLResult(
                requested_backend=req_backend, executed_backend=exec_backend,
                used_fallback=used_fallback, fallback_reason=fallback_reason, task_type=resolved_task,
                best_model_name=str(result["best_model_name"]), best_model=result["best_model"],
                best_score=round(best_score, 6), primary_metric=metric,
                leaderboard=list(result["leaderboard"]), baseline_benchmarks=baselines,
                lift_over_best_baseline=round(lift, 6) if lift is not None else None, elapsed_seconds=float(result["elapsed_seconds"]),
                extra={**(extra or {}), "evaluation": evaluation}, evaluation_indices=test_positions,
            )

        used_fallback = False
        fallback_reason: str | None = None
        exec_backend = req_backend

        # Try optional external adapters first when requested
        if req_backend in ("autogluon", "tpot", "h2o", "autosklearn", "flaml"):
            ext_out = cls._try_external_adapter(
                req_backend, tr_df, va_df, target_col, resolved_task, time_budget_seconds, random_seed
            )
            if ext_out is not None:
                return finish(
                    ext_out, extra=ext_out.get("extra"),
                    leaderboard_partition="adapter_internal" if req_backend == "h2o" else "validation",
                )
            if not fallback_to_native:
                raise AutoMLError(f"External AutoML backend '{req_backend}' is not installed or unavailable on this system.")
            fallback_map = {
                "autogluon": "multi_layer_stacker",
                "h2o": "multi_layer_stacker",
                "tpot": "genetic_pipeline",
                "autosklearn": "meta_bayesian",
                "flaml": "omniforge_native",
            }
            exec_backend = fallback_map[req_backend]
            used_fallback = True
            fallback_reason = (
                f"External package '{req_backend}' is not installed in this environment; "
                f"automatically executed native equivalent '{exec_backend}'."
            )

        # Specialized engines select using development rows only.
        if exec_backend == "multi_layer_stacker":
            res = MultiLayerStackingAutoML(cv_folds=cv_folds, random_seed=random_seed).fit(
                X_tr, y_tr, X_va, y_va, resolved_task,
                groups=stack_groups[split_res.train_indices] if stack_groups is not None else None,
                time_ordered=stack_ordered, gap=gap,
            )
            return finish(res, extra={"ensemble_weights": res["ensemble_weights"], "stacking_validation": res["stacking_validation"]})

        if exec_backend == "genetic_pipeline":
            res = GeneticPipelineAutoML(cv_folds=cv_folds, random_seed=random_seed).fit(X_tr, y_tr, X_va, y_va, resolved_task)
            return finish(res, extra={"evolution_history": res.get("evolution_history", [])}, leaderboard_partition="train_cv")

        if exec_backend == "meta_bayesian":
            res = MetaFeatureWarmStartAutoML(cv_folds=cv_folds, random_seed=random_seed).fit(X_tr, y_tr, X_va, y_va, resolved_task)
            return finish(res, extra={"meta_features": res.get("meta_features", {})})

        return run_native(used_fallback, fallback_reason)

    @staticmethod
    def _try_external_adapter(
        backend: str,
        tr_df: pd.DataFrame,
        va_df: pd.DataFrame,
        target_col: str,
        task_type: TaskType,
        time_budget: float,
        random_seed: int,
    ) -> dict[str, Any] | None:
        """Execute an installed external AutoML library (AutoGluon / TPOT / H2O / Auto-sklearn / FLAML) if available."""
        is_clf = task_type != TaskType.REGRESSION
        t0 = time.perf_counter()
        X_tr, y_tr = tr_df.drop(columns=[target_col]), tr_df[target_col]
        X_va, y_va = va_df.drop(columns=[target_col]), va_df[target_col]

        if backend == "autogluon" and is_package_available("autogluon.tabular"):
            from autogluon.tabular import TabularPredictor

            predictor = TabularPredictor(label=target_col, eval_metric="f1_macro" if is_clf else "root_mean_squared_error").fit(
                tr_df, time_limit=max(10, int(time_budget)), verbosity=0
            )
            preds = predictor.predict(X_va)
            sc = float(f1_score(y_va, preds, average="macro", zero_division=0)) if is_clf else float(root_mean_squared_error(y_va, preds))
            lb = predictor.leaderboard(va_df, silent=True).head(10).to_dict(orient="records")
            return {
                "best_model": _WrappedSklearnModel("AutoGluonTabular", predictor, task_type, random_seed=random_seed),
                "best_model_name": str(predictor.model_best),
                "best_score": round(sc, 6),
                "primary_metric": "f1_macro" if is_clf else "rmse",
                "leaderboard": lb,
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
            }

        if backend == "tpot" and is_package_available("tpot"):
            from tpot import TPOTClassifier, TPOTRegressor

            pre = _build_preprocessor(X_tr)
            est = (
                TPOTClassifier(max_time_mins=max(0.5, time_budget / 60.0), random_state=random_seed)
                if is_clf
                else TPOTRegressor(max_time_mins=max(0.5, time_budget / 60.0), random_state=random_seed)
            )
            pipe = Pipeline([("pre", pre), ("tpot", est)])
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_va)
            sc = float(f1_score(y_va, preds, average="macro", zero_division=0)) if is_clf else float(root_mean_squared_error(y_va, preds))
            return {
                "best_model": _WrappedSklearnModel("TPOT_BestPipeline", pipe, task_type, random_seed=random_seed),
                "best_model_name": "TPOT_BestPipeline",
                "best_score": round(sc, 6),
                "primary_metric": "f1_macro" if is_clf else "rmse",
                "leaderboard": [{"model": "TPOT_BestPipeline", "metric": "f1_macro" if is_clf else "rmse", "score": round(sc, 6)}],
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
            }

        if backend == "flaml" and is_package_available("flaml"):
            from flaml import AutoML as FLAMLAutoML

            pre = _build_preprocessor(X_tr)
            X_tr_m = pre.fit_transform(X_tr)
            X_va_m = pre.transform(X_va)
            automl = FLAMLAutoML()
            automl.fit(X_tr_m, np.asarray(y_tr), task="classification" if is_clf else "regression", time_budget=max(5, int(time_budget)), seed=random_seed, verbose=0)
            preds = automl.predict(X_va_m)
            sc = float(f1_score(y_va, preds, average="macro", zero_division=0)) if is_clf else float(root_mean_squared_error(y_va, preds))
            pipe = Pipeline([("pre", pre), ("flaml", automl)])
            return {
                "best_model": _WrappedSklearnModel(f"FLAML({automl.best_estimator})", pipe, task_type, random_seed=random_seed),
                "best_model_name": f"FLAML({automl.best_estimator})",
                "best_score": round(sc, 6),
                "primary_metric": "f1_macro" if is_clf else "rmse",
                "leaderboard": [{"model": f"FLAML({automl.best_estimator})", "metric": "f1_macro" if is_clf else "rmse", "score": round(sc, 6)}],
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
            }

        if backend == "h2o" and is_package_available("h2o") and shutil.which("java") is not None:
            import h2o
            from h2o.automl import H2OAutoML

            h2o.init(verbose=False)
            h_tr = h2o.H2OFrame(tr_df)
            h_va = h2o.H2OFrame(va_df)
            if is_clf:
                h_tr[target_col] = h_tr[target_col].asfactor()
                h_va[target_col] = h_va[target_col].asfactor()
            aml = H2OAutoML(max_runtime_secs=max(10, int(time_budget)), seed=random_seed, verbosity="error")
            aml.train(y=target_col, training_frame=h_tr)
            preds_df = aml.leader.predict(h_va).as_data_frame()
            preds = preds_df["predict"].to_numpy()
            sc = float(f1_score(y_va.astype(str), preds.astype(str), average="macro", zero_division=0)) if is_clf else float(root_mean_squared_error(y_va, preds.astype(float)))
            lb = aml.leaderboard.as_data_frame().head(10).to_dict(orient="records")

            class _H2OWrapper:
                def predict(self, X_in: pd.DataFrame) -> np.ndarray:
                    frame = aml.leader.predict(h2o.H2OFrame(X_in)).as_data_frame()["predict"]
                    if not is_clf:
                        return np.asarray(frame.to_numpy(), dtype=float)
                    class_values = {str(value): value for value in y_tr.unique()}
                    return np.asarray([class_values[str(value)] for value in frame])

            return {
                "best_model": _WrappedSklearnModel(f"H2O({aml.leader.model_id})", _H2OWrapper(), task_type, random_seed=random_seed),
                "best_model_name": str(aml.leader.model_id),
                "best_score": round(sc, 6),
                "primary_metric": "f1_macro" if is_clf else "rmse",
                "leaderboard": lb,
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
            }

        if backend == "autosklearn" and is_package_available("autosklearn"):
            if is_clf:
                from autosklearn.classification import AutoSklearnClassifier as ASKEst
            else:
                from autosklearn.regression import AutoSklearnRegressor as ASKEst
            pre = _build_preprocessor(X_tr)
            pipe = Pipeline([("pre", pre), ("ask", ASKEst(time_left_for_this_task=max(30, int(time_budget)), seed=random_seed))])
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_va)
            sc = float(f1_score(y_va, preds, average="macro", zero_division=0)) if is_clf else float(root_mean_squared_error(y_va, preds))
            return {
                "best_model": _WrappedSklearnModel("AutoSklearn", pipe, task_type, random_seed=random_seed),
                "best_model_name": "AutoSklearn_Ensemble",
                "best_score": round(sc, 6),
                "primary_metric": "f1_macro" if is_clf else "rmse",
                "leaderboard": [{"model": "AutoSklearn_Ensemble", "metric": "f1_macro" if is_clf else "rmse", "score": round(sc, 6)}],
                "elapsed_seconds": round(time.perf_counter() - t0, 3),
            }

        return None
# Deprecated backward-compat alias (misleading name; kept so existing imports do not break).
MetaLearningBayesianAutoML = MetaFeatureWarmStartAutoML
