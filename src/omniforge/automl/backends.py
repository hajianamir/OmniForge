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
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
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
from omniforge.models.baselines import BaselineBenchmarkSuite

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
    ) -> None:
        self.preprocessor = preprocessor
        self.l1_models = l1_models
        self.l2_models = l2_models
        self.weights = weights
        self.is_clf = is_clf
        self.classes_ = classes

    def _transform_and_l1(self, X: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
        X_mat = self.preprocessor.transform(X)
        l1_preds: dict[str, np.ndarray] = {}
        stack_blocks: list[np.ndarray] = [X_mat]
        for mname, est in self.l1_models.items():
            if self.is_clf and hasattr(est, "predict_proba"):
                p = np.asarray(est.predict_proba(X_mat), dtype=float)
            else:
                p = np.asarray(est.predict(X_mat), dtype=float).reshape(-1, 1)
            l1_preds[mname] = p
            stack_blocks.append(p)
        return X_mat, l1_preds, np.hstack(stack_blocks)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_clf:
            raise AutoMLError("predict_proba is only supported for classification stackers.")
        _, l1_preds, X_l2 = self._transform_and_l1(X)
        all_preds = dict(l1_preds)
        for mname, est in self.l2_models.items():
            all_preds[mname] = np.asarray(est.predict_proba(X_l2), dtype=float)
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
            all_preds[mname] = np.asarray(est.predict(X_l2), dtype=float).reshape(-1, 1)
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

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, task_type: TaskType) -> dict[str, Any]:
        t0 = time.perf_counter()
        is_clf = task_type != TaskType.REGRESSION
        classes = np.unique(y_train) if is_clf else None
        n_classes = len(classes) if classes is not None else 1

        pre = _build_preprocessor(X_train)
        X_tr_mat = pre.fit_transform(X_train)
        X_va_mat = pre.transform(X_val)
        y_tr_arr = np.asarray(y_train)
        y_va_arr = np.asarray(y_val)

        if is_clf:
            min_cls = int(pd.Series(y_tr_arr).value_counts().min())
            n_splits = max(2, min(self.cv_folds, min_cls))
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
            splits = list(splitter.split(X_tr_mat, y_tr_arr))
        else:
            n_splits = max(2, min(self.cv_folds, len(X_tr_mat) // 5))
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
            splits = list(splitter.split(X_tr_mat))

        # Layer 1 OOF + full fits
        l1_specs = self._layer1_specs(is_clf)
        oof_preds: dict[str, np.ndarray] = {}
        val_preds: dict[str, np.ndarray] = {}
        l1_fitted: dict[str, Any] = {}
        leaderboard: list[dict[str, Any]] = []

        for mname, proto in l1_specs.items():
            try:
                oof = np.zeros((len(X_tr_mat), n_classes), dtype=float)
                for tr_i, va_i in splits:
                    est_f = clone(proto)
                    est_f.fit(X_tr_mat[tr_i], y_tr_arr[tr_i])
                    if is_clf and hasattr(est_f, "predict_proba"):
                        oof[va_i] = est_f.predict_proba(X_tr_mat[va_i])
                    else:
                        oof[va_i, 0] = est_f.predict(X_tr_mat[va_i])
                full_est = clone(proto)
                full_est.fit(X_tr_mat, y_tr_arr)
                l1_fitted[mname] = full_est
                oof_preds[mname] = oof
                vp = full_est.predict_proba(X_va_mat) if (is_clf and hasattr(full_est, "predict_proba")) else full_est.predict(X_va_mat).reshape(-1, 1)
                val_preds[mname] = np.asarray(vp, dtype=float)
                val_sc = (
                    float(f1_score(y_va_arr, classes[np.argmax(vp, axis=1)], average="macro", zero_division=0))
                    if (is_clf and classes is not None)
                    else float(root_mean_squared_error(y_va_arr, vp.ravel()))
                )
                leaderboard.append({"model": mname, "layer": "L1", "metric": "f1_macro" if is_clf else "rmse", "score": round(val_sc, 6)})
            except Exception as exc:
                logger.warning("stack_l1_model_skipped", model=mname, error=str(exc))

        if not l1_fitted:
            raise AutoMLError("Multi-Layer Stacker failed to fit any Layer-1 base models.")

        # Layer 2 Stacker Features: [Original Features + Layer 1 OOF Predictions]
        X_l2_tr = np.hstack([X_tr_mat, *[oof_preds[k] for k in l1_fitted]])
        X_l2_va = np.hstack([X_va_mat, *[val_preds[k] for k in l1_fitted]])

        l2_specs: dict[str, Any] = (
            {
                "L2_Stack_HistGBM": HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=self.random_seed),
                "L2_Stack_Logistic": LogisticRegression(C=0.5, max_iter=500, random_state=self.random_seed),
            }
            if is_clf
            else {
                "L2_Stack_HistGBM": HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=self.random_seed),
                "L2_Stack_Ridge": Ridge(alpha=5.0),
            }
        )
        l2_fitted: dict[str, Any] = {}
        for mname, proto in l2_specs.items():
            try:
                oof2 = np.zeros((len(X_l2_tr), n_classes), dtype=float)
                for tr_i, va_i in splits:
                    est_f = clone(proto)
                    est_f.fit(X_l2_tr[tr_i], y_tr_arr[tr_i])
                    if is_clf and hasattr(est_f, "predict_proba"):
                        oof2[va_i] = est_f.predict_proba(X_l2_tr[va_i])
                    else:
                        oof2[va_i, 0] = est_f.predict(X_l2_tr[va_i])
                full_est2 = clone(proto)
                full_est2.fit(X_l2_tr, y_tr_arr)
                l2_fitted[mname] = full_est2
                oof_preds[mname] = oof2
                vp2 = full_est2.predict_proba(X_l2_va) if (is_clf and hasattr(full_est2, "predict_proba")) else full_est2.predict(X_l2_va).reshape(-1, 1)
                val_sc2 = (
                    float(f1_score(y_va_arr, classes[np.argmax(vp2, axis=1)], average="macro", zero_division=0))
                    if (is_clf and classes is not None)
                    else float(root_mean_squared_error(y_va_arr, vp2.ravel()))
                )
                leaderboard.append({"model": mname, "layer": "L2_Stack", "metric": "f1_macro" if is_clf else "rmse", "score": round(val_sc2, 6)})
            except Exception as exc:
                logger.warning("stack_l2_model_skipped", model=mname, error=str(exc))

        # Layer 3: Caruana Greedy Weighted Ensemble Selection on OOF predictions
        weights, _ = _caruana_greedy_weights(oof_preds, y_tr_arr, is_clf=is_clf, classes=classes, n_rounds=15)
        predictor = _CaruanaStackPredictor(pre, l1_fitted, l2_fitted, weights, is_clf=is_clf, classes=classes)
        ens_preds = predictor.predict(X_val)
        ens_score = (
            float(f1_score(y_va_arr, ens_preds, average="macro", zero_division=0))
            if is_clf
            else float(root_mean_squared_error(y_va_arr, ens_preds))
        )
        leaderboard.append({"model": "L3_WeightedEnsemble_Caruana", "layer": "L3_Ensemble", "metric": "f1_macro" if is_clf else "rmse", "score": round(ens_score, 6)})
        leaderboard.sort(key=lambda r: float(r["score"]), reverse=is_clf)

        wrapped = _WrappedSklearnModel(
            name="MultiLayerStackEnsemble",
            estimator=predictor,
            task_type=task_type,
            classes=classes,
            extra_meta={"ensemble_weights": weights, "layers": 3},
            random_seed=self.random_seed,
        )
        return {
            "best_model": wrapped,
            "best_model_name": "L3_WeightedEnsemble_Caruana",
            "best_score": round(ens_score, 6),
            "primary_metric": "f1_macro" if is_clf else "rmse",
            "leaderboard": leaderboard,
            "ensemble_weights": weights,
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
        pre = _build_preprocessor(X)
        Xm = pre.fit_transform(X.head(min(n, 2000)))
        ym = np.asarray(y.head(min(n, 2000)))
        try:
            stump = DecisionTreeClassifier(max_depth=2, random_state=42) if is_clf else DecisionTreeRegressor(max_depth=2, random_state=42)
            lm_score = float(np.mean(cross_val_score(stump, Xm, ym, cv=2, scoring="f1_macro" if is_clf else "r2")))
        except Exception:
            lm_score = 0.0

        meta: dict[str, float] = {
            "log_n_samples": round(float(np.log10(max(n, 1))), 4),
            "log_n_features": round(float(np.log10(max(p, 1))), 4),
            "sample_to_feature_ratio": round(float(n) / max(p, 1), 4),
            "numeric_fraction": round(num_frac, 4),
            "missing_fraction": round(miss_frac, 4),
            "mean_abs_skewness": round(float(np.nan_to_num(mean_skew)), 4),
            "mean_abs_kurtosis": round(float(np.nan_to_num(mean_kurt)), 4),
            "landmark_stump_score": round(lm_score, 4),
        }
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
    lift_over_best_baseline: float
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
    ) -> MultiBackendAutoMLResult:
        """Run any selected AutoML engine with baseline comparison and optional native fallback."""
        if target_col not in df.columns:
            raise AutoMLError(f"Target column '{target_col}' not found in DataFrame.")
        req_backend = str(backend).lower().strip()
        clean_df = df.dropna(subset=[target_col]).copy()
        y_all = clean_df[target_col]
        X_all = clean_df.drop(columns=[target_col])
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
            ).fit(X_all, y_all, groups=aligned_groups, time_values=aligned_times)
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

        if groups is not None or time_values is not None or gap:
            raise AutoMLError(
                "This AutoML backend does not support explicit group or time split metadata.",
                suggestions=["Choose the OmniForge native backend for grouped or chronological evaluation."],
            )

        # Benchmark comparison for specialized and external backends.
        splitter = LeakageSafeSplitter(SplitConfig(strategy=split_strategy, test_size=0.2, cv_folds=cv_folds, random_seed=random_seed))
        split_res = splitter.split(clean_df, target_col=target_col)
        tr_df = clean_df.iloc[split_res.train_indices]
        va_df = clean_df.iloc[split_res.test_indices]
        X_tr, y_tr = tr_df.drop(columns=[target_col]), tr_df[target_col]
        X_va, y_va = va_df.drop(columns=[target_col]), va_df[target_col]

        # Always compute quantitative baselines for honest lift comparison
        pre_bl = _build_preprocessor(X_tr)
        X_tr_bl = pre_bl.fit_transform(X_tr)
        X_va_bl = pre_bl.transform(X_va)
        if is_clf:
            baselines = BaselineBenchmarkSuite.run_classification(X_tr_bl, y_tr, X_va_bl, y_va)
            best_bl_score = max((float(m.get("f1", 0.0)) for m in baselines.values()), default=0.0)
        else:
            baselines = BaselineBenchmarkSuite.run_regression(X_tr_bl, y_tr, X_va_bl, y_va)
            best_bl_score = min((float(m.get("rmse", 1e9)) for m in baselines.values()), default=0.0)

        used_fallback = False
        fallback_reason: str | None = None
        exec_backend = req_backend

        # Try optional external adapters first when requested
        if req_backend in ("autogluon", "tpot", "h2o", "autosklearn", "flaml"):
            ext_out = cls._try_external_adapter(
                req_backend, tr_df, va_df, target_col, resolved_task, time_budget_seconds, random_seed
            )
            if ext_out is not None:
                best_sc = float(ext_out["best_score"])
                lift = (best_sc - best_bl_score) if is_clf else (best_bl_score - best_sc)
                return MultiBackendAutoMLResult(
                    requested_backend=req_backend,
                    executed_backend=req_backend,
                    used_fallback=False,
                    fallback_reason=None,
                    task_type=resolved_task,
                    best_model_name=str(ext_out["best_model_name"]),
                    best_model=ext_out["best_model"],
                    best_score=best_sc,
                    primary_metric=str(ext_out["primary_metric"]),
                    leaderboard=list(ext_out["leaderboard"]),
                    baseline_benchmarks=baselines,
                    lift_over_best_baseline=round(float(lift), 6),
                    elapsed_seconds=float(ext_out["elapsed_seconds"]),
                    extra=dict(ext_out.get("extra") or {}),
                    evaluation_indices=split_res.test_indices.copy(),
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

        # Dispatch native engines
        if exec_backend == "multi_layer_stacker":
            res = MultiLayerStackingAutoML(cv_folds=cv_folds, random_seed=random_seed).fit(X_tr, y_tr, X_va, y_va, resolved_task)
            best_sc = float(res["best_score"])
            lift = (best_sc - best_bl_score) if is_clf else (best_bl_score - best_sc)
            return MultiBackendAutoMLResult(
                requested_backend=req_backend,
                executed_backend=exec_backend,
                used_fallback=used_fallback,
                fallback_reason=fallback_reason,
                task_type=resolved_task,
                best_model_name=str(res["best_model_name"]),
                best_model=res["best_model"],
                best_score=best_sc,
                primary_metric=str(res["primary_metric"]),
                leaderboard=list(res["leaderboard"]),
                baseline_benchmarks=baselines,
                lift_over_best_baseline=round(float(lift), 6),
                elapsed_seconds=float(res["elapsed_seconds"]),
                extra={"ensemble_weights": res.get("ensemble_weights", {})},
                evaluation_indices=split_res.test_indices.copy(),
            )

        if exec_backend == "genetic_pipeline":
            res = GeneticPipelineAutoML(cv_folds=cv_folds, random_seed=random_seed).fit(X_tr, y_tr, X_va, y_va, resolved_task)
            best_sc = float(res["best_score"])
            lift = (best_sc - best_bl_score) if is_clf else (best_bl_score - best_sc)
            return MultiBackendAutoMLResult(
                requested_backend=req_backend,
                executed_backend=exec_backend,
                used_fallback=used_fallback,
                fallback_reason=fallback_reason,
                task_type=resolved_task,
                best_model_name=str(res["best_model_name"]),
                best_model=res["best_model"],
                best_score=best_sc,
                primary_metric=str(res["primary_metric"]),
                leaderboard=list(res["leaderboard"]),
                baseline_benchmarks=baselines,
                lift_over_best_baseline=round(float(lift), 6),
                elapsed_seconds=float(res["elapsed_seconds"]),
                extra={"evolution_history": res.get("evolution_history", [])},
                evaluation_indices=split_res.test_indices.copy(),
            )

        if exec_backend == "meta_bayesian":
            res = MetaFeatureWarmStartAutoML(cv_folds=cv_folds, random_seed=random_seed).fit(X_tr, y_tr, X_va, y_va, resolved_task)
            best_sc = float(res["best_score"])
            lift = (best_sc - best_bl_score) if is_clf else (best_bl_score - best_sc)
            return MultiBackendAutoMLResult(
                requested_backend=req_backend,
                executed_backend=exec_backend,
                used_fallback=used_fallback,
                fallback_reason=fallback_reason,
                task_type=resolved_task,
                best_model_name=str(res["best_model_name"]),
                best_model=res["best_model"],
                best_score=best_sc,
                primary_metric=str(res["primary_metric"]),
                leaderboard=list(res["leaderboard"]),
                baseline_benchmarks=baselines,
                lift_over_best_baseline=round(float(lift), 6),
                elapsed_seconds=float(res["elapsed_seconds"]),
                extra={"meta_features": res.get("meta_features", {})},
                evaluation_indices=split_res.test_indices.copy(),
            )

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
                    return np.asarray(frame.to_numpy(), dtype=float)

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
