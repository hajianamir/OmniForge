"""Enterprise Leakage-Safe AutoML Pipeline Search with Stratified/Group/TimeSeries CV, ColumnTransformer Pipelines, Baselines, HPO & Ensembling."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from omniforge.automl.optimizer import HyperparameterOptimizer
from omniforge.core.carbon import CarbonTracker
from omniforge.core.exceptions import AutoMLError
from omniforge.core.logging import get_logger
from omniforge.core.types import CarbonReport, TaskType
from omniforge.data.splitting import (
    LeakageAuditReport,
    LeakageSafePipelineBuilder,
    LeakageSafeSplitter,
    SplitConfig,
    SplitStrategyType,
)
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.baselines import BaselineBenchmarkSuite
from omniforge.models.classical import ClassicalMLModel
from omniforge.models.deep.cnn import CNN1DModel
from omniforge.models.deep.mlp import MLPModel
from omniforge.models.deep.transformers import TabularTransformerModel
from omniforge.models.ensemble import VotingEnsembleModel

logger = get_logger(__name__)


@dataclass
class AutoMLResult:
    """Comprehensive deliverable returned by :class:`AutoMLPipelineSearch`."""

    task_type: TaskType
    best_model_name: str
    best_model: BaseOmniForgeModel
    best_score: float
    primary_metric: str
    leaderboard: list[dict[str, Any]]
    baseline_benchmarks: dict[str, dict[str, float]]
    lift_over_best_baseline: float
    carbon_report: CarbonReport | None = None
    elapsed_seconds: float = 0.0
    leakage_audit: LeakageAuditReport | None = None
    cv_summary: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # C3 Green-AI: budget constraints & Pareto frontier.
    max_carbon_grams_co2e: float | None = None
    max_inference_latency_ms: float | None = None
    pareto_frontier: list[dict[str, Any]] = field(default_factory=list)
    test_indices: np.ndarray | None = None


class _ChampionEstimator(BaseEstimator):
    """Expose the exact fitted OmniForge model through a scikit-learn pipeline."""

    def __init__(self, model: BaseOmniForgeModel) -> None:
        self.model = model

    def __sklearn_is_fitted__(self) -> bool:
        return self.model.is_fitted_

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> _ChampionEstimator:
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def score(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> float:
        return self.model.score(X, y)

    @property
    def classes_(self) -> np.ndarray | None:
        return self.model.classes_


class AutoMLPipelineSearch:
    """End-to-end leakage-safe AutoML engine supporting Stratified, Group, and TimeSeriesSplit CV with sklearn Pipelines."""

    def __init__(
        self,
        *,
        task_type: TaskType | None = None,
        time_budget_seconds: float = 90.0,
        tune_top_k: int = 2,
        tune_trials: int = 8,
        enable_ensembling: bool = True,
        include_deep: bool = True,
        split_strategy: SplitStrategyType = "auto",
        cv_folds: int = 3,
        group_col: str | None = None,
        time_col: str | None = None,
        gap: int = 0,
        use_cv_pipeline: bool = True,
        select_k_features: int | None = None,
        random_seed: int = 42,
        max_carbon_grams_co2e: float | None = None,
        max_inference_latency_ms: float | None = None,
    ) -> None:
        self.task_type = task_type
        self.time_budget_seconds = float(time_budget_seconds)
        self.tune_top_k = max(1, int(tune_top_k))
        self.tune_trials = max(2, int(tune_trials))
        self.enable_ensembling = enable_ensembling
        self.include_deep = include_deep
        self.split_strategy: SplitStrategyType = split_strategy
        self.cv_folds = max(2, int(cv_folds))
        self.group_col = group_col
        self.time_col = time_col
        self.gap = max(0, int(gap))
        self.use_cv_pipeline = use_cv_pipeline
        self.select_k_features = select_k_features
        self.random_seed = random_seed
        # Green-AI budgets (C3): None means unconstrained.
        self.max_carbon_grams_co2e = (
            None if max_carbon_grams_co2e is None else float(max_carbon_grams_co2e)
        )
        self.max_inference_latency_ms = (
            None if max_inference_latency_ms is None else float(max_inference_latency_ms)
        )

    @staticmethod
    def infer_task_type(y: pd.Series | np.ndarray) -> TaskType:
        """Infer binary classification, multiclass classification, or regression from target."""
        s = pd.Series(y).dropna()
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            return TaskType.BINARY_CLASSIFICATION if s.nunique() <= 2 else TaskType.MULTICLASS_CLASSIFICATION
        if s.nunique() <= 10 and np.all(np.isclose(s, np.round(s))):
            return TaskType.BINARY_CLASSIFICATION if s.nunique() <= 2 else TaskType.MULTICLASS_CLASSIFICATION
        return TaskType.REGRESSION

    @staticmethod
    def pareto_frontier(
        rows: list[dict[str, Any]],
        primary_metric: str,
        *,
        higher_is_better: bool,
    ) -> list[dict[str, Any]]:
        """Compute the non-dominated 3D Pareto frontier (metric × latency × carbon).

        A row dominates another when it is at least as good on all three objectives and
        strictly better on at least one. Metric is maximized when ``higher_is_better``;
        latency (ms per 1k) and carbon (grams CO2eq) are always minimized.
        Rows missing latency/carbon are treated as unconstrained (infinite objective)
        and never dominate fully-measured rows unless they also tie on the available axes.
        """
        if not rows:
            return []

        def objective(r: dict[str, Any]) -> tuple[float, float, float]:
            metric = float(r.get(primary_metric, float("-inf")))
            if not higher_is_better:
                metric = -metric
            latency = float(r.get("inference_latency_ms_per_1k", float("inf")))
            carbon = float(r.get("carbon_gco2e", float("inf")))
            return (metric, -latency, -carbon)  # negate so all-maximize = non-dominated sort

        # Sort descending by primary objective for efficient sweep; then filter dominated.
        ordered = sorted(rows, key=objective, reverse=True)
        frontier: list[dict[str, Any]] = []
        # Non-dominated in 3D: track the Pareto-optimal set via pairwise comparison.
        for r in ordered:
            m, latency, c = objective(r)
            dominated = False
            for f in frontier:
                fm, fl, fc = objective(f)
                if fm >= m and fl >= latency and fc >= c and (fm > m or fl > latency or fc > c):
                    dominated = True
                    break
            if not dominated:
                frontier.append(r)
        # Stable, deterministic order by metric desc.
        frontier.sort(key=objective, reverse=True)
        return frontier

    def _fit_measured(self, model: BaseOmniForgeModel, X_tr: pd.DataFrame, y_tr: pd.Series | np.ndarray) -> float:
        """Fit ``model`` under a CarbonTracker and return emissions in grams CO2eq."""
        tracker = CarbonTracker(f"automl_candidate_{getattr(model, 'name', 'candidate')}", enabled=True)
        with tracker:
            model.fit(X_tr, y_tr)
        return float(tracker.report.emissions_g_co2eq) if tracker.report is not None else 0.0

    def _inference_latency_ms_per_1k(self, model: BaseOmniForgeModel, X_va: pd.DataFrame | np.ndarray) -> float:
        """Measure predict latency on up to 1k validation rows, scaled to ms per 1k rows."""
        bench = X_va[:1000] if hasattr(X_va, "__getitem__") else X_va
        n = max(1, len(bench))
        t0 = time.perf_counter()
        model.predict(bench)
        elapsed_s = time.perf_counter() - t0
        return (elapsed_s / n) * 1_000_000.0

    def _budget_rejected(self, reason: str, carbon_g: float, latency_ms: float) -> bool:
        if self.max_carbon_grams_co2e is not None and carbon_g > self.max_carbon_grams_co2e:
            logger.warning(
                "automl_budget_rejected", reason=reason, budget="carbon",
                carbon_g=round(carbon_g, 4), limit=self.max_carbon_grams_co2e,
            )
            return True
        if self.max_inference_latency_ms is not None and latency_ms > self.max_inference_latency_ms:
            logger.warning(
                "automl_budget_rejected", reason=reason, budget="inference_latency",
                latency_ms=round(latency_ms, 2), limit=self.max_inference_latency_ms,
            )
            return True
        return False

    def _sklearn_estimator_for_cv(
        self,
        alg: str,
        is_clf: bool,
        n_rows: int = 0,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Return a fresh scikit-learn estimator for fold-isolated ColumnTransformer Pipeline CV."""
        hp = dict(params or {})
        if is_clf:
            if alg == "random_forest":
                return RandomForestClassifier(
                    n_estimators=int(hp.get("n_estimators", 100)),
                    max_depth=hp.get("max_depth"),
                    random_state=self.random_seed,
                    n_jobs=-1,
                )
            if alg == "extra_trees":
                return ExtraTreesClassifier(
                    n_estimators=int(hp.get("n_estimators", 100)),
                    max_depth=hp.get("max_depth"),
                    random_state=self.random_seed,
                    n_jobs=-1,
                )
            if alg == "gradient_boosting":
                if n_rows > 3000 and not hp:
                    return HistGradientBoostingClassifier(max_iter=100, random_state=self.random_seed)
                return GradientBoostingClassifier(
                    n_estimators=int(hp.get("n_estimators", 80)),
                    learning_rate=float(hp.get("learning_rate", 0.1)),
                    max_depth=int(hp.get("max_depth", 3)),
                    random_state=self.random_seed,
                )
            return LogisticRegression(max_iter=500, random_state=self.random_seed)
        if alg == "random_forest":
            return RandomForestRegressor(
                n_estimators=int(hp.get("n_estimators", 100)),
                max_depth=hp.get("max_depth"),
                random_state=self.random_seed,
                n_jobs=-1,
            )
        if alg == "extra_trees":
            return ExtraTreesRegressor(
                n_estimators=int(hp.get("n_estimators", 100)),
                max_depth=hp.get("max_depth"),
                random_state=self.random_seed,
                n_jobs=-1,
            )
        if alg == "gradient_boosting":
            if n_rows > 3000 and not hp:
                return HistGradientBoostingRegressor(max_iter=100, random_state=self.random_seed)
            return GradientBoostingRegressor(
                n_estimators=int(hp.get("n_estimators", 80)),
                learning_rate=float(hp.get("learning_rate", 0.1)),
                max_depth=int(hp.get("max_depth", 3)),
                random_state=self.random_seed,
            )
        return Ridge(alpha=float(hp.get("alpha", 1.0)))

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        groups: pd.Series | np.ndarray | None = None,
        time_values: pd.Series | np.ndarray | None = None,
    ) -> AutoMLResult:
        """Execute leakage-safe splitting, fold-isolated Pipeline CV, baselines, candidate screening, HPO, and ensembling."""
        t0 = time.perf_counter()
        X_work = X.reset_index(drop=True).copy()
        y_arr = np.asarray(y)
        task = self.task_type or self.infer_task_type(y_arr)
        is_clf = task in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        primary_metric = "f1_weighted" if is_clf else "rmse"

        # Assemble temporary split frame carrying optional group/time columns so LeakageSafeSplitter audits cleanly
        split_frame = X_work.copy()
        target_tmp_col = "__uap_automl_target__"
        split_frame[target_tmp_col] = y_arr
        eff_group_col = self.group_col
        eff_time_col = self.time_col
        if groups is not None and not eff_group_col:
            eff_group_col = "__uap_automl_group__"
            split_frame[eff_group_col] = np.asarray(groups)
        if eff_group_col and eff_group_col in split_frame.columns:
            split_frame[eff_group_col] = split_frame[eff_group_col].astype("string")
        if time_values is not None and not eff_time_col:
            eff_time_col = "__uap_automl_time__"
            split_frame[eff_time_col] = np.asarray(time_values)

        split_cfg = SplitConfig(
            strategy=self.split_strategy,
            test_size=0.20,
            val_size=0.20,
            cv_folds=self.cv_folds,
            group_col=eff_group_col if (eff_group_col and eff_group_col in split_frame.columns) else None,
            time_col=eff_time_col if (eff_time_col and eff_time_col in split_frame.columns) else None,
            gap=self.gap,
            random_seed=self.random_seed,
        )
        splitter = LeakageSafeSplitter(split_cfg)
        split_res = splitter.split(split_frame, target_col=target_tmp_col)
        tr_idx = split_res.train_indices
        va_idx = split_res.val_indices
        te_idx = split_res.test_indices
        if len(va_idx) == 0:
            raise AutoMLError(
                "AutoML requires separate training, validation and test partitions.",
                suggestions=["Provide more rows or independent groups for the selected split strategy."],
            )
        if split_res.audit_report is not None and not split_res.audit_report.passed:
            raise AutoMLError(
                "AutoML split failed its independence checks.",
                suggestions=["Review the grouping and time columns for the selected split strategy."],
            )
        if is_clf and len(np.unique(y_arr[tr_idx])) < 2:
            raise AutoMLError(
                "AutoML classification training requires at least two classes in the training partition.",
                suggestions=["Provide more examples of each class or review the split strategy."],
            )
        split_cfg = split_cfg.model_copy(update={"strategy": cast(SplitStrategyType, split_res.strategy_used)})

        # Predictor matrix excludes temporary audit columns
        X_tr = X_work.iloc[tr_idx].reset_index(drop=True)
        X_va = X_work.iloc[va_idx].reset_index(drop=True)
        y_tr = y_arr[tr_idx]
        y_va = y_arr[va_idx]

        # Extract group vector strictly on the TRAINING partition (tr_idx) so CV & HPO never touch X_va
        groups_tr: np.ndarray | None = None
        if split_cfg.group_col and split_cfg.group_col in split_frame.columns:
            groups_tr = split_frame.iloc[tr_idx][split_cfg.group_col].astype("string").fillna("__MISSING_GROUP__").to_numpy()

        # Sort X_tr chronologically if time_series CV is active (keeping X_va strictly isolated as outer holdout)
        cv_X = X_tr
        cv_y = y_tr
        cv_groups = groups_tr
        if split_res.strategy_used == "time_series" and split_cfg.time_col and split_cfg.time_col in split_frame.columns:
            tr_time_col = split_frame.iloc[tr_idx][split_cfg.time_col]
            order = pd.to_datetime(tr_time_col, errors="coerce", format="mixed")
            sort_idx = np.argsort(
                order.to_numpy() if order.notna().sum() >= len(order) * 0.8 else tr_time_col.to_numpy(),
                kind="mergesort",
            )
            cv_X = X_tr.iloc[sort_idx].reset_index(drop=True)
            cv_y = y_tr[sort_idx]
            if groups_tr is not None:
                cv_groups = groups_tr[sort_idx]

        def _score_model(
            model: BaseOmniForgeModel, X_eval: pd.DataFrame = X_va, y_eval: np.ndarray = y_va
        ) -> float:
            preds = model.predict(X_eval)
            if is_clf:
                return float(f1_score(y_eval, preds, average="weighted", zero_division=0))
            return float(root_mean_squared_error(y_eval, preds))

        def _fit_with_budget(name: str, model: BaseOmniForgeModel) -> dict[str, float] | None:
            started_at = time.perf_counter()
            carbon_g = self._fit_measured(model, X_tr, y_tr)
            fit_time_s = time.perf_counter() - started_at
            latency_ms = self._inference_latency_ms_per_1k(model, X_va)
            if self._budget_rejected(name, carbon_g, latency_ms):
                return None
            return {
                "fit_time_s": round(fit_time_s, 4),
                "inference_latency_ms_per_1k": round(latency_ms, 4),
                "carbon_gco2e": round(carbon_g, 6),
            }

        cv_summary: list[dict[str, Any]] = []

        with CarbonTracker("automl_pipeline_search") as tracker:
            # 1. Mandatory Strong Baseline Benchmark (strictly fitted on X_tr, evaluated on X_va)
            benchmark = (
                BaselineBenchmarkSuite.run_classification_detailed
                if is_clf else BaselineBenchmarkSuite.run_regression_detailed
            )
            validation_baselines, baseline_models = benchmark(X_tr, y_tr, X_va, y_va, self.random_seed)

            leaderboard: list[dict[str, Any]] = [
                {
                    "model": name,
                    "family": "baseline",
                    primary_metric: scores[primary_metric],
                    "split_strategy": split_res.strategy_used,
                    "fit_time_s": 0.0,
                }
                for name, scores in validation_baselines.items()
            ]
            fitted_candidates: list[tuple[str, float, BaseOmniForgeModel]] = []
            all_fitted_models: dict[str, BaseOmniForgeModel] = dict(baseline_models)

            # 2. Candidate model screening (+ optional fold-isolated sklearn ColumnTransformer Pipeline CV)
            alg_list = (
                ["random_forest", "extra_trees", "gradient_boosting", "xgboost", "lightgbm", "logistic_regression"]
                if is_clf
                else ["random_forest", "extra_trees", "gradient_boosting", "xgboost", "lightgbm", "ridge"]
            )
            cv_supported = {"random_forest", "extra_trees", "gradient_boosting", "logistic_regression", "ridge"}
            for alg in alg_list:
                if time.perf_counter() - t0 > self.time_budget_seconds * 0.55:
                    break
                try:
                    m = ClassicalMLModel(alg, task_type=task, random_seed=self.random_seed, track_carbon=False)
                    measurements = _fit_with_budget(alg, m)
                    if measurements is None:
                        continue
                    sc = _score_model(m)
                    row_entry: dict[str, Any] = {
                        "model": alg,
                        "family": "classical_ml",
                        primary_metric: round(sc, 6),
                        "split_strategy": split_res.strategy_used,
                        **measurements,
                    }
                    # Fold-isolated Pipeline + ColumnTransformer CV (StratifiedKFold / GroupKFold / TimeSeriesSplit)
                    if (
                        self.use_cv_pipeline
                        and alg in cv_supported
                        and len(cv_X) >= 15
                        and (time.perf_counter() - t0 < self.time_budget_seconds * 0.50)
                    ):
                        try:
                            est = self._sklearn_estimator_for_cv(alg, is_clf, n_rows=len(cv_X))
                            cv_res = LeakageSafePipelineBuilder.cross_validate_leakage_free(
                                cv_X,
                                cv_y,
                                est,
                                task_type=task,
                                split_config=split_cfg,
                                groups=cv_groups,
                                select_k_features=self.select_k_features,
                            )
                            row_entry["cv_strategy"] = cv_res["cv_strategy"]
                            row_entry["cv_mean"] = cv_res["test_mean"]
                            row_entry["cv_std"] = cv_res["test_std"]
                            row_entry["overfit_gap"] = cv_res["overfit_gap"]
                            cv_summary.append(
                                {
                                    "model": alg,
                                    "cv_strategy": cv_res["cv_strategy"],
                                    "n_splits": cv_res["n_splits"],
                                    "metric": cv_res["metric"],
                                    "cv_mean": cv_res["test_mean"],
                                    "cv_std": cv_res["test_std"],
                                    "train_mean": cv_res["train_mean"],
                                    "overfit_gap": cv_res["overfit_gap"],
                                }
                            )
                        except Exception as cv_exc:
                            logger.debug("pipeline_cv_skipped", algorithm=alg, error=str(cv_exc))

                    leaderboard.append(row_entry)
                    fitted_candidates.append((alg, sc, m))
                except Exception as exc:
                    logger.warning("automl_candidate_failed", algorithm=alg, error=str(exc))

            if self.include_deep and (time.perf_counter() - t0 < self.time_budget_seconds * 0.75):
                deep_specs: list[tuple[str, BaseOmniForgeModel]] = [
                    (
                        "deep_mlp",
                        MLPModel(
                            hidden_dims=(64, 32),
                            epochs=40,
                            task_type=task,
                            random_seed=self.random_seed,
                            track_carbon=False,
                        ),
                    ),
                    (
                        "deep_cnn1d",
                        CNN1DModel(
                            epochs=30,
                            task_type=task,
                            random_seed=self.random_seed,
                            track_carbon=False,
                        ),
                    ),
                    (
                        "deep_tabular_transformer",
                        TabularTransformerModel(
                            epochs=30,
                            task_type=task,
                            random_seed=self.random_seed,
                            track_carbon=False,
                        ),
                    ),
                ]
                for d_name, d_model in deep_specs:
                    if time.perf_counter() - t0 > self.time_budget_seconds * 0.8:
                        break
                    try:
                        measurements = _fit_with_budget(d_name, d_model)
                        if measurements is None:
                            continue
                        sc = _score_model(d_model)
                        leaderboard.append(
                            {
                                "model": d_name,
                                "family": "deep_learning",
                                primary_metric: round(sc, 6),
                                "split_strategy": split_res.strategy_used,
                                **measurements,
                            }
                        )
                        fitted_candidates.append((d_name, sc, d_model))
                    except Exception as exc:
                        logger.warning("automl_deep_candidate_failed", model=d_name, error=str(exc))

            # 3. Bayesian Hyperparameter Tuning of top tree candidates via Inner CV on (cv_X, cv_y)
            # Inner HPO uses training folds. Validation only screens and ranks candidates.
            fitted_candidates.sort(key=lambda t: t[1], reverse=is_clf)
            top_tunable = [
                name for name, _, _ in fitted_candidates if name in ("random_forest", "gradient_boosting", "extra_trees")
            ][: self.tune_top_k]
            inner_folds: list[tuple[np.ndarray, np.ndarray]] = []
            if top_tunable and len(cv_X) >= 6:
                try:
                    inner_cfg = SplitConfig(
                        strategy=split_cfg.strategy,
                        test_size=0.20,
                        cv_folds=max(2, min(3, self.cv_folds)),
                        group_col=split_cfg.group_col,
                        time_col=split_cfg.time_col,
                        gap=split_cfg.gap,
                        random_seed=self.random_seed,
                    )
                    inner_cv_obj, inner_cv_groups = LeakageSafeSplitter(inner_cfg).build_cv_splitter(
                        cv_X, cv_y, is_classification=is_clf, groups=cv_groups
                    )
                    inner_folds = [
                        (np.asarray(f_tr, dtype=int), np.asarray(f_va, dtype=int))
                        for f_tr, f_va in inner_cv_obj.split(cv_X, cv_y, inner_cv_groups)
                        if len(f_tr) > 0 and len(f_va) > 0
                    ]
                except Exception as exc:
                    logger.debug("inner_hpo_cv_fallback", error=str(exc))

            for alg in top_tunable if inner_folds else ():
                if time.perf_counter() - t0 > self.time_budget_seconds * 0.85:
                    break
                space: dict[str, dict[str, Any]] = (
                    {
                        "n_estimators": {"type": "int", "low": 50, "high": 220},
                        "max_depth": {"type": "int", "low": 3, "high": 12},
                    }
                    if alg in ("random_forest", "extra_trees")
                    else {
                        "n_estimators": {"type": "int", "low": 50, "high": 220},
                        "learning_rate": {"type": "float", "low": 0.02, "high": 0.2, "log": True},
                        "max_depth": {"type": "int", "low": 3, "high": 8},
                    }
                )
                hpo = HyperparameterOptimizer(
                    strategy="bayesian",
                    direction="maximize" if is_clf else "minimize",
                    max_trials=self.tune_trials,
                    time_budget_seconds=max(5.0, self.time_budget_seconds * 0.2),
                    random_seed=self.random_seed,
                )

                def _trial_obj(params: dict[str, Any], _alg: str = alg) -> float:
                    fold_scores: list[float] = []
                    for f_tr, f_va in inner_folds:
                        cand = ClassicalMLModel(
                            _alg, task_type=task, random_seed=self.random_seed, track_carbon=False, **params
                        )
                        cand.fit(cv_X.iloc[f_tr].reset_index(drop=True), cv_y[f_tr])
                        f_preds = cand.predict(cv_X.iloc[f_va].reset_index(drop=True))
                        f_sc = (
                            float(f1_score(cv_y[f_va], f_preds, average="weighted", zero_division=0))
                            if is_clf
                            else float(root_mean_squared_error(cv_y[f_va], f_preds))
                        )
                        fold_scores.append(f_sc)
                    return float(np.mean(fold_scores))

                hpo_res = hpo.optimize(space, _trial_obj)
                tuned = ClassicalMLModel(
                    alg, task_type=task, random_seed=self.random_seed, track_carbon=False, **hpo_res.best_params
                )
                t_name = f"{alg}_tuned"
                measurements = _fit_with_budget(t_name, tuned)
                if measurements is None:
                    continue
                sc = _score_model(tuned)
                leaderboard.append(
                    {
                        "model": t_name,
                        "family": "automl_tuned",
                        primary_metric: round(sc, 6),
                        "split_strategy": split_res.strategy_used,
                        **measurements,
                    }
                )
                fitted_candidates.append((t_name, sc, tuned))

            # 4. Automatic Top-3 Voting Ensemble (deep-clone candidates so all_fitted_models stay unmutated)
            fitted_candidates.sort(key=lambda t: t[1], reverse=is_clf)
            if self.enable_ensembling and len(fitted_candidates) >= 2:
                top_models = [copy.deepcopy(m) for _, _, m in fitted_candidates[:3]]
                ens = VotingEnsembleModel(
                    estimators=top_models, task_type=task, random_seed=self.random_seed, track_carbon=False
                )
                ens_name = "automl_top3_ensemble"
                measurements = _fit_with_budget(ens_name, ens)
                if measurements is not None:
                    sc = _score_model(ens)
                    leaderboard.append(
                        {
                            "model": ens_name,
                            "family": "ensemble",
                            primary_metric: round(sc, 6),
                            "split_strategy": split_res.strategy_used,
                            **measurements,
                        }
                    )
                    fitted_candidates.append((ens_name, sc, ens))

            if not fitted_candidates:
                raise AutoMLError(
                    "No candidate model completed within the configured search limits.",
                    suggestions=["Review candidate failures or increase the time, carbon, or inference latency limit."],
                )

            # Freeze selection before reading the independent test partition.
            for cand_name, _, cand_model in fitted_candidates:
                all_fitted_models[cand_name] = cand_model
            fitted_candidates.sort(key=lambda t: t[1], reverse=is_clf)
            best_name, best_val_score, best_cand_model = fitted_candidates[0]
            development_indices = np.concatenate([tr_idx, va_idx])
            X_dev = X_work.iloc[development_indices].reset_index(drop=True)
            y_dev = y_arr[development_indices]
            X_test = X_work.iloc[te_idx].reset_index(drop=True)
            y_test = y_arr[te_idx]

            best_model = copy.deepcopy(best_cand_model).fit(X_dev, y_dev)
            best_test_score = _score_model(best_model, X_test, y_test)
            baselines, _ = benchmark(X_dev, y_dev, X_test, y_test, self.random_seed)
            if not baselines:
                raise AutoMLError(
                    "No baseline could be evaluated on the independent AutoML test partition.",
                    suggestions=["Review the data and split configuration."],
                )
            best_bl_score = (max if is_clf else min)(row[primary_metric] for row in baselines.values())
            all_fitted_models[best_name] = best_model
            best_model.metadata_.baseline_comparison = baselines
            best_model.metadata_.metrics[primary_metric] = round(best_test_score, 6)
            best_sklearn_pipeline = Pipeline([("model", _ChampionEstimator(best_model))])

            # Extract mathematical equations for baselines, neural networks, and top advanced engines
            from omniforge.evaluation.equations import ModelEquationExtractor

            target_label = str(getattr(y, "name", "") or "y")
            equations_map: dict[str, dict[str, Any]] = {}
            for m_name, m_obj in all_fitted_models.items():
                try:
                    equations_map[m_name] = ModelEquationExtractor.extract(
                        m_obj,
                        X_reference=X_tr,
                        feature_names=list(X_work.columns),
                        target_name=target_label,
                    ).model_dump()
                except Exception as exc:
                    logger.debug("automl_equation_extract_skipped", model=m_name, error=str(exc))

            leaderboard.sort(key=lambda row: row[primary_metric], reverse=is_clf)
            lift = (best_test_score - best_bl_score) if is_clf else (best_bl_score - best_test_score)

        # C3 Green-AI: Pareto frontier over accepted candidates (metric × latency × carbon).
        pareto_rows = self.pareto_frontier(
            [row for row in leaderboard if row["family"] != "baseline"],
            primary_metric,
            higher_is_better=is_clf,
        )

        elapsed = round(time.perf_counter() - t0, 4)
        logger.info(
            "automl_completed",
            best_model=best_name,
            score=round(best_test_score, 4),
            lift=round(lift, 4),
            split_strategy=split_res.strategy_used,
            leakage_audit_passed=split_res.audit_report.passed if split_res.audit_report else True,
        )
        return AutoMLResult(
            task_type=task,
            best_model_name=best_name,
            best_model=best_model,
            best_score=round(best_test_score, 6),
            primary_metric=primary_metric,
            leaderboard=leaderboard,
            baseline_benchmarks=baselines,
            lift_over_best_baseline=round(float(lift), 6),
            carbon_report=tracker.report,
            elapsed_seconds=elapsed,
            leakage_audit=split_res.audit_report,
            cv_summary=cv_summary,
            max_carbon_grams_co2e=self.max_carbon_grams_co2e,
            max_inference_latency_ms=self.max_inference_latency_ms,
            pareto_frontier=pareto_rows,
            test_indices=te_idx.copy(),
            metadata={
                "fitted_models": all_fitted_models,
                "equations": equations_map,
                "feature_names": list(X_work.columns),
                "split_strategy": split_res.strategy_used,
                "sklearn_pipeline": best_sklearn_pipeline,
                "evaluation": {
                    "leaderboard_partition": "validation",
                    "best_score_partition": "test",
                    "validation_score": round(best_val_score, 6),
                    "test_baseline_score": best_bl_score,
                    "validation_baseline_score": (max if is_clf else min)(
                        (row[primary_metric] for row in validation_baselines.values()), default=None,
                    ),
                    "n_train": len(tr_idx),
                    "n_validation": len(va_idx),
                    "n_test": len(te_idx),
                    "n_champion_fit": len(development_indices),
                },
                "cv_summary": cv_summary,
            },
        )
