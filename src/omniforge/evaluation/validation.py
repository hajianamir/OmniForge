"""Cross-validation strategies including Stratified, Group, TimeSeries, Purged, and Combinatorial Purged CV."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit

from omniforge.core.exceptions import EvaluationError
from omniforge.core.types import TaskType
from omniforge.data.cpcv import iter_cpcv_splits
from omniforge.evaluation.metrics import MetricsEvaluator
from omniforge.models.base import BaseOmniForgeModel


class PurgedTimeSeriesSplit:
    """Walk-forward time-series splitter enforcing both purge_gap and embargo_pct (Lopez de Prado 2018)."""

    def __init__(self, n_splits: int = 5, purge_gap: int = 2, embargo_pct: float = 0.02) -> None:
        self.n_splits = max(2, int(n_splits))
        self.purge_gap = max(0, int(purge_gap))
        self.embargo_pct = float(np.clip(embargo_pct, 0.0, 0.45))

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return configured number of walk-forward splits."""
        return self.n_splits

    def split(
        self, X: np.ndarray | pd.DataFrame, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` with strictly enforced ``purge_gap + ceil(n * embargo_pct)`` separation."""
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        if fold_size < 2:
            raise EvaluationError("Dataset too small for PurgedTimeSeriesSplit.")
        embargo_size = int(np.ceil(n * self.embargo_pct)) if self.embargo_pct > 0.0 else 0
        total_gap = self.purge_gap + embargo_size

        yielded = 0
        for k in range(1, self.n_splits + 1):
            test_start = k * fold_size
            test_end = min(n, (k + 1) * fold_size) if k < self.n_splits else n
            train_end = test_start - total_gap
            if train_end < 1 or test_end <= test_start:
                continue
            yielded += 1
            yield np.arange(0, train_end, dtype=int), np.arange(test_start, test_end, dtype=int)

        if yielded == 0:
            raise EvaluationError(
                f"PurgedTimeSeriesSplit could not produce any valid fold for n={n} with "
                f"purge_gap={self.purge_gap} and embargo_size={embargo_size}."
            )


class CombinatorialPurgedKFold:
    """Combinatorial Purged Cross-Validation (CPCV) after Lopez de Prado (2018) for financial ML."""

    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        purge_gap: int = 2,
        embargo_gap: int = 2,
        embargo_pct: float | None = None,
        eval_times: np.ndarray | pd.Series | None = None,
    ) -> None:
        self.n_splits = max(3, int(n_splits))
        self.n_test_groups = min(max(1, int(n_test_groups)), self.n_splits - 1)
        self.purge_gap = max(0, int(purge_gap))
        self.embargo_gap = max(0, int(embargo_gap))
        self.embargo_pct = float(np.clip(embargo_pct, 0.0, 0.45)) if embargo_pct is not None else None
        self.eval_times = eval_times

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """Return exact combinatorial fold count C(n_splits, n_test_groups)."""
        return int(math.comb(self.n_splits, self.n_test_groups))

    def split(
        self,
        X: np.ndarray | pd.DataFrame,
        y: Any = None,
        groups: Any = None,
        *,
        eval_times: np.ndarray | pd.Series | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield all C(n_splits, n_test_groups) purged & embargoed train/test index pairs.

        When ``eval_times`` (or ``self.eval_times``) is supplied, observation ``i`` spans the
        evaluation interval ``[t0_i, t1_i]`` (where ``t0_i`` is ``X.index[i]`` if comparable or ``i``),
        and any training sample whose label interval overlaps a test block's span ``[t0_start, max(t1_test)]``
        is purged strictly per López de Prado (2018, §7.4.1).
        """
        embargo_from_pct = int(np.ceil(len(X) * self.embargo_pct)) if self.embargo_pct else 0
        return iter_cpcv_splits(
            X, n_splits=self.n_splits, n_test_groups=self.n_test_groups, purge_gap=self.purge_gap,
            post_buffer=max(self.embargo_gap, embargo_from_pct),
            eval_times=eval_times if eval_times is not None else self.eval_times,
        )

    @staticmethod
    def compute_pbo(
        strategy_returns: np.ndarray | pd.DataFrame,
        *,
        n_partitions: int = 8,
    ) -> dict[str, Any]:
        """Compute Bailey et al. (2017) CSCV Probability of Backtest Overfitting (PBO).

        Partitions a ``(T, M)`` matrix of ``M >= 2`` candidate strategy return series into an even
        number ``S = n_partitions`` of contiguous sub-matrices, evaluates all ``C(S, S/2)``
        in-sample (IS) / out-of-sample (OOS) splits, selects the IS-optimal strategy ``m*``, and
        computes its relative OOS rank ``omega_bar`` and logit ``lambda = ln(omega_bar / (1 - omega_bar))``.
        """
        mat = np.asarray(strategy_returns, dtype=float)
        if mat.ndim != 2 or mat.shape[1] < 2:
            raise EvaluationError("strategy_returns must be a 2D array of shape (T, M) with M >= 2 strategies.")
        s_parts = max(4, int(n_partitions))
        if s_parts % 2 != 0:
            s_parts += 1
        if mat.shape[0] < s_parts * 2:
            raise EvaluationError(f"Insufficient observations ({mat.shape[0]}) for {s_parts} CSCV partitions.")

        sub_mats = np.array_split(mat, s_parts, axis=0)
        n_strats = mat.shape[1]
        half = s_parts // 2
        logits: list[float] = []
        oos_selected_sharpes: list[float] = []
        is_selected_sharpes: list[float] = []

        def _col_sharpes(arr: np.ndarray) -> np.ndarray:
            means = np.mean(arr, axis=0)
            stds = np.std(arr, axis=0, ddof=1)
            return np.where(stds > 1e-12, means / np.where(stds > 1e-12, stds, 1.0), 0.0)

        all_parts = set(range(s_parts))
        for is_combo in itertools.combinations(range(s_parts), half):
            oos_combo = sorted(all_parts - set(is_combo))
            j_is = np.vstack([sub_mats[i] for i in is_combo])
            j_oos = np.vstack([sub_mats[i] for i in oos_combo])

            r_is = _col_sharpes(j_is)
            r_oos = _col_sharpes(j_oos)
            best_m = int(np.argmax(r_is))
            is_selected_sharpes.append(float(r_is[best_m]))
            oos_selected_sharpes.append(float(r_oos[best_m]))

            # Relative OOS rank in (0, 1): rank 1 = worst, rank M = best => omega_bar = rank / (M + 1)
            ranks = stats.rankdata(r_oos, method="average")
            omega_bar = float(ranks[best_m]) / float(n_strats + 1)
            omega_clamped = float(np.clip(omega_bar, 1e-6, 1.0 - 1e-6))
            logits.append(float(np.log(omega_clamped / (1.0 - omega_clamped))))

        logits_arr = np.asarray(logits, dtype=float)
        pbo = float(np.mean(logits_arr <= 0.0))
        prob_oos_loss = float(np.mean(np.asarray(oos_selected_sharpes) < 0.0))
        slope, intercept, _, _, _ = (
            stats.linregress(is_selected_sharpes, oos_selected_sharpes)
            if len(is_selected_sharpes) >= 2 and np.std(is_selected_sharpes) > 1e-12
            else (0.0, 0.0, 0.0, 1.0, 0.0)
        )
        return {
            "pbo": round(pbo, 6),
            "probability_of_oos_loss": round(prob_oos_loss, 6),
            "median_rank_logit": round(float(np.median(logits_arr)), 6),
            "degradation_slope": round(float(slope), 6),
            "degradation_intercept": round(float(intercept), 6),
            "n_combinations": len(logits),
        }


class CrossValidator:
    """Unified CV runner evaluating any :class:`BaseOmniForgeModel` factory across enterprise CV schemes."""

    def __init__(self, strategy: str = "stratified_kfold", n_splits: int = 5, random_seed: int = 42) -> None:
        self.strategy = strategy.lower()
        self.n_splits = max(2, int(n_splits))
        self.random_seed = random_seed

    def get_splitter(self) -> Any:
        """Instantiate the configured scikit-learn or OmniForge splitter."""
        if self.strategy == "kfold":
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
        if self.strategy == "stratified_kfold":
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
        if self.strategy == "group_kfold":
            return GroupKFold(n_splits=self.n_splits)
        if self.strategy == "stratified_group_kfold":
            return StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_seed)
        if self.strategy == "time_series_split":
            return TimeSeriesSplit(n_splits=self.n_splits)
        if self.strategy == "purged_time_series":
            return PurgedTimeSeriesSplit(n_splits=self.n_splits)
        if self.strategy == "purged_kfold":
            return CombinatorialPurgedKFold(n_splits=self.n_splits, n_test_groups=1)
        if self.strategy == "cpcv":
            return CombinatorialPurgedKFold(n_splits=self.n_splits, n_test_groups=2)
        raise EvaluationError(f"Unknown CV strategy '{self.strategy}'.")

    def cross_validate(
        self,
        model_factory: Any,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        *,
        groups: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run cross-validation and aggregate mean & std across all fold metrics."""
        splitter = self.get_splitter()
        y_arr = np.asarray(y)
        fold_metrics: list[dict[str, float]] = []

        if self.strategy in ("group_kfold", "stratified_group_kfold"):
            if groups is None:
                raise EvaluationError(f"CV strategy '{self.strategy}' requires a non-null groups array.")
            split_iter = splitter.split(X, y_arr, groups)
        else:
            split_iter = splitter.split(X, y_arr)

        for fold_idx, (tr_idx, va_idx) in enumerate(split_iter):
            model: BaseOmniForgeModel = model_factory()
            X_tr = X.iloc[tr_idx] if isinstance(X, pd.DataFrame) else X[tr_idx]
            X_va = X.iloc[va_idx] if isinstance(X, pd.DataFrame) else X[va_idx]
            model.fit(X_tr, y_arr[tr_idx])
            preds = model.predict(X_va)

            if model.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
                proba = model.predict_proba(X_va)
                m = MetricsEvaluator.classification_metrics(y_arr[va_idx], preds, proba)
                scalar_m = {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
            else:
                scalar_m = MetricsEvaluator.regression_metrics(y_arr[va_idx], preds)
            scalar_m["fold"] = float(fold_idx)
            fold_metrics.append(scalar_m)

        if not fold_metrics:
            raise EvaluationError(f"Cross-validation strategy '{self.strategy}' yielded zero valid folds.")

        metric_keys = list(dict.fromkeys(k for fm in fold_metrics for k in fm if k != "fold"))
        summary: dict[str, dict[str, float]] = {}
        for k in metric_keys:
            vals = [
                float(fm[k])
                for fm in fold_metrics
                if k in fm and fm[k] is not None and np.isfinite(float(fm[k]))
            ]
            summary[k] = {
                "mean": round(float(np.mean(vals)), 6) if vals else float("nan"),
                "std": round(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), 6) if vals else float("nan"),
            }

        return {
            "strategy": self.strategy,
            "n_folds_executed": len(fold_metrics),
            "summary": summary,
            "folds": fold_metrics,
        }
