"""Cross-validation strategies including Stratified, Group, TimeSeries, Purged, and Combinatorial Purged CV."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from omniforge.core.exceptions import EvaluationError
from omniforge.core.types import TaskType
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
    ) -> None:
        self.n_splits = max(3, int(n_splits))
        self.n_test_groups = min(max(1, int(n_test_groups)), self.n_splits - 1)
        self.purge_gap = max(0, int(purge_gap))
        self.embargo_gap = max(0, int(embargo_gap))
        self.embargo_pct = float(np.clip(embargo_pct, 0.0, 0.45)) if embargo_pct is not None else None

    def split(
        self, X: np.ndarray | pd.DataFrame, y: Any = None, groups: Any = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield all C(n_splits, n_test_groups) purged & embargoed train/test index pairs."""
        n = len(X)
        blocks = np.array_split(np.arange(n), self.n_splits)
        embargo_from_pct = int(np.ceil(n * self.embargo_pct)) if (self.embargo_pct is not None and self.embargo_pct > 0.0) else 0
        effective_embargo = max(self.embargo_gap, embargo_from_pct)
        for test_combo in itertools.combinations(range(self.n_splits), self.n_test_groups):
            test_idx = np.concatenate([blocks[g] for g in test_combo])
            excluded = set(test_idx.tolist())
            for g in test_combo:
                b_start, b_end = int(blocks[g][0]), int(blocks[g][-1])
                for p in range(max(0, b_start - self.purge_gap), b_start):
                    excluded.add(p)
                for e in range(b_end + 1, min(n, b_end + 1 + effective_embargo)):
                    excluded.add(e)
            train_idx = np.array([i for i in range(n) if i not in excluded], dtype=int)
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


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

        split_iter = (
            splitter.split(X, y_arr, groups)
            if self.strategy == "group_kfold"
            else splitter.split(X, y_arr)
        )

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

        metric_keys = [k for k in fold_metrics[0].keys() if k != "fold"]
        summary: dict[str, dict[str, float]] = {}
        for k in metric_keys:
            vals = [fm[k] for fm in fold_metrics if k in fm]
            summary[k] = {
                "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), 6),
            }

        return {
            "strategy": self.strategy,
            "n_folds_executed": len(fold_metrics),
            "summary": summary,
            "folds": fold_metrics,
        }
