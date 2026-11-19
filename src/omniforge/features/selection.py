"""Multi-method feature selection: variance, correlation, MI, F-test, L1, tree, RFE, and shadow Boruta."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler

from omniforge.core.exceptions import FeatureEngineeringError
from omniforge.core.logging import get_logger

logger = get_logger(__name__)


class FeatureSelectionReport(BaseModel):
    """Structured summary of feature selection scores and retained/dropped columns."""

    method: str
    input_feature_count: int
    selected_feature_count: int
    selected_features: list[str]
    dropped_features: list[str]
    feature_scores: dict[str, float] = Field(default_factory=dict)

    @property
    def scores(self) -> dict[str, float]:
        """Backward-compatible alias for ``feature_scores``."""
        return self.feature_scores


_METHOD_ALIASES: dict[str, str] = {
    "variance_threshold": "variance",
    "correlation_filter": "correlation",
    "anova": "f_test",
    "lasso": "l1",
}


class FeatureSelector:
    """Unified feature selection suite supporting filter, wrapper, embedded, and shadow methods."""

    def __init__(
        self,
        method: str = "tree_importance",
        *,
        task: str = "classification",
        max_features: int | None = None,
        min_variance: float = 1e-8,
        corr_threshold: float = 0.95,
        random_seed: int = 42,
    ) -> None:
        raw_m = str(method).strip().lower()
        self.method = _METHOD_ALIASES.get(raw_m, raw_m)
        self.task = task
        self.max_features = max_features
        self.min_variance = min_variance
        self.corr_threshold = corr_threshold
        self.random_seed = random_seed
        self.selected_features_: list[str] = []
        self.scores_: dict[str, float] = {}
        self.report_: FeatureSelectionReport | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> FeatureSelector:
        """Rank and select features according to ``self.method``."""
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("FeatureSelector expects a pandas DataFrame.")

        num_X = X.select_dtypes(include=[np.number]).fillna(0.0)
        all_cols = [str(c) for c in num_X.columns]
        if not all_cols:
            raise FeatureEngineeringError("No numeric features available for feature selection.")

        arr = num_X.to_numpy(dtype=float)
        k = self.max_features or max(1, int(np.ceil(len(all_cols) * 0.75)))
        k = min(k, len(all_cols))
        scores: dict[str, float] = {}

        if self.method == "variance":
            vt = VarianceThreshold(threshold=self.min_variance)
            vt.fit(arr)
            scores = {c: float(v) for c, v in zip(all_cols, vt.variances_, strict=False)}
            selected = [c for c, keep in zip(all_cols, vt.get_support(), strict=False) if keep]
        elif self.method == "correlation":
            corr = num_X.corr().abs().fillna(0.0)
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = {str(col) for col in upper.columns if any(upper[col] > self.corr_threshold)}
            selected = [c for c in all_cols if c not in to_drop]
            scores = {c: float(1.0 - corr[c].drop(labels=[c], errors="ignore").max()) if len(all_cols) > 1 else 1.0 for c in all_cols}
        elif self.method == "mutual_info":
            if y is None:
                raise FeatureEngineeringError("Target 'y' is required for mutual_info selection.")
            y_arr = np.asarray(y)
            fn = mutual_info_classif if self.task == "classification" else mutual_info_regression
            mi = fn(arr, y_arr, random_state=self.random_seed)
            scores = {c: float(s) for c, s in zip(all_cols, mi, strict=False)}
            selected = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        elif self.method == "f_test":
            if y is None:
                raise FeatureEngineeringError("Target 'y' is required for f_test selection.")
            y_arr = np.asarray(y)
            fn = f_classif if self.task == "classification" else f_regression
            non_const = np.std(arr, axis=0) > 1e-12
            f_vals = np.zeros(len(all_cols), dtype=float)
            if np.any(non_const):
                raw_f, _ = fn(arr[:, non_const], y_arr)
                f_vals[non_const] = np.nan_to_num(raw_f, nan=0.0, posinf=0.0, neginf=0.0)
            scores = {c: float(s) for c, s in zip(all_cols, f_vals, strict=False)}
            selected = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        elif self.method == "l1":
            if y is None:
                raise FeatureEngineeringError("Target 'y' is required for L1 selection.")
            y_arr = np.asarray(y)
            # Standardize features so L1 penalty is strictly scale-invariant across physical units
            arr_scaled = StandardScaler().fit_transform(arr)
            if self.task == "classification":
                try:
                    model = LogisticRegression(
                        l1_ratio=1.0, solver="saga", C=0.5, random_state=self.random_seed, max_iter=500
                    )
                except TypeError:
                    model = LogisticRegression(
                        penalty="l1", solver="liblinear", C=0.5, random_state=self.random_seed, max_iter=500
                    )
                model.fit(arr_scaled, y_arr)
                weights = np.mean(np.abs(model.coef_), axis=0)
            else:
                model = LassoCV(cv=min(5, max(2, len(arr_scaled) // 5)), random_state=self.random_seed)
                model.fit(arr_scaled, y_arr)
                weights = np.abs(model.coef_)
            scores = {c: float(w) for c, w in zip(all_cols, weights, strict=False)}
            non_zero = [c for c, w in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if w > 1e-8]
            selected = non_zero[:k] if non_zero else list(sorted(scores, key=lambda c: scores[c], reverse=True))[:k]
        elif self.method == "rfe":
            if y is None:
                raise FeatureEngineeringError("Target 'y' is required for RFE selection.")
            y_arr = np.asarray(y)
            est: Any = (
                RandomForestClassifier(n_estimators=50, random_state=self.random_seed, n_jobs=-1)
                if self.task == "classification"
                else RandomForestRegressor(n_estimators=50, random_state=self.random_seed, n_jobs=-1)
            )
            rfe = RFE(estimator=est, n_features_to_select=k, step=max(1, len(all_cols) // 5))
            rfe.fit(arr, y_arr)
            scores = {c: float(1.0 / rank) for c, rank in zip(all_cols, rfe.ranking_, strict=False)}
            selected = [c for c, keep in zip(all_cols, rfe.support_, strict=False) if keep]
        elif self.method == "boruta_shadow":
            if y is None:
                raise FeatureEngineeringError("Target 'y' is required for boruta_shadow selection.")
            y_arr = np.asarray(y)
            rng = np.random.default_rng(self.random_seed)
            n_trials = 5
            real_imp_trials = np.zeros((n_trials, len(all_cols)), dtype=float)
            shadow_max_trials = np.zeros(n_trials, dtype=float)
            for t in range(n_trials):
                shadow = np.column_stack([rng.permutation(arr[:, i]) for i in range(arr.shape[1])])
                combined = np.hstack([arr, shadow])
                est = (
                    RandomForestClassifier(n_estimators=50, random_state=self.random_seed + t, n_jobs=-1)
                    if self.task == "classification"
                    else RandomForestRegressor(n_estimators=50, random_state=self.random_seed + t, n_jobs=-1)
                )
                est.fit(combined, y_arr)
                imps = est.feature_importances_
                real_imp_trials[t] = imps[: len(all_cols)]
                shadow_max_trials[t] = float(np.max(imps[len(all_cols) :]))
            mean_real_imps = real_imp_trials.mean(axis=0)
            shadow_threshold = float(np.median(shadow_max_trials))
            scores = {c: float(v) for c, v in zip(all_cols, mean_real_imps, strict=False)}
            # Feature confirmed if it beats shadow max in majority of trials (>= 3 of 5)
            hits = np.sum(real_imp_trials > shadow_max_trials[:, None], axis=0)
            selected = [
                c
                for c, h, v in zip(all_cols, hits, mean_real_imps, strict=False)
                if h >= (n_trials // 2 + 1) or v >= shadow_threshold
            ]
            if not selected:
                selected = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        else:
            # Default: tree_importance
            if y is None:
                raise FeatureEngineeringError("Target 'y' is required for tree_importance selection.")
            y_arr = np.asarray(y)
            est = (
                RandomForestClassifier(n_estimators=80, random_state=self.random_seed, n_jobs=-1)
                if self.task == "classification"
                else RandomForestRegressor(n_estimators=80, random_state=self.random_seed, n_jobs=-1)
            )
            est.fit(arr, y_arr)
            scores = {c: float(v) for c, v in zip(all_cols, est.feature_importances_, strict=False)}
            selected = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]

        self.selected_features_ = selected
        self.scores_ = {k_: round(v_, 6) for k_, v_ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)}
        dropped = [c for c in all_cols if c not in set(selected)]
        self.report_ = FeatureSelectionReport(
            method=self.method,
            input_feature_count=len(all_cols),
            selected_feature_count=len(selected),
            selected_features=selected,
            dropped_features=dropped,
            feature_scores=self.scores_,
        )
        logger.info("features_selected", method=self.method, kept=len(selected), dropped=len(dropped))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Retain only selected features from ``X``."""
        if not self.selected_features_:
            raise FeatureEngineeringError("FeatureSelector must be fitted before transform().")
        present = [c for c in self.selected_features_ if c in X.columns]
        return X[present].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
        """Fit and filter ``X`` in a single step."""
        self.fit(X, y)
        return self.transform(X)

    def fit_transform_split(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray | None,
        X_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Leakage-free feature selection: fit strictly on ``X_train``/``y_train``, then filter both partitions."""
        self.fit(X_train, y_train)
        return self.transform(X_train), self.transform(X_test)

    @staticmethod
    def mrmr_select(
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        k: int = 10,
        task: str = "regression",
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Minimum Redundancy Maximum Relevance (mRMR, Peng et al. 2005) feature selection."""
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        num_X = X.select_dtypes(include=[np.number]).fillna(0.0)
        if num_X.empty:
            raise FeatureEngineeringError("mRMR requires numeric features.")
        k = min(max(1, k), num_X.shape[1])
        y_arr = np.asarray(y)
        if task == "classification":
            rel = mutual_info_classif(num_X, y_arr, random_state=random_state)
        else:
            rel = mutual_info_regression(num_X, y_arr, random_state=random_state)
        rel_s = pd.Series(rel, index=num_X.columns, dtype=float)
        corr_mat = num_X.corr().abs().fillna(0.0)

        selected: list[str] = []
        remaining = list(num_X.columns)
        scores: dict[str, float] = {}

        first = str(rel_s.idxmax())
        selected.append(first)
        remaining.remove(first)
        scores[first] = float(rel_s[first])

        while len(selected) < k and remaining:
            best_feat = remaining[0]
            best_score = -np.inf
            for feat in remaining:
                redundancy = float(corr_mat.loc[feat, selected].mean())
                mrmr_val = float(rel_s[feat]) - redundancy
                if mrmr_val > best_score:
                    best_score = mrmr_val
                    best_feat = feat
            selected.append(best_feat)
            remaining.remove(best_feat)
            scores[best_feat] = round(float(best_score), 6)

        return {"selected_features": selected, "scores": scores}

    @staticmethod
    def boruta_select(
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        task: str = "regression",
        n_iter: int = 15,
        alpha: float = 0.05,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Boruta all-relevant feature selection using shadow features and binomial testing."""
        from scipy.stats import binomtest

        num_X = X.select_dtypes(include=[np.number]).fillna(0.0)
        if num_X.shape[1] < 2:
            raise FeatureEngineeringError("Boruta requires at least 2 numeric features.")
        rng = np.random.default_rng(random_state)
        y_arr = np.asarray(y)
        hits = {str(c): 0 for c in num_X.columns}

        for it in range(n_iter):
            shadow = num_X.apply(lambda col: rng.permutation(col.to_numpy()))
            shadow.columns = [f"_shadow_{c}" for c in num_X.columns]
            comb = pd.concat([num_X, shadow], axis=1)
            if task == "classification":
                rf = RandomForestClassifier(n_estimators=50, random_state=random_state + it, n_jobs=-1)
            else:
                rf = RandomForestRegressor(n_estimators=50, random_state=random_state + it, n_jobs=-1)
            rf.fit(comb, y_arr)
            imps = pd.Series(rf.feature_importances_, index=comb.columns)
            max_shadow = float(imps[[c for c in comb.columns if str(c).startswith("_shadow_")]].max())
            for c in num_X.columns:
                if float(imps[c]) > max_shadow:
                    hits[str(c)] += 1

        confirmed: list[str] = []
        rejected: list[str] = []
        tentative: list[str] = []
        for c, h in hits.items():
            p_val = float(binomtest(h, n=n_iter, p=0.5, alternative="greater").pvalue)
            if p_val < alpha:
                confirmed.append(c)
            elif h < n_iter * 0.25:
                rejected.append(c)
            else:
                tentative.append(c)
        return {"confirmed": confirmed, "tentative": tentative, "rejected": rejected, "hits": hits}

    @staticmethod
    def stability_selection(
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        method: str = "l1",
        task: str = "regression",
        max_features: int = 10,
        n_bootstrap: int = 15,
        threshold: float = 0.6,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Bootstrap stability selection reporting selection frequency per feature."""
        rng = np.random.default_rng(random_state)
        n = len(X)
        counts = {str(c): 0 for c in X.columns}
        y_arr = np.asarray(y)
        for _b in range(n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            sel = FeatureSelector(method=method, task=task, max_features=max_features)
            sel.fit(X.iloc[idx], y_arr[idx])
            for f in sel.selected_features_:
                if f in counts:
                    counts[f] += 1
        freqs = {c: round(cnt / n_bootstrap, 4) for c, cnt in counts.items()}
        stable = [c for c, f in freqs.items() if f >= threshold]
        return {"stable_features": stable, "selection_frequencies": freqs, "threshold": threshold}

