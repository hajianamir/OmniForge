"""Multi-method feature selection: variance, correlation, MI, F-test, L1, tree, RFE, and shadow Boruta."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    VarianceThreshold,
    chi2,
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
    "f_classif": "f_test",
    "f_regression": "f_test",
    "chi_square": "chi2",
    "chisquare": "chi2",
    "lasso": "l1",
    "l1_lasso": "l1",
    "importance": "tree_importance",
    "rf_importance": "tree_importance",
    "boruta": "boruta_shadow",
}

_VALID_SELECTION_METHODS: frozenset[str] = frozenset(
    {
        "variance",
        "correlation",
        "mutual_info",
        "chi2",
        "f_test",
        "l1",
        "tree_importance",
        "rfe",
        "boruta_shadow",
        "mrmr",
    }
)


def _impute_numeric_medians(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Extract numeric/boolean columns and impute missing/non-finite entries with per-column medians."""
    num_X = X.select_dtypes(include=[np.number, "bool"]).copy()
    for c in num_X.columns:
        if pd.api.types.is_bool_dtype(num_X[c]):
            num_X[c] = num_X[c].astype(float)
        else:
            num_X[c] = pd.to_numeric(num_X[c], errors="coerce")
    num_X = num_X.replace([np.inf, -np.inf], np.nan)
    medians: dict[str, float] = {}
    for c in num_X.columns:
        med = float(num_X[c].median()) if num_X[c].notna().any() else 0.0
        medians[str(c)] = med
        if num_X[c].isna().any():
            num_X[c] = num_X[c].fillna(med)
    return num_X, medians


class FeatureSelector:
    """Unified feature selection suite supporting filter, wrapper, embedded, mRMR, and shadow methods."""

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
        canonical = _METHOD_ALIASES.get(raw_m, raw_m)
        if canonical not in _VALID_SELECTION_METHODS:
            raise FeatureEngineeringError(
                f"Unknown feature selection method {method!r}. "
                f"Supported methods: {sorted(_VALID_SELECTION_METHODS)}."
            )
        if task not in ("classification", "regression"):
            raise FeatureEngineeringError(f"Invalid task {task!r}; expected 'classification' or 'regression'.")
        self.method = canonical
        self.task = task
        self.max_features = max_features
        self.min_variance = min_variance
        self.corr_threshold = corr_threshold
        self.random_seed = random_seed
        self.selected_features_: list[str] = []
        self.scores_: dict[str, float] = {}
        self.col_medians_: dict[str, float] = {}
        self.chi2_min_shifts_: dict[str, float] = {}
        self.report_: FeatureSelectionReport | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> FeatureSelector:
        """Rank and select features according to ``self.method``."""
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise FeatureEngineeringError("FeatureSelector expects a non-empty pandas DataFrame.")

        num_X, self.col_medians_ = _impute_numeric_medians(X)
        all_cols = [str(c) for c in num_X.columns]
        if not all_cols:
            raise FeatureEngineeringError("No numeric features available for feature selection.")

        if self.method == "chi2" and self.task != "classification":
            raise FeatureEngineeringError("Feature selection method 'chi2' is only valid for task='classification'.")

        arr = num_X.to_numpy(dtype=float)
        k = self.max_features or max(1, int(np.ceil(len(all_cols) * 0.75)))
        k = min(k, len(all_cols))
        scores: dict[str, float] = {}

        y_arr: np.ndarray | None = None
        if y is not None:
            y_s = pd.Series(y)
            if self.task == "classification" and not pd.api.types.is_numeric_dtype(y_s):
                y_arr = pd.factorize(y_s)[0].astype(int)
            else:
                y_arr = np.asarray(y)

        if self.method == "variance":
            vt = VarianceThreshold(threshold=self.min_variance)
            vt.fit(arr)
            scores = {c: float(v) for c, v in zip(all_cols, vt.variances_, strict=False)}
            kept = [c for c, keep in zip(all_cols, vt.get_support(), strict=False) if keep]
            selected = sorted(kept, key=lambda c: scores[c], reverse=True)
            if self.max_features is not None:
                selected = selected[:k]
        elif self.method == "correlation":
            corr = num_X.corr().abs().fillna(0.0)
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = {str(col) for col in upper.columns if any(upper[col] > self.corr_threshold)}
            selected = [c for c in all_cols if c not in to_drop]
            if self.max_features is not None:
                selected = selected[:k]
            scores = {
                c: float(1.0 - corr[c].drop(labels=[c], errors="ignore").max()) if len(all_cols) > 1 else 1.0
                for c in all_cols
            }
        elif self.method == "chi2":
            if y_arr is None:
                raise FeatureEngineeringError("Target 'y' is required for chi2 selection.")
            col_mins = arr.min(axis=0)
            shifts = np.where(col_mins < 0.0, -col_mins, 0.0)
            self.chi2_min_shifts_ = {c: float(sh) for c, sh in zip(all_cols, shifts, strict=False)}
            arr_nonneg = arr + shifts
            chi2_vals, _ = chi2(arr_nonneg, y_arr)
            chi2_vals = np.nan_to_num(chi2_vals, nan=0.0, posinf=0.0, neginf=0.0)
            scores = {c: float(s) for c, s in zip(all_cols, chi2_vals, strict=False)}
            selected = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        elif self.method == "mutual_info":
            if y_arr is None:
                raise FeatureEngineeringError("Target 'y' is required for mutual_info selection.")
            fn = mutual_info_classif if self.task == "classification" else mutual_info_regression
            mi = fn(arr, y_arr, random_state=self.random_seed)
            scores = {c: float(s) for c, s in zip(all_cols, mi, strict=False)}
            selected = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        elif self.method == "f_test":
            if y_arr is None:
                raise FeatureEngineeringError("Target 'y' is required for f_test selection.")
            fn = f_classif if self.task == "classification" else f_regression
            non_const = np.std(arr, axis=0) > 1e-12
            f_vals = np.zeros(len(all_cols), dtype=float)
            if np.any(non_const):
                raw_f, _ = fn(arr[:, non_const], y_arr)
                f_vals[non_const] = np.nan_to_num(raw_f, nan=0.0, posinf=0.0, neginf=0.0)
            scores = {c: float(s) for c, s in zip(all_cols, f_vals, strict=False)}
            selected = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        elif self.method == "l1":
            if y_arr is None:
                raise FeatureEngineeringError("Target 'y' is required for L1 selection.")
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
            if y_arr is None:
                raise FeatureEngineeringError("Target 'y' is required for RFE selection.")
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
            if y_arr is None:
                raise FeatureEngineeringError("Target 'y' is required for boruta_shadow selection.")
            from scipy.stats import binomtest

            rng = np.random.default_rng(self.random_seed)
            n_trials = 15
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
            hits = np.sum(real_imp_trials > shadow_max_trials[:, None], axis=0)
            scores = {
                c: float(h) / float(n_trials) + 1e-4 * float(v)
                for c, h, v in zip(all_cols, hits, mean_real_imps, strict=False)
            }
            # Kursa & Rudnicki (2010): exact Binomial(n_trials, 0.5) upper-tail confirmation test
            confirmed = [
                c
                for c, h in zip(all_cols, hits, strict=False)
                if float(binomtest(int(h), n=n_trials, p=0.5, alternative="greater").pvalue) < 0.05
            ]
            if confirmed:
                selected = sorted(confirmed, key=lambda c: scores[c], reverse=True)[:k]
            else:
                selected = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        elif self.method == "mrmr":
            if y_arr is None:
                raise FeatureEngineeringError("Target 'y' is required for mrmr selection.")
            mrmr_out = self.mrmr_select(
                num_X,
                y_arr,
                k=k,
                task=self.task,
                random_state=self.random_seed,
            )
            selected = list(mrmr_out["selected_features"])
            scores = dict(mrmr_out.get("all_scores", mrmr_out["scores"]))
        elif self.method == "tree_importance":
            if y_arr is None:
                raise FeatureEngineeringError("Target 'y' is required for tree_importance selection.")
            est = (
                RandomForestClassifier(n_estimators=80, random_state=self.random_seed, n_jobs=-1)
                if self.task == "classification"
                else RandomForestRegressor(n_estimators=80, random_state=self.random_seed, n_jobs=-1)
            )
            est.fit(arr, y_arr)
            scores = {c: float(v) for c, v in zip(all_cols, est.feature_importances_, strict=False)}
            selected = [c for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        else:  # pragma: no cover - guarded by __init__
            raise FeatureEngineeringError(f"Unsupported feature selection method {self.method!r}.")

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
        scheme: Literal["difference", "quotient"] = "difference",
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Minimum Redundancy Maximum Relevance (mRMR; Peng, Long & Ding 2005, IEEE TPAMI).

        Normalizes mutual information relevance onto ``[0, 1]`` so relevance and
        mean absolute correlation redundancy are commensurate under both MID
        (``scheme='difference'``) and MIQ (``scheme='quotient'``).
        """
        if scheme not in ("difference", "quotient"):
            raise FeatureEngineeringError(f"mRMR scheme must be 'difference' or 'quotient', got {scheme!r}.")

        num_X, _ = _impute_numeric_medians(X)
        if num_X.empty:
            raise FeatureEngineeringError("mRMR requires numeric features.")
        k = min(max(1, k), num_X.shape[1])
        y_s = pd.Series(y)
        if task == "classification" and not pd.api.types.is_numeric_dtype(y_s):
            y_arr = pd.factorize(y_s)[0].astype(int)
        else:
            y_arr = np.asarray(y)
        if task == "classification":
            raw_rel = mutual_info_classif(num_X, y_arr, random_state=random_state)
        else:
            raw_rel = mutual_info_regression(num_X, y_arr, random_state=random_state)
        raw_rel = np.nan_to_num(raw_rel, nan=0.0, posinf=0.0, neginf=0.0)
        max_rel = float(np.max(raw_rel)) if len(raw_rel) > 0 else 0.0
        norm_rel = raw_rel / max_rel if max_rel > 1e-12 else np.zeros_like(raw_rel)

        cols = [str(c) for c in num_X.columns]
        rel_s = pd.Series(norm_rel, index=cols, dtype=float)
        raw_rel_s = pd.Series(raw_rel, index=cols, dtype=float)
        corr_mat = num_X.corr().abs().fillna(0.0)
        corr_mat.index = cols
        corr_mat.columns = cols

        selected: list[str] = []
        remaining = list(cols)
        scores: dict[str, float] = {}

        first = str(rel_s.idxmax())
        selected.append(first)
        remaining.remove(first)
        scores[first] = round(float(rel_s[first]), 6)

        while len(selected) < k and remaining:
            best_feat = remaining[0]
            best_score = -np.inf
            for feat in remaining:
                redundancy = float(corr_mat.loc[feat, selected].mean())
                mrmr_val = (
                    float(rel_s[feat]) / max(redundancy, 1e-6)
                    if scheme == "quotient"
                    else float(rel_s[feat]) - redundancy
                )
                if mrmr_val > best_score:
                    best_score = mrmr_val
                    best_feat = feat
            selected.append(best_feat)
            remaining.remove(best_feat)
            scores[best_feat] = round(float(best_score), 6)

        all_scores = dict(scores)
        for feat in remaining:
            redundancy = float(corr_mat.loc[feat, selected].mean()) if selected else 0.0
            mrmr_val = (
                float(rel_s[feat]) / max(redundancy, 1e-6)
                if scheme == "quotient"
                else float(rel_s[feat]) - redundancy
            )
            all_scores[feat] = round(float(mrmr_val), 6)

        return {
            "selected_features": selected,
            "scores": scores,
            "all_scores": all_scores,
            "normalized_relevance": {c: round(float(rel_s[c]), 6) for c in cols},
            "raw_mutual_information": {c: round(float(raw_rel_s[c]), 6) for c in cols},
            "scheme": scheme,
        }

    @staticmethod
    def boruta_select(
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        task: str = "regression",
        n_iter: int = 20,
        alpha: float = 0.05,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Boruta all-relevant feature selection via exact two-tail Binomial(n_iter, 0.5) tests (Kursa & Rudnicki 2010)."""
        from scipy.stats import binomtest

        num_X, _ = _impute_numeric_medians(X)
        if num_X.shape[1] < 2:
            raise FeatureEngineeringError("Boruta requires at least 2 numeric features.")
        rng = np.random.default_rng(random_state)
        y_s = pd.Series(y)
        if task == "classification" and not pd.api.types.is_numeric_dtype(y_s):
            y_arr = pd.factorize(y_s)[0].astype(int)
        else:
            y_arr = np.asarray(y)
        hits = {str(c): 0 for c in num_X.columns}

        for it in range(n_iter):
            shadow = num_X.apply(lambda col: rng.permutation(col.to_numpy()))
            shadow.columns = [f"_shadow_{c}" for c in num_X.columns]
            comb = pd.concat([num_X, shadow], axis=1)
            if task == "classification":
                rf: Any = RandomForestClassifier(n_estimators=50, random_state=random_state + it, n_jobs=-1)
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
        p_values_confirm: dict[str, float] = {}
        p_values_reject: dict[str, float] = {}
        for c, h in hits.items():
            p_conf = float(binomtest(h, n=n_iter, p=0.5, alternative="greater").pvalue)
            p_rej = float(binomtest(h, n=n_iter, p=0.5, alternative="less").pvalue)
            p_values_confirm[c] = round(p_conf, 6)
            p_values_reject[c] = round(p_rej, 6)
            if p_conf < alpha:
                confirmed.append(c)
            elif p_rej < alpha:
                rejected.append(c)
            else:
                tentative.append(c)
        return {
            "confirmed": confirmed,
            "tentative": tentative,
            "rejected": rejected,
            "hits": hits,
            "p_values_confirm": p_values_confirm,
            "p_values_reject": p_values_reject,
            "n_iter": n_iter,
            "alpha": alpha,
        }

    @staticmethod
    def stability_selection(
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        method: str = "l1",
        task: str = "regression",
        max_features: int = 10,
        n_bootstrap: int = 20,
        threshold: float = 0.6,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Meinshausen & Bühlmann (2010, JRSS-B 72(4):417-473) Stability Selection.

        Subsamples ``floor(n / 2)`` observations **without replacement** across
        ``n_bootstrap`` runs with distinct per-subsample seeds and reports the
        Theorem 1 expected false positives (PFER) upper bound.
        """
        if not (0.0 < threshold <= 1.0):
            raise FeatureEngineeringError(f"stability_selection threshold must be in (0, 1], got {threshold}.")
        rng = np.random.default_rng(random_state)
        n = len(X)
        num_cols = [str(c) for c in X.select_dtypes(include=[np.number, "bool"]).columns]
        p = max(len(num_cols), 1)
        counts = {str(c): 0 for c in X.columns}
        y_arr = np.asarray(y)
        subsample_size = max(2, n // 2) if n >= 4 else n
        total_selected = 0

        for b in range(n_bootstrap):
            idx = rng.choice(n, size=subsample_size, replace=False)
            sel = FeatureSelector(
                method=method,
                task=task,
                max_features=max_features,
                random_seed=random_state + b,
            )
            sel.fit(X.iloc[idx], y_arr[idx])
            for f in sel.selected_features_:
                if f in counts:
                    counts[f] += 1
            total_selected += len(sel.selected_features_)

        freqs = {c: round(cnt / max(n_bootstrap, 1), 4) for c, cnt in counts.items()}
        stable = [c for c, f in freqs.items() if f >= threshold]
        q_hat = float(total_selected) / float(max(n_bootstrap, 1))
        pfer_bound = (
            round((q_hat**2) / ((2.0 * threshold - 1.0) * p), 6)
            if threshold > 0.5
            else None
        )
        return {
            "stable_features": stable,
            "selection_frequencies": freqs,
            "threshold": threshold,
            "n_bootstrap": n_bootstrap,
            "subsample_size": int(subsample_size),
            "sampling_without_replacement": True,
            "avg_selected_per_run": round(q_hat, 4),
            "expected_false_positives_bound": pfer_bound,
        }

