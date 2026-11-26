"""Adversarial Robustness Stress Testing and Distribution-Free Split Conformal Prediction."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from omniforge.core.exceptions import EvaluationError
from omniforge.core.logging import get_logger
from omniforge.core.types import TaskType
from omniforge.models.base import BaseOmniForgeModel

logger = get_logger(__name__)


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> tuple[float, bool]:
    """Compute the exact finite-sample split conformal quantile (Angelopoulos & Bates 2021, Thm. D.1).

    Computes the ``k``-th smallest order statistic ``s_(k)`` where
    ``k = ceil((n + 1) * (1 - alpha))``, matching the empirical inverted-CDF quantile
    ``Quantile(s_1, ..., s_n; k / n)`` without NumPy's ``method='higher'`` +1 shift
    or floating-point rounding drift. Returns ``(q_hat, finite_sample_valid)`` where
    ``finite_sample_valid`` is ``True`` iff ``k <= n``. Under exchangeability the
    valid order statistic gives marginal coverage at least ``1 - alpha``.
    An upper bound also requires appropriate tie assumptions. If the rank is
    unavailable, the numerical result is the maximum observed score and is
    not a guaranteed threshold.
    """
    arr = np.asarray(scores, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        raise EvaluationError("Cannot compute conformal quantile on empty nonconformity scores.")
    a = float(np.clip(alpha, 1e-6, 0.999999))
    k = int(np.ceil((n + 1) * (1.0 - a)))
    finite_sample_valid = k <= n
    idx = min(max(k, 1), n) - 1
    return float(np.partition(arr, idx)[idx]), finite_sample_valid


class ConformalPredictor:
    """Split & Mondrian Conformal Prediction with LAC and Adaptive Prediction Sets (APS, Romano et al. 2020)."""

    def __init__(
        self,
        model: BaseOmniForgeModel,
        alpha: float = 0.1,
        *,
        score_method: Literal["lac", "aps"] = "lac",
        allow_empty: bool = False,
        random_seed: int = 42,
    ) -> None:
        self.model = model
        self.alpha = float(np.clip(alpha, 0.001, 0.5))
        self.score_method: Literal["lac", "aps"] = score_method
        self.allow_empty = bool(allow_empty)
        self.random_seed = int(random_seed)
        self.quantile_: float = 0.0
        self.n_calibration_: int = 0
        self.finite_sample_valid_: bool = False
        self.group_quantiles_: dict[str, float] = {}
        self.group_validity_: dict[str, bool] = {}
        self.is_calibrated_: bool = False

    @staticmethod
    def _aps_scores_matrix(proba: np.ndarray, u_vec: np.ndarray) -> np.ndarray:
        """Compute exact Romano, Sesia & Candès (2020, NeurIPS Eq. 7) randomized APS scores s(x, y, u) for all classes."""
        n, k = proba.shape
        out = np.empty((n, k), dtype=float)
        for i in range(n):
            row = np.asarray(proba[i], dtype=float)
            # Deterministic descending sort (mergesort preserves class-index tie breaking)
            order = np.argsort(-row, kind="mergesort")
            sorted_p = row[order]
            cum_before = np.concatenate(([0.0], np.cumsum(sorted_p)[:-1]))
            s_sorted = cum_before + float(u_vec[i]) * sorted_p
            out[i, order] = s_sorted
        return out

    def calibrate(
        self,
        X_cal: pd.DataFrame | np.ndarray,
        y_cal: pd.Series | np.ndarray,
        *,
        groups: np.ndarray | pd.Series | list[Any] | None = None,
    ) -> ConformalPredictor:
        """Compute finite-sample conformal nonconformity quantile(s) (global + optional Mondrian group-conditional)."""
        y_arr = np.asarray(y_cal)
        n = len(y_arr)
        if n < 5:
            raise EvaluationError("ConformalPredictor requires at least 5 calibration samples.")

        if self.model.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            proba = self.model.predict_proba(X_cal)
            classes = list(self.model.classes_) if self.model.classes_ is not None else list(range(proba.shape[1]))
            cls_to_idx = {c: i for i, c in enumerate(classes)}
            if self.score_method == "aps":
                rng = np.random.default_rng(self.random_seed)
                u_cal = rng.uniform(0.0, 1.0, size=n)
                aps_mat = self._aps_scores_matrix(proba, u_cal)
                scores = np.array(
                    [float(aps_mat[i, cls_to_idx[y_arr[i]]]) if y_arr[i] in cls_to_idx else 1.0 for i in range(n)],
                    dtype=float,
                )
            else:
                scores = np.array(
                    [1.0 - float(proba[i, cls_to_idx[y_arr[i]]]) if y_arr[i] in cls_to_idx else 1.0 for i in range(n)],
                    dtype=float,
                )
        else:
            preds = self.model.predict(X_cal).astype(float)
            scores = np.abs(y_arr.astype(float) - preds)

        self.quantile_, self.finite_sample_valid_ = split_conformal_quantile(scores, self.alpha)
        self.n_calibration_ = int(np.isfinite(scores).sum())
        self.group_quantiles_ = {}
        self.group_validity_ = {}
        if groups is not None:
            g_arr = np.asarray(groups).astype(str)
            if len(g_arr) != n:
                raise EvaluationError("groups length must match X_cal / y_cal calibration sample size.")
            for grp in sorted(np.unique(g_arr).tolist()):
                g_scores = scores[g_arr == grp]
                if len(g_scores) >= 2:
                    gq, g_valid = split_conformal_quantile(g_scores, self.alpha)
                    self.group_quantiles_[grp] = gq
                    self.group_validity_[grp] = g_valid

        if not self.finite_sample_valid_:
            logger.warning(
                "conformal_calibration_sample_small",
                n_cal=n,
                alpha=self.alpha,
                min_required=int(np.ceil((1.0 - self.alpha) / self.alpha)),
            )
        self.is_calibrated_ = True
        return self

    def _resolve_thresholds(self, n_rows: int, groups: np.ndarray | pd.Series | list[Any] | None) -> np.ndarray:
        marginal_threshold = self.quantile_ if self.finite_sample_valid_ else np.inf
        if groups is None or not self.group_quantiles_:
            return np.full(n_rows, marginal_threshold, dtype=float)
        g_arr = np.asarray(groups).astype(str)
        if len(g_arr) != n_rows:
            raise EvaluationError("groups length must match X row count.")
        return np.array([
            self.group_quantiles_[g] if self.group_validity_.get(g, False)
            else np.inf if g in self.group_quantiles_ else marginal_threshold
            for g in g_arr
        ], dtype=float)

    def predict_intervals(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        groups: np.ndarray | pd.Series | list[Any] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return distribution-free (global or Mondrian group-conditional) prediction intervals ``[lower, upper]``."""
        if not self.is_calibrated_:
            raise EvaluationError("Call calibrate() before predict_intervals().")
        preds = self.model.predict(X).astype(float)
        q_vec = self._resolve_thresholds(len(preds), groups)
        return {
            "prediction": preds,
            "lower": preds - q_vec,
            "upper": preds + q_vec,
            "interval_width": 2.0 * q_vec,
        }

    def predict_sets(
        self,
        X: pd.DataFrame | np.ndarray,
        *,
        groups: np.ndarray | pd.Series | list[Any] | None = None,
    ) -> list[list[Any]]:
        """Return conformal prediction label sets (LAC or APS, with optional Mondrian group thresholds)."""
        if not self.is_calibrated_:
            raise EvaluationError("Call calibrate() before predict_sets().")
        proba = self.model.predict_proba(X)
        classes = list(self.model.classes_) if self.model.classes_ is not None else list(range(proba.shape[1]))
        q_vec = self._resolve_thresholds(len(proba), groups)
        sets: list[list[Any]] = []

        if self.score_method == "aps":
            rng = np.random.default_rng(self.random_seed + 10_007)
            u_test = rng.uniform(0.0, 1.0, size=len(proba))
            aps_mat = self._aps_scores_matrix(proba, u_test)
            for idx, row_scores in enumerate(aps_mat):
                q_i = float(q_vec[idx])
                included = [classes[j] for j, sc in enumerate(row_scores) if sc <= q_i]
                if not included and not self.allow_empty:
                    included = [classes[int(np.argmax(proba[idx]))]]
                sets.append(included)
            return sets

        for idx, row in enumerate(proba):
            q_i = float(q_vec[idx])
            included = [classes[i] for i, p in enumerate(row) if (1.0 - p) <= q_i]
            if not included and not self.allow_empty:
                included = [classes[int(np.argmax(row))]]
            sets.append(included)
        return sets


class RobustnessTester:
    """Stress-tests model stability under Gaussian noise, feature dropout, and distribution shifts."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def evaluate(
        self,
        model: BaseOmniForgeModel,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        noise_levels: tuple[float, ...] = (0.05, 0.15, 0.30),
        dropout_rates: tuple[float, ...] = (0.05, 0.15),
    ) -> dict[str, Any]:
        """Measure score degradation under adversarial perturbations, missingness, and OOD shifts."""
        rng = np.random.default_rng(self.random_seed)
        base_score = model.score(X, y)
        num_cols = list(X.select_dtypes(include=[np.number]).columns)
        stds = X[num_cols].std().replace(0.0, 1.0).fillna(1.0)

        noise_results: dict[str, float] = {}
        for sigma in noise_levels:
            X_noisy = X.copy()
            for col in num_cols:
                X_noisy[col] = X_noisy[col].astype(float) + rng.normal(0.0, sigma * float(stds[col]), size=len(X))
            noise_results[f"gaussian_sigma_{sigma}"] = round(model.score(X_noisy, y), 6)

        dropout_results: dict[str, float] = {}
        for rate in dropout_rates:
            X_drop = X.copy()
            for col in num_cols:
                mask = rng.uniform(0.0, 1.0, size=len(X)) < rate
                X_drop.loc[mask, col] = 0.0
            dropout_results[f"feature_zeroout_{rate}"] = round(model.score(X_drop, y), 6)

        # Covariate shift (+1.0 std drift across top 3 numeric features)
        X_shift = X.copy()
        for col in num_cols[:3]:
            X_shift[col] = X_shift[col].astype(float) + float(stds[col])
        shift_score = round(model.score(X_shift, y), 6)

        all_stressed = [*noise_results.values(), *dropout_results.values(), shift_score]
        worst_drop = max(0.0, base_score - min(all_stressed)) if all_stressed else 0.0
        robustness_index = round(max(0.0, 100.0 * (1.0 - worst_drop / max(abs(base_score), 1e-6))), 2)

        return {
            "clean_score": round(base_score, 6),
            "noise_stress": noise_results,
            "missingness_stress": dropout_results,
            "ood_shift_score": shift_score,
            "worst_score_drop": round(worst_drop, 6),
            "robustness_index_0_100": robustness_index,
        }


def adaptive_conformal_intervals(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    *,
    alpha: float = 0.10,
    gamma: float = 0.05,
    window: int = 30,
) -> dict[str, Any]:
    """Adaptive Conformal Inference (ACI, Gibbs & Candès 2021, NeurIPS Eq. 2 & Thm. 4.1) for non-exchangeable series.

    Maintains unclipped ``alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`` (naturally bounded in
    ``[-gamma, 1 + gamma]``) so Theorem 4.1's exact deterministic long-run coverage bound
    ``|mean(err_t) - alpha| <= (max(alpha, 1 - alpha) + gamma) / (n * gamma)`` holds under arbitrary distribution shift.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if len(yt) != len(yp) or len(yt) < 5:
        raise ValueError("y_true and y_pred must have identical length >= 5.")
    if not (0.0 < alpha < 1.0) or gamma <= 0.0:
        raise ValueError("alpha must be in (0, 1) and gamma > 0.")

    residuals = np.abs(yt - yp)
    n = len(yt)
    lower = np.empty(n, dtype=float)
    upper = np.empty(n, dtype=float)
    alpha_t = float(alpha)
    covered = np.zeros(n, dtype=bool)
    unbounded_steps = 0

    for t in range(n):
        start = max(0, t - max(2, window))
        cal_res = residuals[start:t]
        m = len(cal_res)

        if alpha_t <= 0.0:
            # Gibbs & Candès (2021, Eq. 2): when alpha_t <= 0, formal conformal set C_t(X_t) = (-inf, +inf) -> err_t = 0.
            # Emit strictly causal finite display envelope from past residuals only (zero peeking at yt[t]!).
            unbounded_steps += 1
            q_t = float(np.max(cal_res) * 3.0 + 1.0) if m > 0 else max(abs(float(yp[t])) * 0.25, 1.0)
            lower[t] = float(yp[t] - q_t)
            upper[t] = float(yp[t] + q_t)
            err_t = 0.0
        elif alpha_t >= 1.0:
            # Gibbs & Candès (2021, Eq. 2): when alpha_t >= 1, formal conformal set C_t(X_t) = empty -> err_t = 1.
            lower[t] = float(yp[t])
            upper[t] = float(yp[t])
            err_t = 1.0
        elif m == 0:
            q_t = max(abs(float(yp[t])) * 0.10, 1e-3)
            lower[t] = float(yp[t] - q_t)
            upper[t] = float(yp[t] + q_t)
            err_t = 0.0 if (lower[t] <= yt[t] <= upper[t]) else 1.0
        else:
            # Finite-sample conformal order statistic k = ceil((m + 1) * (1 - alpha_t))
            k = int(np.ceil((m + 1) * (1.0 - alpha_t)))
            if k > m:
                unbounded_steps += 1
                q_t = float(np.max(cal_res) * 3.0 + 1e-3)
                lower[t] = float(yp[t] - q_t)
                upper[t] = float(yp[t] + q_t)
                err_t = 0.0
            else:
                idx = max(k, 1) - 1
                q_t = float(np.partition(cal_res, idx)[idx])
                lower[t] = float(yp[t] - q_t)
                upper[t] = float(yp[t] + q_t)
                err_t = 0.0 if (lower[t] <= yt[t] <= upper[t]) else 1.0

        covered[t] = err_t == 0.0
        # Strictly unclipped update preserves telescoping sum in Theorem 4.1
        alpha_t = float(alpha_t + gamma * (alpha - err_t))

    eval_start = min(5, n - 1)
    emp_cov = float(np.mean(covered[eval_start:]))
    full_cov = float(np.mean(covered))
    avg_width = float(np.mean(upper[eval_start:] - lower[eval_start:]))
    theo_bound = float((max(alpha, 1.0 - alpha) + gamma) / (n * gamma))
    return {
        "lower": lower,
        "upper": upper,
        "empirical_coverage": round(emp_cov, 6),
        "full_run_coverage": round(full_cov, 6),
        "target_coverage": round(1.0 - alpha, 6),
        "avg_width": round(avg_width, 6),
        "final_alpha_t": round(alpha_t, 6),
        "unbounded_steps": int(unbounded_steps),
        "theoretical_coverage_error_bound": round(theo_bound, 6),
    }


def knn_shapley_valuation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    k: int = 5,
) -> dict[str, Any]:
    """Exact O(N log N) KNN-Shapley training data valuation (Jia et al., 2019) for mislabel/noise detection."""
    X_tr = np.asarray(X_train, dtype=float)
    y_tr = np.asarray(y_train)
    X_va = np.asarray(X_val, dtype=float)
    y_va = np.asarray(y_val)
    n_tr = len(X_tr)
    n_va = len(X_va)
    if n_tr < 2 or n_va < 1:
        return {"evaluable": False, "values": [], "negative_value_count": 0, "worst_indices": []}
    k_eff = max(1, min(k, n_tr))
    shapley = np.zeros(n_tr, dtype=float)

    for j in range(n_va):
        dists = np.linalg.norm(X_tr - X_va[j], axis=1)
        order = np.argsort(dists)
        matches = (y_tr[order] == y_va[j]).astype(float)
        s = np.empty(n_tr, dtype=float)
        s[-1] = matches[-1] / n_tr
        if n_tr > 1:
            ranks = np.arange(1, n_tr, dtype=float)
            deltas = ((matches[:-1] - matches[1:]) / k_eff) * (np.minimum(k_eff, ranks) / ranks)
            s[:-1] = s[-1] + np.cumsum(deltas[::-1])[::-1]
        shapley[order] += s / n_va

    neg_mask = shapley < 0.0
    worst_idx = np.argsort(shapley)[: min(20, n_tr)].tolist()
    return {
        "evaluable": True,
        "n_train": int(n_tr),
        "n_val": int(n_va),
        "k": int(k_eff),
        "values": [round(float(v), 6) for v in shapley],
        "negative_value_count": int(neg_mask.sum()),
        "worst_indices": [int(i) for i in worst_idx],
    }


def venn_abers_calibrate(
    cal_scores: np.ndarray,
    cal_y: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, list[float]]:
    """Inductive Venn-ABERS Predictor (IVAP, Vovk et al.) returning calibrated probability intervals [p0, p1]."""
    from sklearn.isotonic import IsotonicRegression

    cs = np.asarray(cal_scores, dtype=float)
    cy = np.asarray(cal_y, dtype=int)
    ts = np.asarray(test_scores, dtype=float)

    p0_list: list[float] = []
    p1_list: list[float] = []
    p_cal_list: list[float] = []

    for x_t in ts:
        x_aug = np.append(cs, x_t)
        iso0 = IsotonicRegression(out_of_bounds="clip").fit(x_aug, np.append(cy, 0))
        iso1 = IsotonicRegression(out_of_bounds="clip").fit(x_aug, np.append(cy, 1))
        p0 = float(np.clip(iso0.predict([x_t])[0], 0.0, 1.0))
        p1 = float(np.clip(iso1.predict([x_t])[0], 0.0, 1.0))
        denom = 1.0 - p0 + p1
        p_cal = float(p1 / denom) if denom > 1e-12 else 0.5
        p0_list.append(round(p0, 6))
        p1_list.append(round(p1, 6))
        p_cal_list.append(round(p_cal, 6))

    return {"p0": p0_list, "p1": p1_list, "p_calibrated": p_cal_list}


def nested_cv_evaluate(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    *,
    param_grid: list[dict[str, Any]] | None = None,
    outer_splits: int = 3,
    inner_splits: int = 2,
    task: str = "classification",
) -> dict[str, Any]:
    """Leakage-free Nested Cross-Validation (outer evaluation + inner HPO selection)."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score
    from sklearn.model_selection import KFold, StratifiedKFold

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y)
    grid = param_grid or [{"max_depth": 3}, {"max_depth": 6}, {"max_depth": None}]
    is_clf = task == "classification"
    outer_cv = (
        StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
        if is_clf
        else KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    )
    outer_scores: list[float] = []
    chosen_params: list[dict[str, Any]] = []

    for tr_idx, te_idx in outer_cv.split(X_arr, y_arr):
        X_tr, X_te = X_arr[tr_idx], X_arr[te_idx]
        y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]
        inner_cv = (
            StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)
            if is_clf
            else KFold(n_splits=inner_splits, shuffle=True, random_state=42)
        )
        best_p = grid[0]
        best_inner = -float("inf")
        for params in grid:
            in_scores: list[float] = []
            for itr, iva in inner_cv.split(X_tr, y_tr):
                cls = RandomForestClassifier if is_clf else RandomForestRegressor
                m = cls(n_estimators=20, random_state=42, **params).fit(X_tr[itr], y_tr[itr])
                preds = m.predict(X_tr[iva])
                sc = float(accuracy_score(y_tr[iva], preds) if is_clf else r2_score(y_tr[iva], preds))
                in_scores.append(sc)
            mean_in = float(np.mean(in_scores))
            if mean_in > best_inner:
                best_inner = mean_in
                best_p = params
        cls = RandomForestClassifier if is_clf else RandomForestRegressor
        final_m = cls(n_estimators=20, random_state=42, **best_p).fit(X_tr, y_tr)
        out_preds = final_m.predict(X_te)
        out_sc = float(accuracy_score(y_te, out_preds) if is_clf else r2_score(y_te, out_preds))
        outer_scores.append(round(out_sc, 6))
        chosen_params.append(best_p)

    return {
        "outer_scores": outer_scores,
        "mean_outer_score": round(float(np.mean(outer_scores)), 6),
        "std_outer_score": round(float(np.std(outer_scores, ddof=1) if len(outer_scores) > 1 else 0.0), 6),
        "chosen_params_per_fold": chosen_params,
    }



