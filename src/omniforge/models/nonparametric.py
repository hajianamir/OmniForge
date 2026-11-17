"""Publication-grade Non-Parametric Analytical Models with explicit mathematical representations.

Provides distribution-free, non-parametric analytical estimators that complement parametric
linear/logistic models and black-box ensembles:

1. :class:`PenalizedSplineGAMModel` — Penalized Natural B-Spline Generalized Additive Model
   ``y_hat = b0 + sum_j f_j(x_j)``, where each smooth non-parametric function ``f_j(x_j)`` is
   represented as an explicit cubic B-spline basis expansion over data-driven quantile knots.
2. :class:`NadarayaWatsonKernelModel` — Gaussian Kernel Nadaraya-Watson local non-parametric
   smoother with cross-validated bandwidth ``h`` and exact analytical kernel ratio formula.
3. :class:`RBFKernelRidgeModel` — Closed-form RKHS Radial Basis Function Kernel Ridge model
   ``y_hat(x) = sum_i alpha_i * exp(-gamma * ||x - c_i||^2)`` with automatic hyperparameter selection.
4. :class:`DistanceWeightedKNNModel` — Inverse-distance weighted local non-parametric estimator
   with cross-validated neighborhood size ``k``.
5. :class:`IsotonicMonotonicModel` — Non-parametric piecewise-constant monotonic regression
   fitted on the strongest Spearman-correlated predictor with explicit step thresholds.
6. :class:`NonParametricAnalyticalSuite` — Head-to-head benchmark comparing all non-parametric
   analytical models against a parametric linear/logistic reference with explicit non-linearity lift.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import SplineTransformer, StandardScaler

from omniforge.core.types import TaskType
from omniforge.models.base import BaseOmniForgeModel


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    val = 1.0 - ss_res / ss_tot
    return float(val) if math.isfinite(val) else 0.0


def _clean_matrix(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


class PenalizedSplineGAMModel(BaseOmniForgeModel):
    """Non-parametric Penalized B-Spline Generalized Additive Model (GAM) for regression & classification.

    Decomposes predictions into transparent smooth non-linear per-feature functions:
        ``f(x) = beta_0 + f_1(x_1) + f_2(x_2) + ... + f_p(x_p)``
    where each ``f_j(x_j) = sum_{m=1}^{M} w_{jm} B_{jm}(x_j)`` uses penalized B-spline basis
    functions over quantile knots with explicit per-feature effect ranges and non-linearity metrics.
    """

    model_family = "nonparametric_analytical"

    def __init__(
        self,
        *,
        task_type: TaskType = TaskType.REGRESSION,
        n_knots: int = 5,
        degree: int = 3,
        alpha: float = 2.0,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        is_clf = task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        super().__init__(
            name="NonParametric_SplineGAM_Clf" if is_clf else "NonParametric_SplineGAM",
            task_type=task_type,
            random_seed=random_seed,
            n_knots=n_knots,
            degree=degree,
            alpha=alpha,
            **kwargs,
        )
        self.n_knots = max(3, int(n_knots))
        self.degree = max(1, min(3, int(degree)))
        self.alpha = float(max(1e-4, alpha))
        self._is_clf = is_clf
        self._scaler = StandardScaler()
        self._spline: SplineTransformer | None = None
        self._estimator: Any = None
        self.spline_component_summary_: list[dict[str, Any]] = []
        self.effective_df_: float = 0.0
        self.classes_: np.ndarray = np.array([])

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        arr = _clean_matrix(X)
        n_samples, n_features = arr.shape
        # Guard knot count on small datasets
        effective_knots = min(self.n_knots, max(3, n_samples // 10))
        effective_degree = min(self.degree, max(1, effective_knots - 1))

        X_scaled = self._scaler.fit_transform(arr)
        self._spline = SplineTransformer(
            n_knots=effective_knots,
            degree=effective_degree,
            knots="quantile",
            extrapolation="linear",
            include_bias=False,
        )
        B = np.nan_to_num(self._spline.fit_transform(X_scaled), nan=0.0, posinf=0.0, neginf=0.0)
        n_basis_per_feat = B.shape[1] // max(n_features, 1)

        if self._is_clf:
            self._estimator = LogisticRegression(
                C=1.0 / max(self.alpha, 0.05),
                max_iter=1000,
                random_state=self.random_seed,
            )
            self._estimator.fit(B, y)
            self.classes_ = np.asarray(self._estimator.classes_)
            coef_vec = (
                np.mean(np.abs(self._estimator.coef_), axis=0)
                if self._estimator.coef_.ndim > 1
                else np.asarray(self._estimator.coef_, dtype=float).ravel()
            )
            signed_coef = (
                self._estimator.coef_[0]
                if self._estimator.coef_.ndim > 1
                else np.asarray(self._estimator.coef_, dtype=float).ravel()
            )
        else:
            # Select smoothing penalty alpha via fast GCV/holdout heuristic over small grid
            best_a = self.alpha
            if n_samples >= 25:
                y_arr = np.asarray(y, dtype=float)
                split = max(12, int(n_samples * 0.8))
                best_mse = float("inf")
                for cand_a in (0.2, 1.0, 5.0, 20.0):
                    m_try = Ridge(alpha=cand_a, random_state=self.random_seed)
                    m_try.fit(B[:split], y_arr[:split])
                    pred_v = m_try.predict(B[split:])
                    mse = float(np.mean((y_arr[split:] - pred_v) ** 2))
                    if mse < best_mse:
                        best_mse = mse
                        best_a = cand_a
            self.alpha = best_a
            self._estimator = Ridge(alpha=best_a, random_state=self.random_seed)
            self._estimator.fit(B, np.asarray(y, dtype=float))
            signed_coef = np.asarray(self._estimator.coef_, dtype=float).ravel()
            coef_vec = np.abs(signed_coef)

        # Effective degrees of freedom: tr(B (B^T B + alpha I)^-1 B^T)
        try:
            bt_b = B.T @ B
            reg = bt_b + self.alpha * np.eye(bt_b.shape[0])
            hat_diag = np.trace(np.linalg.solve(reg, bt_b))
            self.effective_df_ = round(float(max(1.0, hat_diag)), 3)
        except Exception:
            self.effective_df_ = float(B.shape[1])

        # Compute exact per-feature additive spline contribution f_j(x_j)
        feat_names = list(self.feature_names_) if self.feature_names_ else [f"x_{i}" for i in range(n_features)]
        comp_stds: list[float] = []
        raw_summaries: list[dict[str, Any]] = []
        bsplines = getattr(self._spline, "bsplines_", [])

        for j in range(n_features):
            start = j * n_basis_per_feat
            end = min(B.shape[1], (j + 1) * n_basis_per_feat)
            if start >= end:
                continue
            B_j = B[:, start:end]
            w_j = signed_coef[start:end]
            f_j = B_j @ w_j
            f_std = float(np.std(f_j))
            comp_stds.append(f_std)

            # Measure non-linearity via 1 - R^2 of linear fit to f_j(x_j)
            xj = arr[:, j]
            if float(np.std(xj)) > 1e-12 and f_std > 1e-12:
                corr = float(np.corrcoef(xj, f_j)[0, 1])
                corr = corr if math.isfinite(corr) else 0.0
                nonlin_idx = max(0.0, min(1.0, 1.0 - corr * corr))
                direction = "positive" if corr > 0.15 else ("negative" if corr < -0.15 else "non-monotonic / curved")
            else:
                nonlin_idx = 0.0
                direction = "flat"

            # Recover interior knots in original feature scale
            knots_orig: list[float] = []
            if j < len(bsplines):
                raw_t = np.asarray(bsplines[j].t, dtype=float)
                uniq_t = np.unique(raw_t)
                mean_j = float(self._scaler.mean_[j]) if hasattr(self._scaler, "mean_") else 0.0
                scale_j = float(self._scaler.scale_[j]) if hasattr(self._scaler, "scale_") else 1.0
                knots_orig = [round(float(k * scale_j + mean_j), 4) for k in uniq_t]

            raw_summaries.append(
                {
                    "feature": feat_names[j] if j < len(feat_names) else f"x_{j}",
                    "n_basis": int(end - start),
                    "effect_std": round(f_std, 5),
                    "effect_min": round(float(np.min(f_j)), 4),
                    "effect_max": round(float(np.max(f_j)), 4),
                    "nonlinearity_index": round(nonlin_idx, 4),
                    "shape_direction": direction,
                    "basis_l2_norm": round(float(np.linalg.norm(coef_vec[start:end])), 4),
                    "knots": knots_orig[:7],
                }
            )

        total_var = sum(s * s for s in comp_stds) or 1.0
        for item, s in zip(raw_summaries, comp_stds, strict=False):
            item["variance_share_pct"] = round(float((s * s) / total_var * 100.0), 2)
        raw_summaries.sort(key=lambda r: float(r["variance_share_pct"]), reverse=True)
        self.spline_component_summary_ = raw_summaries

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        assert self._spline is not None and self._estimator is not None
        arr = _clean_matrix(X)
        B = np.nan_to_num(self._spline.transform(self._scaler.transform(arr)), nan=0.0, posinf=0.0, neginf=0.0)
        preds = self._estimator.predict(B)
        return np.asarray(preds) if self._is_clf else np.asarray(preds, dtype=float)

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if not self._is_clf or self._spline is None or self._estimator is None:
            raise NotImplementedError("predict_proba is only available for classification GAM.")
        arr = _clean_matrix(X)
        B = np.nan_to_num(self._spline.transform(self._scaler.transform(arr)), nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(self._estimator.predict_proba(B), dtype=float)


class NadarayaWatsonKernelModel(BaseOmniForgeModel):
    """Non-parametric Nadaraya-Watson Gaussian Kernel Smoother (Regression & Classification).

    Exact analytical kernel estimator:
        ``y_hat(x) = sum_i K_h(x - x_i) * y_i / sum_i K_h(x - x_i)``
        ``K_h(u) = exp(-||u||_2^2 / (2 * h^2))``
    Automatically selects optimal bandwidth ``h`` via Silverman heuristic + holdout cross-validation
    and caps prototype anchors cleanly for fast O(N) inference on large datasets.
    """

    model_family = "nonparametric_analytical"

    def __init__(
        self,
        *,
        task_type: TaskType = TaskType.REGRESSION,
        bandwidth: float | None = None,
        max_anchors: int = 800,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        is_clf = task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        super().__init__(
            name="NonParametric_NadarayaWatson_Clf" if is_clf else "NonParametric_NadarayaWatson",
            task_type=task_type,
            random_seed=random_seed,
            bandwidth=bandwidth,
            max_anchors=max_anchors,
            **kwargs,
        )
        self._is_clf = is_clf
        self._requested_bw = bandwidth
        self.max_anchors = max(50, int(max_anchors))
        self._scaler = StandardScaler()
        self._X_anchors: np.ndarray = np.empty((0, 0), dtype=float)
        self._y_anchors: np.ndarray = np.array([])
        self.classes_: np.ndarray = np.array([])
        self.bandwidth_: float = 1.0
        self.effective_df_: float = 1.0
        self.n_anchors_: int = 0

    def _compute_kernel_preds(self, X_query: np.ndarray, X_ref: np.ndarray, y_ref: np.ndarray, bw: float) -> np.ndarray:
        # Pairwise squared Euclidean distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
        q_norm = np.sum(X_query * X_query, axis=1, keepdims=True)
        r_norm = np.sum(X_ref * X_ref, axis=1, keepdims=True).T
        sq_dists = np.maximum(0.0, q_norm + r_norm - 2.0 * (X_query @ X_ref.T))
        denom = max(2.0 * bw * bw, 1e-8)
        # Log-sum-exp stabilized kernel weights
        logits = -sq_dists / denom
        logits -= np.max(logits, axis=1, keepdims=True)
        W = np.exp(logits)
        w_sum = np.sum(W, axis=1, keepdims=True)
        w_sum = np.where(w_sum <= 1e-12, 1.0, w_sum)
        W_norm = W / w_sum

        if self._is_clf:
            probs = np.zeros((len(X_query), len(self.classes_)), dtype=float)
            for idx_c, cls_v in enumerate(self.classes_):
                indicator = (y_ref == cls_v).astype(float)
                probs[:, idx_c] = W_norm @ indicator
            return probs
        return np.asarray(W_norm @ np.asarray(y_ref, dtype=float), dtype=float)

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        arr = _clean_matrix(X)
        y_arr = np.asarray(y)
        n, d = arr.shape
        X_s = self._scaler.fit_transform(arr)

        if self._is_clf:
            self.classes_ = np.unique(y_arr)

        # Stratified/uniform subsampling of prototype anchors on large datasets
        if n > self.max_anchors:
            rng = np.random.default_rng(self.random_seed)
            idx = rng.choice(n, size=self.max_anchors, replace=False)
            X_use = X_s[idx]
            y_use = y_arr[idx]
        else:
            X_use = X_s
            y_use = y_arr

        # Silverman-inspired multivariate rule of thumb on standardized features
        silverman = float(max(0.25, (len(X_use) ** (-1.0 / (d + 4.0))) * math.sqrt(max(d, 1))))

        if self._requested_bw is not None and self._requested_bw > 0:
            best_bw = float(self._requested_bw)
        elif len(X_use) >= 20:
            # Fast holdout bandwidth selection over 5 scales around Silverman's rule
            cut = max(10, int(len(X_use) * 0.8))
            X_tr, y_tr = X_use[:cut], y_use[:cut]
            X_va, y_va = X_use[cut:], y_use[cut:]
            candidates = [silverman * mult for mult in (0.35, 0.6, 1.0, 1.6, 2.5)]
            best_bw = silverman
            best_score = -float("inf")
            for bw in candidates:
                out = self._compute_kernel_preds(X_va, X_tr, y_tr, bw)
                if self._is_clf:
                    pred_cls = self.classes_[np.argmax(out, axis=1)]
                    sc = float(f1_score(y_va, pred_cls, average="weighted", zero_division=0))
                else:
                    sc = _safe_r2(np.asarray(y_va, dtype=float), out)
                if sc > best_score:
                    best_score = sc
                    best_bw = float(bw)
        else:
            best_bw = silverman

        self.bandwidth_ = round(float(best_bw), 4)
        self._X_anchors = X_use
        self._y_anchors = y_use
        self.n_anchors_ = int(len(X_use))

        # Estimate effective degrees of freedom: trace of smoother matrix S (self-weight sum)
        try:
            n_sub = min(len(X_use), 250)
            Xs_sub = X_use[:n_sub]
            q_n = np.sum(Xs_sub * Xs_sub, axis=1, keepdims=True)
            sq_d = np.maximum(0.0, q_n + q_n.T - 2.0 * (Xs_sub @ Xs_sub.T))
            W_sub = np.exp(-sq_d / max(2.0 * best_bw * best_bw, 1e-8))
            row_sums = np.sum(W_sub, axis=1)
            edf_sub = float(np.sum(1.0 / np.maximum(row_sums, 1.0)))
            self.effective_df_ = round(edf_sub * (len(X_use) / max(n_sub, 1)), 2)
        except Exception:
            self.effective_df_ = float(max(1.0, math.sqrt(len(X_use))))

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        arr = _clean_matrix(X)
        X_s = self._scaler.transform(arr)
        out = self._compute_kernel_preds(X_s, self._X_anchors, self._y_anchors, self.bandwidth_)
        if self._is_clf:
            return np.asarray(self.classes_[np.argmax(out, axis=1)])
        return np.asarray(out, dtype=float)

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if not self._is_clf:
            raise NotImplementedError("predict_proba is only available for classification Nadaraya-Watson.")
        arr = _clean_matrix(X)
        X_s = self._scaler.transform(arr)
        return np.asarray(self._compute_kernel_preds(X_s, self._X_anchors, self._y_anchors, self.bandwidth_), dtype=float)


class RBFKernelRidgeModel(BaseOmniForgeModel):
    """Non-parametric RKHS RBF Kernel Ridge Analytical Model (closed-form dual kernel expansion)."""

    model_family = "nonparametric_analytical"

    def __init__(
        self,
        *,
        alpha: float = 0.5,
        gamma: float | None = None,
        max_anchors: int = 800,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="NonParametric_RBFKernelRidge",
            task_type=TaskType.REGRESSION,
            random_seed=random_seed,
            alpha=alpha,
            gamma=gamma,
            max_anchors=max_anchors,
            **kwargs,
        )
        self.alpha = float(max(1e-4, alpha))
        self.gamma = gamma
        self.max_anchors = max(50, int(max_anchors))
        self._scaler = StandardScaler()
        self._estimator: KernelRidge | None = None
        self.fitted_gamma_: float = 0.1
        self.fitted_alpha_: float = self.alpha
        self.n_support_anchors_: int = 0
        self.top_dual_weights_: list[dict[str, Any]] = []

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        arr = _clean_matrix(X)
        y_arr = np.asarray(y, dtype=float)
        X_s = self._scaler.fit_transform(arr)
        n, d = X_s.shape
        if n > self.max_anchors:
            rng = np.random.default_rng(self.random_seed)
            idx = rng.choice(n, size=self.max_anchors, replace=False)
            X_fit, y_fit = X_s[idx], y_arr[idx]
        else:
            X_fit, y_fit = X_s, y_arr

        base_gamma = self.gamma if (self.gamma is not None and self.gamma > 0) else (1.0 / max(float(d), 1.0))
        best_g, best_a = base_gamma, self.alpha
        if len(X_fit) >= 25 and self.gamma is None:
            cut = max(12, int(len(X_fit) * 0.8))
            best_mse = float("inf")
            for g_mult in (0.25, 1.0, 3.0):
                for a_val in (0.05, 0.5, 2.0):
                    g_try = base_gamma * g_mult
                    kr = KernelRidge(alpha=a_val, kernel="rbf", gamma=g_try)
                    kr.fit(X_fit[:cut], y_fit[:cut])
                    mse = float(np.mean((y_fit[cut:] - kr.predict(X_fit[cut:])) ** 2))
                    if mse < best_mse:
                        best_mse = mse
                        best_g = g_try
                        best_a = a_val

        self.fitted_gamma_ = round(float(best_g), 6)
        self.fitted_alpha_ = round(float(best_a), 5)
        self._estimator = KernelRidge(alpha=best_a, kernel="rbf", gamma=best_g)
        self._estimator.fit(X_fit, y_fit)
        self.n_support_anchors_ = int(len(X_fit))

        dual = np.asarray(self._estimator.dual_coef_, dtype=float).ravel()
        top_idx = np.argsort(np.abs(dual))[::-1][:10]
        self.top_dual_weights_ = [
            {
                "anchor_index": int(i),
                "dual_weight_alpha": round(float(dual[i]), 5),
                "abs_weight": round(float(abs(dual[i])), 5),
                "anchor_target_y": round(float(y_fit[i]), 4),
            }
            for i in top_idx
        ]

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        assert self._estimator is not None
        arr = _clean_matrix(X)
        return np.asarray(self._estimator.predict(self._scaler.transform(arr)), dtype=float)


class DistanceWeightedKNNModel(BaseOmniForgeModel):
    """Non-parametric inverse-distance weighted k-Nearest Neighbors estimator with automatic k tuning."""

    model_family = "nonparametric_analytical"

    def __init__(
        self,
        *,
        task_type: TaskType = TaskType.REGRESSION,
        n_neighbors: int | None = None,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        is_clf = task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        super().__init__(
            name="NonParametric_WeightedKNN_Clf" if is_clf else "NonParametric_WeightedKNN",
            task_type=task_type,
            random_seed=random_seed,
            n_neighbors=n_neighbors,
            **kwargs,
        )
        self._is_clf = is_clf
        self._requested_k = n_neighbors
        self._scaler = StandardScaler()
        self._estimator: Any = None
        self.fitted_k_: int = 5

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        arr = _clean_matrix(X)
        X_s = self._scaler.fit_transform(arr)
        n = len(X_s)

        if self._requested_k is not None:
            best_k = min(max(1, int(self._requested_k)), max(1, n - 1))
        elif n >= 25:
            cut = max(12, int(n * 0.8))
            best_k = min(5, max(1, cut - 1))
            best_score = -float("inf")
            for k in (3, 5, 9, 15):
                if k >= cut:
                    continue
                if self._is_clf:
                    m = KNeighborsClassifier(n_neighbors=k, weights="distance")
                    m.fit(X_s[:cut], y[:cut])
                    sc = float(f1_score(y[cut:], m.predict(X_s[cut:]), average="weighted", zero_division=0))
                else:
                    m = KNeighborsRegressor(n_neighbors=k, weights="distance")
                    m.fit(X_s[:cut], np.asarray(y[:cut], dtype=float))
                    sc = _safe_r2(np.asarray(y[cut:], dtype=float), m.predict(X_s[cut:]))
                if sc > best_score:
                    best_score = sc
                    best_k = k
        else:
            best_k = min(5, max(1, n - 1))

        self.fitted_k_ = int(best_k)
        if self._is_clf:
            self._estimator = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
            self._estimator.fit(X_s, y)
        else:
            self._estimator = KNeighborsRegressor(n_neighbors=best_k, weights="distance")
            self._estimator.fit(X_s, np.asarray(y, dtype=float))

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        assert self._estimator is not None
        arr = _clean_matrix(X)
        preds = self._estimator.predict(self._scaler.transform(arr))
        return np.asarray(preds) if self._is_clf else np.asarray(preds, dtype=float)

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if not self._is_clf or self._estimator is None:
            raise NotImplementedError("predict_proba is only available for classification WeightedKNN.")
        arr = _clean_matrix(X)
        return np.asarray(self._estimator.predict_proba(self._scaler.transform(arr)), dtype=float)


class IsotonicMonotonicModel(BaseOmniForgeModel):
    """Non-parametric monotonic piecewise-constant regression on the strongest Spearman predictor."""

    model_family = "nonparametric_analytical"

    def __init__(self, *, random_seed: int = 42, **kwargs: Any) -> None:
        super().__init__(
            name="NonParametric_IsotonicMonotonic",
            task_type=TaskType.REGRESSION,
            random_seed=random_seed,
            **kwargs,
        )
        self._estimator: IsotonicRegression | None = None
        self.driver_index_: int = 0
        self.driver_feature_: str = "x_0"
        self.driver_spearman_rho_: float = 0.0
        self.increasing_: bool = True
        self.step_thresholds_: list[dict[str, float]] = []

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        arr = _clean_matrix(X)
        y_arr = np.asarray(y, dtype=float)
        n_features = arr.shape[1]
        feat_names = list(self.feature_names_) if self.feature_names_ else [f"x_{i}" for i in range(n_features)]

        best_idx = 0
        best_rho = 0.0
        for j in range(n_features):
            col = arr[:, j]
            if float(np.std(col)) <= 1e-12:
                continue
            rho, _ = stats.spearmanr(col, y_arr)
            if math.isfinite(float(rho)) and abs(float(rho)) > abs(best_rho):
                best_rho = float(rho)
                best_idx = j

        self.driver_index_ = best_idx
        self.driver_feature_ = feat_names[best_idx] if best_idx < len(feat_names) else f"x_{best_idx}"
        self.driver_spearman_rho_ = round(float(best_rho), 4)
        self.increasing_ = best_rho >= 0.0

        self._estimator = IsotonicRegression(increasing=self.increasing_, out_of_bounds="clip")
        self._estimator.fit(arr[:, best_idx], y_arr)

        x_t = np.asarray(getattr(self._estimator, "X_thresholds_", []), dtype=float)
        y_t = np.asarray(getattr(self._estimator, "y_thresholds_", []), dtype=float)
        # Compact representative monotonic curve steps (max 12 knots)
        if len(x_t) > 12:
            pick = np.linspace(0, len(x_t) - 1, 12, dtype=int)
            x_t = x_t[pick]
            y_t = y_t[pick]
        self.step_thresholds_ = [
            {"x_threshold": round(float(xv), 4), "y_hat_step": round(float(yv), 4)}
            for xv, yv in zip(x_t, y_t, strict=False)
        ]

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        assert self._estimator is not None
        arr = _clean_matrix(X)
        return np.asarray(self._estimator.predict(arr[:, self.driver_index_]), dtype=float)


class NonParametricAnalyticalSuite:
    """Benchmarks all Non-Parametric Analytical Models against a Parametric Linear/Logistic reference."""

    @staticmethod
    def run(
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        task_type: TaskType = TaskType.REGRESSION,
        target_name: str = "y",
        cv_folds: int = 5,
        random_seed: int = 42,
        models: list[str] | None = None,
        n_knots: int = 5,
        spline_degree: int = 3,
        bandwidth: float | None = None,
        n_neighbors: int = 11,
        rbf_gamma: float | None = None,
    ) -> dict[str, Any]:
        """Fit, cross-validate, and extract mathematical equations for all non-parametric analytical models."""
        import time as _time

        from omniforge.evaluation.equations import ModelEquationExtractor

        t0 = _time.perf_counter()
        is_clf = task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        y_arr = np.asarray(y)
        n = len(X)
        k_folds = max(2, min(int(cv_folds), 5, n // 4)) if n >= 12 else 2
        sel = {m.strip().lower() for m in models} if models else None
        bw_val: float | str = float(bandwidth) if bandwidth is not None and bandwidth > 0 else "auto"
        gamma_val: float | None = float(rbf_gamma) if rbf_gamma is not None and rbf_gamma > 0 else None

        if is_clf:
            min_cls = int(pd.Series(y_arr).value_counts().min()) if len(y_arr) > 0 else 0
            k_folds = min(k_folds, max(2, min_cls)) if min_cls >= 2 else 2
            splitter: Any = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_seed) if min_cls >= 2 else KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            all_clf: list[tuple[str, str, str, Any]] = [
                ("parametric", "Parametric_LogisticLinear", "Parametric Linear", lambda: _ParametricRefWrapper(is_clf=True, random_seed=random_seed)),
                ("gam_spline", "NonParametric_SplineGAM_Clf", "Non-Parametric Analytical (B-Spline GAM)", lambda: PenalizedSplineGAMModel(task_type=task_type, n_knots=n_knots, degree=spline_degree, random_seed=random_seed)),
                ("nadaraya_watson", "NonParametric_NadarayaWatson_Clf", "Non-Parametric Analytical (Gaussian Kernel)", lambda: NadarayaWatsonKernelModel(task_type=task_type, bandwidth=bw_val, random_seed=random_seed)),
                ("weighted_knn", "NonParametric_WeightedKNN_Clf", "Non-Parametric Analytical (Local k-NN)", lambda: DistanceWeightedKNNModel(task_type=task_type, n_neighbors=n_neighbors, random_seed=random_seed)),
            ]
            candidates: list[tuple[str, str, Any]] = [
                (nm, pdg, fac) for key, nm, pdg, fac in all_clf if key == "parametric" or sel is None or key in sel
            ]
            if len(candidates) == 1:
                candidates = [(nm, pdg, fac) for _, nm, pdg, fac in all_clf]
        else:
            splitter = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
            all_reg: list[tuple[str, str, str, Any]] = [
                ("parametric", "Parametric_RidgeLinear", "Parametric Linear", lambda: _ParametricRefWrapper(is_clf=False, random_seed=random_seed)),
                ("gam_spline", "NonParametric_SplineGAM", "Non-Parametric Analytical (B-Spline GAM)", lambda: PenalizedSplineGAMModel(task_type=TaskType.REGRESSION, n_knots=n_knots, degree=spline_degree, random_seed=random_seed)),
                ("nadaraya_watson", "NonParametric_NadarayaWatson", "Non-Parametric Analytical (Gaussian Kernel)", lambda: NadarayaWatsonKernelModel(task_type=TaskType.REGRESSION, bandwidth=bw_val, random_seed=random_seed)),
                ("kernel_ridge_rbf", "NonParametric_RBFKernelRidge", "Non-Parametric Analytical (RKHS Kernel Ridge)", lambda: RBFKernelRidgeModel(gamma=gamma_val, random_seed=random_seed)),
                ("weighted_knn", "NonParametric_WeightedKNN", "Non-Parametric Analytical (Local k-NN)", lambda: DistanceWeightedKNNModel(task_type=TaskType.REGRESSION, n_neighbors=n_neighbors, random_seed=random_seed)),
                ("isotonic", "NonParametric_IsotonicMonotonic", "Non-Parametric Analytical (Isotonic Step)", lambda: IsotonicMonotonicModel(random_seed=random_seed)),
            ]
            candidates = [
                (nm, pdg, fac) for key, nm, pdg, fac in all_reg if key == "parametric" or sel is None or key in sel
            ]
            if len(candidates) == 1:
                candidates = [(nm, pdg, fac) for _, nm, pdg, fac in all_reg]

        leaderboard: list[dict[str, Any]] = []
        fitted_models: dict[str, BaseOmniForgeModel] = {}
        equations: list[dict[str, Any]] = []
        param_score = 0.0

        splits = list(splitter.split(X, y_arr if is_clf else None))

        for name, paradigm, factory in candidates:
            fold_primary: list[float] = []
            fold_sec: list[float] = []
            try:
                for tr_i, va_i in splits:
                    m = factory()
                    m.fit(X.iloc[tr_i], y_arr[tr_i])
                    preds = m.predict(X.iloc[va_i])
                    if is_clf:
                        fold_primary.append(float(f1_score(y_arr[va_i], preds, average="weighted", zero_division=0)))
                        fold_sec.append(float(accuracy_score(y_arr[va_i], preds)))
                    else:
                        yt = np.asarray(y_arr[va_i], dtype=float)
                        yp = np.asarray(preds, dtype=float)
                        fold_primary.append(_safe_r2(yt, yp))
                        fold_sec.append(float(root_mean_squared_error(yt, yp)))

                full_model = factory()
                full_model.fit(X, y_arr)
                fitted_models[name] = full_model

                mean_p = round(float(np.mean(fold_primary)), 5)
                std_p = round(float(np.std(fold_primary)), 5)
                mean_s = round(float(np.mean(fold_sec)), 5)
                if name.startswith("Parametric_"):
                    param_score = mean_p

                row: dict[str, Any] = {
                    "model_name": name,
                    "paradigm": paradigm,
                    "is_nonparametric": not name.startswith("Parametric_"),
                    "primary_metric": "F1 (Weighted)" if is_clf else "R²",
                    "primary_score": mean_p,
                    "cv_std": std_p,
                    "secondary_metric": "Accuracy" if is_clf else "RMSE",
                    "secondary_score": mean_s,
                }
                if hasattr(full_model, "effective_df_"):
                    row["effective_df"] = float(full_model.effective_df_)
                if hasattr(full_model, "bandwidth_"):
                    row["bandwidth_h"] = float(full_model.bandwidth_)
                if hasattr(full_model, "fitted_k_"):
                    row["k_neighbors"] = int(full_model.fitted_k_)
                leaderboard.append(row)
            except Exception:
                continue

        leaderboard.sort(key=lambda r: float(r["primary_score"]), reverse=True)
        for rank_i, row in enumerate(leaderboard, start=1):
            lift = round(float(row["primary_score"]) - float(param_score), 5)
            row["rank"] = rank_i
            row["lift_vs_parametric"] = lift
            if rank_i == 1:
                row["acceptance_status"] = "ACCEPTED — CHAMPION ANALYTICAL MODEL"
            elif row["is_nonparametric"] and lift > 0.0:
                row["acceptance_status"] = "ACCEPTED — BEATS PARAMETRIC LINEAR"
            elif not row["is_nonparametric"]:
                row["acceptance_status"] = "REFERENCE — PARAMETRIC LINEAR"
            elif float(row["primary_score"]) > 0.0:
                row["acceptance_status"] = "VALIDATED — NON-PARAMETRIC CANDIDATE"
            else:
                row["acceptance_status"] = "REJECTED — BELOW LINEAR REFERENCE"

            m_obj = fitted_models.get(row["model_name"])
            if m_obj is not None:
                eq = ModelEquationExtractor.extract(
                    m_obj,
                    X_reference=X.iloc[: min(len(X), 400)],
                    feature_names=list(X.columns),
                    target_name=target_name,
                )
                eq_d = eq.model_dump()
                eq_d["formula_plain"] = eq_d.get("plain_equation", "")
                eq_d["formula_latex"] = eq_d.get("latex_equation", "")
                eq_d["formula_python"] = (
                    f"# {row['model_name']} ({row['paradigm']})\n"
                    f"# {eq_d.get('plain_equation', '')}\n"
                    f"y_hat = model.predict(X)"
                )
                eq_d["rank"] = rank_i
                eq_d["paradigm"] = row["paradigm"]
                eq_d["model_paradigm"] = row["paradigm"]
                eq_d["acceptance_status"] = row["acceptance_status"]
                eq_d["primary_metric"] = row["primary_metric"]
                eq_d["primary_score"] = row["primary_score"]
                eq_d["naive_floor_score"] = param_score
                eq_d["lift_over_naive"] = lift
                eq_d["selection_criterion"] = (
                    f"{k_folds}-fold CV {row['primary_metric']} vs. Parametric Linear Reference ({param_score:.4f})"
                )
                reasons = [
                    f"{k_folds}-fold CV {row['primary_metric']} = {row['primary_score']:.4f} ± {row['cv_std']:.4f}",
                    f"Lift vs. Parametric Linear Reference = {lift:+.4f}",
                ]
                if "effective_df" in row:
                    reasons.append(f"Effective Non-Parametric Degrees of Freedom (EDF) = {row['effective_df']}")
                if "bandwidth_h" in row:
                    reasons.append(f"Cross-Validated Gaussian Kernel Bandwidth h = {row['bandwidth_h']}")
                eq_d["acceptance_reasons"] = reasons
                equations.append(eq_d)

        champion = leaderboard[0] if leaderboard else {}
        nonparam_best = next((r for r in leaderboard if r.get("is_nonparametric")), {})
        best_np_score = float(nonparam_best.get("primary_score", 0.0)) if nonparam_best else 0.0
        lift_vs_param = round(best_np_score - float(param_score), 5)
        np_beats_param = bool(nonparam_best and lift_vs_param > 0.015)
        primary_metric_name = str(nonparam_best.get("primary_metric") or champion.get("primary_metric") or ("f1_weighted" if is_clf else "r2"))
        linearity_verdict = (
            f"Non-parametric analytical model '{nonparam_best.get('model_name')}' outperforms parametric linear by "
            f"{lift_vs_param:+.4f} {primary_metric_name} — "
            "significant non-linear structure confirmed."
            if np_beats_param
            else "Parametric linear relationship is competitive with non-parametric smoothers; additive/linear terms capture most signal."
        )

        # Return champion fitted non-parametric model + full dict for Studio 14 supervisory registration
        return {
            "target": target_name,
            "task_type": task_type.value,
            "cv_folds": k_folds,
            "n_samples": int(n),
            "primary_metric": primary_metric_name,
            "elapsed_seconds": round(float(_time.perf_counter() - t0), 3),
            "parametric_reference_score": param_score,
            "champion_model": champion.get("model_name", ""),
            "best_model_name": nonparam_best.get("model_name") or champion.get("model_name", ""),
            "champion_paradigm": champion.get("paradigm", ""),
            "champion_score": champion.get("primary_score", 0.0),
            "best_nonparametric_score": best_np_score,
            "lift_vs_parametric": lift_vs_param,
            "nonparametric_outperforms_parametric": np_beats_param,
            "linearity_verdict": linearity_verdict,
            "verdict": linearity_verdict,
            "leaderboard": leaderboard,
            "equations": equations,
            "spline_components": getattr(
                fitted_models.get("NonParametric_SplineGAM_Clf" if is_clf else "NonParametric_SplineGAM"),
                "spline_component_summary_",
                [],
            ),
            "_fitted_models": fitted_models,
        }


class _ParametricRefWrapper(BaseOmniForgeModel):
    """Internal standardized Ridge / Logistic reference wrapper for NonParametricAnalyticalSuite."""

    model_family = "parametric_reference"

    def __init__(self, *, is_clf: bool, random_seed: int = 42) -> None:
        super().__init__(
            name="Parametric_LogisticLinear" if is_clf else "Parametric_RidgeLinear",
            task_type=TaskType.BINARY_CLASSIFICATION if is_clf else TaskType.REGRESSION,
            random_seed=random_seed,
        )
        self._is_clf = is_clf
        self._scaler = StandardScaler()
        self._model: Any = (
            LogisticRegression(C=1.0, max_iter=1000, random_state=random_seed)
            if is_clf
            else Ridge(alpha=1.0, random_state=random_seed)
        )

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        arr = _clean_matrix(X)
        self._model.fit(self._scaler.fit_transform(arr), y)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        arr = _clean_matrix(X)
        preds = self._model.predict(self._scaler.transform(arr))
        return np.asarray(preds) if self._is_clf else np.asarray(preds, dtype=float)

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        if not self._is_clf:
            raise NotImplementedError("predict_proba only available for classification.")
        arr = _clean_matrix(X)
        return np.asarray(self._model.predict_proba(self._scaler.transform(arr)), dtype=float)
