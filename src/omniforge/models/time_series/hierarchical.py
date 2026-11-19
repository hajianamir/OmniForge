"""Coherent Hierarchical Time-Series Reconciliation: Bottom-Up, Top-Down, OLS, WLS, and MinT-Shrink (Schäfer-Strimmer)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.core.exceptions import ForecastingError


class HierarchicalReconciler:
    """Reconciles base forecasts across an aggregation hierarchy S so all levels sum coherently.

    Given summing matrix ``S`` of shape ``(n_total, n_bottom)`` and base forecasts ``y_hat``
    of shape ``(n_total, horizon)``, reconciled forecasts satisfy ``y_tilde = S @ G @ y_hat``.
    """

    def __init__(self, method: str = "ols") -> None:
        self.method = method.lower()  # bottom_up, top_down, ols, wls, wls_var, wls_struct, mint_shrink
        self.last_shrinkage_lambda_: float | None = None

    @staticmethod
    def _schafer_strimmer_shrinkage(residuals: np.ndarray) -> tuple[np.ndarray, float]:
        """Compute Wickramasuriya et al. (2019) MinT-Shrink covariance with optimal Schäfer-Strimmer lambda."""
        E = np.asarray(residuals, dtype=float)
        n_total, T = E.shape
        if T <= 2:
            diag = np.diag(np.var(E, axis=1, ddof=1 if T > 1 else 0) + 1e-6)
            return diag, 1.0
        # Centered residuals
        E_c = E - E.mean(axis=1, keepdims=True)
        W_sample = (E_c @ E_c.T) / max(T - 1, 1)
        stds = np.sqrt(np.maximum(np.diag(W_sample), 1e-12))
        Z = E_c / stds[:, None]  # standardized residuals (n_total, T)
        R_hat = (Z @ Z.T) / max(T - 1, 1)
        # Variance of off-diagonal correlation estimates (Schäfer & Strimmer 2005, Appendix A):
        # Var(r_ij) = (T / (T - 1)^3) * sum_t (z_it * z_jt - r_bar_ij)^2
        r_bar = (Z @ Z.T) / T
        num = 0.0
        den = 0.0
        for i in range(n_total):
            for j in range(i + 1, n_total):
                prod_t = Z[i] * Z[j]
                var_rij = float(np.sum((prod_t - r_bar[i, j]) ** 2)) * (T / ((T - 1) ** 3))
                num += var_rij
                den += float(R_hat[i, j] ** 2)
        lam = float(np.clip(num / max(den, 1e-12), 0.0, 1.0)) if den > 1e-12 else (1.0 if num > 0.0 else 0.0)
        W_diag = np.diag(np.diag(W_sample))
        ridge = max(1e-12, 1e-8 * float(np.mean(np.diag(W_sample))))
        W_shrunk = (1.0 - lam) * W_sample + lam * W_diag + np.eye(n_total) * ridge
        return W_shrunk, lam

    def reconcile(
        self,
        summing_matrix: np.ndarray,
        base_forecasts: np.ndarray,
        *,
        residuals: np.ndarray | None = None,
        historical_bottom_shares: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return coherent reconciled forecasts of shape ``(n_total, horizon)``."""
        S = np.asarray(summing_matrix, dtype=float)
        Y = np.asarray(base_forecasts, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        n_total, n_bottom = S.shape
        if Y.shape[0] != n_total:
            raise ForecastingError(
                f"Base forecasts row count ({Y.shape[0]}) must match summing matrix rows ({n_total})."
            )

        if self.method == "bottom_up":
            bottom = Y[-n_bottom:, :]
            return np.asarray(S @ bottom, dtype=float)

        if self.method == "top_down":
            top = Y[0:1, :]
            if historical_bottom_shares is None:
                shares = np.full((n_bottom, 1), 1.0 / max(n_bottom, 1), dtype=float)
            else:
                sh = np.asarray(historical_bottom_shares, dtype=float).reshape(n_bottom, 1)
                shares = sh / max(float(sh.sum()), 1e-12)
            return np.asarray(S @ (shares @ top), dtype=float)

        if self.method in ("wls", "wls_var", "wls_struct"):
            if self.method != "wls_struct" and residuals is not None and residuals.shape[0] == n_total:
                res_arr = np.asarray(residuals, dtype=float)
                ddof_val = 1 if res_arr.ndim >= 2 and res_arr.shape[1] > 1 else 0
                variances = np.var(res_arr, axis=1, ddof=ddof_val) + 1e-6
            else:
                # Structural WLS: weight proportional to number of bottom series summed
                variances = np.maximum(np.sum(S, axis=1), 1.0)
            W_inv = np.diag(1.0 / variances)
            G = np.linalg.pinv(S.T @ W_inv @ S) @ S.T @ W_inv
            return np.asarray(S @ G @ Y, dtype=float)

        if self.method == "mint_shrink" and residuals is not None and residuals.shape[0] == n_total:
            W, lam = self._schafer_strimmer_shrinkage(residuals)
            self.last_shrinkage_lambda_ = round(lam, 6)
            W_inv = np.linalg.pinv(W)
            G = np.linalg.pinv(S.T @ W_inv @ S) @ S.T @ W_inv
            return np.asarray(S @ G @ Y, dtype=float)

        # Default OLS projection: S (S^T S)^{-1} S^T Y
        G = np.linalg.pinv(S.T @ S) @ S.T
        return np.asarray(S @ G @ Y, dtype=float)

    @staticmethod
    def verify_coherence(summing_matrix: np.ndarray, reconciled: np.ndarray, tol: float = 1e-6) -> dict[str, float | bool]:
        """Verify that upper-level reconciled forecasts equal exact sums of reconciled bottom-level series."""
        S = np.asarray(summing_matrix, dtype=float)
        R = np.asarray(reconciled, dtype=float)
        if R.ndim == 1:
            R = R[:, None]
        n_bottom = S.shape[1]
        reconstructed = S @ R[-n_bottom:, :]
        max_abs_err = float(np.max(np.abs(R - reconstructed)))
        return {"coherent": bool(max_abs_err <= tol), "max_abs_coherence_error": round(max_abs_err, 10)}

    @staticmethod
    def build_summing_matrix(hierarchy_df: pd.DataFrame, level_cols: list[str]) -> tuple[np.ndarray, list[str]]:
        """Construct summing matrix ``S`` and ordered series labels from a tidy hierarchy table."""
        bottom_col = level_cols[-1]
        bottom_ids = sorted(hierarchy_df[bottom_col].astype(str).unique().tolist())
        n_bottom = len(bottom_ids)
        bottom_index = {b: i for i, b in enumerate(bottom_ids)}

        rows: list[np.ndarray] = [np.ones(n_bottom, dtype=float)]
        labels: list[str] = ["Total"]

        for col in level_cols[:-1]:
            for grp, sub in hierarchy_df.groupby(col, observed=True):
                vec = np.zeros(n_bottom, dtype=float)
                for b in sub[bottom_col].astype(str).unique():
                    vec[bottom_index[b]] = 1.0
                rows.append(vec)
                labels.append(f"{col}={grp}")

        for b in bottom_ids:
            vec = np.zeros(n_bottom, dtype=float)
            vec[bottom_index[b]] = 1.0
            rows.append(vec)
            labels.append(f"{bottom_col}={b}")

        return np.vstack(rows), labels
