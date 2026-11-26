"""Coherent Hierarchical Time-Series Reconciliation: Bottom-Up, Top-Down, OLS, WLS, and MinT-Shrink (Schäfer-Strimmer)."""

from __future__ import annotations

from urllib.parse import quote

import numpy as np
import pandas as pd

from omniforge.core.exceptions import ForecastingError


class HierarchicalReconciler:
    """Reconciles base forecasts across an aggregation hierarchy S so all levels sum coherently.

    Given summing matrix ``S`` of shape ``(n_total, n_bottom)`` and base forecasts ``y_hat``
    of shape ``(n_total, horizon)``, reconciled forecasts satisfy ``y_tilde = S @ G @ y_hat``.
    """

    def __init__(self, method: str = "ols") -> None:
        self.method = method.lower()
        if self.method not in {"bottom_up", "top_down", "ols", "wls", "wls_var", "wls_struct", "mint_shrink"}:
            raise ForecastingError("Unknown hierarchical reconciliation method.")
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
        if S.ndim != 2 or not S.size or Y.ndim != 2 or not Y.size or not np.isfinite(S).all() or not np.isfinite(Y).all():
            raise ForecastingError("The summing matrix and forecasts must be nonempty, finite matrices.")
        n_total, n_bottom = S.shape
        if n_total < n_bottom or not np.array_equal(S[-n_bottom:], np.eye(n_bottom)):
            raise ForecastingError("The summing matrix must end with one identity row per bottom-level series.")
        self.last_shrinkage_lambda_ = None
        if Y.shape[0] != n_total:
            raise ForecastingError(
                f"Base forecasts row count ({Y.shape[0]}) must match summing matrix rows ({n_total})."
            )

        if self.method == "bottom_up":
            bottom = Y[-n_bottom:, :]
            return np.asarray(S @ bottom, dtype=float)

        if self.method == "top_down":
            if not np.all(S[0] == 1):
                raise ForecastingError("Top-down reconciliation requires the total series in the first row.")
            top = Y[0:1, :]
            if historical_bottom_shares is None:
                shares = np.full((n_bottom, 1), 1.0 / max(n_bottom, 1), dtype=float)
            else:
                sh = np.asarray(historical_bottom_shares, dtype=float)
                if sh.shape not in {(n_bottom,), (n_bottom, 1)} or not np.isfinite(sh).all() or (sh < 0).any() or sh.sum() <= 0:
                    raise ForecastingError("Historical shares must be finite, nonnegative, and sum to a positive value.")
                shares = sh.reshape(n_bottom, 1) / sh.sum()
            return np.asarray(S @ (shares @ top), dtype=float)

        res_arr = None
        if self.method in {"wls", "wls_var", "mint_shrink"} and residuals is not None:
            res_arr = np.asarray(residuals, dtype=float)
            if res_arr.ndim != 2 or res_arr.shape[0] != n_total or not res_arr.shape[1] or not np.isfinite(res_arr).all():
                raise ForecastingError("Residuals must be finite and aligned to every series in the summing matrix.")
        if self.method in ("wls", "wls_var", "wls_struct"):
            if self.method == "wls_var" and res_arr is None:
                raise ForecastingError("Variance-weighted reconciliation requires aligned residuals.")
            if res_arr is not None:
                ddof_val = 1 if res_arr.ndim >= 2 and res_arr.shape[1] > 1 else 0
                variances = np.var(res_arr, axis=1, ddof=ddof_val) + 1e-6
            else:
                # Structural WLS: weight proportional to number of bottom series summed
                variances = np.maximum(np.sum(S, axis=1), 1.0)
            W_inv = np.diag(1.0 / variances)
            G = np.linalg.pinv(S.T @ W_inv @ S) @ S.T @ W_inv
            return np.asarray(S @ G @ Y, dtype=float)

        if self.method == "mint_shrink":
            if res_arr is None or res_arr.shape[1] < 2:
                raise ForecastingError("MinT shrinkage requires at least two residual observations per series.")
            W, lam = self._schafer_strimmer_shrinkage(res_arr)
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
        """Use full paths as series identities and order bottom columns by path.

        Repeated observations of a path describe the same series. Repeated node
        names under different parents describe separate series.
        """
        if not level_cols or len(set(level_cols)) != len(level_cols) or not hierarchy_df.columns.is_unique:
            raise ForecastingError("Specify distinct hierarchy level columns in a uniquely named table.")
        if any(col not in hierarchy_df.columns for col in level_cols):
            raise ForecastingError("A hierarchy level column is absent from the table.")
        hierarchy = hierarchy_df.loc[:, level_cols]
        if hierarchy.empty or hierarchy.isna().any().any():
            raise ForecastingError("Each hierarchy row must provide an observed identifier at every level.")

        # Escape separators so free-text values cannot alias a different path.
        # Reject values whose distinct source identifiers render identically.
        normalized = hierarchy.map(lambda value: quote(str(value), safe=""))
        for col in level_cols:
            if hierarchy[col].nunique() != normalized[col].nunique():
                raise ForecastingError("Distinct hierarchy identifiers must have distinct textual representations.")
        paths = sorted(set(normalized.itertuples(index=False, name=None)))
        n_bottom = len(paths)
        rows: list[np.ndarray] = [np.ones(n_bottom, dtype=float)]
        labels: list[str] = ["Total"]
        for depth in range(1, len(level_cols) + 1):
            indices: dict[tuple[str, ...], list[int]] = {}
            for index, path in enumerate(paths):
                indices.setdefault(path[:depth], []).append(index)
            for path, columns in indices.items():
                row = np.zeros(n_bottom, dtype=float)
                row[columns] = 1.0
                rows.append(row)
                labels.append("/".join(f"{quote(col, safe='')}={value}" for col, value in zip(level_cols, path, strict=False)))
        return np.vstack(rows), labels
