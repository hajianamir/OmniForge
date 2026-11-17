"""Comprehensive evaluation metrics suite for classification, regression, forecasting, clustering, and finance."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    calinski_harabasz_score,
    cohen_kappa_score,
    confusion_matrix,
    davies_bouldin_score,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)

from omniforge.core.logging import get_logger

logger = get_logger(__name__)


class MetricsEvaluator:
    """Computes task-appropriate enterprise evaluation metrics with numerical edge-case safety."""

    @staticmethod
    def classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute full suite of binary or multiclass classification metrics."""
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        classes = np.unique(yt)
        is_binary = len(classes) <= 2

        metrics: dict[str, Any] = {
            "accuracy": round(float(accuracy_score(yt, yp)), 6),
            "balanced_accuracy": round(float(balanced_accuracy_score(yt, yp)), 6),
            "precision_weighted": round(float(precision_score(yt, yp, average="weighted", zero_division=0)), 6),
            "recall_weighted": round(float(recall_score(yt, yp, average="weighted", zero_division=0)), 6),
            "f1_weighted": round(float(f1_score(yt, yp, average="weighted", zero_division=0)), 6),
            "f1_macro": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 6),
            "mcc": round(float(matthews_corrcoef(yt, yp)), 6),
            "cohen_kappa": round(float(cohen_kappa_score(yt, yp)), 6),
            "confusion_matrix": confusion_matrix(yt, yp).tolist(),
        }

        if y_proba is not None and len(classes) >= 2:
            proba = np.asarray(y_proba, dtype=float)
            try:
                metrics["log_loss"] = round(float(log_loss(yt, np.clip(proba, 1e-12, 1.0 - 1e-12), labels=classes)), 6)
            except Exception as exc:
                logger.debug("log_loss_metric_skipped", error=str(exc))
            if is_binary and proba.ndim == 2 and proba.shape[1] >= 2:
                pos_p = proba[:, 1]
                y_bin = (yt == classes[-1]).astype(int)
                try:
                    auc = float(roc_auc_score(y_bin, pos_p))
                    metrics["auc_roc"] = round(auc, 6)
                    metrics["gini_coefficient"] = round(2.0 * auc - 1.0, 6)
                    metrics["auc_pr"] = round(float(average_precision_score(y_bin, pos_p)), 6)
                    metrics["brier_score"] = round(float(brier_score_loss(y_bin, pos_p)), 6)
                    # Kolmogorov-Smirnov separation statistic
                    pos_scores = np.sort(pos_p[y_bin == 1])
                    neg_scores = np.sort(pos_p[y_bin == 0])
                    if len(pos_scores) > 0 and len(neg_scores) > 0:
                        grid = np.unique(pos_p)
                        cdf_pos = np.searchsorted(pos_scores, grid, side="right") / len(pos_scores)
                        cdf_neg = np.searchsorted(neg_scores, grid, side="right") / len(neg_scores)
                        metrics["ks_statistic"] = round(float(np.max(np.abs(cdf_pos - cdf_neg))), 6)
                except Exception as exc:
                    logger.debug("binary_probability_metrics_skipped", error=str(exc))
            elif not is_binary and proba.ndim == 2 and proba.shape[1] == len(classes):
                try:
                    metrics["auc_roc_ovr"] = round(
                        float(roc_auc_score(yt, proba, multi_class="ovr", average="weighted")), 6
                    )
                except Exception as exc:
                    logger.debug("multiclass_auc_metric_skipped", error=str(exc))
        return metrics

    @staticmethod
    def regression_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, *, n_features: int | None = None
    ) -> dict[str, float]:
        """Compute comprehensive continuous regression metrics."""
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        mse = float(mean_squared_error(yt, yp))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))
        non_zero = np.abs(yt) > 1e-9
        mape = float(np.mean(np.abs((yt[non_zero] - yp[non_zero]) / yt[non_zero])) * 100.0) if np.any(non_zero) else 0.0
        smape = float(np.mean(np.abs(yt - yp) / ((np.abs(yt) + np.abs(yp)) / 2.0 + 1e-9)) * 100.0)

        out = {
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "mse": round(mse, 6),
            "median_ae": round(float(median_absolute_error(yt, yp)), 6),
            "r2": round(r2, 6),
            "explained_variance": round(float(explained_variance_score(yt, yp)), 6),
            "mape": round(mape, 4),
            "smape": round(smape, 4),
            "max_error": round(float(np.max(np.abs(yt - yp))) if len(yt) > 0 else 0.0, 6),
        }
        if n_features is not None and len(yt) > n_features + 1:
            n = len(yt)
            adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - n_features - 1, 1)
            out["adjusted_r2"] = round(float(adj_r2), 6)
        return out

    @staticmethod
    def forecasting_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        y_train: np.ndarray | None = None,
        y_lower: np.ndarray | None = None,
        y_upper: np.ndarray | None = None,
        alpha: float = 0.05,
        seasonal_period: int = 1,
    ) -> dict[str, float]:
        """Compute point and probabilistic forecasting metrics (RMSE, MAE, sMAPE, MASE, Winkler)."""
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        smape = float(np.mean(np.abs(yt - yp) / ((np.abs(yt) + np.abs(yp)) / 2.0 + 1e-9)) * 100.0)
        res: dict[str, float] = {"rmse": round(rmse, 6), "mae": round(mae, 6), "smape": round(smape, 4)}

        if y_train is not None and len(y_train) > seasonal_period:
            tr = np.asarray(y_train, dtype=float).ravel()
            naive_mae = float(np.mean(np.abs(tr[seasonal_period:] - tr[:-seasonal_period])))
            res["mase"] = round(mae / max(naive_mae, 1e-9), 6)

        if y_lower is not None and y_upper is not None:
            lo = np.asarray(y_lower, dtype=float).ravel()
            hi = np.asarray(y_upper, dtype=float).ravel()
            coverage = float(np.mean((yt >= lo) & (yt <= hi)))
            width = hi - lo
            penalty = (2.0 / max(alpha, 1e-6)) * (np.maximum(lo - yt, 0.0) + np.maximum(yt - hi, 0.0))
            winkler = float(np.mean(width + penalty))
            res["interval_coverage"] = round(coverage, 4)
            res["winkler_score"] = round(winkler, 6)
        return res

    @staticmethod
    def clustering_metrics(X: np.ndarray, labels: np.ndarray, *, max_silhouette_samples: int = 4000) -> dict[str, float]:
        """Compute unsupervised internal cluster validity indices (O(N)-safe sampled Silhouette on large datasets)."""
        uniq = np.unique(labels)
        if len(uniq) < 2 or len(uniq) >= len(X):
            return {"silhouette": 0.0, "davies_bouldin": 0.0, "calinski_harabasz": 0.0}
        sample_sz = max_silhouette_samples if len(X) > max_silhouette_samples else None
        sil = float(silhouette_score(X, labels, sample_size=sample_sz, random_state=42))
        return {
            "silhouette": round(sil, 6),
            "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 6),
            "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 6),
        }

    @staticmethod
    def financial_metrics(returns: np.ndarray, periods_per_year: int = 252) -> dict[str, float]:
        """Compute risk-adjusted financial performance metrics from periodic returns."""
        r = np.asarray(returns, dtype=float).ravel()
        if len(r) == 0:
            return {"sharpe_ratio": 0.0, "sortino_ratio": 0.0, "max_drawdown": 0.0}
        mean_r = float(np.mean(r))
        std_r = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
        downside = r[r < 0]
        down_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-9
        sharpe = (mean_r / max(std_r, 1e-9)) * np.sqrt(periods_per_year)
        sortino = (mean_r / max(down_std, 1e-9)) * np.sqrt(periods_per_year)
        # Log-space cumulative equity avoids float64 overflow on extreme stress series
        log_equity = np.cumsum(np.log1p(np.clip(r, -0.999999, 1e6)))
        log_peaks = np.maximum.accumulate(log_equity)
        drawdowns = np.expm1(np.clip(log_equity - log_peaks, -700.0, 0.0))
        max_dd = float(np.min(drawdowns))
        return {
            "annualized_return": round(mean_r * periods_per_year, 6),
            "annualized_volatility": round(std_r * np.sqrt(periods_per_year), 6),
            "sharpe_ratio": round(float(sharpe), 4),
            "sortino_ratio": round(float(sortino), 4),
            "max_drawdown": round(max_dd, 6),
        }
