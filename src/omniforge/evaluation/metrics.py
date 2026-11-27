"""Comprehensive evaluation metrics suite for classification, regression, forecasting, clustering, and finance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import logsumexp
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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

from omniforge.core.exceptions import EvaluationError
from omniforge.core.logging import get_logger

_EULER_MASCHERONI = 0.5772156649015329

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
        all_labels = np.unique(np.concatenate([yt, yp])) if len(yt) > 0 else classes
        is_binary = len(classes) <= 2

        acc = float(accuracy_score(yt, yp))
        if len(all_labels) >= 2:
            bal_acc = float(balanced_accuracy_score(yt, yp))
            mcc_val = float(matthews_corrcoef(yt, yp))
            kappa_raw = float(cohen_kappa_score(yt, yp))
            kappa_val = kappa_raw if np.isfinite(kappa_raw) else 0.0
            cm_list = confusion_matrix(yt, yp, labels=all_labels).tolist()
        else:
            bal_acc = acc
            mcc_val = 0.0
            kappa_val = 0.0
            cm_list = [[int(len(yt))]]

        metrics: dict[str, Any] = {
            "accuracy": round(acc, 6),
            "balanced_accuracy": round(bal_acc, 6),
            "precision_weighted": round(float(precision_score(yt, yp, average="weighted", zero_division=0)), 6),
            "recall_weighted": round(float(recall_score(yt, yp, average="weighted", zero_division=0)), 6),
            "f1_weighted": round(float(f1_score(yt, yp, average="weighted", zero_division=0)), 6),
            "f1_macro": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 6),
            "mcc": round(mcc_val, 6),
            "cohen_kappa": round(kappa_val, 6),
            "confusion_matrix": cm_list,
        }

        if y_proba is not None and len(classes) >= 2:
            proba = np.asarray(y_proba, dtype=float)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                proba_safe = np.clip(proba, 1e-12, 1.0)
                row_sums = proba_safe.sum(axis=1, keepdims=True)
                proba_norm = np.where(row_sums > 0.0, proba_safe / np.where(row_sums > 0.0, row_sums, 1.0), 1.0 / proba.shape[1])
            else:
                proba_norm = np.clip(proba, 1e-12, 1.0 - 1e-12)
            try:
                ll_labels = (
                    np.arange(proba_norm.shape[1])
                    if (
                        proba_norm.ndim == 2
                        and proba_norm.shape[1] > len(classes)
                        and np.issubdtype(yt.dtype, np.integer)
                        and int(np.min(yt)) >= 0
                        and int(np.max(yt)) < proba_norm.shape[1]
                    )
                    else classes
                )
                metrics["log_loss"] = round(float(log_loss(yt, proba_norm, labels=ll_labels)), 6)
            except Exception as exc:
                logger.debug("log_loss_metric_skipped", error=str(exc))
            if is_binary and ((proba.ndim == 2 and proba.shape[1] == 2) or proba.ndim == 1):
                pos_p = proba_norm[:, 1] if proba.ndim == 2 else proba_norm.ravel()
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
                        float(roc_auc_score(yt, proba_norm, multi_class="ovr", average="weighted")), 6
                    )
                except Exception as exc:
                    logger.debug("multiclass_auc_metric_skipped", error=str(exc))
            try:
                cal = MetricsEvaluator.calibration_metrics(yt, proba_norm, n_bins=10, classes=classes)
                metrics["ece"] = cal["ece"]
                metrics["mce"] = cal["mce"]
                metrics["brier_reliability"] = cal["brier_reliability"]
                metrics["brier_resolution"] = cal["brier_resolution"]
            except Exception as exc:
                logger.debug("calibration_metrics_skipped", error=str(exc))
        return metrics

    @staticmethod
    def calibration_metrics(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
        classes: np.ndarray | list[Any] | None = None,
    ) -> dict[str, Any]:
        """Compute ECE, MCE (Naeini et al. 2015; Guo et al. 2017), and Murphy (1973) Brier decomposition.

        For binary classification (1D or 2-column ``y_proba``), evaluates positive-class probability
        calibration against ``o_i = 1[y_i == pos_class]``. For multiclass ``(N, K)`` probabilities,
        evaluates top-label confidence ``f_i = max_k p_{ik}`` against ``o_i = 1[argmax_k p_{ik} == y_i]``.
        """
        yt = np.asarray(y_true)
        proba = np.asarray(y_proba, dtype=float)
        n = len(yt)
        if n == 0:
            return {
                "ece": 0.0,
                "mce": 0.0,
                "brier_reliability": 0.0,
                "brier_resolution": 0.0,
                "brier_uncertainty": 0.0,
                "binned_brier_score": 0.0,
                "reliability_bins": [],
            }
        cls_arr = np.asarray(classes) if classes is not None else np.unique(yt)
        if proba.ndim == 1 or (proba.ndim == 2 and proba.shape[1] == 1):
            conf = np.clip(proba.ravel(), 0.0, 1.0)
            pos_label = cls_arr[-1] if len(cls_arr) >= 2 else 1
            outcomes = (yt == pos_label).astype(float)
        elif proba.ndim == 2 and proba.shape[1] == 2 and len(cls_arr) <= 2:
            row_sums = np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)
            p_norm = np.clip(proba / row_sums, 0.0, 1.0)
            conf = p_norm[:, 1]
            pos_label = cls_arr[-1] if len(cls_arr) >= 2 else 1
            outcomes = (yt == pos_label).astype(float)
        elif proba.ndim == 2 and proba.shape[1] >= 2:
            row_sums = np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)
            p_norm = np.clip(proba / row_sums, 0.0, 1.0)
            pred_idx = np.argmax(p_norm, axis=1)
            conf = np.max(p_norm, axis=1)
            if len(cls_arr) == p_norm.shape[1]:
                pred_labels = cls_arr[pred_idx]
            else:
                pred_labels = pred_idx
            outcomes = (pred_labels == yt).astype(float)
        else:
            raise EvaluationError(f"Unsupported y_proba shape {proba.shape} for calibration_metrics.")

        b_cnt = max(2, int(n_bins))
        if strategy == "quantile" and n >= b_cnt:
            edges = np.quantile(conf, np.linspace(0.0, 1.0, b_cnt + 1))
            edges[0] = 0.0
            edges[-1] = 1.0
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([0.0, 1.0], dtype=float)
        else:
            edges = np.linspace(0.0, 1.0, b_cnt + 1)

        # Assign observations cleanly into [edges[b], edges[b+1]], including 0.0 in bin 0
        bin_ids = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, len(edges) - 2)
        base_rate = float(np.mean(outcomes))
        uncertainty = base_rate * (1.0 - base_rate)
        ece = 0.0
        mce = 0.0
        reliability = 0.0
        resolution = 0.0
        bins_out: list[dict[str, float | int]] = []

        for b in range(len(edges) - 1):
            mask = bin_ids == b
            cnt = int(np.sum(mask))
            if cnt == 0:
                continue
            avg_conf = float(np.mean(conf[mask]))
            avg_acc = float(np.mean(outcomes[mask]))
            gap = abs(avg_acc - avg_conf)
            weight = cnt / n
            ece += weight * gap
            mce = max(mce, gap)
            reliability += weight * ((avg_conf - avg_acc) ** 2)
            resolution += weight * ((avg_acc - base_rate) ** 2)
            bins_out.append(
                {
                    "bin": int(b),
                    "lower": round(float(edges[b]), 6),
                    "upper": round(float(edges[b + 1]), 6),
                    "count": cnt,
                    "mean_confidence": round(avg_conf, 6),
                    "empirical_accuracy": round(avg_acc, 6),
                    "calibration_gap": round(gap, 6),
                }
            )

        binned_bs = reliability - resolution + uncertainty
        return {
            "ece": round(float(ece), 6),
            "mce": round(float(mce), 6),
            "brier_reliability": round(float(reliability), 6),
            "brier_resolution": round(float(resolution), 6),
            "brier_uncertainty": round(float(uncertainty), 6),
            "binned_brier_score": round(float(binned_bs), 6),
            "reliability_bins": bins_out,
        }

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
        r2 = float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan")
        expl_var = float(explained_variance_score(yt, yp)) if len(yt) >= 2 else float("nan")
        non_zero = np.abs(yt) > 1e-12
        mape = float(np.mean(np.abs((yt[non_zero] - yp[non_zero]) / yt[non_zero])) * 100.0) if np.any(non_zero) else float("nan")
        smape_denom = (np.abs(yt) + np.abs(yp)) / 2.0
        smape_terms = np.where(smape_denom > 0.0, np.abs(yt - yp) / np.where(smape_denom > 0.0, smape_denom, 1.0), 0.0)
        smape = float(np.mean(smape_terms) * 100.0) if len(yt) > 0 else 0.0

        out = {
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "mse": round(mse, 6),
            "median_ae": round(float(median_absolute_error(yt, yp)), 6),
            "r2": round(r2, 6) if np.isfinite(r2) else float("nan"),
            "explained_variance": round(expl_var, 6) if np.isfinite(expl_var) else float("nan"),
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
        smape_denom = (np.abs(yt) + np.abs(yp)) / 2.0
        smape_terms = np.where(smape_denom > 0.0, np.abs(yt - yp) / np.where(smape_denom > 0.0, smape_denom, 1.0), 0.0)
        smape = float(np.mean(smape_terms) * 100.0) if len(yt) > 0 else 0.0
        res: dict[str, float] = {"rmse": round(rmse, 6), "mae": round(mae, 6), "smape": round(smape, 4)}

        if y_train is not None and len(y_train) > seasonal_period:
            tr = np.asarray(y_train, dtype=float).ravel()
            naive_mae = float(np.mean(np.abs(tr[seasonal_period:] - tr[:-seasonal_period])))
            res["mase"] = round(mae / max(naive_mae, 1e-9), 6)

        if y_lower is not None and y_upper is not None:
            lo = np.asarray(y_lower, dtype=float).ravel()
            hi = np.asarray(y_upper, dtype=float).ravel()
            if len(lo) != len(yt) or len(hi) != len(yt):
                raise ValueError(
                    f"y_lower and y_upper must have same length as y_true "
                    f"({len(yt)}), got {len(lo)} and {len(hi)}."
                )
            inverted = lo > hi
            if np.any(inverted):
                n_inv = int(inverted.sum())
                raise ValueError(
                    f"Inverted prediction intervals: {n_inv} sample(s) have "
                    f"y_lower > y_upper. First inverted pair: "
                    f"y_lower={float(lo[inverted][0])}, y_upper={float(hi[inverted][0])}."
                )
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
    def financial_metrics(
        returns: np.ndarray,
        periods_per_year: int = 252,
        *,
        mar: float = 0.0,
    ) -> dict[str, float]:
        """Compute risk-adjusted financial performance metrics from periodic returns.

        Uses the authoritative Target Downside Deviation (TDD = sqrt(LPM_2(MAR)))
        defined by Sortino & Price (1994) and Rollinger & Hoffman (2013), dividing
        squared shortfalls below ``mar`` by the full observation count ``N`` rather
        than only the count of losing periods.
        """
        r = np.asarray(returns, dtype=float).ravel()
        r = r[np.isfinite(r)]
        if len(r) == 0:
            return {
                "annualized_return": 0.0,
                "cagr": 0.0,
                "annualized_volatility": 0.0,
                "annualized_downside_deviation": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
                "cvar_95": 0.0,
                "probabilistic_sharpe_ratio": 0.0,
            }
        mean_r = float(np.mean(r))
        excess = r - float(mar)
        mean_excess = float(np.mean(excess))
        std_r = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0

        # Target Downside Deviation (Sortino & Price 1994): sqrt((1/N) * sum(min(0, r_i - MAR)^2))
        shortfall = np.minimum(0.0, excess)
        tdd = float(np.sqrt(np.mean(shortfall**2)))

        p_yr = max(int(periods_per_year), 1)
        ann_scale = float(np.sqrt(p_yr))
        sharpe = (mean_excess / std_r) * ann_scale if std_r > 1e-12 else 0.0
        if tdd > 1e-12:
            sortino = (mean_excess / tdd) * ann_scale
        elif mean_excess > 1e-12:
            sortino = float("inf")
        else:
            sortino = 0.0

        # Log-space cumulative equity starting at initial wealth W_0 = 1.0 (log W_0 = 0.0)
        wiped_out = bool(np.any(r <= -1.0))
        log_equity = np.concatenate(([0.0], np.cumsum(np.log1p(np.clip(r, -0.999999, 1e6)))))
        log_peaks = np.maximum.accumulate(log_equity)
        drawdowns = np.expm1(np.clip(log_equity - log_peaks, -700.0, 0.0))
        max_dd = -1.0 if wiped_out else float(np.min(drawdowns))

        # Exact geometric Compound Annual Growth Rate (CAGR): (W_T)^(periods_per_year / T) - 1
        if wiped_out:
            cagr = -1.0
        else:
            ann_log_growth = float(log_equity[-1]) * (float(p_yr) / float(len(r)))
            cagr = float(np.expm1(np.clip(ann_log_growth, -700.0, 700.0)))

        # Young (1991) Calmar Ratio: CAGR / |MaxDD|
        if abs(max_dd) > 1e-12:
            calmar = cagr / abs(max_dd)
        elif cagr > 1e-12:
            calmar = float("inf")
        else:
            calmar = 0.0

        # Historical 95% Value-at-Risk & Expected Shortfall / CVaR (Artzner et al. 1999; positive loss convention)
        q05 = float(np.quantile(r, 0.05))
        tail_losses = r[r <= q05]
        var_95 = max(0.0, -q05)
        cvar_95 = max(0.0, -float(np.mean(tail_losses))) if len(tail_losses) > 0 else var_95

        psr_info = MetricsEvaluator.deflated_sharpe_ratio(r, benchmark_sr=0.0, n_trials=1, mar=mar)
        return {
            "annualized_return": round(mean_r * p_yr, 6),
            "cagr": round(cagr, 6),
            "annualized_volatility": round(std_r * ann_scale, 6),
            "annualized_downside_deviation": round(tdd * ann_scale, 6),
            "sharpe_ratio": round(float(sharpe), 4) if np.isfinite(sharpe) else float(sharpe),
            "sortino_ratio": round(float(sortino), 4) if np.isfinite(sortino) else float(sortino),
            "calmar_ratio": round(float(calmar), 4) if np.isfinite(calmar) else float(calmar),
            "max_drawdown": round(max_dd, 6),
            "var_95": round(var_95, 6),
            "cvar_95": round(cvar_95, 6),
            "probabilistic_sharpe_ratio": psr_info["probabilistic_sharpe_ratio"],
        }

    @staticmethod
    def deflated_sharpe_ratio(
        returns: np.ndarray,
        *,
        benchmark_sr: float = 0.0,
        n_trials: int = 1,
        trials_sr_std: float | None = None,
        mar: float = 0.0,
    ) -> dict[str, float]:
        """Compute Mertens (2002) Probabilistic Sharpe Ratio & Bailey–López de Prado (2014) Deflated Sharpe Ratio.

        Accounts for non-normal skewness ``gamma_3``, Pearson kurtosis ``gamma_4``, finite sample
        length ``T``, and multiple-testing selection bias across ``n_trials`` candidate strategies.
        ``benchmark_sr`` and ``trials_sr_std`` are expressed in non-annualized per-period Sharpe units.
        """
        r = np.asarray(returns, dtype=float).ravel()
        r = r[np.isfinite(r)]
        t_obs = len(r)
        if t_obs < 3:
            return {
                "non_annualized_sharpe": 0.0,
                "sharpe_std_error": 1.0,
                "expected_max_null_sharpe": float(benchmark_sr),
                "probabilistic_sharpe_ratio": 0.5,
                "deflated_sharpe_ratio": 0.5,
            }
        excess = r - float(mar)
        std_r = float(np.std(excess, ddof=1))
        if std_r <= 1e-12:
            sr_hat = 0.0
            skew_val = 0.0
            kurt_pearson = 3.0
        else:
            sr_hat = float(np.mean(excess) / std_r)
            skew_val = float(stats.skew(excess, bias=False)) if t_obs >= 3 else 0.0
            kurt_pearson = float(stats.kurtosis(excess, fisher=False, bias=False)) if t_obs >= 4 else 3.0
            if not np.isfinite(skew_val):
                skew_val = 0.0
            if not np.isfinite(kurt_pearson):
                kurt_pearson = 3.0

        # Mertens (2002) / Opdyke (2007) exact asymptotic variance of sample Sharpe ratio:
        # Var(SR_hat) = [1 - gamma_3 * SR_hat + ((gamma_4 - 1) / 4) * SR_hat^2] / (T - 1)
        var_num = max(1e-12, 1.0 - skew_val * sr_hat + ((kurt_pearson - 1.0) / 4.0) * (sr_hat**2))
        se_sr = float(np.sqrt(var_num / max(t_obs - 1, 1)))

        z_psr = (sr_hat - float(benchmark_sr)) / max(se_sr, 1e-12)
        psr = float(stats.norm.cdf(z_psr))

        k_trials = max(1, int(n_trials))
        if k_trials <= 1:
            sr_zero = float(benchmark_sr)
            dsr = psr
        else:
            sigma_trials = (
                float(trials_sr_std)
                if (trials_sr_std is not None and trials_sr_std > 0.0)
                else float(np.sqrt(1.0 / max(t_obs - 1, 1)))
            )
            # Bailey & López de Prado (2014, Eq. 8) expected maximum Sharpe under null across K trials
            q1 = float(stats.norm.ppf(1.0 - 1.0 / k_trials))
            q2 = float(stats.norm.ppf(1.0 - 1.0 / (k_trials * np.e)))
            sr_zero = float(benchmark_sr) + sigma_trials * ((1.0 - _EULER_MASCHERONI) * q1 + _EULER_MASCHERONI * q2)
            z_dsr = (sr_hat - sr_zero) / max(se_sr, 1e-12)
            dsr = float(stats.norm.cdf(z_dsr))

        return {
            "non_annualized_sharpe": round(sr_hat, 6),
            "sharpe_std_error": round(se_sr, 6),
            "expected_max_null_sharpe": round(sr_zero, 6),
            "probabilistic_sharpe_ratio": round(psr, 6),
            "deflated_sharpe_ratio": round(dsr, 6),
        }


def evaluate_anomaly_adbench(
    y_true: np.ndarray,
    anomaly_scores: np.ndarray,
    *,
    k: int = 50,
) -> dict[str, float]:
    """Evaluate anomaly detection using ADBench (NeurIPS 2022) standard: AUC-ROC, AUC-PR, and Precision@K."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    yt = np.asarray(y_true, dtype=int)
    sc = np.asarray(anomaly_scores, dtype=float)
    mask = np.isfinite(sc)
    yt = yt[mask]
    sc = sc[mask]
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return {"auc_roc": float("nan"), "auc_pr": float("nan"), "precision_at_k": float("nan"), "k": float(k)}
    k_eff = max(1, min(k, len(yt)))
    top_k_idx = np.argsort(sc)[::-1][:k_eff]
    p_at_k = float(np.mean(yt[top_k_idx] == 1))
    return {
        "auc_roc": round(float(roc_auc_score(yt, sc)), 6),
        "auc_pr": round(float(average_precision_score(yt, sc)), 6),
        "precision_at_k": round(p_at_k, 6),
        "k": float(k_eff),
    }


class ProbabilityCalibrator:
    """Post-hoc probability calibration via Temperature Scaling (Guo et al. 2017), Platt Scaling, or Isotonic Regression.

    Fit on held-out validation predictions. Probability columns follow the
    estimator's class order. For a vector or single column, values denote the
    probability of the second class. Temperature scaling preserves ordering
    within a probability row, up to clipping and numerical precision.
    """

    def __init__(self, method: str = "temperature") -> None:
        m = method.lower().strip()
        if m not in ("temperature", "platt", "isotonic"):
            raise EvaluationError(f"Unsupported calibration method '{method}'. Choose temperature, platt, or isotonic.")
        self.method = m
        self.temperature_: float = 1.0
        self._classes: np.ndarray = np.empty(0)
        self._is_binary_1d: bool = False
        self._platt: LogisticRegression | None = None
        self._isotonic_models: list[IsotonicRegression] = []
        self._is_fitted: bool = False

    @staticmethod
    def _to_logits(proba: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0 - 1e-12)
        if p.ndim == 1:
            log_odds = np.log(p / (1.0 - p))
            return np.column_stack([-0.5 * log_odds, 0.5 * log_odds])
        if p.ndim == 2 and p.shape[1] == 1:
            p1 = p.ravel()
            log_odds = np.log(p1 / (1.0 - p1))
            return np.column_stack([-0.5 * log_odds, 0.5 * log_odds])
        row_sums = np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
        p_norm = np.clip(p / row_sums, 1e-12, 1.0)
        logits = np.log(p_norm)
        return np.asarray(logits - np.mean(logits, axis=1, keepdims=True))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted)
        return np.asarray(exps / np.maximum(np.sum(exps, axis=1, keepdims=True), 1e-12), dtype=float)

    @staticmethod
    def _validate_probabilities(proba: np.ndarray) -> tuple[np.ndarray, int, bool]:
        try:
            values = np.asarray(proba, dtype=float)
        except (TypeError, ValueError) as exc:
            raise EvaluationError("Probabilities must be numeric.") from exc
        if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] == 0) or len(values) == 0:
            raise EvaluationError("Expected a non-empty probability vector or matrix.")
        binary_vector = values.ndim == 1 or values.shape[1] == 1
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise EvaluationError("Probabilities must be finite and in [0, 1].")
        if not binary_vector and not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0):
            raise EvaluationError("Each probability row must sum to one.")
        return values, 2 if binary_vector else values.shape[1], binary_vector

    @property
    def classes_(self) -> np.ndarray:
        """Class labels in the fitted probability-column order."""
        if not self._is_fitted:
            raise EvaluationError("ProbabilityCalibrator must be fitted before reading its class order.")
        return self._classes.copy()

    def fit(
        self, y_val: np.ndarray, proba_val: np.ndarray, *, classes: np.ndarray | list[Any] | None = None,
    ) -> ProbabilityCalibrator:
        """Fit on validation data, using explicit class order when supplied.

        Without `classes`, use the usual sorted estimator order only when
        validation labels cover every probability column. Temperature scaling
        can fit with an absent class if at least two classes are observed and
        the caller supplies the full class order. Platt and isotonic require
        observations for every class.
        """
        pv, n_cols, binary_vector = self._validate_probabilities(proba_val)
        yv = np.asarray(y_val)
        if yv.ndim != 1 or len(yv) != len(pv) or pd.isna(yv).any():
            raise EvaluationError("Supply one observed validation label per probability row.")
        if classes is None:
            try:
                class_order = np.unique(yv)
            except TypeError as exc:
                raise EvaluationError("Supply the estimator's probability-column class order explicitly.") from exc
        else:
            class_order = np.asarray(classes)
        if class_order.ndim != 1 or len(class_order) != n_cols:
            raise EvaluationError("Supply one class label per probability column in the estimator's order.")
        labels = pd.Index(class_order)
        if not labels.is_unique or labels.hasnans:
            raise EvaluationError("Probability-column class labels must be unique and observed.")
        y_idx = labels.get_indexer(yv)
        if (y_idx < 0).any():
            raise EvaluationError("Validation contains a label outside the estimator's class order.")
        n_observed = len(np.unique(y_idx))
        if n_observed < 2 or (self.method != "temperature" and n_observed != n_cols):
            raise EvaluationError(f"Insufficient validation class coverage for {self.method} calibration.")

        logits = self._to_logits(pv)
        temperature = 1.0
        platt = None
        isotonic_models = []
        if self.method == "temperature":
            def _nll(log_t: float) -> float:
                scaled = logits / np.exp(log_t)
                return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y_idx)), y_idx]))

            opt = optimize.minimize_scalar(_nll, bounds=(-3.0, 3.5), method="bounded")
            if not opt.success or not np.isfinite(opt.fun):
                raise EvaluationError("Temperature optimization did not converge to a finite solution.")
            temperature = float(np.exp(opt.x))
        elif self.method == "platt":
            platt = LogisticRegression(C=1e4, max_iter=500, random_state=42)
            feat = (logits[:, 1] - logits[:, 0]).reshape(-1, 1) if n_cols == 2 else logits
            platt.fit(feat, y_idx)
        else:
            probs_2d = self._softmax(logits)
            for k in range(n_cols):
                iso = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
                iso.fit(probs_2d[:, k], (y_idx == k).astype(float))
                isotonic_models.append(iso)

        self._classes = class_order.copy()
        self._is_binary_1d = binary_vector
        self.temperature_ = temperature
        self._platt = platt
        self._isotonic_models = isotonic_models
        self._is_fitted = True
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Transform probabilities in the class order and representation used during fit."""
        if not self._is_fitted:
            raise EvaluationError("ProbabilityCalibrator must be fitted before transform().")
        pv, n_cols, binary_vector = self._validate_probabilities(proba)
        if n_cols != len(self._classes) or binary_vector != self._is_binary_1d:
            raise EvaluationError("The probability schema differs from the fitted calibration input.")
        logits = self._to_logits(pv)

        if self.method == "temperature":
            cal_2d = self._softmax(logits / self.temperature_)
        elif self.method == "platt":
            assert self._platt is not None
            feat = (logits[:, 1] - logits[:, 0]).reshape(-1, 1) if n_cols == 2 else logits
            cal_2d = np.asarray(self._platt.predict_proba(feat), dtype=float)
        else:
            probs_2d = self._softmax(logits)
            stacked = np.column_stack([iso.transform(probs_2d[:, k]) for k, iso in enumerate(self._isotonic_models)])
            cal_2d = stacked / stacked.sum(axis=1, keepdims=True)

        return np.asarray(cal_2d[:, 1] if binary_vector else cal_2d, dtype=float)

    def fit_transform(
        self, y_val: np.ndarray, proba_val: np.ndarray, *, classes: np.ndarray | list[Any] | None = None,
    ) -> np.ndarray:
        """Fit on validation predictions and return calibrated validation probabilities."""
        return self.fit(y_val, proba_val, classes=classes).transform(proba_val)
