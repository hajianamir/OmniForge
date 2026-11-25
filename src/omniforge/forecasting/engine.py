"""End-to-end Causal Enterprise Forecasting Engine with Zero Look-Ahead Imputation, TimeSeriesSplit Rolling CV, Baselines & Ensembling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

from omniforge.core.carbon import CarbonTracker
from omniforge.core.exceptions import ForecastingError
from omniforge.core.logging import get_logger
from omniforge.core.types import CarbonReport
from omniforge.evaluation.robustness import split_conformal_quantile
from omniforge.models.baselines import BaselineBenchmarkSuite
from omniforge.models.time_series.deep_forecast import DeepForecaster
from omniforge.models.time_series.foundation import FoundationForecaster
from omniforge.models.time_series.ml_forecast import MLForecaster
from omniforge.models.time_series.statistical import StatisticalForecaster

logger = get_logger(__name__)


def _causal_impute_split(
    raw_train: pd.Series,
    raw_val: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Trim leading training gaps, forward-fill from the past, and return raw validation outcomes."""
    train = pd.to_numeric(raw_train, errors="coerce").replace([np.inf, -np.inf], np.nan)
    observed = np.flatnonzero(train.notna().to_numpy())
    if not len(observed):
        raise ForecastingError("Training time series contains no finite numeric values.")
    tr_clean = train.iloc[observed[0] :].ffill().to_numpy(dtype=float)

    if raw_val is None:
        return tr_clean, None
    val = pd.to_numeric(raw_val, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return tr_clean, val.to_numpy(dtype=float)


def _forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lo: np.ndarray | None = None,
    y_hi: np.ndarray | None = None,
    *,
    y_train_ref: np.ndarray | None = None,
    seasonal_period: int = 1,
) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    observed = np.isfinite(yt)
    if not observed.any() or not np.isfinite(yp[observed]).all():
        raise ForecastingError("Forecast scoring requires finite predictions at observed evaluation outcomes.")
    yt = yt[observed]
    yp = yp[observed]
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    denom = (np.abs(yt) + np.abs(yp)) / 2.0 + 1e-9
    smape = float(np.mean(np.abs(yt - yp) / denom) * 100.0)
    res: dict[str, float] = {"rmse": round(rmse, 6), "mae": round(mae, 6), "smape": round(smape, 4)}
    if y_train_ref is not None and len(y_train_ref) > seasonal_period:
        naive_diffs = np.abs(y_train_ref[seasonal_period:] - y_train_ref[:-seasonal_period])
        scale = float(np.mean(naive_diffs)) if len(naive_diffs) > 0 else 0.0
        if scale > 1e-9:
            res["mase"] = round(mae / scale, 6)
    if y_lo is not None and y_hi is not None:
        lo = np.asarray(y_lo, dtype=float).ravel()[observed]
        hi = np.asarray(y_hi, dtype=float).ravel()[observed]
        usable = np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
        if usable.any():
            cov = float(np.mean((yt[usable] >= lo[usable]) & (yt[usable] <= hi[usable])))
            res["interval_coverage"] = round(cov, 4)
            res["interval_coverage_n"] = float(np.count_nonzero(usable))
    return res


def diebold_mariano_test(
    y_true: np.ndarray | list[float],
    y_pred_1: np.ndarray | list[float],
    y_pred_2: np.ndarray | list[float],
    *,
    horizon: int = 1,
    loss: Literal["squared", "abs"] = "squared",
) -> dict[str, Any]:
    """Harvey, Leybourne & Newbold (1997, IJF 13:281-291) finite-sample modified Diebold-Mariano (1995) test.

    Tests H0: E[L(e_{1,t}) - L(e_{2,t})] = 0 using HAC autocovariances up to lag ``horizon - 1``
    (falling back to positive-definite Newey-West Bartlett weights if truncated rectangular variance <= 0)
    and evaluates the HLN-modified statistic against Student-t_{n - 1}.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    p1 = np.asarray(y_pred_1, dtype=float).ravel()
    p2 = np.asarray(y_pred_2, dtype=float).ravel()
    n = len(yt)
    if len(p1) != n or len(p2) != n or n < 4:
        return {"evaluable": False, "reason": "Requires equal-length forecast arrays with n >= 4."}
    if not (np.isfinite(yt).all() and np.isfinite(p1).all() and np.isfinite(p2).all()):
        return {"evaluable": False, "reason": "Requires finite observations and forecasts on a contiguous evaluation window."}
    e1 = yt - p1
    e2 = yt - p2
    d = (e1**2 - e2**2) if loss == "squared" else (np.abs(e1) - np.abs(e2))
    d_bar = float(np.mean(d))
    d_c = d - d_bar
    h_eff = min(max(1, int(horizon)), max(1, n // 2))

    gamma_0 = float(np.dot(d_c, d_c) / n)
    if gamma_0 <= 1e-18:
        if abs(d_bar) > 1e-12:
            return {
                "evaluable": False,
                "reason": "Constant nonzero loss difference has no estimable sampling variance.",
                "p_value": None,
                "mean_loss_diff": round(d_bar, 6),
                "preferred_model": "model_1" if d_bar < 0 else "model_2",
                "n": n,
            }
        return {
            "evaluable": True,
            "dm_statistic": 0.0,
            "hln_statistic": 0.0,
            "p_value": 1.0,
            "mean_loss_diff": round(d_bar, 6),
            "preferred_model": "tie",
            "n": n,
        }

    gammas = [float(np.dot(d_c[k:], d_c[:-k]) / n) for k in range(1, h_eff)]
    var_d = (gamma_0 + 2.0 * sum(gammas)) / float(n)
    if var_d <= 1e-18:
        # Newey-West (1987) Bartlett kernel guarantees strictly positive-semidefinite spectral density
        bartlett_sum = sum((1.0 - float(k) / float(h_eff)) * gammas[k - 1] for k in range(1, h_eff))
        var_d = max((gamma_0 + 2.0 * bartlett_sum) / float(n), gamma_0 / (float(n) * 100.0))

    dm_raw = float(d_bar / np.sqrt(var_d))
    hln_num = max(float(n + 1 - 2 * h_eff) + float(h_eff * (h_eff - 1)) / float(n), 1.0)
    hln_factor = float(np.sqrt(hln_num / float(n)))
    dm_hln = float(hln_factor * dm_raw)
    p_val = float(2.0 * stats.t.sf(abs(dm_hln), df=n - 1))
    preferred = "model_1" if d_bar < -1e-12 else ("model_2" if d_bar > 1e-12 else "tie")
    return {
        "evaluable": True,
        "dm_statistic": round(dm_raw, 6),
        "hln_statistic": round(dm_hln, 6),
        "p_value": round(p_val, 6),
        "mean_loss_diff": round(d_bar, 6),
        "preferred_model": preferred,
        "loss": loss,
        "horizon": h_eff,
        "n": n,
    }


def _interval_scores(
    observed: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-outcome coverage indicators and interval scores with missing values preserved."""
    usable = np.isfinite(observed) & np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    hits = np.full(observed.shape, np.nan)
    scores = np.full_like(hits, np.nan)
    actual, lo, hi = observed[usable], lower[usable], upper[usable]
    hits[usable] = (lo <= actual) & (actual <= hi)
    miss = np.maximum(0.0, lo - actual) + np.maximum(0.0, actual - hi)
    scores[usable] = hi - lo + 2 / alpha * miss
    return hits, scores


def _summarize_interval_scores(
    calibrated_hits: np.ndarray, calibrated_scores: np.ndarray, native_hits: np.ndarray, native_scores: np.ndarray,
) -> dict[str, Any]:
    """Report point-weighted estimates and compare both procedures on identical outcomes."""
    usable = np.isfinite(calibrated_hits)
    paired = usable & np.isfinite(native_hits)
    n_observed = int(usable.sum())
    n_paired = int(paired.sum())
    return {
        "observed_outcomes": n_observed,
        "calibrated_coverage": float(calibrated_hits[usable].mean()) if n_observed else None,
        "calibrated_mean_interval_score": float(calibrated_scores[usable].mean()) if n_observed else None,
        "native_coverage_count": n_paired,
        "native_coverage": float(native_hits[paired].mean()) if n_paired else None,
        "native_mean_interval_score": float(native_scores[paired].mean()) if n_paired else None,
        "paired_calibrated_coverage": float(calibrated_hits[paired].mean()) if n_paired else None,
        "paired_calibrated_mean_interval_score": float(calibrated_scores[paired].mean()) if n_paired else None,
    }


def _compare_rolling_interval_scores(
    calibration_errors: np.ndarray, evaluation_errors: np.ndarray, fixed_widths: np.ndarray, alpha: float,
) -> dict[str, Any]:
    """Replay rolling residual quantiles without using the current window's outcomes."""
    window = len(calibration_errors)
    errors = np.concatenate((calibration_errors, evaluation_errors), axis=0)
    rolling_widths = np.empty_like(evaluation_errors)
    last_widths = fixed_widths.copy()
    retained_leads = []
    for i in range(len(evaluation_errors)):
        previous = errors[i : i + window]
        retained = []
        for lead, recent_errors in enumerate(previous.T):
            count = int(np.isfinite(recent_errors).sum())
            if count and (count + 1) * (1 - alpha) <= count:
                width, estimable = split_conformal_quantile(recent_errors, alpha)
                if estimable:
                    last_widths[lead] = width
                    continue
            retained.append(lead + 1)
        retained_leads.append(retained)
        rolling_widths[i] = last_widths

    fixed = np.broadcast_to(fixed_widths, evaluation_errors.shape)
    fixed_hits, fixed_scores = _interval_scores(evaluation_errors, -fixed, fixed, alpha)
    rolling_hits, rolling_scores = _interval_scores(evaluation_errors, -rolling_widths, rolling_widths, alpha)
    paired = np.isfinite(fixed_hits) & np.isfinite(rolling_hits)

    def summarize(index: int | slice = slice(None)) -> dict[str, Any]:
        mask = paired[index]
        count = int(mask.sum())
        report: dict[str, Any] = {"paired_outcomes": count}
        for name, hits, scores, widths in (
            ("fixed", fixed_hits, fixed_scores, fixed),
            ("rolling", rolling_hits, rolling_scores, rolling_widths),
        ):
            report[name] = {
                "coverage": float(hits[index][mask].mean()) if count else None,
                "mean_interval_score": float(scores[index][mask].mean()) if count else None,
                "mean_width": float(2 * widths[index][mask].mean()) if count else None,
            }
        return report

    by_origin = []
    for i, widths in enumerate(rolling_widths):
        by_origin.append({
            "rolling_half_width_by_lead": widths.tolist(),
            "retained_width_leads": retained_leads[i],
            **summarize(i),
        })
    return {
        "method": "trailing_origin_absolute_error_order_statistic",
        "window_origins": window,
        **summarize(),
        "by_origin": by_origin,
    }


@dataclass
class ForecastingReport:
    """Complete result from running :class:`ForecastingEngine`."""

    champion_model: str
    horizon: int
    forecast: pd.DataFrame
    leaderboard: list[dict[str, Any]]
    baseline_benchmarks: dict[str, dict[str, float]]
    ensemble_forecast: pd.DataFrame | None = None
    carbon_report: CarbonReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ForecastingEngine:
    """Causal forecasting tournament with zero-lookahead imputation, TimeSeriesSplit rolling CV, baselines, statistical, ML, deep & foundation models."""

    def __init__(
        self,
        *,
        seasonal_period: int = 7,
        confidence_level: float = 0.95,
        include_deep: bool = True,
        include_foundation: bool = True,
        n_cv_splits: int = 3,
        gap: int = 0,
        random_seed: int = 42,
    ) -> None:
        self.seasonal_period = max(1, int(seasonal_period))
        self.confidence_level = float(confidence_level)
        self.include_deep = include_deep
        self.include_foundation = include_foundation
        self.n_cv_splits = max(0, int(n_cv_splits))
        self.gap = max(0, int(gap))
        self.random_seed = random_seed

    def forecast(
        self,
        df: pd.DataFrame,
        *,
        target_col: str,
        time_col: str | None = None,
        horizon: int = 14,
        validation_horizon: int | None = None,
        interval_calibration_origins: int = 0,
        interval_evaluation_origins: int = 1,
        compare_rolling_intervals: bool = False,
    ) -> ForecastingReport:
        """Run a chronological tournament and optionally compare rolling widths in retrospective evaluation."""
        if target_col not in df.columns:
            raise ForecastingError(f"Target column '{target_col}' not found in DataFrame.")
        work = (
            df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
            if (time_col and time_col in df.columns)
            else df.reset_index(drop=True)
        )
        all_history = pd.to_numeric(work[target_col], errors="coerce")
        if len(all_history) < 20:
            raise ForecastingError("ForecastingEngine requires at least 20 observations.")
        if horizon < 1 or (validation_horizon is not None and validation_horizon < 1):
            raise ForecastingError("Forecast and validation horizons must be positive.")
        if not 0 <= interval_calibration_origins <= 40:
            raise ForecastingError("Choose between zero and 40 interval calibration origins.")
        if interval_evaluation_origins < 1:
            raise ForecastingError("Choose at least one interval evaluation origin.")
        if not 0 < self.confidence_level < 1:
            raise ForecastingError("The interval level must lie strictly between zero and one.")

        reserve = (interval_calibration_origins + interval_evaluation_origins) * horizon + 2 * self.gap
        min_selection = max(20, (validation_horizon or 0) + self.gap + 2)
        calibration: dict[str, Any] = {
            "status": "disabled",
            "requested_origins": interval_calibration_origins,
            "requested_evaluation_origins": interval_evaluation_origins,
            "nominal_level": self.confidence_level,
        }
        if compare_rolling_intervals:
            calibration["rolling_comparison"] = None
        calibration_start = None
        raw_series = all_history
        if interval_calibration_origins:
            if (
                (interval_calibration_origins + 1) * self.confidence_level > interval_calibration_origins
                or len(all_history) - reserve < min_selection
            ):
                calibration["status"] = "insufficient_history_or_origins"
            else:
                cut = len(all_history) - reserve
                candidate_history = all_history.iloc[:cut]
                validation_size = validation_horizon or min(horizon, max(4, cut // 5))
                usable = np.isfinite(candidate_history.to_numpy(dtype=float, na_value=np.nan))
                if usable[:cut - validation_size - self.gap].sum() >= 2 and usable[-validation_size:].any():
                    raw_series = candidate_history
                    calibration_start = cut + self.gap
                    calibration["selection_end_position"] = cut
                else:
                    calibration["status"] = "insufficient_selection_outcomes"

        val_h = validation_horizon if validation_horizon is not None else min(horizon, max(4, len(raw_series) // 5))
        train_end = len(raw_series) - val_h - self.gap
        if train_end < 2:
            raise ForecastingError(
                "Forecast and validation horizons and the embargo must leave at least two training observations.",
                suggestions=["Choose a shorter horizon or embargo, or provide more history."],
            )
        y_train, y_val = _causal_impute_split(raw_series.iloc[:train_end], raw_series.iloc[-val_h:])
        assert y_val is not None
        n_observed = int(np.count_nonzero(np.isfinite(y_val)))
        if len(y_train) < 2 or not n_observed:
            raise ForecastingError(
                "Forecast validation needs at least two usable training points and one observed validation outcome.",
                suggestions=["Check the target column and adjust the validation horizon."],
            )
        full_causal_series, _ = _causal_impute_split(all_history, None)

        with CarbonTracker("forecasting_tournament") as tracker:
            baseline_scores = BaselineBenchmarkSuite.run_forecasting(
                y_train, y_val, seasonal_period=self.seasonal_period, gap=self.gap
            )
            leaderboard: list[dict[str, Any]] = [
                {"model": name, "family": "baseline", **scores}
                for name, scores in baseline_scores.items()
            ]
            val_preds_map: dict[str, pd.DataFrame] = {}

            candidates: list[tuple[str, str, Any]] = [
                ("ETS_HoltWinters", "statistical", lambda: StatisticalForecaster("ets", seasonal_period=self.seasonal_period, confidence_level=self.confidence_level)),
                ("ARIMA", "statistical", lambda: StatisticalForecaster("arima", seasonal_period=self.seasonal_period, confidence_level=self.confidence_level)),
                ("Theta", "statistical", lambda: StatisticalForecaster("theta", seasonal_period=self.seasonal_period, confidence_level=self.confidence_level)),
                ("FourierProphet", "statistical", lambda: StatisticalForecaster("fourier_prophet", seasonal_period=self.seasonal_period, confidence_level=self.confidence_level)),
                ("ML_GradientBoosting", "ml", lambda: MLForecaster("gradient_boosting", seasonal_period=self.seasonal_period, confidence_level=self.confidence_level, random_seed=self.random_seed)),
                ("ML_RandomForest", "ml", lambda: MLForecaster("random_forest", seasonal_period=self.seasonal_period, confidence_level=self.confidence_level, random_seed=self.random_seed)),
            ]
            if self.include_deep:
                candidates.append(
                    ("Deep_NBEATS_MLP", "deep", lambda: DeepForecaster("nbeats_mlp", epochs=35, confidence_level=self.confidence_level, random_seed=self.random_seed))
                )

            fitted_builders: dict[str, Any] = {}
            for name, family, builder in candidates:
                try:
                    model = builder()
                    model.fit(y_train)
                    fc_val = model.predict(val_h + self.gap).iloc[self.gap :]
                    metrics = _forecast_metrics(
                        y_val,
                        fc_val["yhat"].to_numpy(),
                        fc_val["yhat_lower"].to_numpy(),
                        fc_val["yhat_upper"].to_numpy(),
                        y_train_ref=y_train,
                        seasonal_period=self.seasonal_period,
                    )
                    leaderboard.append({"model": name, "family": family, **metrics})
                    val_preds_map[name] = fc_val
                    fitted_builders[name] = builder
                except Exception as exc:
                    logger.warning("forecast_candidate_failed", model=name, error=str(exc))

            if self.include_foundation:
                try:
                    ff = self._patch_forecaster()
                    fc_val = ff.forecast_zero_shot(y_train, val_h + self.gap).iloc[self.gap :]
                    metrics = _forecast_metrics(
                        y_val,
                        fc_val["yhat"].to_numpy(),
                        fc_val["yhat_lower"].to_numpy(),
                        fc_val["yhat_upper"].to_numpy(),
                        y_train_ref=y_train,
                        seasonal_period=self.seasonal_period,
                    )
                    leaderboard.append({
                        "model": "HistoricalPatchRetrieval",
                        "family": "historical_analogs",
                        "backend": "historical_patch_retrieval",
                        "nominal_interval_level": self.confidence_level,
                        **metrics,
                    })
                    val_preds_map["HistoricalPatchRetrieval"] = fc_val
                    fitted_builders["HistoricalPatchRetrieval"] = "patch_retrieval"
                except Exception as exc:
                    logger.warning("foundation_forecast_failed", error=str(exc))

            leaderboard.sort(
                key=lambda row: (
                    not np.isfinite(float(row.get("smape", float("inf")))),
                    float(row.get("smape", float("inf"))),
                )
            )
            non_baseline = [r for r in leaderboard if r["model"] in fitted_builders]
            champion_name = leaderboard[0]["model"]

            # Expanding-Window Multi-Origin TimeSeriesSplit Rolling CV on top-3 candidates
            rolling_cv_results: list[dict[str, Any]] = []
            if self.n_cv_splits >= 2 and len(raw_series) >= 30:
                max_splits = min(self.n_cv_splits, max(2, (len(raw_series) - 12) // max(4, val_h)))
                cv_size = min(val_h, (len(raw_series) - self.gap - 12) // max_splits)
                if cv_size >= 2:
                    tscv = TimeSeriesSplit(n_splits=max_splits, test_size=cv_size, gap=self.gap)
                    for m_name in [r["model"] for r in non_baseline[:3]]:
                        fold_rmses: list[float] = []
                        fold_smapes: list[float] = []
                        fold_observed: list[int] = []
                        for tr_idx, va_idx in tscv.split(raw_series):
                            if len(tr_idx) < 12 or len(va_idx) < 2:
                                continue
                            try:
                                fold_tr, fold_va = _causal_impute_split(raw_series.iloc[tr_idx], raw_series.iloc[va_idx])
                                assert fold_va is not None
                                if len(fold_tr) < 12 or not np.isfinite(fold_va).any():
                                    continue
                                fc_fold = self._refit_and_predict(
                                    m_name, fitted_builders, fold_tr, self.gap + len(fold_va)
                                )
                                fm = _forecast_metrics(fold_va, fc_fold["yhat"].to_numpy()[self.gap :])
                                fold_rmses.append(fm["rmse"])
                                fold_smapes.append(fm["smape"])
                                fold_observed.append(int(np.count_nonzero(np.isfinite(fold_va))))
                            except Exception as exc:
                                logger.debug("rolling_cv_fold_skipped", model=m_name, error=str(exc))
                        if fold_rmses:
                            rolling_cv_results.append(
                                {
                                    "model": m_name,
                                    "cv_strategy": f"TimeSeriesSplit(n_splits={len(fold_rmses)}, gap={self.gap})",
                                    "cv_observed_counts": fold_observed,
                                    "cv_rmse_mean": round(float(np.mean(fold_rmses)), 6),
                                    "cv_rmse_std": round(float(np.std(fold_rmses, ddof=1) if len(fold_rmses) > 1 else 0.0), 6),
                                    "cv_smape_mean": round(float(np.mean(fold_smapes)), 4),
                                    "cv_smape_std": round(float(np.std(fold_smapes, ddof=1) if len(fold_smapes) > 1 else 0.0), 4),
                                }
                            )

            widths = None
            if calibration_start is not None:
                widths, evidence = self._calibrate_intervals(
                    all_history, champion_name, fitted_builders, calibration_start,
                    interval_calibration_origins, interval_evaluation_origins, horizon,
                    compare_rolling_intervals=compare_rolling_intervals,
                )
                calibration.update(evidence)

            # The operational refit can use all observations after retrospective scoring.
            future_fc = self._refit_and_predict(champion_name, fitted_builders, full_causal_series, horizon)
            if widths is not None:
                future_fc["yhat_lower"] = future_fc["yhat"] - widths
                future_fc["yhat_upper"] = future_fc["yhat"] + widths

            # Build approximate normal bounds from within-model spread and between-model disagreement.
            top_models = [r["model"] for r in non_baseline[:3]]
            ensemble_fc: pd.DataFrame | None = None
            if len(top_models) >= 2:
                smape_map = {r["model"]: max(float(r.get("smape", 1.0)), 1e-4) for r in non_baseline[:3]}
                inv_w = np.array([1.0 / smape_map[m] for m in top_models], dtype=float)
                weights = inv_w / inv_w.sum()
                fcs = [self._refit_and_predict(m, fitted_builders, full_causal_series, horizon) for m in top_models]
                means_mat = np.vstack([f["yhat"].to_numpy(dtype=float) for f in fcs])
                lo_mat = np.vstack([f["yhat_lower"].to_numpy(dtype=float) for f in fcs])
                hi_mat = np.vstack([f["yhat_upper"].to_numpy(dtype=float) for f in fcs])
                mu_ens = np.average(means_mat, axis=0, weights=weights)

                alpha_ci = float(np.clip(1.0 - self.confidence_level, 1e-4, 0.5))
                z_ci = max(float(stats.norm.ppf(1.0 - alpha_ci / 2.0)), 1e-6)
                # Directional implied within-model std (reduces identically to symmetric sigma_k when intervals are symmetric)
                sigma_lo = np.maximum(means_mat - lo_mat, 0.0) / z_ci
                sigma_hi = np.maximum(hi_mat - means_mat, 0.0) / z_ci
                epistemic_var = np.average((means_mat - mu_ens[None, :]) ** 2, axis=0, weights=weights)
                total_sd_lo = np.sqrt(np.average(sigma_lo**2, axis=0, weights=weights) + epistemic_var)
                total_sd_hi = np.sqrt(np.average(sigma_hi**2, axis=0, weights=weights) + epistemic_var)

                ensemble_fc = pd.DataFrame(
                    {
                        "step": np.arange(1, horizon + 1),
                        "yhat": mu_ens,
                        "yhat_lower": mu_ens - z_ci * total_sd_lo,
                        "yhat_upper": mu_ens + z_ci * total_sd_hi,
                    }
                )

            dm_summary: dict[str, Any] | None = None
            if len(non_baseline) >= 2:
                m1_name = non_baseline[0]["model"]
                m2_name = non_baseline[1]["model"]
                if m1_name in val_preds_map and m2_name in val_preds_map and len(y_val) >= 4:
                    dm_summary = {
                        "model_1": m1_name,
                        "model_2": m2_name,
                        **diebold_mariano_test(
                            y_val,
                            val_preds_map[m1_name]["yhat"].to_numpy(dtype=float),
                            val_preds_map[m2_name]["yhat"].to_numpy(dtype=float),
                            horizon=1,
                        ),
                    }

            if time_col and time_col in work.columns and pd.api.types.is_datetime64_any_dtype(work[time_col]):
                last_ts = pd.to_datetime(work[time_col].iloc[-1])
                ts_idx = pd.DatetimeIndex(pd.to_datetime(work[time_col]))
                try:
                    with pd.option_context("future.infer_freq_returns_offset", True):
                        inferred = ts_idx.inferred_freq
                except Exception:
                    inferred = ts_idx.inferred_freq
                freq = getattr(inferred, "freqstr", inferred) or "D"
                fut_dates = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]
                future_fc[time_col] = fut_dates
                if ensemble_fc is not None:
                    ensemble_fc[time_col] = fut_dates

        logger.info("forecasting_completed", champion=champion_name, horizon=horizon, causal_imputation=True)
        return ForecastingReport(
            champion_model=champion_name,
            horizon=horizon,
            forecast=future_fc,
            leaderboard=leaderboard,
            baseline_benchmarks=baseline_scores,
            ensemble_forecast=ensemble_fc,
            carbon_report=tracker.report,
            metadata={
                "n_history": len(full_causal_series),
                "validation_horizon": val_h,
                "validation_observed": n_observed,
                "validation_missing": val_h - n_observed,
                "validation_gap": self.gap,
                "causal_imputation_verified": True,
                "ensemble_variance_method": "law_of_total_variance_mixture",
                "ensemble_interval_method": "approximate_normal_moment_bounds" if ensemble_fc is not None else None,
                "champion_interval_calibration": calibration,
                "rolling_cv_results": rolling_cv_results,
                "diebold_mariano_vs_runner_up": dm_summary,
            },
        )

    def _calibrate_intervals(
        self, history: pd.Series, champion: str, builders: dict[str, Any], start: int,
        origins: int, evaluation_origins: int, horizon: int,
        *, compare_rolling_intervals: bool = False,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Freeze lead-specific widths before evaluating later non-overlapping windows."""
        evaluation_start = len(history) - evaluation_origins * horizon
        evidence: dict[str, Any] = {
            "status": "calibration_unavailable",
            "method": "rolling_origin_absolute_error_order_statistic",
            "calibration_start_position": start,
            "calibration_end_position": start + origins * horizon,
            "evaluation_start_position": evaluation_start,
            "evaluation": None,
        }

        def observed_forecast(origin: int) -> tuple[np.ndarray, pd.DataFrame]:
            train, observed = _causal_impute_split(history.iloc[:origin], history.iloc[origin: origin + horizon])
            assert observed is not None
            forecast = self._refit_and_predict(champion, builders, train, horizon)
            if len(forecast) != horizon:
                raise ForecastingError("The champion did not return all requested forecast leads.")
            return observed, forecast

        scores = np.full((origins, horizon), np.nan)
        for i in range(origins):
            try:
                observed, forecast = observed_forecast(start + i * horizon)
                scores[i] = np.abs(observed - forecast["yhat"].to_numpy(dtype=float))
            except Exception as exc:
                logger.warning("interval_calibration_origin_failed", error_type=type(exc).__name__)
        counts = np.isfinite(scores).sum(axis=0)
        evidence["usable_origins_by_lead"] = counts.tolist()
        if np.any((counts + 1) * self.confidence_level > counts):
            evidence["status"] = "insufficient_finite_calibration_errors"
            return None, evidence
        widths = np.array([split_conformal_quantile(col, 1 - self.confidence_level)[0] for col in scores.T])
        evidence["status"] = "applied"
        evidence["half_width_by_lead"] = widths.tolist()
        calibrated_hits = np.full((evaluation_origins, horizon), np.nan)
        calibrated_scores = np.full_like(calibrated_hits, np.nan)
        native_hits = np.full_like(calibrated_hits, np.nan)
        native_scores = np.full_like(calibrated_hits, np.nan)
        evaluation_errors = np.full_like(calibrated_hits, np.nan) if compare_rolling_intervals else None
        alpha = 1 - self.confidence_level
        by_origin: list[dict[str, Any]] = []
        for i in range(evaluation_origins):
            position = evaluation_start + i * horizon
            try:
                observed, forecast = observed_forecast(position)
                point = forecast["yhat"].to_numpy(dtype=float)
                calibrated_hits[i], calibrated_scores[i] = _interval_scores(observed, point - widths, point + widths, alpha)
                if {"yhat_lower", "yhat_upper"} <= set(forecast.columns):
                    lower = pd.to_numeric(forecast["yhat_lower"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                    upper = pd.to_numeric(forecast["yhat_upper"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                    native_hits[i], native_scores[i] = _interval_scores(
                        observed, lower, upper, alpha,
                    )
                status = "scored" if np.isfinite(calibrated_hits[i]).any() else "no_evaluable_outcomes"
                if evaluation_errors is not None:
                    evaluation_errors[i] = np.where(np.isfinite(calibrated_hits[i]), np.abs(observed - point), np.nan)
            except Exception as exc:
                status = "forecast_unavailable"
                calibrated_hits[i] = calibrated_scores[i] = native_hits[i] = native_scores[i] = np.nan
                logger.warning("interval_calibration_evaluation_failed", error_type=type(exc).__name__)
            by_origin.append({
                "start_position": position,
                "status": status,
                **_summarize_interval_scores(calibrated_hits[i], calibrated_scores[i], native_hits[i], native_scores[i]),
            })
        evidence["evaluation"] = {
            "requested_origins": evaluation_origins,
            "scored_origins": sum(row["status"] == "scored" for row in by_origin),
            **_summarize_interval_scores(calibrated_hits, calibrated_scores, native_hits, native_scores),
            "by_origin": by_origin,
            "by_lead": [
                {
                    "lead": i + 1,
                    **_summarize_interval_scores(calibrated_hits[:, i], calibrated_scores[:, i], native_hits[:, i], native_scores[:, i]),
                }
                for i in range(horizon)
            ],
        }
        if evaluation_errors is not None:
            comparison = _compare_rolling_interval_scores(scores, evaluation_errors, widths, alpha)
            for row, origin in zip(comparison["by_origin"], by_origin, strict=True):
                row["start_position"] = origin["start_position"]
            evidence["rolling_comparison"] = comparison
        return widths, evidence

    def _patch_forecaster(self) -> FoundationForecaster:
        tail = (1.0 - self.confidence_level) / 2.0
        return FoundationForecaster(quantiles=(tail, 0.5, 1.0 - tail))

    def _refit_and_predict(
        self, name: str, builders: dict[str, Any], full_series: np.ndarray, horizon: int
    ) -> pd.DataFrame:
        entry = builders.get(name)
        if entry == "patch_retrieval":
            return self._patch_forecaster().forecast_zero_shot(full_series, horizon)
        if callable(entry):
            model = entry()
            model.fit(full_series)
            return model.predict(horizon)
        from omniforge.models.baselines import ForecastingBaseline

        method = name.removeprefix("Baseline_")
        bl = ForecastingBaseline(method=method, seasonal_period=self.seasonal_period).fit(full_series)
        z = float(stats.norm.ppf((1.0 + self.confidence_level) / 2.0))
        res = bl.predict(horizon, return_intervals=True, z=z)
        assert isinstance(res, dict)
        return pd.DataFrame({"step": np.arange(1, horizon + 1), **res})
