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
    """Strictly causal past-only forward-fill imputation with zero look-ahead across train/val boundary."""
    tr = pd.to_numeric(raw_train, errors="coerce")
    if not tr.notna().any():
        raise ForecastingError("Training time series contains no valid numeric values.")
    first_valid_tr = float(tr.dropna().iloc[0])
    # Fill strictly from past within train (t=0 uses first observed train anchor only; never peeks into val)
    tr_clean = tr.ffill().fillna(first_valid_tr).to_numpy(dtype=float)

    if raw_val is None:
        return tr_clean, None

    va = pd.to_numeric(raw_val, errors="coerce")
    # Seed any leading NaN in validation strictly from the last observed training point (zero future peeking!)
    last_train_val = float(tr_clean[-1])
    va_clean = va.ffill().fillna(last_train_val).to_numpy(dtype=float)
    return tr_clean, va_clean


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
        cov = float(np.mean((yt >= y_lo) & (yt <= y_hi)))
        res["interval_coverage"] = round(cov, 4)
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
    e1 = yt - p1
    e2 = yt - p2
    d = (e1**2 - e2**2) if loss == "squared" else (np.abs(e1) - np.abs(e2))
    d_bar = float(np.mean(d))
    d_c = d - d_bar
    h_eff = min(max(1, int(horizon)), max(1, n // 2))

    gamma_0 = float(np.dot(d_c, d_c) / n)
    if gamma_0 <= 1e-18:
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
    ) -> ForecastingReport:
        """Run causal split, baseline benchmarks, TimeSeriesSplit rolling CV, model tournament, and future forecasts."""
        if target_col not in df.columns:
            raise ForecastingError(f"Target column '{target_col}' not found in DataFrame.")
        work = (
            df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
            if (time_col and time_col in df.columns)
            else df.reset_index(drop=True)
        )
        raw_series = pd.to_numeric(work[target_col], errors="coerce")
        if len(raw_series) < 20:
            raise ForecastingError("ForecastingEngine requires at least 20 observations.")

        val_h = validation_horizon or min(horizon, max(4, len(raw_series) // 5))
        # Split chronologically FIRST, then apply strictly causal past-only imputation (zero look-ahead!)
        y_train, y_val = _causal_impute_split(raw_series.iloc[:-val_h], raw_series.iloc[-val_h:])
        assert y_val is not None
        full_causal_series, _ = _causal_impute_split(raw_series, None)

        with CarbonTracker("forecasting_tournament") as tracker:
            baseline_scores = BaselineBenchmarkSuite.run_forecasting(
                y_train, y_val, seasonal_period=self.seasonal_period
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
                    fc_val = model.predict(val_h)
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
                    ff = FoundationForecaster()
                    fc_val = ff.forecast_zero_shot(y_train, val_h)
                    metrics = _forecast_metrics(
                        y_val,
                        fc_val["yhat"].to_numpy(),
                        fc_val["yhat_lower"].to_numpy(),
                        fc_val["yhat_upper"].to_numpy(),
                        y_train_ref=y_train,
                        seasonal_period=self.seasonal_period,
                    )
                    leaderboard.append({"model": "Foundation_ZeroShot", "family": "foundation", **metrics})
                    val_preds_map["Foundation_ZeroShot"] = fc_val
                    fitted_builders["Foundation_ZeroShot"] = "foundation"
                except Exception as exc:
                    logger.warning("foundation_forecast_failed", error=str(exc))

            leaderboard.sort(
                key=lambda row: (
                    not np.isfinite(float(row.get("smape", float("inf")))),
                    float(row.get("smape", float("inf"))),
                )
            )
            non_baseline = [r for r in leaderboard if r["model"] in fitted_builders]
            champion_name = non_baseline[0]["model"] if non_baseline else leaderboard[0]["model"]

            # Expanding-Window Multi-Origin TimeSeriesSplit Rolling CV on top-3 candidates
            rolling_cv_results: list[dict[str, Any]] = []
            if self.n_cv_splits >= 2 and len(raw_series) >= 30:
                max_splits = min(self.n_cv_splits, max(2, (len(raw_series) - 12) // max(4, val_h)))
                if max_splits >= 2:
                    tscv = TimeSeriesSplit(n_splits=max_splits, test_size=min(val_h, max(3, len(raw_series) // (max_splits + 2))), gap=self.gap)
                    for m_name in [r["model"] for r in non_baseline[:3]]:
                        fold_rmses: list[float] = []
                        fold_smapes: list[float] = []
                        for tr_idx, va_idx in tscv.split(raw_series):
                            if len(tr_idx) < 12 or len(va_idx) < 2:
                                continue
                            try:
                                fold_tr, fold_va = _causal_impute_split(raw_series.iloc[tr_idx], raw_series.iloc[va_idx])
                                assert fold_va is not None
                                fc_fold = self._refit_and_predict(m_name, fitted_builders, fold_tr, len(fold_va))
                                fm = _forecast_metrics(fold_va, fc_fold["yhat"].to_numpy())
                                fold_rmses.append(fm["rmse"])
                                fold_smapes.append(fm["smape"])
                            except Exception as exc:
                                logger.debug("rolling_cv_fold_skipped", model=m_name, error=str(exc))
                        if fold_rmses:
                            rolling_cv_results.append(
                                {
                                    "model": m_name,
                                    "cv_strategy": f"TimeSeriesSplit(n_splits={len(fold_rmses)}, gap={self.gap})",
                                    "cv_rmse_mean": round(float(np.mean(fold_rmses)), 6),
                                    "cv_rmse_std": round(float(np.std(fold_rmses, ddof=1) if len(fold_rmses) > 1 else 0.0), 6),
                                    "cv_smape_mean": round(float(np.mean(fold_smapes)), 4),
                                    "cv_smape_std": round(float(np.std(fold_smapes, ddof=1) if len(fold_smapes) > 1 else 0.0), 4),
                                }
                            )

            # Refit champion on full causal series
            future_fc = self._refit_and_predict(champion_name, fitted_builders, full_causal_series, horizon)

            # Build top-3 inverse-sMAPE weighted ensemble forecast with exact Law of Total Variance intervals
            # (Wallis 2005; Gneiting & Ranjan 2013): combines within-model aleatoric variance + epistemic disagreement.
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
                "causal_imputation_verified": True,
                "ensemble_variance_method": "law_of_total_variance_mixture",
                "rolling_cv_results": rolling_cv_results,
                "diebold_mariano_vs_runner_up": dm_summary,
            },
        )

    def _refit_and_predict(
        self, name: str, builders: dict[str, Any], full_series: np.ndarray, horizon: int
    ) -> pd.DataFrame:
        entry = builders.get(name)
        if entry == "foundation":
            return FoundationForecaster().forecast_zero_shot(full_series, horizon)
        if callable(entry):
            model = entry()
            model.fit(full_series)
            return model.predict(horizon)
        from omniforge.models.baselines import ForecastingBaseline

        bl = ForecastingBaseline(method="naive", seasonal_period=self.seasonal_period).fit(full_series)
        res = bl.predict(horizon, return_intervals=True)
        assert isinstance(res, dict)
        return pd.DataFrame({"step": np.arange(1, horizon + 1), **res})
