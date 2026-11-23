"""Walk-Forward Time-Series (with Embargo Gap & Causal Imputation) & Trading Strategy Backtesting Engine."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from omniforge.core.exceptions import EvaluationError
from omniforge.evaluation.metrics import MetricsEvaluator


class BacktestingEngine:
    """Expanding and rolling-window walk-forward backtesting for forecasting and financial strategies."""

    def walk_forward_forecast(
        self,
        series: pd.Series | np.ndarray,
        forecaster_factory: Callable[[], Any],
        *,
        initial_train_size: int = 60,
        horizon: int = 7,
        step_size: int = 7,
        gap: int = 0,
        window_type: str = "expanding",
    ) -> dict[str, Any]:
        """Run multi-origin walk-forward backtesting with optional embargo ``gap`` and strictly causal imputation."""
        raw = pd.to_numeric(pd.Series(series), errors="coerce")
        gap_n = max(0, int(gap))
        if len(raw) < initial_train_size + gap_n + horizon:
            raise EvaluationError(
                f"Series length ({len(raw)}) insufficient for initial_train_size ({initial_train_size}) + gap ({gap_n}) + horizon ({horizon})."
            )

        folds: list[dict[str, Any]] = []
        all_actuals: list[float] = []
        all_preds: list[float] = []
        cursor = initial_train_size

        while cursor + gap_n + horizon <= len(raw):
            start_idx = 0 if window_type == "expanding" else max(0, cursor - initial_train_size)
            tr_raw = raw.iloc[start_idx:cursor]
            te_raw = raw.iloc[cursor + gap_n : cursor + gap_n + horizon]
            if not tr_raw.notna().any():
                cursor += max(1, step_size)
                continue
            first_valid = float(tr_raw.dropna().iloc[0])
            train_y = tr_raw.ffill().fillna(first_valid).to_numpy(dtype=float)
            # Keep NaN actuals as-is — do not forward-fill test targets. Missing actuals
            # must not be scored (otherwise rmse/mae/smape/mase would be falsely 0.0).
            test_y = te_raw.to_numpy(dtype=float)

            model = forecaster_factory()
            model.fit(train_y)
            fc = model.predict(horizon + gap_n)
            if isinstance(fc, pd.DataFrame):
                yhat = fc["yhat"].to_numpy(dtype=float)[gap_n : gap_n + horizon]
                lo = fc["yhat_lower"].to_numpy(dtype=float)[gap_n : gap_n + horizon] if "yhat_lower" in fc else None
                hi = fc["yhat_upper"].to_numpy(dtype=float)[gap_n : gap_n + horizon] if "yhat_upper" in fc else None
            else:
                yhat = np.asarray(fc, dtype=float).ravel()[gap_n : gap_n + horizon]
                lo, hi = None, None

            # Score only positions with valid (non-NaN) actuals.
            valid_mask = ~np.isnan(test_y)
            if valid_mask.any():
                test_y_valid = test_y[valid_mask]
                yhat_valid = yhat[valid_mask]
                lo_valid = lo[valid_mask] if lo is not None else None
                hi_valid = hi[valid_mask] if hi is not None else None
                m = MetricsEvaluator.forecasting_metrics(
                    test_y_valid, yhat_valid, y_train=train_y, y_lower=lo_valid, y_upper=hi_valid
                )
            else:
                m = {"rmse": float("nan"), "mae": float("nan"), "smape": float("nan"), "mase": float("nan")}
            folds.append({"cutoff_index": cursor, "train_size": len(train_y), "gap": gap_n, **m})
            all_actuals.extend(test_y[valid_mask].tolist())
            all_preds.extend(yhat[valid_mask].tolist())
            cursor += max(1, step_size)

        if not folds:
            raise EvaluationError("No valid walk-forward folds could be evaluated (all training windows were empty/NaN).")
        overall = MetricsEvaluator.forecasting_metrics(np.asarray(all_actuals), np.asarray(all_preds))
        fold_summary: dict[str, float] = {}
        for metric_key in ("rmse", "mae", "smape", "mase", "interval_coverage"):
            vals = [float(f[metric_key]) for f in folds if metric_key in f and np.isfinite(float(f[metric_key]))]
            if vals:
                fold_summary[f"{metric_key}_mean"] = round(float(np.mean(vals)), 6)
                fold_summary[f"{metric_key}_std"] = round(float(np.std(vals, ddof=1 if len(vals) > 1 else 0)), 6)

        return {
            "window_type": window_type,
            "horizon": horizon,
            "gap": gap_n,
            "n_origins": len(folds),
            "causal_imputation_verified": True,
            "overall_metrics": overall,
            "fold_summary": fold_summary,
            "fold_metrics": folds,
        }

    @staticmethod
    def backtest_trading_signals(
        prices: pd.Series | np.ndarray,
        signals: pd.Series | np.ndarray,
        *,
        transaction_cost_bps: float = 5.0,
        slippage_bps: float = 2.0,
        risk_free_rate_annual: float = 0.0,
        periods_per_year: int = 252,
        n_trials: int = 1,
    ) -> dict[str, Any]:
        """Backtest directional trading signals (-1, 0, +1) with realistic transaction costs, slippage, drawdown duration, DSR & Buy-and-Hold baseline."""
        p = np.asarray(prices, dtype=float).ravel()
        sig = np.asarray(signals, dtype=float).ravel()
        if len(p) != len(sig) or len(p) < 3:
            raise EvaluationError("prices and signals must have equal length >= 3.")
        asset_returns = np.diff(p) / np.maximum(p[:-1], 1e-12)
        pos = sig[:-1]  # Strictly lagged signal at t applied to return (t -> t+1): zero look-ahead!
        turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
        total_bps = (transaction_cost_bps + slippage_bps) / 10_000.0
        cost = turnover * total_bps
        strat_returns = pos * asset_returns - cost
        rf_per_period = risk_free_rate_annual / max(periods_per_year, 1)

        fin = MetricsEvaluator.financial_metrics(
            strat_returns, periods_per_year=periods_per_year, mar=rf_per_period
        )
        dsr_info = MetricsEvaluator.deflated_sharpe_ratio(
            strat_returns, benchmark_sr=0.0, n_trials=max(1, int(n_trials)), mar=rf_per_period
        )
        fin["deflated_sharpe_ratio"] = dsr_info["deflated_sharpe_ratio"]
        safe_strat = np.clip(strat_returns, -0.999999, 10.0)
        safe_asset = np.clip(asset_returns, -0.999999, 10.0)
        equity = np.concatenate(([1.0], np.exp(np.clip(np.cumsum(np.log1p(safe_strat)), -50.0, 50.0))))
        bh_equity = np.concatenate(([1.0], np.exp(np.clip(np.cumsum(np.log1p(safe_asset)), -50.0, 50.0))))
        running_peak = np.maximum.accumulate(equity)
        underwater = equity[1:] < running_peak[1:]
        max_dd_duration = 0
        cur_dur = 0
        for flag in underwater:
            cur_dur = (cur_dur + 1) if flag else 0
            max_dd_duration = max(max_dd_duration, cur_dur)

        active_mask = pos != 0
        hit_rate = float(np.mean(strat_returns[active_mask] > 0)) if np.any(active_mask) else 0.0
        gross_profit = float(np.sum(strat_returns[strat_returns > 0]))
        gross_loss = float(np.abs(np.sum(strat_returns[strat_returns < 0])))
        profit_factor = gross_profit / max(gross_loss, 1e-12)
        return {
            **fin,
            "cumulative_return": round(float(equity[-1] - 1.0), 6),
            "buy_and_hold_return": round(float(bh_equity[-1] - 1.0), 6),
            "excess_return_vs_buy_and_hold": round(float(equity[-1] - bh_equity[-1]), 6),
            "hit_rate": round(hit_rate, 4),
            "profit_factor": round(min(profit_factor, 999.0), 4),
            "max_drawdown_duration_periods": int(max_dd_duration),
            "avg_turnover": round(float(np.mean(turnover)), 4),
            "total_cost_drag": round(float(np.sum(cost)), 6),
            "n_periods": len(strat_returns),
        }
