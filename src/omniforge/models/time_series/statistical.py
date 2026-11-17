"""Statistical time-series forecasting: ETS/Holt-Winters, ARIMA/SARIMAX, Theta, Croston, Fourier-Prophet."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from omniforge.core.exceptions import ForecastingError
from omniforge.core.logging import get_logger
from omniforge.core.utils import is_package_available

logger = get_logger(__name__)


class StatisticalForecaster:
    """Unified probabilistic statistical forecaster covering ETS, ARIMA, Theta, Croston, and Fourier-Trend."""

    def __init__(
        self,
        method: str = "ets",
        *,
        seasonal_period: int = 7,
        arima_order: tuple[int, int, int] = (1, 1, 1),
        fourier_order: int = 3,
        confidence_level: float = 0.95,
    ) -> None:
        self.method = method.lower()
        self.seasonal_period = max(1, int(seasonal_period))
        self.arima_order = arima_order
        self.fourier_order = int(fourier_order)
        self.confidence_level = float(confidence_level)
        self._history: np.ndarray = np.array([], dtype=float)
        self._fitted_model: Any = None
        self._residual_std: float = 1.0
        self._theta_params: dict[str, float] = {}
        self._croston_rate: float = 0.0

    def fit(self, y: pd.Series | np.ndarray) -> StatisticalForecaster:
        """Fit statistical forecasting model to a 1D target series (strictly causal imputation)."""
        s_y = pd.to_numeric(pd.Series(y), errors="coerce")
        if not s_y.notna().any():
            raise ForecastingError("StatisticalForecaster target series contains no valid numeric values.")
        arr = s_y.ffill().fillna(float(s_y.dropna().iloc[0])).to_numpy(dtype=float).ravel()
        if len(arr) < 4:
            raise ForecastingError("StatisticalForecaster requires at least 4 historical observations.")
        self._history = arr

        if self.method == "prophet" and is_package_available("prophet"):
            from prophet import Prophet  # type: ignore[import-not-found]

            df_p = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=len(arr), freq="D"), "y": arr})
            m = Prophet(interval_width=self.confidence_level)
            m.fit(df_p)
            self._fitted_model = m
            return self

        if self.method == "ets":
            self._fit_ets(arr)
        elif self.method in ("arima", "sarimax"):
            self._fit_arima(arr)
        elif self.method == "theta":
            self._fit_theta(arr)
        elif self.method == "croston":
            self._fit_croston(arr)
        else:
            # Fourier-Prophet style additive trend + multi-harmonic seasonality
            self._fit_fourier_trend(arr)

        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """Forecast ``horizon`` steps ahead with point estimate and calibrated prediction intervals."""
        if len(self._history) == 0:
            raise ForecastingError("Call fit() before predict().")
        h = max(1, int(horizon))
        z = self._z_score(self.confidence_level)
        steps = np.sqrt(np.arange(1, h + 1, dtype=float))

        if self.method == "prophet" and self._fitted_model is not None and is_package_available("prophet"):
            future = self._fitted_model.make_future_dataframe(periods=h)
            fc = self._fitted_model.predict(future).tail(h)
            return pd.DataFrame(
                {
                    "step": np.arange(1, h + 1),
                    "yhat": fc["yhat"].to_numpy(dtype=float),
                    "yhat_lower": fc["yhat_lower"].to_numpy(dtype=float),
                    "yhat_upper": fc["yhat_upper"].to_numpy(dtype=float),
                }
            )

        if self.method == "ets" and self._fitted_model is not None:
            yhat = np.asarray(self._fitted_model.forecast(h), dtype=float)
        elif self.method in ("arima", "sarimax") and self._fitted_model is not None:
            fc_res = self._fitted_model.get_forecast(steps=h)
            yhat = np.asarray(fc_res.predicted_mean, dtype=float)
            ci = np.asarray(fc_res.conf_int(alpha=1.0 - self.confidence_level), dtype=float)
            return pd.DataFrame(
                {"step": np.arange(1, h + 1), "yhat": yhat, "yhat_lower": ci[:, 0], "yhat_upper": ci[:, 1]}
            )
        elif self.method == "theta":
            n = len(self._history)
            t_future = np.arange(n, n + h, dtype=float)
            drift = self._theta_params["slope"] * 0.5
            level = self._theta_params["ses_level"]
            yhat = level + drift * (t_future - (n - 1) + (1.0 / max(self._theta_params["alpha"], 0.05) - 1.0))
        elif self.method == "croston":
            yhat = np.full(h, self._croston_rate, dtype=float)
        else:
            n = len(self._history)
            X_fut = self._build_fourier_matrix(np.arange(n, n + h, dtype=float))
            yhat = np.asarray(self._fitted_model.predict(X_fut), dtype=float)

        half_w = z * self._residual_std * steps
        return pd.DataFrame(
            {
                "step": np.arange(1, h + 1),
                "yhat": yhat,
                "yhat_lower": yhat - half_w,
                "yhat_upper": yhat + half_w,
            }
        )

    def _fit_ets(self, arr: np.ndarray) -> None:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            use_seasonal = len(arr) >= 2 * self.seasonal_period and self.seasonal_period > 1
            model = ExponentialSmoothing(
                arr,
                trend="add",
                seasonal="add" if use_seasonal else None,
                seasonal_periods=self.seasonal_period if use_seasonal else None,
                initialization_method="estimated",
            )
            res = model.fit(optimized=True)
            self._fitted_model = res
            self._residual_std = float(np.std(res.resid)) or float(np.std(np.diff(arr)))
        except Exception as exc:
            logger.warning("ets_fallback_to_fourier", reason=str(exc))
            self.method = "fourier_prophet"
            self._fit_fourier_trend(arr)

    def _fit_arima(self, arr: np.ndarray) -> None:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX

            model = SARIMAX(
                arr,
                order=self.arima_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            self._fitted_model = res
            self._residual_std = float(np.std(res.resid)) or 1.0
        except Exception as exc:
            logger.warning("arima_fallback_to_fourier", reason=str(exc))
            self.method = "fourier_prophet"
            self._fit_fourier_trend(arr)

    def _fit_theta(self, arr: np.ndarray) -> None:
        t = np.arange(len(arr), dtype=float)
        slope, _ = np.polyfit(t, arr, 1)
        alpha = 0.25
        level = arr[0]
        fitted = np.empty_like(arr)
        for i, val in enumerate(arr):
            fitted[i] = level
            level = alpha * val + (1.0 - alpha) * level
        self._theta_params = {"slope": float(slope), "ses_level": float(level), "alpha": alpha}
        self._residual_std = float(np.std(arr - fitted)) or 1.0

    def _fit_croston(self, arr: np.ndarray, alpha: float = 0.2) -> None:
        """Croston's method for intermittent demand forecasting."""
        non_zero_idx = np.where(arr > 0)[0]
        if len(non_zero_idx) == 0:
            self._croston_rate = 0.0
            self._residual_std = 0.0
            return
        z_est = float(arr[non_zero_idx[0]])
        p_est = float(non_zero_idx[0] + 1)
        last_idx = int(non_zero_idx[0])
        for idx in non_zero_idx[1:]:
            q = float(idx - last_idx)
            z_est = alpha * float(arr[idx]) + (1.0 - alpha) * z_est
            p_est = alpha * q + (1.0 - alpha) * p_est
            last_idx = int(idx)
        self._croston_rate = z_est / max(p_est, 1e-6)
        self._residual_std = float(np.std(arr - self._croston_rate))

    def _build_fourier_matrix(self, t: np.ndarray) -> np.ndarray:
        cols: list[np.ndarray] = [t / max(len(self._history), 1)]
        p = max(self.seasonal_period, 2)
        for k in range(1, self.fourier_order + 1):
            angle = 2.0 * np.pi * k * t / p
            cols.append(np.sin(angle))
            cols.append(np.cos(angle))
        return np.column_stack(cols)

    def _fit_fourier_trend(self, arr: np.ndarray) -> None:
        t = np.arange(len(arr), dtype=float)
        X = self._build_fourier_matrix(t)
        reg = Ridge(alpha=1.0)
        reg.fit(X, arr)
        self._fitted_model = reg
        self._residual_std = float(np.std(arr - reg.predict(X))) or 1.0

    @staticmethod
    def _z_score(conf: float) -> float:
        from scipy.stats import norm

        return float(norm.ppf(0.5 + min(max(conf, 0.5), 0.999) / 2.0))
