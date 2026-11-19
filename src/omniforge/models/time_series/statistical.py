"""Statistical time-series forecasting: ETS/Holt-Winters, ARIMA/SARIMAX, Theta, Croston, Fourier-Prophet."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
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
            from prophet import Prophet

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
            h_idx = np.arange(1, h + 1, dtype=float)
            slope = self._theta_params["slope"]
            level = self._theta_params["ses_level"]
            alpha = self._theta_params["alpha"]
            # Exact Hyndman & Billah (2003, Int. J. Forecasting 19:287-290) Eq. 5 finite-sample Theta drift:
            # X_hat_{T+h|T} = ell_T + 0.5 * b_0 * [ (h - 1) + (1 - (1 - alpha)^T) / alpha ]
            if alpha <= 1e-8:
                drift_offset = float(n)
            else:
                drift_offset = (1.0 - (1.0 - alpha) ** n) / alpha
            yhat = level + 0.5 * slope * ((h_idx - 1.0) + drift_offset)
            # Exact SES-with-drift h-step-ahead forecast error std: sigma * sqrt(1 + (h - 1) * alpha^2)
            theta_half_w = z * self._residual_std * np.sqrt(1.0 + (h_idx - 1.0) * (alpha**2))
            return pd.DataFrame(
                {
                    "step": np.arange(1, h + 1),
                    "yhat": yhat,
                    "yhat_lower": yhat - theta_half_w,
                    "yhat_upper": yhat + theta_half_w,
                }
            )
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
            ddof = 1 if len(res.resid) > 1 else 0
            self._residual_std = float(np.std(res.resid, ddof=ddof)) or float(np.std(np.diff(arr), ddof=1 if len(arr) > 2 else 0))
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
            # Drop diffuse Kalman state-space burn-in residuals (p + d / loglikelihood_burn)
            p_ord, d_ord, _ = self.arima_order
            burn = max(int(p_ord) + int(d_ord), int(getattr(res, "loglikelihood_burn", int(p_ord) + int(d_ord))))
            burn = min(max(0, burn), max(0, len(res.resid) - 2))
            eff_resid = np.asarray(res.resid[burn:], dtype=float)
            ddof = 1 if len(eff_resid) > 1 else 0
            self._residual_std = float(np.std(eff_resid, ddof=ddof)) or 1.0
        except Exception as exc:
            logger.warning("arima_fallback_to_fourier", reason=str(exc))
            self.method = "fourier_prophet"
            self._fit_fourier_trend(arr)

    def _fit_theta(self, arr: np.ndarray) -> None:
        """Fit Hyndman & Billah (2003) Theta method (theta=2) with SSE-optimal SES alpha."""
        n = len(arr)
        t = np.arange(n, dtype=float)
        slope, _ = np.polyfit(t, arr, 1)

        def _ses_sse(alpha_val: float) -> float:
            lvl = float(arr[0])
            sse = 0.0
            for i in range(1, n):
                err = float(arr[i]) - lvl
                sse += err * err
                lvl = alpha_val * float(arr[i]) + (1.0 - alpha_val) * lvl
            return sse

        opt = minimize_scalar(_ses_sse, bounds=(1e-4, 1.0), method="bounded")
        alpha = float(np.clip(opt.x, 1e-4, 1.0))

        level = float(arr[0])
        one_step_errors = np.empty(max(n - 1, 1), dtype=float)
        for i in range(n):
            if i > 0:
                one_step_errors[i - 1] = float(arr[i]) - level
            level = alpha * float(arr[i]) + (1.0 - alpha) * level

        self._theta_params = {"slope": float(slope), "ses_level": float(level), "alpha": alpha}
        # Unbiased 1-step-ahead innovation std (excluding trivial t=0 initialization error)
        dof = max(len(one_step_errors) - 1, 1)
        sigma = float(np.sqrt(np.sum(one_step_errors**2) / float(dof)))
        self._residual_std = sigma if sigma > 1e-12 else (float(np.std(np.diff(arr), ddof=1 if n > 2 else 0)) or 1.0)

    def _fit_croston(self, arr: np.ndarray, alpha: float = 0.2) -> None:
        """Syntetos-Boylan Approximation (SBA) bias-corrected Croston method for intermittent demand."""
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
        # Syntetos & Boylan (2005) bias correction factor (1 - alpha / 2) eliminates Croston positive bias
        self._croston_rate = (1.0 - 0.5 * alpha) * (z_est / max(p_est, 1e-6))
        self._residual_std = float(np.std(arr - self._croston_rate, ddof=1 if len(arr) > 1 else 0))

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
