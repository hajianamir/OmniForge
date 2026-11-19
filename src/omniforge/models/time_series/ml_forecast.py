"""Machine Learning recursive & direct lag-feature forecaster with conformal intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from omniforge.core.exceptions import ForecastingError
from omniforge.core.types import TaskType
from omniforge.models.classical import ClassicalMLModel


class MLForecaster:
    """Recursive lag + rolling + Fourier ML forecaster supporting exogenous regressors."""

    def __init__(
        self,
        algorithm: str = "gradient_boosting",
        *,
        lags: list[int] | None = None,
        seasonal_period: int = 7,
        confidence_level: float = 0.95,
        random_seed: int = 42,
    ) -> None:
        self.algorithm = algorithm
        self.seasonal_period = max(1, int(seasonal_period))
        self.lags = sorted(lags or [1, 2, 3, self.seasonal_period, self.seasonal_period * 2])
        self.confidence_level = float(confidence_level)
        self.random_seed = random_seed
        self._model: ClassicalMLModel | None = None
        self._history: np.ndarray = np.array([], dtype=float)
        self._conformal_q: float = 1.0
        self._max_lag: int = 1

    def _make_row(self, hist: np.ndarray, t_idx: int) -> np.ndarray:
        feats: list[float] = []
        for lag in self.lags:
            val = hist[-lag] if len(hist) >= lag else hist[0]
            feats.append(float(val))
        w3 = hist[-min(3, len(hist)) :]
        w7 = hist[-min(self.seasonal_period, len(hist)) :]
        feats.extend(
            [
                float(np.mean(w3)),
                float(np.std(w3)),
                float(np.mean(w7)),
                float(np.std(w7)),
                float(t_idx),
                float(np.sin(2.0 * np.pi * t_idx / max(self.seasonal_period, 2))),
                float(np.cos(2.0 * np.pi * t_idx / max(self.seasonal_period, 2))),
            ]
        )
        return np.asarray(feats, dtype=float)

    def fit(self, y: pd.Series | np.ndarray, exog: pd.DataFrame | None = None) -> MLForecaster:
        """Fit lag-engineered ML regressor and calibrate split conformal residual quantile (causal imputation)."""
        s_y = pd.to_numeric(pd.Series(y), errors="coerce")
        if not s_y.notna().any():
            raise ForecastingError("MLForecaster target series contains no valid numeric values.")
        arr = s_y.ffill().fillna(float(s_y.dropna().iloc[0])).to_numpy(dtype=float).ravel()
        self._max_lag = min(max(self.lags), max(1, len(arr) // 3))
        self.lags = [lag for lag in self.lags if lag <= self._max_lag] or [1]
        if len(arr) <= self._max_lag + 4:
            raise ForecastingError("Series too short for MLForecaster lag window.")
        self._history = arr.copy()

        rows: list[np.ndarray] = []
        targets: list[float] = []
        exog_arr = exog.to_numpy(dtype=float) if exog is not None else None

        for t in range(self._max_lag, len(arr)):
            base = self._make_row(arr[:t], t)
            if exog_arr is not None:
                base = np.concatenate([base, exog_arr[t]])
            rows.append(base)
            targets.append(float(arr[t]))

        X_mat = np.vstack(rows)
        y_vec = np.asarray(targets, dtype=float)

        # Hold out last 20% for split conformal interval calibration (Angelopoulos & Bates 2021)
        alpha = float(np.clip(1.0 - self.confidence_level, 1e-4, 0.5))
        cal_size = max(5, int(len(X_mat) * 0.2))
        if len(X_mat) > cal_size + 10:
            cal_model = ClassicalMLModel(
                algorithm=self.algorithm,
                task_type=TaskType.REGRESSION,
                random_seed=self.random_seed,
                track_carbon=False,
            )
            cal_model.fit(X_mat[:-cal_size], y_vec[:-cal_size])
            cal_resid = np.abs(y_vec[-cal_size:] - cal_model.predict(X_mat[-cal_size:]))
            n_cal = len(cal_resid)
            k = min(max(int(np.ceil((n_cal + 1) * (1.0 - alpha))), 1), n_cal)
            self._conformal_q = float(np.partition(cal_resid, k - 1)[k - 1])
        else:
            z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
            self._conformal_q = float(np.std(np.diff(arr), ddof=1)) * z_crit if len(arr) > 2 else 1.0

        self._model = ClassicalMLModel(
            algorithm=self.algorithm,
            task_type=TaskType.REGRESSION,
            random_seed=self.random_seed,
            track_carbon=False,
        )
        self._model.fit(X_mat, y_vec)
        return self

    def predict(self, horizon: int, future_exog: pd.DataFrame | None = None) -> pd.DataFrame:
        """Recursively roll forward ``horizon`` steps with conformal uncertainty bounds."""
        if self._model is None:
            raise ForecastingError("MLForecaster must be fitted before predict().")
        h = max(1, int(horizon))
        buffer = list(self._history)
        n_start = len(self._history)
        exog_arr = future_exog.to_numpy(dtype=float) if future_exog is not None else None
        preds = np.empty(h, dtype=float)

        for step in range(h):
            row = self._make_row(np.asarray(buffer, dtype=float), n_start + step)
            if exog_arr is not None and step < len(exog_arr):
                row = np.concatenate([row, exog_arr[step]])
            yhat = float(self._model.predict(row.reshape(1, -1))[0])
            preds[step] = yhat
            buffer.append(yhat)

        growth = np.sqrt(np.arange(1, h + 1, dtype=float))
        half_w = self._conformal_q * np.minimum(growth, 3.0)
        return pd.DataFrame(
            {
                "step": np.arange(1, h + 1),
                "yhat": preds,
                "yhat_lower": preds - half_w,
                "yhat_upper": preds + half_w,
            }
        )
