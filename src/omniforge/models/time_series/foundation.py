"""Chronos-style Zero-Shot Context/Patch Foundation Time-Series Forecaster."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from omniforge.core.exceptions import ForecastingError
from omniforge.core.utils import is_package_available


class FoundationForecaster:
    """Zero-shot probabilistic foundation forecaster with Chronos hook and local Patch-Retrieval engine.

    When Amazon ``chronos`` is installed and ``use_pretrained_chronos=True``, delegates to
    ``ChronosPipeline``. Otherwise runs a fast, zero-shot multi-scale normalized patch-similarity
    probabilistic sampler (combining scale-normalized k-NN historical trajectories + damped linear
    spectral continuation) to generate full quantile distributions without training.
    """

    def __init__(
        self,
        *,
        patch_length: int = 12,
        top_k_patches: int = 8,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        use_pretrained_chronos: bool = False,
        chronos_model_id: str = "amazon/chronos-t5-tiny",
    ) -> None:
        self.patch_length = max(4, int(patch_length))
        self.top_k_patches = max(2, int(top_k_patches))
        self.quantiles = tuple(float(q) for q in quantiles)
        self.use_pretrained_chronos = use_pretrained_chronos and is_package_available("chronos")
        self.chronos_model_id = chronos_model_id
        self._pipeline: Any = None

    def forecast_zero_shot(self, context: pd.Series | np.ndarray, horizon: int) -> pd.DataFrame:
        """Produce zero-shot probabilistic forecasts directly from raw context history (causal imputation)."""
        s_ctx = pd.to_numeric(pd.Series(context), errors="coerce")
        if not s_ctx.notna().any():
            raise ForecastingError("FoundationForecaster context contains no valid numeric values.")
        arr = s_ctx.ffill().fillna(float(s_ctx.dropna().iloc[0])).to_numpy(dtype=float).ravel()
        h = max(1, int(horizon))
        if len(arr) < 6:
            raise ForecastingError("FoundationForecaster requires at least 6 context observations.")

        if self.use_pretrained_chronos:
            import torch
            from chronos import ChronosPipeline

            if self._pipeline is None:
                self._pipeline = ChronosPipeline.from_pretrained(self.chronos_model_id)
            samples = self._pipeline.predict(torch.tensor(arr, dtype=torch.float32), h).numpy()[0]
            return self._summarize_samples(samples, h)

        # Zero-shot scale-normalized patch-similarity trajectory ensemble
        p_len = min(self.patch_length, max(4, len(arr) // 3))
        query = arr[-p_len:]
        q_mean = float(np.mean(query))
        q_std_raw = float(np.std(query, ddof=1)) if p_len > 1 else 0.0
        q_flat = q_std_raw < 1e-8 * max(abs(q_mean), 1.0)
        q_std = q_std_raw if not q_flat else 1.0
        q_norm = (query - q_mean) / q_std

        trajectories: list[np.ndarray] = []
        weights: list[float] = []

        # Collect candidate historical patches that have at least `h` future steps
        max_start = len(arr) - p_len - h
        if max_start >= 1:
            dists: list[tuple[float, int]] = []
            for start in range(0, max_start + 1):
                cand = arr[start : start + p_len]
                c_mean = float(np.mean(cand))
                c_std_raw = float(np.std(cand, ddof=1)) if p_len > 1 else 0.0
                if c_std_raw < 1e-8 * max(abs(c_mean), 1.0):
                    # Flat historical patch: avoid 1e6 scale explosion by comparing raw level deviation
                    dist = float(np.sqrt(np.mean((q_norm - 0.0) ** 2))) + 1.0
                else:
                    c_norm = (cand - c_mean) / c_std_raw
                    dist = float(np.sqrt(np.mean((q_norm - c_norm) ** 2)))
                dists.append((dist, start))
            dists.sort(key=lambda pair: pair[0])

            for dist, start in dists[: self.top_k_patches]:
                cand = arr[start : start + p_len]
                fut = arr[start + p_len : start + p_len + h]
                c_mean = float(np.mean(cand))
                c_std_raw = float(np.std(cand, ddof=1)) if p_len > 1 else 0.0
                if q_flat or c_std_raw < 1e-8 * max(abs(c_mean), 1.0):
                    # Additive shift continuation when either query or historical patch is flat (no division blow-up)
                    trajectories.append(arr[-1] + (fut - cand[-1]))
                else:
                    # Rescale future trajectory back to current query scale and anchor to last level
                    rescaled = ((fut - c_mean) / c_std_raw) * q_std + q_mean
                    anchor_shift = arr[-1] - (((cand[-1] - c_mean) / c_std_raw) * q_std + q_mean)
                    trajectories.append(rescaled + 0.5 * anchor_shift)
                weights.append(float(np.exp(-dist)))

        # Add damped local trend continuation path for robustness
        t_ctx = np.arange(len(arr), dtype=float)
        slope, _ = np.polyfit(t_ctx[-min(len(arr), 28) :], arr[-min(len(arr), 28) :], 1)
        damp = np.cumsum(0.92 ** np.arange(1, h + 1))
        trend_path = arr[-1] + slope * damp
        trajectories.append(trend_path)
        weights.append(0.8)

        # Draw bootstrap noise paths around weighted trajectories
        rng = np.random.default_rng(42)
        w_arr = np.asarray(weights, dtype=float)
        w_arr /= np.sum(w_arr)
        traj_mat = np.vstack(trajectories)
        diffs_tail = np.diff(arr[-min(len(arr), 30) :])
        resid_scale = float(np.std(diffs_tail, ddof=1)) if len(diffs_tail) > 1 else 0.0
        n_draws = 100
        draws = np.empty((n_draws, h), dtype=float)
        step_scale = np.sqrt(np.arange(1, h + 1, dtype=float))
        for i in range(n_draws):
            idx = int(rng.choice(len(trajectories), p=w_arr))
            draws[i] = traj_mat[idx] + rng.normal(0.0, resid_scale * 0.5 * step_scale)

        return self._summarize_samples(draws, h)

    def _summarize_samples(self, samples: np.ndarray, horizon: int) -> pd.DataFrame:
        out: dict[str, Any] = {
            "step": np.arange(1, horizon + 1),
            "yhat": np.median(samples, axis=0),
            "yhat_lower": np.quantile(samples, min(self.quantiles), axis=0),
            "yhat_upper": np.quantile(samples, max(self.quantiles), axis=0),
        }
        for q in self.quantiles:
            out[f"q{int(round(q * 100)):02d}"] = np.quantile(samples, q, axis=0)
        return pd.DataFrame(out)
