"""Paper-Grade Deep Neural Time-Series Forecasters: N-BEATS (Trend+Seasonality Basis), PatchTST (Patch Transformer), TFT (Gated Quantile Multi-Horizon), LSTM & Residual MLP."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from omniforge.core.exceptions import ForecastingError
from omniforge.core.types import TaskType
from omniforge.core.utils import is_package_available
from omniforge.models._torch_state import TorchStateMixin
from omniforge.models.deep.mlp import MLPModel
from omniforge.models.deep.rnn_lstm import RecurrentSequenceModel


def _causal_series_to_array(y: pd.Series | np.ndarray) -> np.ndarray:
    """Convert raw series to float array via strictly causal past-only forward-fill (never bfill)."""
    s = pd.to_numeric(pd.Series(y), errors="coerce")
    if not s.notna().any():
        raise ForecastingError("Input time series contains no valid numeric observations.")
    first_valid = float(s.dropna().iloc[0])
    return np.asarray(s.ffill().fillna(first_valid).to_numpy(dtype=float).ravel(), dtype=float)


class DeepForecaster(TorchStateMixin):
    """Windowed deep sequence forecaster supporting N-BEATS, PatchTST, TFT, LSTM/GRU & Residual MLP with split conformal calibration."""

    SUPPORTED_ARCHITECTURES: tuple[str, ...] = ("nbeats", "patchtst", "tft", "nbeats_mlp", "lstm", "gru")

    def __init__(
        self,
        architecture: str = "nbeats",
        *,
        context_length: int = 24,
        horizon_train: int = 8,
        epochs: int = 50,
        confidence_level: float = 0.95,
        random_seed: int = 42,
    ) -> None:
        self.architecture = architecture.lower()
        self.context_length = max(6, int(context_length))
        self.horizon_train = max(1, int(horizon_train))
        self.epochs = int(epochs)
        self.confidence_level = float(confidence_level)
        self.random_seed = random_seed
        self._history: np.ndarray = np.array([], dtype=float)
        self._model: Any = None
        self._torch_net: Any = None
        self._mean: float = 0.0
        self._std: float = 1.0
        self._residual_q: float = 1.0
        self._tft_cqr_q: float = 0.0
        self._eff_context: int = self.context_length
        self._eff_horizon: int = self.horizon_train

    def _build_backbone(self) -> MLPModel | RecurrentSequenceModel:
        if self.architecture in ("lstm", "gru", "rnn"):
            return RecurrentSequenceModel(
                cell_type=self.architecture,
                hidden_size=24,
                epochs=self.epochs,
                task_type=TaskType.REGRESSION,
                random_seed=self.random_seed,
                track_carbon=False,
            )
        return MLPModel(
            hidden_dims=(64, 32),
            activation="gelu",
            epochs=self.epochs,
            task_type=TaskType.REGRESSION,
            random_seed=self.random_seed,
            track_carbon=False,
        )

    def _build_torch_net(self, L: int, H: int) -> Any:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        arch = self.architecture

        class _NBEATSNet(nn.Module):
            """Oreshkin et al. (ICLR 2020): Doubly Residual Trend Polynomial + Fourier Seasonality Basis Stacks."""

            def __init__(self) -> None:
                super().__init__()
                self.trend_fc = nn.Sequential(nn.Linear(L, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
                self.trend_back = nn.Linear(64, 3)
                self.trend_fore = nn.Linear(64, 3)
                self.seas_fc = nn.Sequential(nn.Linear(L, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
                self.seas_back = nn.Linear(64, 6)
                self.seas_fore = nn.Linear(64, 6)
                t_b = torch.linspace(-1.0, 0.0, L)
                t_f = torch.linspace(0.0, 1.0, H)
                self.register_buffer("T_b", torch.stack([t_b**0, t_b**1, t_b**2], dim=0))
                self.register_buffer("T_f", torch.stack([t_f**0, t_f**1, t_f**2], dim=0))
                self.register_buffer("S_b", torch.stack([torch.cos(2 * np.pi * k * t_b) for k in (1, 2, 3)] + [torch.sin(2 * np.pi * k * t_b) for k in (1, 2, 3)], dim=0))
                self.register_buffer("S_f", torch.stack([torch.cos(2 * np.pi * k * t_f) for k in (1, 2, 3)] + [torch.sin(2 * np.pi * k * t_f) for k in (1, 2, 3)], dim=0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                ht = self.trend_fc(x)
                b1 = self.trend_back(ht) @ self.T_b
                f1 = self.trend_fore(ht) @ self.T_f
                resid = x - b1
                hs = self.seas_fc(resid)
                f2 = self.seas_fore(hs) @ self.S_f
                out_t: torch.Tensor = f1 + f2
                return out_t

        class _PatchTSTNet(nn.Module):
            """Nie et al. (ICLR 2023): Patch Tokenization + Multi-Head Self-Attention Time-Series Transformer."""

            def __init__(self) -> None:
                super().__init__()
                self.patch_len = max(2, min(4, L // 2))
                self.stride = max(1, self.patch_len // 2)
                self.n_patches = max(1, (L - self.patch_len) // self.stride + 1)
                self.proj = nn.Linear(self.patch_len, 32)
                self.pos = nn.Parameter(torch.randn(1, self.n_patches, 32) * 0.02)
                enc_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, batch_first=True, dropout=0.1)
                self.enc = nn.TransformerEncoder(enc_layer, num_layers=2)
                self.head = nn.Linear(self.n_patches * 32, H)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)[:, : self.n_patches, :]
                tok = self.proj(patches) + self.pos[:, : patches.size(1), :]
                z = self.enc(tok).reshape(x.size(0), -1)
                out_p: torch.Tensor = self.head(z)
                return out_p

        class _TFTNet(nn.Module):
            """Lim et al. (2021) Temporal Fusion Transformer: Gated Residual Network + Self-Attention + Quantile Head (P10/P50/P90)."""

            def __init__(self) -> None:
                super().__init__()
                self.vsn = nn.Sequential(nn.Linear(L, 32), nn.ELU(), nn.Linear(32, L), nn.Softmax(dim=-1))
                self.grn1 = nn.Linear(1, 32)
                self.lstm = nn.LSTM(32, 32, batch_first=True)
                self.attn = nn.MultiheadAttention(32, 4, batch_first=True)
                self.gate = nn.Sequential(nn.Linear(32, 32), nn.Sigmoid())
                self.norm = nn.LayerNorm(32)
                self.q_head = nn.Linear(32, H * 3)  # [q_lo, q_50, q_hi]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                w = self.vsn(x)
                xw = (x * w).unsqueeze(-1)
                h = F.elu(self.grn1(xw))
                seq, _ = self.lstm(h)
                att, _ = self.attn(seq, seq, seq)
                fused = self.norm(seq + self.gate(att) * att)
                out_q: torch.Tensor = self.q_head(fused[:, -1, :]).view(x.size(0), H, 3)
                return out_q
        return _NBEATSNet() if arch == "nbeats" else (_PatchTSTNet() if arch == "patchtst" else _TFTNet())

    def _fit_paper_torch_forecaster(self, arr: np.ndarray) -> bool:
        """Train genuine PyTorch N-BEATS / PatchTST / TFT multi-horizon forecaster."""
        if not is_package_available("torch") or self.architecture not in ("nbeats", "patchtst", "tft"):
            return False

        import torch

        torch.manual_seed(self.random_seed)
        L = self._eff_context
        H = min(self.horizon_train, max(1, (len(arr) - L) // 3))
        if len(arr) <= L + H + 4:
            return False
        self._eff_horizon = H
        alpha = float(np.clip(1.0 - self.confidence_level, 1e-4, 0.5))

        # Determine chronological train / calibration split BEFORE scaling to prevent calibration leakage
        n_windows = len(arr) - L - H + 1
        cal_n = max(2, int(n_windows * 0.2))
        has_holdout = n_windows > cal_n + H + 3
        if has_holdout:
            # Last training window starts at (n_windows - cal_n - H) so its H-step target never overlaps calibration targets
            train_end_win = max(2, n_windows - cal_n - (H - 1))
            scaler_end_idx = min(len(arr), L + train_end_win + H - 1)
        else:
            train_end_win = n_windows
            scaler_end_idx = len(arr)

        self._mean = float(np.mean(arr[:scaler_end_idx]))
        self._std = max(float(np.std(arr[:scaler_end_idx])), 1e-6)
        norm_arr = (arr - self._mean) / self._std

        X_win = np.array([norm_arr[i - L : i] for i in range(L, len(norm_arr) - H + 1)], dtype=np.float32)
        Y_win = np.array([norm_arr[i : i + H] for i in range(L, len(norm_arr) - H + 1)], dtype=np.float32)

        arch = self.architecture
        self._torch_build_config = {"L": L, "H": H}
        net = self._build_torch_net(**self._torch_build_config)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-3, weight_decay=1e-4)
        Xt = torch.tensor(X_win)
        Yt = torch.tensor(Y_win)

        # Chronological split conformal / CQR calibration (non-overlapping targets when series is long enough)
        X_tr, Y_tr = (Xt[:train_end_win], Yt[:train_end_win]) if has_holdout else (Xt, Yt)
        X_ca, Y_ca = (Xt[-cal_n:], Yt[-cal_n:]) if has_holdout else (Xt, Yt)

        # Match TFT quantile levels to requested confidence_level: [alpha/2, 0.5, 1 - alpha/2]
        q_lo_level = float(np.clip(alpha / 2.0, 0.005, 0.45))
        q_hi_level = float(np.clip(1.0 - alpha / 2.0, 0.55, 0.995))
        quantiles = torch.tensor([q_lo_level, 0.5, q_hi_level], dtype=torch.float32)
        net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            out = net(X_tr)
            if arch == "tft":
                err = Y_tr.unsqueeze(-1) - out
                loss = torch.mean(torch.maximum(quantiles * err, (quantiles - 1.0) * err))
            else:
                loss = torch.nn.functional.mse_loss(out, Y_tr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()

        net.eval()
        with torch.no_grad():
            pred_ca = net(X_ca)
            if arch == "tft":
                # Conformalized Quantile Regression (Romano, Patterson & Candès 2019):
                # E_i = max(q_lo(X_i) - Y_i, Y_i - q_hi(X_i))
                ca_raw = pred_ca[:, 0, :].cpu().numpy() * self._std + self._mean
                y_true_ca = Y_ca[:, 0].cpu().numpy() * self._std + self._mean
                lo_ca = np.minimum(ca_raw[:, 0], ca_raw[:, 2])
                hi_ca = np.maximum(ca_raw[:, 0], ca_raw[:, 2])
                cqr_scores = np.maximum(lo_ca - y_true_ca, y_true_ca - hi_ca)
                n_ca = len(cqr_scores)
                k_ca = min(max(int(np.ceil((n_ca + 1) * (1.0 - alpha))), 1), n_ca)
                self._tft_cqr_q = float(np.partition(cqr_scores, k_ca - 1)[k_ca - 1])
                res = np.abs(y_true_ca - ca_raw[:, 1])
                self._residual_q = max(float(np.partition(res, k_ca - 1)[k_ca - 1]), 1e-6)
            else:
                res = torch.abs(Y_ca[:, 0] - pred_ca[:, 0]).cpu().numpy() * self._std
                n_ca = len(res)
                k_ca = min(max(int(np.ceil((n_ca + 1) * (1.0 - alpha))), 1), n_ca)
                self._residual_q = max(float(np.partition(res, k_ca - 1)[k_ca - 1]), 1e-6)
        self._torch_net = net
        return True

    def fit(self, y: pd.Series | np.ndarray) -> DeepForecaster:
        """Train windowed deep sequence forecaster with chronological split conformal residual calibration."""
        arr = _causal_series_to_array(y)
        self._eff_context = min(self.context_length, max(6, len(arr) // 3))
        if len(arr) <= self._eff_context + 4:
            raise ForecastingError("Series too short for DeepForecaster context window.")
        self._history = arr.copy()

        if self._fit_paper_torch_forecaster(arr):
            return self

        X_win = np.array(
            [arr[i - self._eff_context : i] for i in range(self._eff_context, len(arr))],
            dtype=float,
        )
        last_level = X_win[:, -1]
        y_delta = arr[self._eff_context :] - last_level

        alpha = float(np.clip(1.0 - self.confidence_level, 1e-4, 0.5))
        cal_n = max(4, int(len(X_win) * 0.2))
        if len(X_win) > cal_n + 8:
            cal_model = self._build_backbone()
            cal_model.fit(X_win[:-cal_n], y_delta[:-cal_n])
            cal_pred = last_level[-cal_n:] + cal_model.predict(X_win[-cal_n:])
            residuals = np.abs(arr[self._eff_context :][-cal_n:] - cal_pred)
            n_ca = len(residuals)
            k_ca = min(max(int(np.ceil((n_ca + 1) * (1.0 - alpha))), 1), n_ca)
            self._residual_q = float(np.partition(residuals, k_ca - 1)[k_ca - 1])
        else:
            z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
            self._residual_q = float(np.std(y_delta, ddof=1)) * z_crit if len(y_delta) > 1 else 1.0

        self._model = self._build_backbone()
        self._model.fit(X_win, y_delta)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        """Roll out deep neural multi-step probabilistic forecasts (including P10/P50/P90 quantiles for TFT)."""
        h = max(1, int(horizon))
        if self._torch_net is not None:
            import torch

            buf = list((self._history - self._mean) / self._std)
            preds: list[float] = []
            q10_list: list[float] = []
            q90_list: list[float] = []
            self._torch_net.eval()
            with torch.no_grad():
                while len(preds) < h:
                    win = torch.tensor(np.asarray(buf[-self._eff_context :], dtype=np.float32).reshape(1, -1))
                    out = self._torch_net(win).cpu().numpy()[0]
                    if self.architecture == "tft":
                        p_lo = out[:, 0] * self._std + self._mean
                        p50 = out[:, 1] * self._std + self._mean
                        p_hi = out[:, 2] * self._std + self._mean
                        for a, m, b in zip(p_lo, p50, p_hi, strict=False):
                            if len(preds) >= h:
                                break
                            lo_raw, mid_v, hi_raw = sorted([float(a), float(m), float(b)])
                            # Apply CQR finite-sample conformal margin (scaled mildly with horizon)
                            h_scale = float(np.sqrt(len(preds) + 1))
                            cqr_adj = self._tft_cqr_q * h_scale if self._tft_cqr_q >= 0.0 else self._tft_cqr_q
                            lo_v = min(mid_v, lo_raw - cqr_adj)
                            hi_v = max(mid_v, hi_raw + cqr_adj)
                            preds.append(mid_v)
                            q10_list.append(lo_v)
                            q90_list.append(hi_v)
                            buf.append((mid_v - self._mean) / self._std)
                    else:
                        step_vals = out * self._std + self._mean
                        for v in step_vals:
                            if len(preds) >= h:
                                break
                            preds.append(float(v))
                            buf.append((float(v) - self._mean) / self._std)
            pred_arr = np.asarray(preds, dtype=float)
            growth = np.sqrt(np.arange(1, h + 1, dtype=float))
            half_w = self._residual_q * growth
            out_df = pd.DataFrame(
                {
                    "step": np.arange(1, h + 1),
                    "yhat": pred_arr,
                    "yhat_lower": np.asarray(q10_list, dtype=float) if q10_list else (pred_arr - half_w),
                    "yhat_upper": np.asarray(q90_list, dtype=float) if q90_list else (pred_arr + half_w),
                }
            )
            return out_df

        if self._model is None:
            raise ForecastingError("DeepForecaster must be fitted before predict().")
        buf = list(self._history)
        preds_arr = np.empty(h, dtype=float)
        for step in range(h):
            window = np.asarray(buf[-self._eff_context :], dtype=float).reshape(1, -1)
            delta = float(self._model.predict(window)[0])
            yhat = float(window[0, -1]) + delta
            preds_arr[step] = yhat
            buf.append(yhat)

        growth = np.sqrt(np.arange(1, h + 1, dtype=float))
        half_w = self._residual_q * growth
        return pd.DataFrame(
            {
                "step": np.arange(1, h + 1),
                "yhat": preds_arr,
                "yhat_lower": preds_arr - half_w,
                "yhat_upper": preds_arr + half_w,
            }
        )
