"""Deep Autoencoder for representation learning, dimensionality reduction, and anomaly detection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from omniforge.core.types import TaskType
from omniforge.models.base import BaseOmniForgeModel


class AutoencoderModel(BaseOmniForgeModel):
    """Symmetric Deep Autoencoder trained with Adam MSE reconstruction loss."""

    model_family = "deep_autoencoder"
    framework = "omniforge_numpy_autograd"

    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dim: int = 32,
        epochs: int = 80,
        learning_rate: float = 1e-2,
        batch_size: int = 64,
        contamination: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="DeepAutoencoder",
            task_type=TaskType.ANOMALY_DETECTION,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            contamination=contamination,
            **kwargs,
        )
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.contamination = float(contamination)
        self._scaler = StandardScaler()
        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []
        self.threshold_: float = 0.0
        self.loss_history_: list[float] = []

    def _fit(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any) -> None:
        X_scaled = self._scaler.fit_transform(X).astype(np.float64)
        n, d = X_scaled.shape
        h_dim = max(2, min(self.hidden_dim, max(d * 2, 8)))
        z_dim = max(1, min(self.latent_dim, h_dim))
        dims = [d, h_dim, z_dim, h_dim, d]

        rng = np.random.default_rng(self.random_seed)
        self._weights = [
            rng.normal(0.0, np.sqrt(2.0 / max(dims[i], 1)), size=(dims[i], dims[i + 1]))
            for i in range(len(dims) - 1)
        ]
        self._biases = [np.zeros((1, dims[i + 1]), dtype=np.float64) for i in range(len(dims) - 1)]

        m_w = [np.zeros_like(w) for w in self._weights]
        v_w = [np.zeros_like(w) for w in self._weights]
        m_b = [np.zeros_like(b) for b in self._biases]
        v_b = [np.zeros_like(b) for b in self._biases]
        step = 0
        self.loss_history_ = []

        for _ in range(self.epochs):
            perm = rng.permutation(n)
            ep_loss = 0.0
            batches = 0
            for start in range(0, n, self.batch_size):
                xb = X_scaled[perm[start : start + self.batch_size]]
                bs = len(xb)
                acts: list[np.ndarray] = [xb]
                pre: list[np.ndarray] = []
                for li in range(len(self._weights) - 1):
                    z = acts[-1] @ self._weights[li] + self._biases[li]
                    pre.append(z)
                    acts.append(np.tanh(z))
                recon = acts[-1] @ self._weights[-1] + self._biases[-1]
                diff = recon - xb
                ep_loss += float(np.mean(diff**2))
                batches += 1

                delta = (2.0 * diff) / (bs * d)
                grads_w: list[np.ndarray] = [np.empty(0)] * len(self._weights)
                grads_b: list[np.ndarray] = [np.empty(0)] * len(self._biases)
                for li in range(len(self._weights) - 1, -1, -1):
                    grads_w[li] = acts[li].T @ delta + 1e-4 * self._weights[li]
                    grads_b[li] = np.sum(delta, axis=0, keepdims=True)
                    if li > 0:
                        t = np.tanh(pre[li - 1])
                        delta = (delta @ self._weights[li].T) * (1.0 - t * t)

                step += 1
                for li in range(len(self._weights)):
                    m_w[li] = 0.9 * m_w[li] + 0.1 * grads_w[li]
                    v_w[li] = 0.999 * v_w[li] + 0.001 * (grads_w[li] ** 2)
                    m_b[li] = 0.9 * m_b[li] + 0.1 * grads_b[li]
                    v_b[li] = 0.999 * v_b[li] + 0.001 * (grads_b[li] ** 2)
                    mw_h = m_w[li] / (1.0 - 0.9**step)
                    vw_h = v_w[li] / (1.0 - 0.999**step)
                    mb_h = m_b[li] / (1.0 - 0.9**step)
                    vb_h = v_b[li] / (1.0 - 0.999**step)
                    self._weights[li] -= self.learning_rate * mw_h / (np.sqrt(vw_h) + 1e-8)
                    self._biases[li] -= self.learning_rate * mb_h / (np.sqrt(vb_h) + 1e-8)

            self.loss_history_.append(round(ep_loss / max(batches, 1), 6))

        errors = self._compute_recon_error(X_scaled)
        self.threshold_ = float(np.quantile(errors, 1.0 - self.contamination))

    def encode(self, X: Any) -> np.ndarray:
        """Project input rows into the compressed bottleneck latent representation."""
        self._check_is_fitted()
        X_arr, _ = self._prepare_X(X, fitting=False)
        h = self._scaler.transform(X_arr)
        for li in range(2):
            h = np.tanh(h @ self._weights[li] + self._biases[li])
        return np.asarray(h, dtype=float)

    def reconstruct(self, X: Any) -> np.ndarray:
        """Encode and decode ``X`` back into original feature space."""
        self._check_is_fitted()
        X_arr, _ = self._prepare_X(X, fitting=False)
        X_scaled = self._scaler.transform(X_arr)
        h = X_scaled
        for li in range(len(self._weights) - 1):
            h = np.tanh(h @ self._weights[li] + self._biases[li])
        recon_scaled = h @ self._weights[-1] + self._biases[-1]
        return np.asarray(self._scaler.inverse_transform(recon_scaled), dtype=float)

    def reconstruction_error(self, X: Any) -> np.ndarray:
        """Return per-sample MSE reconstruction error (higher indicates stronger anomaly)."""
        self._check_is_fitted()
        X_arr, _ = self._prepare_X(X, fitting=False)
        return self._compute_recon_error(self._scaler.transform(X_arr))

    def _compute_recon_error(self, X_scaled: np.ndarray) -> np.ndarray:
        h = X_scaled
        for li in range(len(self._weights) - 1):
            h = np.tanh(h @ self._weights[li] + self._biases[li])
        recon = h @ self._weights[-1] + self._biases[-1]
        return np.asarray(np.mean((recon - X_scaled) ** 2, axis=1), dtype=float)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Predict anomaly labels (1 = anomaly, 0 = normal) based on calibrated threshold."""
        errors = self._compute_recon_error(self._scaler.transform(X))
        return (errors > self.threshold_).astype(int)
