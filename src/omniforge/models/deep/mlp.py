"""Deep Multi-Layer Perceptron (MLP) with PyTorch backend and full NumPy Adam backprop engine."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from omniforge.core.types import TaskType
from omniforge.core.utils import is_package_available
from omniforge.models._torch_state import TorchStateMixin
from omniforge.models.base import BaseOmniForgeModel


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(shifted)
    return np.asarray(exps / np.maximum(np.sum(exps, axis=1, keepdims=True), 1e-12), dtype=np.float64)


def _activate(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "tanh":
        return np.asarray(np.tanh(z), dtype=np.float64)
    if kind == "gelu":
        zc = np.clip(z, -1e3, 1e3)
        u = np.sqrt(2.0 / np.pi) * (zc + 0.044715 * (zc**3))
        return np.asarray(0.5 * z * (1.0 + np.tanh(u)), dtype=np.float64)
    return np.asarray(np.maximum(0.0, z), dtype=np.float64)


def _activate_grad(z: np.ndarray, kind: str) -> np.ndarray:
    if kind == "tanh":
        t = np.tanh(z)
        return np.asarray(1.0 - t * t, dtype=np.float64)
    if kind == "gelu":
        # Exact analytical derivative of tanh-GELU (Hendrycks & Gimpel 2016), overflow-safe for |z| >> 1
        zc = np.clip(z, -1e3, 1e3)
        c = np.sqrt(2.0 / np.pi)
        u = c * (zc + 0.044715 * (zc**3))
        th = np.tanh(u)
        sech2 = 1.0 - th * th
        du_dz = c * (1.0 + 3.0 * 0.044715 * (zc**2))
        return np.asarray(0.5 * (1.0 + th) + 0.5 * zc * sech2 * du_dz, dtype=np.float64)
    return (z > 0.0).astype(np.float64)


class MLPModel(TorchStateMixin, BaseOmniForgeModel):
    """Deep Feedforward MLP for classification and regression with Dropout, L2 decay, and Adam."""

    model_family = "deep_mlp"
    framework = "pytorch_or_numpy"

    def __init__(
        self,
        hidden_dims: tuple[int, ...] | list[int] = (64, 32),
        *,
        activation: str = "relu",
        dropout: float = 0.1,
        learning_rate: float = 1e-2,
        weight_decay: float = 1e-4,
        epochs: int = 80,
        batch_size: int = 64,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="DeepMLP",
            task_type=task_type,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs,
        )
        self.hidden_dims = list(hidden_dims)
        self.activation = activation.lower()
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self._scaler = StandardScaler()
        self._weights: list[np.ndarray] = []
        self._biases: list[np.ndarray] = []
        self._torch_net: Any = None
        self._label_to_idx: dict[Any, int] = {}
        self._idx_to_label: dict[int, Any] = {}
        self.loss_history_: list[float] = []

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        X_scaled = self._scaler.fit_transform(X).astype(np.float64)
        is_clf = self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

        if is_clf:
            classes = np.unique(y)
            self._label_to_idx = {c: i for i, c in enumerate(classes)}
            self._idx_to_label = {i: c for i, c in enumerate(classes)}
            y_idx = np.array([self._label_to_idx[v] for v in y], dtype=int)
            out_dim = len(classes)
        else:
            y_idx = y.astype(np.float64)
            out_dim = 1

        if is_package_available("torch"):
            self._fit_torch(X_scaled, y_idx, out_dim, is_clf)
            return

        self.framework = "omniforge_numpy_autograd"
        self.metadata_.framework = self.framework
        self._fit_numpy(X_scaled, y_idx, out_dim, is_clf)

    def _build_torch_net(self, in_dim: int, out_dim: int) -> Any:
        import torch.nn as nn

        act_cls = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}.get(self.activation, nn.ReLU)

        layers: list[nn.Module] = []
        prev = in_dim
        for h in self.hidden_dims:
            layers.extend([nn.Linear(prev, h), act_cls()])
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def _fit_torch(self, X: np.ndarray, y: np.ndarray, out_dim: int, is_clf: bool) -> None:
        import torch
        import torch.nn as nn

        self.framework = "pytorch"
        self.metadata_.framework = self.framework
        torch.manual_seed(self.random_seed)
        self._torch_build_config = {"in_dim": X.shape[1], "out_dim": out_dim}
        net = self._build_torch_net(**self._torch_build_config)

        optimizer = torch.optim.AdamW(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        criterion: Any = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = (
            torch.tensor(y, dtype=torch.long)
            if is_clf
            else torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        )

        self.loss_history_ = []
        n = len(X)
        net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                optimizer.zero_grad()
                out = net(X_t[idx])
                loss = criterion(out, y_t[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                batches += 1
            self.loss_history_.append(round(epoch_loss / max(batches, 1), 6))
        net.eval()
        self._torch_net = net

    def _fit_numpy(self, X: np.ndarray, y: np.ndarray, out_dim: int, is_clf: bool) -> None:
        rng = np.random.default_rng(self.random_seed)
        dims = [X.shape[1], *self.hidden_dims, out_dim]
        self._weights = [
            rng.normal(0.0, np.sqrt(2.0 / max(dims[i], 1)), size=(dims[i], dims[i + 1]))
            for i in range(len(dims) - 1)
        ]
        self._biases = [np.zeros((1, dims[i + 1]), dtype=np.float64) for i in range(len(dims) - 1)]

        # Adam optimizer moments
        m_w = [np.zeros_like(w) for w in self._weights]
        v_w = [np.zeros_like(w) for w in self._weights]
        m_b = [np.zeros_like(b) for b in self._biases]
        v_b = [np.zeros_like(b) for b in self._biases]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        n = X.shape[0]
        self.loss_history_ = []

        for _ in range(self.epochs):
            perm = rng.permutation(n)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = X[idx]
                yb = y[idx]
                bs = len(xb)

                # Forward pass
                acts: list[np.ndarray] = [xb]
                pre_acts: list[np.ndarray] = []
                drop_masks: list[np.ndarray | None] = []
                for li in range(len(self._weights) - 1):
                    z = acts[-1] @ self._weights[li] + self._biases[li]
                    pre_acts.append(z)
                    a = _activate(z, self.activation)
                    if self.dropout > 0.0:
                        mask = (rng.uniform(0.0, 1.0, size=a.shape) >= self.dropout).astype(np.float64) / (
                            1.0 - self.dropout
                        )
                        a = a * mask
                        drop_masks.append(mask)
                    else:
                        drop_masks.append(None)
                    acts.append(a)

                logits = acts[-1] @ self._weights[-1] + self._biases[-1]
                if is_clf:
                    probs = _softmax(logits)
                    one_hot = np.zeros_like(probs)
                    one_hot[np.arange(bs), yb.astype(int)] = 1.0
                    loss = -float(np.mean(np.sum(one_hot * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)))
                    delta = (probs - one_hot) / bs
                else:
                    y_col = yb.reshape(-1, 1)
                    diff = logits - y_col
                    loss = float(np.mean(diff**2))
                    delta = (2.0 * diff) / bs

                epoch_loss += loss
                n_batches += 1

                # Backpropagation
                grads_w: list[np.ndarray] = [np.empty(0)] * len(self._weights)
                grads_b: list[np.ndarray] = [np.empty(0)] * len(self._biases)

                for li in range(len(self._weights) - 1, -1, -1):
                    # Pure data-loss gradient (decoupled from weight decay per Loshchilov & Hutter 2019 AdamW)
                    grads_w[li] = acts[li].T @ delta
                    grads_b[li] = np.sum(delta, axis=0, keepdims=True)
                    if li > 0:
                        delta = delta @ self._weights[li].T
                        if drop_masks[li - 1] is not None:
                            delta = delta * drop_masks[li - 1]
                        delta = delta * _activate_grad(pre_acts[li - 1], self.activation)

                step += 1
                for li in range(len(self._weights)):
                    m_w[li] = beta1 * m_w[li] + (1.0 - beta1) * grads_w[li]
                    v_w[li] = beta2 * v_w[li] + (1.0 - beta2) * (grads_w[li] ** 2)
                    m_b[li] = beta1 * m_b[li] + (1.0 - beta1) * grads_b[li]
                    v_b[li] = beta2 * v_b[li] + (1.0 - beta2) * (grads_b[li] ** 2)

                    mw_hat = m_w[li] / (1.0 - beta1**step)
                    vw_hat = v_w[li] / (1.0 - beta2**step)
                    mb_hat = m_b[li] / (1.0 - beta1**step)
                    vb_hat = v_b[li] / (1.0 - beta2**step)

                    # Decoupled weight decay (AdamW): w <- (1 - lr * lambda) * w - lr * m_hat / (sqrt(v_hat) + eps)
                    if self.weight_decay > 0.0:
                        self._weights[li] *= 1.0 - self.learning_rate * self.weight_decay
                    self._weights[li] -= self.learning_rate * mw_hat / (np.sqrt(vw_hat) + eps)
                    self._biases[li] -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)

            self.loss_history_.append(round(epoch_loss / max(n_batches, 1), 6))

    def _forward_logits(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X).astype(np.float64)
        if self._torch_net is not None:
            import torch

            with torch.no_grad():
                return np.asarray(
                    self._torch_net(torch.tensor(X_scaled, dtype=torch.float32)).numpy(),
                    dtype=np.float64,
                )
        h = X_scaled
        for li in range(len(self._weights) - 1):
            h = _activate(h @ self._weights[li] + self._biases[li], self.activation)
        return np.asarray(h @ self._weights[-1] + self._biases[-1], dtype=np.float64)

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return _softmax(self._forward_logits(X))

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        logits = self._forward_logits(X)
        if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            idx = np.argmax(logits, axis=1)
            return np.array([self._idx_to_label[int(i)] for i in idx])
        return logits.ravel()
