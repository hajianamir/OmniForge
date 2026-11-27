"""1D/2D Convolutional Neural Network (CNN) for sequences, signals, and image tensors."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from omniforge.core.types import TaskType
from omniforge.core.utils import is_package_available
from omniforge.models._torch_state import TorchStateMixin
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.deep.mlp import MLPModel


class CNN1DModel(TorchStateMixin, BaseOmniForgeModel):
    """Multi-scale 1D Convolutional Neural Network for temporal/signal/tabular classification & regression."""

    model_family = "deep_cnn"
    framework = "pytorch_or_numpy"

    def __init__(
        self,
        num_filters: int = 16,
        kernel_sizes: tuple[int, ...] = (2, 3),
        epochs: int = 40,
        learning_rate: float = 1e-2,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="DeepCNN1D",
            task_type=task_type,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
            epochs=epochs,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.num_filters = int(num_filters)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self._scaler = StandardScaler()
        self._kernels: list[np.ndarray] = []
        self._head: MLPModel | None = None
        self._torch_model: Any = None
        self._idx_to_label: dict[int, Any] = {}

    def _conv1d_pool_features(self, X: np.ndarray) -> np.ndarray:
        """Apply multi-scale 1D convolutions + ReLU + Global Average & Max Pooling."""
        n_samples, seq_len = X.shape
        pooled_blocks: list[np.ndarray] = []
        for kernel in self._kernels:
            k_len = kernel.shape[1]
            if seq_len < k_len:
                padded = np.pad(X, ((0, 0), (0, k_len - seq_len)), mode="edge")
            else:
                padded = X
            # Sliding windows: (n_samples, out_len, k_len)
            windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=k_len, axis=1)
            # Convolve with filters: (n_samples, out_len, n_filters)
            conv = np.maximum(0.0, np.einsum("nlk,fk->nlf", windows, kernel))
            avg_pool = conv.mean(axis=1)
            max_pool = conv.max(axis=1)
            pooled_blocks.extend([avg_pool, max_pool])
        return np.hstack(pooled_blocks)

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        X_scaled = self._scaler.fit_transform(X)
        is_clf = self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

        if is_package_available("torch") and X_scaled.shape[1] >= 2:
            self._fit_torch(X_scaled, y, is_clf)
            return

        self.framework = "omniforge_numpy_conv1d"
        self.metadata_.framework = self.framework
        rng = np.random.default_rng(self.random_seed)
        self._kernels = [
            rng.normal(0.0, np.sqrt(2.0 / max(k, 1)), size=(self.num_filters, k))
            for k in self.kernel_sizes
        ]
        conv_feats = self._conv1d_pool_features(X_scaled)
        self._head = MLPModel(
            hidden_dims=(32,),
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            task_type=self.task_type,
            random_seed=self.random_seed,
            track_carbon=False,
        )
        self._head.fit(conv_feats, y)
        self.loss_history_: list[float] = list(self._head.loss_history_)

    _torch_model_attribute = "_torch_model"

    def _build_torch_net(self, in_dim: int, out_dim: int) -> Any:
        import torch
        import torch.nn as nn

        k_size = min(max(self.kernel_sizes), in_dim)
        class _Net(nn.Module):
            def __init__(self, n_filters: int, kernel: int, out_d: int) -> None:
                super().__init__()
                self.conv = nn.Conv1d(1, n_filters, kernel_size=kernel, padding=kernel // 2)
                self.act = nn.ReLU()
                self.fc = nn.Linear(n_filters * 2, out_d)

            def forward(self, inp: Any) -> Any:
                h = self.act(self.conv(inp))
                pooled = torch.cat([h.mean(dim=-1), h.max(dim=-1).values], dim=-1)
                return self.fc(pooled)
        return _Net(self.num_filters, k_size, out_dim)

    def _fit_torch(self, X: np.ndarray, y: np.ndarray, is_clf: bool) -> None:
        import torch
        import torch.nn as nn

        self.framework = "pytorch"
        self.metadata_.framework = self.framework
        torch.manual_seed(self.random_seed)

        if is_clf:
            classes = np.unique(y)
            label_to_idx = {c: i for i, c in enumerate(classes)}
            self._idx_to_label = {i: c for i, c in enumerate(classes)}
            y_t = torch.tensor([label_to_idx[v] for v in y], dtype=torch.long)
            out_dim = len(classes)
        else:
            y_t = torch.tensor(y.astype(np.float32).reshape(-1, 1), dtype=torch.float32)
            out_dim = 1

        self._torch_build_config = {"in_dim": X.shape[1], "out_dim": out_dim}
        model = self._build_torch_net(**self._torch_build_config)
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn: Any = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()
        X_t = torch.tensor(X.astype(np.float32)).unsqueeze(1)

        model.train()
        self.loss_history_ = []
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(model(X_t), y_t)
            loss.backward()
            opt.step()
            self.loss_history_.append(round(float(loss.item()), 6))
        model.eval()
        self._torch_model = model

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        if self._torch_model is not None:
            import torch

            with torch.no_grad():
                out = self._torch_model(torch.tensor(X_scaled.astype(np.float32)).unsqueeze(1)).numpy()
            if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
                return np.array([self._idx_to_label[int(i)] for i in np.argmax(out, axis=1)])
            return np.asarray(out.ravel(), dtype=float)
        assert self._head is not None
        return self._head.predict(self._conv1d_pool_features(X_scaled))

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        if self._torch_model is not None:
            import torch

            with torch.no_grad():
                logits = self._torch_model(torch.tensor(X_scaled.astype(np.float32)).unsqueeze(1))
                return np.asarray(torch.softmax(logits, dim=-1).numpy(), dtype=float)
        assert self._head is not None
        return self._head.predict_proba(self._conv1d_pool_features(X_scaled))
