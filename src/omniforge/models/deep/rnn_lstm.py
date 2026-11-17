"""Recurrent Neural Networks (RNN, LSTM, GRU) for sequential classification and regression."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from omniforge.core.types import TaskType
from omniforge.core.utils import is_package_available
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.deep.mlp import MLPModel


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class RecurrentSequenceModel(BaseOmniForgeModel):
    """Sequence model supporting vanilla RNN, LSTM, and GRU gated recurrent units."""

    model_family = "deep_recurrent"
    framework = "pytorch_or_numpy"

    def __init__(
        self,
        cell_type: str = "lstm",
        hidden_size: int = 32,
        num_layers: int = 1,
        epochs: int = 40,
        learning_rate: float = 1e-2,
        task_type: TaskType = TaskType.REGRESSION,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=f"Deep{cell_type.upper()}",
            task_type=task_type,
            cell_type=cell_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            epochs=epochs,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.cell_type = cell_type.lower()
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self._scaler = StandardScaler()
        self._gate_weights: dict[str, np.ndarray] = {}
        self._readout: MLPModel | None = None
        self._torch_net: Any = None
        self._idx_to_label: dict[int, Any] = {}

    def _init_numpy_cell(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        h = self.hidden_size
        scale = 1.0 / np.sqrt(max(h, 1))
        if self.cell_type == "lstm":
            self._gate_weights = {
                "W_x": rng.normal(0.0, scale, size=(1, 4 * h)),
                "W_h": rng.normal(0.0, scale, size=(h, 4 * h)),
                "b": np.zeros((1, 4 * h)),
            }
        elif self.cell_type == "gru":
            self._gate_weights = {
                "W_x": rng.normal(0.0, scale, size=(1, 3 * h)),
                "W_h": rng.normal(0.0, scale, size=(h, 3 * h)),
                "b": np.zeros((1, 3 * h)),
            }
        else:
            self._gate_weights = {
                "W_x": rng.normal(0.0, scale, size=(1, h)),
                "W_h": rng.normal(0.0, scale, size=(h, h)),
                "b": np.zeros((1, h)),
            }

    def _encode_sequence_numpy(self, X: np.ndarray) -> np.ndarray:
        """Roll the gated recurrent cell across feature time-steps and concatenate final + mean states."""
        n_samples, seq_len = X.shape
        h_dim = self.hidden_size
        h = np.zeros((n_samples, h_dim), dtype=np.float64)
        c = np.zeros((n_samples, h_dim), dtype=np.float64)
        h_accum = np.zeros_like(h)

        W_x, W_h, b = self._gate_weights["W_x"], self._gate_weights["W_h"], self._gate_weights["b"]

        for t in range(seq_len):
            xt = X[:, t : t + 1]
            if self.cell_type == "lstm":
                gates = xt @ W_x + h @ W_h + b
                i_g = _sigmoid(gates[:, 0:h_dim])
                f_g = _sigmoid(gates[:, h_dim : 2 * h_dim] + 1.0)
                g_g = np.tanh(gates[:, 2 * h_dim : 3 * h_dim])
                o_g = _sigmoid(gates[:, 3 * h_dim : 4 * h_dim])
                c = f_g * c + i_g * g_g
                h = o_g * np.tanh(c)
            elif self.cell_type == "gru":
                gates = xt @ W_x + h @ W_h + b
                z_g = _sigmoid(gates[:, 0:h_dim])
                r_g = _sigmoid(gates[:, h_dim : 2 * h_dim])
                cand_x = xt @ W_x[:, 2 * h_dim :]
                cand_h = (r_g * h) @ W_h[:, 2 * h_dim :] + b[:, 2 * h_dim :]
                n_g = np.tanh(cand_x + cand_h)
                h = (1.0 - z_g) * n_g + z_g * h
            else:
                h = np.tanh(xt @ W_x + h @ W_h + b)
            h_accum += h

        return np.hstack([h, h_accum / max(seq_len, 1)])

    @staticmethod
    def _ensure_2d(X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        return arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        X_scaled = self._scaler.fit_transform(self._ensure_2d(X))
        is_clf = self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

        if is_package_available("torch"):
            self._fit_torch(X_scaled, y, is_clf)
            return

        self.framework = f"uap_numpy_{self.cell_type}"
        self.metadata_.framework = self.framework
        self._init_numpy_cell()
        seq_repr = self._encode_sequence_numpy(X_scaled)
        self._readout = MLPModel(
            hidden_dims=(32,),
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            task_type=self.task_type,
            random_seed=self.random_seed,
            track_carbon=False,
        )
        self._readout.fit(seq_repr, y)

    def _fit_torch(self, X: np.ndarray, y: np.ndarray, is_clf: bool) -> None:
        import torch  # type: ignore[import-not-found]
        import torch.nn as nn  # type: ignore[import-not-found]

        self.framework = "pytorch"
        self.metadata_.framework = self.framework
        torch.manual_seed(self.random_seed)
        if is_clf:
            classes = np.unique(y)
            l2i = {c: i for i, c in enumerate(classes)}
            self._idx_to_label = {i: c for i, c in enumerate(classes)}
            y_t = torch.tensor([l2i[v] for v in y], dtype=torch.long)
            out_dim = len(classes)
        else:
            y_t = torch.tensor(y.astype(np.float32).reshape(-1, 1), dtype=torch.float32)
            out_dim = 1

        cell_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}.get(self.cell_type, nn.LSTM)

        class _RNNNet(nn.Module):
            def __init__(self, hidden: int, layers: int, out_d: int) -> None:
                super().__init__()
                self.rnn = cell_cls(1, hidden, num_layers=layers, batch_first=True)
                self.fc = nn.Linear(hidden, out_d)

            def forward(self, inp: Any) -> Any:
                out, _ = self.rnn(inp)
                return self.fc(out[:, -1, :])

        net = _RNNNet(self.hidden_size, self.num_layers, out_dim)
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_fn: Any = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()
        X_t = torch.tensor(X.astype(np.float32)).unsqueeze(-1)
        net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(net(X_t), y_t)
            loss.backward()
            opt.step()
        net.eval()
        self._torch_net = net

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X_scaled = self._scaler.transform(self._ensure_2d(X))
        if self._torch_net is not None:
            import torch  # type: ignore[import-not-found]

            with torch.no_grad():
                out = self._torch_net(torch.tensor(X_scaled.astype(np.float32)).unsqueeze(-1)).numpy()
            if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
                return np.array([self._idx_to_label[int(i)] for i in np.argmax(out, axis=1)])
            return out.ravel()
        assert self._readout is not None
        return self._readout.predict(self._encode_sequence_numpy(X_scaled))

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X_scaled = self._scaler.transform(self._ensure_2d(X))
        if self._torch_net is not None:
            import torch  # type: ignore[import-not-found]

            with torch.no_grad():
                logits = self._torch_net(torch.tensor(X_scaled.astype(np.float32)).unsqueeze(-1))
                return torch.softmax(logits, dim=-1).numpy().astype(float)
        assert self._readout is not None
        return self._readout.predict_proba(self._encode_sequence_numpy(X_scaled))
