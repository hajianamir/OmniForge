"""Paper-Grade ResNet-Tabular Architecture (Gorishniy et al., NeurIPS 2021): Pre-Activation Residual Skip Blocks with BatchNorm, SiLU & Dropout."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from omniforge.core.types import TaskType
from omniforge.core.utils import is_package_available
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.deep.mlp import MLPModel


class ResNetTabularModel(BaseOmniForgeModel):
    """Exact Gorishniy et al. (NeurIPS 2021) Tabular ResNet with PyTorch AdamW/CosineLR & NumPy fallback."""

    model_family = "deep_resnet_tabular"
    framework = "pytorch_or_numpy"

    def __init__(
        self,
        d_main: int = 128,
        d_hidden: int = 256,
        n_blocks: int = 3,
        dropout_first: float = 0.2,
        dropout_second: float = 0.1,
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-4,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="ResNetTabular",
            task_type=task_type,
            d_main=d_main,
            d_hidden=d_hidden,
            n_blocks=n_blocks,
            dropout_first=dropout_first,
            dropout_second=dropout_second,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            **kwargs,
        )
        self.d_main = max(16, int(d_main))
        self.d_hidden = max(16, int(d_hidden))
        self.n_blocks = max(1, int(n_blocks))
        self.dropout_first = float(dropout_first)
        self.dropout_second = float(dropout_second)
        self.epochs = int(epochs)
        self.batch_size = max(16, int(batch_size))
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self._scaler = StandardScaler()
        self._torch_net: Any = None
        self._fallback: MLPModel | None = None
        self._idx_to_label: dict[int, Any] = {}

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        _ = kwargs
        assert y is not None
        X_s = self._scaler.fit_transform(X).astype(np.float32)
        is_clf = self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

        if is_clf:
            classes = np.unique(y)
            self._classes = classes
            lbl_to_idx = {c: i for i, c in enumerate(classes)}
            self._idx_to_label = {i: c for i, c in enumerate(classes)}
            y_enc = np.array([lbl_to_idx[v] for v in y], dtype=np.int64)
            out_dim = len(classes)
        else:
            y_enc = np.asarray(y, dtype=np.float32)
            out_dim = 1

        if is_package_available("torch"):
            import torch
            import torch.nn as nn

            torch.manual_seed(self.random_seed)
            in_dim = X_s.shape[1]
            d_m, d_h = self.d_main, self.d_hidden
            dp1, dp2 = self.dropout_first, self.dropout_second

            class _ResBlock(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.block = nn.Sequential(
                        nn.BatchNorm1d(d_m),
                        nn.Linear(d_m, d_h),
                        nn.SiLU(),
                        nn.Dropout(dp1),
                        nn.Linear(d_h, d_m),
                        nn.Dropout(dp2),
                    )

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    out: torch.Tensor = x + self.block(x)
                    return out

            net = nn.Sequential(
                nn.Linear(in_dim, d_m),
                *[_ResBlock() for _ in range(self.n_blocks)],
                nn.BatchNorm1d(d_m),
                nn.SiLU(),
                nn.Linear(d_m, out_dim),
            )
            opt = torch.optim.AdamW(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.epochs, 1))
            loss_fn: Any = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()

            Xt = torch.tensor(X_s, dtype=torch.float32)
            yt = torch.tensor(y_enc, dtype=torch.long if is_clf else torch.float32)
            n = len(Xt)
            bs = min(self.batch_size, max(16, n))

            net.train()
            for _ in range(self.epochs):
                perm = torch.randperm(n)
                for start in range(0, n, bs):
                    idx = perm[start : start + bs]
                    if len(idx) < 2:
                        continue
                    opt.zero_grad()
                    out = net(Xt[idx])
                    loss = loss_fn(out, yt[idx]) if is_clf else loss_fn(out.squeeze(-1), yt[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                    opt.step()
                sched.step()
            net.eval()
            self._torch_net = net
        else:
            self._fallback = MLPModel(
                hidden_dims=(self.d_main, self.d_main),
                activation="gelu",
                epochs=self.epochs,
                learning_rate=self.learning_rate,
                task_type=self.task_type,
                random_seed=self.random_seed,
                track_carbon=False,
            )
            self._fallback.fit(X_s, y)

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        _ = kwargs
        X_s = self._scaler.transform(X).astype(np.float32)
        if self._torch_net is not None:
            import torch

            self._torch_net.eval()
            with torch.no_grad():
                logits = self._torch_net(torch.tensor(X_s, dtype=torch.float32))
                return np.asarray(torch.softmax(logits, dim=-1).cpu().numpy(), dtype=float)
        assert self._fallback is not None
        return self._fallback.predict_proba(X_s)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        _ = kwargs
        X_s = self._scaler.transform(X).astype(np.float32)
        if self._torch_net is not None:
            import torch

            self._torch_net.eval()
            with torch.no_grad():
                out = self._torch_net(torch.tensor(X_s, dtype=torch.float32))
                if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
                    idx = torch.argmax(out, dim=-1).cpu().numpy()
                    return np.array([self._idx_to_label[int(i)] for i in idx])
                return np.asarray(out.squeeze(-1).cpu().numpy(), dtype=float)
        assert self._fallback is not None
        return self._fallback.predict(X_s)
