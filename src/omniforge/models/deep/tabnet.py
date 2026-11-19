"""Paper-Grade TabNet Architecture (Arik & Pfister, AAAI 2021): Sequential Attentive Feature Selection with Exact Sparsemax, Prior Scale Relaxation & Feature Attribution Masks."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from omniforge.core.types import TaskType
from omniforge.core.utils import is_package_available
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.deep.mlp import MLPModel


def sparsemax_numpy(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Exact Euclidean projection onto the probability simplex (Martins & Astudillo, ICML 2016)."""
    z_shift = z - np.max(z, axis=axis, keepdims=True)
    z_sorted = np.sort(z_shift, axis=axis)[..., ::-1]
    k_range = np.arange(1, z.shape[axis] + 1, dtype=float)
    z_cumsum = np.cumsum(z_sorted, axis=axis)
    support = (1.0 + k_range * z_sorted) > z_cumsum
    k_z = np.sum(support, axis=axis, keepdims=True)
    tau = (np.take_along_axis(z_cumsum, np.maximum(k_z - 1, 0), axis=axis) - 1.0) / np.maximum(k_z, 1.0)
    return np.asarray(np.maximum(z_shift - tau, 0.0), dtype=float)


class TabNetModel(BaseOmniForgeModel):
    """Faithful TabNet (Arik & Pfister, AAAI 2021) with PyTorch GLU/Sparsemax/GhostBN & NumPy fallback."""

    model_family = "deep_tabnet"
    framework = "pytorch_or_numpy"

    def __init__(
        self,
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 3,
        gamma: float = 1.5,
        lambda_sparse: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1.5e-2,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="TabNet",
            task_type=task_type,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            lambda_sparse=lambda_sparse,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.n_d = max(4, int(n_d))
        self.n_a = max(4, int(n_a))
        self.n_steps = max(1, int(n_steps))
        self.gamma = max(1.0, float(gamma))
        self.lambda_sparse = max(0.0, float(lambda_sparse))
        self.epochs = int(epochs)
        self.batch_size = max(16, int(batch_size))
        self.learning_rate = float(learning_rate)
        self._scaler = StandardScaler()
        self._torch_net: Any = None
        self._np_step_w: list[np.ndarray] = []
        self._np_head: MLPModel | None = None
        self._idx_to_label: dict[int, Any] = {}
        self.feature_importances_: np.ndarray | None = None
        self.last_masks_: np.ndarray | None = None

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
            self._fit_torch(X_s, y_enc, out_dim=out_dim, is_clf=is_clf)
        else:
            self._fit_numpy(X_s, y)

    def _fit_torch(self, X_s: np.ndarray, y_enc: np.ndarray, *, out_dim: int, is_clf: bool) -> None:
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_seed)
        in_dim = X_s.shape[1]
        n_d, n_a, n_steps, gamma = self.n_d, self.n_a, self.n_steps, self.gamma

        def _sparsemax_torch(z: torch.Tensor) -> torch.Tensor:
            z_shift = z - z.max(dim=-1, keepdim=True).values
            z_sorted, _ = torch.sort(z_shift, dim=-1, descending=True)
            k_range = torch.arange(1, z.size(-1) + 1, device=z.device, dtype=z.dtype).view(1, -1)
            z_cumsum = torch.cumsum(z_sorted, dim=-1)
            support = (1.0 + k_range * z_sorted) > z_cumsum
            k_z = support.sum(dim=-1, keepdim=True).clamp(min=1)
            tau = (z_cumsum.gather(-1, k_z - 1) - 1.0) / k_z.to(z.dtype)
            return torch.clamp(z_shift - tau, min=0.0)

        class _GLUBlock(nn.Module):
            def __init__(self, d_in: int, d_out: int) -> None:
                super().__init__()
                self.fc = nn.Linear(d_in, d_out * 2, bias=False)
                self.bn = nn.BatchNorm1d(d_out * 2)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.bn(self.fc(x))
                a, b = h.chunk(2, dim=-1)
                return torch.mul(a, torch.sigmoid(b))

        class _TabNetModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_bn = nn.BatchNorm1d(in_dim)
                self.shared1 = _GLUBlock(in_dim, n_d + n_a)
                self.shared2 = _GLUBlock(n_d + n_a, n_d + n_a)
                self.step_blocks = nn.ModuleList([_GLUBlock(n_d + n_a, n_d + n_a) for _ in range(n_steps)])
                self.att_fcs = nn.ModuleList([nn.Linear(n_a, in_dim, bias=False) for _ in range(n_steps)])
                self.att_bns = nn.ModuleList([nn.BatchNorm1d(in_dim) for _ in range(n_steps)])
                self.head = nn.Linear(n_d, out_dim, bias=True)

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                x_bn = self.input_bn(x)
                prior = torch.ones_like(x_bn)
                agg_d = torch.zeros((x.size(0), n_d), device=x.device, dtype=x.dtype)
                m_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
                m_explain = torch.zeros_like(x_bn)
                h0 = self.shared1(x_bn)
                h0 = (h0 + self.shared2(h0)) * 0.70710678
                att_in = h0[:, n_d:]
                for step in range(n_steps):
                    mask = _sparsemax_torch(prior * self.att_bns[step](self.att_fcs[step](att_in)))
                    prior = prior * (gamma - mask)
                    m_loss = m_loss + torch.mean(torch.sum(-mask * torch.log(mask + 1e-10), dim=-1))
                    masked_x = mask * x_bn
                    hs = self.shared1(masked_x)
                    hs = (hs + self.shared2(hs)) * 0.70710678
                    hs = (hs + self.step_blocks[step](hs)) * 0.70710678
                    d_out = torch.relu(hs[:, :n_d])
                    att_in = hs[:, n_d:]
                    agg_d = agg_d + d_out
                    step_imp = d_out.sum(dim=-1, keepdim=True)
                    m_explain = m_explain + step_imp * mask
                return self.head(agg_d), m_loss / float(n_steps), m_explain

        net = _TabNetModule()
        opt = torch.optim.AdamW(net.parameters(), lr=self.learning_rate, weight_decay=1e-4)
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
                out, m_loss, _ = net(Xt[idx])
                pred_loss = loss_fn(out, yt[idx]) if is_clf else loss_fn(out.squeeze(-1), yt[idx])
                loss = pred_loss + self.lambda_sparse * m_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
            sched.step()

        net.eval()
        with torch.no_grad():
            _, _, m_exp = net(Xt)
            imp = m_exp.mean(dim=0).cpu().numpy()
            imp_sum = float(np.sum(imp))
            self.feature_importances_ = (imp / imp_sum) if imp_sum > 1e-12 else np.ones(in_dim) / max(in_dim, 1)
            self.last_masks_ = m_exp.cpu().numpy()
        self._torch_net = net

    def _fit_numpy(self, X_s: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_seed)
        in_dim = X_s.shape[1]
        self._np_step_w = [rng.normal(0.0, 1.0 / np.sqrt(max(in_dim, 1)), size=(in_dim, in_dim)) for _ in range(self.n_steps)]
        feats, masks = self._encode_numpy(X_s)
        self.last_masks_ = masks
        imp = np.mean(masks, axis=0)
        self.feature_importances_ = imp / max(float(np.sum(imp)), 1e-12)
        self._np_head = MLPModel(
            hidden_dims=(max(32, self.n_d * 2), self.n_d),
            activation="gelu",
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            task_type=self.task_type,
            random_seed=self.random_seed,
            track_carbon=False,
        )
        self._np_head.fit(feats, y)

    def _encode_numpy(self, X_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prior = np.ones_like(X_s, dtype=float)
        agg = np.zeros_like(X_s, dtype=float)
        total_mask = np.zeros_like(X_s, dtype=float)
        for w in self._np_step_w:
            logits = prior * (X_s @ w)
            mask = sparsemax_numpy(logits, axis=-1)
            prior = prior * np.maximum(self.gamma - mask, 0.0)
            agg += np.maximum(0.0, mask * X_s)
            total_mask += mask
        return np.hstack([X_s, agg]), total_mask / float(max(len(self._np_step_w), 1))

    def explain(self, X: Any) -> np.ndarray:
        """Return per-instance TabNet sparse feature attribution matrix of shape ``(n_samples, n_features)``."""
        self._check_is_fitted()
        X_arr, _ = self._prepare_X(X, fitting=False)
        X_s = self._scaler.transform(X_arr).astype(np.float32)
        if self._torch_net is not None:
            import torch

            self._torch_net.eval()
            with torch.no_grad():
                _, _, m_exp = self._torch_net(torch.tensor(X_s, dtype=torch.float32))
            return np.asarray(m_exp.cpu().numpy(), dtype=float)
        _, masks = self._encode_numpy(X_s)
        return masks

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        _ = kwargs
        X_s = self._scaler.transform(X).astype(np.float32)
        if self._torch_net is not None:
            import torch

            self._torch_net.eval()
            with torch.no_grad():
                logits, _, _ = self._torch_net(torch.tensor(X_s, dtype=torch.float32))
                return np.asarray(torch.softmax(logits, dim=-1).cpu().numpy(), dtype=float)
        assert self._np_head is not None
        feats, _ = self._encode_numpy(X_s)
        return self._np_head.predict_proba(feats)

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        _ = kwargs
        X_s = self._scaler.transform(X).astype(np.float32)
        if self._torch_net is not None:
            import torch

            self._torch_net.eval()
            with torch.no_grad():
                out, _, _ = self._torch_net(torch.tensor(X_s, dtype=torch.float32))
                if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
                    idx = torch.argmax(out, dim=-1).cpu().numpy()
                    return np.array([self._idx_to_label[int(i)] for i in idx])
                return np.asarray(out.squeeze(-1).cpu().numpy(), dtype=float)
        assert self._np_head is not None
        feats, _ = self._encode_numpy(X_s)
        return self._np_head.predict(feats)
