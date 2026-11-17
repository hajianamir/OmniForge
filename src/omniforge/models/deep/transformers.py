"""FT-Transformer style Multi-Head Self-Attention architecture for tabular and sequence data."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from omniforge.core.types import TaskType
from omniforge.core.utils import is_package_available
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.deep.mlp import MLPModel


def _layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


class TabularTransformerModel(BaseOmniForgeModel):
    """Feature Tokenizer + Multi-Head Self-Attention Transformer for tabular & time-series tasks."""

    model_family = "deep_transformer"
    framework = "pytorch_or_numpy"

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        epochs: int = 50,
        learning_rate: float = 1e-2,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="TabularTransformer",
            task_type=task_type,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            epochs=epochs,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.d_model = int(d_model)
        self.n_heads = max(1, int(n_heads))
        if self.d_model % self.n_heads != 0:
            self.d_model = ((self.d_model // self.n_heads) + 1) * self.n_heads
        self.n_layers = max(1, int(n_layers))
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self._scaler = StandardScaler()
        self._token_w: np.ndarray | None = None
        self._token_b: np.ndarray | None = None
        self._blocks: list[dict[str, np.ndarray]] = []
        self._head: MLPModel | None = None
        self._torch_net: Any = None
        self._idx_to_label: dict[int, Any] = {}
        self.last_attention_weights_: np.ndarray | None = None

    def _init_numpy_transformer(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_seed)
        d = self.d_model
        scale = 1.0 / np.sqrt(d)
        self._token_w = rng.normal(0.0, scale, size=(n_features, d))
        self._token_b = rng.normal(0.0, 0.02, size=(n_features, d))
        self._blocks = []
        for _ in range(self.n_layers):
            self._blocks.append(
                {
                    "W_q": rng.normal(0.0, scale, size=(d, d)),
                    "W_k": rng.normal(0.0, scale, size=(d, d)),
                    "W_v": rng.normal(0.0, scale, size=(d, d)),
                    "W_o": rng.normal(0.0, scale, size=(d, d)),
                    "W_ff1": rng.normal(0.0, scale, size=(d, d * 2)),
                    "W_ff2": rng.normal(0.0, scale, size=(d * 2, d)),
                }
            )

    def _encode_numpy(self, X: np.ndarray) -> np.ndarray:
        """Feature tokenization + Multi-Head Scaled Dot-Product Self-Attention + FFN."""
        assert self._token_w is not None and self._token_b is not None
        # Tokenize each scalar feature into a d_model vector: (N, F, D)
        tokens = X[:, :, None] * self._token_w[None, :, :] + self._token_b[None, :, :]
        n, f, d = tokens.shape
        head_dim = d // self.n_heads

        h = tokens
        attn_weights = None
        for blk in self._blocks:
            q = (h @ blk["W_q"]).reshape(n, f, self.n_heads, head_dim).transpose(0, 2, 1, 3)
            k = (h @ blk["W_k"]).reshape(n, f, self.n_heads, head_dim).transpose(0, 2, 1, 3)
            v = (h @ blk["W_v"]).reshape(n, f, self.n_heads, head_dim).transpose(0, 2, 1, 3)
            scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
            scores -= np.max(scores, axis=-1, keepdims=True)
            attn = np.exp(scores)
            attn /= np.maximum(np.sum(attn, axis=-1, keepdims=True), 1e-12)
            attn_weights = attn.mean(axis=1)  # (N, F, F)
            ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(n, f, d)
            h = _layer_norm(h + ctx @ blk["W_o"])
            ff = np.maximum(0.0, h @ blk["W_ff1"]) @ blk["W_ff2"]
            h = _layer_norm(h + ff)

        self.last_attention_weights_ = attn_weights
        # Concatenate attention-pooled representation with original scaled features (residual skip)
        return np.hstack([h.mean(axis=1), X])

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        X_scaled = self._scaler.fit_transform(X)
        is_clf = self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

        if is_package_available("torch"):
            self._fit_torch(X_scaled, y, is_clf)
            return

        self.framework = "uap_numpy_transformer"
        self.metadata_.framework = self.framework
        self._init_numpy_transformer(X_scaled.shape[1])
        encoded = self._encode_numpy(X_scaled)
        self._head = MLPModel(
            hidden_dims=(self.d_model,),
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            task_type=self.task_type,
            random_seed=self.random_seed,
            track_carbon=False,
        )
        self._head.fit(encoded, y)
        self.loss_history_: list[float] = list(self._head.loss_history_)

    def _fit_torch(self, X: np.ndarray, y: np.ndarray, is_clf: bool) -> None:
        import torch  # type: ignore[import-not-found]
        import torch.nn as nn  # type: ignore[import-not-found]
        import torch.nn.functional as F  # type: ignore[import-not-found]

        self.framework = "pytorch_ft_transformer"
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

        d_mod, heads, layers_n = self.d_model, self.n_heads, self.n_layers
        n_feats = X.shape[1]

        class _PreNormGEGLUBlock(nn.Module):
            """Gorishniy et al. (NeurIPS 2021) Pre-Norm MHSA + GEGLU FFN Transformer Block."""

            def __init__(self, first_layer: bool = False) -> None:
                super().__init__()
                self.first_layer = first_layer
                self.norm1 = nn.Identity() if first_layer else nn.LayerNorm(d_mod)
                self.attn = nn.MultiheadAttention(d_mod, heads, dropout=0.1, batch_first=True)
                self.norm2 = nn.LayerNorm(d_mod)
                self.ffn_in = nn.Linear(d_mod, d_mod * 4)
                self.ffn_drop = nn.Dropout(0.1)
                self.ffn_out = nn.Linear(d_mod * 2, d_mod)
                self.last_attn_weights: Any = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.norm1(x)
                attn_out, attn_w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
                self.last_attn_weights = attn_w
                x = x + attn_out
                u, v = self.ffn_in(self.norm2(x)).chunk(2, dim=-1)
                return x + self.ffn_out(self.ffn_drop(u * F.gelu(v)))

        class _FTTransformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feat_w = nn.Parameter(torch.randn(n_feats, d_mod) * (1.0 / np.sqrt(d_mod)))
                self.feat_b = nn.Parameter(torch.zeros(n_feats, d_mod))
                self.cls_token = nn.Parameter(torch.randn(1, 1, d_mod) * 0.02)
                self.blocks = nn.ModuleList([_PreNormGEGLUBlock(first_layer=(i == 0)) for i in range(layers_n)])
                self.head_norm = nn.LayerNorm(d_mod)
                self.head = nn.Linear(d_mod, out_dim)

            def forward(self, inp: Any) -> Any:
                tok = inp.unsqueeze(-1) * self.feat_w.unsqueeze(0) + self.feat_b.unsqueeze(0)
                cls = self.cls_token.expand(inp.size(0), -1, -1)
                seq = torch.cat([cls, tok], dim=1)
                for blk in self.blocks:
                    seq = blk(seq)
                cls_out = F.relu(self.head_norm(seq[:, 0, :]))
                return self.head(cls_out)

        net = _FTTransformer()
        opt = torch.optim.AdamW(net.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(self.epochs, 1))
        loss_fn: Any = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()
        X_t = torch.tensor(X.astype(np.float32))
        n = len(X_t)
        bs = min(128, max(16, n))

        net.train()
        self.loss_history_ = []
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                opt.zero_grad()
                loss = loss_fn(net(X_t[idx]), y_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
                ep_loss += float(loss.item())
                n_batches += 1
            sched.step()
            self.loss_history_.append(round(ep_loss / max(n_batches, 1), 6))

        net.eval()
        with torch.no_grad():
            _ = net(X_t[: min(n, 256)])
            # Extract [CLS]-to-feature attention importance from final Transformer layer
            w_last = net.blocks[-1].last_attn_weights
            if w_last is not None:
                cls_to_feat = w_last[:, 0, 1:].mean(dim=0).cpu().numpy()
                s_val = float(np.sum(cls_to_feat))
                self.feature_importances_ = (cls_to_feat / s_val) if s_val > 1e-12 else np.ones(n_feats) / max(n_feats, 1)
                self.last_attention_weights_ = w_last.cpu().numpy()
        self._torch_net = net

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        if self._torch_net is not None:
            import torch  # type: ignore[import-not-found]

            with torch.no_grad():
                out = self._torch_net(torch.tensor(X_scaled.astype(np.float32))).numpy()
            if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
                return np.array([self._idx_to_label[int(i)] for i in np.argmax(out, axis=1)])
            return out.ravel()
        assert self._head is not None
        return self._head.predict(self._encode_numpy(X_scaled))

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        if self._torch_net is not None:
            import torch  # type: ignore[import-not-found]

            with torch.no_grad():
                logits = self._torch_net(torch.tensor(X_scaled.astype(np.float32)))
                return torch.softmax(logits, dim=-1).numpy().astype(float)
        assert self._head is not None
        return self._head.predict_proba(self._encode_numpy(X_scaled))


FTTransformerModel = TabularTransformerModel

