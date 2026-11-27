"""FT-Transformer style Multi-Head Self-Attention architecture for tabular and sequence data."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from omniforge.core.types import TaskType
from omniforge.core.utils import is_package_available
from omniforge.models._torch_state import TorchStateMixin
from omniforge.models.base import BaseOmniForgeModel


def _layer_norm_with_cache(x: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=-1, keepdims=True)
    xc = x - mean
    inv_std = 1.0 / np.sqrt(np.mean(xc * xc, axis=-1, keepdims=True) + eps)
    x_hat = xc * inv_std
    return np.asarray(x_hat, dtype=np.float64), x_hat, inv_std


def _layer_norm_backward(dy: np.ndarray, x_hat: np.ndarray, inv_std: np.ndarray) -> np.ndarray:
    return np.asarray(
        inv_std * (dy - np.mean(dy, axis=-1, keepdims=True) - x_hat * np.mean(dy * x_hat, axis=-1, keepdims=True)),
        dtype=np.float64,
    )


def _softmax_last(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return np.asarray(exps / np.maximum(np.sum(exps, axis=-1, keepdims=True), 1e-12), dtype=np.float64)


class TabularTransformerModel(TorchStateMixin, BaseOmniForgeModel):
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
        weight_decay: float = 1e-4,
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
            weight_decay=weight_decay,
            **kwargs,
        )
        self.d_model = int(d_model)
        self.n_heads = max(1, int(n_heads))
        if self.d_model % self.n_heads != 0:
            self.d_model = ((self.d_model // self.n_heads) + 1) * self.n_heads
        self.n_layers = max(1, int(n_layers))
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.weight_decay = max(0.0, float(weight_decay))
        self._scaler = StandardScaler()
        self._np_params: dict[str, np.ndarray] = {}
        self._torch_net: Any = None
        self._idx_to_label: dict[int, Any] = {}
        self.last_attention_weights_: np.ndarray | None = None

    def _init_numpy_transformer(self, n_features: int, out_dim: int) -> None:
        rng = np.random.default_rng(self.random_seed)
        d = self.d_model
        scale = 1.0 / np.sqrt(max(d, 1))
        params: dict[str, np.ndarray] = {
            "token_w": rng.normal(0.0, scale, size=(n_features, d)).astype(np.float64),
            "token_b": np.zeros((n_features, d), dtype=np.float64),
            "cls_token": rng.normal(0.0, 0.02, size=(1, 1, d)).astype(np.float64),
            "W_head": rng.normal(0.0, 1.0 / np.sqrt(d + n_features), size=(d + n_features, out_dim)).astype(np.float64),
            "b_head": np.zeros((1, out_dim), dtype=np.float64),
        }
        for li in range(self.n_layers):
            params[f"W_q_{li}"] = rng.normal(0.0, scale, size=(d, d)).astype(np.float64)
            params[f"W_k_{li}"] = rng.normal(0.0, scale, size=(d, d)).astype(np.float64)
            params[f"W_v_{li}"] = rng.normal(0.0, scale, size=(d, d)).astype(np.float64)
            params[f"W_o_{li}"] = rng.normal(0.0, scale, size=(d, d)).astype(np.float64)
            params[f"W_ff1_{li}"] = rng.normal(0.0, np.sqrt(2.0 / d), size=(d, d * 2)).astype(np.float64)
            params[f"b_ff1_{li}"] = np.zeros((1, 1, d * 2), dtype=np.float64)
            params[f"W_ff2_{li}"] = rng.normal(0.0, scale, size=(d * 2, d)).astype(np.float64)
            params[f"b_ff2_{li}"] = np.zeros((1, 1, d), dtype=np.float64)
        self._np_params = params

    def _forward_numpy_with_cache(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Forward pass of Pre-Norm [CLS] FT-Transformer returning logits and exact backprop cache."""
        p = self._np_params
        xb = np.asarray(X, dtype=np.float64)
        b_sz, n_f = xb.shape
        d = self.d_model
        heads = self.n_heads
        head_dim = d // heads
        attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

        feat_tokens = xb[:, :, None] * p["token_w"][None, :, :] + p["token_b"][None, :, :]
        cls_tok = np.broadcast_to(p["cls_token"], (b_sz, 1, d))
        h = np.concatenate([cls_tok, feat_tokens], axis=1)  # (B, L=F+1, D)
        seq_len = h.shape[1]

        layer_caches: list[dict[str, np.ndarray]] = []
        last_attn = None
        for li in range(self.n_layers):
            h_norm, h_hat, h_inv = _layer_norm_with_cache(h)
            q = (h_norm @ p[f"W_q_{li}"]).reshape(b_sz, seq_len, heads, head_dim).transpose(0, 2, 1, 3)
            k = (h_norm @ p[f"W_k_{li}"]).reshape(b_sz, seq_len, heads, head_dim).transpose(0, 2, 1, 3)
            v = (h_norm @ p[f"W_v_{li}"]).reshape(b_sz, seq_len, heads, head_dim).transpose(0, 2, 1, 3)

            scores = (q @ k.transpose(0, 1, 3, 2)) * attn_scale
            attn = _softmax_last(scores)
            last_attn = attn.mean(axis=1)  # (B, L, L)
            av = attn @ v
            ctx = av.transpose(0, 2, 1, 3).reshape(b_sz, seq_len, d)
            h_mid = h + ctx @ p[f"W_o_{li}"]

            m_norm, m_hat, m_inv = _layer_norm_with_cache(h_mid)
            ff_pre = m_norm @ p[f"W_ff1_{li}"] + p[f"b_ff1_{li}"]
            ff_act = np.maximum(0.0, ff_pre)
            h_next = h_mid + ff_act @ p[f"W_ff2_{li}"] + p[f"b_ff2_{li}"]

            layer_caches.append(
                {
                    "h_norm": h_norm,
                    "h_hat": h_hat,
                    "h_inv": h_inv,
                    "q": q,
                    "k": k,
                    "v": v,
                    "attn": attn,
                    "ctx": ctx,
                    "m_norm": m_norm,
                    "m_hat": m_hat,
                    "m_inv": m_inv,
                    "ff_pre": ff_pre,
                    "ff_act": ff_act,
                }
            )
            h = h_next

        cls_out, cls_hat, cls_inv = _layer_norm_with_cache(h[:, 0, :])
        feat_cat = np.hstack([cls_out, xb])
        logits = feat_cat @ p["W_head"] + p["b_head"]
        cache = {
            "xb": xb,
            "seq_len": seq_len,
            "layer_caches": layer_caches,
            "cls_hat": cls_hat,
            "cls_inv": cls_inv,
            "feat_cat": feat_cat,
            "last_attn": last_attn,
        }
        return logits, cache

    def _backward_numpy(self, d_logits: np.ndarray, cache: dict[str, Any]) -> dict[str, np.ndarray]:
        """Exact analytical backpropagation through head, Pre-Norm FFN, Multi-Head Attention, and Feature Tokenizer."""
        p = self._np_params
        xb: np.ndarray = cache["xb"]
        b_sz = xb.shape[0]
        seq_len = int(cache["seq_len"])
        d = self.d_model
        heads = self.n_heads
        head_dim = d // heads
        attn_scale = 1.0 / np.sqrt(max(head_dim, 1))

        grads: dict[str, np.ndarray] = {}
        feat_cat: np.ndarray = cache["feat_cat"]
        grads["W_head"] = feat_cat.T @ d_logits
        grads["b_head"] = np.sum(d_logits, axis=0, keepdims=True)

        d_feat_cat = d_logits @ p["W_head"].T
        d_cls_out = d_feat_cat[:, :d]
        dh = np.zeros((b_sz, seq_len, d), dtype=np.float64)
        dh[:, 0, :] = _layer_norm_backward(d_cls_out, cache["cls_hat"], cache["cls_inv"])

        layer_caches: list[dict[str, np.ndarray]] = cache["layer_caches"]
        for li in range(self.n_layers - 1, -1, -1):
            lc = layer_caches[li]
            # FFN residual backward
            d_h_mid = dh.copy()
            ff_act = lc["ff_act"]
            ff_pre = lc["ff_pre"]
            m_norm = lc["m_norm"]
            grads[f"W_ff2_{li}"] = ff_act.reshape(-1, d * 2).T @ dh.reshape(-1, d)
            grads[f"b_ff2_{li}"] = np.sum(dh, axis=(0, 1), keepdims=True)
            d_ff_act = dh @ p[f"W_ff2_{li}"].T
            d_ff_pre = d_ff_act * (ff_pre > 0.0)
            grads[f"W_ff1_{li}"] = m_norm.reshape(-1, d).T @ d_ff_pre.reshape(-1, d * 2)
            grads[f"b_ff1_{li}"] = np.sum(d_ff_pre, axis=(0, 1), keepdims=True)
            d_m_norm = d_ff_pre @ p[f"W_ff1_{li}"].T
            d_h_mid += _layer_norm_backward(d_m_norm, lc["m_hat"], lc["m_inv"])

            # Multi-Head Self-Attention residual backward
            dh_prev = d_h_mid.copy()
            ctx = lc["ctx"]
            grads[f"W_o_{li}"] = ctx.reshape(-1, d).T @ d_h_mid.reshape(-1, d)
            d_ctx = d_h_mid @ p[f"W_o_{li}"].T
            d_av = d_ctx.reshape(b_sz, seq_len, heads, head_dim).transpose(0, 2, 1, 3)

            attn = lc["attn"]
            q, k, v = lc["q"], lc["k"], lc["v"]
            d_attn = d_av @ v.transpose(0, 1, 3, 2)
            d_v = attn.transpose(0, 1, 3, 2) @ d_av
            d_scores = attn * (d_attn - np.sum(d_attn * attn, axis=-1, keepdims=True)) * attn_scale
            d_q = d_scores @ k
            d_k = d_scores.transpose(0, 1, 3, 2) @ q

            dq_flat = d_q.transpose(0, 2, 1, 3).reshape(b_sz, seq_len, d)
            dk_flat = d_k.transpose(0, 2, 1, 3).reshape(b_sz, seq_len, d)
            dv_flat = d_v.transpose(0, 2, 1, 3).reshape(b_sz, seq_len, d)

            h_norm = lc["h_norm"]
            h_flat = h_norm.reshape(-1, d)
            grads[f"W_q_{li}"] = h_flat.T @ dq_flat.reshape(-1, d)
            grads[f"W_k_{li}"] = h_flat.T @ dk_flat.reshape(-1, d)
            grads[f"W_v_{li}"] = h_flat.T @ dv_flat.reshape(-1, d)

            d_h_norm = dq_flat @ p[f"W_q_{li}"].T + dk_flat @ p[f"W_k_{li}"].T + dv_flat @ p[f"W_v_{li}"].T
            dh = dh_prev + _layer_norm_backward(d_h_norm, lc["h_hat"], lc["h_inv"])

        grads["cls_token"] = np.sum(dh[:, 0:1, :], axis=0, keepdims=True)
        d_feat_tok = dh[:, 1:, :]
        grads["token_w"] = np.sum(d_feat_tok * xb[:, :, None], axis=0)
        grads["token_b"] = np.sum(d_feat_tok, axis=0)
        return grads

    def _fit_numpy(self, X_scaled: np.ndarray, y: np.ndarray, is_clf: bool) -> None:
        self.framework = "omniforge_numpy_transformer"
        self.metadata_.framework = self.framework
        n, n_feats = X_scaled.shape
        if is_clf:
            classes = np.unique(y)
            l2i = {c: i for i, c in enumerate(classes)}
            self._idx_to_label = {i: c for i, c in enumerate(classes)}
            y_idx = np.array([l2i[v] for v in y], dtype=int)
            out_dim = max(2, len(classes))
        else:
            y_idx = np.empty(0, dtype=int)
            out_dim = 1

        self._init_numpy_transformer(n_feats, out_dim)
        m_state = {k: np.zeros_like(v) for k, v in self._np_params.items()}
        v_state = {k: np.zeros_like(v) for k, v in self._np_params.items()}
        rng = np.random.default_rng(self.random_seed)
        bs = min(128, max(16, n))
        step = 0
        self.loss_history_ = []
        y_float = y.astype(np.float64).reshape(-1, 1) if not is_clf else np.empty((0, 1))

        for _ in range(self.epochs):
            perm = rng.permutation(n)
            ep_loss = 0.0
            n_batches = 0
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                xb = X_scaled[idx]
                cur_bs = len(xb)
                logits, cache = self._forward_numpy_with_cache(xb)
                if is_clf:
                    probs = _softmax_last(logits)
                    yb = y_idx[idx]
                    ep_loss += float(-np.mean(np.log(np.clip(probs[np.arange(cur_bs), yb], 1e-12, 1.0))))
                    d_logits = probs.copy()
                    d_logits[np.arange(cur_bs), yb] -= 1.0
                    d_logits /= float(cur_bs)
                else:
                    yb_f = y_float[idx]
                    diff = logits - yb_f
                    ep_loss += float(np.mean(diff**2))
                    d_logits = (2.0 * diff) / float(cur_bs)

                grads = self._backward_numpy(d_logits, cache)
                # Global gradient norm clipping (max_norm = 2.0) matching PyTorch trainer
                sq_norm = sum(float(np.sum(g * g)) for g in grads.values())
                tot_norm = float(np.sqrt(max(sq_norm, 0.0)))
                if tot_norm > 2.0:
                    clip_coef = 2.0 / (tot_norm + 1e-12)
                    for k_name in grads:
                        grads[k_name] *= clip_coef

                step += 1
                b1_corr = 1.0 - (0.9**step)
                b2_corr = 1.0 - (0.999**step)
                for k_name, param in self._np_params.items():
                    g = grads[k_name]
                    m_state[k_name] = 0.9 * m_state[k_name] + 0.1 * g
                    v_state[k_name] = 0.999 * v_state[k_name] + 0.001 * (g * g)
                    m_hat = m_state[k_name] / b1_corr
                    v_hat = v_state[k_name] / b2_corr
                    # Decoupled AdamW weight decay on projection matrices (not biases / cls_token)
                    if self.weight_decay > 0.0 and k_name.startswith(("W_", "token_w")):
                        param *= 1.0 - self.learning_rate * self.weight_decay
                    param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
                n_batches += 1

            self.loss_history_.append(round(ep_loss / max(n_batches, 1), 6))

        _, eval_cache = self._forward_numpy_with_cache(X_scaled[: min(n, 256)])
        last_attn = eval_cache["last_attn"]
        if last_attn is not None:
            self.last_attention_weights_ = last_attn
            cls_to_feat = np.mean(last_attn[:, 0, 1:], axis=0)
            s_val = float(np.sum(cls_to_feat))
            self.feature_importances_ = (cls_to_feat / s_val) if s_val > 1e-12 else np.ones(n_feats) / max(n_feats, 1)

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        X_scaled = self._scaler.fit_transform(X)
        is_clf = self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

        if is_package_available("torch"):
            self._fit_torch(X_scaled, y, is_clf)
            return
        self._fit_numpy(X_scaled, y, is_clf)

    def _build_torch_net(self, n_feats: int, out_dim: int) -> Any:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        d_mod, heads, layers_n = self.d_model, self.n_heads, self.n_layers

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
                out: torch.Tensor = x + self.ffn_out(self.ffn_drop(u * F.gelu(v)))
                return out

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
        return _FTTransformer()

    def _fit_torch(self, X: np.ndarray, y: np.ndarray, is_clf: bool) -> None:
        import torch
        import torch.nn as nn

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

        n_feats = X.shape[1]
        self._torch_build_config = {"n_feats": n_feats, "out_dim": out_dim}
        net = self._build_torch_net(**self._torch_build_config)
        opt = torch.optim.AdamW(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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
            last_blk: Any = net.blocks[-1]
            w_last: Any = last_blk.last_attn_weights
            if w_last is not None:
                cls_to_feat = w_last[:, 0, 1:].mean(dim=0).cpu().numpy()
                s_val = float(np.sum(cls_to_feat))
                self.feature_importances_ = (cls_to_feat / s_val) if s_val > 1e-12 else np.ones(n_feats) / max(n_feats, 1)
                self.last_attention_weights_ = w_last.cpu().numpy()
        self._torch_net = net

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        if self._torch_net is not None:
            import torch

            with torch.no_grad():
                out = self._torch_net(torch.tensor(X_scaled.astype(np.float32))).numpy()
            if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
                return np.array([self._idx_to_label[int(i)] for i in np.argmax(out, axis=1)])
            return np.asarray(out.ravel(), dtype=float)
        logits, cache = self._forward_numpy_with_cache(X_scaled)
        if cache["last_attn"] is not None:
            self.last_attention_weights_ = cache["last_attn"]
        if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            return np.array([self._idx_to_label[int(i)] for i in np.argmax(logits, axis=1)])
        return np.asarray(logits.ravel(), dtype=float)

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        if self._torch_net is not None:
            import torch

            with torch.no_grad():
                logits = self._torch_net(torch.tensor(X_scaled.astype(np.float32)))
                return np.asarray(torch.softmax(logits, dim=-1).numpy(), dtype=float)
        logits_np, cache = self._forward_numpy_with_cache(X_scaled)
        if cache["last_attn"] is not None:
            self.last_attention_weights_ = cache["last_attn"]
        return _softmax_last(logits_np)


FTTransformerModel = TabularTransformerModel

