"""Paper-Grade Graph Neural Networks (GCN, GraphSAGE, Multi-Head GAT) & Graph Autoencoder (GAE / VGAE) for Node Classification & Neural Link Prediction."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from omniforge.core.types import TaskType
from omniforge.core.utils import is_package_available
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.deep.mlp import MLPModel


class GraphNeuralNetworkModel(BaseOmniForgeModel):
    """End-to-end GCN (Kipf & Welling 2017), GraphSAGE (Hamilton 2017) & Multi-Head GAT (Velickovic 2018) with PyTorch & NumPy fallback."""

    model_family = "deep_gnn"
    framework = "pytorch_or_numpy_gnn"

    def __init__(
        self,
        architecture: Literal["gcn", "graphsage", "gat"] = "gcn",
        hidden_dim: int = 32,
        n_hops: int = 2,
        n_heads: int = 4,
        knn_k: int = 5,
        epochs: int = 60,
        learning_rate: float = 1e-2,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=f"DeepGNN_{architecture.upper()}",
            task_type=task_type,
            architecture=architecture,
            hidden_dim=hidden_dim,
            n_hops=n_hops,
            n_heads=n_heads,
            knn_k=knn_k,
            epochs=epochs,
            learning_rate=learning_rate,
            **kwargs,
        )
        self.architecture = str(architecture).lower()
        self.hidden_dim = max(8, int(hidden_dim))
        self.n_hops = max(1, int(n_hops))
        self.n_heads = max(1, int(n_heads))
        self.knn_k = max(2, int(knn_k))
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self._scaler = StandardScaler()
        self._proj_weights: list[np.ndarray] = []
        self._head: MLPModel | None = None
        self._torch_net: Any = None
        self._idx_to_label: dict[int, Any] = {}

    @staticmethod
    def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
        """Compute symmetric normalized adjacency D^{-1/2} (A + I) D^{-1/2}."""
        n = adj.shape[0]
        a_tilde = adj.astype(np.float64) + np.eye(n, dtype=np.float64)
        deg = np.sum(a_tilde, axis=1)
        deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
        return (deg_inv_sqrt[:, None] * a_tilde) * deg_inv_sqrt[None, :]

    @staticmethod
    def row_normalize_adjacency(adj: np.ndarray) -> np.ndarray:
        """Compute row-normalized mean-aggregator adjacency D^{-1} A for GraphSAGE."""
        deg = np.sum(adj, axis=1, keepdims=True)
        return np.divide(adj.astype(np.float64), np.where(deg > 0, deg, 1.0))

    def _build_knn_adjacency(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if n <= 1:
            return np.ones((n, n), dtype=np.float64)
        k = min(self.knn_k, n)
        nn = NearestNeighbors(n_neighbors=k).fit(X)
        adj = nn.kneighbors_graph(X, mode="connectivity").toarray()
        return np.maximum(adj, adj.T)

    def _propagate_numpy(self, X: np.ndarray, adjacency: np.ndarray | None = None) -> np.ndarray:
        adj = adjacency if adjacency is not None else self._build_knn_adjacency(X)
        a_norm = self.row_normalize_adjacency(adj) if self.architecture == "graphsage" else self.normalize_adjacency(adj)
        h = X
        layer_outputs = [X]
        for w in self._proj_weights:
            if self.architecture == "graphsage":
                neigh = a_norm @ h
                h = np.maximum(0.0, np.hstack([h, neigh]) @ w)
            else:
                h = np.maximum(0.0, a_norm @ h @ w)
            layer_outputs.append(h)
        return np.hstack(layer_outputs)

    def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: Any) -> None:
        assert y is not None
        X_scaled = self._scaler.fit_transform(X)
        adjacency = kwargs.get("adjacency")
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

        if is_package_available("torch") and len(X_scaled) <= 4000:
            self._fit_torch(X_scaled, y_enc, adjacency=adjacency, out_dim=out_dim, is_clf=is_clf)
            return

        rng = np.random.default_rng(self.random_seed)
        self._proj_weights = []
        in_d = X_scaled.shape[1]
        for _ in range(self.n_hops):
            eff_in = in_d * 2 if self.architecture == "graphsage" else in_d
            w = rng.normal(0.0, np.sqrt(2.0 / max(eff_in, 1)), size=(eff_in, self.hidden_dim))
            self._proj_weights.append(w)
            in_d = self.hidden_dim

        emb = self._propagate_numpy(X_scaled, adjacency=adjacency)
        self._head = MLPModel(
            hidden_dims=(self.hidden_dim,),
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            task_type=self.task_type,
            random_seed=self.random_seed,
            track_carbon=False,
        )
        self._head.fit(emb, y)

    def _fit_torch(
        self,
        X_s: np.ndarray,
        y_enc: np.ndarray,
        *,
        adjacency: np.ndarray | None,
        out_dim: int,
        is_clf: bool,
    ) -> None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        torch.manual_seed(self.random_seed)
        adj = adjacency if adjacency is not None else self._build_knn_adjacency(X_s)
        in_dim = X_s.shape[1]
        h_dim = self.hidden_dim
        arch = self.architecture
        n_heads = self.n_heads

        class _GATLayer(nn.Module):
            """Velickovic et al. (ICLR 2018) Multi-Head Graph Attention Layer."""

            def __init__(self, d_in: int, d_out: int, heads: int) -> None:
                super().__init__()
                self.heads = heads
                self.d_h = max(4, d_out // heads)
                self.W = nn.Linear(d_in, self.heads * self.d_h, bias=False)
                self.a_src = nn.Parameter(torch.randn(self.heads, self.d_h) * 0.1)
                self.a_dst = nn.Parameter(torch.randn(self.heads, self.d_h) * 0.1)

            def forward(self, x: torch.Tensor, a_mask: torch.Tensor) -> torch.Tensor:
                n = x.size(0)
                h = self.W(x).view(n, self.heads, self.d_h)
                e_src = (h * self.a_src.unsqueeze(0)).sum(dim=-1)  # (N, H)
                e_dst = (h * self.a_dst.unsqueeze(0)).sum(dim=-1)  # (N, H)
                e = F.leaky_relu(e_src.unsqueeze(1) + e_dst.unsqueeze(0), 0.2)  # (N, N, H)
                mask = (a_mask > 0).unsqueeze(-1)
                e = e.masked_fill(~mask, -1e9)
                alpha = torch.softmax(e, dim=1)  # (N, N, H)
                out = torch.einsum("ijh,jhd->ihd", alpha, h).reshape(n, self.heads * self.d_h)
                return F.elu(out)

        class _GNNNet(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                if arch == "gat":
                    self.l1: Any = _GATLayer(in_dim, h_dim, n_heads)
                    d1 = self.l1.heads * self.l1.d_h
                    self.l2: Any = _GATLayer(d1, h_dim, 1)
                    d2 = self.l2.heads * self.l2.d_h
                elif arch == "graphsage":
                    self.l1 = nn.Linear(in_dim * 2, h_dim)
                    self.l2 = nn.Linear(h_dim * 2, h_dim)
                    d2 = h_dim
                else:
                    self.l1 = nn.Linear(in_dim, h_dim)
                    self.l2 = nn.Linear(h_dim, h_dim)
                    d2 = h_dim
                self.head = nn.Linear(d2, out_dim)

            def forward(self, x: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
                if arch == "gat":
                    h = self.l1(x, a_t)
                    h = self.l2(h, a_t)
                elif arch == "graphsage":
                    h = F.relu(self.l1(torch.cat([x, a_t @ x], dim=-1)))
                    h = F.dropout(h, p=0.15, training=self.training)
                    h = F.relu(self.l2(torch.cat([h, a_t @ h], dim=-1)))
                else:
                    h = F.relu(a_t @ self.l1(x))
                    h = F.dropout(h, p=0.15, training=self.training)
                    h = F.relu(a_t @ self.l2(h))
                return self.head(h)

        a_mat = (
            adj + np.eye(len(adj))
            if arch == "gat"
            else (self.row_normalize_adjacency(adj) if arch == "graphsage" else self.normalize_adjacency(adj))
        )
        Xt = torch.tensor(X_s, dtype=torch.float32)
        At = torch.tensor(a_mat, dtype=torch.float32)
        yt = torch.tensor(y_enc, dtype=torch.long if is_clf else torch.float32)

        net = _GNNNet()
        opt = torch.optim.AdamW(net.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        loss_fn: Any = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()
        net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            out = net(Xt, At)
            loss = loss_fn(out, yt) if is_clf else loss_fn(out.squeeze(-1), yt)
            loss.backward()
            opt.step()
        net.eval()
        self._torch_net = net

    def _predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X_s = self._scaler.transform(X)
        adjacency = kwargs.get("adjacency")
        if self._torch_net is not None:
            import torch

            adj = adjacency if adjacency is not None else self._build_knn_adjacency(X_s)
            a_mat = (
                adj + np.eye(len(adj))
                if self.architecture == "gat"
                else (self.row_normalize_adjacency(adj) if self.architecture == "graphsage" else self.normalize_adjacency(adj))
            )
            with torch.no_grad():
                out = self._torch_net(torch.tensor(X_s, dtype=torch.float32), torch.tensor(a_mat, dtype=torch.float32))
                if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
                    idx = torch.argmax(out, dim=-1).cpu().numpy()
                    return np.array([self._idx_to_label[int(i)] for i in idx])
                return out.squeeze(-1).cpu().numpy()
        assert self._head is not None
        emb = self._propagate_numpy(X_s, adjacency=adjacency)
        return self._head.predict(emb)

    def _predict_proba(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X_s = self._scaler.transform(X)
        adjacency = kwargs.get("adjacency")
        if self._torch_net is not None:
            import torch

            adj = adjacency if adjacency is not None else self._build_knn_adjacency(X_s)
            a_mat = (
                adj + np.eye(len(adj))
                if self.architecture == "gat"
                else (self.row_normalize_adjacency(adj) if self.architecture == "graphsage" else self.normalize_adjacency(adj))
            )
            with torch.no_grad():
                logits = self._torch_net(torch.tensor(X_s, dtype=torch.float32), torch.tensor(a_mat, dtype=torch.float32))
                return torch.softmax(logits, dim=-1).cpu().numpy()
        assert self._head is not None
        emb = self._propagate_numpy(X_s, adjacency=adjacency)
        return self._head.predict_proba(emb)


class GraphAutoencoderLinkModel:
    """Variational / Standard Graph Autoencoder (Kipf & Welling, 2016) for Neural Link Prediction."""

    def __init__(
        self,
        embedding_dim: int = 16,
        hidden_dim: int = 32,
        variational: bool = True,
        epochs: int = 80,
        learning_rate: float = 1e-2,
        random_seed: int = 42,
    ) -> None:
        self.embedding_dim = max(4, int(embedding_dim))
        self.hidden_dim = max(8, int(hidden_dim))
        self.variational = bool(variational)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.random_seed = int(random_seed)
        self.embeddings_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    def fit_embeddings(self, adj: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        """Train GAE/VGAE encoder on adjacency matrix to learn link-predictive latent embeddings Z."""
        n = adj.shape[0]
        X = features.astype(np.float32) if features is not None else np.eye(n, dtype=np.float32)
        a_norm = GraphNeuralNetworkModel.normalize_adjacency(adj).astype(np.float32)

        if is_package_available("torch") and n <= 2500:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            torch.manual_seed(self.random_seed)
            in_d = X.shape[1]
            h_d, z_d = self.hidden_dim, self.embedding_dim
            is_var = self.variational

            class _GAE(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.gc1 = nn.Linear(in_d, h_d)
                    self.gc_mu = nn.Linear(h_d, z_d)
                    self.gc_logstd = nn.Linear(h_d, z_d) if is_var else None

                def encode(self, x_t: torch.Tensor, a_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
                    h = F.relu(a_t @ self.gc1(x_t))
                    mu = a_t @ self.gc_mu(h)
                    logstd = (a_t @ self.gc_logstd(h)).clamp(-5.0, 2.0) if self.gc_logstd is not None else None
                    return mu, logstd

                def forward(self, x_t: torch.Tensor, a_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
                    mu, logstd = self.encode(x_t, a_t)
                    if logstd is not None and self.training:
                        z = mu + torch.randn_like(mu) * torch.exp(logstd)
                    else:
                        z = mu
                    adj_logits = z @ z.T
                    return adj_logits, mu, logstd

            net = _GAE()
            opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
            Xt = torch.tensor(X)
            At = torch.tensor(a_norm)
            target_adj = torch.tensor((adj > 0).astype(np.float32))
            pos_cnt = float(target_adj.sum().item())
            neg_cnt = max(float(n * n) - pos_cnt, 1.0)
            pos_weight = torch.tensor([min(neg_cnt / max(pos_cnt, 1.0), 20.0)])

            net.train()
            self.loss_history_ = []
            for _ in range(self.epochs):
                opt.zero_grad()
                logits, mu, logstd = net(Xt, At)
                bce = F.binary_cross_entropy_with_logits(logits, target_adj, pos_weight=pos_weight)
                if logstd is not None:
                    kl = -0.5 / max(n, 1) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - torch.exp(2 * logstd), dim=1))
                    loss = bce + 0.1 * kl
                else:
                    loss = bce
                loss.backward()
                opt.step()
                self.loss_history_.append(round(float(loss.item()), 6))

            net.eval()
            with torch.no_grad():
                mu, _ = net.encode(Xt, At)
                self.embeddings_ = mu.cpu().numpy().astype(float)
            return self.embeddings_

        # Spectral PMI + SVD deterministic fallback when PyTorch is unavailable
        u, s, _ = np.linalg.svd(a_norm @ X, full_matrices=False)
        k = min(self.embedding_dim, u.shape[1])
        self.embeddings_ = (u[:, :k] * np.sqrt(np.maximum(s[:k], 1e-9))).astype(float)
        return self.embeddings_

    def score_pairs(self, pairs: list[tuple[int, int]]) -> np.ndarray:
        """Compute calibrated sigmoid link probabilities sigma(z_u^T z_v) for candidate node index pairs."""
        if self.embeddings_ is None:
            raise ValueError("GraphAutoencoderLinkModel must call fit_embeddings() first.")
        Z = self.embeddings_
        scores = np.array([float(np.dot(Z[u], Z[v])) for u, v in pairs], dtype=float)
        return 1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))

    def evaluate_link_split(
        self,
        train_adj: np.ndarray,
        pos_test_pairs: list[tuple[int, int]],
        neg_test_pairs: list[tuple[int, int]],
        features: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train on train_adj and evaluate held-out link prediction ROC-AUC & Average Precision."""
        self.fit_embeddings(train_adj, features=features)
        pairs = [*pos_test_pairs, *neg_test_pairs]
        y_true = np.array([1] * len(pos_test_pairs) + [0] * len(neg_test_pairs), dtype=int)
        probs = self.score_pairs(pairs)
        return {
            "roc_auc": round(float(roc_auc_score(y_true, probs)), 6) if len(np.unique(y_true)) == 2 else 0.5,
            "average_precision": round(float(average_precision_score(y_true, probs)), 6) if len(np.unique(y_true)) == 2 else 0.5,
        }
