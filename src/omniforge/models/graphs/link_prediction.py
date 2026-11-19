"""Full Graph Link Prediction & Graph ML Suite: 8 Topological Scorers + RRF Ensemble, Supervised Link Prediction (Edge Operators + ROC-AUC/AP), Neural GAE/VGAE Link Prediction, K-Core/Bridges & Influence Maximization."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from omniforge.core.exceptions import ModelError
from omniforge.models.deep.gnn import GraphAutoencoderLinkModel
from omniforge.models.graphs.analytics import GraphAnalyticsEngine


class GraphLinkPredictionSuite(GraphAnalyticsEngine):
    """Production Graph Link Prediction & Graph ML Suite combining Topological, Supervised Embedding & Neural GAE/VGAE methods."""

    TOPOLOGICAL_METHODS: tuple[str, ...] = (
        "adamic_adar",
        "resource_allocation",
        "jaccard",
        "common_neighbors",
        "preferential_attachment",
        "katz",
        "personalized_pagerank",
        "lhn",
        "ensemble_rrf",
    )

    @staticmethod
    def _katz_matrix(undir: nx.Graph, nodes: list[Any], beta: float = 0.05, max_len: int = 4) -> np.ndarray:
        adj = np.asarray(nx.to_numpy_array(undir, nodelist=nodes, dtype=float), dtype=float)
        # Ensure strict Katz convergence: beta_eff < 1 / spectral_radius(A)
        max_deg = float(np.max(np.sum(adj, axis=1))) if adj.size > 0 else 1.0
        beta_eff = min(float(beta), 0.85 / max(max_deg, 1.0))
        acc: np.ndarray = np.zeros_like(adj)
        pwr = adj.copy()
        b = beta_eff
        for _ in range(max_len):
            acc += b * pwr
            pwr = pwr @ adj
            b *= beta_eff
        return acc

    def predict_links_advanced(
        self,
        graph: nx.Graph,
        *,
        method: str = "ensemble_rrf",
        top_k: int = 15,
        candidate_limit: int = 5000,
    ) -> list[dict[str, Any]]:
        """Score non-edges using 8 topological scorers or Reciprocal Rank Fusion (ensemble_rrf)."""
        undir = graph.to_undirected()
        nodes = list(undir.nodes())
        n = len(nodes)
        if n < 2:
            return []
        non_edges: list[tuple[Any, Any]] = []
        for idx, pair in enumerate(nx.non_edges(undir)):
            non_edges.append(pair)
            if idx + 1 >= candidate_limit:
                break
        if not non_edges:
            return []

        node_to_idx = {nd: i for i, nd in enumerate(nodes)}
        m = method.lower()

        def _score_single(single_m: str) -> dict[tuple[Any, Any], float]:
            if single_m == "jaccard":
                return {(u, v): float(s) for u, v, s in nx.jaccard_coefficient(undir, non_edges)}
            if single_m == "resource_allocation":
                return {(u, v): float(s) for u, v, s in nx.resource_allocation_index(undir, non_edges)}
            if single_m == "preferential_attachment":
                return {(u, v): float(s) for u, v, s in nx.preferential_attachment(undir, non_edges)}
            if single_m == "common_neighbors":
                return {(u, v): float(len(list(nx.common_neighbors(undir, u, v)))) for u, v in non_edges}
            if single_m == "lhn":
                out: dict[tuple[Any, Any], float] = {}
                for u, v in non_edges:
                    cn = len(list(nx.common_neighbors(undir, u, v)))
                    denom = max(undir.degree(u) * undir.degree(v), 1)
                    out[(u, v)] = float(cn / denom)
                return out
            if single_m == "katz":
                km = self._katz_matrix(undir, nodes)
                return {(u, v): float(km[node_to_idx[u], node_to_idx[v]]) for u, v in non_edges}
            if single_m == "personalized_pagerank":
                # Fast 2-step random-walk stationary similarity approximation for all pairs
                adj = nx.to_numpy_array(undir, nodelist=nodes, dtype=float)
                deg = adj.sum(axis=1, keepdims=True)
                P = np.divide(adj, np.where(deg > 0, deg, 1.0))
                alpha = 0.15
                ppr = alpha * (np.eye(n) + (1 - alpha) * P + ((1 - alpha) ** 2) * (P @ P) + ((1 - alpha) ** 3) * (P @ P @ P))
                sym = 0.5 * (ppr + ppr.T)
                return {(u, v): float(sym[node_to_idx[u], node_to_idx[v]]) for u, v in non_edges}
            return {(u, v): float(s) for u, v, s in nx.adamic_adar_index(undir, non_edges)}

        if m == "ensemble_rrf":
            sub_methods = ("adamic_adar", "resource_allocation", "jaccard", "katz", "personalized_pagerank")
            rrf_scores: dict[tuple[Any, Any], float] = dict.fromkeys(non_edges, 0.0)
            per_method_raw: dict[str, dict[tuple[Any, Any], float]] = {}
            for sm in sub_methods:
                smap = _score_single(sm)
                per_method_raw[sm] = smap
                ranked_pairs = sorted(smap.items(), key=lambda kv: kv[1], reverse=True)
                for rank_i, (pair, _) in enumerate(ranked_pairs, start=1):
                    rrf_scores[pair] += 1.0 / (60.0 + rank_i)
            ordered = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            return [
                {
                    "source": u,
                    "target": v,
                    "score": round(float(sc), 6),
                    "method": "ensemble_rrf",
                    "adamic_adar": round(float(per_method_raw["adamic_adar"].get((u, v), 0.0)), 4),
                    "jaccard": round(float(per_method_raw["jaccard"].get((u, v), 0.0)), 4),
                    "katz": round(float(per_method_raw["katz"].get((u, v), 0.0)), 5),
                }
                for (u, v), sc in ordered
            ]

        smap = _score_single(m)
        ordered = sorted(smap.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [{"source": u, "target": v, "score": round(float(sc), 6), "method": m} for (u, v), sc in ordered]

    @staticmethod
    def _edge_operator(emb_u: np.ndarray, emb_v: np.ndarray, op: str) -> np.ndarray:
        if op == "l1":
            return np.asarray(np.abs(emb_u - emb_v), dtype=float)
        if op == "l2":
            return np.asarray((emb_u - emb_v) ** 2, dtype=float)
        if op == "average":
            return np.asarray(0.5 * (emb_u + emb_v), dtype=float)
        return np.asarray(emb_u * emb_v, dtype=float)  # Hadamard (Grover & Leskovec, Node2Vec 2016)

    def train_supervised_link_predictor(
        self,
        graph: nx.Graph,
        *,
        test_ratio: float = 0.2,
        edge_operator: str = "hadamard",
        classifier: str = "rf",
        embedding_dim: int = 16,
        top_k: int = 15,
    ) -> dict[str, Any]:
        """Spanning-tree-safe Supervised Link Prediction with NetMF embeddings, topological pair features & ROC-AUC/AP evaluation."""
        undir = graph.to_undirected()
        edges = list(undir.edges())
        if len(edges) < 6:
            raise ModelError("Graph requires at least 6 edges for supervised train/test link prediction.")
        rng = np.random.default_rng(self.random_seed)

        # Preserve a spanning forest in training edges so graph connectivity is maintained
        mst_edges = set(tuple(sorted((u, v), key=str)) for u, v in nx.minimum_spanning_edges(undir, data=False))
        removable = [e for e in edges if tuple(sorted((e[0], e[1]), key=str)) not in mst_edges]
        n_test = max(2, min(len(removable), int(len(edges) * test_ratio)))
        if len(removable) < 2:
            # Fallback for tree-like graphs: hold out random subset of edges
            perm_idx = rng.permutation(len(edges))
            n_test = max(2, int(len(edges) * test_ratio))
            pos_test = [edges[int(i)] for i in perm_idx[:n_test]]
            pos_train = [edges[int(i)] for i in perm_idx[n_test:]]
        else:
            perm_idx = rng.permutation(len(removable))
            pos_test = [removable[int(i)] for i in perm_idx[:n_test]]
            test_set = {tuple(sorted((u, v), key=str)) for u, v in pos_test}
            pos_train = [e for e in edges if tuple(sorted((e[0], e[1]), key=str)) not in test_set]

        train_g = nx.Graph()
        train_g.add_nodes_from(undir.nodes())
        train_g.add_edges_from(pos_train)

        # Sample negative non-edges (balanced)
        all_non_edges = list(nx.non_edges(undir))
        if len(all_non_edges) < len(pos_train) + len(pos_test):
            neg_pool = all_non_edges
        else:
            pick = rng.choice(len(all_non_edges), size=len(pos_train) + len(pos_test), replace=False)
            neg_pool = [all_non_edges[int(i)] for i in pick]
        split_neg = max(1, min(len(neg_pool) - 1, len(pos_train)))
        neg_train = neg_pool[:split_neg]
        neg_test = neg_pool[split_neg : split_neg + len(pos_test)] or neg_pool[-max(1, len(pos_test)) :]

        # Compute NetMF spectral embeddings strictly on train_g (zero test-edge leakage!)
        emb_map = self.compute_node_embeddings(train_g, embedding_dim=embedding_dim)
        dim = len(next(iter(emb_map.values()))) if emb_map else 1

        def _pair_features(pairs: list[tuple[Any, Any]]) -> np.ndarray:
            rows = []
            for u, v in pairs:
                eu = emb_map.get(u, np.zeros(dim))
                ev = emb_map.get(v, np.zeros(dim))
                op_vec = self._edge_operator(eu, ev, edge_operator)
                cn = float(len(list(nx.common_neighbors(train_g, u, v))))
                du, dv = float(train_g.degree(u)), float(train_g.degree(v))
                aa = sum(1.0 / np.log(max(train_g.degree(w), 2)) for w in nx.common_neighbors(train_g, u, v))
                jac = cn / max(du + dv - cn, 1.0)
                rows.append(np.concatenate([op_vec, [cn, du * dv, aa, jac]]))
            return np.vstack(rows) if rows else np.zeros((0, dim + 4))

        X_tr = _pair_features([*pos_train, *neg_train])
        y_tr = np.array([1] * len(pos_train) + [0] * len(neg_train), dtype=int)
        X_te = _pair_features([*pos_test, *neg_test])
        y_te = np.array([1] * len(pos_test) + [0] * len(neg_test), dtype=int)

        clf_k = classifier.lower()
        if clf_k == "hgb":
            clf: Any = HistGradientBoostingClassifier(max_iter=100, random_state=self.random_seed)
        elif clf_k == "logistic":
            clf = LogisticRegression(max_iter=500, random_state=self.random_seed)
        else:
            clf = RandomForestClassifier(n_estimators=120, random_state=self.random_seed, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        te_probs = clf.predict_proba(X_te)[:, 1]
        roc_auc = float(roc_auc_score(y_te, te_probs)) if len(np.unique(y_te)) == 2 else 0.5
        ap = float(average_precision_score(y_te, te_probs)) if len(np.unique(y_te)) == 2 else 0.5

        # Score candidate unseen non-edges on full graph by refitting classifier on full-graph embedding basis
        # (prevents SVD sign/rotation mismatch between train_g embeddings and undir embeddings)
        full_emb = self.compute_node_embeddings(undir, embedding_dim=embedding_dim)
        full_dim = len(next(iter(full_emb.values()))) if full_emb else 1

        def _full_pair_features(pairs: list[tuple[Any, Any]]) -> np.ndarray:
            rows_f = []
            for u, v in pairs:
                eu = full_emb.get(u, np.zeros(full_dim))
                ev = full_emb.get(v, np.zeros(full_dim))
                op_vec = self._edge_operator(eu, ev, edge_operator)
                cn = float(len(list(nx.common_neighbors(undir, u, v))))
                du, dv = float(undir.degree(u)), float(undir.degree(v))
                aa = sum(1.0 / np.log(max(undir.degree(w), 2)) for w in nx.common_neighbors(undir, u, v))
                jac = cn / max(du + dv - cn, 1.0)
                rows_f.append(np.concatenate([op_vec, [cn, du * dv, aa, jac]]))
            return np.vstack(rows_f) if rows_f else np.zeros((0, full_dim + 4))

        if len(all_non_edges) <= 2500:
            cand_edges = all_non_edges
        else:
            cand_pick = rng.choice(len(all_non_edges), size=2500, replace=False)
            cand_edges = [all_non_edges[int(i)] for i in cand_pick]

        top_preds: list[dict[str, Any]] = []
        if cand_edges:
            neg_full_n = min(len(all_non_edges), len(edges))
            neg_full_pick = rng.choice(len(all_non_edges), size=neg_full_n, replace=False)
            neg_full = [all_non_edges[int(i)] for i in neg_full_pick]
            X_full_tr = _full_pair_features([*edges, *neg_full])
            y_full_tr = np.array([1] * len(edges) + [0] * len(neg_full), dtype=int)
            if clf_k == "hgb":
                clf_full: Any = HistGradientBoostingClassifier(max_iter=100, random_state=self.random_seed)
            elif clf_k == "logistic":
                clf_full = LogisticRegression(max_iter=500, random_state=self.random_seed)
            else:
                clf_full = RandomForestClassifier(n_estimators=120, random_state=self.random_seed, n_jobs=-1)
            clf_full.fit(X_full_tr, y_full_tr)
            cand_probs = clf_full.predict_proba(_full_pair_features(cand_edges))[:, 1]
            order = np.argsort(cand_probs)[::-1][:top_k]
            top_preds = [
                {"source": cand_edges[int(i)][0], "target": cand_edges[int(i)][1], "probability": round(float(cand_probs[int(i)]), 5), "method": f"supervised_{clf_k}_{edge_operator}"}
                for i in order
            ]

        return {
            "metrics": {
                "roc_auc": round(roc_auc, 5),
                "average_precision": round(ap, 5),
                "train_edges": len(pos_train),
                "test_edges": len(pos_test),
                "edge_operator": edge_operator,
                "classifier": clf_k,
            },
            "top_predictions": top_preds,
        }

    def predict_links_neural_gae(
        self,
        graph: nx.Graph,
        *,
        variational: bool = True,
        embedding_dim: int = 16,
        epochs: int = 60,
        top_k: int = 15,
    ) -> dict[str, Any]:
        """Train Neural Graph Autoencoder (GAE/VGAE) for link prediction with held-out ROC-AUC & top predictions."""
        undir = graph.to_undirected()
        nodes = list(undir.nodes())
        n = len(nodes)
        if n < 4 or undir.number_of_edges() < 4:
            raise ModelError("Graph requires at least 4 nodes and 4 edges for Neural GAE/VGAE.")
        node_to_idx = {nd: i for i, nd in enumerate(nodes)}
        raw_edges = list(undir.edges())
        rng = np.random.default_rng(self.random_seed)
        # Preserve spanning forest connectivity during held-out edge split
        mst_edges = {tuple(sorted((u, v), key=str)) for u, v in nx.minimum_spanning_edges(undir, data=False)}
        removable = [e for e in raw_edges if tuple(sorted((e[0], e[1]), key=str)) not in mst_edges]
        n_test = max(1, int(len(raw_edges) * 0.2))
        if len(removable) >= n_test:
            perm = rng.permutation(len(removable))
            pos_test_raw = [removable[int(i)] for i in perm[:n_test]]
            test_key = {tuple(sorted((u, v), key=str)) for u, v in pos_test_raw}
            pos_train_raw = [e for e in raw_edges if tuple(sorted((e[0], e[1]), key=str)) not in test_key]
        else:
            perm = rng.permutation(len(raw_edges))
            pos_test_raw = [raw_edges[int(i)] for i in perm[:n_test]]
            pos_train_raw = [raw_edges[int(i)] for i in perm[n_test:]]

        pos_test = [(node_to_idx[u], node_to_idx[v]) for u, v in pos_test_raw]
        pos_train = [(node_to_idx[u], node_to_idx[v]) for u, v in pos_train_raw]

        train_adj = np.zeros((n, n), dtype=float)
        for u, v in pos_train:
            train_adj[u, v] = 1.0
            train_adj[v, u] = 1.0
        full_adj = nx.to_numpy_array(undir, nodelist=nodes, dtype=float)
        all_non_edges_idx = [(node_to_idx[u], node_to_idx[v]) for u, v in nx.non_edges(undir)]
        # Sample held-out negatives UNIFORMLY at random across all non-edges (not lexicographical node-0 slice!)
        if len(all_non_edges_idx) > len(pos_test):
            neg_pick = rng.choice(len(all_non_edges_idx), size=len(pos_test), replace=False)
            neg_test = [all_non_edges_idx[int(i)] for i in neg_pick]
        else:
            neg_test = list(all_non_edges_idx)
        if len(all_non_edges_idx) > 2000:
            cand_pick = rng.choice(len(all_non_edges_idx), size=2000, replace=False)
            non_edges_idx = [all_non_edges_idx[int(i)] for i in cand_pick]
        else:
            non_edges_idx = all_non_edges_idx

        gae = GraphAutoencoderLinkModel(embedding_dim=embedding_dim, variational=variational, epochs=epochs, random_seed=self.random_seed)
        eval_metrics = gae.evaluate_link_split(train_adj, pos_test, neg_test) if neg_test else {"roc_auc": 0.5, "average_precision": 0.5}
        gae.fit_embeddings(full_adj)
        probs = gae.score_pairs(non_edges_idx) if non_edges_idx else np.array([])
        order = np.argsort(probs)[::-1][:top_k]
        top_preds = [
            {"source": nodes[non_edges_idx[int(i)][0]], "target": nodes[non_edges_idx[int(i)][1]], "probability": round(float(probs[int(i)]), 5), "method": "neural_vgae" if variational else "neural_gae"}
            for i in order
        ]
        return {"metrics": eval_metrics, "top_predictions": top_preds}

    @staticmethod
    def analyze_structure_and_influence(graph: nx.Graph, *, top_influencers: int = 5) -> dict[str, Any]:
        """Compute K-Core decomposition, Bridges, Articulation Points, Connected Components & Top Influential Seed Nodes."""
        undir = graph.to_undirected()
        undir.remove_edges_from(nx.selfloop_edges(undir))
        core_num = nx.core_number(undir) if undir.number_of_nodes() > 0 else {}
        bridges = [(str(u), str(v)) for u, v in list(nx.bridges(undir))[:50]]
        art_pts = [str(n) for n in list(nx.articulation_points(undir))[:50]]
        pr = nx.pagerank(undir) if undir.number_of_nodes() > 0 else {}
        # Greedy degree-discount seed selection for influence maximization
        deg_rem = {n: undir.degree(n) for n in undir.nodes()}
        selected_seeds: list[dict[str, Any]] = []
        chosen: set[Any] = set()
        for rank in range(1, min(top_influencers, undir.number_of_nodes()) + 1):
            cands = [(n, deg_rem[n] + 10.0 * pr.get(n, 0.0)) for n in undir.nodes() if n not in chosen]
            if not cands:
                break
            best_n, best_sc = max(cands, key=lambda kv: kv[1])
            chosen.add(best_n)
            for nb in undir.neighbors(best_n):
                if nb in deg_rem:
                    deg_rem[nb] = max(0, deg_rem[nb] - 1)
            selected_seeds.append({"rank": rank, "node": str(best_n), "degree": int(undir.degree(best_n)), "pagerank": round(float(pr.get(best_n, 0.0)), 5), "k_core": int(core_num.get(best_n, 0)), "influence_score": round(float(best_sc), 4)})
        return {
            "n_nodes": undir.number_of_nodes(),
            "n_edges": undir.number_of_edges(),
            "density": round(float(nx.density(undir)), 6),
            "n_components": nx.number_connected_components(undir),
            "max_k_core": max(core_num.values()) if core_num else 0,
            "n_bridges": len(bridges),
            "bridges": bridges[:20],
            "articulation_points": art_pts[:20],
            "top_influencers": selected_seeds,
        }
