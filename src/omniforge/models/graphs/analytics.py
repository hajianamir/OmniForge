"""Enterprise Graph Analytics & Graph ML: Centrality, Communities, Link Prediction, Node Embeddings."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from omniforge.core.exceptions import ModelError


class GraphAnalyticsEngine:
    """Comprehensive graph intelligence suite built on NetworkX and spectral/random-walk embeddings."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    @staticmethod
    def build_graph(
        edges_df: pd.DataFrame,
        *,
        source_col: str = "source",
        target_col: str = "target",
        weight_col: str | None = None,
        directed: bool = False,
    ) -> nx.Graph:
        """Construct a NetworkX graph from an edge-list DataFrame."""
        if source_col not in edges_df.columns or target_col not in edges_df.columns:
            raise ModelError(f"Edge DataFrame must contain '{source_col}' and '{target_col}'.")
        graph: nx.Graph = nx.DiGraph() if directed else nx.Graph()
        for _, row in edges_df.iterrows():
            w = float(row[weight_col]) if (weight_col and weight_col in edges_df.columns) else 1.0
            graph.add_edge(row[source_col], row[target_col], weight=w)
        return graph

    @staticmethod
    def compute_centralities(graph: nx.Graph) -> pd.DataFrame:
        """Compute PageRank, Degree, Betweenness, Closeness, and Clustering centralities per node."""
        if graph.number_of_nodes() == 0:
            return pd.DataFrame()
        pr = nx.pagerank(graph)
        deg = nx.degree_centrality(graph)
        bet = nx.betweenness_centrality(graph)
        clo = nx.closeness_centrality(graph)
        clust = nx.clustering(graph.to_undirected())
        rows = [
            {
                "node": node,
                "degree": graph.degree(node),
                "degree_centrality": round(float(deg[node]), 6),
                "pagerank": round(float(pr[node]), 6),
                "betweenness": round(float(bet[node]), 6),
                "closeness": round(float(clo[node]), 6),
                "clustering_coeff": round(float(clust[node]), 6),
            }
            for node in graph.nodes()
        ]
        return pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)

    @staticmethod
    def detect_communities(graph: nx.Graph) -> dict[str, Any]:
        """Detect graph communities using Clauset-Newman-Moore greedy modularity optimization."""
        undir = graph.to_undirected()
        if undir.number_of_edges() == 0:
            comms = [{n} for n in undir.nodes()]
            mod = 0.0
        else:
            comms = list(nx.algorithms.community.greedy_modularity_communities(undir))
            mod = float(nx.algorithms.community.modularity(undir, comms))
        node_to_comm: dict[Any, int] = {}
        for cid, members in enumerate(comms):
            for node in members:
                node_to_comm[node] = cid
        return {
            "n_communities": len(comms),
            "modularity": round(mod, 6),
            "node_community_map": node_to_comm,
            "community_sizes": [len(c) for c in comms],
        }

    @staticmethod
    def predict_links(
        graph: nx.Graph, *, method: str = "adamic_adar", top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Score non-existent edges for link prediction (adamic_adar, jaccard, preferential_attachment)."""
        undir = graph.to_undirected()
        non_edges = list(nx.non_edges(undir))
        if not non_edges:
            return []
        if method == "jaccard":
            preds = nx.jaccard_coefficient(undir, non_edges)
        elif method == "preferential_attachment":
            preds = nx.preferential_attachment(undir, non_edges)
        else:
            preds = nx.adamic_adar_index(undir, non_edges)
        scored = sorted(
            [{"source": u, "target": v, "score": round(float(s), 6)} for u, v, s in preds],
            key=lambda d: d["score"],
            reverse=True,
        )
        return scored[:top_k]

    def compute_node_embeddings(
        self, graph: nx.Graph, *, embedding_dim: int = 16, walk_steps: int = 3
    ) -> dict[Any, np.ndarray]:
        """Compute NetMF / random-walk PMI spectral node embeddings via TruncatedSVD."""
        nodes = list(graph.nodes())
        n = len(nodes)
        if n == 0:
            return {}
        adj = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
        deg = adj.sum(axis=1, keepdims=True)
        P = np.divide(adj, np.where(deg > 0, deg, 1.0))
        # Multi-step transition matrix sum approximating DeepWalk/Node2Vec co-occurrence
        accum = np.zeros_like(P)
        power = P.copy()
        for _ in range(max(1, walk_steps)):
            accum += power
            power = power @ P
        pmi = np.log(np.maximum(accum * n / max(walk_steps, 1), 1.0))
        k = max(1, min(embedding_dim, n - 1)) if n > 1 else 1
        if n > 2 and k < n:
            emb = TruncatedSVD(n_components=k, random_state=self.random_seed).fit_transform(pmi)
        else:
            emb = pmi[:, :k]
        return {node: emb[i].astype(np.float32) for i, node in enumerate(nodes)}

    @staticmethod
    def detect_anomalous_nodes(graph: nx.Graph, z_threshold: float = 2.5) -> pd.DataFrame:
        """Identify structurally anomalous nodes via z-scores on degree, PageRank, and ego density."""
        df = GraphAnalyticsEngine.compute_centralities(graph)
        if len(df) < 3:
            df["anomaly_score"] = 0.0
            df["is_anomaly"] = 0
            return df
        feats = df[["degree", "pagerank", "betweenness"]].to_numpy(dtype=float)
        stds = np.maximum(feats.std(axis=0), 1e-9)
        z = np.abs((feats - feats.mean(axis=0)) / stds)
        df["anomaly_score"] = np.round(z.max(axis=1), 4)
        df["is_anomaly"] = (df["anomaly_score"] > z_threshold).astype(int)
        return df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
