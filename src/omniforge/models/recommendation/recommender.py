"""Enterprise Recommendation Systems: SVD Matrix Factorization, Item-KNN, Content-Based, Hybrid."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from omniforge.core.exceptions import ModelError


class HybridRecommender:
    """Hybrid recommender combining SVD collaborative filtering, item-KNN, and content similarity."""

    def __init__(
        self,
        n_factors: int = 16,
        cf_weight: float = 0.65,
        content_weight: float = 0.35,
        random_seed: int = 42,
    ) -> None:
        self.n_factors = int(n_factors)
        self.cf_weight = float(cf_weight)
        self.content_weight = float(content_weight)
        self.random_seed = random_seed

        self._user_to_idx: dict[Any, int] = {}
        self._item_to_idx: dict[Any, int] = {}
        self._idx_to_item: dict[int, Any] = {}
        self._interaction_mat: np.ndarray = np.empty((0, 0))
        self._observed_mask: np.ndarray = np.empty((0, 0), dtype=bool)
        self._cf_scores: np.ndarray = np.empty((0, 0))
        self._item_sim: np.ndarray = np.empty((0, 0))
        self._popularity_scores: np.ndarray = np.empty(0)
        self._is_fitted = False

    def fit(
        self,
        interactions: pd.DataFrame,
        *,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating",
        item_metadata: pd.DataFrame | None = None,
        text_col: str | None = None,
    ) -> HybridRecommender:
        """Fit collaborative SVD and optional content-based TF-IDF similarity matrix."""
        for c in (user_col, item_col):
            if c not in interactions.columns:
                raise ModelError(f"Column '{c}' required in interactions DataFrame.")

        users = sorted(interactions[user_col].unique().tolist(), key=str)
        items = sorted(interactions[item_col].unique().tolist(), key=str)
        self._user_to_idx = {u: i for i, u in enumerate(users)}
        self._item_to_idx = {it: i for i, it in enumerate(items)}
        self._idx_to_item = {i: it for i, it in enumerate(items)}

        mat = np.zeros((len(users), len(items)), dtype=float)
        counts = np.zeros((len(users), len(items)), dtype=float)
        has_rating = rating_col in interactions.columns
        for row_u, row_it, row_r in zip(
            interactions[user_col],
            interactions[item_col],
            interactions[rating_col] if has_rating else [1.0] * len(interactions),
            strict=False,
        ):
            val = float(row_r) if has_rating else 1.0
            if not np.isfinite(val):
                continue
            u_i = self._user_to_idx[row_u]
            it_i = self._item_to_idx[row_it]
            mat[u_i, it_i] += val
            counts[u_i, it_i] += 1.0

        observed = counts > 0
        if has_rating:
            mat = np.where(observed, mat / np.where(observed, counts, 1.0), 0.0)
        self._interaction_mat = mat
        self._observed_mask = observed
        pop_raw = observed.sum(axis=0).astype(float)
        self._popularity_scores = pop_raw / max(float(pop_raw.sum()), 1e-12)

        # Collaborative filtering via TruncatedSVD (with user mean centering over observed entries)
        obs_counts = observed.sum(axis=1)
        user_means = np.where(obs_counts > 0, (mat * observed).sum(axis=1) / np.maximum(obs_counts, 1), 0.0)
        centered = np.where(observed, mat - user_means[:, None], 0.0)
        k = max(1, min(self.n_factors, min(mat.shape) - 1))
        if min(mat.shape) > 2:
            svd = TruncatedSVD(n_components=k, random_state=self.random_seed)
            u_factors = svd.fit_transform(centered)
            recon = u_factors @ svd.components_ + user_means[:, None]
        else:
            recon = mat.copy()
        self._cf_scores = recon

        # Item-item similarity (content TF-IDF if provided, else collaborative item cosine)
        if item_metadata is not None and text_col and text_col in item_metadata.columns and item_col in item_metadata.columns:
            meta_map = dict(zip(item_metadata[item_col], item_metadata[text_col].fillna("").astype(str), strict=False))
            corpus = [meta_map.get(it, "") for it in items]
            tfidf = TfidfVectorizer(max_features=2000).fit_transform(corpus)
            self._item_sim = cosine_similarity(tfidf)
        else:
            self._item_sim = cosine_similarity(mat.T) if mat.shape[1] > 0 else np.empty((0, 0))
        if self._item_sim.size > 0:
            np.fill_diagonal(self._item_sim, 0.0)

        self._is_fitted = True
        return self

    def recommend(
        self,
        user_id: Any,
        *,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[dict[str, Any]]:
        """Generate Top-K item recommendations with hybrid scores and cold-start fallback."""
        if not self._is_fitted:
            raise ModelError("HybridRecommender must be fitted before recommend().")

        n_items = len(self._idx_to_item)
        if user_id not in self._user_to_idx:
            # Cold-start fallback: popularity ranking
            order = np.argsort(-self._popularity_scores)[: min(top_k, n_items)]
            return [
                {"item_id": self._idx_to_item[int(i)], "score": round(float(self._popularity_scores[i]), 6), "strategy": "cold_start_popularity"}
                for i in order
            ]

        u_idx = self._user_to_idx[user_id]
        user_hist = self._interaction_mat[u_idx]
        seen_mask = self._observed_mask[u_idx]
        cand_mask = (~seen_mask) if exclude_seen else np.ones(n_items, dtype=bool)
        if not np.any(cand_mask):
            return []

        cf = self._cf_scores[u_idx]
        cf_c = cf[cand_mask]
        cf_span = float(cf_c.max() - cf_c.min())
        cf_norm = (cf - float(cf_c.min())) / cf_span if cf_span > 1e-12 else np.full(n_items, 0.5, dtype=float)

        if self._item_sim.size > 0:
            # Sarwar et al. (2001) similarity-weighted item-KNN score (self-diagonal already zeroed)
            sim_num = user_hist @ self._item_sim
            sim_den = np.asarray(seen_mask, dtype=float) @ np.abs(self._item_sim)
            content_raw = np.where(sim_den > 1e-12, sim_num / np.where(sim_den > 1e-12, sim_den, 1.0), 0.0)
        else:
            content_raw = np.zeros(n_items, dtype=float)
        c_c = content_raw[cand_mask]
        c_span = float(c_c.max() - c_c.min())
        c_norm = (content_raw - float(c_c.min())) / c_span if c_span > 1e-12 else np.full(n_items, 0.5, dtype=float)

        w_sum = max(self.cf_weight + self.content_weight, 1e-12)
        scores = (self.cf_weight * cf_norm + self.content_weight * c_norm) / w_sum

        if exclude_seen:
            scores = np.where(seen_mask, -np.inf, scores)

        order = np.argsort(-scores)[: min(top_k, n_items)]
        return [
            {"item_id": self._idx_to_item[int(i)], "score": round(float(scores[i]), 6), "strategy": "hybrid_cf_content"}
            for i in order
            if np.isfinite(scores[i])
        ]

    @staticmethod
    def evaluate_ranking(
        recommended_items: list[Any], relevant_items: set[Any], k: int = 10
    ) -> dict[str, float]:
        """Compute deduplicated Precision@K, Recall@K, and binary NDCG@K ranking metrics."""
        k_eff = max(1, int(k))
        # Deduplicate preserving rank order so repeated recommendations cannot inflate Recall/NDCG > 1.0
        top = list(dict.fromkeys(recommended_items))[:k_eff]
        hits = [1.0 if item in relevant_items else 0.0 for item in top]
        precision = sum(hits) / max(len(top), 1)
        recall = sum(hits) / max(len(relevant_items), 1) if relevant_items else 0.0
        dcg = sum(h / np.log2(idx + 2.0) for idx, h in enumerate(hits))
        ideal_hits = [1.0] * min(len(relevant_items), k_eff)
        idcg = sum(h / np.log2(idx + 2.0) for idx, h in enumerate(ideal_hits))
        ndcg = dcg / max(idcg, 1e-12) if idcg > 0.0 else 0.0
        return {
            f"precision_at_{k_eff}": round(precision, 4),
            f"recall_at_{k_eff}": round(recall, 4),
            f"ndcg_at_{k_eff}": round(ndcg, 4),
        }
