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
        has_rating = rating_col in interactions.columns
        for _, row in interactions.iterrows():
            u_i = self._user_to_idx[row[user_col]]
            it_i = self._item_to_idx[row[item_col]]
            mat[u_i, it_i] = float(row[rating_col]) if has_rating else 1.0

        self._interaction_mat = mat
        self._popularity_scores = mat.sum(axis=0) / max(float(mat.sum()), 1e-12)

        # Collaborative filtering via TruncatedSVD (with user mean centering)
        user_means = np.where(mat.sum(axis=1) > 0, mat.sum(axis=1) / np.maximum((mat > 0).sum(axis=1), 1), 0.0)
        centered = np.where(mat > 0, mat - user_means[:, None], 0.0)
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
        cf = self._cf_scores[u_idx]
        cf_norm = (cf - cf.min()) / max(float(cf.max() - cf.min()), 1e-12)

        content_raw = user_hist @ self._item_sim if self._item_sim.size > 0 else np.zeros(n_items)
        c_norm = (content_raw - content_raw.min()) / max(float(content_raw.max() - content_raw.min()), 1e-12)
        scores = self.cf_weight * cf_norm + self.content_weight * c_norm

        if exclude_seen:
            scores = np.where(user_hist > 0, -np.inf, scores)

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
        """Compute Precision@K, Recall@K, and NDCG@K ranking metrics."""
        top = recommended_items[:k]
        hits = [1.0 if item in relevant_items else 0.0 for item in top]
        precision = sum(hits) / max(len(top), 1)
        recall = sum(hits) / max(len(relevant_items), 1)
        dcg = sum(h / np.log2(idx + 2.0) for idx, h in enumerate(hits))
        ideal_hits = [1.0] * min(len(relevant_items), k)
        idcg = sum(h / np.log2(idx + 2.0) for idx, h in enumerate(ideal_hits))
        ndcg = dcg / max(idcg, 1e-12)
        return {
            f"precision_at_{k}": round(precision, 4),
            f"recall_at_{k}": round(recall, 4),
            f"ndcg_at_{k}": round(ndcg, 4),
        }
