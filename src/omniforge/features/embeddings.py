"""Text, categorical entity, and dense tabular embedding generators with semantic search."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from omniforge.core.exceptions import FeatureEngineeringError
from omniforge.core.utils import is_package_available


class EmbeddingGenerator:
    """Produces dense vector representations for text, categorical entities, and tabular rows."""

    def __init__(self, embedding_dim: int = 32, random_seed: int = 42) -> None:
        self.embedding_dim = embedding_dim
        self.random_seed = random_seed
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._pca: PCA | None = None
        self._scaler: StandardScaler | None = None
        self._entity_embeddings: dict[str, np.ndarray] = {}
        self._mode: str = ""

    def fit_text(self, texts: list[str], *, use_sentence_transformers: bool = False) -> EmbeddingGenerator:
        """Fit a semantic text embedding pipeline (SentenceTransformers or TF-IDF + LSA/SVD)."""
        cleaned = [str(t) if t is not None else "" for t in texts]
        if use_sentence_transformers and is_package_available("sentence_transformers"):
            self._mode = "sentence_transformers"
            return self

        self._vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        tfidf = self._vectorizer.fit_transform(cleaned)
        n_comp = max(1, min(self.embedding_dim, tfidf.shape[1] - 1, max(1, tfidf.shape[0] - 1)))
        if tfidf.shape[1] > 1 and n_comp >= 1:
            self._svd = TruncatedSVD(n_components=n_comp, random_state=self.random_seed)
            self._svd.fit(tfidf)
        else:
            self._svd = None
        self._mode = "tfidf_svd"
        return self

    def transform_text(self, texts: list[str]) -> np.ndarray:
        """Encode ``texts`` into L2-normalized dense embeddings."""
        cleaned = [str(t) if t is not None else "" for t in texts]
        if self._mode == "sentence_transformers":
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            model = SentenceTransformer("all-MiniLM-L6-v2")
            return np.asarray(model.encode(cleaned, normalize_embeddings=True), dtype=np.float32)

        if self._vectorizer is None:
            raise FeatureEngineeringError("Call fit_text() before transform_text().")
        tfidf = self._vectorizer.transform(cleaned)
        dense = self._svd.transform(tfidf) if self._svd is not None else tfidf.toarray()
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        return (dense / np.maximum(norms, 1e-12)).astype(np.float32)

    def fit_transform_text(self, texts: list[str]) -> np.ndarray:
        """Fit text embedder and return dense embeddings."""
        self.fit_text(texts)
        return self.transform_text(texts)

    def fit_tabular(self, df: pd.DataFrame) -> EmbeddingGenerator:
        """Fit a dense tabular projection via standardized PCA."""
        num = df.select_dtypes(include=[np.number]).fillna(0.0)
        if num.shape[1] == 0:
            raise FeatureEngineeringError("No numeric columns available for tabular embeddings.")
        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(num.to_numpy(dtype=float))
        n_comp = max(1, min(self.embedding_dim, scaled.shape[0], scaled.shape[1]))
        self._pca = PCA(n_components=n_comp, random_state=self.random_seed)
        self._pca.fit(scaled)
        self._mode = "tabular_pca"
        return self

    def transform_tabular(self, df: pd.DataFrame) -> np.ndarray:
        """Project tabular rows into dense embedding vectors."""
        if self._scaler is None or self._pca is None:
            raise FeatureEngineeringError("Call fit_tabular() before transform_tabular().")
        num = df.select_dtypes(include=[np.number]).fillna(0.0)
        scaled = self._scaler.transform(num.to_numpy(dtype=float))
        return np.asarray(self._pca.transform(scaled), dtype=np.float32)

    def fit_entity_embeddings(
        self, df: pd.DataFrame, entity_col: str, context_cols: list[str]
    ) -> dict[str, np.ndarray]:
        """Learn dense entity embeddings from aggregated numeric context profiles via SVD."""
        if entity_col not in df.columns:
            raise FeatureEngineeringError(f"Entity column '{entity_col}' not found.")
        valid_ctx = [c for c in context_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if not valid_ctx:
            raise FeatureEngineeringError("At least one numeric context column is required.")
        grouped = df.groupby(entity_col, observed=True)[valid_ctx].mean().fillna(0.0)
        mat = StandardScaler().fit_transform(grouped.to_numpy(dtype=float))
        dim = max(1, min(self.embedding_dim, mat.shape[0], mat.shape[1]))
        if dim < mat.shape[1]:
            emb = PCA(n_components=dim, random_state=self.random_seed).fit_transform(mat)
        else:
            emb = mat
        self._entity_embeddings = {
            str(entity): emb[idx].astype(np.float32) for idx, entity in enumerate(grouped.index)
        }
        return self._entity_embeddings

    @staticmethod
    def cosine_search(
        query_vec: np.ndarray, corpus_vecs: np.ndarray, top_k: int = 5
    ) -> list[tuple[int, float]]:
        """Return ``(index, cosine_similarity)`` for the ``top_k`` closest vectors."""
        q = np.asarray(query_vec, dtype=float).ravel()
        c = np.asarray(corpus_vecs, dtype=float)
        q_norm = np.linalg.norm(q) + 1e-12
        c_norms = np.linalg.norm(c, axis=1) + 1e-12
        sims = (c @ q) / (c_norms * q_norm)
        order = np.argsort(-sims)[: min(top_k, len(sims))]
        return [(int(idx), float(sims[idx])) for idx in order]
