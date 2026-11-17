"""Enterprise NLP Toolkit: sentiment, topic modeling, TextRank summarization, NER, keyphrases, classification."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from omniforge.core.exceptions import ModelError

_POSITIVE_LEXICON: frozenset[str] = frozenset(
    {
        "good", "great", "excellent", "strong", "positive", "growth", "profit", "improve",
        "improved", "success", "successful", "gain", "gains", "outperform", "efficient",
        "reliable", "innovative", "clean", "fast", "accurate", "happy", "satisfied", "best",
        "robust", "healthy", "opportunity", "optimal", "benefit", "increase", "record",
    }
)
_NEGATIVE_LEXICON: frozenset[str] = frozenset(
    {
        "bad", "poor", "weak", "negative", "loss", "losses", "decline", "fail", "failed",
        "failure", "risk", "risks", "slow", "error", "errors", "bug", "crash", "delay",
        "delayed", "fraud", "breach", "critical", "severe", "drop", "decrease", "unhappy",
        "defect", "churn", "default", "warning", "unstable", "volatile", "issue", "problem",
    }
)
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "have",
        "has", "had", "not", "but", "you", "your", "our", "their", "they", "them", "been",
        "will", "would", "could", "should", "into", "about", "over", "after", "before",
        "between", "under", "while", "where", "which", "when", "what", "there", "these",
        "those", "than", "then", "also", "more", "most", "some", "such", "only", "very",
    }
)
_ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "URL": re.compile(r"https?://[^\s]+"),
    "MONEY": re.compile(r"(?:[$€£¥]\s?\d+(?:,\d{3})*(?:\.\d+)?(?:\s?[MBKmbk])?|\b\d+(?:\.\d+)?\s?(?:USD|EUR|GBP)\b)"),
    "PERCENT": re.compile(r"\b\d+(?:\.\d+)?%"),
    "DATE": re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b"),
}


class NLPEngine:
    """Comprehensive NLP toolkit for enterprise text analytics and document intelligence."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self._clf_vec: TfidfVectorizer | None = None
        self._clf_model: LogisticRegression | None = None

    @staticmethod
    def preprocess(text: str) -> str:
        """Clean and normalize raw text."""
        s = re.sub(r"<[^>]+>", " ", str(text))
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def analyze_sentiment(texts: list[str]) -> list[dict[str, Any]]:
        """Score sentiment polarity [-1, 1] and label (positive/neutral/negative)."""
        results: list[dict[str, Any]] = []
        for raw in texts:
            tokens = re.findall(r"[a-zA-Z]+", str(raw).lower())
            pos = sum(1 for tok in tokens if tok in _POSITIVE_LEXICON)
            neg = sum(1 for tok in tokens if tok in _NEGATIVE_LEXICON)
            denom = max(pos + neg, 1)
            polarity = (pos - neg) / denom if (pos + neg) > 0 else 0.0
            label = "positive" if polarity > 0.15 else ("negative" if polarity < -0.15 else "neutral")
            results.append(
                {
                    "label": label,
                    "polarity": round(float(polarity), 4),
                    "positive_hits": pos,
                    "negative_hits": neg,
                }
            )
        return results

    @staticmethod
    def extract_entities(text: str) -> list[dict[str, str]]:
        """Extract structured entities (money, dates, percentages, emails, URLs, proper nouns)."""
        entities: list[dict[str, str]] = []
        for label, pat in _ENTITY_PATTERNS.items():
            for match in pat.findall(text):
                entities.append({"text": str(match), "label": label})
        # Simple multi-word Proper Noun span extraction
        for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text):
            entities.append({"text": str(match), "label": "PROPER_NOUN"})
        return entities

    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Extract ranked domain keywords via length- and frequency-weighted scoring."""
        tokens = [
            tok
            for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
            if tok not in _STOPWORDS
        ]
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        scored = [
            (word, round((cnt / total) * (1.0 + 0.1 * min(len(word), 12)), 5))
            for word, cnt in counts.items()
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def summarize_extractive(text: str, max_sentences: int = 3) -> str:
        """Perform graph-based TextRank extractive summarization over TF-IDF sentence similarity."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.strip()) > 10]
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(sentences)
        sim = cosine_similarity(mat)
        np.fill_diagonal(sim, 0.0)
        # Power-iteration PageRank over sentence similarity graph
        scores = np.ones(len(sentences), dtype=float) / len(sentences)
        row_sums = sim.sum(axis=1, keepdims=True)
        trans = np.divide(sim, np.where(row_sums > 0, row_sums, 1.0))
        for _ in range(25):
            scores = 0.15 / len(sentences) + 0.85 * (trans.T @ scores)
        top_idx = sorted(np.argsort(-scores)[:max_sentences].tolist())
        return " ".join(sentences[i] for i in top_idx)

    def discover_topics(
        self, documents: list[str], *, n_topics: int = 4, top_words: int = 6
    ) -> dict[str, Any]:
        """Discover latent topics across a document corpus using TF-IDF + NMF."""
        if len(documents) < 2:
            raise ModelError("At least 2 documents are required for topic modeling.")
        vec = TfidfVectorizer(max_features=2000, stop_words="english", min_df=1)
        tfidf = vec.fit_transform([self.preprocess(d) for d in documents])
        k = max(1, min(n_topics, tfidf.shape[0], tfidf.shape[1]))
        nmf = NMF(n_components=k, random_state=self.random_seed, init="nndsvda", max_iter=300)
        W = nmf.fit_transform(tfidf)
        H = nmf.components_
        vocab = vec.get_feature_names_out()
        topics: list[dict[str, Any]] = []
        for idx in range(k):
            order = np.argsort(-H[idx])[: min(top_words, len(vocab))]
            topics.append(
                {
                    "topic_id": idx,
                    "top_terms": [str(vocab[j]) for j in order],
                    "weights": [round(float(H[idx, j]), 4) for j in order],
                }
            )
        doc_topics = np.argmax(W, axis=1).tolist()
        return {"n_topics": k, "topics": topics, "document_topics": doc_topics}

    def fit_classifier(self, texts: list[str], labels: list[Any]) -> NLPEngine:
        """Fit a fast TF-IDF n-gram + regularized Logistic Regression text classifier."""
        self._clf_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
        X = self._clf_vec.fit_transform([self.preprocess(t) for t in texts])
        self._clf_model = LogisticRegression(C=2.0, max_iter=1000, random_state=self.random_seed)
        self._clf_model.fit(X, labels)
        return self

    def predict_classifier(self, texts: list[str]) -> np.ndarray:
        """Predict text categories using fitted classifier."""
        if self._clf_vec is None or self._clf_model is None:
            raise ModelError("Call fit_classifier() before predict_classifier().")
        X = self._clf_vec.transform([self.preprocess(t) for t in texts])
        return np.asarray(self._clf_model.predict(X))
