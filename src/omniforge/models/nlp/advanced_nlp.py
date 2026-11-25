"""Extended Enterprise NLP Suite: Zero-Shot Classification, Extractive QA, Dense Semantic Search, Emotion & Aspect Sentiment, Language Detection, Toxicity/PII Screening, Document Clustering & MMR Summarization."""

from __future__ import annotations

import importlib.metadata
import re
from collections import Counter
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from omniforge.core.logging import get_logger
from omniforge.core.utils import is_package_available
from omniforge.models.nlp.nlp_engine import NLPEngine

logger = get_logger(__name__)

_EMOTION_LEXICONS: dict[str, frozenset[str]] = {
    "joy": frozenset({"happy", "great", "excellent", "love", "delighted", "success", "celebrate", "wonderful", "win", "profit", "growth", "thrilled", "positive", "best"}),
    "trust": frozenset({"reliable", "secure", "trusted", "verified", "compliant", "stable", "robust", "proven", "safe", "honest", "transparent", "consistent", "confident"}),
    "anticipation": frozenset({"expect", "upcoming", "forecast", "plan", "future", "launch", "opportunity", "roadmap", "next", "predict", "target", "soon", "momentum"}),
    "anger": frozenset({"angry", "furious", "unacceptable", "broken", "terrible", "worst", "scam", "fraud", "outrage", "violation", "breach", "hate", "awful"}),
    "fear": frozenset({"risk", "threat", "danger", "crisis", "warning", "volatile", "crash", "collapse", "vulnerable", "uncertain", "panic", "concern", "default"}),
    "sadness": frozenset({"loss", "decline", "fail", "failed", "disappointed", "unfortunate", "drop", "churn", "miss", "sad", "regret", "downturn", "weak"}),
}

_TOXIC_TERMS: frozenset[str] = frozenset(
    {"idiot", "stupid", "moron", "hate", "kill", "abuse", "harass", "scam", "fraudulent", "worthless", "trash", "garbage", "pathetic"}
)

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

_LANG_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset({"the", "and", "is", "in", "to", "of", "for", "with", "that", "this", "from", "are"}),
    "de": frozenset({"der", "die", "das", "und", "ist", "ein", "eine", "mit", "für", "nicht", "auf", "von"}),
    "fr": frozenset({"le", "la", "les", "de", "des", "et", "est", "dans", "pour", "avec", "une", "pas"}),
    "es": frozenset({"el", "la", "los", "las", "de", "y", "es", "en", "para", "con", "una", "por"}),
    "tr": frozenset({"ve", "bir", "bu", "için", "ile", "olan", "olarak", "daha", "çok", "gibi"}),
}


def _ver(pkg: str) -> str | None:
    try:
        return importlib.metadata.version(pkg)
    except Exception:
        return None


def probe_nlp_backends() -> list[dict[str, Any]]:
    """Probe installed neural NLP libraries alongside OmniForge's native offline engines."""
    return [
        {"id": "transformers", "name": "HuggingFace Transformers (Zero-Shot / QA / BART / NER)", "available": is_package_available("transformers"), "version": _ver("transformers")},
        {"id": "sentence_transformers", "name": "Sentence-Transformers (Dense Semantic Embeddings)", "available": is_package_available("sentence_transformers"), "version": _ver("sentence-transformers")},
        {"id": "spacy", "name": "spaCy Industrial NLP", "available": is_package_available("spacy"), "version": _ver("spacy")},
            {"id": "native_nlp", "name": "OmniForge Advanced NLP Suite (Offline Always-On)", "available": True, "version": "4.0.0"},
    ]


class AdvancedNLPSuite(NLPEngine):
    """Full-spectrum Enterprise NLP Suite extending NLPEngine with Zero-Shot, QA, Semantic Search, Emotions, Language ID, PII/Toxicity, LDA & Document Clustering."""

    def __init__(self, random_seed: int = 42, use_pretrained: bool = False) -> None:
        super().__init__(random_seed=random_seed)
        self.use_pretrained = bool(use_pretrained)
        self._hf_pipelines: dict[str, Any] = {}

    def zero_shot_classify(
        self,
        texts: list[str],
        candidate_labels: list[str],
        *,
        multi_label: bool = False,
    ) -> list[dict[str, Any]]:
        """Classify arbitrary texts into arbitrary unseen labels (HuggingFace NLI if enabled+installed, else TF-IDF char+word semantic prototype scoring)."""
        labels = [str(lb).strip() for lb in candidate_labels if str(lb).strip()]
        if not texts or not labels:
            return []

        if self.use_pretrained and is_package_available("transformers"):
            try:
                from transformers import pipeline

                if "zs" not in self._hf_pipelines:
                    self._hf_pipelines["zs"] = pipeline("zero-shot-classification")
                pipe = self._hf_pipelines["zs"]
                out = []
                for txt in texts:
                    res = pipe(txt, candidate_labels=labels, multi_label=multi_label)
                    scores = {str(lb): round(float(sc), 5) for lb, sc in zip(res["labels"], res["scores"], strict=False)}
                    out.append({"text": txt[:140], "top_label": res["labels"][0], "confidence": round(float(res["scores"][0]), 5), "scores": scores, "backend": "transformers_nli"})
                return out
            except Exception as exc:
                # Backend exception text may contain user documents or credentials.
                logger.warning(
                    "nlp_backend_failed",
                    operation="zero_shot_classification",
                    backend="transformers",
                    error_type=type(exc).__name__,
                    fallback="native_semantic_tfidf",
                )

        # Calibrated hybrid word + character n-gram TF-IDF semantic similarity
        corpus = [*[self.preprocess(t) for t in texts], *[self.preprocess(lb) for lb in labels]]
        vec_w = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        try:
            mw = vec_w.fit_transform(corpus)
            mc = vec_c.fit_transform(corpus)
            n_t = len(texts)
            sim = 0.65 * cosine_similarity(mw[:n_t], mw[n_t:]) + 0.35 * cosine_similarity(mc[:n_t], mc[n_t:])
        except ValueError:
            sim = np.ones((len(texts), len(labels)), dtype=float) / max(len(labels), 1)

        results: list[dict[str, Any]] = []
        for i, txt in enumerate(texts):
            row_s = sim[i]
            if multi_label:
                probs = 1.0 / (1.0 + np.exp(-6.0 * (row_s - 0.12)))
            else:
                scaled = (row_s - np.max(row_s)) * 8.0
                ex = np.exp(scaled)
                probs = ex / np.maximum(np.sum(ex), 1e-12)
            order = np.argsort(probs)[::-1]
            score_map = {labels[int(j)]: round(float(probs[int(j)]), 5) for j in order}
            best_j = int(order[0])
            results.append(
                {
                    "text": txt[:140],
                    "top_label": labels[best_j],
                    "confidence": round(float(probs[best_j]), 5),
                    "scores": score_map,
                    "backend": "native_semantic_tfidf",
                }
            )
        return results

    def answer_question(self, question: str, context: str, *, top_k: int = 3) -> dict[str, Any]:
        """Extractive Question Answering returning ranked answer spans, character offsets & confidence."""
        q_clean = self.preprocess(question)
        ctx_clean = self.preprocess(context)
        if not q_clean or not ctx_clean:
            return {"answer": "", "confidence": 0.0, "candidates": [], "backend": "none"}

        if self.use_pretrained and is_package_available("transformers"):
            try:
                from transformers import pipeline

                if "qa" not in self._hf_pipelines:
                    self._hf_pipelines["qa"] = pipeline("question-answering")
                res = self._hf_pipelines["qa"](question=q_clean, context=ctx_clean)
                return {
                    "answer": str(res["answer"]),
                    "confidence": round(float(res["score"]), 5),
                    "start": int(res["start"]),
                    "end": int(res["end"]),
                    "candidates": [res],
                    "backend": "transformers_qa",
                }
            except Exception as exc:
                logger.warning(
                    "nlp_backend_failed",
                    operation="question_answering",
                    backend="transformers",
                    error_type=type(exc).__name__,
                    fallback="native_extractive_qa",
                )

        # BM25-weighted sentence + entity-type boost extractive QA
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctx_clean) if len(s.strip()) > 3]
        if not sents:
            sents = [ctx_clean]
        q_tokens = set(re.findall(r"[a-zA-Z0-9]+", q_clean.lower()))
        q_lower = q_clean.lower()
        wants_num = any(w in q_lower for w in ("how much", "how many", "cost", "price", "revenue", "percent", "rate", "score", "number"))
        wants_date = any(w in q_lower for w in ("when", "date", "year", "month"))

        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            mat = vec.fit_transform([q_clean, *sents])
            cos_scores = cosine_similarity(mat[0:1], mat[1:]).ravel()
        except ValueError:
            cos_scores = np.zeros(len(sents), dtype=float)

        cands: list[dict[str, Any]] = []
        cursor = 0
        for idx, sent in enumerate(sents):
            pos = ctx_clean.find(sent, cursor)
            if pos >= 0:
                cursor = pos + len(sent)
            s_tokens = set(re.findall(r"[a-zA-Z0-9]+", sent.lower()))
            overlap = len(q_tokens & s_tokens) / max(len(q_tokens), 1)
            bonus = 0.0
            if wants_num and re.search(r"[$€£¥]?\d+(?:[.,]\d+)?%?", sent):
                bonus += 0.18
            if wants_date and re.search(r"\b(?:\d{4}|\d{1,2}/\d{1,2}/\d{2,4})\b", sent):
                bonus += 0.18
            raw_sc = 0.55 * float(cos_scores[idx]) + 0.45 * overlap + bonus
            conf = float(1.0 / (1.0 + np.exp(-5.0 * (raw_sc - 0.25))))
            cands.append({"answer": sent, "confidence": round(conf, 5), "start": max(pos, 0), "end": max(pos, 0) + len(sent)})

        cands.sort(key=lambda r: r["confidence"], reverse=True)
        best = cands[0]
        return {
            "answer": best["answer"],
            "confidence": best["confidence"],
            "start": best["start"],
            "end": best["end"],
            "candidates": cands[:top_k],
            "backend": "native_extractive_qa",
        }

    def encode_embeddings(self, texts: list[str], *, dim: int = 32) -> tuple[np.ndarray, str]:
        """Encode texts into L2-normalized dense embeddings via Sentence-Transformers or SVD-TFIDF."""
        cleaned = [self.preprocess(t) for t in texts]
        if self.use_pretrained and is_package_available("sentence_transformers"):
            try:
                from sentence_transformers import SentenceTransformer

                if "st" not in self._hf_pipelines:
                    self._hf_pipelines["st"] = SentenceTransformer("all-MiniLM-L6-v2")
                emb = self._hf_pipelines["st"].encode(cleaned, normalize_embeddings=True)
                return np.asarray(emb, dtype=float), "sentence_transformers"
            except Exception as exc:
                logger.warning(
                    "nlp_backend_failed",
                    operation="embeddings",
                    backend="sentence_transformers",
                    error_type=type(exc).__name__,
                    fallback="native_tfidf_svd",
                )

        vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), sublinear_tf=True)
        mat = vec.fit_transform(cleaned)
        n_comp = min(int(dim), mat.shape[0] - 1, mat.shape[1] - 1)
        if n_comp >= 2:
            svd = TruncatedSVD(n_components=n_comp, random_state=self.random_seed)
            dense = svd.fit_transform(mat)
        else:
            dense = mat.toarray()
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        dense = np.divide(dense, np.where(norms > 1e-12, norms, 1.0))
        return dense.astype(float), "native_tfidf_svd"

    def semantic_search(self, query: str, corpus: list[str], *, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve top-K semantically relevant documents from a corpus for a query."""
        if not corpus:
            return []
        embs, backend = self.encode_embeddings([query, *corpus])
        sims = cosine_similarity(embs[0:1], embs[1:]).ravel()
        order = np.argsort(sims)[::-1][:top_k]
        return [
            {"rank": r + 1, "doc_index": int(idx), "similarity": round(float(sims[int(idx)]), 5), "text": corpus[int(idx)][:240], "backend": backend}
            for r, idx in enumerate(order)
        ]

    def detect_emotions_and_aspects(self, texts: list[str], *, aspects: list[str] | None = None) -> list[dict[str, Any]]:
        """Score 6-class Plutchik emotions and optional aspect-specific sentiment."""
        out: list[dict[str, Any]] = []
        sent_list = self.analyze_sentiment(texts)
        for txt, sent in zip(texts, sent_list, strict=False):
            toks = re.findall(r"[a-zA-Z]+", str(txt).lower())
            emo_counts = {emo: sum(1 for t in toks if t in lex) for emo, lex in _EMOTION_LEXICONS.items()}
            total = sum(emo_counts.values())
            emo_scores = {k: round(v / total, 4) if total > 0 else 0.0 for k, v in emo_counts.items()}
            dominant = max(emo_scores.items(), key=lambda kv: kv[1])[0] if total > 0 else "neutral"
            aspect_map: dict[str, dict[str, Any]] = {}
            if aspects:
                sents = re.split(r"(?<=[.!?])\s+", str(txt))
                for asp in aspects:
                    matched = [s for s in sents if asp.lower() in s.lower()]
                    if matched:
                        a_s = self.analyze_sentiment([" ".join(matched)])[0]
                        aspect_map[asp] = {"label": a_s["label"], "polarity": a_s["polarity"], "mentions": len(matched)}
            out.append(
                {
                    "text": str(txt)[:140],
                    "sentiment": sent["label"],
                    "polarity": sent["polarity"],
                    "dominant_emotion": dominant,
                    "emotions": emo_scores,
                    "aspects": aspect_map,
                }
            )
        return out

    @staticmethod
    def detect_language(texts: list[str]) -> list[dict[str, Any]]:
        """Detect language via Unicode script analysis + stopword scoring."""
        results: list[dict[str, Any]] = []
        for txt in texts:
            s = str(txt)
            if re.search(r"[\u067e\u0686\u0698\u06af\u06a9\u06cc]", s):
                results.append({"language": "fa", "name": "Persian", "confidence": 0.96})
                continue
            if re.search(r"[\u0600-\u06FF]", s):
                results.append({"language": "ar", "name": "Arabic", "confidence": 0.92})
                continue
            if re.search(r"[\u0400-\u04FF]", s):
                results.append({"language": "ru", "name": "Russian", "confidence": 0.94})
                continue
            if re.search(r"[\u4E00-\u9FFF]", s):
                results.append({"language": "zh", "name": "Chinese", "confidence": 0.95})
                continue
            if re.search(r"[\u3040-\u30FF]", s):
                results.append({"language": "ja", "name": "Japanese", "confidence": 0.95})
                continue
            toks = re.findall(r"[a-zA-Zäöüßéèêàçñğıüşöç]+", s.lower())
            best_lang, best_hits = "en", 0
            for code, sw in _LANG_STOPWORDS.items():
                hits = sum(1 for t in toks if t in sw)
                if hits > best_hits:
                    best_hits = hits
                    best_lang = code
            names = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "tr": "Turkish"}
            conf = min(0.98, 0.55 + 0.08 * best_hits)
            results.append({"language": best_lang, "name": names.get(best_lang, best_lang), "confidence": round(conf, 3)})
        return results

    @staticmethod
    def screen_toxicity_and_pii(texts: list[str]) -> list[dict[str, Any]]:
        """Return redacted text, PII categories and counts, and a toxicity assessment."""
        out: list[dict[str, Any]] = []
        for txt in texts:
            s = str(txt)
            pii_found: list[dict[str, str]] = []
            redacted = s
            for kind, pat in _PII_PATTERNS.items():
                pii_found.extend({"type": kind} for _ in pat.finditer(s))
                redacted = pat.sub(f"[{kind}_REDACTED]", redacted)
            toks = re.findall(r"[a-zA-Z]+", s.lower())
            tox_hits = [t for t in toks if t in _TOXIC_TERMS]
            tox_score = min(1.0, len(tox_hits) / max(len(toks) * 0.15, 1.0)) if tox_hits else 0.0
            out.append(
                {
                    "pii_count": len(pii_found),
                    "pii_entities": pii_found,
                    "toxicity_score": round(float(tox_score), 4),
                    "flagged": bool(pii_found or tox_score >= 0.25),
                    "redacted_text": redacted,
                }
            )
        return out

    def extract_topics_lda(self, texts: list[str], *, n_topics: int = 4, top_words: int = 8) -> list[dict[str, Any]]:
        """Probabilistic Latent Dirichlet Allocation (LDA) topic discovery."""
        cleaned = [self.preprocess(t) for t in texts if str(t).strip()]
        if len(cleaned) < 2:
            return []
        vec = CountVectorizer(max_features=1000, stop_words="english")
        dtm = vec.fit_transform(cleaned)
        k = max(1, min(int(n_topics), len(cleaned), dtm.shape[1]))
        lda = LatentDirichletAllocation(n_components=k, random_state=self.random_seed, max_iter=20)
        lda.fit(dtm)
        vocab = vec.get_feature_names_out()
        topics: list[dict[str, Any]] = []
        for tid, comp in enumerate(lda.components_):
            top_idx = np.argsort(comp)[::-1][:top_words]
            topics.append({"topic_id": tid, "keywords": [str(vocab[i]) for i in top_idx], "weight": round(float(comp.sum()), 4)})
        return topics

    def cluster_documents(self, texts: list[str], *, n_clusters: int = 3) -> dict[str, Any]:
        """Cluster documents in dense semantic embedding space and label each cluster with top terms."""
        if len(texts) < 2:
            return {"clusters": [], "assignments": []}
        embs, backend = self.encode_embeddings(texts)
        k = max(2, min(int(n_clusters), len(texts)))
        labels = KMeans(n_clusters=k, n_init="auto", random_state=self.random_seed).fit_predict(embs)
        clusters: list[dict[str, Any]] = []
        for cid in range(k):
            members = [texts[i] for i in range(len(texts)) if int(labels[i]) == cid]
            words = Counter(w for m in members for w in re.findall(r"[a-zA-Z]{4,}", m.lower()))
            top_kw = [w for w, _ in words.most_common(6)]
            clusters.append({"cluster_id": cid, "size": len(members), "top_terms": top_kw, "sample_doc": members[0][:140] if members else ""})
        return {"backend": backend, "clusters": clusters, "assignments": [int(c) for c in labels]}

    def find_near_duplicates(self, texts: list[str], *, threshold: float = 0.78) -> list[dict[str, Any]]:
        """Detect near-duplicate documents via pairwise cosine similarity."""
        if len(texts) < 2:
            return []
        embs, _ = self.encode_embeddings(texts)
        sim = cosine_similarity(embs)
        pairs: list[dict[str, Any]] = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                if float(sim[i, j]) >= threshold:
                    pairs.append({"doc_a": i, "doc_b": j, "similarity": round(float(sim[i, j]), 5), "preview_a": texts[i][:90], "preview_b": texts[j][:90]})
        pairs.sort(key=lambda r: r["similarity"], reverse=True)
        return pairs

    def summarize(self, text: str, *, max_sentences: int = 3) -> dict[str, Any]:
        """Summarize text via optional pretrained HuggingFace seq2seq model or TF-IDF TextRank centrality."""
        clean = str(text or "").strip()
        if not clean:
            return {"backend": "native_textrank", "summary": "", "sentences_selected": 0}

        if self.use_pretrained and is_package_available("transformers"):
            try:
                from transformers import pipeline

                if "summarizer" not in self._hf_pipelines:
                    self._hf_pipelines["summarizer"] = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
                res = self._hf_pipelines["summarizer"](clean[:3000], max_length=130, min_length=25, do_sample=False)
                if res:
                    return {"backend": "transformers:distilbart-cnn-12-6", "summary": str(res[0].get("summary_text", "")), "sentences_selected": max_sentences}
            except Exception as exc:
                logger.warning(
                    "nlp_backend_failed",
                    operation="summarization",
                    backend="transformers",
                    error_type=type(exc).__name__,
                    fallback="native_textrank",
                )

        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if len(s.strip()) > 10]
        if len(sents) <= max_sentences:
            return {"backend": "native_textrank", "summary": " ".join(sents) if sents else clean[:400], "sentences_selected": len(sents)}

        vec = TfidfVectorizer(max_features=1024, stop_words="english")
        try:
            mat = vec.fit_transform(sents)
            sim = cosine_similarity(mat)
            np.fill_diagonal(sim, 0.0)
            scores = sim.sum(axis=1)
        except Exception:
            scores = np.array([len(s) for s in sents], dtype=float)
        top_idx = sorted(np.argsort(-scores)[:max_sentences].tolist())
        selected = [sents[i] for i in top_idx]
        return {"backend": "native_textrank", "summary": " ".join(selected), "sentences_selected": len(selected)}

