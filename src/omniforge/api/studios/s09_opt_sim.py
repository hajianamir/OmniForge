"""Studio 09 — Specialized AI (Extended NLP, Vision Foundation Hub, Graph Link Prediction, RecSys, Survival).

Offline-first implementation: every route works without network access or GPU.
Optional deep backends (transformers / sentence-transformers / torch / open_clip / timm)
are probed and used only when installed; otherwise calibrated native fallbacks run.
"""

from __future__ import annotations

import io
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from omniforge.api.studios.schemas import TwinControlReq
from omniforge.simulation import ClosedLoopMPCController, build_default_twin

router: APIRouter = APIRouter(tags=["workbench-s09"])

_DEFAULT_NLP_TEXTS: list[str] = [
    "Quarterly revenue grew 18% with strong customer trust and reliable cloud delivery.",
    "Security alert: user john.doe@example.com reported a potential fraud breach from 192.168.1.44.",
    "Customer churn increased after terrible service outage and unacceptable billing delays.",
]
_DEFAULT_LABELS: list[str] = ["Finance", "Security", "Operations", "Customer Support"]
_DEFAULT_QUESTION = "What grew by 18%?"
_MAX_UPLOAD_BYTES = 15 * 1024 * 1024


def _wb() -> Any:
    from omniforge.api import workbench as wb

    return wb


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class NLPReq(BaseModel):
    task: str = Field(default="overview", description="overview|zero_shot|qa|semantic_search|emotions|pii_toxicity|topics|cluster")
    custom_texts: list[str] = Field(default_factory=list)
    candidate_labels: list[str] = Field(default_factory=list)
    question: str = ""
    query: str = ""


class GraphReq(BaseModel):
    source_col: str | None = None
    target_col: str | None = None
    link_method: str = "ensemble_rrf"
    edge_operator: str = "hadamard"
    run_supervised: bool = True
    run_neural_gae: bool = True
    top_k: int = Field(default=12, ge=1, le=50)


class RecommendReq(BaseModel):
    top_k: int = Field(default=12, ge=1, le=25)


class SurvivalReq(BaseModel):
    duration_col: str | None = None
    event_col: str | None = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _clean_texts(req_texts: list[str]) -> list[str]:
    texts = [str(t).strip() for t in (req_texts or []) if str(t).strip()]
    return texts if texts else list(_DEFAULT_NLP_TEXTS)


def _clean_labels(req_labels: list[str]) -> list[str]:
    labels = [str(lb).strip() for lb in (req_labels or []) if str(lb).strip()]
    return labels if labels else list(_DEFAULT_LABELS)


def _nlp_suite() -> Any:
    from omniforge.models.nlp.advanced_nlp import AdvancedNLPSuite

    return AdvancedNLPSuite(random_seed=42, use_pretrained=False)


def _nlp_backends() -> list[dict[str, Any]]:
    try:
        from omniforge.models.nlp.advanced_nlp import probe_nlp_backends

        return list(probe_nlp_backends())
    except Exception as exc:  # pragma: no cover - defensive
        return [{"id": "native_nlp", "name": "OmniForge Advanced NLP Suite (Offline Always-On)", "available": True, "version": "unknown", "probe_error": str(exc)}]


def _vision_hub() -> Any:
    from omniforge.models.computer_vision.foundation import VisionFoundationHub

    return VisionFoundationHub(image_size=(224, 224), random_seed=42)


def _vision_backends() -> dict[str, Any]:
    try:
        from omniforge.models.computer_vision.foundation import probe_vision_backends

        return dict(probe_vision_backends())
    except Exception as exc:  # pragma: no cover - defensive
        return {"device": "cpu", "backends": [{"id": "native_pyramid", "name": "OmniForge Spatial-Pyramid Descriptor (Offline Always-On)", "available": True, "version": "unknown", "probe_error": str(exc)}]}


def _graph_suite() -> Any:
    from omniforge.models.graphs.link_prediction import GraphLinkPredictionSuite

    return GraphLinkPredictionSuite(random_seed=42)


def _safe(section: str, fn: Any, fallback: Any) -> Any:
    """Run an analysis section, degrading to a structured fallback instead of HTTP 500."""
    try:
        return fn()
    except Exception as exc:
        return {**(fallback if isinstance(fallback, dict) else {"data": fallback}), "section": section, "error": f"{type(exc).__name__}: {exc}"}


def _synthetic_benchmark_scene() -> np.ndarray:
    """Deterministic 256x256 RGB benchmark scene (sky gradient + sun + grass + building + tree)."""
    from PIL import Image, ImageDraw

    w, h = 256, 256
    horizon = int(h * 0.55)
    img = Image.new("RGB", (w, h))
    px = img.load()
    if px is None:  # pragma: no cover - PIL always exposes pixel access for new images
        raise HTTPException(status_code=500, detail="Failed to synthesize benchmark scene.")
    for y in range(h):
        if y < horizon:
            t = y / max(horizon, 1)
            col = (int(74 + (191 - 74) * t), int(144 + (227 - 144) * t), int(217 + (255 - 217) * t))
        else:
            t = (y - horizon) / max(h - horizon, 1)
            col = (int(66 + (38 - 66) * t), int(138 + (74 - 138) * t), int(66 + (38 - 66) * t))
        for x in range(w):
            px[x, y] = col
    draw = ImageDraw.Draw(img)
    # Sun
    draw.ellipse((196, 22, 240, 66), fill=(255, 238, 158))
    # Clouds
    for cx, cy, cw in ((30, 34, 64), (96, 62, 78), (150, 28, 54)):
        draw.ellipse((cx, cy, cx + cw, cy + 16), fill=(244, 248, 255))
        draw.ellipse((cx + cw // 4, cy - 8, cx + (3 * cw) // 4, cy + 12), fill=(244, 248, 255))
    # Grass tufts
    for gx in range(8, w, 13):
        draw.polygon([(gx, h), (gx + 4, h - 14), (gx + 8, h)], fill=(48, 110, 48))
    # Building with window grid
    bx0, by0, bx1, by1 = 148, 96, 236, 214
    draw.rectangle((bx0, by0, bx1, by1), fill=(126, 126, 136), outline=(64, 64, 72), width=2)
    for r in range(4):
        for c in range(4):
            wx = bx0 + 8 + c * 20
            wy = by0 + 10 + r * 26
            lit = (r * 4 + c) % 3 == 0
            draw.rectangle((wx, wy, wx + 10, wy + 12), fill=(255, 240, 170) if lit else (32, 38, 56))
    # Tree
    draw.rectangle((36, 150, 46, 214), fill=(84, 56, 34))
    draw.ellipse((12, 96, 72, 164), fill=(34, 96, 44))
    draw.ellipse((26, 78, 58, 122), fill=(44, 116, 52))
    arr: np.ndarray = np.asarray(img, dtype=np.uint8)
    return arr


_MAX_IMAGE_PIXELS = 40_000_000  # ~40 MP decoded cap; blocks decompression bombs before allocation


def _decode_image_upload(file_bytes: bytes, filename: str | None) -> np.ndarray:
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(file_bytes))
        # Inspect dimensions from the header ONLY; decoding happens after the pixel-count guard.
        width, height = int(img.width or 0), int(img.height or 0)
        if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=400, detail=f"Image dimensions {width}x{height} exceed the decoded-pixel limit.")
        img.load()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Uploaded file is not a decodable image: {type(exc).__name__}.") from exc
    rgb = img.convert("RGB")
    if rgb.width > 4096 or rgb.height > 4096:
        rgb.thumbnail((4096, 4096))
    return np.asarray(rgb, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Backends probe
# ---------------------------------------------------------------------------
@router.get("/s09/backends")
def s09_backends() -> dict[str, Any]:
    """Probe optional deep NLP / Vision backends plus always-on native engines."""
    nlp = _nlp_backends()
    vision = _vision_backends()
    return {"nlp": nlp, "text": nlp, "vision": vision}


# ---------------------------------------------------------------------------
# Extended NLP
# ---------------------------------------------------------------------------
@router.post("/s09/nlp")
def s09_nlp(req: NLPReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    wb = _wb()
    sid = wb._sid(x_session_id)
    state = wb.get_session_state(sid)

    texts = _clean_texts(req.custom_texts)
    labels = _clean_labels(req.candidate_labels)
    question = (req.question or "").strip() or _DEFAULT_QUESTION
    query = (req.query or "").strip() or question
    task = (req.task or "overview").strip().lower()
    suite = _nlp_suite()
    context = " ".join(texts)

    def _zero_shot() -> Any:
        return suite.zero_shot_classify(texts, labels)

    def _qa() -> Any:
        return suite.answer_question(question, context)

    def _search() -> Any:
        return suite.semantic_search(query, texts)

    def _emotions() -> Any:
        return suite.detect_emotions_and_aspects(texts)

    def _pii() -> Any:
        return suite.screen_toxicity_and_pii(texts)

    def _topics() -> Any:
        return suite.extract_topics_lda(texts, n_topics=4, top_words=8)

    def _cluster() -> Any:
        cluster = suite.cluster_documents(texts, n_clusters=min(3, len(texts)))
        cluster["near_duplicates"] = suite.find_near_duplicates(texts, threshold=0.78)
        return cluster

    result: dict[str, Any] = {
        "task": task,
        "texts_analyzed": len(texts),
        "backends": _nlp_backends(),
    }

    if task == "overview":
        # Every sub-analysis is isolated: one failure must not take down the suite.
        result["zero_shot"] = _safe("zero_shot", _zero_shot, [])
        result["qa"] = _safe("qa", _qa, {})
        result["semantic_search"] = _safe("semantic_search", _search, [])
        result["emotions"] = _safe("emotions", _emotions, [])
        result["pii_toxicity"] = _safe("pii_toxicity", _pii, [])
        result["topics"] = _safe("topics", _topics, [])
        result["cluster"] = _safe("cluster", _cluster, {})
        result["language"] = _safe("language", lambda: suite.detect_language(texts), [])
    elif task == "zero_shot":
        result["zero_shot"] = _safe("zero_shot", _zero_shot, [])
    elif task == "qa":
        result["qa"] = _safe("qa", _qa, {})
    elif task == "semantic_search":
        result["semantic_search"] = _safe("semantic_search", _search, [])
    elif task == "emotions":
        result["emotions"] = _safe("emotions", _emotions, [])
    elif task == "pii_toxicity":
        result["pii_toxicity"] = _safe("pii_toxicity", _pii, [])
    elif task == "topics":
        result["topics"] = _safe("topics", _topics, [])
    elif task == "cluster":
        result["cluster"] = _safe("cluster", _cluster, {})
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown NLP task '{task}'. Expected one of: overview, zero_shot, qa, semantic_search, emotions, pii_toxicity, topics, cluster.",
        )

    # The overview contract requires a non-empty zero_shot section; enforce it loudly.
    if task == "overview" and not result.get("zero_shot"):
        result["zero_shot"] = [{"text": texts[0][:140], "top_label": labels[0], "confidence": 0.0, "scores": {lb: 0.0 for lb in labels}, "backend": "unavailable"}]

    wb.record_lineage(state, "Studio 09 NLP", f"Ran NLP task '{task}' on {len(texts)} document(s)", "Backends: native/optional probed")
    payload: dict[str, Any] = wb._cache_and_respond(state, sid, "s09_nlp", result)
    return payload


# ---------------------------------------------------------------------------
# Vision Foundation Hub
# ---------------------------------------------------------------------------
@router.post("/s09/vision")
async def s09_vision(
    candidate_labels: str = Form(default="sky, forest, building, industrial, document, dark"),
    n_segments: int = Form(default=4),
    file: UploadFile | None = File(default=None),
    x_session_id: str | None = Header(default="default"),
) -> dict[str, Any]:
    wb = _wb()
    sid = wb._sid(x_session_id)
    state = wb.get_session_state(sid)

    labels = [lb.strip() for lb in str(candidate_labels or "").split(",") if lb.strip()] or ["sky", "forest", "building"]
    n_segments = int(np.clip(int(n_segments), 2, 12))

    synthetic = file is None
    if file is not None:
        raw = await wb.read_upload_bounded(file, max_bytes=_MAX_UPLOAD_BYTES)
        rgb = _decode_image_upload(raw, file.filename)
        filename = file.filename
    else:
        rgb = _synthetic_benchmark_scene()
        filename = None

    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    hub = _vision_hub()

    def _zero_shot() -> Any:
        return hub.zero_shot_classify(rgb, labels)

    def _detect() -> Any:
        return hub.detect_objects(rgb, confidence_threshold=0.25)

    def _segment() -> Any:
        return hub.segment_image(rgb, n_segments=n_segments)

    def _embedding() -> Any:
        res = hub.extract_foundation_embedding(rgb, pretrained=False)
        emb = res.get("embedding")
        dim = int(res.get("embedding_dim", 0) or 0)
        norm = float(np.linalg.norm(emb)) if emb is not None and dim else 0.0
        return {"backend": res.get("backend"), "dim": dim, "norm": round(norm, 6), "image_shape": [h, w, 3]}

    result = {
        "image": {"width": w, "height": h, "mode": "RGB", "synthetic": bool(synthetic), "filename": filename},
        "zero_shot": _safe("zero_shot", _zero_shot, {"backend": "unavailable", "predictions": [], "top_label": None}),
        "detections": _safe("detections", _detect, {"backend": "unavailable", "n_detections": 0, "detections": []}),
        "segmentation": _safe("segmentation", _segment, {"backend": "unavailable", "n_segments": 0, "regions": []}),
        "embedding": _safe("embedding", _embedding, {"backend": "unavailable", "dim": 0, "norm": 0.0, "image_shape": [h, w, 3]}),
        "backends": _vision_backends(),
    }

    wb.record_lineage(state, "Studio 09 Vision", f"Vision hub run on {'synthetic benchmark scene' if synthetic else (filename or 'uploaded image')} ({w}x{h})", "Zero-shot/detect/segment/embed")
    payload: dict[str, Any] = wb._cache_and_respond(state, sid, "s09_vision", result)
    return payload


# ---------------------------------------------------------------------------
# Graph link prediction & structure
# ---------------------------------------------------------------------------
def _pick_edge_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    """Choose two node columns for an edge list. Returns (col_a, col_b, strategy).

    If the low-cardinality pair would produce a complete graph (every possible
    edge already present, so there are no non-edges to score for link
    prediction), raise ``ValueError`` so the caller falls back to the
    correlation-graph strategy instead of returning an empty link list.
    """
    low_card: list[str] = []
    for c in df.columns:
        nun = int(df[c].nunique(dropna=True))
        is_texty = pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(df[c])
        if 1 < nun <= 50 and (is_texty or nun <= 20):
            low_card.append(c)
    if len(low_card) >= 2:
        col_a, col_b = low_card[0], low_card[1]
        edges = df[[col_a, col_b]].dropna()
        g = nx.Graph()
        for _, row in edges.iterrows():
            u, v = str(row[col_a]).strip(), str(row[col_b]).strip()
            if u and v and u != v:
                g.add_edge(u, v)
        if g.number_of_nodes() >= 2:
            max_edges = g.number_of_nodes() * (g.number_of_nodes() - 1) // 2
            if g.number_of_edges() >= max_edges:
                raise ValueError("complete_graph_no_nonedges")
        return col_a, col_b, "low_cardinality_columns"
    raise ValueError("no_low_cardinality_pair")


def _correlation_graph(df: pd.DataFrame, min_edges: int = 6) -> tuple[nx.Graph, list[dict[str, Any]], str, str]:
    """Fallback graph for all-numeric frames: nodes = numeric columns, edges = strongest |Pearson r| pairs."""
    from omniforge.core.exceptions import ModelError

    num = df.select_dtypes(include=[np.number])
    num = num.replace([np.inf, -np.inf], np.nan)
    cols = [c for c in num.columns if num[c].dropna().nunique() > 1]
    if len(cols) < 2:
        raise ModelError("Correlation-graph fallback requires at least two non-constant numeric columns.")
    corr = num[cols].corr(numeric_only=True).abs()
    pairs = sorted(
        ((float(corr.loc[a, b]), a, b) for i, a in enumerate(cols) for b in cols[i + 1 :]),
        key=lambda item: item[0],
        reverse=True,
    )
    g = nx.Graph()
    g.add_nodes_from(str(c) for c in cols)
    picked: list[dict[str, Any]] = []
    total_pairs = len(pairs)
    target_edges = min(max(min_edges, len(cols) // 2), total_pairs)
    for r_abs, a, b in pairs:
        if len(picked) >= target_edges:
            break
        g.add_edge(str(a), str(b), weight=round(r_abs, 6))
        picked.append({"source": str(a), "target": str(b), "abs_corr": round(r_abs, 6)})
    return g, picked, "Abs-Pearson correlation graph", "correlation_top_pairs"


def _build_graph(df: pd.DataFrame, source_col: str | None, target_col: str | None) -> tuple[nx.Graph, dict[str, Any]]:
    """Build the analysis graph, auto-detecting edge columns when not provided."""
    if (source_col is None) != (target_col is None):
        raise HTTPException(status_code=400, detail="Provide both source_col and target_col, or neither to auto-detect.")
    if source_col is not None and target_col is not None:
        for c in (source_col, target_col):
            if c not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{c}' not found in active dataset.")
        if source_col == target_col:
            raise HTTPException(status_code=400, detail="source_col and target_col must be different columns.")
        edges = df[[source_col, target_col]].dropna()
        g = nx.Graph()
        self_loops = 0
        for _, row in edges.iterrows():
            u, v = str(row[source_col]).strip(), str(row[target_col]).strip()
            if not u or not v:
                continue
            if u == v:
                self_loops += 1
                continue
            g.add_edge(u, v)
        if g.number_of_nodes() < 2 or g.number_of_edges() < 1:
            raise HTTPException(status_code=400, detail="Edge-list columns produced no usable edges (all rows empty or self-loops).")
        meta = {
            "strategy": "explicit_edge_columns",
            "source_col": source_col,
            "target_col": target_col,
            "rows_used": int(len(edges)),
            "self_loops_dropped": int(self_loops),
        }
        return g, meta
    try:
        col_a, col_b, strategy = _pick_edge_columns(df)
        g, meta = _build_graph(df, col_a, col_b)
        meta = {**meta, "strategy": strategy}
        return g, meta
    except ValueError:
        g, pairs, label, strategy = _correlation_graph(df)
        return g, {"strategy": strategy, "source_col": None, "target_col": None, "edges": pairs, "note": label}


def _graph_visualization(g: nx.Graph, communities: list[set[Any]], *, max_nodes: int = 120) -> dict[str, Any]:
    """Deterministic spring layout (seeded) for the NetworkGraphSVG component."""
    nodes = list(g.nodes())
    if len(nodes) > max_nodes:
        keep = sorted(nodes, key=lambda n: g.degree(n), reverse=True)[:max_nodes]
        g = g.subgraph(keep).copy()
        nodes = list(g.nodes())
    comm_of: dict[Any, int] = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            if node in g:
                comm_of[node] = idx
    try:
        pos = nx.spring_layout(g, seed=42)
    except Exception:
        pos = {n: (float(i), 0.0) for i, n in enumerate(nodes)}
    vis_nodes = [
        {
            "id": str(n),
            "x": round(float(pos[n][0]), 6),
            "y": round(float(pos[n][1]), 6),
            "degree": int(g.degree(n)),
            "community": int(comm_of.get(n, 0)),
        }
        for n in nodes
    ]
    vis_edges = [{"source": str(u), "target": str(v)} for u, v in g.edges()]
    return {"nodes": vis_nodes, "edges": vis_edges}


@router.post("/s09/graph")
def s09_graph(req: GraphReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    wb = _wb()
    sid = wb._sid(x_session_id)
    state = wb.get_session_state(sid)
    df = wb._require_df(state)

    suite = _graph_suite()
    graph, graph_meta = _build_graph(df, req.source_col, req.target_col)
    undir = graph.to_undirected()
    undir.remove_edges_from(nx.selfloop_edges(undir))

    structure = suite.analyze_structure_and_influence(undir, top_influencers=5)

    try:
        communities = list(nx.community.greedy_modularity_communities(undir)) if undir.number_of_edges() > 0 else [set(undir.nodes())]
    except Exception:
        communities = [set(undir.nodes())]
    community_payload = {
        "algorithm": "greedy_modularity",
        "n_communities": len(communities),
        "assignments": [{"community_id": i, "size": len(c), "members": sorted(str(n) for n in c)[:50]} for i, c in enumerate(communities)],
    }

    try:
        centrality_df = suite.compute_centralities(undir)
        centrality: Any = centrality_df.head(10).to_dict("records") if centrality_df is not None and not centrality_df.empty else []
    except Exception:
        centrality = []

    visualization = _graph_visualization(undir, communities)

    method = (req.link_method or "ensemble_rrf").strip().lower()
    topological_links = suite.predict_links_advanced(undir, method=method, top_k=req.top_k)

    supervised: dict[str, Any]
    if req.run_supervised:
        try:
            supervised = suite.train_supervised_link_predictor(undir, edge_operator=(req.edge_operator or "hadamard").lower(), classifier="rf", top_k=req.top_k)
            supervised["available"] = True
        except Exception as exc:
            supervised = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "metrics": {"roc_auc": None, "average_precision": None, "edge_operator": (req.edge_operator or "hadamard").lower(), "classifier": "rf"},
                "top_predictions": [],
            }
    else:
        supervised = {"available": False, "reason": "skipped by request", "metrics": {"roc_auc": None}, "top_predictions": []}

    neural: dict[str, Any]
    if req.run_neural_gae:
        try:
            neural = suite.predict_links_neural_gae(undir, variational=True, embedding_dim=16, epochs=40, top_k=req.top_k)
            neural["available"] = True
        except Exception as exc:
            neural = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "metrics": {"roc_auc": None, "average_precision": None},
                "top_predictions": [],
            }
    else:
        neural = {"available": False, "reason": "skipped by request", "metrics": {"roc_auc": None}, "top_predictions": []}

    result = {
        "graph": graph_meta,
        "structure": structure,
        "visualization": visualization,
        "topological_links": topological_links,
        "supervised_link_prediction": supervised,
        "neural_gae_link_prediction": neural,
        "centrality": centrality,
        "community": community_payload,
        "link_method": method,
        "edge_operator": (req.edge_operator or "hadamard").lower(),
        "n_topological_links": len(topological_links),
    }
    wb.record_lineage(state, "Studio 09 Graph", f"Graph link prediction ({method}) on {undir.number_of_nodes()} nodes / {undir.number_of_edges()} edges", f"Strategy={graph_meta.get('strategy')}")
    payload: dict[str, Any] = wb._cache_and_respond(state, sid, "s09_graph", result)
    return payload


# ---------------------------------------------------------------------------
# Hybrid recommender (dataset-grounded, offline)
# ---------------------------------------------------------------------------
def _task_profile(df: pd.DataFrame, target: str | None) -> dict[str, Any]:
    n_rows, n_cols = df.shape
    num_cols = list(df.select_dtypes(include=[np.number]).columns)
    cat_cols = [c for c in df.columns if c not in num_cols]
    task_type = None
    if target and target in df.columns:
        try:
            task_type = str(_wb()._infer_task(df[target]).value)
        except Exception:
            task_type = None
    return {"n_rows": int(n_rows), "n_cols": int(n_cols), "n_numeric": len(num_cols), "n_categorical": len(cat_cols), "task_type": task_type}


def _algorithm_catalog(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic rule-based prior over tabular algorithm suitability."""
    is_reg = profile["task_type"] == "regression"
    n_rows, n_feat = profile["n_rows"], profile["n_cols"]
    high_dim = n_feat > max(n_rows, 1)
    small_data = n_rows < 300
    big_data = n_rows >= 1000
    if is_reg:
        catalog = [
            ("gradient_boosting_regressor", 0.88, "Strong default for tabular regression; captures non-linear interactions."),
            ("ridge_regression", 0.74 if not high_dim else 0.82, "L2-regularized linear model; excels in high-dimensional settings." if high_dim else "Stable linear baseline with multicollinearity protection."),
            ("random_forest_regressor", 0.84, "Robust to scale/outliers; minimal preprocessing required."),
            ("hist_gradient_boosting", 0.90 if big_data else 0.80, "Fast binned GBDT; shines on larger samples." if big_data else "Strong GBDT; gains grow with sample size."),
            ("multi_layer_stacker", 0.85 if big_data else 0.68, "High ceiling via stacked ensembles." if big_data else "Data-hungry stacker; below its ceiling on small samples."),
            ("knn_regressor", 0.52, "Cheap local baseline; degrades in high dimensions."),
        ]
    else:
        catalog = [
            ("gradient_boosting_classifier", 0.88, "Strong default for tabular classification."),
            ("logistic_regression", 0.78 if not high_dim else 0.85, "Calibrated, interpretable linear baseline." if not high_dim else "Regularized linear model first choice in high dimensions."),
            ("random_forest_classifier", 0.84, "Robust baseline; minimal preprocessing required."),
            ("xgb_hist_classifier", 0.90 if big_data else 0.80, "Binned histogram GBDT; best on larger samples." if big_data else "Strong GBDT; prefer once n grows."),
            ("multi_layer_stacker", 0.85 if big_data else 0.66, "High ceiling stacked ensemble." if big_data else "Data-hungry stacker; capped on small samples."),
            ("naive_bayes", 0.55 if not high_dim else 0.72, "Fast baseline for high-dimensional sparse signals." if high_dim else "Independence assumption rarely holds for dense tabular data."),
            ("knn_classifier", 0.50, "Explainable local baseline; degrades with dimension."),
        ]
    if small_data:
        catalog = [(name, (score - 0.08 if "stacker" in name else score), reason) for name, score, reason in catalog]
    return [{"item": name, "category": "algorithm", "score": round(float(score), 3), "reason": reason} for name, score, reason in catalog]


def _feature_signals(df: pd.DataFrame, target: str | None, limit: int = 5) -> list[dict[str, Any]]:
    """Signal-strength ranking of predictors vs the target (|Pearson r|, factorized for low-card categoricals)."""
    if not target or target not in df.columns:
        return []
    y = df[target]
    y_num = pd.to_numeric(y, errors="coerce")
    y_is_num = bool(y_num.notna().mean() > 0.9 and y_num.dropna().nunique() > 2)
    out: list[dict[str, Any]] = []

    def _corr(a: pd.Series, b: pd.Series) -> float | None:
        joined = pd.concat([a, b], axis=1, ignore_index=True).dropna()
        if len(joined) < 10:
            return None
        x, yv = joined.iloc[:, 0].to_numpy(dtype=float), joined.iloc[:, 1].to_numpy(dtype=float)
        if float(np.std(x)) <= 1e-12 or float(np.std(yv)) <= 1e-12:
            return None
        return float(abs(np.corrcoef(x, yv)[0, 1]))

    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        try:
            if pd.api.types.is_numeric_dtype(s):
                x = pd.to_numeric(s, errors="coerce")
                if y_is_num:
                    r = _corr(x, y_num)
                    if r is not None:
                        out.append({"item": c, "category": "feature", "score": round(r, 4), "reason": f"|Pearson r| with target '{target}'"})
                else:
                    # Binary/multiclass low-unique numeric target: point-biserial via factorized target codes.
                    r = _corr(x, pd.Series(pd.factorize(y.astype(str))[0], index=y.index, dtype=float))
                    if r is not None:
                        out.append({"item": c, "category": "feature", "score": round(r, 4), "reason": f"Association with target '{target}' (factorized)"})
            else:
                nun = int(s.nunique(dropna=True))
                if 1 < nun <= 50:
                    codes = pd.Series(pd.factorize(s.astype(str))[0].astype(float), index=s.index)
                    if y_is_num:
                        r = _corr(codes, y_num)
                    else:
                        r = _corr(codes, pd.Series(pd.factorize(y.astype(str))[0].astype(float), index=y.index))
                    if r is not None:
                        out.append({"item": c, "category": "feature", "score": round(r, 4), "reason": f"Association with target '{target}' (factorized)"})
        except Exception:
            continue
    out.sort(key=lambda r: float(r["score"]), reverse=True)
    return out[:limit]


@router.post("/s09/recommend")
def s09_recommend(req: RecommendReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    wb = _wb()
    sid = wb._sid(x_session_id)
    state = wb.get_session_state(sid)
    top_k = int(req.top_k)

    df = state.get("uap_df")
    has_df = isinstance(df, pd.DataFrame) and not df.empty
    target = None
    if has_df:
        targets = wb.get_selected_targets(state, df)
        target = targets[0] if targets else None

    if has_df:
        assert isinstance(df, pd.DataFrame)
        profile = _task_profile(df, target)
        recommendations = _algorithm_catalog(profile)
        recommendations.extend(_feature_signals(df, target))
        strategy = "hybrid_rules_prior+feature_signal"
        n_rows, n_features = int(df.shape[0]), int(df.shape[1])
    else:
        profile = {"n_rows": 0, "n_cols": 0, "n_numeric": 0, "n_categorical": 0, "task_type": None}
        recommendations = _algorithm_catalog({**profile, "n_rows": 500, "n_cols": 12})
        strategy = "hybrid_rules_prior_no_dataset"
        n_rows, n_features = 0, 0

    recommendations.sort(key=lambda r: float(r["score"]), reverse=True)
    recommendations = recommendations[:top_k]
    for rank, item in enumerate(recommendations, start=1):
        item["rank"] = rank

    result = {
        "recommendations": recommendations,
        "strategy": strategy,
        "task_type": profile.get("task_type"),
        "n_rows": n_rows,
        "n_features": n_features,
        "target": target,
    }
    wb.record_lineage(state, "Studio 09 Recommend", f"Hybrid recommendations ({strategy}) for target={target or 'unset'}", f"n_rows={n_rows} n_features={n_features}")
    payload: dict[str, Any] = wb._cache_and_respond(state, sid, "s09_recommend", result)
    return payload


# ---------------------------------------------------------------------------
# Survival analysis (Kaplan-Meier + Cox PH)
# ---------------------------------------------------------------------------
@router.post("/s09/survival")
def s09_survival(req: SurvivalReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    from omniforge.core.exceptions import ModelError
    from omniforge.models.survival.survival import CoxPHModel, KaplanMeierEstimator

    wb = _wb()
    sid = wb._sid(x_session_id)
    state = wb.get_session_state(sid)
    df = wb._require_df(state)

    duration_col = (req.duration_col or "").strip() or None
    event_col = (req.event_col or "").strip() or None
    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    if not numeric_cols:
        raise HTTPException(status_code=400, detail="Survival analysis requires at least two numeric columns (duration + event).")
    duration_col = duration_col or numeric_cols[0]
    event_col = event_col or (numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
    for c in (duration_col, event_col):
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{c}' not found in active dataset.")
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise HTTPException(status_code=400, detail=f"Column '{c}' must be numeric for survival analysis.")
    if duration_col == event_col:
        raise HTTPException(status_code=400, detail="duration_col and event_col must be different columns.")

    work = df[[duration_col, event_col] + [c for c in numeric_cols if c not in (duration_col, event_col)]].copy()
    work[duration_col] = pd.to_numeric(work[duration_col], errors="coerce")
    work[event_col] = pd.to_numeric(work[event_col], errors="coerce")
    work = work.dropna(subset=[duration_col, event_col])
    n_before = int(len(work))
    work = work[np.isfinite(work[duration_col]) & np.isfinite(work[event_col])]
    work = work[work[duration_col] > 0]
    n_dropped = n_before - int(len(work))
    if len(work) < 10:
        raise HTTPException(status_code=400, detail=f"Survival analysis needs at least 10 rows with positive, finite durations after filtering (got {len(work)}).")
    if int((work[event_col] > 0.5).sum()) < 1:
        raise HTTPException(status_code=400, detail="Event column contains no positive events (values > 0.5).")

    durations = work[duration_col].to_numpy(dtype=float)
    events = (work[event_col].to_numpy(dtype=float) > 0.5).astype(int)

    km = KaplanMeierEstimator(alpha=0.05)
    try:
        km.fit(durations, events)
    except ModelError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    km_curve = km.survival_table_.to_dict("records")
    median_survival = km.median_survival_time_

    feature_cols = [c for c in work.columns if c not in (duration_col, event_col)][:8]
    cox_payload: dict[str, Any] | None
    cox_note: str | None = None
    if feature_cols:
        cox_df = work[[*feature_cols, duration_col, event_col]].copy()
        for c in feature_cols:
            cox_df[c] = pd.to_numeric(cox_df[c], errors="coerce")
        med = cox_df[feature_cols].median(numeric_only=True).fillna(0.0)
        cox_df[feature_cols] = cox_df[feature_cols].fillna(med).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        cox_df = cox_df.rename(columns={duration_col: "duration", event_col: "event"})
        try:
            cox = CoxPHModel(l2_penalty=1e-4, max_iter=100, tol=1e-7, ties="efron")
            cox.fit(cox_df, duration_col="duration", event_col="event", feature_cols=feature_cols)
            cox_payload = {
                "hazard_ratios": {str(k): round(float(v), 6) for k, v in cox.hazard_ratios_.items()},
                "hazard_ratio_ci": {str(k): [round(float(v[0]), 6), round(float(v[1]), 6)] for k, v in cox.hazard_ratio_ci_.items()},
                "p_values": {str(k): round(float(v), 6) for k, v in cox.p_values_.items()},
                "standard_errors": {name: round(float(se), 6) for name, se in zip(cox.feature_names_, cox.standard_errors_, strict=False)},
                "concordance_index": round(float(cox.concordance_index_), 6),
                "log_partial_likelihood": round(float(cox.log_partial_likelihood_), 6),
                "ties": "efron",
                "feature_cols": list(cox.feature_names_),
            }
        except Exception as exc:
            cox_payload = None
            cox_note = f"CoxPH fit skipped: {type(exc).__name__}: {exc}"
    else:
        cox_note = "No additional numeric predictor columns available for Cox PH."

    result = {
        "duration_col": duration_col,
        "event_col": event_col,
        "n_samples": int(len(work)),
        "n_events": int(events.sum()),
        "n_dropped_rows": int(n_dropped),
        "median_survival": None if median_survival is None else round(float(median_survival), 6),
        "km_curve": km_curve,
        "cox": cox_payload,
        "note": cox_note,
    }
    wb.record_lineage(state, "Studio 09 Survival", f"Kaplan-Meier + CoxPH on durations='{duration_col}' events='{event_col}'", f"n={len(work)} events={int(events.sum())}")
    payload: dict[str, Any] = wb._cache_and_respond(state, sid, "s09_survival", result)
    return payload


# --------------------------------------------------------------------------- Phase 6 C7: Closed-Loop Digital Twin MPC


@router.post("/s09/twin-control")
def s09_twin_control(
    req: TwinControlReq,
    x_session_id: str | None = Header(default=None),
) -> dict[str, Any]:
    """Run closed-loop MPC on a linear Digital Twin (receding-horizon control).

    Solves the constrained quadratic program at each step, applies the first control
    action, assimilates the next measurement, and repeats. Returns trajectory charts
    suitable for the interactive Studio 09 twin-control panel.
    """
    sid = _wb()._sid(x_session_id or "default")
    state = _wb().get_session_state(sid)

    # Build a default or state-cached twin.
    twin = state.get("uap_digital_twin")
    if twin is None or twin.A[0, 1] != req.dt:
        twin = build_default_twin(n_x=req.n_x, n_u=req.n_u, dt=req.dt)
        state["uap_digital_twin"] = twin

    controller = ClosedLoopMPCController(
        twin,
        horizon=req.horizon,
        Q=np.diag(req.Q_diag) if req.Q_diag else None,
        R=np.diag(req.R_diag) if req.R_diag else None,
        u_min=np.asarray(req.u_min, dtype=float) if req.u_min else None,
        u_max=np.asarray(req.u_max, dtype=float) if req.u_max else None,
        x_min=np.asarray(req.x_min, dtype=float) if req.x_min else None,
        x_max=np.asarray(req.x_max, dtype=float) if req.x_max else None,
        random_seed=42,
    )

    # Deterministic measurement stream if none supplied.
    if req.measurements:
        measurements = [np.asarray(m, dtype=float) for m in req.measurements]
        if len(measurements) < req.n_steps:
            # Tile to required length.
            rng = np.random.default_rng(7)
            while len(measurements) < req.n_steps:
                measurements.append(measurements[-1] + rng.normal(0, 0.05, size=req.n_x))
        measurements = measurements[: req.n_steps]
    else:
        rng = np.random.default_rng(1234)
        measurements = [rng.normal(0.0, 0.5, size=req.n_x) for _ in range(req.n_steps)]

    setpoint = None
    if req.setpoint:
        sp = np.asarray(req.setpoint, dtype=float).ravel()
        if sp.size >= req.n_x:
            setpoint = sp[: req.n_x]
        else:
            setpoint = np.pad(sp, (0, req.n_x - sp.size), mode="edge")

    report = controller.run_closed_loop(measurements, setpoint=setpoint)

    # Cache a copy of the controller for warm-start continuation.
    state["uap_mpc_controller"] = controller
    state["uap_twin_control_report"] = report.model_dump()

    result = {
        "twin_control": report.model_dump(),
        "twin_dimensions": {"n_x": req.n_x, "n_u": req.n_u},
        "horizon": req.horizon,
        "n_steps_applied": report.n_steps_applied,
        "all_constraints_satisfied": report.all_constraints_satisfied,
        "constraint_violations": report.constraint_violations,
        "total_cost": report.total_cost,
        "trajectory_chart": {
            "state_trajectory": report.state_trajectory,
            "control_history": report.control_history,
            "setpoint": report.setpoint,
            "stage_costs": report.stage_costs,
        },
    }
    wb = _wb()
    wb.record_lineage(
        state,
        "Studio 09 Twin Control",
        f"Closed-loop MPC (H={req.horizon}, {report.n_steps_applied} steps)",
        f"Constraints satisfied={report.all_constraints_satisfied}, total_cost={report.total_cost:.4f}",
    )
    payload: dict[str, Any] = wb._cache_and_respond(state, sid, "s09_twin_control", result)
    return payload
