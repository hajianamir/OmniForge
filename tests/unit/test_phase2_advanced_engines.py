"""Unit tests for Phase 1A Performance Engine & ALL 5 Phase 2 Advanced AI/ML Engine Suites."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from omniforge.automl.backends import (
    AutoMLBackendHub,
    GeneticPipelineAutoML,
    MetaFeatureWarmStartAutoML,
    MetaLearningBayesianAutoML,
    MultiLayerStackingAutoML,
    probe_automl_backends,
)
from omniforge.core.performance import (
    TTLCache,
    compact_scatter_payload,
    dataframe_fingerprint,
    downsample_table_for_plot,
    smart_sample_df,
)
from omniforge.core.types import TaskType
from omniforge.models.computer_vision.foundation import VisionFoundationHub, probe_vision_backends
from omniforge.models.deep import (
    FTTransformerModel,
    GraphAutoencoderLinkModel,
    GraphNeuralNetworkModel,
    ResNetTabularModel,
    TabNetModel,
)
from omniforge.models.graphs.link_prediction import GraphLinkPredictionSuite
from omniforge.models.nlp.advanced_nlp import AdvancedNLPSuite, probe_nlp_backends
from omniforge.models.time_series.deep_forecast import DeepForecaster


def _make_clf_df(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    x3 = rng.uniform(-2, 2, size=n)
    x4 = x1 * 0.7 - x2 * 0.5 + rng.normal(0, 0.3, size=n)
    y = ((x1 + 0.8 * x4 - 0.5 * x2) > 0.1).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "target": y})


def test_phase1a_performance_primitives() -> None:
    df = _make_clf_df(1200)
    fp1 = dataframe_fingerprint(df)
    fp2 = dataframe_fingerprint(df.copy())
    assert fp1 == fp2
    sampled, flag = smart_sample_df(df, max_rows=300, stratify_col="target")
    assert flag is True and len(sampled) == 300
    ds = downsample_table_for_plot(df, max_rows=200)
    assert len(ds) == 200
    sc = compact_scatter_payload(df["x1"].to_numpy(), df["x2"].to_numpy(), groups=df["target"].to_numpy(), max_points=250)
    assert sc["rendered_points"] == 250 and sc["downsampled"] is True
    cache: TTLCache[int] = TTLCache(max_entries=4, ttl_seconds=60.0)
    cache.set("k1", 42)
    assert cache.get("k1") == 42


def test_phase2_1_multi_backend_automl_hub() -> None:
    backends = probe_automl_backends()
    ids = {b["id"] for b in backends}
    assert {
        "omniforge_native",
        "multi_layer_stacker",
        "genetic_pipeline",
        "meta_bayesian",
        "autogluon",
        "tpot",
        "h2o",
        "autosklearn",
        "flaml",
    } <= ids

    df = _make_clf_df(180)
    X, y = df.drop(columns=["target"]), df["target"]
    X_tr, X_va = X.iloc[:140], X.iloc[140:]
    y_tr, y_va = y.iloc[:140], y.iloc[140:]

    # 1. Native Multi-Layer Stacking + Caruana Greedy Ensemble
    stk = MultiLayerStackingAutoML(cv_folds=3)
    stk_res = stk.fit(X_tr, y_tr, X_va, y_va, task_type=TaskType.BINARY_CLASSIFICATION)
    assert stk_res["best_score"] >= 0.70
    assert len(stk_res["leaderboard"]) >= 3

    # 2. Native Genetic Pipeline Search
    gen = GeneticPipelineAutoML(population_size=6, generations=2, cv_folds=3)
    gen_res = gen.fit(X_tr, y_tr, X_va, y_va, task_type=TaskType.BINARY_CLASSIFICATION)
    assert gen_res["best_score"] >= 0.65

    # 3. Native Meta-Feature Warm-Start portfolio (formerly misnamed …Bayesian…)
    assert MetaLearningBayesianAutoML is MetaFeatureWarmStartAutoML  # deprecated alias retained
    meta = MetaFeatureWarmStartAutoML(cv_folds=3)
    meta_res = meta.fit(X_tr, y_tr, X_va, y_va, task_type=TaskType.BINARY_CLASSIFICATION)
    assert meta_res["best_score"] >= 0.65

    # 4. Hub dispatch + graceful fallback for unavailable external backend
    hub_res = AutoMLBackendHub.run(df, target_col="target", backend="multi_layer_stacker", time_budget_seconds=15.0)
    assert hub_res.best_score >= 0.65
    assert hub_res.executed_backend == "multi_layer_stacker"
    fb_res = AutoMLBackendHub.run(df, target_col="target", backend="autosklearn", time_budget_seconds=15.0)
    assert fb_res.best_model is not None and fb_res.used_fallback is True


def test_phase2_2_paper_grade_deep_learning_and_forecasting() -> None:
    df = _make_clf_df(120)
    X = df.drop(columns=["target"]).to_numpy()
    y = df["target"].to_numpy()

    # TabNet with Sparsemax + feature importance masks
    tabnet = TabNetModel(n_d=8, n_a=8, n_steps=3, epochs=8, task_type=TaskType.BINARY_CLASSIFICATION, track_carbon=False)
    tabnet.fit(X[:90], y[:90])
    preds_tn = tabnet.predict(X[90:])
    assert len(preds_tn) == 30
    masks = tabnet.explain(X[90:])
    assert masks.shape == (30, X.shape[1])

    # ResNet-Tabular
    rn = ResNetTabularModel(d_hidden=24, n_blocks=2, epochs=8, task_type=TaskType.BINARY_CLASSIFICATION, track_carbon=False)
    rn.fit(X[:90], y[:90])
    assert len(rn.predict(X[90:])) == 30

    # FT-Transformer with [CLS] token + Pre-Norm GEGLU
    ft = FTTransformerModel(d_model=16, n_heads=4, n_layers=1, epochs=6, task_type=TaskType.BINARY_CLASSIFICATION, track_carbon=False)
    ft.fit(X[:90], y[:90])
    assert len(ft.predict(X[90:])) == 30

    # GNN (GCN, GraphSAGE, Multi-Head GAT)
    for arch in ("gcn", "graphsage", "gat"):
        gnn = GraphNeuralNetworkModel(architecture=arch, hidden_dim=16, epochs=8, task_type=TaskType.BINARY_CLASSIFICATION, track_carbon=False)
        gnn.fit(X[:60], y[:60])
        assert len(gnn.predict(X[60:80])) == 20

    # Neural GAE/VGAE Link Prediction
    g = nx.karate_club_graph()
    adj = nx.to_numpy_array(g)
    gae = GraphAutoencoderLinkModel(embedding_dim=8, variational=True, epochs=15)
    emb = gae.fit_embeddings(adj)
    assert emb.shape == (g.number_of_nodes(), 8)
    probs = gae.score_pairs([(0, 1), (0, 33)])
    assert probs.shape == (2,) and np.all((probs >= 0.0) & (probs <= 1.0))

    # Deep Forecasters: N-BEATS, PatchTST, TFT (quantiles)
    t = np.arange(80, dtype=float)
    series = pd.Series(50.0 + 0.3 * t + 6.0 * np.sin(2 * np.pi * t / 12.0))
    for arch in ("nbeats", "patchtst", "tft"):
        fc = DeepForecaster(architecture=arch, context_length=16, epochs=10).fit(series)
        out = fc.predict(horizon=8)
        assert len(out) == 8 and {"step", "yhat", "yhat_lower", "yhat_upper"} <= set(out.columns)


def test_phase2_3_vision_foundation_hub() -> None:
    probe = probe_vision_backends()
    assert "backends" in probe and any(b["id"] == "native_pyramid" and b["available"] for b in probe["backends"])

    hub = VisionFoundationHub(image_size=(96, 96))
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[:45, :, :] = [50, 140, 230]
    img[45:, :, :] = [40, 160, 60]
    img[30:66, 30:66, :] = [240, 90, 60]

    zs = hub.zero_shot_classify(img, ["sky", "forest", "dark night", "industrial"])
    assert zs["top_label"] is not None and abs(sum(p["probability"] for p in zs["predictions"]) - 1.0) < 1e-3

    emb = hub.extract_foundation_embedding(img, pretrained=False)
    assert emb["embedding_dim"] >= 64

    dets = hub.detect_objects(img)
    assert "detections" in dets

    segs = hub.segment_image(img, n_segments=3)
    assert segs["n_segments"] >= 2

    imgs = [img, 255 - img, img // 2, np.roll(img, 12, axis=1), np.flip(img, axis=0), np.flip(255 - img, axis=1)]
    labels = ["day", "inv", "day", "day", "day", "inv"]
    tr = hub.train_transfer_classifier(imgs, labels)
    assert tr["n_samples"] == 6


def test_phase2_4_extended_nlp_suite() -> None:
    backends = probe_nlp_backends()
    assert any(b["id"] == "native_nlp" and b["available"] for b in backends)

    suite = AdvancedNLPSuite(use_pretrained=False)
    docs = [
        "Quarterly revenue grew 18% with strong customer trust and reliable cloud delivery.",
        "Security alert: user john.doe@example.com reported a fraud breach from 192.168.1.44.",
        "Customer churn spiked due to terrible outage and unacceptable billing errors.",
        "Quarterly revenue rose 18% driven by strong customer trust and reliable cloud systems.",
    ]
    zs = suite.zero_shot_classify(docs, ["Finance", "Security", "Customer Support"])
    assert len(zs) == 4 and zs[0]["top_label"] in {"Finance", "Security", "Customer Support"}

    qa = suite.answer_question("What grew by 18%?", " ".join(docs))
    assert "18%" in qa["answer"] and qa["confidence"] > 0.3

    ss = suite.semantic_search("fraud breach security alert", docs, top_k=2)
    assert len(ss) == 2 and ss[0]["doc_index"] == 1

    emo = suite.detect_emotions_and_aspects(docs, aspects=["revenue", "security"])
    assert len(emo) == 4 and "emotions" in emo[0]

    langs = suite.detect_language(["The quick brown fox jumps over the lazy dog", "Der schnelle braune Fuchs mit dem Hund"])
    assert langs[0]["language"] == "en" and langs[1]["language"] == "de"

    pii = suite.screen_toxicity_and_pii(docs)
    assert pii[1]["pii_count"] >= 2 and "[EMAIL_REDACTED]" in pii[1]["redacted_text"]

    lda = suite.extract_topics_lda(docs, n_topics=2)
    assert len(lda) == 2
    cl = suite.cluster_documents(docs, n_clusters=2)
    assert len(cl["clusters"]) == 2
    dups = suite.find_near_duplicates(docs, threshold=0.55)
    assert any(d["doc_a"] == 0 and d["doc_b"] == 3 for d in dups)


def test_phase2_5_full_graph_link_prediction_suite() -> None:
    suite = GraphLinkPredictionSuite()
    g = nx.karate_club_graph()

    for m in ("adamic_adar", "resource_allocation", "jaccard", "common_neighbors", "preferential_attachment", "katz", "personalized_pagerank", "lhn", "ensemble_rrf"):
        preds = suite.predict_links_advanced(g, method=m, top_k=5)
        assert len(preds) == 5

    sup = suite.train_supervised_link_predictor(g, edge_operator="hadamard", classifier="rf", top_k=5)
    assert 0.0 <= sup["metrics"]["roc_auc"] <= 1.0
    assert len(sup["top_predictions"]) == 5

    neural = suite.predict_links_neural_gae(g, variational=True, epochs=15, top_k=5)
    assert 0.0 <= neural["metrics"]["roc_auc"] <= 1.0
    assert len(neural["top_predictions"]) == 5

    struct = suite.analyze_structure_and_influence(g, top_influencers=4)
    assert struct["n_nodes"] == g.number_of_nodes() and len(struct["top_influencers"]) == 4
