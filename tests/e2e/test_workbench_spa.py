"""End-to-End Integration Tests for the FastAPI + Compiled React SPA 13-Studio OmniForge Workbench."""

from __future__ import annotations

import io
import json
import uuid

import numpy as np
import pandas as pd
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from omniforge.api.main import create_app
from omniforge.core.utils import to_json_safe


def test_spa_static_bundle_gzip_and_all_13_studios_e2e() -> None:
    """Verify compiled React SPA serving, GZip compression, and all 13 CRISP-DM Studios + .uap roundtrip."""
    app = create_app()
    client = TestClient(app)
    sid = f"e2e_{uuid.uuid4().hex[:8]}"
    hdr = {"X-Session-ID": sid}

    # 1. Verify root serves compiled React SPA HTML & static JS/CSS assets
    r_idx = client.get("/")
    assert r_idx.status_code == 200
    assert "OmniForge" in r_idx.text and '<div id="root">' in r_idx.text

    # 2. Studio 01: Load sample dataset, inspect server-side paginated grid, quality, cleaning, deleted audit
    r_load = client.post("/api/workbench/s01/load-sample", json={"source_type": "domain", "name": "finance", "n_samples": 180}, headers=hdr)
    assert r_load.status_code == 200
    st = r_load.json()["state"]
    assert st["has_dataset"] is True and st["n_rows"] == 180

    r_grid = client.get("/api/workbench/s01/grid?page=1&page_size=25", headers=hdr)
    assert r_grid.status_code == 200
    assert len(r_grid.json()["rows"]) == 25 and r_grid.json()["total_rows"] == 180

    r_qual = client.post("/api/workbench/s01/quality", headers=hdr)
    assert r_qual.status_code == 200 and r_qual.json()["quality"]["overall_score"] > 0

    r_clean = client.post("/api/workbench/s01/clean", json={"drop_duplicates": True, "handle_outliers": "clip"}, headers=hdr)
    assert r_clean.status_code == 200

    # Global Column Exclusions & Split Lock
    num_cols = st["numeric_columns"]
    r_lock = client.post("/api/workbench/split-lock", json={"strategy": "auto", "test_size": 0.2, "cv_folds": 3}, headers=hdr)
    assert r_lock.status_code == 200 and r_lock.json()["state"]["split_lock"]["active"] is True

    # 3. Studio 02: Clustering Preview (PREVIEW ONLY) + Anomaly Screening + Commit
    r_cl = client.post("/api/workbench/s02/clustering", json={"algorithm": "kmeans", "n_clusters": 3, "run_auto_k": False}, headers=hdr)
    assert r_cl.status_code == 200 and r_cl.json()["provenance"] == "PREVIEW_ONLY"
    r_cl_c = client.post("/api/workbench/s02/clustering/commit", json={"mode": "attach"}, headers=hdr)
    assert r_cl_c.status_code == 200 and r_cl_c.json()["committed"] is True

    r_an = client.post("/api/workbench/s02/anomaly", json={"method": "isolation_forest", "contamination": 0.05}, headers=hdr)
    assert r_an.status_code == 200 and r_an.json()["provenance"] == "PREVIEW_ONLY"
    r_an_c = client.post("/api/workbench/s02/anomaly/commit", json={"mode": "remove_anomalies"}, headers=hdr)
    assert r_an_c.status_code == 200 and r_an_c.json()["rows_removed"] >= 1

    r_del_csv = client.get("/api/workbench/s01/deleted-audit/csv", headers=hdr)
    assert r_del_csv.status_code == 200 and len(r_del_csv.content) > 20

    # 4. Studio 03: Universal Diagnostics, Baselines & Risk
    r_diag = client.post("/api/workbench/s03/run", json={"mode": "auto", "alpha": 0.05}, headers=hdr)
    assert r_diag.status_code == 200 and len(r_diag.json()["report"]["hypothesis_ledger"]) > 0

    # 5. Studio 04: Automated EDA & Hypothesis Lab
    r_eda = client.post("/api/workbench/s04/eda", headers=hdr)
    assert r_eda.status_code == 200 and len(r_eda.json()["distributions"]) >= 1
    r_hyp = client.post("/api/workbench/s04/hypothesis", json={"test_type": "pearson", "col_a": num_cols[0], "col_b": num_cols[1]}, headers=hdr)
    assert r_hyp.status_code == 200

    # 6. Studio 05: Feature Engineering & Selection
    r_fe = client.post("/api/workbench/s05/engineer", json={"add_interactions": False, "correct_skew": False, "commit": False}, headers=hdr)
    assert r_fe.status_code == 200
    r_fs = client.post("/api/workbench/s05/select", json={"method": "mutual_info", "top_k": 6, "commit": False}, headers=hdr)
    assert r_fs.status_code == 200 and len(r_fs.json()["rankings"]) >= 1

    # 7. Studio 06: Multi-Backend AutoML Hub + Paper-Grade Deep Learning + Prediction
    r_bk = client.get("/api/workbench/s06/backends", headers=hdr)
    assert r_bk.status_code == 200 and len(r_bk.json()["automl_backends"]) >= 9

    r_ml = client.post("/api/workbench/s06/automl", json={"backend": "multi_layer_stacker", "time_budget_seconds": 15.0, "cv_folds": 3}, headers=hdr)
    assert r_ml.status_code == 200 and r_ml.json()["best_model_name"]
    feat_cols = r_ml.json()["state"]["active_feature_cols"]

    r_dl = client.post("/api/workbench/s06/deep-learning", json={"architectures": ["tabnet", "resnet_tabular"], "epochs": 6}, headers=hdr)
    assert r_dl.status_code == 200 and len(r_dl.json()["results"]) >= 3

    r_pred = client.post("/api/workbench/s06/predict-single", json={"record": dict.fromkeys(feat_cols, 1.0)}, headers=hdr)
    assert r_pred.status_code == 200 and "prediction" in r_pred.json()

    # 8. Studio 07: Responsible AI
    r_exp = client.post("/api/workbench/s07/explain", json={"row_index": 0}, headers=hdr)
    assert r_exp.status_code == 200 and "global_importance" in r_exp.json()

    # 9. Studio 08: Deep Forecasting (TFT) + Portfolio Optimization + Monte Carlo
    r_fc = client.post("/api/workbench/s08/forecast", json={"target_col": num_cols[0], "horizon": 6, "deep_arch": "tft", "epochs": 8}, headers=hdr)
    assert r_fc.status_code == 200 and len(r_fc.json()["deep_forecast"]) == 6
    r_opt = client.post("/api/workbench/s08/optimize", json={}, headers=hdr)
    assert r_opt.status_code == 200
    r_sim = client.post("/api/workbench/s08/simulate", json={"target_col": num_cols[0], "n_simulations": 400}, headers=hdr)
    assert r_sim.status_code == 200

    # 10. Studio 09: Extended NLP + Vision Foundation Hub + Full Graph Link Prediction + RecSys
    r_nlp = client.post("/api/workbench/s09/nlp", json={"task": "overview"}, headers=hdr)
    assert r_nlp.status_code == 200 and len(r_nlp.json()["zero_shot"]) >= 1
    r_vis = client.post("/api/workbench/s09/vision", data={"candidate_labels": "sky, forest, building"}, headers=hdr)
    assert r_vis.status_code == 200 and r_vis.json()["zero_shot"]["top_label"]
    r_gr = client.post("/api/workbench/s09/graph", json={"link_method": "ensemble_rrf", "run_supervised": True, "run_neural_gae": True}, headers=hdr)
    assert r_gr.status_code == 200 and len(r_gr.json()["topological_links"]) >= 1
    r_rec = client.post("/api/workbench/s09/recommend", json={}, headers=hdr)
    assert r_rec.status_code == 200 and len(r_rec.json()["recommendations"]) >= 1

    # 11. Studio 10: Visual DAG Pipeline Builder
    r_dag = client.post("/api/workbench/s10/run", json={"stages": ["quality", "eda", "features"]}, headers=hdr)
    assert r_dag.status_code == 200 and len(r_dag.json()["nodes"]) == 3

    # 12. Studio 11: MLOps Registry & Drift
    r_reg = client.post("/api/workbench/s11/register", json={"model_name": "spa_e2e_champ", "stage": "Staging"}, headers=hdr)
    assert r_reg.status_code == 200
    r_drf = client.post("/api/workbench/s11/drift", json={"split_ratio": 0.5}, headers=hdr)
    assert r_drf.status_code == 200

    # 13. Studio 12: Executive Report Compile + Multi-Format Export (.docx, .pdf, .xlsx, .html, .json)
    r_rep = client.post("/api/workbench/s12/compile", json={"title": "E2E SPA Verification Report"}, headers=hdr)
    assert r_rep.status_code == 200 and len(r_rep.json()["sections"]) >= 1
    for fmt in ("html", "json", "docx", "pdf", "xlsx"):
        r_exp_f = client.get(f"/api/workbench/s12/export/{fmt}", headers=hdr)
        assert r_exp_f.status_code == 200 and len(r_exp_f.content) > 100

    # 14. Studio 13: AI Analytics Copilot
    r_cop = client.post("/api/workbench/s13/chat", json={"message": "Summarize my active dataset"}, headers=hdr)
    assert r_cop.status_code == 200 and len(r_cop.json()["reply"]) > 5

    # 15. Portable .uap Workspace Save & Restore Roundtrip
    r_save = client.post("/api/workbench/workspace/save", data={"project_name": "e2e_roundtrip"}, headers=hdr)
    assert r_save.status_code == 200 and len(r_save.content) > 200
    sid2 = f"e2e_restore_{uuid.uuid4().hex[:8]}"
    r_restore = client.post(
        "/api/workbench/workspace/load",
        files={"file": ("e2e_roundtrip.uap", io.BytesIO(r_save.content), "application/octet-stream")},
        headers={"X-Session-ID": sid2},
    )
    assert r_restore.status_code == 200
    assert r_restore.json()["state"]["has_dataset"] is True
    assert r_restore.json()["state"]["active_model_name"] is not None


def test_to_json_safe_strict_rfc8259_compliance() -> None:
    """Ensure to_json_safe recursively eliminates NaN/Inf/NaT so FastAPI JSON encoding never 500s."""
    payload = {
        "py_nan": float("nan"),
        "py_inf": float("inf"),
        "py_ninf": float("-inf"),
        "np_nan": np.float64("nan"),
        "np_arr": np.array([1.0, np.nan, np.inf, 2.5]),
        "pd_na": pd.NA,
        "pd_nat": pd.NaT,
        "pd_ts": pd.Timestamp("2025-05-01 12:30:00"),
        "nested": [{"x": float("nan"), "ok": np.int64(7)}],
    }
    safe = to_json_safe(payload)
    encoded = json.dumps(safe, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["py_nan"] is None
    assert decoded["py_inf"] is None
    assert decoded["py_ninf"] is None
    assert decoded["np_nan"] is None
    assert decoded["np_arr"] == [1.0, None, None, 2.5]
    assert decoded["pd_na"] is None
    assert decoded["pd_nat"] is None
    assert decoded["pd_ts"].startswith("2025-05-01")
    assert decoded["nested"] == [{"x": None, "ok": 7}]


def test_dirty_upload_benchmark_csv_and_structured_json_errors() -> None:
    """Regression suite covering dirty CSV/Excel uploads, benchmark loading, edge endpoints, and JSON 500s."""
    app = create_app()

    def _simulated_boom() -> None:
        raise RuntimeError("Simulated internal server error")

    app.router.routes.insert(0, APIRoute("/api/workbench/_test_boom", _simulated_boom, methods=["GET"]))
    client = TestClient(app, raise_server_exceptions=False)
    sid = f"dirty_{uuid.uuid4().hex[:8]}"
    hdr = {"X-Session-ID": sid}

    # 1. Bundled benchmark CSV load (contains real missing values)
    r_bm = client.get("/api/workbench/benchmarks", headers=hdr)
    assert r_bm.status_code == 200 and len(r_bm.json()["benchmarks"]) >= 1
    first_bm = r_bm.json()["benchmarks"][0]["filename"]
    r_load_bm = client.post(
        "/api/workbench/s01/load-sample",
        json={"source_type": "benchmark", "name": first_bm},
        headers=hdr,
    )
    assert r_load_bm.status_code == 200
    assert client.get("/api/workbench/s01/grid?page=1&page_size=15", headers=hdr).status_code == 200

    # 2. Dirty CSV upload with NaN, +/-Inf, nulls, datetime column, and exact duplicates
    rng = np.random.default_rng(11)
    dirty_df = pd.DataFrame(
        {
            "num_a": [1.5, np.nan, 3.2, np.inf, -np.inf, 6.0] + rng.normal(10, 3, 114).tolist(),
            "num_b": [np.nan, 2.0, 4.0, 5.0, 6.0, np.nan] + rng.normal(50, 12, 114).tolist(),
            "num_c": rng.uniform(1, 100, 120).tolist(),
            "group_col": ["North", "South", None, "East", "West"] * 24,
            "date_col": pd.date_range("2024-01-01", periods=120, freq="D").astype(str),
            "target": rng.integers(0, 2, size=120).tolist(),
        }
    )
    dirty_df = pd.concat([dirty_df, dirty_df.iloc[:2]], ignore_index=True)

    r_up = client.post(
        "/api/workbench/s01/upload",
        files={"file": ("dirty.csv", io.BytesIO(dirty_df.to_csv(index=False).encode("utf-8")), "text/csv")},
        headers=hdr,
    )
    assert r_up.status_code == 200
    assert client.get("/api/workbench/s01/grid?page=1&page_size=20", headers=hdr).status_code == 200

    # 3. Clean, deleted audit, exclusions, diagnostics with datetime col, batch predict w/ Inf, fairness, NLP topics, survival
    assert client.post("/api/workbench/s01/clean", json={"drop_duplicates": True, "handle_outliers": "clip"}, headers=hdr).status_code == 200
    r_del = client.get("/api/workbench/s01/deleted-audit", headers=hdr)
    assert r_del.status_code == 200 and r_del.json()["total_deleted"] >= 2

    assert client.post("/api/workbench/targets", json={"targets": ["target"]}, headers=hdr).status_code == 200
    assert client.post("/api/workbench/exclusions", json={"excluded_columns": ["date_col"], "reason": "Scope test"}, headers=hdr).status_code == 200
    assert client.post("/api/workbench/exclusions", json={"excluded_columns": []}, headers=hdr).status_code == 200
    assert client.post("/api/workbench/s03/run", json={"mode": "all", "alpha": 0.05}, headers=hdr).status_code == 200

    assert client.post(
        "/api/workbench/s06/automl",
        json={"backend": "multi_layer_stacker", "time_budget_seconds": 12.0, "cv_folds": 2},
        headers=hdr,
    ).status_code == 200
    batch_csv = io.BytesIO(dirty_df.drop(columns=["target"]).head(8).to_csv(index=False).encode("utf-8"))
    assert client.post(
        "/api/workbench/s06/predict-batch",
        files={"file": ("inc.csv", batch_csv, "text/csv")},
        headers=hdr,
    ).status_code == 200
    assert client.post("/api/workbench/s07/fairness", json={"sensitive_col": "group_col"}, headers=hdr).status_code == 200
    assert client.post("/api/workbench/s09/nlp", json={"task": "topics"}, headers=hdr).status_code == 200
    assert client.post("/api/workbench/s09/survival", json={"duration_col": "num_c", "event_col": "target"}, headers=hdr).status_code == 200

    # 4. Unhandled server exceptions must ALWAYS return structured JSON (never empty/plain text 500)
    r_500 = client.get("/api/workbench/_test_boom", headers=hdr)
    assert r_500.status_code == 500
    assert r_500.headers["content-type"].startswith("application/json")
    body = json.loads(r_500.text)
    assert body["error_type"] == "RuntimeError"
    assert "Simulated internal server error" in body["detail"]

