"""Comprehensive E2E regression tests covering all 8 OmniForge Task 33 requirements."""

from __future__ import annotations

import io
import uuid

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from omniforge.api.auth import get_api_key
from omniforge.api.main import create_app


def _make_multi_sheet_excel() -> bytes:
    """Create a real multi-sheet .xlsx workbook in memory for Excel connector & sheet-picker tests."""
    rng = np.random.default_rng(77)
    n = 120
    x1 = rng.normal(100, 15, n)
    x2 = rng.normal(40, 8, n)
    y = 2.4 * x1 - 1.5 * x2 + 25.0 + rng.normal(0, 3.0, n)
    df_train = pd.DataFrame(
        {
            "sensor_a": np.round(x1, 3),
            "sensor_b": np.round(x2, 3),
            "zone": rng.choice(["NORTH", "SOUTH", "EAST"], size=n),
            "const_col": "STATIC",
            "output_val": np.round(y, 3),
        }
    )
    # Inject real defects for automatic intake audit verification
    df_train.loc[[2, 15, 29], "sensor_a"] = np.nan
    df_train.loc[[5, 18], "sensor_b"] = 350.0  # extreme outliers
    df_train = pd.concat([df_train, df_train.iloc[[0, 1]]], ignore_index=True)

    # Second worksheet: incoming live operational batch with explicit fault excursions
    x1_live = rng.normal(100, 14, 30)
    x2_live = rng.normal(40, 7, 30)
    y_live = 2.4 * x1_live - 1.5 * x2_live + 25.0 + rng.normal(0, 2.5, 30)
    y_live[[4, 11, 22]] -= 85.0  # deliberate real-world fault excursions
    df_live = pd.DataFrame(
        {
            "sensor_a": np.round(x1_live, 3),
            "sensor_b": np.round(x2_live, 3),
            "zone": rng.choice(["NORTH", "SOUTH", "EAST"], size=30),
            "const_col": "STATIC",
            "output_val": np.round(y_live, 3),
        }
    )

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_train.to_excel(writer, sheet_name="Historical_Train", index=False)
        df_live.to_excel(writer, sheet_name="Live_Ops_Batch", index=False)
    return buf.getvalue()


def test_task33_full_omniforge_workflow_and_all_8_requirements() -> None:
    """Verify all 8 Task 33 requirements end-to-end across FastAPI workbench endpoints."""
    app = create_app()
    client = TestClient(app, headers={"X-API-Key": get_api_key()})
    hdr = {"X-Session-ID": f"task33_{uuid.uuid4().hex[:8]}"}

    # --- Req #4: OmniForge rebranding & compiled SPA served cleanly ---
    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "OmniForge" in r_home.text
    r_st = client.get("/api/workbench/state", headers=hdr)
    assert r_st.status_code == 200
    assert r_st.json()["platform"]["name"] == "OmniForge"

    # --- Req #2 & #8: Multi-sheet Excel inspection, ingestion & IMMEDIATE Automatic Intake Defect Audit ---
    xlsx_bytes = _make_multi_sheet_excel()
    r_sheets = client.post(
        "/api/workbench/s01/excel-sheets",
        files={"file": ("plant_ops.xlsx", io.BytesIO(xlsx_bytes), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        headers=hdr,
    )
    assert r_sheets.status_code == 200
    assert r_sheets.json()["sheets"] == ["Historical_Train", "Live_Ops_Batch"]

    r_up = client.post(
        "/api/workbench/s01/upload",
        data={"sheet_name": "Historical_Train"},
        files={"file": ("plant_ops.xlsx", io.BytesIO(xlsx_bytes), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        headers=hdr,
    )
    assert r_up.status_code == 200
    ia = r_up.json()["intake_audit"]
    assert ia is not None
    assert ia["defect_counts"]["total"] >= 3  # missing + duplicates + constant col + outliers
    assert len(ia["defects"]) >= 3
    assert len(ia["statistical_pretests"]) >= 2

    # --- Req #1: Restored & Expanded Data Cleaning (Dry-Run Preview + Auto / Complete / Custom + Export) ---
    r_prev = client.post("/api/workbench/s01/clean", json={"mode": "auto", "preview_only": True}, headers=hdr)
    assert r_prev.status_code == 200 and r_prev.json()["preview_only"] is True
    r_cln = client.post("/api/workbench/s01/clean", json={"mode": "auto", "preview_only": False}, headers=hdr)
    assert r_cln.status_code == 200
    assert r_cln.json()["state"]["studio_completed"]["01"] is True
    assert client.get("/api/workbench/s01/export-clean/csv", headers=hdr).status_code == 200
    assert client.get("/api/workbench/s01/export-clean/xlsx", headers=hdr).status_code == 200

    # --- Req #5: Studio 02 Clustering Bug Fix (Row deletion ACTUALLY removes rows & green LED NEVER turns off!) ---
    r_cl = client.post("/api/workbench/s02/clustering", json={"mode": "complete", "n_clusters": 4}, headers=hdr)
    assert r_cl.status_code == 200
    cl_data = r_cl.json()
    assert len(cl_data["algorithm_benchmark"]) >= 3
    choices = cl_data["cluster_choices"]
    assert len(choices) >= 2
    smallest = min(choices, key=lambda c: c["count"])
    rows_before = cl_data["state"]["n_rows"]

    # Reject empty selection on remove_clusters
    r_empty = client.post(
        "/api/workbench/s02/clustering/commit",
        json={"mode": "remove_clusters", "remove_cluster_ids": []},
        headers=hdr,
    )
    assert r_empty.status_code == 400

    # Delete smallest cluster and confirm exact row count drop + green LED stays ON
    r_del = client.post(
        "/api/workbench/s02/clustering/commit",
        json={"mode": "remove_clusters", "remove_cluster_ids": [smallest["cluster_id"]]},
        headers=hdr,
    )
    assert r_del.status_code == 200
    assert r_del.json()["rows_removed"] == smallest["count"]
    assert r_del.json()["state"]["n_rows"] == rows_before - smallest["count"]
    assert r_del.json()["state"]["studio_completed"]["02"] is True

    # --- Req #1: Restored Data Mining Lab (Association Rules, Subgroup Discovery, Manifold) ---
    r_mine = client.post(
        "/api/workbench/s02/mining",
        json={"mode": "complete", "manifold_method": "pca"},
        headers=hdr,
    )
    assert r_mine.status_code == 200
    assert set(r_mine.json()["tasks_executed"]) == {"association_rules", "subgroups", "manifold"}
    assert "scatter" in r_mine.json()["manifold"]

    # --- Req #5 & #6: Advance to Studio 03 -> Studio 02 green light STAYS ON + Ranked Baseline Equations ---
    r_diag = client.post("/api/workbench/s03/run", json={"mode": "all"}, headers=hdr)
    assert r_diag.status_code == 200
    diag_state = r_diag.json()["state"]
    assert diag_state["studio_completed"]["01"] is True
    assert diag_state["studio_completed"]["02"] is True  # BUG FIX VERIFIED!
    assert diag_state["studio_completed"]["03"] is True

    eqs = r_diag.json()["report"]["baseline_equations"]
    assert "output_val" in eqs
    ranked_eqs = eqs["output_val"]
    assert len(ranked_eqs) >= 3
    assert ranked_eqs[0]["rank"] == 1
    assert "ACCEPTED" in ranked_eqs[0]["acceptance_status"]
    assert ranked_eqs[0]["primary_score"] > 0.7
    assert ranked_eqs[0]["lift_over_naive"] > 0.5
    assert ranked_eqs[0]["selection_criterion"]
    assert len(ranked_eqs[0]["acceptance_reasons"]) >= 1

    # --- Req #1: Expanded 17-test Hypothesis Lab (Complete Battery + Distribution Fit) ---
    r_hyp = client.post("/api/workbench/s04/hypothesis", json={"mode": "complete"}, headers=hdr)
    assert r_hyp.status_code == 200 and r_hyp.json()["total_tests"] >= 8
    r_fit = client.post(
        "/api/workbench/s04/hypothesis",
        json={"mode": "custom", "test_type": "distribution_fit", "col_a": "output_val"},
        headers=hdr,
    )
    assert r_fit.status_code == 200 and len(r_fit.json()["result"]["extra_table"]) >= 3

    # --- Req #8: Studio 14 Live Prediction, Reality-vs-Baseline Tolerance Alerts & Multi-Source Connectors ---
    # Quick-train a neural network too so Baseline + Neural Net are both selectable
    r_qt = client.post("/api/workbench/s14/quick-train", json={"method": "mlp", "epochs": 8}, headers=hdr)
    assert r_qt.status_code == 200
    r_mods = client.get("/api/workbench/s14/models", headers=hdr)
    assert r_mods.status_code == 200
    models = r_mods.json()["models"]
    families = {m["family"] for m in models}
    assert "Baseline" in families
    assert any("Neural" in f for f in families)

    # Evaluate Live Excel Worksheet ('Live_Ops_Batch') with strict tolerance -> must detect the 3 injected faults!
    ridge_model = next(m for m in models if "Ridge" in m["label"])
    r_ev = client.post(
        "/api/workbench/s14/evaluate-file",
        data={
            "sheet_name": "Live_Ops_Batch",
            "model_id": ridge_model["model_id"],
            "actual_col": "output_val",
            "max_pct_error": "8.0",
            "max_sigma_z": "2.5",
        },
        files={"file": ("plant_ops.xlsx", io.BytesIO(xlsx_bytes), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        headers=hdr,
    )
    assert r_ev.status_code == 200, r_ev.text
    ev = r_ev.json()
    assert ev["system_status"] in ("CRITICAL_ALARM", "WARNING")
    assert ev["kpis"]["breach_count"] >= 3
    assert len(ev["alarms"]) >= 3
    assert len(ev["trend_series"]) == 30
    assert len(ev["spc_points"]) == 30
    assert ev["state"]["studio_completed"]["14"] is True

    # Verify SQL Connector + SQL Supervisory Evaluation too
    r_demo_sql = client.post("/api/workbench/s01/sql/demo", headers=hdr)
    assert r_demo_sql.status_code == 200
    sql_url = r_demo_sql.json()["connection_url"]
    r_sql_insp = client.post("/api/workbench/s01/sql/inspect", json={"connection_url": sql_url}, headers=hdr)
    assert r_sql_insp.status_code == 200 and len(r_sql_insp.json()["tables"]) >= 3

    # Verify Single Live Reading & Supervisory CSV/XLSX export
    r_s14_csv = client.get("/api/workbench/s14/export/csv", headers=hdr)
    r_s14_xlsx = client.get("/api/workbench/s14/export/xlsx", headers=hdr)
    assert r_s14_csv.status_code == 200 and len(r_s14_csv.content) > 200
    assert r_s14_xlsx.status_code == 200 and len(r_s14_xlsx.content) > 500

    # --- Req #7: Customizable Reporting Center (Catalog + Custom Sections + PDF, XLSX, DOCX, CSV Export) ---
    r_cat = client.get("/api/workbench/s12/catalog", headers=hdr)
    assert r_cat.status_code == 200
    sec_keys = [s["key"] for s in r_cat.json()["sections"]]
    assert "intake_audit" in sec_keys and "future_predictions" in sec_keys

    r_comp = client.post(
        "/api/workbench/s12/compile",
        json={
            "title": "OmniForge Task 33 Full Audit Report",
            "selected_sections": sec_keys,
            "include_charts": True,
            "include_tables": True,
        },
        headers=hdr,
    )
    assert r_comp.status_code == 200
    comp = r_comp.json()
    assert comp["n_charts"] >= 2 and len(comp["chart_previews"]) >= 2
    assert "Intake Defect Audit" in comp["table_names"]
    assert "Supervisory Tolerance Alarms" in comp["table_names"]

    for fmt in ("pdf", "xlsx", "docx", "csv"):
        r_exp = client.get(f"/api/workbench/s12/export/{fmt}", headers=hdr)
        assert r_exp.status_code == 200 and len(r_exp.content) > 250

    # --- Req #4: Portable .omniforge project archive export & restore ---
    r_save = client.post(
        "/api/workbench/workspace/save",
        data={"project_name": "task33_snapshot"},
        headers=hdr,
    )
    assert r_save.status_code == 200
    assert ".omniforge" in r_save.headers.get("content-disposition", "")
    r_load = client.post(
        "/api/workbench/workspace/load",
        files={"file": ("task33_snapshot.omniforge", io.BytesIO(r_save.content), "application/octet-stream")},
        headers={"X-Session-ID": f"restore_{uuid.uuid4().hex[:8]}"},
    )
    assert r_load.status_code == 200
    assert r_load.json()["state"]["has_dataset"] is True
