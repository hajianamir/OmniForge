"""Workbench chart payload persistence and scope invalidation through the API."""

from __future__ import annotations

import io
import uuid

import pandas as pd
from fastapi.testclient import TestClient

from omniforge.api.auth import get_api_key
from omniforge.api.main import create_app
from omniforge.api.workbench import _sid
from omniforge.core.types import Principal
from omniforge.ui.state import get_active_split_frames, get_session_state


def _client() -> tuple[TestClient, dict[str, str]]:
    return TestClient(create_app(), headers={"X-API-Key": get_api_key()}), {"X-Session-ID": f"charts-{uuid.uuid4().hex}"}


def _upload(client: TestClient, headers: dict[str, str]) -> None:
    df = pd.DataFrame({
        "alpha": range(80), "beta": [(i * 7) % 31 for i in range(80)],
        "constant": [2] * 80, "target": [i * 3 + i % 4 for i in range(80)],
    })
    res = client.post(
        "/api/workbench/s01/upload",
        headers=headers,
        files={"file": ("chart-data.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")},
    )
    assert res.status_code == 200, res.text


def test_real_chart_payloads_roundtrip_and_do_not_cross_sessions() -> None:
    client, headers = _client()
    _upload(client, headers)
    eda = client.post("/api/workbench/s04/eda", headers=headers)
    assert eda.status_code == 200, eda.text
    eda_payload = eda.json()
    assert eda_payload["distributions"] and eda_payload["correlation_matrix"]["values"]
    matrix = eda_payload["correlation_matrix"]
    constant = matrix["columns"].index("constant")
    assert matrix["values"][constant][constant] is None  # undefined, not a zero correlation
    assert "s04_eda" in eda_payload["state"]["cached_result_keys"]

    manifold = client.post(
        "/api/workbench/s02/mining", headers=headers,
        json={"mode": "custom", "tasks": ["manifold"], "manifold_method": "svd"},
    )
    assert manifold.status_code == 200, manifold.text
    projection = manifold.json()["manifold"]
    assert projection["method"].lower() == "svd"
    assert projection["scatter"]["x"]
    assert projection["total_explained_pct"] == sum(projection["explained_variance_pct"])

    saved = client.post("/api/workbench/workspace/save", headers=headers, data={"project_name": "charts"})
    assert saved.status_code == 200, saved.text
    other = {"X-Session-ID": f"charts-{uuid.uuid4().hex}"}
    assert client.get("/api/workbench/results", headers=other).json()["results"] == {}
    restored = client.post(
        "/api/workbench/workspace/load", headers=other,
        files={"file": ("charts.omniforge", io.BytesIO(saved.content), "application/octet-stream")},
    )
    assert restored.status_code == 200, restored.text
    results = client.get("/api/workbench/results", headers=other).json()["results"]
    assert results["s04_eda"]["distributions"] == eda_payload["distributions"]
    assert results["s02_mining"]["manifold"] == projection

    changed = client.post("/api/workbench/exclusions", headers=other, json={"excluded_columns": ["alpha"]})
    assert changed.status_code == 200, changed.text
    assert "s04_eda" not in changed.json()["state"]["cached_result_keys"]
    assert "s02_mining" not in changed.json()["state"]["cached_result_keys"]
    assert "s04_eda" not in client.get("/api/workbench/results", headers=other).json()["results"]
    # Invalidating the restored session does not alter the original session.
    assert "s04_eda" in client.get("/api/workbench/results", headers=headers).json()["results"]


def test_advanced_mining_roundtrips_and_is_invalidated_with_its_study_data() -> None:
    client, headers = _client()
    _upload(client, headers)
    response = client.post("/api/workbench/s02/advanced_mining", headers=headers, json={})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["target"] == "target"
    assert "cluster_stability_ari" in body
    assert "s02_advanced_mining" in body["state"]["cached_result_keys"]
    original = {key: value for key, value in body.items() if key != "state"}
    assert client.get("/api/workbench/results", headers=headers).json()["results"]["s02_advanced_mining"] == original

    # This analysis uses the full study data, not the locked train/test split.
    locked = client.post(
        "/api/workbench/split-lock", headers=headers,
        json={"strategy": "random", "test_size": 0.2, "cv_folds": 3},
    )
    assert locked.status_code == 200, locked.text
    assert "s02_advanced_mining" in locked.json()["state"]["cached_result_keys"]
    saved = client.post("/api/workbench/workspace/save", headers=headers, data={"project_name": "advanced-mining"})
    assert saved.status_code == 200, saved.text
    other = {"X-Session-ID": f"charts-{uuid.uuid4().hex}"}
    restored = client.post(
        "/api/workbench/workspace/load", headers=other,
        files={"file": ("advanced.omniforge", io.BytesIO(saved.content), "application/octet-stream")},
    )
    assert restored.status_code == 200, restored.text
    assert client.get("/api/workbench/results", headers=other).json()["results"]["s02_advanced_mining"] == original
    changed = client.post("/api/workbench/exclusions", headers=other, json={"excluded_columns": ["alpha"]})
    assert changed.status_code == 200, changed.text
    assert "s02_advanced_mining" not in changed.json()["state"]["cached_result_keys"]
    assert "s02_advanced_mining" not in client.get("/api/workbench/results", headers=other).json()["results"]


def test_feature_engineering_reports_real_split_provenance() -> None:
    client, headers = _client()
    _upload(client, headers)
    req = {"mode": "custom", "correct_skew": False, "add_interactions": False, "add_polynomials": False, "commit": False}
    unlocked = client.post("/api/workbench/s05/engineer", headers=headers, json=req)
    assert unlocked.status_code == 200, unlocked.text
    assert unlocked.json()["split_locked"] is False
    assert "not active" in unlocked.json()["leakage_warning"].lower()

    lock = client.post("/api/workbench/split-lock", headers=headers, json={"strategy": "random", "test_size": 0.2, "cv_folds": 3})
    assert lock.status_code == 200, lock.text
    assert "s05_engineer" not in lock.json()["state"]["cached_result_keys"]
    locked = client.post("/api/workbench/s05/engineer", headers=headers, json=req)
    assert locked.status_code == 200, locked.text
    assert locked.json()["split_locked"] is True
    assert locked.json()["leakage_warning"] is None
    unlock = client.delete("/api/workbench/split-lock", headers=headers)
    assert unlock.status_code == 200, unlock.text
    assert "s05_engineer" not in unlock.json()["state"]["cached_result_keys"]


def test_committed_engineering_invalidates_old_study_charts_and_rerun_uses_new_columns() -> None:
    client, headers = _client()
    _upload(client, headers)
    assert client.post("/api/workbench/s04/eda", headers=headers).status_code == 200
    mining = client.post(
        "/api/workbench/s02/mining", headers=headers,
        json={"mode": "custom", "tasks": ["manifold"], "manifold_method": "pca"},
    )
    assert mining.status_code == 200, mining.text
    selection = client.post(
        "/api/workbench/s05/select", headers=headers,
        json={"mode": "custom", "method": "f_test", "top_k": 1, "commit": False},
    )
    assert selection.status_code == 200, selection.text
    engineering = client.post(
        "/api/workbench/s05/engineer", headers=headers,
        json={"mode": "custom", "correct_skew": False, "add_interactions": False,
              "add_polynomials": True, "add_datetime_features": False, "commit": True},
    )
    assert engineering.status_code == 200, engineering.text
    generated = engineering.json()["new_features"]
    assert generated
    keys = engineering.json()["state"]["cached_result_keys"]
    assert "s05_engineer" in keys
    assert not {"s02_mining", "s04_eda", "s05_select"} & set(keys)
    principal = Principal(subject="api_key_principal", roles=["admin"], tenant_id="default-tenant")
    state = get_session_state(_sid(headers["X-Session-ID"], principal))
    assert state["uap_eda_report"] is None and state["uap_mining_result"] is None
    assert state["uap_fs_scores"] is None

    refreshed = client.post("/api/workbench/s04/eda", headers=headers)
    assert refreshed.status_code == 200, refreshed.text
    assert set(generated) & {item["column"] for item in refreshed.json()["distributions"]}
    assert client.get("/api/workbench/results", headers=headers).json()["results"]["s04_eda"]["distributions"] == refreshed.json()["distributions"]


def test_noop_selection_does_not_discard_existing_charts() -> None:
    client, headers = _client()
    _upload(client, headers)
    assert client.post("/api/workbench/s04/eda", headers=headers).status_code == 200
    selected = client.post(
        "/api/workbench/s05/select", headers=headers,
        json={"mode": "custom", "method": "f_test", "top_k": 10, "commit": True},
    )
    assert selected.status_code == 200, selected.text
    assert selected.json()["state"]["columns"] == ["alpha", "beta", "constant", "target"]
    assert selected.json()["state"]["can_undo"] is False
    assert {"s04_eda", "s05_select"} <= set(selected.json()["state"]["cached_result_keys"])


def test_committed_selection_is_cached_and_preserves_locked_row_membership() -> None:
    client, headers = _client()
    _upload(client, headers)
    assert client.post("/api/workbench/s04/eda", headers=headers).status_code == 200
    locked = client.post(
        "/api/workbench/split-lock", headers=headers,
        json={"strategy": "random", "test_size": 0.2, "cv_folds": 3},
    )
    assert locked.status_code == 200, locked.text
    principal = Principal(subject="api_key_principal", roles=["admin"], tenant_id="default-tenant")
    state = get_session_state(_sid(headers["X-Session-ID"], principal))
    train_before, test_before, _ = get_active_split_frames(state)
    assert test_before is not None

    selected = client.post(
        "/api/workbench/s05/select", headers=headers,
        json={"mode": "custom", "method": "f_test", "top_k": 1, "commit": True},
    )
    assert selected.status_code == 200, selected.text
    payload = selected.json()
    assert payload["selected_features"]
    assert "s05_select" in payload["state"]["cached_result_keys"]
    assert "s04_eda" not in payload["state"]["cached_result_keys"]
    assert state["uap_fs_selected_cols"] == payload["selected_features"]
    assert state["uap_fs_scores"] is not None
    assert list(state["uap_df"].columns) != ["alpha", "beta", "constant", "target"]
    train_after, test_after, lock = get_active_split_frames(state)
    assert test_after is not None and lock is not None
    assert set(train_after["target"]) == set(train_before["target"])
    assert set(test_after["target"]) == set(test_before["target"])
    assert client.get("/api/workbench/results", headers=headers).json()["results"]["s05_select"]["rankings"] == payload["rankings"]

    saved = client.post("/api/workbench/workspace/save", headers=headers, data={"project_name": "feature-selection"})
    assert saved.status_code == 200, saved.text
    other = {"X-Session-ID": f"charts-{uuid.uuid4().hex}"}
    restored = client.post(
        "/api/workbench/workspace/load", headers=other,
        files={"file": ("selection.omniforge", io.BytesIO(saved.content), "application/octet-stream")},
    )
    assert restored.status_code == 200, restored.text
    assert restored.json()["state"]["split_lock"]["active"] is True
    assert client.get("/api/workbench/results", headers=other).json()["results"]["s05_select"]["rankings"] == payload["rankings"]


def test_standalone_nlp_results_do_not_depend_on_workspace_dataset() -> None:
    client, headers = _client()
    nlp = client.post(
        "/api/workbench/s09/nlp", headers=headers,
        json={"task": "qa", "custom_texts": ["The river is blue."], "question": "What color is the river?"},
    )
    assert nlp.status_code == 200, nlp.text
    assert not nlp.json()["state"]["has_dataset"]
    original = client.get("/api/workbench/results", headers=headers).json()["results"]["s09_nlp"]
    _upload(client, headers)
    assert client.get("/api/workbench/results", headers=headers).json()["results"]["s09_nlp"] == original
    changed = client.post("/api/workbench/exclusions", headers=headers, json={"excluded_columns": ["alpha"]})
    assert changed.status_code == 200, changed.text
    assert client.get("/api/workbench/results", headers=headers).json()["results"]["s09_nlp"] == original


def test_supervisory_current_evaluation_and_models_are_invalidated_but_history_is_historical() -> None:
    client, headers = _client()
    _upload(client, headers)
    trained = client.post("/api/workbench/s14/quick-train", headers=headers, json={"method": "ridge"})
    assert trained.status_code == 200, trained.text
    assert trained.json()["models"]
    evaluated = client.post(
        "/api/workbench/s14/evaluate-demo-batch", headers=headers,
        json={"n_rows": 20, "inject_fault_pct": 10, "noise_scale": 0.01},
    )
    assert evaluated.status_code == 200, evaluated.text
    assert evaluated.json()["trend_series"]
    status = client.get("/api/workbench/s14/models", headers=headers).json()
    assert status["models"] and status["last_evaluation"] and len(status["history"]) == 1

    locked = client.post(
        "/api/workbench/split-lock", headers=headers,
        json={"strategy": "random", "test_size": 0.2, "cv_folds": 3},
    )
    assert locked.status_code == 200, locked.text
    status = client.get("/api/workbench/s14/models", headers=headers).json()
    assert status["models"] == [] and status["last_evaluation"] is None
    assert len(status["history"]) == 1  # past evaluation, not a current chart or model
    assert "s14_supervisory" not in client.get("/api/workbench/results", headers=headers).json()["results"]

    # Historical entries remain available in the ledger after a new evaluation,
    # and the actual latest chart survives a signed save/restore.
    assert client.post("/api/workbench/s14/quick-train", headers=headers, json={"method": "ridge"}).status_code == 200
    latest = client.post(
        "/api/workbench/s14/evaluate-demo-batch", headers=headers,
        json={"n_rows": 15, "inject_fault_pct": 0, "noise_scale": 0},
    )
    assert latest.status_code == 200, latest.text
    assert len(latest.json()["history"]) == 2
    saved = client.post("/api/workbench/workspace/save", headers=headers, data={"project_name": "monitor"})
    assert saved.status_code == 200, saved.text
    other = {"X-Session-ID": f"charts-{uuid.uuid4().hex}"}
    restored = client.post(
        "/api/workbench/workspace/load", headers=other,
        files={"file": ("monitor.omniforge", io.BytesIO(saved.content), "application/octet-stream")},
    )
    assert restored.status_code == 200, restored.text
    restored_result = client.get("/api/workbench/results", headers=other).json()["results"]["s14_supervisory"]
    assert restored_result["trend_series"] == latest.json()["trend_series"]
    assert restored_result["history"] == latest.json()["history"]
    changed = client.post("/api/workbench/exclusions", headers=other, json={"excluded_columns": ["alpha"]})
    assert changed.status_code == 200, changed.text
    status = client.get("/api/workbench/s14/models", headers=other).json()
    assert status["last_evaluation"] is None and status["models"] == []
    assert "s14_supervisory" not in client.get("/api/workbench/results", headers=other).json()["results"]


def test_committed_screening_previews_match_server_cache_and_are_marked_historical() -> None:
    client, headers = _client()
    _upload(client, headers)
    locked = client.post(
        "/api/workbench/split-lock", headers=headers,
        json={"strategy": "random", "test_size": 0.2, "cv_folds": 3},
    )
    assert locked.status_code == 200, locked.text
    principal = Principal(subject="api_key_principal", roles=["admin"], tenant_id="default-tenant")
    state = get_session_state(_sid(headers["X-Session-ID"], principal))
    train_before, test_before, _ = get_active_split_frames(state)
    assert test_before is not None
    original_train, original_test = set(train_before["alpha"]), set(test_before["alpha"])
    clustering = client.post(
        "/api/workbench/s02/clustering", headers=headers,
        json={"mode": "custom", "algorithm": "kmeans", "n_clusters": 3, "run_auto_k": False},
    )
    assert clustering.status_code == 200, clustering.text
    original = clustering.json()
    assert original["scatter"]["x"]
    anomaly = client.post("/api/workbench/s02/anomaly", headers=headers, json={"mode": "custom", "method": "mad"})
    assert anomaly.status_code == 200, anomaly.text
    mining = client.post(
        "/api/workbench/s02/mining", headers=headers,
        json={"mode": "custom", "tasks": ["manifold"], "manifold_method": "pca"},
    )
    assert mining.status_code == 200, mining.text
    cluster_id = min(original["cluster_choices"], key=lambda choice: choice["count"])["cluster_id"]
    committed = client.post(
        "/api/workbench/s02/clustering/commit", headers=headers,
        json={"mode": "remove_clusters", "remove_cluster_ids": [cluster_id], "attach_cluster_id": False},
    )
    assert committed.status_code == 200, committed.text
    result = committed.json()["result"]
    assert result["provenance"] == "COMMITTED" and result["committed_action"]
    assert result["scatter"] == original["scatter"]  # explicitly labelled pre-commit preview
    assert result["n_clusters"] == original["n_clusters"]
    assert cluster_id not in [choice["cluster_id"] for choice in result["cluster_choices"]]
    cached = client.get("/api/workbench/results", headers=headers).json()["results"]
    assert cached["s02_clustering"] == result
    assert "s02_anomaly" not in cached and "s02_mining" not in cached
    survivors = set(state["uap_df"]["alpha"])
    train, test, lock = get_active_split_frames(state)
    assert lock is not None and test is not None
    assert set(train["alpha"]) == original_train & survivors
    assert set(test["alpha"]) == original_test & survivors

    anomaly = client.post("/api/workbench/s02/anomaly", headers=headers, json={"mode": "custom", "method": "mad"})
    assert anomaly.status_code == 200, anomaly.text
    original_anomaly = anomaly.json()
    removed = client.post("/api/workbench/s02/anomaly/commit", headers=headers, json={"mode": "remove_anomalies"})
    assert removed.status_code == 200, removed.text
    result = removed.json()["result"]
    assert result["committed_mode"] == "remove_anomalies" and result["committed_action"]
    assert result["n_anomalies"] == original_anomaly["n_anomalies"]
    assert result["scatter"] == original_anomaly["scatter"]
    cached = client.get("/api/workbench/results", headers=headers).json()["results"]
    assert cached["s02_anomaly"] == result
    assert "s02_clustering" not in cached
    survivors = set(state["uap_df"]["alpha"])
    train, test, lock = get_active_split_frames(state)
    assert lock is not None and test is not None
    assert set(train["alpha"]) == original_train & survivors
    assert set(test["alpha"]) == original_test & survivors
    assert state["uap_clustering_result"] is None
    assert state["uap_s02_cluster_committed"] is True  # provenance is still retained


def test_messy_excel_workbook_produces_charts_across_studios() -> None:
    """Real-world Excel sheets with title banners, formatted numbers, and sentinels produce valid studio chart payloads."""
    client, headers = _client()
    rows = [
        ["Q4 Enterprise Telemetry Report", None, None, None, None],
        [None, None, None, None, None],
        ["Region", "Revenue_USD", "Operating_Cost", "Defect_Rate", "Risk_Band"],
    ]
    for i in range(45):
        rev = "-" if i == 3 else f"${1200 + i * 35:,.2f}"
        cost = "#N/A" if i == 7 else f"{640 + (i % 11) * 28:,.1f}"
        defect = f"{1.5 + (i % 6) * 0.8:.2f}%"
        band = ["Low", "Medium", "High"][i % 3]
        region = ["North", "South", "East", "West"][i % 4]
        rows.append([region, rev, cost, defect, band])

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        pd.DataFrame({"note": ["cover"]}).to_excel(writer, sheet_name="Cover", index=False)
        pd.DataFrame(rows).to_excel(writer, sheet_name="Telemetry_Data", index=False, header=False)
    excel_bytes = excel_buf.getvalue()

    up = client.post(
        "/api/workbench/s01/upload",
        headers=headers,
        data={"sheet_name": "Telemetry_Data"},
        files={"file": ("messy_report.xlsx", io.BytesIO(excel_bytes), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert up.status_code == 200, up.text
    ws_state = up.json()["state"]
    assert ws_state["n_rows"] == 45
    assert set(["Revenue_USD", "Operating_Cost", "Defect_Rate"]).issubset(set(ws_state["numeric_columns"]))

    # Studio 00: relationships + correlation matrix
    r00 = client.get("/api/workbench/s00/data_relationships", headers=headers)
    assert r00.status_code == 200, r00.text
    d00 = r00.json()
    assert len(d00["pairs"]) > 0
    assert len(d00["target_relationships"]) > 0
    assert len(d00["correlation_matrix"]["columns"]) >= 3

    # Studio 02: clustering, anomaly, manifold scatter
    r02_cl = client.post("/api/workbench/s02/clustering", headers=headers, json={"mode": "auto", "n_clusters": 3})
    assert r02_cl.status_code == 200, r02_cl.text
    assert r02_cl.json()["scatter"]["rendered_points"] == 45

    r02_an = client.post("/api/workbench/s02/anomaly", headers=headers, json={"mode": "auto", "contamination": 0.1})
    assert r02_an.status_code == 200, r02_an.text
    assert r02_an.json()["scatter"]["rendered_points"] == 45

    # Studio 04: numeric distributions + categorical distributions + missingness + correlation matrix
    r04 = client.post("/api/workbench/s04/eda", headers=headers)
    assert r04.status_code == 200, r04.text
    d04 = r04.json()
    assert len(d04["distributions"]) >= 3
    assert len(d04["categorical_distributions"]) >= 1
    assert len(d04["missing_bars"]) >= 1
    assert len(d04["correlation_matrix"]["columns"]) >= 3

    # Studio 08: optimize & simulate cache keys + histogram
    r08_opt = client.post("/api/workbench/s08/optimize", headers=headers, json={})
    assert r08_opt.status_code == 200, r08_opt.text
    assert "s08_optimize" in r08_opt.json()["state"]["cached_result_keys"]
    assert r08_opt.json()["optimization"]["solution"]

    r08_sim = client.post("/api/workbench/s08/simulate", headers=headers, json={"target_col": "Revenue_USD", "n_simulations": 300})
    assert r08_sim.status_code == 200, r08_sim.text
    assert "s08_simulate" in r08_sim.json()["state"]["cached_result_keys"]
    assert len(r08_sim.json()["simulation"]["histogram"]) >= 10


def test_purely_categorical_dataset_still_renders_eda_and_scatter_charts() -> None:
    """Datasets with zero native numeric columns still produce Studio 00, 02, and 04 chart payloads."""
    client, headers = _client()
    df = pd.DataFrame(
        {
            "department": [["Ops", "Sales", "Eng", "Finance"][i % 4] for i in range(40)],
            "shift": [["Day", "Swing", "Night"][i % 3] for i in range(40)],
            "tier": [["Gold", "Silver", "Bronze"][(i * 2) % 3] for i in range(40)],
            "outcome": ["Pass" if i % 3 else "Fail" for i in range(40)],
        }
    )
    up = client.post(
        "/api/workbench/s01/upload",
        headers=headers,
        files={"file": ("categorical.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")},
    )
    assert up.status_code == 200, up.text

    r00 = client.get("/api/workbench/s00/data_relationships", headers=headers)
    assert r00.status_code == 200 and len(r00.json()["pairs"]) > 0

    r02_cl = client.post("/api/workbench/s02/clustering", headers=headers, json={"mode": "custom", "algorithm": "kmeans", "n_clusters": 3})
    assert r02_cl.status_code == 200 and r02_cl.json()["scatter"]["rendered_points"] == 40

    r02_an = client.post("/api/workbench/s02/anomaly", headers=headers, json={"mode": "custom", "method": "isolation_forest"})
    assert r02_an.status_code == 200 and r02_an.json()["scatter"]["rendered_points"] == 40

    r02_mn = client.post("/api/workbench/s02/mining", headers=headers, json={"mode": "custom", "tasks": ["manifold"], "manifold_method": "pca"})
    assert r02_mn.status_code == 200 and r02_mn.json()["manifold"]["scatter"]["rendered_points"] == 40

    r04 = client.post("/api/workbench/s04/eda", headers=headers)
    assert r04.status_code == 200 and len(r04.json()["categorical_distributions"]) == 4


def test_compiled_static_spa_ships_studio_charts_across_lifecycle() -> None:
    """Compiled static_app chunks include interactive chart titles across Studios 00-11."""
    from omniforge.ui.app import get_static_app_dir

    assets_dir = get_static_app_dir() / "assets"
    combined_js = "\n".join(p.read_text(encoding="utf-8") for p in sorted(assets_dir.glob("*.js")))
    expected_markers = [
        "Target Correlation Profile",
        "6D Intake Quality & Readiness Profile",
        "Cleaning Remediation Impact",
        "Ranked Baseline Model Performance",
        "Category Breakdown:",
        "AutoML Candidate Leaderboard",
        "Neural Training Loss Convergence Curves",
        "Partial Dependence Curve (PDP)",
        "Optimal Markowitz Portfolio Allocation Weights",
        "Monte Carlo Simulated Outcome Distribution",
        "Kaplan-Meier Survival Probability Curve S(t)",
        "DAG Stage Execution Telemetry",
        "Feature Population Stability Index (PSI) Drift Profile",
    ]
    for marker in expected_markers:
        assert marker in combined_js, f"Missing compiled chart marker in static_app bundle: {marker}"

