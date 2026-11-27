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



def test_small_eda_and_hypothesis_keep_exclusions_after_cleaning() -> None:
    client, headers = _client()
    df = pd.DataFrame({"Skip Me": [None] * 4, "Load Value": [1e30] * 4, "Group Label": ["a", "b", "a", "b"]})
    uploaded = client.post(
        "/api/workbench/s01/upload", headers=headers,
        files={"file": ("small.csv", df.to_csv(index=False).encode(), "text/csv")},
    )
    assert uploaded.status_code == 200, uploaded.text
    excluded = client.post("/api/workbench/exclusions", headers=headers, json={"excluded_columns": ["Skip Me"]})
    assert excluded.json()["state"]["excluded_columns"] == ["Skip Me"]
    eda = client.post("/api/workbench/s04/eda", headers=headers)
    assert eda.status_code == 200, eda.text
    assert eda.json()["state"]["excluded_columns"] == ["Skip Me"]
    assert eda.json()["distributions"][0]["bins"] == [{"label": "1e+30", "count": 4}]
    assert eda.json()["categorical_distributions"][0]["column"] == "Group Label"

    for preview in (True, False):
        cleaned = client.post(
            "/api/workbench/s01/clean", headers=headers,
            json={"preview_only": preview, "standardize_column_names": True, "drop_duplicates": False},
        )
        assert cleaned.status_code == 200, cleaned.text
        assert cleaned.json()["state"]["excluded_columns"] == (["Skip Me"] if preview else ["skip_me"])

    for mode in ("custom", "auto", "complete"):
        hyp = client.post(
            "/api/workbench/s04/hypothesis", headers=headers,
            json={"mode": mode, "test_type": "shapiro", "col_a": "load_value"},
        )
        assert hyp.status_code == 200, hyp.text
        result = hyp.json()
        assert result["state"]["excluded_columns"] == ["skip_me"]
        if mode == "custom":
            assert result["result"]["p_value"] is None
            assert result["result"]["evaluable"] is False
        else:
            assert any(row["test_type"] == "shapiro" for row in result["skipped"])

    denied = client.post(
        "/api/workbench/s04/hypothesis", headers=headers,
        json={"mode": "custom", "test_type": "pearson", "col_a": "load_value", "col_b": "skip_me"},
    )
    assert denied.status_code == 400
    rerun = client.post("/api/workbench/s04/eda", headers=headers)
    assert rerun.status_code == 200, rerun.text
    assert rerun.json()["state"]["excluded_columns"] == ["skip_me"]
    assert "skip_me" not in [row["column"] for row in rerun.json()["missing_bars"]]


def test_studio03_benchmarks_use_the_same_locked_cohort_and_studio05_preview_is_nonmutating() -> None:
    client, headers = _client()
    _upload(client, headers)
    lock = client.post("/api/workbench/split-lock", headers=headers, json={"strategy": "random", "test_size": 0.25})
    assert lock.status_code == 200, lock.text
    diagnostics = client.post(
        "/api/workbench/s03/run", headers=headers,
        json={"mode": "custom", "selected_modules": ["baseline_benchmark"], "holdout_ratio": 0.4},
    )
    assert diagnostics.status_code == 200, diagnostics.text
    report = diagnostics.json()["report"]
    assert report["baseline_split"]["target"] == {"source": "split_lock", "training_rows": 60, "holdout_rows": 20}
    assert report["baseline_feature_cols"]["target"] == ["alpha", "beta"]
    selection = {"mode": "custom", "method": "f_test", "top_k": 1, "target": "target"}
    preview = client.post("/api/workbench/s05/select", headers=headers, json=selection)
    assert preview.status_code == 200, preview.text
    assert preview.json()["fit_scope"] == "train_only"
    assert preview.json()["selected_source_columns"] == ["alpha"]
    assert preview.json()["committed"] is False
    assert preview.json()["state"]["columns"] == ["alpha", "beta", "constant", "target"]
    committed = client.post("/api/workbench/s05/select", headers=headers, json={**selection, "commit": True})
    assert committed.status_code == 200, committed.text
    assert committed.json()["committed"] is True
    assert committed.json()["state"]["columns"] == ["alpha", "target"]
    assert committed.json()["state"]["split_lock"]["active"] is True


def test_studio04_battery_returns_finite_adjusted_results_on_larger_fixture() -> None:
    import time

    import numpy as np
    from statsmodels.stats.multitest import multipletests

    client, headers = _client()
    rng = np.random.default_rng(41)
    n = 12_000
    fixture = pd.DataFrame({
        "normal": rng.standard_normal(n), "lognormal": rng.lognormal(size=n),
        "uniform": rng.uniform(size=n), "poisson": rng.poisson(lam=7, size=n),
        "group": rng.choice(["left", "right", "middle"], size=n),
    })
    uploaded = client.post(
        "/api/workbench/s01/upload", headers=headers,
        files={"file": ("synthetic-battery.csv", io.BytesIO(fixture.to_csv(index=False).encode()), "text/csv")},
    )
    assert uploaded.status_code == 200, uploaded.text
    start = time.perf_counter()
    response = client.post("/api/workbench/s04/hypothesis", headers=headers, json={"mode": "complete"})
    elapsed = time.perf_counter() - start
    assert response.status_code == 200, response.text
    payload = response.json()
    battery = payload["battery"]
    assert payload["total_tests"] == len(battery) > 10
    rejected, adjusted, _, _ = multipletests([row["p_value"] for row in battery], method="fdr_bh")
    np.testing.assert_allclose([row["p_value_fdr"] for row in battery], adjusted)
    assert payload["significant_tests"] == int(rejected.sum())
    print(f"Studio 04 complete battery: {n:,} rows, {len(battery)} tests, API round trip {elapsed:.3f}s")


def test_studio01_preserves_temporal_gap_rows_and_split_keys_on_cleaning() -> None:
    client, headers = _client()
    n = 35
    source = pd.DataFrame({
        "Event Time": pd.date_range("2022-01-01", periods=n, freq="D"),
        "sensor reading": [float(i) if i % 3 else None for i in range(n)],
        "target": [float(i) for i in range(n)],
    })
    upload = client.post("/api/workbench/s01/upload", headers=headers, files={
        "file": ("temporal.csv", source.to_csv(index=False).encode(), "text/csv"),
    })
    assert upload.status_code == 200, upload.text
    lock = client.post("/api/workbench/split-lock", headers=headers, json={
        "strategy": "time_series", "time_col": "Event Time", "target_col": "target", "test_size": 0.2, "gap": 3,
    })
    assert lock.status_code == 200, lock.text
    assert lock.json()["state"]["split_lock"]["active"]
    cleaned = client.post("/api/workbench/s01/clean", headers=headers, json={
        "mode": "custom", "impute_numeric": "bfill", "drop_duplicates": False,
        "standardize_column_names": True,
    })
    assert cleaned.status_code == 200, cleaned.text
    body = cleaned.json()
    assert body["rows_removed"] == 0
    assert body["cleaning_report"]["rows_reserved_by_split"] == 3
    assert body["cleaning_report"]["effective_numeric_imputation"] == "causal_ffill"
    assert body["state"]["split_lock"]["active"]
    assert body["state"]["columns"] == ["Event Time", "sensor_reading", "target", "sensor_reading_is_outlier"]


def test_audit_flags_survive_saved_workspace_without_entering_baseline_or_automl() -> None:
    client, headers = _client()
    frame = pd.DataFrame({
        "signal": [None if i % 8 == 0 else i for i in range(80)],
        "signal__is_missing": [i * 2 for i in range(80)],
        "temp": [(i * 7) % 33 for i in range(80)],
        "target": [i % 2 for i in range(80)],
    })
    upload = client.post("/api/workbench/s01/upload", headers=headers, files={
        "file": ("audit-columns.csv", frame.to_csv(index=False).encode(), "text/csv"),
    })
    assert upload.status_code == 200, upload.text
    lock = client.post("/api/workbench/split-lock", headers=headers, json={"strategy": "stratified", "test_size": 0.25})
    assert lock.status_code == 200, lock.text
    clean = client.post("/api/workbench/s01/clean", headers=headers, json={
        "mode": "custom", "add_missing_indicators": True, "standardize_column_names": True,
        "drop_duplicates": False, "handle_outliers": "flag",
    })
    assert clean.status_code == 200, clean.text
    summary = clean.json()["state"]
    flags = set(summary["audit_only_columns"])
    assert "signal_2_is_missing" in flags
    assert "signal_is_outlier" in flags
    assert "signal_is_missing" not in flags
    assert not any(c.startswith("target_") for c in flags)
    assert flags <= set(summary["columns"])
    assert clean.json()["cleaning_report"]["split_lock_leakage_safe"]
    denied = client.post("/api/workbench/targets", headers=headers, json={"targets": ["signal_2_is_missing"]})
    assert denied.status_code == 400
    csv = client.get("/api/workbench/s01/export-clean/csv", headers=headers)
    assert csv.status_code == 200, csv.text
    exported = pd.read_csv(io.BytesIO(csv.content))
    assert flags <= set(exported.columns)
    assert exported["signal_is_missing"].tolist() == frame["signal__is_missing"].tolist()

    saved = client.post("/api/workbench/workspace/save", headers=headers, data={"project_name": "audit-columns"})
    assert saved.status_code == 200, saved.text
    restored_headers = {"X-Session-ID": f"audit-{uuid.uuid4().hex}"}
    restored = client.post("/api/workbench/workspace/load", headers=restored_headers, files={
        "file": ("audit-columns.omniforge", saved.content, "application/octet-stream"),
    })
    assert restored.status_code == 200, restored.text
    assert set(restored.json()["state"]["audit_only_columns"]) == flags

    baseline = client.post("/api/workbench/s03/run", headers=restored_headers, json={
        "mode": "custom", "selected_modules": ["baseline_benchmark"],
    })
    assert baseline.status_code == 200, baseline.text
    assert baseline.json()["report"]["baseline_feature_cols"]["target"] == ["signal", "signal_is_missing", "temp"]

    automl = client.post("/api/workbench/s06/automl", headers=restored_headers, json={
        "target": "target", "backend": "omniforge_native", "time_budget_seconds": 5,
    })
    assert automl.status_code == 200, automl.text
    assert automl.json()["state"]["active_input_feature_cols"] == ["signal", "signal_is_missing", "temp"]


def test_mining_requires_explicit_target_for_multiple_outcomes() -> None:
    client, headers = _client()
    frame = pd.DataFrame({
        "x": range(60), "z": [(i * 3) % 7 for i in range(60)],
        "outcome_a": [i % 3 for i in range(60)], "outcome_b": [i * 3.0 + i % 4 for i in range(60)],
    })
    upload = client.post("/api/workbench/s01/upload", headers=headers, files={
        "file": ("mining.csv", frame.to_csv(index=False).encode(), "text/csv"),
    })
    assert upload.status_code == 200, upload.text
    targets = client.post("/api/workbench/targets", headers=headers, json={"targets": ["outcome_a", "outcome_b"]})
    assert targets.status_code == 200, targets.text
    req = {"mode": "custom", "tasks": ["subgroups", "manifold"]}
    assert client.post("/api/workbench/s02/mining", headers=headers, json=req).status_code == 400
    assert client.post("/api/workbench/s02/advanced_mining", headers=headers).status_code == 400
    assert client.post("/api/workbench/s06/automl", headers=headers, json={}).status_code == 400
    assert client.post("/api/workbench/s02/mining", headers=headers, json={**req, "target": "x"}).status_code == 400
    mined = client.post("/api/workbench/s02/mining", headers=headers, json={**req, "target": "outcome_b"})
    assert mined.status_code == 200, mined.text
    assert mined.json()["target"] == "outcome_b"
    assert {row["feature"] for row in mined.json()["manifold"]["loadings"]} == {"x", "z"}
    advanced = client.post("/api/workbench/s02/advanced_mining?target_col=outcome_b", headers=headers)
    assert advanced.status_code == 200, advanced.text
    assert advanced.json()["target"] == "outcome_b"
    assert set(advanced.json()["distance_correlation"]) == {"x", "z"}


def test_diagnostics_catalog_custom_executes_real_module() -> None:
    client, headers = _client()
    _upload(client, headers)
    catalog = client.get('/api/workbench/s03/catalog', headers=headers).json()
    assert 'normality_distribution' in catalog['executable_modules']
    assert 'automated_eda' not in catalog['executable_modules']
    response = client.post('/api/workbench/s03/run', headers=headers, json={
        'mode': 'custom', 'selected_modules': ['normality_distribution'],
    })
    assert response.status_code == 200, response.text
    report = response.json()['report']
    assert report['executed_modules'] == ['normality_distribution']
    assert report['total_tests'] > 0


def test_diagnostics_rejects_silent_dispatch_fallbacks() -> None:
    client, headers = _client()
    _upload(client, headers)
    for payload in [
        {'mode': 'Custom'},
        {'mode': 'custom'},
        {'mode': 'custom', 'selected_modules': ['statistical_tests']},
        {'mode': 'custom', 'selected_modules': 'normality_distribution'},
        {'mode': 'custom', 'selected_modules': ['automated_eda']},
        {'mode': 'auto', 'selected_modules': ['baseline_benchmark']},
    ]:
        assert client.post('/api/workbench/s03/run', headers=headers, json=payload).status_code == 422, payload


def test_feature_selection_run_scope_leaves_study_and_split_intact() -> None:
    client, headers = _client()
    _upload(client, headers)
    locked = client.post('/api/workbench/split-lock', headers=headers, json={'strategy': 'random'})
    assert locked.status_code == 200, locked.text
    response = client.post('/api/workbench/s05/select', headers=headers, json={
        'target': 'target', 'method': 'f_test', 'top_k': 1, 'apply_to_run': True,
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['committed'] is False
    assert result['run_scope']['ignored_columns'] == ['beta', 'constant']
    assert result['state']['columns'] == ['alpha', 'beta', 'constant', 'target']
    assert result['state']['split_lock'] == locked.json()['state']['split_lock']
    run = client.post('/api/workbench/s03/run', headers=headers, json={
        'mode': 'custom', 'selected_modules': ['baseline_benchmark'], **result['run_scope'],
    })
    assert run.status_code == 200, run.text
    assert run.json()['report']['baseline_feature_cols'] == {'target': ['alpha']}


def test_deep_benchmark_discloses_class_mix_and_reference_provenance() -> None:
    client, headers = _client()
    frame = pd.DataFrame({'x': range(500), 'target': [0] * 450 + [1] * 50})
    uploaded = client.post('/api/workbench/s01/upload', headers=headers, files={
        'file': ('imbalanced.csv', io.BytesIO(frame.to_csv(index=False).encode()), 'text/csv'),
    })
    assert uploaded.status_code == 200, uploaded.text
    response = client.post('/api/workbench/s06/deep-learning', headers=headers, json={
        'target': 'target', 'architectures': [], 'max_rows': 100,
    })
    assert response.status_code == 200, response.text
    body = response.json()
    evaluation = body['evaluation']
    assert evaluation['holdout_sampling'] == 'stratified_without_replacement'
    assert evaluation['holdout_class_counts'] == {'0': 18, '1': 2}
    assert evaluation['available_holdout_class_counts'] == {'0': 90, '1': 10}
    assert body['baseline_evidence']['Baseline_most_frequent']['status'] == 'scored'
    analytical = [row for row in body['results'] if 'Nadaraya' in row['model']]
    assert analytical and analytical[0]['framework'] != 'sklearn'
