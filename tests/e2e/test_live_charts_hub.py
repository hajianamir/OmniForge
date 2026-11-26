"""End-to-end tests for the On-Demand Live Professional Chart Engine & SVG/PNG/XLSX Exporter."""

from __future__ import annotations

import io
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import openpyxl
import pandas as pd
from fastapi.testclient import TestClient

from omniforge.api.auth import get_api_key
from omniforge.api.main import create_app
from omniforge.reporting.live_charts import LIVE_CHART_SPECS

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _make_client() -> tuple[TestClient, dict[str, str]]:
    client = TestClient(create_app(), headers={"X-API-Key": get_api_key()})
    headers = {"X-Session-ID": f"live-charts-{uuid.uuid4().hex}"}
    return client, headers


def _upload_csv_df(client: TestClient, headers: dict[str, str], df: pd.DataFrame, filename: str = "dataset.csv") -> None:
    res = client.post(
        "/api/workbench/s01/upload",
        headers=headers,
        files={"file": (filename, io.BytesIO(df.to_csv(index=False).encode("utf-8")), "text/csv")},
    )
    assert res.status_code == 200, res.text


def test_catalog_filters_by_studio_and_tracks_readiness() -> None:
    client, headers = _make_client()

    # Before dataset upload: has_dataset=False and all charts marked ready=False
    r0 = client.get("/api/workbench/charts/catalog?studio_id=04", headers=headers)
    assert r0.status_code == 200, r0.text
    body0 = r0.json()
    assert body0["has_dataset"] is False
    assert {c["id"] for c in body0["charts"]} == {spec.id for spec in LIVE_CHART_SPECS if "04" in spec.studios}
    assert all(c["ready"] is False for c in body0["charts"])

    # Upload a live tabular dataset
    df = pd.DataFrame(
        {
            "sensor_a": [10.0 + i * 0.7 for i in range(60)],
            "sensor_b": [50.0 - (i % 11) * 1.4 for i in range(60)],
            "zone": ["Zone-A", "Zone-B", "Zone-C"] * 20,
            "yield_target": [100.0 + i * 1.2 + (i % 5) for i in range(60)],
        }
    )
    _upload_csv_df(client, headers, df, "sensors.csv")

    r1 = client.get("/api/workbench/charts/catalog?studio_id=04", headers=headers)
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1["has_dataset"] is True
    by_id = {c["id"]: c for c in body1["charts"]}

    # Dataset-driven charts are immediately ready; model-dependent charts remain gated until training
    assert by_id["normality_qq_plot"]["ready"] is True
    assert by_id["empirical_ecdf_ks"]["ready"] is True
    assert by_id["stratified_violin_box"]["ready"] is True
    assert all(c["recommended_for_studio"] and "04" in c["studios"] for c in body1["charts"])
    report_gallery = client.get("/api/workbench/charts/catalog?studio_id=12", headers=headers).json()
    assert len(report_gallery["charts"]) == len(LIVE_CHART_SPECS)
    outside = next(spec.id for spec in LIVE_CHART_SPECS if "04" not in spec.studios)
    for endpoint, extra in (("generate", {}), ("export", {"format": "svg"})):
        denied = client.post(
            f"/api/workbench/charts/{endpoint}", headers=headers, json={"chart_id": outside, "studio_id": "s04", **extra},
        )
        assert denied.status_code == 400
        assert "outside this studio" in denied.json()["detail"]


def test_all_regression_and_dataset_chart_types_generate_and_export_svg_png_xlsx() -> None:
    client, headers = _make_client()

    # Build a rich real tabular dataset with continuous, skewed positive, and categorical columns
    rows = []
    regions = ["North", "South", "East", "West"]
    tiers = ["Enterprise", "Growth", "Starter"]
    for i in range(120):
        reg = regions[i % len(regions)]
        tier = tiers[(i * 2 + (i // 7)) % len(tiers)]
        treat = 1 if i % 2 == 0 else 0
        temp = 50.0 + (i % 19) * 2.3 + (4.0 if reg == "North" else -2.0)
        pressure = 100.0 + ((i * 7) % 29) * 1.8
        flow = 15.0 + ((i * 5) % 23) * 1.4
        defect_rate = max(0.2, 8.5 - 0.04 * temp + 0.03 * pressure + (1.5 if i % 11 == 0 else 0.0))
        revenue = 250.0 + 3.8 * temp + 2.1 * pressure + 4.5 * flow + 28.0 * treat - (18.0 if tier == "Starter" else 0.0)
        rows.append(
            {
                "region": reg if i % 17 != 0 else None,
                "tier": tier,
                "treatment": treat,
                "temperature": round(temp, 3),
                "pressure": round(pressure, 3),
                "flow_rate": round(flow, 3),
                "defect_rate": round(defect_rate, 3),
                "revenue": round(revenue, 3),
            }
        )
    _upload_csv_df(client, headers, pd.DataFrame(rows), "plant_operations.csv")

    # Configure regression target = revenue
    r_cfg = client.post("/api/workbench/targets", headers=headers, json={"targets": ["revenue"]})
    assert r_cfg.status_code == 200, r_cfg.text

    # Run upstream studios so workflow-backed charts also use real computed studio outputs
    assert client.post("/api/workbench/s01/quality", headers=headers).status_code == 200
    assert client.post("/api/workbench/s01/clean", headers=headers, json={"mode": "auto"}).status_code == 200
    assert client.post("/api/workbench/s02/clustering", headers=headers, json={"mode": "auto", "n_clusters": 3}).status_code == 200
    assert client.post("/api/workbench/s03/run", headers=headers, json={"mode": "auto"}).status_code == 200
    assert client.post("/api/workbench/s05/select", headers=headers, json={"mode": "complete", "top_k": 5}).status_code == 200
    assert client.post(
        "/api/workbench/s06/automl",
        headers=headers,
        json={"backend": "omniforge_native", "time_budget_seconds": 8.0},
    ).status_code == 200
    assert client.post("/api/workbench/s07/explain", headers=headers, json={"row_index": 2}).status_code == 200
    assert client.post("/api/workbench/s07/fairness", headers=headers, json={"sensitive_col": "region"}).status_code == 200
    assert client.post(
        "/api/workbench/s08/causal",
        headers=headers,
        json={"treatment_col": "treatment", "outcome_col": "revenue"},
    ).status_code == 200
    assert client.post("/api/workbench/s08/optimize", headers=headers, json={"risk_aversion": 1.5}).status_code == 200
    assert client.post("/api/workbench/s11/drift", headers=headers, json={}).status_code == 200

    cat = client.get("/api/workbench/charts/catalog?studio_id=all", headers=headers).json()
    ready_regression_specs = [c for c in cat["charts"] if c["ready"]]
    ready_ids = {c["id"] for c in ready_regression_specs}

    expected_ready_regression = {
        "dataset_readiness_gauge",
        "quality_6d_radar",
        "categorical_share_donut",
        "data_lifecycle_funnel",
        "stacked_composition_bar",
        "empirical_ecdf_ks",
        "stratified_violin_box",
        "normality_qq_plot",
        "ridgeline_kde_joyplot",
        "lorenz_gini_curve",
        "bivariate_regression_ci",
        "correlation_heatmap_svg",
        "multivariate_parallel_coords",
        "alluvial_sankey_flow",
        "group_centroid_radar",
        "feature_correlation_forest",
        "standardized_effect_forest",
        "feature_importance_pareto",
        "model_attribution_waterfall",
        "portfolio_allocation_donut",
        "model_residual_diagnostics",
        "bland_altman_agreement",
        "cusum_sequential_control",
        "missing_variance_pareto",
        "model_multimetric_radar",
    }
    assert expected_ready_regression.issubset(ready_ids), {cid: next(c["missing_reason"] for c in cat["charts"] if c["id"] == cid) for cid in expected_ready_regression - ready_ids}

    # Generate EVERY ready chart individually (single-chart on-demand engine) & verify payload integrity
    for spec in ready_regression_specs:
        cid = spec["id"]
        r_gen = client.post(
            "/api/workbench/charts/generate",
            headers=headers,
            json={
                "chart_id": cid,
                "studio_id": "12",
                "feature_x": "temperature" if spec["supports_feature_x"] else None,
                "feature_y": "revenue" if spec["supports_feature_y"] else None,
                "group_col": "region" if spec["supports_group_by"] else None,
            },
        )
        assert r_gen.status_code == 200, f"{cid} failed: {r_gen.text}"
        payload = r_gen.json()
        assert payload["chart_id"] == cid
        assert payload["chart_type"] == spec["chart_type"]
        assert payload["title"]
        assert isinstance(payload["data"], dict) and len(payload["data"]) > 0
        assert len(payload["table_columns"]) >= 2
        assert len(payload["table_rows"]) >= 1

    # Export one representative chart per chart_type in SVG (light & dark), PNG, and XLSX
    exported_types: set[str] = set()
    for spec in ready_regression_specs:
        ctype = spec["chart_type"]
        if ctype in exported_types:
            continue
        exported_types.add(ctype)
        cid = spec["id"]

        for theme in ("dark", "light"):
            r_svg = client.post(
                "/api/workbench/charts/export",
                headers=headers,
                json={"chart_id": cid, "format": "svg", "theme": theme},
            )
            assert r_svg.status_code == 200, f"SVG export failed for {cid} ({ctype}): {r_svg.text}"
            assert "image/svg+xml" in r_svg.headers["content-type"]
            ET.fromstring(r_svg.content)  # Must be strictly well-formed XML SVG

        r_png = client.post(
            "/api/workbench/charts/export",
            headers=headers,
            json={"chart_id": cid, "format": "png", "theme": "dark"},
        )
        assert r_png.status_code == 200, f"PNG export failed for {cid} ({ctype}): {r_png.text}"
        assert r_png.content.startswith(PNG_MAGIC)
        assert len(r_png.content) > 2000

        r_xlsx = client.post(
            "/api/workbench/charts/export",
            headers=headers,
            json={"chart_id": cid, "format": "xlsx", "theme": "light"},
        )
        assert r_xlsx.status_code == 200, f"XLSX export failed for {cid} ({ctype}): {r_xlsx.text}"
        wb = openpyxl.load_workbook(io.BytesIO(r_xlsx.content))
        assert {"Chart_Visual", "Chart_Data", "Statistical_Metadata"}.issubset(set(wb.sheetnames))
        assert wb["Chart_Data"].max_row >= 2
        assert len(wb["Chart_Visual"]._images) >= 1


def test_classification_model_diagnostic_charts_and_exports() -> None:
    client, headers = _make_client()

    # Build a clean binary classification dataset and train a real AutoML classifier
    clf_rows = []
    for i in range(110):
        x1 = (i % 23) * 0.45 - 4.5
        x2 = ((i * 5) % 19) * 0.38 - 3.2
        x3 = ((i * 3) % 13) * 0.52
        score = 1.4 * x1 - 0.9 * x2 + 0.5 * x3
        label = "HighRisk" if score > 0.3 else "LowRisk"
        clf_rows.append({"x1": round(x1, 3), "x2": round(x2, 3), "x3": round(x3, 3), "segment": "A" if i % 2 == 0 else "B", "risk_class": label})

    _upload_csv_df(client, headers, pd.DataFrame(clf_rows), "credit_risk.csv")
    assert client.post("/api/workbench/targets", headers=headers, json={"targets": ["risk_class"]}).status_code == 200
    assert client.post(
        "/api/workbench/s06/automl",
        headers=headers,
        json={"backend": "omniforge_native", "time_budget_seconds": 8.0},
    ).status_code == 200

    clf_chart_ids = [
        "classifier_roc_curve",
        "classifier_pr_curve",
        "classifier_calibration",
        "classifier_confusion_matrix",
        "classifier_gain_lift",
    ]
    for cid in clf_chart_ids:
        r_gen = client.post("/api/workbench/charts/generate", headers=headers, json={"chart_id": cid, "studio_id": "all"})
        assert r_gen.status_code == 200, f"{cid} failed: {r_gen.text}"
        body = r_gen.json()
        assert body["chart_id"] == cid
        assert len(body["table_rows"]) >= 2

        for fmt in ("svg", "png", "xlsx"):
            r_exp = client.post("/api/workbench/charts/export", headers=headers, json={"chart_id": cid, "format": fmt})
            assert r_exp.status_code == 200, f"{cid} {fmt} export failed: {r_exp.text}"
            if fmt == "svg":
                ET.fromstring(r_exp.content)
            elif fmt == "png":
                assert r_exp.content.startswith(PNG_MAGIC)
            else:
                wb = openpyxl.load_workbook(io.BytesIO(r_exp.content))
                assert "Chart_Data" in wb.sheetnames


def test_purely_categorical_and_formula_injection_xlsx_safety() -> None:
    client, headers = _make_client()

    # Dataset with ZERO numeric columns and dangerous spreadsheet-formula-prefixed category labels
    df = pd.DataFrame(
        {
            "dept": ["=CMD|' /C calc'!A0", "+Sales", "-Support", "@Ops"] * 15,
            "outcome": ["Approved", "Rejected", "Approved", "Pending"] * 15,
            "region": ["EMEA", "AMER", "APAC"] * 20,
        }
    )
    _upload_csv_df(client, headers, df, "pure_categorical.csv")

    for cid in ("categorical_share_donut", "stacked_composition_bar", "alluvial_sankey_flow", "empirical_ecdf_ks"):
        r_gen = client.post("/api/workbench/charts/generate", headers=headers, json={"chart_id": cid, "feature_x": "dept"})
        assert r_gen.status_code == 200, r_gen.text

    r_xlsx = client.post(
        "/api/workbench/charts/export",
        headers=headers,
        json={"chart_id": "categorical_share_donut", "format": "xlsx", "feature_x": "dept"},
    )
    assert r_xlsx.status_code == 200, r_xlsx.text
    wb = openpyxl.load_workbook(io.BytesIO(r_xlsx.content))
    ws_data = wb["Chart_Data"]
    col_a_vals = [str(ws_data.cell(row=r, column=1).value or "") for r in range(2, ws_data.max_row + 1)]
    # Verify CWE-1236 spreadsheet formula neutralization prefixed leading '=', '+', '-', '@' with "'"
    assert not any(v.startswith(("=", "+", "-", "@")) for v in col_a_vals)


def test_compiled_spa_ships_live_dropdown_chart_generator_and_25_chart_types() -> None:
    assets_dir = Path("src/omniforge/ui/static_app/assets")
    js_bundle = "\n".join(p.read_text(encoding="utf-8") for p in assets_dir.glob("*.js"))
    required_markers = [
        "/api/workbench/charts/catalog",
        "/api/workbench/charts/generate",
        "/api/workbench/charts/export",
        "Select a professional chart to generate on demand",
        "Choose Professional Chart Type (Only Selected Chart Is Generated)",
    ]
    for marker in required_markers:
        assert marker in js_bundle, f"Missing compiled SPA marker: {marker}"
