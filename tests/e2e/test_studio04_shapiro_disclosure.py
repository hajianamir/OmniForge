"""Studio 04 exposes the actual scope of each capped Shapiro-Wilk result."""

from __future__ import annotations

from io import BytesIO
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import stats

from omniforge.api.auth import get_api_key
from omniforge.api.main import create_app


def _upload(n: int) -> tuple[TestClient, np.ndarray]:
    client = TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": f"shapiro-{uuid4().hex}"})
    values = np.random.default_rng(123).normal(size=n)
    df = pd.DataFrame({"complete": values, "partial": values}).assign(constant=1)
    df.loc[:119, "partial"] = np.nan
    uploaded = client.post(
        "/api/workbench/s01/upload",
        files={"file": ("shapiro.csv", df.to_csv(index=False).encode(), "text/csv")},
    )
    assert uploaded.status_code == 200, uploaded.text
    return client, values


@pytest.mark.parametrize("n", [5000, 5001, 5100])
def test_shapiro_boundary_counts_for_custom_and_per_column_batteries(n: int) -> None:
    client, values = _upload(n)
    expected = stats.shapiro(values[:5000])
    for mode in ("custom", "auto", "complete"):
        res = client.post(
            "/api/workbench/s04/hypothesis", json={"mode": mode, "test_type": "shapiro", "col_a": "complete"}
        )
        assert res.status_code == 200, res.text
        body = res.json()
        by_column = {"complete": body["result"]} if mode == "custom" else {
            row["scope"]: row for row in body["battery"] if row["test_type"] == "shapiro"
        }
        row = by_column["complete"]
        assert row["statistic"] == round(expected.statistic, 6)
        assert row["p_value"] == pytest.approx(expected.pvalue)
        assert row["details"]["sampled_from"] == n
        assert row["details"]["n_scanned"] == 5000
        assert row["shapiro_sample"] == {
            "source_rows": n, "available_rows": n, "tested_rows": 5000, "omitted_rows": n - 5000,
            "cap": 5000, "cap_omitted_rows": n - 5000, "cap_reason": "Shapiro-Wilk p-value accuracy",
            "selection": "first_non_missing_in_study_order",
        }
        assert row["sample_disclosure"].startswith(f"{n:,} available / 5,000 tested / {n - 5000:,} omitted")
        if mode != "custom":
            partial = by_column["partial"]["shapiro_sample"]
            assert (partial["source_rows"], partial["available_rows"], partial["tested_rows"]) == (n, n - 120, n - 120)
            assert partial["omitted_rows"] == partial["cap_omitted_rows"] == 0
            assert f"{n:,} source rows" in by_column["partial"]["sample_disclosure"]
            assert all(r.get("sample_disclosure") is None for r in body["battery"] if r["test_type"] != "shapiro")
            skipped = next(r for r in body["skipped"] if r["scope"] == "constant" and r["test_type"] == "shapiro")
            assert skipped["shapiro_sample"]["tested_rows"] == 0
            assert skipped["shapiro_sample"]["cap_omitted_rows"] == 0


def test_shapiro_disclosure_reaches_report_preview_and_exports() -> None:
    client, _ = _upload(5100)
    battery = client.post("/api/workbench/s04/hypothesis", json={"mode": "auto"})
    assert battery.status_code == 200, battery.text
    report = client.post(
        "/api/workbench/s12/compile",
        json={"selected_sections": ["eda_hypothesis"], "include_tables": True, "include_charts": False},
    )
    assert report.status_code == 200, report.text
    preview = report.json()["table_previews"]["Hypothesis Test Battery"]
    note = "5,100 available / 5,000 tested / 100 omitted"
    assert preview["preview"][0]["sample_disclosure"].startswith(note)
    # Wide document formats retain only the leading columns.
    assert preview["columns"].index("sample_disclosure") < 7
    for fmt in ("json", "html", "md", "csv"):
        exported = client.get(f"/api/workbench/s12/export/{fmt}")
        assert exported.status_code == 200, exported.text
        assert note in exported.text
    for fmt, prefix in (("docx", "word/document.xml"), ("xlsx", "xl/worksheets/")):
        exported = client.get(f"/api/workbench/s12/export/{fmt}")
        assert exported.status_code == 200, exported.text
        with ZipFile(BytesIO(exported.content)) as doc:
            assert any(note.encode() in doc.read(name) for name in doc.namelist() if name.startswith(prefix))


@pytest.mark.parametrize("n", [5000, 5001, 5100])
def test_qq_plot_discloses_shapiro_sample_separately_from_plot_size(n: int) -> None:
    client, values = _upload(n)
    chart = {"chart_id": "normality_qq_plot", "studio_id": "04", "feature_x": "complete"}
    response = client.post("/api/workbench/charts/generate", json=chart)
    assert response.status_code == 200, response.text
    body = response.json()
    expected = stats.shapiro(np.sort(values)[np.linspace(0, n - 1, 5000).round().astype(int)])
    assert body["kpis"]["Shapiro-Wilk W"] == round(expected.statistic, 4)
    assert body["kpis"]["Sample Size n"] == n
    assert body["shapiro_sample"]["tested_rows"] == 5000
    assert body["shapiro_sample"]["selection"] == "evenly_spaced_sorted_ranks"
    note = f"{n:,} available / 5,000 tested / {n - 5000:,} omitted"
    assert note in body["subtitle"]
    if n == 5100:
        exported = client.post("/api/workbench/charts/export", json={**chart, "format": "svg"})
        assert exported.status_code == 200, exported.text
        assert note in exported.text
        spreadsheet = client.post("/api/workbench/charts/export", json={**chart, "format": "xlsx"})
        assert spreadsheet.status_code == 200, spreadsheet.text
        with ZipFile(BytesIO(spreadsheet.content)) as book:
            assert any(note.encode() in book.read(name) for name in book.namelist() if name.startswith("xl/worksheets/"))
        constant = client.post("/api/workbench/charts/generate", json={**chart, "feature_x": "constant"})
        assert constant.status_code == 200, constant.text
        result = constant.json()
        assert result["shapiro_sample"]["tested_rows"] == 0
        assert result["kpis"]["Shapiro p-value"] is None
        assert "not run (no variation)" in result["subtitle"]
