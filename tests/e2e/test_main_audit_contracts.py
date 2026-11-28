"""Regression coverage for the independent audit of main at 4385647."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scipy import stats
from statsmodels.stats.multitest import multipletests

from omniforge.api.auth import get_api_key
from omniforge.api.main import create_app


def _client(frame: pd.DataFrame | None = None) -> TestClient:
    client = TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": uuid4().hex})
    if frame is not None:
        response = client.post(
            "/api/workbench/s01/upload", files={"file": ("audit.csv", frame.to_csv(index=False), "text/csv")},
        )
        assert response.status_code == 200, response.text
    return client


@pytest.fixture()
def diagnostics_client() -> TestClient:
    rng = np.random.default_rng(610)
    x = np.linspace(-3, 3, 120)
    client = _client(pd.DataFrame({
        "x": x, "x2": 0.92 * x + rng.normal(0, 0.2, len(x)),
        "ignored_candidate": np.sin(x), "target": 2 * x + rng.normal(0, 0.4, len(x)),
    }))
    assert client.post("/api/workbench/targets", json={"targets": ["target"]}).status_code == 200
    return client


@pytest.mark.parametrize("method", ["shapiro_wilk", "dagostino_pearson", "anderson_darling", "kolmogorov_smirnov"])
def test_studio03_canonical_controls_run_with_holm(diagnostics_client: TestClient, method: str) -> None:
    response = diagnostics_client.post("/api/workbench/s03/run", json={
        "mode": "custom", "selected_modules": ["normality_distribution"],
        "fdr_method": "holm", "parametric_mode": "non_parametric", "normality_method": method,
    })
    assert response.status_code == 200, response.text
    report = response.json()["report"]
    assert report["criteria"]["parametric_mode"] == "non_parametric"
    assert report["criteria"]["normality_method"] == method
    assert report["correction_family"]["method"] == "holm"
    evaluated = [row for row in report["hypothesis_ledger"] if row["p_value"] is not None]
    assert evaluated and report["correction_family"]["n_evaluable"] == len(evaluated)
    expected_name = "lilliefors_ks" if method == "kolmogorov_smirnov" else method
    assert {row["test_name"] for row in evaluated} == {expected_name}
    expected = multipletests([row["p_value"] for row in evaluated], method="holm")[1]
    np.testing.assert_allclose([row["adjusted_p_value"] for row in evaluated], expected)


@pytest.mark.parametrize(("field", "value"), [
    ("fdr_method", "unsupported"), ("parametric_mode", "nonparametric"),
    ("normality_method", "shapiro"), ("normality_method", "dagostino"),
    ("normality_method", "anderson"), ("normality_method", "ks"),
    ("correlation_method", "unsupported"), ("outlier_method", "unsupported"), ("vif_threshold", 0.0),
])
def test_studio03_rejects_unsupported_criteria_at_request_boundary(field: str, value: str | float) -> None:
    # No dataset is loaded: the request must be rejected before the route handler.
    response = _client().post("/api/workbench/s03/run", json={field: value})
    assert response.status_code == 422, response.text
    assert response.json()["error_code"] == "REQUEST_VALIDATION_ERROR"
    assert response.json()["detail"]


def test_studio03_returns_vif_map_with_run_threshold(diagnostics_client: TestClient) -> None:
    response = diagnostics_client.post("/api/workbench/s03/run", json={
        "mode": "custom", "selected_modules": ["multicollinearity_vif"], "vif_threshold": 11,
    })
    assert response.status_code == 200, response.text
    report = response.json()["report"]
    assert report["vif_scores"]["x"] == pytest.approx(65.2236, abs=0.001)
    assert report["vif_scores"]["x2"] == pytest.approx(65.0584, abs=0.001)
    assert report["criteria"]["vif_threshold"] == 11


def test_studio08_rejects_the_audit_three_arm_fixture_and_keeps_binary_aipw() -> None:
    rng = np.random.default_rng(3201)
    n = 300
    x = rng.normal(size=n)
    treatment = np.tile([0.0, 1.0, 2.0], n // 3)
    order = rng.permutation(n)
    x, treatment = x[order], treatment[order]
    y = 1.25 * x + 2.0 * treatment + rng.normal(0, 0.2, size=n)
    frame = pd.DataFrame({"treatment": treatment, "outcome": y, "x": x})
    request = {"treatment_col": "treatment", "outcome_col": "outcome", "confounders": ["x"]}

    rejected = _client(frame).post("/api/workbench/s08/causal", json=request)
    assert rejected.status_code == 400, rejected.text
    body = rejected.json()
    assert "binary treatment coded 0 (control) and 1 (treated)" in body["message"]
    assert "causal" not in body

    accepted = _client(frame[treatment < 2]).post("/api/workbench/s08/causal", json=request)
    assert accepted.status_code == 200, accepted.text
    result = accepted.json()["causal"]
    assert (result["method"], result["n_treated"], result["n_control"]) == ("aipw", 100, 100)
    assert result["ate"] == pytest.approx(2.0, abs=0.08)
    assert result["ci_lower"] < result["ate"] < result["ci_upper"]


def test_studio06_describes_audit_linear_null_lift_without_an_inferential_claim() -> None:
    rng = np.random.default_rng(4)
    x = rng.uniform(-2.0, 2.0, 72)
    y = 1.5 * x + rng.normal(0.0, 2.0, 72)
    client = _client(pd.DataFrame({"x": x, "target": y}))
    assert client.post("/api/workbench/targets", json={"targets": ["target"]}).status_code == 200
    response = client.post("/api/workbench/s06/nonparametric", json={
        "target": "target", "cv_folds": 5, "models": ["gam_spline"],
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["lift_vs_parametric"] == pytest.approx(0.0210, abs=0.001)
    assert result["nonparametric_outperforms_parametric"] is True
    assert result["cv_lift_heuristic_threshold"] == 0.015
    verdict = result["verdict"]
    assert verdict == result["linearity_verdict"]
    assert "exceeds the heuristic threshold" in verdict
    assert "not a statistical test of nonlinearity" in verdict
    assert "significant" not in verdict


def test_studio01_benford_api_preserves_small_positive_p_value() -> None:
    digits = np.arange(1, 10)
    values = np.repeat(digits + 0.25, digits)
    client = _client(pd.DataFrame({"amount": values}))
    response = client.post("/api/workbench/s01/advanced_audit")
    assert response.status_code == 200, response.text
    result = response.json()["benford_tests"]["amount"]
    expected = stats.chisquare(digits, f_exp=len(values) * np.log10(1 + 1 / digits)).pvalue
    assert 0 < result["p_value"] < 1e-6
    assert result["p_value"] == pytest.approx(expected, rel=1e-12, abs=0)
    assert result["conforms_benford"] is False


def test_studio03_documented_operations_match_openapi() -> None:
    paths = create_app().openapi()["paths"]
    prefix = "/api/workbench"
    registered = {
        (path.removeprefix(prefix), method.upper())
        for path, operations in paths.items() if path.startswith(f"{prefix}/s03/")
        for method in operations
    }
    docs = Path(__file__).resolve().parents[2] / "docs"
    for filename in ("user_guide.md", "api_reference.md", "architecture.md"):
        documented = set()
        for line in (docs / filename).read_text(encoding="utf-8").splitlines():
            cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
            route = next((cell for cell in cells if re.fullmatch(r"/s03/\w+", cell)), None)
            if route is not None:
                verb = next(cell for cell in cells if cell in {"GET", "POST", "PUT", "PATCH", "DELETE"})
                documented.add((route, verb))
        assert documented == registered, filename
