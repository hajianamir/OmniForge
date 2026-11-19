"""Task 35 end-to-end regression tests: real equations, immediate server cache ordering, and SPA headers."""

from __future__ import annotations

from fastapi.testclient import TestClient

from omniforge.api.auth import get_api_key
from omniforge.api.main import create_app


def _client_and_headers() -> tuple[TestClient, dict[str, str]]:
    app = create_app()
    client = TestClient(app, headers={"X-API-Key": get_api_key()})
    headers = {"X-Session-ID": "task35_e2e_session"}
    return client, headers


def test_task35_real_equations_studio03_and_studio06_automl() -> None:
    """Studio 03 baseline equations and Studio 06 AutoML equations must expose real formulas (never empty fallback)."""
    client, headers = _client_and_headers()
    r_load = client.post(
        "/api/workbench/s01/load-sample",
        json={"source_type": "domain", "name": "energy", "n_samples": 180},
        headers=headers,
    )
    assert r_load.status_code == 200, r_load.text

    # Studio 03 Diagnostics — every ranked baseline equation must have non-empty real formula_plain/latex/python
    r_diag = client.post(
        "/api/workbench/s03/run",
        json={"mode": "complete", "alpha": 0.05, "holdout_ratio": 0.2},
        headers=headers,
    )
    assert r_diag.status_code == 200, r_diag.text
    diag_data = r_diag.json()
    assert "s03_diagnostics" in diag_data["state"]["cached_result_keys"]
    eq_map = diag_data["report"]["baseline_equations"]
    assert eq_map, "Expected ranked baseline_equations for target"
    for target_name, eq_list in eq_map.items():
        assert len(eq_list) >= 3, f"Expected >=3 ranked baselines for {target_name}"
        for eq in eq_list:
            fp = str(eq.get("formula_plain") or "").strip()
            fl = str(eq.get("formula_latex") or "").strip()
            fpy = str(eq.get("formula_python") or "").strip()
            assert fp, f"Empty formula_plain for {eq.get('model_name')}"
            assert fl, f"Empty formula_latex for {eq.get('model_name')}"
            assert fpy, f"Empty formula_python for {eq.get('model_name')}"
            assert fp.lower() != "non-parametric analytical model", f"Bogus placeholder on {eq.get('model_name')}"
            if "ridge" in str(eq.get("model_name", "")).lower():
                assert eq.get("structured_terms") or eq.get("coefficients"), "Ridge equation must expose real coefficients"

    # Studio 06 AutoML — ModelEquationExtractor must succeed (no y_reference TypeError)
    r_aml = client.post(
        "/api/workbench/s06/automl",
        json={"backend": "native_stacking", "time_budget_seconds": 15, "cv_folds": 3},
        headers=headers,
    )
    assert r_aml.status_code == 200, r_aml.text
    aml_data = r_aml.json()
    assert "s06_automl" in aml_data["state"]["cached_result_keys"]
    aml_eq = aml_data.get("equation") or {}
    assert str(aml_eq.get("formula_plain") or "").strip(), f"AutoML equation was empty: {aml_eq}"
    assert str(aml_eq.get("formula_latex") or "").strip(), f"AutoML LaTeX equation was empty: {aml_eq}"


def test_task35_immediate_cache_ordering_across_studios() -> None:
    """Every studio POST endpoint must save its cache key BEFORE building state.cached_result_keys."""
    client, headers = _client_and_headers()
    assert client.post(
        "/api/workbench/s01/load-sample",
        json={"source_type": "domain", "name": "energy", "n_samples": 160},
        headers=headers,
    ).status_code == 200

    # 1. Studio 01 Quality Profile must immediately include s01_quality in returned state.cached_result_keys
    r_q = client.post("/api/workbench/s01/profile", headers=headers)
    assert r_q.status_code == 200, r_q.text
    assert "s01_quality" in r_q.json()["state"]["cached_result_keys"]

    # 2. Studio 01 Clean Dry-Run Preview must also be cached immediately
    r_prev = client.post("/api/workbench/s01/clean", json={"mode": "auto", "preview_only": True}, headers=headers)
    assert r_prev.status_code == 200, r_prev.text
    assert "s01_clean" in r_prev.json()["state"]["cached_result_keys"]
    assert "s01_quality" in r_prev.json()["state"]["cached_result_keys"]

    # 3. Studio 01 Clean Commit must include s01_clean & preserve s01_quality
    r_clean = client.post("/api/workbench/s01/clean", json={"mode": "auto", "preview_only": False}, headers=headers)
    assert r_clean.status_code == 200, r_clean.text
    assert "s01_clean" in r_clean.json()["state"]["cached_result_keys"]
    assert "s01_quality" in r_clean.json()["state"]["cached_result_keys"]

    # 4. Studio 02 Clustering + Anomaly + Mining all persist simultaneously
    r_cl = client.post("/api/workbench/s02/clustering", json={"mode": "auto", "algorithm": "kmeans", "n_clusters": 3}, headers=headers)
    assert r_cl.status_code == 200, r_cl.text
    assert "s02_clustering" in r_cl.json()["state"]["cached_result_keys"]

    r_an = client.post("/api/workbench/s02/anomaly", json={"mode": "auto"}, headers=headers)
    assert r_an.status_code == 200, r_an.text
    an_keys = r_an.json()["state"]["cached_result_keys"]
    assert {"s01_quality", "s01_clean", "s02_clustering", "s02_anomaly"}.issubset(set(an_keys))

    # 5. Studio 04 EDA + Hypothesis + Studio 06 Non-Parametric + Studio 14 Supervisory all persist simultaneously
    r_eda = client.post("/api/workbench/s04/eda", headers=headers)
    assert r_eda.status_code == 200, r_eda.text
    assert "s04_eda" in r_eda.json()["state"]["cached_result_keys"]

    r_hyp = client.post("/api/workbench/s04/hypothesis", json={"mode": "auto", "alpha": 0.05}, headers=headers)
    assert r_hyp.status_code == 200, r_hyp.text
    assert {"s04_eda", "s04_hypothesis"}.issubset(set(r_hyp.json()["state"]["cached_result_keys"]))

    r_np = client.post("/api/workbench/s06/nonparametric", json={"mode": "complete"}, headers=headers)
    assert r_np.status_code == 200, r_np.text
    assert "s06_nonparametric" in r_np.json()["state"]["cached_result_keys"]

    r_sup = client.post(
        "/api/workbench/s14/evaluate-demo-batch",
        json={"n_rows": 25, "inject_fault_pct": 10.0, "noise_scale": 0.05},
        headers=headers,
    )
    assert r_sup.status_code == 200, r_sup.text
    sup_keys = set(r_sup.json()["state"]["cached_result_keys"])
    assert {"s01_quality", "s01_clean", "s02_clustering", "s02_anomaly", "s04_eda", "s04_hypothesis", "s06_nonparametric", "s14_supervisory"}.issubset(sup_keys)

    # GET /api/workbench/results must return every single one of those cached payloads
    r_all = client.get("/api/workbench/results", headers=headers)
    assert r_all.status_code == 200
    all_res = r_all.json()["results"]
    for k in ("s01_quality", "s01_clean", "s02_clustering", "s02_anomaly", "s04_eda", "s04_hypothesis", "s06_nonparametric", "s14_supervisory"):
        assert k in all_res and all_res[k], f"Missing cached payload for {k}"


def test_task35_spa_index_no_cache_header() -> None:
    """SPA index.html must send Cache-Control: no-store so browsers never load stale bundles."""
    client, _ = _client_and_headers()
    r = client.get("/")
    assert r.status_code == 200
    assert "no-store" in r.headers.get("cache-control", "").lower()
