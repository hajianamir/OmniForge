"""Pass 3 regression tests for multi-tenant BOLA, privilege escalation, registry CWE-22, and studio routers."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from omniforge.api.auth import get_api_key
from omniforge.api.jobs import BackgroundJobManager
from omniforge.api.main import create_app
from omniforge.api.workbench import read_upload_bounded
from omniforge.core.exceptions import SecurityError
from omniforge.mlops.registry import ModelRegistry
from omniforge.models.baselines import LogisticBaseline


def _mint(client: TestClient, *, username: str, role: str, tenant_id: str) -> dict[str, str]:
    resp = client.post(
        "/api/v1/auth/token",
        json={"username": username, "role": role, "tenant_id": tenant_id},
        headers={"X-API-Key": get_api_key()},
    )
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def test_tenant_admin_cannot_escalate_to_platform_admin_or_cross_tenant() -> None:
    """Tenant-scoped admin must not mint platform_admin or foreign-tenant tokens (CWE-269 / P1-12)."""
    client = TestClient(create_app())
    alpha_admin = _mint(client, username="alice", role="admin", tenant_id="tenant-alpha")

    # Self-escalation to platform_admin must be forbidden
    r_esc = client.post(
        "/api/v1/auth/token",
        json={"username": "alice_root", "role": "platform_admin", "tenant_id": "tenant-alpha"},
        headers=alpha_admin,
    )
    assert r_esc.status_code == 403

    # Cross-tenant minting to tenant-beta must be forbidden
    r_cross = client.post(
        "/api/v1/auth/token",
        json={"username": "bob_spy", "role": "admin", "tenant_id": "tenant-beta"},
        headers=alpha_admin,
    )
    assert r_cross.status_code == 403

    # Same-tenant non-platform_admin minting succeeds
    r_same = client.post(
        "/api/v1/auth/token",
        json={"username": "alice_analyst", "role": "analyst", "tenant_id": "tenant-alpha"},
        headers=alpha_admin,
    )
    assert r_same.status_code == 200
    assert r_same.json()["tenant_id"] == "tenant-alpha"

    # platform_admin CAN mint cross-tenant tokens
    plat_admin = _mint(client, username="ops_root", role="platform_admin", tenant_id="ops")
    r_plat = client.post(
        "/api/v1/auth/token",
        json={"username": "beta_lead", "role": "admin", "tenant_id": "tenant-beta"},
        headers=plat_admin,
    )
    assert r_plat.status_code == 200
    assert r_plat.json()["tenant_id"] == "tenant-beta"


def test_api_v1_copilot_and_upload_do_not_leak_across_tenants() -> None:
    """Dataset uploaded by Tenant Alpha must never be visible in Tenant Beta's /api/v1/copilot/chat."""
    client = TestClient(create_app())
    alpha_hdr = _mint(client, username="u_alpha", role="admin", tenant_id="tenant-cop-a")
    beta_hdr = _mint(client, username="u_beta", role="admin", tenant_id="tenant-cop-b")

    csv_bytes = b"secret_alpha_metric,flag\n101.5,1\n202.5,0\n303.5,1\n404.5,0\n"
    r_up = client.post(
        "/api/v1/data/upload",
        files={"file": ("alpha_confidential.csv", io.BytesIO(csv_bytes), "text/csv")},
        headers=alpha_hdr,
    )
    assert r_up.status_code == 200

    r_alpha_chat = client.post(
        "/api/v1/copilot/chat",
        json={"message": "What is my current workspace status?"},
        headers=alpha_hdr,
    )
    assert r_alpha_chat.status_code == 200
    assert "1 dataset(s)" in r_alpha_chat.json()["reply"]

    r_beta_chat = client.post(
        "/api/v1/copilot/chat",
        json={"message": "What is my current workspace status?"},
        headers=beta_hdr,
    )
    assert r_beta_chat.status_code == 200
    assert "0 dataset(s)" in r_beta_chat.json()["reply"]


def test_model_registry_blocks_path_traversal_and_isolates_tenants(tmp_path: Path) -> None:
    """ModelRegistry must block CWE-22 model_name traversal and Studio 11 must isolate tenants."""
    X = pd.DataFrame({"a": [0.1, 0.2, 0.8, 0.9, 0.15, 0.85], "b": [1.0, 1.1, 0.1, 0.2, 0.9, 0.3]})
    y = np.array([0, 0, 1, 1, 0, 1])
    fitted = LogisticBaseline(random_seed=42).fit(X, y)

    reg = ModelRegistry(db_path=tmp_path / "reg.db", artifact_dir=tmp_path / "artifacts")
    for bad_name in ("../escape", "../../etc/passwd", "/tmp/abs_escape", "bad/slash", ".hidden"):
        with pytest.raises(SecurityError):
            reg.register_model(fitted, bad_name)

    # Raw fitted sklearn estimator round-trip (signed HMAC + safe_joblib_load_bytes via SklearnEstimatorAdapter)
    from sklearn.ensemble import RandomForestClassifier

    from omniforge.core.types import ModelStage

    raw_rf = RandomForestClassifier(n_estimators=10, random_state=7).fit(X, y)
    reg.register_model(raw_rf, "rf_wrapped", stage=ModelStage.STAGING)
    loaded_rf, _ = reg.load_model("rf_wrapped")
    assert np.array_equal(loaded_rf.predict(X), raw_rf.predict(X))

    client = TestClient(create_app())
    t1 = {**_mint(client, username="ds1", role="admin", tenant_id="iso-tenant-1"), "X-Session-ID": "s1"}
    t2 = {**_mint(client, username="ds2", role="admin", tenant_id="iso-tenant-2"), "X-Session-ID": "s1"}

    # Load sample + quick-train active model in tenant 1
    assert client.post("/api/workbench/s01/load-sample", json={"source_type": "domain", "name": "finance", "n_samples": 60}, headers=t1).status_code == 200
    assert client.post("/api/workbench/targets", json={"targets": ["default_flag"]}, headers=t1).status_code == 200
    assert client.post("/api/workbench/s14/quick-train", json={"method": "random_forest", "target": "default_flag"}, headers=t1).status_code == 200

    # Traversal model_name rejected via API with 403
    r_trav = client.post("/api/workbench/s11/register", json={"model_name": "../pwn", "stage": "Staging"}, headers=t1)
    assert r_trav.status_code == 403

    # Valid registration in tenant 1 (including lowercase stage handling)
    r_reg = client.post("/api/workbench/s11/register", json={"model_name": "t1_credit_champ", "stage": "staging"}, headers=t1)
    assert r_reg.status_code == 200

    # Tenant 2 must NOT see Tenant 1's registered model or be able to promote/archive it
    r_ov2 = client.get("/api/workbench/s11/overview", headers=t2)
    assert r_ov2.status_code == 200
    assert all(m["model_name"] != "t1_credit_champ" for m in r_ov2.json()["registered_models"])

    r_tamper = client.post(
        "/api/workbench/s11/promote",
        json={"model_name": "t1_credit_champ", "version": 1, "stage": "Archived"},
        headers=t2,
    )
    assert r_tamper.status_code == 400

    # Invalid stage string returns 400 instead of 500
    r_bad_stage = client.post(
        "/api/workbench/s11/promote",
        json={"model_name": "t1_credit_champ", "version": 1, "stage": "NotAStage"},
        headers=t1,
    )
    assert r_bad_stage.status_code == 400

    # Lowercase stage transition succeeds for Tenant 1
    r_prom = client.post(
        "/api/workbench/s11/promote",
        json={"model_name": "t1_credit_champ", "version": 1, "stage": "production"},
        headers=t1,
    )
    assert r_prom.status_code == 200
    assert r_prom.json()["updated"]["stage"] == "Production"


def test_background_jobs_isolated_by_owner_key_and_upload_bounded() -> None:
    """BackgroundJobManager must enforce owner_key visibility, and read_upload_bounded must cap bytes."""
    mgr = BackgroundJobManager(max_workers=1)
    rec_a = mgr.submit("alpha_job", lambda: {"secret": 42}, owner_key="job:alpha")
    assert mgr.get(rec_a.job_id, owner_key="job:alpha") is not None
    assert mgr.get(rec_a.job_id, owner_key="job:beta") is None
    assert mgr.get(rec_a.job_id, owner_key=None) is not None  # platform_admin query
    assert len(mgr.list_recent(owner_key="job:beta")) == 0
    assert len(mgr.list_recent(owner_key="job:alpha")) == 1

    import asyncio

    big = UploadFile(filename="big.csv", file=io.BytesIO(b"x" * 2048))
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(read_upload_bounded(big, max_bytes=1024))
    assert exc_info.value.status_code == 413


def test_zero_mypy_ignore_errors_and_clean_openapi_schema() -> None:
    """No src/ file may carry file-wide mypy ignore-errors/disable-error-code, and OpenAPI schema must compile."""
    src_root = Path(__file__).resolve().parents[2] / "src"
    offenders: list[str] = []
    for py_file in src_root.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        if "mypy: ignore-errors" in text or "mypy: disable-error-code" in text:
            offenders.append(str(py_file.relative_to(src_root)))
    assert offenders == [], f"Unexpected mypy file-level suppressions: {offenders}"

    app = create_app()
    schema = app.openapi()
    paths = schema.get("paths", {})
    assert "/api/workbench/s13/chat" in paths
    assert "/api/workbench/s04/hypothesis" in paths
    assert "/api/workbench/s14/quick-train" in paths


def test_studio13_copilot_extracts_reply_and_suggested_actions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Studio 13 /s13/chat must extract CopilotResponse.reply cleanly rather than stringifying the model."""
    monkeypatch.setenv("OMNIFORGE_REGISTRY_DIR", str(tmp_path / "reg"))
    monkeypatch.setenv("OMNIFORGE_TRACKING_DIR", str(tmp_path / "trk"))

    app = create_app()
    client = TestClient(app)
    hdrs = {**_mint(client, username="analyst-1", role="data_scientist", tenant_id="t-s13"), "X-Session-Id": "sess-s13"}

    r_load = client.post(
        "/api/workbench/s01/load-sample",
        json={"source_type": "domain", "name": "finance", "n_samples": 60},
        headers=hdrs,
    )
    assert r_load.status_code == 200

    r_chat = client.post(
        "/api/workbench/s13/chat",
        json={"message": "Profile data quality and suggest next steps."},
        headers=hdrs,
    )
    assert r_chat.status_code == 200
    body = r_chat.json()
    assert not body["reply"].startswith("reply=")
    assert "provider_used=" not in body["reply"]
    assert body["provider_used"] in ("local_deterministic_planner", "local")
    assert isinstance(body["suggested_actions"], list) and len(body["suggested_actions"]) >= 1
    assert isinstance(body["tool_calls"], list) and any(
        (tc.get("tool") == "profile_quality" if isinstance(tc, dict) else tc == "profile_quality")
        for tc in body["tool_calls"]
    )


def test_studios_s04_s08_s12_s14_regressions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify s04 Lilliefors/Stephens p-values, s08->s12 forecast catalog/report wiring, and s14 small-sample split."""
    monkeypatch.setenv("OMNIFORGE_REGISTRY_DIR", str(tmp_path / "reg"))
    monkeypatch.setenv("OMNIFORGE_TRACKING_DIR", str(tmp_path / "trk"))

    app = create_app()
    client = TestClient(app)
    hdrs = {**_mint(client, username="analyst-2", role="data_scientist", tenant_id="t-studios"), "X-Session-Id": "sess-studios"}

    rng = np.random.default_rng(12345)
    df_norm = pd.DataFrame(
        {
            "gauss": rng.normal(loc=10.0, scale=2.0, size=60),
            "x2": rng.normal(loc=0.0, scale=1.0, size=60),
            "target": rng.normal(loc=5.0, scale=1.0, size=60),
        }
    )
    buf = io.BytesIO()
    df_norm.to_csv(buf, index=False)
    buf.seek(0)
    r_up = client.post(
        "/api/workbench/s01/upload",
        files={"file": ("norm.csv", buf.getvalue(), "text/csv")},
        headers=hdrs,
    )
    assert r_up.status_code == 200

    # Studio 04 Anderson-Darling returns continuous Stephens (1986) p-value (> 0.15 on clean normal data)
    r_ad = client.post(
        "/api/workbench/s04/hypothesis",
        json={"mode": "custom", "test_type": "anderson", "col_a": "gauss", "alpha": 0.05},
        headers=hdrs,
    )
    assert r_ad.status_code == 200
    ad_res = r_ad.json()["result"]
    assert ad_res["p_value"] > 0.15
    assert ad_res["significant"] is False

    # Studio 04 KS Normality delegates to composite-null Lilliefors test
    r_ks = client.post(
        "/api/workbench/s04/hypothesis",
        json={"mode": "custom", "test_type": "ks_normality", "col_a": "gauss", "alpha": 0.05},
        headers=hdrs,
    )
    assert r_ks.status_code == 200
    assert "Lilliefors" in r_ks.json()["result"]["test_name"]

    # Studio 01 advanced_audit + Studio 02 advanced_mining serialize cleanly via to_json_object
    assert client.post("/api/workbench/s01/advanced_audit", headers=hdrs).status_code == 200
    assert client.post("/api/workbench/s02/advanced_mining", headers=hdrs).status_code == 200

    # Studio 08 forecast populates uap_forecast_report and Studio 12 catalog/compile picks it up
    r_fc = client.post(
        "/api/workbench/s08/forecast",
        json={"target_col": "target", "horizon": 5, "deep_arch": "mlp_lag" if False else "nbeats", "epochs": 5},
        headers=hdrs,
    )
    assert r_fc.status_code == 200
    r_cat = client.get("/api/workbench/s12/catalog", headers=hdrs)
    assert r_cat.status_code == 200
    fc_sec = next(s for s in r_cat.json()["sections"] if s["key"] == "decision_intelligence")
    assert fc_sec["has_data"] is True

    r_comp = client.post(
        "/api/workbench/s12/compile",
        json={"selected_sections": ["executive_summary", "decision_intelligence"]},
        headers=hdrs,
    )
    assert r_comp.status_code == 200
    assert "Deep Time-Series Forecast" in r_comp.json()["table_names"]

    # Small 10-row dataset works cleanly in Studio 14 quick-train without empty test split (old max(10, 8)=10 bug)
    hdrs_small = {
        **_mint(client, username="analyst-2", role="data_scientist", tenant_id="t-studios"),
        "X-Session-Id": "sess-small10",
    }
    small_df = df_norm.iloc[:10].copy()
    sbuf = io.BytesIO()
    small_df.to_csv(sbuf, index=False)
    assert (
        client.post(
            "/api/workbench/s01/upload",
            files={"file": ("small10.csv", sbuf.getvalue(), "text/csv")},
            headers=hdrs_small,
        ).status_code
        == 200
    )
    client.post("/api/workbench/targets", json={"targets": ["target"]}, headers=hdrs_small)
    r_qt_small = client.post(
        "/api/workbench/s14/quick-train",
        json={"method": "ridge", "target": "target"},
        headers=hdrs_small,
    )
    assert r_qt_small.status_code == 200

