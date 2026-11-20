"""Dedicated P0 Security, Cryptography, Deserialization, Auth/CSRF, SQL/SSRF, XSS & Runtime Regression Tests."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import pickle
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError
from typer.testing import CliRunner

from omniforge.api.auth import WORKBENCH_AUTH_COOKIE, WORKBENCH_CSRF_COOKIE, get_api_key
from omniforge.api.main import create_app
from omniforge.api.workbench import router as workbench_router
from omniforge.cli.main import cli
from omniforge.config.settings import APISettings, get_settings
from omniforge.core.exceptions import DataIngestionError, SecurityError
from omniforge.core.logging import configure_logging, get_logger
from omniforge.core.security import (
    ArtifactSigner,
    EncryptionManager,
    JWTManager,
    safe_joblib_load_bytes,
)
from omniforge.core.types import Principal
from omniforge.core.workspace_session import WorkspaceSessionManager
from omniforge.data.connectors import (
    _create_readonly_sql_engine,
    access_inspect_database,
    ensure_demo_sqlite_database,
    sql_inspect_database,
    sql_load_dataframe,
)
from omniforge.data.ingestion import DataIngestionEngine
from omniforge.data.versioning import DatasetVersionManager
from omniforge.eda.automated_eda import AutomatedEDA
from omniforge.ui.state import export_workspace_project_bytes, restore_workspace_project_bytes


class _ExploitPayload:
    """Synthetic CWE-502 RCE payload used to prove deserialization defenses block execution."""

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return (os.system, ("echo SHOULD_NEVER_EXECUTE",))


def test_p0_1_aes256gcm_jwt_persistence_and_production_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify AES-256-GCM encryption, persisted keys across restarts, and production fail-closed."""
    monkeypatch.setenv("OMNIFORGE_KEY_DIR", str(tmp_path))
    monkeypatch.delenv("OMNIFORGE_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("OMNIFORGE_SECURITY__ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("OMNIFORGE_JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("OMNIFORGE_SECURITY__JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("OMNIFORGE_ARTIFACT_SIGNING_KEY", raising=False)
    monkeypatch.delenv("OMNIFORGE_SECURITY__ARTIFACT_SIGNING_KEY", raising=False)
    monkeypatch.setenv("OMNIFORGE_ENVIRONMENT", "development")
    get_settings.cache_clear()
    try:
        enc1 = EncryptionManager()
        key_file = tmp_path / "encryption.key"
        assert key_file.is_file()
        if os.name == "posix":
            assert (key_file.stat().st_mode & 0o777) == 0o600

        ct1 = enc1.encrypt_bytes(b"enterprise-secret")
        ct2 = enc1.encrypt_bytes(b"enterprise-secret")
        assert base64.urlsafe_b64decode(ct1).startswith(b"OF2GCM\x01") and ct1 != ct2  # unique 96-bit nonces

        enc2 = EncryptionManager()
        assert enc2.decrypt_bytes(ct1) == b"enterprise-secret"

        # Tampered ciphertext MUST fail authentication tag verification
        tampered = ct1[:-1] + bytes([ct1[-1] ^ 0xFF])
        with pytest.raises(SecurityError):
            enc2.decrypt_bytes(tampered)

        # Persisted JWT secret survives restart
        j1 = JWTManager()
        tok = j1.create_token(Principal(subject="alice", roles=["analyst"], tenant_id="t1"))
        j2 = JWTManager()
        assert j2.verify_token(tok).subject == "alice"

        # Production mode MUST fail closed when secrets are unset
        monkeypatch.setenv("OMNIFORGE_ENVIRONMENT", "production")
        get_settings.cache_clear()
        with pytest.raises(SecurityError):
            EncryptionManager()
        with pytest.raises(SecurityError):
            JWTManager()
        with pytest.raises(SecurityError):
            ArtifactSigner()
    finally:
        get_settings.cache_clear()


def test_p0_2_deserialization_rce_hmac_xxe_and_upload_blocks(tmp_path: Path) -> None:
    """Verify RestrictedUnpickler, HMAC verification, ZIP-bomb guard, XXE block, and pickle upload rejection."""
    evil_bytes = pickle.dumps(_ExploitPayload())
    with pytest.raises(SecurityError):
        safe_joblib_load_bytes(evil_bytes)

    # Signed workspace round-trip vs unsigned/tampered rejection
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "lbl": ["a", "b", "c"]})
    state: dict[str, Any] = {"uap_df": df, "uap_raw_df": df.copy(), "uap_dataset_name": "sec_test"}
    signed_zip, _manifest = export_workspace_project_bytes(state, project_name="signed_proj")
    target_state: dict[str, Any] = {}
    info = restore_workspace_project_bytes(target_state, signed_zip)
    assert info["format_version"] == "2.0"

    # Strip HMAC signature from archive -> MUST raise SecurityError
    buf = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(signed_zip), "r") as zin, zipfile.ZipFile(buf, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "manifest.json":
                m = json.loads(data.decode("utf-8"))
                assert m.get("payload_hmac_sha256")
                m["payload_hmac_sha256"] = ""
                data = json.dumps(m).encode("utf-8")
            zout.writestr(item, data)
    with pytest.raises(SecurityError):
        WorkspaceSessionManager.load_from_bytes(buf.getvalue())

    # .pkl ingestion blocked by default + XML XXE blocked
    pkl_path = tmp_path / "malicious.pkl"
    pkl_path.write_bytes(evil_bytes)
    with pytest.raises(DataIngestionError, match="CWE-502"):
        DataIngestionEngine().ingest(pkl_path)

    xxe_path = tmp_path / "xxe.xml"
    xxe_path.write_text(
        '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><r><v>&xxe;</v></r>',
        encoding="utf-8",
    )
    with pytest.raises(DataIngestionError, match="(?i)doctype|entity|xxe"):
        DataIngestionEngine().ingest(xxe_path)

    # DatasetVersionManager uses safe Parquet/JSON and rejects .pkl snapshots
    vm = DatasetVersionManager(store_dir=tmp_path / "versions")
    snap = vm.commit(df, dataset_name="safe_ds")
    assert Path(snap.storage_path).suffix in {".parquet", ".json"}
    checked_df, _ = vm.checkout("safe_ds")
    assert len(checked_df) == 3

    # Upload endpoints block .pkl/.joblib/.dill with HTTP 400
    app = create_app()
    client = TestClient(app, headers={"X-API-Key": get_api_key()})
    for ext in (".pkl", ".pickle", ".joblib", ".dill"):
        r = client.post(
            "/api/workbench/s01/upload",
            files={"file": (f"bad{ext}", io.BytesIO(b"data"), "application/octet-stream")},
        )
        assert r.status_code == 400, (ext, r.status_code)


def test_p0_3_and_p0_4_auth_rbac_csrf_cors_health_and_500_hygiene() -> None:
    """Verify all operational APIs require auth, CSRF enforced on cookie mutations, strict CORS, and safe 500/health."""
    app = create_app()

    # 1. Every operational /api/* endpoint MUST return 401 unauthenticated
    anon = TestClient(app, raise_server_exceptions=False)
    v1_routes = [r for r in app.routes if isinstance(r, APIRoute) and r.path.startswith("/api/")]
    wb_routes = [r for r in workbench_router.routes if isinstance(r, APIRoute)]
    for route in v1_routes + wb_routes:
        if route.path in ("/api/v1/health", "/api/v1/version"):
            continue
        url = route.path
        for p, v in [
            ("{job_id}", "1"),
            ("{fmt}", "csv"),
            ("{endpoint}", "ep1"),
            ("{model_name}", "m1"),
            ("{version}", "1"),
            ("{formula_format}", "python"),
            ("{domain_name}", "finance"),
        ]:
            url = url.replace(p, v)
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            resp = anon.request(method, url, json={} if method in ("POST", "PUT", "PATCH") else None)
            assert resp.status_code == 401, (method, route.path, resp.status_code)

    # 2. Local SPA cookie + CSRF double-submit enforcement (requires explicit opt-in)
    from omniforge.config.settings import get_settings

    sec = get_settings().security
    prev_opt_in = sec.auto_bootstrap_local_ui
    sec.auto_bootstrap_local_ui = True
    try:
        browser = TestClient(app)
        assert browser.get("/").status_code == 200
        assert WORKBENCH_AUTH_COOKIE in browser.cookies and WORKBENCH_CSRF_COOKIE in browser.cookies
        assert browser.get("/api/workbench/state").status_code == 200
        assert browser.post("/api/workbench/theme", json={"theme": "light"}).status_code == 403
        csrf = browser.cookies[WORKBENCH_CSRF_COOKIE]
        assert browser.post("/api/workbench/theme", json={"theme": "light"}, headers={"X-CSRF-Token": csrf}).status_code == 200
    finally:
        sec.auto_bootstrap_local_ui = prev_opt_in

    # 3. CORS forbids wildcard+credentials and rejects untrusted origins
    assert APISettings(cors_origins=["*"], cors_allow_credentials=True).cors_allow_credentials is False
    r_evil = anon.options(
        "/api/workbench/state",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert r_evil.status_code == 400 and "access-control-allow-origin" not in r_evil.headers

    # 4. /health hides static_dir and 500 handler hides raw exception text
    r_h = anon.get("/health")
    assert r_h.status_code == 200 and "static_dir" not in r_h.json() and r_h.json()["spa_ready"] is True
    assert r_h.headers.get("x-content-type-options") == "nosniff"
    assert r_h.headers.get("x-frame-options") == "SAMEORIGIN"

    def _boom() -> None:
        raise RuntimeError("INTERNAL_SECRET_DO_NOT_LEAK")

    app.router.routes.insert(0, APIRoute("/api/workbench/_sec_test_boom", _boom, methods=["GET"]))
    authed = TestClient(app, raise_server_exceptions=False, headers={"X-API-Key": get_api_key()})
    r_500 = authed.get("/api/workbench/_sec_test_boom")
    assert r_500.status_code == 500
    body_500 = r_500.json()
    assert body_500["error_type"] == "InternalServerError"
    assert "INTERNAL_SECRET_DO_NOT_LEAK" not in r_500.text
    assert "RuntimeError" not in r_500.text


def test_p0_5_and_p0_6_sql_readonly_ssrf_traversal_and_xss(tmp_path: Path) -> None:
    """Verify 3-layer SQLite read-only, SSRF/Access blocks, benchmark traversal block, and XSS escaping."""
    db_path = ensure_demo_sqlite_database()
    demo_url = f"sqlite:///{db_path.as_posix()}"
    assert sql_inspect_database(demo_url)["connected"] is True
    assert len(sql_load_dataframe(demo_url, table_name="industrial_turbine_telemetry", limit=5)) == 5

    # Validator blocks PRAGMA, stacked queries, DDL/DML
    for evil_sql in (
        "PRAGMA database_list",
        "SELECT 1; ATTACH DATABASE '/tmp/pwn.db' AS pwn;",
        "DELETE FROM industrial_turbine_telemetry",
    ):
        with pytest.raises(SecurityError):
            sql_load_dataframe(demo_url, query=evil_sql)

    # C-level SQLite authorizer independently blocks ATTACH/PRAGMA/DDL
    eng, _ = _create_readonly_sql_engine(demo_url, allow_introspection_pragmas=False)
    try:
        with eng.connect() as conn:
            with pytest.raises(DatabaseError):
                conn.execute(text("ATTACH DATABASE '/tmp/bypass.db' AS pwn"))
    finally:
        eng.dispose()

    # SSRF & out-of-workspace DB blocks
    for evil_url in (
        "sqlite:////etc/passwd",
        "postgresql://u:p@169.254.169.254:5432/meta",
        "postgresql://u:p@127.0.0.1:5432/internal",
        "mysql://u:p@example.com/db?local_infile=1",
    ):
        with pytest.raises(SecurityError):
            sql_inspect_database(evil_url)
    with pytest.raises(SecurityError):
        access_inspect_database("/etc/passwd")

    # Stored XSS escaping in AutomatedEDA.to_html()
    df_xss = pd.DataFrame({
        "<script>alert(1)</script>": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "<img src=x onerror=alert(2)>": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    })
    html_out = AutomatedEDA().run(df_xss, generate_figures=False).to_html()
    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_out

    # Compiled SPA has zero document.write and enforces CSP + CSRF.
    # The bundle is code-split (P2-7), so scan EVERY emitted chunk — the CSRF
    # header lives in the shared entry chunk and never in a lazy studio chunk —
    # instead of assuming a single monolithic index-*.js.
    repo_root = Path(__file__).resolve().parents[2]
    static_dir = repo_root / "src/omniforge/ui/static_app"
    entry_chunks = list((static_dir / "assets").glob("index-*.js"))
    assert len(entry_chunks) == 1, f"expected exactly one entry chunk, found {entry_chunks}"
    all_chunks = list((static_dir / "assets").glob("*.js"))
    assert all_chunks, "no compiled JS chunks found - run `npm run build` in frontend/"
    sources = {chunk.name: chunk.read_text(encoding="utf-8") for chunk in all_chunks}
    assert all("document.write" not in src for src in sources.values()), "document.write leaked into a chunk"
    assert "X-CSRF-Token" in sources[entry_chunks[0].name], "CSRF header missing from the SPA entry chunk"
    assert any("Content-Security-Policy" in src for src in sources.values()), "CSP string missing from the SPA bundle"
    # The HTML shell must reference only assets that actually exist on disk.
    index_html = (static_dir / "index.html").read_text(encoding="utf-8")
    referenced = set(re.findall(r'(?:src|href)="/(assets/[^"]+)"', index_html))
    assert referenced, "index.html references no hashed assets"
    missing = sorted(ref for ref in referenced if not (static_dir / ref).is_file())
    assert not missing, f"index.html references missing assets: {missing}"


def test_p0_7_p0_8_p0_9_runtime_logging_docker_and_examples_hygiene(tmp_path: Path) -> None:
    """Verify R1-R5 runtime crash fixes, R13 logging hierarchy, Docker entrypoint, and lightweight examples/."""
    app = create_app()
    client = TestClient(app, headers={"X-API-Key": get_api_key()})
    hdr = {"X-Session-ID": "p0_runtime_regression"}

    r0 = client.post("/api/workbench/s01/load-sample", json={"source_type": "domain", "name": "finance", "n_samples": 100}, headers=hdr)
    assert r0.status_code == 200
    num_col = r0.json()["state"]["numeric_columns"][0]

    # R1: fdr_method -> multiple_testing_correction & mode="auto" list[str]
    r1 = client.post("/api/workbench/s03/run", json={"mode": "auto", "fdr_method": "bonferroni"}, headers=hdr)
    assert r1.status_code == 200
    assert r1.json()["report"]["criteria"]["multiple_testing_correction"] == "bonferroni"

    # R2: s08/simulate returns MonteCarloResult dict (not 2-tuple list)
    r2 = client.post("/api/workbench/s08/simulate", json={"target_col": num_col, "n_simulations": 250, "horizon_steps": 4}, headers=hdr)
    assert r2.status_code == 200 and isinstance(r2.json()["simulation"], dict)
    assert "var_95" in r2.json()["simulation"]

    # R3: s09/graph builds graph from real dataset columns
    edge_df = pd.DataFrame({"u": ["A", "A", "B", "C", "D", "E"], "v": ["B", "C", "D", "E", "F", "A"]})
    assert client.post(
        "/api/workbench/s01/upload",
        files={"file": ("g.csv", io.BytesIO(edge_df.to_csv(index=False).encode("utf-8")), "text/csv")},
        headers=hdr,
    ).status_code == 200
    r3 = client.post("/api/workbench/s09/graph", json={"source_col": "u", "target_col": "v", "run_supervised": False, "run_neural_gae": False}, headers=hdr)
    assert r3.status_code == 200 and r3.json()["structure"]["n_nodes"] == 6

    # R4: /api/v1/pipelines/execute serializes PipelineExecutionResult via to_dict()
    r4 = client.post(
        "/api/v1/pipelines/execute",
        json={
            "pipeline_spec": {"name": "reg_dag", "steps": [{"step_id": "s1", "step_type": "clean", "config": {}}]},
            "records": [{"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}],
        },
    )
    assert r4.status_code == 200 and r4.json()["status"] == "success"

    # R5: CLI `omniforge clean` works without AttributeError
    in_csv = tmp_path / "in.csv"
    out_csv = tmp_path / "out.csv"
    pd.DataFrame({"x": [1.0, 1.0, 2.0], "y": ["a", "a", "b"]}).to_csv(in_csv, index=False)
    res_cli = CliRunner().invoke(cli, ["clean", str(in_csv), "--out", str(out_csv)])
    normalized_out = " ".join(res_cli.output.split())
    assert res_cli.exit_code == 0 and "Rows: 3 -> 2" in normalized_out

    # R13: logging hierarchy uses "omniforge" root without double-prefixing
    log_file = tmp_path / "test.log"
    configure_logging(level="INFO", log_format="json", log_file=log_file)
    lg = get_logger("omniforge.api.main")
    assert lg._std_logger.name == "omniforge.api.main"
    assert logging.getLogger("omniforge").level == logging.INFO
    lg.info("p0_log_probe", secret_token="sk-abcdefghijklmno123456")
    for h in logging.getLogger("omniforge").handlers:
        h.flush()
    log_text = log_file.read_text(encoding="utf-8")
    assert "p0_log_probe" in log_text and "sk-abcdefghijklmno123456" not in log_text

    # P0-9 & P0-0: Docker entrypoint, compose, and lightweight examples/
    repo_root = Path(__file__).resolve().parents[2]
    ep = (repo_root / "docker/entrypoint.sh").read_text(encoding="utf-8")
    assert "exec omniforge" in ep and "exec uap" not in ep and "/var/lib/uap" not in ep
    assert "OMNIFORGE_API_WORKERS:-1" in ep
    dc = yaml.safe_load((repo_root / "docker/docker-compose.yml").read_text(encoding="utf-8"))
    assert "omniforge-api" in dc["services"] and "omniforge-ui" in dc["services"]

    leftover_data = [
        p for p in (repo_root / "examples").rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".xlsm"}
    ]
    assert leftover_data == [], f"Found leftover CSV/XLSX in examples/: {leftover_data}"
