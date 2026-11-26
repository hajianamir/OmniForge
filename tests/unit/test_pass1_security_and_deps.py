"""Regression tests for Three-Pass Codebase Audit — Pass 1 (Security, SSRF, XML, SQL, Archive & Deps)."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from omniforge.core.exceptions import DataIngestionError, DependencyMissingError, SecurityError
from omniforge.data.connectors import _validate_remote_db_host, validate_outbound_http_url
from omniforge.data.ingestion import DataIngestionEngine, fast_read_tabular


def test_outbound_http_blocks_metadata_loopback_and_private_ips(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct SSRF targets (cloud IMDS, loopback, RFC1918, unspecified) must raise SecurityError."""
    monkeypatch.delenv("OMNIFORGE_ALLOW_PRIVATE_HTTP_HOSTS", raising=False)
    monkeypatch.delenv("OMNIFORGE_ALLOWED_HTTP_HOSTS", raising=False)

    engine = DataIngestionEngine()
    blocked_urls = [
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://169.254.169.254/bucket/dump.csv",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://127.0.0.1:8080/api/v1/internal",
        "http://127.0.0.1:8080/export.csv",
        "http://[::1]:8000/secret.json",
        "http://10.0.0.5/internal.parquet",
        "http://0.0.0.0/health",
    ]
    for bad_url in blocked_urls:
        with pytest.raises(SecurityError):
            engine.ingest(bad_url)


def test_ipv4_mapped_ipv6_metadata_blocked_even_when_private_hosts_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IPv4-mapped IPv6 link-local/metadata (::ffff:169.254.169.254) must never bypass private-host opt-in."""
    monkeypatch.setenv("OMNIFORGE_ALLOW_PRIVATE_HTTP_HOSTS", "true")
    monkeypatch.setenv("OMNIFORGE_ALLOW_PRIVATE_DB_HOSTS", "true")

    # Private RFC1918 is permitted under explicit opt-in:
    assert validate_outbound_http_url("http://10.10.1.5:8080/data.csv") == "http://10.10.1.5:8080/data.csv"

    # Link-local / cloud metadata via IPv4-mapped IPv6 MUST remain unconditionally blocked:
    with pytest.raises(SecurityError):
        validate_outbound_http_url("http://[::ffff:169.254.169.254]/latest/meta-data/")
    with pytest.raises(SecurityError):
        validate_outbound_http_url("http://[::ffff:0.0.0.0]/")
    with pytest.raises(SecurityError):
        _validate_remote_db_host("::ffff:169.254.169.254", 5432)
    with pytest.raises(SecurityError):
        _validate_remote_db_host("::ffff:0.0.0.0", 5432)
    # 6to4 transition embedding 169.254.169.254 (2002:a9fe:a9fe::1):
    with pytest.raises(SecurityError):
        validate_outbound_http_url("http://[2002:a9fe:a9fe::1]/latest/meta-data/")
    with pytest.raises(SecurityError):
        _validate_remote_db_host("2002:a9fe:a9fe::1", 5432)


def test_http_redirect_to_metadata_blocked_and_cross_origin_auth_stripped() -> None:
    """302 redirect to link-local/private IP must be blocked before hop 2; cross-origin redirects strip Auth."""
    engine = DataIngestionEngine()

    def ssrf_redirect_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "93.184.216.34":
            return httpx.Response(
                302,
                headers={"Location": "http://169.254.169.254/latest/meta-data/"},
            )
        raise AssertionError("SSRF target must never be contacted")

    with pytest.raises(SecurityError):
        engine.ingest(
            "https://93.184.216.34/api/records",
            _transport=httpx.MockTransport(ssrf_redirect_handler),
        )

    seen_headers_on_target: dict[str, str] = {}

    def cross_origin_redirect_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "93.184.216.34":
            return httpx.Response(
                302,
                headers={"Location": "https://1.1.1.1/public.csv"},
            )
        seen_headers_on_target.update(dict(request.headers))
        return httpx.Response(
            200,
            headers={"Content-Type": "text/csv"},
            content=b"a,b\n1,2\n3,4\n",
        )

    res = engine.ingest(
        "https://93.184.216.34/download.csv",
        headers={"Authorization": "Bearer super-secret", "X-Custom": "keep-me"},
        _transport=httpx.MockTransport(cross_origin_redirect_handler),
    )
    assert len(res.data) == 2
    assert "authorization" not in {k.lower() for k in seen_headers_on_target}
    assert seen_headers_on_target.get("x-custom") == "keep-me"


def test_http_response_size_cap_enforced() -> None:
    """Oversized HTTP responses must be rejected before exhausting memory."""
    engine = DataIngestionEngine()

    def huge_stream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "text/csv"}, content=b"x\n" * 2048)

    with pytest.raises(DataIngestionError, match="safety limit"):
        engine.ingest(
            "https://93.184.216.34/huge.csv",
            max_bytes=512,
            _transport=httpx.MockTransport(huge_stream_handler),
        )

    def loop_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://93.184.216.34/loop.csv"})

    with pytest.raises(DataIngestionError, match="maximum HTTP redirects"):
        engine.ingest(
            "https://93.184.216.34/loop.csv",
            max_redirects=2,
            _transport=httpx.MockTransport(loop_handler),
        )


def test_sanitize_uri_preserves_host_and_masks_query_secrets() -> None:
    """Password masking must not corrupt matching substrings in scheme/hostname, and must redact query tokens."""
    uri = "postgresql://admin:postgres@postgres.internal:5432/postgres_db?api_key=s3cr3t&sslmode=require"
    cleaned = DataIngestionEngine._sanitize_uri(uri)
    assert cleaned == "postgresql://admin:***@postgres.internal:5432/postgres_db?api_key=%2A%2A%2A&sslmode=require" or (
        cleaned == "postgresql://admin:***@postgres.internal:5432/postgres_db?api_key=***&sslmode=require"
    )
    assert "s3cr3t" not in cleaned
    assert "postgres.internal" in cleaned
    assert cleaned.startswith("postgresql://")


def test_xml_ingestion_defusedxml_fallback_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """XML fallback must parse via defusedxml when pandas read_xml fails, and fail closed if defusedxml is absent."""
    xml_path = tmp_path / "records.xml"
    xml_path.write_text("<root><row><a>10</a><b>ok</b></row></root>", encoding="utf-8")

    # 1. Force pandas.read_xml failure so fallback path executes via defusedxml
    def _boom(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise ValueError("lxml missing")

    monkeypatch.setattr(pd, "read_xml", _boom)
    engine = DataIngestionEngine(optimize_dtypes=False)
    res = engine.ingest(xml_path)
    assert len(res.data) == 1
    assert str(res.data.iloc[0]["b"]) == "ok"

    # 2. Simulate missing defusedxml -> must raise DependencyMissingError (never stdlib ET)
    real_import = builtins.__import__

    def _block_defusedxml(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "defusedxml" or name.startswith("defusedxml."):
            raise ImportError("No module named 'defusedxml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "defusedxml", raising=False)
    monkeypatch.delitem(sys.modules, "defusedxml.ElementTree", raising=False)
    monkeypatch.setattr(builtins, "__import__", _block_defusedxml)

    with pytest.raises(DependencyMissingError, match="defusedxml"):
        engine.ingest(xml_path)


def test_fast_read_tabular_handles_apostrophe_in_path(tmp_path: Path) -> None:
    """Paths containing single quotes must not break DuckDB SQL literal escaping."""
    sub = tmp_path / "client's_folder"
    sub.mkdir()
    csv_file = sub / "q4's_metrics.csv"
    csv_file.write_text("col_a,col_b\n1,2\n3,4\n", encoding="utf-8")

    for backend in ("auto", "pandas", "duckdb"):
        if backend == "duckdb":
            try:
                import duckdb  # noqa: F401
            except ImportError:
                continue
        df, used = fast_read_tabular(csv_file, backend=backend)
        assert len(df) == 2 and list(df.columns) == ["col_a", "col_b"]


def test_workspace_archive_spoofed_zip_header_bomb_and_member_hygiene(tmp_path: Path) -> None:
    """Spoofed ZipInfo.file_size headers, duplicate entries, traversal entries, and symlinks must be rejected."""
    import io
    import warnings
    import zipfile

    from omniforge.core.exceptions import DataValidationError
    from omniforge.core.workspace_session import (
        MANIFEST_FILENAME,
        STATE_FILENAME,
        WorkspaceManifest,
        WorkspaceSessionManager,
    )

    manifest_json = WorkspaceManifest(
        project_name="test_proj",
        saved_at_utc="2026-11-21T00:00:00Z",
    ).model_dump_json().encode("utf-8")

    # 1a. Bounded stream reader enforces max_bytes mid-stream and oversized archives fail closed
    big_payload = b"\x00" * (200 * 1024)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_FILENAME, manifest_json)
        zf.writestr(STATE_FILENAME, big_payload)
    valid_zip = buf.getvalue()

    with pytest.raises(DataValidationError, match="exceeds safety limit"):
        WorkspaceSessionManager.load_from_bytes(valid_zip, max_uncompressed_bytes=32 * 1024)

    with zipfile.ZipFile(io.BytesIO(valid_zip), mode="r") as zf_check:
        with pytest.raises(DataValidationError, match="exceeds safety limit"):
            WorkspaceSessionManager._read_zip_member_bounded(zf_check, STATE_FILENAME, max_bytes=16 * 1024)

    # 1b. Spoofed central-directory file_size header must fail closed with DataValidationError
    raw_zip = bytearray(valid_zip)
    orig_le = (200 * 1024).to_bytes(4, "little")
    fake_le = (16).to_bytes(4, "little")
    assert orig_le in raw_zip
    spoofed_zip = bytes(raw_zip).replace(orig_le, fake_le)
    with pytest.raises(DataValidationError):
        WorkspaceSessionManager.load_from_bytes(spoofed_zip, max_uncompressed_bytes=32 * 1024)

    # 1c. Non-DEFLATE/STORED compression (ZIP_LZMA / ZIP_BZIP2, which bypass chunked read bounds) must be blocked
    lzma_buf = io.BytesIO()
    with zipfile.ZipFile(lzma_buf, mode="w", compression=zipfile.ZIP_LZMA) as zf:
        zf.writestr(MANIFEST_FILENAME, manifest_json)
        zf.writestr(STATE_FILENAME, b"payload")
    with pytest.raises(DataValidationError, match="unsupported ZIP compression method"):
        WorkspaceSessionManager.inspect_bytes(lzma_buf.getvalue())

    # 2. Duplicate ZIP member names (entry shadowing) must be rejected
    dup_buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(dup_buf, mode="w") as zf:
            zf.writestr(MANIFEST_FILENAME, manifest_json)
            zf.writestr(MANIFEST_FILENAME, manifest_json)
            zf.writestr(STATE_FILENAME, b"stub")
    with pytest.raises(DataValidationError, match="duplicate ZIP member"):
        WorkspaceSessionManager.inspect_bytes(dup_buf.getvalue())

    # 3. Path traversal member names must be rejected
    trav_buf = io.BytesIO()
    with zipfile.ZipFile(trav_buf, mode="w") as zf:
        zf.writestr(MANIFEST_FILENAME, manifest_json)
        zf.writestr(STATE_FILENAME, b"stub")
        zf.writestr("../escape.txt", b"bad")
    with pytest.raises(DataValidationError, match="unsafe ZIP member path"):
        WorkspaceSessionManager.inspect_bytes(trav_buf.getvalue())

    # 4. delete_local_project must refuse symlinks
    victim = tmp_path / "important.omniforge"
    victim.write_bytes(b"keep")
    link = tmp_path / "link.omniforge"
    link.symlink_to(victim)
    assert WorkspaceSessionManager.delete_local_project(link) is False
    assert victim.exists()


def test_future_scoring_sql_readonly_and_credential_redaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FutureTargetPredictor.load_from_sql must block DDL/DML/stacked SQL and redact credentials on error."""
    from omniforge.mlops.future_scoring import FutureTargetPredictor

    monkeypatch.setenv("OMNIFORGE_ALLOWED_DB_DIRS", str(tmp_path))

    ref = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "feat_b": ["x", "y", "x"]})
    db_path = tmp_path / "future_test.sqlite"
    uri, query, _ = FutureTargetPredictor.create_sample_sqlite_db(
        ref, ["feat_a", "feat_b"], db_path, table_name="future;drop"
    )
    assert query == 'SELECT * FROM "future_drop"'
    loaded = FutureTargetPredictor.load_from_sql(uri, query)
    assert len(loaded) == 12

    for bad_sql in (
        "DROP TABLE future_drop",
        "DELETE FROM future_drop",
        'SELECT * FROM "future_drop"; DROP TABLE "future_drop"',
        "PRAGMA writable_schema = ON",
    ):
        with pytest.raises(SecurityError):
            FutureTargetPredictor.load_from_sql(uri, bad_sql)

    with pytest.raises((DataIngestionError, SecurityError)) as exc_info:
        FutureTargetPredictor.load_from_sql(
            "sqlite://admin:super_secret_pw@nonexistent_dir_xyz/missing.db",
            "SELECT 1",
        )
    assert "super_secret_pw" not in str(exc_info.value)

    with pytest.raises((DataIngestionError, SecurityError)) as w_exc:
        FutureTargetPredictor.write_predictions_to_sql(
            loaded,
            "sqlite://admin:super_secret_pw@nonexistent_dir_xyz/missing.db",
        )
    assert "super_secret_pw" not in str(w_exc.value)


def test_core_only_imports_succeed_without_sqlalchemy_or_matplotlib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """omniforge.mlops and omniforge.reporting must import cleanly without optional sqlalchemy/matplotlib."""
    import importlib

    import omniforge.mlops.future_scoring as fs_mod
    import omniforge.reporting.charts as charts_mod

    real_import = builtins.__import__

    def _block_optional(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("sqlalchemy", "matplotlib") or name.startswith(("sqlalchemy.", "matplotlib.")):
            raise ImportError(f"Blocked optional module {name}")
        return real_import(name, *args, **kwargs)

    for key in list(sys.modules):
        if key in ("sqlalchemy", "matplotlib") or key.startswith(("sqlalchemy.", "matplotlib.")):
            monkeypatch.delitem(sys.modules, key, raising=False)
    monkeypatch.setattr(builtins, "__import__", _block_optional)

    reloaded_fs = importlib.reload(fs_mod)
    reloaded_charts = importlib.reload(charts_mod)

    # Non-SQL helper works in core-only install:
    tmpl = reloaded_fs.FutureTargetPredictor.build_manual_template(
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}), ["x"], n_rows=2
    )
    assert len(tmpl) == 2

    # SQL and chart rendering fail closed with DependencyMissingError:
    with pytest.raises(DependencyMissingError, match="sqlalchemy"):
        reloaded_fs.FutureTargetPredictor.load_from_sql("sqlite:///:memory:", "SELECT 1")

    reloaded_charts.plt._mod = None
    with pytest.raises(DependencyMissingError, match="matplotlib"):
        _ = reloaded_charts.plt.subplots


def test_capabilities_core_pack_includes_defusedxml_and_intake_audit_clears_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CORE_RUNTIME must include defusedxml and failed intake audit refreshes must never leave stale reports."""
    from omniforge.capabilities import CORE_RUNTIME
    from omniforge.data.intake_audit import InitialIntakeAuditor
    from omniforge.ui.state import get_session_state, revert_to_raw_dataset

    assert "defusedxml" in CORE_RUNTIME.imports
    assert any(d.startswith("defusedxml>=") for d in CORE_RUNTIME.dists)

    st = get_session_state("test_stale_intake_clear")
    st["uap_raw_df"] = pd.DataFrame({"a": [1, 2, 3]})
    st["uap_df"] = pd.DataFrame({"a": [1, 2]})
    st["uap_intake_audit"] = {"stale": True}

    def _raise_audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("simulated audit failure")

    monkeypatch.setattr(InitialIntakeAuditor, "audit", _raise_audit)
    assert revert_to_raw_dataset(st) is True
    assert st["uap_intake_audit"] is None


def test_top_level_package_lazy_exports_and_stdlib_only_capabilities() -> None:
    """omniforge.__all__ symbols must resolve lazily, and omniforge.capabilities must work without numpy."""
    import subprocess

    import omniforge

    for sym in omniforge.__all__:
        assert getattr(omniforge, sym) is not None
    with pytest.raises(AttributeError):
        _ = omniforge.__nonexistent_symbol_xyz__  # type: ignore[attr-defined]

    repo_root = Path(__file__).resolve().parents[2]
    # Simulate bare embeddable Python where numpy/pandas are absent (sys.modules['numpy'] = None):
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(repo_root / 'src')!r}); "
        "sys.modules['numpy'] = None; "
        "from omniforge.capabilities import ALL_PACKS, CORE_RUNTIME; "
        "assert len(ALL_PACKS) >= 10 and 'defusedxml' in CORE_RUNTIME.imports"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr



