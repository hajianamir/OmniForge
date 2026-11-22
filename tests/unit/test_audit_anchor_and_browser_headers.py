"""Regression tests for the P1-11 browser-security headers and the optional
out-of-band audit-log checkpoint anchor (RFC 9162-style).

These tests pin the *default* posture (CSP is Report-Only, HSTS on, anchor opt-in)
and the anchor's tamper-detection semantics: a log re-chained without the checkpoint
key must fail `verify_anchor` even when the in-band chain verifies.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from omniforge.api.main import create_app
from omniforge.config.settings import get_settings
from omniforge.core.exceptions import SecurityError
from omniforge.core.security import AuditCheckpointAnchor, AuditLogger

CHECKPOINT_KEY = "anchor-test-key-0123456789abcdef-test-only"


def _events(log_path: Path, count: int, *, anchor_path: Path | None = None, anchor_every: int | None = None):
    logger = AuditLogger(log_path=log_path, anchor_path=anchor_path, checkpoint_key=CHECKPOINT_KEY, anchor_every=anchor_every)
    for i in range(count):
        logger.log_event(action=f"action_{i}", resource_type="dataset", resource_id=f"ds_{i}", details={"n": i})
    return logger


def test_security_headers_csp_report_only_and_hsts_present() -> None:
    app = create_app()
    anon = TestClient(app, raise_server_exceptions=False)
    r = anon.get("/health")
    assert r.status_code == 200
    csp = r.headers.get("content-security-policy-report-only")
    assert csp is not None and "default-src 'self'" in csp and "object-src 'none'" in csp
    hsts = r.headers.get("strict-transport-security", "")
    assert hsts.startswith("max-age=") and "max-age=0" not in hsts
    # Pre-existing hardening headers are still emitted.
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "SAMEORIGIN"
    # Report-Only means the browser never blocks; the enforced header must NOT appear.
    assert "content-security-policy" not in {k.lower() for k in r.headers if k.lower() != "content-security-policy-report-only"}


def test_security_header_settings_are_report_only_by_default() -> None:
    api = get_settings().api
    assert api.csp_report_only_enabled is True
    assert "default-src 'self'" in api.csp_report_only_policy
    assert api.hsts_enabled is True and api.hsts_max_age > 0


def test_anchor_disabled_by_default(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=log)
    assert logger.anchor_enabled is False
    logger.log_event(action="a", resource_type="r", resource_id="x")
    ok, checked, detail = logger.verify_anchor()
    assert (ok, checked) == (True, 0) and "not configured" in detail
    chain_ok, count, _ = logger.verify_with_anchor()
    assert chain_ok and count == 1


def test_anchor_every_event_and_verify_ok(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    anchor = tmp_path / "anchor.jsonl"
    logger = _events(log, 3, anchor_path=anchor)
    assert logger.anchor_enabled is True
    lines = [ln for ln in anchor.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 3
    records = [json.loads(ln) for ln in lines]
    assert [r["seq"] for r in records] == [1, 2, 3]
    assert [r["record_count"] for r in records] == [1, 2, 3]
    ok, checked, detail = logger.verify_anchor()
    assert ok and checked == 3 and "anchored through record 3 of 3" in detail
    chain_ok, count, detail = logger.verify_with_anchor()
    assert chain_ok and count == 3 and "record 3 of 3" in detail


def test_anchor_cadence_every_n(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    anchor = tmp_path / "anchor.jsonl"
    _events(log, 5, anchor_path=anchor, anchor_every=2)
    records = [json.loads(ln) for ln in anchor.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert [r["record_count"] for r in records] == [2, 4]
    ok, checked, _detail = AuditCheckpointAnchor.verify_against_log(log, anchor, CHECKPOINT_KEY)
    assert ok and checked == 2


def test_anchor_detects_log_rechained_without_checkpoint_key(tmp_path: Path) -> None:
    """Attacker rewrites the JSONL with its own valid in-band chain; anchor must still reject it."""
    victim = tmp_path / "victim.jsonl"
    anchor = tmp_path / "anchor.jsonl"
    _events(victim, 2, anchor_path=anchor)

    attacker_log = tmp_path / "attacker" / "audit.jsonl"
    _events(attacker_log, 2)  # same default in-band signer => valid in-band chain
    shutil.copyfile(attacker_log, victim)

    chain_ok, _ = AuditLogger.verify_file(victim)
    assert chain_ok, "test premise: in-band chain alone cannot detect the re-chain"
    ok, _checked, detail = AuditCheckpointAnchor.verify_against_log(victim, anchor, CHECKPOINT_KEY)
    assert not ok and "disagrees" in detail
    reloaded = AuditLogger(log_path=victim, anchor_path=anchor, checkpoint_key=CHECKPOINT_KEY)
    chain_ok, _count, detail = reloaded.verify_with_anchor()
    assert not chain_ok and "disagrees" in detail


def test_anchor_tampered_file_refused_at_load(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    anchor = tmp_path / "anchor.jsonl"
    _events(log, 2, anchor_path=anchor)
    # Flip one character inside the first MAC, keeping JSON valid.
    text = anchor.read_text(encoding="utf-8")
    idx = text.find('"mac"') + 7
    tampered = text[:idx] + ("0" if text[idx] != "0" else "1") + text[idx + 1 :]
    anchor.write_text(tampered, encoding="utf-8")
    with pytest.raises(SecurityError):
        AuditLogger(log_path=log, anchor_path=anchor, checkpoint_key=CHECKPOINT_KEY)


def test_anchor_missing_file_fails_verification(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    anchor = tmp_path / "anchor.jsonl"
    logger = _events(log, 1, anchor_path=anchor)
    anchor.unlink()
    ok, _checked, detail = logger.verify_anchor()
    assert not ok and "not found" in detail


def test_anchor_trailing_checkpoint_rollback_is_visible_in_coverage(tmp_path: Path) -> None:
    """Deleting the tail checkpoint is not a chain break, but must reduce reported coverage."""
    log = tmp_path / "audit.jsonl"
    anchor = tmp_path / "anchor.jsonl"
    logger = _events(log, 2, anchor_path=anchor)
    lines = anchor.read_text(encoding="utf-8").splitlines()
    anchor.write_text(lines[0] + "\n", encoding="utf-8")  # drop the record_count=2 checkpoint
    ok, checked, detail = logger.verify_anchor()
    assert ok and checked == 1 and "record 1 of 2" in detail


def test_anchor_env_configuration_and_explicit_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log = tmp_path / "audit.jsonl"
    anchor = tmp_path / "anchor.jsonl"
    monkeypatch.setenv("OMNIFORGE_AUDIT__ANCHOR_PATH", str(anchor))
    monkeypatch.setenv("OMNIFORGE_AUDIT__ANCHOR_EVERY", "1")
    monkeypatch.setenv("OMNIFORGE_AUDIT__CHECKPOINT_KEY", CHECKPOINT_KEY)
    logger = AuditLogger(log_path=log)
    assert logger.anchor_enabled is True
    logger.log_event(action="a", resource_type="r", resource_id="x")
    ok, _checked, _detail = logger.verify_anchor()
    assert ok
    monkeypatch.setenv("OMNIFORGE_AUDIT__ANCHOR_PATH", "off")
    assert AuditLogger(log_path=tmp_path / "audit2.jsonl").anchor_enabled is False


def test_anchor_invalid_every_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIFORGE_AUDIT__ANCHOR_EVERY", "abc")
    with pytest.raises(SecurityError):
        AuditLogger(log_path=tmp_path / "a.jsonl")
    monkeypatch.setenv("OMNIFORGE_AUDIT__ANCHOR_EVERY", "0")
    with pytest.raises(SecurityError):
        AuditLogger(log_path=tmp_path / "b.jsonl")


def test_anchor_production_requires_checkpoint_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIFORGE_ENVIRONMENT", "production")
    monkeypatch.delenv("OMNIFORGE_AUDIT__CHECKPOINT_KEY", raising=False)
    with pytest.raises(SecurityError):
        AuditLogger(log_path=tmp_path / "prod.jsonl", anchor_path=tmp_path / "anchor.jsonl")


def test_in_band_flip_one_char_still_detected(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    _events(log, 2)
    raw = log.read_text(encoding="utf-8")
    idx = raw.find('"resource_id"') + 16
    tampered = raw[:idx] + ("A" if raw[idx] != "A" else "B") + raw[idx + 1 :]
    log.write_text(tampered, encoding="utf-8")
    ok, count = AuditLogger.verify_file(log)
    assert not ok and count < 2
