"""Unit tests for Core Infrastructure, Security, Carbon Tracking, and Config Composition."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from omniforge.config.loader import load_settings
from omniforge.core.carbon import CarbonTracker
from omniforge.core.exceptions import AuthorizationError, DataIngestionError, ErrorSeverity
from omniforge.core.logging import redact_sensitive_value
from omniforge.core.security import (
    AccessController,
    AuditLogger,
    ComplianceValidator,
    EncryptionManager,
    JWTManager,
    PIIRedactor,
)
from omniforge.core.types import Principal
from omniforge.version import get_version_info


def test_version_and_exception_hierarchy() -> None:
    info = get_version_info()
    assert info["name"] == "OmniForge"
    assert info["version"] == "4.0.0"
    err = DataIngestionError("Bad CSV", context={"path": "x.csv"}, suggestions=["Check delimiter"])
    payload = err.to_dict()
    assert payload["error_code"].startswith("OmniForge-")
    assert payload["severity"] == ErrorSeverity.HIGH.value
    assert "trace_id" in payload and len(payload["trace_id"]) > 0
    assert payload["suggestions"] == ["Check delimiter"]


def test_secret_redaction_and_security(workspace_tmp: Path) -> None:
    redacted = redact_sensitive_value({"api_key": "sk-123456", "nested": {"password": "secret"}, "safe": "ok"})
    assert redacted["api_key"] == "***REDACTED***"
    assert redacted["nested"]["password"] == "***REDACTED***"
    assert redacted["safe"] == "ok"

    enc = EncryptionManager(key="unit-test-secret-key-at-least-32-chars!!")
    cipher = enc.encrypt_bytes(b"sensitive-payload")
    assert enc.decrypt_bytes(cipher) == b"sensitive-payload"

    jwt_mgr = JWTManager(secret_key="jwt-unit-test-secret-key-at-least-32-chars!!")
    principal = Principal(subject="alice", roles=["data_scientist"], tenant_id="acme")
    tok = jwt_mgr.create_token(principal)
    decoded = jwt_mgr.verify_token(tok)
    assert decoded.subject == "alice"
    assert decoded.tenant_id == "acme"

    ac = AccessController()
    assert ac.authorize(decoded, "model:train", resource_tenant_id="acme") is True
    viewer = Principal(subject="bob", roles=["viewer"], tenant_id="acme")
    with pytest.raises(AuthorizationError):
        ac.authorize(viewer, "model:deploy")

    audit = AuditLogger(log_path=workspace_tmp / "audit.jsonl")
    audit.log_event(actor="alice", action="train_model", resource_type="model", resource_id="credit_v1", status="success")
    audit.log_event(actor="alice", action="promote", resource_type="model", resource_id="credit_v1", status="success")
    valid, count = audit.verify_chain()
    assert valid is True and count == 2

    pii_df = pd.DataFrame({"customer_email": ["a@b.com"], "ssn_number": ["123-45-6789"], "amount": [50.0]})
    detected = PIIRedactor().scan_dataframe(pii_df)
    assert "customer_email" in detected and "ssn_number" in detected and "amount" not in detected

    comp = ComplianceValidator(profiles=["GDPR", "HIPAA"]).evaluate_dataset(pii_df)
    assert isinstance(comp, dict)


def test_config_composition_and_carbon_tracker() -> None:
    settings = load_settings(domain="finance", overrides=["automl.time_budget_seconds=90"])
    assert settings.automl.time_budget_seconds == 90
    assert settings.domain_overrides["domain"]["name"] == "finance"

    with CarbonTracker(operation_name="unit_test_task", country_iso="DEU") as tracker:
        _ = sum(i * i for i in range(100_000))
    assert tracker.report is not None
    assert tracker.report.duration_seconds > 0
    assert 0.0 <= tracker.report.green_score <= 100.0
