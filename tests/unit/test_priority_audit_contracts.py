"""Regression tests for Workbench permissions and bounded audit disclosures."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from omniforge.api import workbench as wb
from omniforge.api.auth import jwt_manager
from omniforge.api.main import create_app
from omniforge.core.exceptions import ComplianceError, DataValidationError
from omniforge.core.security import AuditCheckpointAnchor, AuditLogger, ComplianceValidator, PIIRedactor
from omniforge.core.types import Principal, TaskType
from omniforge.data.synthetic import SyntheticDataGenerator
from omniforge.models.classical import ClassicalMLModel
from omniforge.models.nlp.advanced_nlp import AdvancedNLPSuite


def test_workbench_mutations_require_action_permissions() -> None:
    client = TestClient(create_app())
    tenant = f"registry-test-{uuid4().hex}"
    viewer = Principal(subject="viewer", roles=["viewer"], tenant_id=tenant)
    scientist = Principal(subject="scientist", roles=["data_scientist"], tenant_id=tenant)
    operator = Principal(subject="operator", roles=["mlops_engineer"], tenant_id=tenant)

    def headers(principal: Principal) -> dict[str, str]:
        return {"Authorization": f"Bearer {jwt_manager.create_token(principal)}"}

    for endpoint, payload in [
        ("s02/clustering/commit", {}),
        ("s05/engineer", {"commit": True}),
        ("s06/automl", {}),
        ("s06/nonparametric", {}),
        ("s06/deep-learning", {}),
        ("s10/run", {"stages": ["quality"]}),
        ("s11/register", {}),
        ("s11/rollback", {"model_name": "audit-model"}),
        ("s12/compile", {}),
        ("s13/chat", {"message": "Summarize the study"}),
        ("s14/quick-train", {}),
    ]:
        assert client.post(f"/api/workbench/{endpoint}", json=payload, headers=headers(viewer)).status_code == 403
    files = {"file": ("demo.csv", b"x,y\n1,2\n", "text/csv")}
    assert client.post("/api/workbench/s01/upload", files=files, headers=headers(viewer)).status_code == 403
    assert client.post("/api/workbench/s01/upload", files=files, headers=headers(scientist)).status_code == 200
    assert client.post("/api/workbench/s10/run", json={"stages": ["automl"]}, headers=headers(operator)).status_code == 403

    frame = pd.DataFrame({"x": range(12)})
    model = ClassicalMLModel("ridge", task_type=TaskType.REGRESSION).fit(frame, frame["x"])
    for principal in (scientist, operator):
        wb.get_session_state(wb._sid("default", principal))["uap_active_model"] = model
    register = "/api/workbench/s11/register"
    model_name = "audit-model"
    staging = client.post(register, json={"model_name": model_name, "stage": "Staging"}, headers=headers(scientist))
    assert staging.status_code == 200, staging.text
    production = {"model_name": model_name, "stage": "Production"}
    assert client.post(register, json=production, headers=headers(scientist)).status_code == 403
    allowed = client.post(register, json=production, headers=headers(operator))
    assert allowed.status_code == 200, allowed.text
    allowed = client.post(register, json=production, headers=headers(operator))
    assert allowed.status_code == 200, allowed.text
    rollback = "/api/workbench/s11/rollback"
    assert client.post(rollback, json={"model_name": model_name}, headers=headers(scientist)).status_code == 403
    restored = client.post(rollback, json={"model_name": model_name}, headers=headers(operator))
    assert restored.status_code == 200, restored.text
    assert restored.json()["restored"]["version"] == 2


@pytest.mark.parametrize("contents", ["", " \n"])
def test_nonempty_log_requires_anchor_evidence(tmp_path: Path, contents: str) -> None:
    log_path, anchor_path = tmp_path / "audit.jsonl", tmp_path / "anchor.jsonl"
    logger = AuditLogger(log_path)
    logger.log_event(action="read", resource_type="dataset", resource_id="test")
    anchor_path.write_text(contents, encoding="utf-8")
    ok, checked, detail = AuditCheckpointAnchor.verify_against_log(log_path, anchor_path)
    assert (ok, checked) == (False, 0)
    assert "empty" in detail


def test_nlp_screening_returns_categories_without_raw_matches() -> None:
    address = "alex@example.invalid"
    result = AdvancedNLPSuite().screen_toxicity_and_pii([f"Contact {address}."])[0]
    assert result["pii_count"] == 1
    assert result["pii_entities"] == [{"type": "EMAIL"}]
    assert "[EMAIL_REDACTED]" in result["redacted_text"]
    assert address not in repr(result)


def test_pii_screening_reports_coverage_and_checks_identifier_formats() -> None:
    df = pd.DataFrame({
        "notes": ["neutral"] * 250 + ["person@example.invalid"],
        "reference": [1] * 250 + [4111111111111111],
    })
    scanner = PIIRedactor()
    scan = scanner.scan_dataframe_report(df, sample_size=100)
    assert scan["mode"] == "systematic_sample"
    assert (scan["rows_scanned"], scan["total_rows"]) == (100, 251)
    assert scan["columns"] == {"notes": ["value_pattern:email"], "reference": ["value_pattern:credit_card"]}
    assert scanner.scan_dataframe_report(df, sample_size=None)["mode"] == "all_rows"
    assert scanner.detect_text_pii("999.999.999.999 and 9999-9999-9999-9999").keys().isdisjoint({"ipv4", "credit_card"})
    assert scanner.redact_text("4111 1111 1111 1111") == "[REDACTED]:CREDIT_CARD"


def test_compliance_screening_only_reports_supported_checks_and_explicit_evidence() -> None:
    df = pd.DataFrame({"measurement": [1.0, 2.0]})
    validator = ComplianceValidator(["GDPR", "HIPAA", "EU AI Act"])
    report = validator.evaluate_dataset(df)
    assert report["assessment_status"] == "review_required"
    assert report["compliant"] is None
    assert report["profiles_evaluated"] == ["GDPR", "HIPAA"]
    assert report["unsupported_profiles"] == ["EU AI ACT"]
    assert any("Encryption evidence" in item for item in report["missing_evidence"])
    with pytest.raises(ComplianceError, match="lacks sufficient evidence"):
        validator.evaluate_dataset(df, strict=True)
    evaluated = ComplianceValidator(["HIPAA"]).evaluate_dataset(df, encryption_enabled=True, strict=True)
    assert evaluated["assessment_status"] == "checks_passed"
    assert evaluated["compliant"] is None


def test_privacy_budget_requires_public_domains_and_fixed_bounds() -> None:
    frame = pd.DataFrame({"amount": [1.0, 100000.0], "kind": ["public", "outside_domain"]})
    gen = SyntheticDataGenerator()
    with pytest.raises(DataValidationError, match="public bounds"):
        gen.fit_generate_copula(frame, dp_epsilon=1.0)
    with pytest.raises(DataValidationError, match="public category domain"):
        gen.fit_generate_copula(frame, dp_epsilon=1.0, dp_bounds={"amount": (2.0, 2.0)})
    released = gen.fit_generate_copula(
        frame, n_samples=30, dp_epsilon=1.0,
        dp_bounds={"amount": (2.0, 2.0)}, dp_categories={"kind": ["public"]},
    )
    assert released["amount"].eq(2.0).all() and released["kind"].eq("public").all()
    assert released.attrs["is_synthetic"] is True


def test_categorical_leakage_check_requires_repeated_transferable_mapping() -> None:
    from omniforge.core.contracts import LeakageGuard

    frame = pd.DataFrame({
        "identifier": range(16),
        "proxy": ["left", "right"] * 8,
        "unstable": ["red", "blue"] * 4 + ["blue", "red"] * 4,
        "label": ["a", "b"] * 8,
    })
    result = LeakageGuard.validate_split(frame.iloc[:8], frame.iloc[8:], target_col="label")
    assert result.target_leakage_columns == ["proxy"]


def test_partial_correlation_accounts_for_intercept_and_redundant_controls() -> None:
    import numpy as np

    from omniforge.mlops.root_cause import CausalDiscovery

    rng = np.random.default_rng(42)
    z = rng.normal(100, 1, 2000)
    frame = pd.DataFrame({"x": 2 * z + 150 + rng.normal(size=2000),
                          "y": 3 * z - 150 + rng.normal(size=2000),
                          "z": z, "duplicate": 2 * z + 20})
    design = np.column_stack([np.ones(len(z)), z])
    outcomes = frame[["x", "y"]].to_numpy()
    residual = outcomes - design @ np.linalg.lstsq(design, outcomes, rcond=None)[0]
    expected_r = np.corrcoef(residual.T)[0, 1]
    ordinary = CausalDiscovery._partial_corr(frame, "x", "y", ["z"])
    redundant = CausalDiscovery._partial_corr(frame, "x", "y", ["z", "duplicate"])
    assert ordinary[0] == pytest.approx(expected_r)
    assert redundant == pytest.approx(ordinary)
