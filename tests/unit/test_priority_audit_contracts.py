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
from omniforge.core.security import AuditCheckpointAnchor, AuditLogger
from omniforge.core.types import Principal, TaskType
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
        ("s06/automl", {}),
        ("s06/nonparametric", {}),
        ("s06/deep-learning", {}),
        ("s10/run", {"stages": ["quality"]}),
        ("s11/register", {}),
        ("s11/rollback", {"model_name": "audit-model"}),
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
