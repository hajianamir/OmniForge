"""Regression coverage for registry stage validation and undefined fairness metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from omniforge.core.exceptions import MLOpsError
from omniforge.core.types import ModelStage
from omniforge.evaluation.fairness import FairnessAuditor
from omniforge.mlops.registry import ModelRegistry


def _registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(db_path=tmp_path / "registry.db", artifact_dir=tmp_path / "artifacts")


def _model() -> DummyRegressor:
    return DummyRegressor().fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0, 3.0]))


def test_registry_accepts_stage_strings_and_enforces_governance(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    with pytest.raises(MLOpsError, match="fairness gate"):
        registry.register_model(_model(), "credit", stage=" production ", fairness_passed=False)
    assert registry.list_versions() == []
    assert not (tmp_path / "artifacts" / "credit").exists()

    first = registry.register_model(_model(), "credit", stage="production", fairness_passed=True)
    second = registry.register_model(_model(), "credit", stage=ModelStage.PRODUCTION, fairness_passed=True)
    assert first.stage is ModelStage.PRODUCTION
    assert second.stage is ModelStage.PRODUCTION
    versions = {version.version: version.stage for version in registry.list_versions("credit")}
    assert versions == {1: ModelStage.ARCHIVED, 2: ModelStage.PRODUCTION}

    updated = registry.transition_stage("credit", second.version, " staging ")
    assert updated.stage is ModelStage.STAGING
    with pytest.raises(MLOpsError, match="Invalid model stage"):
        registry.transition_stage("credit", second.version, "wrong")
    assert registry.get_version("credit", second.version).stage is ModelStage.STAGING


def test_registry_rejects_invalid_stage_before_saving(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    with pytest.raises(MLOpsError, match="Invalid model stage"):
        registry.register_model(_model(), "credit", stage="not-a-stage")
    assert registry.list_versions() == []
    assert not (tmp_path / "artifacts" / "credit").exists()


def test_registry_roundtrips_undefined_equalized_odds(tmp_path: Path) -> None:
    report = FairnessAuditor().audit(
        y_true=np.array([0, 0, 0, 1]),
        y_pred=np.array([0, 1, 0, 1]),
        sensitive_features=np.array(["a", "a", "b", "b"]),
    )
    assert report.passed is False
    assert report.equalized_odds_diff is None
    registry = _registry(tmp_path)
    version = registry.register_model(_model(), "credit", fairness_report=report)
    assert version.fairness_passed is False
    assert version.metrics["equalized_odds_diff"] is None
    stored = registry.list_versions("credit")[0]
    assert stored.metrics["equalized_odds_diff"] is None
    assert json.loads(stored.model_dump_json())["metrics"]["equalized_odds_diff"] is None


def test_strict_gate_rejects_unbound_and_reassigned_evidence(tmp_path: Path) -> None:
    import pandas as pd

    from omniforge.mlops.fairness_evidence import FairnessEvidence

    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "artifacts", require_fairness_gate=True)
    X_test = pd.DataFrame({"x": [4.0, 5.0, 6.0, 7.0]}, index=[4, 5, 6, 7])
    y_test = pd.Series([0, 1, 0, 1], index=X_test.index)
    groups = pd.Series(["A", "A", "B", "B"], index=X_test.index)
    report = FairnessAuditor().audit(y_test, y_test, groups)
    assert report.passed
    model = _model()
    evidence = FairnessEvidence.capture(model, X_test, y_test, groups, report)
    with pytest.raises(MLOpsError, match="does not match the model"):
        registry.register_model(DummyRegressor(strategy="median").fit([[0], [1]], [0, 8]), "credit",
                                fairness_evidence=evidence)
    with pytest.raises(MLOpsError, match="fairness gate"):
        registry.register_model(model, "credit", stage="Production", fairness_passed=True)
    version = registry.register_model(model, "credit", stage="Staging", fairness_evidence=evidence, encrypt=True)
    assert version.fairness_evidence is not None
    assert version.fairness_evidence.holdout_fingerprint == evidence.holdout_fingerprint
    assert version.fairness_evidence.artifact_checksum == version.checksum_sha256
    assert version.fairness_evidence.assessment_status == "thresholds_met"
    legacy_report = report.model_copy(update={"assessment_status": None})
    legacy_evidence = FairnessEvidence.capture(model, X_test, y_test, groups, legacy_report)
    legacy = registry.register_model(model, "legacy", stage="Staging", fairness_evidence=legacy_evidence)
    with pytest.raises(MLOpsError, match="requires a current fairness assessment"):
        registry.transition_stage("legacy", legacy.version, "Production")
    unbound = registry.register_model(model, "credit", stage="Staging", fairness_passed=True)
    with pytest.raises(MLOpsError, match="saved with the target version"):
        registry.transition_stage("credit", unbound.version, "Production", fairness_report=report)
    with pytest.raises(MLOpsError, match="fairness gate"):
        registry.transition_stage("credit", unbound.version, "Production")
    promoted = registry.transition_stage("credit", version.version, "Production")
    assert promoted.stage is ModelStage.PRODUCTION
    with pytest.raises(MLOpsError, match="fairness gate"):
        registry.rollback_production("credit", target_version=unbound.version)
    assert registry.get_version("credit", version.version).stage is ModelStage.PRODUCTION
    Path(version.artifact_path).write_bytes(b"incomplete catalog artifact")
    with pytest.raises(MLOpsError, match="artifact does not match"):
        registry.transition_stage("credit", version.version, "Production")


def test_workbench_fairness_is_bound_to_trained_holdout_and_registration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd
    from fastapi.testclient import TestClient
    from sklearn.linear_model import LinearRegression, LogisticRegression

    from omniforge.api import workbench as wb
    from omniforge.api.auth import get_api_key
    from omniforge.api.main import create_app
    from omniforge.config.settings import get_settings
    from omniforge.core.types import Principal
    from omniforge.ui.state import load_new_dataset, record_active_holdout

    registry = ModelRegistry(tmp_path / "registry.db", tmp_path / "artifacts", require_fairness_gate=True)
    monkeypatch.setattr(wb, "_SHARED_REGISTRY", registry)
    session = "test-bound-fairness"
    principal = Principal(subject="api_key_principal", roles=["admin"], tenant_id=get_settings().default_tenant_id)
    state = wb.get_session_state(wb._sid(session, principal))
    data = pd.DataFrame({"x": [-6, 6, -5, 5, -4, 4, -3, 3], "target": [0, 1] * 4,
                         "group": ["A", "B", "A", "B", "A", "A", "B", "B"],
                         "unit": [1, 2, 1, 2, 3, 4, 3, 4]})
    load_new_dataset(state, data, "evidence", "test", default_target="target")
    train, test = data.iloc[:4], data.iloc[4:]
    state["uap_active_model"] = LogisticRegression().fit(train[["x"]], train["target"])
    state["uap_active_X_test"] = test[["x"]]
    state["uap_active_y_test"] = test["target"]
    record_active_holdout(state)
    client = TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": session})
    fairness = client.post("/api/workbench/s07/fairness", json={"sensitive_col": "group"})
    assert fairness.status_code == 200, fairness.text
    evidence = fairness.json()["promotion_evidence"]
    assert evidence["n_holdout_rows"] == 4 and fairness.json()["fairness"]["passed"]
    state["uap_active_X_test"] = test[["x"]] * 2
    invalid = client.post("/api/workbench/s11/register", json={"model_name": "credit"})
    assert invalid.status_code == 409, invalid.text
    state["uap_active_X_test"] = test[["x"]]
    registered = client.post("/api/workbench/s11/register", json={"model_name": "credit"})
    assert registered.status_code == 200, registered.text
    stored = registered.json()["registered"]["fairness_evidence"]
    assert stored["holdout_fingerprint"] == evidence["holdout_fingerprint"]
    assert stored["version"] == 1
    state["uap_fairness_result"] = {"passed": False}
    state["uap_fairness_evidence"] = None
    promoted = client.post("/api/workbench/s11/promote", json={"model_name": "credit", "version": 1, "stage": "Production"})
    assert promoted.status_code == 200, promoted.text
    assert promoted.json()["updated"]["fairness_passed"] is True

    clustered = client.post("/api/workbench/s07/fairness", json={"sensitive_col": "group", "cluster_col": "unit"})
    assert clustered.status_code == 200, clustered.text
    assert clustered.json()["fairness"]["n_clusters"] == 2
    assert clustered.json()["promotion_evidence"]["holdout_fingerprint"] != evidence["holdout_fingerprint"]
    state["uap_df"].loc[test.index[0], "unit"] = 5
    stale = client.post("/api/workbench/s11/register", json={"model_name": "clustered"})
    assert stale.status_code == 409, stale.text
    state["uap_df"].loc[test.index[0], "unit"] = 3
    clustered_version = client.post("/api/workbench/s11/register", json={"model_name": "clustered"})
    assert clustered_version.status_code == 200, clustered_version.text
    clustered_evidence = clustered_version.json()["registered"]["fairness_evidence"]
    assert clustered_evidence["cluster_attribute"] == "unit"
    assert clustered_evidence["n_clusters"] == 2

    state["uap_active_model"] = LinearRegression().fit(train[["x"]], train["target"])
    diagnostic = client.post("/api/workbench/s07/fairness", json={"sensitive_col": "group"})
    assert diagnostic.status_code == 200, diagnostic.text
    assert diagnostic.json()["outcome_definition"]["method"] == "at_or_above_holdout_median"
    assert diagnostic.json()["fairness"]["assessment_status"] == "insufficient_evidence"
    assert not diagnostic.json()["fairness"]["passed"]


def test_workspace_root_rebases_default_paths_but_keeps_overrides(tmp_path: Path) -> None:
    from omniforge.config.settings import PathsSettings

    workspace = tmp_path / "mounted"
    explicit_registry = tmp_path / "separate" / "registry.db"
    paths = PathsSettings(workspace_dir=workspace, registry_db_path=explicit_registry)
    assert paths.registry_db_path == explicit_registry
    assert paths.artifact_store_dir == workspace / "artifacts"
    assert paths.tracking_db_path == workspace / "mlops_tracking.db"
    assert paths.audit_log_path == workspace / "audit.jsonl"
