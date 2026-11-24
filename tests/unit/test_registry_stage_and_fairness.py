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
