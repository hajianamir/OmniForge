"""Regressions for model identity, scientific provenance and Workbench boundaries."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from omniforge.api.auth import get_api_key, jwt_manager
from omniforge.api.main import create_app
from omniforge.core.exceptions import AuthorizationError, DataError, MLOpsError
from omniforge.core.security import AccessController
from omniforge.core.types import Principal, TaskType
from omniforge.data.cleaning import CleaningConfig, DataCleaner
from omniforge.data.connectors import validate_readonly_sql
from omniforge.data.versioning import DatasetVersionManager
from omniforge.evaluation.equations import _build_external_deployment_formulas
from omniforge.evaluation.interpretability import InterpretabilityEngine
from omniforge.mlops.monitoring import ModelMonitor
from omniforge.mlops.registry import ModelRegistry
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.classical import ClassicalMLModel
from omniforge.pipelines.base import FunctionStep, PipelineContext
from omniforge.pipelines.dag import PipelineDAG
from omniforge.pipelines.executor import PipelineExecutor
from omniforge.reporting.live_charts import _build_classifier_calibration, _resolve_model_eval


def test_classical_updates_rebuild_the_actual_estimator(tmp_path: Path) -> None:
    X = pd.DataFrame({"x": np.arange(24, dtype=float)})
    y = X["x"] ** 2
    model = ClassicalMLModel("ridge", task_type=TaskType.REGRESSION, track_carbon=False).fit(X, y)
    assert model.set_params(algorithm="random_forest", n_estimators=5) is model
    assert not model.is_fitted_
    assert not model.scale_features
    model.fit(X, y)
    assert isinstance(model._estimator, RandomForestRegressor)
    assert model._estimator.n_estimators == 5
    assert clone(model).get_params() == model.get_params()
    restored = BaseOmniForgeModel.load(model.save(tmp_path / "model.joblib"))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    assert restored.metadata_.hyperparameters["algorithm"] == "random_forest"


def test_cache_identity_includes_the_callable(tmp_path: Path) -> None:
    executor = PipelineExecutor(cache_dir=tmp_path, track_carbon=False)

    def produce_one(_context: PipelineContext) -> int:
        return 1

    def produce_two(_context: PipelineContext) -> int:
        return 2

    first_dag = PipelineDAG("identity").add_step(
        FunctionStep("output", produce_one, cacheable=True, implementation_version="1")
    )
    changed_dag = PipelineDAG("identity").add_step(
        FunctionStep("output", produce_two, cacheable=True, implementation_version="1")
    )
    assert executor.execute(first_dag).context.step_outputs["output"] == 1
    assert executor.execute(first_dag).step_records[0].cache_hit
    changed = executor.execute(changed_dag)
    assert changed.context.step_outputs["output"] == 2
    assert not changed.step_records[0].cache_hit


def test_fitted_outlier_flags_use_same_schema_on_holdout() -> None:
    train = pd.DataFrame({"value": [1., 2., 3., 4., 5., 6.]})
    holdout = pd.DataFrame({"value": [7., 1000.]})
    X_train, X_test, _, report = DataCleaner(CleaningConfig(outlier_action="flag")).fit_transform_split(train, holdout)
    assert list(X_train.columns) == list(X_test.columns)
    assert X_train["value__is_outlier"].tolist() == [0] * 6
    assert X_test["value__is_outlier"].tolist() == [0, 1]
    assert X_test["value"].tolist() == [7.0, 1000.0]
    assert report.outliers_detected == {"value": 1}


@pytest.mark.parametrize("row", [[np.nan, np.nan], [np.inf, 0.], [.1, .5], [0., .5, .5], [-.1, 1.1]])
def test_invalid_probabilities_cannot_be_used_for_calibration(row: list[float]) -> None:
    class ProbabilityModel:
        classes_ = np.array([0, 1])

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.tile(row, (len(X), 1))

    X = pd.DataFrame({"x": [1., 2., 3., 4.]})
    target = pd.Series([0, 1, 0, 1], name="target")
    state = {
        "uap_active_model": ProbabilityModel(),
        "uap_active_X_test": X,
        "uap_active_y_test": target,
        "uap_targets": ["target"],
    }
    result = _resolve_model_eval(state, X.assign(target=target))
    assert result is not None
    assert result.y_proba is None
    with pytest.raises(ValueError, match="valid class probabilities"):
        _build_classifier_calibration(result)


def test_small_calibration_does_not_report_guaranteed_interval() -> None:
    X = pd.DataFrame({"x": [1., 2., 3., 4.]})
    model = ClassicalMLModel("ridge", task_type=TaskType.REGRESSION, track_carbon=False).fit(X, np.zeros(4))
    result = InterpretabilityEngine().conformal_explain(model, X.iloc[:1], np.array([.1]), X.iloc[1:2], alpha=.01)
    assert result["nominal_coverage"] == .99
    assert result["coverage_guarantee"] is None
    assert result["interval"] is None
    assert result["finite_sample_valid"] is False


def test_exported_formula_parses_with_ordinary_quoted_names() -> None:
    formula = _build_external_deployment_formulas(
        model_name='Model "B"',
        target_name='User\'s "Total"\nAmount (EUR)',
        task_type="regression",
        equation_type="exact_closed_form",
        intercept=1.,
        coefficients={"Cost (€)": 2.},
    )
    compile(formula["formula_python"], "<test export>", "exec")


def test_dataset_snapshot_checks_persisted_bytes(tmp_path: Path) -> None:
    manager = DatasetVersionManager(store_dir=tmp_path)
    frame = pd.DataFrame({"value": [1., 2.]})
    version = manager.commit(frame, "snapshot")
    stored, _ = manager.checkout("snapshot")
    pd.testing.assert_frame_equal(frame, stored)
    with Path(version.storage_path).open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(DataError, match="checksum"):
        manager.checkout("snapshot")


def test_registry_import_preserves_models_and_refuses_collisions(tmp_path: Path) -> None:
    legacy = ModelRegistry(tmp_path / "old.db", tmp_path / "old")
    current = ModelRegistry(tmp_path / "new.db", tmp_path / "new")
    X = pd.DataFrame({"x": [1., 2., 3., 4.]})
    model = ClassicalMLModel("ridge", task_type=TaskType.REGRESSION, track_carbon=False).fit(X, X["x"])
    old = legacy.register_model(model, "old_model")
    current.register_model(model, "new_model")
    current.import_legacy_registry(legacy.db_path)
    current.import_legacy_registry(legacy.db_path)
    assert [row.model_name for row in current.list_versions()] == ["new_model", "old_model"]
    loaded, imported = current.load_model("old_model")
    assert imported.checksum_sha256 == old.checksum_sha256
    assert loaded.metadata_.artifact_checksum == old.checksum_sha256
    conflict = ModelRegistry(tmp_path / "conflict.db", tmp_path / "conflict")
    conflict.register_model(model, "old_model")
    with pytest.raises(MLOpsError, match="version collision"):
        current.import_legacy_registry(conflict.db_path)


def test_integer_regression_does_not_trigger_accuracy_rollback() -> None:
    monitor = ModelMonitor("count_regression", task_type=TaskType.REGRESSION)
    for actual in range(10):
        monitor.record_inference(latency_ms=1., actual=actual, prediction=actual + 1)
    health = monitor.evaluate_health(auto_rollback=True)
    assert health["live_mae"] == 1.
    assert health["live_accuracy"] is None
    assert health["active_alerts"] == []
    assert not health["rollback_executed"]


def test_cross_tenant_audit_records_final_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    principal = Principal(subject="tester", roles=["platform_admin"], tenant_id="alpha")
    events: list[dict[str, Any]] = []
    controller = AccessController()
    monkeypatch.setattr(controller, "_audit_cross_tenant", lambda *args, **kwargs: events.append(kwargs))
    with pytest.raises(AuthorizationError):
        controller.authorize(principal, "data:read", resource_tenant_id="beta", required_attributes={"approval": True})
    assert controller.authorize(principal, "data:read", resource_tenant_id="beta")
    assert [entry["status"] for entry in events] == ["denied", "success"]


def test_api_uses_tabular_upload_policy_and_restricts_report_export() -> None:
    client = TestClient(create_app(), headers={"X-API-Key": get_api_key()})
    with BytesIO() as workbook:
        pd.DataFrame({"a": [1, 2]}).to_excel(workbook, index=False)
        accepted = client.post("/api/v1/data/upload", files={"file": ("dataset.xlsx", workbook.getvalue())})
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["columns"] == ["a"]
    for route in ("/api/v1/data/upload", "/api/workbench/s01/upload", "/api/workbench/s14/evaluate-file"):
        rejected = client.post(route, files={"file": ("dataset.hdf5", b"invalid test data")})
        assert rejected.status_code == 400, rejected.text
    viewer = Principal(subject=f"viewer-{uuid4()}", roles=["viewer"], tenant_id="phase1-test")
    with TestClient(create_app(), headers={"Authorization": f"Bearer {jwt_manager.create_token(viewer)}"}) as restricted:
        unavailable = restricted.get("/api/workbench/s12/export/html")
    assert unavailable.status_code == 403


def test_read_only_sql_accepts_ordinary_comparison_values() -> None:
    query = "SELECT * FROM contacts WHERE email = 'reader@example.com' AND url <> 'https://example.org/path'"
    assert validate_readonly_sql(query) == query


def test_studio09_reads_selected_text_and_interaction_columns() -> None:
    headers = {"X-API-Key": get_api_key(), "X-Session-ID": f"specialized-{uuid4()}"}
    client = TestClient(create_app(), headers=headers)
    response = client.post(
        "/api/workbench/s01/upload",
        files={"file": ("interactions.csv", b"user,item,review\nalex,a,green orchard\njamie,b,citrus grove\njamie,c,spring blossoms\n")},
    )
    assert response.status_code == 200, response.text
    nlp = client.post("/api/workbench/s09/nlp", json={"task": "semantic_search", "text_col": "review", "query": "orchard"})
    assert nlp.status_code == 200, nlp.text
    assert nlp.json()["text_source"] == "active_dataset"
    assert nlp.json()["text_column"] == "review"
    assert nlp.json()["texts_analyzed"] == 3
    recs = client.post("/api/workbench/s09/recommend", json={"user_col": "user", "item_col": "item", "user_id": "alex"})
    assert recs.status_code == 200, recs.text
    assert recs.json()["strategy"] == "fitted_item_recommender"
    assert {row["item"] for row in recs.json()["recommendations"]} == {"b", "c"}
