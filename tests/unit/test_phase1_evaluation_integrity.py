"""Regression coverage for model, fitted inputs, cohorts and operational limits."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression, Ridge

from omniforge.api import workbench as wb
from omniforge.api.auth import get_api_key, jwt_manager
from omniforge.api.main import create_app
from omniforge.core.exceptions import EvaluationError
from omniforge.core.types import Principal
from omniforge.features.inference import WorkbenchInputPreprocessor
from omniforge.mlops.supervisory_monitor import SupervisoryPredictionMonitor
from omniforge.reporting.live_charts import _resolve_model_eval
from omniforge.ui.state import load_new_dataset, record_active_holdout


def test_chart_applies_non_champion_encoding_and_requires_its_outcome() -> None:
    data = pd.DataFrame({"category": ["a", "b"] * 24, "outcome": [0, 1] * 24, "sibling": [2, 3] * 24})
    raw = data[["category"]]
    prep = WorkbenchInputPreprocessor(mode="numeric").fit(raw.iloc[:30])
    encoded = prep.transform(raw)
    model = LogisticRegression().fit(encoded.iloc[:30], data["outcome"].iloc[:30])
    state = {"uap_baseline_models": {"outcome": {"logit": {
        "model": model, "feature_cols": list(encoded), "input_preprocessor": prep,
    }}}, "uap_targets": ["outcome", "sibling"]}
    result = _resolve_model_eval(state, data)
    assert result is not None
    assert "holdout status unverified" in result.eval_scope
    assert np.array_equal(result.y_pred, model.predict(encoded.iloc[36:]))
    assert np.allclose(result.y_proba, model.predict_proba(encoded.iloc[36:]))
    errors: list[str] = []
    assert _resolve_model_eval(state, data.drop(columns="outcome"), unavailable=errors) is None
    assert "outcome is absent" in errors[0]
    assert _resolve_model_eval(state, data.drop(columns="category")) is None


def _regression_state() -> tuple[dict, pd.DataFrame]:
    X = pd.DataFrame({"input": np.arange(30, dtype=float)})
    target = pd.Series(3 * X["input"] + np.sin(X["input"]), name="outcome")
    model = Ridge().fit(X.iloc[:20], target.iloc[:20])
    data = X.assign(outcome=target)
    state = {
        "uap_df": data, "uap_active_model": model, "uap_active_model_target": "outcome",
        "uap_active_feature_cols": ["input"], "uap_active_X_test": X.iloc[20:].copy(),
        "uap_active_y_test": target.iloc[20:].copy(), "uap_targets": ["outcome", "sibling"],
        "uap_baseline_models": {"sibling": {"ridge": {"model": model, "feature_cols": ["input"]}}},
    }
    record_active_holdout(state)
    return state, data


def test_saved_holdout_requires_aligned_ids_and_same_model() -> None:
    state, data = _regression_state()
    assert "Held-Out" in _resolve_model_eval(state, data).eval_scope
    state["uap_active_y_test"] = state["uap_active_y_test"].iloc[::-1]
    assert _resolve_model_eval(state, data) is None
    state["uap_active_y_test"] = state["uap_active_y_test"].iloc[::-1]
    state["uap_active_model"] = Ridge().fit(data[["input"]], data["outcome"])
    assert _resolve_model_eval(state, data) is None


def test_spc_uses_frozen_residual_reference_and_reports_missing_observations() -> None:
    from omniforge.reporting.live_charts import generate_live_chart

    state, data = _regression_state()
    monitor = SupervisoryPredictionMonitor()
    batch = data.iloc[:6].copy()
    first = monitor.evaluate_incoming(state, batch, model_id="active_champion")
    cusum = generate_live_chart(state, chart_id="cusum_sequential_control", studio_id="14")
    assert cusum["kpis"]["Baseline SD σ₀"] == round(first["spc_limits"]["sigma"], 4)
    batch.loc[batch.index[1:], "outcome"] += [10, -20, 30, -40, 50]
    batch.loc[batch.index[0], "outcome"] = np.nan
    shifted = monitor.evaluate_incoming(state, batch, model_id="active_champion")
    assert first["spc_limits"] == shifted["spc_limits"]
    assert shifted["spc_limits"]["sigma"] > 0
    assert shifted["spc_points"][1]["status"] == "BREACH"
    assert shifted["telemetry_rows"][0]["residual"] is None
    assert shifted["telemetry_rows"][0]["status"] == "UNAVAILABLE"
    with pytest.raises(ValueError, match="complete sequential residual series"):
        generate_live_chart(state, chart_id="cusum_sequential_control", studio_id="14")
    assert shifted["kpis"]["evaluated_rows"] == 5
    assert shifted["kpis"]["comparison_rows"] == 0
    assert shifted["kpis"]["baseline_label"] is None
    assert "model_vs_baseline_lift_pct" not in shifted["kpis"]
    with pytest.raises(EvaluationError, match="different outcome"):
        monitor.evaluate_incoming(state, batch, model_id="active_champion", baseline_model_id="baseline::sibling::ridge")
    with pytest.raises(EvaluationError, match="requested model"):
        monitor.evaluate_incoming(state, batch, model_id="unknown")
    pending = monitor.evaluate_incoming(state, batch[["input"]], model_id="active_champion")
    assert pending["system_status"] == "UNAVAILABLE"
    assert pending["kpis"]["compliance_rate_pct"] is None


def test_invalid_predictions_cannot_pass_compliance() -> None:
    class InvalidModel:
        _estimator_type = "regressor"

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), np.nan)

    state, data = _regression_state()
    state["uap_active_model"] = InvalidModel()
    result = SupervisoryPredictionMonitor().evaluate_incoming(state, data, model_id="active_champion")
    assert result["system_status"] == "UNAVAILABLE"
    assert result["spc_limits"] is None
    assert result["telemetry_rows"][0]["model_pred"] is None
    assert result["spc_points"][0]["residual"] is None
    assert not result["alarms"]


def test_quick_train_protects_sibling_outcomes_and_uses_stratification() -> None:
    sid = uuid4().hex
    principal = Principal(subject="api_key_principal", roles=["admin"], tenant_id="default-tenant")
    state = wb.get_session_state(wb._sid(sid, principal))
    frame = pd.DataFrame({"input": range(60), "outcome": ["a"] * 48 + ["b"] * 12, "sibling": range(60)})
    load_new_dataset(state, frame, "sorted", "test fixture", default_targets=["outcome", "sibling"])
    with TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": sid}) as client:
        assert client.post("/api/workbench/s14/quick-train", json={}).status_code == 400
        response = client.post("/api/workbench/s14/quick-train", json={"target": "outcome"})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["evaluation"]["source"] == "stratified_split"
        assert result["models"][0]["feature_cols"] == ["input"]
        assert client.post("/api/workbench/s11/root-cause", json={}).status_code == 400
        simulation = client.post("/api/workbench/s14/evaluate-demo-batch", json={"n_rows": 17})
        assert simulation.status_code == 200, simulation.text
        for endpoint in ("s11", "s14"):
            diagnosis = client.post(f"/api/workbench/{endpoint}/root-cause", json={})
            assert diagnosis.status_code == 200, diagnosis.text
            assert diagnosis.json()["cohorts"]["n_current"] == 17
    assert set(next(iter(state["uap_baseline_models"]["outcome"].values()))["evaluation_holdout"][1]) == {"a", "b"}


def test_export_and_operational_mutation_permissions() -> None:
    viewer = Principal(subject="readonly", roles=["viewer"], tenant_id=uuid4().hex)
    headers = {"Authorization": f"Bearer {jwt_manager.create_token(viewer)}"}
    with TestClient(create_app(), headers=headers) as client:
        assert client.post("/api/workbench/workspace/save", data={}).status_code == 403
        assert client.post("/api/workbench/charts/export", json={"chart_id": "any"}).status_code == 403
        assert client.post("/api/workbench/s14/evaluate-demo-batch", json={}).status_code == 403
        assert client.post("/api/workbench/s14/evaluate-single", json={"record": {}}).status_code == 403
        assert client.post("/api/workbench/s09/twin-control", json={}).status_code == 403


def test_readonly_connector_pins_postgres_and_sets_mysql_session_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any

    from sqlalchemy import event

    from omniforge.data import connectors

    class Engine:
        pass

    created: list[tuple[Any, dict[str, Any]]] = []
    callbacks: list[Any] = []
    executed: list[str] = []

    def create_engine(url: Any, **options: Any) -> Engine:
        created.append((url, options))
        return Engine()

    def listen(engine: Engine, event_name: str, *, insert: bool) -> Any:
        assert isinstance(engine, Engine)
        assert event_name == "connect" and insert
        return callbacks.append

    class Cursor:
        def execute(self, sql: str) -> None:
            executed.append(sql)

        def close(self) -> None:
            executed.append("closed")

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

    monkeypatch.setattr("sqlalchemy.create_engine", create_engine)
    monkeypatch.setattr(event, "listens_for", listen)
    monkeypatch.setattr(connectors, "_validate_remote_db_host", lambda host, port: ("2001:db8::5", "192.0.2.5"))
    connectors._create_readonly_sql_engine("postgresql+psycopg://localhost/study?sslmode=verify-full")
    pg_url, pg_options = created.pop()
    assert pg_url.host == "localhost,localhost"
    assert pg_url.query["hostaddr"] == "2001:db8::5,192.0.2.5"
    assert pg_url.query["sslmode"] == "verify-full"
    assert pg_options["execution_options"]["postgresql_readonly"]
    connectors._create_readonly_sql_engine("mysql+pymysql://localhost/study")
    assert len(callbacks) == 1
    callbacks[0](Connection(), None)
    assert executed == ["SET SESSION TRANSACTION READ ONLY", "closed"]


def test_legacy_registry_retries_after_empty_source(tmp_path: Path) -> None:
    import sqlite3

    from omniforge.core.exceptions import MLOpsError
    from omniforge.mlops.registry import ModelRegistry

    previous = tmp_path / "legacy.db"
    previous.touch()
    store = ModelRegistry(db_path=tmp_path / "canonical.db", artifact_dir=tmp_path / "artifacts")
    store.import_legacy_registry(previous)
    with sqlite3.connect(previous) as conn:
        conn.execute("CREATE TABLE unexpected (value TEXT)")
    with pytest.raises(MLOpsError, match="missing its model version table"):
        store.import_legacy_registry(previous)


def test_chart_library_errors_do_not_expose_exception_values(monkeypatch: pytest.MonkeyPatch) -> None:
    from omniforge.api.studios import charts_hub

    sentinel = "synthetic-private-exception-value"

    def failing_chart(*args: object, **kwargs: object) -> None:
        raise ValueError(sentinel)

    monkeypatch.setattr(charts_hub, "generate_live_chart", failing_chart)
    with TestClient(create_app(), headers={"X-API-Key": get_api_key()}) as client:
        response = client.post("/api/workbench/charts/generate", json={"chart_id": "dataset_readiness_gauge"})
    assert response.status_code == 400
    assert sentinel not in response.text
    assert response.json()["detail"] == "The chart could not be produced from the supplied data."


def test_score_gauge_reports_observed_metric_without_an_invented_baseline() -> None:
    from omniforge.reporting.live_charts import generate_live_chart

    state, _ = _regression_state()
    state["uap_active_model_metrics"] = {"baseline_score": 0.99}
    score = generate_live_chart(state, chart_id="model_champion_gauge", studio_id="14")
    expected = state["uap_active_model"].score(state["uap_active_X_test"], state["uap_active_y_test"])
    assert score["data"]["value"] == round(expected * 100, 2)
    assert score["data"]["thresholds"] == []
    assert "Baseline Reference (%)" not in score["kpis"]

    state["uap_active_model_target"] = "missing_outcome"
    state["uap_supervisory_result"] = {"kpis": {"model_label": "some other model", "mape_pct": 2}}
    with pytest.raises(ValueError, match="outcome is absent"):
        generate_live_chart(state, chart_id="model_champion_gauge", studio_id="14")


def test_score_only_report_labels_compliance_unavailable() -> None:
    from omniforge.api.studios.s12_reports import _compile_report_payload
    from omniforge.api.studios.schemas import ReportCompileReq

    state, data = _regression_state()
    SupervisoryPredictionMonitor().evaluate_incoming(state, data[["input"]], model_id="active_champion")
    report = _compile_report_payload(state, ReportCompileReq(selected_sections=["future_predictions"], include_charts=False))
    section = next(section for section in report["sections"] if section["section_id"] == "future_predictions")
    assert "compliance unavailable" in section["body"]
    assert "100.00% compliance" not in section["body"]
    assert "Unevaluable records: 30" in section["body"]


def test_integer_valued_regression_remains_regression_after_workspace_restore() -> None:
    from omniforge.reporting.live_charts import generate_live_chart
    from omniforge.ui.state import export_workspace_project_bytes, restore_workspace_project_bytes

    X = pd.DataFrame({"input": np.arange(20, dtype=float)})
    y = pd.Series(np.square(X["input"]), name="outcome")
    study = X.assign(outcome=y)
    state: dict = {}
    load_new_dataset(state, study, "counts", "test fixture", default_target="outcome")
    state.update({
        "uap_active_model": Ridge().fit(X.iloc[:14], y.iloc[:14]),
        "uap_active_model_target": "outcome", "uap_active_feature_cols": ["input"],
        "uap_active_X_test": X.iloc[14:], "uap_active_y_test": y.iloc[14:],
    })
    record_active_holdout(state)
    original = generate_live_chart(state, chart_id="model_champion_gauge")
    assert "R²" in original["subtitle"]
    archive, _ = export_workspace_project_bytes(state)
    restored: dict = {}
    restore_workspace_project_bytes(restored, archive)
    assert restored["uap_active_holdout_model"] is restored["uap_active_model"]
    after = generate_live_chart(restored, chart_id="model_champion_gauge")
    assert "Held-Out Evaluation Set (n=6)" in after["subtitle"]
    assert after["data"] == original["data"]
    monitored = SupervisoryPredictionMonitor().evaluate_incoming(restored, study.iloc[14:], model_id="active_champion")
    assert monitored["kpis"]["task_mode"] == "regression"
