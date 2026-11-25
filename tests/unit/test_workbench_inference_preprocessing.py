"""Raw Studio 06 inputs reuse the preparation fitted during model training."""

from __future__ import annotations

import io
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from omniforge.api import workbench as wb
from omniforge.api.auth import get_api_key
from omniforge.api.jobs import BackgroundJobManager
from omniforge.api.main import create_app
from omniforge.automl.backends import _build_preprocessor
from omniforge.config.settings import get_settings
from omniforge.core.types import Principal
from omniforge.core.workspace_session import WorkspaceSessionManager
from omniforge.features.inference import WorkbenchInputPreprocessor
from omniforge.ui.state import get_session_state, load_new_dataset


def _training_frame() -> pd.DataFrame:
    n = 60
    return pd.DataFrame(
        {
            "amount": np.full(n, 4.0),
            "segment": [["blue", "red", "NA", "001"][i % 4] for i in range(n)],
            "when": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "target": [int(i % 4 in (1, 3)) for i in range(n)],
        }
    )


def test_studio06_single_and_batch_use_fitted_training_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Known/unseen categories, timestamps and missing fields agree with direct model inference."""
    frame = _training_frame()
    session = f"raw-{uuid.uuid4().hex}"
    principal = Principal(subject="api_key_principal", roles=["admin"], tenant_id=get_settings().default_tenant_id)
    state = get_session_state(wb._sid(session, principal))
    load_new_dataset(state, frame, "test", "test", default_target="target")

    def train(df: pd.DataFrame, target_col: str, **kwargs: object) -> SimpleNamespace:
        X = df.drop(columns=[target_col])
        pipe = Pipeline([("pre", _build_preprocessor(X)), ("model", LogisticRegression())])
        pipe.fit(X, df[target_col])
        return SimpleNamespace(
            best_model=pipe, best_model_name="logistic", executed_backend="test", requested_backend="test",
            best_score=1.0, lift_over_best_baseline=0.2, primary_metric="accuracy", task_type=kwargs["task_type"],
            leaderboard=[], elapsed_seconds=0.1, fallback_reason=None, extra={},
        )

    monkeypatch.setattr("omniforge.api.studios.s06_automl.AutoMLBackendHub.run", train)
    client = TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": session})
    trained = client.post("/api/workbench/s06/automl", json={"target": "target", "backend": "test"})
    assert trained.status_code == 200, trained.text
    summary = trained.json()["state"]
    assert summary["active_input_feature_types"]["when"] == "datetime"
    assert summary["active_input_feature_types"]["segment"] == "categorical"

    records = [
        {"amount": "4", "segment": "red", "when": "2024-01-20T00:00:00Z"},
        {"amount": 4, "segment": "blue", "when": "2024-01-20T01:00:00+01:00"},
        {"amount": 4, "segment": "NA", "when": 1705708800},
        {"amount": 4, "segment": "001", "when": 1705708800},
        {"amount": None, "segment": "unseen", "when": None},
    ]
    preprocessor = state["uap_active_input_preprocessor"]
    assert isinstance(preprocessor, WorkbenchInputPreprocessor)
    expected_input = preprocessor.transform(pd.DataFrame(records))
    expected = state["uap_active_model"].predict(expected_input)
    assert expected[0] != expected[1]  # categories must not silently become zero

    single_predictions = []
    for record in records:
        response = client.post("/api/workbench/s06/predict-single", json={"record": record})
        assert response.status_code == 200, response.text
        single_predictions.append(response.json()["prediction"])
    assert single_predictions == expected.tolist()

    raw_csv = pd.DataFrame(records).to_csv(index=False).encode()
    batch = client.post("/api/workbench/s06/predict-batch", files={"file": ("records.csv", io.BytesIO(raw_csv), "text/csv")})
    assert batch.status_code == 200, batch.text
    assert [row["predicted_target"] for row in batch.json()["preview"]] == expected.tolist()
    invalid = client.post("/api/workbench/s06/predict-single", json={"record": {"amount": "not-a-number"}})
    assert invalid.status_code == 422
    invalid_date = client.post("/api/workbench/s06/predict-single", json={"record": {"when": "not-a-date"}})
    assert invalid_date.status_code == 422

    manager = BackgroundJobManager(db_path=tmp_path / "jobs.db")
    observer = BackgroundJobManager(db_path=tmp_path / "jobs.db")
    monkeypatch.setattr(wb, "job_manager", manager)
    try:
        submitted = client.post("/api/workbench/s06/automl", json={"target": "target", "backend": "test", "async_job": True})
        assert submitted.status_code == 200, submitted.text
        job_id = submitted.json()["job"]["job_id"]
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            polled = client.get(f"/api/workbench/jobs/{job_id}")
            assert polled.status_code == 200
            if polled.json()["status"] in ("completed", "failed"):
                break
            time.sleep(0.02)
        assert polled.json()["status"] == "completed", polled.text
        assert polled.json()["result"]["best_model_name"] == "logistic"
        shared = observer.get(job_id, owner_key=wb.current_job_owner_key(principal))
        assert shared is not None and shared.status == "completed"
    finally:
        manager.shutdown()
        observer.shutdown()


def test_running_automl_cancellation_does_not_publish_a_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frame = _training_frame()
    session = f"cancel-raw-{uuid.uuid4().hex}"
    principal = Principal(subject="api_key_principal", roles=["admin"], tenant_id=get_settings().default_tenant_id)
    state = get_session_state(wb._sid(session, principal))
    load_new_dataset(state, frame, "test", "test", default_target="target")
    started, release = threading.Event(), threading.Event()

    def long_training(*args: object, **kwargs: object) -> None:
        started.set()
        assert release.wait(timeout=8)

    monkeypatch.setattr("omniforge.api.studios.s06_automl.AutoMLBackendHub.run", long_training)
    manager = BackgroundJobManager(max_workers=1, db_path=tmp_path / "jobs.db")
    monkeypatch.setattr(wb, "job_manager", manager)
    client = TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": session})
    try:
        submitted = client.post("/api/workbench/s06/automl", json={"target": "target", "async_job": True})
        assert submitted.status_code == 200, submitted.text
        job_id = submitted.json()["job"]["job_id"]
        assert started.wait(timeout=3)
        cancelled = client.post(f"/api/workbench/jobs/{job_id}/cancel")
        assert cancelled.status_code == 202
        assert cancelled.json()["status"] == "cancelling"
        release.set()
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            polled = client.get(f"/api/workbench/jobs/{job_id}")
            if polled.json()["status"] in ("cancelled", "failed", "completed"):
                break
            time.sleep(0.02)
        assert polled.json()["status"] == "cancelled", polled.text
        assert state["uap_active_model"] is None
    finally:
        release.set()
        manager.shutdown()


def test_numeric_models_reuse_fitted_encoding_and_training_medians() -> None:
    frame = _training_frame().drop(columns="target")
    frame.loc[0, "amount"] = np.nan
    preprocessor = WorkbenchInputPreprocessor(mode="numeric").fit(frame)
    train = preprocessor.transform(frame)
    inputs = pd.DataFrame([
        {"segment": "red", "when": "2024-01-02T00:00:00Z"},
        {"segment": "unknown", "when": None},
    ])
    transformed = preprocessor.transform(inputs)
    assert list(transformed.columns) == list(train.columns)
    assert transformed.loc[0, "amount"] == 4.0
    assert transformed.loc[0, "segment_red"] == 1.0
    assert transformed.loc[1, ["segment_red", "segment_blue"]].sum() == 0.0
    assert transformed.loc[0, "when"] == train.loc[1, "when"]

    archive, manifest = WorkspaceSessionManager.export_to_bytes({"uap_active_input_preprocessor": preprocessor})
    assert "uap_active_input_preprocessor" in manifest.saved_keys
    restored, _ = WorkspaceSessionManager.load_from_bytes(archive)
    pd.testing.assert_frame_equal(restored["uap_active_input_preprocessor"].transform(inputs), transformed)


def test_real_automl_scores_raw_categorical_and_datetime_records() -> None:
    n = 80
    frame = pd.DataFrame({
        "amount": np.arange(n, dtype=float),
        "segment": ["red" if i % 2 else "blue" for i in range(n)],
        "when": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
        "target": [i % 2 for i in range(n)],
    })
    session = f"real-raw-{uuid.uuid4().hex}"
    principal = Principal(subject="api_key_principal", roles=["admin"], tenant_id=get_settings().default_tenant_id)
    state = get_session_state(wb._sid(session, principal))
    load_new_dataset(state, frame, "test", "test", default_target="target")
    client = TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": session})
    trained = client.post(
        "/api/workbench/s06/automl", json={"target": "target", "backend": "multi_layer_stacker", "time_budget_seconds": 5, "cv_folds": 2}
    )
    assert trained.status_code == 200, trained.text
    rows = [
        {"amount": 5, "segment": "red", "when": "2024-01-05T00:00:00Z"},
        {"amount": 5, "segment": "blue", "when": "2024-01-05T00:00:00Z"},
    ]
    expected = state["uap_active_model"].predict(state["uap_active_input_preprocessor"].transform(pd.DataFrame(rows))).tolist()
    assert expected[0] != expected[1]
    singles = [client.post("/api/workbench/s06/predict-single", json={"record": row}) for row in rows]
    assert all(response.status_code == 200 for response in singles)
    assert [response.json()["prediction"] for response in singles] == expected
    batch = client.post(
        "/api/workbench/s06/predict-batch",
        files={"file": ("records.csv", io.BytesIO(pd.DataFrame(rows).to_csv(index=False).encode()), "text/csv")},
    )
    assert batch.status_code == 200, batch.text
    assert [row["predicted_target"] for row in batch.json()["preview"]] == expected


def test_native_automl_respects_chronological_split_lock_and_excluded_date() -> None:
    n = 80
    frame = pd.DataFrame({
        "amount": np.arange(n, dtype=float),
        "when": pd.date_range("2024-01-01", periods=n, freq="D"),
        "target": [i % 2 for i in range(n)],
    })
    session = f"locked-{uuid.uuid4().hex}"
    principal = Principal(subject="api_key_principal", roles=["admin"], tenant_id=get_settings().default_tenant_id)
    state = get_session_state(wb._sid(session, principal))
    load_new_dataset(state, frame, "test", "test", default_target="target")
    client = TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": session})
    excluded = client.post("/api/workbench/exclusions", json={"excluded_columns": ["when"]})
    assert excluded.status_code == 200, excluded.text
    locked = client.post(
        "/api/workbench/split-lock",
        json={"strategy": "time_series", "time_col": "when", "gap": 2, "test_size": 0.2, "cv_folds": 2},
    )
    assert locked.status_code == 200, locked.text
    result = client.post(
        "/api/workbench/s06/automl",
        json={"target": "target", "backend": "omniforge_native", "time_budget_seconds": 10, "cv_folds": 2},
    )
    assert result.status_code == 200, result.text
    payload = result.json()
    evaluation = payload["metadata"]["evaluation"]
    assert evaluation["best_score_partition"] == "test"
    assert evaluation["test_partition_source"] == "supplied_holdout"
    assert evaluation["leaderboard_partition"] == "validation"
    train_rows, lock_test, _ = wb.get_active_split_frames(state, frame)
    assert lock_test is not None
    held_out = state["uap_active_X_test"]
    pd.testing.assert_index_equal(held_out.index, lock_test.index)
    pd.testing.assert_series_equal(state["uap_active_y_test"], lock_test["target"])
    assert evaluation["n_test"] == len(lock_test)
    assert set(held_out.index).isdisjoint(train_rows.index)
    assert list(held_out.columns) == ["amount"]
    assert payload["state"]["excluded_columns"] == ["when"]


def test_studio10_automl_reserves_the_locked_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from omniforge.automl.backends import AutoMLBackendHub

    frame = pd.DataFrame({"row": np.arange(60), "target": np.tile([0, 1], 30)})
    session = f"dag-lock-{uuid.uuid4().hex}"
    principal = Principal(subject="api_key_principal", roles=["admin"], tenant_id=get_settings().default_tenant_id)
    state = get_session_state(wb._sid(session, principal))
    load_new_dataset(state, frame, "test", "test", default_target="target")
    client = TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": session})
    locked = client.post("/api/workbench/split-lock", json={"strategy": "stratified", "test_size": 0.25, "cv_folds": 2})
    assert locked.status_code == 200, locked.text
    train, test, _ = wb.get_active_split_frames(state, frame)
    assert test is not None
    seen: list[int] = []

    def record_input(data: pd.DataFrame, **options: object) -> SimpleNamespace:
        seen.extend(data["row"].tolist())
        return SimpleNamespace(best_model_name="fixture", best_score=0.8, lift_over_best_baseline=0.1)

    monkeypatch.setattr(AutoMLBackendHub, "run", record_input)
    result = client.post("/api/workbench/s10/run", json={"target": "target", "stages": ["automl"]})
    assert result.status_code == 200, result.text
    assert seen == train["row"].tolist()
    assert set(seen).isdisjoint(test["row"])
    step = result.json()["execution"]["step_outputs"]["step_1_automl"]
    assert step["score_partition"] == "validation"
    assert step["reserved_test_rows"] == len(test)
