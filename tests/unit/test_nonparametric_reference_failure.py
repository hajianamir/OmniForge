"""V-01: replay the auditor's forced-reference-failure scenario without false acceptance."""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import omniforge.models.nonparametric as nonparametric
from omniforge.api.auth import get_api_key
from omniforge.api.main import create_app
from omniforge.core.types import TaskType
from omniforge.evaluation.equations import ModelEquationExtractor


class FailingReference:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        _ = X, y
        raise RuntimeError("forced reference failure")


class ExactQuadratic:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> ExactQuadratic:
        _ = X, y
        return self

    @staticmethod
    def predict(X: pd.DataFrame) -> np.ndarray:
        return X["x"].to_numpy(dtype=float) ** 2


@pytest.fixture
def forced_reference_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nonparametric, "_ParametricRefWrapper", FailingReference)
    monkeypatch.setattr(nonparametric, "DistanceWeightedKNNModel", ExactQuadratic)
    monkeypatch.setattr(
        ModelEquationExtractor,
        "extract",
        staticmethod(lambda *args, **kwargs: SimpleNamespace(model_dump=lambda: {})),
    )


def assert_comparison_unavailable(result: dict[str, Any]) -> None:
    assert result["parametric_reference_score"] is None
    assert result["parametric_reference_status"] == "unavailable"
    assert result["comparison_status"] == "unavailable"
    assert result["best_model_name"] == "NonParametric_WeightedKNN"
    assert result["best_nonparametric_score"] == 1.0
    assert result["lift_vs_parametric"] is None
    assert result["nonparametric_outperforms_parametric"] is None
    assert not result["champion_model"]
    assert result["champion_score"] is None
    assert result["leaderboard"][0]["acceptance_status"] == "NOT ASSESSED — PARAMETRIC REFERENCE UNAVAILABLE"
    assert result["leaderboard"][0]["lift_vs_parametric"] is None
    assert "forced reference failure" not in result["verdict"]
    assert "naive_floor_score" not in result["equations"][0]
    assert result["equations"][0]["lift_over_naive"] is None


def test_auditor_reference_failure_does_not_accept_candidate(forced_reference_failure: None) -> None:
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    y = X["x"].to_numpy(dtype=float) ** 2
    result = nonparametric.NonParametricAnalyticalSuite.run(
        X, y, task_type=TaskType.REGRESSION, cv_folds=3, models=["weighted_knn"]
    )
    assert_comparison_unavailable(result)
    assert "NonParametric_WeightedKNN" in result["_fitted_models"]


def test_studio06_no_auto_promotion_after_reference_failure(forced_reference_failure: None) -> None:
    client = TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": "v01-reference-failure"})
    X = pd.DataFrame({"x": np.arange(24, dtype=float)})
    frame = X.assign(target=X["x"] ** 2)
    uploaded = client.post(
        "/api/workbench/s01/upload",
        files={"file": ("quadratic.csv", io.BytesIO(frame.to_csv(index=False).encode()), "text/csv")},
    )
    assert uploaded.status_code == 200, uploaded.text
    picked = client.post("/api/workbench/targets", json={"targets": ["target"]})
    assert picked.status_code == 200, picked.text
    response = client.post(
        "/api/workbench/s06/nonparametric", json={"target": "target", "models": ["weighted_knn"], "cv_folds": 3}
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert_comparison_unavailable(result)
    assert result["state"]["active_model_name"] is None
    assert result["state"]["available_models_count"] == 1


def test_a_measured_zero_score_remains_a_valid_reference(
    forced_reference_failure: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExactClass(ExactQuadratic):
        @staticmethod
        def predict(X: pd.DataFrame) -> np.ndarray:
            return X["x"].to_numpy(dtype=int) % 2

    class ReversedClass(ExactClass):
        @staticmethod
        def predict(X: pd.DataFrame) -> np.ndarray:
            return 1 - ExactClass.predict(X)

    monkeypatch.setattr(nonparametric, "_ParametricRefWrapper", ReversedClass)
    monkeypatch.setattr(nonparametric, "DistanceWeightedKNNModel", ExactClass)
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    result = nonparametric.NonParametricAnalyticalSuite.run(
        X, ExactClass.predict(X), task_type=TaskType.BINARY_CLASSIFICATION, cv_folds=3, models=["weighted_knn"]
    )
    assert result["parametric_reference_score"] == 0.0
    assert result["parametric_reference_status"] == "available"
    assert result["comparison_status"] == "available"
    assert result["lift_vs_parametric"] == 1.0
    assert result["nonparametric_outperforms_parametric"] is True


def test_reference_fold_nonfinite_predictions_cannot_create_zero_floor(
    forced_reference_failure: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NonfiniteReference(ExactQuadratic):
        @staticmethod
        def predict(X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), np.nan)

    monkeypatch.setattr(nonparametric, "_ParametricRefWrapper", NonfiniteReference)
    X = pd.DataFrame({"x": np.arange(12, dtype=float)})
    result = nonparametric.NonParametricAnalyticalSuite.run(
        X, ExactQuadratic.predict(X), task_type=TaskType.REGRESSION, cv_folds=3, models=["weighted_knn"]
    )
    assert_comparison_unavailable(result)
