"""Regression oracles for portable models, equations, and numerical Workbench flows."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from omniforge.api import workbench as wb
from omniforge.api.auth import get_api_key
from omniforge.api.main import create_app
from omniforge.core.exceptions import DependencyMissingError, ModelTrainingError
from omniforge.core.types import Principal, TaskType
from omniforge.evaluation.equations import ModelEquationExtractor
from omniforge.models.anomaly.detector import AnomalyDetector
from omniforge.models.base import BaseOmniForgeModel, SklearnEstimatorAdapter
from omniforge.models.classical import ClassicalMLModel
from omniforge.models.clustering.clusterer import ClusteringEngine
from omniforge.models.deep.gnn import GraphNeuralNetworkModel
from omniforge.models.deep.mlp import MLPModel
from omniforge.models.ensemble import VotingEnsembleModel
from omniforge.ui.state import load_new_dataset


def test_sklearn_clone_pipeline_search_and_effective_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("omniforge.models.deep.mlp.is_package_available", lambda _: False)
    rng = np.random.default_rng(27)
    X = rng.normal(size=(60, 3))
    y = np.where(X[:, 0] > 0, "yes", "no")
    classifier = ClassicalMLModel("logistic_regression", track_carbon=False)
    regressor = ClassicalMLModel("ridge", task_type=TaskType.REGRESSION, track_carbon=False)
    neural = MLPModel(hidden_dims=[5], epochs=2, track_carbon=False)
    ensemble = VotingEnsembleModel(estimators=[classifier, neural], weights=[1., 1.], track_carbon=False)
    assert is_classifier(classifier) and is_regressor(regressor)
    assert clone(ensemble).get_params()["weights"] == [1., 1.]
    assert clone(neural).get_params()["hidden_dims"] == [5]
    adapter = SklearnEstimatorAdapter(Ridge())
    assert not clone(adapter).is_fitted_
    assert clone(adapter).fit(X, X[:, 0]).predict(X).shape == (len(X),)
    assert clone(AnomalyDetector(method=" MAD ")).method == "mad"
    assert clone(ClusteringEngine(algorithm=" KMeans ")).algorithm == "kmeans"
    cv = cross_val_score(Pipeline([("model", ensemble)]), X, y, cv=2)
    assert np.isfinite(cv).all()
    result = GridSearchCV(Pipeline([("model", regressor)]), {"model__alpha": [0.1, 1.]}, cv=2).fit(X, X[:, 0] + X[:, 1])
    assert result.best_estimator_.named_steps["model"]._estimator.alpha == result.best_params_["model__alpha"]
    neural.set_params(hidden_dims=(6,), dropout=0.0)
    assert neural.hidden_dims == [6] and neural.dropout == 0.0
    with pytest.raises(ModelTrainingError, match="Unsupported"):
        neural.set_params(nonexistent=0)


def test_new_category_fit_distinguishes_null_from_strings_and_unknowns() -> None:
    model = ClassicalMLModel("ridge", task_type=TaskType.REGRESSION, track_carbon=False)
    X = pd.DataFrame({0: pd.Categorical([None, "MISSING", "nan", "x"])})
    matrix, _ = model._prepare_X(X, fitting=True)
    assert len(np.unique(matrix)) == 4
    tested, _ = model._prepare_X(pd.DataFrame({0: pd.array([pd.NA, "MISSING", "new"], dtype="string")}), fitting=False)
    assert tested[0, 0] == matrix[0, 0]
    assert tested[1, 0] == matrix[1, 0]
    assert tested[2, 0] == -1


def test_surrogate_class_and_held_out_r2_match_python_and_sql_exports() -> None:
    class Reference:
        classes_ = np.array(["absent", "actual class", "third"])

        def predict_proba(self, X):
            numeric = X.iloc[:, 0].fillna(0).to_numpy()
            category = X.iloc[:, 1].fillna("").eq("O'Reilly").to_numpy()
            p = .3 + .02 * numeric + .08 * category
            return np.column_stack([np.full(len(p), .13), p, .87 - p])

    frame = pd.DataFrame({"in\"put": np.linspace(-2, 2, 100), "label": ["O'Reilly", None, "c", "d"] * 25})
    frame.loc[[3, 4], 'in"put'] = np.nan
    target = "patient's outcome"
    model = Reference()
    report = ModelEquationExtractor.extract(model, X_reference=frame, target_name=target)
    validation = report.surrogate_validation
    assert validation["response_class"] == "actual class"
    assert report.response_class == "actual class"
    assert "actual class" in report.surrogate_plain_equation
    assert report.surrogate_fidelity_r2 is not None
    order = np.random.default_rng(42).permutation(len(frame))
    reference = frame.iloc[order[-validation["n_evaluation"]:]]
    reference_prob = model.predict_proba(reference)[:, 1]

    namespace: dict = {}
    exec(compile(report.formula_python, "<generated equation>", "exec"), namespace)
    predictor = next(value for key, value in namespace.items() if key.startswith("predict_"))
    rows = reference.astype(object).where(reference.notna(), None).to_dict("records")
    emitted = np.array([predictor(row) for row in rows])
    assert report.surrogate_fidelity_r2 == pytest.approx(r2_score(reference_prob, emitted), abs=1e-10)
    assert validation["mae"] == pytest.approx(np.abs(reference_prob - emitted).mean(), abs=1e-10)
    # SQL scoring operates on the documented encoded input schema.
    prepared = {}
    for name, rule in validation["input_encoding"].items():
        column = reference[rule["feature"]]
        if rule["kind"] == "numeric":
            prepared[name] = column.fillna(rule["fill"]).to_numpy()
        elif rule["category"] is None:
            prepared[name] = column.isna().to_numpy(dtype=float)
        else:
            prepared[name] = column.eq(rule["category"]).fillna(False).to_numpy(dtype=float)
    with sqlite3.connect(":memory:") as connection:
        pd.DataFrame(prepared).to_sql("your_table", connection, index=False)
        actual = pd.read_sql_query(report.formula_sql, connection).iloc[:, 0].to_numpy()
    np.testing.assert_allclose(actual, emitted, atol=1e-12)
    assert report.formula_excel.splitlines()[0].startswith("=MIN(")
    assert "fmin" in report.formula_c and "min(" in report.formula_matlab


def test_constant_surrogate_does_not_publish_perfect_r_squared() -> None:
    class Constant:
        def predict(self, X):
            return np.full(len(X), 9.0)

    report = ModelEquationExtractor.extract(Constant(), X_reference=pd.DataFrame({"x": range(15)}))
    assert report.surrogate_validation["mae"] == 0
    assert report.surrogate_fidelity_r2 is None
    assert "R² unavailable" in report.external_usage_summary


def test_nonlinear_preprocessing_and_hinge_loss_do_not_publish_exact_probabilities() -> None:
    frame = pd.DataFrame({"x": np.linspace(-3, 3, 50)})
    model = Pipeline([("polynomial", PolynomialFeatures(2)), ("ridge", Ridge())]).fit(frame, frame["x"] ** 2)
    report = ModelEquationExtractor.extract(model, X_reference=frame)
    assert report.equation_type != "exact_closed_form"
    assert report.surrogate_validation["n_evaluation"] > 0

    hinge = SGDClassifier(loss="hinge", random_state=24).fit(frame, (frame["x"] > 0).astype(int))
    report = ModelEquationExtractor.extract(hinge, X_reference=frame)
    assert report.equation_type != "exact_closed_form"
    assert not report.formula_python


def test_gnn_is_batch_stable_and_explicit_graph_requires_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("omniforge.models.deep.gnn.is_package_available", lambda _: False)
    monkeypatch.setattr("omniforge.models.deep.mlp.is_package_available", lambda _: False)
    rng = np.random.default_rng(19)
    X = rng.normal(size=(28, 4))
    y = X[:, 0] + X[:, 1]
    model = GraphNeuralNetworkModel(epochs=2, task_type=TaskType.REGRESSION, track_carbon=False).fit(X, y)
    np.testing.assert_allclose(model.predict(X[:4]), np.concatenate([model.predict(row[None]) for row in X[:4]]))
    assert "GCN fixed projection" in ModelEquationExtractor.extract(model).plain_equation
    with pytest.raises(DependencyMissingError, match="requires"):
        GraphNeuralNetworkModel(architecture="gat", epochs=2, track_carbon=False).fit(X, y > 0)
    explicit = GraphNeuralNetworkModel(epochs=2, task_type=TaskType.REGRESSION, track_carbon=False).fit(X, y, adjacency=np.eye(len(X)))
    with pytest.raises(ModelTrainingError, match="requires a graph"):
        explicit.predict(X[:4])
    assert explicit.predict(X[:4], adjacency=np.eye(4)).shape == (4,)


def test_twin_input_validation_and_portfolio_return_provenance() -> None:
    principal = Principal(subject="api_key_principal", roles=["admin"], tenant_id="default-tenant")
    sid = uuid4().hex
    state = wb.get_session_state(wb._sid(sid, principal))
    study = pd.DataFrame({"first": [1., 2., 4., 8.], "second": [2., 3., 4., 6.]})
    load_new_dataset(state, study, "return data", "controlled fixture")
    with TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": sid}) as client:
        response = client.post("/api/workbench/s08/optimize", json={"asset_cols": ["first", "second"]})
        assert response.status_code == 200, response.text
        returns = response.json()["optimization"]["return_estimation"]
        assert returns["n_intervals"] == 3
        assert returns["expected_returns"]["first"] == 1.0
        endpoint = "/api/workbench/s09/twin-control"
        for bad in ({"Q_diag": [1.]}, {"R_diag": [0.]}, {"measurements": [[0., 1.]], "n_steps": 2}):
            assert client.post(endpoint, json=bad).status_code == 422
        observed = {"measurements": [[2., 1.], [3., 1.]], "n_steps": 2}
        result = client.post(endpoint, json=observed)
        assert result.status_code == 200, result.text
        data = result.json()
        assert data["provenance"]["measurement_source"] == "observed"
        endpoint_state = data["trajectory_chart"]["state_trajectory"][-1]
        repeat = client.post(endpoint, json=observed).json()
        assert repeat["trajectory_chart"]["state_trajectory"] == data["trajectory_chart"]["state_trajectory"]
        resumed = client.post(endpoint, json={**observed, "state_mode": "resume"}).json()
        assert resumed["provenance"]["initial_state"] == endpoint_state
        assert client.post(endpoint, json={**observed, "state_mode": "resume", "dt": 2}).status_code == 409


def test_torch_state_roundtrip_uses_existing_signed_model_path(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    X = np.random.default_rng(42).normal(size=(24, 3))
    model = MLPModel(epochs=2, task_type=TaskType.REGRESSION, track_carbon=False).fit(X, X[:, 0])
    assert model.framework == "pytorch"
    path = model.save(tmp_path / "torch-model.joblib")
    restored = BaseOmniForgeModel.load(path, expected_artifact_checksum=model.metadata_.artifact_checksum)
    np.testing.assert_allclose(restored.predict(X), model.predict(X))

    graph = GraphNeuralNetworkModel(architecture="gat", n_hops=3, epochs=2, track_carbon=False).fit(X, X[:, 0] > 0)
    assert len(graph._torch_net.layers) == 3
    assert all(layer.a_src.grad is not None for layer in graph._torch_net.layers)
    report = ModelEquationExtractor.extract(graph)
    assert "GAT[hops=3" in report.plain_equation
    assert report.parameter_count == sum(p.numel() for p in graph._torch_net.parameters())
    graph_path = graph.save(tmp_path / "graph.joblib")
    restored_graph = BaseOmniForgeModel.load(graph_path, expected_artifact_checksum=graph.metadata_.artifact_checksum)
    np.testing.assert_allclose(restored_graph.predict_proba(X), graph.predict_proba(X))
