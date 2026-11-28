"""Independent evaluation, champion export and DAG replay regressions."""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from omniforge.automl.pipeline_search import AutoMLPipelineSearch, _ChampionEstimator
from omniforge.core.security import ArtifactSigner, safe_joblib_load_bytes
from omniforge.core.types import TaskType
from omniforge.models.classical import ClassicalMLModel
from omniforge.models.deep.mlp import MLPModel
from omniforge.models.ensemble import VotingEnsembleModel
from omniforge.pipelines.base import FunctionStep, PipelineContext
from omniforge.pipelines.dag import PipelineDAG
from omniforge.pipelines.executor import PipelineExecutor


@pytest.mark.parametrize("strategy", ["stratified", "group", "time_series"])
def test_native_automl_selects_before_independent_evaluation(
    strategy: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"row": np.arange(96), "feature": rng.normal(size=96)})
    y = rng.permutation(np.tile([0, 1], 48))
    groups = X["row"] // 6
    fits: list[tuple[ClassicalMLModel, set[int]]] = []
    predictions: list[tuple[ClassicalMLModel, set[int]]] = []
    fit = ClassicalMLModel.fit
    predict = ClassicalMLModel.predict

    def track_fit(self: ClassicalMLModel, frame: pd.DataFrame, labels: np.ndarray) -> ClassicalMLModel:
        fits.append((self, set(frame["row"])))
        return fit(self, frame, labels)

    def track_predict(self: ClassicalMLModel, frame: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        predictions.append((self, set(frame["row"])))
        return predict(self, frame, **kwargs)

    monkeypatch.setattr(ClassicalMLModel, "fit", track_fit)
    monkeypatch.setattr(ClassicalMLModel, "predict", track_predict)
    result = AutoMLPipelineSearch(
        task_type=TaskType.BINARY_CLASSIFICATION,
        split_strategy=strategy,
        gap=2 if strategy == "time_series" else 0,
        include_deep=False,
        enable_ensembling=False,
        use_cv_pipeline=False,
        tune_trials=2,
        tune_top_k=1,
    ).fit(X, y, groups=groups if strategy == "group" else None)

    assert result.test_indices is not None
    test_rows = set(result.test_indices)
    assert all(not (rows & test_rows) for _, rows in fits)
    assert [(model, rows) for model, rows in predictions if rows & test_rows] == [(result.best_model, test_rows)]
    champion_fit = next(rows for model, rows in fits if model is result.best_model)
    assert len(champion_fit) == result.metadata["evaluation"]["n_champion_fit"]
    if strategy == "group":
        assert set(groups.iloc[list(champion_fit)]).isdisjoint(groups.iloc[result.test_indices])
    if strategy == "time_series":
        assert min(test_rows) - max(champion_fit) == 3
    assert result.leakage_audit is not None and result.leakage_audit.passed
    expected_score = f1_score(
        y[result.test_indices], result.best_model.predict(X.iloc[result.test_indices]), average="weighted",
    )
    assert result.best_score == round(expected_score, 6)
    assert result.best_score == result.best_model.metadata_.metrics["f1_weighted"]
    np.testing.assert_array_equal(result.metadata["sklearn_pipeline"].predict(X), result.best_model.predict(X))


@pytest.mark.parametrize("family", ["tuned_classical", "deep", "ensemble", "regression"])
def test_champion_pipeline_roundtrip_keeps_predictions(family: str) -> None:
    X = pd.DataFrame({"n": np.arange(16), "category": ["a", "b"] * 8})
    y = np.tile([0, 1], 8)
    if family == "regression":
        model = ClassicalMLModel("ridge", task_type=TaskType.REGRESSION, track_carbon=False)
        y = X["n"].to_numpy() * 1.5
    elif family == "deep":
        model = MLPModel(hidden_dims=(8,), epochs=2, task_type=TaskType.BINARY_CLASSIFICATION, track_carbon=False)
    else:
        tree = ClassicalMLModel(
            "random_forest", task_type=TaskType.BINARY_CLASSIFICATION, track_carbon=False, n_estimators=4,
        )
        model = tree if family == "tuned_classical" else VotingEnsembleModel(
            estimators=[
                tree, ClassicalMLModel("logistic_regression", task_type=TaskType.BINARY_CLASSIFICATION, track_carbon=False),
            ],
            task_type=TaskType.BINARY_CLASSIFICATION,
            track_carbon=False,
        )
    model.fit(X, y)
    frame = pd.DataFrame({"n": [3, 14], "category": ["unseen", "a"]})
    pipeline = Pipeline([("model", _ChampionEstimator(model))])

    serialized = io.BytesIO()
    joblib.dump(pipeline, serialized)
    payload = serialized.getvalue()
    signer = ArtifactSigner()
    signature = signer.sign_bytes(payload)
    assert signer.verify_bytes(payload, signature)
    loaded = safe_joblib_load_bytes(payload)

    np.testing.assert_array_equal(loaded.predict(frame), model.predict(frame))
    if family != "regression":
        np.testing.assert_allclose(loaded.predict_proba(frame), model.predict_proba(frame))


def test_pipeline_retry_and_cache_replay_preserve_mutation_and_deletion(tmp_path: Path) -> None:
    attempts = 0

    def mutate(ctx: PipelineContext) -> dict[str, list[int]]:
        nonlocal attempts
        attempts += 1
        ctx.artifacts["payload"]["values"].append(2)
        del ctx.artifacts["obsolete"]
        ctx.params["updated"] = True
        if attempts == 1:
            raise RuntimeError("retry this step")
        return ctx.artifacts["payload"]

    def observe(ctx: PipelineContext) -> list[int]:
        assert ctx.step_outputs["mutate"] is ctx.artifacts["payload"]
        assert "obsolete" not in ctx.artifacts
        assert ctx.params["updated"]
        return ctx.artifacts["payload"]["values"]

    dag = PipelineDAG("mutable").add_step(FunctionStep("mutate", mutate, cacheable=True, retries=1, implementation_version="1"))
    dag.add_step(FunctionStep("observe", observe, depends_on=["mutate"]))
    executor = PipelineExecutor(cache_dir=tmp_path, track_carbon=False)
    inputs = {"payload": {"values": [1]}, "obsolete": "value"}
    cold = executor.execute(dag, initial_artifacts=inputs)
    warm = executor.execute(dag, initial_artifacts=inputs)

    assert cold.step_records[0].metadata["attempts"] == 2
    assert warm.step_records[0].cache_hit
    assert attempts == 2
    assert cold.context.step_outputs["observe"] == warm.context.step_outputs["observe"] == [1, 2]
    assert inputs == {"payload": {"values": [1]}, "obsolete": "value"}


def test_pipeline_concurrent_publish_keeps_complete_cache_entries(tmp_path: Path) -> None:
    barrier = Barrier(2)

    def step(ctx: PipelineContext) -> int:
        barrier.wait(timeout=5)
        ctx.set("value", 9)
        return 9

    dag = PipelineDAG("parallel").add_step(FunctionStep("value", step, cacheable=True, implementation_version="1"))
    executor = PipelineExecutor(cache_dir=tmp_path, track_carbon=False)
    with ThreadPoolExecutor(max_workers=2) as workers:
        runs = list(workers.map(lambda _: executor.execute(dag), range(2)))
    assert all(run.status == "success" for run in runs)
    assert len(list(tmp_path.iterdir())) == 1
    warm = executor.execute(dag)
    assert warm.step_records[0].cache_hit and warm.context.get("value") == 9


def test_supplied_holdout_labels_do_not_change_automl_selection() -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"x": rng.normal(size=60)})
    y = pd.Series((X["x"] > 0).astype(int))
    X_test = pd.DataFrame({"x": [-4.0, -2.0, 2.0, 4.0]}, index=[100, 101, 102, 103])
    y_test = pd.Series([0, 0, 1, 1], index=X_test.index)

    def run(labels: pd.Series) -> Any:
        return AutoMLPipelineSearch(
            task_type=TaskType.BINARY_CLASSIFICATION,
            include_deep=False,
            enable_ensembling=False,
            use_cv_pipeline=False,
            tune_trials=1,
            tune_top_k=1,
        ).fit(X, y, holdout=(X_test, labels))

    correct = run(y_test)
    flipped = run(1 - y_test)
    assert correct.best_model_name == flipped.best_model_name
    assert correct.metadata["evaluation"]["validation_score"] == flipped.metadata["evaluation"]["validation_score"]
    np.testing.assert_array_equal(correct.best_model.predict(X_test), flipped.best_model.predict(X_test))
    assert correct.best_score != flipped.best_score
    assert correct.metadata["evaluation"]["test_partition_source"] == "supplied_holdout"


def test_specialized_automl_scores_external_holdout_with_matching_baseline() -> None:
    from omniforge.automl.backends import AutoMLBackendHub

    rng = np.random.default_rng(17)
    data = pd.DataFrame({"x": rng.normal(size=60)})
    data["label"] = (data["x"] > 0).astype(int)
    X_test = pd.DataFrame({"x": [-4.0, -2.0, 2.0, 4.0]}, index=[100, 101, 102, 103])
    y_test = pd.Series([0, 0, 1, 1], index=X_test.index)
    result = AutoMLBackendHub.run(data, "label", backend="meta_bayesian", cv_folds=2, holdout=(X_test, y_test))
    evaluation = result.extra["evaluation"]
    assert evaluation["best_score_partition"] == "test"
    assert evaluation["n_train"] + evaluation["n_validation"] == len(data)
    assert evaluation["n_test"] == len(X_test)
    assert result.primary_metric == "f1_weighted"
    assert evaluation["lift_comparison"]["status"] == "unavailable"
    expected = f1_score(y_test, result.best_model.predict(X_test), average="weighted")
    assert result.best_score == round(expected, 6)
    strongest = max(metrics["f1_weighted"] for metrics in result.baseline_benchmarks.values())
    assert result.lift_over_best_baseline == pytest.approx(expected - strongest, abs=1e-6)
