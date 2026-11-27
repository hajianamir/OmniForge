"""Independent oracles for probability, optimization and nested validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from sklearn.linear_model import Ridge

from omniforge.automl.backends import MultiLayerStackingAutoML
from omniforge.automl.optimizer import HyperparameterOptimizer
from omniforge.core.exceptions import AutoMLError, EvaluationError, ForecastingError
from omniforge.core.types import TaskType
from omniforge.evaluation.metrics import ProbabilityCalibrator
from omniforge.models.time_series.hierarchical import HierarchicalReconciler


def test_temperature_maps_explicit_classes_including_an_unobserved_class() -> None:
    probabilities = np.array([[.05, .90, .05], [.08, .15, .77], [.10, .80, .10], [.05, .10, .85]])
    y = np.array(["second", "first", "second", "first"])
    labels = ["unseen", "second", "first"]
    calibrator = ProbabilityCalibrator().fit(y, probabilities, classes=labels)
    np.testing.assert_array_equal(calibrator.classes_, labels)
    expected = softmax(np.log(probabilities) / calibrator.temperature_, axis=1)
    np.testing.assert_allclose(calibrator.transform(probabilities), expected)
    reordered = [2, 0, 1]
    shuffled = ProbabilityCalibrator().fit(y, probabilities[:, reordered], classes=[labels[i] for i in reordered])
    np.testing.assert_allclose(shuffled.transform(probabilities[:, reordered]), expected[:, reordered])
    with pytest.raises(EvaluationError, match="one class label"):
        ProbabilityCalibrator().fit(y, probabilities)


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibration_requires_class_coverage_and_preserves_nonconsecutive_labels(method: str) -> None:
    y = np.array([70, 4, 70, 4])
    p = np.array([.05, .90, .15, .85])
    calibrator = ProbabilityCalibrator(method).fit(y, p, classes=[70, 4])
    np.testing.assert_array_equal(calibrator.classes_, [70, 4])
    scores = calibrator.transform(p)
    assert scores[1] > scores[0]
    with pytest.raises(EvaluationError, match="class coverage"):
        ProbabilityCalibrator(method).fit(np.full(4, 70), p, classes=[70, 4])


@pytest.mark.parametrize("p", [[[.3, .3], [.1, .9]], [[np.inf, 0], [.1, .9]], [[1.1, -.1], [.1, .9]], [[], []]])
def test_calibration_rejects_bad_probabilities_on_fit_and_transform(p: list[list[float]]) -> None:
    labels = np.array([0, 1])
    with pytest.raises(EvaluationError):
        ProbabilityCalibrator().fit(labels, np.asarray(p))
    calibrated = ProbabilityCalibrator().fit(labels, np.array([[.8, .2], [.1, .9]]))
    with pytest.raises(EvaluationError):
        calibrated.transform(np.asarray(p))
    with pytest.raises(EvaluationError, match="schema"):
        calibrated.transform(np.array([.2, .9]))


@pytest.mark.parametrize("strategy", ["grid", "random", "bayesian"])
@pytest.mark.parametrize("direction", ["maximize", "minimize"])
def test_optimizer_excludes_failed_trials_and_compares_unrounded_values(strategy: str, direction: str) -> None:
    scores = iter([np.nan, np.inf, None, .10000010, .10000020])

    def objective(_) -> float:
        value = next(scores)
        if value is None:
            raise RuntimeError("private trial details")
        return value

    optimizer = HyperparameterOptimizer(strategy=strategy, direction=direction, max_trials=5)
    result = optimizer.optimize({"x": {"type": "categorical", "choices": list(range(5))}}, objective)
    assert result.best_score == (.10000020 if direction == "maximize" else .10000010)
    assert [trial.status for trial in result.trials] == ["nonfinite", "nonfinite", "failed", "completed", "completed"]
    assert [trial.score for trial in result.trials[:3]] == [None, None, None]
    assert "private trial details" not in result.model_dump_json()
    with pytest.raises(AutoMLError, match="finite objective"):
        optimizer.optimize({"x": {"type": "categorical", "choices": [1, 2]}}, lambda _: np.nan)


@pytest.mark.parametrize("chronological", [False, True])
def test_nested_stacking_provenance_respects_group_and_time_scopes(monkeypatch: pytest.MonkeyPatch, chronological: bool) -> None:
    X = pd.DataFrame({"signal": np.linspace(-2, 2, 120), "kind": ["p", "q"] * 60})
    y = X["signal"] * 3 + np.sin(X["signal"])
    groups = np.repeat(np.arange(11), 10)
    options: dict[str, Any] = {"time_ordered": True, "gap": 2} if chronological else {"groups": groups}
    stacker = MultiLayerStackingAutoML(cv_folds=3)
    monkeypatch.setattr(stacker, "_layer1_specs", lambda _: {"L1_Ridge": Ridge()})
    result = stacker.fit(X.iloc[:110], y.iloc[:110], X.iloc[110:], y.iloc[110:], TaskType.REGRESSION, **options)
    assert result["stacking_validation"]["layer_two_candidates"]
    assert result["stacking_validation"]["n_oof_rows"] < 110 if chronological else result["stacking_validation"]["n_oof_rows"] == 110
    for fold in result["fold_provenance"]:
        outer_train, outer_validation = set(fold["train"]), set(fold["validation"])
        assert outer_train.isdisjoint(outer_validation)
        assert fold["inner_folds"]
        for split in [fold, *fold["inner_folds"]]:
            train, validation = split["train"], split["validation"]
            assert set(train).isdisjoint(validation)
            if chronological:
                assert max(train) + 2 < min(validation)
            else:
                assert set(groups[train]).isdisjoint(groups[validation])
        for inner in fold["inner_folds"]:
            assert set(inner["train"]) | set(inner["validation"]) <= outer_train
    model = result["best_model"]
    held_out = X.iloc[110:]
    np.testing.assert_allclose(model.predict(held_out), model.predict(held_out.to_numpy()))


def test_hierarchy_distinguishes_repeated_names_under_different_parents() -> None:
    hierarchy = pd.DataFrame({
        "region": ["south", "north", "north", "north"],
        "branch": ["central", "central", "other", "central"],
        "site": ["main", "main", "annex", "main"],
    })
    S, labels = HierarchicalReconciler.build_summing_matrix(hierarchy, ["region", "branch", "site"])
    assert labels[-3:] == [
        "region=north/branch=central/site=main", "region=north/branch=other/site=annex", "region=south/branch=central/site=main",
    ]
    np.testing.assert_array_equal(S[-3:], np.eye(3))
    np.testing.assert_array_equal(S @ np.array([2., 3., 10.]), [15, 5, 10, 2, 3, 10, 2, 3, 10])
    forecasts = np.arange(len(S), dtype=float)
    fitted = HierarchicalReconciler("ols").reconcile(S, forecasts)
    assert HierarchicalReconciler.verify_coherence(S, fitted)["coherent"]
    with pytest.raises(ForecastingError, match="requires at least two residual"):
        HierarchicalReconciler("mint_shrink").reconcile(S, forecasts)


@pytest.mark.parametrize("design", ["group", "time_series"])
def test_backend_hub_preserves_explicit_stacking_split(monkeypatch: pytest.MonkeyPatch, design: str) -> None:
    from omniforge.automl.backends import AutoMLBackendHub

    frame = pd.DataFrame({"signal": np.linspace(-3, 3, 100)})
    frame["outcome"] = 2 * frame["signal"] + np.sin(frame["signal"])
    options: dict[str, Any]
    if design == "group":
        design_vector = np.repeat(np.arange(10), 10)
        options = {"groups": design_vector}
    else:
        design_vector = pd.date_range("2025-01-01", periods=len(frame))
        options = {"time_values": design_vector, "gap": 2}
        order = np.random.default_rng(42).permutation(len(frame))
        frame = frame.iloc[order]
        options["time_values"] = design_vector[order]

    original = MultiLayerStackingAutoML.fit

    def observe(self, X_train, y_train, X_val, y_val, task_type, **kwargs):
        assert list(X_train.columns) == list(X_val.columns) == ["signal"]
        if design == "group":
            assert set(design_vector[X_train.index]).isdisjoint(design_vector[X_val.index])
            np.testing.assert_array_equal(kwargs["groups"], design_vector[X_train.index])
        else:
            times = design_vector[X_train.index]
            assert times.is_monotonic_increasing
            assert max(X_train.index) + 2 < min(X_val.index)
            assert kwargs["time_ordered"] and kwargs["gap"] == 2
        return original(self, X_train, y_train, X_val, y_val, task_type, **kwargs)

    monkeypatch.setattr(MultiLayerStackingAutoML, "fit", observe)
    monkeypatch.setattr(MultiLayerStackingAutoML, "_layer1_specs", lambda *_: {"L1_Ridge": Ridge()})
    result = AutoMLBackendHub.run(
        frame, "outcome", backend="multi_layer_stacker", task_type=TaskType.REGRESSION,
        split_strategy=design, **options,
    )
    assert result.extra["stacking_validation"]["layer_two_candidates"]
    assert np.isfinite(result.best_score)


def test_small_inner_fold_excludes_only_the_base_model_that_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    class MinimumRowsRidge(Ridge):
        def fit(self, X, y, sample_weight=None):
            if len(X) < 15:
                raise ValueError("More inner training observations are required.")
            return super().fit(X, y, sample_weight=sample_weight)

    X = pd.DataFrame({"x": np.linspace(-2, 2, 120)})
    y = X["x"] * 3 + np.sin(X["x"])
    stacker = MultiLayerStackingAutoML(cv_folds=3)
    monkeypatch.setattr(stacker, "_layer1_specs", lambda _: {"always_fits": Ridge(), "needs_more": MinimumRowsRidge()})
    result = stacker.fit(
        X.iloc[:110], y.iloc[:110], X.iloc[110:], y.iloc[110:], TaskType.REGRESSION,
        time_ordered=True, gap=2,
    )
    assert {row["model"] for row in result["leaderboard"] if row["layer"] == "L1"} == {"always_fits", "needs_more"}
    assert result["stacking_validation"]["layer_two_candidates"]
    assert result["stacking_validation"]["layer_two_inputs"] == ["always_fits"]
    assert np.isfinite(result["best_model"].predict(X.iloc[110:])).all()
