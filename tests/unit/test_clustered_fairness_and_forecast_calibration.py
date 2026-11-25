"""Independent checks for grouped fairness and chronological calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omniforge.core.exceptions import EvaluationError
from omniforge.evaluation.fairness import FairnessAuditor
from omniforge.forecasting.engine import ForecastingEngine, _compare_rolling_interval_scores


def test_clustered_fairness_does_not_count_duplicates_as_independent() -> None:
    labels = np.tile([0, 1], 30)
    groups = np.repeat(["A", "B"], 30)
    units = np.tile(np.arange(30), 2)
    audit = FairnessAuditor()
    clustered = audit.audit(labels, labels, groups, cluster_ids=units)
    repeated = [np.repeat(arr, 16) for arr in (labels, groups, units)]
    replicated = audit.audit(repeated[0], repeated[0], repeated[1], cluster_ids=repeated[2])
    iid = audit.audit(repeated[0], repeated[0], repeated[1])

    assert clustered.passed and replicated.passed
    assert clustered.n_clusters == replicated.n_clusters == 30
    assert replicated.group_metrics["A"]["effective_cluster_count"] == pytest.approx(30)
    assert clustered.statistical_parity_p_value is None
    assert clustered.selection_interval_method == "bonferroni_weighted_hoeffding_independent_clusters"
    assert clustered.demographic_parity_ci_95 == replicated.demographic_parity_ci_95
    assert clustered.disparate_impact_ci_95 == replicated.disparate_impact_ci_95
    expected_margin = np.sqrt(np.log(2 * 2 / .05) / (2 * 30))
    assert clustered.demographic_parity_ci_95 == pytest.approx((0, 2 * expected_margin), abs=1e-6)
    assert iid.demographic_parity_ci_95 is not None
    assert iid.demographic_parity_ci_95[1] < clustered.demographic_parity_ci_95[1]


def test_missing_or_single_cluster_is_reported_without_iid_fallback() -> None:
    labels = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    groups = pd.Series(["A"] * 4 + ["B"] * 4)
    units = pd.Series(["one"] * 4 + ["two", "two", "three", None], name="site")
    audit = FairnessAuditor()
    report = audit.audit(labels, labels, groups, cluster_ids=units)
    assert report.cluster_attribute == "site"
    assert report.missing_cluster_count == 1
    assert report.assessment_status == "insufficient_evidence"
    assert report.selection_interval_method is None
    assert report.statistical_parity_p_value is None
    with pytest.raises(EvaluationError, match="aligned"):
        audit.audit(labels, labels, groups, cluster_ids=units.sort_index(ascending=False))


class _LinearForecaster:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def fit(self, series: np.ndarray) -> None:
        self.last = series[-1]

    def predict(self, horizon: int) -> pd.DataFrame:
        point = self.last + np.arange(1, horizon + 1)
        return pd.DataFrame({"yhat": point, "yhat_lower": point - 1, "yhat_upper": point + 1})


@pytest.mark.parametrize("degraded", [False, True])
def test_calibration_window_precedes_independent_forecast_evaluation(monkeypatch: pytest.MonkeyPatch, degraded: bool) -> None:
    monkeypatch.setattr("omniforge.forecasting.engine.StatisticalForecaster", _LinearForecaster)
    monkeypatch.setattr("omniforge.forecasting.engine.MLForecaster", _LinearForecaster)
    values = np.arange(102, dtype=float)
    values[-3:] += 100
    if degraded:
        values[75] = np.nan
    engine = ForecastingEngine(confidence_level=0.90, gap=2, include_deep=False, include_foundation=False, n_cv_splits=0)
    result = engine.forecast(pd.DataFrame({"y": values}), target_col="y", horizon=3, interval_calibration_origins=9)
    assert result.champion_model == "Baseline_drift"
    assert result.leaderboard[0]["rmse"] == 0
    facts = result.metadata["champion_interval_calibration"]
    assert facts["selection_end_position"] == 68
    assert facts["calibration_start_position"] == 70
    assert facts["calibration_end_position"] == 97
    assert facts["evaluation_start_position"] == 99
    if degraded:
        assert facts["status"] == "insufficient_finite_calibration_errors"
        assert facts["usable_origins_by_lead"] == [9, 9, 8]
        assert facts["evaluation"] is None
    else:
        assert facts["status"] == "applied"
        assert facts["half_width_by_lead"] == pytest.approx([0, 0, 0])
        assert facts["evaluation"]["observed_outcomes"] == 3
        assert facts["evaluation"]["calibrated_coverage"] == 0
        assert facts["evaluation"]["calibrated_mean_interval_score"] == pytest.approx(2000)
        np.testing.assert_allclose(result.forecast["yhat_upper"], result.forecast["yhat"])


def test_short_series_keeps_nominal_bounds_when_calibration_is_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("omniforge.forecasting.engine.StatisticalForecaster", _LinearForecaster)
    monkeypatch.setattr("omniforge.forecasting.engine.MLForecaster", _LinearForecaster)
    history = pd.DataFrame({"y": np.arange(40, dtype=float)})
    engine = ForecastingEngine(include_deep=False, include_foundation=False, n_cv_splits=0)
    default = engine.forecast(history, target_col="y", horizon=4)
    requested = engine.forecast(history, target_col="y", horizon=4, interval_calibration_origins=20)
    pd.testing.assert_frame_equal(default.forecast, requested.forecast)
    assert requested.metadata["champion_interval_calibration"]["status"] == "insufficient_history_or_origins"


def test_multiple_evaluation_origins_keep_frozen_widths_and_pair_comparisons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("omniforge.forecasting.engine.StatisticalForecaster", _LinearForecaster)
    monkeypatch.setattr("omniforge.forecasting.engine.MLForecaster", _LinearForecaster)
    engine = ForecastingEngine(confidence_level=0.9, gap=2, include_deep=False, include_foundation=False, n_cv_splits=0)
    fit_history = []

    def predict(name, builders, train: np.ndarray, horizon: int) -> pd.DataFrame:
        if name == "Baseline_drift":
            fit_history.append((len(train), train[-1]))
            if len(train) == 104:
                raise ValueError("unavailable evaluation fold")
        point = np.arange(len(train), len(train) + horizon, dtype=float)
        upper = point + 1
        if len(train) == 107:
            upper = upper.astype(object)
            upper[1] = "unavailable"
        return pd.DataFrame({"step": np.arange(1, horizon + 1), "yhat": point, "yhat_lower": point - 1, "yhat_upper": upper})

    monkeypatch.setattr(engine, "_refit_and_predict", predict)
    values = np.arange(110, dtype=float)
    values[[102, 107, 108]] += [2, 0.5, 2]
    values[103] = np.nan
    result = engine.forecast(
        pd.DataFrame({"y": values}), target_col="y", horizon=3,
        interval_calibration_origins=9, interval_evaluation_origins=3,
        compare_rolling_intervals=True,
    )
    facts = result.metadata["champion_interval_calibration"]
    assert result.champion_model == "Baseline_drift" and result.leaderboard[0]["rmse"] == 0
    assert facts["selection_end_position"] == 70
    assert facts["calibration_end_position"] == 99
    assert facts["half_width_by_lead"] == [0, 0, 0]
    evaluation = facts["evaluation"]
    assert [(row["start_position"], row["status"]) for row in evaluation["by_origin"]] == [
        (101, "scored"), (104, "forecast_unavailable"), (107, "scored"),
    ]
    assert fit_history[-4:] == [(101, 100), (104, 104), (107, 106), (110, 109)]
    assert evaluation["scored_origins"] == 2
    assert evaluation["observed_outcomes"] == 5 and evaluation["native_coverage_count"] == 4
    assert evaluation["calibrated_coverage"] == pytest.approx(0.4)
    assert evaluation["paired_calibrated_coverage"] == pytest.approx(0.5)
    assert evaluation["native_coverage"] == pytest.approx(0.75)
    assert evaluation["calibrated_mean_interval_score"] == pytest.approx(18)
    assert evaluation["native_mean_interval_score"] == pytest.approx(7)
    assert [row["observed_outcomes"] for row in evaluation["by_lead"]] == [2, 2, 1]
    comparison = facts["rolling_comparison"]
    assert comparison["paired_outcomes"] == 5
    assert [row["start_position"] for row in comparison["by_origin"]] == [101, 104, 107]
    assert comparison["by_origin"][1]["paired_outcomes"] == 0
    assert comparison["fixed"]["coverage"] == evaluation["calibrated_coverage"]
    np.testing.assert_allclose(result.forecast["yhat"], result.forecast["yhat_upper"])


def test_trailing_interval_expands_after_shock_then_releases_it() -> None:
    evaluation = np.ones((12, 2))
    evaluation[0] = [20, 2]
    evaluation[1, 1] = np.nan
    result = _compare_rolling_interval_scores(np.ones((9, 2)), evaluation, np.ones(2), 0.10)
    windows = result["by_origin"]

    assert windows[0]["rolling_half_width_by_lead"] == [1, 1]
    assert windows[0]["rolling"]["coverage"] == 0
    assert windows[1]["rolling_half_width_by_lead"] == [20, 2]
    assert windows[2]["retained_width_leads"] == [2]
    assert windows[2]["rolling_half_width_by_lead"] == [20, 2]
    assert windows[10]["rolling_half_width_by_lead"] == [1, 2]
    assert windows[11]["rolling_half_width_by_lead"] == [1, 1]
    assert result["paired_outcomes"] == 23
    assert result["rolling"]["mean_width"] > result["fixed"]["mean_width"]
