"""Independent checks for grouped fairness and chronological calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omniforge.core.exceptions import EvaluationError
from omniforge.evaluation.fairness import FairnessAuditor
from omniforge.forecasting.engine import ForecastingEngine


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
