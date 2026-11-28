"""Post-PR-33 audit boundary and numerical regressions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from omniforge.core.task_inference import infer_supervised_task
from omniforge.data import splitting
from omniforge.data.cleaning import robust_mahalanobis_outliers
from omniforge.eda.statistics import StatisticalAnalyzer, friedman_nemenyi_test
from omniforge.evaluation.validation import CombinatorialPurgedKFold
from omniforge.forecasting.engine import ForecastingEngine
from omniforge.models.time_series.ml_forecast import MLForecaster
from omniforge.reporting.live_charts import _infer_task_type, get_live_chart_catalog


def test_mcd_auditor_small_gaussian_fixture_is_ineligible() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        result = robust_mahalanobis_outliers(pd.DataFrame(rng.normal(size=(60, 3)), columns=list("abc")))
        assert result["evaluable"] is False
        assert result["minimum_complete_rows"] == 100
        assert result["n_scanned"] == result["sampled_from"] == 60
        assert "n_outliers" not in result


def test_mcd_reports_flags_not_calibrated_false_positive_rate_at_dimension_gate() -> None:
    frame = pd.DataFrame(np.random.default_rng(17).normal(size=(121, 6)))
    frame.iloc[-1, 0] = np.inf
    assert not robust_mahalanobis_outliers(frame.iloc[1:])["evaluable"]
    result = robust_mahalanobis_outliers(frame)
    assert result["evaluable"] is True
    assert result["minimum_complete_rows"] == result["n_scanned"] == 120
    assert result["sampled_from"] == 121
    assert result["cutoff_tail_probability"] == 0.01
    assert result["false_positive_rate_calibrated"] is False
    assert result["flagged_fraction"] == result["n_outliers"] / 120


@pytest.mark.parametrize(("n", "method"), [(10, "heuristic_first_difference_band"), (80, "heuristic_refit_one_step_residual_band")])
def test_native_ml_band_provenance_and_horizon_scaling(n: int, method: str) -> None:
    history = np.sin(np.arange(n)) + 0.08 * np.arange(n)
    forecast = MLForecaster("ridge", seasonal_period=3).fit(history).predict(12)
    assert forecast.attrs["native_interval"] == {
        "method": method, "coverage_guaranteed": False, "horizon_scaling": "sqrt_step_capped_at_3",
    }
    width = forecast["yhat_upper"] - forecast["yhat"]
    assert width.iloc[0] > 0
    np.testing.assert_allclose(width / width.iloc[0], np.minimum(np.sqrt(np.arange(1, 13)), 3))
    np.testing.assert_allclose(width, forecast["yhat"] - forecast["yhat_lower"])


class _QuadraticML:
    """Deterministic winning candidate to exercise native/rolling delivery policy."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def fit(self, values: np.ndarray) -> None:
        self.last = np.sqrt(values[-1])

    def predict(self, horizon: int) -> pd.DataFrame:
        point = (self.last + np.arange(1, horizon + 1)) ** 2
        result = pd.DataFrame({"step": np.arange(1, horizon + 1), "yhat": point, "yhat_lower": point - 2, "yhat_upper": point + 2})
        result.attrs["native_interval"] = {"method": "heuristic_refit_one_step_residual_band", "coverage_guaranteed": False}
        return result


@pytest.mark.parametrize("origins", [0, 9])
def test_forecast_report_keeps_native_provenance_separate_from_rolling_bounds(monkeypatch: pytest.MonkeyPatch, origins: int) -> None:
    monkeypatch.setattr("omniforge.forecasting.engine.MLForecaster", _QuadraticML)
    engine = ForecastingEngine(confidence_level=0.90, n_cv_splits=0, include_deep=False, include_foundation=False)
    result = engine.forecast(pd.DataFrame({"y": np.arange(102, dtype=float) ** 2}), target_col="y", horizon=3,
                             interval_calibration_origins=origins)
    assert result.champion_model.startswith("ML_")
    assert result.leaderboard[0]["native_interval"]["coverage_guaranteed"] is False
    if origins:
        assert result.metadata["champion_interval_calibration"]["status"] == "applied"
        assert result.metadata["champion_native_interval"] is None
        assert "native_interval" not in result.forecast.attrs
        np.testing.assert_allclose(result.forecast["yhat_upper"], result.forecast["yhat"])
    else:
        assert result.metadata["champion_interval_calibration"]["status"] == "disabled"
        assert result.metadata["champion_native_interval"] == result.forecast.attrs["native_interval"]
        np.testing.assert_allclose(result.forecast["yhat_upper"] - result.forecast["yhat"], 2)


def test_friedman_chi_square_does_not_override_iman_davenport_decision() -> None:
    scores = np.array([[3, 2, 1], [3, 2, 1], [2, 1, 3], [1, 3, 2]])
    chi2, chi_p = stats.friedmanchisquare(*scores.T)
    blocks, models = scores.shape
    f_p = stats.f.sf((blocks - 1) * chi2 / (blocks * (models - 1) - chi2), models - 1, (models - 1) * (blocks - 1))
    # Deliberately choose alpha between the two approximations to expose the old OR.
    alpha = 0.8
    assert chi_p < alpha < f_p
    result = friedman_nemenyi_test(scores, alpha=alpha)
    assert result["omnibus_decision_method"] == "iman_davenport_f"
    assert result["p_value"] == pytest.approx(chi_p, abs=1e-6)
    assert result["omnibus_p_value"] == pytest.approx(f_p, abs=1e-6)
    assert result["significant"] is False
    assert result["post_hoc_status"] == "omnibus_not_rejected"
    assert not any(pair["significant"] for pair in result["pairwise_differences"])


def test_explicit_legacy_cpcv_name_preserves_policy_and_import() -> None:
    assert splitting.CombinatorialPurgedKFold is splitting.LegacyCombinatorialPurgedKFold
    assert splitting.LegacyCombinatorialPurgedKFold().get_n_splits() == 15
    assert CombinatorialPurgedKFold().get_n_splits() == 10
    positions = np.zeros((12, 1))
    legacy = splitting.LegacyCombinatorialPurgedKFold(n_splits=3, purge_gap=20)
    canonical = CombinatorialPurgedKFold(n_splits=3, purge_gap=20)
    assert len(legacy.split(positions)) == 3
    assert all(not len(train) for train, _ in legacy.split(positions))
    canonical_splits = list(canonical.split(positions))
    assert len(canonical_splits) == 1
    np.testing.assert_array_equal(canonical_splits[0][0], [10, 11])


def test_chart_task_inference_uses_canonical_rule_and_does_not_label_missing_target() -> None:
    for target in (pd.Series([], dtype=float), pd.Series([1.0000001, 2.0000001] * 8)):
        assert _infer_task_type(target) == infer_supervised_task(target)
    for target, expected in (([np.nan] * 12, "unsupervised"), ([1.0000001, 2.0000001] * 6, "regression")):
        state = {"uap_df": pd.DataFrame({"x": np.arange(12), "target": target}), "uap_targets": ["target"]}
        columns = get_live_chart_catalog(state)["columns"]
        assert columns["primary_target"] == "target"
        assert columns["task_type"] == expected


@pytest.mark.parametrize(("table", "p_method"), [
    ([[18, 182], [32, 168]], "yates_corrected_chi_square"),
    ([[1, 2], [4, 1]], "fisher_exact_two_sided"),
    ([[20, 10, 0], [10, 20, 0], [5, 15, 10]], "pearson_chi_square"),
])
def test_cramers_v_variant_is_separate_from_association_p_value(table: list[list[int]], p_method: str) -> None:
    counts = np.asarray(table)
    i, j = np.indices(counts.shape)
    a = pd.Series(np.repeat(i.ravel(), counts.ravel()))
    b = pd.Series(np.repeat(j.ravel(), counts.ravel()))
    result = StatisticalAnalyzer.chi_square_independence(a, b)
    chi = stats.chi2_contingency(counts, correction=False).statistic
    assert result.effect_size == pytest.approx(np.sqrt(chi / (counts.sum() * (min(counts.shape) - 1))), abs=5e-5)
    assert result.details["effect_size_variant"] == "uncorrected_pearson_cramers_v"
    assert result.details["p_value_method"] == p_method
    expected_p = stats.fisher_exact(counts).pvalue if p_method == "fisher_exact_two_sided" else stats.chi2_contingency(counts).pvalue
    assert result.p_value == pytest.approx(expected_p)


def test_studio01_reports_small_sample_ineligibility_and_screening_scope() -> None:
    from uuid import uuid4

    from fastapi.testclient import TestClient

    from omniforge.api.auth import get_api_key
    from omniforge.api.main import create_app

    client = TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": uuid4().hex})
    rng = np.random.default_rng(0)
    for n in (60, 100):
        frame = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
        upload = client.post("/api/workbench/s01/upload", files={"file": ("mcd.csv", frame.to_csv(index=False), "text/csv")})
        assert upload.status_code == 200, upload.text
        response = client.post("/api/workbench/s01/advanced_audit")
        assert response.status_code == 200, response.text
        payload = response.json()["robust_mahalanobis"]
        assert payload["n_scanned"] == n
        assert payload["evaluable"] is (n == 100)
        if payload["evaluable"]:
            assert payload["false_positive_rate_calibrated"] is False
            assert payload["flagged_fraction"] == payload["n_outliers"] / n
        else:
            assert "n_outliers" not in payload


@pytest.mark.parametrize(("scores", "reason"), [
    (np.zeros((4, 2)), "three models"),
    (np.full((4, 3), np.nan), "finite"),
    (np.tile(np.arange(4)[:, None], (1, 3)), "tie"),
])
def test_friedman_reports_unevaluable_degenerate_inputs(scores: np.ndarray, reason: str) -> None:
    result = friedman_nemenyi_test(scores)
    assert result["evaluable"] is False
    assert reason in result["reason"]
