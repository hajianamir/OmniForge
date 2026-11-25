"""Regressions for inference, statistical evaluation, and configuration contracts."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX

from omniforge.api.studios.schemas import TwinControlReq
from omniforge.config.loader import ConfigLoader
from omniforge.core.exceptions import ConfigurationError, ForecastingError, MLOpsError, SimulationError
from omniforge.evaluation.fairness import FairnessAuditor
from omniforge.mlops.registry import ModelRegistry
from omniforge.mlops.root_cause import CausalDiscovery, DriftRootCauseAnalyzer
from omniforge.mlops.serving import ModelServingEngine
from omniforge.models.base import SklearnEstimatorAdapter
from omniforge.models.time_series.statistical import StatisticalForecaster
from omniforge.simulation import ClosedLoopMPCController, build_default_twin


def test_array_serving_keeps_primary_schema_when_canary_reorders_columns(tmp_path: Path) -> None:
    records = pd.DataFrame({"z": [0.0, 2.0, 0.0, 1.0], "a": [0.0, 0.0, 7.0, 4.0]})
    targets = records["z"] * 3 + records["a"] * 8
    primary = SklearnEstimatorAdapter(LinearRegression().fit(records, targets))
    canary = SklearnEstimatorAdapter(LinearRegression().fit(records[["a", "z"]], targets))
    engine = ModelServingEngine(registry=ModelRegistry(tmp_path / "r.db", tmp_path / "artifacts"))
    engine.deploy_model("score", primary, canary_model=canary, canary_traffic_ratio=1.0)
    result = engine.predict("score", records.to_numpy())
    assert result["variant"] == "canary"
    assert result["predictions"] == pytest.approx(targets)
    with pytest.raises(MLOpsError, match="Expected 2 array features"):
        engine.predict("score", records[["a"]].to_numpy())


def test_sarimax_forecasts_with_fitted_season_and_future_regressors() -> None:
    rng = np.random.default_rng(26)
    regressor = pd.DataFrame({"factor": rng.normal(size=80)})
    series = pd.Series(np.tile([-4.0, 1.0, 5.0, -2.0], 20) + 2 * regressor["factor"] + rng.normal(0, 0.15, 80))
    series.iloc[:2] = np.nan
    order = (0, 0, 0)
    model = StatisticalForecaster(method="sarimax", arima_order=order, seasonal_period=4).fit(series, exog=regressor)
    expected = SARIMAX(
        series.iloc[2:].to_numpy(), exog=regressor.iloc[2:].to_numpy(), order=order,
        seasonal_order=(1, 0, 0, 4), enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    future = pd.DataFrame({"factor": [1.0, -1.0, 2.0]})
    forecast = model.predict(3, exog=future)
    np.testing.assert_allclose(forecast["yhat"], expected.get_forecast(3, exog=future.to_numpy()).predicted_mean)
    with pytest.raises(ForecastingError, match="Supply future regressors"):
        model.predict(3)
    with pytest.raises(ForecastingError, match="match the fitted columns"):
        model.predict(3, exog=future.rename(columns={"factor": "other"}))


def test_fairness_requires_complete_demographics_and_selection_variation() -> None:
    labels = np.tile([0, 1], 3)
    groups = pd.Series(["A", "A", "Missing", "Missing", pd.NA, None])
    report = FairnessAuditor().audit(labels, labels, groups)
    assert report.assessment_status == "insufficient_evidence" and not report.passed
    assert report.missing_sensitive_count == 2
    assert set(report.group_metrics) == {"A", "Missing"}
    assert report.group_metrics["Missing"]["count"] == 2
    for prediction in (0, 1):
        degenerate = FairnessAuditor().audit(labels, np.full(6, prediction), np.repeat(["A", "B", "C"], 2))
        assert degenerate.assessment_status == "insufficient_evidence"
        assert degenerate.statistical_parity_p_value is None and not degenerate.passed


def test_drift_reports_observed_prediction_shift_independently_of_shap(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = pd.DataFrame({"time": range(30), "x": np.linspace(0, 1, 30), "y": np.linspace(0, 1, 30)})
    shifted = np.linspace(3.0, 3.5, 20)
    cur = pd.DataFrame({"time": range(30, 50), "x": shifted, "y": shifted})
    seen: set[str] = set()

    def onset(frame: pd.DataFrame, time_col: str, column: str) -> int:
        assert frame[time_col].tolist() == list(range(len(frame)))
        seen.add(column)
        return 1 if column == "x" else 2

    monkeypatch.setattr(CausalDiscovery, "_rolling_window_drift_onset", staticmethod(onset))

    class Model:
        def predict(self, rows: pd.DataFrame) -> np.ndarray:
            return 2 * rows["x"].to_numpy()

    report = DriftRootCauseAnalyzer().analyze(ref, cur, model=Model(), time_col="time")
    assert report.prediction_shift == pytest.approx(5.5)
    assert report.prediction_response == "numeric_prediction"
    assert report.attributed_prediction_shift is not None
    assert report.dag_edges == [["x", "y"]] and seen == {"x", "y"}
    tiny = DriftRootCauseAnalyzer().analyze(ref, cur.head(2), model=Model(), time_col="time")
    assert tiny.prediction_shift == pytest.approx(2 * cur.head(2)["x"].mean() - 1)
    assert tiny.attributed_prediction_shift is None
    assert "could not be assessed" in tiny.remediation_prescriptions[0]


def test_observational_cycle_removes_weakest_existing_edge() -> None:
    parents = {"a": {"c"}, "b": {"a"}, "c": {"b"}}
    assert CausalDiscovery._break_cycles(parents, {"a": 1.0, "b": 1.1, "c": 2.0}) == {
        "a": {"c"}, "b": set(), "c": {"b"},
    }


def test_failed_twin_solve_keeps_actuator_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SimulationError, match="n_x=2 and n_u=1"):
        build_default_twin(n_x=3)
    with pytest.raises(ValueError, match="n_u"):
        TwinControlReq(n_u=2)
    calls = []

    def fail(**kwargs):
        calls.append(kwargs["bounds"][0])
        return SimpleNamespace(success=False, x=np.full_like(kwargs["x0"], 500), message="solver stopped")

    monkeypatch.setattr("omniforge.simulation.closed_loop.minimize", fail)
    twin = build_default_twin()
    controller = ClosedLoopMPCController(twin, horizon=2, u_min=np.array([-0.1]), u_max=np.array([0.1]))
    report = controller.solve(np.array([1.0, 0.0]))
    assert calls == [(-0.1, 0.1), (-0.1, 0.1)]
    assert report.feasibility == "solver_failed" and not report.success
    np.testing.assert_allclose(report.full_control_sequence, 0.0)
    np.testing.assert_allclose(report.state_trajectory, [[1, 0], [1, 0], [1, 0]])


def test_config_honors_environment_then_cli_and_validates_domain_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "default.yaml").write_text("platform:\n  random_seed: 17\napi:\n  port: 8130\n")
    (tmp_path / "domains").mkdir()
    (tmp_path / "domains" / "domain-1.yaml").write_text("domain: demo\n")
    monkeypatch.setenv("OMNIFORGE_RANDOM_SEED", "91")
    monkeypatch.setenv("OMNIFORGE_API__HOST", "localhost")
    loader = ConfigLoader(tmp_path)
    merged = loader.compose(domain="DOMAIN-1")
    assert merged.random_seed == 91
    assert (merged.api.host, merged.api.port) == ("localhost", 8130)
    assert merged.domain_overrides == {"domain": "demo"}
    assert loader.compose(overrides=["platform.random_seed=32"]).random_seed == 32
    with pytest.raises(ConfigurationError, match="Domain names"):
        loader.compose(domain="invalid domain")
