"""Regression checks for provenance and uncertainty in audited metrics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from omniforge.core.carbon import CarbonTracker
from omniforge.core.exceptions import DataValidationError
from omniforge.core.types import ModelMetadata, TaskType
from omniforge.data.quality import DataQualityProfiler
from omniforge.evaluation.fairness import FairnessAuditor
from omniforge.evaluation.model_cards import ModelCardGenerator
from omniforge.models.computer_vision.foundation import VisionFoundationHub


def test_quality_reference_accuracy_and_rule_denominators() -> None:
    df = pd.DataFrame({"amount": [21, 22, 23, 24], "note": ["a", "b", None, "d"]}, index=[4, 2, 3, 1])
    trusted = pd.DataFrame({"amount": [24, 93, 91, 92], "note": ["d", "c", "a", "x"]}, index=[1, 3, 4, 2])
    profiler = DataQualityProfiler()
    unmeasured = profiler.profile(df)
    assert unmeasured.dimensions["accuracy"] is None
    assert "accuracy" not in unmeasured.effective_weights

    report = profiler.profile(df, reference_rules={"amount": (20, 25)}, reference_data=trusted)
    assert report.reference_comparisons == 7
    assert report.dimensions["accuracy"] == pytest.approx(3 / 7 * 100, abs=0.01)
    assert report.rule_conformance == 100.0 and report.rule_checks == 4
    assert report.column_scores[0].statistical_plausibility == 100.0
    assert report.column_scores[0].reference_accuracy == 25.0
    assert sum(report.effective_weights.values()) == pytest.approx(1)
    assert profiler.profile(df, reference_data=df).dimensions["accuracy"] == 100.0

    different_weights = DataQualityProfiler(weights={"accuracy": 1}).profile(
        df, reference_rules={"amount": (20, 25)}, reference_data=trusted,
    )
    assert different_weights.overall_score == different_weights.dimensions["accuracy"]
    with pytest.raises(DataValidationError, match="unique"):
        profiler.profile(df, reference_data=pd.concat([trusted, trusted]))


def test_fairness_intervals_cover_equal_rates_with_unequal_group_sizes() -> None:
    group = np.repeat(["rare", "middle", "large"], [2, 10, 40])
    predictions = np.tile([0, 1], 26)
    report = FairnessAuditor().audit(predictions, predictions, group)
    assert report.demographic_parity_diff == 0.0
    assert report.disparate_impact_ratio == 1.0
    assert report.selection_interval_method == "bonferroni_clopper_pearson_independent_rows"
    assert report.statistical_parity_test == "pearson_chi_square_monte_carlo"
    assert report.demographic_parity_ci_95 is not None
    assert report.disparate_impact_ci_95 is not None
    tail = 0.05 / 6
    low = stats.beta.ppf(tail, [1, 5, 20], [2, 6, 21])
    high = stats.beta.ppf(1 - tail, [2, 6, 21], [1, 5, 20])
    assert report.demographic_parity_ci_95 == pytest.approx((0.0, high.max() - low.min()), abs=1e-6)
    assert report.disparate_impact_ci_95 == pytest.approx((low.min() / high.max(), 1.0), abs=1e-6)
    assert report.group_metrics["rare"]["selected_count"] == 1

    zero = FairnessAuditor().audit(predictions, np.zeros(52), group)
    assert zero.statistical_parity_p_value is None
    assert zero.disparate_impact_ci_95 == (0, 1)


def test_vision_fallback_abstains_for_unrelated_and_tied_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("omniforge.models.computer_vision.foundation.is_package_available", lambda _: False)
    hub = VisionFoundationHub(image_size=(32, 32))
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    unrelated = hub.zero_shot_classify(black, ["airplane", "carpet"])
    assert unrelated["top_label"] is None
    assert unrelated["predictions"] == []
    tied = hub.zero_shot_classify(black, ["black", "dark"])
    assert tied["top_label"] is None
    assert tied["score_kind"] == "visual_cue_heuristic"
    assert all("probability" not in item for item in tied["predictions"])

    monkeypatch.setattr(hub._base_cv, "detect_objects", lambda *_args, **_kwargs: [{"bbox": [0, 0, 10, 10], "area_px": 80}])
    detection = hub.detect_objects(black)
    assert detection["score_kind"] == "foreground_area_fraction"
    assert detection["detections"][0]["area_fraction"] == pytest.approx(80 / 1024, abs=1e-6)
    assert "confidence" not in detection["detections"][0]


def test_carbon_disabled_and_gpu_attribution() -> None:
    with CarbonTracker("sample", enabled=False) as off:
        pass
    assert off.report is not None
    metadata = ModelMetadata(name="example", model_family="test", framework="test", task_type=TaskType.REGRESSION)
    metadata.carbon_report = off.report
    card = ModelCardGenerator.build_model_card(metadata).to_markdown()
    assert "Unmeasured (tracking disabled)" in card and "100.0/100" not in card
    assert off.report.metadata["energy_source"] == "unmeasured"
    with CarbonTracker("cpu_only") as cpu:
        pass
    assert cpu.report is not None
    assert cpu.report.gpu_power_watts == 0
    assert cpu.report.metadata["gpu_attribution"] == "unavailable"
    with CarbonTracker("gpu", gpu_power_reader=lambda: 42.5) as gpu:
        pass
    assert gpu.report is not None
    assert gpu.report.gpu_power_watts == 42.5
    assert gpu.report.metadata["gpu_attribution"] == "workload_telemetry"


def test_carbon_uses_backend_energy_with_independent_provenance() -> None:
    class Tracker:
        final_emissions_data = SimpleNamespace(energy_consumed=0.125)

        def stop(self) -> float:
            return 0.0

    tracker = CarbonTracker("backend", grid_intensity_g_per_kwh=700)
    tracker.start()
    tracker._cc_tracker = Tracker()
    report = tracker.stop()
    assert report.energy_kwh == 0.125
    assert report.emissions_g_co2eq == 0
    assert report.metadata["emissions_source"] == report.metadata["energy_source"] == "codecarbon"
