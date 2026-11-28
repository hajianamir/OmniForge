"""Evidence must not silently degrade when baseline fitting fails."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omniforge.eda.diagnostics import UniversalDiagnosticsReport, UniversalDiagnosticsSuite
from omniforge.models.baselines import DummyRegressorBaseline, RidgeLinearBaseline
from omniforge.reporting.narrative import NarrativeGenerator


def _failed_fit(*args: object, **kwargs: object) -> None:
    raise RuntimeError('sensitive estimator input')


def _diagnostics() -> UniversalDiagnosticsReport:
    x = np.linspace(-2, 2, 80)
    frame = pd.DataFrame({'x': x, 'target': x**2 + x})
    return UniversalDiagnosticsSuite().run(frame, target_cols=['target'], selected_modules=['baseline_benchmark'])


def test_strong_baseline_failure_is_observable_and_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RidgeLinearBaseline, 'fit', _failed_fit)
    report = _diagnostics()
    failure = report.baseline_evidence['target']['Baseline_RidgeLinear']
    assert failure['status'] == 'failed'
    assert 'sensitive' not in failure['error']
    assert report.baseline_evidence['target']['Baseline_mean']['status'] == 'scored'


def test_naive_floor_unavailable_without_fitted_dummies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DummyRegressorBaseline, 'fit', _failed_fit)
    report = _diagnostics()
    equations = report.baseline_equations['target']
    assert equations
    for row in equations:
        assert row['naive_floor_status'] == 'unavailable'
        assert row['naive_floor_score'] is None
        assert row['lift_over_naive'] is None
        assert 'ACCEPTED' not in row['acceptance_status']


def test_narrative_consumes_the_risk_payload_keys() -> None:
    result = NarrativeGenerator().generate_executive_narrative(
        title='Risk', selected_sections=['decision_intelligence'],
        decision_summary={'structural_risk_index': 32.25, 'risk_band': 'Moderate'},
    )
    assert '32.25' in result['sections'][0]['body']
    assert 'Moderate' in result['sections'][0]['body']


def test_unavailable_module_is_not_attested() -> None:
    report = UniversalDiagnosticsSuite().run(
        pd.DataFrame({'label': ['a', 'b']}), selected_modules=['normality_distribution'],
    )
    assert not report.executed_modules
    assert report.module_status['normality_distribution'] == 'unavailable'


def test_reference_failure_blocks_native_automl_lift(monkeypatch: pytest.MonkeyPatch) -> None:
    from omniforge.automl.pipeline_search import AutoMLPipelineSearch

    monkeypatch.setattr(RidgeLinearBaseline, 'fit', _failed_fit)
    x = np.linspace(-2, 2, 100)
    result = AutoMLPipelineSearch(time_budget_seconds=10, cv_folds=2).fit(pd.DataFrame({'x': x}), pd.Series(x**2 + x))
    assert result.lift_over_best_baseline is None
    evidence = result.metadata['evaluation']['baseline_evidence']['test']
    assert evidence['Baseline_RidgeLinear']['status'] == 'failed'
    assert evidence['Baseline_mean']['status'] == 'scored'
