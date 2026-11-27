"""Regression tests for audit sampling, statistical, and documentation contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omniforge.api.studios.schemas import AutoMLRunReq, DeepBenchReq, DiagnosticsReq, NonParametricRunReq
from omniforge.eda.diagnostics import UniversalDiagnosticsSuite


@pytest.mark.parametrize('schema', [AutoMLRunReq, DeepBenchReq, DiagnosticsReq, NonParametricRunReq])
def test_model_requests_accept_an_ignored_column_scope(schema: type) -> None:
    assert schema(ignored_columns=['sensor']).ignored_columns == ['sensor']


def test_diagnostics_scopes_only_baseline_inputs_and_drops_train_constants() -> None:
    x = np.linspace(-1, 1, 80)
    df = pd.DataFrame({'x': x, 'ignored': x**3, 'constant': 1, 'y': x**2 + x})
    report = UniversalDiagnosticsSuite().run(
        df, target_cols=['y'], selected_modules=['baseline_benchmark', 'outlier_anomaly_scan'],
        ignored_columns=['ignored'], baseline_partitions=(df.iloc[:60], df.iloc[60:]),
    )
    assert report.baseline_feature_cols == {'y': ['x']}
    assert 'ignored' in [row['Column'] for row in report.outlier_profiles]


def test_sparse_baseline_declares_selected_support() -> None:
    x = np.linspace(-2, 2, 125)
    frame = pd.DataFrame({'signal': x, 'noise': np.random.default_rng(11).normal(size=len(x)), 'target': x * 2})
    report = UniversalDiagnosticsSuite().run(frame, target_cols=['target'], selected_modules=['baseline_benchmark'])
    sparse = next(row for row in report.baseline_equations['target'] if 'Lasso' in row['model_name'])
    assert sparse['reference_kind'] == 'parsimonious_l1'
    assert 'signal' in sparse['selected_support']
    assert sparse['n_features'] == len(sparse['selected_support'])


def test_tiny_holdout_does_not_certify_a_baseline_winner() -> None:
    x = np.arange(80, dtype=float)
    frame = pd.DataFrame({'x': x, 'target': x + 0.1})
    report = UniversalDiagnosticsSuite().run(
        frame, target_cols=['target'], selected_modules=['baseline_benchmark'],
        baseline_partitions=(frame.iloc[:-2], frame.iloc[-2:]),
    )
    rows = report.baseline_equations['target']
    assert any(row['lift_over_naive'] > 0 for row in rows)
    assert all('ACCEPTED' not in row['acceptance_status'] for row in rows)
    assert rows[0]['lift_comparison']['status'] == 'unavailable'


def test_paired_bootstrap_preserves_reference_pairing() -> None:
    from omniforge.evaluation.paired_comparison import paired_score_interval

    y = np.linspace(-1, 1, 40)
    result = paired_score_interval(y, y, y + 1, metric='rmse', split_strategy='random', comparisons=4, reference_name='reference')
    assert result['n_pairs'] == 40
    assert result['ci_lower'] == pytest.approx(1)
    assert result['ci_upper'] == pytest.approx(1)
    assert result['supports_improvement'] is True
