"""Audit regressions for diagnostics and report contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

from omniforge.api.workbench import _infer_task
from omniforge.automl.pipeline_search import AutoMLPipelineSearch
from omniforge.core.types import TaskType
from omniforge.data.splitting import LeakageSafeSplitter
from omniforge.eda.diagnostics import UniversalDiagnosticsSuite
from omniforge.eda.statistics import StatisticalAnalyzer


def test_diagnostics_discloses_sample_and_cap() -> None:
    frame = pd.DataFrame(np.arange(25_001 * 31).reshape(-1, 31), columns=[f'x{i}' for i in range(31)])
    report = UniversalDiagnosticsSuite().run(frame, selected_modules=['outlier_anomaly_scan'])
    assert report.n_scanned == 25_000
    assert report.sampled_from == 25_001
    assert report.columns_skipped['outlier_anomaly_scan'] == ['x30']


def test_task_inference_agrees_for_integer_targets_with_eleven_levels() -> None:
    target = pd.Series(np.tile(np.arange(11), 5))
    assert _infer_task(target) == AutoMLPipelineSearch.infer_task_type(target) == TaskType.REGRESSION
    assert not UniversalDiagnosticsSuite._is_classification_series(target)
    assert LeakageSafeSplitter().resolve_strategy(pd.DataFrame({'y': target}), 'y') == 'random'


def test_fdr_family_is_identified_for_each_ledger_test() -> None:
    df = pd.DataFrame({'a': np.arange(60), 'b': np.arange(60)**2})
    report = UniversalDiagnosticsSuite().run(df, selected_modules=['normality_distribution', 'correlation_association'])
    assert report.hypothesis_ledger
    assert all(row['correction_family'] == 'studio03_global_evaluable_ledger' for row in report.hypothesis_ledger)


def test_cramers_v_uses_uncorrected_pearson_statistic() -> None:
    table = np.array([[22, 10], [10, 22]])
    a = pd.Series(np.repeat([0, 0, 1, 1], table.ravel()))
    b = pd.Series(np.repeat([0, 1, 0, 1], table.ravel()))
    result = StatisticalAnalyzer.chi_square_independence(a, b)
    expected = np.sqrt(chi2_contingency(table, correction=False).statistic / table.sum())
    assert result.effect_size == pytest.approx(expected, abs=0.00005)


def test_numeric_workbench_uses_bounded_nominal_encoding() -> None:
    from omniforge.features.inference import WorkbenchInputPreprocessor

    frame = pd.DataFrame({'category': [f'level{i:02}' for i in range(45)]})
    pre = WorkbenchInputPreprocessor(mode='numeric').fit(frame)
    encoded = pre.transform(frame)
    assert encoded.shape[1] == 32
    assert 'category_code' not in encoded
    assert (encoded.sum(axis=1) == 1).all()
    assert pre.transform(pd.DataFrame({'category': ['unseen']})).to_numpy().sum() == 0


def test_legacy_tabular_preparation_does_not_drop_thirteenth_category() -> None:
    from omniforge.api.workbench import _prepare_xy

    frame = pd.DataFrame({f'cat{i}': ['a', 'b'] * 10 for i in range(13)})
    frame['target'] = np.arange(20, dtype=float)
    X, _, _ = _prepare_xy(frame, 'target')
    assert 'cat12_a' in X.columns


def test_legacy_cpcv_delegates_label_interval_purge() -> None:
    from omniforge.data.splitting import CombinatorialPurgedKFold as LegacyCPCV
    from omniforge.evaluation.validation import CombinatorialPurgedKFold

    X = pd.DataFrame({'x': np.arange(60)})
    eval_times = np.arange(60) + 2
    legacy = LegacyCPCV(n_splits=5, n_test_splits=2, embargo_pct=0.05)
    canonical = CombinatorialPurgedKFold(n_splits=5, n_test_groups=2, purge_gap=0, embargo_gap=0, embargo_pct=0.05)
    lhs = legacy.split(X, eval_times=eval_times)
    rhs = list(canonical.split(X, eval_times=eval_times))
    assert len(lhs) == len(rhs) == 10
    for (left_train, left_test), (right_train, right_test) in zip(lhs, rhs, strict=True):
        np.testing.assert_array_equal(left_train, right_train)
        np.testing.assert_array_equal(left_test, right_test)


def test_shapiro_nested_sample_is_disclosed_on_ledger_row() -> None:
    from omniforge.eda.diagnostics import StatisticalCriteriaConfig

    frame = pd.DataFrame({'x': np.random.default_rng(12).normal(size=5_100)})
    report = UniversalDiagnosticsSuite().run(
        frame, selected_modules=['normality_distribution'],
        criteria=StatisticalCriteriaConfig(normality_method='shapiro_wilk', bootstrap_samples=50),
    )
    assert report.n_scanned == 5_100
    assert report.hypothesis_ledger[0]['n_scanned'] == 5_000
    assert report.hypothesis_ledger[0]['sampled_from'] == 5_100
