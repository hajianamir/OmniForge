"""Audit regressions for integrity, documentation, and residual contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from omniforge.api.auth import get_api_key
from omniforge.api.main import create_app
from omniforge.data.cleaning import robust_mahalanobis_outliers
from omniforge.eda.diagnostics import UniversalDiagnosticsSuite
from omniforge.models.baselines import ForecastingBaseline
from omniforge.ui.state import lock_leakage_safe_split


def test_mcd_requires_complete_case_sample_well_above_dimension() -> None:
    frame = pd.DataFrame(np.random.default_rng(42).normal(size=(25, 4)))
    frame.iloc[0, 0] = np.nan
    result = robust_mahalanobis_outliers(frame)
    assert not result['evaluable']
    assert result['n_scanned'] == 24 and result['sampled_from'] == 25


def test_studio03_documentation_matches_routes() -> None:
    reference = Path('docs/api_reference.md').read_text()
    assert '`/s03/eda`' not in reference
    assert '`/s03/run`' in reference
    assert '`/s03/catalog`' in reference


def test_naive_forecast_does_not_claim_conformal_coverage() -> None:
    doc = ForecastingBaseline.predict.__doc__ or ''
    assert 'random-walk' in doc
    assert 'conformal-like' not in doc.lower()


def test_split_lock_does_not_pass_with_exact_cross_partition_duplicates() -> None:
    frame = pd.DataFrame({'predictor': [10] * 40, 'target': [3] * 40})
    audit = lock_leakage_safe_split({}, frame, strategy='random', target_col='target')['audit_report']
    assert audit['exact_duplicate_leak_count'] > 0
    assert not audit['passed']


def test_request_validation_has_safe_envelope() -> None:
    client = TestClient(create_app(), headers={'X-API-Key': get_api_key()})
    response = client.post('/api/workbench/s03/run', json={'mode': 'never-echo-this-sentinel'})
    assert response.status_code == 422
    assert response.json()['error_code'] == 'REQUEST_VALIDATION_ERROR'
    assert response.json()['trace_id']
    assert 'never-echo-this-sentinel' not in response.text


def test_risk_composite_is_explicitly_an_uncalibrated_screen() -> None:
    frame = pd.DataFrame({'x': np.arange(60), 'y': np.arange(60)**2})
    report = UniversalDiagnosticsSuite().run(frame, selected_modules=['risk_uncertainty_audit'])
    assert report.risk_assessment is not None
    assert report.risk_assessment['score_kind'] == 'uncalibrated_screening_heuristic'
    assert report.risk_assessment['component_caps']['outlier_tail_risk'] == 25


def test_operator_docs_distinguish_se_methods_and_sdk_only_controls() -> None:
    guide = Path('docs/user_guide.md').read_text()
    register = Path('docs/RISK_REGISTER.md').read_text()
    assert 'SATE variance floor' in guide
    assert 'sd(psi, ddof=1) / sqrt(n)' in guide
    for doc in (guide, register):
        assert 'implemented, not yet exposed' in doc
        assert 'Venn–Abers' in doc
    chapter = Path('frontend/src/data/handbookChaptersPart1.ts').read_text()
    assert 'stable row_id identity' not in chapter
    assert 'zero-based row positions' in chapter


def test_pinned_logistic_paths_do_not_have_deprecated_penalty_fallbacks() -> None:
    for filename in ('src/omniforge/models/baselines.py', 'src/omniforge/features/selection.py'):
        source = Path(filename).read_text()
        assert 'penalty="l1"' not in source


def test_sklearn_category_cap_includes_the_infrequent_bucket() -> None:
    from omniforge.automl.backends import _build_preprocessor

    X = pd.DataFrame({'machine': [f'm{i}' for i in range(45)]})
    preprocessor = _build_preprocessor(X).fit(X)
    encoded = preprocessor.transform(X)
    assert encoded.shape == (45, 32)
    encoder = preprocessor.named_transformers_['cat'].named_steps['ohe']
    assert len(encoder.infrequent_categories_[0]) == 14


def test_benford_first_digits_agree_across_decimal_and_scientific_notation() -> None:
    from omniforge.data.cleaning import benford_digit_test

    digits = pd.Series(np.repeat(np.arange(1, 10, dtype=float) + 0.25, np.arange(1, 10)))
    assert benford_digit_test(digits) == benford_digit_test(digits * 1e-210) == benford_digit_test(digits * 1e208)
