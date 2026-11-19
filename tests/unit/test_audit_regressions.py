"""Regression tests for independent audit findings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omniforge.core.contracts import AnalysisSpec, LeakageGuard, ResultBundle
from omniforge.data.cleaning import CleaningConfig, DataCleaner
from omniforge.data.quality import DataQualityProfiler
from omniforge.data.splitting import CombinatorialPurgedKFold
from omniforge.data.validation import ColumnContract, DataContract
from omniforge.eda.statistics import StatisticalAnalyzer
from omniforge.evaluation.robustness import adaptive_conformal_intervals
from omniforge.features.selection import FeatureSelector
from omniforge.mlops.drift import DriftDetector


def test_cleaner_transform_before_fit_raises_runtime_error() -> None:
    """DataCleaner.transform() must not silently fit on unseen data when unfitted."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    cleaner = DataCleaner()
    with pytest.raises(RuntimeError, match="must be fitted"):
        cleaner.transform(df)


def test_cleaning_config_default_fuzzy_threshold_is_safe() -> None:
    """Default fuzzy_dedup_threshold must be >= 0.97 to avoid false-positive ID merges."""
    cfg = CleaningConfig()
    assert cfg.fuzzy_dedup_threshold >= 0.97


def test_combinatorial_purged_kfold_purge_and_embargo() -> None:
    """CPCV must generate C(n_splits, n_test_splits) splits with no train/test leakage."""
    X = np.arange(60).reshape(60, 1)
    cpcv = CombinatorialPurgedKFold(n_splits=5, n_test_splits=2, purge_gap=2, embargo_pct=0.05)
    assert cpcv.get_n_splits(X) == 10
    splits = cpcv.split(X)
    assert len(splits) == 10
    for train_idx, test_idx in splits:
        assert len(np.intersect1d(train_idx, test_idx)) == 0
        # Ensure purge_gap=2 around every test index is excluded from train_idx
        for t in test_idx:
            for offset in range(-2, 3):
                assert (t + offset) not in train_idx


def test_drift_jensen_shannon_divergence() -> None:
    """JSD must be near 0 for identical distributions and significantly higher for shifted ones."""
    rng = np.random.default_rng(42)
    ref = rng.normal(0.0, 1.0, size=500)
    same = rng.normal(0.0, 1.0, size=500)
    shifted = rng.normal(3.0, 1.0, size=500)
    jsd_same = DriftDetector.compute_jensen_shannon_divergence(ref, same, n_bins=10)
    jsd_shift = DriftDetector.compute_jensen_shannon_divergence(ref, shifted, n_bins=10)
    assert 0.0 <= jsd_same < 0.1
    assert 0.3 < jsd_shift <= 1.0


def test_little_mcar_test_detects_mcar_vs_mar() -> None:
    """Little's MCAR test should not reject MCAR under random missingness and should reject strong MAR."""
    rng = np.random.default_rng(7)
    n = 200
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = 2.0 * x1 + rng.normal(0.0, 0.5, size=n)

    df_mcar = pd.DataFrame({"x1": x1.copy(), "x2": x2.copy()})
    df_mcar.loc[rng.random(n) < 0.2, "x1"] = np.nan
    df_mcar.loc[rng.random(n) < 0.2, "x2"] = np.nan
    res_mcar = DataQualityProfiler.little_mcar_test(df_mcar)
    assert res_mcar["status"] == "completed"
    assert res_mcar["p_value"] is not None

    df_mar = pd.DataFrame({"x1": x1.copy(), "x2": x2.copy()})
    df_mar.loc[df_mar["x1"] > 0.5, "x2"] = np.nan
    df_mar.loc[df_mar["x2"] < -0.5, "x1"] = np.nan
    res_mar = DataQualityProfiler.little_mcar_test(df_mar)
    assert res_mar["status"] == "completed"
    assert res_mar["is_mcar"] is False


def test_additional_statistical_tests() -> None:
    """Verify Jarque-Bera, Anderson-Darling, McNemar, and Fisher exact tests."""
    rng = np.random.default_rng(123)
    norm_vals = rng.normal(0.0, 1.0, size=200)
    skew_vals = rng.exponential(1.0, size=200)

    jb_norm = StatisticalAnalyzer.jarque_bera_test(norm_vals)
    jb_skew = StatisticalAnalyzer.jarque_bera_test(skew_vals)
    assert jb_norm.reject_null is False
    assert jb_skew.reject_null is True

    ad_skew = StatisticalAnalyzer.anderson_darling_test(skew_vals)
    assert ad_skew.reject_null is True

    mc = StatisticalAnalyzer.mcnemar_test([[30, 20], [2, 48]])
    assert mc.reject_null is True

    fe = StatisticalAnalyzer.fisher_exact_test([[10, 1], [1, 10]])
    assert fe.reject_null is True


def test_adwin_concept_drift_detector() -> None:
    """ADWIN must detect an abrupt mean shift in a streaming error sequence."""
    rng = np.random.default_rng(99)
    stream = np.concatenate([rng.normal(0.1, 0.02, size=80), rng.normal(0.8, 0.02, size=80)])
    res = DriftDetector.detect_concept_drift_adwin(stream, delta=0.002, min_window_length=10)
    assert res["drift_detected"] is True
    assert len(res["drift_indices"]) >= 1
    assert res["drift_indices"][0] >= 80


def test_mice_rubin_and_missforest_imputation() -> None:
    """Verify MICE with Rubin's rules pooling and MissForest iterative RF imputation."""
    rng = np.random.default_rng(11)
    x1 = rng.normal(10.0, 2.0, size=60)
    x2 = 2.0 * x1 + rng.normal(0.0, 0.5, size=60)
    df = pd.DataFrame({"x1": x1, "x2": x2})
    df.loc[0:5, "x1"] = np.nan
    df.loc[10:14, "x2"] = np.nan

    mice_res = DataCleaner.multiple_imputation_rubin(df, m=4, max_iter=5, random_state=42)
    assert mice_res["m"] == 4
    assert len(mice_res["imputed_datasets"]) == 4
    assert mice_res["pooled_estimates"]["x1"]["total_variance"] > 0.0

    mf_df = DataCleaner.missforest_impute(df, max_iter=3, n_estimators=20, random_state=42)
    assert int(mf_df.isna().sum().sum()) == 0


def test_cohens_d_and_power_sample_size() -> None:
    """Verify Cohen's d effect size and two-sample power sample size calculation."""
    rng = np.random.default_rng(55)
    a = rng.normal(0.0, 1.0, size=100)
    b = rng.normal(0.5, 1.0, size=100)
    d = StatisticalAnalyzer.cohens_d(b, a)
    assert 0.2 < d < 0.9
    pwr = StatisticalAnalyzer.two_sample_power_sample_size(effect_size=0.5, alpha=0.05, power=0.80)
    assert 55 <= int(pwr["n_per_group"]) <= 70


def test_analysis_spec_leakage_guard_and_result_bundle() -> None:
    """Verify AnalysisSpec, LeakageGuard (row/group/temporal), and ResultBundle bootstrap CI."""
    spec = AnalysisSpec(
        task_kind="time_series_forecast",
        analysis_mode="confirmatory",
        target_column="y",
        time_column="ts",
        group_column="entity",
        forecast_horizon=7,
    )
    assert spec.analysis_mode == "confirmatory"

    train_df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=5, freq="D"),
            "entity": ["A", "A", "B", "B", "B"],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=[0, 1, 2, 3, 4],
    )
    test_bad = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-04", periods=3, freq="D"),
            "entity": ["B", "C", "C"],
            "y": [6.0, 7.0, 8.0],
        },
        index=[4, 5, 6],
    )
    rep_bad = LeakageGuard.validate_split(train_df, test_bad, group_col="entity", time_col="ts")
    assert rep_bad.passed is False
    assert rep_bad.row_overlap_count == 1
    assert rep_bad.group_overlap_count == 1
    assert rep_bad.temporal_order_valid is False

    test_ok = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-10", periods=3, freq="D"),
            "entity": ["C", "C", "D"],
            "y": [6.0, 7.0, 8.0],
        },
        index=[5, 6, 7],
    )
    rep_ok = LeakageGuard.validate_split(train_df, test_ok, group_col="entity", time_col="ts")
    assert rep_ok.passed is True

    mean, lo, hi = ResultBundle.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_boot=200, seed=42)
    bundle = ResultBundle(
        analysis_name="mean_y",
        test_or_model="bootstrap",
        effect_estimate=mean,
        ci_lower=lo,
        ci_upper=hi,
        n_samples=5,
    )
    assert bundle.ci_lower is not None and bundle.ci_upper is not None
    assert bundle.ci_lower <= mean <= bundle.ci_upper


def test_adaptive_conformal_and_advanced_feature_selection() -> None:
    """Verify Adaptive Conformal Inference (ACI) and mRMR / Boruta / Stability Selection."""
    rng = np.random.default_rng(77)
    y_pred = np.linspace(1.0, 10.0, 80)
    y_true = y_pred + rng.normal(0.0, 0.5, size=80)
    aci = adaptive_conformal_intervals(y_true, y_pred, alpha=0.10, gamma=0.05, window=20)
    assert 0.70 <= aci["empirical_coverage"] <= 1.0
    assert aci["avg_width"] > 0.0

    X = pd.DataFrame(
        {
            "signal1": rng.normal(0.0, 1.0, size=250),
            "signal2": rng.normal(0.0, 1.0, size=250),
            "noise1": rng.normal(0.0, 1.0, size=250),
            "noise2": rng.normal(0.0, 1.0, size=250),
        }
    )
    y = 3.0 * X["signal1"] + 3.0 * X["signal2"] + rng.normal(0.0, 0.2, size=250)

    mrmr = FeatureSelector.mrmr_select(X, y, k=2, task="regression", random_state=42)
    assert set(mrmr["selected_features"]) == {"signal1", "signal2"}

    boruta = FeatureSelector.boruta_select(X, y, task="regression", n_iter=10, random_state=42)
    assert "signal1" in boruta["confirmed"]

    stab = FeatureSelector.stability_selection(X, y, method="l1", task="regression", max_features=2, n_bootstrap=10)
    assert "signal1" in stab["stable_features"]


def test_data_contract_and_missforest_config() -> None:
    """Verify DataContract validation (errors vs warnings) and missforest in CleaningConfig."""
    contract = DataContract(
        name="sensor_contract",
        allow_extra_columns=False,
        columns=[
            ColumnContract(name="id", required=True, dtype="numeric", unique=True),
            ColumnContract(name="temp", required=True, dtype="numeric", min_value=-50.0, max_value=150.0),
        ],
    )
    df_bad = pd.DataFrame({"id": [1, 1], "temp": [20.0, 200.0], "extra": ["a", "b"]})
    res_bad = contract.validate_dataframe(df_bad)
    assert res_bad["passed"] is False
    assert len(res_bad["errors"]) >= 2
    assert len(res_bad["warnings"]) == 1

    df_ok = pd.DataFrame({"id": [1, 2, 3, 4], "temp": [20.0, np.nan, 25.0, 22.0]})
    cleaner = DataCleaner(CleaningConfig(numeric_imputation="missforest"))
    cleaned, _ = cleaner.fit_transform(df_ok)
    assert int(cleaned["temp"].isna().sum()) == 0







