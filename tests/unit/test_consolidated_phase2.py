"""Regressions for split-aware cleaning, Studio statistics, and model evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from omniforge.api.studios.s04_stats import _run_single_stat_test
from omniforge.data.cleaning import CleaningConfig, DataCleaner
from omniforge.domains import default_domain_registry
from omniforge.eda.statistics import StatisticalAnalyzer
from omniforge.models.clustering.clusterer import ClusteringEngine
from omniforge.models.computer_vision import foundation
from omniforge.pipelines.base import PipelineContext
from omniforge.pipelines.builder import _builtin_step_handlers
from omniforge.reporting.live_charts import _build_feature_correlation_forest


def test_temporal_cleaner_prevents_lookahead_and_preserves_labels() -> None:
    train = pd.DataFrame({
        "time": [4, 1, 3, 2], "value": [40.0, np.nan, np.nan, 20.0], "target": [1.0, np.nan, 3, 4],
    }, index=[8, 8, 9, 10])
    holdout = pd.DataFrame({"time": [5, 3, 6, 7], "value": [np.nan, np.nan, 60.0, np.nan], "target": [0, 1, 2, 3]})
    cleaner = DataCleaner(CleaningConfig(
        time_column="time", numeric_imputation="bfill", constant_fill_numeric=-1,
        exclude_columns=["target"], outlier_action="none", drop_exact_duplicates=False,
    ))
    clean_train, report = cleaner.fit_transform(train)
    clean_holdout, _ = cleaner.transform(holdout)
    assert clean_train["value"].tolist() == [40, -1, 20, 20]
    assert clean_holdout["value"].tolist() == [40, -1, 60, 60]
    pd.testing.assert_series_equal(clean_train["target"], train["target"])
    assert report.effective_numeric_imputation == "causal_ffill"


def test_declarative_cleaning_fits_only_split_training_data() -> None:
    handlers = _builtin_step_handlers()
    data = pd.DataFrame({"value": np.arange(20, dtype=float), "target": np.arange(20, dtype=float)})
    ctx = PipelineContext()
    ctx.set("raw_data", data)
    handlers["split"](ctx, target_column="target", strategy="time_series", time_col="target", val_size=0.2)
    train = ctx.get("train_data").copy()
    validation = ctx.get("val_data").copy()
    test = ctx.get("test_data").copy()
    validation["value"] = np.nan
    test["value"] = 1e9
    ctx.set("val_data", validation)
    ctx.set("test_data", test)
    handlers["clean"](ctx, outlier_action="none", drop_high_missing_cols=False, drop_exact_duplicates=False)
    assert ctx.get("preprocessing_fit_scope") == "train_only"
    np.testing.assert_allclose(ctx.get("val_data")["value"], train["value"].median())
    pd.testing.assert_series_equal(ctx.get("val_data")["target"], validation["target"])


def test_valid_hypotheses_match_statsmodels_without_counting_missing_results() -> None:
    p = [0.01666666665, np.nan, 0.045, 0.8, np.inf]
    adjusted = StatisticalAnalyzer.adjust_p_values(p)
    expected = multipletests([p[0], p[2], p[3]], method="fdr_bh")[1]
    np.testing.assert_allclose(np.take(adjusted, [0, 2, 3]), expected, rtol=1e-14)
    assert np.isnan(adjusted[1]) and np.isnan(adjusted[4])


def test_independent_ks_does_not_require_overlapping_rows() -> None:
    a, b = np.arange(6), np.arange(7) + 10
    frame = pd.DataFrame({"a": [*a, *([np.nan] * len(b))], "b": [*([np.nan] * len(a)), *b]})
    observed = _run_single_stat_test(frame, "ks_2samp", "a", "b", 0.05)
    expected = stats.ks_2samp(a, b)
    assert observed["sample_sizes"] == [6, 7]
    assert observed["p_value"] == expected.pvalue
    with pytest.raises(HTTPException, match="paired numeric rows"):
        _run_single_stat_test(frame, "paired_t", "a", "b", 0.05)


def test_clustering_reuses_training_schema_for_unknown_categories_and_missing_values() -> None:
    train = pd.DataFrame({"x": [0.0, 0.5, np.nan, 10, 11, 12], "group": ["a", "a", "a", "b", "b", "b"]})
    test = pd.DataFrame({"x": [1.0, np.nan, 11.0], "group": ["unseen", "b", "a"]})
    model = ClusteringEngine(algorithm="kmeans", n_clusters=2)
    _, labels = model.fit_predict_split(train, test)
    first = model.predict(test.iloc[[0]])[0]
    assert labels.tolist() == model.predict(test[["group", "x"]]).tolist()
    assert first == labels[0]
    assert model._col_medians_["x"] == train["x"].median()


def test_hopkins_matches_independent_pairwise_distance_calculation() -> None:
    data = np.random.default_rng(23).normal(size=(40, 3))
    sample_size, seed = 7, 11
    result = ClusteringEngine.hopkins_statistic(pd.DataFrame(data), sample_size=sample_size, random_state=seed)
    standardized = StandardScaler().fit_transform(data)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(data), sample_size, replace=False)
    uniform = rng.uniform(standardized.min(axis=0), standardized.max(axis=0), size=(sample_size, 3))
    artificial = np.min(cdist(uniform, standardized), axis=1)
    observed = cdist(standardized[sample], standardized)
    observed[np.arange(sample_size), sample] = np.inf
    u, w = np.sum(artificial**3), np.sum(np.min(observed, axis=1)**3)
    assert result["hopkins"] == pytest.approx(u / (u + w))
    assert ClusteringEngine.hopkins_statistic(pd.DataFrame(np.zeros((40, 2))))["hopkins"] is None


def test_correlation_chart_preserves_the_scipy_estimate_and_interval() -> None:
    x = np.arange(15, dtype=float)
    frame = pd.DataFrame({"reference": x, "identical": x, "other": x**2})
    chart = _build_feature_correlation_forest(frame, "reference", str)
    assert "Approximate Fisher-z" in chart["subtitle"]
    for row in chart["data"]["rows"]:
        result = stats.pearsonr(x, frame[row["label"]])
        interval = result.confidence_interval()
        assert (row["estimate"], row["ci_low"], row["ci_high"], row["p_value"]) == (
            result.statistic, interval.low, interval.high, result.pvalue,
        )


def test_vision_cv_scale_is_fitted_per_fold_and_then_on_all_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    class RecordingScaler(StandardScaler):
        def fit(self, X, y=None, sample_weight=None):
            rows_seen.append(len(X))
            return super().fit(X, y, sample_weight=sample_weight)

    rows_seen: list[int] = []
    monkeypatch.setattr(foundation, "StandardScaler", RecordingScaler)
    hub = foundation.VisionFoundationHub()
    monkeypatch.setattr(hub, "extract_foundation_embedding", lambda image, **_: {
        "backend": "test_fixture_embedding", "embedding": image,
    })
    features = [np.array([i, i**2]) for i in range(12)]
    result = hub.train_transfer_classifier(features, [0] * 6 + [1] * 6)
    assert rows_seen == [8, 8, 8, 12]
    assert result["backend"] == "test_fixture_embedding"
    assert result["cv_accuracy"] is not None


@pytest.mark.parametrize(("domain", "excluded", "descriptive_kpi"), [
    ("energy", {"net_load_mw"}, "mean_net_load_mw"),
    ("materials", {"specific_strength"}, "specific_strength_ratio"),
    ("real_estate", {"gross_rent_yield_pct", "price_per_sqm"}, "median_price_per_sqm"),
])
def test_default_target_derived_domain_indicators_remain_descriptive(domain, excluded, descriptive_kpi) -> None:
    plugin = default_domain_registry.get(domain)
    sample = plugin.generate_sample_dataset(30)
    engineered = plugin.engineer_domain_features(sample)
    assert excluded.isdisjoint(engineered.columns)
    assert np.isfinite(plugin.compute_domain_kpis(sample)[descriptive_kpi])
