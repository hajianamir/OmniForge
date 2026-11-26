"""Unit tests for Automated EDA, Feature Engineering, Feature Selection & Embeddings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from omniforge.data.cleaning import DataCleaner
from omniforge.eda.automated_eda import AutomatedEDA
from omniforge.eda.diagnostics import (
    DIAGNOSTIC_TEST_CATALOG,
    StatisticalCriteriaConfig,
    UniversalDiagnosticsSuite,
)
from omniforge.eda.statistics import StatisticalAnalyzer
from omniforge.features.embeddings import EmbeddingGenerator
from omniforge.features.engineering import FeatureConfig, FeatureEngineeringPipeline
from omniforge.features.selection import FeatureSelector


def test_automated_eda_and_statistics(classification_df: pd.DataFrame, workspace_tmp: Path) -> None:
    cleaned, _ = DataCleaner().fit_transform(classification_df)
    rep = AutomatedEDA(theme="dark").run(cleaned, target_column="target", generate_figures=True)
    assert rep.profile.n_rows == len(cleaned)
    assert len(rep.figures) > 0
    html_p = workspace_tmp / "eda.html"
    rep.to_html(html_p)
    assert html_p.exists() and html_p.stat().st_size > 500

    ttest = StatisticalAnalyzer.two_sample_test(
        cleaned.loc[cleaned["target"] == 0, "feature_0"],
        cleaned.loc[cleaned["target"] == 1, "feature_0"],
    )
    assert 0.0 <= ttest.p_value <= 1.0


def test_universal_diagnostics_criteria_baselines_and_risk(classification_df: pd.DataFrame) -> None:
    """Verify criteria-driven statistical testing, multiple-testing corrections, baselines, and risk audit."""
    cleaned, _ = DataCleaner().fit_transform(classification_df)

    # Verify individual new hypothesis tests
    one_s = StatisticalAnalyzer.one_sample_t_test(cleaned["feature_0"], popmean=0.0, alpha=0.01)
    assert one_s.test_name == "one_sample_t_test" and one_s.ci_low is not None
    lev = StatisticalAnalyzer.levene_variance_test(cleaned, "feature_0", "target", alpha=0.05)
    assert lev.test_name == "levene_variance_test"
    corr = StatisticalAnalyzer.correlation_test(cleaned["feature_0"], cleaned["feature_1"], method="spearman", alpha=0.05)
    assert corr.test_name == "spearman_correlation"
    ad = StatisticalAnalyzer.normality_test(cleaned["feature_0"], alpha=0.05, method="anderson_darling")
    assert ad.test_name == "anderson_darling"
    lb = StatisticalAnalyzer.ljung_box_autocorrelation_test(cleaned["feature_0"], alpha=0.05)
    assert lb.test_name == "ljung_box_autocorrelation"
    adj_bonf = StatisticalAnalyzer.adjust_p_values([0.01, 0.04, 0.20], method="bonferroni")
    assert adj_bonf == pytest.approx([0.03, 0.12, 0.6])
    adj_fdr = StatisticalAnalyzer.adjust_p_values([0.01, 0.04, 0.20], method="fdr_bh")
    assert adj_fdr[0] <= adj_fdr[1] <= adj_fdr[2]

    # Auto recommendation engine
    rec_mods = UniversalDiagnosticsSuite.recommend_modules(cleaned, target_cols=["target"])
    assert "data_quality_integrity" in rec_mods
    assert "baseline_benchmark" in rec_mods
    assert "risk_uncertainty_audit" in rec_mods

    # Run full battery with strict alpha=0.01 & FDR correction + baselines + risk audit
    criteria = StatisticalCriteriaConfig(
        alpha=0.01,
        confidence_level=0.99,
        multiple_testing_correction="fdr_bh",
        parametric_mode="auto",
        risk_tail_alpha=0.01,
    )
    all_modules = [*DIAGNOSTIC_TEST_CATALOG.keys(), "baseline_benchmark", "risk_uncertainty_audit"]
    diag = UniversalDiagnosticsSuite().run(
        cleaned,
        target_cols=["target"],
        selected_modules=all_modules,
        criteria=criteria,
    )
    assert diag.total_tests > 0
    assert diag.quality_report is not None
    assert len(diag.distribution_profiles) > 0
    assert len(diag.outlier_profiles) > 0
    assert len(diag.vif_scores) > 0
    assert "target" in diag.baseline_benchmarks
    assert "Baseline_LogisticRegression" in diag.baseline_benchmarks["target"]
    assert diag.risk_assessment is not None
    assert 0.0 <= diag.risk_assessment["structural_risk_index"] <= 100.0
    assert len(diag.risk_assessment["tail_risk_table"]) > 0
    assert len(diag.risk_assessment["target_uncertainty"]) == 1


def test_feature_engineering_selection_and_embeddings(classification_df: pd.DataFrame) -> None:
    cleaned, _ = DataCleaner().fit_transform(classification_df)
    X = cleaned.drop(columns=["target"])
    y = cleaned["target"]

    fe = FeatureEngineeringPipeline(
        FeatureConfig(create_interactions=True, max_interaction_features=4, scale_numeric="standard")
    )
    X_train_t = fe.fit_transform(X.iloc[:160], y.iloc[:160])
    X_val_t = fe.transform(X.iloc[160:])
    assert list(X_train_t.columns) == list(X_val_t.columns)
    assert X_train_t.shape[1] >= X.shape[1]

    selector = FeatureSelector(method="mutual_info", max_features=6)
    X_sel = selector.fit_transform(X_train_t, y.iloc[:160])
    assert X_sel.shape[1] <= 6
    assert len(selector.report_.selected_features) == X_sel.shape[1]

    emb = EmbeddingGenerator(embedding_dim=4).fit_transform_text(
        [
            "credit risk default analysis",
            "portfolio return optimization",
            "energy load forecast model",
            "clinical patient survival outcome",
            "network intrusion anomaly alert",
        ]
    )
    assert emb.shape[0] == 5 and emb.shape[1] <= 4
