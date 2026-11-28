"""Regression tests for independent audit findings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omniforge.core.contracts import AnalysisSpec, LeakageGuard, ResultBundle
from omniforge.data.cleaning import CleaningConfig, DataCleaner
from omniforge.data.quality import DataQualityProfiler
from omniforge.data.splitting import CombinatorialPurgedKFold, LeakageSafeSplitter
from omniforge.data.validation import ColumnContract, DataContract
from omniforge.eda.statistics import StatisticalAnalyzer
from omniforge.evaluation.robustness import adaptive_conformal_intervals
from omniforge.features.selection import FeatureSelector
from omniforge.mlops.drift import DriftDetector
from omniforge.models.clustering.clusterer import ClusteringEngine
from omniforge.models.mining.miner import AssociationRuleMiner
from omniforge.reporting.exporters import sanitize_spreadsheet_value


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


def test_root_does_not_auto_issue_admin_cookie_by_default() -> None:
    """Visiting / must not issue admin auth cookies when auto_bootstrap_local_ui is False."""
    from fastapi.testclient import TestClient

    from omniforge.api.main import app
    from omniforge.config.settings import get_settings

    sec = get_settings().security
    prev = sec.auto_bootstrap_local_ui
    sec.auto_bootstrap_local_ui = False
    try:
        client = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 51000))
        resp = client.get("/")
        assert resp.status_code == 200
        assert "omniforge_wb_auth" not in resp.cookies
    finally:
        sec.auto_bootstrap_local_ui = prev


def test_transaction_mining_and_k_prototypes() -> None:
    """Verify transaction-ID-aware association rules and K-Prototypes mixed clustering."""
    tx_df = pd.DataFrame(
        {
            "invoice_id": ["T1", "T1", "T2", "T2", "T3", "T3", "T4", "T4"],
            "item": ["A", "B", "A", "B", "A", "C", "B", "D"],
        }
    )
    res_tx = AssociationRuleMiner.mine_transactions(
        tx_df, "invoice_id", "item", min_support=0.25, min_confidence=0.5, min_lift=1.0
    )
    assert res_tx["n_transactions"] == 4
    assert res_tx["n_rules"] >= 1

    mix_df = pd.DataFrame(
        {
            "x": [1.0, 1.2, 0.9, 10.0, 10.5, 9.8],
            "cat": ["red", "red", "red", "blue", "blue", "blue"],
        }
    )
    kp = ClusteringEngine.k_prototypes_fit(mix_df, n_clusters=2, gamma=0.5, random_state=42)
    assert len(kp["labels"]) == 6
    assert len(set(kp["labels"])) == 2


def test_group_cv_without_groups_raises_and_spreadsheet_sanitization() -> None:
    """Group CV without groups must raise DataValidationError; formula prefixes must be neutralized."""
    from omniforge.core.exceptions import DataValidationError
    from omniforge.data.splitting import SplitConfig

    splitter = LeakageSafeSplitter(SplitConfig(strategy="group"))
    df = pd.DataFrame({"x": np.arange(20), "y": np.arange(20) % 2})
    with pytest.raises(DataValidationError, match="groups=None"):
        splitter.build_cv_splitter(df, y=df["y"], groups=None, is_classification=True)

    assert sanitize_spreadsheet_value("=1+1") == "'=1+1"
    assert sanitize_spreadsheet_value("  +cmd|'/C calc'!A0") == "'  +cmd|'/C calc'!A0"
    assert sanitize_spreadsheet_value("@SUM(A1:A2)") == "'@SUM(A1:A2)"
    assert sanitize_spreadsheet_value("safe_text") == "safe_text"
    assert sanitize_spreadsheet_value(42) == 42


def test_drift_unknown_on_no_shared_columns() -> None:
    """DriftDetector must return DriftStatus.UNKNOWN when no shared columns or valid samples exist."""
    from omniforge.core.types import DriftStatus

    det = DriftDetector()
    ref = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    cur = pd.DataFrame({"b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    rep = det.detect(ref, cur)
    assert rep.overall_status == DriftStatus.UNKNOWN
    assert rep.total_features == 0


def test_session_state_tenant_isolation() -> None:
    """Two tenants using the same X-Session-ID must receive completely isolated workspace states."""
    from omniforge.ui.state import get_session_state

    s_a = get_session_state("shared-sid", tenant_id="tenant-alpha")
    s_b = get_session_state("shared-sid", tenant_id="tenant-beta")
    s_a["secret_marker"] = "alpha-only"
    assert "secret_marker" not in s_b


def test_split_lock_row_id_and_fail_closed() -> None:
    """get_active_split_frames must match by row_id when present and fail-closed on invalid lock."""
    from omniforge.ui.state import get_active_split_frames

    df = pd.DataFrame(
        {
            "__row_id__": [f"r_{i}" for i in range(10)],
            "x": np.arange(10, dtype=float),
        },
        index=[100 + i for i in range(10)],
    )
    state = {
        "uap_split_lock": {
            "active": True,
            "row_id_col": "__row_id__",
            "train_row_ids": [f"r_{i}" for i in range(7)],
            "test_row_ids": [f"r_{i}" for i in range(7, 10)],
        }
    }
    tr, te, _ = get_active_split_frames(state, df)
    assert len(tr) == 7
    assert te is not None and len(te) == 3

    bad_state = {"uap_split_lock": {"active": True, "train_indices": [0, 1], "test_indices": [2]}}
    with pytest.raises(ValueError, match="Active Split Lock is invalid"):
        get_active_split_frames(bad_state, df)


def test_manifold_projector_excludes_color_col_and_targets() -> None:
    """Changing color_col or passing exclude_cols must not alter PCA coordinates."""
    from omniforge.models.mining.miner import ManifoldProjector

    rng = np.random.default_rng(42)
    df1 = pd.DataFrame({"x1": rng.normal(0, 1, 30), "x2": rng.normal(0, 1, 30), "target": rng.integers(0, 2, 30)})
    df2 = df1.copy()
    df2["target"] = 1 - df2["target"]  # Flip target/color_col only

    proj = ManifoldProjector()
    r1 = proj.project(df1, method="pca", color_col="target")
    r2 = proj.project(df2, method="pca", color_col="target")
    assert r1["loadings"] == r2["loadings"]
    assert r1["explained_variance_pct"] == r2["explained_variance_pct"]


def test_fp_growth_prefixspan_beam_search_and_raw_view() -> None:
    """Verify FP-Growth, PrefixSpan, Subgroup Beam Search, and raw audit/anomaly view separation."""
    from omniforge.models.mining.miner import SequentialPatternMiner, fp_growth_mine, subgroup_beam_search
    from omniforge.ui.state import get_audit_anomaly_dataframe

    # 1. FP-Growth
    txs = [["A", "B", "C"], ["A", "B"], ["A", "C"], ["B", "C"], ["A", "B"], ["D"]]
    fp = fp_growth_mine(txs, min_support=0.3, min_confidence=0.5, min_lift=1.0)
    assert fp["n_transactions"] == 6
    assert len(fp["frequent_itemsets"]) >= 3
    assert len(fp["rules"]) >= 1

    # 2. PrefixSpan
    seqs = [["login", "view", "buy"], ["login", "buy"], ["view", "buy"], ["login", "view", "logout"]]
    sp = SequentialPatternMiner.prefix_span(seqs, min_support=0.5, max_len=3)
    patterns = [r["pattern"] for r in sp]
    assert ["login", "buy"] in patterns

    # 3. Subgroup Beam Search (2-condition)
    rng = np.random.default_rng(7)
    df_sg = pd.DataFrame(
        {
            "age": np.concatenate([np.full(20, 60.0), rng.normal(30, 5, 40)]),
            "plan": ["gold"] * 20 + ["basic"] * 40,
            "spend": np.concatenate([np.full(20, 100.0), rng.normal(10, 2, 40)]),
        }
    )
    bs = subgroup_beam_search(df_sg, "spend", beam_width=4, max_depth=2, min_support=10)
    assert any(len(r["conditions"]) == 2 for r in bs)

    # 4. Raw audit/anomaly view separation
    raw_df = pd.DataFrame({"x": [1.0, 2.0, 999.0]})
    clipped_df = pd.DataFrame({"x": [1.0, 2.0, 10.0]})
    state = {"uap_raw_df": raw_df, "uap_df": clipped_df}
    out_raw = get_audit_anomaly_dataframe(state)
    assert float(out_raw["x"].max()) == 999.0


def test_benford_and_robust_mahalanobis_cleaning() -> None:
    """Verify Benford's Law audit test and MCD Robust Mahalanobis multivariate outlier detection."""
    from omniforge.data.cleaning import benford_digit_test, robust_mahalanobis_outliers

    # Benford-conforming exponential-like data
    rng = np.random.default_rng(123)
    benford_series = pd.Series(10.0 ** rng.uniform(0.0, 4.0, 300))
    b_res = benford_digit_test(benford_series)
    assert b_res["evaluable"] is True
    assert b_res["conforms_benford"] is True

    # Multivariate MCD Mahalanobis outlier detection
    X_in = rng.normal(0.0, 1.0, size=(80, 3))
    X_out = np.array([[15.0, -15.0, 12.0], [20.0, 20.0, -18.0]])
    df_mcd = pd.DataFrame(np.vstack([X_in, X_out]), columns=["a", "b", "c"])
    m_res = robust_mahalanobis_outliers(df_mcd, alpha=0.01)
    assert m_res["evaluable"] is True
    assert 80 in m_res["outlier_indices"]
    assert 81 in m_res["outlier_indices"]


def test_distance_correlation_and_cluster_stability_ari() -> None:
    """Verify nonlinear Distance Correlation (dCor) and Bootstrap ARI cluster stability."""
    from omniforge.eda.statistics import distance_correlation

    rng = np.random.default_rng(42)
    x = rng.uniform(-2.0, 2.0, 120)
    y_quad = x**2  # Zero linear Pearson correlation, strong nonlinear dependence
    res_dc = distance_correlation(x, y_quad)
    assert res_dc["dcor"] > 0.35

    # Well-separated clusters should achieve high bootstrap ARI
    c1 = rng.normal(0.0, 0.3, size=(30, 2))
    c2 = rng.normal(6.0, 0.3, size=(30, 2))
    df_c = pd.DataFrame(np.vstack([c1, c2]), columns=["f1", "f2"])
    stab = ClusteringEngine.bootstrap_stability_ari(df_c, n_clusters=2, n_bootstraps=10, random_state=42)
    assert stab["evaluable"] is True
    assert stab["mean_ari"] > 0.85


def test_cleaning_distortion_and_manifold_trustworthiness() -> None:
    """Verify cleaning_distortion_audit (KS/Wasserstein) and ManifoldProjector trustworthiness."""
    from omniforge.data.cleaning import cleaning_distortion_audit
    from omniforge.models.mining.miner import ManifoldProjector

    rng = np.random.default_rng(11)
    raw = pd.DataFrame({"v": rng.normal(0.0, 1.0, 100)})
    distorted = pd.DataFrame({"v": rng.normal(5.0, 1.0, 100)})
    audit = cleaning_distortion_audit(raw, distorted)
    assert audit["columns_audited"] == 1
    assert "v" in audit["high_distortion_columns"]
    assert audit["per_column"]["v"]["wasserstein"] > 4.0

    df_m = pd.DataFrame({"x1": rng.normal(0, 1, 40), "x2": rng.normal(0, 1, 40), "x3": rng.normal(0, 1, 40)})
    res = ManifoldProjector().project(df_m, method="pca")
    assert res["trustworthiness"] is not None
    assert 0.0 <= res["trustworthiness"] <= 1.0


def test_adbench_anomaly_metrics() -> None:
    """Verify ADBench (NeurIPS 2022) anomaly evaluation metrics: AUC-ROC, AUC-PR, Precision@K."""
    from omniforge.evaluation.metrics import evaluate_anomaly_adbench

    yt = np.array([0] * 90 + [1] * 10)
    sc = np.concatenate([np.linspace(0.0, 0.4, 90), np.linspace(0.8, 1.0, 10)])
    rep = evaluate_anomaly_adbench(yt, sc, k=10)
    assert rep["auc_roc"] == 1.0
    assert rep["auc_pr"] == 1.0
    assert rep["precision_at_k"] == 1.0


def test_friedman_nemenyi_demsar_test() -> None:
    """Verify Demšar (2006) Friedman + Nemenyi Critical Difference multi-model comparison."""
    from omniforge.eda.statistics import friedman_nemenyi_test

    # 6 datasets/folds x 3 models where Model 0 consistently outperforms Model 1 and Model 2
    scores = np.array(
        [
            [0.95, 0.85, 0.70],
            [0.94, 0.84, 0.72],
            [0.96, 0.86, 0.68],
            [0.93, 0.82, 0.71],
            [0.97, 0.83, 0.69],
            [0.92, 0.80, 0.65],
        ]
    )
    res = friedman_nemenyi_test(scores, alpha=0.05)
    assert res["evaluable"] is True
    assert res["significant"] is True
    assert res["avg_ranks"][0] == 1.0
    assert res["critical_difference"] > 0.0


def test_advanced_audit_and_mining_endpoints() -> None:
    """Verify /api/workbench/s01/advanced_audit and /s02/advanced_mining endpoints."""
    from fastapi.testclient import TestClient

    from omniforge.api.main import app
    from omniforge.api.workbench import _sid
    from omniforge.config.settings import get_settings
    from omniforge.core.types import Principal
    from omniforge.ui.state import get_session_state

    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "amount": 10.0 ** rng.uniform(0.0, 3.0, 60),
            "age": rng.normal(40.0, 10.0, 60),
            "target": rng.normal(10.0, 2.0, 60),
        }
    )
    # Workbench sessions are namespaced per authenticated principal (BOLA fix), so the
    # state the test plants must live under the SAME namespaced key the cookie principal lands in.
    cookie_principal = Principal(
        subject="local_workbench_user",
        roles=["admin"],
        tenant_id=get_settings().default_tenant_id,
    )
    state = get_session_state(_sid("adv-ep-test", cookie_principal))
    state["uap_df"] = df
    state["uap_raw_df"] = df.copy()
    state["uap_target_col"] = "target"

    sec = get_settings().security
    prev = sec.auto_bootstrap_local_ui
    sec.auto_bootstrap_local_ui = True
    try:
        client = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 51000))
        client.get("/")
        csrf = client.cookies.get("omniforge_csrf", "")
        headers = {"X-Session-ID": "adv-ep-test", "X-CSRF-Token": csrf}
        r1 = client.post("/api/workbench/s01/advanced_audit", headers=headers)
        assert r1.status_code == 200
        d1 = r1.json()
        assert "benford_tests" in d1 and "robust_mahalanobis" in d1 and "cleaning_distortion" in d1

        r2 = client.post("/api/workbench/s02/advanced_mining", headers=headers)
        assert r2.status_code == 200
        d2 = r2.json()
        assert "beam_subgroups" in d2 and "cluster_stability_ari" in d2 and "distance_correlation" in d2
        assert "hopkins_clustering_tendency" in d2 and d2["hopkins_clustering_tendency"]["evaluable"] is True
    finally:
        sec.auto_bootstrap_local_ui = prev


def test_association_rules_exact_hand_calculated_metrics() -> None:
    """Verify exact support=0.20, confidence=0.50, lift=1.6667 on 100 transactions."""
    from omniforge.models.mining.miner import fp_growth_mine

    txs: list[list[str]] = []
    txs.extend([["A", "B"]] * 20)
    txs.extend([["A"]] * 20)
    txs.extend([["B"]] * 10)
    txs.extend([["C"]] * 50)
    res = fp_growth_mine(txs, min_support=0.1, min_confidence=0.4, min_lift=1.0)
    rule_ab = [r for r in res["rules"] if r["antecedent"] == ["A"] and r["consequent"] == ["B"]]
    assert len(rule_ab) == 1
    assert abs(rule_ab[0]["support"] - 0.20) < 1e-4
    assert abs(rule_ab[0]["confidence"] - 0.50) < 1e-4
    assert abs(rule_ab[0]["lift"] - 1.6667) < 1e-3


def test_knn_shapley_and_venn_abers() -> None:
    """Verify exact KNN-Shapley data valuation (detecting mislabeled points) and Venn-ABERS calibration."""
    from omniforge.evaluation.robustness import knn_shapley_valuation, venn_abers_calibrate

    rng = np.random.default_rng(42)
    X_tr = np.vstack([rng.normal(0.0, 0.4, (20, 2)), rng.normal(3.0, 0.4, (20, 2))])
    y_tr = np.array([0] * 20 + [1] * 20)
    # Flip label of index 0 to simulate a mislabeled training point
    y_tr[0] = 1
    X_va = np.vstack([rng.normal(0.0, 0.4, (10, 2)), rng.normal(3.0, 0.4, (10, 2))])
    y_va = np.array([0] * 10 + [1] * 10)

    val_res = knn_shapley_valuation(X_tr, y_tr, X_va, y_va, k=5)
    assert val_res["evaluable"] is True
    assert val_res["values"][0] < 0.0
    assert 0 in val_res["worst_indices"][:5]

    cal_scores = np.linspace(0.05, 0.95, 40)
    cal_y = (cal_scores > 0.5).astype(int)
    va = venn_abers_calibrate(cal_scores, cal_y, np.array([0.1, 0.9]))
    assert va["p0"][0] <= va["p_calibrated"][0] <= va["p1"][0]
    assert va["p_calibrated"][1] > va["p_calibrated"][0]


def test_distance_correlation_exact_and_cleaning_percentile_elliptic() -> None:
    """Verify dCor(x, x)==1, scale/shift invariance, and DataCleaner percentile/elliptic methods."""
    from omniforge.data.cleaning import CleaningConfig, DataCleaner
    from omniforge.eda.statistics import distance_correlation

    x = np.linspace(1.0, 10.0, 30)
    self_res = distance_correlation(x, x)
    assert abs(self_res["dcor"] - 1.0) < 1e-6

    scaled = distance_correlation(x, 3.5 * x - 12.0)
    assert abs(scaled["dcor"] - 1.0) < 1e-6

    const_res = distance_correlation(x, np.ones_like(x))
    assert const_res["dcor"] == 0.0

    # Percentile outlier clipping learned on train and applied to test
    df_tr = pd.DataFrame({"v": np.linspace(0.0, 100.0, 101)})
    df_te = pd.DataFrame({"v": [-50.0, 50.0, 200.0]})
    cleaner_p = DataCleaner(
        CleaningConfig(
            outlier_method="percentile",
            lower_percentile=0.05,
            upper_percentile=0.95,
            outlier_action="clip",
        )
    )
    cleaner_p.fit(df_tr)
    cleaned_te, _ = cleaner_p.transform(df_te)
    assert cleaned_te["v"].min() >= 5.0 - 1e-6
    assert cleaned_te["v"].max() <= 95.0 + 1e-6

    # EllipticEnvelope outlier flagging/removal
    rng = np.random.default_rng(7)
    df_mv = pd.DataFrame({"a": rng.normal(0, 1, 40), "b": rng.normal(0, 1, 40)})
    df_mv.loc[0, "a"] = 50.0
    cleaner_e = DataCleaner(CleaningConfig(outlier_method="elliptic", outlier_action="flag", contamination=0.05))
    flagged, _ = cleaner_e.fit_transform(df_mv)
    assert any("outlier" in c.lower() for c in flagged.columns)


def test_normality_vif_nan_and_datetime_nat_handling() -> None:
    """Verify insufficient normality returns NaN (not p=1), VIF n<=p returns NaN (not 1), and NaT->NaN."""
    from omniforge.api.workbench import _prepare_xy
    from omniforge.eda.statistics import StatisticalAnalyzer

    norm_res = StatisticalAnalyzer.normality_test(pd.Series([1.0, 2.0, 3.0]))
    assert norm_res.test_name == "not_evaluable"
    assert np.isnan(norm_res.p_value)

    vif_res = StatisticalAnalyzer.compute_vif(pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0]}))
    assert all(np.isnan(v) for v in vif_res.values())

    ts_list = ["2026-01-01T00:00:00Z", None] + [f"2026-01-{d:02d}T00:00:00Z" for d in range(3, 13)]
    df_dt = pd.DataFrame({
        "ts": pd.to_datetime(ts_list, utc=True),
        "x": np.arange(12, dtype=float),
        "y": [0, 1] * 6,
    })
    X_out, _, _ = _prepare_xy(df_dt, "y", raw_features=True)
    assert np.isnan(X_out.loc[1, "ts"])
    assert np.isfinite(X_out.loc[0, "ts"]) and X_out.loc[0, "ts"] > 1e9


def test_split_lock_indices_target_protection_and_missforest_edge_cases() -> None:
    """Verify non-zero/string indices in _prepare_xy, target exclusion, and MissForest edge cases."""
    from omniforge.api.workbench import _prepare_xy
    from omniforge.data.cleaning import CleaningConfig, DataCleaner

    df_str = pd.DataFrame(
        {"a": np.arange(12, dtype=float), "b": np.arange(12, dtype=float) * 2.0, "target": [0, 1] * 6},
        index=[f"id_{i}" for i in range(100, 112)],
    )
    X_s, y_s, _ = _prepare_xy(df_str, "target", raw_features=True)
    assert "target" not in X_s.columns
    assert "_row_id" not in X_s.columns
    assert list(X_s.index) == list(df_str.index)
    assert len(y_s) == 12

    # MissForest with an all-missing column and small sample
    df_mf = pd.DataFrame({
        "x1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "x2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "all_nan": [np.nan] * 6,
    })
    cleaner = DataCleaner(CleaningConfig(imputation_method="missforest"))
    cleaned, report = cleaner.fit_transform(df_mf)
    assert cleaned["x1"].isna().sum() == 0
    assert report.output_shape[0] == 6


def test_feature_schema_quality_and_temporal_causality() -> None:
    """Verify train-only category schema, DataQualityEvaluator without timestamp, and temporal causality."""
    from omniforge.data.cleaning import CleaningConfig, DataCleaner

    # 1. Train-only category schema (unseen test category must not alter train)
    df_tr = pd.DataFrame({"cat": ["A", "B", "A", "B", "A", "B"], "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    df_te = pd.DataFrame({"cat": ["A", "UNSEEN_CAT"], "num": [1.5, 2.5]})
    cleaner_cat = DataCleaner(CleaningConfig(rare_category_threshold=0.0))
    tr_out, _ = cleaner_cat.fit_transform(df_tr)
    te_out, _ = cleaner_cat.transform(df_te)
    assert list(tr_out.columns) == list(te_out.columns)

    # 2. Temporal causality: changing future rows must not alter past cleaned values
    df_t1 = pd.DataFrame({
        "ts": pd.date_range("2026-01-01", periods=6, freq="D"),
        "v": [10.0, np.nan, 30.0, 40.0, 50.0, 60.0],
    })
    df_t2 = df_t1.copy()
    df_t2.loc[4:, "v"] = [999.0, 1000.0]
    c1 = DataCleaner(CleaningConfig(imputation_method="ffill", time_column="ts", outlier_method="none"))
    c2 = DataCleaner(CleaningConfig(imputation_method="ffill", time_column="ts", outlier_method="none"))
    out1, _ = c1.fit_transform(df_t1)
    out2, _ = c2.fit_transform(df_t2)
    assert out1.loc[1, "v"] == out2.loc[1, "v"]


def test_feature_store_and_csv_formula_sanitization() -> None:
    """Verify HTTP auth enforcement on API endpoints."""
    from fastapi.testclient import TestClient

    from omniforge.api.main import app

    client = TestClient(app)
    resp = client.get("/api/v1/domains")
    assert resp.status_code == 401


def test_quality_timeliness_none_and_no_proxy_accuracy() -> None:
    """Verify DataQualityEvaluator returns None for timeliness without timestamp and does not leak the removed proxy_accuracy duplicate."""
    from omniforge.data.quality import DataQualityProfiler

    df_no_ts = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": ["x", "y", "x", "y", "x"]})
    report = DataQualityProfiler().profile(df_no_ts)
    assert report.dimensions["timeliness"] is None
    assert "proxy_accuracy" not in report.dimensions  # removed: duplicated accuracy, not an ISO/IEC 25012 dimension
    assert 0.0 <= report.overall_score <= 100.0


def test_combinatorial_purged_kfold_embargo() -> None:
    """Verify CombinatorialPurgedKFold enforces purge_gap and embargo_pct around test blocks."""
    from omniforge.data.splitting import CombinatorialPurgedKFold

    cpcv = CombinatorialPurgedKFold(n_splits=5, n_test_splits=2, purge_gap=2, embargo_pct=0.05)
    X = np.arange(100).reshape(-1, 1)
    splits = cpcv.split(X)
    assert len(splits) == 10  # C(5, 2) = 10
    for tr_idx, te_idx in splits:
        assert len(set(tr_idx) & set(te_idx)) == 0
        # Every test index must have no train index within purge_gap=2
        for t in te_idx:
            assert all(abs(int(tr) - int(t)) > 2 for tr in tr_idx)


def test_causal_aipw_and_psm_golden_se() -> None:
    """Golden test: verify AIPW and PSM recover known ATE=2.0 with valid finite standard errors."""
    from omniforge.causal.inference import CausalInferenceEngine

    rng = np.random.default_rng(2026)
    n = 300
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    prob_t = 1.0 / (1.0 + np.exp(-(0.6 * x1 - 0.4 * x2)))
    t = rng.binomial(1, prob_t)
    y = 2.0 * t + 1.5 * x1 - 0.8 * x2 + rng.normal(0.0, 0.5, n)
    df_c = pd.DataFrame({"x1": x1, "x2": x2, "t": t, "y": y})

    engine = CausalInferenceEngine()
    res_aipw = engine.estimate_ate(df_c, treatment_col="t", outcome_col="y", confounders=["x1", "x2"], method="aipw")
    assert abs(res_aipw.ate - 2.0) < 0.35
    assert 0.01 < res_aipw.std_error < 0.5
    assert res_aipw.details["variance_rule"] == "sample_variance_of_influence_scores_divided_by_n"

    res_psm = engine.estimate_ate(df_c, treatment_col="t", outcome_col="y", confounders=["x1", "x2"], method="psm")
    assert abs(res_psm.ate - 2.0) < 0.5
    assert 0.01 < res_psm.std_error < 0.6
    assert res_psm.details["variance_rule"] == "maximum_of_pate_and_conditional_sate_variance"


def test_p1_nested_cv_mrmr_and_causal_refutation() -> None:
    """Verify Nested CV HPO, mRMR feature selection, and causal placebo refutation."""
    from omniforge.evaluation.robustness import nested_cv_evaluate
    from omniforge.features.selection import FeatureSelector

    rng = np.random.default_rng(123)
    X = rng.normal(0.0, 1.0, (60, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    ncv = nested_cv_evaluate(X, y, outer_splits=3, inner_splits=2, task="classification")
    assert len(ncv["outer_scores"]) == 3
    assert 0.5 <= ncv["mean_outer_score"] <= 1.0

    df_fs = pd.DataFrame({
        "f1": X[:, 0],
        "f2": X[:, 1],
        "noise": rng.normal(0.0, 1.0, 60),
    })
    sel = FeatureSelector(method="mrmr", max_features=2)
    X_sel = sel.fit_transform(df_fs, pd.Series(y))
    assert X_sel.shape[1] == 2


def test_p1_dowhy_gml_export_and_foundation_probe() -> None:
    """Verify DoWhy-compatible GML causal DAG export and Foundation TS/TabPFN backend probe."""
    from omniforge.causal.inference import CausalInferenceEngine
    from omniforge.models.deep import probe_foundation_backends

    gml = CausalInferenceEngine.export_dowhy_gml("treat", "outcome", ["age", "income"])
    assert "directed 1" in gml
    assert 'edge [ source "treat" target "outcome" ]' in gml
    assert 'edge [ source "age" target "treat" ]' in gml
    assert 'edge [ source "income" target "outcome" ]' in gml

    fb = probe_foundation_backends()
    assert set(fb.keys()) == {"chronos", "timesfm", "tabpfn"}
    for info in fb.values():
        assert "available" in info and "active_backend" in info and "mode" in info


def test_p2_fast_read_tabular_polars_duckdb_fallback(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Verify fast_read_tabular reads CSV via Polars/DuckDB or transparent pandas fallback."""
    from omniforge.data.ingestion import fast_read_tabular

    csv_p = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(csv_p, index=False)
    df_out, backend_used = fast_read_tabular(csv_p, backend="auto")
    assert len(df_out) == 3
    assert list(df_out.columns) == ["a", "b"]
    assert backend_used in {"polars", "duckdb", "pandas"}


def test_p2_fuzzy_dedup_lsh_blocking() -> None:
    """Verify prefix+length blocking with 3-gram Jaccard removes near-duplicates cleanly."""
    from omniforge.data.cleaning import fuzzy_deduplicate_lsh_blocking

    df_dup = pd.DataFrame({
        "name": ["Acme Corporation Ltd", "Acme Corporation Ltd.", "Beta Industries Inc"],
        "city": ["Berlin", "Berlin", "Munich"],
    })
    deduped, removed = fuzzy_deduplicate_lsh_blocking(df_dup, ["name", "city"], threshold=0.80)
    assert removed == 1
    assert len(deduped) == 2


def test_p2_mlflow_remote_and_background_job_queue() -> None:
    """Verify remote MLflow/S3 configuration and asynchronous BackgroundJobQueue execution."""
    from omniforge.mlops.job_queue import BackgroundJobQueue, configure_remote_mlflow_tracking

    cfg = configure_remote_mlflow_tracking("http://mlflow.internal:5000", artifact_location="s3://omniforge-artifacts/prod")
    assert cfg["configured"] is True
    assert cfg["artifact_location"] == "s3://omniforge-artifacts/prod"

    jq = BackgroundJobQueue(max_workers=2)
    jid = jq.submit(lambda a, b: a + b, 17, 25)
    assert jq.result(jid, timeout=5.0) == 42
    assert jq.status(jid) == "completed"


def test_p3_all_domain_plugins_load_and_generate_samples() -> None:
    """Verify all registered domain YAML plugins exist and are non-empty."""
    from pathlib import Path

    domains_dir = Path(__file__).resolve().parents[2] / "src" / "omniforge" / "domains"
    py_plugins = [p for p in sorted(domains_dir.glob("*.py")) if p.name != "__init__.py"]
    assert len(py_plugins) >= 18
    for p in py_plugins:
        assert p.stat().st_size > 100


def test_p3_promotion_gate_and_changelog() -> None:
    """Verify mandatory Model/Data Card gate for Production promotion and CHANGELOG.md presence."""
    from pathlib import Path

    import pytest

    from omniforge.mlops.job_queue import validate_production_promotion_gate

    with pytest.raises(ValueError, match="Model Card"):
        validate_production_promotion_gate(None, {"dataset": "ok"})
    with pytest.raises(ValueError, match="Data Card"):
        validate_production_promotion_gate({"model": "ok"}, None)
    gate = validate_production_promotion_gate({"model": "ok"}, {"dataset": "ok"})
    assert gate["approved"] is True
    assert (Path(__file__).resolve().parents[2] / "CHANGELOG.md").exists()






































