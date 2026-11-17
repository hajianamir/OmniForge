"""Comprehensive anti-leakage, CV pipeline isolation, causal forecasting & bug-fix verification tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.automl.pipeline_search import AutoMLPipelineSearch
from omniforge.core.types import TaskType
from omniforge.data.cleaning import CleaningConfig, DataCleaner
from omniforge.data.splitting import LeakageSafePipelineBuilder, LeakageSafeSplitter, SplitConfig
from omniforge.evaluation.backtesting import BacktestingEngine
from omniforge.features.engineering import FeatureConfig, FeatureEngineeringPipeline
from omniforge.features.selection import FeatureSelector
from omniforge.forecasting.engine import ForecastingEngine
from omniforge.mlops.continuous_training import ContinuousTrainingOrchestrator
from omniforge.mlops.registry import ModelRegistry
from omniforge.mlops.tracking import ExperimentTracker
from omniforge.models.anomaly.detector import AnomalyDetector
from omniforge.models.classical import ClassicalMLModel
from omniforge.models.clustering.clusterer import ClusteringEngine
from omniforge.models.time_series.deep_forecast import DeepForecaster
from omniforge.models.time_series.hierarchical import HierarchicalReconciler


def test_leakage_safe_splitter_stratified_group_and_time_series() -> None:
    rng = np.random.default_rng(7)
    n = 180
    df = pd.DataFrame(
        {
            "entity_id": np.repeat(np.arange(30), 6),
            "ts": pd.date_range("2025-01-01", periods=n, freq="D"),
            "x1": rng.normal(10.0, 2.0, n),
            "x2": rng.normal(50.0, 10.0, n),
            "cat": rng.choice(["A", "B", "C"], n),
            "y": rng.integers(0, 2, n),
        }
    )

    # 1. Stratified split
    res_strat = LeakageSafeSplitter(SplitConfig(strategy="stratified", test_size=0.25, random_seed=42)).split(
        df, target_col="y"
    )
    assert res_strat.audit_report is not None and res_strat.audit_report.passed
    assert set(res_strat.train_indices).isdisjoint(set(res_strat.test_indices))

    # 2. Group split: ZERO shared entity_id between train and test
    res_grp = LeakageSafeSplitter(SplitConfig(strategy="group", test_size=0.25, group_col="entity_id")).split(
        df, target_col="y"
    )
    assert res_grp.audit_report is not None and res_grp.audit_report.passed
    tr_entities = set(res_grp.train_df["entity_id"])
    te_entities = set(res_grp.test_df["entity_id"])
    assert tr_entities.isdisjoint(te_entities)

    # 3. Time-series split with embargo gap=5
    res_ts = LeakageSafeSplitter(
        SplitConfig(strategy="time_series", test_size=0.2, time_col="ts", gap=5)
    ).split(df, target_col="y")
    assert res_ts.audit_report is not None and res_ts.audit_report.passed
    assert res_ts.train_df["ts"].max() < res_ts.test_df["ts"].min()
    assert len(res_ts.train_df) + len(res_ts.test_df) == n - 5

    # 4. Fold-isolated sklearn Pipeline + ColumnTransformer CV
    from sklearn.ensemble import RandomForestClassifier

    X = df[["x1", "x2", "cat"]].copy()
    X.loc[[0, 15, 40], "x1"] = np.nan
    cv_out = LeakageSafePipelineBuilder.cross_validate_leakage_free(
        X,
        df["y"],
        estimator=RandomForestClassifier(n_estimators=20, random_state=42),
        task_type=TaskType.BINARY_CLASSIFICATION,
        split_config=SplitConfig(strategy="group", cv_folds=4, group_col="entity_id"),
        groups=df["entity_id"].to_numpy(),
    )
    assert cv_out["n_splits"] == 4
    assert cv_out["cv_strategy"] == "GroupKFold"
    assert len(cv_out["test_scores"]) == 4
    pipe_fitted = LeakageSafePipelineBuilder.build_ml_pipeline(
        X,
        RandomForestClassifier(n_estimators=20, random_state=42),
        task_type=TaskType.BINARY_CLASSIFICATION,
    ).fit(X, df["y"])
    # Deployable sklearn pipeline handles unseen categories + NaN cleanly
    unseen = pd.DataFrame({"x1": [np.nan], "x2": [48.0], "cat": ["UNSEEN_CAT"]})
    preds = pipe_fitted.predict(unseen)
    assert len(preds) == 1


def test_data_cleaner_anti_leakage_aliases_iterative_and_causal_ts() -> None:
    rng = np.random.default_rng(11)
    # 1. Alias validation check
    cfg_alias = CleaningConfig(
        drop_duplicates=True,  # type: ignore[call-arg]
        normalize_strings=True,  # type: ignore[call-arg]
        outlier_action="drop",  # type: ignore[arg-type]
        categorical_imputation="mode",  # type: ignore[arg-type]
    )
    assert cfg_alias.drop_exact_duplicates is True
    assert cfg_alias.clean_strings is True
    assert cfg_alias.outlier_action == "remove"
    assert cfg_alias.categorical_imputation == "most_frequent"

    # 2. Fit strictly on Train; Test extreme outlier must NOT shift Train imputer/LOF/IForest
    n = 120
    x = rng.normal(100.0, 5.0, n)
    df_tr = pd.DataFrame({"a": x, "b": 2.0 * x + rng.normal(0.0, 0.5, n)})
    df_tr.loc[[3, 10], "a"] = np.nan
    df_te = pd.DataFrame({"a": [np.nan, 101.0, 1e9], "b": [200.0, 202.0, 1e9]})

    cleaner_if = DataCleaner(CleaningConfig(numeric_imputation="iterative", outlier_method="lof", outlier_action="flag"))
    tr_clean, te_clean, rep_tr, rep_te = cleaner_if.fit_transform_split(df_tr, df_te)
    assert tr_clean["a"].isna().sum() == 0
    assert te_clean["a"].isna().sum() == 0
    assert te_clean.loc[2, "__is_multivariate_outlier"] == 1
    assert "outliers:lof:flag" in rep_tr.steps_executed
    assert "outliers:lof:flag" in rep_te.steps_executed
    # Train-fitted LOF flagged Test's 1e9 extreme row while Train median/imputer stayed immune to 1e9!
    assert float(te_clean.loc[0, "a"]) < 200.0

    # 3. Causal time-series imputation: future values never leak backward into Train
    ts_tr = pd.DataFrame({"t": [1, 2, 3, 4], "val": [10.0, np.nan, 14.0, np.nan]})
    ts_te = pd.DataFrame({"t": [5, 6], "val": [np.nan, 9999.0]})
    c_ts = DataCleaner(
        CleaningConfig(numeric_imputation="bfill", causal_time_series=True, time_column="t", outlier_method="none")
    )
    tr_c, te_c, _, _ = c_ts.fit_transform_split(ts_tr, ts_te)
    assert float(tr_c.loc[1, "val"]) == 10.0  # past-only ffill, NOT 14.0!
    assert float(te_c.loc[0, "val"]) == 14.0  # anchored strictly on Train's last observed value, NOT 9999.0!


def test_anomaly_detector_zero_batch_leakage_and_causal_ts() -> None:
    rng = np.random.default_rng(21)
    X_tr = pd.DataFrame(rng.normal(0.0, 1.0, (120, 4)), columns=["f1", "f2", "f3", "f4"])
    X_te = pd.DataFrame(
        np.vstack([rng.normal(0.0, 1.0, (10, 4)), np.full((2, 4), 25.0)]),
        columns=["f1", "f2", "f3", "f4"],
    )

    for method in ("isolation_forest", "lof", "zscore", "ensemble"):
        det = AnomalyDetector(method=method, contamination=0.05).fit(X_tr)
        batch_scores = det.anomaly_scores(X_te)
        single_score_0 = det.anomaly_scores(X_te.iloc[[0]])[0]
        single_score_last = det.anomaly_scores(X_te.iloc[[-1]])[0]
        # CRITICAL PROOF OF ZERO TEST-BATCH LEAKAGE: single-row score == batch score!
        assert np.isclose(single_score_0, batch_scores[0], atol=1e-9), f"{method} leaked test-batch stats!"
        assert np.isclose(single_score_last, batch_scores[-1], atol=1e-9), f"{method} leaked test-batch stats!"
        assert det.predict(X_te)[-1] == 1

    # Strictly causal time-series anomaly scores: mutating future t=59 must NOT alter scores at t=0..58!
    base_s = pd.Series(np.sin(np.linspace(0, 6, 60)) + rng.normal(0, 0.05, 60))
    mut_s = base_s.copy()
    mut_s.iloc[-1] = 1_000_000.0
    r1 = AnomalyDetector.detect_time_series_anomalies(base_s, window=10)
    r2 = AnomalyDetector.detect_time_series_anomalies(mut_s, window=10)
    assert np.allclose(r1["anomaly_score"].iloc[:-1].to_numpy(), r2["anomaly_score"].iloc[:-1].to_numpy(), atol=1e-12)
    assert bool(r2["is_anomaly"].iloc[-1]) is True


def test_clustering_inductive_dbscan_and_feature_engineering_selection() -> None:
    rng = np.random.default_rng(33)
    c1 = rng.normal(-5.0, 0.3, (50, 2))
    c2 = rng.normal(5.0, 0.3, (50, 2))
    df_tr = pd.DataFrame(np.vstack([c1, c2]), columns=["x", "y"])
    df_te = pd.DataFrame([[-5.1, -4.9], [5.1, 4.9], [500.0, 500.0]], columns=["x", "y"])

    cl = ClusteringEngine(algorithm="dbscan", eps=0.8, min_samples=4)
    res_tr, te_labels = cl.fit_predict_split(df_tr, df_te, feature_cols=["x", "y"], run_auto_k=False)
    assert res_tr.n_clusters_found >= 2
    assert te_labels[0] != -1 and te_labels[1] != -1 and te_labels[0] != te_labels[1]
    assert int(te_labels[2]) == -1  # Far OOD point honestly flagged as noise!

    # Verify fit_predict_split accepts k_min & k_max when run_auto_k=True
    km = ClusteringEngine(algorithm="kmeans", n_clusters=2, random_seed=42)
    km_res, km_te = km.fit_predict_split(df_tr, df_te, feature_cols=["x", "y"], run_auto_k=True, k_min=2, k_max=4)
    assert 2 <= km_res.n_clusters_found <= 4
    assert len(km_te) == len(df_te)

    # FeatureEngineeringPipeline: Test NaN uses Train median, NOT Test median; poly-only works
    fe_tr = pd.DataFrame({"a": [10.0, 12.0, 14.0, 16.0], "b": [1.0, 2.0, 3.0, 4.0]})
    fe_te = pd.DataFrame({"a": [np.nan, 9999.0], "b": [5.0, 6.0]})
    pipe = FeatureEngineeringPipeline(FeatureConfig(scale_numeric="none", create_polynomials=True, create_interactions=False))
    tr_out, te_out = pipe.fit_transform_split(fe_tr, fe_te)
    assert "a_pow2" in tr_out.columns
    assert np.isclose(float(te_out.loc[0, "a"]), 13.0, atol=1e-6)  # Train median (13), NOT Test value (9999)!

    # FeatureSelector aliases dispatch properly & expose .scores == .feature_scores
    y_bin = pd.Series([0, 0, 1, 1] * 15)
    X_fs = pd.DataFrame(
        {
            "signal": y_bin.to_numpy(dtype=float) + rng.normal(0, 0.1, 60),
            "dup_signal": y_bin.to_numpy(dtype=float) + rng.normal(0, 0.01, 60),
            "const": np.ones(60),
            "noise": rng.normal(0, 1, 60),
        }
    )
    for alias, canonical in [
        ("anova", "f_test"),
        ("lasso", "l1"),
        ("variance_threshold", "variance"),
        ("correlation_filter", "correlation"),
    ]:
        sel = FeatureSelector(method=alias, task="classification", max_features=2)
        assert sel.method == canonical
        sel.fit(X_fs, y_bin)
        assert sel.report_ is not None
        assert sel.report_.scores == sel.report_.feature_scores


def test_automl_forecasting_hierarchical_backtest_and_ct_anti_leakage(tmp_path) -> None:
    rng = np.random.default_rng(44)
    n = 120
    X = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, n),
            "f2": rng.normal(0, 1, n),
            "cat": rng.choice(["north", "south", "east"], n),
        }
    )
    y = pd.Series((X["f1"] * 1.8 - X["f2"] > 0.0).astype(int), name="target")
    groups = pd.Series(np.repeat(np.arange(20), 6))

    # 1. AutoML with Group split + fold-isolated Pipeline CV
    am = AutoMLPipelineSearch(
        time_budget_seconds=12.0,
        include_deep=False,
        tune_trials=2,
        split_strategy="group",
        cv_folds=3,
        use_cv_pipeline=True,
    ).fit(X, y, groups=groups)
    assert am.leakage_audit is not None and am.leakage_audit.passed is True
    assert len(am.cv_summary) >= 1
    assert "sklearn_pipeline" in am.metadata

    # 2. Causal Forecasting + DeepForecaster + Rolling CV
    t = np.arange(70, dtype=float)
    s = pd.Series(20.0 + 0.15 * t + 3.0 * np.sin(2 * np.pi * t / 7.0) + rng.normal(0, 0.2, len(t)))
    s.iloc[[5, 25, 65]] = np.nan
    f_rep = ForecastingEngine(seasonal_period=7, include_deep=True, include_foundation=False, n_cv_splits=3).forecast(
        pd.DataFrame({"y": s}), target_col="y", horizon=5, validation_horizon=5
    )
    assert f_rep.metadata.get("causal_imputation_verified") is True
    assert len(f_rep.metadata.get("rolling_cv_results") or []) >= 1

    df_fc = DeepForecaster("nbeats_mlp", epochs=12).fit(s.iloc[:-5]).predict(5)
    assert len(df_fc) == 5

    # 3. Backtesting with embargo gap + trading benchmark
    bt = BacktestingEngine().walk_forward_forecast(
        s, lambda: DeepForecaster("nbeats_mlp", epochs=10), initial_train_size=40, horizon=4, step_size=6, gap=2
    )
    assert bt["gap"] == 2 and bt["causal_imputation_verified"] is True and bt["n_origins"] >= 2
    tb = BacktestingEngine.backtest_trading_signals(
        prices=100.0 + np.cumsum(rng.normal(0.1, 1.0, 80)),
        signals=np.ones(80),
        slippage_bps=3.0,
    )
    assert "buy_and_hold_return" in tb and "max_drawdown_duration_periods" in tb

    # 4. Hierarchical MinT-Shrink optimal Schäfer-Strimmer lambda + coherence
    S = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    base_fc = np.array([[105.0], [62.0], [40.0], [30.0], [31.0], [41.0]])
    resids = rng.normal(0, 1.0, (6, 25))
    hr = HierarchicalReconciler("mint_shrink")
    rec = hr.reconcile(S, base_fc, residuals=resids)
    assert hr.last_shrinkage_lambda_ is not None and 0.0 <= hr.last_shrinkage_lambda_ <= 1.0
    assert HierarchicalReconciler.verify_coherence(S, rec)["coherent"] is True

    # 5. ContinuousTrainingOrchestrator leakage-free unseen holdout evaluation
    reg = ModelRegistry(tmp_path / "reg")
    trk = ExperimentTracker(tmp_path / "trk")
    ref_df = X.iloc[:70].copy()
    ref_df["target"] = y.iloc[:70].values
    new_df = X.iloc[70:].copy()
    new_df["target"] = y.iloc[70:].values
    ct = ContinuousTrainingOrchestrator(reg, trk)
    ct_res = ct.evaluate_and_retrain(
        model_name="demo_clf",
        reference_df=ref_df,
        new_labeled_df=new_df,
        target_col="target",
        trainer_fn=lambda Xf, yf: ClassicalMLModel("random_forest", task_type=TaskType.BINARY_CLASSIFICATION, track_carbon=False).fit(Xf, yf),
        force_retrain=True,
    )
    assert ct_res["retrained"] is True
    assert ct_res["leakage_free_evaluation"] is True
    assert ct_res["holdout_rows"] > 0
