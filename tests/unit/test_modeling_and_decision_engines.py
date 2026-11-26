"""Unit tests covering Baselines, Classical, Ensembles, Deep Learning, Specialized Models, Forecasting, Causal, OR & Simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omniforge.causal.inference import CausalInferenceEngine
from omniforge.core.types import TaskType
from omniforge.data.cleaning import DataCleaner
from omniforge.forecasting.engine import ForecastingEngine, diebold_mariano_test
from omniforge.models.anomaly.detector import AnomalyDetector
from omniforge.models.baselines import BaselineBenchmarkSuite
from omniforge.models.classical import ClassicalMLModel
from omniforge.models.computer_vision.vision import ComputerVisionEngine
from omniforge.models.deep import (
    AutoencoderModel,
    CNN1DModel,
    GraphNeuralNetworkModel,
    MLPModel,
    MultimodalFusionModel,
    RecurrentSequenceModel,
    TabularTransformerModel,
)
from omniforge.models.ensemble import StackingEnsembleModel, VotingEnsembleModel
from omniforge.models.graphs.analytics import GraphAnalyticsEngine
from omniforge.models.nlp.nlp_engine import NLPEngine
from omniforge.models.recommendation.recommender import HybridRecommender
from omniforge.models.registry import create_model
from omniforge.models.survival.survival import CoxPHModel, KaplanMeierEstimator
from omniforge.optimization.solvers import OptimizationEngine
from omniforge.simulation.engines import (
    AgentBasedSimulator,
    DigitalTwinEngine,
    DiscreteEventSimulator,
    MonteCarloSimulator,
)


def test_baselines_classical_ensembles_and_save_load(
    classification_df: pd.DataFrame, workspace_tmp: Path
) -> None:
    df, _ = DataCleaner().fit_transform(classification_df)
    X, y = df.drop(columns=["target"]), df["target"]
    X_tr, X_te, y_tr, y_te = X.iloc[:160], X.iloc[160:], y.iloc[:160], y.iloc[160:]

    suite_res = BaselineBenchmarkSuite.run_classification(X_tr, y_tr, X_te, y_te)
    assert len(suite_res) >= 4

    rf = create_model("random_forest", task_type=TaskType.BINARY_CLASSIFICATION, n_estimators=30)
    rf.fit(X_tr, y_tr)
    preds = rf.predict(X_te)
    probs = rf.predict_proba(X_te)
    assert len(preds) == len(X_te)
    assert probs.shape == (len(X_te), 2)

    enc_key = "test-model-encryption-key-32-chars-minimum"
    saved = rf.save(workspace_tmp / "rf_model.joblib", encrypt=True, encryption_key=enc_key)
    reloaded = ClassicalMLModel.load(saved, encrypted=True, encryption_key=enc_key)
    np.testing.assert_array_equal(preds, reloaded.predict(X_te))

    ens_v = VotingEnsembleModel(task_type=TaskType.BINARY_CLASSIFICATION).fit(X_tr, y_tr)
    assert len(ens_v.predict(X_te)) == len(X_te)
    ens_s = StackingEnsembleModel(task_type=TaskType.BINARY_CLASSIFICATION).fit(X_tr, y_tr)
    assert len(ens_s.predict(X_te)) == len(X_te)


def test_deep_learning_models_suite(classification_df: pd.DataFrame) -> None:
    df, _ = DataCleaner().fit_transform(classification_df)
    X = df.select_dtypes(include=[np.number]).drop(columns=["target"], errors="ignore").to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=int)

    mlp = MLPModel(task_type=TaskType.BINARY_CLASSIFICATION, epochs=6, hidden_dims=(32, 16)).fit(X[:150], y[:150])
    assert len(mlp.predict(X[150:])) == len(X) - 150

    ae = AutoencoderModel(latent_dim=3, epochs=6).fit(X[:150])
    assert len(ae.reconstruction_error(X[150:])) == len(X) - 150

    rng = np.random.default_rng(1)
    sig_X = rng.normal(0, 1, size=(40, 16))
    sig_y = (sig_X[:, :4].mean(axis=1) > 0).astype(int)
    cnn = CNN1DModel(task_type=TaskType.BINARY_CLASSIFICATION, epochs=4).fit(sig_X[:30], sig_y[:30])
    assert len(cnn.predict(sig_X[30:])) == 10

    seq_X = rng.normal(0, 1, size=(40, 8, 4))
    seq_y = (seq_X[:, -1, 0] > 0).astype(int)
    rnn = RecurrentSequenceModel(task_type=TaskType.BINARY_CLASSIFICATION, epochs=4).fit(seq_X[:30], seq_y[:30])
    assert len(rnn.predict(seq_X[30:])) == 10

    trans = TabularTransformerModel(task_type=TaskType.BINARY_CLASSIFICATION, epochs=4).fit(X[:80], y[:80])
    assert len(trans.predict(X[80:100])) == 20

    n_nodes = 20
    feats = rng.normal(0, 1, size=(n_nodes, 5))
    adj = np.eye(n_nodes)
    for i in range(n_nodes - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1.0
    node_y = (feats[:, 0] > 0).astype(int)
    gnn = GraphNeuralNetworkModel(task_type=TaskType.BINARY_CLASSIFICATION, epochs=6).fit(feats, node_y, adjacency=adj)
    assert len(gnn.predict(feats, adjacency=adj)) == n_nodes

    mm = MultimodalFusionModel(task_type=TaskType.BINARY_CLASSIFICATION, epochs=5).fit_multimodal(
        {"tabular": X[:60], "signal": rng.normal(0, 1, size=(60, 4))}, y[:60]
    )
    assert len(mm.predict_multimodal({"tabular": X[60:80], "signal": rng.normal(0, 1, size=(20, 4))})) == 20


def test_forecasting_causal_optimization_simulation_and_specialized(
    timeseries_df: pd.DataFrame, causal_df: pd.DataFrame
) -> None:
    # Forecasting
    fc = ForecastingEngine(include_deep=False).forecast(timeseries_df, target_col="y", time_col="timestamp", horizon=10)
    assert len(fc.forecast) == 10
    assert len(fc.leaderboard) >= 4

    # Causal
    c_res = CausalInferenceEngine(bootstrap_rounds=20).estimate_ate(
        causal_df,
        treatment_col="treatment",
        outcome_col="outcome",
        confounders=["confounder_1", "confounder_2"],
        method="aipw",
    )
    assert abs(c_res.ate - 3.2) < 1.5

    # Optimization (LP, Knapsack, Hungarian, Markowitz, VRP)
    opt = OptimizationEngine()
    lp = opt.solve_lp([-3.0, -2.0], A_ub=[[1.0, 1.0], [2.0, 1.0]], b_ub=[4.0, 5.0])
    assert lp.optimal is True
    ks = opt.solve_knapsack([60, 100, 120], [10, 20, 30], capacity=50)
    assert ks.objective_value >= 220.0
    asg = opt.solve_assignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
    assert asg.optimal is True
    port = opt.optimize_portfolio([0.08, 0.12], [[0.04, 0.01], [0.01, 0.09]])
    assert abs(sum(port.solution.values()) - 1.0) < 1e-4
    coords = np.array([(0.0, 0.0), (1.0, 2.0), (2.0, 1.0), (-1.0, 2.0)])
    dist_mat = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
    vrp = opt.solve_vrp_savings(dist_mat, [0, 3, 4, 2], vehicle_capacity=6)
    assert vrp.status == "heuristic_feasible"
    assert not vrp.optimal
    assert sorted(customer for route in vrp.solution["routes"] for customer in route[1:-1]) == [1, 2, 3]

    # Simulation
    mc_res, _ = MonteCarloSimulator(n_simulations=500).run(
        {"x": {"dist": "normal", "mean": 10.0, "std": 2.0}}, lambda s: s["x"] * 2.0
    )
    assert 18.0 < mc_res.mean < 22.0
    des = DiscreteEventSimulator().simulate_queue(arrival_rate=0.8, service_rate=1.2, n_servers=1, horizon_time=200.0)
    assert des["completed_entities"] > 0
    abm = AgentBasedSimulator(n_agents=40).simulate_diffusion(steps=10)
    assert len(abm) == 11 and "active" in abm.columns
    twin = DigitalTwinEngine(
        A=np.array([[1.0]]),
        B=np.array([[0.5]]),
        H=np.array([[1.0]]),
        Q=np.array([[0.01]]),
        R=np.array([[0.05]]),
        x0=np.array([20.0]),
    )
    est = twin.assimilate(np.array([20.5]), control=np.array([1.0]))
    assert est is not None

    # Recommender, Survival, Anomaly, NLP, CV, Graph
    inter = pd.DataFrame(
        {"user_id": ["u1", "u1", "u2", "u2", "u3"], "item_id": ["i1", "i2", "i2", "i3", "i1"], "rating": [5, 4, 5, 3, 4]}
    )
    rec = HybridRecommender(n_factors=2).fit(inter)
    assert len(rec.recommend("u1", top_k=2)) >= 1

    km = KaplanMeierEstimator().fit([2, 4, 6, 8, 10], [1, 0, 1, 1, 0])
    assert len(km.survival_table_) > 0
    cox_df = pd.DataFrame({"x1": [1, 2, 3, 4, 5, 6], "duration": [10, 8, 7, 5, 3, 2], "event": [1, 1, 0, 1, 1, 1]})
    cox = CoxPHModel().fit(cox_df, duration_col="duration", event_col="event")
    assert len(cox.predict_risk(pd.DataFrame({"x1": [1, 6]}))) == 2

    anom = AnomalyDetector(method="ensemble").fit(np.random.default_rng(0).normal(0, 1, size=(80, 4)))
    assert len(anom.predict(np.zeros((5, 4)))) == 5

    nlp = NLPEngine()
    sent = nlp.analyze_sentiment(["This platform is excellent and reliable!", "Terrible broken slow experience."])
    assert sent[0]["label"] == "positive" and sent[1]["label"] == "negative"

    cv = ComputerVisionEngine()
    imgs = [np.zeros((24, 24, 3), dtype=np.uint8) + i * 20 for i in range(1, 11)]
    cv.fit_classifier(imgs, [0] * 5 + [1] * 5)
    assert len(cv.predict_classifier(imgs[:2])) == 2

    ga = GraphAnalyticsEngine()
    edges_df = pd.DataFrame({"source": ["A", "B", "C", "C"], "target": ["B", "C", "A", "D"], "weight": [1.0, 1.0, 1.0, 1.0]})
    g = ga.build_graph(edges_df, weight_col="weight")
    assert g.number_of_nodes() == 4


def test_clustering_engine_all_algorithms_autok_and_profiling() -> None:
    """Verify ClusteringEngine across all 7 algorithms, Auto-K optimization, comparison, PCA & ANOVA profiling."""
    from omniforge.models.clustering import SUPPORTED_CLUSTERING_ALGORITHMS, ClusteringEngine

    rng = np.random.default_rng(42)
    c0 = rng.normal(loc=[-6.0, -6.0, 0.0], scale=0.6, size=(40, 3))
    c1 = rng.normal(loc=[6.0, 6.0, 5.0], scale=0.6, size=(40, 3))
    c2 = rng.normal(loc=[0.0, 8.0, -5.0], scale=0.6, size=(40, 3))
    df = pd.DataFrame(np.vstack([c0, c1, c2]), columns=["f1", "f2", "f3"])

    # Auto-K should identify 3 clusters with high silhouette score
    probe = ClusteringEngine(algorithm="kmeans", scaler="standard")
    auto_res = probe.auto_select_k(df, k_min=2, k_max=6, algorithm="kmeans")
    assert auto_res["best_k"] == 3
    assert auto_res["best_silhouette"] > 0.65
    assert len(auto_res["curve"]) == 5

    # All 7 supported clustering algorithms run end-to-end cleanly
    for algo in SUPPORTED_CLUSTERING_ALGORITHMS:
        eng = ClusteringEngine(algorithm=algo, n_clusters=3, eps=0.8, min_samples=4)
        res = eng.fit_predict_with_profile(df, feature_cols=["f1", "f2", "f3"], run_auto_k=False)
        assert res.algorithm == algo
        assert len(res.labels) == len(df)
        assert res.n_clusters_found >= 2, f"{algo} found too few clusters: {res.n_clusters_found}"
        assert res.pca_2d is not None and len(res.pca_2d) == len(df)
        assert len(res.cluster_profiles) >= 2
        assert len(res.feature_separation_tests) == 3
        assert all(float(t["p-value (Sig)"]) < 0.05 for t in res.feature_separation_tests)
        # Predict on new samples for every algorithm
        new_preds = eng.predict(df.iloc[:6])
        assert len(new_preds) == 6

    # GMM soft cluster probabilities sum to 1.0
    gmm_eng = ClusteringEngine(algorithm="gmm", n_clusters=3).fit(df)
    probs = gmm_eng.predict_proba_clusters(df.iloc[:5])
    assert probs.shape == (5, 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    # Multi-algorithm comparison benchmark
    cmp_df = probe.compare_algorithms(df, n_clusters=3, eps=0.8, min_samples=4)
    assert len(cmp_df) == 5
    assert set(cmp_df["algorithm"]) == {"kmeans", "agglomerative", "gmm", "birch", "dbscan"}
    assert float(cmp_df.iloc[0]["silhouette"]) > 0.65


def test_model_equations_and_future_target_prediction_manual_file_sql(tmp_path, monkeypatch) -> None:
    """Verify exact mathematical equations (Baseline, Neural, Advanced) and Manual/File/SQL future scoring."""
    monkeypatch.setenv("OMNIFORGE_ALLOWED_DB_DIRS", str(tmp_path))
    from omniforge.data.validation import DataValidator, ValidationRule
    from omniforge.evaluation.equations import ModelEquationExtractor
    from omniforge.mlops.future_scoring import FutureTargetPredictor
    from omniforge.models.baselines import (
        BaselineBenchmarkSuite,
        ForecastingBaseline,
        LogisticBaseline,
        RidgeLinearBaseline,
    )
    from omniforge.models.clustering import ClusteringEngine
    from omniforge.models.deep.cnn import CNN1DModel
    from omniforge.models.deep.gnn import GraphNeuralNetworkModel
    from omniforge.models.deep.transformers import TabularTransformerModel

    # 0. ValidationReport canonical fields + backward-compatible aliases
    tiny_df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    vrep = DataValidator([ValidationRule(name="a_nn", rule_type="not_null", column="a")]).validate(tiny_df)
    assert vrep.passed is True
    assert vrep.passed_count == vrep.passed_checks == 1
    assert vrep.total_expectations == vrep.total_checks == 1
    assert vrep.failed_count == vrep.failed_checks == 0

    # 1. Exact unstandardized linear baseline equation numerically reproduces Ridge predictions
    rng = np.random.default_rng(7)
    X_reg = pd.DataFrame(
        {
            "hours": rng.uniform(10.0, 50.0, size=120),
            "temp": rng.normal(25.0, 5.0, size=120),
            "pressure": rng.normal(100.0, 12.0, size=120),
        }
    )
    y_reg = 12.5 + 3.0 * X_reg["hours"] - 1.5 * X_reg["temp"] + 0.4 * X_reg["pressure"] + rng.normal(0.0, 0.5, size=120)

    ridge_bl = RidgeLinearBaseline(alpha=0.1).fit(X_reg, y_reg)
    eq_ridge = ModelEquationExtractor.extract(ridge_bl, X_reference=X_reg, target_name="yield_out")
    assert eq_ridge.category == "baseline"
    assert eq_ridge.equation_type == "exact_closed_form"
    assert eq_ridge.intercept is not None
    assert len(eq_ridge.coefficients_table) == 3
    # Verify exact mathematical equivalence: beta0 + sum(beta_j * x_j) == ridge_bl.predict(X_reg)
    coef_map = {r["Feature"]: float(r["Raw Coefficient (beta)"]) for r in eq_ridge.coefficients_table}
    reconstructed = eq_ridge.intercept + sum(coef_map[c] * X_reg[c].to_numpy() for c in X_reg.columns)
    assert np.allclose(reconstructed, ridge_bl.predict(X_reg), atol=1e-4)

    # 2. Logistic baseline equation (classification) + Forecasting baseline equation
    y_clf = (y_reg > float(y_reg.median())).astype(int)
    log_bl = LogisticBaseline().fit(X_reg, y_clf)
    eq_log = ModelEquationExtractor.extract(log_bl, X_reference=X_reg, target_name="is_high")
    assert eq_log.category == "baseline" and "sigma" in eq_log.latex_equation
    assert any("Odds Ratio" in k for k in eq_log.coefficients_table[0])

    fc_bl = ForecastingBaseline(method="drift").fit(y_reg.to_numpy()[:40])
    eq_fc = ModelEquationExtractor.extract(fc_bl, target_name="yield_out")
    assert eq_fc.category == "baseline" and "h" in eq_fc.plain_equation

    # 3. Neural Network equations (MLP, CNN1D, Tabular Transformer, GCN)
    for dl_cls in (MLPModel, CNN1DModel, TabularTransformerModel, GraphNeuralNetworkModel):
        dl_m = dl_cls(epochs=10, task_type=TaskType.REGRESSION, track_carbon=False).fit(X_reg.iloc[:80], y_reg.iloc[:80])
        eq_dl = ModelEquationExtractor.extract(dl_m, X_reference=X_reg.iloc[:80], target_name="yield_out")
        assert eq_dl.category == "deep_learning"
        assert len(eq_dl.architecture_latex) >= 2
        assert len(eq_dl.layer_summary) >= 1
        assert eq_dl.parameter_count is not None and eq_dl.parameter_count > 0
        assert eq_dl.surrogate_plain_equation is not None
        assert eq_dl.surrogate_fidelity_r2 is not None

    # 4. Advanced Classical & Voting Ensemble equations
    rf_m = ClassicalMLModel("random_forest", task_type=TaskType.REGRESSION, track_carbon=False).fit(X_reg.iloc[:90], y_reg.iloc[:90])
    eq_rf = ModelEquationExtractor.extract(rf_m, X_reference=X_reg.iloc[:90], target_name="yield_out")
    assert eq_rf.category == "classical_ml"
    assert eq_rf.decision_rules_text is not None and "hours" in eq_rf.decision_rules_text
    assert eq_rf.surrogate_fidelity_r2 is not None and eq_rf.surrogate_fidelity_r2 > 0.7

    ens_m = VotingEnsembleModel(estimators=[ridge_bl, rf_m], task_type=TaskType.REGRESSION, track_carbon=False).fit(
        X_reg.iloc[:90], y_reg.iloc[:90]
    )
    eq_ens = ModelEquationExtractor.extract(ens_m, X_reference=X_reg.iloc[:90], target_name="yield_out")
    assert eq_ens.category == "ensemble"
    assert "0.50" in eq_ens.plain_equation

    # 5. Operational Future Target Prediction across Manual, File & SQL + Cluster & OOD context
    ref_df = X_reg.copy()
    ref_df["yield_out"] = y_reg
    ref_df["is_high"] = y_clf
    feat_cols = ["hours", "temp", "pressure"]

    _, bl_models_reg = BaselineBenchmarkSuite.run_regression_detailed(X_reg.iloc[:90], y_reg.iloc[:90], X_reg.iloc[90:], y_reg.iloc[90:])
    mlp_reg = MLPModel(hidden_dims=(32, 16), epochs=15, task_type=TaskType.REGRESSION, track_carbon=False).fit(X_reg.iloc[:90], y_reg.iloc[:90])
    clust_res = ClusteringEngine(algorithm="kmeans", n_clusters=3).fit_predict_with_profile(ref_df, feature_cols=feat_cols, run_auto_k=False)

    # 5a. Manual scenario template
    manual_df = FutureTargetPredictor.build_manual_template(ref_df, feat_cols, n_rows=3)
    assert len(manual_df) == 3 and list(manual_df.columns) == feat_cols

    # 5b. Live SQL round-trip: create SQLite DB -> load_from_sql -> predict_future -> write_predictions_to_sql
    db_file = tmp_path / "future_test.db"
    uri, query, _ = FutureTargetPredictor.create_sample_sqlite_db(ref_df, feat_cols, db_file, n_rows=10)
    sql_incoming = FutureTargetPredictor.load_from_sql(uri, query)
    assert len(sql_incoming) == 10

    f_res = FutureTargetPredictor.predict_future(
        sql_incoming,
        ref_df,
        feat_cols,
        target_models={"yield_out": rf_m, "is_high": log_bl},
        secondary_models_by_target={"yield_out": {**bl_models_reg, "deep_mlp": mlp_reg}},
        clustering_result=clust_res,
        source_type="sql",
    )
    assert f_res.n_rows_scored == 10 and f_res.source_type == "sql"
    assert "future_yield_out" in f_res.scored_df.columns
    assert "baseline_yield_out" in f_res.scored_df.columns
    assert "neural_yield_out" in f_res.scored_df.columns
    assert "consensus_yield_out" in f_res.scored_df.columns
    assert "future_is_high" in f_res.scored_df.columns
    assert "assigned_segment" in f_res.scored_df.columns
    assert "analytical_data_risk" in f_res.scored_df.columns
    assert len(f_res.executive_outlook) >= 2

    write_info = FutureTargetPredictor.write_predictions_to_sql(f_res.scored_df, uri, table_name="scored_future")
    assert write_info["rows_written"] == 10
    reloaded = FutureTargetPredictor.load_from_sql(uri, "SELECT * FROM scored_future")
    assert len(reloaded) == 10 and "future_yield_out" in reloaded.columns





def test_forecast_rolling_cv_predicts_through_embargo(monkeypatch: pytest.MonkeyPatch) -> None:
    class ExactLinearForecaster:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, y):
            self.last = y[-1]

        def predict(self, horizon):
            values = self.last + np.arange(1, horizon + 1)
            return pd.DataFrame({"yhat": values, "yhat_lower": values, "yhat_upper": values})

    monkeypatch.setattr("omniforge.forecasting.engine.StatisticalForecaster", ExactLinearForecaster)
    monkeypatch.setattr("omniforge.forecasting.engine.MLForecaster", ExactLinearForecaster)
    values = np.arange(60, dtype=float)
    values[:2] = np.nan
    values[-3] = np.nan
    report = ForecastingEngine(gap=3, include_deep=False, include_foundation=False).forecast(
        pd.DataFrame({"y": values}), target_col="y", horizon=6,
    )
    assert report.metadata["n_history"] == 58
    assert report.metadata["validation_observed"] == 5
    assert report.metadata["validation_missing"] == 1
    assert report.metadata["validation_gap"] == 3
    assert report.baseline_benchmarks["Baseline_drift"]["rmse"] == 0
    assert report.champion_model == "Baseline_drift"
    assert report.forecast["yhat"].tolist() == list(range(60, 66))
    assert report.leaderboard[0]["rmse"] == 0
    assert report.metadata["diebold_mariano_vs_runner_up"]["evaluable"] is False
    scores = report.metadata["rolling_cv_results"]
    assert scores and all(row["cv_rmse_mean"] == 0 for row in scores)


def test_dm_constant_nonzero_loss_difference_has_no_p_value() -> None:
    result = diebold_mariano_test(np.zeros(8), np.zeros(8), np.ones(8))
    assert result["evaluable"] is False
    assert result["p_value"] is None
    assert result["mean_loss_diff"] == -1
    assert result["preferred_model"] == "model_1"


def test_nonlinear_solver_reports_local_convergence() -> None:
    def objective(x: np.ndarray) -> float:
        return float((x[0] ** 2 - 1) ** 2 + 0.4 * x[0])

    result = OptimizationEngine().solve_nonlinear(objective, [1.0])
    assert result.status == "locally_converged"
    assert not result.optimal
    assert result.objective_value > objective(np.array([-1.0]))
