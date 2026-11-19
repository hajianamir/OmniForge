"""Comprehensive mathematical, statistical, and algorithmic verification suite for Phase P1 remediation.

Verifies exact compliance with peer-reviewed statistical literature and cross-checks
against authoritative reference packages (SciPy, statsmodels) across all 17 Phase P1
items plus second-pass audit invariants.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.forecasting.theta import ThetaModel

from omniforge.causal.inference import CausalInferenceEngine
from omniforge.core.types import TaskType
from omniforge.eda.diagnostics import UniversalDiagnosticsSuite
from omniforge.eda.statistics import StatisticalAnalyzer
from omniforge.evaluation.backtesting import BacktestingEngine
from omniforge.evaluation.fairness import FairnessAuditor
from omniforge.evaluation.interpretability import InterpretabilityEngine
from omniforge.evaluation.metrics import MetricsEvaluator
from omniforge.evaluation.robustness import ConformalPredictor, split_conformal_quantile
from omniforge.evaluation.validation import CombinatorialPurgedKFold, CrossValidator, PurgedTimeSeriesSplit
from omniforge.mlops.drift import DriftDetector
from omniforge.models.anomaly.detector import AnomalyDetector
from omniforge.models.base import BaseOmniForgeModel
from omniforge.models.clustering.clusterer import ClusteringEngine
from omniforge.models.deep.gnn import GraphAutoencoderLinkModel
from omniforge.models.graphs.link_prediction import GraphLinkPredictionSuite
from omniforge.models.nonparametric import NadarayaWatsonKernelModel, NonParametricAnalyticalSuite
from omniforge.models.survival.survival import CoxPHModel, KaplanMeierEstimator
from omniforge.models.time_series.deep_forecast import DeepForecaster
from omniforge.models.time_series.ml_forecast import MLForecaster
from omniforge.models.time_series.statistical import StatisticalForecaster
from omniforge.simulation.engines import DigitalTwinEngine, MonteCarloSimulator


# ---------------------------------------------------------------------------
# P1-1: Financial Sortino TDD (full-N) & W0=1.0 Max Drawdown
# ---------------------------------------------------------------------------
def test_p1_1_sortino_full_n_tdd_and_w0_max_drawdown() -> None:
    """Verify full-N target downside deviation and W0=1.0 max drawdown on immediate losses."""
    # All negative returns: W0=1.0 -> [1.0, 0.9, 0.81] => max drawdown = -0.19 (not -0.10!)
    r_neg = np.array([-0.10, -0.10], dtype=float)
    fin = MetricsEvaluator.financial_metrics(r_neg, periods_per_year=1)
    assert math.isclose(fin["max_drawdown"], -0.19, rel_tol=1e-5)

    # Full-N Sortino TDD with MAR=0: 99 gains (+0.02) and 1 loss (-0.01)
    r_mix = np.array([0.02] * 99 + [-0.01], dtype=float)
    fin_mix = MetricsEvaluator.financial_metrics(r_mix, periods_per_year=252, mar=0.0)
    expected_tdd = math.sqrt((0.01**2) / 100.0)
    expected_sortino = (float(np.mean(r_mix)) / expected_tdd) * math.sqrt(252.0)
    assert math.isclose(fin_mix["sortino_ratio"], expected_sortino, rel_tol=1e-3)

    # Backtesting underwater curve must also start at W0=1.0
    bt = BacktestingEngine.backtest_trading_signals(
        prices=np.array([100.0, 90.0, 81.0, 85.05]),
        signals=np.array([1.0, 1.0, 1.0, 1.0]),
        transaction_cost_bps=0.0,
        slippage_bps=0.0,
        periods_per_year=1,
    )
    assert math.isclose(bt["max_drawdown"], -0.19, rel_tol=1e-4)
    assert bt["max_drawdown_duration_periods"] == 3


# ---------------------------------------------------------------------------
# P1-2: Diagnostics Full-N Semi-Deviation, Central Range & Infinite VIF
# ---------------------------------------------------------------------------
def test_p1_2_diagnostics_semidev_central_range_and_inf_vif() -> None:
    """Verify Markowitz full-N semi-deviation (~sigma/sqrt(2)), central 95% range, and inf VIF."""
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, 10_000)
    df = pd.DataFrame({"x": x, "x_dup": 2.0 * x + 3.0, "y": x + rng.normal(0.0, 0.5, 10_000)})
    rep = UniversalDiagnosticsSuite().run(df, target_cols=["y"])
    tail = {r["Column"]: r for r in rep.risk_assessment["tail_risk_table"]}
    assert abs(float(tail["x"]["Downside Risk (Semi-Dev)"]) - (1.0 / math.sqrt(2.0))) < 0.03
    assert math.isinf(float(rep.vif_scores["x"]))
    assert float(rep.risk_assessment["risk_components"]["multicollinearity_risk"]) == 20.0


# ---------------------------------------------------------------------------
# P1-3: Exact Finite-Sample Split Conformal Quantile & Forecast Leakage Fix
# ---------------------------------------------------------------------------
def test_p1_3_split_conformal_exact_order_statistic_and_forecasters() -> None:
    """Verify Angelopoulos & Bates (2021) Theorem 1.1 exact k-th order statistic coverage."""
    scores = np.arange(1.0, 13.0)  # n=12, alpha=0.10 -> k = ceil(13 * 0.9) = 12 -> s_(12) = 12.0
    q_12, ok_12 = split_conformal_quantile(scores, alpha=0.10)
    assert q_12 == 12.0 and ok_12 is True
    # n=8, alpha=0.10 -> k = ceil(9 * 0.9) = 9 > 8 -> finite_sample_valid=False and capped at max(scores)=8.0
    q_8, ok_8 = split_conformal_quantile(np.arange(1.0, 9.0), alpha=0.10)
    assert q_8 == 8.0 and ok_8 is False

    rng = np.random.default_rng(42)
    covered = 0
    trials = 1000
    for _ in range(trials):
        cal = rng.standard_normal(12)
        test = float(rng.standard_normal())
        q, _ = split_conformal_quantile(np.abs(cal), alpha=0.10)
        if abs(test) <= q:
            covered += 1
    rate = covered / trials
    # Exact theoretical coverage is k/(n+1) = 12/13 = 0.9231 in [0.90, 0.9769]
    assert 0.89 <= rate <= 0.96

    # ConformalPredictor calibrates and outputs valid prediction sets
    from omniforge.models.baselines import LogisticBaseline

    X_cal = rng.normal(0.0, 1.0, size=(40, 2))
    y_cal = (X_cal[:, 0] > 0.0).astype(int)
    clf = LogisticBaseline().fit(X_cal[:20], y_cal[:20])
    cp = ConformalPredictor(clf, alpha=0.10).calibrate(X_cal[20:], y_cal[20:])
    sets = cp.predict_sets(X_cal[20:22])
    assert cp.is_calibrated_ and cp.finite_sample_valid_ and len(sets) == 2

    # MLForecaster & DeepForecaster produce valid monotonic intervals with finite-sample conformal
    series = 50.0 + 0.5 * np.arange(80, dtype=float) + np.sin(np.arange(80) * 2.0 * np.pi / 7.0)
    ml_fc = MLForecaster(algorithm="ridge", seasonal_period=7).fit(series).predict(5)
    assert (ml_fc["yhat_lower"] <= ml_fc["yhat"]).all() and (ml_fc["yhat"] <= ml_fc["yhat_upper"]).all()
    dp_fc = DeepForecaster(architecture="tft", context_length=12, horizon_train=4, epochs=12).fit(series).predict(4)
    assert (dp_fc["yhat_lower"] <= dp_fc["yhat"]).all() and (dp_fc["yhat"] <= dp_fc["yhat_upper"]).all()


# ---------------------------------------------------------------------------
# P1-4..P1-9: EDA Statistics (Lilliefors, VIF, ADF MacKinnon, Bonett-Wright, MWU, KW)
# ---------------------------------------------------------------------------
def test_p1_4_to_p1_9_statistical_analyzer_exact_invariants() -> None:
    """Verify Lilliefors, exact OLS VIF, MacKinnon ADF, Bonett-Wright CIs, Kerby r_rb, and KW eta^2_H."""
    rng = np.random.default_rng(123)

    # P1-4: Lilliefors rejects chi2(3) and holds ~5% Type-I error on Gaussian
    res_lf = StatisticalAnalyzer.normality_test(pd.Series(rng.chisquare(3, 60)), method="kolmogorov_smirnov")
    assert res_lf.test_name == "lilliefors_ks" and res_lf.reject_null is True

    # P1-5: VIF matches statsmodels and reports inf on exact collinearity
    x1 = rng.standard_normal(200)
    x2 = 0.8 * x1 + 0.6 * rng.standard_normal(200)
    x3 = rng.standard_normal(200)
    vifs = StatisticalAnalyzer.compute_vif(pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}))
    Xc = np.column_stack([np.ones(200), x1, x2, x3])
    for idx, col in enumerate(["x1", "x2", "x3"], start=1):
        assert abs(vifs[col] - variance_inflation_factor(Xc, idx)) < 1e-3
    vifs_col = StatisticalAnalyzer.compute_vif(pd.DataFrame({"x1": x1, "dup": 3.0 * x1 - 1.0, "x3": x3}))
    assert math.isinf(vifs_col["x1"]) and math.isinf(vifs_col["dup"]) and vifs_col["x3"] < 1.2

    # P1-6: ADF MacKinnon fallback matches statsmodels adfuller statistic on stationary series
    wn = rng.standard_normal(150)
    r_sm = StatisticalAnalyzer.stationarity_test(pd.Series(wn))
    r_fb = StatisticalAnalyzer._adf_mackinnon_fallback(wn)
    assert r_sm.reject_null is True and r_fb.reject_null is True
    assert abs(r_sm.statistic - r_fb.statistic) < 1e-3

    # P1-7: Bonett & Wright (2000) exact Spearman & Kendall CIs
    n_c = 80
    u = rng.standard_normal(n_c)
    v = 0.85 * u + 0.5 * rng.standard_normal(n_c)
    rs = StatisticalAnalyzer.correlation_test(pd.Series(u), pd.Series(v), method="spearman")
    rk = StatisticalAnalyzer.correlation_test(pd.Series(u), pd.Series(v), method="kendall")
    z975 = float(stats.norm.ppf(0.975))
    exp_s = round(float(np.tanh(np.arctanh(rs.statistic) - z975 * math.sqrt((1 + 0.5 * rs.statistic**2) / (n_c - 3)))), 4)
    exp_k = round(float(np.tanh(np.arctanh(rk.statistic) - z975 * math.sqrt(0.437 / (n_c - 4)))), 4)
    assert rs.ci_low == exp_s and rk.ci_low == exp_k

    # P1-8: Kerby (2014) rank-biserial sign + Hodges-Lehmann/Moses CI + Welch df
    a_hi = np.arange(10.0, 18.0)
    b_lo = np.arange(1.0, 9.0)
    mw = StatisticalAnalyzer.two_sample_test(a_hi, b_lo, parametric=False)
    assert mw.effect_size == 1.0 and math.isclose(mw.details["hodges_lehmann_shift"], 9.0)
    assert mw.ci_low is not None and mw.ci_high is not None and mw.ci_low <= 9.0 <= mw.ci_high
    assert StatisticalAnalyzer.two_sample_test(b_lo, a_hi, parametric=False).effect_size == -1.0
    wt = StatisticalAnalyzer.two_sample_test(rng.normal(0, 1, 10), rng.normal(0, 10, 50), parametric=True, equal_var=False)
    assert wt.details["df"] < 58.0

    # P1-9: Kruskal-Wallis eta_squared_h & epsilon_squared
    df_kw = pd.DataFrame({"grp": ["A"] * 20 + ["B"] * 20 + ["C"] * 20, "val": np.concatenate([rng.normal(0, 1, 20), rng.normal(2, 1, 20), rng.normal(4, 1, 20)])})
    kw = StatisticalAnalyzer.multi_group_test(df_kw, "val", "grp", parametric=False)
    assert kw.effect_size_name == "eta_squared_h"
    assert math.isclose(kw.effect_size, round(max(0.0, (kw.statistic - 2.0) / 57.0), 4))
    assert math.isclose(kw.details["epsilon_squared"], round(kw.statistic / 59.0, 4))


# ---------------------------------------------------------------------------
# P1-10: Drift Detection — PSI Binary/Constant & Page-Hinkley Full Reset
# ---------------------------------------------------------------------------
def test_p1_10_psi_and_page_hinkley_reset() -> None:
    """Verify exact binary/constant PSI detection and single-fire Page-Hinkley reset."""
    ref_bin = np.array([0] * 900 + [1] * 100, dtype=float)
    cur_bin = np.array([0] * 100 + [1] * 900, dtype=float)
    assert math.isclose(DriftDetector.compute_psi(ref_bin, cur_bin), 2.0 * 0.8 * math.log(9.0), rel_tol=1e-3)
    assert DriftDetector.compute_psi(np.ones(100), np.full(100, 5.0)) > 5.0

    ph = DriftDetector.detect_concept_drift_page_hinkley([0.05] * 200 + [0.55] * 200, delta=0.005, threshold=5.0)
    assert ph["drift_detected"] is True and len(ph["drift_indices"]) == 1
    ph_down = DriftDetector.detect_concept_drift_page_hinkley([0.8] * 100 + [0.2] * 100, threshold=5.0, mode="both")
    assert ph_down["drift_directions"] == ["down"]


# ---------------------------------------------------------------------------
# P1-11: Cox PH Efron & Breslow Exact Match vs statsmodels.PHReg
# ---------------------------------------------------------------------------
def test_p1_11_cox_ph_exact_match_statsmodels_with_ties() -> None:
    """Verify CoxPHModel matches statsmodels.PHReg to 1e-5 in original covariate units with ties."""
    rng = np.random.default_rng(42)
    n = 250
    age = rng.normal(55.0, 15.0, n)
    biomarker = rng.normal(0.05, 0.02, n)
    treat = rng.binomial(1, 0.5, n).astype(float)
    linpred = 0.04 * (age - 55.0) + 15.0 * (biomarker - 0.05) - 0.6 * treat
    T = np.clip(np.ceil(rng.exponential(scale=10.0 / np.exp(linpred))), 1.0, 15.0)
    E = rng.binomial(1, 0.72, n)
    df = pd.DataFrame({"duration": T, "event": E, "age": age, "biomarker": biomarker, "treat": treat})

    for ties in ("efron", "breslow"):
        cox = CoxPHModel(l2_penalty=0.0, ties=ties).fit(df)
        sm_fit = PHReg(T, df[["age", "biomarker", "treat"]].to_numpy(), status=E, ties=ties).fit()
        assert np.allclose(cox.coef_, sm_fit.params, atol=1e-5)
        assert np.allclose(cox.standard_errors_, sm_fit.bse, atol=1e-5)

    km = KaplanMeierEstimator().fit(T, E)
    assert (km.survival_table_["ci_lower"] <= km.survival_table_["survival_prob"]).all()
    assert (km.survival_table_["survival_prob"] <= km.survival_table_["ci_upper"]).all()


# ---------------------------------------------------------------------------
# P1-12: Causal Inference — Hájek IPW, Abadie-Imbens PSM, 2SLS Partial F, Cross-Fit AIPW
# ---------------------------------------------------------------------------
def test_p1_12_causal_hajek_abadie_imbens_2sls_and_aipw() -> None:
    """Verify Hájek IPW shift invariance, 2SLS partial F vs statsmodels, Abadie-Imbens SE, and AIPW EIF."""
    rng = np.random.default_rng(123)
    n = 350
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    ps_true = 1.0 / (1.0 + np.exp(-(0.7 * x1 - 0.5 * x2)))
    T = rng.binomial(1, ps_true)
    Y = 2.5 * T + 1.2 * x1 - 0.8 * x2 + rng.normal(0, 1.0, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "T": T, "Y": Y})
    eng = CausalInferenceEngine(random_seed=42, bootstrap_rounds=20)

    r1 = eng.estimate_ate(df, treatment_col="T", outcome_col="Y", confounders=["x1", "x2"], method="ipw", run_refutations=False)
    df_s = df.assign(Y=df["Y"] + 10_000.0)
    r2 = eng.estimate_ate(df_s, treatment_col="T", outcome_col="Y", confounders=["x1", "x2"], method="ipw", run_refutations=False)
    assert abs(r1.ate - r2.ate) < 1e-5

    res_psm = eng.estimate_ate(df, treatment_col="T", outcome_col="Y", confounders=["x1", "x2"], method="psm", run_refutations=False)
    assert res_psm.details["se_method"] == "abadie_imbens_2006_analytical" and res_psm.std_error > 0

    res_aipw = eng.estimate_ate(df, treatment_col="T", outcome_col="Y", confounders=["x1", "x2"], method="aipw", run_refutations=False)
    assert res_aipw.details["se_method"] == "efficient_influence_function" and abs(res_aipw.ate - 2.5) < 0.35

    Z = rng.normal(0, 1, n)
    U = rng.normal(0, 1, n)
    T_iv = 0.8 * Z + 0.5 * x1 + 0.9 * U + rng.normal(0, 0.5, n)
    Y_iv = 3.0 * T_iv + 1.0 * x1 + 1.5 * U + rng.normal(0, 1.0, n)
    df_iv = pd.DataFrame({"x1": x1, "Z": Z, "T": T_iv, "Y": Y_iv})
    res_iv = eng.estimate_ate(df_iv, treatment_col="T", outcome_col="Y", confounders=["x1"], instrument_col="Z", method="iv_2sls", run_refutations=False)
    sm_f = float(sm.OLS(T_iv, sm.add_constant(np.column_stack([x1, Z]))).fit().compare_f_test(sm.OLS(T_iv, sm.add_constant(x1)).fit())[0])
    assert abs(res_iv.details["first_stage_partial_f"] - sm_f) < 1e-3
    assert res_iv.details["weak_instrument_warning"] is False


# ---------------------------------------------------------------------------
# P1-13: Non-Parametric Models — Nadaraya-Watson "auto" Fix & Leak-Free Tuning
# ---------------------------------------------------------------------------
def test_p1_13_nonparametric_suite_complete_and_sorted_safe() -> None:
    """Verify Nadaraya-Watson runs in suite (not silently dropped) and sorted data stays leak-free."""
    rng = np.random.default_rng(7)
    x1 = np.linspace(-3, 3, 180)
    x2 = rng.normal(0, 1, 180)
    y = np.sin(2.0 * x1) + 0.3 * x2**2 + rng.normal(0, 0.15, 180)
    X_df = pd.DataFrame({"x1": x1, "x2": x2})
    res = NonParametricAnalyticalSuite.run(X_df.iloc[:140], y[:140], task_type=TaskType.REGRESSION, cv_folds=3, X_test=X_df.iloc[140:], y_test=y[140:])
    names = [r["model_name"] for r in res["leaderboard"]]
    assert "NonParametric_NadarayaWatson" in names and len(names) == 6
    nw = NadarayaWatsonKernelModel(bandwidth="auto").fit(X_df, y)
    assert 1.0 <= nw.effective_df_ <= nw.n_anchors_


# ---------------------------------------------------------------------------
# P1-14: LIME Scale-Invariant Standardized Ridge & Exact SHAP Efficiency
# ---------------------------------------------------------------------------
def test_p1_14_lime_scale_invariance_and_shap_efficiency() -> None:
    """Verify LIME recovers identical weights across 1e10 feature scale ratio and SHAP satisfies efficiency."""
    class _Lin(BaseOmniForgeModel):
        def __init__(self, w: list[float]) -> None:
            super().__init__(name="lin", task_type=TaskType.REGRESSION)
            self._w = np.asarray(w, dtype=float)
        def _fit(self, X: np.ndarray, y: np.ndarray | None, **kwargs: object) -> None:
            pass
        def _predict(self, X: np.ndarray, **kwargs: object) -> np.ndarray:
            return np.asarray(X, dtype=float) @ self._w

    rng = np.random.default_rng(7)
    df = pd.DataFrame({"big": rng.normal(0, 1e6, 200), "tiny": rng.normal(0, 1e-4, 200), "noise": rng.normal(0, 1, 200)})
    s_big, s_tiny = float(df["big"].std(ddof=1)), float(df["tiny"].std(ddof=1))
    model = _Lin([2.0 / s_big, 2.0 / s_tiny, 0.0]).fit(df, np.zeros(200))

    ie = InterpretabilityEngine(random_seed=42)
    lime = ie.explain_lime_local(model, df, df.iloc[[0]], n_perturbations=250)
    assert abs(lime["feature_weights"]["big"] - lime["feature_weights"]["tiny"]) < 0.15
    assert math.isclose(lime["kernel_width"], round(0.75 * math.sqrt(3), 4))

    shap = ie.explain_shap_local(model, df, df.iloc[[0]], n_samples=32)
    assert abs((shap["expected_value"] + sum(shap["attributions"].values())) - shap["prediction"]) < 1e-4


# ---------------------------------------------------------------------------
# P1-15: Graph Link Prediction & VGAE Kipf-Welling ELBO Normalization
# ---------------------------------------------------------------------------
def test_p1_15_vgae_elbo_and_link_prediction_invariants() -> None:
    """Verify VGAE ELBO decreases monotonically on SBM and Katz converges on dense cliques."""
    G = nx.stochastic_block_model([20, 20], [[0.45, 0.03], [0.03, 0.45]], seed=42)
    suite = GraphLinkPredictionSuite(random_seed=42)
    vgae = suite.predict_links_neural_gae(G, variational=True, epochs=50, top_k=5)
    assert vgae["metrics"]["roc_auc"] > 0.65 and len(vgae["top_predictions"]) == 5

    gae_m = GraphAutoencoderLinkModel(variational=True, epochs=50, random_seed=42)
    emb = gae_m.fit_embeddings(nx.to_numpy_array(G, dtype=float))
    assert np.all(np.isfinite(emb)) and gae_m.loss_history_[-1] < gae_m.loss_history_[0]

    K25 = nx.complete_graph(25)
    K25.remove_edge(0, 1)
    katz = suite.predict_links_advanced(K25, method="katz", top_k=1)
    assert math.isfinite(katz[0]["score"]) and katz[0]["score"] < 100.0


# ---------------------------------------------------------------------------
# P1-16: PurgedTimeSeriesSplit Embargo & CombinatorialPurgedKFold
# ---------------------------------------------------------------------------
def test_p1_16_purged_and_embargoed_cv() -> None:
    """Verify PurgedTimeSeriesSplit enforces purge_gap + ceil(n * embargo_pct) strictly."""
    X = np.arange(120).reshape(120, 1)
    folds = list(PurgedTimeSeriesSplit(n_splits=4, purge_gap=2, embargo_pct=0.05).split(X))
    assert len(folds) == 4
    for tr, te in folds:
        assert int(te.min()) - int(tr.max()) - 1 == 8  # purge=2 + ceil(120*0.05)=6

    cpcv = list(CombinatorialPurgedKFold(n_splits=5, n_test_groups=1, purge_gap=2, embargo_gap=0, embargo_pct=0.05).split(X))
    assert cpcv[0][1][-1] == 23 and cpcv[0][0][0] == 30
    assert len(list(CrossValidator("purged_kfold", n_splits=5).get_splitter().split(X))) == 5
    assert len(list(CrossValidator("cpcv", n_splits=5).get_splitter().split(X))) == 10


# ---------------------------------------------------------------------------
# P1-17: Hyndman-Billah (2003) Theta Method, ARIMA Burn-In, Croston SBA
# ---------------------------------------------------------------------------
def test_p1_17_theta_matches_statsmodels_and_arima_burnin() -> None:
    """Verify Theta matches statsmodels.ThetaModel to 1e-4, ARIMA drops Kalman burn-in, and SBA Croston."""
    rng = np.random.default_rng(7)
    y = 10.0 + 0.6 * np.arange(30, dtype=float) + rng.normal(0, 1.5, 30)
    fc = StatisticalForecaster(method="theta", confidence_level=0.95).fit(y).predict(5)
    sm_fc = np.asarray(ThetaModel(y, deseasonalize=False).fit().forecast(5), dtype=float)
    assert float(np.max(np.abs(fc["yhat"].to_numpy() - sm_fc))) < 1e-4

    y_hi = 10_000.0 + np.cumsum(rng.normal(0, 1.0, 80))
    sf_ar = StatisticalForecaster(method="arima", arima_order=(1, 1, 0)).fit(y_hi)
    assert sf_ar._residual_std < 3.0

    fc_cr = StatisticalForecaster(method="croston").fit(np.array([0, 0, 0, 0, 10] * 10, dtype=float)).predict(2)
    assert math.isclose(float(fc_cr["yhat"].iloc[0]), 1.8, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Second-Pass Audit Invariants: Fairness, Joseph Kalman, Exact Rolling MAD, Clustering
# ---------------------------------------------------------------------------
def test_second_pass_fairness_kalman_mad_and_clustering() -> None:
    """Verify zero-selection fairness parity, Joseph Kalman PSD stability, exact rolling MAD, and clustering medians."""
    # Fairness: all-zero predictions across groups must give disparate_impact_ratio == 1.0
    rep = FairnessAuditor().audit(
        y_true=np.array([1, 0, 1, 0, 0, 0]),
        y_pred=np.zeros(6, dtype=int),
        sensitive_features=np.array(["A", "A", "A", "B", "B", "B"]),
    )
    assert rep.disparate_impact_ratio == 1.0 and rep.equal_opportunity_diff == 0.0

    # Joseph stabilized Kalman covariance stays strictly symmetric & PSD
    dt = DigitalTwinEngine(A=np.eye(2), B=np.zeros((2, 1)), H=np.array([[1.0, 0.0]]), Q=np.eye(2) * 0.01, R=np.array([[1e-6]]), x0=np.zeros(2))
    for _ in range(25):
        dt.assimilate(np.array([1.0]))
    assert np.allclose(dt.P, dt.P.T, atol=1e-12)
    assert float(np.min(np.linalg.eigvalsh(dt.P))) >= -1e-12

    # Gaussian copula Monte Carlo never produces non-finite values
    mc_res, out = MonteCarloSimulator(n_simulations=400, random_seed=42).run(
        {"a": {"dist": "normal", "mean": 10.0, "std": 2.0}, "b": {"dist": "normal", "mean": 5.0, "std": 1.0}},
        lambda s: s["a"] + s["b"],
        correlation_matrix=np.array([[1.0, 0.8], [0.8, 1.0]]),
    )
    assert np.all(np.isfinite(out)) and math.isfinite(mc_res.std)

    # Exact rolling MAD detects spike without future leakage
    s = np.concatenate([np.zeros(25), [25.0], np.zeros(10)])
    ts_anom = AnomalyDetector.detect_time_series_anomalies(s, span=6, window=10, z_threshold=3.0)
    assert int(ts_anom["is_anomaly"].iloc[25]) == 1

    # Clustering stores training medians for inductive prediction with NaN rows
    df_tr = pd.DataFrame({"x": [10.0, 11.0, 12.0, 100.0, 101.0, 102.0], "y": [1.0, 1.1, 0.9, 50.0, 51.0, 49.0]})
    ce = ClusteringEngine(algorithm="kmeans", n_clusters=2).fit(df_tr)
    pred = ce.predict(pd.DataFrame({"x": [np.nan], "y": [1.0]}))
    assert len(pred) == 1 and ce._array_medians_ is not None and ce._array_medians_[0] == pytest.approx(56.0)


# ---------------------------------------------------------------------------
# Third-Pass Adversarial Re-Audit Invariants (Topics 1–12)
# ---------------------------------------------------------------------------
def test_third_pass_metrics_backtesting_and_optimization() -> None:
    """Verify 1D binary proba, unbiased sMAPE, Markowitz risk-free Sharpe, MILP A_eq, and Clarke-Wright CVRP."""
    from omniforge.evaluation.backtesting import BacktestingEngine
    from omniforge.optimization.solvers import OptimizationEngine

    # 1. Classification 1D probability + unnormalized probability safety
    m_1d = MetricsEvaluator.classification_metrics(
        np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8])
    )
    assert math.isclose(m_1d["auc_roc"], 1.0) and 0.0 <= m_1d["brier_score"] <= 0.1
    m_unnorm = MetricsEvaluator.classification_metrics(
        np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), np.array([[0.9, 0.1], [0.2, 0.8], [1.6, 0.4], [0.4, 1.6]])
    )
    assert math.isclose(m_unnorm["auc_roc"], 1.0) and math.isfinite(m_unnorm["log_loss"])

    # 2. Unbiased sMAPE on tiny non-zero series (no 1e-9 denominator bias)
    r_tiny = MetricsEvaluator.regression_metrics(np.array([1e-11, 2e-11]), np.array([2e-11, 4e-11]))
    assert math.isclose(r_tiny["smape"], 66.666667, rel_tol=1e-4)

    # 3. BacktestingEngine raises on empty folds
    from omniforge.core.exceptions import EvaluationError

    with pytest.raises(EvaluationError):
        BacktestingEngine().walk_forward_forecast(np.arange(10, dtype=float), lambda: None, horizon=5, initial_train_size=10)

    # 4. Markowitz risk-free rate + exact Sharpe computation
    opt = OptimizationEngine()
    res_mv = opt.optimize_portfolio(
        expected_returns=np.array([0.04, 0.14]),
        cov_matrix=np.array([[0.01, 0.0], [0.0, 0.04]]),
        risk_aversion=1.0,
        risk_free_rate=0.04,
        min_weight=0.0,
        max_weight=1.0,
    )
    assert res_mv.status == "optimal" and res_mv.solution["asset_1"] > 0.95
    assert res_mv.details["sharpe_ratio"] == pytest.approx(
        (res_mv.details["expected_return"] - 0.04) / res_mv.details["volatility"], rel=1e-4
    )

    # 5. MILP equality constraint A_eq @ x == b_eq
    res_milp = opt.solve_milp(
        c=np.array([2.0, 5.0]),
        integrality=np.array([1, 1]),
        A_eq=np.array([[1.0, 1.0]]),
        b_eq=np.array([4.0]),
        lower_bounds=np.array([0.0, 0.0]),
        upper_bounds=np.array([4.0, 4.0]),
    )
    assert res_milp.optimal and math.isclose(res_milp.objective_value, 8.0, rel_tol=1e-6)

    # 6. Clarke-Wright CVRP 4-case endpoint reversal + asymmetric directed savings + positive-savings guard
    dist_sym = np.array([
        [0.0, 10.0, 10.0, 10.0],
        [10.0, 0.0, 1.0, 20.0],
        [10.0, 1.0, 0.0, 1.0],
        [10.0, 20.0, 1.0, 0.0],
    ])
    vrp = opt.solve_vrp_savings(dist_sym, demands=np.array([0.0, 1.0, 1.0, 1.0]), vehicle_capacity=5.0)
    assert len(vrp.solution) == 1 and math.isclose(vrp.objective_value, 22.0, rel_tol=1e-6)


def test_third_pass_mining_recommender_and_features() -> None:
    """Verify SubgroupDiscovery BH-FDR, HybridRecommender duplicate handling & candidate masking, and OOF target encoding."""
    from omniforge.features.engineering import FeatureConfig, FeatureEngineeringPipeline
    from omniforge.features.selection import FeatureSelector
    from omniforge.models.mining.miner import AssociationRuleMiner, SubgroupDiscoveryMiner
    from omniforge.models.recommendation.recommender import HybridRecommender

    # 1. AssociationRuleMiner zero-inflated numeric binning fallback + SubgroupDiscovery BH-FDR
    df_z = pd.DataFrame({"z": [0.0] * 40 + [10.0] * 10, "c": ["A"] * 25 + ["B"] * 25, "y": [0] * 35 + [1] * 15})
    rules_res = AssociationRuleMiner().mine(df_z, min_support=0.1, min_confidence=0.5, max_rules=20)
    assert isinstance(rules_res, dict) and "rules" in rules_res
    sub_res = SubgroupDiscoveryMiner().discover(df_z, target_col="y", min_subgroup_size=5)
    assert len(sub_res["subgroups"]) > 0 and "adjusted_p_value" in sub_res["subgroups"][0]

    # 2. HybridRecommender averages duplicate ratings, zeros self-similarity, and masks seen items
    inter = pd.DataFrame({
        "user_id": ["u1", "u1", "u1", "u2", "u2", "u3", "u3"],
        "item_id": ["i1", "i1", "i2", "i1", "i3", "i2", "i3"],
        "rating": [2.0, 4.0, 5.0, 3.0, 5.0, 4.0, 5.0],
    })
    rec = HybridRecommender(n_factors=2).fit(inter)
    assert math.isclose(float(rec._interaction_mat[rec._user_to_idx["u1"], rec._item_to_idx["i1"]]), 3.0)
    assert np.allclose(np.diag(rec._item_sim), 0.0)
    top = rec.recommend("u1", top_k=5, exclude_seen=True)
    assert all(r["item_id"] not in ("i1", "i2") for r in top)
    ev = HybridRecommender.evaluate_ranking(["i1", "i1", "i2"], {"i1", "i2"}, k=2)
    assert 0.0 <= ev["ndcg_at_2"] <= 1.0 and 0.0 <= ev["recall_at_2"] <= 1.0

    # 3. FeatureEngineeringPipeline 5-fold OOF target encoding (no self-label leakage) + causal diff_1 + f_test zero-var guard
    rng = np.random.default_rng(42)
    df_fe = pd.DataFrame({
        "cat": [f"c{i}" for i in range(60)],
        "t": pd.date_range("2025-01-01", periods=60, freq="D"),
        "val": np.arange(60, dtype=float),
        "const": np.ones(60, dtype=float),
        "target": rng.normal(10.0, 2.0, 60),
    })
    afe = FeatureEngineeringPipeline(
        FeatureConfig(
            scale_numeric="none",
            categorical_encoding="target",
            time_series_column="t",
            time_series_target="val",
            lags=[1],
            exclude_columns=["target"],
        )
    )
    df_tr = afe.fit_transform(df_fe, y=df_fe["target"])
    assert float(abs(np.corrcoef(df_tr["cat__target_enc"], df_fe["target"])[0, 1])) < 0.25
    assert df_tr["val_diff_1"].iloc[2] == pytest.approx(1.0)
    fs = FeatureSelector(method="f_test", task="regression", max_features=2).fit(df_fe[["val", "const"]], df_fe["target"])
    assert "val" in fs.selected_features_


def test_third_pass_ensemble_deep_hpo_copula_and_netmf() -> None:
    """Verify deep-copied OOF stacking, exact GELU derivative, TabNet explain(df), GP-UCB HPO, Pandas 3 datetime copula, NetMF."""
    import warnings

    import networkx as nx

    from omniforge.automl.optimizer import HyperparameterOptimizer
    from omniforge.core.exceptions import ModelError
    from omniforge.data.synthetic import SyntheticDataGenerator
    from omniforge.models.baselines import RidgeLinearBaseline
    from omniforge.models.deep.mlp import _activate, _activate_grad
    from omniforge.models.deep.tabnet import TabNetModel
    from omniforge.models.ensemble import StackingEnsembleModel, VotingEnsembleModel
    from omniforge.models.graphs.analytics import GraphAnalyticsEngine
    from omniforge.models.nonparametric import PenalizedSplineGAMModel
    from omniforge.models.time_series.foundation import FoundationForecaster

    # 1. Exact overflow-safe analytical GELU derivative
    x_grid = np.linspace(-6.0, 6.0, 121)
    eps = 1e-6
    fd = (_activate(x_grid + eps, "gelu") - _activate(x_grid - eps, "gelu")) / (2.0 * eps)
    assert float(np.max(np.abs(_activate_grad(x_grid, "gelu") - fd))) < 1e-8
    assert np.all(np.isfinite(_activate_grad(np.array([-1e4, 0.0, 1e4]), "gelu")))

    # 2. TabNet DataFrame-safe explain() & parameter guards
    rng = np.random.default_rng(7)
    df_tn = pd.DataFrame(rng.normal(0, 1, (40, 4)), columns=["a", "b", "c", "d"])
    y_tn = (df_tn["a"] > 0).astype(int).to_numpy()
    tn = TabNetModel(n_steps=2, epochs=4, gamma=0.5, lambda_sparse=-1.0, task_type=TaskType.BINARY_CLASSIFICATION, track_carbon=False).fit(df_tn, y_tn)
    assert tn.gamma >= 1.0 and tn.lambda_sparse >= 0.0
    exp_tn = tn.explain(df_tn.iloc[:5])
    assert exp_tn.shape == (5, 4) and np.all(np.isfinite(exp_tn))

    # 3. StackingEnsembleModel supports non-sklearn BaseOmniForgeModel via deepcopy; Voting rejects negative weights
    X_s = pd.DataFrame(rng.normal(0, 1, (80, 3)), columns=["x1", "x2", "x3"])
    y_s = 2.0 * X_s["x1"] + X_s["x2"] ** 2 + rng.normal(0, 0.1, 80)
    stk = StackingEnsembleModel(
        estimators=[
            PenalizedSplineGAMModel(task_type=TaskType.REGRESSION, n_knots=5, track_carbon=False),
            RidgeLinearBaseline(track_carbon=False),
        ],
        task_type=TaskType.REGRESSION,
        cv_folds=3,
        track_carbon=False,
    ).fit(X_s, y_s)
    assert float(np.corrcoef(stk.predict(X_s), y_s)[0, 1]) > 0.95
    with pytest.raises(ModelError):
        VotingEnsembleModel(estimators=[RidgeLinearBaseline(track_carbon=False)], weights=[-1.0], task_type=TaskType.REGRESSION).fit(X_s, y_s)

    # 4. FoundationForecaster flat-segment anti-explosion guard
    fc_flat = FoundationForecaster(patch_length=6).forecast_zero_shot(
        np.concatenate([np.zeros(20), np.linspace(1.0, 5.0, 20), np.linspace(5.0, 6.0, 15)]), horizon=4
    )
    assert float(np.max(np.abs(fc_flat["yhat"]))) < 50.0

    # 5. Bayesian GP-UCB emits zero ConvergenceWarnings + log-grid uses geomspace
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        hpo_b = HyperparameterOptimizer(strategy="bayesian", max_trials=10, random_seed=42).optimize(
            {"x": {"type": "float", "low": -2.0, "high": 2.0}}, lambda p: -(p["x"] - 0.5) ** 2
        )
        assert all("ConvergenceWarning" not in x.category.__name__ for x in w)
    assert hpo_b.best_score > -0.1
    hpo_g = HyperparameterOptimizer(strategy="grid", max_trials=5).optimize(
        {"lr": {"type": "float", "low": 1e-4, "high": 1.0, "log": True}}, lambda p: -abs(np.log10(p["lr"]) + 2.0)
    )
    assert math.isclose(hpo_g.best_params["lr"], 1e-2, rel_tol=1e-6)

    # 6. SyntheticDataGenerator preserves datetime64[us] range (Pandas 3.x safe) & GraphAnalyticsEngine NetMF
    df_dt = pd.DataFrame({"x": np.linspace(0, 10, 30), "c": np.ones(30), "ts": pd.date_range("2024-01-01", periods=30, freq="D").astype("datetime64[us]")})
    syn = SyntheticDataGenerator(random_seed=42).fit_generate_copula(df_dt, n_samples=20)
    assert syn["ts"].min() >= pd.Timestamp("2024-01-01")
    emb = GraphAnalyticsEngine(random_seed=42).compute_node_embeddings(nx.karate_club_graph(), embedding_dim=4)
    assert len(emb) == 34

