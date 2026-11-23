"""Phase 3 Econometric, Conformal, Causal, DuckDB Confinement & Governance verification suite (OmniForge v4.0.0).

Verifies literature-exact implementations across all Phase 3 pillars:
1. Canonical SemVer 4.0.0 alignment across SDK, pyproject, native hubs, exporters & SPA manifests.
2. Exact directional CIs, non-central t power/sample size (Cohen 1988), Székely-Rizzo (2013, 2014)
   unbiased U-centered distance correlation, Demšar (2006) Studentized-range Nemenyi + Iman-Davenport F_F,
   AIC/BIC distribution ranking, and strict 2x2 Fisher's exact guard.
3. Romano, Sesia & Candès (2020) randomized APS, Mondrian group-conditional conformal calibration,
   and Gibbs & Candès (2021) strictly causal unclipped Adaptive Conformal Inference (ACI).
4. Cinelli & Hazlett (2020) omitted-variable sensitivity bounds (R^2_{Y~T|X}, RV_q, RV_{q,alpha}),
   Austin (2009) propensity-weighted SMD balance, regular DeepForecaster grid preservation, and IV wiring.
5. Harvey-Leybourne-Newbold (1997) modified Diebold-Mariano HAC test, Law of Total Variance forecast
   ensembles, Kruskal (1958) Spearman-to-Gaussian copula, Wachter/DiCE inverse-MAD L1 counterfactuals,
   and Ledoit-Wolf (2004) portfolio covariance shrinkage.
6. DuckDB engine confinement (`enable_external_access=False`, workspace path confinement, scan denylist)
   and NIST AI RMF ModelRegistry accountability metadata, fairness promotion gate, and automated SLA rollback.
"""

from __future__ import annotations

import json
import math
import sqlite3
import tomllib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from scipy import stats as sp_stats

import omniforge
from omniforge.api.studios.s04_stats import _run_single_stat_test
from omniforge.automl.backends import probe_automl_backends
from omniforge.causal.inference import CausalInferenceEngine
from omniforge.core.exceptions import DataIngestionError, MLOpsError, SecurityError
from omniforge.core.types import ModelStage, TaskType
from omniforge.data.connectors import sql_inspect_database, sql_load_dataframe
from omniforge.data.ingestion import DataIngestionEngine, fast_read_tabular
from omniforge.eda.statistics import StatisticalAnalyzer, distance_correlation, friedman_nemenyi_test
from omniforge.evaluation.fairness import FairnessReport
from omniforge.evaluation.interpretability import InterpretabilityEngine
from omniforge.evaluation.robustness import ConformalPredictor, adaptive_conformal_intervals
from omniforge.forecasting.engine import ForecastingEngine, diebold_mariano_test
from omniforge.mlops.continuous_training import ContinuousTrainingOrchestrator
from omniforge.mlops.monitoring import ModelMonitor
from omniforge.mlops.registry import ModelRegistry
from omniforge.mlops.tracking import ExperimentTracker
from omniforge.models.classical import ClassicalMLModel
from omniforge.models.computer_vision.foundation import probe_vision_backends
from omniforge.models.nlp.advanced_nlp import probe_nlp_backends
from omniforge.optimization.solvers import OptimizationEngine
from omniforge.simulation.engines import MonteCarloSimulator as MCSimulator
from omniforge.version import get_version_info

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_3_1_0_canonical_alignment() -> None:
    """Verify canonical v4.0.0 version across SDK, pyproject, SPA manifests, and native hubs."""
    assert omniforge.__version__ == "4.0.0"
    assert get_version_info()["version"] == "4.0.0"

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == "4.0.0"

    pkg_json = json.loads((REPO_ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
    pkg_lock = json.loads((REPO_ROOT / "frontend" / "package-lock.json").read_text(encoding="utf-8"))
    assert pkg_json["version"] == "4.0.0"
    assert pkg_lock["version"] == "4.0.0"
    assert pkg_lock["packages"][""]["version"] == "4.0.0"

    native_automl = {b["id"]: b["version"] for b in probe_automl_backends() if b["category"].startswith("Native")}
    assert all(v == "4.0.0" for v in native_automl.values())
    vis_native = [b for b in probe_vision_backends()["backends"] if b["id"] == "native_pyramid"][0]
    nlp_native = [b for b in probe_nlp_backends() if b["id"] == "native_nlp"][0]
    assert vis_native["version"] == "4.0.0"
    assert nlp_native["version"] == "4.0.0"


def test_exact_directional_cis_and_noncentral_t_power() -> None:
    """Verify exact one-sided Student/Welch CIs and exact scipy.stats.nct power/sample size."""
    rng = np.random.default_rng(101)
    x = pd.Series(rng.normal(loc=2.5, scale=1.2, size=25))
    y = pd.Series(rng.normal(loc=1.1, scale=1.7, size=30))

    # One-sample one-sided CIs vs SciPy reference
    for alt in ("two-sided", "greater", "less"):
        res_1 = StatisticalAnalyzer.one_sample_t_test(x, popmean=1.0, alpha=0.05, alternative=alt)
        ref_1 = sp_stats.ttest_1samp(x.to_numpy(), popmean=1.0, alternative=alt).confidence_interval(0.95)
        if math.isfinite(ref_1.low):
            assert res_1.ci_low == pytest.approx(float(ref_1.low), abs=1e-4)
        else:
            assert res_1.ci_low == -math.inf
        if math.isfinite(ref_1.high):
            assert res_1.ci_high == pytest.approx(float(ref_1.high), abs=1e-4)
        else:
            assert res_1.ci_high == math.inf

    # Two-sample Welch one-sided CIs vs SciPy reference
    for alt in ("two-sided", "greater", "less"):
        res_2 = StatisticalAnalyzer.two_sample_test(x, y, parametric=True, equal_var=False, alpha=0.05, alternative=alt)
        ref_2 = sp_stats.ttest_ind(x.to_numpy(), y.to_numpy(), equal_var=False, alternative=alt).confidence_interval(0.95)
        if math.isfinite(ref_2.low):
            assert res_2.ci_low == pytest.approx(float(ref_2.low), abs=1e-4)
        else:
            assert res_2.ci_low == -math.inf
        if math.isfinite(ref_2.high):
            assert res_2.ci_high == pytest.approx(float(ref_2.high), abs=1e-4)
        else:
            assert res_2.ci_high == math.inf

    # Cohen (1988) / G*Power exact non-central t two-sample sample sizes (alpha=0.05, power=0.80, two-sided)
    expected_n = {0.2: 394, 0.5: 64, 0.8: 26, 1.2: 12, 1.5: 9}
    for d_val, n_ref in expected_n.items():
        pw_res = StatisticalAnalyzer.two_sample_power_sample_size(effect_size=d_val, alpha=0.05, power=0.80)
        assert pw_res["n_per_group"] == n_ref
        assert pw_res["achieved_power"] >= 0.80


def test_unbiased_distance_correlation_and_demsar_nemenyi() -> None:
    """Verify Székely-Rizzo (2013, 2014) U-centered dCor, t-test/permutation p-values, and Demšar Nemenyi."""
    rng = np.random.default_rng(2026)
    # Under independence, biased V-statistic dCor > 0 while unbiased U-centered dCor^2 centers at 0
    biased_vals = []
    unbiased_vals = []
    for _ in range(15):
        u = pd.Series(rng.normal(size=45))
        v = pd.Series(rng.normal(size=45))
        res_uv = distance_correlation(u, v)
        biased_vals.append(float(res_uv["dcor"] or 0.0))
        unbiased_vals.append(float(res_uv["unbiased_dcor_sq"] or 0.0))
    assert float(np.mean(biased_vals)) > 0.15
    assert abs(float(np.mean(unbiased_vals))) < 0.05

    # Nonlinear parabolic dependence Y = X^2 with near-zero Pearson r
    x_nl = pd.Series(rng.uniform(-2.0, 2.0, size=90))
    y_nl = pd.Series(x_nl.to_numpy() ** 2 + rng.normal(0.0, 0.08, size=90))
    dc = distance_correlation(x_nl, y_nl, n_permutations=199, random_seed=7)
    assert dc["unbiased_dcor"] is not None and float(dc["unbiased_dcor"]) > 0.40
    assert dc["p_value"] is not None and float(dc["p_value"]) < 1e-4
    assert dc["permutation_p_value"] is not None and float(dc["permutation_p_value"]) <= 0.02

    # Demšar (2006) Table 5 exact Studentized-range Nemenyi critical values + k > 10 support
    blocks = rng.normal(0.0, 0.2, size=(20, 1))
    scores_5 = pd.DataFrame(
        blocks + np.array([[0.90, 0.85, 0.78, 0.65, 0.50]]) + rng.normal(0.0, 0.02, size=(20, 5)),
        columns=["m1", "m2", "m3", "m4", "m5"],
    )
    fn5 = friedman_nemenyi_test(scores_5, alpha=0.05)
    assert fn5["q_alpha"] == pytest.approx(2.728, abs=1e-3)
    assert fn5["significant"] is True
    assert fn5["iman_davenport_f"] > fn5["friedman_stat"] / 4.0

    # Verify k = 12 (previously clipped to k=10!) and alpha = 0.01 work via exact studentized_range
    scores_12 = pd.DataFrame(rng.normal(size=(15, 12)), columns=[f"alg_{i}" for i in range(12)])
    fn12 = friedman_nemenyi_test(scores_12, alpha=0.01)
    expected_q12 = float(sp_stats.studentized_range.ppf(0.99, 12, np.inf) / math.sqrt(2.0))
    assert fn12["q_alpha"] == pytest.approx(expected_q12, abs=1e-4)


def test_s04_aic_distribution_ranking_and_fisher_2x2_guard() -> None:
    """Verify AIC-based distribution fit ranking and strict HTTP 400 guard on R x C > 2x2 Fisher exact."""
    rng = np.random.default_rng(55)
    df_stat = pd.DataFrame(
        {
            "pos_skew": rng.lognormal(mean=1.2, sigma=0.75, size=350),
            "cat3_a": rng.choice(["A", "B", "C"], size=350),
            "cat3_b": rng.choice(["X", "Y", "Z"], size=350),
        }
    )
    fit_res = _run_single_stat_test(df_stat, test_type="distribution_fit", col_a="pos_skew", col_b=None, alpha=0.05)
    assert fit_res["extra_table"][0]["distribution"] == "Log-Normal"
    assert "aic" in fit_res["extra_table"][0] and "bic" in fit_res["extra_table"][0]
    aics = [row["aic"] for row in fit_res["extra_table"]]
    assert aics == sorted(aics)

    with pytest.raises(HTTPException) as exc_info:
        _run_single_stat_test(df_stat, test_type="fisher_exact", col_a="cat3_a", col_b="cat3_b", alpha=0.05)
    assert exc_info.value.status_code == 400
    assert "2x2" in str(exc_info.value.detail)


def test_conformal_aps_mondrian_and_strictly_causal_aci() -> None:
    """Verify Romano et al. (2020) APS, Mondrian conditional calibration, and Gibbs-Candès (2021) causal ACI."""
    rng = np.random.default_rng(303)
    # 3-class classification for APS conformal prediction sets
    n = 600
    X_cls = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    logits = np.column_stack([1.8 * X_cls["x1"], 1.8 * X_cls["x2"], -1.2 * (X_cls["x1"] + X_cls["x2"])])
    probs_true = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_true /= probs_true.sum(axis=1, keepdims=True)
    y_cls = pd.Series([rng.choice([0, 1, 2], p=p) for p in probs_true])

    clf = ClassicalMLModel("logistic_regression", task_type=TaskType.MULTICLASS_CLASSIFICATION).fit(
        X_cls.iloc[:250], y_cls.iloc[:250]
    )
    cp_aps = ConformalPredictor(clf, alpha=0.10, score_method="aps", allow_empty=False, random_seed=11).calibrate(
        X_cls.iloc[250:450], y_cls.iloc[250:450]
    )
    sets_aps = cp_aps.predict_sets(X_cls.iloc[450:])
    y_test_cls = y_cls.iloc[450:].to_numpy()
    emp_aps_cov = float(np.mean([yt in s for yt, s in zip(y_test_cls, sets_aps, strict=True)]))
    assert emp_aps_cov >= 0.84
    assert all(len(s) >= 1 for s in sets_aps)

    # Mondrian group-conditional regression conformal under severe group heteroscedasticity
    n_reg = 700
    grp = np.where(np.arange(n_reg) % 5 == 0, "high_noise_minority", "low_noise_majority")
    x_r = rng.uniform(-2.0, 2.0, size=n_reg)
    noise_scale = np.where(grp == "high_noise_minority", 4.5, 0.25)
    y_r = pd.Series(2.0 * x_r + rng.normal(0.0, noise_scale))
    X_r = pd.DataFrame({"x": x_r})

    reg = ClassicalMLModel("ridge_regressor", task_type=TaskType.REGRESSION).fit(X_r.iloc[:200], y_r.iloc[:200])
    cp_marg = ConformalPredictor(reg, alpha=0.10).calibrate(X_r.iloc[200:450], y_r.iloc[200:450])
    cp_mond = ConformalPredictor(reg, alpha=0.10).calibrate(
        X_r.iloc[200:450], y_r.iloc[200:450], groups=grp[200:450]
    )
    X_te, y_te, g_te = X_r.iloc[450:], y_r.iloc[450:].to_numpy(), grp[450:]
    min_mask = g_te == "high_noise_minority"
    marg_res = cp_marg.predict_intervals(X_te)
    mond_res = cp_mond.predict_intervals(X_te, groups=g_te)
    marg_min_cov = float(np.mean((y_te[min_mask] >= marg_res["lower"][min_mask]) & (y_te[min_mask] <= marg_res["upper"][min_mask])))
    mond_min_cov = float(np.mean((y_te[min_mask] >= mond_res["lower"][min_mask]) & (y_te[min_mask] <= mond_res["upper"][min_mask])))
    assert marg_min_cov <= 0.65  # Marginal conformal severely under-covers minority high-noise group
    assert mond_min_cov >= 0.80  # Mondrian conditional conformal restores target coverage
    assert mond_min_cov - marg_min_cov >= 0.20

    # Strictly causal Gibbs & Candès (2021) ACI: mutating y_true[t] MUST NOT alter interval[t]!
    t_len = 160
    preds_stream = np.zeros(t_len)
    y_stream_a = np.concatenate([rng.normal(0.0, 0.5, 80), rng.normal(0.0, 3.0, 80)])
    y_stream_b = y_stream_a.copy()
    y_stream_b[120] = 1.0e9  # Massive future shock at step 120

    aci_a = adaptive_conformal_intervals(y_stream_a, preds_stream, alpha=0.10, gamma=0.05)
    aci_b = adaptive_conformal_intervals(y_stream_b, preds_stream, alpha=0.10, gamma=0.05)
    # At step 120, prediction interval MUST be identical before y_true[120] is revealed!
    assert float(aci_a["lower"][120]) == float(aci_b["lower"][120])
    assert float(aci_a["upper"][120]) == float(aci_b["upper"][120])
    # Only at step 121 (after observing y_true[120]) does interval widen in response to the miss
    width_a_121 = float(aci_a["upper"][121] - aci_a["lower"][121])
    width_b_121 = float(aci_b["upper"][121] - aci_b["lower"][121])
    assert width_b_121 >= width_a_121
    assert abs(aci_a["full_run_coverage"] - 0.90) <= aci_a["theoretical_coverage_error_bound"] + 0.05


def test_cinelli_hazlett_sensitivity_and_austin_smd_balance() -> None:
    """Verify Cinelli & Hazlett (2020) exact RV_q / RV_{q,alpha} benchmark and Austin (2009) SMD balance."""
    # Published benchmark: t = 9.78, df = 783 -> R^2_{Y~D|X} = 0.108859, RV_q = 0.293727, RV_{q,0.05} = 0.243049
    sens = CausalInferenceEngine.compute_sensitivity_bounds(ate=9.78, std_error=1.0, dof=783, alpha=0.05)
    assert sens["partial_r2_treatment_outcome"] == pytest.approx(0.108859, abs=1e-4)
    assert sens["robustness_value_q"] == pytest.approx(0.293727, abs=1e-3)
    assert sens["robustness_value_q_alpha"] == pytest.approx(0.243049, abs=1e-3)

    # Confounded observational data: verify ATE recovery, sensitivity attachment, and SMD reduction
    rng = np.random.default_rng(42)
    n = 600
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)
    p_tr = 1.0 / (1.0 + np.exp(-(1.1 * x1 - 0.8 * x2)))
    tr = (rng.uniform(size=n) < p_tr).astype(int)
    y = 2.5 * tr + 1.4 * x1 - 1.0 * x2 + rng.normal(0.0, 0.6, size=n)
    df_c = pd.DataFrame({"treat": tr, "outcome": y, "x1": x1, "x2": x2})

    eng = CausalInferenceEngine(random_seed=42)
    ate_res = eng.estimate_ate(
        df_c, treatment_col="treat", outcome_col="outcome", confounders=["x1", "x2"], method="aipw", run_refutations=True
    )
    assert ate_res.ate == pytest.approx(2.5, abs=0.25)
    assert ate_res.details["sensitivity_rv_q"] > 0.20
    assert ate_res.details["max_abs_smd_unadjusted"] > 0.50
    assert ate_res.details["max_abs_smd_adjusted"] < 0.15
    assert ate_res.details["covariate_balance_achieved"] is True
    assert "unobserved_confounder_sensitivity" in ate_res.refutations
    assert ate_res.refutations["unobserved_confounder_sensitivity"]["robustness_value_q_alpha"] > 0.15


def test_diebold_mariano_total_variance_spearman_copula_wachter_and_ledoit_wolf() -> None:
    """Verify HLN Diebold-Mariano, Law of Total Variance ensemble, Spearman copula, Wachter CF, and Ledoit-Wolf."""
    rng = np.random.default_rng(99)
    y_true_dm = rng.normal(10.0, 1.0, size=80)
    pred_good = y_true_dm + rng.normal(0.0, 0.35, size=80)
    pred_bad = y_true_dm + rng.normal(0.0, 1.60, size=80)
    dm = diebold_mariano_test(y_true_dm, pred_good, pred_bad, horizon=3, loss="squared")
    assert dm["hln_statistic"] < -2.5
    assert dm["p_value"] < 0.02
    assert dm["preferred_model"] == "model_1"

    dm_tie = diebold_mariano_test(y_true_dm, pred_good, pred_good, horizon=1)
    assert dm_tie["dm_statistic"] == 0.0 and dm_tie["p_value"] == 1.0

    # ForecastingEngine attaches Diebold-Mariano & Law of Total Variance intervals
    t_idx = pd.date_range("2025-01-01", periods=96, freq="D")
    ts_df = pd.DataFrame({"ds": t_idx, "y": 50.0 + 0.3 * np.arange(96) + 6.0 * np.sin(2 * np.pi * np.arange(96) / 7.0) + rng.normal(0.0, 0.8, 96)})
    fc_res = ForecastingEngine(seasonal_period=7, include_deep=False, include_foundation=False).forecast(
        ts_df, target_col="y", time_col="ds", horizon=10
    )
    assert fc_res.metadata["ensemble_variance_method"] == "law_of_total_variance_mixture"
    assert fc_res.metadata["diebold_mariano_vs_runner_up"] is not None
    assert fc_res.ensemble_forecast is not None
    assert all(
        hi > lo
        for lo, hi in zip(
            fc_res.ensemble_forecast["yhat_lower"].to_numpy(),
            fc_res.ensemble_forecast["yhat_upper"].to_numpy(),
            strict=True,
        )
    )

    # Kruskal (1958) Spearman-to-Gaussian copula on non-Gaussian marginals (lognormal + beta)
    captured_draws: dict[str, np.ndarray] = {}

    def _capture_fn(d: dict[str, np.ndarray]) -> np.ndarray:
        captured_draws.update(d)
        return d["sev"] * d["rate"]

    mc_res, _ = MCSimulator(n_simulations=15_000, random_seed=123).run(
        input_specs={
            "sev": {"dist": "lognormal", "mean": 1.0, "sigma": 0.9},
            "rate": {"dist": "beta", "a": 2.0, "b": 5.0},
        },
        model_fn=_capture_fn,
        correlation_matrix=np.array([[1.0, 0.75], [0.75, 1.0]]),
        correlation_type="spearman",
    )
    assert mc_res.n_simulations == 15_000
    emp_spearman = float(sp_stats.spearmanr(captured_draws["sev"], captured_draws["rate"]).statistic)
    assert emp_spearman == pytest.approx(0.75, abs=0.025)

    # Wachter / DiCE inverse-MAD L1 counterfactual with immutable_features & domain bounds
    X_bg = pd.DataFrame(
        {
            "age": rng.uniform(22.0, 65.0, size=200),
            "income": rng.uniform(30.0, 150.0, size=200),
            "savings": rng.uniform(0.0, 80.0, size=200),
        }
    )
    y_bg = ((X_bg["income"] + 1.2 * X_bg["savings"]) > 130.0).astype(int)
    cf_model = ClassicalMLModel("logistic_regression", task_type=TaskType.BINARY_CLASSIFICATION).fit(X_bg, y_bg)
    inst = pd.DataFrame({"age": [29.0], "income": [45.0], "savings": [10.0]})
    cf = InterpretabilityEngine(random_seed=42).generate_counterfactual(
        cf_model,
        X_bg,
        inst,
        desired_class=1,
        immutable_features=["age"],
        feature_bounds={"income": (30.0, 150.0), "savings": (0.0, 80.0)},
    )
    assert cf["found"] is True
    assert cf["distance_metric"] == "wachter_l1_mad"
    assert "age" not in cf["changes"]
    for feat_name, chg in cf["changes"].items():
        lo_b, hi_b = (30.0, 150.0) if feat_name == "income" else (0.0, 80.0)
        assert lo_b - 1e-6 <= chg["to"] <= hi_b + 1e-6

    # Ledoit-Wolf (2004) portfolio shrinkage when p > T (12 assets, only 8 time periods -> rank-deficient sample cov)
    ret_hist = rng.normal(0.01, 0.04, size=(8, 12))
    mu_12 = np.mean(ret_hist, axis=0)
    port = OptimizationEngine().optimize_portfolio(
        mu_12,
        returns_history=ret_hist,
        covariance_estimator="ledoit_wolf",
        min_weight=0.0,
        max_weight=0.35,
    )
    assert port.optimal is True
    assert port.details["covariance_estimator"] == "ledoit_wolf_2004"
    assert 0.0 < port.details["shrinkage_intensity"] <= 1.0
    assert sum(port.solution.values()) == pytest.approx(1.0, abs=1e-4)


def test_duckdb_confinement_and_sql_free_tabular_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RR-039 DuckDB workspace confinement, external-access C++ sandbox, and SQL-free fast_read_tabular."""
    ws_dir = tmp_path / "allowed_ws"
    ws_dir.mkdir()
    monkeypatch.setenv("OMNIFORGE_ALLOWED_DB_DIRS", str(ws_dir))

    db_file = ws_dir / "analytics.duckdb"
    with duckdb.connect(str(db_file)) as c:
        c.execute("CREATE TABLE metrics (k VARCHAR, v DOUBLE); INSERT INTO metrics VALUES ('auc', 0.94)")

    info = sql_inspect_database(f"duckdb:///{db_file.as_posix()}")
    assert info["connected"] is True and info["dialect"] == "duckdb"
    assert "metrics" in info["table_names"]

    df_ok = sql_load_dataframe(f"duckdb:///{db_file.as_posix()}", table_name="metrics")
    assert len(df_ok) == 1 and float(df_ok["v"].iloc[0]) == pytest.approx(0.94)

    # 1. Path traversal outside OMNIFORGE_ALLOWED_DATA_DIRS must fail with SecurityError
    outside_db = tmp_path / "outside.duckdb"
    with duckdb.connect(str(outside_db)) as c:
        c.execute("CREATE TABLE t (x INT)")
    with pytest.raises(SecurityError):
        sql_load_dataframe(f"duckdb:///{outside_db.as_posix()}", table_name="t")

    # 2. DuckDB external file-scan functions blocked in connector SQL validator
    with pytest.raises(SecurityError):
        sql_load_dataframe(f"duckdb:///{db_file.as_posix()}", query="SELECT * FROM read_csv_auto('/etc/hosts')")

    # 3. Defense-in-depth: SQL validator blocks read_csv_auto AND DuckDB C++ engine blocks replacement-scan file paths
    csv_secret = tmp_path / "secret.csv"
    csv_secret.write_text("col\n42\n", encoding="utf-8")
    with pytest.raises(SecurityError, match="file-access SQL statements are blocked"):
        DataIngestionEngine().ingest(
            f"duckdb:///{db_file.as_posix()}",
            query=f"SELECT * FROM read_csv_auto('{csv_secret.as_posix()}')",
        )
    with pytest.raises(DataIngestionError) as duck_exc:
        DataIngestionEngine().ingest(
            f"duckdb:///{db_file.as_posix()}",
            query=f"SELECT * FROM '{csv_secret.as_posix()}'",
        )
    assert "disabled by configuration" in str(duck_exc.value.__cause__)

    # 4. fast_read_tabular handles paths with embedded single quotes without SQL injection
    tricky_csv = ws_dir / "client's_quotes.csv"
    pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]}).to_csv(tricky_csv, index=False)
    df_fast, used_backend = fast_read_tabular(tricky_csv, backend="duckdb")
    assert used_backend == "duckdb" and len(df_fast) == 2


def test_registry_governance_fairness_gate_and_automated_rollback(tmp_path: Path) -> None:
    """Verify RR-010/021/031/038 NIST AI RMF fields, SQLite migration, fairness gate, and automated SLA rollback."""
    db_path = tmp_path / "legacy_reg.db"
    art_dir = tmp_path / "artifacts"

    # Simulate pre-v3.1.0 SQLite schema to verify seamless ALTER TABLE migration
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE model_versions (
                model_name TEXT NOT NULL,
                version INTEGER NOT NULL,
                stage TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                checksum_sha256 TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                description TEXT NOT NULL,
                registered_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (model_name, version)
            )
            """
        )
        conn.commit()

    reg = ModelRegistry(db_path=db_path, artifact_dir=art_dir, require_fairness_gate=True)

    rng = np.random.default_rng(17)
    X_m = pd.DataFrame({"f1": rng.normal(size=60), "f2": rng.normal(size=60)})
    y_m = (X_m["f1"] + X_m["f2"] > 0).astype(int)
    m_v1 = ClassicalMLModel("logistic_regression", task_type=TaskType.BINARY_CLASSIFICATION).fit(X_m, y_m)
    m_v2 = ClassicalMLModel("logistic_regression", task_type=TaskType.BINARY_CLASSIFICATION).fit(X_m, y_m)

    fair_pass = FairnessReport(
        protected_attribute="gender",
        demographic_parity_diff=0.03,
        disparate_impact_ratio=0.92,
        equal_opportunity_diff=0.02,
        equalized_odds_diff=0.04,
        group_metrics={},
        passed=True,
    )
    fair_fail = FairnessReport(
        protected_attribute="gender",
        demographic_parity_diff=0.35,
        disparate_impact_ratio=0.48,
        equal_opportunity_diff=0.29,
        equalized_odds_diff=0.31,
        group_metrics={},
        passed=False,
    )

    v1 = reg.register_model(
        m_v1,
        "credit_scorer",
        owner="risk-analytics-lead",
        intended_use="Retail credit underwriting",
        approved_domains=["finance", "retail"],
        risk_tier="high",
        fairness_report=fair_pass,
        stage=ModelStage.PRODUCTION,
    )
    assert v1.version == 1 and v1.stage == ModelStage.PRODUCTION
    assert v1.owner == "risk-analytics-lead"
    assert v1.risk_tier == "high"
    assert v1.approved_domains == ["finance", "retail"]
    assert v1.fairness_passed is True

    # v2 fails fairness audit -> registering at STAGING succeeds, promoting to PRODUCTION is BLOCKED!
    v2 = reg.register_model(
        m_v2,
        "credit_scorer",
        owner="risk-analytics-lead",
        risk_tier="high",
        fairness_report=fair_fail,
        stage=ModelStage.STAGING,
    )
    assert v2.fairness_passed is False
    with pytest.raises(MLOpsError, match="fairness gate"):
        reg.transition_stage("credit_scorer", v2.version, ModelStage.PRODUCTION)

    # Explicit auditable override_reason permits emergency promotion and records override tag
    v2_overridden = reg.transition_stage(
        "credit_scorer",
        v2.version,
        ModelStage.PRODUCTION,
        override_reason="Approved temporary exception CAB-2026-114",
    )
    assert v2_overridden.stage == ModelStage.PRODUCTION
    assert v2_overridden.tags["fairness_gate_override"] == "Approved temporary exception CAB-2026-114"
    assert reg.get_version("credit_scorer", 1).stage == ModelStage.ARCHIVED

    # Live SLA degradation on v2 automatically rolls PRODUCTION back to fair v1!
    mon = ModelMonitor("credit_scorer", min_accuracy=0.80)
    for _ in range(8):
        mon.record_inference(latency_ms=12.0, prediction=1, actual=0)  # 0% live accuracy < 0.80 SLA
    tracker = ExperimentTracker(db_path=tmp_path / "exp_track.db")
    ct = ContinuousTrainingOrchestrator(reg, tracker, kill_switch=False, auto_promote=False)
    health = ct.monitor_and_rollback(mon, min_feedback_for_rollback=5)
    assert health["status"] == "degraded"
    assert health["rollback_executed"] is True
    assert health["rolled_back_to_version"] == 1
    assert reg.get_version("credit_scorer", 1).stage == ModelStage.PRODUCTION
    assert reg.get_version("credit_scorer", 2).stage == ModelStage.ARCHIVED
