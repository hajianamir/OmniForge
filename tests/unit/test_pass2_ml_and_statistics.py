"""Pass 2 regression tests for statistical tests, baseline models, evaluation, causal, and Shapley speedups."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from omniforge.eda.statistics import StatisticalAnalyzer
from omniforge.models.baselines import DummyClassifierBaseline, DummyRegressorBaseline, LogisticBaseline


def test_logistic_baseline_emits_no_sklearn_warnings_for_binary_and_multiclass() -> None:
    """LogisticBaseline must use l1_ratio=1.0 + saga without sklearn penalty/l1_ratio warnings."""
    rng = np.random.default_rng(42)
    X_bin = pd.DataFrame(
        {
            "num_1": rng.normal(0.0, 1.0, size=90),
            "num_2": rng.normal(0.5, 1.2, size=90),
            "cat_1": rng.choice(["a", "b", "c"], size=90),
        }
    )
    y_bin = (X_bin["num_1"] + 0.5 * X_bin["num_2"] > 0.1).astype(int).to_numpy()
    y_multi = np.digitize(X_bin["num_1"].to_numpy(), bins=[-0.4, 0.4])

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.filterwarnings("error", message=".*Inconsistent values.*", category=UserWarning)

        bin_model = LogisticBaseline(random_seed=7).fit(X_bin, y_bin, C=0.8)
        proba_bin = bin_model.predict_proba(X_bin)
        preds_bin = bin_model.predict(X_bin)
        assert proba_bin.shape == (90, 2)
        assert np.allclose(proba_bin.sum(axis=1), 1.0)
        assert len(preds_bin) == 90

        multi_model = LogisticBaseline(random_seed=7).fit(X_bin, y_multi, C=0.8)
        proba_multi = multi_model.predict_proba(X_bin)
        preds_multi = multi_model.predict(X_bin)
        assert proba_multi.shape == (90, 3)
        assert np.allclose(proba_multi.sum(axis=1), 1.0)
        assert set(preds_multi).issubset({0, 1, 2})


def test_anderson_darling_test_interpolated_pvalue_and_zero_scipy_warnings() -> None:
    """anderson_darling_test must emit zero SciPy FutureWarnings and return continuous p-values."""
    rng = np.random.default_rng(2026)
    normal_sample = rng.normal(loc=10.0, scale=2.0, size=150)
    skewed_sample = rng.exponential(scale=2.0, size=150)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        res_norm = StatisticalAnalyzer.anderson_darling_test(normal_sample, alpha=0.05)
        res_skew = StatisticalAnalyzer.anderson_darling_test(skewed_sample, alpha=0.01)
        res_norm_via_api = StatisticalAnalyzer.normality_test(
            pd.Series(skewed_sample), alpha=0.001, method="anderson_darling"
        )
        extreme_outlier = np.concatenate([np.zeros(499), [1e6]])
        res_extreme = StatisticalAnalyzer.anderson_darling_test(extreme_outlier, alpha=0.001)

    assert res_norm.reject_null is False
    assert 0.15 < res_norm.p_value < 0.999999

    assert res_skew.reject_null is True
    assert 0.0 <= res_skew.p_value < 0.01
    assert res_norm_via_api.reject_null is True
    assert 0.0 <= res_norm_via_api.p_value < 0.001
    assert res_extreme.reject_null is True
    assert res_extreme.p_value == 0.0

    # Constant (zero-variance) arrays must reject normality cleanly without NaNs or RuntimeWarnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        const_ad = StatisticalAnalyzer.anderson_darling_test(np.ones(25))
        const_jb = StatisticalAnalyzer.jarque_bera_test(np.ones(25))
    assert const_ad.reject_null is True and const_ad.p_value == 0.0
    assert const_jb.reject_null is True and const_jb.p_value == 0.0


def test_mcnemar_test_edwards_continuity_correction_when_discordant_counts_equal() -> None:
    """Edwards continuity-corrected McNemar chi-square must be 0.0 when |b - c| <= 1."""
    # Symmetric discordant counts b == c == 10 -> |b - c| = 0 -> statistic must be 0.0 (not 1/20 = 0.05!)
    sym = StatisticalAnalyzer.mcnemar_test([[40, 10], [10, 40]])
    assert sym.statistic == 0.0
    assert sym.p_value == 1.0
    assert sym.reject_null is False

    # |b - c| == 1 -> max(0, 1 - 1)^2 / n_disc == 0.0
    near_sym = StatisticalAnalyzer.mcnemar_test([[40, 11], [10, 39]])
    assert near_sym.statistic == 0.0
    assert near_sym.reject_null is False

    # Zero discordant counts
    zero_disc = StatisticalAnalyzer.mcnemar_test([[50, 0], [0, 50]])
    assert zero_disc.statistic == 0.0
    assert zero_disc.p_value == 1.0

    # Negative counts must raise ValueError
    with pytest.raises(ValueError, match="negative"):
        StatisticalAnalyzer.mcnemar_test([[10, -1], [2, 10]])


def test_metrics_evaluator_multiclass_partial_fold_and_single_row_regression() -> None:
    """3-class proba on 2-class fold must not slice column 1 as binary; 1-row regression emits no warning."""
    from sklearn.exceptions import UndefinedMetricWarning

    from omniforge.evaluation.metrics import MetricsEvaluator

    # 3-class model evaluated on a validation fold observing only classes {0, 2}
    y_true = np.array([0, 0, 2, 2])
    y_pred = np.array([0, 0, 2, 2])
    y_proba = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.85, 0.10, 0.05],
            [0.05, 0.05, 0.90],
            [0.05, 0.10, 0.85],
        ]
    )
    m = MetricsEvaluator.classification_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == 1.0
    assert "log_loss" in m and m["log_loss"] < 0.2
    # Must NOT have treated 3-column y_proba[:, 1] as binary P(y == 2)
    assert "auc_roc" not in m
    assert "brier_score" not in m

    with warnings.catch_warnings():
        warnings.simplefilter("error", UndefinedMetricWarning)
        reg_1 = MetricsEvaluator.regression_metrics(np.array([5.0]), np.array([4.5]))
    assert reg_1["mae"] == 0.5
    assert np.isnan(reg_1["r2"])
    assert np.isnan(reg_1["explained_variance"])


def test_cross_validator_ignores_nan_fold_mape_and_unions_metric_keys() -> None:
    """CrossValidator summary must aggregate finite fold metrics and union keys across folds."""
    from omniforge.evaluation.validation import CrossValidator

    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 12)})
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    y_reg = pd.Series([0.0, 0.0, 0.0, 0.0, 10.0, 12.0, 14.0, 16.0, 20.0, 22.0, 24.0, 26.0])
    cv = CrossValidator(strategy="group_kfold", n_splits=3, random_seed=42)
    res_reg = cv.cross_validate(lambda: DummyRegressorBaseline(), X, y_reg, groups=groups)
    assert res_reg["n_folds_executed"] == 3
    assert any(np.isnan(f["mape"]) for f in res_reg["folds"])
    assert np.isfinite(res_reg["summary"]["mape"]["mean"])
    assert np.isfinite(res_reg["summary"]["mape"]["std"])

    # Group 2 (first validation fold in sklearn GroupKFold) has only class 0 -> Fold 0 omits auc_roc,
    # while Folds 1 and 2 contain both classes -> summary must still include auc_roc!
    y_clf = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
    res_clf = cv.cross_validate(lambda: DummyClassifierBaseline(), X, y_clf, groups=groups)
    assert "auc_roc" not in res_clf["folds"][0]
    assert "auc_roc" in res_clf["summary"]
    assert np.isfinite(res_clf["summary"]["auc_roc"]["mean"])


def test_causal_inference_guards_empty_and_categorical_confounders() -> None:
    """CausalInferenceEngine must guard leakage columns and handle empty or categorical confounders."""
    from omniforge.causal.inference import CausalInferenceEngine
    from omniforge.core.exceptions import CausalInferenceError

    rng = np.random.default_rng(123)
    n = 80
    seg = rng.choice(["enterprise", "midmarket", "smb"], size=n)
    x_num = rng.normal(0.0, 1.0, size=n)
    z_inst = rng.binomial(1, 0.5, size=n)
    t = ((0.8 * z_inst + 0.4 * x_num + rng.normal(0, 0.5, size=n)) > 0.3).astype(int)
    y = 2.5 * t + 1.2 * x_num + (seg == "enterprise").astype(float) + rng.normal(0, 0.3, size=n)
    df = pd.DataFrame({"T": t, "Y": y, "Z": z_inst, "seg": seg, "x_num": x_num})

    eng = CausalInferenceEngine(random_seed=7, bootstrap_rounds=12)

    with pytest.raises(CausalInferenceError, match="Unsupported"):
        eng.estimate_ate(df, treatment_col="T", outcome_col="Y", confounders=["x_num"], method="not_a_method")

    with pytest.raises(CausalInferenceError, match="Confounders cannot include"):
        eng.estimate_ate(df, treatment_col="T", outcome_col="Y", confounders=["x_num", "T"])

    with pytest.raises(CausalInferenceError, match="Confounders cannot include"):
        eng.estimate_ate(
            df, treatment_col="T", outcome_col="Y", confounders=["Z"], instrument_col="Z", method="iv_2sls"
        )

    # Categorical + numeric confounders
    res_cat = eng.estimate_ate(df, treatment_col="T", outcome_col="Y", confounders=["seg", "x_num"], method="aipw")
    assert 1.5 < res_cat.ate < 3.5

    # Empty confounders (unadjusted RCT / simple IV-2SLS) including refutations
    res_empty_aipw = eng.estimate_ate(df, treatment_col="T", outcome_col="Y", confounders=[], method="aipw")
    assert np.isfinite(res_empty_aipw.ate)
    res_empty_iv = eng.estimate_ate(
        df, treatment_col="T", outcome_col="Y", confounders=[], instrument_col="Z", method="iv_2sls"
    )
    assert np.isfinite(res_empty_iv.ate)

    gml = CausalInferenceEngine.export_dowhy_gml('t"1', "y", ['c"1', 'c"1', "y"])
    assert 'id "t\\"1"' in gml
    assert gml.count('id "c\\"1"') == 1


def test_vectorized_shap_pdp_and_knn_shapley_exact_equivalence_and_batching() -> None:
    """Vectorized Monte Carlo SHAP, PDP, and KNN-Shapley must match exact recurrences and batch calls."""
    from omniforge.evaluation.interpretability import InterpretabilityEngine
    from omniforge.evaluation.robustness import knn_shapley_valuation, nested_cv_evaluate

    rng = np.random.default_rng(99)
    X = pd.DataFrame(
        {
            "f0": rng.normal(0.0, 1.0, size=60),
            "f1": rng.normal(1.0, 2.0, size=60),
            "f2": rng.uniform(-1.0, 1.0, size=60),
            "cat": rng.choice(["red", "blue"], size=60),
        }
    )
    y = (X["f0"] - 0.7 * X["f1"] + (X["cat"] == "red").astype(float) > 0.0).astype(int).to_numpy()
    model = LogisticBaseline(random_seed=11).fit(X, y)

    call_sizes: list[int] = []
    orig_predict_proba = model.predict_proba

    def _counting_predict_proba(df_in: pd.DataFrame) -> np.ndarray:
        call_sizes.append(len(df_in))
        return orig_predict_proba(df_in)

    model.predict_proba = _counting_predict_proba  # type: ignore[method-assign]

    ie = InterpretabilityEngine(random_seed=42)
    shap_out = ie.explain_shap_local(model, X, X.iloc[[0]], n_samples=32)
    # 1 call for instance + 1 batched call of 32 * (4 + 1) = 160 rows (instead of 161 separate calls!)
    assert call_sizes == [1, 32 * (4 + 1)]
    # Shapley efficiency axiom: expected_value + sum(attributions) == prediction
    total_attr = sum(shap_out["attributions"].values())
    assert abs((shap_out["expected_value"] + total_attr) - shap_out["prediction"]) < 1e-4

    call_sizes.clear()
    pdp_out = ie.partial_dependence(model, X, "f0", grid_resolution=10)
    assert call_sizes == [len(X) * 10]
    assert len(pdp_out["grid"]) == 10 and len(pdp_out["pdp_mean"]) == 10

    # KNN-Shapley exact match against scalar backward recurrence
    X_tr = rng.normal(size=(35, 3))
    y_tr = rng.integers(0, 3, size=35)
    X_va = rng.normal(size=(12, 3))
    y_va = rng.integers(0, 3, size=12)
    k = 5
    ks = knn_shapley_valuation(X_tr, y_tr, X_va, y_va, k=k)

    ref_shapley = np.zeros(len(X_tr), dtype=float)
    for j in range(len(X_va)):
        order = np.argsort(np.linalg.norm(X_tr - X_va[j], axis=1))
        s = np.zeros(len(X_tr), dtype=float)
        s[-1] = (1.0 if y_tr[order[-1]] == y_va[j] else 0.0) / len(X_tr)
        for i in range(len(X_tr) - 2, -1, -1):
            m_curr = 1.0 if y_tr[order[i]] == y_va[j] else 0.0
            m_next = 1.0 if y_tr[order[i + 1]] == y_va[j] else 0.0
            rank = i + 1
            s[i] = s[i + 1] + ((m_curr - m_next) / k) * (min(k, rank) / rank)
        ref_shapley[order] += s / len(X_va)

    assert np.allclose(ks["values"], np.round(ref_shapley, 6), atol=1e-6)

    ncv = nested_cv_evaluate(X_tr, (y_tr > 0).astype(int), outer_splits=3, inner_splits=2)
    expected_sample_std = round(float(np.std(ncv["outer_scores"], ddof=1)), 6)
    assert ncv["std_outer_score"] == expected_sample_std

    # CategoricalDtype rare-category handling in vectorized explain_shap_local
    X_cat = X.copy()
    X_cat["cat"] = pd.Categorical(X_cat["cat"], categories=["red", "blue"])
    inst_rare = X_cat.iloc[[0]].copy()
    inst_rare["cat"] = pd.Categorical(["green"], categories=["red", "blue", "green"])
    shap_rare = ie.explain_shap_local(model, X_cat, inst_rare, n_samples=8)
    assert set(shap_rare["attributions"].keys()) == set(X_cat.columns)


