"""Phase 5 acceptance: C3 Green-AI Pareto AutoML, C10 Explainability Diff + Conformal, C8 Zero-Trust PII.

These tests exercise the contracted Phase 5 acceptance gate directly (components are
deterministic and do not require the heavy AutoML end-to-end pipeline):
  * Pareto dominance correctness under latency/carbon budgets (C3).
  * Exact split-conformal coverage >= 1 - alpha on exchangeable holdout (C10).
  * explain_diff shift detection between two models (e.g. Ridge vs GBDT) (C10).
  * Zero raw PII leakage under pii_policy="pseudonymize"/"block" (C8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omniforge.automl.pipeline_search import AutoMLPipelineSearch
from omniforge.core.security import DataFramePIIGuard, SecurityError
from omniforge.core.types import TaskType
from omniforge.evaluation.interpretability import InterpretabilityEngine
from omniforge.evaluation.robustness import split_conformal_quantile


class _StubModel:
    """Duck-typed BaseOmniForgeModel stand-in (predict / predict_proba / task_type / classes_)."""

    def __init__(self, coef, intercept=0.0, task_type=TaskType.BINARY_CLASSIFICATION,
                 classes_=(0, 1), name="stub"):
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)
        self.task_type = task_type
        self.classes_ = list(classes_)
        self.name = name

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.coef + self.intercept

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = X @ self.coef + self.intercept
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])


# --------------------------------------------------------------------------- C3: Pareto + budgets


def test_pareto_frontier_three_dimensional_dominance():
    rows = [
        {"model": "a", "f1": 0.90, "inference_latency_ms_per_1k": 10.0, "carbon_gco2e": 1.0},
        {"model": "b", "f1": 0.80, "inference_latency_ms_per_1k": 5.0,  "carbon_gco2e": 0.5},
        {"model": "c", "f1": 0.95, "inference_latency_ms_per_1k": 8.0,  "carbon_gco2e": 0.8},
        {"model": "d", "f1": 0.85, "inference_latency_ms_per_1k": 4.0,  "carbon_gco2e": 0.4},
    ]
    frontier = AutoMLPipelineSearch.pareto_frontier(rows, "f1", higher_is_better=True)
    names = {r["model"] for r in frontier}
    # c dominates a (better f1, lower latency, lower carbon); d dominates b.
    # c and d do not dominate each other (c wins f1, d wins latency+carbon) -> both survive.
    assert names == {"c", "d"}, names


def test_pareto_frontier_empty_and_regression_direction():
    assert AutoMLPipelineSearch.pareto_frontier([], "rmse", higher_is_better=False) == []
    rows = [
        {"model": "a", "rmse": 2.0, "inference_latency_ms_per_1k": 10.0, "carbon_gco2e": 1.0},
        {"model": "b", "rmse": 1.5, "inference_latency_ms_per_1k": 5.0,  "carbon_gco2e": 0.5},
    ]
    # lower rmse is better; b has lower rmse, lower latency, lower carbon -> dominates a.
    frontier = AutoMLPipelineSearch.pareto_frontier(rows, "rmse", higher_is_better=False)
    assert [r["model"] for r in frontier] == ["b"]


def test_budget_rejection_under_tight_latency_and_carbon():
    tight = AutoMLPipelineSearch(
        task_type=TaskType.BINARY_CLASSIFICATION,
        max_carbon_grams_co2e=1.0,
        max_inference_latency_ms=100.0,
    )
    assert tight._budget_rejected("x", carbon_g=2.0, latency_ms=10.0) is True   # carbon over
    assert tight._budget_rejected("x", carbon_g=0.5, latency_ms=200.0) is True  # latency over
    assert tight._budget_rejected("x", carbon_g=0.5, latency_ms=10.0) is False  # under both

    unconstrained = AutoMLPipelineSearch(task_type=TaskType.BINARY_CLASSIFICATION)
    assert unconstrained.max_carbon_grams_co2e is None
    assert unconstrained.max_inference_latency_ms is None
    assert unconstrained._budget_rejected("x", carbon_g=1e9, latency_ms=1e9) is False


# --------------------------------------------------------------------------- C10: conformal coverage


def test_split_conformal_quantile_finite_sample_guarantee():
    rng = np.random.default_rng(7)
    scores = np.abs(rng.normal(0.0, 1.0, 100))
    alpha = 0.1
    q_hat, valid = split_conformal_quantile(scores, alpha)
    assert valid is True
    # The quantile is an order statistic of the calibration scores.
    assert q_hat in scores
    # At least (1 - alpha) of calibration scores fall below the quantile.
    assert float(np.mean(scores <= q_hat)) >= 1.0 - alpha


def test_conformal_coverage_on_exchangeable_holdout():
    rng = np.random.default_rng(3)
    alpha = 0.1
    n_cal, n_hold, trials = 500, 400, 40
    coverages = []
    for _ in range(trials):
        y_cal = rng.normal(0.0, 1.0, n_cal)
        pred_cal = np.full(n_cal, y_cal.mean())
        scores = np.abs(y_cal - pred_cal)
        q_hat, _ = split_conformal_quantile(scores, alpha)
        y_hold = rng.normal(0.0, 1.0, n_hold)
        pred_hold = np.full(n_hold, y_cal.mean())
        coverages.append(float(np.mean(np.abs(y_hold - pred_hold) <= q_hat)))
    mean_coverage = float(np.mean(coverages))
    assert mean_coverage >= 1.0 - alpha - 0.03, mean_coverage


# --------------------------------------------------------------------------- C10: explain_diff shift detection


def _bg_frame(n_rows=200, n_feat=20, seed=1):
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_feat)]
    return pd.DataFrame(rng.normal(0.0, 1.0, size=(n_rows, n_feat)), columns=cols)


def test_explain_diff_identical_models_not_significant():
    n_feat = 20
    X_bg = _bg_frame(n_feat=n_feat)
    inst = pd.DataFrame(np.zeros((1, n_feat)), columns=[f"f{i}" for i in range(n_feat)])
    m1 = _StubModel(np.ones(n_feat))
    m2 = _StubModel(np.ones(n_feat))
    eng = InterpretabilityEngine(random_seed=11)
    out = eng.explain_diff(m1, m2, X_bg, inst, n_perturbations=300, n_permutations=500, alpha=0.05)
    assert out["global_shift"] == 0.0
    assert out["mean_abs_shift"] == 0.0
    assert out["p_value"] == 1.0
    assert out["significant"] is False
    assert out["prediction_a"] == out["prediction_b"]
    assert out["per_feature"] and out["top_shifted_features"]


def test_explain_diff_different_models_detect_shift():
    n_feat = 20
    X_bg = _bg_frame(n_feat=n_feat, seed=2)
    inst = pd.DataFrame(np.zeros((1, n_feat)), columns=[f"f{i}" for i in range(n_feat)])
    m_ridge = _StubModel(np.ones(n_feat))          # attends to every feature equally
    m_gbdt = _StubModel(2.0 * np.ones(n_feat))     # stronger, uniform shift vs ridge
    eng = InterpretabilityEngine(random_seed=12)
    out = eng.explain_diff(m_ridge, m_gbdt, X_bg, inst, n_perturbations=400, n_permutations=600, alpha=0.05)
    assert out["mean_abs_shift"] > 0.0
    assert out["significant"] is True
    assert out["p_value"] < 0.05
    assert len(out["top_shifted_features"]) == n_feat
    assert all(abs(v["shift"]) > 0 for v in out["per_feature"].values())
    assert "prediction_diff" in out


# --------------------------------------------------------------------------- C10: conformal explanations


def test_conformal_explain_regression_interval():
    rng = np.random.default_rng(5)
    coef = np.array([1.0, -1.0, 0.5, 0.0])
    X = rng.normal(0.0, 1.0, size=(120, 4))
    y = X @ coef + rng.normal(0.0, 0.3, 120)
    cols = [f"x{i}" for i in range(4)]
    model = _StubModel(coef, task_type=TaskType.REGRESSION)
    eng = InterpretabilityEngine(random_seed=5)
    out = eng.conformal_explain(
        model,
        pd.DataFrame(X[:80], columns=cols),
        pd.Series(y[:80]),
        pd.DataFrame(X[80:81], columns=cols),
        alpha=0.1,
    )
    assert out["method"] == "conformal_lac_interval"
    assert out["coverage_guarantee"] == 0.9
    assert out["finite_sample_valid"] is True
    lo, hi = out["interval"]
    assert lo <= out["prediction"] <= hi
    assert out["half_width"] == (hi - lo) / 2
    assert out["n_calibration"] == 80


def test_conformal_explain_classification_prediction_set():
    rng = np.random.default_rng(9)
    X = rng.normal(0.0, 1.0, size=(80, 3))
    y = (X @ np.array([1.0, 1.0, 0.0]) > 0).astype(int)
    cols = [f"x{i}" for i in range(3)]
    model = _StubModel(np.array([1.0, 1.0, 0.0]), task_type=TaskType.BINARY_CLASSIFICATION,
                       classes_=(0, 1))
    eng = InterpretabilityEngine(random_seed=9)
    out = eng.conformal_explain(
        model,
        pd.DataFrame(X, columns=cols),
        pd.Series(y),
        pd.DataFrame(X[:1], columns=cols),
        alpha=0.1,
    )
    assert out["method"] == "conformal_aps_prediction_set"
    assert out["coverage_guarantee"] == 0.9
    assert isinstance(out["prediction_set"], list)
    assert out["n_calibration"] == 80


# --------------------------------------------------------------------------- C8: Zero-Trust PII guard


def _pii_df():
    return pd.DataFrame({
        "name": ["alice", "bob", "carol"],
        "ssn": ["123-45-6789", "987-65-4321", "555-55-5555"],
        "email": ["a@x.com", "b@y.com", "c@z.com"],
        "amount": [10.0, 20.0, 30.0],
    })


_RAW_PII = ("123-45-6789", "987-65-4321", "555-55-5555", "a@x.com", "b@y.com", "c@z.com")


def test_pii_guard_detect_returns_counts_only_no_raw_values():
    guard = DataFramePIIGuard(tenant_id="t1", pii_policy="pseudonymize")
    rep = guard.detect(_pii_df())
    assert rep["ssn"]["pii_cells"] == 3
    assert rep["email"]["pii_cells"] == 3
    assert "amount" not in rep
    flat = str(rep)
    for raw in _RAW_PII:
        assert raw not in flat


def test_pii_guard_pseudonymize_zero_raw_leakage_deterministic():
    guard = DataFramePIIGuard(tenant_id="t1", pii_policy="pseudonymize")
    out, rep = guard.transform(_pii_df())
    assert list(out["amount"]) == [10.0, 20.0, 30.0]  # non-PII untouched
    for col in ("ssn", "email"):
        for v in out[col]:
            assert v not in _RAW_PII
            s = str(v)
            assert len(s) == 16 and all(c in "0123456789abcdef" for c in s), s
    # deterministic: same input -> same pseudonyms (HMAC, no random cache)
    out2, _ = guard.transform(_pii_df())
    assert out["ssn"].tolist() == out2["ssn"].tolist()
    assert out["email"].tolist() == out2["email"].tolist()
    assert rep["pii_detected"] is True


def test_pii_guard_block_fail_closed():
    guard = DataFramePIIGuard(tenant_id="t1", pii_policy="block")
    with pytest.raises(SecurityError):
        guard.transform(_pii_df())
    clean = pd.DataFrame({"amount": [1.0, 2.0], "name": ["x", "y"]})
    out, rep = guard.transform(clean)
    assert rep["pii_detected"] is False
    assert list(out["amount"]) == [1.0, 2.0]


def test_pii_guard_audit_logs_no_raw_pii():
    events = []

    class _FakeAudit:
        def log_event(self, *, action, resource_type, resource_id, actor, role,
                      tenant_id, status, details):
            events.append(details)

    guard = DataFramePIIGuard(tenant_id="t1", pii_policy="pseudonymize",
                              audit_logger=_FakeAudit())
    guard.transform(_pii_df())
    assert events, "expected an audit event to be emitted"
    for d in events:
        s = str(d)
        for raw in _RAW_PII:
            assert raw not in s, f"raw PII leaked into audit details: {s}"


def test_pii_guard_unknown_policy_rejected():
    with pytest.raises(SecurityError):
        DataFramePIIGuard(tenant_id="t1", pii_policy="nonsense")
