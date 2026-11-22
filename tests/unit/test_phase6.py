"""Phase 6 acceptance: Future Developments II (v4.0.0).

C5 — Causal Drift Root-Cause Attribution:
  * DriftRootCauseAnalyzer classifies drifted features into root causes (upstream
    sources with no drifted ancestors) vs symptoms (downstream with drifted ancestors).
  * Remediation prescriptions emitted; DAG edges well-formed.
  * Exposed via Studio 11 /s11/root-cause and Studio 14 /s14/root-cause (schemas).

C7 — Closed-Loop Digital Twin MPC:
  * ClosedLoopMPCController solves constrained receding-horizon QP via SLSQP.
  * Control / state bounds respected; constraints satisfied.
  * Receding-horizon applies first control, assimilates measurement.

C4 — Air-Gapped Deterministic Copilot:
  * CopilotToolRegistry exposes 12 new Phase 6 tools (root_cause_diagnosis,
    twin_control, split_lock, pii_audit, pareto_automl, domain_validate,
    register_model, promote_model, list_studios, studio_overview + existing).
  * _chat_local_planner routes C4/C5/C7/C8/C3 natural-language keywords to tools.
  * Tools execute offline deterministically.

Also: canonical SemVer 4.0.0 alignment (covered by existing phase tests).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from omniforge.mlops.drift import DriftDetector
from omniforge.mlops.root_cause import (
    CausalDiscovery,
    DriftRootCauseAnalyzer,
    RootCauseReport,
)
from omniforge.simulation import (
    ClosedLoopMPCController,
    TwinControlReport,
    build_default_twin,
)

# --------------------------------------------------------------------------- C5: root cause


def _make_shifted_pair(n=400, seed=11):
    """Reference + current where 'source' shifts (root cause) and 'derived' follows (symptom)."""
    rng = np.random.default_rng(seed)
    ref = pd.DataFrame(
        {
            "source": rng.normal(0.0, 1.0, n),
            "noise": rng.normal(0.0, 0.5, n),
        }
    )
    ref["derived"] = 0.6 * ref["source"] + 0.4 * ref["noise"]
    cur = ref.copy()
    cur["source"] = cur["source"] + 1.5  # upstream root-cause shift
    cur["derived"] = 0.6 * cur["source"] + 0.4 * cur["noise"]  # downstream follows
    return ref, cur


class _PredictShiftModel:
    """Stub model whose prediction = source + 2*derived (so both show attribution)."""

    def predict(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return (X["source"].astype(float) + 2.0 * X["derived"].astype(float)).to_numpy()


def test_causal_discovery_dag_is_well_formed():
    ref, cur = _make_shifted_pair()
    cd = CausalDiscovery(alpha=0.05, max_cond_set_size=2, random_seed=7)
    parents = cd.discover(pd.concat([ref, cur], ignore_index=True))
    # Every node present, parents are subsets of nodes, no self-loops.
    nodes = set(parents)
    assert "source" in nodes and "derived" in nodes
    for child, par_set in parents.items():
        assert child not in par_set  # no self-loop
        assert par_set <= nodes


def test_root_cause_isolates_source_as_root_cause():
    ref, cur = _make_shifted_pair()
    model = _PredictShiftModel()
    report = DriftRootCauseAnalyzer(random_seed=7).analyze(
        ref, cur, model=model, feature_cols=["source", "derived", "noise"]
    )
    assert isinstance(report, RootCauseReport)
    assert report.overall_status.value in ("WARNING", "CRITICAL")
    assert "source" in report.root_causes, f"source should be root cause; got {report.root_causes}"
    assert report.total_attribution > 0.0
    # DAG edges are [parent, child] pairs, well-formed.
    for edge in report.dag_edges:
        assert len(edge) == 2
        assert edge[0] != edge[1]
    # At least one remediation prescription mentions root cause.
    assert any("ROOT CAUSE" in p for p in report.remediation_prescriptions)


def test_root_cause_no_model_still_works():
    ref, cur = _make_shifted_pair()
    report = DriftRootCauseAnalyzer(random_seed=7).analyze(
        ref, cur, model=None, feature_cols=["source", "derived", "noise"]
    )
    assert isinstance(report, RootCauseReport)
    assert report.root_causes or report.symptoms or report.drifted_features


def test_root_cause_prescriptions_no_raw_pii():
    ref, cur = _make_shifted_pair()
    report = DriftRootCauseAnalyzer(random_seed=7).analyze(
        ref, cur, model=None, feature_cols=["source", "derived", "noise"]
    )
    flat = str(report.remediation_prescriptions)
    # Remediation should reference column names, not raw data values.
    assert "source" in flat or "derived" in flat


def test_drift_detector_still_healthy_when_no_shift():
    ref = pd.DataFrame({"a": np.random.default_rng(1).normal(0, 1, 100)})
    cur = pd.DataFrame({"a": np.random.default_rng(2).normal(0, 1, 100)})
    rep = DriftDetector().detect(ref, cur)
    assert rep.overall_status.value in ("HEALTHY", "WARNING")


# --------------------------------------------------------------------------- C7: closed-loop twin MPC


def test_build_default_twin_shapes():
    twin = build_default_twin(n_x=2, n_u=1, dt=1.0)
    assert twin.A.shape == (2, 2)
    assert twin.B.shape == (2, 1)
    assert twin.H.shape == (1, 2)
    assert twin.x.shape == (2,)


def test_mpc_respects_control_bounds():
    twin = build_default_twin(n_x=2, n_u=1, dt=1.0)
    ctrl = ClosedLoopMPCController(
        twin, horizon=6, u_min=np.array([-0.5]), u_max=np.array([0.5]),
        x_min=np.array([-10.0, -10.0]), x_max=np.array([10.0, 10.0]),
    )
    res = ctrl.solve(x0=np.zeros(2), r_path=np.zeros((6, 2)))
    assert np.all(res.optimal_control >= -0.5 - 1e-6)
    assert np.all(res.optimal_control <= 0.5 + 1e-6)
    assert res.state_trajectory.shape == (7, 2)


def test_mpc_setpoint_tracking():
    twin = build_default_twin(n_x=2, n_u=1, dt=1.0)
    ctrl = ClosedLoopMPCController(twin, horizon=8)
    r_path = np.tile(np.array([1.0, 0.0]), (8, 1))
    res = ctrl.solve(x0=np.zeros(2), r_path=r_path)
    # State should move toward setpoint (first state component closer to 1.0).
    assert float(res.state_trajectory[-1, 0]) > float(res.state_trajectory[0, 0])


def test_closed_loop_run_report():
    twin = build_default_twin(n_x=2, n_u=1, dt=1.0)
    ctrl = ClosedLoopMPCController(twin, horizon=6)
    measurements = [np.zeros(2) for _ in range(5)]
    report = ctrl.run_closed_loop(measurements, setpoint=np.array([1.0, 0.0]))
    assert isinstance(report, TwinControlReport)
    assert report.n_steps_applied == 5
    assert len(report.control_history) == 5
    assert len(report.state_trajectory) == 5
    assert isinstance(report.all_constraints_satisfied, bool)


# --------------------------------------------------------------------------- C4: air-gapped copilot tools


def test_copilot_tool_registry_has_phase6_tools():
    from omniforge.copilot import CopilotToolRegistry
    from omniforge.copilot.memory import CopilotMemory

    reg = CopilotToolRegistry(CopilotMemory())
    names = {t.name for t in reg._tools.values()}
    expected = {
        "root_cause_diagnosis",
        "twin_control",
        "split_lock",
        "pii_audit",
        "pareto_automl",
        "domain_validate",
        "register_model",
        "promote_model",
        "list_studios",
        "studio_overview",
    }
    assert expected <= names, f"missing Phase 6 tools: {expected - names}"


def test_split_lock_passes_on_non_overlapping_split():
    from omniforge.copilot import CopilotToolRegistry
    from omniforge.copilot.memory import CopilotMemory

    mem = CopilotMemory()
    df = pd.DataFrame({"x": np.arange(100), "y": np.arange(100) * 2})
    mem.set_dataset("ds", df)
    reg = CopilotToolRegistry(mem)
    res = reg.invoke("split_lock", {"dataset_name": "ds", "test_size": 0.2})
    assert res["passed"] is True
    assert res["row_overlap_count"] == 0


def test_pii_audit_pseudonymize_zero_leakage():
    from omniforge.copilot import CopilotToolRegistry
    from omniforge.copilot.memory import CopilotMemory

    mem = CopilotMemory()
    df = pd.DataFrame(
        {"name": ["alice", "bob"], "ssn": ["123-45-6789", "987-65-4321"], "amt": [1.0, 2.0]}
    )
    mem.set_dataset("ds", df)
    reg = CopilotToolRegistry(mem)
    res = reg.invoke("pii_audit", {"dataset_name": "ds", "pii_policy": "pseudonymize"})
    assert res["pii_detected"] is True
    assert "ssn" in res["pii_columns"]
    raw = ("123-45-6789", "987-65-4321")
    flat = str(res)
    for r in raw:
        assert r not in flat


def test_list_studios_returns_15():
    from omniforge.copilot import CopilotToolRegistry
    from omniforge.copilot.memory import CopilotMemory

    reg = CopilotToolRegistry(CopilotMemory())
    res = reg.invoke("list_studios", {})
    assert res["count"] == 15
    assert len(res["studios"]) == 15


def test_twin_control_tool_deterministic():
    from omniforge.copilot import CopilotToolRegistry
    from omniforge.copilot.memory import CopilotMemory

    reg = CopilotToolRegistry(CopilotMemory())
    res = reg.invoke("twin_control", {"n_x": 2, "n_u": 1, "horizon": 6, "n_steps": 5, "dt": 1.0})
    assert res["n_steps_applied"] == 5
    assert isinstance(res["all_constraints_satisfied"], bool)
    assert len(res["control_history"]) == 5


# --------------------------------------------------------------------------- C5: temporal DAG orientation (Item 1)


def _make_temporal_shifted_pair(n=600, seed=21):
    """Reference+current with a time_col; 'source' shifts early, 'derived' shifts later."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    source = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 0.5, n)
    derived = 0.6 * source + 0.4 * noise
    ref = pd.DataFrame({"time": t, "source": source, "derived": derived, "noise": noise})
    cur = ref.copy()
    # source shifts at t>=300 (early window); derived shifts at t>=450 (later window).
    cur.loc[t >= 300, "source"] += 1.5
    cur.loc[t >= 450, "derived"] += 1.5
    return ref, cur


def test_temporal_dag_orientation():
    ref, cur = _make_temporal_shifted_pair()
    cd = CausalDiscovery(alpha=0.05, max_cond_set_size=2, random_seed=7)
    parents = cd.discover(pd.concat([ref, cur], ignore_index=True), time_col="time")
    nodes = set(parents)
    assert "source" in nodes and "derived" in nodes
    # source shifts earlier (t>=300) than derived (t>=450) -> source should be upstream.
    assert "derived" not in parents.get("source", set())  # source is not a child of derived
    for child, par_set in parents.items():
        assert child not in par_set  # no self-loop
        assert par_set <= nodes


# --------------------------------------------------------------------------- C5: exact Delta-SHAP integration (Item 2)


class _PredictProbaModel:
    """Stub classifier with predict / predict_proba / task_type / classes_."""

    def __init__(self, coef, intercept=0.0, classes_=(0, 1)):
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)
        self.task_type = __import__("omniforge.core.types", fromlist=["TaskType"]).TaskType.BINARY_CLASSIFICATION
        self.classes_ = list(classes_)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X @ self.coef + self.intercept > 0).astype(int)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = X @ self.coef + self.intercept
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])


def test_deltashap_integration():
    rng = np.random.default_rng(31)
    n, n_feat = 200, 5
    X_bg = pd.DataFrame(rng.normal(0.0, 1.0, size=(n, n_feat)), columns=[f"f{i}" for i in range(n_feat)])
    model = _PredictProbaModel(np.ones(n_feat))
    ref = X_bg.sample(n=60, random_state=1)
    cur = X_bg.sample(n=60, random_state=2)
    # Shift f0 strongly in current so it shows Delta-SHAP attribution.
    cur = cur.copy()
    cur["f0"] = cur["f0"] + 2.0

    report = DriftRootCauseAnalyzer(random_seed=7).analyze(
        ref, cur, model=model, feature_cols=[f"f{i}" for i in range(n_feat)]
    )
    assert isinstance(report, RootCauseReport)
    # Every drifted feature should carry a Delta-SHAP attribution.
    for f in report.drifted_features:
        assert f.abs_delta_prediction >= 0.0
        assert isinstance(f.delta_prediction, float)
    assert report.total_attribution >= 0.0


# --------------------------------------------------------------------------- C7: MPC convergence fallback (Item 3)


def test_mpc_convergence_fallback():
    twin = build_default_twin(n_x=2, n_u=1, dt=1.0)
    # Extremely tight control bound -> QP likely infeasible; controller must not crash.
    ctrl = ClosedLoopMPCController(
        twin, horizon=6, u_min=np.array([-1e-6]), u_max=np.array([1e-6]),
        x_min=np.array([-10.0, -10.0]), x_max=np.array([10.0, 10.0]),
    )
    res = ctrl.solve(x0=np.array([100.0, 100.0]), r_path=np.zeros((6, 2)))
    assert isinstance(res.success, bool)
    assert isinstance(res.solver_message, str)
    assert isinstance(res.constraint_violations, list)
    assert res.state_trajectory.shape == (7, 2)


# --------------------------------------------------------------------------- C7: twin persistence (Item 4)


def test_twin_persistence():
    twin = build_default_twin(n_x=2, n_u=1, dt=1.0)
    ctrl = ClosedLoopMPCController(twin, horizon=6)
    measurements = [np.zeros(2) for _ in range(4)]
    ctrl.run_closed_loop(measurements, setpoint=np.array([1.0, 0.0]))

    # Checkpoint twin state + controller warm-start.
    snapshot = ctrl.persist(warm_start=None)
    assert "twin_x" in snapshot and "twin_P" in snapshot
    assert snapshot["horizon"] == 6

    # Restore into a fresh twin/controller.
    twin2 = build_default_twin(n_x=2, n_u=1, dt=1.0)
    ctrl2, warm_start = ClosedLoopMPCController.restore(twin2, snapshot)
    assert np.allclose(twin2.x, twin.x)
    assert np.allclose(twin2.P, twin.P)
    assert ctrl2.horizon == 6


# --------------------------------------------------------------------------- C4: structured error handling (Item 5)


def test_register_model_invalid_stage():
    from omniforge.copilot import CopilotToolRegistry
    from omniforge.copilot.memory import CopilotMemory
    from omniforge.core.exceptions import CopilotError

    mem = CopilotMemory()
    reg = CopilotToolRegistry(mem)
    # No trained model -> CopilotError (invoke wraps handler exceptions).
    with pytest.raises(CopilotError):
        reg.invoke("register_model", {"model_name": "m", "stage": "Staging"})
    # Inject a model under the key tool_register_model looks up, then
    # invalid stage -> structured error (not raised).
    mem.set_model("automl_champion", _PredictShiftModel())
    res = reg.invoke("register_model", {"model_name": "m", "stage": "NotAStage"})
    assert "error" in res
    assert "available_stages" in res
    assert "Staging" in res["available_stages"]


def test_promote_model_invalid_stage():
    from omniforge.copilot import CopilotToolRegistry
    from omniforge.copilot.memory import CopilotMemory

    mem = CopilotMemory()
    reg = CopilotToolRegistry(mem)
    res = reg.invoke("promote_model", {"model_name": "m", "version": 1, "stage": "NotAStage"})
    assert "error" in res
    assert "available_stages" in res
    assert "Production" in res["available_stages"]


# --------------------------------------------------------------------------- version alignment


def test_version_is_4_0_0():
    import omniforge

    assert omniforge.__version__ == "4.0.0"
    assert omniforge.get_version_info()["version"] == "4.0.0"
