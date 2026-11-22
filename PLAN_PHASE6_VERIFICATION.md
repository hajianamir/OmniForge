# Phase 6 (v4.0.0) — Verification Plan: "Not Done" Items

This plan converts the "What Was NOT Done / Needs Further Development" findings into
actionable implementation items, each with code changes + verification tests.

## Items

| # | Area | Item | Type | Verification |
|---|------|------|------|--------------|
| 1 | C5 | Temporal DAG orientation via rolling-window onset-of-drift | Enhancement | `test_temporal_dag_orientation` |
| 2 | C5 | Exact ΔSHAP integration (replace feature-replacement approx.) | Enhancement | `test_deltashap_integration` |
| 3 | C7 | MPC convergence fallback (bound relaxation on SLSQP failure) | Enhancement | `test_mpc_convergence_fallback` |
| 4 | C7 | Twin/controller persistence across session calls | Enhancement | `test_twin_persistence` |
| 5 | C4 | Structured error handling for `register_model`/`promote_model` | Fix | `test_register_model_invalid_stage` |
| 6 | CI | Confirm GitHub Actions 5/5 green on HEAD (external; cannot run in-container) | Gate | `gh api .../commits/<sha>/status` |

Each item below lists: (a) code change, (b) test, (c) acceptance criterion.

---

## Item 1 — C5 Temporal DAG Orientation

### Code change
`src/omniforge/mlops/root_cause.py` — `CausalDiscovery`:
- Add `_rolling_window_drift_onset(df, time_col)` returning per-feature "first shift step" via rolling PSI over time windows.
- In `discover()`, when `time_col` is provided, break orientation ties by earliest drift onset (earliest-shift feature = upstream root cause).

### Test
`tests/unit/test_phase6.py` — `test_temporal_dag_orientation`:
- Build reference/current frames with a `time_col` where `source` shifts at t=100 and `derived` shifts at t=200.
- Run `CausalDiscovery.discover(df, time_col=time_col)`.
- Assert `derived`'s parent set includes `source` (source upstream of derived).
- Assert no self-loops.

### Acceptance
Orientation respects temporal causality: the feature that shifts first is classified upstream.

---

## Item 2 — C5 Exact ΔSHAP Integration

### Code change
`src/omniforge/mlops/root_cause.py` — `DriftRootCauseAnalyzer.analyze`:
- When `model` is provided, compute Delta-attribution via `InterpretabilityEngine.explain_shap_local` (exact Monte Carlo Shapley) instead of feature-replacement approximation.
- Fall back to feature-replacement if SHAP is unavailable.

### Test
`tests/unit/test_phase6.py` — `test_deltashap_integration`:
- Stub model with `predict`/`predict_proba`/`task_type`/`classes_`.
- Run `DriftRootCauseAnalyzer.analyze(ref, cur, model=stub, feature_cols=[...])`.
- Assert every drifted feature has non-zero `abs_delta_prediction` attribution.
- Assert `total_attribution > 0`.

### Acceptance
Attribution uses exact local Shapley values (not rough feature replacement).

---

## Item 3 — C7 MPC Convergence Fallback

### Code change
`src/omniforge/simulation/closed_loop.py` — `ClosedLoopMPCController.solve`:
- When SLSQP reports `success=False`, retry up to 3 times with bounds relaxed by 10% each iteration.
- If all retries fail, return result with `success=False` and `constraint_violations` listing the failure (do not raise).

### Test
`tests/unit/test_phase6.py` — `test_mpc_convergence_fallback`:
- Build a twin with extremely tight `u_max` that makes the QP infeasible.
- Call `ctrl.solve(x0, r_path)`.
- Assert result contains `success` (bool), `solver_message`, `constraint_violations` (list).
- Assert it does NOT raise an exception.

### Acceptance
Controller never crashes on infeasible QP; returns structured failure report.

---

## Item 4 — C7 Twin Persistence

### Code change
`src/omniforge/simulation/closed_loop.py` — `ClosedLoopMPCController`:
- Add `persist(state, key)` / `restore(state, key)` static helpers to save/restore `twin.x`, `P`, and controller warm-start in a dict (workbench session state).
- `run_closed_loop` accepts optional `state_key` to auto-persist after each step.

### Test
`tests/unit/test_phase6.py` — `test_twin_persistence`:
- Build twin + controller, run 3 steps.
- Persist to a dict under key `"twin_1"`.
- Build a fresh twin/controller, restore from dict.
- Assert restored twin state equals pre-restore state.

### Acceptance
Twin state can be checkpointed and restored across session boundaries.

---

## Item 5 — C4 Structured Error Handling

### Code change
`src/omniforge/copilot/tools.py` — `tool_register_model`, `tool_promote_model`:
- Wrap `ModelStage(stage)` in try/except; on invalid stage return `{"error": ..., "available_stages": [...]}` instead of raising.

### Test
`tests/unit/test_phase6.py` — `test_register_model_invalid_stage`:
- Copilot memory with no model → `tool_register_model` returns structured error (not raise).
- With a model + invalid stage → returns `{"error": ..., "available_stages": [...]}`.

### Acceptance
Invalid stage never propagates an unhandled exception; structured error returned.

---

## Item 6 — CI Gate (external)

- Check `gh api repos/hajianamir/OmniForge/commits/<HEAD>/status` → state `success`, 5 contexts.
- Cannot run in-container (no pytest/runners); must be verified on GitHub Actions.
