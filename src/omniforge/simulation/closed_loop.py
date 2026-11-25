"""Closed-Loop Digital Twin Model-Predictive Control (C7 — Future Developments II).

Constrained receding-horizon MPC:

    u*_{t:t+H} = argmin  sum_{k=0}^{H-1} || xhat_{t+k+1|t} - r_{t+k} ||_Q^2
                             + || u_{t+k} ||_R^2
    subject to  u_min <= u_{t+k} <= u_max
                x_min <= xhat_{t+k+1|t} <= x_max   (state safety bounds; softened via slack if infeasible)
                xhat_{t+1|t}   = A xhat_{t|t} + B u_t
                xhat_{t+k+1|t} = A xhat_{t+k|t} + B u_{t+k}

Solved with :func:`scipy.optimize.minimize` (SLSQP). Deterministic, offline, and safe
for air-gapped execution. Exposed via ``POST /api/workbench/s09/twin-control`` with
interactive trajectory charts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import minimize

from omniforge.core.exceptions import SimulationError
from omniforge.core.logging import get_logger
from omniforge.simulation.engines import DigitalTwinEngine

logger = get_logger(__name__)
MPCFeasibility = Literal["hard_feasible", "soft_state_feasible", "solver_failed"]


# --------------------------------------------------------------------------- result models


@dataclass
class MPCSolveResult:
    """Outcome of a single MPC solve at one time step."""

    optimal_control: np.ndarray  # first control action u_t (receding-horizon apply)
    full_control_sequence: np.ndarray  # u_{t:t+H} (all H steps)
    state_trajectory: np.ndarray  # xhat_{t:t+H+1} under optimal control
    cost: float
    success: bool
    solver_message: str
    constraint_violations: list[str] = field(default_factory=list)
    solve_time_ms: float = 0.0
    slack_used: float = 0.0
    feasibility: MPCFeasibility = "solver_failed"


class TwinControlReport(BaseModel):
    """Closed-loop twin-control session report across all applied steps."""

    horizon: int
    n_steps_applied: int
    setpoint: list[list[float]]  # per-step applied setpoints
    state_trajectory: list[list[float]]  # per-step applied states
    control_history: list[list[float]]  # per-step applied controls
    stage_costs: list[float]
    total_cost: float
    all_constraints_satisfied: bool
    constraint_violations: list[str] = Field(default_factory=list)
    step_feasibility: list[MPCFeasibility] = Field(default_factory=list)
    solved_at: str = Field(default_factory=lambda: __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat())


# --------------------------------------------------------------------------- MPC controller


class ClosedLoopMPCController:
    """Constrained receding-horizon MPC controller for a linear state-space Digital Twin.

    At each step the controller solves a finite-horizon quadratic program over the
    predicted state/control trajectory, applies only the first control action
    (receding horizon), assimilates the next measurement via the twin's Kalman filter,
    and repeats. Input limits are hard constraints. State bounds can use reported slack.
    """

    def __init__(
        self,
        twin: DigitalTwinEngine,
        *,
        horizon: int = 10,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        u_min: np.ndarray | None = None,
        u_max: np.ndarray | None = None,
        x_min: np.ndarray | None = None,
        x_max: np.ndarray | None = None,
        random_seed: int = 42,
    ) -> None:
        self.twin = twin
        self.n_x = int(twin.A.shape[0])
        self.n_u = int(twin.B.shape[1])
        self.horizon = max(1, int(horizon))
        self.Q = (
            np.eye(self.n_x, dtype=float) if Q is None else np.asarray(Q, dtype=float)
        )
        self.R = (
            np.eye(self.n_u, dtype=float) if R is None else np.asarray(R, dtype=float)
        )
        self.u_min = (
            -np.inf * np.ones(self.n_u)
            if u_min is None
            else np.asarray(u_min, dtype=float).ravel()
        )
        self.u_max = (
            np.inf * np.ones(self.n_u)
            if u_max is None
            else np.asarray(u_max, dtype=float).ravel()
        )
        self.x_min = (
            -np.inf * np.ones(self.n_x)
            if x_min is None
            else np.asarray(x_min, dtype=float).ravel()
        )
        self.x_max = (
            np.inf * np.ones(self.n_x)
            if x_max is None
            else np.asarray(x_max, dtype=float).ravel()
        )
        self.random_seed = int(random_seed)
        if self.Q.shape != (self.n_x, self.n_x):
            raise SimulationError("Q must be shape (n_x, n_x).")
        if self.R.shape != (self.n_u, self.n_u):
            raise SimulationError("R must be shape (n_u, n_u).")
        if self.u_min.shape != (self.n_u,) or self.u_max.shape != (self.n_u,):
            raise SimulationError(f"u_min and u_max must have length {self.n_u}.")
        if self.x_min.shape != (self.n_x,) or self.x_max.shape != (self.n_x,):
            raise SimulationError(f"x_min and x_max must have length {self.n_x}.")
        if np.any(self.u_min > self.u_max):
            raise SimulationError("u_min cannot exceed u_max.")
        if np.any(self.x_min > self.x_max):
            raise SimulationError("x_min cannot exceed x_max.")

    # -- decision-vector helpers -----------------------------------------------
    def _unpack(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flattened decision vector z = [u_0 ... u_{H-1}, x_1 ... x_H] -> (U, X)."""
        U = z[: self.horizon * self.n_u].reshape(self.horizon, self.n_u)
        X = z[self.horizon * self.n_u :].reshape(self.horizon, self.n_x)
        return U, X

    def _pack(self, U: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.concatenate([U.ravel(), X.ravel()])

    def _rollout_open_loop(self, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Roll state dynamics forward from ``x0`` under control sequence ``U`` -> ``X`` (H, n_x)."""
        X = np.empty((self.horizon, self.n_x), dtype=float)
        prev = x0
        for k in range(self.horizon):
            prev = self.twin.A @ prev + self.twin.B @ U[k]
            X[k] = prev
        return X

    @staticmethod
    def _to_scipy_bounds(lo: np.ndarray, hi: np.ndarray, horizon: int) -> list[tuple[float | None, float | None]]:
        step_bounds = [
            (
                float(lo_j) if np.isfinite(lo_j) else None,
                float(hi_j) if np.isfinite(hi_j) else None,
            )
            for lo_j, hi_j in zip(lo, hi, strict=False)
        ]
        return step_bounds * horizon

    # -- objective -------------------------------------------------------------
    def _objective(self, z: np.ndarray, r_path: np.ndarray) -> float:
        U, X = self._unpack(z)
        tracking = X - r_path
        q_cost = float(np.sum((tracking @ self.Q) * tracking))
        r_cost = float(np.sum((U @ self.R) * U))
        return q_cost + r_cost

    def _objective_grad(self, z: np.ndarray, r_path: np.ndarray) -> np.ndarray:
        """Exact analytic gradient of ``_objective`` wrt independent decision coordinates ``(U, X)``."""
        U, X = self._unpack(z)
        tracking = X - r_path
        grad_X = tracking @ (self.Q + self.Q.T)
        grad_U = U @ (self.R + self.R.T)
        return self._pack(grad_U, grad_X)

    # -- constraints -----------------------------------------------------------
    def _dynamics_constraints(self, z: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """Equality: x_1 = A x_0 + B u_0 and x_{k+1} = A x_k + B u_k for k = 1..H-1."""
        U, X = self._unpack(z)
        x0_vec = (
            np.asarray(self.twin.x, dtype=float).ravel()
            if x0 is None
            else np.asarray(x0, dtype=float).ravel()
        )
        residuals = np.empty((self.horizon, self.n_x), dtype=float)
        residuals[0] = X[0] - (self.twin.A @ x0_vec + self.twin.B @ U[0])
        for k in range(1, self.horizon):
            residuals[k] = X[k] - (self.twin.A @ X[k - 1] + self.twin.B @ U[k])
        return residuals.ravel()

    def solve(
        self,
        x0: np.ndarray,
        r_path: np.ndarray | None = None,
        *,
        warm_start: np.ndarray | None = None,
        slack_penalty: float = 1e4,
    ) -> MPCSolveResult:
        """Solve one MPC step from current state ``x0`` over setpoint path ``r_path``.

        Enforces hard input bounds ``u_min <= u_k <= u_max`` and attempts hard state
        safety bounds first. When disturbances drive the plant outside the 1-step
        reachable safe set, Stage 2 softens state bounds via non-negative slack
        variables ``x_min - eps <= x_k <= x_max + eps`` with an exact L1 + smooth L2
        penalty (Scokaert & Rawlings 1999; Kerrigan & Maciejowski 2000).
        """
        n_x, n_u, H = self.n_x, self.n_u, self.horizon
        if r_path is None:
            r_path = np.zeros((H, n_x), dtype=float)
        else:
            r_path = np.asarray(r_path, dtype=float)
            if r_path.ndim == 1:
                r_path = np.tile(r_path.reshape(1, -1), (H, 1))
            if r_path.shape != (H, n_x):
                raise SimulationError(f"r_path must be shape ({H}, {n_x}), got {r_path.shape}.")
        x0 = np.asarray(x0, dtype=float).ravel()
        if x0.shape != (n_x,):
            raise SimulationError(f"x0 must be shape ({n_x},), got {x0.shape}.")
        if not np.isfinite(x0).all() or not np.isfinite(r_path).all() or not np.isfinite(slack_penalty):
            raise SimulationError("Initial state, reference path, and slack penalty must be finite.")

        n_z_base = H * (n_u + n_x)
        # Dynamically consistent initial guess so equality constraints hold at iteration 0.
        if warm_start is not None and warm_start.size == n_z_base:
            U_ws, _ = self._unpack(np.asarray(warm_start, dtype=float).ravel())
            U0 = np.clip(U_ws, self.u_min, self.u_max)
        else:
            U0 = np.clip(np.zeros((H, n_u), dtype=float), self.u_min, self.u_max)
        X0 = self._rollout_open_loop(x0, U0)

        # Stage-2 decision vector: z_s = [U (H*n_u), X (H*n_x), slack_x (H*n_x)]
        slack_size = H * n_x
        eff_penalty = max(float(slack_penalty), 1.0)

        def _objective_with_slack(z: np.ndarray, r_ref: np.ndarray) -> float:
            base_cost = self._objective(z[:n_z_base], r_ref)
            slack = z[n_z_base:]
            return base_cost + float(eff_penalty * (np.sum(slack) + np.sum(slack * slack)))

        def _objective_grad_with_slack(z: np.ndarray, r_ref: np.ndarray) -> np.ndarray:
            grad_base = self._objective_grad(z[:n_z_base], r_ref)
            slack = z[n_z_base:]
            grad_slack = eff_penalty * (1.0 + 2.0 * slack)
            return np.concatenate([grad_base, grad_slack])

        def _dynamics_constraints_with_slack(z: np.ndarray) -> np.ndarray:
            return self._dynamics_constraints(z[:n_z_base], x0)

        def _dynamics_jacobian_with_slack(z: np.ndarray) -> np.ndarray:
            J_base = self._dynamics_jacobian(z[:n_z_base], x0)
            return np.hstack([J_base, np.zeros((J_base.shape[0], slack_size), dtype=float)])

        def _build_soft_state_inequalities(
            x_lo: np.ndarray, x_hi: np.ndarray
        ) -> tuple[list[tuple[int, int, float, float]], np.ndarray]:
            """Build linear inequalities c(z_s) >= 0 softening x_lo - s <= X <= x_hi + s."""
            specs: list[tuple[int, int, float, float]] = []
            # Each spec: (x_idx, s_idx, x_sign, bound_offset) where x_sign*z[x_idx] + z[s_idx] + bound_offset >= 0
            for k in range(H):
                for j in range(n_x):
                    x_idx = H * n_u + k * n_x + j
                    s_idx = n_z_base + k * n_x + j
                    if np.isfinite(x_lo[j]):
                        specs.append((x_idx, s_idx, 1.0, -float(x_lo[j])))
                    if np.isfinite(x_hi[j]):
                        specs.append((x_idx, s_idx, -1.0, float(x_hi[j])))
            J_ineq = np.zeros((len(specs), n_z_base + slack_size), dtype=float)
            for row_i, (x_idx, s_idx, x_sign, _offset) in enumerate(specs):
                J_ineq[row_i, x_idx] = x_sign
                J_ineq[row_i, s_idx] = 1.0
            return specs, J_ineq

        def _solve_with_bounds(
            u_lo: np.ndarray,
            u_hi: np.ndarray,
            x_lo: np.ndarray,
            x_hi: np.ndarray,
            use_slack: bool = False,
        ) -> Any:
            u_step = np.clip(U0, u_lo, u_hi)
            x_step = self._rollout_open_loop(x0, u_step)
            z_init = self._pack(u_step, x_step)
            if use_slack:
                bds: list[tuple[float | None, float | None]] = []
                bds.extend(self._to_scipy_bounds(u_lo, u_hi, H))
                bds.extend([(None, None)] * (H * n_x))
                bds.extend([(0.0, None)] * slack_size)

                low_viol = np.where(np.isfinite(x_lo), np.maximum(0.0, x_lo - x_step), 0.0)
                high_viol = np.where(np.isfinite(x_hi), np.maximum(0.0, x_step - x_hi), 0.0)
                s_init = np.maximum(low_viol, high_viol).ravel() + 1e-9
                z0_slack = np.concatenate([z_init, s_init])

                ineq_specs, J_ineq = _build_soft_state_inequalities(x_lo, x_hi)
                constraints: list[dict[str, Any]] = [
                    {
                        "type": "eq",
                        "fun": _dynamics_constraints_with_slack,
                        "jac": _dynamics_jacobian_with_slack,
                    }
                ]
                if ineq_specs:
                    x_indices = np.array([sp[0] for sp in ineq_specs], dtype=int)
                    s_indices = np.array([sp[1] for sp in ineq_specs], dtype=int)
                    x_signs = np.array([sp[2] for sp in ineq_specs], dtype=float)
                    offsets = np.array([sp[3] for sp in ineq_specs], dtype=float)
                    constraints.append(
                        {
                            "type": "ineq",
                            "fun": lambda z: x_signs * z[x_indices] + z[s_indices] + offsets,
                            "jac": lambda _z: J_ineq,
                        }
                    )

                # Preserve the minimizer while keeping the initial gradient in a
                # numerically tractable range for SLSQP's line search.
                scale = max(1.0, float(np.max(np.abs(_objective_grad_with_slack(z0_slack, r_path)))))
                return minimize(
                    fun=lambda z: _objective_with_slack(z, r_path) / scale,
                    x0=z0_slack,
                    jac=lambda z: _objective_grad_with_slack(z, r_path) / scale,
                    method="SLSQP",
                    bounds=bds,
                    constraints=constraints,
                    options={"maxiter": 300, "ftol": 1e-10, "disp": False},
                )

            bds_hard = self._to_scipy_bounds(u_lo, u_hi, H) + self._to_scipy_bounds(x_lo, x_hi, H)
            return minimize(
                fun=lambda z: self._objective(z, r_path),
                x0=z_init,
                jac=lambda z: self._objective_grad(z, r_path),
                method="SLSQP",
                bounds=bds_hard,
                constraints=[
                    {
                        "type": "eq",
                        "fun": lambda z: self._dynamics_constraints(z, x0),
                        "jac": lambda z: self._dynamics_jacobian(z, x0),
                    }
                ],
                options={"maxiter": 200, "ftol": 1e-9, "disp": False},
            )

        start_time = time.perf_counter()

        # Stage 1: Try hard constraints without slack.
        result = _solve_with_bounds(self.u_min, self.u_max, self.x_min, self.x_max)
        U, X = self._unpack(result.x) if getattr(result, "x", None) is not None else (U0, X0)
        dyn_residual = float(np.max(np.abs(self._dynamics_constraints(self._pack(U, X), x0))))
        U = np.clip(U, self.u_min, self.u_max)
        X = self._rollout_open_loop(x0, U)
        state_excess = np.maximum(self.x_min - X, X - self.x_max)
        feasibility: MPCFeasibility = "solver_failed"
        if result.success and dyn_residual <= 1e-6 and np.all(state_excess <= 1e-6):
            feasibility = "hard_feasible"

        # Stage 2: If hard bounds are infeasible, soften state safety bounds with slack variables.
        slack_used = 0.0
        if feasibility == "solver_failed":
            result = _solve_with_bounds(
                self.u_min, self.u_max, self.x_min, self.x_max, use_slack=True
            )
            if result.success:
                z_slack = result.x
                U_s, X_s = self._unpack(z_slack[:n_z_base])
                candidate = np.clip(U_s, self.u_min, self.u_max)
                trajectory = self._rollout_open_loop(x0, candidate)
                slack = z_slack[n_z_base:].reshape(H, n_x)
                state_excess = np.maximum(self.x_min - trajectory, trajectory - self.x_max)
                if (
                    np.all(np.abs(U_s - candidate) <= 1e-6)
                    and np.all(np.abs(X_s - trajectory) <= 1e-6)
                    and np.all(state_excess <= slack + 1e-6)
                    and np.all(slack >= -1e-6)
                ):
                    U, X = candidate, trajectory
                    slack_used = max(0.0, float(state_excess.max()))
                    feasibility = "soft_state_feasible" if slack_used > 1e-6 else "hard_feasible"

        success = feasibility != "solver_failed"
        if not success:
            # Return the bounded, dynamically consistent initial trajectory.
            U, X = U0, X0

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        violations: list[str] = []
        if not success:
            violations.append(f"solver returned no verified feasible solution: {result.message}")
        # Explicit bound check against ORIGINAL (unrelaxed) bounds.
        for k in range(H):
            for j in range(n_u):
                if U[k, j] < self.u_min[j] - 1e-6 or U[k, j] > self.u_max[j] + 1e-6:
                    violations.append(
                        f"u[{k}][{j}]={U[k, j]:.6f} outside [{self.u_min[j]}, {self.u_max[j]}]"
                    )
            for j in range(n_x):
                if X[k, j] < self.x_min[j] - 1e-6 or X[k, j] > self.x_max[j] + 1e-6:
                    violations.append(
                        f"x[{k}][{j}]={X[k, j]:.6f} outside [{self.x_min[j]}, {self.x_max[j]}]"
                    )

        traj_cost = self._objective(self._pack(U, X), r_path) if success else float("inf")
        return MPCSolveResult(
            optimal_control=U[0].copy(),
            full_control_sequence=U.copy(),
            state_trajectory=np.vstack([x0.reshape(1, -1), X]),
            cost=traj_cost,
            success=success,
            solver_message=str(result.message),
            constraint_violations=violations,
            solve_time_ms=round(elapsed_ms, 2),
            slack_used=round(slack_used, 6),
            feasibility=feasibility,
        )

    def _dynamics_jacobian(self, z: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """Analytic Jacobian of the initial-state + horizon dynamics equality constraints."""
        _ = (z, x0)
        n_x, n_u, H = self.n_x, self.n_u, self.horizon
        n_dyn = H * n_x
        n_z = H * (n_u + n_x)
        J = np.zeros((n_dyn, n_z), dtype=float)
        # k = 0: X[0] - (A x0 + B U[0]) = 0
        J[0:n_x, 0:n_u] = -self.twin.B
        J[0:n_x, H * n_u : H * n_u + n_x] = np.eye(n_x)
        # k = 1..H-1: X[k] - (A X[k-1] + B U[k]) = 0
        for k in range(1, H):
            row = k * n_x
            J[row : row + n_x, k * n_u : (k + 1) * n_u] = -self.twin.B
            J[row : row + n_x, H * n_u + (k - 1) * n_x : H * n_u + k * n_x] = -self.twin.A
            J[row : row + n_x, H * n_u + k * n_x : H * n_u + (k + 1) * n_x] = np.eye(n_x)
        return J

    # -- closed-loop simulation ------------------------------------------------
    def run_closed_loop(
        self,
        measurements: np.ndarray | list[np.ndarray],
        setpoint: np.ndarray | None = None,
        *,
        state: dict[str, Any] | None = None,
        state_key: str | None = None,
    ) -> TwinControlReport:
        """Run closed-loop MPC over a sequence of measurements.

        For each measurement: solve MPC from current twin state, apply first control,
        assimilate the measurement via Kalman filter, record trajectory. Returns a
        :class:`TwinControlReport` for charting / audit.

        When ``state`` and ``state_key`` are supplied, the twin state and controller
        warm-start are checkpointed into ``state[state_key]`` after every step so the
        closed-loop session can be restored across workbench session boundaries via
        :meth:`restore`.
        """
        meas = [np.asarray(m, dtype=float).ravel() for m in measurements]
        n_steps = len(meas)
        if n_steps == 0:
            raise SimulationError("At least one measurement is required for closed-loop control.")

        # Build setpoint array: one row per step, padded to horizon with last value.
        if setpoint is None:
            setpoint_arr = np.zeros((n_steps, self.n_x), dtype=float)
        else:
            sp = np.asarray(setpoint, dtype=float)
            if sp.ndim == 1:
                sp = sp.reshape(1, -1)
            if sp.shape[0] >= n_steps:
                setpoint_arr = sp[:n_steps]
            else:
                # Tile/ repeat the available setpoints to cover all steps.
                reps = int(np.ceil(n_steps / sp.shape[0]))
                setpoint_arr = np.tile(sp, (reps, 1))[:n_steps]

        state_traj: list[list[float]] = []
        control_hist: list[list[float]] = []
        stage_costs: list[float] = []
        step_feasibility: list[MPCFeasibility] = []
        all_violations: list[str] = []
        all_satisfied = True

        warm_start = None
        total_cost = 0.0

        for t in range(n_steps):
            z_t = self.twin.x.copy()
            # r_path for this step: broadcast current step's setpoint across horizon.
            r_path_t = np.tile(setpoint_arr[t], (self.horizon, 1))

            solve_res = self.solve(z_t, r_path_t, warm_start=warm_start)
            step_feasibility.append(solve_res.feasibility)
            if not solve_res.success:
                all_satisfied = False
            all_violations.extend(solve_res.constraint_violations)

            u_applied = solve_res.optimal_control
            control_hist.append(u_applied.tolist())

            # Stage cost at current state vs setpoint.
            tracking = z_t - setpoint_arr[t]
            stage_cost = float(tracking @ self.Q @ tracking + u_applied @ self.R @ u_applied)
            stage_costs.append(stage_cost)
            total_cost += stage_cost

            # Apply control to twin (open-loop prediction), then assimilate measurement.
            predicted = self.twin.A @ z_t + self.twin.B @ u_applied
            state_traj.append(predicted.tolist())
            self.twin.assimilate(meas[t], control=u_applied)

            # Build warm-start for next step: shift control sequence and state trajectory,
            # padding the terminal control/state so warm_start matches H * (n_u + n_x).
            u_tail = solve_res.full_control_sequence[-1:]
            next_controls = np.vstack([solve_res.full_control_sequence[1:], u_tail])
            x_terminal_next = (self.twin.A @ solve_res.state_trajectory[-1] + self.twin.B @ u_tail[0]).reshape(1, -1)
            next_states = np.vstack([solve_res.state_trajectory[2:], x_terminal_next])
            warm_start = self._pack(next_controls, next_states)

            # Checkpoint twin state + controller warm-start for session persistence.
            if state is not None and state_key is not None:
                state[state_key] = self.persist(warm_start=warm_start)

        return TwinControlReport(
            horizon=self.horizon,
            n_steps_applied=n_steps,
            setpoint=setpoint_arr.tolist(),
            state_trajectory=state_traj,
            control_history=control_hist,
            stage_costs=stage_costs,
            total_cost=round(total_cost, 6),
            all_constraints_satisfied=all_satisfied and len(all_violations) == 0,
            constraint_violations=all_violations,
            step_feasibility=step_feasibility,
        )

    # -- session persistence ----------------------------------------------------
    def persist(self, *, warm_start: np.ndarray | None = None) -> dict[str, Any]:
        """Checkpoint twin state + controller warm-start into a serializable dict.

        Suitable for storage in workbench session state and restoration via
        :meth:`restore`.
        """
        return {
            "twin_x": self.twin.x.tolist(),
            "twin_P": self.twin.P.tolist(),
            "horizon": self.horizon,
            "Q": self.Q.tolist(),
            "R": self.R.tolist(),
            "warm_start": warm_start.tolist() if warm_start is not None else None,
            "u_min": self.u_min.tolist(),
            "u_max": self.u_max.tolist(),
            "x_min": self.x_min.tolist(),
            "x_max": self.x_max.tolist(),
        }

    @staticmethod
    def restore(
        twin: DigitalTwinEngine,
        payload: dict[str, Any],
    ) -> tuple[ClosedLoopMPCController, np.ndarray | None]:
        """Restore a controller from a :meth:`persist` payload, re-bound to ``twin``.

        Re-instantiates the controller with the same bounds and warm-start so a
        closed-loop session can resume across workbench session boundaries.
        """
        ctrl = ClosedLoopMPCController(
            twin,
            horizon=int(payload.get("horizon", 10)),
            Q=np.asarray(payload["Q"], dtype=float) if payload.get("Q") is not None else None,
            R=np.asarray(payload["R"], dtype=float) if payload.get("R") is not None else None,
            u_min=np.asarray(payload["u_min"], dtype=float),
            u_max=np.asarray(payload["u_max"], dtype=float),
            x_min=np.asarray(payload["x_min"], dtype=float),
            x_max=np.asarray(payload["x_max"], dtype=float),
        )
        twin.x = np.asarray(payload["twin_x"], dtype=float)
        twin.P = np.asarray(payload["twin_P"], dtype=float)
        warm_start = (
            np.asarray(payload["warm_start"], dtype=float) if payload.get("warm_start") is not None else None
        )
        return ctrl, warm_start


def build_default_twin(
    n_x: int = 2,
    n_u: int = 1,
    *,
    dt: float = 1.0,
    random_seed: int = 42,
) -> DigitalTwinEngine:
    """Construct a second-order linear Digital Twin with two states and one input.

    Discretized double integrator with mild process/measurement noise:
        x_{k+1} = [1 dt; 0 1] x_k + [0.5 dt^2; dt] u_k + w
        y_k     = x_k + v
    """
    if (n_x, n_u) != (2, 1):
        raise SimulationError("The default double-integrator twin requires n_x=2 and n_u=1.")
    if not np.isfinite(dt) or dt <= 0:
        raise SimulationError("The default twin time step must be finite and positive.")
    A = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    B = np.array([[0.5 * dt * dt], [dt]], dtype=float)
    H = np.eye(n_x, dtype=float)
    Q = np.eye(n_x, dtype=float) * 0.01
    R = np.eye(n_x, dtype=float) * 0.1
    x0 = np.zeros(n_x, dtype=float)
    return DigitalTwinEngine(A=A, B=B, H=H, Q=Q, R=R, x0=x0)
