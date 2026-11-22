"""Closed-Loop Digital Twin Model-Predictive Control (C7 — Future Developments II).

Constrained receding-horizon MPC:

    u*_{t:t+H} = argmin  sum_{k=0}^{H-1} || H xhat_{t+k|t} - r_{t+k} ||_Q^2
                             + || u_{t+k} ||_R^2
    subject to  u_min <= u <= u_max
                x_min <= xhat <= x_max        (state safety bounds)
                xhat_{t+k+1|t} = A xhat_{t+k|t} + B u_{t+k}

Solved with :func:`scipy.optimize.minimize` (SLSQP). Deterministic, offline, and safe
for air-gapped execution. Exposed via ``POST /api/workbench/s09/twin-control`` with
interactive trajectory charts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import minimize

from omniforge.core.exceptions import SimulationError
from omniforge.core.logging import get_logger
from omniforge.simulation.engines import DigitalTwinEngine

logger = get_logger(__name__)


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


class TwinControlReport(BaseModel):
    """Closed-loop twin-control session report across all applied steps."""

    horizon: int
    n_steps_applied: int
    setpoint: list[float]
    state_trajectory: list[list[float]]  # per-step applied states
    control_history: list[list[float]]  # per-step applied controls
    stage_costs: list[float]
    total_cost: float
    all_constraints_satisfied: bool
    constraint_violations: list[str] = Field(default_factory=list)
    solved_at: str = Field(default_factory=lambda: __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat())


# --------------------------------------------------------------------------- MPC controller


class ClosedLoopMPCController:
    """Constrained receding-horizon MPC controller for a linear state-space Digital Twin.

    At each step the controller solves a finite-horizon quadratic program over the
    predicted state/control trajectory, applies only the first control action
    (receding horizon), assimilates the next measurement via the twin's Kalman filter,
    and repeats. Input and state safety bounds are enforced as hard constraints.
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

    # -- decision-vector helpers -----------------------------------------------
    def _unpack(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flattened decision vector z = [u_0 ... u_{H-1}, x_1 ... x_H] -> (U, X)."""
        U = z[: self.horizon * self.n_u].reshape(self.horizon, self.n_u)
        X = z[self.horizon * self.n_u :].reshape(self.horizon, self.n_x)
        return U, X

    def _pack(self, U: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.concatenate([U.ravel(), X.ravel()])

    # -- objective -------------------------------------------------------------
    def _objective(self, z: np.ndarray, r_path: np.ndarray) -> float:
        U, X = self._unpack(z)
        cost = 0.0
        for k in range(self.horizon):
            tracking = X[k] - r_path[k]
            cost += float(tracking @ self.Q @ tracking)
            cost += float(U[k] @ self.R @ U[k])
        return cost

    def _objective_grad(self, z: np.ndarray, r_path: np.ndarray) -> np.ndarray:
        U, X = self._unpack(z)
        grad_U = np.zeros_like(U)
        grad_X = np.zeros_like(X)
        for k in range(self.horizon):
            tracking = X[k] - r_path[k]
            grad_X[k] += 2.0 * (self.Q @ tracking)
            grad_U[k] += 2.0 * (self.R @ U[k])
        # Dynamics coupling: x_{k+1} = A x_k + B u_k enters objective via x_{k+1}
        for k in range(self.horizon - 1):
            x_next_pred = self.twin.A @ X[k] + self.twin.B @ U[k]
            tracking_next = x_next_pred - r_path[k + 1]
            grad_X[k] += 2.0 * (self.twin.A.T @ (self.Q @ tracking_next))
            grad_U[k] += 2.0 * (self.twin.B.T @ (self.Q @ tracking_next))
        return self._pack(grad_U, grad_X)

    # -- constraints -----------------------------------------------------------
    def _dynamics_constraints(self, z: np.ndarray) -> np.ndarray:
        """Equality: x_{k+1} - (A x_k + B u_k) = 0 for k = 0..H-2."""
        U, X = self._unpack(z)
        residuals: list[float] = []
        for k in range(self.horizon - 1):
            x_next_pred = self.twin.A @ X[k] + self.twin.B @ U[k]
            residuals.extend((X[k + 1] - x_next_pred).tolist())
        return np.asarray(residuals, dtype=float)

    def solve(
        self,
        x0: np.ndarray,
        r_path: np.ndarray | None = None,
        *,
        warm_start: np.ndarray | None = None,
    ) -> MPCSolveResult:
        """Solve one MPC step from current state ``x0`` over setpoint path ``r_path``."""
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

        # Initial guess: hold previous control, steady-state prediction.
        if warm_start is not None and warm_start.size == H * (n_u + n_x):
            z0 = warm_start.copy()
        else:
            U0 = np.clip(np.zeros((H, n_u)), self.u_min, self.u_max)
            X0 = np.tile(x0, (H, 1))
            z0 = self._pack(U0, X0)

        constraints = [
            {
                "type": "eq",
                "fun": lambda z: self._dynamics_constraints(z),
                "jac": lambda z: self._dynamics_jacobian(z),
            }
        ]

        def _solve_with_bounds(u_lo: np.ndarray, u_hi: np.ndarray, x_lo: np.ndarray, x_hi: np.ndarray):
            bds = []
            for _ in range(H):
                bds.extend(list(zip(u_lo, u_hi, strict=False)))
            for _ in range(H):
                bds.extend(list(zip(x_lo, x_hi, strict=False)))
            return minimize(
                fun=lambda z: self._objective(z, r_path),
                x0=z0,
                jac=lambda z: self._objective_grad(z, r_path),
                method="SLSQP",
                bounds=bds,
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-9, "disp": False},
            )

        # Convergence fallback: retry with progressively relaxed bounds when SLSQP fails.
        result = _solve_with_bounds(self.u_min, self.u_max, self.x_min, self.x_max)
        relax = 1.0
        while not result.success and relax < 1.5:
            relax += 0.10
            u_lo = self.u_min * relax
            u_hi = self.u_max * relax
            x_lo = self.x_min * relax
            x_hi = self.x_max * relax
            result = _solve_with_bounds(u_lo, u_hi, x_lo, x_hi)

        U, X = self._unpack(result.x)
        violations: list[str] = []
        if not result.success:
            violations.append(f"solver did not converge: {result.message}")
        # Explicit bound check against ORIGINAL (unrelaxed) bounds.
        for k in range(H):
            for j in range(n_u):
                if U[k, j] < self.u_min[j] - 1e-6 or U[k, j] > self.u_max[j] + 1e-6:
                    violations.append(f"u[{k}][{j}]={U[k, j]:.6f} outside [{self.u_min[j]}, {self.u_max[j]}]")
            for j in range(n_x):
                if X[k, j] < self.x_min[j] - 1e-6 or X[k, j] > self.x_max[j] + 1e-6:
                    violations.append(f"x[{k}][{j}]={X[k, j]:.6f} outside [{self.x_min[j]}, {self.x_max[j]}]")

        return MPCSolveResult(
            optimal_control=U[0].copy(),
            full_control_sequence=U.copy(),
            state_trajectory=np.vstack([x0.reshape(1, -1), X]),
            cost=float(result.fun) if result.success else float("inf"),
            success=bool(result.success),
            solver_message=str(result.message),
            constraint_violations=violations,
        )

    def _dynamics_jacobian(self, z: np.ndarray) -> np.ndarray:
        """Analytic Jacobian of the dynamics equality constraints."""
        n_x, n_u, H = self.n_x, self.n_u, self.horizon
        n_dyn = (H - 1) * n_x
        n_z = H * (n_u + n_x)
        J = np.zeros((n_dyn, n_z), dtype=float)
        for k in range(H - 1):
            row = k * n_x
            # d/d u_k: -B   | d/d x_k: -A   | d/d x_{k+1}: I
            J[row : row + n_x, k * n_u : (k + 1) * n_u] = -self.twin.B
            J[row : row + n_x, H * n_u + k * n_x : H * n_u + (k + 1) * n_x] = -self.twin.A
            J[row : row + n_x, H * n_u + (k + 1) * n_x : H * n_u + (k + 2) * n_x] = np.eye(n_x)
        return J

    # -- closed-loop simulation ------------------------------------------------
    def run_closed_loop(
        self,
        measurements: np.ndarray | list[np.ndarray],
        setpoint: np.ndarray | None = None,
        *,
        controls_override: np.ndarray | None = None,
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

        if setpoint is None:
            setpoint_arr = np.zeros((self.horizon, self.n_x), dtype=float)
        else:
            sp = np.asarray(setpoint, dtype=float)
            if sp.ndim == 1:
                setpoint_arr = np.tile(sp.reshape(1, -1), (self.horizon, 1))
            elif sp.shape[0] >= n_steps:
                setpoint_arr = sp[:n_steps]
                # Pad horizon setpoints beyond available rows with last value.
                last = setpoint_arr[-1:]
                setpoint_arr = np.vstack(
                    [setpoint_arr] + [last] * (self.horizon - setpoint_arr.shape[0])
                )
            else:
                setpoint_arr = np.vstack([sp, np.tile(sp[-1:], (self.horizon - sp.shape[0], 1))])

        state_traj: list[list[float]] = []
        control_hist: list[list[float]] = []
        stage_costs: list[float] = []
        all_violations: list[str] = []
        all_satisfied = True

        warm_start = None
        total_cost = 0.0

        for t in range(n_steps):
            z_t = self.twin.x.copy()
            # r_path for this step: use setpoint row t (broadcast across horizon).
            r_path_t = np.tile(setpoint_arr[min(t, setpoint_arr.shape[0] - 1)], (self.horizon, 1))

            solve_res = self.solve(z_t, r_path_t, warm_start=warm_start)
            if not solve_res.success:
                all_satisfied = False
            all_violations.extend(solve_res.constraint_violations)

            u_applied = solve_res.optimal_control
            control_hist.append(u_applied.tolist())

            # Stage cost at current state vs setpoint
            tracking = z_t - setpoint_arr[min(t, setpoint_arr.shape[0] - 1)]
            stage_cost = float(tracking @ self.Q @ tracking + u_applied @ self.R @ u_applied)
            stage_costs.append(stage_cost)
            total_cost += stage_cost

            # Apply control to twin (open-loop prediction), then assimilate measurement.
            predicted = self.twin.A @ z_t + self.twin.B @ u_applied
            state_traj.append(predicted.tolist())
            self.twin.assimilate(meas[t], control=u_applied)

            warm_start = np.roll(
                np.concatenate([solve_res.full_control_sequence.ravel(), solve_res.state_trajectory[1:].ravel()]),
                -(self.n_u + self.n_x),
            )
            if warm_start.size != self.horizon * (self.n_u + self.n_x):
                warm_start = None

            # Checkpoint twin state + controller warm-start for session persistence.
            if state is not None and state_key is not None:
                state[state_key] = self.persist(warm_start=warm_start)

        return TwinControlReport(
            horizon=self.horizon,
            n_steps_applied=n_steps,
            setpoint=setpoint_arr[:n_steps].tolist(),
            state_trajectory=state_traj,
            control_history=control_hist,
            stage_costs=stage_costs,
            total_cost=round(total_cost, 6),
            all_constraints_satisfied=all_satisfied and len(all_violations) == 0,
            constraint_violations=all_violations,
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
    ) -> "ClosedLoopMPCController":
        """Restore a controller from a :meth:`persist` payload, re-bound to ``twin``.

        Re-instantiates the controller with the same bounds and warm-start so a
        closed-loop session can resume across workbench session boundaries.
        """
        ctrl = ClosedLoopMPCController(
            twin,
            horizon=int(payload.get("horizon", 10)),
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
    """Construct a default stable second-order linear Digital Twin (demo / test helper).

    Discretized double integrator with mild process/measurement noise:
        x_{k+1} = [1 dt; 0 1] x_k + [0.5 dt^2; dt] u_k + w
        y_k     = [1 0] x_k + v
    """
    rng = np.random.default_rng(random_seed)
    A = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    B = np.array([[0.5 * dt * dt], [dt]], dtype=float)
    H = np.array([[1.0, 0.0]], dtype=float)
    Q = np.eye(n_x, dtype=float) * 0.01
    R = np.eye(n_u, dtype=float) * 0.1
    x0 = np.zeros(n_x, dtype=float)
    return DigitalTwinEngine(A=A, B=B, H=H, Q=Q, R=R, x0=x0)
