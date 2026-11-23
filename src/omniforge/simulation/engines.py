"""Enterprise Simulation Suite: Monte Carlo, Discrete-Event, Agent-Based, and Digital Twins."""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from omniforge.core.exceptions import SimulationError
from omniforge.core.logging import get_logger

logger = get_logger(__name__)


class MonteCarloResult(BaseModel):
    """Summary statistics, risk quantiles, and sensitivity drivers from Monte Carlo simulation."""

    n_simulations: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    var_95: float
    cvar_95: float
    percentiles: dict[str, float]
    sensitivity_rank_corr: dict[str, float] = Field(default_factory=dict)


class MonteCarloSimulator:
    """Monte Carlo risk & uncertainty engine supporting correlated inputs and Tornado sensitivity."""

    def __init__(self, n_simulations: int = 5000, random_seed: int = 42) -> None:
        self.n_simulations = max(100, int(n_simulations))
        self.random_seed = random_seed

    def run(
        self,
        input_specs: dict[str, dict[str, Any]],
        model_fn: Callable[[dict[str, np.ndarray]], np.ndarray],
        *,
        correlation_matrix: np.ndarray | None = None,
        correlation_type: Literal["spearman", "pearson_latent"] = "spearman",
    ) -> tuple[MonteCarloResult, np.ndarray]:
        """Sample stochastic inputs (optionally correlated via Kruskal-calibrated Gaussian copula) and evaluate ``model_fn``.

        Supported distributions in ``input_specs``:
            ``normal`` (mean, std), ``lognormal`` (mean, sigma), ``uniform`` (low, high),
            ``triangular`` (left, mode, right), ``poisson`` (lam), ``beta`` (a, b).
        """
        rng = np.random.default_rng(self.random_seed)
        names = list(input_specs.keys())
        d = len(names)
        if d == 0:
            raise SimulationError("MonteCarloSimulator requires at least one stochastic input.")

        # Generate correlated uniform draws via Gaussian copula if correlation matrix supplied
        if correlation_matrix is not None:
            corr = np.asarray(correlation_matrix, dtype=float)
            if corr.shape != (d, d):
                raise SimulationError("correlation_matrix shape must match number of input_specs.")
            corr = np.clip(0.5 * (corr + corr.T), -1.0, 1.0)
            np.fill_diagonal(corr, 1.0)
            if correlation_type == "spearman":
                # Kruskal (1958, JASA 53:814-861) / Clemen & Reilly (1999) exact Spearman-to-Gaussian copula mapping:
                # rho_Gaussian = 2 * sin(pi * rho_Spearman / 6), preserving rank correlation across arbitrary marginals.
                corr = 2.0 * np.sin((np.pi / 6.0) * corr)
                np.fill_diagonal(corr, 1.0)
            eigvals, eigvecs = np.linalg.eigh(corr)
            eigvals = np.clip(eigvals, 1e-6, None)
            corr = (eigvecs * eigvals) @ eigvecs.T
            diag_std = np.sqrt(np.maximum(np.diag(corr), 1e-12))
            corr = corr / np.outer(diag_std, diag_std)
            corr = 0.5 * (corr + corr.T)
            np.fill_diagonal(corr, 1.0)
            z = rng.multivariate_normal(np.zeros(d), corr, size=self.n_simulations)
            u_draws = np.clip(stats.norm.cdf(z), 1e-9, 1.0 - 1e-9)
        else:
            # Stratified Latin Hypercube style sampling per dimension
            u_draws = np.empty((self.n_simulations, d), dtype=float)
            for j in range(d):
                strata = (np.arange(self.n_simulations) + rng.uniform(0.0, 1.0, size=self.n_simulations)) / self.n_simulations
                rng.shuffle(strata)
                u_draws[:, j] = np.clip(strata, 1e-6, 1.0 - 1e-6)

        samples: dict[str, np.ndarray] = {}
        for j, name in enumerate(names):
            spec = input_specs[name]
            dist = str(spec.get("dist", "normal")).lower()
            u = u_draws[:, j]
            if dist == "normal":
                samples[name] = stats.norm.ppf(u, loc=float(spec.get("mean", 0.0)), scale=max(float(spec.get("std", 1.0)), 1e-12))
            elif dist == "lognormal":
                samples[name] = stats.lognorm.ppf(u, s=max(float(spec.get("sigma", 0.5)), 1e-12), scale=np.exp(float(spec.get("mean", 0.0))))
            elif dist == "uniform":
                low, high = float(spec.get("low", 0.0)), float(spec.get("high", 1.0))
                if high < low:
                    raise SimulationError(f"Uniform input '{name}' requires high >= low.")
                samples[name] = low + u * (high - low)
            elif dist == "triangular":
                left, mode, right = float(spec["left"]), float(spec["mode"]), float(spec["right"])
                if not (left <= mode <= right) or right <= left:
                    raise SimulationError(f"Triangular input '{name}' requires left <= mode <= right and right > left.")
                c = (mode - left) / (right - left)
                samples[name] = stats.triang.ppf(u, c=c, loc=left, scale=right - left)
            elif dist == "poisson":
                samples[name] = stats.poisson.ppf(u, mu=max(float(spec.get("lam", 5.0)), 1e-12))
            elif dist == "beta":
                samples[name] = stats.beta.ppf(u, a=max(float(spec.get("a", 2.0)), 1e-12), b=max(float(spec.get("b", 2.0)), 1e-12))
            else:
                raise SimulationError(f"Unsupported distribution '{dist}' for input '{name}'.")

        outcomes = np.asarray(model_fn(samples), dtype=float).ravel()
        q05 = float(np.quantile(outcomes, 0.05))
        tail = outcomes[outcomes <= q05]
        cvar_95 = float(np.mean(tail)) if len(tail) > 0 else q05

        sens: dict[str, float] = {}
        out_const = float(np.std(outcomes)) <= 1e-12
        for name in names:
            if out_const or float(np.std(samples[name])) <= 1e-12:
                sens[name] = 0.0
            else:
                rho, _ = stats.spearmanr(samples[name], outcomes)
                sens[name] = round(0.0 if np.isnan(rho) else float(rho), 4)
        sens = dict(sorted(sens.items(), key=lambda kv: abs(kv[1]), reverse=True))

        result = MonteCarloResult(
            n_simulations=self.n_simulations,
            mean=round(float(np.mean(outcomes)), 6),
            std=round(float(np.std(outcomes, ddof=1) if len(outcomes) > 1 else 0.0), 6),
            median=round(float(np.median(outcomes)), 6),
            min=round(float(np.min(outcomes)), 6),
            max=round(float(np.max(outcomes)), 6),
            var_95=round(q05, 6),
            cvar_95=round(cvar_95, 6),
            percentiles={
                f"p{p:02d}": round(float(np.quantile(outcomes, p / 100.0)), 6)
                for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
            },
            sensitivity_rank_corr=sens,
        )
        return result, outcomes


class DiscreteEventSimulator:
    """Priority-queue Discrete-Event Simulation (DES) engine for multi-server queuing systems."""

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed

    def simulate_queue(
        self,
        *,
        arrival_rate: float,
        service_rate: float,
        n_servers: int = 2,
        horizon_time: float = 500.0,
    ) -> dict[str, Any]:
        """Simulate a multi-server M/M/c queuing network over ``horizon_time``."""
        if arrival_rate <= 0 or service_rate <= 0 or n_servers < 1:
            raise SimulationError("arrival_rate, service_rate must be > 0 and n_servers >= 1.")
        rng = np.random.default_rng(self.random_seed)

        # Priority queue events: (time, event_type, entity_id)
        # event_type: 0 = departure, 1 = arrival
        events: list[tuple[float, int, int]] = []
        t = float(rng.exponential(1.0 / arrival_rate))
        entity_id = 0
        while t < horizon_time:
            heapq.heappush(events, (t, 1, entity_id))
            entity_id += 1
            t += float(rng.exponential(1.0 / arrival_rate))

        busy_servers = 0
        waiting_queue: list[tuple[int, float]] = []
        wait_times: list[float] = []
        system_times: list[float] = []
        arrival_time_map: dict[int, float] = {}
        last_time = 0.0
        busy_area = 0.0
        queue_area = 0.0
        max_queue_len = 0
        completed = 0

        while events:
            now, etype, eid = heapq.heappop(events)
            # Hard horizon: stop processing events scheduled beyond horizon_time.
            # Entries still in service/waiting at horizon are not counted as completed.
            if now > horizon_time:
                break
            dt = now - last_time
            busy_area += busy_servers * dt
            queue_area += len(waiting_queue) * dt
            last_time = now

            if etype == 1:  # Arrival
                arrival_time_map[eid] = now
                if busy_servers < n_servers:
                    busy_servers += 1
                    wait_times.append(0.0)
                    svc = float(rng.exponential(1.0 / service_rate))
                    heapq.heappush(events, (now + svc, 0, eid))
                else:
                    waiting_queue.append((eid, now))
                    max_queue_len = max(max_queue_len, len(waiting_queue))
            else:  # Departure
                completed += 1
                system_times.append(now - arrival_time_map[eid])
                if waiting_queue:
                    next_eid, arr_t = waiting_queue.pop(0)
                    wait_times.append(now - arr_t)
                    svc = float(rng.exponential(1.0 / service_rate))
                    heapq.heappush(events, (now + svc, 0, next_eid))
                else:
                    busy_servers -= 1

        total_t = max(last_time, 1e-9)
        return {
            "total_arrivals": entity_id,
            "completed_entities": completed,
            "throughput_per_time_unit": round(completed / total_t, 4),
            "avg_wait_time": round(float(np.mean(wait_times)) if wait_times else 0.0, 4),
            "p95_wait_time": round(float(np.quantile(wait_times, 0.95)) if wait_times else 0.0, 4),
            "avg_system_time": round(float(np.mean(system_times)) if system_times else 0.0, 4),
            "server_utilization_pct": round((busy_area / (n_servers * total_t)) * 100.0, 2),
            "avg_queue_length": round(queue_area / total_t, 4),
            "max_queue_length": max_queue_len,
        }


class AgentBasedSimulator:
    """Network Agent-Based Model (ABM) for adoption, contagion, and market diffusion dynamics."""

    def __init__(self, n_agents: int = 200, random_seed: int = 42) -> None:
        self.n_agents = max(10, int(n_agents))
        self.random_seed = random_seed

    def simulate_diffusion(
        self,
        *,
        initial_active_fraction: float = 0.05,
        transmission_prob: float = 0.15,
        recovery_prob: float = 0.05,
        avg_contacts: int = 6,
        steps: int = 50,
    ) -> pd.DataFrame:
        """Simulate SIR / innovation diffusion over a Watts-Strogatz small-world social network."""
        import networkx as nx

        rng = np.random.default_rng(self.random_seed)
        k = min(max(2, avg_contacts), self.n_agents - 1)
        if k % 2 == 1:
            k = max(2, k - 1)
        graph = nx.watts_strogatz_graph(self.n_agents, k, 0.15, seed=self.random_seed)

        # States: 0 = Susceptible, 1 = Active/Infected, 2 = Recovered/Saturated
        states = np.zeros(self.n_agents, dtype=int)
        n_init = max(1, int(self.n_agents * initial_active_fraction))
        states[rng.choice(self.n_agents, size=n_init, replace=False)] = 1

        records: list[dict[str, int]] = []
        for step in range(steps + 1):
            records.append(
                {
                    "step": step,
                    "susceptible": int(np.sum(states == 0)),
                    "active": int(np.sum(states == 1)),
                    "recovered": int(np.sum(states == 2)),
                }
            )
            next_states = states.copy()
            active_nodes = np.where(states == 1)[0]
            for node in active_nodes:
                for nb in graph.neighbors(int(node)):
                    if states[nb] == 0 and rng.uniform() < transmission_prob:
                        next_states[nb] = 1
                if rng.uniform() < recovery_prob:
                    next_states[node] = 2
            states = next_states

        return pd.DataFrame(records)


class DigitalTwinEngine:
    """State-space Digital Twin combining Kalman Filter state assimilation and what-if control simulation."""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray,
    ) -> None:
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.x = np.asarray(x0, dtype=float).ravel()
        self.P = np.eye(len(self.x), dtype=float)

    def assimilate(self, measurement: np.ndarray, control: np.ndarray | None = None) -> np.ndarray:
        """Run one Kalman predict-update step assimilating live telemetry into the twin state."""
        u = np.zeros(self.B.shape[1], dtype=float) if control is None else np.asarray(control, dtype=float).ravel()
        z = np.asarray(measurement, dtype=float).ravel()

        # Predict
        x_pred = self.A @ self.x + self.B @ u
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Update
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.pinv(S)
        self.x = x_pred + K @ y
        # Joseph stabilized covariance update (Bucy & Joseph 1968): guarantees symmetry & PSD
        ikh = np.eye(len(self.x), dtype=float) - K @ self.H
        self.P = ikh @ P_pred @ ikh.T + K @ self.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()

    def simulate_scenario(
        self, control_sequence: np.ndarray, horizon: int | None = None
    ) -> np.ndarray:
        """Project future digital twin state trajectories under a proposed control policy."""
        u_seq = np.asarray(control_sequence, dtype=float)
        if u_seq.ndim == 1:
            u_seq = u_seq.reshape(-1, self.B.shape[1])
        steps = horizon or len(u_seq)
        state = self.x.copy()
        trajectory = np.empty((steps, len(state)), dtype=float)
        for t in range(steps):
            u = u_seq[min(t, len(u_seq) - 1)]
            state = self.A @ state + self.B @ u
            trajectory[t] = state
        return trajectory
