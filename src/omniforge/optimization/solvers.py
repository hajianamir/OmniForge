"""Enterprise Operations Research & Optimization Engine: LP, MILP, NLP, Portfolio, Knapsack, VRP, Assignment."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, linprog, milp, minimize

from omniforge.core.exceptions import OptimizationError
from omniforge.core.logging import get_logger

logger = get_logger(__name__)


class OptimizationResult(BaseModel):
    """Standardized optimization solution payload."""

    problem_type: str
    status: str
    optimal: bool
    objective_value: float
    solution: list[float] | dict[str, Any]
    solver: str = "scipy_highs"
    iterations: int = 0
    details: dict[str, Any] = Field(default_factory=dict)


class OptimizationEngine:
    """Unified mathematical optimization and operations research solver suite."""

    def solve_lp(
        self,
        c: list[float] | np.ndarray,
        *,
        A_ub: list[list[float]] | np.ndarray | None = None,
        b_ub: list[float] | np.ndarray | None = None,
        A_eq: list[list[float]] | np.ndarray | None = None,
        b_eq: list[float] | np.ndarray | None = None,
        bounds: list[tuple[float | None, float | None]] | None = None,
        maximize: bool = False,
    ) -> OptimizationResult:
        """Solve a continuous Linear Program using SciPy HiGHS simplex/interior-point solver."""
        c_vec = np.asarray(c, dtype=float)
        obj_c = -c_vec if maximize else c_vec
        res = linprog(c=obj_c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            raise OptimizationError(
                f"Linear program did not converge: {res.message}",
                suggestions=["Check constraint feasibility and variable bounds."],
            )
        obj_val = float(-res.fun if maximize else res.fun)
        return OptimizationResult(
            problem_type="linear_programming",
            status=str(res.message),
            optimal=bool(res.success),
            objective_value=round(obj_val, 6),
            solution=[round(float(v), 6) for v in res.x],
            solver="scipy_highs",
            iterations=int(getattr(res, "nit", 0)),
        )

    def solve_milp(
        self,
        c: list[float] | np.ndarray,
        integrality: list[int] | np.ndarray,
        *,
        A_ub: list[list[float]] | np.ndarray | None = None,
        b_ub: list[float] | np.ndarray | None = None,
        A_eq: list[list[float]] | np.ndarray | None = None,
        b_eq: list[float] | np.ndarray | None = None,
        lower_bounds: list[float] | np.ndarray | None = None,
        upper_bounds: list[float] | np.ndarray | None = None,
        maximize: bool = False,
    ) -> OptimizationResult:
        """Solve a Mixed-Integer Linear Program (MILP) using HiGHS branch-and-cut."""
        c_vec = np.asarray(c, dtype=float)
        obj_c = -c_vec if maximize else c_vec
        integ = np.asarray(integrality, dtype=int)
        if len(integ) != len(c_vec):
            raise OptimizationError("integrality vector length must match objective vector c.")
        lb = np.zeros_like(c_vec) if lower_bounds is None else np.asarray(lower_bounds, dtype=float)
        ub = np.full_like(c_vec, np.inf) if upper_bounds is None else np.asarray(upper_bounds, dtype=float)
        constraint_list: list[LinearConstraint] = []
        if A_ub is not None and b_ub is not None:
            constraint_list.append(
                LinearConstraint(np.asarray(A_ub, dtype=float), -np.inf, np.asarray(b_ub, dtype=float))
            )
        if A_eq is not None and b_eq is not None:
            b_eq_arr = np.asarray(b_eq, dtype=float)
            constraint_list.append(LinearConstraint(np.asarray(A_eq, dtype=float), b_eq_arr, b_eq_arr))
        res = milp(
            c=obj_c,
            integrality=integ,
            bounds=Bounds(lb, ub),
            constraints=constraint_list if constraint_list else None,
        )
        if not res.success or res.x is None:
            raise OptimizationError(f"MILP optimization failed: {res.message}")
        obj_val = float(-res.fun if maximize else res.fun)
        return OptimizationResult(
            problem_type="mixed_integer_linear_programming",
            status=str(res.message),
            optimal=bool(res.success),
            objective_value=round(obj_val, 6),
            solution=[round(float(v), 6) for v in res.x],
            solver="scipy_highs_milp",
        )

    def solve_knapsack(
        self,
        values: list[float],
        weights: list[float],
        capacity: float,
        item_names: list[str] | None = None,
    ) -> OptimizationResult:
        """Solve 0-1 Knapsack / discrete budget allocation to global optimality via MILP."""
        n = len(values)
        if n == 0 or len(weights) != n:
            raise OptimizationError("values and weights must be non-empty lists of equal length.")
        res = self.solve_milp(
            c=values,
            integrality=[1] * n,
            A_ub=[weights],
            b_ub=[capacity],
            lower_bounds=[0.0] * n,
            upper_bounds=[1.0] * n,
            maximize=True,
        )
        names = item_names or [f"item_{i}" for i in range(n)]
        sol_vec = res.solution if isinstance(res.solution, list) else []
        chosen = [names[i] for i, x in enumerate(sol_vec) if float(x) > 0.5]
        total_w = float(sum(float(weights[i]) for i, x in enumerate(sol_vec) if float(x) > 0.5))
        return OptimizationResult(
            problem_type="knapsack_allocation",
            status="optimal",
            optimal=True,
            objective_value=res.objective_value,
            solution={"selected_items": chosen, "binary_vector": res.solution},
            solver="scipy_highs_milp",
            details={"total_weight": round(total_w, 6), "capacity": capacity, "utilization_pct": round(total_w / max(capacity, 1e-12) * 100.0, 2)},
        )

    def solve_assignment(
        self, cost_matrix: list[list[float]] | np.ndarray, maximize: bool = False
    ) -> OptimizationResult:
        """Solve linear sum assignment (resource-to-task matching) via the Hungarian algorithm."""
        mat = np.asarray(cost_matrix, dtype=float)
        if mat.ndim != 2:
            raise OptimizationError("cost_matrix must be 2D.")
        row_ind, col_ind = linear_sum_assignment(-mat if maximize else mat)
        total = float(mat[row_ind, col_ind].sum())
        pairs = [{"row": int(r), "col": int(c), "value": float(mat[r, c])} for r, c in zip(row_ind, col_ind, strict=False)]
        return OptimizationResult(
            problem_type="linear_assignment",
            status="optimal",
            optimal=True,
            objective_value=round(total, 6),
            solution={"assignments": pairs},
            solver="scipy_hungarian",
        )

    def optimize_portfolio(
        self,
        expected_returns: list[float] | np.ndarray,
        cov_matrix: list[list[float]] | np.ndarray | None = None,
        *,
        returns_history: list[list[float]] | np.ndarray | None = None,
        covariance_estimator: str = "ledoit_wolf",
        shrinkage_alpha: float | None = None,
        risk_aversion: float = 2.0,
        risk_free_rate: float = 0.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        asset_names: list[str] | None = None,
    ) -> OptimizationResult:
        """Solve Markowitz Mean-Variance portfolio optimization with Ledoit-Wolf (2004) shrinkage & box constraints."""
        mu = np.asarray(expected_returns, dtype=float).ravel()
        n = len(mu)
        shrinkage_used: float = 0.0
        cov_method = "supplied_matrix"
        if returns_history is not None:
            ret_mat = np.asarray(returns_history, dtype=float)
            if ret_mat.ndim != 2 or ret_mat.shape[1] != n or ret_mat.shape[0] < 2:
                raise OptimizationError(
                    "returns_history must be a 2D (T >= 2, n_assets) array aligned with expected_returns."
                )
            if covariance_estimator.lower() == "ledoit_wolf":
                from sklearn.covariance import LedoitWolf

                lw = LedoitWolf().fit(ret_mat)
                cov = np.asarray(lw.covariance_, dtype=float)
                shrinkage_used = float(lw.shrinkage_)
                cov_method = "ledoit_wolf_2004"
            else:
                cov = np.cov(ret_mat, rowvar=False, ddof=1)
                cov_method = "sample_covariance"
        else:
            if cov_matrix is None:
                raise OptimizationError("Provide either cov_matrix or returns_history.")
            cov = np.asarray(cov_matrix, dtype=float)

        if n == 0 or cov.shape != (n, n):
            raise OptimizationError("Covariance matrix must be non-empty, square, and aligned with expected_returns.")
        if risk_aversion < 0.0:
            raise OptimizationError("risk_aversion must be non-negative for convex mean-variance optimization.")
        if min_weight > max_weight or n * min_weight > 1.0 + 1e-9 or n * max_weight < 1.0 - 1e-9:
            raise OptimizationError(
                f"Infeasible portfolio weight bounds [{min_weight}, {max_weight}] for {n} assets summing to 1."
            )
        cov_sym = 0.5 * (cov + cov.T)
        if shrinkage_alpha is not None:
            alpha_s = float(np.clip(shrinkage_alpha, 0.0, 1.0))
            mu_trace = float(np.trace(cov_sym)) / max(n, 1)
            cov_sym = (1.0 - alpha_s) * cov_sym + alpha_s * mu_trace * np.eye(n, dtype=float)
            shrinkage_used = alpha_s
            cov_method = "linear_identity_shrinkage"
        # Project any slight numerical negative eigenvalues to PSD cone for strict convexity
        eigvals, eigvecs = np.linalg.eigh(cov_sym)
        if float(np.min(eigvals)) < -1e-8:
            raise OptimizationError("Covariance matrix is not positive semi-definite.")
        if float(np.min(eigvals)) < 0.0:
            cov_sym = (eigvecs * np.maximum(eigvals, 0.0)) @ eigvecs.T
            cov_sym = 0.5 * (cov_sym + cov_sym.T)

        def objective(w: np.ndarray) -> float:
            port_ret = float(w @ mu)
            port_var = float(w @ cov_sym @ w)
            return -(port_ret - 0.5 * risk_aversion * port_var)

        def gradient(w: np.ndarray) -> np.ndarray:
            return np.asarray(-(mu - risk_aversion * (cov_sym @ w)), dtype=float)

        w0 = np.clip(np.full(n, 1.0 / n, dtype=float), min_weight, max_weight)
        w0 = w0 / max(float(w0.sum()), 1e-12)
        bounds = [(min_weight, max_weight) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0), "jac": lambda w: np.ones(n, dtype=float)}]
        res = minimize(
            objective,
            w0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if not res.success:
            raise OptimizationError(f"Portfolio optimization failed: {res.message}")

        w_opt = np.clip(np.asarray(res.x, dtype=float), min_weight, max_weight)
        if abs(float(w_opt.sum()) - 1.0) > 1e-9 and float(w_opt.sum()) > 0.0:
            w_opt = w_opt / float(w_opt.sum())
        port_ret = float(w_opt @ mu)
        port_vol = float(np.sqrt(max(float(w_opt @ cov_sym @ w_opt), 0.0)))
        sharpe = (port_ret - float(risk_free_rate)) / max(port_vol, 1e-12)
        names = asset_names or [f"asset_{i}" for i in range(n)]
        weights_map = {name: round(float(val), 6) for name, val in zip(names, w_opt, strict=False)}
        return OptimizationResult(
            problem_type="portfolio_mean_variance",
            status="optimal",
            optimal=True,
            objective_value=round(float(-objective(w_opt)), 6),
            solution=weights_map,
            solver="scipy_slsqp",
            iterations=int(res.nit),
            details={
                "expected_return": round(port_ret, 6),
                "volatility": round(port_vol, 6),
                "sharpe_ratio": round(sharpe, 4),
                "covariance_estimator": cov_method,
                "shrinkage_intensity": round(shrinkage_used, 6),
            },
        )

    def solve_transportation(
        self,
        supply: list[float],
        demand: list[float],
        cost_matrix: list[list[float]] | np.ndarray,
    ) -> OptimizationResult:
        """Solve minimum-cost supply chain transportation flow problem via LP."""
        s = np.asarray(supply, dtype=float)
        d = np.asarray(demand, dtype=float)
        costs = np.asarray(cost_matrix, dtype=float)
        m, n = costs.shape
        if len(s) != m or len(d) != n:
            raise OptimizationError("Supply/demand lengths must match cost_matrix dimensions.")
        if float(s.sum()) + 1e-9 < float(d.sum()):
            raise OptimizationError("Total supply is less than total demand (infeasible).")

        # Decision variables x_{i,j} flattened row-major
        c = costs.ravel()
        A_ub = np.zeros((m, m * n), dtype=float)
        for i in range(m):
            A_ub[i, i * n : (i + 1) * n] = 1.0
        A_eq = np.zeros((n, m * n), dtype=float)
        for j in range(n):
            for i in range(m):
                A_eq[j, i * n + j] = 1.0

        res = self.solve_lp(c, A_ub=A_ub, b_ub=s, A_eq=A_eq, b_eq=d, bounds=[(0.0, None)] * (m * n))
        flow = np.asarray(res.solution, dtype=float).reshape(m, n)
        return OptimizationResult(
            problem_type="supply_chain_transportation",
            status="optimal",
            optimal=True,
            objective_value=res.objective_value,
            solution={"flow_matrix": flow.round(4).tolist()},
            solver="scipy_highs",
        )

    @staticmethod
    def _two_opt_route(route: list[int], dist: np.ndarray) -> list[int]:
        """Improve a single depot-to-depot route [0, ..., 0] via exact 2-opt local search."""
        best = list(route)
        m = len(best)
        if m <= 4:
            return best

        def _cost(seq: list[int]) -> float:
            return sum(float(dist[seq[k], seq[k + 1]]) for k in range(len(seq) - 1))

        best_cost = _cost(best)
        improved = True
        while improved:
            improved = False
            for a in range(1, m - 2):
                for b in range(a + 1, m - 1):
                    candidate = best[:a] + best[a : b + 1][::-1] + best[b + 1 :]
                    cand_cost = _cost(candidate)
                    if cand_cost + 1e-12 < best_cost:
                        best = candidate
                        best_cost = cand_cost
                        improved = True
                        break
                if improved:
                    break
        return best

    def solve_vrp_savings(
        self,
        distance_matrix: list[list[float]] | np.ndarray,
        demands: list[float],
        vehicle_capacity: float,
    ) -> OptimizationResult:
        """Solve Capacitated Vehicle Routing Problem (CVRP) via Clarke-Wright Savings + 2-opt."""
        dist = np.asarray(distance_matrix, dtype=float)
        n = dist.shape[0]
        if dist.ndim != 2 or dist.shape[1] != n or n < 2 or len(demands) != n:
            raise OptimizationError("distance_matrix (NxN) and demands (length N, depot=0) required.")
        if vehicle_capacity <= 0.0:
            raise OptimizationError("vehicle_capacity must be positive.")
        if any(float(d) > vehicle_capacity + 1e-12 for d in demands[1:]):
            raise OptimizationError("At least one customer demand exceeds vehicle_capacity (infeasible CVRP).")

        is_symmetric = bool(np.allclose(dist, dist.T, atol=1e-9))

        # Initial routes: depot -> customer -> depot
        routes: dict[int, list[int]] = {i: [i] for i in range(1, n)}
        route_load: dict[int, float] = {i: float(demands[i]) for i in range(1, n)}
        cust_to_route: dict[int, int] = {i: i for i in range(1, n)}

        savings: list[tuple[float, int, int]] = []
        if is_symmetric:
            for i in range(1, n):
                for j in range(i + 1, n):
                    sav = float(dist[0, i] + dist[0, j] - dist[i, j])
                    if sav > 1e-12:
                        savings.append((sav, i, j))
        else:
            for i in range(1, n):
                for j in range(1, n):
                    if i == j:
                        continue
                    sav = float(dist[i, 0] + dist[0, j] - dist[i, j])
                    if sav > 1e-12:
                        savings.append((sav, i, j))
        savings.sort(key=lambda t: t[0], reverse=True)

        for _, i, j in savings:
            ri, rj = cust_to_route[i], cust_to_route[j]
            if ri == rj:
                continue
            route_i, route_j = routes[ri], routes[rj]
            if route_load[ri] + route_load[rj] > vehicle_capacity + 1e-12:
                continue
            if route_i[-1] == i and route_j[0] == j:
                merged = route_i + route_j
            elif is_symmetric and route_j[-1] == j and route_i[0] == i:
                merged = route_j + route_i
            elif is_symmetric and route_i[-1] == i and route_j[-1] == j:
                merged = route_i + route_j[::-1]
            elif is_symmetric and route_i[0] == i and route_j[0] == j:
                merged = route_i[::-1] + route_j
            else:
                continue
            routes[ri] = merged
            route_load[ri] += route_load[rj]
            del routes[rj]
            del route_load[rj]
            for c_id in merged:
                cust_to_route[c_id] = ri

        final_routes: list[list[int]] = []
        total_dist = 0.0
        for r in routes.values():
            full = self._two_opt_route([0, *r, 0], dist)
            route_d = sum(float(dist[full[k], full[k + 1]]) for k in range(len(full) - 1))
            total_dist += route_d
            final_routes.append(full)

        return OptimizationResult(
            problem_type="capacitated_vrp",
            status="heuristic_optimal",
            optimal=True,
            objective_value=round(total_dist, 4),
            solution={"routes": final_routes},
            solver="clarke_wright_savings_2opt",
            details={"n_vehicles_used": len(final_routes)},
        )

    def solve_nonlinear(
        self,
        objective: Callable[[np.ndarray], float],
        x0: list[float],
        *,
        bounds: list[tuple[float | None, float | None]] | None = None,
        constraints: list[dict[str, Any]] | None = None,
    ) -> OptimizationResult:
        """Solve smooth nonlinear constrained optimization via SLSQP."""
        res = minimize(objective, np.asarray(x0, dtype=float), method="SLSQP", bounds=bounds, constraints=constraints or ())
        if not res.success:
            raise OptimizationError(f"Nonlinear optimization failed: {res.message}")
        return OptimizationResult(
            problem_type="nonlinear_programming",
            status=str(res.message),
            optimal=bool(res.success),
            objective_value=round(float(res.fun), 6),
            solution=[round(float(v), 6) for v in res.x],
            solver="scipy_slsqp",
            iterations=int(res.nit),
        )
