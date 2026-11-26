"""Hyperparameter Optimization (Bayesian GP-EI surrogate, Random, Grid, Optuna hook)."""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from omniforge.core.exceptions import AutoMLError
from omniforge.core.logging import get_logger
from omniforge.core.utils import is_package_available

logger = get_logger(__name__)


class OptimizationTrial(BaseModel):
    """Single HPO trial record."""

    trial_id: int
    params: dict[str, Any]
    score: float | None
    status: Literal["completed", "nonfinite", "failed"] = "completed"
    duration_seconds: float


class HPOResult(BaseModel):
    """Aggregated hyperparameter optimization result."""

    strategy: str
    direction: str
    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    trials: list[OptimizationTrial] = Field(default_factory=list)


class HyperparameterOptimizer:
    """Supports Bayesian (GP-EI / GP-UCB with LHS & one-hot kernels), Random Search, Grid Search, and native Optuna TPE."""

    def __init__(
        self,
        *,
        strategy: str = "bayesian",  # bayesian, random, grid, optuna
        acquisition: str = "ei",  # ei (Jones et al. 1998) or ucb (Srinivas et al. 2010)
        direction: str = "maximize",
        max_trials: int = 20,
        time_budget_seconds: float = 120.0,
        random_seed: int = 42,
    ) -> None:
        self.strategy = strategy.lower()
        self.acquisition = acquisition.lower()
        self.direction = direction.lower()
        if self.direction not in {"maximize", "minimize"}:
            raise AutoMLError("HPO direction must be maximize or minimize.")
        self.max_trials = max(2, int(max_trials))
        self.time_budget_seconds = float(time_budget_seconds)
        self.random_seed = random_seed

    def optimize(
        self,
        search_space: dict[str, dict[str, Any]],
        objective_fn: Callable[[dict[str, Any]], float],
    ) -> HPOResult:
        """Optimize ``objective_fn`` over ``search_space``.

        Each entry in ``search_space`` has ``type`` in ``{"int", "float", "categorical"}``
        with ``low``/``high`` (and optional ``log=True``) or ``choices``.
        """
        if not search_space:
            raise AutoMLError("search_space cannot be empty.")

        if self.strategy == "optuna" and is_package_available("optuna"):
            return self._optimize_optuna(search_space, objective_fn)
        if self.strategy == "grid":
            return self._optimize_grid(search_space, objective_fn)
        if self.strategy == "bayesian":
            return self._optimize_bayesian(search_space, objective_fn)
        return self._optimize_random(search_space, objective_fn)

    def _sample_random_config(
        self, search_space: dict[str, dict[str, Any]], rng: np.random.Generator
    ) -> dict[str, Any]:
        cfg: dict[str, Any] = {}
        for name, spec in search_space.items():
            stype = spec.get("type", "float")
            if stype == "categorical":
                choices = spec["choices"]
                cfg[name] = choices[int(rng.integers(0, len(choices)))]
            elif stype == "int":
                cfg[name] = int(rng.integers(int(spec["low"]), int(spec["high"]) + 1))
            else:
                low, high = float(spec["low"]), float(spec["high"])
                if spec.get("log", False) and low > 0:
                    cfg[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
                else:
                    cfg[name] = float(rng.uniform(low, high))
        return cfg

    def _sample_lhs_configs(
        self, search_space: dict[str, dict[str, Any]], n_samples: int, rng: np.random.Generator
    ) -> list[dict[str, Any]]:
        """Generate space-filling Stratified Latin Hypercube (McKay et al. 1979) candidate configurations."""
        m = max(1, int(n_samples))
        keys = list(search_space.keys())
        n_dims = len(keys)
        u_mat = np.zeros((m, n_dims), dtype=float)
        for j in range(n_dims):
            strata = (np.arange(m, dtype=float) + rng.uniform(0.0, 1.0, size=m)) / float(m)
            rng.shuffle(strata)
            u_mat[:, j] = strata

        configs: list[dict[str, Any]] = []
        for i in range(m):
            cfg: dict[str, Any] = {}
            for j, name in enumerate(keys):
                spec = search_space[name]
                u = float(np.clip(u_mat[i, j], 0.0, 1.0 - 1e-12))
                stype = spec.get("type", "float")
                if stype == "categorical":
                    choices = spec["choices"]
                    cfg[name] = choices[min(int(u * len(choices)), len(choices) - 1)]
                elif stype == "int":
                    lo_i, hi_i = int(spec["low"]), int(spec["high"])
                    cfg[name] = int(lo_i + int(u * (hi_i - lo_i + 1)))
                else:
                    low, high = float(spec["low"]), float(spec["high"])
                    if spec.get("log", False) and low > 0:
                        cfg[name] = float(np.exp(np.log(low) + u * (np.log(high) - np.log(low))))
                    else:
                        cfg[name] = float(low + u * (high - low))
            configs.append(cfg)
        return configs

    def _encode_config(self, cfg: dict[str, Any], search_space: dict[str, dict[str, Any]]) -> np.ndarray:
        """Encode hyperparameters into unit-metric GP space with symmetric 1/sqrt(2) one-hot categorical embeddings."""
        vec: list[float] = []
        inv_sqrt2 = float(1.0 / np.sqrt(2.0))
        for name, spec in search_space.items():
            val = cfg[name]
            stype = spec.get("type", "float")
            if stype == "categorical":
                choices = spec["choices"]
                if len(choices) <= 1:
                    vec.append(0.0)
                else:
                    idx = choices.index(val)
                    for c_i in range(len(choices)):
                        vec.append(inv_sqrt2 if c_i == idx else 0.0)
            else:
                low, high = float(spec["low"]), float(spec["high"])
                if spec.get("log", False) and low > 0:
                    vec.append((np.log(float(val)) - np.log(low)) / max(np.log(high) - np.log(low), 1e-12))
                else:
                    vec.append((float(val) - low) / max(high - low, 1e-12))
        return np.asarray(vec, dtype=float)

    @staticmethod
    def _evaluate_trial(
        trial_id: int, config: dict[str, Any], objective_fn: Callable[[dict[str, Any]], float],
    ) -> OptimizationTrial:
        start = time.perf_counter()
        status: Literal["completed", "nonfinite", "failed"] = "completed"
        score = None
        try:
            value = float(objective_fn(dict(config)))
            if np.isfinite(value):
                score = value
            else:
                status = "nonfinite"
        except Exception as exc:
            status = "failed"
            logger.warning("hpo_trial_failed", trial_id=trial_id, error_type=type(exc).__name__)
        return OptimizationTrial(
            trial_id=trial_id, params=config, score=score, status=status,
            duration_seconds=round(time.perf_counter() - start, 4),
        )

    def _optimize_random(
        self, search_space: dict[str, dict[str, Any]], objective_fn: Callable[[dict[str, Any]], float]
    ) -> HPOResult:
        rng = np.random.default_rng(self.random_seed)
        trials: list[OptimizationTrial] = []
        t0 = time.perf_counter()
        for tid in range(self.max_trials):
            if time.perf_counter() - t0 > self.time_budget_seconds:
                break
            cfg = self._sample_random_config(search_space, rng)
            trials.append(self._evaluate_trial(tid, cfg, objective_fn))
        return self._summarize("random", trials)

    def _optimize_bayesian(
        self, search_space: dict[str, dict[str, Any]], objective_fn: Callable[[dict[str, Any]], float]
    ) -> HPOResult:
        """Gaussian Process Bayesian Optimization with Latin Hypercube warm-start & Analytic EI / GP-UCB."""
        rng = np.random.default_rng(self.random_seed)
        n_init = min(5, max(3, self.max_trials // 3))
        init_configs = self._sample_lhs_configs(search_space, n_init, rng)
        trials: list[OptimizationTrial] = []
        X_obs: list[np.ndarray] = []
        y_obs: list[float] = []
        sign = 1.0 if self.direction == "maximize" else -1.0
        t0 = time.perf_counter()

        for tid in range(self.max_trials):
            if time.perf_counter() - t0 > self.time_budget_seconds:
                break
            if tid < len(init_configs) or len(X_obs) < 3:
                cfg = init_configs[tid] if tid < len(init_configs) else self._sample_random_config(search_space, rng)
            else:
                kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                    length_scale=0.5, length_scale_bounds=(1e-2, 1e3), nu=2.5
                )
                # Avoid marginal-likelihood degeneracy when too few observations exist to identify length-scale
                opt_mode: str | None = "fmin_l_bfgs_b" if len(X_obs) >= 6 else None
                y_signed = np.asarray(y_obs, dtype=float) * sign
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-4,
                    optimizer=opt_mode,
                    normalize_y=True,
                    random_state=self.random_seed,
                )
                X_mat = np.vstack(X_obs)
                gp.fit(X_mat, y_signed)
                cands = self._sample_lhs_configs(search_space, 64, rng)
                X_cands = np.vstack([self._encode_config(c, search_space) for c in cands])
                mu, std = gp.predict(X_cands, return_std=True)
                if self.acquisition == "ucb":
                    # Srinivas et al. (2010, ICML) sublinear-regret GP-UCB exploration schedule
                    d_eff = max(1, X_mat.shape[1])
                    beta_t = 2.0 * np.log(d_eff * ((tid + 1) ** 2) * (np.pi**2) / 0.30)
                    acq = mu + np.sqrt(max(beta_t, 1.0)) * std
                else:
                    # Jones, Schonlau & Welch (1998) analytic Expected Improvement (EI)
                    f_best = float(np.max(y_signed))
                    xi = 0.01 * max(float(np.std(y_signed)), 1e-6)
                    imp = mu - f_best - xi
                    safe_std = np.where(std > 1e-12, std, 1.0)
                    z = np.where(std > 1e-12, imp / safe_std, 0.0)
                    acq = np.where(std > 1e-12, imp * stats.norm.cdf(z) + std * stats.norm.pdf(z), np.maximum(imp, 0.0))
                cfg = cands[int(np.argmax(acq))]

            trial_result = self._evaluate_trial(tid, cfg, objective_fn)
            trials.append(trial_result)
            if trial_result.score is not None:
                X_obs.append(self._encode_config(cfg, search_space))
                y_obs.append(trial_result.score)
        strat_label = "bayesian_gp_ucb" if self.acquisition == "ucb" else "bayesian_gp_ei"
        return self._summarize(strat_label, trials)

    def _optimize_grid(
        self, search_space: dict[str, dict[str, Any]], objective_fn: Callable[[dict[str, Any]], float]
    ) -> HPOResult:
        keys = list(search_space.keys())
        axes: list[list[Any]] = []
        for k in keys:
            spec = search_space[k]
            if spec.get("type") == "categorical":
                axes.append(list(spec["choices"]))
            elif spec.get("type") == "int":
                axes.append(sorted(set(int(v) for v in np.linspace(spec["low"], spec["high"], 3))))
            elif spec.get("log", False) and float(spec["low"]) > 0.0 and float(spec["high"]) > 0.0:
                axes.append([float(v) for v in np.geomspace(float(spec["low"]), float(spec["high"]), 3)])
            else:
                axes.append([float(v) for v in np.linspace(spec["low"], spec["high"], 3)])

        trials: list[OptimizationTrial] = []
        t0 = time.perf_counter()
        for tid, combo in enumerate(itertools.product(*axes)):
            if tid >= self.max_trials or (time.perf_counter() - t0 > self.time_budget_seconds):
                break
            cfg = dict(zip(keys, combo, strict=False))
            trials.append(self._evaluate_trial(tid, cfg, objective_fn))
        return self._summarize("grid", trials)

    def _optimize_optuna(
        self, search_space: dict[str, dict[str, Any]], objective_fn: Callable[[dict[str, Any]], float]
    ) -> HPOResult:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        trials: list[OptimizationTrial] = []

        def _obj(trial: Any) -> float:
            cfg: dict[str, Any] = {}
            for name, spec in search_space.items():
                stype = spec.get("type", "float")
                if stype == "categorical":
                    cfg[name] = trial.suggest_categorical(name, spec["choices"])
                elif stype == "int":
                    cfg[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
                else:
                    cfg[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
            result = self._evaluate_trial(trial.number, cfg, objective_fn)
            trials.append(result)
            # Optuna records NaN objectives as failed and excludes them from TPE.
            return result.score if result.score is not None else float("nan")

        study = optuna.create_study(direction=self.direction, sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(_obj, n_trials=self.max_trials, timeout=self.time_budget_seconds)
        return self._summarize("optuna_tpe", trials)

    def _summarize(self, strategy: str, trials: list[OptimizationTrial]) -> HPOResult:
        if not trials:
            raise AutoMLError("No HPO trials completed within budget.")
        scored = [(trial.score, trial) for trial in trials
                  if trial.status == "completed" and trial.score is not None and np.isfinite(trial.score)]
        if not scored:
            raise AutoMLError("No HPO trial produced a finite objective.",
                              context={"n_trials": len(trials), "trial_statuses": [trial.status for trial in trials]})
        choose = max if self.direction == "maximize" else min
        best_score, best = choose(scored, key=lambda entry: entry[0])
        logger.info("hpo_completed", strategy=strategy, trials=len(trials), best_score=best_score)
        return HPOResult(
            strategy=strategy,
            direction=self.direction,
            best_params=best.params,
            best_score=best_score,
            n_trials=len(trials),
            trials=trials,
        )
