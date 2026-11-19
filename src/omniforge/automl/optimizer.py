"""Hyperparameter Optimization (Bayesian GP-EI surrogate, Random, Grid, Optuna hook)."""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
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
    score: float
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
    """Supports Bayesian (GP-UCB/EI), Random Search, Grid Search, and native Optuna TPE."""

    def __init__(
        self,
        *,
        strategy: str = "bayesian",  # bayesian, random, grid, optuna
        direction: str = "maximize",
        max_trials: int = 20,
        time_budget_seconds: float = 120.0,
        random_seed: int = 42,
    ) -> None:
        self.strategy = strategy.lower()
        self.direction = direction.lower()
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

    def _encode_config(self, cfg: dict[str, Any], search_space: dict[str, dict[str, Any]]) -> np.ndarray:
        vec: list[float] = []
        for name, spec in search_space.items():
            val = cfg[name]
            stype = spec.get("type", "float")
            if stype == "categorical":
                choices = spec["choices"]
                vec.append(float(choices.index(val)) / max(len(choices) - 1, 1))
            else:
                low, high = float(spec["low"]), float(spec["high"])
                if spec.get("log", False) and low > 0:
                    vec.append((np.log(float(val)) - np.log(low)) / max(np.log(high) - np.log(low), 1e-12))
                else:
                    vec.append((float(val) - low) / max(high - low, 1e-12))
        return np.asarray(vec, dtype=float)

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
            ts = time.perf_counter()
            score = float(objective_fn(cfg))
            trials.append(OptimizationTrial(trial_id=tid, params=cfg, score=round(score, 6), duration_seconds=round(time.perf_counter() - ts, 4)))
        return self._summarize("random", trials)

    def _optimize_bayesian(
        self, search_space: dict[str, dict[str, Any]], objective_fn: Callable[[dict[str, Any]], float]
    ) -> HPOResult:
        """Gaussian Process Upper Confidence Bound (GP-UCB) Bayesian Optimization."""
        rng = np.random.default_rng(self.random_seed)
        n_init = min(5, max(2, self.max_trials // 3))
        trials: list[OptimizationTrial] = []
        X_obs: list[np.ndarray] = []
        y_obs: list[float] = []
        sign = 1.0 if self.direction == "maximize" else -1.0
        t0 = time.perf_counter()

        for tid in range(self.max_trials):
            if time.perf_counter() - t0 > self.time_budget_seconds:
                break
            if tid < n_init or len(X_obs) < 3:
                cfg = self._sample_random_config(search_space, rng)
            else:
                kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                    length_scale=0.5, length_scale_bounds=(1e-2, 1e3), nu=2.5
                )
                # Avoid marginal-likelihood degeneracy when too few observations exist to identify length-scale
                opt_mode: str | None = "fmin_l_bfgs_b" if len(X_obs) >= 6 else None
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-4,
                    optimizer=opt_mode,
                    normalize_y=True,
                    random_state=self.random_seed,
                )
                gp.fit(np.vstack(X_obs), np.asarray(y_obs, dtype=float) * sign)
                cands = [self._sample_random_config(search_space, rng) for _ in range(40)]
                X_cands = np.vstack([self._encode_config(c, search_space) for c in cands])
                mu, std = gp.predict(X_cands, return_std=True)
                ucb = mu + 1.96 * std
                cfg = cands[int(np.argmax(ucb))]

            ts = time.perf_counter()
            score = float(objective_fn(cfg))
            if np.isfinite(score):
                X_obs.append(self._encode_config(cfg, search_space))
                y_obs.append(score)
                trials.append(OptimizationTrial(trial_id=tid, params=cfg, score=round(score, 6), duration_seconds=round(time.perf_counter() - ts, 4)))
        return self._summarize("bayesian_gp_ucb", trials)

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
            ts = time.perf_counter()
            score = float(objective_fn(cfg))
            trials.append(OptimizationTrial(trial_id=tid, params=cfg, score=round(score, 6), duration_seconds=round(time.perf_counter() - ts, 4)))
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
            ts = time.perf_counter()
            val = float(objective_fn(cfg))
            trials.append(OptimizationTrial(trial_id=trial.number, params=cfg, score=round(val, 6), duration_seconds=round(time.perf_counter() - ts, 4)))
            return val

        study = optuna.create_study(direction=self.direction, sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        study.optimize(_obj, n_trials=self.max_trials, timeout=self.time_budget_seconds)
        return self._summarize("optuna_tpe", trials)

    def _summarize(self, strategy: str, trials: list[OptimizationTrial]) -> HPOResult:
        if not trials:
            raise AutoMLError("No HPO trials completed within budget.")
        best = max(trials, key=lambda t: t.score) if self.direction == "maximize" else min(trials, key=lambda t: t.score)
        logger.info("hpo_completed", strategy=strategy, trials=len(trials), best_score=best.score)
        return HPOResult(
            strategy=strategy,
            direction=self.direction,
            best_params=best.params,
            best_score=best.score,
            n_trials=len(trials),
            trials=trials,
        )
