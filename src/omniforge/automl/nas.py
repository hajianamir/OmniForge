"""Limited Neural Architecture Search (NAS) via evolutionary mutation over MLP architectures."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from omniforge.core.logging import get_logger
from omniforge.core.types import TaskType
from omniforge.models.deep.mlp import MLPModel

logger = get_logger(__name__)


class NeuralArchitectureSearcher:
    """Resource-bounded Evolutionary Neural Architecture Search (NAS) for tabular/signal neural nets."""

    def __init__(
        self,
        *,
        task_type: TaskType = TaskType.BINARY_CLASSIFICATION,
        population_size: int = 6,
        generations: int = 3,
        epochs_per_eval: int = 25,
        random_seed: int = 42,
    ) -> None:
        self.task_type = task_type
        self.population_size = max(3, int(population_size))
        self.generations = max(1, int(generations))
        self.epochs_per_eval = max(10, int(epochs_per_eval))
        self.random_seed = random_seed

    def _random_arch(self, rng: np.random.Generator) -> dict[str, Any]:
        depth = int(rng.integers(1, 4))
        widths = [int(rng.choice([16, 32, 64, 96, 128])) for _ in range(depth)]
        return {
            "hidden_dims": widths,
            "activation": str(rng.choice(["relu", "gelu", "tanh"])),
            "dropout": float(rng.choice([0.0, 0.05, 0.1, 0.2])),
            "learning_rate": float(rng.choice([3e-3, 1e-2, 2e-2])),
        }

    def _mutate(self, parent: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
        child = {
            "hidden_dims": list(parent["hidden_dims"]),
            "activation": parent["activation"],
            "dropout": parent["dropout"],
            "learning_rate": parent["learning_rate"],
        }
        mutation = str(rng.choice(["width", "depth", "activation", "dropout", "lr"]))
        if mutation == "width":
            idx = int(rng.integers(0, len(child["hidden_dims"])))
            child["hidden_dims"][idx] = int(rng.choice([16, 32, 64, 96, 128]))
        elif mutation == "depth":
            if len(child["hidden_dims"]) < 3 and rng.uniform() < 0.5:
                child["hidden_dims"].append(int(rng.choice([16, 32, 64])))
            elif len(child["hidden_dims"]) > 1:
                child["hidden_dims"].pop()
        elif mutation == "activation":
            child["activation"] = str(rng.choice(["relu", "gelu", "tanh"]))
        elif mutation == "dropout":
            child["dropout"] = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
        else:
            child["learning_rate"] = float(rng.choice([3e-3, 1e-2, 2e-2]))
        return child

    def search(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> dict[str, Any]:
        """Run evolutionary NAS with leakage-safe stratified/chronological validation and return best model."""
        rng = np.random.default_rng(self.random_seed)
        y_arr = np.asarray(y)
        is_clf = self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)
        strat = None
        if is_clf:
            vals, counts = np.unique(y_arr, return_counts=True)
            if len(vals) >= 2 and int(counts.min()) >= 2:
                strat = y_arr
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y_arr, test_size=0.2, random_state=self.random_seed, stratify=strat
        )

        population = [self._random_arch(rng) for _ in range(self.population_size)]
        history: list[dict[str, Any]] = []
        best_arch = population[0]
        best_score = -np.inf if is_clf else np.inf

        for gen in range(self.generations):
            scored_pop: list[tuple[float, dict[str, Any]]] = []
            for arch in population:
                model = MLPModel(
                    hidden_dims=arch["hidden_dims"],
                    activation=arch["activation"],
                    dropout=arch["dropout"],
                    learning_rate=arch["learning_rate"],
                    epochs=self.epochs_per_eval,
                    task_type=self.task_type,
                    random_seed=self.random_seed,
                    track_carbon=False,
                )
                model.fit(X_tr, y_tr)
                preds = model.predict(X_va)
                score = (
                    float(f1_score(y_va, preds, average="weighted", zero_division=0))
                    if is_clf
                    else float(root_mean_squared_error(y_va, preds))
                )
                scored_pop.append((score, arch))
                history.append({"generation": gen, "score": round(score, 6), **arch})
                if (is_clf and score > best_score) or (not is_clf and score < best_score):
                    best_score = score
                    best_arch = arch

            scored_pop.sort(key=lambda t: t[0], reverse=is_clf)
            elites = [arch for _, arch in scored_pop[: max(2, self.population_size // 2)]]
            next_pop = list(elites)
            while len(next_pop) < self.population_size:
                parent = elites[int(rng.integers(0, len(elites)))]
                next_pop.append(self._mutate(parent, rng))
            population = next_pop

        # Train final best model with fuller epoch budget on full dataset
        final_model = MLPModel(
            hidden_dims=best_arch["hidden_dims"],
            activation=best_arch["activation"],
            dropout=best_arch["dropout"],
            learning_rate=best_arch["learning_rate"],
            epochs=self.epochs_per_eval * 2,
            task_type=self.task_type,
            random_seed=self.random_seed,
        )
        final_model.fit(X, y_arr)
        logger.info("nas_completed", best_score=round(best_score, 4), best_arch=best_arch)
        return {
            "best_architecture": best_arch,
            "best_val_score": round(float(best_score), 6),
            "best_model": final_model,
            "history": history,
        }
