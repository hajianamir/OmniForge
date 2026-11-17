"""Sports Analytics & Biomechanics Domain Plugin: Expected Goals (xG), Injury Load, Win Probability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class SportsDomainPlugin(BaseDomainPlugin):
    """Enterprise Sports Science & Performance Analytics plugin."""

    metadata = DomainMetadata(
        name="sports",
        display_name="Sports Analytics & Performance",
        description="Win probability modeling, Expected Goals (xG), athlete workload ACWR injury risk, and scouting.",
        compliance_frameworks=["GDPR", "Athlete-Biometrics"],
        kpis=["win_rate", "mean_acwr_load_ratio", "mean_expected_goals"],
        recommended_models=["logistic_regression", "gradient_boosting", "monte_carlo"],
        default_target="match_won",
        config_file="configs/domains/sports.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"acute_load", "chronic_load"}.issubset(out.columns):
            out["acwr_ratio"] = out["acute_load"] / np.maximum(out["chronic_load"], 1.0)
        if {"shots_on_target", "shots_total"}.issubset(out.columns):
            out["shot_accuracy"] = out["shots_on_target"] / np.maximum(out["shots_total"], 1.0)
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if "match_won" in df.columns:
            kpis["win_rate"] = round(float(df["match_won"].mean()), 4)
        if {"acute_load", "chronic_load"}.issubset(df.columns):
            kpis["mean_acwr_load_ratio"] = round(float((df["acute_load"] / np.maximum(df["chronic_load"], 1.0)).mean()), 3)
        if "expected_goals" in df.columns:
            kpis["mean_expected_goals"] = round(float(df["expected_goals"].mean()), 3)
        return kpis

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        shots = rng.integers(4, 24, n_samples)
        on_target = np.clip(np.round(shots * rng.uniform(0.2, 0.65, n_samples)), 0, shots).astype(int)
        possession = np.clip(rng.normal(52, 10, n_samples), 25, 78)
        xg = 0.08 * shots + 0.22 * on_target + rng.uniform(0, 0.5, n_samples)
        acute = rng.uniform(300, 750, n_samples)
        chronic = rng.uniform(350, 650, n_samples)
        logit = -2.0 + 1.15 * xg + 0.03 * (possession - 50)
        won = (rng.uniform(size=n_samples) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
        return pd.DataFrame(
            {
                "shots_total": shots,
                "shots_on_target": on_target,
                "possession_pct": np.round(possession, 1),
                "expected_goals": np.round(xg, 2),
                "acute_load": np.round(acute, 1),
                "chronic_load": np.round(chronic, 1),
                "match_won": won,
            }
        )


plugin = SportsDomainPlugin()
