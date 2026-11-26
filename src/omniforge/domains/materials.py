"""Materials Science & Chemistry Domain Plugin: Property Prediction, Formulation Optimization, Bandgap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class MaterialsDomainPlugin(BaseDomainPlugin):
    """Computational Materials Science & Formulation Analytics plugin."""

    metadata = DomainMetadata(
        name="materials",
        display_name="Materials Science & Chemistry",
        description="Alloy tensile strength & bandgap regression, crystal-graph modeling, and Bayesian formulation discovery.",
        compliance_frameworks=["REACH", "ISO-17025"],
        kpis=["mean_tensile_strength_mpa", "specific_strength_ratio"],
        recommended_models=["gradient_boosting", "gnn", "hyperparameter_optimizer"],
        default_target="tensile_strength_mpa",
        config_file="configs/domains/materials.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if "tensile_strength_mpa" in df.columns:
            kpis["mean_tensile_strength_mpa"] = round(float(df["tensile_strength_mpa"].mean()), 2)
        if {"tensile_strength_mpa", "density_g_cm3"}.issubset(df.columns):
            kpis["specific_strength_ratio"] = round(
                float((df["tensile_strength_mpa"] / np.maximum(df["density_g_cm3"], 0.1)).mean()), 2
            )
        return kpis

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        carbon = rng.uniform(0.05, 1.2, n_samples)
        chromium = rng.uniform(0.0, 18.0, n_samples)
        anneal_temp = rng.uniform(400, 1050, n_samples)
        density = rng.uniform(2.7, 8.2, n_samples)
        strength = 280.0 + 340.0 * carbon + 12.0 * chromium - 0.18 * (anneal_temp - 650.0) ** 2 / 100.0 + rng.normal(0, 22, n_samples)
        return pd.DataFrame(
            {
                "carbon_pct": np.round(carbon, 3),
                "chromium_pct": np.round(chromium, 2),
                "anneal_temp_c": np.round(anneal_temp, 1),
                "density_g_cm3": np.round(density, 3),
                "tensile_strength_mpa": np.round(np.maximum(strength, 120.0), 2),
            }
        )


plugin = MaterialsDomainPlugin()
