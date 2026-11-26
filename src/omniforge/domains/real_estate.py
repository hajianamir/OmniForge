"""Real Estate & PropTech Domain Plugin: Automated Valuation Models (AVM), Cap Rate, Rent Yield."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class RealEstateDomainPlugin(BaseDomainPlugin):
    """Enterprise Real Estate & Automated Valuation Model (AVM) plugin (Fair Housing Act)."""

    metadata = DomainMetadata(
        name="real_estate",
        display_name="Real Estate & PropTech",
        description="Automated Valuation Models (AVM), cap-rate forecasting, rent yield optimization, and fair-housing audits.",
        compliance_frameworks=["Fair-Housing-Act", "USPTO-AVM"],
        kpis=["median_price_per_sqm", "mean_gross_rent_yield_pct", "median_valuation"],
        recommended_models=["gradient_boosting", "random_forest", "ridge"],
        default_target="sale_price",
        config_file="configs/domains/real_estate.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if {"sale_price", "area_sqm"}.issubset(df.columns):
            kpis["median_price_per_sqm"] = round(float((df["sale_price"] / np.maximum(df["area_sqm"], 1.0)).median()), 2)
            kpis["median_valuation"] = round(float(df["sale_price"].median()), 2)
        if {"annual_rent", "sale_price"}.issubset(df.columns):
            kpis["mean_gross_rent_yield_pct"] = round(
                float((100.0 * df["annual_rent"] / np.maximum(df["sale_price"], 1.0)).mean()), 2
            )
        return kpis

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        area = rng.uniform(40, 260, n_samples)
        rooms = np.clip(np.round(area / 32.0 + rng.normal(0, 0.5, n_samples)), 1, 8).astype(int)
        dist_center = rng.uniform(0.5, 25.0, n_samples)
        year_built = rng.integers(1955, 2025, n_samples)
        price = 55_000 + 3800 * area - 5200 * dist_center + 850 * (year_built - 1950) + rng.normal(0, 28_000, n_samples)
        price = np.maximum(price, 80_000)
        rent = price * rng.uniform(0.035, 0.065, n_samples)
        return pd.DataFrame(
            {
                "area_sqm": np.round(area, 1),
                "rooms": rooms,
                "distance_to_center_km": np.round(dist_center, 2),
                "year_built": year_built,
                "annual_rent": np.round(rent, 2),
                "sale_price": np.round(price, 2),
            }
        )


plugin = RealEstateDomainPlugin()
