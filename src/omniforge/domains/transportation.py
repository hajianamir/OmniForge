"""Transportation, Logistics & Supply Chain Domain Plugin: ETA, VRP Routing, On-Time Delivery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class TransportationDomainPlugin(BaseDomainPlugin):
    """Enterprise Transportation & Supply Chain Analytics plugin (ISO-28000)."""

    metadata = DomainMetadata(
        name="transportation",
        display_name="Transportation & Supply Chain",
        description="ETA prediction, Vehicle Routing (CVRP), freight cost optimization, and on-time delivery tracking.",
        compliance_frameworks=["ISO-28000", "DOT-Safety"],
        kpis=["on_time_delivery_rate", "avg_transit_hours", "cost_per_km"],
        recommended_models=["cvrp_solver", "transportation_lp", "gradient_boosting"],
        default_target="delayed",
        config_file="configs/domains/transportation.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"distance_km", "planned_hours"}.issubset(out.columns):
            out["planned_speed_kmh"] = out["distance_km"] / np.maximum(out["planned_hours"], 0.1)
        if {"freight_cost", "distance_km"}.issubset(out.columns):
            out["cost_per_km"] = out["freight_cost"] / np.maximum(out["distance_km"], 1.0)
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if "delayed" in df.columns:
            kpis["on_time_delivery_rate"] = round(float(1.0 - df["delayed"].mean()), 4)
        if "actual_hours" in df.columns:
            kpis["avg_transit_hours"] = round(float(df["actual_hours"].mean()), 2)
        if {"freight_cost", "distance_km"}.issubset(df.columns):
            kpis["cost_per_km"] = round(float(df["freight_cost"].sum() / max(df["distance_km"].sum(), 1.0)), 3)
        return kpis

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        dist = rng.uniform(50, 1800, n_samples)
        planned = dist / rng.uniform(55, 75, n_samples)
        weather_idx = rng.uniform(0, 1, n_samples)
        actual = planned * (1.0 + 0.25 * weather_idx + rng.normal(0.02, 0.08, n_samples))
        cost = dist * rng.uniform(1.4, 2.4, n_samples)
        delayed = (actual > planned * 1.1).astype(int)
        return pd.DataFrame(
            {
                "distance_km": np.round(dist, 1),
                "planned_hours": np.round(planned, 2),
                "actual_hours": np.round(np.maximum(actual, 0.5), 2),
                "weather_severity": np.round(weather_idx, 3),
                "freight_cost": np.round(cost, 2),
                "delayed": delayed,
            }
        )


plugin = TransportationDomainPlugin()
