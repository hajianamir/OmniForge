"""Energy, Utilities & Smart Grid Domain Plugin: Load Forecasting, Solar/Wind Generation, Carbon Intensity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.data.validation import ValidationRule
from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class EnergyDomainPlugin(BaseDomainPlugin):
    """Enterprise Energy & Utilities Analytics plugin (NERC-CIP / ISO-50001)."""

    metadata = DomainMetadata(
        name="energy",
        display_name="Energy, Utilities & Smart Grid",
        description="Grid load forecasting, renewable generation modeling, battery dispatch optimization, and outage risk.",
        compliance_frameworks=["NERC-CIP", "ISO-50001"],
        kpis=["load_factor", "peak_demand_mw", "renewable_penetration_pct", "grid_carbon_intensity_g_kwh"],
        recommended_models=["forecasting_engine", "gradient_boosting", "linear_programming"],
        default_target="load_mw",
        config_file="configs/domains/energy.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "temperature_c" in out.columns:
            out["cooling_degree_days"] = np.maximum(out["temperature_c"] - 18.0, 0.0)
            out["heating_degree_days"] = np.maximum(18.0 - out["temperature_c"], 0.0)
        if {"solar_mw", "wind_mw", "load_mw"}.issubset(out.columns):
            out["net_load_mw"] = out["load_mw"] - (out["solar_mw"] + out["wind_mw"])
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if "load_mw" in df.columns:
            peak = float(df["load_mw"].max())
            avg = float(df["load_mw"].mean())
            kpis["peak_demand_mw"] = round(peak, 3)
            kpis["load_factor"] = round(avg / max(peak, 1e-9), 4)
        if {"solar_mw", "wind_mw", "load_mw"}.issubset(df.columns):
            ren = float((df["solar_mw"] + df["wind_mw"]).sum())
            tot = float(df["load_mw"].sum())
            kpis["renewable_penetration_pct"] = round(100.0 * ren / max(tot, 1e-9), 2)
        return kpis

    def get_validation_rules(self) -> list[ValidationRule]:
        return [ValidationRule(name="load_non_negative", rule_type="range", column="load_mw", params={"min": 0})]

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        hours = np.arange(n_samples)
        temp = 18.0 + 9.0 * np.sin(2 * np.pi * hours / 24.0) + rng.normal(0, 1.5, n_samples)
        solar = np.maximum(0.0, 45.0 * np.sin(np.pi * (hours % 24) / 24.0) + rng.normal(0, 3.0, n_samples))
        wind = np.maximum(0.0, rng.normal(25.0, 8.0, n_samples))
        load = 120.0 + 3.5 * np.abs(temp - 18.0) + 20.0 * np.sin(2 * np.pi * hours / 24.0) + rng.normal(0, 4.0, n_samples)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=n_samples, freq="h"),
                "temperature_c": np.round(temp, 2),
                "solar_mw": np.round(solar, 2),
                "wind_mw": np.round(wind, 2),
                "load_mw": np.round(np.maximum(load, 10.0), 2),
            }
        )


plugin = EnergyDomainPlugin()
