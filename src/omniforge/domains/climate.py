"""Climate, ESG & Environmental Science Domain Plugin: Emissions, Anomaly Indices, Extreme Weather."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class ClimateDomainPlugin(BaseDomainPlugin):
    """Enterprise Climate Science & ESG Analytics plugin (TCFD / CSRD / GHG-Protocol)."""

    metadata = DomainMetadata(
        name="climate",
        display_name="Climate Science & ESG Analytics",
        description="Temperature anomaly modeling, Scope 1/2/3 GHG accounting, air quality forecasting, and climate risk.",
        compliance_frameworks=["TCFD", "CSRD", "GHG-Protocol"],
        kpis=["mean_temp_anomaly_c", "total_ghg_tco2eq", "extreme_event_rate"],
        recommended_models=["forecasting_engine", "monte_carlo", "gradient_boosting"],
        default_target="temp_anomaly_c",
        config_file="configs/domains/climate.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"scope1_tco2", "scope2_tco2", "scope3_tco2"}.issubset(out.columns):
            out["total_ghg_tco2"] = out["scope1_tco2"] + out["scope2_tco2"] + out["scope3_tco2"]
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if "temp_anomaly_c" in df.columns:
            kpis["mean_temp_anomaly_c"] = round(float(df["temp_anomaly_c"].mean()), 4)
            kpis["extreme_event_rate"] = round(float((df["temp_anomaly_c"] > 1.5).mean()), 4)
        if {"scope1_tco2", "scope2_tco2", "scope3_tco2"}.issubset(df.columns):
            kpis["total_ghg_tco2eq"] = round(float((df["scope1_tco2"] + df["scope2_tco2"] + df["scope3_tco2"]).sum()), 2)
        return kpis

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        co2_ppm = np.linspace(385.0, 428.0, n_samples) + rng.normal(0, 1.8, n_samples)
        s1 = rng.uniform(50, 400, n_samples)
        s2 = rng.uniform(30, 250, n_samples)
        s3 = rng.uniform(200, 1200, n_samples)
        anom = 0.03 * (co2_ppm - 380.0) + rng.normal(0.3, 0.25, n_samples)
        return pd.DataFrame(
            {
                "co2_ppm": np.round(co2_ppm, 2),
                "scope1_tco2": np.round(s1, 2),
                "scope2_tco2": np.round(s2, 2),
                "scope3_tco2": np.round(s3, 2),
                "temp_anomaly_c": np.round(anom, 3),
            }
        )


plugin = ClimateDomainPlugin()
