"""Agriculture & Precision AgTech Domain Plugin: Crop Yield, NDVI Vegetation, Irrigation Optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class AgricultureDomainPlugin(BaseDomainPlugin):
    """Precision Agriculture & Agronomy Analytics plugin (GAP / FAO-Sustainability)."""

    metadata = DomainMetadata(
        name="agriculture",
        display_name="Agriculture & Precision AgTech",
        description="Crop yield forecasting, NDVI satellite vigor monitoring, soil moisture modeling, and irrigation scheduling.",
        compliance_frameworks=["GLOBALG.A.P.", "ISO-14001"],
        kpis=["mean_yield_tons_per_ha", "water_use_efficiency", "mean_ndvi"],
        recommended_models=["random_forest", "gradient_boosting", "forecasting_engine"],
        default_target="yield_tons_ha",
        config_file="configs/domains/agriculture.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"nir_band", "red_band"}.issubset(out.columns):
            out["ndvi"] = (out["nir_band"] - out["red_band"]) / np.maximum(out["nir_band"] + out["red_band"], 1e-6)
        if {"rainfall_mm", "irrigation_mm"}.issubset(out.columns):
            out["total_water_mm"] = out["rainfall_mm"] + out["irrigation_mm"]
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if "yield_tons_ha" in df.columns:
            kpis["mean_yield_tons_per_ha"] = round(float(df["yield_tons_ha"].mean()), 3)
        if {"yield_tons_ha", "rainfall_mm", "irrigation_mm"}.issubset(df.columns):
            water = (df["rainfall_mm"] + df["irrigation_mm"]).sum()
            kpis["water_use_efficiency"] = round(float(df["yield_tons_ha"].sum() / max(water, 1.0) * 1000.0), 4)
        return kpis

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        rain = rng.uniform(180, 650, n_samples)
        irrig = rng.uniform(50, 300, n_samples)
        nir = rng.uniform(0.45, 0.85, n_samples)
        red = rng.uniform(0.08, 0.25, n_samples)
        nitrogen = rng.uniform(60, 220, n_samples)
        ndvi = (nir - red) / (nir + red)
        yld = 2.0 + 6.5 * ndvi + 0.005 * (rain + irrig) + 0.012 * nitrogen + rng.normal(0, 0.5, n_samples)
        return pd.DataFrame(
            {
                "rainfall_mm": np.round(rain, 1),
                "irrigation_mm": np.round(irrig, 1),
                "nir_band": np.round(nir, 3),
                "red_band": np.round(red, 3),
                "nitrogen_kg_ha": np.round(nitrogen, 1),
                "yield_tons_ha": np.round(np.maximum(yld, 0.5), 2),
            }
        )


plugin = AgricultureDomainPlugin()
