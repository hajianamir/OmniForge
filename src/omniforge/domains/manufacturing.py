"""Manufacturing & Industry 4.0 Domain Plugin: Predictive Maintenance, RUL, OEE, Quality Yield."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class ManufacturingDomainPlugin(BaseDomainPlugin):
    """Enterprise Manufacturing & Predictive Maintenance plugin (ISO-9001 / IATF-16949)."""

    metadata = DomainMetadata(
        name="manufacturing",
        display_name="Manufacturing & Industry 4.0",
        description="Predictive maintenance, Remaining Useful Life (RUL), OEE tracking, and defect anomaly detection.",
        compliance_frameworks=["ISO-9001", "IATF-16949"],
        kpis=["oee_score", "failure_rate", "mean_vibration_rms", "yield_pct"],
        recommended_models=["anomaly_detector", "gradient_boosting", "cox_ph", "digital_twin"],
        default_target="machine_failure",
        config_file="configs/domains/manufacturing.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"torque_nm", "rpm"}.issubset(out.columns):
            out["mechanical_power_kw"] = (out["torque_nm"] * out["rpm"]) / 9549.0
        if {"process_temp_c", "ambient_temp_c"}.issubset(out.columns):
            out["thermal_delta_c"] = out["process_temp_c"] - out["ambient_temp_c"]
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if {"availability", "performance", "quality_rate"}.issubset(df.columns):
            kpis["oee_score"] = round(float((df["availability"] * df["performance"] * df["quality_rate"]).mean()), 4)
        if "machine_failure" in df.columns:
            kpis["failure_rate"] = round(float(df["machine_failure"].mean()), 4)
        if "vibration_rms" in df.columns:
            kpis["mean_vibration_rms"] = round(float(df["vibration_rms"].mean()), 4)
        return kpis

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        rpm = rng.normal(1500, 180, n_samples)
        torque = rng.normal(42, 9, n_samples)
        vib = np.abs(rng.normal(2.2, 0.9, n_samples))
        proc_temp = rng.normal(75, 7, n_samples)
        amb_temp = rng.normal(24, 2, n_samples)
        wear = rng.uniform(0, 240, n_samples)
        logit = -5.5 + 0.9 * vib + 0.015 * wear + 0.04 * (proc_temp - amb_temp)
        failure = (rng.uniform(size=n_samples) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
        return pd.DataFrame(
            {
                "rpm": np.round(rpm, 1),
                "torque_nm": np.round(torque, 2),
                "vibration_rms": np.round(vib, 3),
                "process_temp_c": np.round(proc_temp, 2),
                "ambient_temp_c": np.round(amb_temp, 2),
                "tool_wear_min": np.round(wear, 1),
                "availability": np.round(rng.uniform(0.85, 0.99, n_samples), 3),
                "performance": np.round(rng.uniform(0.82, 0.98, n_samples), 3),
                "quality_rate": np.round(rng.uniform(0.92, 0.999, n_samples), 4),
                "machine_failure": failure,
            }
        )


plugin = ManufacturingDomainPlugin()
