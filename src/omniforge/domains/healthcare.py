"""Healthcare & Clinical Analytics Domain Plugin: Readmission Risk, Survival, Clinical Fairness (HIPAA)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.data.validation import ValidationRule
from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class HealthcareDomainPlugin(BaseDomainPlugin):
    """Enterprise Healthcare & Clinical Risk Analytics plugin (HIPAA / FDA-SaMD / GDPR)."""

    metadata = DomainMetadata(
        name="healthcare",
        display_name="Healthcare & Clinical Analytics",
        description="Patient readmission prediction, ICU triage risk, clinical survival analysis, and treatment effects.",
        compliance_frameworks=["HIPAA", "GDPR", "FDA-21CFR11"],
        kpis=["readmission_rate", "mean_length_of_stay_days", "high_risk_cohort_pct"],
        recommended_models=["logistic_regression", "random_forest", "cox_ph", "kaplan_meier"],
        default_target="readmitted_30d",
        config_file="configs/domains/healthcare.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"weight_kg", "height_m"}.issubset(out.columns):
            out["bmi"] = out["weight_kg"] / np.maximum(out["height_m"] ** 2, 0.5)
        if {"systolic_bp", "diastolic_bp"}.issubset(out.columns):
            out["mean_arterial_pressure"] = (out["systolic_bp"] + 2.0 * out["diastolic_bp"]) / 3.0
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if "readmitted_30d" in df.columns:
            kpis["readmission_rate"] = round(float(df["readmitted_30d"].mean()), 4)
        if "length_of_stay_days" in df.columns:
            kpis["mean_length_of_stay_days"] = round(float(df["length_of_stay_days"].mean()), 2)
        if predictions is not None:
            kpis["high_risk_cohort_pct"] = round(float(np.mean(np.asarray(predictions) >= 0.5) * 100.0), 2)
        return kpis

    def get_validation_rules(self) -> list[ValidationRule]:
        return [
            ValidationRule(name="age_range", rule_type="range", column="age", params={"min": 0, "max": 120}),
            ValidationRule(name="systolic_bp_range", rule_type="range", column="systolic_bp", params={"min": 50, "max": 260}),
        ]

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        age = rng.integers(20, 88, size=n_samples)
        prior_admits = rng.poisson(1.2, size=n_samples)
        los = np.clip(rng.gamma(2.5, 2.0, size=n_samples), 1.0, 30.0)
        sbp = rng.normal(128, 18, size=n_samples)
        dbp = rng.normal(78, 11, size=n_samples)
        weight = rng.normal(78, 15, size=n_samples)
        height = rng.normal(1.71, 0.09, size=n_samples)
        logit = -3.4 + 0.03 * age + 0.55 * prior_admits + 0.08 * los
        readmitted = (rng.uniform(size=n_samples) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
        return pd.DataFrame(
            {
                "age": age,
                "prior_admissions": prior_admits,
                "length_of_stay_days": np.round(los, 1),
                "systolic_bp": np.round(sbp, 1),
                "diastolic_bp": np.round(dbp, 1),
                "weight_kg": np.round(weight, 1),
                "height_m": np.round(height, 2),
                "readmitted_30d": readmitted,
            }
        )


plugin = HealthcareDomainPlugin()
