"""Human Resources & People Analytics Domain Plugin: Attrition, Fairness Pay Equity, Workforce Planning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class HRDomainPlugin(BaseDomainPlugin):
    """Enterprise People Analytics plugin (EEOC / EU-AI-Act High-Risk / GDPR)."""

    metadata = DomainMetadata(
        name="hr",
        display_name="HR & People Analytics",
        description="Employee attrition risk, workforce capacity planning, and mandatory EEOC/EU-AI-Act fairness auditing.",
        compliance_frameworks=["EEOC", "EU-AI-Act", "GDPR"],
        kpis=["attrition_rate", "mean_engagement_score", "avg_tenure_years"],
        recommended_models=["logistic_regression", "random_forest", "fairness_auditor"],
        default_target="attrition",
        config_file="configs/domains/hr.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"salary", "market_median_salary"}.issubset(out.columns):
            out["compa_ratio"] = out["salary"] / np.maximum(out["market_median_salary"], 1.0)
        if {"overtime_hours", "tenure_years"}.issubset(out.columns):
            out["burnout_index"] = out["overtime_hours"] / np.maximum(out["tenure_years"] + 1.0, 1.0)
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if "attrition" in df.columns:
            kpis["attrition_rate"] = round(float(df["attrition"].mean()), 4)
        if "engagement_score" in df.columns:
            kpis["mean_engagement_score"] = round(float(df["engagement_score"].mean()), 2)
        if "tenure_years" in df.columns:
            kpis["avg_tenure_years"] = round(float(df["tenure_years"].mean()), 2)
        return kpis

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        tenure = np.round(rng.uniform(0.3, 14.0, n_samples), 1)
        salary = rng.uniform(55_000, 165_000, n_samples)
        market = salary * rng.uniform(0.9, 1.15, n_samples)
        overtime = rng.uniform(0, 22, n_samples)
        engage = np.clip(rng.normal(7.0, 1.6, n_samples), 1.0, 10.0)
        logit = -1.8 + 0.12 * overtime - 0.45 * engage - 0.1 * tenure
        attrition = (rng.uniform(size=n_samples) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
        return pd.DataFrame(
            {
                "tenure_years": tenure,
                "salary": np.round(salary, 2),
                "market_median_salary": np.round(market, 2),
                "overtime_hours": np.round(overtime, 1),
                "engagement_score": np.round(engage, 2),
                "attrition": attrition,
            }
        )


plugin = HRDomainPlugin()
