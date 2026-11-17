"""Government & Public Sector Analytics Domain Plugin: Policy Impact, Service Equity, Budget Allocation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class GovernmentDomainPlugin(BaseDomainPlugin):
    """Public Sector, Policy Evaluation & Civic Equity plugin (FedRAMP / EU-AI-Act / FOIA)."""

    metadata = DomainMetadata(
        name="government",
        display_name="Government & Public Policy",
        description="Causal policy impact evaluation, municipal service allocation, tax compliance, and equity auditing.",
        compliance_frameworks=["FedRAMP", "EU-AI-Act", "GDPR"],
        kpis=["service_sla_compliance_rate", "budget_utilization_pct", "mean_response_days"],
        recommended_models=["causal_inference", "fairness_auditor", "linear_programming"],
        default_target="sla_met",
        config_file="configs/domains/government.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"budget_spent", "budget_allocated"}.issubset(out.columns):
            out["budget_utilization_ratio"] = out["budget_spent"] / np.maximum(out["budget_allocated"], 1.0)
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if "sla_met" in df.columns:
            kpis["service_sla_compliance_rate"] = round(float(df["sla_met"].mean()), 4)
        if {"budget_spent", "budget_allocated"}.issubset(df.columns):
            kpis["budget_utilization_pct"] = round(
                float(100.0 * df["budget_spent"].sum() / max(df["budget_allocated"].sum(), 1.0)), 2
            )
        if "response_days" in df.columns:
            kpis["mean_response_days"] = round(float(df["response_days"].mean()), 2)
        return kpis

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        pop = rng.integers(5_000, 250_000, n_samples)
        alloc = rng.uniform(100_000, 2_000_000, n_samples)
        spent = alloc * rng.uniform(0.7, 1.05, n_samples)
        staff = rng.integers(5, 90, n_samples)
        resp_days = np.clip(14.0 - 0.1 * staff + 0.00003 * pop + rng.normal(0, 2.5, n_samples), 1.0, 45.0)
        sla_met = (resp_days <= 10.0).astype(int)
        return pd.DataFrame(
            {
                "district_population": pop,
                "budget_allocated": np.round(alloc, 2),
                "budget_spent": np.round(spent, 2),
                "staff_count": staff,
                "response_days": np.round(resp_days, 1),
                "sla_met": sla_met,
            }
        )


plugin = GovernmentDomainPlugin()
