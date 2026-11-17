"""Retail & E-Commerce Domain Plugin: Demand Forecasting, CLV, Churn, Dynamic Pricing, Recommendations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class RetailDomainPlugin(BaseDomainPlugin):
    """Enterprise Retail & E-Commerce Analytics plugin (GDPR / CCPA)."""

    metadata = DomainMetadata(
        name="retail",
        display_name="Retail & E-Commerce",
        description="Demand forecasting, RFM customer lifetime value, personalized recommendations, and price elasticity.",
        compliance_frameworks=["GDPR", "CCPA"],
        kpis=["average_order_value", "conversion_rate", "gross_margin_pct", "repeat_purchase_rate"],
        recommended_models=["hybrid_recommender", "gradient_boosting", "forecasting_engine"],
        default_target="converted",
        config_file="configs/domains/retail.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"monetary_total", "order_count"}.issubset(out.columns):
            out["avg_order_value"] = out["monetary_total"] / np.maximum(out["order_count"], 1.0)
        if {"unit_price", "unit_cost"}.issubset(out.columns):
            out["unit_margin_pct"] = (out["unit_price"] - out["unit_cost"]) / np.maximum(out["unit_price"], 1e-6)
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if {"monetary_total", "order_count"}.issubset(df.columns):
            kpis["average_order_value"] = round(float(df["monetary_total"].sum() / max(df["order_count"].sum(), 1)), 2)
        if "converted" in df.columns:
            kpis["conversion_rate"] = round(float(df["converted"].mean()), 4)
        return kpis

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        recency = rng.integers(1, 180, n_samples)
        orders = rng.poisson(4, n_samples) + 1
        unit_price = rng.uniform(15, 220, n_samples)
        unit_cost = unit_price * rng.uniform(0.45, 0.75, n_samples)
        monetary = orders * unit_price * rng.uniform(0.8, 1.4, n_samples)
        logit = -1.2 - 0.015 * recency + 0.35 * orders - 0.004 * unit_price
        converted = (rng.uniform(size=n_samples) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
        return pd.DataFrame(
            {
                "recency_days": recency,
                "order_count": orders,
                "unit_price": np.round(unit_price, 2),
                "unit_cost": np.round(unit_cost, 2),
                "monetary_total": np.round(monetary, 2),
                "converted": converted,
            }
        )


plugin = RetailDomainPlugin()
