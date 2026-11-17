"""Finance, Banking & Quant Domain Plugin: Credit Risk, Fraud, VaR, Sharpe, Portfolio Analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.data.validation import ValidationRule
from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class FinanceDomainPlugin(BaseDomainPlugin):
    """Enterprise Finance & Quantitative Analytics plugin (SOX / Basel III / PCI-DSS)."""

    metadata = DomainMetadata(
        name="finance",
        display_name="Finance & Quantitative Analytics",
        description="Credit scoring, AML/fraud detection, algorithmic risk, Value-at-Risk, and portfolio optimization.",
        compliance_frameworks=["SOX", "Basel-III", "PCI-DSS", "GDPR"],
        kpis=["sharpe_ratio", "value_at_risk_95", "expected_shortfall_95", "default_rate", "expected_loss_ratio"],
        recommended_models=["gradient_boosting", "random_forest", "logistic_regression", "markowitz_optimizer"],
        default_target="default_flag",
        config_file="configs/domains/finance.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"debt", "income"}.issubset(out.columns):
            out["debt_to_income_ratio"] = out["debt"] / np.maximum(out["income"], 1.0)
        if {"loan_amount", "collateral_value"}.issubset(out.columns):
            out["loan_to_value_ratio"] = out["loan_amount"] / np.maximum(out["collateral_value"], 1.0)
        if {"credit_used", "credit_limit"}.issubset(out.columns):
            out["credit_utilization"] = out["credit_used"] / np.maximum(out["credit_limit"], 1.0)
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if "daily_return" in df.columns:
            rets = df["daily_return"].dropna().to_numpy(dtype=float)
            if len(rets) > 2:
                std = float(np.std(rets, ddof=1))
                kpis["sharpe_ratio"] = round(float(np.mean(rets) / max(std, 1e-9) * np.sqrt(252)), 4)
                var95 = float(np.quantile(rets, 0.05))
                kpis["value_at_risk_95"] = round(var95, 6)
                tail = rets[rets <= var95]
                kpis["expected_shortfall_95"] = round(float(np.mean(tail)) if len(tail) else var95, 6)
        if "default_flag" in df.columns:
            kpis["default_rate"] = round(float(df["default_flag"].mean()), 4)
        if "loan_amount" in df.columns and predictions is not None:
            pd_prob = np.clip(np.asarray(predictions, dtype=float), 0.0, 1.0)
            lgd = 0.45
            ead = df["loan_amount"].to_numpy(dtype=float)[: len(pd_prob)]
            kpis["expected_loss_total"] = round(float(np.sum(pd_prob * lgd * ead)), 2)
        return kpis

    def get_validation_rules(self) -> list[ValidationRule]:
        return [
            ValidationRule(name="income_positive", rule_type="range", column="income", params={"min": 0}),
            ValidationRule(name="credit_score_bounds", rule_type="range", column="credit_score", params={"min": 300, "max": 850}),
        ]

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        income = rng.lognormal(mean=11.0, sigma=0.45, size=n_samples)
        debt = income * rng.uniform(0.05, 0.75, size=n_samples)
        credit_score = np.clip(rng.normal(690, 75, size=n_samples), 320, 850)
        loan_amount = rng.uniform(5_000, 120_000, size=n_samples)
        collateral = loan_amount * rng.uniform(0.8, 1.8, size=n_samples)
        daily_return = rng.normal(0.0005, 0.015, size=n_samples)
        logit = -3.0 + 3.2 * (debt / np.maximum(income, 1.0)) - 0.006 * (credit_score - 650)
        default_flag = (rng.uniform(size=n_samples) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
        return pd.DataFrame(
            {
                "income": np.round(income, 2),
                "debt": np.round(debt, 2),
                "credit_score": np.round(credit_score, 0),
                "loan_amount": np.round(loan_amount, 2),
                "collateral_value": np.round(collateral, 2),
                "daily_return": np.round(daily_return, 5),
                "default_flag": default_flag,
            }
        )


plugin = FinanceDomainPlugin()
