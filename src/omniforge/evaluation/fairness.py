"""Responsible AI Fairness Auditing & Bias Mitigation (Demographic Parity, Equalized Odds, Reweighing)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class FairnessReport(BaseModel):
    """Comprehensive algorithmic fairness assessment across protected demographic groups."""

    protected_attribute: str
    passed: bool
    demographic_parity_diff: float
    disparate_impact_ratio: float
    equalized_odds_diff: float
    equal_opportunity_diff: float
    group_metrics: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class FairnessAuditor:
    """Evaluates and mitigates group fairness disparities across binary classification models."""

    def __init__(
        self,
        *,
        disparate_impact_min: float = 0.80,
        demographic_parity_max: float = 0.10,
        equalized_odds_max: float = 0.10,
    ) -> None:
        self.disparate_impact_min = disparate_impact_min
        self.demographic_parity_max = demographic_parity_max
        self.equalized_odds_max = equalized_odds_max

    def audit(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        sensitive_features: np.ndarray | pd.Series,
        *,
        attribute_name: str = "sensitive_group",
        positive_label: Any = 1,
    ) -> FairnessReport:
        """Compute group-level TPR, FPR, selection rate, disparate impact, and equalized odds."""
        yt = (np.asarray(y_true) == positive_label).astype(int)
        yp = (np.asarray(y_pred) == positive_label).astype(int)
        sf = np.asarray(sensitive_features).astype(str)

        groups = sorted(np.unique(sf).tolist())
        group_metrics: dict[str, dict[str, float | None]] = {}
        sel_rates: list[float] = []
        tprs: list[float] = []
        fprs: list[float] = []

        for grp in groups:
            mask = sf == grp
            g_yt = yt[mask]
            g_yp = yp[mask]
            sel_rate = float(np.mean(g_yp)) if len(g_yp) > 0 else 0.0
            pos_mask = g_yt == 1
            neg_mask = g_yt == 0
            tpr_val: float | None = float(np.mean(g_yp[pos_mask])) if np.any(pos_mask) else None
            fpr_val: float | None = float(np.mean(g_yp[neg_mask])) if np.any(neg_mask) else None
            acc = float(np.mean(g_yp == g_yt)) if len(g_yt) > 0 else 0.0
            prec = float(np.mean(g_yt[g_yp == 1])) if np.any(g_yp == 1) else 0.0

            group_metrics[grp] = {
                "count": float(int(np.sum(mask))),
                "selection_rate": round(sel_rate, 6),
                "tpr": round(tpr_val, 6) if tpr_val is not None else None,
                "fpr": round(fpr_val, 6) if fpr_val is not None else None,
                "accuracy": round(acc, 6),
                "precision": round(prec, 6),
            }
            sel_rates.append(sel_rate)
            if tpr_val is not None:
                tprs.append(tpr_val)
            if fpr_val is not None:
                fprs.append(fpr_val)

        dp_diff = float(max(sel_rates) - min(sel_rates)) if sel_rates else 0.0
        if sel_rates:
            max_sel, min_sel = max(sel_rates), min(sel_rates)
            # When all groups have 0 selection rate, parity is exact (ratio = 1.0, matching Fairlearn)
            di_ratio = 1.0 if max_sel <= 1e-12 else float(min_sel / max_sel)
        else:
            di_ratio = 1.0
        eopp_diff = float(max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0
        fpr_diff = float(max(fprs) - min(fprs)) if len(fprs) >= 2 else 0.0
        eodds_diff = max(eopp_diff, fpr_diff)

        violations: list[str] = []
        recommendations: list[str] = []
        if di_ratio < self.disparate_impact_min:
            violations.append(
                f"Disparate impact ratio ({di_ratio:.3f}) below 4/5ths threshold ({self.disparate_impact_min:.2f})."
            )
            recommendations.append("Apply Kamiran-Calders reweighing or group threshold optimization.")
        if dp_diff > self.demographic_parity_max:
            violations.append(
                f"Demographic parity difference ({dp_diff:.3f}) exceeds limit ({self.demographic_parity_max:.2f})."
            )
        if eodds_diff > self.equalized_odds_max:
            violations.append(
                f"Equalized odds difference ({eodds_diff:.3f}) exceeds limit ({self.equalized_odds_max:.2f})."
            )
            recommendations.append("Inspect group-specific ROC operating points and calibrate thresholds.")
        if not recommendations:
            recommendations.append("Model satisfies all configured demographic and equalized odds gates.")

        return FairnessReport(
            protected_attribute=attribute_name,
            passed=len(violations) == 0,
            demographic_parity_diff=round(dp_diff, 6),
            disparate_impact_ratio=round(di_ratio, 6),
            equalized_odds_diff=round(eodds_diff, 6),
            equal_opportunity_diff=round(eopp_diff, 6),
            group_metrics=group_metrics,
            violations=violations,
            recommendations=recommendations,
        )

    @staticmethod
    def compute_reweighing_weights(
        y_true: np.ndarray | pd.Series, sensitive_features: np.ndarray | pd.Series
    ) -> np.ndarray:
        """Compute Kamiran & Calders (2012) pre-processing sample weights to enforce statistical parity."""
        y = np.asarray(y_true)
        s = np.asarray(sensitive_features).astype(str)
        if len(y) == 0:
            return np.array([], dtype=float)
        n = float(len(y))
        weights = np.ones(len(y), dtype=float)
        for grp in np.unique(s):
            p_s = float(np.sum(s == grp)) / n
            for label in np.unique(y):
                p_y = float(np.sum(y == label)) / n
                mask = (s == grp) & (y == label)
                p_sy = float(np.sum(mask)) / n
                if p_sy > 0:
                    weights[mask] = (p_s * p_y) / p_sy
        return weights

    @staticmethod
    def optimize_group_thresholds(
        y_proba: np.ndarray,
        sensitive_features: np.ndarray | pd.Series,
        target_selection_rate: float | None = None,
    ) -> dict[str, float]:
        """Compute group-specific probability thresholds equalizing selection rates across groups."""
        p_arr = np.asarray(y_proba, dtype=float)
        p = p_arr[:, 1] if (p_arr.ndim == 2 and p_arr.shape[1] >= 2) else p_arr.ravel()
        s = np.asarray(sensitive_features).astype(str)
        if len(p) == 0:
            return {}
        target_rate = float(target_selection_rate) if target_selection_rate is not None else float(np.mean(p >= 0.5))
        target_rate = float(np.clip(target_rate, 0.05, 0.95))
        thresholds: dict[str, float] = {}
        for grp in sorted(np.unique(s).tolist()):
            gp = p[s == grp]
            thresholds[grp] = round(float(np.quantile(gp, 1.0 - target_rate)), 6) if len(gp) > 0 else 0.5
        return thresholds
