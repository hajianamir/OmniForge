"""Responsible AI Fairness Auditing & Bias Mitigation (Demographic Parity, Equalized Odds, Reweighing)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from scipy import stats

from omniforge.core.exceptions import EvaluationError


class FairnessReport(BaseModel):
    """Comprehensive algorithmic fairness assessment across protected demographic groups."""

    protected_attribute: str
    passed: bool
    demographic_parity_diff: float
    disparate_impact_ratio: float
    equalized_odds_diff: float | None
    equal_opportunity_diff: float | None
    disparate_impact_ci_95: tuple[float, float] | None = None
    demographic_parity_ci_95: tuple[float, float] | None = None
    selection_interval_method: str | None = None
    statistical_parity_p_value: float | None = None
    statistical_parity_test: str | None = None
    statistically_significant_disparity: bool = False
    generalized_entropy_index: float = 0.0
    within_group_gei: float = 0.0
    between_group_gei: float = 0.0
    theil_index: float = 0.0
    group_metrics: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    @field_validator(
        "equalized_odds_diff",
        "equal_opportunity_diff",
        mode="before",
    )
    @classmethod
    def _nan_to_none(cls, v: Any) -> Any:
        """Convert NaN floats to None for valid JSON serialization."""
        if isinstance(v, float) and math.isnan(v):
            return None
        return v


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

        if any(values.ndim != 1 for values in (yt, yp, sf)) or len(yt) != len(yp) or len(yt) != len(sf):
            raise EvaluationError("Fairness auditing requires aligned, one-dimensional labels, predictions and groups.")
        groups = sorted(np.unique(sf).tolist())
        group_metrics: dict[str, dict[str, float | None]] = {}
        sel_rates: list[float] = []
        tprs: list[float] = []
        fprs: list[float] = []

        # An empty input has no evaluable comparisons.
        if len(yt) == 0 or len(groups) == 0:
            return FairnessReport(
                protected_attribute=attribute_name,
                passed=False,
                demographic_parity_diff=0.0,
                disparate_impact_ratio=1.0,
                equalized_odds_diff=None,
                equal_opportunity_diff=None,
                violations=["Insufficient data: empty input or no groups to compare"],
                recommendations=[
                    "Provide non-empty predictions with at least two sensitive groups."
                ],
            )

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
                "selected_count": float(g_yp.sum()),
                "positive_count": float(pos_mask.sum()),
                "negative_count": float(neg_mask.sum()),
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

        di_ci: tuple[float, float] | None = None
        dp_ci: tuple[float, float] | None = None
        parity_p: float | None = None
        parity_test: str | None = None
        if len(groups) >= 2:
            # Bonferroni simultaneous Clopper-Pearson intervals cover all population
            # selection rates. Project the joint bounds onto their range and min/max
            # ratio, so coverage does not depend on picking the observed extrema.
            counts = np.array([[int(group_metrics[g]["selected_count"] or 0), int(group_metrics[g]["count"] or 0)] for g in groups])
            selected, total = counts.T
            tail = 0.05 / (2.0 * len(groups))
            lower = np.where(selected > 0, stats.beta.ppf(tail, selected, total - selected + 1), 0.0)
            upper = np.where(selected < total, stats.beta.ppf(1.0 - tail, selected + 1, total - selected), 1.0)
            dp_ci = (
                math.floor(max(0.0, float(lower.max() - upper.min())) * 1e6) / 1e6,
                math.ceil(float(upper.max() - lower.min()) * 1e6) / 1e6,
            )
            di_upper = min(1.0, float(upper.min() / lower.max())) if lower.max() > 0 else 1.0
            di_ci = (
                math.floor(float(lower.min() / upper.max()) * 1e6) / 1e6,
                math.ceil(di_upper * 1e6) / 1e6,
            )

            contingency = np.column_stack([selected, total - selected])
            if np.all(contingency.sum(axis=0) > 0):
                if len(groups) == 2:
                    parity_test = "fisher_exact"
                    parity_p = float(stats.fisher_exact(contingency).pvalue)
                else:
                    parity_test = "pearson_chi_square"
                    result = stats.chi2_contingency(contingency, correction=False)
                    if (result.expected_freq < 5).any():
                        parity_test = "pearson_chi_square_monte_carlo"
                        method = stats.MonteCarloMethod(n_resamples=9999, rng=np.random.default_rng(42))
                        result = stats.chi2_contingency(contingency, correction=False, method=method)
                    parity_p = float(result.pvalue)

        # Speicher et al. (2018, KDD) Generalized Entropy Index (alpha=2) & Theil Index (alpha=1)
        # Benefit vector b_i = y_pred_i - y_true_i + 1 in {0, 1, 2}; E(2) = E_within(2) + E_between(2)
        gei_total = 0.0
        gei_within = 0.0
        gei_between = 0.0
        theil_val = 0.0
        if len(yt) > 0:
            benefit = (yp - yt + 1).astype(float)
            mu = float(np.mean(benefit))
            if mu > 1e-12:
                n_tot = float(len(benefit))
                gei_total = float(np.mean(((benefit / mu) ** 2) - 1.0) / 2.0)
                pos_b = benefit[benefit > 0.0]
                theil_val = float(np.sum((pos_b / mu) * np.log(pos_b / mu)) / n_tot) if len(pos_b) > 0 else 0.0
                for grp in groups:
                    b_g = benefit[sf == grp]
                    if len(b_g) == 0:
                        continue
                    w_g = float(len(b_g)) / n_tot
                    mu_g = float(np.mean(b_g))
                    gei_between += 0.5 * w_g * (((mu_g / mu) ** 2) - 1.0)
                    if mu_g > 1e-12:
                        gei_g = float(np.mean(((b_g / mu_g) ** 2) - 1.0) / 2.0)
                        gei_within += w_g * ((mu_g / mu) ** 2) * gei_g

        violations: list[str] = []
        recommendations: list[str] = []
        if len(groups) < 2:
            violations.append("Group fairness requires at least two observed groups.")
            eopp_diff = eodds_diff = float("nan")

        # Equalized odds requires comparable TPR and FPR across >= 2 groups.
        # If some groups lack positives or negatives (null TPR/FPR), the metric
        # is undefined for those groups and must be flagged as a violation rather
        # than silently reported as 0.0 (which would falsely appear "fair").
        if len(groups) >= 2:
            if len(tprs) < len(groups):
                violations.append(
                    f"Equalized odds undefined: {len(groups) - len(tprs)} group(s) have no positive labels (TPR unavailable)."
                )
                eopp_diff = float("nan")
                eodds_diff = float("nan")
            if len(fprs) < len(groups):
                violations.append(
                    f"Equalized odds undefined: {len(groups) - len(fprs)} group(s) have no negative labels (FPR unavailable)."
                )
                fpr_diff = float("nan")
                eodds_diff = float("nan")

        if di_ratio < self.disparate_impact_min:
            violations.append(
                f"Observed disparate impact ratio ({di_ratio:.3f}) below the configured threshold ({self.disparate_impact_min:.2f})."
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
            if violations:
                recommendations.append(
                    "Review and remediate the listed fairness violations before deployment."
                )
            else:
                recommendations.append(
                    "Observed selection and error rates met the configured thresholds for this sample."
                )

        return FairnessReport(
            protected_attribute=attribute_name,
            passed=len(violations) == 0,
            demographic_parity_diff=round(dp_diff, 6),
            disparate_impact_ratio=round(di_ratio, 6),
            equalized_odds_diff=round(eodds_diff, 6),
            equal_opportunity_diff=round(eopp_diff, 6),
            disparate_impact_ci_95=di_ci,
            demographic_parity_ci_95=dp_ci,
            selection_interval_method="bonferroni_clopper_pearson_independent_rows" if di_ci is not None else None,
            statistical_parity_p_value=round(parity_p, 6) if parity_p is not None else None,
            statistical_parity_test=parity_test,
            statistically_significant_disparity=parity_p is not None and parity_p < 0.05,
            generalized_entropy_index=round(max(0.0, gei_total), 6),
            within_group_gei=round(max(0.0, gei_within), 6),
            between_group_gei=round(max(0.0, gei_between), 6),
            theil_index=round(max(0.0, theil_val), 6),
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
        *,
        y_true: np.ndarray | pd.Series | None = None,
        criterion: str = "demographic_parity",
        positive_label: Any = 1,
    ) -> dict[str, float]:
        """Compute group-specific post-processing thresholds (Demographic Parity, Equal Opportunity, or Equalized Odds).

        Implements quantile-matched selection-rate parity for ``criterion="demographic_parity"``
        and Hardt, Price & Srebro (2016, NeurIPS) group ROC operating-point matching for
        ``criterion="equal_opportunity"`` and ``criterion="equalized_odds"``.
        """
        crit = criterion.lower().strip()
        if crit not in ("demographic_parity", "equal_opportunity", "equalized_odds"):
            raise EvaluationError(
                f"Unsupported threshold criterion '{criterion}'. Choose demographic_parity, equal_opportunity, or equalized_odds."
            )
        p_arr = np.asarray(y_proba, dtype=float)
        p = p_arr[:, 1] if (p_arr.ndim == 2 and p_arr.shape[1] >= 2) else p_arr.ravel()
        s = np.asarray(sensitive_features).astype(str)
        if len(p) == 0:
            return {}
        groups = sorted(np.unique(s).tolist())

        if crit == "demographic_parity":
            target_rate = float(target_selection_rate) if target_selection_rate is not None else float(np.mean(p >= 0.5))
            target_rate = float(np.clip(target_rate, 0.05, 0.95))
            thresholds: dict[str, float] = {}
            for grp in groups:
                gp = p[s == grp]
                thresholds[grp] = round(float(np.quantile(gp, 1.0 - target_rate)), 6) if len(gp) > 0 else 0.5
            return thresholds

        if y_true is None:
            raise EvaluationError(f"y_true is required for criterion='{crit}'.")
        yt = (np.asarray(y_true) == positive_label).astype(int)
        if len(yt) != len(p):
            raise EvaluationError("y_true and y_proba must have identical length.")

        grid = np.linspace(0.02, 0.98, 193)
        pos_all = yt == 1
        neg_all = yt == 0
        target_tpr = (
            float(np.clip(target_selection_rate, 0.05, 0.95))
            if target_selection_rate is not None
            else (float(np.mean(p[pos_all] >= 0.5)) if np.any(pos_all) else 0.5)
        )
        target_fpr = float(np.mean(p[neg_all] >= 0.5)) if np.any(neg_all) else 0.2

        # For Equalized Odds, refine (target_tpr, target_fpr) toward the feasible intersection across groups
        if crit == "equalized_odds":
            best_shared_loss = float("inf")
            for cand_tpr in np.linspace(max(0.1, target_tpr - 0.25), min(0.95, target_tpr + 0.20), 18):
                for cand_fpr in np.linspace(max(0.02, target_fpr - 0.15), min(0.60, target_fpr + 0.20), 14):
                    grp_gaps: list[float] = []
                    grp_errs: list[float] = []
                    for grp in groups:
                        g_mask = s == grp
                        gp = p[g_mask]
                        gy = yt[g_mask]
                        if len(gp) == 0:
                            continue
                        preds_grid = gp[None, :] >= grid[:, None]
                        g_pos = gy == 1
                        g_neg = gy == 0
                        tprs_g = np.mean(preds_grid[:, g_pos], axis=1) if np.any(g_pos) else np.full(len(grid), cand_tpr)
                        fprs_g = np.mean(preds_grid[:, g_neg], axis=1) if np.any(g_neg) else np.full(len(grid), cand_fpr)
                        errs_g = np.mean(preds_grid != gy[None, :], axis=1)
                        gap_vec = np.maximum(np.abs(tprs_g - cand_tpr), np.abs(fprs_g - cand_fpr))
                        best_i = int(np.argmin(gap_vec + 0.05 * errs_g))
                        grp_gaps.append(float(gap_vec[best_i]))
                        grp_errs.append(float(errs_g[best_i]))
                    obj = (max(grp_gaps) if grp_gaps else 0.0) + 0.25 * (float(np.mean(grp_errs)) if grp_errs else 0.0)
                    if obj < best_shared_loss:
                        best_shared_loss = obj
                        target_tpr = float(cand_tpr)
                        target_fpr = float(cand_fpr)

        out_thresholds: dict[str, float] = {}
        for grp in groups:
            g_mask = s == grp
            gp = p[g_mask]
            gy = yt[g_mask]
            if len(gp) == 0:
                out_thresholds[grp] = 0.5
                continue
            cand_t = np.unique(np.concatenate([grid, np.quantile(gp, np.linspace(0.02, 0.98, 49))]))
            preds_mat = gp[None, :] >= cand_t[:, None]
            g_pos = gy == 1
            g_neg = gy == 0
            tprs_g = np.mean(preds_mat[:, g_pos], axis=1) if np.any(g_pos) else np.full(len(cand_t), target_tpr)
            fprs_g = np.mean(preds_mat[:, g_neg], axis=1) if np.any(g_neg) else np.full(len(cand_t), target_fpr)
            errs_g = np.mean(preds_mat != gy[None, :], axis=1)
            if crit == "equal_opportunity":
                loss_vec = np.abs(tprs_g - target_tpr) + 0.05 * errs_g
            else:
                loss_vec = np.maximum(np.abs(tprs_g - target_tpr), np.abs(fprs_g - target_fpr)) + 0.05 * errs_g
            out_thresholds[grp] = round(float(cand_t[int(np.argmin(loss_vec))]), 6)
        return out_thresholds
