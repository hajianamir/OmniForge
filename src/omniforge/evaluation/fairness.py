"""Responsible AI Fairness Auditing & Bias Mitigation (Demographic Parity, Equalized Odds, Reweighing)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from omniforge.core.exceptions import EvaluationError


class FairnessReport(BaseModel):
    """Comprehensive algorithmic fairness assessment across protected demographic groups."""

    protected_attribute: str
    passed: bool
    demographic_parity_diff: float
    disparate_impact_ratio: float
    equalized_odds_diff: float
    equal_opportunity_diff: float
    disparate_impact_ci_95: tuple[float, float] = (0.0, 1.0)
    demographic_parity_ci_95: tuple[float, float] = (0.0, 1.0)
    statistical_parity_p_value: float = 1.0
    statistically_significant_disparity: bool = False
    generalized_entropy_index: float = 0.0
    within_group_gei: float = 0.0
    between_group_gei: float = 0.0
    theil_index: float = 0.0
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

        # Statistical inference on selection parity: Katz (1978) log-RR CI, Agresti-Caffo (2000) CI & Fisher/Chi2 test
        di_ci: tuple[float, float] = (round(di_ratio, 6), round(di_ratio, 6))
        dp_ci: tuple[float, float] = (round(dp_diff, 6), round(dp_diff, 6))
        parity_p = 1.0
        if len(groups) >= 2 and len(yp) > 0:
            group_counts = [(grp, int(np.sum(yp[sf == grp] == 1)), int(np.sum(sf == grp))) for grp in groups]
            min_grp = min(group_counts, key=lambda t: (t[1] / max(t[2], 1), t[2]))
            max_grp = max(group_counts, key=lambda t: (t[1] / max(t[2], 1), -t[2]))
            a1, n1 = min_grp[1], min_grp[2]
            a2, n2 = max_grp[1], max_grp[2]
            if n1 > 0 and n2 > 0:
                # Katz et al. (1978, Biometrics) 95% CI with Haldane-Anscombe +0.5 continuity correction on zero cells
                use_cc = a1 in (0, n1) or a2 in (0, n2)
                c1 = a1 + (0.5 if use_cc else 0.0)
                m1 = n1 + (1.0 if use_cc else 0.0)
                c2 = a2 + (0.5 if use_cc else 0.0)
                m2 = n2 + (1.0 if use_cc else 0.0)
                p1_rr = c1 / m1
                p2_rr = c2 / m2
                if p1_rr > 0.0 and p2_rr > 0.0:
                    se_log_rr = float(np.sqrt(max(0.0, (1.0 - p1_rr) / c1 + (1.0 - p2_rr) / c2)))
                    log_rr = float(np.log(p1_rr / p2_rr))
                    z95 = 1.959963984540054
                    di_ci = (
                        round(float(np.exp(np.clip(log_rr - z95 * se_log_rr, -20.0, 20.0))), 6),
                        round(float(np.exp(np.clip(log_rr + z95 * se_log_rr, -20.0, 20.0))), 6),
                    )
                # Agresti & Caffo (2000, American Statistician) +1/+2 95% CI for risk difference (p_max - p_min)
                p1_ac = (a1 + 1.0) / (n1 + 2.0)
                p2_ac = (a2 + 1.0) / (n2 + 2.0)
                diff_ac = p2_ac - p1_ac
                se_ac = float(np.sqrt(p1_ac * (1.0 - p1_ac) / (n1 + 2.0) + p2_ac * (1.0 - p2_ac) / (n2 + 2.0)))
                dp_ci = (
                    round(float(np.clip(diff_ac - 1.959963984540054 * se_ac, -1.0, 1.0)), 6),
                    round(float(np.clip(diff_ac + 1.959963984540054 * se_ac, -1.0, 1.0)), 6),
                )

            # Exact Fisher test for 2x2 table or Pearson chi-square test for Gx2 contingency table
            contingency = np.array([[a, n - a] for _, a, n in group_counts if n > 0], dtype=int)
            if contingency.shape[0] >= 2 and np.all(contingency.sum(axis=0) > 0):
                try:
                    if contingency.shape[0] == 2:
                        _, p_raw = stats.fisher_exact(contingency, alternative="two-sided")
                    else:
                        _, p_raw, _, _ = stats.chi2_contingency(contingency)
                    if np.isfinite(p_raw):
                        parity_p = float(p_raw)
                except Exception:
                    parity_p = 1.0

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
        if di_ratio < self.disparate_impact_min:
            violations.append(
                f"Disparate impact ratio ({di_ratio:.3f}, 95% CI [{di_ci[0]:.3f}, {di_ci[1]:.3f}]) below 4/5ths threshold ({self.disparate_impact_min:.2f})."
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
            disparate_impact_ci_95=di_ci,
            demographic_parity_ci_95=dp_ci,
            statistical_parity_p_value=round(parity_p, 6),
            statistically_significant_disparity=bool(parity_p < 0.05),
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
