"""Parametric, non-parametric, correlation, variance, autocorrelation, stationarity, and VIF statistical testing engine."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats


class HypothesisTestResult(BaseModel):
    """Standardized output of a statistical hypothesis test with effect size, CI, and multiple-testing support."""

    test_name: str
    statistic: float
    p_value: float
    alpha: float = 0.05
    reject_null: bool
    adjusted_p_value: float | None = None
    effect_size: float | None = None
    effect_size_name: str | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    interpretation: str
    details: dict[str, Any] = Field(default_factory=dict)


class StatisticalAnalyzer:
    """Comprehensive statistical testing toolkit for EDA, validation, and scientific inference."""

    @staticmethod
    def adjust_p_values(
        p_values: list[float],
        method: Literal["none", "bonferroni", "fdr_bh"] = "fdr_bh",
    ) -> list[float]:
        """Adjust a family of raw p-values for multiple hypothesis testing."""
        m = len(p_values)
        if m == 0:
            return []
        arr = np.clip(np.asarray(p_values, dtype=float), 0.0, 1.0)
        if method == "none":
            return [round(float(p), 6) for p in arr]
        if method == "bonferroni":
            return [round(float(min(1.0, p * m)), 6) for p in arr]
        # Benjamini-Hochberg FDR step-up procedure
        order = np.argsort(arr)
        ranked = arr[order]
        adj = np.empty(m, dtype=float)
        running_min = 1.0
        for i in range(m - 1, -1, -1):
            val = ranked[i] * m / float(i + 1)
            running_min = min(running_min, val)
            adj[i] = min(1.0, running_min)
        restored = np.empty(m, dtype=float)
        restored[order] = adj
        return [round(float(p), 6) for p in restored]

    @staticmethod
    def normality_test(
        series: pd.Series,
        alpha: float = 0.05,
        method: Literal["auto", "shapiro_wilk", "dagostino_pearson", "kolmogorov_smirnov", "anderson_darling"] = "auto",
    ) -> HypothesisTestResult:
        """Run Shapiro-Wilk, D'Agostino-Pearson, Kolmogorov-Smirnov, or Anderson-Darling normality test."""
        vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) < 8:
            return HypothesisTestResult(
                test_name="not_evaluable",
                statistic=float("nan"),
                p_value=float("nan"),
                alpha=alpha,
                reject_null=False,
                interpretation="Not evaluable: insufficient observations (<8) for normality testing.",
            )

        std = float(np.std(vals, ddof=1))
        if std <= 1e-12:
            return HypothesisTestResult(
                test_name=method if method != "auto" else "shapiro_wilk",
                statistic=0.0,
                p_value=0.0,
                alpha=alpha,
                reject_null=True,
                interpretation="Constant column (zero variance); strictly non-normal.",
            )

        chosen = method
        if chosen == "auto":
            chosen = "shapiro_wilk" if len(vals) <= 5000 else "dagostino_pearson"

        if chosen == "shapiro_wilk":
            sample = vals if len(vals) <= 5000 else np.random.default_rng(42).choice(vals, size=5000, replace=False)
            stat, p_val = stats.shapiro(sample)
            name = "shapiro_wilk"
        elif chosen == "dagostino_pearson":
            stat, p_val = stats.normaltest(vals)
            name = "dagostino_pearson"
        elif chosen == "kolmogorov_smirnov":
            # Lilliefors (1967) / Dallal & Wilkinson (1986) corrected KS test for normality
            # with sample-estimated mean and standard deviation (never raw kstest(z, "norm")!).
            try:
                from statsmodels.stats.diagnostic import lilliefors

                stat_lf, p_lf = lilliefors(vals, dist="norm", pvalmethod="table")
                stat = float(stat_lf)
                p_val = float(np.clip(p_lf, 0.0, 1.0))
            except Exception:
                z = np.sort((vals - float(np.mean(vals))) / std)
                n_obs = float(len(z))
                cdf_vals = stats.norm.cdf(z)
                d_plus = float(np.max(np.arange(1.0, n_obs + 1.0) / n_obs - cdf_vals))
                d_minus = float(np.max(cdf_vals - np.arange(0.0, n_obs) / n_obs))
                stat = max(d_plus, d_minus)
                # Dallal & Wilkinson (1986) analytic approximation for p <= 0.10,
                # paired with Stephens (1974) / R nortest::lillie.test polynomial for p > 0.10.
                n_eff = n_obs
                d_eff = stat
                if n_eff > 100.0:
                    d_eff = stat * ((n_eff / 100.0) ** 0.49)
                    n_eff = 100.0
                p_dw = float(
                    np.exp(
                        -7.01256 * (d_eff**2) * (n_eff + 2.78019)
                        + 2.99587 * d_eff * np.sqrt(n_eff + 2.78019)
                        - 0.122119
                        + 0.974598 / np.sqrt(n_eff)
                        + 1.67997 / n_eff
                    )
                )
                if p_dw <= 0.10:
                    p_val = float(np.clip(p_dw, 0.0, 1.0))
                else:
                    z_mod = stat * (np.sqrt(n_obs) - 0.01 + 0.85 / np.sqrt(n_obs))
                    if z_mod <= 0.302:
                        p_val = 1.0
                    elif z_mod <= 0.500:
                        p_val = float(
                            2.76773
                            - 19.828315 * z_mod
                            + 80.709644 * (z_mod**2)
                            - 138.55152 * (z_mod**3)
                            + 81.218052 * (z_mod**4)
                        )
                    elif z_mod <= 0.900:
                        p_val = float(
                            -4.901232
                            + 40.662806 * z_mod
                            - 97.490286 * (z_mod**2)
                            + 94.029866 * (z_mod**3)
                            - 32.355711 * (z_mod**4)
                        )
                    else:
                        p_val = float(np.exp(1.2872 - 5.9030 * z_mod - 0.0200 * (z_mod**2)))
                    p_val = float(np.clip(p_val, 0.0, 1.0))
            name = "lilliefors_ks"
        else:
            try:
                ad = stats.anderson(vals, dist="norm", method="interpolate")
                stat = float(ad.statistic)
                p_val = float(np.clip(ad.pvalue, 0.0, 1.0))
                reject = bool(p_val < alpha)
                extra_det: dict[str, Any] = {
                    "skewness": round(float(stats.skew(vals)), 4),
                    "kurtosis": round(float(stats.kurtosis(vals)), 4),
                }
            except TypeError:
                ad = stats.anderson(vals, dist="norm")
                stat = float(ad.statistic)
                sig_levels = np.asarray(ad.significance_level, dtype=float) / 100.0
                crit_vals = np.asarray(ad.critical_values, dtype=float)
                idx = int(np.argmin(np.abs(sig_levels - alpha)))
                crit = float(crit_vals[idx])
                reject = bool(stat > crit)
                a_star = stat * (1.0 + 0.75 / len(vals) + 2.25 / (len(vals) ** 2))
                if a_star >= 0.6:
                    p_val = float(np.exp(1.2937 - 5.709 * a_star + 0.0186 * (a_star**2)))
                elif a_star >= 0.34:
                    p_val = float(np.exp(0.9177 - 4.279 * a_star - 1.38 * (a_star**2)))
                elif a_star >= 0.2:
                    p_val = float(1.0 - np.exp(-8.318 + 42.796 * a_star - 59.938 * (a_star**2)))
                else:
                    p_val = float(1.0 - np.exp(-13.436 + 101.14 * a_star - 223.73 * (a_star**2)))
                p_val = float(np.clip(p_val, 0.0, 1.0))
                extra_det = {
                    "critical_value": round(crit, 4),
                    "skewness": round(float(stats.skew(vals)), 4),
                    "kurtosis": round(float(stats.kurtosis(vals)), 4),
                }
            return HypothesisTestResult(
                test_name="anderson_darling",
                statistic=round(stat, 6),
                p_value=round(p_val, 6),
                alpha=alpha,
                reject_null=reject,
                interpretation=(
                    f"Distribution significantly deviates from normality (A²={stat:.3f}, p={p_val:.4g} < alpha={alpha})."
                    if reject
                    else f"No statistically significant departure from normality (A²={stat:.3f}, p={p_val:.4g} >= alpha={alpha})."
                ),
                details=extra_det,
            )

        reject = bool(p_val < alpha)
        return HypothesisTestResult(
            test_name=name,
            statistic=round(float(stat), 6),
            p_value=round(float(p_val), 6),
            alpha=alpha,
            reject_null=reject,
            interpretation=(
                f"Distribution significantly deviates from normality (p={p_val:.4g} < alpha={alpha})."
                if reject
                else f"No statistically significant departure from normality (p={p_val:.4g} >= alpha={alpha})."
            ),
            details={"skewness": round(float(stats.skew(vals)), 4), "kurtosis": round(float(stats.kurtosis(vals)), 4)},
        )

    @staticmethod
    def one_sample_t_test(
        series: pd.Series,
        popmean: float = 0.0,
        *,
        alpha: float = 0.05,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> HypothesisTestResult:
        """Test whether a numeric sample mean differs significantly from a hypothesized benchmark value."""
        vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) < 3:
            return HypothesisTestResult(
                test_name="one_sample_t_test",
                statistic=0.0,
                p_value=1.0,
                alpha=alpha,
                reject_null=False,
                interpretation="Insufficient observations (<3) for one-sample t-test.",
            )
        stat, p_val = stats.ttest_1samp(vals, popmean=popmean, alternative=alternative)
        mean_val = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1))
        se = sd / np.sqrt(len(vals))
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=len(vals) - 1))
        ci_low = mean_val - t_crit * se
        ci_high = mean_val + t_crit * se
        cohens_d = float((mean_val - popmean) / sd) if sd > 1e-12 else 0.0
        reject = bool(p_val < alpha)
        return HypothesisTestResult(
            test_name="one_sample_t_test",
            statistic=round(float(stat), 6),
            p_value=round(float(p_val), 6),
            alpha=alpha,
            reject_null=reject,
            effect_size=round(cohens_d, 4),
            effect_size_name="cohens_d",
            ci_low=round(ci_low, 4),
            ci_high=round(ci_high, 4),
            interpretation=(
                f"Sample mean ({mean_val:.4g}) differs significantly from {popmean:.4g} (p={p_val:.4g}, d={cohens_d:.2f})."
                if reject
                else f"No significant difference between sample mean ({mean_val:.4g}) and {popmean:.4g} (p={p_val:.4g})."
            ),
            details={"sample_mean": round(mean_val, 6), "hypothesized_mean": popmean, "n": len(vals)},
        )

    @staticmethod
    def levene_variance_test(
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        *,
        alpha: float = 0.05,
        center: Literal["median", "mean", "trimmed"] = "median",
    ) -> HypothesisTestResult:
        """Test homogeneity of variances across groups (Brown-Forsythe/Levene test)."""
        groups = [
            pd.to_numeric(g[value_col], errors="coerce").dropna().to_numpy(dtype=float)
            for _, g in df.groupby(group_col, observed=True)
            if len(pd.to_numeric(g[value_col], errors="coerce").dropna()) >= 2
        ]
        if len(groups) < 2:
            return HypothesisTestResult(
                test_name="levene_variance_test",
                statistic=0.0,
                p_value=1.0,
                alpha=alpha,
                reject_null=False,
                interpretation="Fewer than 2 valid groups for variance homogeneity test.",
            )
        stat, p_val = stats.levene(*groups, center=center)
        reject = bool(p_val < alpha)
        return HypothesisTestResult(
            test_name="levene_variance_test",
            statistic=round(float(stat), 6),
            p_value=round(float(p_val), 6),
            alpha=alpha,
            reject_null=reject,
            interpretation=(
                f"Unequal group variances detected (p={p_val:.4g}); Welch/non-parametric tests recommended."
                if reject
                else f"Group variances appear homogeneous (p={p_val:.4g})."
            ),
            details={"n_groups": len(groups), "center": center},
        )

    @staticmethod
    def two_sample_test(
        group_a: np.ndarray | pd.Series,
        group_b: np.ndarray | pd.Series,
        *,
        parametric: bool = True,
        equal_var: bool = False,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
        alpha: float = 0.05,
    ) -> HypothesisTestResult:
        """Compare two samples via Welch/Student t-test (with Cohen's d & CI) or Mann-Whitney U."""
        a = np.asarray(pd.to_numeric(pd.Series(group_a), errors="coerce").dropna(), dtype=float)
        b = np.asarray(pd.to_numeric(pd.Series(group_b), errors="coerce").dropna(), dtype=float)
        if len(a) < 2 or len(b) < 2:
            return HypothesisTestResult(
                test_name="two_sample_test",
                statistic=0.0,
                p_value=1.0,
                alpha=alpha,
                reject_null=False,
                interpretation="Insufficient observations in one or both groups.",
            )

        n_a, n_b = len(a), len(b)
        mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
        var_a = float(np.var(a, ddof=1))
        var_b = float(np.var(b, ddof=1))
        diff_mean = mean_a - mean_b
        pooled_sd = float(np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)))

        if parametric:
            stat, p_val = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
            name = "student_t_test" if equal_var else "welch_t_test"
            effect_val = float(diff_mean / pooled_sd) if pooled_sd > 1e-12 else 0.0
            es_name = "cohens_d"
            if equal_var:
                se_diff = float(pooled_sd * np.sqrt(1.0 / n_a + 1.0 / n_b))
                df_t = float(max(n_a + n_b - 2, 1))
            else:
                se_a2 = var_a / n_a
                se_b2 = var_b / n_b
                se_diff = float(np.sqrt(se_a2 + se_b2))
                # Exact Welch-Satterthwaite effective degrees of freedom
                ws_denom = (se_a2**2) / max(n_a - 1, 1) + (se_b2**2) / max(n_b - 1, 1)
                df_t = float(((se_a2 + se_b2) ** 2) / ws_denom) if ws_denom > 1e-18 else float(max(n_a + n_b - 2, 1))
            t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=max(df_t, 1.0)))
            ci_low = diff_mean - t_crit * se_diff
            ci_high = diff_mean + t_crit * se_diff
            extra_details: dict[str, Any] = {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "mean_diff": round(diff_mean, 6),
                "df": round(df_t, 4),
                "n_a": n_a,
                "n_b": n_b,
            }
        else:
            stat, p_val = stats.mannwhitneyu(a, b, alternative=alternative)
            name = "mann_whitney_u"
            # Kerby (2014) simple difference rank-biserial correlation:
            # r_rb = 2 * U_1 / (n_a * n_b) - 1  (+1 when a > b, -1 when a < b)
            effect_val = float(np.clip((2.0 * float(stat)) / max(n_a * n_b, 1) - 1.0, -1.0, 1.0))
            es_name = "rank_biserial_r"

            # Distribution-free Hodges-Lehmann location shift & Moses CI on pairwise differences (a_i - b_j)
            max_exact_pairs = 200_000
            if n_a * n_b <= max_exact_pairs:
                pw_diffs = np.sort((a[:, None] - b[None, :]).ravel())
                hl_shift = float(np.median(pw_diffs))
                n_pairs = len(pw_diffs)
                z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
                se_u = float(np.sqrt(n_a * n_b * (n_a + n_b + 1.0) / 12.0))
                # Moses (1965) / Conover (1999) rank indices for Hodges-Lehmann CI
                k_lo = int(np.floor(0.5 * n_pairs - z_crit * se_u))
                k_lo = max(0, min(k_lo, n_pairs - 1))
                k_hi = max(k_lo, min(n_pairs - 1 - k_lo, n_pairs - 1))
                ci_low = float(pw_diffs[k_lo])
                ci_high = float(pw_diffs[k_hi])
            else:
                rng_hl = np.random.default_rng(42)
                n_sub = 400
                sub_a = rng_hl.choice(a, size=min(n_a, n_sub), replace=False)
                sub_b = rng_hl.choice(b, size=min(n_b, n_sub), replace=False)
                pw_diffs = np.sort((sub_a[:, None] - sub_b[None, :]).ravel())
                hl_shift = float(np.median(pw_diffs))
                z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
                # Scale empirical Hodges-Lehmann CI width using full-sample Mann-Whitney variance
                n_sa, n_sb = len(sub_a), len(sub_b)
                se_u_sub = float(np.sqrt(n_sa * n_sb * (n_sa + n_sb + 1.0) / 12.0))
                frac_half = z_crit * se_u_sub / max(n_sa * n_sb, 1) * np.sqrt((n_sa + n_sb) / max(n_a + n_b, 1))
                q_lo = max(0.0, 0.5 - frac_half)
                q_hi = min(1.0, 0.5 + frac_half)
                ci_low = float(np.quantile(pw_diffs, q_lo))
                ci_high = float(np.quantile(pw_diffs, q_hi))

            extra_details = {
                "mean_a": mean_a,
                "mean_b": mean_b,
                "median_a": float(np.median(a)),
                "median_b": float(np.median(b)),
                "hodges_lehmann_shift": round(hl_shift, 6),
                "n_a": n_a,
                "n_b": n_b,
            }

        reject = bool(p_val < alpha)
        return HypothesisTestResult(
            test_name=name,
            statistic=round(float(stat), 6),
            p_value=round(float(p_val), 6),
            alpha=alpha,
            reject_null=reject,
            effect_size=round(effect_val, 4),
            effect_size_name=es_name,
            ci_low=round(ci_low, 4),
            ci_high=round(ci_high, 4),
            interpretation=(
                f"Statistically significant difference between groups (p={p_val:.4g}, {es_name}={effect_val:.2f})."
                if reject
                else f"No statistically significant difference between groups (p={p_val:.4g})."
            ),
            details=extra_details,
        )

    @staticmethod
    def multi_group_test(
        df: pd.DataFrame,
        value_col: str,
        group_col: str,
        *,
        parametric: bool = True,
        alpha: float = 0.05,
    ) -> HypothesisTestResult:
        """Compare >=2 groups using one-way ANOVA (eta-squared) or Kruskal-Wallis H-test (epsilon-squared)."""
        groups = [
            pd.to_numeric(g[value_col], errors="coerce").dropna().to_numpy(dtype=float)
            for _, g in df.groupby(group_col, observed=True)
            if len(pd.to_numeric(g[value_col], errors="coerce").dropna()) >= 2
        ]
        if len(groups) < 2:
            return HypothesisTestResult(
                test_name="anova",
                statistic=0.0,
                p_value=1.0,
                alpha=alpha,
                reject_null=False,
                interpretation="Fewer than 2 valid groups available.",
            )
        all_vals = np.concatenate(groups)
        n_total = len(all_vals)
        k_groups = len(groups)

        if parametric:
            stat, p_val = stats.f_oneway(*groups)
            name = "one_way_anova"
            grand_mean = float(np.mean(all_vals))
            ss_between = float(sum(len(g) * (float(np.mean(g)) - grand_mean) ** 2 for g in groups))
            ss_total = float(np.sum((all_vals - grand_mean) ** 2))
            eff = ss_between / ss_total if ss_total > 1e-12 else 0.0
            es_name = "eta_squared"
            extra_multi: dict[str, Any] = {"n_groups": k_groups, "n_total": n_total}
        else:
            stat, p_val = stats.kruskal(*groups)
            name = "kruskal_wallis"
            # Tomczak & Tomczak (2014):
            # Rank eta-squared:   eta^2_H = (H - k + 1) / (N - k)
            # Rank epsilon-squared: eps^2 = H / ((N^2 - 1)/(N + 1)) = H / (N - 1)
            eta2_h = float(np.clip((float(stat) - k_groups + 1.0) / max(n_total - k_groups, 1), 0.0, 1.0))
            eps2 = float(np.clip(float(stat) / max(n_total - 1, 1), 0.0, 1.0))
            eff = eta2_h
            es_name = "eta_squared_h"
            extra_multi = {
                "n_groups": k_groups,
                "n_total": n_total,
                "eta_squared_h": round(eta2_h, 4),
                "epsilon_squared": round(eps2, 4),
            }

        reject = bool(p_val < alpha)
        return HypothesisTestResult(
            test_name=name,
            statistic=round(float(stat), 6),
            p_value=round(float(p_val), 6),
            alpha=alpha,
            reject_null=reject,
            effect_size=round(eff, 4),
            effect_size_name=es_name,
            interpretation=(
                f"Significant variation across {k_groups} groups of '{group_col}' (p={p_val:.4g}, {es_name}={eff:.3f})."
                if reject
                else f"No significant variation across groups of '{group_col}' (p={p_val:.4g})."
            ),
            details=extra_multi,
        )

    @staticmethod
    def correlation_test(
        series_a: pd.Series,
        series_b: pd.Series,
        *,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        alpha: float = 0.05,
    ) -> HypothesisTestResult:
        """Compute Pearson r, Spearman rho, or Kendall tau correlation with exact p-value and Fisher-z CI."""
        pair = pd.DataFrame(
            {"a": pd.to_numeric(series_a, errors="coerce"), "b": pd.to_numeric(series_b, errors="coerce")}
        ).dropna()
        n = len(pair)
        if n < 4 or float(pair["a"].std()) <= 1e-12 or float(pair["b"].std()) <= 1e-12:
            return HypothesisTestResult(
                test_name=f"{method}_correlation",
                statistic=0.0,
                p_value=1.0,
                alpha=alpha,
                reject_null=False,
                effect_size=0.0,
                effect_size_name=method,
                interpretation="Insufficient non-constant paired observations for correlation test.",
            )

        a_arr = pair["a"].to_numpy(dtype=float)
        b_arr = pair["b"].to_numpy(dtype=float)
        if method == "pearson":
            coef, p_val = stats.pearsonr(a_arr, b_arr)
        elif method == "spearman":
            coef, p_val = stats.spearmanr(a_arr, b_arr)
        else:
            coef, p_val = stats.kendalltau(a_arr, b_arr)

        coef_f = float(np.nan_to_num(coef, nan=0.0))
        p_f = float(np.nan_to_num(p_val, nan=1.0))

        # Fisher-z confidence interval with method-specific Bonett & Wright (2000) standard errors:
        #   Pearson r:  SE(z) = 1 / sqrt(n - 3)
        #   Spearman r: SE(z) = sqrt((1 + r^2 / 2) / (n - 3))
        #   Kendall t:  SE(z) = sqrt(0.437 / (n - 4))
        ci_low, ci_high = None, None
        min_n_ci = 5 if method == "kendall" else 4
        if n >= min_n_ci and abs(coef_f) < 0.9999:
            z_val = float(np.arctanh(coef_f))
            if method == "pearson":
                se_z = float(1.0 / np.sqrt(n - 3.0))
            elif method == "spearman":
                se_z = float(np.sqrt((1.0 + 0.5 * (coef_f**2)) / (n - 3.0)))
            else:
                se_z = float(np.sqrt(0.437 / (n - 4.0)))
            z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
            ci_low = round(float(np.tanh(z_val - z_crit * se_z)), 4)
            ci_high = round(float(np.tanh(z_val + z_crit * se_z)), 4)

        reject = bool(p_f < alpha)
        return HypothesisTestResult(
            test_name=f"{method}_correlation",
            statistic=round(coef_f, 6),
            p_value=round(p_f, 6),
            alpha=alpha,
            reject_null=reject,
            effect_size=round(abs(coef_f), 4),
            effect_size_name=f"abs_{method}",
            ci_low=ci_low,
            ci_high=ci_high,
            interpretation=(
                f"Significant {method} association (r={coef_f:+.3f}, p={p_f:.4g})."
                if reject
                else f"No statistically significant {method} association (r={coef_f:+.3f}, p={p_f:.4g})."
            ),
            details={"n_pairs": n, "signed_correlation": round(coef_f, 6)},
        )

    @staticmethod
    def chi_square_independence(
        series_a: pd.Series, series_b: pd.Series, alpha: float = 0.05
    ) -> HypothesisTestResult:
        """Test categorical association via Pearson Chi-Square (or Fisher's Exact for sparse 2x2) and Cramér's V."""
        ct = pd.crosstab(series_a, series_b)
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return HypothesisTestResult(
                test_name="chi_square_independence",
                statistic=0.0,
                p_value=1.0,
                alpha=alpha,
                reject_null=False,
                effect_size=0.0,
                effect_size_name="cramers_v",
                interpretation="Contingency table has fewer than 2 rows or columns.",
            )
        chi2, p_val, dof, expected = stats.chi2_contingency(ct)
        n = float(ct.to_numpy().sum())
        min_dim = max(min(ct.shape) - 1, 1)
        cramers_v = float(np.sqrt(chi2 / max(n * min_dim, 1e-12)))

        test_name = "chi_square_independence"
        stat_out = float(chi2)
        if ct.shape == (2, 2) and float(np.min(expected)) < 5.0:
            odds_ratio, fisher_p = stats.fisher_exact(ct.to_numpy())
            test_name = "fisher_exact_test"
            stat_out = float(odds_ratio) if np.isfinite(odds_ratio) else float(chi2)
            p_val = float(fisher_p)

        reject = bool(p_val < alpha)
        return HypothesisTestResult(
            test_name=test_name,
            statistic=round(stat_out, 6),
            p_value=round(float(p_val), 6),
            alpha=alpha,
            reject_null=reject,
            effect_size=round(cramers_v, 4),
            effect_size_name="cramers_v",
            interpretation=(
                f"Significant categorical association (p={p_val:.4g}, Cramér's V={cramers_v:.3f})."
                if reject
                else f"No significant categorical association (p={p_val:.4g})."
            ),
            details={"dof": int(dof), "shape": list(ct.shape)},
        )

    @staticmethod
    def compute_vif(df: pd.DataFrame) -> dict[str, float]:
        """Compute exact Variance Inflation Factor (VIF_j = 1 / (1 - R_j^2)) via OLS regression.

        Uses SVD-backed ``np.linalg.lstsq`` on centered predictors rather than ``pinv(R)``,
        because Moore-Penrose pseudoinverse under-reports singular / perfectly collinear
        correlation matrices (e.g. returning 0.25 -> 1.0 instead of ``inf`` for exact duplicates).
        """
        num_df = df.select_dtypes(include=[np.number]).dropna()
        cols = list(num_df.columns)
        if len(cols) < 2 or len(num_df) <= len(cols):
            return {str(c): float("nan") for c in cols}

        X = num_df.to_numpy(dtype=float)
        stds = X.std(axis=0)
        valid_mask = stds > 1e-12
        valid_cols = [cols[i] for i in range(len(cols)) if valid_mask[i]]
        if len(valid_cols) < 2:
            return {str(c): float("nan") for c in cols}

        # Standardize valid predictors (zero mean, unit std -> no intercept needed after centering)
        Z = (X[:, valid_mask] - X[:, valid_mask].mean(axis=0)) / stds[valid_mask]
        n_rows, p = Z.shape
        out: dict[str, float] = {}
        for j, col in enumerate(valid_cols):
            y_j = Z[:, j]
            X_other = np.delete(Z, j, axis=1)
            # Add explicit intercept column for full numerical accuracy
            design = np.column_stack([np.ones(n_rows, dtype=float), X_other])
            coef, *_ = np.linalg.lstsq(design, y_j, rcond=None)
            resid = y_j - design @ coef
            ss_res = float(np.dot(resid, resid))
            ss_tot = float(np.dot(y_j, y_j))
            r2_j = 1.0 - (ss_res / max(ss_tot, 1e-15))
            one_minus_r2 = 1.0 - r2_j
            if one_minus_r2 <= 1e-12:
                out[str(col)] = float("inf")
            else:
                out[str(col)] = round(float(max(1.0, 1.0 / one_minus_r2)), 4)
        return out

    @staticmethod
    def ljung_box_autocorrelation_test(series: pd.Series, lags: int = 10, alpha: float = 0.05) -> HypothesisTestResult:
        """Run Ljung-Box Q-test for serial autocorrelation / departure from white noise."""
        vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        n = len(vals)
        eff_lags = min(lags, max(1, n // 5))
        if n < 12 or float(np.std(vals)) <= 1e-12:
            return HypothesisTestResult(
                test_name="ljung_box_autocorrelation",
                statistic=0.0,
                p_value=1.0,
                alpha=alpha,
                reject_null=False,
                interpretation="Insufficient non-constant observations (<12) for Ljung-Box test.",
            )
        centered = vals - np.mean(vals)
        denom = float(np.dot(centered, centered))
        q_stat = 0.0
        for k in range(1, eff_lags + 1):
            rk = float(np.dot(centered[:-k], centered[k:]) / denom)
            q_stat += (rk**2) / float(n - k)
        q_stat *= float(n * (n + 2))
        p_val = float(1.0 - stats.chi2.cdf(q_stat, df=eff_lags))
        reject = bool(p_val < alpha)
        return HypothesisTestResult(
            test_name="ljung_box_autocorrelation",
            statistic=round(q_stat, 6),
            p_value=round(p_val, 6),
            alpha=alpha,
            reject_null=reject,
            interpretation=(
                f"Significant serial autocorrelation detected up to lag {eff_lags} (Q={q_stat:.2f}, p={p_val:.4g})."
                if reject
                else f"No significant serial autocorrelation detected (white-noise consistent, p={p_val:.4g})."
            ),
            details={"lags": eff_lags, "n": n},
        )

    @staticmethod
    def stationarity_test(series: pd.Series, alpha: float = 0.05) -> HypothesisTestResult:
        """Run Augmented Dickey-Fuller test (via statsmodels or AR(1) t-stat fallback)."""
        vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        if len(vals) < 12 or float(np.std(vals)) <= 1e-12:
            return HypothesisTestResult(
                test_name="adf_stationarity",
                statistic=0.0,
                p_value=1.0,
                alpha=alpha,
                reject_null=False,
                interpretation="Insufficient non-constant observations (<12) for ADF stationarity test.",
            )
        try:
            from statsmodels.tsa.stattools import adfuller

            try:
                adf_res = adfuller(vals, autolag="AIC", result_object=True)
                adf_stat = float(adf_res.stat)
                p_val = float(adf_res.pvalue)
                used_lag = int(adf_res.usedlag)
                nobs = int(adf_res.nobs)
                crit_vals = dict(adf_res.critical_values)
            except TypeError:
                adf_stat, p_val, used_lag, nobs, crit_vals, _ = adfuller(vals, autolag="AIC")
            reject = bool(p_val < alpha)
            return HypothesisTestResult(
                test_name="augmented_dickey_fuller",
                statistic=round(float(adf_stat), 6),
                p_value=round(float(p_val), 6),
                alpha=alpha,
                reject_null=reject,
                interpretation=(
                    f"Series appears stationary (ADF={adf_stat:.3f}, p={p_val:.4g})."
                    if reject
                    else f"Series is non-stationary; differencing recommended (ADF={adf_stat:.3f}, p={p_val:.4g})."
                ),
                details={"used_lag": int(used_lag), "nobs": int(nobs), "critical_values": crit_vals},
            )
        except Exception:
            return StatisticalAnalyzer._adf_mackinnon_fallback(vals, alpha=alpha)

    @staticmethod
    def _adf_mackinnon_fallback(vals: np.ndarray, *, alpha: float = 0.05) -> HypothesisTestResult:
        """Pure-NumPy Augmented Dickey-Fuller test using MacKinnon (1994, 2010) tau_c distribution.

        Never uses Student-t critical values (which severely over-reject unit roots).
        Selects augmentation lag order via AIC up to Schwert's rule and evaluates p-values
        against MacKinnon (1994, 2010) finite-sample response-surface critical values.
        """
        diff = np.diff(vals)
        n_diff = len(diff)
        # Schwert (1989) upper lag bound capped for small series
        max_lag = min(int(np.floor(12.0 * ((len(vals) / 100.0) ** 0.25))), max(0, (n_diff - 4) // 3))

        # Select optimal lag on a common sample [max_lag:] so AIC values are strictly comparable
        best_aic = float("inf")
        best_lag = 0
        y_common = diff[max_lag:]
        n_common = len(y_common)
        for p_lag in range(0, max_lag + 1):
            if n_common <= p_lag + 3:
                continue
            cols_c: list[np.ndarray] = [
                np.ones(n_common, dtype=float),
                vals[max_lag : len(vals) - 1],
            ]
            for j in range(1, p_lag + 1):
                cols_c.append(diff[max_lag - j : n_diff - j])
            X_c = np.column_stack(cols_c)
            beta_c, *_ = np.linalg.lstsq(X_c, y_common, rcond=None)
            resid_c = y_common - X_c @ beta_c
            sigma2_c = max(float(np.dot(resid_c, resid_c)) / float(n_common), 1e-18)
            aic_c = float(n_common * np.log(sigma2_c) + 2.0 * X_c.shape[1])
            if aic_c < best_aic:
                best_aic = aic_c
                best_lag = p_lag

        # Fit final ADF regression at best_lag using full available observations [best_lag:]
        y_dep = diff[best_lag:]
        best_nobs = len(y_dep)
        cols_x: list[np.ndarray] = [
            np.ones(best_nobs, dtype=float),
            vals[best_lag : len(vals) - 1],
        ]
        for j in range(1, best_lag + 1):
            cols_x.append(diff[best_lag - j : n_diff - j])
        X_mat = np.column_stack(cols_x)
        beta, *_ = np.linalg.lstsq(X_mat, y_dep, rcond=None)
        resid = y_dep - X_mat @ beta
        rss = float(np.dot(resid, resid))
        dof = max(best_nobs - X_mat.shape[1], 1)
        s2 = rss / float(dof)
        cov = s2 * np.linalg.pinv(X_mat.T @ X_mat)
        se_gamma = float(np.sqrt(max(cov[1, 1], 1e-18)))
        best_t = float(beta[1] / se_gamma)

        # MacKinnon (2010, Table 2, N=1, constant-only tau_c) response surface:
        # CV(T) = b_inf + b_1 / T + b_2 / T^2 + b_3 / T^3
        T = float(max(best_nobs, 10))
        cv_1 = -3.43035 - 6.5393 / T - 16.786 / (T**2) - 79.433 / (T**3)
        cv_5 = -2.86154 - 2.8903 / T - 4.234 / (T**2) - 40.040 / (T**3)
        cv_10 = -2.56677 - 1.5384 / T - 2.809 / (T**2)
        crit_vals = {"1%": round(cv_1, 6), "5%": round(cv_5, 6), "10%": round(cv_10, 6)}

        # Finite-sample offset relative to asymptotic 5% critical value (-2.86154)
        # mapped onto calibrated MacKinnon (1994) asymptotic tau_c CDF in probit space
        tau_adj = best_t - (cv_5 - (-2.86154))
        mk_taus = np.array(
            [
                -5.0, -4.2, -3.7, -3.43035, -3.15, -2.86154, -2.56677, -2.25, -1.95,
                -1.65, -1.35, -1.0, -0.65, -0.3, 0.0, 0.4, 0.8, 1.5,
            ],
            dtype=float,
        )
        mk_probs = np.array(
            [
                0.000022, 0.000659, 0.004113, 0.009967, 0.023068, 0.050007, 0.100062,
                0.188601, 0.308917, 0.457040, 0.605998, 0.753264, 0.859279, 0.925504,
                0.958532, 0.981495, 0.991636, 0.997524,
            ],
            dtype=float,
        )
        mk_probits = stats.norm.ppf(mk_probs)
        if tau_adj <= mk_taus[0]:
            p_mk = 1e-6
        elif tau_adj >= mk_taus[-1]:
            p_mk = 0.999
        else:
            p_mk = float(stats.norm.cdf(np.interp(tau_adj, mk_taus, mk_probits)))
        p_mk = float(np.clip(p_mk, 0.0, 1.0))

        reject = bool(p_mk < alpha)
        return HypothesisTestResult(
            test_name="augmented_dickey_fuller_mackinnon",
            statistic=round(best_t, 6),
            p_value=round(p_mk, 6),
            alpha=alpha,
            reject_null=reject,
            interpretation=(
                f"Series appears stationary (ADF={best_t:.3f}, p={p_mk:.4g})."
                if reject
                else f"Series is non-stationary; differencing recommended (ADF={best_t:.3f}, p={p_mk:.4g})."
            ),
            details={"used_lag": int(best_lag), "nobs": int(best_nobs), "critical_values": crit_vals},
        )

    @staticmethod
    def jarque_bera_test(values: np.ndarray | pd.Series, alpha: float = 0.05) -> HypothesisTestResult:
        """Jarque-Bera goodness-of-fit test for skewness and kurtosis matching normality."""
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 3:
            raise ValueError("Jarque-Bera test requires at least 3 finite observations.")
        stat, p_val = stats.jarque_bera(arr)
        reject = bool(p_val < alpha)
        return HypothesisTestResult(
            test_name="jarque_bera",
            statistic=round(float(stat), 6),
            p_value=round(float(p_val), 6),
            alpha=alpha,
            reject_null=reject,
            interpretation=(
                f"Reject normality (JB={stat:.3f}, p={p_val:.4g})."
                if reject
                else f"Consistent with normality (JB={stat:.3f}, p={p_val:.4g})."
            ),
            details={"n": int(len(arr))},
        )

    @staticmethod
    def anderson_darling_test(values: np.ndarray | pd.Series, alpha: float = 0.05) -> HypothesisTestResult:
        """Anderson-Darling test for normality (sensitive to distribution tails)."""
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 5:
            raise ValueError("Anderson-Darling test requires at least 5 finite observations.")
        res = stats.anderson(arr, dist="norm")
        stat = float(res.statistic)
        # Map alpha to closest standard significance level (15%, 10%, 5%, 2.5%, 1%)
        levels = np.asarray(res.significance_level, dtype=float) / 100.0
        idx = int(np.argmin(np.abs(levels - alpha)))
        crit = float(res.critical_values[idx])
        reject = bool(stat > crit)
        return HypothesisTestResult(
            test_name="anderson_darling",
            statistic=round(stat, 6),
            p_value=round(float(levels[idx]) if reject else 1.0, 6),
            alpha=alpha,
            reject_null=reject,
            interpretation=(
                f"Reject normality at alpha={levels[idx]:.3f} (A2={stat:.3f} > crit={crit:.3f})."
                if reject
                else f"Consistent with normality at alpha={levels[idx]:.3f} (A2={stat:.3f} <= crit={crit:.3f})."
            ),
            details={"critical_value": crit, "matched_alpha": float(levels[idx])},
        )

    @staticmethod
    def mcnemar_test(table_2x2: np.ndarray | list[list[int]], alpha: float = 0.05) -> HypothesisTestResult:
        """Exact McNemar test for paired nominal 2x2 contingency table."""
        tbl = np.asarray(table_2x2, dtype=int)
        if tbl.shape != (2, 2):
            raise ValueError("McNemar test requires a 2x2 contingency table.")
        b, c = int(tbl[0, 1]), int(tbl[1, 0])
        n_disc = b + c
        if n_disc == 0:
            p_val = 1.0
            stat = 0.0
        else:
            p_val = float(stats.binomtest(min(b, c), n=n_disc, p=0.5, alternative="two-sided").pvalue)
            stat = float(((abs(b - c) - 1.0) ** 2) / n_disc) if n_disc > 0 else 0.0
        reject = bool(p_val < alpha)
        return HypothesisTestResult(
            test_name="mcnemar_exact",
            statistic=round(stat, 6),
            p_value=round(p_val, 6),
            alpha=alpha,
            reject_null=reject,
            interpretation=(
                f"Significant paired difference (b={b}, c={c}, p={p_val:.4g})."
                if reject
                else f"No significant paired difference (b={b}, c={c}, p={p_val:.4g})."
            ),
            details={"b": b, "c": c, "discordant_pairs": n_disc},
        )

    @staticmethod
    def fisher_exact_test(table_2x2: np.ndarray | list[list[int]], alpha: float = 0.05) -> HypothesisTestResult:
        """Fisher's exact test for 2x2 contingency tables (small-sample safe)."""
        tbl = np.asarray(table_2x2, dtype=int)
        if tbl.shape != (2, 2):
            raise ValueError("Fisher's exact test requires a 2x2 contingency table.")
        odds_ratio, p_val = stats.fisher_exact(tbl)
        reject = bool(p_val < alpha)
        return HypothesisTestResult(
            test_name="fisher_exact",
            statistic=round(float(odds_ratio), 6) if np.isfinite(odds_ratio) else float("inf"),
            p_value=round(float(p_val), 6),
            alpha=alpha,
            reject_null=reject,
            interpretation=(
                f"Significant association (OR={odds_ratio:.3f}, p={p_val:.4g})."
                if reject
                else f"No significant association (OR={odds_ratio:.3f}, p={p_val:.4g})."
            ),
            details={"odds_ratio": float(odds_ratio)},
        )

    @staticmethod
    def cohens_d(group_a: np.ndarray | pd.Series, group_b: np.ndarray | pd.Series) -> float:
        """Compute pooled-SD Cohen's d effect size between two independent samples."""
        a = np.asarray(group_a, dtype=float)
        b = np.asarray(group_b, dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 2 or len(b) < 2:
            raise ValueError("Cohen's d requires at least 2 finite observations per group.")
        var_a = float(np.var(a, ddof=1))
        var_b = float(np.var(b, ddof=1))
        pooled_sd = float(np.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)))
        if pooled_sd == 0.0:
            return 0.0
        return float((np.mean(a) - np.mean(b)) / pooled_sd)

    @staticmethod
    def two_sample_power_sample_size(
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> dict[str, float | int]:
        """Normal-approximation sample size per group for two-sided two-sample test."""
        d = abs(float(effect_size))
        if d <= 0.0:
            raise ValueError("effect_size must be non-zero.")
        if not (0.0 < alpha < 1.0) or not (0.0 < power < 1.0):
            raise ValueError("alpha and power must be in (0, 1).")
        z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
        z_beta = float(stats.norm.ppf(power))
        n_per_group = int(np.ceil(2.0 * ((z_alpha + z_beta) / d) ** 2))
        return {
            "effect_size": round(d, 6),
            "alpha": float(alpha),
            "target_power": float(power),
            "n_per_group": n_per_group,
            "total_n": int(2 * n_per_group),
        }


def distance_correlation(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> dict[str, float]:
    """Compute Székely-Rizzo-Bakirov (2007) distance correlation (dCor) for linear & nonlinear dependence."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    n = len(xa)
    if n < 4:
        return {"dcor": 0.0, "dcov": 0.0, "n": float(n)}
    a = np.abs(xa[:, None] - xa[None, :])
    b = np.abs(ya[:, None] - ya[None, :])
    a_cent = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    b_cent = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2_xy = float(np.mean(a_cent * b_cent))
    dcov2_xx = float(np.mean(a_cent * a_cent))
    dcov2_yy = float(np.mean(b_cent * b_cent))
    denom_sq = float(np.sqrt(max(dcov2_xx, 0.0) * max(dcov2_yy, 0.0)))
    if denom_sq <= 1e-12:
        return {"dcor": 0.0, "dcov": 0.0, "n": float(n)}
    dcor = float(np.sqrt(max(dcov2_xy, 0.0) / denom_sq))
    dcor = float(np.clip(dcor, 0.0, 1.0))
    return {
        "dcor": round(dcor, 6),
        "dcov": round(float(np.sqrt(max(dcov2_xy, 0.0))), 6),
        "n": float(n),
    }


def friedman_nemenyi_test(score_matrix: np.ndarray | pd.DataFrame, *, alpha: float = 0.05) -> dict[str, Any]:
    """Demšar (2006) Friedman test with Nemenyi post-hoc Critical Difference (CD) for multi-model comparison."""
    from scipy import stats

    mat = np.asarray(score_matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] < 3 or mat.shape[1] < 2:
        return {"evaluable": False, "reason": "Requires >= 3 datasets/folds and >= 2 models."}
    n_datasets, k_models = mat.shape
    stat, p_val = stats.friedmanchisquare(*[mat[:, j] for j in range(k_models)])
    # Average ranks (higher score = rank 1)
    ranks = np.array([stats.rankdata(-row) for row in mat])
    avg_ranks = np.mean(ranks, axis=0)
    # Approximate Nemenyi q_alpha via studentized range / normal critical value
    q_alpha = float(stats.norm.ppf(1.0 - alpha / (k_models * (k_models - 1)))) if k_models > 2 else 1.96
    cd = float(q_alpha * np.sqrt(k_models * (k_models + 1) / (6.0 * n_datasets)))
    return {
        "evaluable": True,
        "n_datasets": int(n_datasets),
        "k_models": int(k_models),
        "friedman_stat": round(float(stat), 6),
        "p_value": round(float(p_val), 6),
        "significant": bool(p_val < alpha),
        "avg_ranks": [round(float(r), 4) for r in avg_ranks],
        "critical_difference": round(cd, 4),
    }




