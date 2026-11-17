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
                test_name="shapiro_wilk",
                statistic=0.0,
                p_value=1.0,
                alpha=alpha,
                reject_null=False,
                interpretation="Insufficient observations (<8) for normality testing.",
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
            z = (vals - np.mean(vals)) / std
            stat, p_val = stats.kstest(z, "norm")
            name = "kolmogorov_smirnov"
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

        diff_mean = float(np.mean(a) - np.mean(b))
        pooled_sd = float(
            np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / max(len(a) + len(b) - 2, 1))
        )
        cohens_d = float(diff_mean / pooled_sd) if pooled_sd > 1e-12 else 0.0
        se_diff = float(np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=max(len(a) + len(b) - 2, 1)))
        ci_low = diff_mean - t_crit * se_diff
        ci_high = diff_mean + t_crit * se_diff

        if parametric:
            stat, p_val = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
            name = "student_t_test" if equal_var else "welch_t_test"
            es_name = "cohens_d"
        else:
            stat, p_val = stats.mannwhitneyu(a, b, alternative=alternative)
            name = "mann_whitney_u"
            # Rank-biserial correlation effect size
            cohens_d = float(1.0 - (2.0 * float(stat)) / max(len(a) * len(b), 1))
            es_name = "rank_biserial_r"

        reject = bool(p_val < alpha)
        return HypothesisTestResult(
            test_name=name,
            statistic=round(float(stat), 6),
            p_value=round(float(p_val), 6),
            alpha=alpha,
            reject_null=reject,
            effect_size=round(cohens_d, 4),
            effect_size_name=es_name,
            ci_low=round(ci_low, 4),
            ci_high=round(ci_high, 4),
            interpretation=(
                f"Statistically significant difference between groups (p={p_val:.4g}, {es_name}={cohens_d:.2f})."
                if reject
                else f"No statistically significant difference between groups (p={p_val:.4g})."
            ),
            details={"mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)), "n_a": len(a), "n_b": len(b)},
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
        else:
            stat, p_val = stats.kruskal(*groups)
            name = "kruskal_wallis"
            eff = float(max(0.0, (float(stat) - k_groups + 1.0) / max(n_total - k_groups, 1)))
            es_name = "epsilon_squared"

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
            details={"n_groups": k_groups, "n_total": n_total},
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

        # Fisher-z confidence interval for correlation coefficient
        ci_low, ci_high = None, None
        if n > 4 and abs(coef_f) < 0.9999:
            z_val = float(np.arctanh(coef_f))
            se_z = 1.0 / np.sqrt(n - 3.0)
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
        """Compute Variance Inflation Factor (VIF) for numeric features to diagnose multicollinearity."""
        num_df = df.select_dtypes(include=[np.number]).dropna()
        cols = list(num_df.columns)
        if len(cols) < 2 or len(num_df) <= len(cols):
            return {str(c): 1.0 for c in cols}

        X = num_df.to_numpy(dtype=float)
        stds = X.std(axis=0)
        valid_mask = stds > 1e-12
        valid_cols = [cols[i] for i in range(len(cols)) if valid_mask[i]]
        if len(valid_cols) < 2:
            return {str(c): 1.0 for c in cols}

        X_std = (X[:, valid_mask] - X[:, valid_mask].mean(axis=0)) / stds[valid_mask]
        corr = np.corrcoef(X_std, rowvar=False)
        try:
            inv_corr = np.linalg.pinv(corr)
            vifs = np.diag(inv_corr)
        except np.linalg.LinAlgError:
            vifs = np.ones(len(valid_cols))

        return {
            str(col): round(float(max(1.0, vif)), 4) for col, vif in zip(valid_cols, vifs, strict=False)
        }

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
            diff = np.diff(vals)
            lagged = vals[:-1]
            X = np.column_stack([np.ones(len(lagged)), lagged])
            beta, *_ = np.linalg.lstsq(X, diff, rcond=None)
            resid = diff - X @ beta
            s2 = float(np.sum(resid**2) / max(len(diff) - 2, 1))
            cov = s2 * np.linalg.pinv(X.T @ X)
            se = float(np.sqrt(max(cov[1, 1], 1e-12)))
            t_stat = float(beta[1] / se)
            p_approx = float(stats.t.cdf(t_stat, df=max(len(diff) - 2, 1)))
            reject = bool(p_approx < alpha)
            return HypothesisTestResult(
                test_name="dickey_fuller_ar1",
                statistic=round(t_stat, 6),
                p_value=round(p_approx, 6),
                alpha=alpha,
                reject_null=reject,
                interpretation="Series appears stationary." if reject else "Series appears non-stationary.",
            )
