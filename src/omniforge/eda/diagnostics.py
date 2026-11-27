"""Universal domain-agnostic statistical diagnostics, hypothesis ledger, baseline benchmarking & risk engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.model_selection import train_test_split

from omniforge.core.exceptions import DataValidationError
from omniforge.core.performance import smart_sample_df
from omniforge.core.task_inference import infer_supervised_task
from omniforge.core.types import TaskType
from omniforge.data.quality import DataQualityProfiler
from omniforge.eda.statistics import HypothesisTestResult, StatisticalAnalyzer
from omniforge.evaluation.paired_comparison import paired_score_interval
from omniforge.features.inference import WorkbenchInputPreprocessor, nonconstant_predictors
from omniforge.models.baselines import (
    BaselineBenchmarkSuite,
    BaselineEvidence,
    baseline_feature_evidence,
    strongest_scored_reference,
)

# Complete catalog of selectable diagnostic & analytical modules (domain-agnostic)
DIAGNOSTIC_TEST_CATALOG: dict[str, str] = {
    "data_quality_integrity": "6D Data Quality, Missingness, Duplicates & Integrity Audit",
    "normality_distribution": "Normality, Distribution Shape & Bootstrap Confidence Intervals",
    "outlier_anomaly_scan": "Outlier & Extreme Value Scan (IQR / Z-Score / Modified Z-Score)",
    "correlation_association": "Correlation & Categorical Association Significance Tests",
    "group_difference_tests": "Group & Target Difference Hypothesis Tests (T-test / Welch / Mann-Whitney / ANOVA / Kruskal / Levene)",
    "multicollinearity_vif": "Multicollinearity Diagnostics (Variance Inflation Factor - VIF)",
    "stationarity_autocorrelation": "Sequential Stationarity (ADF) & Autocorrelation (Ljung-Box) Tests",
}

DOWNSTREAM_MODULE_CATALOG: dict[str, str] = {
    "baseline_benchmark": "Establish Quantitative Baselines (Dummy / Linear / Tree Stump Floor)",
    "risk_uncertainty_audit": "Quantitative Risk & Uncertainty Audit (VaR / CVaR, Structural Risk & Conformal Bounds)",
    "clustering_segmentation": "Unsupervised Clustering & Natural Segment Discovery (Auto-K + Silhouette / PCA)",
    "automated_eda": "Automated EDA Visual Report & Insights",
    "automl_benchmark": "Baseline-First AutoML Tournament (Single or Multi-Target)",
    "model_card": "Governance & Compliance Model Card",
}


EXECUTABLE_MODULES = {
    **DIAGNOSTIC_TEST_CATALOG,
    "baseline_benchmark": DOWNSTREAM_MODULE_CATALOG["baseline_benchmark"],
    "risk_uncertainty_audit": "Empirical Tail Risk (VaR / CVaR) and Structural Screening",
}


class StatisticalCriteriaConfig(BaseModel):
    """User-configurable statistical testing criteria, thresholds, and risk parameters."""

    alpha: float = Field(default=0.05, gt=0.0, lt=0.5, description="Significance level (Sig / p-value cutoff)")
    confidence_level: float = Field(default=0.95, gt=0.5, lt=0.9999, description="Confidence level for intervals")
    multiple_testing_correction: Literal["none", "bonferroni", "fdr_bh"] = "fdr_bh"
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"
    parametric_mode: Literal["auto", "parametric", "non_parametric"] = "auto"
    normality_method: Literal["auto", "shapiro_wilk", "dagostino_pearson", "kolmogorov_smirnov", "anderson_darling"] = "auto"
    correlation_method: Literal["auto", "pearson", "spearman", "kendall"] = "auto"
    outlier_method: Literal["iqr", "zscore", "modified_zscore"] = "iqr"
    vif_threshold: float = Field(default=5.0, ge=1.5, le=50.0)
    risk_tail_alpha: float = Field(default=0.05, gt=0.001, lt=0.5)
    bootstrap_samples: int = Field(default=250, ge=50, le=2000)
    holdout_ratio: float = Field(default=0.20, ge=0.10, le=0.45)


@dataclass
class UniversalDiagnosticsReport:
    """Structured output of the universal domain-agnostic diagnostics, baseline, and risk suite."""

    criteria: StatisticalCriteriaConfig
    executed_modules: list[str]
    n_scanned: int = 0
    sampled_from: int = 0
    columns_skipped: dict[str, list[str]] = field(default_factory=dict)
    scan_limits: dict[str, Any] = field(default_factory=dict)
    correction_family: dict[str, Any] | None = None
    module_status: dict[str, str] = field(default_factory=dict)
    hypothesis_ledger: list[dict[str, Any]] = field(default_factory=list)
    quality_report: dict[str, Any] | None = None
    distribution_profiles: list[dict[str, Any]] = field(default_factory=list)
    outlier_profiles: list[dict[str, Any]] = field(default_factory=list)
    vif_scores: dict[str, float] = field(default_factory=dict)
    baseline_benchmarks: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    baseline_evidence: dict[str, BaselineEvidence] = field(default_factory=dict)
    baseline_equations: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    baseline_feature_cols: dict[str, list[str]] = field(default_factory=dict)
    baseline_split: dict[str, dict[str, Any]] = field(default_factory=dict)
    risk_assessment: dict[str, Any] | None = None
    recommendations: list[str] = field(default_factory=list)

    @property
    def total_tests(self) -> int:
        return len(self.hypothesis_ledger)

    @property
    def significant_tests(self) -> int:
        return sum(1 for row in self.hypothesis_ledger if row.get("decision") == "Reject H0")


class UniversalDiagnosticsSuite:
    """Domain-agnostic full-dataset statistical testing, baseline establishment, and risk assessment suite."""

    @staticmethod
    def recommend_modules(
        df: pd.DataFrame,
        target_cols: list[str] | None = None,
    ) -> dict[str, str]:
        """Inspect dataset structure and return recommended modules mapped to transparent rationales."""
        targets = [t for t in (target_cols or []) if t in df.columns]
        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        cat_cols = [c for c in df.columns if c not in num_cols or df[c].nunique(dropna=True) <= 12]
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

        recs: dict[str, str] = {
            "data_quality_integrity": "Always recommended to audit completeness, uniqueness, validity, and schema health.",
        }
        if num_cols:
            recs["normality_distribution"] = (
                f"Detected {len(num_cols)} numeric column(s); checks normality, skewness, and confidence bounds."
            )
            recs["outlier_anomaly_scan"] = (
                f"Scans {len(num_cols)} numeric column(s) for extreme values that can distort tests and models."
            )
        if len(num_cols) >= 2 or len(cat_cols) >= 2:
            recs["correlation_association"] = (
                "Tests statistical significance of numeric correlations and categorical associations."
            )
        if num_cols and cat_cols:
            recs["group_difference_tests"] = (
                "Compares numeric distributions across categorical groups/targets using variance & mean/rank tests."
            )
        if len(num_cols) >= 2:
            recs["multicollinearity_vif"] = (
                f"Evaluates collinearity across {len(num_cols)} numeric columns using VIF."
            )
        if dt_cols or (len(df) >= 24 and len(num_cols) >= 1 and any(k in c.lower() for c in df.columns for k in ("date", "time", "ds", "period", "step"))):
            recs["stationarity_autocorrelation"] = (
                "Detected sequential/time-indexed structure; runs ADF stationarity and Ljung-Box tests."
            )

        recs["risk_uncertainty_audit"] = (
            "Quantifies tail risk (VaR/CVaR), structural data risk score, and uncertainty bounds."
        )
        if len(num_cols) >= 2:
            recs["clustering_segmentation"] = (
                f"Discovers natural latent segments across {len(num_cols)} numeric feature(s) with Auto-K Silhouette validation."
            )
        if targets and len(df.columns) > len(targets):
            recs["baseline_benchmark"] = (
                f"Establishes quantitative baseline performance floors for target(s): {', '.join(targets)}."
            )
            recs["automated_eda"] = "Generates interactive distributions, relationships, and executive narrative."
        return recs

    @staticmethod
    def _is_classification_series(series: pd.Series) -> bool:
        return infer_supervised_task(series) != TaskType.REGRESSION

    @staticmethod
    def _bootstrap_mean_ci(
        vals: np.ndarray,
        confidence_level: float,
        n_boot: int,
        seed: int = 42,
    ) -> tuple[float, float]:
        if len(vals) < 3:
            m = float(np.mean(vals)) if len(vals) > 0 else 0.0
            return m, m
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
        means = vals[idx].mean(axis=1)
        tail = (1.0 - confidence_level) / 2.0
        return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))

    @staticmethod
    def _detect_outliers(vals: np.ndarray, method: Literal["iqr", "zscore", "modified_zscore"]) -> tuple[int, float, float, float]:
        if len(vals) < 4:
            return 0, 0.0, float(np.min(vals)) if len(vals) else 0.0, float(np.max(vals)) if len(vals) else 0.0
        if method == "iqr":
            q1, q3 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (vals < low) | (vals > high)
        elif method == "zscore":
            mean_v, std_v = float(np.mean(vals)), float(np.std(vals, ddof=1))
            if std_v <= 1e-12:
                return 0, 0.0, mean_v, mean_v
            low, high = mean_v - 3.0 * std_v, mean_v + 3.0 * std_v
            mask = np.abs((vals - mean_v) / std_v) > 3.0
        else:
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            if mad <= 1e-12:
                return 0, 0.0, med, med
            mod_z = 0.6745 * (vals - med) / mad
            low, high = med - 3.5 * mad / 0.6745, med + 3.5 * mad / 0.6745
            mask = np.abs(mod_z) > 3.5
        cnt = int(np.sum(mask))
        return cnt, round(float(cnt / len(vals) * 100.0), 2), round(low, 4), round(high, 4)

    def run(
        self,
        df: pd.DataFrame,
        *,
        target_cols: list[str] | None = None,
        selected_modules: list[str] | None = None,
        criteria: StatisticalCriteriaConfig | None = None,
        baseline_partitions: tuple[pd.DataFrame, pd.DataFrame] | None = None,
        ignored_columns: list[str] | None = None,
        baseline_split_strategy: str | None = None,
    ) -> UniversalDiagnosticsReport:
        """Execute user-selected statistical tests, baseline benchmarks, and risk audits with exact criteria."""
        cfg = criteria or StatisticalCriteriaConfig()
        targets = [t for t in (target_cols or []) if t in df.columns]
        modules = list(selected_modules) if selected_modules is not None else list(self.recommend_modules(df, targets).keys())

        unknown = set(modules) - EXECUTABLE_MODULES.keys() if selected_modules is not None else set()
        if unknown:
            raise DataValidationError(
                "Unknown or non-executable diagnostic modules.",
                suggestions=["Choose modules from the executable diagnostics catalog."],
            )
        modules = [module for module in dict.fromkeys(modules) if module in EXECUTABLE_MODULES]
        report = UniversalDiagnosticsReport(criteria=cfg, executed_modules=[])
        # Bound heavy statistical scans deterministically on huge tables (>25k rows) for sub-second UX
        work_df, was_sampled = smart_sample_df(
            df,
            max_rows=25_000,
            stratify_col=targets[0] if targets and targets[0] in df.columns else None,
            seed=42,
        )
        report.n_scanned = len(work_df)
        report.sampled_from = len(df)
        report.scan_limits["row_sampling"] = {"sampled": was_sampled, "max_rows": 25_000, "seed": 42}

        def bounded_columns(key: str, candidates: list[str], limit: int) -> list[str]:
            report.scan_limits[key] = {"max_columns": limit, "eligible_columns": len(candidates)}
            report.columns_skipped[key] = candidates[limit:]
            return candidates[:limit]

        num_cols = list(work_df.select_dtypes(include=[np.number]).columns)
        cat_cols = [
            c
            for c in work_df.columns
            if (c not in num_cols or (2 <= work_df[c].nunique(dropna=True) <= 10))
            and 2 <= work_df[c].nunique(dropna=True) <= 25
        ]
        # Use work_df for statistical inference while DataQualityProfiler still profiles full df (cached)
        full_df = df
        df = work_df

        raw_ledger: list[dict[str, Any]] = []
        normality_flags: dict[str, bool] = {}

        def _append_test(family: str, variable: str, res: HypothesisTestResult) -> None:
            raw_ledger.append(
                {
                    "family": family,
                    "variable": variable,
                    "test_name": res.test_name,
                    "statistic": res.statistic,
                    "p_value": res.p_value,
                    "adjusted_p_value": res.p_value,
                    "alpha": cfg.alpha,
                    "effect_size": res.effect_size,
                    "effect_size_name": res.effect_size_name or "",
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    "decision": "Reject H0" if res.p_value < cfg.alpha else "Fail to Reject H0",
                    "interpretation": res.interpretation,
                }
            )

        # 1. Data Quality & Integrity
        if "data_quality_integrity" in modules:
            q_rep = DataQualityProfiler().profile(full_df)
            report.quality_report = q_rep.model_dump()
            if q_rep.overall_score < 85.0:
                report.recommendations.append(
                    f"Data quality score is {q_rep.overall_score:.1f}/100 ({q_rep.grade}); review missingness/cleaning in Studio 01."
                )

        # 2. Normality, Shape & Bootstrap CIs
        if "normality_distribution" in modules or "group_difference_tests" in modules or "correlation_association" in modules:
            for col in bounded_columns("normality_distribution", num_cols, 30):
                vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
                n_res = StatisticalAnalyzer.normality_test(df[col], alpha=cfg.alpha, method=cfg.normality_method)
                normality_flags[col] = not n_res.reject_null
                if "normality_distribution" in modules:
                    _append_test("Normality", col, n_res)
                    ci_l, ci_h = self._bootstrap_mean_ci(vals, cfg.confidence_level, cfg.bootstrap_samples)
                    skew_v = float(stats.skew(vals)) if len(vals) >= 3 and np.std(vals) > 1e-12 else 0.0
                    kurt_v = float(stats.kurtosis(vals)) if len(vals) >= 4 and np.std(vals) > 1e-12 else 0.0
                    report.distribution_profiles.append(
                        {
                            "Column": col,
                            "N": len(vals),
                            "Mean": round(float(np.mean(vals)), 4) if len(vals) else 0.0,
                            "Median": round(float(np.median(vals)), 4) if len(vals) else 0.0,
                            "Std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0,
                            "Skewness": round(skew_v, 3),
                            "Excess Kurtosis": round(kurt_v, 3),
                            f"CI Low ({int(cfg.confidence_level * 100)}%)": round(ci_l, 4),
                            f"CI High ({int(cfg.confidence_level * 100)}%)": round(ci_h, 4),
                            "Normality Test": n_res.test_name,
                            "Normality p-value": n_res.p_value,
                            "Normal?": "Yes" if not n_res.reject_null else "No",
                        }
                    )

        # 3. Outlier & Extreme Value Scan
        if "outlier_anomaly_scan" in modules:
            for col in bounded_columns("outlier_anomaly_scan", num_cols, 30):
                vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
                cnt, pct, low, high = self._detect_outliers(vals, cfg.outlier_method)
                report.outlier_profiles.append(
                    {
                        "Column": col,
                        "Method": cfg.outlier_method,
                        "Outlier Count": cnt,
                        "Outlier %": pct,
                        "Lower Fence": low,
                        "Upper Fence": high,
                        "Severity": "High" if pct >= 5.0 else ("Moderate" if pct >= 1.5 else "Low"),
                    }
                )
            high_out = [r["Column"] for r in report.outlier_profiles if r["Outlier %"] >= 5.0]
            if high_out:
                report.recommendations.append(
                    f"High outlier rate (>=5%) detected in {', '.join(high_out[:5])}; consider robust scaling or outlier clipping."
                )

        # 4. Correlation & Categorical Association Tests
        if "correlation_association" in modules:
            # Prioritize target-vs-feature pairs first, then top numeric pairs (cap at 35 pairs for responsiveness)
            num_pairs: list[tuple[str, str]] = []
            seen_pairs: set[tuple[str, str]] = set()
            num_targets = [t for t in targets if t in num_cols]
            for t in num_targets:
                for c in num_cols:
                    if c != t:
                        key = (min(t, c), max(t, c))
                        if key not in seen_pairs:
                            seen_pairs.add(key)
                            num_pairs.append((c, t))
            for i, c1 in enumerate(num_cols[:12]):
                for c2 in num_cols[i + 1 : 12]:
                    key = (min(c1, c2), max(c1, c2))
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        num_pairs.append((c1, c2))

            selected_num_pairs = num_pairs[:35]
            scanned_num = {c for pair in selected_num_pairs for c in pair}
            report.columns_skipped["correlation_numeric"] = [c for c in num_cols if c not in scanned_num]
            report.scan_limits["correlation_numeric"] = {
                "non_target_candidate_columns": 12, "max_pairs": 35, "n_pairs": len(selected_num_pairs),
                "all_possible_pairs": len(num_cols) * (len(num_cols) - 1) // 2,
            }
            for c1, c2 in selected_num_pairs:
                if cfg.correlation_method == "auto":
                    both_normal = normality_flags.get(c1, False) and normality_flags.get(c2, False)
                    corr_m: Literal["pearson", "spearman", "kendall"] = (
                        "pearson"
                        if (cfg.parametric_mode == "parametric" or (cfg.parametric_mode == "auto" and both_normal))
                        else "spearman"
                    )
                else:
                    corr_m = cfg.correlation_method
                c_res = StatisticalAnalyzer.correlation_test(df[c1], df[c2], method=corr_m, alpha=cfg.alpha)
                _append_test("Correlation", f"{c1} vs {c2}", c_res)

            # Categorical independence tests (Chi-Square / Fisher Exact + Cramér's V)
            cat_pairs: list[tuple[str, str]] = []
            for i, a in enumerate(cat_cols[:10]):
                for b in cat_cols[i + 1 : 10]:
                    cat_pairs.append((a, b))
            selected_cat_pairs = cat_pairs[:20]
            scanned_cat = {c for pair in selected_cat_pairs for c in pair}
            report.columns_skipped["correlation_categorical"] = [c for c in cat_cols if c not in scanned_cat]
            report.scan_limits["correlation_categorical"] = {
                "candidate_columns": 10, "max_pairs": 20, "n_pairs": len(selected_cat_pairs),
                "all_possible_pairs": len(cat_cols) * (len(cat_cols) - 1) // 2,
            }
            for a, b in selected_cat_pairs:
                chi_res = StatisticalAnalyzer.chi_square_independence(df[a], df[b], alpha=cfg.alpha)
                _append_test("Categorical Association", f"{a} vs {b}", chi_res)

        # 5. Group & Target Difference Hypothesis Tests (Levene + Student/Welch T-test / Mann-Whitney U / ANOVA / Kruskal)
        if "group_difference_tests" in modules:
            group_candidates = [t for t in targets if t in cat_cols] + [c for c in cat_cols if c not in targets]
            group_cols_used = bounded_columns("group_columns", group_candidates, 4)
            numeric_predictors = bounded_columns("group_numeric_predictors", [c for c in num_cols if c not in group_cols_used], 15)
            report.scan_limits["group_columns"]["max_levels"] = 12
            for g_col in group_cols_used:
                n_grps = int(df[g_col].nunique(dropna=True))
                if n_grps < 2 or n_grps > 12:
                    report.columns_skipped["group_columns"].append(g_col)
                    continue
                for v_col in numeric_predictors:
                    lev = StatisticalAnalyzer.levene_variance_test(df, v_col, g_col, alpha=cfg.alpha)
                    _append_test("Variance Homogeneity", f"{v_col} by {g_col}", lev)

                    is_normal = normality_flags.get(v_col, False)
                    use_parametric = (
                        True
                        if cfg.parametric_mode == "parametric"
                        else (False if cfg.parametric_mode == "non_parametric" else is_normal)
                    )
                    if n_grps == 2:
                        uniq = [u for u in df[g_col].dropna().unique()]
                        g_a = df.loc[df[g_col] == uniq[0], v_col]
                        g_b = df.loc[df[g_col] == uniq[1], v_col]
                        res = StatisticalAnalyzer.two_sample_test(
                            g_a,
                            g_b,
                            parametric=use_parametric,
                            equal_var=not lev.reject_null,
                            alternative=cfg.alternative,
                            alpha=cfg.alpha,
                        )
                    else:
                        res = StatisticalAnalyzer.multi_group_test(
                            df,
                            v_col,
                            g_col,
                            parametric=use_parametric,
                            alpha=cfg.alpha,
                        )
                    _append_test("Group Comparison", f"{v_col} by {g_col}", res)

        # 6. Multicollinearity (VIF)
        if "multicollinearity_vif" in modules and len(num_cols) >= 2:
            pred_num_df = df[[c for c in num_cols if c not in targets]] if len([c for c in num_cols if c not in targets]) >= 2 else df[num_cols]
            report.vif_scores = StatisticalAnalyzer.compute_vif(pred_num_df)
            high_vif = [k for k, v in report.vif_scores.items() if v >= cfg.vif_threshold]
            if high_vif:
                report.recommendations.append(
                    f"Multicollinearity alert (VIF >= {cfg.vif_threshold:.1f}) in {', '.join(high_vif[:5])}; consider regularized models or feature pruning."
                )

        # 7. Sequential Stationarity (ADF) & Autocorrelation (Ljung-Box)
        if "stationarity_autocorrelation" in modules and num_cols:
            seq_cols = [t for t in targets if t in num_cols] + [c for c in num_cols if c not in targets]
            for col in bounded_columns("stationarity_autocorrelation", seq_cols, 8):
                adf_res = StatisticalAnalyzer.stationarity_test(df[col], alpha=cfg.alpha)
                lb_res = StatisticalAnalyzer.ljung_box_autocorrelation_test(df[col], alpha=cfg.alpha)
                _append_test("Stationarity", col, adf_res)
                _append_test("Autocorrelation", col, lb_res)

        # Apply user-selected multiple-testing correction across the complete hypothesis ledger
        if raw_ledger:
            raw_ps = [float(r["p_value"]) for r in raw_ledger]
            adj_ps = StatisticalAnalyzer.adjust_p_values(raw_ps, method=cfg.multiple_testing_correction)
            family = "studio03_global_evaluable_ledger" if cfg.multiple_testing_correction != "none" else None
            report.correction_family = {
                "id": family, "method": cfg.multiple_testing_correction,
                "n_evaluable": int(np.isfinite(adj_ps).sum()),
            }
            for row, adj_p in zip(raw_ledger, adj_ps, strict=False):
                evaluable = bool(np.isfinite(adj_p))
                row["correction_family"] = family
                row["adjusted_p_value"] = adj_p if evaluable else None
                if not evaluable:
                    row["p_value"] = None
                    row["decision"] = "Not evaluable"
                else:
                    row["decision"] = "Reject H0" if adj_p < cfg.alpha else "Fail to Reject H0"
            report.hypothesis_ledger = raw_ledger

        # 8. Quantitative Baseline Benchmarks + Explicit Mathematical Baseline Equations (per selected target)
        self.last_baseline_models_: dict[str, dict[str, Any]] = {}
        if "baseline_benchmark" in modules and targets:
            from omniforge.evaluation.equations import ModelEquationExtractor

            excluded = set(targets) | set(ignored_columns or ())
            eligible_predictors = [c for c in full_df.columns if c not in excluded]
            if eligible_predictors:
                for t_col in targets:
                    strat: pd.Series | None = None
                    if baseline_partitions is None:
                        work = full_df.dropna(subset=[t_col]).copy()
                        if len(work) < 12:
                            continue
                        target = work[t_col]
                        strat = target if self._is_classification_series(target) and int(target.value_counts().min()) >= 2 else None
                        try:
                            train_positions, test_positions = train_test_split(
                                np.arange(len(work)), test_size=cfg.holdout_ratio, random_state=42, stratify=strat,
                            )
                        except ValueError:
                            report.recommendations.append(f"Skipped baseline for '{t_col}': cannot form an evaluable holdout.")
                            continue
                        train, test = work.iloc[train_positions], work.iloc[test_positions]
                        partition_source = "diagnostics_holdout"
                    else:
                        train, test = (part.dropna(subset=[t_col]) for part in baseline_partitions)
                        partition_source = "split_lock"
                    if len(train) < 2 or len(test) < 2:
                        report.recommendations.append(f"Skipped baseline for '{t_col}': too few observed targets in the fixed partitions.")
                        continue
                    y_tr, y_val = train[t_col], test[t_col]
                    is_clf = self._is_classification_series(y_tr)
                    pred_cols = nonconstant_predictors(train[eligible_predictors])
                    if not pred_cols:
                        report.recommendations.append(f"Skipped baseline for '{t_col}': no nonconstant predictors in the training partition.")
                        continue

                    preprocessor = WorkbenchInputPreprocessor(mode="numeric").fit(train[pred_cols])
                    X_tr = preprocessor.transform(train[pred_cols])
                    X_val = preprocessor.transform(test[pred_cols])
                    report.baseline_split[t_col] = {
                        "source": partition_source,
                        "training_rows": len(train),
                        "holdout_rows": len(test),
                    }

                    if is_clf and int(pd.Series(y_tr).nunique(dropna=True)) < 2:
                        report.recommendations.append(
                            f"Skipped classification baselines for '{t_col}': training split contains only 1 class after row filtering."
                        )
                        continue
                    if not is_clf and float(np.nanstd(pd.to_numeric(y_tr, errors="coerce"))) == 0.0:
                        report.recommendations.append(
                            f"Skipped regression baselines for '{t_col}': target has zero variance in the training split."
                        )
                        continue

                    baseline_evidence: BaselineEvidence = {}
                    baseline_predictions: dict[str, np.ndarray] = {}
                    report.baseline_evidence[t_col] = baseline_evidence
                    if is_clf:
                        bl_metrics, bl_models = BaselineBenchmarkSuite.run_classification_detailed(
                            X_tr, y_tr, X_val, y_val, evidence=baseline_evidence, predictions=baseline_predictions,
                        )
                    else:
                        bl_metrics, bl_models = BaselineBenchmarkSuite.run_regression_detailed(
                            X_tr, y_tr, X_val, y_val, evidence=baseline_evidence, predictions=baseline_predictions,
                        )
                    report.baseline_benchmarks[t_col] = bl_metrics
                    report.baseline_feature_cols[t_col] = list(pred_cols)
                    # Preserve the exact scored models and the fitted raw-input schema.
                    raw_models: dict[str, Any] = {}
                    primary_metric = "f1_weighted" if is_clf else "r2"
                    higher_is_better = True
                    criterion_label = (
                        "Holdout F1-Weighted (Higher is Better · Tie-Break: Accuracy & Log-Loss)"
                        if is_clf
                        else "Holdout R² & RMSE (Higher R² / Lower RMSE is Better)"
                    )

                    # Determine naive floor baseline score for explicit lift & acceptance criteria
                    naive_reference = strongest_scored_reference(
                        bl_metrics, baseline_evidence, primary_metric, higher_is_better=True, role="naive",
                    )
                    naive_floor = naive_reference[1] if naive_reference is not None else None
                    max_vif = max((float(v) for v in report.vif_scores.values() if not np.isnan(v)), default=1.0)

                    # Rank all baselines best -> worst
                    def _rank_key(
                        item: tuple[str, dict[str, float]],
                        _is_clf: bool = is_clf,
                    ) -> tuple[float, float]:
                        _, md = item
                        if _is_clf:
                            return (float(md.get("f1_weighted", 0.0)), float(md.get("accuracy", 0.0)))
                        return (float(md.get("r2", -1e9)), -float(md.get("rmse", 1e9)))

                    ranked_names = [nm for nm, _ in sorted(bl_metrics.items(), key=_rank_key, reverse=True)]
                    rank_lookup = {nm: idx + 1 for idx, nm in enumerate(ranked_names)}

                    enriched_eqs: list[dict[str, Any]] = []
                    for bl_name in ranked_names:
                        bl_m = bl_models.get(bl_name)
                        if bl_m is None:
                            continue
                        eq_rep = ModelEquationExtractor.extract(
                            bl_m,
                            X_reference=X_tr,
                            feature_names=list(X_tr.columns),
                            target_name=t_col,
                        )
                        eq_d = eq_rep.model_dump()
                        m_scores = dict(bl_metrics.get(bl_name, {}))
                        p_score = float(m_scores.get(primary_metric, 0.0))
                        rank_num = int(rank_lookup.get(bl_name, len(ranked_names)))
                        is_naive = baseline_evidence[bl_name]["role"] == "naive"
                        lift_abs = round(p_score - naive_floor, 5) if naive_floor is not None else None
                        lift_pct = (
                            round(lift_abs / abs(naive_floor) * 100.0, 2)
                            if lift_abs is not None and naive_floor is not None and abs(naive_floor) > 1e-4 else None
                        )

                        comparison: dict[str, Any] = {"status": "unavailable", "reason": "No evaluable naive reference."}
                        if naive_reference is not None and not is_naive:
                            strategy = baseline_split_strategy if baseline_partitions is not None else (
                                "stratified" if strat is not None else "random"
                            )
                            comparison = paired_score_interval(
                                y_val, baseline_predictions[bl_name], baseline_predictions[naive_reference[0]],
                                metric=primary_metric, split_strategy=strategy, reference_name=naive_reference[0],
                                comparisons=len(bl_metrics) * sum(row["role"] == "naive" for row in baseline_evidence.values()),
                                confidence=cfg.confidence_level,
                            )
                        if is_naive:
                            acc_status = "REFERENCE — NAIVE FLOOR" if naive_reference else "REFERENCE — NAIVE FAMILY INCOMPLETE"
                        elif comparison.get("supports_improvement"):
                            acc_status = "ACCEPTED — CHAMPION BASELINE" if rank_num == 1 else "ACCEPTED — VALIDATED BASELINE"
                        elif comparison["status"] == "available":
                            acc_status = "BENCHMARKED — IMPROVEMENT NOT ESTABLISHED"
                        else:
                            acc_status = "BENCHMARKED — COMPARISON UNAVAILABLE"

                        reasons: list[str] = [
                            f"Ranked #{rank_num} of {len(ranked_names)} candidates on {len(test)} held-out rows ({partition_source}).",
                            f"Primary Criterion ({criterion_label}): {primary_metric.upper()} = {p_score:.4f}"
                            + (f" | RMSE = {float(m_scores.get('rmse', 0.0)):.4f}, MAE = {float(m_scores.get('mae', 0.0)):.4f}" if not is_clf else f" | Accuracy = {float(m_scores.get('accuracy', 0.0)):.4f}"),
                        ]
                        if naive_floor is None or lift_abs is None:
                            reasons.append("Naive floor unavailable: not all designated dummy references were scored.")
                        elif is_naive:
                            reasons.append(
                                "Retained as mandatory zero-skill statistical reference floor — all downstream ML/Neural models must statistically beat this floor."
                            )
                        elif lift_abs > 0:
                            pct_txt = f" ({lift_pct:+.1f}%)" if lift_pct is not None else ""
                            reasons.append(f"Beats Naive Floor ({naive_floor:.4f}) by +{lift_abs:.4f}{pct_txt}, on this holdout.")
                        else:
                            reasons.append(f"Did not beat Naive Floor ({naive_floor:.4f}); flagged for review or non-linear modeling in Studio 06.")

                        low_name = bl_name.lower()
                        if "nonparametric" in low_name or "spline" in low_name or "nadaraya" in low_name or "knn" in low_name or "kernel" in low_name or "isotonic" in low_name:
                            model_paradigm = "Non-Parametric Analytical"
                            if "spline" in low_name:
                                reasons.append("Distribution-free Penalized B-Spline GAM captures smooth non-linear per-feature curves without normality or linearity assumptions.")
                            elif "nadaraya" in low_name:
                                reasons.append("Distribution-free Gaussian Kernel Nadaraya-Watson smoother computes exact locally weighted conditional expectations with CV bandwidth.")
                            elif "knn" in low_name:
                                reasons.append("Non-parametric inverse-distance weighted local k-NN adapts freely to multi-modal and non-linear response manifolds.")
                            else:
                                reasons.append("Distribution-free non-parametric analytical model requiring zero parametric shape assumptions.")
                        elif "ridge" in low_name or "logistic" in low_name:
                            model_paradigm = "Parametric Linear"
                            if max_vif >= cfg.vif_threshold:
                                reasons.append(f"L2 regularization specifically suitable because predictor multicollinearity was detected (max VIF = {max_vif:.2f} >= {cfg.vif_threshold:.1f}).")
                            else:
                                reasons.append("Closed-form regularized parametric equation provides full coefficient auditability and stable variance.")
                        elif "stump" in low_name or "tree" in low_name:
                            model_paradigm = "Tree Rule"
                            reasons.append("Captures non-linear single-feature threshold splits without normality or linearity assumptions.")
                        else:
                            model_paradigm = "Naive Floor"

                        # Extract structured sorted terms for modern visual equation rendering
                        structured_terms: list[dict[str, Any]] = []
                        raw_coefs = eq_d.get("coefficients") or {}
                        if isinstance(raw_coefs, dict):
                            for feat_k, coef_v in raw_coefs.items():
                                if isinstance(coef_v, (int, float)) and np.isfinite(coef_v) and abs(float(coef_v)) > 1e-12:
                                    structured_terms.append(
                                        {
                                            "feature": str(feat_k),
                                            "coefficient": round(float(coef_v), 6),
                                            "abs_weight": round(abs(float(coef_v)), 6),
                                            "direction": "+" if float(coef_v) >= 0 else "-",
                                        }
                                    )
                            structured_terms.sort(key=lambda r: r["abs_weight"], reverse=True)

                        eq_d.update(
                            {
                                **baseline_feature_evidence(bl_m),
                                "rank": rank_num,
                                "total_candidates": len(ranked_names),
                                "model_paradigm": model_paradigm,
                                "task_type": "classification" if is_clf else "regression",
                                "primary_metric": primary_metric,
                                "primary_score": round(p_score, 5),
                                "higher_is_better": higher_is_better,
                                "selection_criterion": criterion_label,
                                "naive_floor_score": naive_floor,
                                "naive_floor_status": "available" if naive_reference is not None else "unavailable",
                                "naive_floor_model": naive_reference[0] if naive_reference is not None else None,
                                "lift_over_naive": lift_abs,
                                "lift_over_naive_pct": lift_pct,
                                "lift_comparison": comparison,
                                "acceptance_status": acc_status,
                                "acceptance_reasons": reasons,
                                "metrics": {k: round(float(v), 5) for k, v in m_scores.items() if isinstance(v, (int, float)) and np.isfinite(v)},
                                "structured_terms": structured_terms[:16],
                                "holdout_rows": int(len(X_val)),
                                "train_rows": int(len(X_tr)),
                            }
                        )
                        enriched_eqs.append(eq_d)

                        raw_models[bl_name] = {
                            "model": bl_m,
                            "feature_cols": list(X_tr.columns),
                            "input_preprocessor": preprocessor,
                            "evaluation_holdout": (X_val, y_val),
                            "metrics": eq_d["metrics"],
                        }
                    report.baseline_equations[t_col] = enriched_eqs
                    self.last_baseline_models_[t_col] = raw_models
            elif len(full_df.columns) == len(targets) and len(full_df) >= 20:
                # Forecasting baselines use chronological partitions for standalone series.
                for t_col in targets:
                    if pd.api.types.is_numeric_dtype(df[t_col]):
                        if baseline_partitions is None:
                            vals = full_df[t_col].dropna().to_numpy(dtype=float)
                            split_i = max(10, int(len(vals) * (1.0 - cfg.holdout_ratio)))
                            train_vals, test_vals = vals[:split_i], vals[split_i:]
                            source = "diagnostics_chronological_holdout"
                        else:
                            train_vals, test_vals = (
                                part[t_col].dropna().to_numpy(dtype=float) for part in baseline_partitions
                            )
                            source = "split_lock"
                        if len(train_vals) < 10 or len(test_vals) < 2:
                            continue
                        report.baseline_split[t_col] = {
                            "source": source, "training_rows": len(train_vals), "holdout_rows": len(test_vals),
                        }
                        fc_metrics, fc_models = BaselineBenchmarkSuite.run_forecasting_detailed(
                            train_vals, test_vals
                        )
                        report.baseline_benchmarks[t_col] = fc_metrics
                        self.last_baseline_models_[t_col] = fc_models
                        for fc_m in fc_models.values():
                            eq_rep = ModelEquationExtractor.extract(fc_m, target_name=t_col)
                            report.baseline_equations.setdefault(t_col, []).append(eq_rep.model_dump())

        # 9. Quantitative Risk & Uncertainty Assessment (VaR, CVaR, Tail Risk, Structural Risk Index, Target Uncertainty)
        if "risk_uncertainty_audit" in modules:
            tail_columns = bounded_columns("risk_tail_columns", num_cols, 25)
            risk_outlier_columns = bounded_columns("risk_outlier_columns", num_cols, 15) if not report.outlier_profiles else []
            report.risk_assessment = self._compute_risk_assessment(
                df, targets, num_cols, report, cfg, tail_columns=tail_columns, risk_outlier_columns=risk_outlier_columns,
            )

        tested_families = {row["family"] for row in raw_ledger if row["decision"] != "Not evaluable"}
        evidence = {
            "data_quality_integrity": bool(report.quality_report),
            "normality_distribution": bool(report.distribution_profiles),
            "outlier_anomaly_scan": bool(report.outlier_profiles),
            "correlation_association": bool({"Correlation", "Categorical Association"} & tested_families),
            "group_difference_tests": "Group Comparison" in tested_families,
            "multicollinearity_vif": bool(report.vif_scores),
            "stationarity_autocorrelation": bool({"Stationarity", "Autocorrelation"} & tested_families),
            "baseline_benchmark": any(report.baseline_benchmarks.values()),
            "risk_uncertainty_audit": report.risk_assessment is not None,
        }
        report.module_status = {module: "executed" if evidence[module] else "unavailable" for module in modules}
        report.executed_modules = [module for module in modules if evidence[module]]
        return report

    def _compute_risk_assessment(
        self,
        df: pd.DataFrame,
        targets: list[str],
        num_cols: list[str],
        report: UniversalDiagnosticsReport,
        cfg: StatisticalCriteriaConfig,
        *,
        tail_columns: list[str],
        risk_outlier_columns: list[str],
    ) -> dict[str, Any]:
        """Compute quantitative tail risk (VaR/CVaR), structural data risk index (0-100), and target uncertainty."""
        tail_rows: list[dict[str, Any]] = []
        fat_tail_count = 0
        for col in tail_columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) < 5:
                continue
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1))
            var_low = float(np.quantile(vals, cfg.risk_tail_alpha))
            var_high = float(np.quantile(vals, 1.0 - cfg.risk_tail_alpha))
            low_tail = vals[vals <= var_low]
            high_tail = vals[vals >= var_high]
            cvar_low = float(np.mean(low_tail)) if len(low_tail) else var_low
            cvar_high = float(np.mean(high_tail)) if len(high_tail) else var_high
            # Below-mean semi-deviation: sqrt((1/N) * sum(min(0, x_i - mean)^2)) (Markowitz 1959; Sortino & Price 1994)
            downside_std = float(np.sqrt(np.mean(np.minimum(0.0, vals - mean_v) ** 2)))
            kurt = float(stats.kurtosis(vals)) if std_v > 1e-12 else 0.0
            if kurt > 3.0:
                fat_tail_count += 1
            tail_rows.append(
                {
                    "Column": col,
                    "Mean": round(mean_v, 4),
                    "Volatility (Std)": round(std_v, 4),
                    "Downside Risk (Semi-Dev)": round(downside_std, 4),
                    f"Lower VaR ({int(cfg.risk_tail_alpha * 100)}%)": round(var_low, 4),
                    f"Lower CVaR ({int(cfg.risk_tail_alpha * 100)}%)": round(cvar_low, 4),
                    f"Upper VaR ({int((1 - cfg.risk_tail_alpha) * 100)}%)": round(var_high, 4),
                    f"Upper CVaR ({int((1 - cfg.risk_tail_alpha) * 100)}%)": round(cvar_high, 4),
                    "Excess Kurtosis": round(kurt, 3),
                    "Tail Profile": "Fat-Tailed (High Risk)" if kurt > 3.0 else ("Elevated" if kurt > 1.0 else "Normal-Like"),
                }
            )

        # Structural Data Risk Index (0 = lowest risk, 100 = highest risk)
        missing_rate = float(df.isna().sum().sum() / max(df.shape[0] * max(df.shape[1], 1), 1))
        dup_rate = float(df.duplicated().mean()) if len(df) > 0 else 0.0
        avg_out_pct = (
            float(np.mean([r["Outlier %"] for r in report.outlier_profiles]))
            if report.outlier_profiles
            else float(
                np.mean([self._detect_outliers(pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float), cfg.outlier_method)[1] for c in risk_outlier_columns])
                if risk_outlier_columns
                else 0.0
            )
        )
        vif_map = report.vif_scores or (StatisticalAnalyzer.compute_vif(df[num_cols]) if len(num_cols) >= 2 else {})
        max_vif = float(max(vif_map.values())) if vif_map else 1.0
        sample_to_feat = float(len(df) / max(df.shape[1], 1))

        risk_components = {
            "missingness_risk": round(min(25.0, missing_rate * 120.0), 2),
            "duplication_risk": round(min(15.0, dup_rate * 100.0), 2),
            "outlier_tail_risk": round(min(25.0, avg_out_pct * 2.5 + fat_tail_count * 1.5), 2),
            "multicollinearity_risk": round(min(20.0, max(0.0, (max_vif - 2.0) * 2.0)), 2),
            "sample_sparsity_risk": round(min(15.0, max(0.0, (15.0 - sample_to_feat) * 1.2)), 2),
        }
        total_risk = round(min(100.0, sum(risk_components.values())), 2)
        risk_band = (
            "Critical" if total_risk >= 70 else ("High" if total_risk >= 45 else ("Moderate" if total_risk >= 20 else "Low"))
        )

        # Target-level uncertainty & imbalance risk
        target_uncertainty: list[dict[str, Any]] = []
        for t_col in targets:
            s = df[t_col].dropna()
            if len(s) == 0:
                continue
            if self._is_classification_series(s):
                probs = s.value_counts(normalize=True).to_numpy(dtype=float)
                min_share = float(np.min(probs))
                max_share = float(np.max(probs))
                norm_entropy = float(-np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0))) / max(np.log2(len(probs)), 1e-12)) if len(probs) > 1 else 0.0
                imbalance_ratio = round(max_share / max(min_share, 1e-12), 2)
                target_uncertainty.append(
                    {
                        "Target": t_col,
                        "Type": "Classification",
                        "Risk Metric": f"Imbalance Ratio = {imbalance_ratio}:1 | Minority Share = {min_share * 100:.1f}%",
                        "Normalized Entropy": round(norm_entropy, 3),
                        "Risk Level": "High" if imbalance_ratio >= 5.0 else ("Moderate" if imbalance_ratio >= 2.5 else "Low"),
                    }
                )
            else:
                vals = s.to_numpy(dtype=float)
                tail_p = max(0.0, min(0.4999, (1.0 - cfg.confidence_level) / 2.0))
                q_low = float(np.quantile(vals, tail_p))
                q_high = float(np.quantile(vals, 1.0 - tail_p))
                cv = float(np.std(vals, ddof=1) / max(abs(float(np.mean(vals))), 1e-12)) if len(vals) > 1 else 0.0
                target_uncertainty.append(
                    {
                        "Target": t_col,
                        "Type": "Regression",
                        "Risk Metric": f"{round(cfg.confidence_level * 100, 1):g}% Empirical Range [{q_low:.3g}, {q_high:.3g}] | Coeff. of Variation = {cv:.2f}",
                        "Normalized Entropy": None,
                        "Risk Level": "High" if cv >= 1.5 else ("Moderate" if cv >= 0.7 else "Low"),
                    }
                )

        return {
            "structural_risk_index": total_risk,
            "risk_band": risk_band,
            "confidence_level": cfg.confidence_level,
            "tail_alpha": cfg.risk_tail_alpha,
            "risk_components": risk_components,
            "tail_risk_table": tail_rows,
            "target_uncertainty": target_uncertainty,
        }
