"""Automatic Immediate Post-Ingestion Diagnostics, Statistical Pre-Tests & Defect Auditor for OmniForge."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from omniforge.core.logging import get_logger
from omniforge.core.performance import smart_sample_df
from omniforge.core.utils import to_json_object
from omniforge.data.format_audit import DataFormatAuditor
from omniforge.data.quality import DataQualityProfiler

logger = get_logger(__name__)


class InitialIntakeAuditor:
    """Runs an immediate, comprehensive intake health audit & statistical pre-test battery upon ingestion.

    Detects formatting anomalies, missingness patterns, non-finite values, duplicates, outliers,
    normality violations, severe skewness, high multicollinearity, class imbalance, and target-leakage
    risks right after data enters OmniForge, while generating a recommended auto-cleaning configuration.
    """

    def audit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str | None = None,
        max_sample_rows: int = 20_000,
    ) -> dict[str, Any]:
        """Execute full immediate intake audit and return a JSON-safe structured report."""
        if df is None or df.empty:
            return {
                "overall_health": "Empty",
                "quality_score": 0.0,
                "quality_grade": "F",
                "n_rows": 0,
                "n_cols": 0,
                "defect_counts": {"critical": 0, "high": 0, "medium": 0, "low": 0, "total": 0},
                "defects": [],
                "statistical_pretests": [],
                "outlier_summary": [],
                "correlation_alerts": [],
                "auto_clean_preset": {},
                "summary_bullets": ["No rows present in dataset."],
            }

        n_rows, n_cols = int(len(df)), int(df.shape[1])
        sample_df, was_sampled = smart_sample_df(df, max_rows=max_sample_rows, stratify_col=target_col if target_col in df.columns else None)

        # 1. Structural & Format Audit
        format_report = DataFormatAuditor().audit(df)

        # 2. 6D Quality Scorecard
        quality_report = DataQualityProfiler().profile(sample_df)

        defects: list[dict[str, Any]] = []

        # Promote format audit findings into unified defects list
        for issue in format_report.issues:
            sev = str(getattr(issue, "severity", "medium")).lower()
            if sev not in {"critical", "high", "medium", "low"}:
                sev = "medium"
            defects.append(
                {
                    "severity": sev,
                    "category": "Format & Schema",
                    "column": str(getattr(issue, "column", "*")),
                    "finding": str(getattr(issue, "issue_type", "Format issue")),
                    "metric": f"{getattr(issue, 'affected_rows', 0):,} rows ({getattr(issue, 'affected_pct', 0.0):.1f}%)",
                    "recommended_fix": str(getattr(issue, "recommendation", "Apply Auto-Clean type/string normalization.")),
                }
            )

        # 3. Duplicates Check
        dup_count = int(df.duplicated().sum()) if n_rows <= 250_000 else int(sample_df.duplicated().sum())
        if dup_count > 0:
            dup_pct = round(dup_count / max(n_rows, 1) * 100.0, 2)
            defects.append(
                {
                    "severity": "high" if dup_pct >= 5.0 else "medium",
                    "category": "Duplicates",
                    "column": "ALL_ROWS",
                    "finding": f"Exact duplicate rows detected ({dup_count:,} rows)",
                    "metric": f"{dup_pct:.2f}% duplicate rate",
                    "recommended_fix": "Enable 'Drop Exact Duplicates' in Leakage-Safe Cleaning (logged in Deleted Audit).",
                }
            )

        # 4. Missingness, Constant Columns & Non-Finite (+/-Inf) Scan
        high_missing_cols: list[str] = []
        any_missing = False
        any_inf = False
        constant_cols: list[str] = []
        for col in df.columns:
            ser = df[col]
            miss_n = int(ser.isna().sum())
            miss_pct = miss_n / max(n_rows, 1) * 100.0
            if miss_n > 0:
                any_missing = True
                if miss_pct >= 50.0:
                    high_missing_cols.append(str(col))
                    defects.append(
                        {
                            "severity": "critical" if miss_pct >= 80.0 else "high",
                            "category": "Missing Data",
                            "column": str(col),
                            "finding": f"Severe missingness ({miss_pct:.1f}% missing)",
                            "metric": f"{miss_n:,} / {n_rows:,} null",
                            "recommended_fix": "Drop column (>80% missing) or apply KNN/Iterative imputation + missing indicator.",
                        }
                    )
                elif miss_pct >= 5.0:
                    defects.append(
                        {
                            "severity": "medium",
                            "category": "Missing Data",
                            "column": str(col),
                            "finding": f"Moderate missing values ({miss_pct:.1f}%)",
                            "metric": f"{miss_n:,} null cells",
                            "recommended_fix": "Impute using Median/KNN (numeric) or Most Frequent (categorical) inside Split Lock.",
                        }
                    )

            n_uniq = int(ser.nunique(dropna=True))
            if n_uniq <= 1 and n_rows > 1:
                constant_cols.append(str(col))
                defects.append(
                    {
                        "severity": "medium",
                        "category": "Zero Variance",
                        "column": str(col),
                        "finding": "Constant / zero-variance column carries no predictive signal",
                        "metric": f"{n_uniq} unique value(s)",
                        "recommended_fix": "Exclude or drop constant column before modeling.",
                    }
                )

            if pd.api.types.is_numeric_dtype(ser):
                arr = pd.to_numeric(ser, errors="coerce").to_numpy(dtype=float)
                inf_n = int(np.isinf(arr).sum())
                if inf_n > 0:
                    any_inf = True
                    defects.append(
                        {
                            "severity": "critical",
                            "category": "Non-Finite Values",
                            "column": str(col),
                            "finding": f"Contains {inf_n:,} +/-Infinity values that break numerical models",
                            "metric": f"{inf_n:,} non-finite cells",
                            "recommended_fix": "Convert +/-Inf to NaN and apply median/winsorized imputation immediately.",
                        }
                    )

        # 5. Immediate Statistical Pre-Tests & Outlier Screening on Numeric Columns
        num_cols = list(sample_df.select_dtypes(include=[np.number]).columns)
        stat_tests: list[dict[str, Any]] = []
        outlier_summary: list[dict[str, Any]] = []
        any_heavy_outliers = False

        for col in num_cols[:25]:
            vals = pd.to_numeric(sample_df[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 8:
                continue
            q1, med, q3 = np.percentile(vals, [25.0, 50.0, 75.0])
            iqr = float(q3 - q1)
            iqr_outliers = int(np.sum((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr))) if iqr > 1e-12 else 0
            mad = float(np.median(np.abs(vals - med)))
            if mad > 1e-12:
                mod_z = 0.6745 * (vals - med) / mad
                mad_outliers = int(np.sum(np.abs(mod_z) > 3.5))
            else:
                mad_outliers = 0
            out_pct = round(iqr_outliers / max(len(vals), 1) * 100.0, 2)
            skew = float(stats.skew(vals)) if np.std(vals) > 1e-12 else 0.0
            kurt = float(stats.kurtosis(vals)) if np.std(vals) > 1e-12 else 0.0

            # Normality test (Shapiro <= 5000, D'Agostino-Pearson > 5000)
            test_name = "Shapiro-Wilk"
            try:
                if len(vals) <= 5000:
                    stat_v, p_val = stats.shapiro(vals[:5000])
                else:
                    test_name = "D'Agostino-Pearson"
                    stat_v, p_val = stats.normaltest(vals)
                stat_v, p_val = float(stat_v), float(p_val)
            except Exception:
                stat_v, p_val = 0.0, 1.0

            is_normal = bool(p_val >= 0.05)
            stat_tests.append(
                {
                    "column": str(col),
                    "test": test_name,
                    "statistic": round(stat_v, 4) if math.isfinite(stat_v) else 0.0,
                    "p_value": round(p_val, 6) if math.isfinite(p_val) else 1.0,
                    "normal_at_05": "Normal" if is_normal else "Non-Normal",
                    "skewness": round(skew, 3) if math.isfinite(skew) else 0.0,
                    "kurtosis": round(kurt, 3) if math.isfinite(kurt) else 0.0,
                    "iqr_outlier_pct": out_pct,
                }
            )
            outlier_summary.append(
                {
                    "column": str(col),
                    "iqr_outliers": iqr_outliers,
                    "iqr_pct": out_pct,
                    "mad_outliers": mad_outliers,
                    "min": round(float(np.min(vals)), 4),
                    "median": round(float(med), 4),
                    "max": round(float(np.max(vals)), 4),
                }
            )
            if out_pct >= 5.0:
                any_heavy_outliers = True
                defects.append(
                    {
                        "severity": "high" if out_pct >= 10.0 else "medium",
                        "category": "Outliers",
                        "column": str(col),
                        "finding": f"Heavy extreme values ({iqr_outliers:,} IQR outliers, skew={skew:.2f})",
                        "metric": f"{out_pct:.2f}% outside 1.5×IQR",
                        "recommended_fix": "Winsorize/Clip outliers or use Robust scaling / Studio 02 Anomaly Screening.",
                    }
                )
            elif abs(skew) >= 2.0:
                defects.append(
                    {
                        "severity": "low",
                        "category": "Distribution Skew",
                        "column": str(col),
                        "finding": f"Strongly skewed distribution (skew={skew:.2f}, {test_name} p={p_val:.4g})",
                        "metric": f"skew={skew:.2f}",
                        "recommended_fix": "Apply Yeo-Johnson/Log transform in Studio 05 or use non-parametric tests.",
                    }
                )

        # 6. High Multicollinearity & Potential Target Leakage Alerts
        corr_alerts: list[dict[str, Any]] = []
        if len(num_cols) >= 2:
            sub_num = sample_df[num_cols[:25]].replace([np.inf, -np.inf], np.nan)
            corr_mat = sub_num.corr().fillna(0.0)
            c_names = list(corr_mat.columns)
            for i in range(len(c_names)):
                for j in range(i + 1, len(c_names)):
                    r = float(corr_mat.iloc[i, j])
                    if abs(r) >= 0.85:
                        a, b = str(c_names[i]), str(c_names[j])
                        is_leak = target_col is not None and target_col in (a, b) and abs(r) >= 0.96
                        corr_alerts.append(
                            {
                                "feature_a": a,
                                "feature_b": b,
                                "pearson_r": round(r, 4),
                                "alert_type": "Potential Target Leakage" if is_leak else "High Multicollinearity",
                            }
                        )
                        defects.append(
                            {
                                "severity": "critical" if is_leak else "medium",
                                "category": "Leakage Risk" if is_leak else "Multicollinearity",
                                "column": f"{a} ↔ {b}",
                                "finding": (
                                    f"Near-deterministic correlation with target (|r|={abs(r):.3f}) — inspect for leakage!"
                                    if is_leak
                                    else f"Redundant collinear feature pair (r={r:.3f})"
                                ),
                                "metric": f"r = {r:.4f}",
                                "recommended_fix": (
                                    "Exclude leaky column via Global Column Exclusions before modeling."
                                    if is_leak
                                    else "Use regularized models (Ridge/Lasso) or VIF pruning in Studio 05."
                                ),
                            }
                        )

        # 7. Target Class Imbalance Screening
        if target_col and target_col in df.columns:
            t_ser = df[target_col].dropna()
            n_t_uniq = int(t_ser.nunique())
            if 2 <= n_t_uniq <= 20 and len(t_ser) > 0:
                vc = t_ser.value_counts(normalize=True)
                min_share = float(vc.min()) * 100.0
                max_share = float(vc.max()) * 100.0
                if min_share < 10.0:
                    defects.append(
                        {
                            "severity": "high" if min_share < 3.0 else "medium",
                            "category": "Class Imbalance",
                            "column": str(target_col),
                            "finding": f"Imbalanced target classes (minority class = {min_share:.2f}%, majority = {max_share:.2f}%)",
                            "metric": f"{n_t_uniq} classes · min {min_share:.1f}%",
                            "recommended_fix": "Use Stratified Split Lock and optimize F1-Weighted / ROC-AUC instead of raw Accuracy.",
                        }
                    )

        # Sort defects by severity priority
        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        defects.sort(key=lambda d: (sev_order.get(d["severity"], 4), d["category"], d["column"]))
        counts = {
            "critical": sum(1 for d in defects if d["severity"] == "critical"),
            "high": sum(1 for d in defects if d["severity"] == "high"),
            "medium": sum(1 for d in defects if d["severity"] == "medium"),
            "low": sum(1 for d in defects if d["severity"] == "low"),
            "total": len(defects),
        }

        if counts["critical"] > 0:
            health = "Critical Issues Detected"
        elif counts["high"] > 0:
            health = "Action Recommended"
        elif counts["medium"] > 0:
            health = "Minor Issues Detected"
        else:
            health = "Clean & Analysis-Ready"

        auto_clean_preset = {
            "mode": "auto",
            "standardize_missing_tokens": True,
            "clean_strings": True,
            "strip_html": True,
            "infer_types": True,
            "drop_exact_duplicates": dup_count > 0,
            "drop_high_missing_cols": len(high_missing_cols) > 0,
            "missing_col_threshold": 0.80,
            "numeric_imputation": "median" if (any_missing or any_inf) else "median",
            "categorical_imputation": "most_frequent",
            "outlier_method": "iqr" if any_heavy_outliers else "none",
            "outlier_action": "clip" if any_heavy_outliers else "none",
            "outlier_threshold": 1.5,
            "drop_constant_columns": len(constant_cols) > 0,
        }

        summary_bullets = [
            f"Quality Score: {quality_report.overall_score:.1f}/100 (Grade {quality_report.grade}) across {n_rows:,} rows × {n_cols} columns.",
            f"Detected {counts['total']} diagnostic findings: {counts['critical']} critical, {counts['high']} high, {counts['medium']} medium, {counts['low']} low.",
            f"Duplicates: {dup_count:,} | Constant cols: {len(constant_cols)} | High-correlation alerts: {len(corr_alerts)}.",
        ]
        if was_sampled:
            summary_bullets.append(f"Statistical pre-tests computed on stratified sample of {len(sample_df):,} rows for instant intake speed.")

        logger.info(
            "initial_intake_audit_completed",
            rows=n_rows,
            cols=n_cols,
            health=health,
            total_defects=counts["total"],
            critical=counts["critical"],
        )

        return to_json_object(
            {
                "overall_health": health,
                "quality_score": round(float(quality_report.overall_score), 2),
                "quality_grade": str(quality_report.grade),
                "dimensions": quality_report.dimensions,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "duplicate_rows": dup_count,
                "constant_columns": constant_cols,
                "high_missing_columns": high_missing_cols,
                "format_health": format_report.health_status,
                "defect_counts": counts,
                "defects": defects[:120],
                "statistical_pretests": stat_tests,
                "outlier_summary": outlier_summary,
                "correlation_alerts": corr_alerts[:40],
                "auto_clean_preset": auto_clean_preset,
                "summary_bullets": summary_bullets,
            }
        )
