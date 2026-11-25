"""Multi-dimensional enterprise Data Quality Profiler (6 canonical quality dimensions)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from omniforge.core.exceptions import DataValidationError
from omniforge.core.logging import get_logger
from omniforge.core.performance import GLOBAL_DF_CACHE, compute_df_fingerprint

logger = get_logger(__name__)


class ColumnQualityScore(BaseModel):
    """Granular quality metrics for an individual column."""

    column: str
    dtype: str
    completeness: float
    uniqueness: float
    validity: float
    consistency: float
    outlier_ratio: float
    overall_score: float
    reference_accuracy: float | None = None
    reference_comparisons: int = 0
    rule_conformance: float | None = None
    rule_checks: int = 0
    statistical_plausibility: float | None = None
    issues: list[str] = Field(default_factory=list)


class DataQualityReport(BaseModel):
    """Comprehensive six-dimensional Data Quality scorecard and remediation plan."""

    overall_score: float = Field(ge=0.0, le=100.0)
    grade: str
    dimensions: dict[str, float | None]
    row_count: int
    column_count: int
    duplicate_row_ratio: float
    missing_cell_ratio: float
    column_scores: list[ColumnQualityScore] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    effective_weights: dict[str, float] = Field(default_factory=dict)
    reference_comparisons: int = 0
    rule_conformance: float | None = None
    rule_checks: int = 0
    profiled_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DataQualityProfiler:
    """Computes completeness, uniqueness, validity, consistency, accuracy, and timeliness scores."""

    DEFAULT_WEIGHTS: dict[str, float] = {
        "completeness": 0.25,
        "uniqueness": 0.15,
        "validity": 0.20,
        "consistency": 0.15,
        "accuracy": 0.15,
        "timeliness": 0.10,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        raw = dict(weights or self.DEFAULT_WEIGHTS)
        total = sum(raw.values()) or 1.0
        self.weights = {k: v / total for k, v in raw.items()}

    def profile(
        self,
        df: pd.DataFrame,
        *,
        timestamp_column: str | None = None,
        max_freshness_days: float = 365.0,
        reference_rules: dict[str, tuple[float, float]] | None = None,
        reference_data: pd.DataFrame | None = None,
        key_columns: list[str] | None = None,
    ) -> DataQualityReport:
        """Profile a DataFrame across all six enterprise data quality dimensions.

        Args:
            df: Input DataFrame.
            timestamp_column: Optional datetime column used to score timeliness.
            max_freshness_days: Age threshold beyond which timeliness degrades.
            reference_rules: Optional inclusive numeric ranges for rule conformance.
            reference_data: Trusted values keyed by matching index and column labels.
                Only pairs with two observed values enter exact-match reference accuracy.
            key_columns: Optional primary key columns whose uniqueness is scored strictly.

        Returns:
            :class:`DataQualityReport` with scores on a 0–100 scale and recommendations.
        """
        import json

        if reference_data is not None and (
            not df.index.is_unique or not reference_data.index.is_unique
            or not df.columns.is_unique or not reference_data.columns.is_unique
        ):
            raise DataValidationError(
                "Reference accuracy requires unique row identifiers and column labels.",
                suggestions=["Index both frames by the same unique row identifier."],
            )
        cache_params = json.dumps({
            "timestamp_column": timestamp_column,
            "max_freshness_days": max_freshness_days,
            "rules": reference_rules,
            "reference": compute_df_fingerprint(reference_data) if reference_data is not None else None,
            "key_columns": key_columns,
            "weights": self.weights,
        }, sort_keys=True)
        hit, cached = GLOBAL_DF_CACHE.get("data_quality", df, cache_params)
        if hit and isinstance(cached, DataQualityReport):
            return cached

        n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
        if n_rows == 0 or n_cols == 0:
            return DataQualityReport(
                overall_score=0.0,
                grade="F",
                dimensions={k: None for k in self.DEFAULT_WEIGHTS},
                row_count=n_rows,
                column_count=n_cols,
                duplicate_row_ratio=0.0,
                missing_cell_ratio=1.0,
                recommendations=["Dataset is empty; verify upstream ingestion pipeline."],
            )

        missing_cell_ratio = float(df.isna().sum().sum()) / float(n_rows * n_cols)
        completeness = max(0.0, (1.0 - missing_cell_ratio) * 100.0)

        try:
            dup_ratio = float(df.duplicated(subset=key_columns).mean())
        except TypeError:
            dup_ratio = float(df.astype(str).duplicated(subset=key_columns).mean())
        uniqueness = max(0.0, (1.0 - dup_ratio) * 100.0)

        col_scores: list[ColumnQualityScore] = []
        validity_parts: list[float] = []
        consistency_parts: list[float] = []
        reference_matches = reference_comparisons = 0
        rule_passed = rule_checks = 0
        outlier_columns: list[str] = []
        recommendations: list[str] = []

        rules = reference_rules or {}

        for col in df.columns:
            series = df[col]
            col_name = str(col)
            col_issues: list[str] = []

            col_comp = float(series.notna().mean()) * 100.0
            if col_comp < 90.0:
                col_issues.append(f"High missing rate ({100.0 - col_comp:.1f}%).")

            try:
                col_uniq = float(series.nunique(dropna=True)) / max(int(series.notna().sum()), 1) * 100.0
            except TypeError:
                col_uniq = 100.0

            # Validity: finite numerics, no placeholder tokens in strings, no NaT
            col_valid = 100.0
            outlier_ratio = 0.0
            col_consistency = 100.0
            col_accuracy: float | None = None
            col_plausibility: float | None = None
            col_conformance: float | None = None
            col_comparisons = col_rule_checks = 0

            if reference_data is not None and col in reference_data:
                trusted = reference_data[col].reindex(df.index)
                comparable = series.notna() & trusted.notna()
                col_comparisons = int(comparable.sum())
                if col_comparisons:
                    matches = int(series[comparable].astype(object).eq(trusted[comparable].astype(object)).sum())
                    col_accuracy = matches / col_comparisons * 100.0
                    reference_matches += matches
                    reference_comparisons += col_comparisons

            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                vals = series.dropna().astype(float)
                if len(vals) > 0:
                    inf_ratio = float(np.isinf(vals).mean())
                    col_valid = max(0.0, (1.0 - inf_ratio) * 100.0)
                    if inf_ratio > 0:
                        col_issues.append(f"Contains infinite values ({inf_ratio*100:.1f}%).")

                    finite = vals[np.isfinite(vals)]
                    if len(finite) >= 4:
                        q1, q3 = float(finite.quantile(0.25)), float(finite.quantile(0.75))
                        iqr = q3 - q1
                        if np.isfinite(iqr) and iqr > 0:
                            out_mask = (finite < q1 - 3.0 * iqr) | (finite > q3 + 3.0 * iqr)
                            outlier_ratio = float(out_mask.mean())
                            col_plausibility = (1.0 - outlier_ratio) * 100.0
                            if outlier_ratio > 0.05:
                                outlier_columns.append(col_name)
                                col_issues.append(f"Extreme outlier density ({outlier_ratio*100:.1f}%).")

                    if col_name in rules:
                        low, high = rules[col_name]
                        passed = int((np.isfinite(vals) & vals.between(low, high)).sum())
                        col_rule_checks = len(vals)
                        col_conformance = passed / col_rule_checks * 100.0
                        rule_passed += passed
                        rule_checks += col_rule_checks
                        if col_conformance < 100.0:
                            col_issues.append(f"Domain range [{low}, {high}] violated by {100-col_conformance:.1f}% of values.")
            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 25_000:
                    non_null = non_null.sample(n=25_000, random_state=42)
                non_null = non_null.astype(str)
                if len(non_null) > 0:
                    # Consistency checks case/whitespace fragmentation in categoricals
                    raw_unique = non_null.nunique()
                    norm_unique = non_null.str.strip().str.lower().nunique()
                    if raw_unique > 0 and norm_unique < raw_unique:
                        frag = (raw_unique - norm_unique) / raw_unique
                        col_consistency = max(0.0, (1.0 - frag) * 100.0)
                        col_issues.append("Inconsistent casing or trailing whitespace across categories.")
                    bad_tokens = non_null.str.strip().str.lower().isin({"na", "n/a", "null", "none", "?", ""}).mean()
                    col_valid = max(0.0, (1.0 - float(bad_tokens)) * 100.0)
                    if bad_tokens > 0:
                        col_issues.append(f"Contains unparsed null placeholder strings ({bad_tokens*100:.1f}%).")

            validity_parts.append(col_valid)
            consistency_parts.append(col_consistency)
            col_total = 0.35 * col_comp + 0.25 * col_valid + 0.20 * col_consistency
            col_overall = round(
                col_total / 0.80 if col_accuracy is None else col_total + 0.20 * col_accuracy, 2
            )
            col_scores.append(
                ColumnQualityScore(
                    column=col_name,
                    dtype=str(series.dtype),
                    completeness=round(col_comp, 2),
                    uniqueness=round(col_uniq, 2),
                    validity=round(col_valid, 2),
                    consistency=round(col_consistency, 2),
                    outlier_ratio=round(outlier_ratio, 4),
                    overall_score=col_overall,
                    reference_accuracy=round(col_accuracy, 2) if col_accuracy is not None else None,
                    reference_comparisons=col_comparisons,
                    rule_conformance=round(col_conformance, 2) if col_conformance is not None else None,
                    rule_checks=col_rule_checks,
                    statistical_plausibility=round(col_plausibility, 2) if col_plausibility is not None else None,
                    issues=col_issues,
                )
            )

        validity = float(np.mean(validity_parts)) if validity_parts else 100.0
        consistency = float(np.mean(consistency_parts)) if consistency_parts else 100.0
        accuracy = reference_matches / reference_comparisons * 100.0 if reference_comparisons else None
        timeliness = self._compute_timeliness(df, timestamp_column, max_freshness_days)

        dimensions: dict[str, Any] = {
            "completeness": round(completeness, 2),
            "uniqueness": round(uniqueness, 2),
            "validity": round(validity, 2),
            "consistency": round(consistency, 2),
            "accuracy": round(accuracy, 2) if accuracy is not None else None,
            "timeliness": round(timeliness, 2) if timeliness is not None else None,
        }
        evaluable_weights = {
            k: w for k, w in self.weights.items() if dimensions.get(k) is not None
        }
        total_w = sum(evaluable_weights.values()) or 1.0
        effective_weights = {k: w / total_w for k, w in evaluable_weights.items()}
        overall = round(
            sum(float(dimensions[k]) * w for k, w in effective_weights.items()),
            2,
        )
        grade = self._score_to_grade(overall)

        if completeness < 95.0:
            recommendations.append(
                f"Completeness is {completeness:.1f}%. Configure DataCleaner imputation and missing indicators."
            )
        if dup_ratio > 0.01:
            recommendations.append(
                f"Detected {dup_ratio*100:.2f}% duplicate rows. Enable exact or fuzzy deduplication."
            )
        if consistency < 95.0:
            recommendations.append(
                "String casing/whitespace inconsistencies found. Enable string cleaning and normalization."
            )
        if outlier_columns:
            recommendations.append("Review extreme observations in the columns flagged for statistical plausibility.")
        if rule_passed < rule_checks:
            recommendations.append("Review observed values that violate configured domain ranges.")
        if accuracy is not None and accuracy < 100.0:
            recommendations.append("Investigate disagreements with the trusted reference data.")
        if timeliness is not None and timeliness < 85.0:
            recommendations.append(
                "Dataset freshness is below threshold. Schedule incremental ingestion or check pipeline SLA."
            )
        if not recommendations:
            recommendations.append("No remediation was suggested by the checks that could be evaluated.")

        report = DataQualityReport(
            overall_score=overall,
            grade=grade,
            dimensions=dimensions,
            row_count=n_rows,
            column_count=n_cols,
            duplicate_row_ratio=round(dup_ratio, 4),
            missing_cell_ratio=round(missing_cell_ratio, 4),
            column_scores=col_scores,
            effective_weights=effective_weights,
            reference_comparisons=reference_comparisons,
            rule_conformance=round(rule_passed / rule_checks * 100.0, 2) if rule_checks else None,
            rule_checks=rule_checks,
            recommendations=recommendations,
        )
        logger.info("data_quality_profiled", overall_score=overall, grade=grade)
        GLOBAL_DF_CACHE.set("data_quality", df, report, cache_params)
        return report

    @staticmethod
    def _compute_timeliness(
        df: pd.DataFrame, timestamp_column: str | None, max_freshness_days: float
    ) -> float | None:
        ts_col = timestamp_column
        if ts_col is None:
            dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
            ts_col = str(dt_cols[0]) if dt_cols else None
        if ts_col is None or ts_col not in df.columns:
            return None
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dropna()
        if len(ts) == 0:
            return None
        latest = pd.Timestamp(ts.max()).to_pydatetime()
        now = datetime.now(UTC)
        age_days = max(0.0, (now - latest).total_seconds() / 86400.0)
        if age_days <= max_freshness_days * 0.25:
            return 100.0
        decay = max(0.0, 1.0 - age_days / max(max_freshness_days * 2.0, 1.0))
        return float(round(decay * 100.0, 2))

    @staticmethod
    def _score_to_grade(score: float) -> str:
        if score >= 95.0:
            return "A+"
        if score >= 90.0:
            return "A"
        if score >= 80.0:
            return "B"
        if score >= 70.0:
            return "C"
        if score >= 60.0:
            return "D"
        return "F"

    @staticmethod
    def little_mcar_test(
        df: pd.DataFrame,
        alpha: float = 0.05,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> dict[str, Any]:
        """Little's (1988, JASA 83:1198-1202) MCAR test via Multivariate Normal EM MLE.

        Includes both partially missing and fully observed numeric covariates so MAR
        mechanisms dependent on fully observed predictors are properly detected.
        """
        from scipy import stats

        num_df = df.select_dtypes(include=[np.number]).copy()
        # Exclude boolean columns, completely missing columns, and zero-variance columns
        keep_cols = [
            c
            for c in num_df.columns
            if not pd.api.types.is_bool_dtype(num_df[c])
            and not num_df[c].isna().all()
            and int(num_df[c].dropna().nunique()) > 1
        ]
        num_df = num_df[keep_cols].replace([np.inf, -np.inf], np.nan)
        has_any_missing = bool(num_df.isna().any().any()) if num_df.shape[1] > 0 else False
        # Drop rows where ALL numeric variables are missing (p_j = 0 contributes nothing to likelihood)
        if num_df.shape[1] > 0:
            num_df = num_df.loc[~num_df.isna().all(axis=1)]

        if num_df.shape[1] < 2 or len(num_df) < 10 or not has_any_missing:
            return {
                "status": "not_applicable",
                "reason": "Need at least 2 non-constant numeric columns, >= 10 non-empty rows, and at least 1 missing value.",
                "statistic": None,
                "df": None,
                "p_value": None,
                "is_mcar": None,
            }

        X_mat = num_df.to_numpy(dtype=float)
        obs_mat = ~np.isnan(X_mat)
        n_rows, p = X_mat.shape

        # Group rows by missingness pattern once
        pattern_groups: dict[tuple[bool, ...], np.ndarray] = {}
        for row_idx in range(n_rows):
            pat = tuple(bool(v) for v in obs_mat[row_idx])
            if pat not in pattern_groups:
                pattern_groups[pat] = []  # type: ignore[assignment]
            pattern_groups[pat].append(row_idx)  # type: ignore[attr-defined]
        pattern_indices = {pat: np.asarray(idxs, dtype=int) for pat, idxs in pattern_groups.items()}

        if len(pattern_indices) <= 1:
            return {
                "status": "not_applicable",
                "reason": "Insufficient distinct missingness patterns for Little's MCAR test.",
                "statistic": None,
                "df": 0,
                "p_value": None,
                "is_mcar": None,
            }

        # Initialize EM from observed column means & diagonal/sample covariance
        mu = np.nanmean(X_mat, axis=0)
        if np.any(~np.isfinite(mu)):
            return {
                "status": "not_applicable",
                "reason": "Insufficient observations to initialize column means.",
                "statistic": None,
                "df": None,
                "p_value": None,
                "is_mcar": None,
            }
        X_init = np.where(obs_mat, X_mat, mu)
        centered_init = X_init - mu
        sigma = (centered_init.T @ centered_init) / max(n_rows, 1)
        diag_scale = max(float(np.mean(np.diag(sigma))), 1e-6)
        jitter = 1e-8 * diag_scale * np.eye(p)
        sigma = sigma + jitter

        em_converged = False
        em_iters = 0
        for it in range(1, max(1, max_iter) + 1):
            em_iters = it
            X_expected = X_mat.copy()
            cov_cond_sum = np.zeros((p, p), dtype=float)

            for pat, idxs in pattern_indices.items():
                obs_mask = np.asarray(pat, dtype=bool)
                miss_mask = ~obs_mask
                if not np.any(miss_mask):
                    continue
                if not np.any(obs_mask):
                    X_expected[np.ix_(idxs, miss_mask)] = mu[miss_mask]
                    cov_cond_sum[np.ix_(miss_mask, miss_mask)] += len(idxs) * sigma[np.ix_(miss_mask, miss_mask)]
                    continue

                sigma_oo = sigma[np.ix_(obs_mask, obs_mask)]
                sigma_mo = sigma[np.ix_(miss_mask, obs_mask)]
                sigma_mm = sigma[np.ix_(miss_mask, miss_mask)]
                sigma_oo_inv = np.linalg.pinv(sigma_oo, rcond=1e-10)
                beta_mo = sigma_mo @ sigma_oo_inv  # (p_miss, p_obs)

                diff_obs = X_mat[np.ix_(idxs, obs_mask)] - mu[obs_mask]
                X_expected[np.ix_(idxs, miss_mask)] = mu[miss_mask] + diff_obs @ beta_mo.T

                cond_cov = sigma_mm - beta_mo @ sigma_mo.T
                cond_cov = 0.5 * (cond_cov + cond_cov.T)
                cov_cond_sum[np.ix_(miss_mask, miss_mask)] += len(idxs) * cond_cov

            mu_next = X_expected.mean(axis=0)
            centered = X_expected - mu_next
            sigma_next = ((centered.T @ centered) + cov_cond_sum) / float(n_rows)
            sigma_next = 0.5 * (sigma_next + sigma_next.T) + jitter

            denom = max(float(np.linalg.norm(mu)) + float(np.linalg.norm(sigma, ord="fro")), 1e-12)
            delta = (float(np.linalg.norm(mu_next - mu)) + float(np.linalg.norm(sigma_next - sigma, ord="fro"))) / denom
            mu, sigma = mu_next, sigma_next
            if delta < tol:
                em_converged = True
                break

        # Little (1988, p. 1199, Eq. 2): finite-sample bias-adjusted ML covariance tilde{Sigma} = (N / (N - 1)) * Sigma_ML
        sigma_tilde = sigma * (float(n_rows) / float(max(n_rows - 1, 1)))

        chi2_stat = 0.0
        total_obs_dims = 0
        n_patterns = 0
        for pat, idxs in pattern_indices.items():
            obs_mask = np.asarray(pat, dtype=bool)
            p_j = int(obs_mask.sum())
            if p_j == 0 or len(idxs) == 0:
                continue
            n_patterns += 1
            total_obs_dims += p_j
            y_bar_j = X_mat[np.ix_(idxs, obs_mask)].mean(axis=0)
            diff = y_bar_j - mu[obs_mask]
            cov_j = sigma_tilde[np.ix_(obs_mask, obs_mask)]
            cov_inv = np.linalg.pinv(cov_j, rcond=1e-10)
            quad = float(len(idxs) * (diff @ cov_inv @ diff))
            chi2_stat += max(0.0, quad)

        dof = total_obs_dims - p
        if dof <= 0 or n_patterns <= 1:
            return {
                "status": "not_applicable",
                "reason": "Insufficient distinct missingness patterns for Little's MCAR test.",
                "statistic": round(chi2_stat, 6),
                "df": int(max(0, dof)),
                "p_value": None,
                "is_mcar": None,
            }

        p_val = float(stats.chi2.sf(chi2_stat, df=dof))
        return {
            "status": "completed",
            "statistic": round(chi2_stat, 6),
            "df": int(dof),
            "p_value": round(p_val, 6),
            "alpha": float(alpha),
            "is_mcar": bool(p_val >= alpha),
            "n_patterns": int(n_patterns),
            "estimator": "mvn_em_mle",
            "em_iterations": int(em_iters),
            "em_converged": bool(em_converged),
        }

