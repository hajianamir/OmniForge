"""Great Expectations & Pandera-style declarative data validation engine for OmniForge."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from omniforge.core.exceptions import DataValidationError
from omniforge.core.logging import get_logger
from omniforge.core.types import ValidationSeverity

logger = get_logger(__name__)


class ExpectationResult(BaseModel):
    """Outcome of evaluating a single data expectation rule."""

    expectation_name: str
    column: str | None = None
    passed: bool
    severity: ValidationSeverity = ValidationSeverity.ERROR
    observed_value: Any = None
    expected_value: Any = None
    unexpected_count: int = 0
    unexpected_ratio: float = 0.0
    sample_failing_indices: list[int] = Field(default_factory=list)
    message: str = ""
    remediation: str = ""


class ValidationReport(BaseModel):
    """Aggregated validation report across an entire expectation suite."""

    suite_name: str
    passed: bool
    total_expectations: int
    passed_count: int
    failed_count: int
    pass_rate: float
    critical_failures: int = 0
    error_failures: int = 0
    warning_failures: int = 0
    results: list[ExpectationResult] = Field(default_factory=list)
    remediation_summary: list[str] = Field(default_factory=list)
    validated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def passed_checks(self) -> int:
        """Backward-compatible alias for :attr:`passed_count`."""
        return self.passed_count

    @property
    def total_checks(self) -> int:
        """Backward-compatible alias for :attr:`total_expectations`."""
        return self.total_expectations

    @property
    def failed_checks(self) -> int:
        """Backward-compatible alias for :attr:`failed_count`."""
        return self.failed_count


class Expectation(BaseModel):
    """Serializable declaration of a Great Expectations / Pandera validation rule."""

    expectation_type: str
    column: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)
    severity: ValidationSeverity = ValidationSeverity.ERROR
    description: str = ""


class ValidationRule(BaseModel):
    """Concise shorthand validation rule convertible into a Great Expectations :class:`Expectation`."""

    name: str
    rule_type: str  # range, not_null, in_set, unique, regex
    column: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    severity: ValidationSeverity = ValidationSeverity.ERROR

    def to_expectation(self) -> Expectation:
        """Convert this concise rule into a canonical :class:`Expectation`."""
        rt = self.rule_type.lower()
        if rt == "range":
            kw: dict[str, Any] = {}
            if "min" in self.params:
                kw["min_value"] = self.params["min"]
            if "max" in self.params:
                kw["max_value"] = self.params["max"]
            return Expectation(
                expectation_type="expect_column_values_to_be_between",
                column=self.column,
                kwargs=kw,
                severity=self.severity,
                description=self.name,
            )
        if rt == "not_null":
            return Expectation(
                expectation_type="expect_column_values_to_not_be_null",
                column=self.column,
                kwargs=dict(self.params),
                severity=self.severity,
                description=self.name,
            )
        if rt == "in_set":
            vals = self.params.get("values", self.params.get("value_set", []))
            return Expectation(
                expectation_type="expect_column_values_to_be_in_set",
                column=self.column,
                kwargs={"value_set": vals},
                severity=self.severity,
                description=self.name,
            )
        if rt == "unique":
            return Expectation(
                expectation_type="expect_column_values_to_be_unique",
                column=self.column,
                kwargs={},
                severity=self.severity,
                description=self.name,
            )
        return Expectation(
            expectation_type=self.rule_type,
            column=self.column,
            kwargs=dict(self.params),
            severity=self.severity,
            description=self.name,
        )


class DataValidator:
    """Declarative expectation suite, schema checker, auto-suite generator, and statistical validator."""

    def __init__(
        self,
        rules_or_name: str | list[Expectation | ValidationRule] = "uap_validation_suite",
        *,
        suite_name: str = "uap_validation_suite",
    ) -> None:
        if isinstance(rules_or_name, str):
            self.suite_name = rules_or_name
            self.expectations: list[Expectation] = []
        else:
            self.suite_name = suite_name
            self.expectations = [
                r.to_expectation() if isinstance(r, ValidationRule) else r for r in rules_or_name
            ]
        self._custom_validators: dict[str, Callable[[pd.DataFrame, dict[str, Any]], ExpectationResult]] = {}

    def add_expectation(
        self,
        expectation_type: str,
        column: str | None = None,
        *,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        description: str = "",
        **kwargs: Any,
    ) -> DataValidator:
        """Append a built-in or custom expectation to the suite (fluent builder)."""
        self.expectations.append(
            Expectation(
                expectation_type=expectation_type,
                column=column,
                kwargs=kwargs,
                severity=severity,
                description=description,
            )
        )
        return self

    def register_custom_validator(
        self, name: str, fn: Callable[[pd.DataFrame, dict[str, Any]], ExpectationResult]
    ) -> None:
        """Register a user-defined custom validation function."""
        self._custom_validators[name] = fn

    @classmethod
    def auto_generate_suite(
        cls,
        df: pd.DataFrame,
        suite_name: str = "auto_generated_suite",
        *,
        null_tolerance: float = 0.05,
        range_margin_pct: float = 0.15,
    ) -> DataValidator:
        """Infer a baseline Great Expectations-style validation suite from a reference DataFrame."""
        validator = cls(suite_name=suite_name)
        validator.add_expectation(
            "expect_table_row_count_to_be_between",
            min_value=max(1, int(len(df) * 0.1)) if len(df) > 0 else 0,
            severity=ValidationSeverity.ERROR,
        )
        validator.add_expectation(
            "expect_table_columns_to_match_set",
            column_set=[str(c) for c in df.columns],
            severity=ValidationSeverity.CRITICAL,
        )

        for col in df.columns:
            series = df[col]
            col_name = str(col)
            validator.add_expectation("expect_column_to_exist", column=col_name, severity=ValidationSeverity.CRITICAL)

            null_ratio = float(series.isna().mean()) if len(df) > 0 else 0.0
            if null_ratio == 0.0:
                validator.add_expectation(
                    "expect_column_values_to_not_be_null",
                    column=col_name,
                    mostly=max(0.0, 1.0 - null_tolerance),
                    severity=ValidationSeverity.ERROR,
                )
            else:
                validator.add_expectation(
                    "expect_column_null_ratio_to_be_below",
                    column=col_name,
                    max_null_ratio=min(1.0, null_ratio + null_tolerance),
                    severity=ValidationSeverity.WARNING,
                )

            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                non_null = series.dropna().astype(float)
                if len(non_null) > 0:
                    cmin, cmax = float(non_null.min()), float(non_null.max())
                    span = max(cmax - cmin, abs(cmax) * 0.1, 1.0)
                    validator.add_expectation(
                        "expect_column_values_to_be_between",
                        column=col_name,
                        min_value=cmin - span * range_margin_pct,
                        max_value=cmax + span * range_margin_pct,
                        severity=ValidationSeverity.WARNING,
                    )
            elif series.nunique(dropna=True) <= 20 and len(series.dropna()) > 0:
                allowed = [str(v) for v in series.dropna().unique().tolist()]
                validator.add_expectation(
                    "expect_column_values_to_be_in_set",
                    column=col_name,
                    value_set=allowed,
                    severity=ValidationSeverity.WARNING,
                )

        return validator

    def validate(self, df: pd.DataFrame, *, raise_on_failure: bool = False) -> ValidationReport:
        """Evaluate every configured expectation against ``df``."""
        results: list[ExpectationResult] = []
        for exp in self.expectations:
            results.append(self._evaluate_one(df, exp))

        passed_count = sum(1 for r in results if r.passed)
        failed = [r for r in results if not r.passed]
        critical_f = sum(1 for r in failed if r.severity == ValidationSeverity.CRITICAL)
        error_f = sum(1 for r in failed if r.severity == ValidationSeverity.ERROR)
        warning_f = sum(1 for r in failed if r.severity == ValidationSeverity.WARNING)
        overall_passed = critical_f == 0 and error_f == 0
        total = len(results)

        remediations = [r.remediation for r in failed if r.remediation]
        report = ValidationReport(
            suite_name=self.suite_name,
            passed=overall_passed,
            total_expectations=total,
            passed_count=passed_count,
            failed_count=len(failed),
            pass_rate=round(passed_count / max(total, 1), 4),
            critical_failures=critical_f,
            error_failures=error_f,
            warning_failures=warning_f,
            results=results,
            remediation_summary=remediations,
        )

        logger.info(
            "data_validated",
            suite=self.suite_name,
            passed=overall_passed,
            pass_rate=report.pass_rate,
            failures=len(failed),
        )
        if raise_on_failure and not overall_passed:
            raise DataValidationError(
                f"Validation suite '{self.suite_name}' failed with {critical_f} critical and {error_f} error failures.",
                suggestions=remediations[:5],
                context={"pass_rate": report.pass_rate, "failed_count": len(failed)},
            )
        return report

    def _evaluate_one(self, df: pd.DataFrame, exp: Expectation) -> ExpectationResult:
        etype = exp.expectation_type
        col = exp.column
        kw = exp.kwargs
        sev = exp.severity

        if etype in self._custom_validators:
            return self._custom_validators[etype](df, kw)

        if etype == "expect_table_row_count_to_be_between":
            n = len(df)
            min_v = kw.get("min_value", 0)
            max_v = kw.get("max_value", float("inf"))
            ok = min_v <= n <= max_v
            return ExpectationResult(
                expectation_name=etype,
                passed=ok,
                severity=sev,
                observed_value=n,
                expected_value={"min": min_v, "max": max_v},
                message=f"Row count {n} within [{min_v}, {max_v}]" if ok else f"Row count {n} outside [{min_v}, {max_v}]",
                remediation="" if ok else "Check upstream ingestion filters or pipeline completeness.",
            )

        if etype == "expect_table_columns_to_match_set":
            expected_set = set(kw.get("column_set", []))
            actual_set = {str(c) for c in df.columns}
            missing = sorted(expected_set - actual_set)
            extra = sorted(actual_set - expected_set)
            ok = len(missing) == 0 and (kw.get("allow_extra", True) or len(extra) == 0)
            return ExpectationResult(
                expectation_name=etype,
                passed=ok,
                severity=sev,
                observed_value=sorted(actual_set),
                expected_value=sorted(expected_set),
                message="Schema columns match" if ok else f"Missing columns: {missing}; Extra columns: {extra}",
                remediation="" if ok else f"Align schema with upstream producer. Missing: {missing}",
            )

        if etype == "expect_column_to_exist":
            ok = col is not None and col in df.columns
            return ExpectationResult(
                expectation_name=etype,
                column=col,
                passed=ok,
                severity=sev,
                observed_value=ok,
                expected_value=True,
                message=f"Column '{col}' exists" if ok else f"Required column '{col}' is missing",
                remediation="" if ok else f"Add or rename column '{col}' during ingestion.",
            )

        if col is None or col not in df.columns:
            return ExpectationResult(
                expectation_name=etype,
                column=col,
                passed=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Column '{col}' not found for expectation '{etype}'.",
                remediation=f"Ensure column '{col}' exists before running '{etype}'.",
            )

        series = df[col]
        n_rows = max(len(series), 1)

        if etype == "expect_column_values_to_not_be_null":
            mostly = float(kw.get("mostly", 1.0))
            failing_mask = series.isna()
            fail_cnt = int(failing_mask.sum())
            non_null_ratio = 1.0 - fail_cnt / n_rows
            ok = non_null_ratio >= mostly
            return ExpectationResult(
                expectation_name=etype,
                column=col,
                passed=ok,
                severity=sev,
                observed_value=round(non_null_ratio, 4),
                expected_value=mostly,
                unexpected_count=fail_cnt,
                unexpected_ratio=round(fail_cnt / n_rows, 4),
                sample_failing_indices=df.index[failing_mask].tolist()[:10],
                message=f"Non-null ratio {non_null_ratio:.4f} >= {mostly}" if ok else f"Found {fail_cnt} nulls in '{col}'",
                remediation="" if ok else f"Impute or drop missing values in '{col}' using DataCleaner.",
            )

        if etype == "expect_column_null_ratio_to_be_below":
            max_ratio = float(kw.get("max_null_ratio", 0.1))
            null_ratio = float(series.isna().mean()) if len(series) > 0 else 0.0
            ok = null_ratio <= max_ratio
            return ExpectationResult(
                expectation_name=etype,
                column=col,
                passed=ok,
                severity=sev,
                observed_value=round(null_ratio, 4),
                expected_value=max_ratio,
                unexpected_count=int(series.isna().sum()),
                unexpected_ratio=round(null_ratio, 4),
                message=f"Null ratio {null_ratio:.4f} <= {max_ratio}" if ok else f"Null ratio {null_ratio:.4f} exceeds {max_ratio}",
                remediation="" if ok else f"Investigate missing data surge in '{col}' or apply imputation.",
            )

        if etype == "expect_column_values_to_be_unique":
            dup_mask = series.duplicated(keep=False) & series.notna()
            fail_cnt = int(dup_mask.sum())
            ok = fail_cnt == 0
            return ExpectationResult(
                expectation_name=etype,
                column=col,
                passed=ok,
                severity=sev,
                observed_value=fail_cnt == 0,
                expected_value=True,
                unexpected_count=fail_cnt,
                unexpected_ratio=round(fail_cnt / n_rows, 4),
                sample_failing_indices=df.index[dup_mask].tolist()[:10],
                message=f"Column '{col}' values are unique" if ok else f"Found {fail_cnt} duplicate entries in '{col}'",
                remediation="" if ok else f"Deduplicate on key '{col}' before modeling.",
            )

        if etype == "expect_column_values_to_be_between":
            min_v = kw.get("min_value", -np.inf)
            max_v = kw.get("max_value", np.inf)
            mostly = float(kw.get("mostly", 1.0))
            num = pd.to_numeric(series, errors="coerce")
            valid_non_null = num.notna()
            fail_mask = valid_non_null & ((num < min_v) | (num > max_v))
            fail_cnt = int(fail_mask.sum())
            denom = max(int(valid_non_null.sum()), 1)
            ok = (1.0 - fail_cnt / denom) >= mostly
            return ExpectationResult(
                expectation_name=etype,
                column=col,
                passed=ok,
                severity=sev,
                observed_value={"min": float(num.min()) if valid_non_null.any() else None, "max": float(num.max()) if valid_non_null.any() else None},
                expected_value={"min": min_v, "max": max_v},
                unexpected_count=fail_cnt,
                unexpected_ratio=round(fail_cnt / denom, 4),
                sample_failing_indices=df.index[fail_mask].tolist()[:10],
                message=f"'{col}' within [{min_v}, {max_v}]" if ok else f"{fail_cnt} values in '{col}' outside [{min_v}, {max_v}]",
                remediation="" if ok else f"Clip or inspect out-of-range values in '{col}'.",
            )

        if etype == "expect_column_values_to_be_in_set":
            allowed = set(str(v) for v in kw.get("value_set", []))
            non_null_mask = series.notna()
            str_vals = series.astype(str)
            fail_mask = non_null_mask & (~str_vals.isin(allowed))
            fail_cnt = int(fail_mask.sum())
            ok = fail_cnt == 0
            return ExpectationResult(
                expectation_name=etype,
                column=col,
                passed=ok,
                severity=sev,
                observed_value=sorted(str_vals[non_null_mask].unique().tolist())[:20],
                expected_value=sorted(allowed)[:20],
                unexpected_count=fail_cnt,
                unexpected_ratio=round(fail_cnt / n_rows, 4),
                sample_failing_indices=df.index[fail_mask].tolist()[:10],
                message=f"All values in '{col}' belong to allowed set" if ok else f"Found {fail_cnt} unexpected categories in '{col}'",
                remediation="" if ok else f"Normalize unknown category labels in '{col}'.",
            )

        if etype == "expect_column_values_to_match_regex":
            pattern = re.compile(str(kw["regex"]))
            non_null_mask = series.notna()
            fail_mask = non_null_mask & (~series.astype(str).apply(lambda s: bool(pattern.search(s))))
            fail_cnt = int(fail_mask.sum())
            ok = fail_cnt == 0
            return ExpectationResult(
                expectation_name=etype,
                column=col,
                passed=ok,
                severity=sev,
                unexpected_count=fail_cnt,
                unexpected_ratio=round(fail_cnt / n_rows, 4),
                sample_failing_indices=df.index[fail_mask].tolist()[:10],
                message=f"Values match regex in '{col}'" if ok else f"{fail_cnt} values failed regex in '{col}'",
                remediation="" if ok else f"Clean malformed string formats in '{col}'.",
            )

        if etype == "expect_column_dtype":
            expected_kind = str(kw.get("dtype", "numeric")).lower()
            ok = (
                (expected_kind == "numeric" and pd.api.types.is_numeric_dtype(series))
                or (expected_kind == "integer" and pd.api.types.is_integer_dtype(series))
                or (expected_kind == "float" and pd.api.types.is_float_dtype(series))
                or (expected_kind == "datetime" and pd.api.types.is_datetime64_any_dtype(series))
                or (expected_kind == "string" and (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)))
                or (expected_kind == "boolean" and pd.api.types.is_bool_dtype(series))
                or (expected_kind == str(series.dtype).lower())
            )
            return ExpectationResult(
                expectation_name=etype,
                column=col,
                passed=ok,
                severity=sev,
                observed_value=str(series.dtype),
                expected_value=expected_kind,
                message=f"Column '{col}' dtype is {series.dtype}" if ok else f"Expected '{expected_kind}' for '{col}', got '{series.dtype}'",
                remediation="" if ok else f"Cast column '{col}' to '{expected_kind}' during ingestion or cleaning.",
            )

        if etype == "expect_column_values_to_be_increasing":
            non_null = series.dropna()
            ok = bool(non_null.is_monotonic_increasing)
            return ExpectationResult(
                expectation_name=etype,
                column=col,
                passed=ok,
                severity=sev,
                observed_value=ok,
                expected_value=True,
                message=f"Column '{col}' is monotonically increasing" if ok else f"Column '{col}' is not monotonically increasing",
                remediation="" if ok else f"Sort dataset by '{col}' before time-series modeling.",
            )

        if etype == "expect_column_distribution_ks_test":
            reference = np.asarray(kw["reference_values"], dtype=float)
            observed = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
            alpha = float(kw.get("alpha", 0.05))
            if len(reference) == 0 or len(observed) == 0:
                ok, stat, p_val = False, 1.0, 0.0
            else:
                ks_res = stats.ks_2samp(reference, observed)
                stat, p_val = float(ks_res.statistic), float(ks_res.pvalue)
                ok = p_val >= alpha
            return ExpectationResult(
                expectation_name=etype,
                column=col,
                passed=ok,
                severity=sev,
                observed_value={"ks_statistic": round(stat, 4), "p_value": round(p_val, 6)},
                expected_value={"alpha": alpha},
                message=f"KS test passed (p={p_val:.4f})" if ok else f"Significant distribution shift detected in '{col}' (p={p_val:.6f})",
                remediation="" if ok else f"Trigger drift investigation or model retraining for '{col}'.",
            )

        if etype == "expect_column_pair_values_a_to_be_greater_than_b":
            col_b = str(kw["column_b"])
            if col_b not in df.columns:
                return ExpectationResult(
                    expectation_name=etype,
                    column=col,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Comparison column '{col_b}' missing.",
                )
            or_equal = bool(kw.get("or_equal", True))
            a = pd.to_numeric(df[col], errors="coerce")
            b = pd.to_numeric(df[col_b], errors="coerce")
            both_valid = a.notna() & b.notna()
            fail_mask = both_valid & (~(a >= b) if or_equal else ~(a > b))
            fail_cnt = int(fail_mask.sum())
            ok = fail_cnt == 0
            return ExpectationResult(
                expectation_name=etype,
                column=f"{col} vs {col_b}",
                passed=ok,
                severity=sev,
                unexpected_count=fail_cnt,
                unexpected_ratio=round(fail_cnt / n_rows, 4),
                sample_failing_indices=df.index[fail_mask].tolist()[:10],
                message=f"'{col}' >= '{col_b}' holds" if ok else f"{fail_cnt} rows violate '{col}' >= '{col_b}'",
                remediation="" if ok else f"Reconcile cross-column business invariant between '{col}' and '{col_b}'.",
            )

        return ExpectationResult(
            expectation_name=etype,
            column=col,
            passed=False,
            severity=ValidationSeverity.ERROR,
            message=f"Unknown expectation type '{etype}'.",
            remediation="Register custom validator or verify expectation_type spelling.",
        )
