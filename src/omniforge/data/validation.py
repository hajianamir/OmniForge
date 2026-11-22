"""Great Expectations & Pandera-style declarative data validation engine for OmniForge."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

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


_BUILTIN_EXPECTATION_TYPES: frozenset[str] = frozenset(
    {
        "expect_table_row_count_to_be_between",
        "expect_table_columns_to_match_set",
        "expect_column_to_exist",
        "expect_column_values_to_not_be_null",
        "expect_column_null_ratio_to_be_below",
        "expect_column_values_to_be_unique",
        "expect_column_values_to_be_between",
        "expect_column_values_to_be_in_set",
        "expect_column_values_to_match_regex",
        "expect_column_dtype",
        "expect_column_values_to_be_increasing",
        "expect_column_distribution_ks_test",
        "expect_column_pair_values_a_to_be_greater_than_b",
    }
)

_SUPPORTED_RULE_SHORTHANDS: tuple[str, ...] = (
    "range",
    "not_null",
    "null_ratio",
    "in_set",
    "unique",
    "regex",
    "dtype",
    "increasing",
    "ks_test",
    "pair_greater",
    "column_exists",
    "row_count",
    "columns_match_set",
)


class ValidationRule(BaseModel):
    """Concise shorthand validation rule convertible into a Great Expectations :class:`Expectation`.

    Aligned with ISO/IEC 25012:2008 data-quality dimensions (Completeness, Uniqueness, Validity,
    Consistency, Accuracy) and Great Expectations Core expectation semantics.

    Supported ``rule_type`` shorthands:
        ``range`` (``min``/``min_value``, ``max``/``max_value``, ``mostly``),
        ``not_null`` (``mostly``),
        ``null_ratio`` (``max_null_ratio``/``max_ratio``),
        ``in_set`` (``values``/``value_set``),
        ``unique``,
        ``regex`` (``regex``/``pattern``, ``mostly``),
        ``dtype`` (``dtype``/``expected_dtype``),
        ``increasing`` (``strict``),
        ``ks_test`` (``reference_values``, ``alpha``),
        ``pair_greater`` (``column_b`` plus ``column``/``column_a``, ``or_equal``),
        ``column_exists``, ``row_count``, ``columns_match_set``,
        any canonical ``expect_*`` identifier, or ``custom:<name>`` for registered custom validators.
    """

    name: str
    rule_type: str
    column: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    severity: ValidationSeverity = ValidationSeverity.ERROR

    def to_expectation(self) -> Expectation:
        """Convert this concise rule into a validated canonical :class:`Expectation`."""
        rt = self.rule_type.strip().lower()
        p = dict(self.params)

        if rt in ("range", "between"):
            kw: dict[str, Any] = {}
            if "min" in p or "min_value" in p:
                kw["min_value"] = p.get("min_value", p.get("min"))
            if "max" in p or "max_value" in p:
                kw["max_value"] = p.get("max_value", p.get("max"))
            if "mostly" in p:
                mostly = float(p["mostly"])
                if not 0.0 <= mostly <= 1.0:
                    raise DataValidationError(f"Rule '{self.name}': 'mostly' must be in [0.0, 1.0], got {mostly}.")
                kw["mostly"] = mostly
            if "min_value" in kw and "max_value" in kw and kw["min_value"] is not None and kw["max_value"] is not None:
                if float(kw["min_value"]) > float(kw["max_value"]):
                    raise DataValidationError(
                        f"Rule '{self.name}': min ({kw['min_value']}) cannot exceed max ({kw['max_value']})."
                    )
            return Expectation(
                expectation_type="expect_column_values_to_be_between",
                column=self.column,
                kwargs=kw,
                severity=self.severity,
                description=self.name,
            )

        if rt in ("not_null", "non_null"):
            kw_nn: dict[str, Any] = {}
            if "mostly" in p:
                mostly = float(p["mostly"])
                if not 0.0 <= mostly <= 1.0:
                    raise DataValidationError(f"Rule '{self.name}': 'mostly' must be in [0.0, 1.0], got {mostly}.")
                kw_nn["mostly"] = mostly
            return Expectation(
                expectation_type="expect_column_values_to_not_be_null",
                column=self.column,
                kwargs=kw_nn,
                severity=self.severity,
                description=self.name,
            )

        if rt in ("null_ratio", "max_null_ratio"):
            raw_max = p.get("max_null_ratio", p.get("max_ratio", 0.1))
            max_r = float(raw_max)
            if not 0.0 <= max_r <= 1.0:
                raise DataValidationError(
                    f"Rule '{self.name}': 'max_null_ratio' must be in [0.0, 1.0], got {max_r}."
                )
            return Expectation(
                expectation_type="expect_column_null_ratio_to_be_below",
                column=self.column,
                kwargs={"max_null_ratio": max_r},
                severity=self.severity,
                description=self.name,
            )

        if rt == "in_set":
            vals = p.get("values", p.get("value_set", p.get("allowed_values")))
            if vals is None or isinstance(vals, (str, bytes)) or not hasattr(vals, "__iter__"):
                raise DataValidationError(
                    f"Rule '{self.name}' (in_set) requires an iterable 'values' or 'value_set' parameter."
                )
            return Expectation(
                expectation_type="expect_column_values_to_be_in_set",
                column=self.column,
                kwargs={"value_set": list(vals)},
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

        if rt in ("regex", "match_regex"):
            raw_pat = p.get("regex", p.get("pattern"))
            if not isinstance(raw_pat, str) or raw_pat == "":
                raise DataValidationError(
                    f"Rule '{self.name}' (regex) requires a non-empty string 'regex' or 'pattern' parameter."
                )
            try:
                re.compile(raw_pat)
            except re.error as exc:
                raise DataValidationError(
                    f"Rule '{self.name}' has invalid regular expression {raw_pat!r}: {exc}",
                    cause=exc,
                ) from exc
            kw_rx: dict[str, Any] = {"regex": raw_pat}
            if "mostly" in p:
                mostly = float(p["mostly"])
                if not 0.0 <= mostly <= 1.0:
                    raise DataValidationError(f"Rule '{self.name}': 'mostly' must be in [0.0, 1.0], got {mostly}.")
                kw_rx["mostly"] = mostly
            return Expectation(
                expectation_type="expect_column_values_to_match_regex",
                column=self.column,
                kwargs=kw_rx,
                severity=self.severity,
                description=self.name,
            )

        if rt in ("dtype", "type"):
            raw_dt = p.get("dtype", p.get("expected_dtype", p.get("type")))
            if not isinstance(raw_dt, str) or not raw_dt.strip():
                raise DataValidationError(
                    f"Rule '{self.name}' (dtype) requires a non-empty 'dtype' or 'expected_dtype' parameter."
                )
            return Expectation(
                expectation_type="expect_column_dtype",
                column=self.column,
                kwargs={"dtype": raw_dt.strip()},
                severity=self.severity,
                description=self.name,
            )

        if rt in ("increasing", "monotonic_increasing"):
            return Expectation(
                expectation_type="expect_column_values_to_be_increasing",
                column=self.column,
                kwargs={"strict": bool(p.get("strict", False))},
                severity=self.severity,
                description=self.name,
            )

        if rt in ("ks_test", "distribution_ks"):
            ref_vals = p.get("reference_values", p.get("reference"))
            if ref_vals is None:
                raise DataValidationError(
                    f"Rule '{self.name}' (ks_test) requires 'reference_values' in params."
                )
            alpha = float(p.get("alpha", 0.05))
            if not 0.0 < alpha < 1.0:
                raise DataValidationError(f"Rule '{self.name}' (ks_test): 'alpha' must be in (0, 1), got {alpha}.")
            return Expectation(
                expectation_type="expect_column_distribution_ks_test",
                column=self.column,
                kwargs={"reference_values": list(np.asarray(ref_vals, dtype=float).ravel()), "alpha": alpha},
                severity=self.severity,
                description=self.name,
            )

        if rt in ("pair_greater", "column_pair_greater"):
            col_a = self.column or p.get("column_a")
            col_b = p.get("column_b", p.get("other_column"))
            if not isinstance(col_a, str) or not col_a.strip() or not isinstance(col_b, str) or not col_b.strip():
                raise DataValidationError(
                    f"Rule '{self.name}' (pair_greater) requires both 'column' (or 'column_a') and 'column_b'."
                )
            return Expectation(
                expectation_type="expect_column_pair_values_a_to_be_greater_than_b",
                column=col_a.strip(),
                kwargs={"column_b": col_b.strip(), "or_equal": bool(p.get("or_equal", True))},
                severity=self.severity,
                description=self.name,
            )

        if rt in ("column_exists", "exists"):
            return Expectation(
                expectation_type="expect_column_to_exist",
                column=self.column,
                kwargs={},
                severity=self.severity,
                description=self.name,
            )

        if rt == "row_count":
            kw_rc: dict[str, Any] = {}
            min_rc = p.get("min_value", p.get("min"))
            max_rc = p.get("max_value", p.get("max"))
            if min_rc is not None:
                kw_rc["min_value"] = int(min_rc)
            if max_rc is not None:
                kw_rc["max_value"] = int(max_rc)
            if "min_value" in kw_rc and "max_value" in kw_rc and kw_rc["min_value"] > kw_rc["max_value"]:
                raise DataValidationError(
                    f"Rule '{self.name}' (row_count) has min_value ({kw_rc['min_value']}) > max_value ({kw_rc['max_value']})."
                )
            return Expectation(
                expectation_type="expect_table_row_count_to_be_between",
                column=None,
                kwargs=kw_rc,
                severity=self.severity,
                description=self.name,
            )

        if rt == "columns_match_set":
            cols = p.get("column_set", p.get("columns"))
            if cols is None or isinstance(cols, (str, bytes)) or not hasattr(cols, "__iter__"):
                raise DataValidationError(
                    f"Rule '{self.name}' (columns_match_set) requires an iterable 'column_set' parameter."
                )
            return Expectation(
                expectation_type="expect_table_columns_to_match_set",
                column=None,
                kwargs={"column_set": [str(c) for c in cols], "allow_extra": bool(p.get("allow_extra", True))},
                severity=self.severity,
                description=self.name,
            )

        if rt in _BUILTIN_EXPECTATION_TYPES:
            return Expectation(
                expectation_type=rt,
                column=self.column,
                kwargs=p,
                severity=self.severity,
                description=self.name,
            )

        if self.rule_type.startswith("custom:"):
            custom_name = self.rule_type.split(":", 1)[1].strip()
            if not custom_name:
                raise DataValidationError(f"Rule '{self.name}' has empty custom validator name.")
            return Expectation(
                expectation_type=custom_name,
                column=self.column,
                kwargs=p,
                severity=self.severity,
                description=self.name,
            )

        raise DataValidationError(
            f"Unsupported validation rule_type '{self.rule_type}' in rule '{self.name}'.",
            suggestions=[
                f"Use one of: {', '.join(_SUPPORTED_RULE_SHORTHANDS)}, a canonical 'expect_*' type, or 'custom:<name>'."
            ],
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
            raw_pat = kw.get("regex", kw.get("pattern"))
            if not isinstance(raw_pat, str) or raw_pat == "":
                return ExpectationResult(
                    expectation_name=etype,
                    column=col,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Missing or empty 'regex' pattern for column '{col}'.",
                    remediation="Provide a valid regular expression in kwargs['regex'].",
                )
            try:
                pattern = re.compile(raw_pat)
            except re.error as exc:
                return ExpectationResult(
                    expectation_name=etype,
                    column=col,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid regex {raw_pat!r} for '{col}': {exc}",
                    remediation="Correct regular expression syntax.",
                )
            mostly = float(kw.get("mostly", 1.0))
            non_null_mask = series.notna()
            fail_mask = non_null_mask & (~series.astype(str).apply(lambda s: bool(pattern.search(s))))
            fail_cnt = int(fail_mask.sum())
            denom = max(int(non_null_mask.sum()), 1)
            ok = (1.0 - fail_cnt / denom) >= mostly if non_null_mask.any() else True
            return ExpectationResult(
                expectation_name=etype,
                column=col,
                passed=ok,
                severity=sev,
                observed_value=round(1.0 - fail_cnt / denom, 4) if non_null_mask.any() else 1.0,
                expected_value={"regex": raw_pat, "mostly": mostly},
                unexpected_count=fail_cnt,
                unexpected_ratio=round(fail_cnt / denom, 4) if non_null_mask.any() else 0.0,
                sample_failing_indices=df.index[fail_mask].tolist()[:10],
                message=f"Values match regex in '{col}'" if ok else f"{fail_cnt} values failed regex in '{col}'",
                remediation="" if ok else f"Clean malformed string formats in '{col}'.",
            )

        if etype == "expect_column_dtype":
            expected_kind = str(kw.get("dtype", kw.get("expected_dtype", "numeric"))).lower().strip()
            is_bool = bool(pd.api.types.is_bool_dtype(series))
            ok = (
                (expected_kind == "numeric" and pd.api.types.is_numeric_dtype(series) and not is_bool)
                or (expected_kind in ("integer", "int") and pd.api.types.is_integer_dtype(series) and not is_bool)
                or (expected_kind == "float" and pd.api.types.is_float_dtype(series))
                or (expected_kind == "datetime" and pd.api.types.is_datetime64_any_dtype(series))
                or (expected_kind in ("string", "str") and (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)))
                or (expected_kind in ("boolean", "bool") and is_bool)
                or (expected_kind in ("categorical", "category") and isinstance(series.dtype, pd.CategoricalDtype))
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
            strict = bool(kw.get("strict", False))
            non_null = series.dropna()
            ok = bool(non_null.is_monotonic_increasing) and (not strict or bool(non_null.is_unique))
            label = "strictly increasing" if strict else "monotonically increasing"
            return ExpectationResult(
                expectation_name=etype,
                column=col,
                passed=ok,
                severity=sev,
                observed_value=ok,
                expected_value={"increasing": True, "strict": strict},
                message=f"Column '{col}' is {label}" if ok else f"Column '{col}' is not {label}",
                remediation="" if ok else f"Sort or deduplicate dataset by '{col}' before time-series modeling.",
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
            col_b = str(kw.get("column_b", ""))
            if not col_b or col_b not in df.columns:
                return ExpectationResult(
                    expectation_name=etype,
                    column=col,
                    passed=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Comparison column '{col_b}' missing.",
                )
            or_equal = bool(kw.get("or_equal", True))
            op_sym = ">=" if or_equal else ">"
            s_a, s_b = df[col], df[col_b]
            raw_both_non_null = s_a.notna() & s_b.notna()
            if pd.api.types.is_datetime64_any_dtype(s_a) or pd.api.types.is_datetime64_any_dtype(s_b):
                a_cmp = pd.to_datetime(s_a, errors="coerce", format="mixed")
                b_cmp = pd.to_datetime(s_b, errors="coerce", format="mixed")
            else:
                a_num = pd.to_numeric(s_a, errors="coerce")
                b_num = pd.to_numeric(s_b, errors="coerce")
                if raw_both_non_null.any() and not (a_num.notna() & b_num.notna()).any():
                    a_cmp = pd.to_datetime(s_a, errors="coerce", format="mixed")
                    b_cmp = pd.to_datetime(s_b, errors="coerce", format="mixed")
                else:
                    a_cmp, b_cmp = a_num, b_num
            both_parsed = a_cmp.notna() & b_cmp.notna()
            unparseable_mask = raw_both_non_null & (~both_parsed)
            order_fail_mask = both_parsed & (~(a_cmp >= b_cmp) if or_equal else ~(a_cmp > b_cmp))
            fail_mask = unparseable_mask | order_fail_mask
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
                message=f"'{col}' {op_sym} '{col_b}' holds" if ok else f"{fail_cnt} rows violate '{col}' {op_sym} '{col_b}'",
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


class ColumnContract(BaseModel):
    """Formal schema & domain contract for a single column."""

    name: str
    required: bool = True
    dtype: Literal["numeric", "string", "datetime", "boolean", "any"] = "any"
    unique: bool = False
    max_missing_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[Any] | None = None


class DataContract(BaseModel):
    """Dataset-level Data Contract separating hard errors from soft warnings."""

    name: str = "default_contract"
    columns: list[ColumnContract] = Field(default_factory=list)
    allow_extra_columns: bool = True

    def validate_dataframe(self, df: pd.DataFrame) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        spec_names = {c.name for c in self.columns}

        if not self.allow_extra_columns:
            extra = [str(c) for c in df.columns if str(c) not in spec_names]
            if extra:
                warnings.append(f"Unexpected extra columns: {extra}")

        for col_spec in self.columns:
            if col_spec.name not in df.columns:
                if col_spec.required:
                    errors.append(f"Missing required column '{col_spec.name}'.")
                else:
                    warnings.append(f"Optional column '{col_spec.name}' is absent.")
                continue

            s = df[col_spec.name]
            miss_ratio = float(s.isna().mean()) if len(s) > 0 else 0.0
            if miss_ratio > col_spec.max_missing_ratio:
                errors.append(
                    f"Column '{col_spec.name}' missing ratio {miss_ratio:.3f} > {col_spec.max_missing_ratio:.3f}."
                )

            if col_spec.unique and s.dropna().duplicated().any():
                errors.append(f"Column '{col_spec.name}' violates uniqueness constraint.")

            if col_spec.dtype == "numeric" and not pd.api.types.is_numeric_dtype(s):
                errors.append(f"Column '{col_spec.name}' expected numeric dtype, got {s.dtype}.")
            elif col_spec.dtype == "datetime" and not pd.api.types.is_datetime64_any_dtype(s):
                warnings.append(f"Column '{col_spec.name}' is not datetime64 (dtype={s.dtype}).")

            if pd.api.types.is_numeric_dtype(s):
                non_na = s.dropna()
                if not non_na.empty:
                    if col_spec.min_value is not None and float(non_na.min()) < col_spec.min_value:
                        errors.append(
                            f"Column '{col_spec.name}' min {float(non_na.min())} < {col_spec.min_value}."
                        )
                    if col_spec.max_value is not None and float(non_na.max()) > col_spec.max_value:
                        errors.append(
                            f"Column '{col_spec.name}' max {float(non_na.max())} > {col_spec.max_value}."
                        )

            if col_spec.allowed_values is not None:
                bad = set(s.dropna().unique().tolist()) - set(col_spec.allowed_values)
                if bad:
                    errors.append(f"Column '{col_spec.name}' contains disallowed values: {sorted(map(str, bad))[:5]}.")

        return {"passed": len(errors) == 0, "errors": errors, "warnings": warnings}

