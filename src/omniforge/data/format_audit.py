"""Automatic post-import dataset format, schema & encoding defect auditor.

Runs automatically upon data ingestion in OmniForge to surface column-level and
value-level formatting issues (numeric strings, mixed types, inconsistent date
formats, hidden null tokens, whitespace/casing inconsistencies, control/HTML/
mojibake artifacts, non-finite numbers, constant/empty columns, suspected IDs,
and header naming defects) before statistical testing or modeling begins.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from omniforge.core.logging import get_logger
from omniforge.core.performance import GLOBAL_DF_CACHE

logger = get_logger(__name__)

SeverityLevel = Literal["high", "medium", "low"]
SemanticType = Literal[
    "numeric",
    "numeric_as_text",
    "datetime",
    "datetime_as_text",
    "boolean",
    "boolean_as_text",
    "categorical",
    "identifier",
    "free_text",
    "empty",
    "mixed",
]

# Non-standard missing-value sentinel strings frequently exported by Excel/ERP/CSV systems
_HIDDEN_MISSING_TOKENS: frozenset[str] = frozenset(
    {
        "",
        "na",
        "n/a",
        "n.a.",
        "nan",
        "null",
        "none",
        "nil",
        "-",
        "--",
        "---",
        "?",
        "??",
        "missing",
        "unknown",
        "undefined",
        "#n/a",
        "#na",
        "#value!",
        "#ref!",
        "#div/0!",
        "#name?",
        "#null!",
        "inf",
        "-inf",
    }
)

_BOOL_TOKENS: frozenset[str] = frozenset({"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1"})
_PURE_NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_FORMATTED_NUM_RE = re.compile(r"^[\s$€£¥₹+-]*\d{1,3}(?:,\d{3})*(?:\.\d+)?%?[\s]*$")
_CURRENCY_OR_PCT_RE = re.compile(r"^[\s]*[$€£¥₹+-]?\s*\d[\d,]*(?:\.\d+)?\s*%?[\s]*$")
_EURO_DECIMAL_RE = re.compile(r"^\s*[+-]?(?:\d{1,3}(?:\.\d{3})+,\d+|\d+,\d{1,2})\s*$")
_HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*>")
_CTRL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_NEWLINE_TAB_RE = re.compile(r"[\r\n\t]")
_MULTI_SPACE_RE = re.compile(r" {2,}")
_MOJIBAKE_RE = re.compile(r"\ufffd|Ã[\x80-\xbf]|â€[™“”–—]|Â[\xa0-\xbf]")
_UNNAMED_COL_RE = re.compile(r"^Unnamed:\s*\d+$", re.IGNORECASE)
_ID_NAME_RE = re.compile(r"(?:^|_)(?:id|uuid|guid|code|key|index|idx|serial|ssn|account_no|record_no)(?:$|_)", re.IGNORECASE)

# Recognizable date format families used to detect mixed/unparsed date strings
_DATE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ISO YYYY-MM-DD", re.compile(r"^\d{4}-\d{1,2}-\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?$")),
    ("Slash YYYY/MM/DD", re.compile(r"^\d{4}/\d{1,2}/\d{1,2}(?: \d{1,2}:\d{2}(?::\d{2})?)?$")),
    ("Slash MM/DD/YYYY or DD/MM/YYYY", re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}(?: \d{1,2}:\d{2}(?::\d{2})?)?$")),
    ("Dash DD-MM-YYYY or MM-DD-YYYY", re.compile(r"^\d{1,2}-\d{1,2}-\d{4}(?: \d{1,2}:\d{2}(?::\d{2})?)?$")),
    ("Dot DD.MM.YYYY", re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}(?: \d{1,2}:\d{2}(?::\d{2})?)?$")),
    ("Named Month", re.compile(r"^(?:\d{1,2}[\s-]+[A-Za-z]{3,9}[\s,-]+\d{2,4}|[A-Za-z]{3,9}[\s]+\d{1,2},?[\s]+\d{2,4})$")),
]


class FormatIssue(BaseModel):
    """Single detected formatting, type, encoding, or schema defect."""

    column: str
    issue_code: str
    category: str
    severity: SeverityLevel
    affected_count: int = 0
    affected_pct: float = 0.0
    description: str
    sample_values: list[str] = Field(default_factory=list)
    recommendation: str
    auto_fixable: bool = False


class ColumnFormatProfile(BaseModel):
    """Per-column format health and inferred semantic type summary."""

    column: str
    raw_dtype: str
    semantic_type: SemanticType
    missing_count: int = 0
    hidden_missing_count: int = 0
    unique_count: int = 0
    issue_count: int = 0
    highest_severity: Literal["none", "low", "medium", "high"] = "none"
    issues: list[FormatIssue] = Field(default_factory=list)


class DatasetFormatAuditReport(BaseModel):
    """Complete automatic post-import format audit report across all columns."""

    audited_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    row_count: int
    column_count: int
    total_issues: int = 0
    high_severity_count: int = 0
    medium_severity_count: int = 0
    low_severity_count: int = 0
    clean_columns_count: int = 0
    health_status: str = "Clean"
    suggested_exclude_columns: list[str] = Field(default_factory=list)
    column_profiles: list[ColumnFormatProfile] = Field(default_factory=list)
    issues: list[FormatIssue] = Field(default_factory=list)
    summary_notes: list[str] = Field(default_factory=list)


def _severity_rank(sev: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(sev, 0)


def _looks_numeric_string(val: str) -> bool:
    s = val.strip()
    if not s:
        return False
    if _PURE_NUM_RE.match(s) or _FORMATTED_NUM_RE.match(s) or _CURRENCY_OR_PCT_RE.match(s):
        cleaned = re.sub(r"[$€£¥₹,\s%]", "", s)
        try:
            float(cleaned)
            return True
        except ValueError:
            return False
    return False


def _match_date_family(val: str) -> str | None:
    s = val.strip()
    if len(s) < 6 or len(s) > 40:
        return None
    for family, pat in _DATE_PATTERNS:
        if pat.match(s):
            return family
    return None


class DataFormatAuditor:
    """Inspects any tabular DataFrame immediately after import for formatting and schema defects."""

    def __init__(self, max_eval_rows: int = 15_000, max_samples: int = 5, use_cache: bool = True) -> None:
        self.max_eval_rows = max(500, int(max_eval_rows))
        self.max_samples = max(1, int(max_samples))
        self.use_cache = bool(use_cache)

    def audit(self, df: pd.DataFrame) -> DatasetFormatAuditReport:
        """Execute all format, encoding, type-consistency, and schema defect checks.

        Args:
            df: Newly imported (or updated) pandas DataFrame.

        Returns:
            :class:`DatasetFormatAuditReport` detailing every detected issue per column.
        """
        if self.use_cache:
            hit, cached = GLOBAL_DF_CACHE.get("format_audit", df, f"{self.max_eval_rows}:{self.max_samples}")
            if hit and isinstance(cached, DatasetFormatAuditReport):
                return cached
        n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
        if n_cols == 0:
            return DatasetFormatAuditReport(
                row_count=n_rows,
                column_count=0,
                health_status="Critical Format Defects",
                summary_notes=["Dataset has zero columns; verify file delimiter and encoding."],
            )

        eval_df = df if n_rows <= self.max_eval_rows else df.sample(n=self.max_eval_rows, random_state=42)
        eval_n = max(len(eval_df), 1)
        scale_factor = float(n_rows) / float(eval_n) if n_rows > len(eval_df) else 1.0

        all_issues: list[FormatIssue] = []
        col_profiles: list[ColumnFormatProfile] = []
        suggested_exclude: list[str] = []

        # 1. Duplicate column header detection
        col_name_counts = Counter(str(c) for c in df.columns)
        dup_headers = {c for c, cnt in col_name_counts.items() if cnt > 1}

        for col_idx in range(n_cols):
            raw_col = df.columns[col_idx]
            col_name = str(raw_col)
            series = eval_df.iloc[:, col_idx]
            col_issues: list[FormatIssue] = []

            # --- Header Checks ---
            if col_name in dup_headers:
                col_issues.append(
                    FormatIssue(
                        column=col_name,
                        issue_code="DUPLICATE_COLUMN_NAME",
                        category="Header Defect",
                        severity="high",
                        affected_count=col_name_counts[col_name],
                        affected_pct=100.0,
                        description=f"Column header '{col_name}' appears {col_name_counts[col_name]} times in the dataset.",
                        sample_values=[col_name],
                        recommendation="Rename duplicate columns uniquely before running statistical tests or modeling.",
                        auto_fixable=False,
                    )
                )
            if _UNNAMED_COL_RE.match(col_name) or col_name.strip() == "":
                col_issues.append(
                    FormatIssue(
                        column=col_name,
                        issue_code="UNNAMED_INDEX_COLUMN",
                        category="Header Defect",
                        severity="medium",
                        affected_count=n_rows,
                        affected_pct=100.0,
                        description=f"Column '{col_name}' appears to be an exported CSV/Excel row-index artifact without a header.",
                        sample_values=[col_name],
                        recommendation="Exclude this index artifact from studies (Study Scope panel) or drop it during cleaning.",
                        auto_fixable=True,
                    )
                )
                if col_name not in suggested_exclude:
                    suggested_exclude.append(col_name)
            elif col_name != col_name.strip() or _MULTI_SPACE_RE.search(col_name) or _CTRL_CHAR_RE.search(col_name):
                col_issues.append(
                    FormatIssue(
                        column=col_name,
                        issue_code="WHITESPACE_IN_HEADER",
                        category="Header Defect",
                        severity="low",
                        affected_count=1,
                        affected_pct=0.0,
                        description=f"Column header '{col_name}' contains leading/trailing spaces, repeated spaces, or control characters.",
                        sample_values=[repr(col_name)],
                        recommendation="Trim whitespace from column headers to avoid formula and query reference errors.",
                        auto_fixable=True,
                    )
                )

            missing_cnt = int(round(int(series.isna().sum()) * scale_factor))
            non_null = series.dropna()
            n_non_null = len(non_null)
            try:
                unique_cnt = int(non_null.nunique())
            except TypeError:
                unique_cnt = int(non_null.astype(str).nunique())

            hidden_missing_cnt = 0
            semantic_type: SemanticType = "categorical"

            # --- Empty / Constant Column Checks ---
            if n_non_null == 0:
                semantic_type = "empty"
                col_issues.append(
                    FormatIssue(
                        column=col_name,
                        issue_code="ALL_EMPTY_COLUMN",
                        category="Empty / Constant Column",
                        severity="high",
                        affected_count=n_rows,
                        affected_pct=100.0,
                        description=f"Column '{col_name}' is 100% empty (all values are missing/NaN).",
                        sample_values=["NaN"],
                        recommendation="Exclude from studies or drop high-missing columns in Studio 01 Tab 3 (Cleaning).",
                        auto_fixable=True,
                    )
                )
                if col_name not in suggested_exclude:
                    suggested_exclude.append(col_name)
            else:
                if unique_cnt == 1 and n_rows >= 5:
                    sample_val = str(non_null.iloc[0])[:60]
                    col_issues.append(
                        FormatIssue(
                            column=col_name,
                            issue_code="CONSTANT_COLUMN",
                            category="Empty / Constant Column",
                            severity="medium",
                            affected_count=n_rows,
                            affected_pct=100.0,
                            description=f"Column '{col_name}' has zero variance (every non-null record equals '{sample_val}').",
                            sample_values=[sample_val],
                            recommendation="Exclude from statistical tests and modeling — constant features add zero information.",
                            auto_fixable=False,
                        )
                    )
                    if col_name not in suggested_exclude:
                        suggested_exclude.append(col_name)

                # --- Branch by Pandas Dtype ---
                if pd.api.types.is_bool_dtype(series):
                    semantic_type = "boolean"
                elif pd.api.types.is_datetime64_any_dtype(series):
                    semantic_type = "datetime"
                elif pd.api.types.is_numeric_dtype(series):
                    semantic_type = "numeric"
                    arr = pd.to_numeric(non_null, errors="coerce")
                    inf_mask = np.isinf(arr.to_numpy(dtype=float, na_value=np.nan))
                    inf_cnt = int(np.sum(inf_mask))
                    if inf_cnt > 0:
                        scaled_inf = int(round(inf_cnt * scale_factor))
                        col_issues.append(
                            FormatIssue(
                                column=col_name,
                                issue_code="INFINITE_NUMERIC_VALUES",
                                category="Numeric Format Defect",
                                severity="high",
                                affected_count=scaled_inf,
                                affected_pct=round((inf_cnt / eval_n) * 100.0, 2),
                                description=f"Column '{col_name}' contains {scaled_inf:,} infinite (+inf / -inf) value(s).",
                                sample_values=[str(v) for v in non_null[inf_mask].head(self.max_samples).tolist()],
                                recommendation="Replace infinite values with NaN and impute before statistical tests or model training.",
                                auto_fixable=True,
                            )
                        )

                    # Detect suspected integer ID / monotonic index column
                    if (
                        n_rows >= 20
                        and unique_cnt == n_non_null
                        and missing_cnt == 0
                        and (_ID_NAME_RE.search(col_name) is not None or _UNNAMED_COL_RE.match(col_name) is not None)
                    ):
                        semantic_type = "identifier"
                        col_issues.append(
                            FormatIssue(
                                column=col_name,
                                issue_code="SUSPECTED_ID_COLUMN",
                                category="Identifier / Non-Analytical Column",
                                severity="low",
                                affected_count=n_rows,
                                affected_pct=100.0,
                                description=f"Column '{col_name}' has 100% unique values ({unique_cnt:,}) and an identifier-style name.",
                                sample_values=[str(v) for v in non_null.head(self.max_samples).tolist()],
                                recommendation="Exclude identifier columns in Study Scope so they do not distort correlations or leak into models.",
                                auto_fixable=False,
                            )
                        )
                        if col_name not in suggested_exclude:
                            suggested_exclude.append(col_name)
                else:
                    # Object / String / Categorical / Mixed column inspection
                    semantic_type, hidden_missing_cnt = self._audit_object_column(
                        col_name=col_name,
                        non_null=non_null,
                        eval_n=eval_n,
                        n_rows=n_rows,
                        scale_factor=scale_factor,
                        unique_cnt=unique_cnt,
                        col_issues=col_issues,
                        suggested_exclude=suggested_exclude,
                    )

            # High missing rate warning (non-empty columns with >= 25% real NaNs)
            if 0 < n_non_null and (missing_cnt / max(n_rows, 1)) >= 0.25:
                miss_pct = round((missing_cnt / max(n_rows, 1)) * 100.0, 2)
                col_issues.append(
                    FormatIssue(
                        column=col_name,
                        issue_code="HIGH_MISSING_VALUES",
                        category="Missingness Defect",
                        severity="high" if miss_pct >= 50.0 else "medium",
                        affected_count=missing_cnt,
                        affected_pct=miss_pct,
                        description=f"Column '{col_name}' is missing {missing_cnt:,} values ({miss_pct:.1f}%).",
                        sample_values=["NaN"],
                        recommendation="Impute missing values in Studio 01 Tab 3 (Cleaning) or exclude this column if uninformative.",
                        auto_fixable=True,
                    )
                )

            highest: Literal["none", "low", "medium", "high"] = "none"
            for iss in col_issues:
                if _severity_rank(iss.severity) > _severity_rank(highest):
                    highest = iss.severity

            col_profiles.append(
                ColumnFormatProfile(
                    column=col_name,
                    raw_dtype=str(series.dtype),
                    semantic_type=semantic_type,
                    missing_count=missing_cnt,
                    hidden_missing_count=hidden_missing_cnt,
                    unique_count=unique_cnt,
                    issue_count=len(col_issues),
                    highest_severity=highest,
                    issues=col_issues,
                )
            )
            all_issues.extend(col_issues)

        # Sort issues: high -> medium -> low, then by affected_count descending
        all_issues.sort(key=lambda i: (-_severity_rank(i.severity), -i.affected_count, i.column))

        high_cnt = sum(1 for i in all_issues if i.severity == "high")
        med_cnt = sum(1 for i in all_issues if i.severity == "medium")
        low_cnt = sum(1 for i in all_issues if i.severity == "low")
        clean_cols = sum(1 for cp in col_profiles if cp.issue_count == 0)

        if high_cnt > 0:
            health = "Critical Format Defects" if high_cnt >= 3 else "Attention Required"
        elif med_cnt > 0:
            health = "Minor Format Issues"
        elif low_cnt > 0:
            health = "Minor Format Advisories"
        else:
            health = "Clean"

        notes: list[str] = []
        if not all_issues:
            notes.append("All columns passed automatic format, schema, type-consistency, and encoding checks cleanly.")
        else:
            notes.append(
                f"Detected {len(all_issues)} format/schema finding(s) across {n_cols - clean_cols} of {n_cols} column(s) "
                f"({high_cnt} high, {med_cnt} medium, {low_cnt} low severity)."
            )
            auto_fix = sum(1 for i in all_issues if i.auto_fixable)
            if auto_fix > 0:
                notes.append(
                    f"{auto_fix} issue(s) can be automatically repaired in Studio 01 → Tab 3 (Cleaning & Diff)."
                )
            if suggested_exclude:
                notes.append(
                    f"{len(suggested_exclude)} column(s) look like IDs, empty, or constant columns "
                    f"({', '.join(suggested_exclude[:6])}) and can be excluded in one click via Study Scope."
                )

        report = DatasetFormatAuditReport(
            row_count=n_rows,
            column_count=n_cols,
            total_issues=len(all_issues),
            high_severity_count=high_cnt,
            medium_severity_count=med_cnt,
            low_severity_count=low_cnt,
            clean_columns_count=clean_cols,
            health_status=health,
            suggested_exclude_columns=suggested_exclude,
            column_profiles=col_profiles,
            issues=all_issues,
            summary_notes=notes,
        )
        logger.info(
            "format_audit_complete",
            rows=n_rows,
            cols=n_cols,
            total_issues=len(all_issues),
            high=high_cnt,
            medium=med_cnt,
            low=low_cnt,
            health_status=health,
        )
        if self.use_cache:
            GLOBAL_DF_CACHE.set("format_audit", df, report, f"{self.max_eval_rows}:{self.max_samples}")
        return report

    def _audit_object_column(
        self,
        *,
        col_name: str,
        non_null: pd.Series,
        eval_n: int,
        n_rows: int,
        scale_factor: float,
        unique_cnt: int,
        col_issues: list[FormatIssue],
        suggested_exclude: list[str],
    ) -> tuple[SemanticType, int]:
        """Inspect object/string/categorical series for format, mixed-type, date, whitespace & encoding defects."""
        raw_vals = non_null.tolist()
        str_vals = [str(v) for v in raw_vals]

        # Detect raw Python mixed object types (e.g. ints/floats alongside non-numeric strings)
        py_types = Counter(
            "number"
            if isinstance(v, (int, float, np.number)) and not isinstance(v, bool)
            else ("bool" if isinstance(v, bool) else type(v).__name__)
            for v in raw_vals
        )

        # 1. Hidden / non-standard missing tokens ("N/A", "null", "?", "-", "", whitespace-only)
        hidden_samples: list[str] = []
        hidden_cnt = 0
        valid_str_vals: list[str] = []
        for s in str_vals:
            stripped = s.strip()
            if stripped.lower() in _HIDDEN_MISSING_TOKENS:
                hidden_cnt += 1
                if len(hidden_samples) < self.max_samples:
                    hidden_samples.append(repr(s) if stripped == "" else s)
            else:
                valid_str_vals.append(s)

        scaled_hidden = int(round(hidden_cnt * scale_factor))
        if hidden_cnt > 0:
            col_issues.append(
                FormatIssue(
                    column=col_name,
                    issue_code="HIDDEN_MISSING_TOKENS",
                    category="Hidden Missing Values",
                    severity="high" if (hidden_cnt / eval_n) >= 0.05 else "medium",
                    affected_count=scaled_hidden,
                    affected_pct=round((hidden_cnt / eval_n) * 100.0, 2),
                    description=(
                        f"Column '{col_name}' contains {scaled_hidden:,} placeholder/sentinel string(s) "
                        f"representing missing data (e.g., N/A, null, ?, '-', blank)."
                    ),
                    sample_values=hidden_samples,
                    recommendation="Enable 'Standardize Missing Tokens' in Studio 01 Tab 3 (Cleaning) to convert these to true NaN before imputation.",
                    auto_fixable=True,
                )
            )

        if not valid_str_vals:
            return "empty", scaled_hidden

        n_valid = len(valid_str_vals)
        semantic_type: SemanticType = "categorical"

        # 2. Numeric stored as text / Currency / Percentages / European decimal commas / Mixed numeric+text
        euro_matches = [s for s in valid_str_vals if _EURO_DECIMAL_RE.match(s)]
        num_matches = [s for s in valid_str_vals if _looks_numeric_string(s) and not _EURO_DECIMAL_RE.match(s)]
        num_ratio = len(num_matches) / n_valid
        euro_ratio = len(euro_matches) / n_valid

        if euro_ratio >= 0.65:
            semantic_type = "numeric_as_text"
            col_issues.append(
                FormatIssue(
                    column=col_name,
                    issue_code="EUROPEAN_DECIMAL_COMMA",
                    category="Numeric Stored as Text",
                    severity="high",
                    affected_count=int(round(len(euro_matches) * scale_factor)),
                    affected_pct=round((len(euro_matches) / eval_n) * 100.0, 2),
                    description=(
                        f"Column '{col_name}' uses comma decimal separators (e.g., '12,50'), causing numbers to be imported as text."
                    ),
                    sample_values=euro_matches[: self.max_samples],
                    recommendation="Normalize decimal commas to dots ('.') and cast to numeric before statistical tests or modeling.",
                    auto_fixable=True,
                )
            )
        elif num_ratio >= 0.85:
            semantic_type = "numeric_as_text"
            has_currency_or_comma = any(re.search(r"[$€£¥₹,%]", s) for s in num_matches[:200])
            bad_examples = [s for s in valid_str_vals if not _looks_numeric_string(s)]
            detail = (
                "formatted numeric strings (currency symbols, commas, or % signs)"
                if has_currency_or_comma
                else "numeric values stored as text/object strings"
            )
            col_issues.append(
                FormatIssue(
                    column=col_name,
                    issue_code="NUMERIC_STORED_AS_TEXT",
                    category="Numeric Stored as Text",
                    severity="high",
                    affected_count=int(round(len(num_matches) * scale_factor)),
                    affected_pct=round((len(num_matches) / eval_n) * 100.0, 2),
                    description=f"Column '{col_name}' contains {detail} ({num_ratio * 100:.1f}% numeric-like).",
                    sample_values=(bad_examples[:2] + num_matches[:3])[: self.max_samples],
                    recommendation="Run Data Cleaning (Tab 3) with automatic type inference to strip currency/commas and cast to numeric.",
                    auto_fixable=True,
                )
            )
            if bad_examples:
                col_issues.append(
                    FormatIssue(
                        column=col_name,
                        issue_code="MIXED_TYPES_IN_COLUMN",
                        category="Mixed Data Types",
                        severity="high",
                        affected_count=int(round(len(bad_examples) * scale_factor)),
                        affected_pct=round((len(bad_examples) / eval_n) * 100.0, 2),
                        description=(
                            f"Column '{col_name}' is predominantly numeric ({num_ratio * 100:.1f}%) but contains "
                            f"{int(round(len(bad_examples) * scale_factor)):,} non-numeric text value(s)."
                        ),
                        sample_values=list(dict.fromkeys(bad_examples))[: self.max_samples],
                        recommendation="Inspect and repair malformed text entries or allow Cleaning to coerce invalid entries to NaN and impute.",
                        auto_fixable=True,
                    )
                )
        elif 0.15 <= num_ratio < 0.85 or (len(py_types) > 1 and "number" in py_types and "str" in py_types):
            semantic_type = "mixed"
            non_num = [s for s in valid_str_vals if not _looks_numeric_string(s)]
            col_issues.append(
                FormatIssue(
                    column=col_name,
                    issue_code="MIXED_TYPES_IN_COLUMN",
                    category="Mixed Data Types",
                    severity="high",
                    affected_count=int(round(min(len(num_matches), len(non_num)) * scale_factor)) or int(round(n_valid * scale_factor)),
                    affected_pct=round((min(len(num_matches), len(non_num)) / eval_n) * 100.0, 2),
                    description=(
                        f"Column '{col_name}' mixes numeric values ({len(num_matches):,}) and text strings ({len(non_num):,}) in the same field."
                    ),
                    sample_values=(num_matches[:2] + non_num[:3])[: self.max_samples],
                    recommendation="Split or standardize mixed-type values before statistical tests; otherwise tests treat the entire column as categorical text.",
                    auto_fixable=False,
                )
            )

        # 3. Date / Datetime stored as text & inconsistent date formats
        if semantic_type == "categorical":
            family_hits: Counter[str] = Counter()
            date_matched_count = 0
            unmatched_date_samples: list[str] = []
            matched_samples_by_fam: dict[str, str] = {}
            for s in valid_str_vals:
                fam = _match_date_family(s)
                if fam is not None:
                    family_hits[fam] += 1
                    date_matched_count += 1
                    matched_samples_by_fam.setdefault(fam, s)
                elif len(unmatched_date_samples) < self.max_samples:
                    unmatched_date_samples.append(s)

            date_ratio = date_matched_count / n_valid
            if date_ratio >= 0.60:
                semantic_type = "datetime_as_text"
                if len(family_hits) > 1:
                    fams_desc = ", ".join(f"{k} ({v})" for k, v in family_hits.most_common(4))
                    col_issues.append(
                        FormatIssue(
                            column=col_name,
                            issue_code="INCONSISTENT_DATE_FORMATS",
                            category="Inconsistent Date Formats",
                            severity="high",
                            affected_count=int(round(date_matched_count * scale_factor)),
                            affected_pct=round((date_matched_count / eval_n) * 100.0, 2),
                            description=(
                                f"Column '{col_name}' mixes {len(family_hits)} different date string formats: {fams_desc}."
                            ),
                            sample_values=list(matched_samples_by_fam.values())[: self.max_samples],
                            recommendation="Standardize all dates to ISO-8601 (YYYY-MM-DD) in Cleaning before time-series splitting or forecasting.",
                            auto_fixable=True,
                        )
                    )
                else:
                    sole_fam = next(iter(family_hits))
                    col_issues.append(
                        FormatIssue(
                            column=col_name,
                            issue_code="UNPARSED_DATE_TEXT",
                            category="Date Stored as Text",
                            severity="medium",
                            affected_count=int(round(date_matched_count * scale_factor)),
                            affected_pct=round((date_matched_count / eval_n) * 100.0, 2),
                            description=f"Column '{col_name}' contains {sole_fam} date strings stored as plain text instead of datetime64.",
                            sample_values=list(matched_samples_by_fam.values())[: self.max_samples],
                            recommendation="Run Cleaning (Tab 3) to parse text dates into native datetime type for temporal analysis.",
                            auto_fixable=True,
                        )
                    )
                if unmatched_date_samples and date_ratio < 0.995:
                    bad_dt_cnt = n_valid - date_matched_count
                    col_issues.append(
                        FormatIssue(
                            column=col_name,
                            issue_code="MALFORMED_DATE_ENTRIES",
                            category="Inconsistent Date Formats",
                            severity="high",
                            affected_count=int(round(bad_dt_cnt * scale_factor)),
                            affected_pct=round((bad_dt_cnt / eval_n) * 100.0, 2),
                            description=f"Column '{col_name}' has {int(round(bad_dt_cnt * scale_factor)):,} value(s) that do not match a valid date pattern.",
                            sample_values=unmatched_date_samples[: self.max_samples],
                            recommendation="Inspect malformed date strings; invalid entries will become NaT when parsed.",
                            auto_fixable=False,
                        )
                    )

        # 4. Boolean stored as text (e.g. "Yes"/"No", "True"/"false", "Y"/"N")
        if semantic_type == "categorical":
            lower_set = {s.strip().lower() for s in valid_str_vals}
            if 1 <= len(lower_set) <= 4 and lower_set.issubset(_BOOL_TOKENS):
                semantic_type = "boolean_as_text"
                raw_distinct = sorted({s.strip() for s in valid_str_vals})
                if len(raw_distinct) > 2:
                    col_issues.append(
                        FormatIssue(
                            column=col_name,
                            issue_code="INCONSISTENT_BOOLEAN_ENCODING",
                            category="Inconsistent Boolean Encoding",
                            severity="medium",
                            affected_count=int(round(n_valid * scale_factor)),
                            affected_pct=round((n_valid / eval_n) * 100.0, 2),
                            description=f"Boolean column '{col_name}' uses {len(raw_distinct)} variant representations: {', '.join(raw_distinct[:6])}.",
                            sample_values=raw_distinct[: self.max_samples],
                            recommendation="Standardize boolean strings to canonical True/False or 1/0 before analysis.",
                            auto_fixable=True,
                        )
                    )

        # 5. Leading / trailing / repeated internal whitespace
        ws_bad = [s for s in valid_str_vals if s != s.strip() or _MULTI_SPACE_RE.search(s)]
        if ws_bad:
            col_issues.append(
                FormatIssue(
                    column=col_name,
                    issue_code="EXCESS_WHITESPACE",
                    category="Whitespace Defect",
                    severity="medium" if (len(ws_bad) / eval_n) >= 0.05 else "low",
                    affected_count=int(round(len(ws_bad) * scale_factor)),
                    affected_pct=round((len(ws_bad) / eval_n) * 100.0, 2),
                    description=f"Column '{col_name}' has {int(round(len(ws_bad) * scale_factor)):,} value(s) with leading, trailing, or repeated spaces.",
                    sample_values=[repr(x) for x in list(dict.fromkeys(ws_bad))[: self.max_samples]],
                    recommendation="Enable 'Trim & Normalize Strings' in Studio 01 Tab 3 (Cleaning) to avoid split categories.",
                    auto_fixable=True,
                )
            )

        # 6. Inconsistent letter casing / near-duplicate category spellings ("Male" vs "male" vs "MALE")
        if semantic_type in ("categorical", "boolean_as_text") and unique_cnt <= 500:
            norm_groups: dict[str, set[str]] = {}
            for s in valid_str_vals:
                stripped = s.strip()
                norm_groups.setdefault(stripped.lower(), set()).add(stripped)
            case_conflicts = {k: sorted(v) for k, v in norm_groups.items() if len(v) > 1}
            if case_conflicts:
                conflict_examples = [" / ".join(vs[:3]) for vs in list(case_conflicts.values())[: self.max_samples]]
                affected_rows = sum(1 for s in valid_str_vals if s.strip().lower() in case_conflicts)
                col_issues.append(
                    FormatIssue(
                        column=col_name,
                        issue_code="INCONSISTENT_CASING",
                        category="Inconsistent Casing",
                        severity="medium",
                        affected_count=int(round(affected_rows * scale_factor)),
                        affected_pct=round((affected_rows / eval_n) * 100.0, 2),
                        description=(
                            f"Column '{col_name}' has {len(case_conflicts)} category group(s) with inconsistent capitalization "
                            f"(creates duplicate categories in group-by tests and encoding)."
                        ),
                        sample_values=conflict_examples,
                        recommendation="Standardize category casing during cleaning so identical labels are merged cleanly.",
                        auto_fixable=True,
                    )
                )

        # 7. Control characters, embedded newlines/tabs, HTML markup, or mojibake encoding artifacts
        ctrl_html_bad = [
            s
            for s in valid_str_vals
            if _CTRL_CHAR_RE.search(s) or _NEWLINE_TAB_RE.search(s) or _HTML_TAG_RE.search(s)
        ]
        if ctrl_html_bad:
            col_issues.append(
                FormatIssue(
                    column=col_name,
                    issue_code="CONTROL_OR_HTML_CHARS",
                    category="Special / Markup Characters",
                    severity="medium",
                    affected_count=int(round(len(ctrl_html_bad) * scale_factor)),
                    affected_pct=round((len(ctrl_html_bad) / eval_n) * 100.0, 2),
                    description=f"Column '{col_name}' contains {int(round(len(ctrl_html_bad) * scale_factor)):,} value(s) with HTML tags, tabs, newlines, or control characters.",
                    sample_values=[repr(x[:60]) for x in list(dict.fromkeys(ctrl_html_bad))[: self.max_samples]],
                    recommendation="Enable 'Trim & Normalize Strings' in Cleaning to strip HTML tags and control characters.",
                    auto_fixable=True,
                )
            )

        mojibake_bad = [s for s in valid_str_vals if _MOJIBAKE_RE.search(s)]
        if mojibake_bad:
            col_issues.append(
                FormatIssue(
                    column=col_name,
                    issue_code="ENCODING_MOJIBAKE",
                    category="Encoding Defect",
                    severity="high",
                    affected_count=int(round(len(mojibake_bad) * scale_factor)),
                    affected_pct=round((len(mojibake_bad) / eval_n) * 100.0, 2),
                    description=f"Column '{col_name}' contains {int(round(len(mojibake_bad) * scale_factor)):,} text value(s) with garbled character-encoding (mojibake / replacement character).",
                    sample_values=list(dict.fromkeys(mojibake_bad))[: self.max_samples],
                    recommendation="Re-export source CSV/Excel using UTF-8 encoding or clean corrupted strings before NLP/grouping.",
                    auto_fixable=False,
                )
            )

        # 8. Identifier / Free-Text high-cardinality text column advisory
        if semantic_type == "categorical" and n_rows >= 20:
            uniq_ratio = unique_cnt / max(n_valid, 1)
            avg_len = float(np.mean([len(s) for s in valid_str_vals[:500]]))
            is_named_id = _ID_NAME_RE.search(col_name) is not None
            if (is_named_id and uniq_ratio >= 0.90) or (uniq_ratio >= 0.98 and avg_len <= 40.0):
                semantic_type = "identifier"
                col_issues.append(
                    FormatIssue(
                        column=col_name,
                        issue_code="SUSPECTED_ID_COLUMN",
                        category="Identifier / Non-Analytical Column",
                        severity="low",
                        affected_count=n_rows,
                        affected_pct=100.0,
                        description=f"Text column '{col_name}' is {uniq_ratio * 100:.1f}% unique ({unique_cnt:,} distinct values) and behaves like a record ID/key.",
                        sample_values=valid_str_vals[: self.max_samples],
                        recommendation="Exclude this identifier column in Study Scope so it does not inflate cardinality or overfit models.",
                        auto_fixable=False,
                    )
                )
                if col_name not in suggested_exclude:
                    suggested_exclude.append(col_name)
            elif avg_len >= 60.0 and uniq_ratio >= 0.5:
                semantic_type = "free_text"

        return semantic_type, scaled_hidden
