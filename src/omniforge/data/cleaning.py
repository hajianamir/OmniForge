"""Comprehensive fit/transform data cleaning pipeline with full audit reporting."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.neighbors import LocalOutlierFactor

from omniforge.core.exceptions import DataCleaningError
from omniforge.core.logging import get_logger
from omniforge.core.performance import parallel_n_jobs
from omniforge.core.utils import compute_dataframe_checksum

logger = get_logger(__name__)

_DEFAULT_MISSING_TOKENS: frozenset[str] = frozenset(
    {"", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "missing", "unknown"}
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_CTRL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_CURRENCY_RE = re.compile(r"^[\s$€£¥,+-]*\d[\d,]*(?:\.\d+)?%?[\s]*$")


class CleaningConfig(BaseModel):
    """Configuration controlling all stages of :class:`DataCleaner`."""

    standardize_missing_tokens: bool = True
    clean_strings: bool = True
    lowercase_strings: bool = False
    strip_html: bool = True
    infer_types: bool = True
    drop_high_missing_cols: bool = True
    missing_col_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    numeric_imputation: str = "median"  # mean, median, constant, knn, iterative, missforest, ffill, bfill, interpolate, none
    categorical_imputation: str = "most_frequent"  # most_frequent, mode, constant, none
    constant_fill_numeric: float = 0.0
    constant_fill_categorical: str = "MISSING"
    add_missing_indicators: bool = False
    drop_exact_duplicates: bool = True
    fuzzy_dedup_column: str | None = None
    fuzzy_dedup_threshold: float = Field(default=0.97, ge=0.5, le=1.0)
    outlier_method: str = "iqr"  # iqr, zscore, mad, percentile, isolation_forest, lof, elliptic, none
    outlier_action: str = "clip"  # clip, remove, drop, nan, flag, none
    outlier_threshold: float = 1.5
    lower_percentile: float = Field(default=0.01, ge=0.0, le=0.49)
    upper_percentile: float = Field(default=0.99, ge=0.51, le=1.0)
    rare_category_threshold: float = Field(default=0.0, ge=0.0, le=0.5)
    contamination: float = Field(default=0.05, gt=0.0, lt=0.5)
    causal_time_series: bool = True
    time_column: str | None = None
    domain_bounds: dict[str, tuple[float | None, float | None]] = Field(default_factory=dict)
    exclude_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_aliases(cls, data: Any) -> Any:
        """Accept ergonomic aliases cleanly so UI and programmatic callers stay 100% consistent."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        if "drop_duplicates" in out and "drop_exact_duplicates" not in out:
            out["drop_exact_duplicates"] = out.pop("drop_duplicates")
        if "normalize_strings" in out and "clean_strings" not in out:
            out["clean_strings"] = out.pop("normalize_strings")
        oa = str(out.get("outlier_action", "")).lower().strip()
        if oa == "drop":
            out["outlier_action"] = "remove"
        elif oa == "winsorize":
            out["outlier_action"] = "clip"
        om = str(out.get("outlier_method", "")).lower().strip()
        if om in {"modified_zscore", "robust_zscore"}:
            out["outlier_method"] = "mad"
        ni = str(out.get("numeric_imputation", "")).lower().strip()
        if ni == "zero":
            out["numeric_imputation"] = "constant"
            out.setdefault("constant_fill_numeric", 0.0)
        ci = str(out.get("categorical_imputation", "")).lower().strip()
        if ci == "mode":
            out["categorical_imputation"] = "most_frequent"
        elif ci == "missing_token":
            out["categorical_imputation"] = "constant"
        return out


class CleaningReport(BaseModel):
    """Structured audit report describing every mutation applied during cleaning."""

    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    input_checksum: str
    output_checksum: str
    columns_dropped: list[str] = Field(default_factory=list)
    columns_added: list[str] = Field(default_factory=list)
    type_conversions: dict[str, dict[str, str]] = Field(default_factory=dict)
    missing_values_before: int = 0
    missing_values_after: int = 0
    imputed_columns: dict[str, int] = Field(default_factory=dict)
    exact_duplicates_removed: int = 0
    fuzzy_duplicates_removed: int = 0
    outliers_detected: dict[str, int] = Field(default_factory=dict)
    outlier_action_applied: str = "none"
    removed_rows_count: int = 0
    removed_rows_by_reason: dict[str, int] = Field(default_factory=dict)
    removed_row_samples: list[dict[str, Any]] = Field(default_factory=list)
    strings_cleaned_columns: list[str] = Field(default_factory=list)
    steps_executed: list[str] = Field(default_factory=list)


@dataclass
class _FittedCleanerState:
    dropped_cols: list[str] = field(default_factory=list)
    numeric_fill_values: dict[str, float] = field(default_factory=dict)
    causal_last_values: dict[str, float] = field(default_factory=dict)
    categorical_fill_values: dict[str, Any] = field(default_factory=dict)
    frequent_categories: dict[str, set[str]] = field(default_factory=dict)
    knn_imputer: KNNImputer | None = None
    knn_columns: list[str] = field(default_factory=list)
    iterative_imputer: IterativeImputer | None = None
    iterative_columns: list[str] = field(default_factory=list)
    missforest_models: dict[str, tuple[Any, list[str]]] = field(default_factory=dict)
    missing_indicator_cols: list[str] = field(default_factory=list)
    outlier_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    multivariate_detector: Any | None = None
    multivariate_columns: list[str] = field(default_factory=list)
    multivariate_medians: dict[str, float] = field(default_factory=dict)
    inferred_numeric_cols: list[str] = field(default_factory=list)
    inferred_datetime_cols: list[str] = field(default_factory=list)


class DataCleaner:
    """Leakage-safe, auditable data cleaning pipeline supporting fit/transform."""

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config = config or CleaningConfig()
        self._state = _FittedCleanerState()
        self._is_fitted = False
        self.last_report: CleaningReport | None = None
        self.last_removed_rows_: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def apply_domain_bounds(
        df: pd.DataFrame,
        domain_bounds: dict[str, tuple[float | None, float | None]],
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Convert physically impossible values outside explicit domain bounds to NaN (zero leakage, pre-split safe)."""
        work = df.copy()
        invalid_counts: dict[str, int] = {}
        for col, (lo, hi) in domain_bounds.items():
            if col not in work.columns or not pd.api.types.is_numeric_dtype(work[col]):
                continue
            ser = pd.to_numeric(work[col], errors="coerce")
            mask = pd.Series(False, index=work.index)
            if lo is not None:
                mask |= ser < float(lo)
            if hi is not None:
                mask |= ser > float(hi)
            cnt = int(mask.sum())
            if cnt > 0:
                work.loc[mask, col] = np.nan
                invalid_counts[str(col)] = cnt
        return work, invalid_counts

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        """Fit cleaning statistics on ``df`` and return cleaned data plus audit report."""
        self.fit(df)
        return self.transform(df)

    def fit_transform_split(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport, CleaningReport]:
        """Leakage-safe cleaning: fit ALL statistical parameters on ``df_train`` only, then transform Train and Test."""
        self.fit(df_train)
        clean_train, rep_train = self.transform(df_train)
        rem_tr = self.last_removed_rows_.copy()
        clean_test, rep_test = self.transform(df_test)
        rem_te = self.last_removed_rows_.copy()
        chunks = [c for c in (rem_tr, rem_te) if not c.empty]
        self.last_removed_rows_ = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        return clean_train, clean_test, rep_train, rep_test

    def fit(self, df: pd.DataFrame) -> DataCleaner:
        """Learn imputation values, type conversions, and outlier bounds strictly from training data."""
        if not isinstance(df, pd.DataFrame):
            raise DataCleaningError("DataCleaner expects a pandas DataFrame.")

        work = df.copy()
        self._state = _FittedCleanerState()
        excluded = set(self.config.exclude_columns)

        # Canonical Stage 1 & 2: Clean strings BEFORE standardizing missing sentinel tokens
        # so HTML/control-char/whitespace-wrapped sentinels (e.g. "<span>N/A</span>", " null ") become NaN.
        if self.config.clean_strings:
            work, _ = self._clean_strings(work, excluded)
        if self.config.standardize_missing_tokens:
            work = self._standardize_missing(work, excluded)

        # Canonical Stage 3: Infer and cast numeric/datetime column types strictly from Train
        if self.config.infer_types:
            work, _ = self._infer_and_cast_types(work, excluded, fit_mode=True)

        # Canonical Stage 4: Replace non-finite +/-inf values in numeric columns with NaN
        for col in work.select_dtypes(include=[np.number]).columns:
            if col not in excluded:
                arr_vals = work[col].to_numpy(dtype=float, na_value=np.nan)
                if np.isinf(arr_vals).any():
                    work[col] = work[col].replace([np.inf, -np.inf], np.nan)

        # Canonical Stage 5: Apply explicit physical/domain bounds
        if self.config.domain_bounds:
            work, _ = self.apply_domain_bounds(work, self.config.domain_bounds)

        # Canonical Stage 6: Drop columns exceeding missing-value threshold BEFORE row deduplication
        if self.config.drop_high_missing_cols and len(work) > 0:
            for col in work.columns:
                if col in excluded:
                    continue
                if float(work[col].isna().mean()) >= self.config.missing_col_threshold:
                    self._state.dropped_cols.append(str(col))
            work = work.drop(columns=self._state.dropped_cols, errors="ignore")

        # Canonical Stage 7 & 8: Exact and fuzzy deduplication matching transform() stage order
        if self.config.drop_exact_duplicates and len(work) > 0:
            work = work.drop_duplicates()
        if (
            self.config.fuzzy_dedup_column
            and self.config.fuzzy_dedup_column in work.columns
            and len(work) > 1
        ):
            work, _, _ = self._fuzzy_deduplicate(
                work, self.config.fuzzy_dedup_column, self.config.fuzzy_dedup_threshold
            )

        # Canonical Stage 9: Record pre-outlier missingness indicator columns
        if self.config.add_missing_indicators:
            for col in work.columns:
                if col not in excluded and work[col].isna().any():
                    self._state.missing_indicator_cols.append(str(col))

        base_cols = list(work.columns)
        num_cols = [
            c for c in work.select_dtypes(include=[np.number]).columns if c not in excluded
        ]
        cat_cols = [
            c
            for c in base_cols
            if c not in num_cols
            and c not in excluded
            and not pd.api.types.is_datetime64_any_dtype(work[c])
        ]

        # Canonical Stage 10: Learn univariate & multivariate outlier detectors strictly on Train,
        # then apply outlier treatment BEFORE fitting Stage 11 imputers (matching transform() order
        # so removed/clipped/NaN'd training outliers never contaminate learned imputation statistics).
        for col in num_cols:
            vals = work[col].dropna().astype(float)
            if len(vals) < 4:
                continue
            if self.config.outlier_method == "iqr":
                q1, q3 = float(vals.quantile(0.25)), float(vals.quantile(0.75))
                iqr = q3 - q1
                if iqr > 0:
                    self._state.outlier_bounds[str(col)] = (
                        q1 - self.config.outlier_threshold * iqr,
                        q3 + self.config.outlier_threshold * iqr,
                    )
            elif self.config.outlier_method == "zscore":
                mean, std = float(vals.mean()), float(vals.std(ddof=0))
                if std > 0:
                    self._state.outlier_bounds[str(col)] = (
                        mean - self.config.outlier_threshold * std,
                        mean + self.config.outlier_threshold * std,
                    )
            elif self.config.outlier_method == "mad":
                med = float(vals.median())
                mad = float(np.median(np.abs(vals - med)))
                if mad > 0:
                    scaled = 1.4826 * mad
                    self._state.outlier_bounds[str(col)] = (
                        med - self.config.outlier_threshold * scaled,
                        med + self.config.outlier_threshold * scaled,
                    )
            elif self.config.outlier_method == "percentile":
                lo_q = float(vals.quantile(self.config.lower_percentile))
                hi_q = float(vals.quantile(self.config.upper_percentile))
                self._state.outlier_bounds[str(col)] = (lo_q, hi_q)

        if self.config.outlier_method in ("isolation_forest", "lof", "elliptic", "elliptic_envelope") and num_cols and len(work) >= 10:
            mat = work[num_cols].copy()
            mv_meds: dict[str, float] = {}
            for c in num_cols:
                med = float(mat[c].median()) if mat[c].notna().any() else 0.0
                mv_meds[str(c)] = med
                mat[c] = mat[c].fillna(med)
            arr = mat.to_numpy(dtype=float)
            if self.config.outlier_method == "isolation_forest":
                det: Any = IsolationForest(
                    contamination=self.config.contamination,
                    random_state=42,
                    n_estimators=120,
                    n_jobs=parallel_n_jobs(),
                ).fit(arr)
            elif self.config.outlier_method in ("elliptic", "elliptic_envelope"):
                det = EllipticEnvelope(
                    contamination=self.config.contamination,
                    random_state=42,
                ).fit(arr)
            else:
                n_neighbors = min(20, max(2, len(work) - 1))
                det = LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    contamination=self.config.contamination,
                    novelty=True,
                    n_jobs=parallel_n_jobs(),
                ).fit(arr)
            self._state.multivariate_detector = det
            self._state.multivariate_columns = list(num_cols)
            self._state.multivariate_medians = mv_meds

        if self.config.outlier_method != "none" and self.config.outlier_action != "none" and len(work) > 0:
            work, _, outlier_flag_cols, _, _, _ = self._handle_outliers(work, excluded)
            if outlier_flag_cols:
                work = work.drop(columns=outlier_flag_cols, errors="ignore")

        # Canonical Stage 11: Fit causal anchors & univariate/multivariate imputers on post-outlier Train data
        ts_work = (
            work.sort_values(self.config.time_column, kind="mergesort")
            if (self.config.time_column and self.config.time_column in work.columns)
            else work
        )
        for c in num_cols:
            valid_c = ts_work[c].dropna()
            if len(valid_c) > 0:
                self._state.causal_last_values[str(c)] = float(valid_c.iloc[-1])

        # Always store fallback per-column numeric fill values (needed for leading NaNs in causal ffill too)
        for col in num_cols:
            non_null = work[col].dropna()
            if len(non_null) == 0:
                fill_val = self.config.constant_fill_numeric
            elif self.config.numeric_imputation == "mean":
                fill_val = float(non_null.mean())
            elif self.config.numeric_imputation == "constant":
                fill_val = self.config.constant_fill_numeric
            else:
                fill_val = float(non_null.median())
            self._state.numeric_fill_values[str(col)] = fill_val

        if self.config.numeric_imputation == "knn" and num_cols and len(work) >= 5:
            usable = [c for c in num_cols if not work[c].isna().all()]
            if usable:
                n_neighbors = min(5, max(1, len(work) - 1))
                imputer = KNNImputer(n_neighbors=n_neighbors)
                imputer.fit(work[usable])
                self._state.knn_imputer = imputer
                self._state.knn_columns = usable
        elif self.config.numeric_imputation == "iterative" and num_cols and len(work) >= 5:
            usable = [c for c in num_cols if not work[c].isna().all()]
            if usable:
                iter_imp = IterativeImputer(max_iter=10, random_state=42)
                iter_imp.fit(work[usable])
                self._state.iterative_imputer = iter_imp
                self._state.iterative_columns = usable
        elif self.config.numeric_imputation == "missforest" and num_cols and len(work) >= 5:
            from sklearn.ensemble import RandomForestRegressor

            usable = [c for c in num_cols if not work[c].isna().all()]
            if len(usable) >= 2:
                mf_work = work[usable].copy()
                for c in usable:
                    mf_work[c] = mf_work[c].fillna(self._state.numeric_fill_values.get(c, 0.0))
                for c in usable:
                    obs_mask = work[c].notna().to_numpy()
                    if obs_mask.sum() < 3:
                        continue
                    pred_cols = [oc for oc in usable if oc != c]
                    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                    rf.fit(mf_work.loc[obs_mask, pred_cols], work.loc[obs_mask, c].to_numpy(dtype=float))
                    self._state.missforest_models[c] = (rf, pred_cols)
        elif self.config.numeric_imputation in ("knn", "iterative", "missforest") and len(work) < 5:
            logger.warning(
                "imputation_fallback_to_median",
                requested=self.config.numeric_imputation,
                rows=len(work),
            )

        for col in cat_cols:
            non_null = work[col].dropna()
            if len(non_null) == 0 or self.config.categorical_imputation == "constant":
                self._state.categorical_fill_values[str(col)] = self.config.constant_fill_categorical
            else:
                mode_series = non_null.mode()
                self._state.categorical_fill_values[str(col)] = (
                    mode_series.iloc[0] if len(mode_series) > 0 else self.config.constant_fill_categorical
                )

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
        """Apply fitted cleaning rules to ``df`` and return cleaned DataFrame and audit report."""
        if not self._is_fitted:
            raise RuntimeError(
                "DataCleaner must be fitted before calling transform(). Call fit() or fit_transform() first."
            )

        input_shape = (int(df.shape[0]), int(df.shape[1]))
        input_checksum = compute_dataframe_checksum(df)
        missing_before = int(df.isna().sum().sum())
        steps: list[str] = []
        excluded = set(self.config.exclude_columns)

        work = df.copy()
        orig_row_ids = np.asarray(df.index.tolist(), dtype=object)
        removed_chunks: list[pd.DataFrame] = []
        removed_by_reason: dict[str, int] = {}

        def _capture_removed(
            sub_df: pd.DataFrame,
            orig_ids: np.ndarray,
            reasons: list[str] | str,
            *,
            stage: str = "Data Cleaning",
        ) -> None:
            if sub_df.empty:
                return
            rec = sub_df.copy().reset_index(drop=True)
            reason_list = [reasons] * len(rec) if isinstance(reasons, str) else list(reasons)
            rec.insert(0, "_deletion_reason", reason_list)
            rec.insert(0, "_deletion_stage", stage)
            rec.insert(0, "_original_row_index", list(orig_ids))
            removed_chunks.append(rec)
            for r in reason_list:
                removed_by_reason[r] = removed_by_reason.get(r, 0) + 1

        # Canonical Stage 1 & 2: Clean strings BEFORE standardizing missing tokens (identical to fit())
        cleaned_str_cols: list[str] = []
        if self.config.clean_strings:
            work, cleaned_str_cols = self._clean_strings(work, excluded)
            steps.append("clean_strings")

        if self.config.standardize_missing_tokens:
            work = self._standardize_missing(work, excluded)
            steps.append("standardize_missing_tokens")

        # Canonical Stage 3: Infer and cast types
        type_conversions: dict[str, dict[str, str]] = {}
        if self.config.infer_types:
            work, type_conversions = self._infer_and_cast_types(work, excluded, fit_mode=False)
            steps.append("infer_and_cast_types")

        # Canonical Stage 4: Replace non-finite +/-inf values in numeric columns with NaN
        nonfinite_replaced = 0
        for col in work.select_dtypes(include=[np.number]).columns:
            if col not in excluded:
                arr_vals = work[col].to_numpy(dtype=float, na_value=np.nan)
                inf_mask = np.isinf(arr_vals)
                if inf_mask.any():
                    nonfinite_replaced += int(inf_mask.sum())
                    work[col] = work[col].replace([np.inf, -np.inf], np.nan)
        if nonfinite_replaced > 0:
            steps.append("replace_nonfinite_with_nan")

        # Canonical Stage 5: Apply explicit physical/domain bounds
        if self.config.domain_bounds:
            work, domain_invalids = self.apply_domain_bounds(work, self.config.domain_bounds)
            if domain_invalids:
                steps.append(f"apply_domain_bounds:{sum(domain_invalids.values())}_coerced_to_nan")

        # Canonical Stage 6: Drop high-missing columns learned during fit()
        dropped_present = [c for c in self._state.dropped_cols if c in work.columns]
        if dropped_present:
            work = work.drop(columns=dropped_present)
            steps.append("drop_high_missing_columns")

        # Canonical Stage 7: Exact deduplication
        exact_dups_removed = 0
        if self.config.drop_exact_duplicates and len(work) > 0:
            dup_mask = work.duplicated()
            if dup_mask.any():
                dup_arr = dup_mask.to_numpy(dtype=bool)
                _capture_removed(
                    work.loc[dup_arr],
                    orig_row_ids[dup_arr],
                    "Exact Duplicate Row",
                    stage="Exact Deduplication",
                )
                work = work.loc[~dup_arr].copy()
                orig_row_ids = orig_row_ids[~dup_arr]
                exact_dups_removed = int(dup_arr.sum())
            steps.append("drop_exact_duplicates")

        # Canonical Stage 8: Fuzzy deduplication
        fuzzy_dups_removed = 0
        if (
            self.config.fuzzy_dedup_column
            and self.config.fuzzy_dedup_column in work.columns
            and len(work) > 1
        ):
            pre_fuzzy = work
            work, fuzzy_dups_removed, fuzzy_keep_mask = self._fuzzy_deduplicate(
                pre_fuzzy, self.config.fuzzy_dedup_column, self.config.fuzzy_dedup_threshold
            )
            if fuzzy_dups_removed > 0:
                rem_mask = ~np.asarray(fuzzy_keep_mask, dtype=bool)
                _capture_removed(
                    pre_fuzzy.loc[rem_mask],
                    orig_row_ids[rem_mask],
                    f"Fuzzy Duplicate ({self.config.fuzzy_dedup_column})",
                    stage="Fuzzy Deduplication",
                )
                orig_row_ids = orig_row_ids[~rem_mask]
            steps.append("fuzzy_deduplicate")

        # Canonical Stage 9: Add missingness indicator columns
        added_cols: list[str] = []
        if self.config.add_missing_indicators:
            for col in self._state.missing_indicator_cols:
                if col in work.columns:
                    ind_name = f"{col}__is_missing"
                    work[ind_name] = work[col].isna().astype(np.int8)
                    added_cols.append(ind_name)
            if added_cols:
                steps.append("add_missing_indicators")

        # Canonical Stage 10: Outlier detection & handling (before imputation)
        outliers_detected: dict[str, int] = {}
        if self.config.outlier_method != "none" and self.config.outlier_action != "none":
            work, outliers_detected, outlier_flags, out_rem_df, out_rem_pos, out_rem_reasons = self._handle_outliers(
                work, excluded
            )
            if not out_rem_df.empty and len(out_rem_pos) == len(out_rem_df):
                _capture_removed(
                    out_rem_df,
                    orig_row_ids[out_rem_pos],
                    out_rem_reasons,
                    stage="Outlier Removal",
                )
                keep_pos = np.ones(len(orig_row_ids), dtype=bool)
                keep_pos[out_rem_pos] = False
                orig_row_ids = orig_row_ids[keep_pos]
            added_cols.extend(outlier_flags)
            steps.append(f"outliers:{self.config.outlier_method}:{self.config.outlier_action}")

        # Canonical Stage 11: Impute remaining missing values using post-outlier Train statistics
        imputed_counts: dict[str, int] = {}
        work, imputed_counts = self._impute_missing(work, excluded)
        if imputed_counts:
            steps.append("impute_missing")

        if removed_chunks:
            self.last_removed_rows_ = pd.concat(removed_chunks, ignore_index=True)
        else:
            self.last_removed_rows_ = pd.DataFrame()

        removed_samples: list[dict[str, Any]] = []
        if not self.last_removed_rows_.empty:
            for rec_row in self.last_removed_rows_.head(50).to_dict(orient="records"):
                removed_samples.append(
                    {
                        str(k): (None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else (str(v) if isinstance(v, pd.Timestamp) else v))
                        for k, v in rec_row.items()
                    }
                )

        missing_after = int(work.isna().sum().sum())
        report = CleaningReport(
            input_shape=input_shape,
            output_shape=(int(work.shape[0]), int(work.shape[1])),
            input_checksum=input_checksum,
            output_checksum=compute_dataframe_checksum(work),
            columns_dropped=dropped_present,
            columns_added=added_cols,
            type_conversions=type_conversions,
            missing_values_before=missing_before,
            missing_values_after=missing_after,
            imputed_columns=imputed_counts,
            exact_duplicates_removed=exact_dups_removed,
            fuzzy_duplicates_removed=fuzzy_dups_removed,
            outliers_detected=outliers_detected,
            outlier_action_applied=self.config.outlier_action,
            removed_rows_count=int(len(self.last_removed_rows_)),
            removed_rows_by_reason=removed_by_reason,
            removed_row_samples=removed_samples,
            strings_cleaned_columns=cleaned_str_cols,
            steps_executed=steps,
        )
        self.last_report = report
        logger.info(
            "data_cleaned",
            rows_in=input_shape[0],
            rows_out=report.output_shape[0],
            missing_before=missing_before,
            missing_after=missing_after,
            duplicates_removed=exact_dups_removed + fuzzy_dups_removed,
        )
        return work, report

    def _standardize_missing(self, df: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
        for col in df.columns:
            if col in excluded:
                continue
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].apply(
                    lambda v: np.nan
                    if isinstance(v, str) and v.strip().lower() in _DEFAULT_MISSING_TOKENS
                    else v
                )
        return df

    def _clean_strings(self, df: pd.DataFrame, excluded: set[str]) -> tuple[pd.DataFrame, list[str]]:
        cleaned: list[str] = []
        for col in df.columns:
            if col in excluded:
                continue
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                def _clean_val(val: Any) -> Any:
                    if not isinstance(val, str):
                        return val
                    s = _HTML_TAG_RE.sub(" ", val) if self.config.strip_html else val
                    s = _CTRL_CHAR_RE.sub("", s)
                    s = _MULTI_SPACE_RE.sub(" ", s).strip()
                    if self.config.lowercase_strings:
                        s = s.lower()
                    return np.nan if s == "" else s

                df[col] = df[col].apply(_clean_val)
                cleaned.append(str(col))
        return df, cleaned

    def _infer_and_cast_types(
        self, df: pd.DataFrame, excluded: set[str], *, fit_mode: bool
    ) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
        conversions: dict[str, dict[str, str]] = {}
        for col in df.columns:
            if col in excluded:
                continue
            if not (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])):
                continue
            non_null = df[col].dropna().astype(str)
            if len(non_null) == 0:
                continue
            orig_dtype = str(df[col].dtype)

            # Attempt numeric coercion (including currency/percentage formatted strings)
            should_numeric = (
                col in self._state.inferred_numeric_cols
                if not fit_mode
                else float(non_null.apply(lambda s: bool(_CURRENCY_RE.match(s))).mean()) >= 0.85
            )
            if should_numeric:
                cleaned_num = non_null.str.replace(r"[$€£¥,\s%]", "", regex=True)
                converted = pd.to_numeric(
                    df[col].astype(str).str.replace(r"[$€£¥,\s%]", "", regex=True), errors="coerce"
                )
                if len(cleaned_num) > 0:
                    df[col] = converted
                    if fit_mode and str(col) not in self._state.inferred_numeric_cols:
                        self._state.inferred_numeric_cols.append(str(col))
                    conversions[str(col)] = {"from": orig_dtype, "to": str(df[col].dtype)}
                    continue

            # Attempt datetime coercion on date-like columns
            should_dt = col in self._state.inferred_datetime_cols
            if fit_mode and not should_dt:
                sample = non_null.head(30)
                if sample.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", regex=True).mean() >= 0.8:
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                    if parsed.notna().mean() >= 0.85:
                        should_dt = True
                        self._state.inferred_datetime_cols.append(str(col))
            if should_dt:
                df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
                conversions[str(col)] = {"from": orig_dtype, "to": str(df[col].dtype)}

        return df, conversions

    def _fuzzy_deduplicate(
        self, df: pd.DataFrame, column: str, threshold: float
    ) -> tuple[pd.DataFrame, int, list[bool]]:
        values = df[column].fillna("").astype(str).tolist()
        keep_mask = [True] * len(values)
        seen: list[str] = []
        removed = 0
        for idx, val in enumerate(values):
            norm = val.strip().lower()
            if not norm:
                continue
            is_dup = False
            for prev in seen:
                if abs(len(prev) - len(norm)) / max(len(prev), len(norm), 1) > (1.0 - threshold):
                    continue
                if SequenceMatcher(None, prev, norm).ratio() >= threshold:
                    is_dup = True
                    break
            if is_dup:
                keep_mask[idx] = False
                removed += 1
            else:
                seen.append(norm)
        return df.loc[keep_mask].copy(), removed, keep_mask

    def _handle_outliers(
        self, df: pd.DataFrame, excluded: set[str]
    ) -> tuple[pd.DataFrame, dict[str, int], list[str], pd.DataFrame, np.ndarray, list[str]]:
        detected: dict[str, int] = {}
        flag_cols: list[str] = []
        removed_df = pd.DataFrame()
        removed_pos = np.array([], dtype=int)
        removed_reasons: list[str] = []
        method = self.config.outlier_method
        action = self.config.outlier_action

        if method in ("iqr", "zscore", "mad", "percentile"):
            rows_to_drop = pd.Series(False, index=df.index)
            per_row_cols: dict[int, list[str]] = {}
            for col, (low, high) in self._state.outlier_bounds.items():
                if col not in df.columns or col in excluded:
                    continue
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                series = df[col].astype(float)
                mask = (series < low) | (series > high)
                count = int(mask.sum())
                if count > 0:
                    detected[col] = count
                    if action == "clip":
                        df[col] = series.clip(lower=low, upper=high)
                    elif action == "nan":
                        df.loc[mask, col] = np.nan
                    elif action == "remove":
                        rows_to_drop |= mask
                        for pos in np.flatnonzero(mask.to_numpy(dtype=bool)):
                            per_row_cols.setdefault(int(pos), []).append(col)
                    elif action == "flag":
                        flag_name = f"{col}__is_outlier"
                        df[flag_name] = mask.astype(np.int8)
                        flag_cols.append(flag_name)
            if action == "remove" and rows_to_drop.any():
                drop_arr = rows_to_drop.to_numpy(dtype=bool)
                removed_pos = np.flatnonzero(drop_arr)
                removed_df = df.iloc[removed_pos].copy()
                removed_reasons = [
                    f"Outlier Removed ({method.upper()}: {', '.join(per_row_cols.get(int(p), [])[:4])})"
                    for p in removed_pos
                ]
                df = df.loc[~drop_arr].copy()
            return df, detected, flag_cols, removed_df, removed_pos, removed_reasons

        # Multivariate outlier detection (Isolation Forest / LOF novelty=True) using Train-fitted detector
        if (
            method in ("isolation_forest", "lof", "elliptic", "elliptic_envelope")
            and self._state.multivariate_detector is not None
            and self._state.multivariate_columns
        ):
            cols_present = [c for c in self._state.multivariate_columns if c in df.columns]
            if len(cols_present) == len(self._state.multivariate_columns) and len(df) > 0:
                matrix = df[cols_present].copy()
                for c in cols_present:
                    med = self._state.multivariate_medians.get(c, 0.0)
                    matrix[c] = pd.to_numeric(matrix[c], errors="coerce").fillna(med)
                preds = self._state.multivariate_detector.predict(matrix.to_numpy(dtype=float))
                mask = pd.Series(preds == -1, index=df.index)
                count = int(mask.sum())
                if count > 0:
                    detected["__multivariate__"] = count
                    if action == "remove":
                        drop_arr = mask.to_numpy(dtype=bool)
                        removed_pos = np.flatnonzero(drop_arr)
                        removed_df = df.iloc[removed_pos].copy()
                        removed_reasons = [f"Multivariate Outlier Removed ({method.upper()})"] * len(removed_pos)
                        df = df.loc[~drop_arr].copy()
                    elif action == "flag":
                        df["__is_multivariate_outlier"] = mask.astype(np.int8)
                        flag_cols.append("__is_multivariate_outlier")
                    elif action == "nan":
                        df.loc[mask, cols_present] = np.nan
                    elif action == "clip":
                        # Clip flagged rows to Train IQR bounds if available or training medians
                        for c in cols_present:
                            med = self._state.multivariate_medians.get(c, 0.0)
                            df.loc[mask, c] = med
        return df, detected, flag_cols, removed_df, removed_pos, removed_reasons

    def _impute_missing(
        self, df: pd.DataFrame, excluded: set[str]
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        imputed: dict[str, int] = {}

        if (
            self.config.numeric_imputation == "knn"
            and self._state.knn_imputer is not None
            and self._state.knn_columns
        ):
            cols = [c for c in self._state.knn_columns if c in df.columns]
            if len(cols) == len(self._state.knn_columns):
                before_nulls = {c: int(df[c].isna().sum()) for c in cols}
                transformed = self._state.knn_imputer.transform(df[cols])
                df[cols] = transformed
                for c, cnt in before_nulls.items():
                    if cnt > 0:
                        imputed[c] = cnt
        elif (
            self.config.numeric_imputation == "iterative"
            and self._state.iterative_imputer is not None
            and self._state.iterative_columns
        ):
            cols = [c for c in self._state.iterative_columns if c in df.columns]
            if len(cols) == len(self._state.iterative_columns):
                before_nulls = {c: int(df[c].isna().sum()) for c in cols}
                transformed = self._state.iterative_imputer.transform(df[cols])
                df[cols] = transformed
                for c, cnt in before_nulls.items():
                    if cnt > 0:
                        imputed[c] = cnt
        elif self.config.numeric_imputation == "missforest" and self._state.missforest_models:
            num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in excluded]
            mf_in = df[num_cols].copy()
            for c in num_cols:
                mf_in[c] = mf_in[c].fillna(self._state.numeric_fill_values.get(c, 0.0))
            for c, (rf, pred_cols) in self._state.missforest_models.items():
                if c in df.columns and c not in excluded and all(pc in mf_in.columns for pc in pred_cols):
                    na_mask = df[c].isna().to_numpy()
                    if na_mask.any():
                        df.loc[na_mask, c] = rf.predict(mf_in.loc[na_mask, pred_cols])
                        imputed[c] = int(na_mask.sum())
        elif self.config.numeric_imputation in ("ffill", "bfill", "interpolate"):
            # Sort chronologically if time_column provided, restore index ordering afterwards
            orig_index = df.index
            was_sorted = False
            if self.config.time_column and self.config.time_column in df.columns:
                df = df.sort_values(self.config.time_column, kind="mergesort")
                was_sorted = True
            for col in df.select_dtypes(include=[np.number]).columns:
                if col in excluded:
                    continue
                nulls = int(df[col].isna().sum())
                if nulls == 0:
                    continue
                fallback = self._state.causal_last_values.get(
                    str(col),
                    self._state.numeric_fill_values.get(str(col), self.config.constant_fill_numeric),
                )
                if self.config.causal_time_series:
                    # Strictly causal past-only forward-fill anchored on Train's last valid value: NEVER peek into t+1
                    df[col] = df[col].ffill().fillna(fallback)
                elif self.config.numeric_imputation == "ffill":
                    df[col] = df[col].ffill().fillna(fallback)
                elif self.config.numeric_imputation == "bfill":
                    df[col] = df[col].bfill().fillna(fallback)
                else:
                    df[col] = df[col].interpolate(method="linear").ffill().fillna(fallback)
                imputed[str(col)] = nulls
            if was_sorted:
                df = df.loc[orig_index]
        elif self.config.numeric_imputation != "none":
            for col, val in self._state.numeric_fill_values.items():
                if col in df.columns and col not in excluded:
                    nulls = int(df[col].isna().sum())
                    if nulls > 0:
                        df[col] = df[col].fillna(val)
                        imputed[col] = nulls

        if self.config.categorical_imputation != "none":
            for col, val in self._state.categorical_fill_values.items():
                if col in df.columns and col not in excluded:
                    nulls = int(df[col].isna().sum())
                    if nulls > 0:
                        if isinstance(df[col].dtype, pd.CategoricalDtype):
                            if val not in df[col].cat.categories:
                                df[col] = df[col].cat.add_categories([val])
                        df[col] = df[col].fillna(val)
                        imputed[col] = nulls

        return df, imputed

    @staticmethod
    def multiple_imputation_rubin(
        df: pd.DataFrame,
        m: int = 5,
        max_iter: int = 10,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Real MICE (Multiple Imputation by Chained Equations) with Rubin's Rules pooling."""

        num_df = df.select_dtypes(include=[np.number]).copy()
        if num_df.empty or m < 2:
            raise ValueError("multiple_imputation_rubin requires numeric columns and m >= 2.")

        imputed_datasets: list[pd.DataFrame] = []
        means_per_m: list[np.ndarray] = []
        vars_per_m: list[np.ndarray] = []
        n = len(num_df)

        for i in range(m):
            imp = IterativeImputer(
                max_iter=max_iter,
                sample_posterior=True,
                random_state=random_state + i,
            )
            arr = imp.fit_transform(num_df)
            df_i = pd.DataFrame(arr, columns=num_df.columns, index=num_df.index)
            imputed_datasets.append(df_i)
            means_per_m.append(arr.mean(axis=0))
            vars_per_m.append(arr.var(axis=0, ddof=1) / max(n, 1))

        q_mat = np.vstack(means_per_m)
        u_mat = np.vstack(vars_per_m)
        q_bar = q_mat.mean(axis=0)
        u_bar = u_mat.mean(axis=0)
        b_var = q_mat.var(axis=0, ddof=1)
        total_var = u_bar + (1.0 + 1.0 / m) * b_var

        pooled = {
            str(col): {
                "pooled_mean": round(float(q_bar[j]), 6),
                "within_variance": round(float(u_bar[j]), 6),
                "between_variance": round(float(b_var[j]), 6),
                "total_variance": round(float(total_var[j]), 6),
                "pooled_se": round(float(np.sqrt(max(0.0, total_var[j]))), 6),
            }
            for j, col in enumerate(num_df.columns)
        }
        return {"m": int(m), "pooled_estimates": pooled, "imputed_datasets": imputed_datasets}

    @staticmethod
    def pool_rubin_estimates(
        estimates: list[np.ndarray],
        variances: list[np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Generic Rubin's Rules pooling for any downstream estimator (e.g. regression coefficients)."""
        m = len(estimates)
        if m < 2 or len(variances) != m:
            raise ValueError("pool_rubin_estimates requires m >= 2 estimates and matching variances.")
        q_mat = np.vstack([np.asarray(e, dtype=float) for e in estimates])
        u_mat = np.vstack([np.asarray(v, dtype=float) for v in variances])
        q_bar = q_mat.mean(axis=0)
        u_bar = u_mat.mean(axis=0)
        b_var = q_mat.var(axis=0, ddof=1)
        total_var = u_bar + (1.0 + 1.0 / m) * b_var
        return {
            "pooled_estimate": q_bar,
            "within_variance": u_bar,
            "between_variance": b_var,
            "total_variance": total_var,
            "pooled_se": np.sqrt(np.maximum(0.0, total_var)),
        }

    @staticmethod
    def missforest_impute(
        df: pd.DataFrame,
        max_iter: int = 5,
        n_estimators: int = 50,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """Iterative non-parametric Random Forest imputation (MissForest)."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        work = df.copy()
        cols_with_na = [c for c in work.columns if work[c].isna().any()]
        if not cols_with_na:
            return work

        num_cols = list(work.select_dtypes(include=[np.number]).columns)
        cat_cols = [c for c in work.columns if c not in num_cols]
        na_masks = {c: work[c].isna().to_numpy() for c in cols_with_na}

        # Initial fill
        for c in num_cols:
            if work[c].isna().any():
                med = work[c].median()
                work[c] = work[c].fillna(0.0 if pd.isna(med) else float(med))
        for c in cat_cols:
            if work[c].isna().any():
                mode_s = work[c].mode(dropna=True)
                _fv = "MISSING" if mode_s.empty else mode_s.iloc[0]
                if isinstance(work[c].dtype, pd.CategoricalDtype) and _fv not in work[c].cat.categories:
                    work[c] = work[c].cat.add_categories([_fv])
                work[c] = work[c].fillna(_fv)

        # Encode categoricals numerically for predictor matrix
        for _ in range(max_iter):
            for c in cols_with_na:
                mask = na_masks[c]
                if mask.all() or not mask.any():
                    continue
                other_cols = [oc for oc in work.columns if oc != c]
                if not other_cols:
                    continue
                X_enc = pd.get_dummies(work[other_cols], drop_first=False, dtype=float)
                X_obs = X_enc.loc[~mask]
                X_mis = X_enc.loc[mask]
                y_obs = work.loc[~mask, c]
                if c in num_cols:
                    rf = RandomForestRegressor(
                        n_estimators=n_estimators,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                    rf.fit(X_obs, y_obs.to_numpy(dtype=float))
                    work.loc[mask, c] = rf.predict(X_mis)
                else:
                    rf_c = RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=random_state,
                        n_jobs=-1,
                    )
                    rf_c.fit(X_obs, y_obs.astype(str))
                    work.loc[mask, c] = rf_c.predict(X_mis)
        return work


def benford_digit_test(series: pd.Series) -> dict[str, Any]:
    """First-digit Benford's Law conformity test (Chi-square) for financial/audit data screening."""
    from scipy import stats

    vals = pd.to_numeric(series, errors="coerce").dropna().abs()
    vals = vals[vals > 0]
    if len(vals) < 30:
        return {"evaluable": False, "reason": "Requires >= 30 positive observations.", "n": int(len(vals))}
    first_digits = vals.astype(str).str.lstrip("0.").str[0]
    digits = first_digits[first_digits.isin([str(d) for d in range(1, 10)])].astype(int)
    n = len(digits)
    if n < 30:
        return {"evaluable": False, "reason": "Insufficient valid leading digits.", "n": int(n)}
    obs = np.array([int((digits == d).sum()) for d in range(1, 10)], dtype=float)
    exp = np.array([n * np.log10(1.0 + 1.0 / d) for d in range(1, 10)], dtype=float)
    chi2_stat, p_val = stats.chisquare(obs, f_exp=exp)
    return {
        "evaluable": True,
        "n": int(n),
        "chi2": round(float(chi2_stat), 4),
        "p_value": round(float(p_val), 6),
        "conforms_benford": bool(p_val >= 0.05),
    }


def robust_mahalanobis_outliers(
    df: pd.DataFrame,
    *,
    alpha: float = 0.01,
    random_state: int = 42,
) -> dict[str, Any]:
    """Multivariate outlier detection using Minimum Covariance Determinant (MCD) Mahalanobis distance."""
    from scipy import stats
    from sklearn.covariance import MinCovDet

    num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()
    p = num.shape[1]
    if p < 2 or len(num) < max(20, p + 5):
        return {"evaluable": False, "reason": "Requires >= 2 numeric columns and sufficient complete rows."}
    X = num.to_numpy(dtype=float)
    try:
        mcd = MinCovDet(random_state=random_state).fit(X)
        dist_sq = mcd.mahalanobis(X)
    except Exception:
        cov = np.cov(X, rowvar=False) + np.eye(p) * 1e-6
        inv_cov = np.linalg.pinv(cov)
        diff = X - np.mean(X, axis=0)
        dist_sq = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    cutoff = float(stats.chi2.ppf(1.0 - alpha, df=p))
    flags = dist_sq > cutoff
    return {
        "evaluable": True,
        "n_rows": int(len(num)),
        "n_features": int(p),
        "chi2_cutoff": round(cutoff, 4),
        "n_outliers": int(flags.sum()),
        "outlier_indices": num.index[flags].tolist(),
    }


def cleaning_distortion_audit(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict[str, Any]:
    """Quantify statistical distortion introduced by data cleaning (KS, Wasserstein, median shift)."""
    from scipy import stats

    shared_num = [
        c
        for c in raw_df.select_dtypes(include=[np.number]).columns
        if c in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[c])
    ]
    per_col: dict[str, dict[str, float]] = {}
    for col in shared_num:
        r = pd.to_numeric(raw_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        c = pd.to_numeric(cleaned_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(r) < 5 or len(c) < 5:
            continue
        ks = stats.ks_2samp(r, c)
        w_dist = float(stats.wasserstein_distance(r, c))
        per_col[col] = {
            "ks_stat": round(float(ks.statistic), 6),
            "ks_pvalue": round(float(ks.pvalue), 6),
            "wasserstein": round(w_dist, 6),
            "median_shift": round(float(np.median(c) - np.median(r)), 6),
        }
    return {
        "columns_audited": len(per_col),
        "per_column": per_col,
        "high_distortion_columns": [k for k, v in per_col.items() if v["ks_pvalue"] < 0.01],
    }


def fuzzy_deduplicate_lsh_blocking(
    df: pd.DataFrame,
    cols: list[str],
    *,
    threshold: float = 0.85,
) -> tuple[pd.DataFrame, int]:
    """Near-linear fuzzy deduplication using prefix+length blocking and 3-gram Jaccard similarity."""
    if df.empty or not cols:
        return df.copy(), 0
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return df.copy(), 0

    def _shingles(text: str) -> set[str]:
        s = " ".join(text.lower().split())
        if len(s) < 3:
            return {s} if s else set()
        return {s[i : i + 3] for i in range(len(s) - 2)}

    blocks: dict[tuple[str, int], list[int]] = {}
    keys: list[str] = []
    for idx_pos, row_vals in enumerate(df[valid_cols].astype(str).itertuples(index=False, name=None)):
        txt = " | ".join(row_vals).strip().lower()
        keys.append(txt)
        b_key = (txt[:3], len(txt) // 4)
        blocks.setdefault(b_key, []).append(idx_pos)

    keep_mask = np.ones(len(df), dtype=bool)
    for idxs in blocks.values():
        if len(idxs) < 2:
            continue
        kept_shingles: list[set[str]] = []
        for pos in idxs:
            sh = _shingles(keys[pos])
            is_dup = False
            for ref in kept_shingles:
                union = len(sh | ref)
                if union > 0 and (len(sh & ref) / union) >= threshold:
                    is_dup = True
                    break
            if is_dup:
                keep_mask[pos] = False
            else:
                kept_shingles.append(sh)

    removed = int((~keep_mask).sum())
    return df.iloc[keep_mask].copy(), removed




