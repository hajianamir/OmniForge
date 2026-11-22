"""Leakage-safe automated and manual feature engineering for tabular and time-series data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler

from omniforge.core.exceptions import FeatureEngineeringError
from omniforge.core.logging import get_logger

logger = get_logger(__name__)


class FeatureConfig(BaseModel):
    """Configuration for :class:`FeatureEngineeringPipeline`."""

    scale_numeric: str = "standard"  # standard, robust, minmax, none
    correct_skew: bool = False
    skew_threshold: float = 1.5
    max_onehot_cardinality: int = 12
    categorical_encoding: str = "auto"  # auto, onehot, frequency, target
    target_encoding_smoothing: float = 10.0
    create_interactions: bool = False
    max_interaction_features: int = 5
    create_polynomials: bool = False
    polynomial_degree: int = 2
    extract_datetime_features: bool = True
    add_cyclical_time: bool = True
    time_series_column: str | None = None
    time_series_target: str | None = None
    time_series_group_col: str | None = None
    lags: list[int] = Field(default_factory=list)
    rolling_windows: list[int] = Field(default_factory=list)
    fourier_periods: list[int] = Field(default_factory=list)
    fourier_order: int = 2
    group_aggregations: dict[str, list[str]] = Field(default_factory=dict)
    exclude_columns: list[str] = Field(default_factory=list)
    random_seed: int = 42


@dataclass
class _FittedFeatureState:
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    datetime_cols: list[str] = field(default_factory=list)
    skewed_cols: list[str] = field(default_factory=list)
    skewed_medians: dict[str, float] = field(default_factory=dict)
    raw_numeric_medians: dict[str, float] = field(default_factory=dict)
    power_transformer: PowerTransformer | None = None
    scaler: Any = None
    scaled_cols: list[str] = field(default_factory=list)
    onehot_categories: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    freq_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    target_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    global_target_mean: float = 0.0
    interaction_cols: list[str] = field(default_factory=list)
    group_agg_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    group_agg_defaults: dict[str, float] = field(default_factory=dict)
    column_medians: dict[str, float] = field(default_factory=dict)
    output_columns: list[str] = field(default_factory=list)
    train_row_count: int = 0
    time_origin_ns: float | None = None
    time_step_ns: float | None = None


class FeatureEngineeringPipeline:
    """Comprehensive fit/transform feature engineering engine for tabular & time-series data."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self._state = _FittedFeatureState()
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> FeatureEngineeringPipeline:
        """Fit encoders, scalers, target statistics, and interaction selectors on ``X``."""
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError("FeatureEngineeringPipeline requires a pandas DataFrame.")

        self._state = _FittedFeatureState()
        self._state.train_row_count = len(X)
        excluded = set(self.config.exclude_columns)

        for col in X.columns:
            if col in excluded:
                continue
            s = X[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                self._state.datetime_cols.append(str(col))
            elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                col_s = str(col)
                self._state.numeric_cols.append(col_s)
                num_vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
                self._state.raw_numeric_medians[col_s] = float(num_vals.median()) if num_vals.notna().any() else 0.0
            else:
                self._state.categorical_cols.append(str(col))

        if self.config.time_series_column and self.config.time_series_column in X.columns:
            ts_parsed = pd.to_datetime(X[self.config.time_series_column], errors="coerce").dropna()
            if len(ts_parsed) >= 2:
                ns_vals = ts_parsed.astype("datetime64[ns]").astype("int64").to_numpy(dtype=float)
                diffs = np.diff(np.sort(ns_vals))
                pos_diffs = diffs[diffs > 0]
                if len(pos_diffs) > 0:
                    self._state.time_origin_ns = float(ns_vals[0])
                    self._state.time_step_ns = float(np.median(pos_diffs))

        y_num: np.ndarray | None = None
        y_is_discrete = False
        if y is not None:
            y_series = pd.Series(y)
            if pd.api.types.is_numeric_dtype(y_series) and not pd.api.types.is_bool_dtype(y_series):
                y_num = y_series.to_numpy(dtype=float)
                non_na_y = pd.Series(y_num).dropna()
                y_is_discrete = int(non_na_y.nunique()) <= 15 and bool(np.all(np.isclose(non_na_y, np.round(non_na_y))))
            else:
                y_num = pd.factorize(y_series)[0].astype(float)
                y_is_discrete = True
            self._state.global_target_mean = float(np.nanmean(y_num)) if len(y_num) > 0 else 0.0

        # Categorical encodings (with shuffled/stratified OOF target encoding cached for leak-free training transform)
        oof_target_enc: dict[str, np.ndarray] = {}
        for col in self._state.categorical_cols:
            _s_raw = X[col]
            if isinstance(_s_raw.dtype, pd.CategoricalDtype) and "MISSING" not in _s_raw.cat.categories:
                _s_raw = _s_raw.cat.add_categories(["MISSING"])
            s = _s_raw.fillna("MISSING").astype(str)
            n_uniq = int(s.nunique())
            enc_mode = self.config.categorical_encoding
            if enc_mode == "onehot" or (enc_mode == "auto" and n_uniq <= self.config.max_onehot_cardinality):
                cats = sorted(s.unique().tolist())[: self.config.max_onehot_cardinality]
                used_col_names: set[str] = set()
                cat_pairs: list[tuple[str, str]] = []
                for idx_c, cat in enumerate(cats):
                    base_slug = "".join(ch if ch.isalnum() else "_" for ch in str(cat))[:32] or "empty"
                    cand_col = f"{col}__{base_slug}"
                    if cand_col in used_col_names:
                        cand_col = f"{col}__{base_slug}_{idx_c}"
                    used_col_names.add(cand_col)
                    cat_pairs.append((cat, cand_col))
                self._state.onehot_categories[col] = cat_pairs
            else:
                freqs = (s.value_counts(normalize=True)).to_dict()
                self._state.freq_maps[col] = {str(k): float(v) for k, v in freqs.items()}
                if (enc_mode == "target" or enc_mode == "auto") and y_num is not None and len(y_num) == len(s):
                    smooth = float(self.config.target_encoding_smoothing)
                    gmean = self._state.global_target_mean
                    tmp = pd.DataFrame({"cat": s.to_numpy(), "y": y_num})
                    stats_df = tmp.groupby("cat")["y"].agg(["count", "mean"])
                    smoothed = (stats_df["count"] * stats_df["mean"] + smooth * gmean) / (
                        stats_df["count"] + smooth
                    )
                    self._state.target_maps[col] = {str(k): float(v) for k, v in smoothed.items()}

                    # Shuffled/Stratified Out-of-Fold Empirical Bayes smoothing (Micci-Barreca 2001)
                    n_rows = len(s)
                    n_folds = min(5, max(2, n_rows // 4))
                    oof_vals = np.full(n_rows, gmean, dtype=float)
                    if n_rows >= 4:
                        cat_arr = s.to_numpy()
                        valid_y = np.nan_to_num(y_num, nan=gmean)
                        if y_is_discrete:
                            _, cls_counts = np.unique(valid_y, return_counts=True)
                            min_cls = int(cls_counts.min()) if len(cls_counts) > 0 else 0
                        else:
                            min_cls = 0
                        if y_is_discrete and min_cls >= n_folds:
                            cv_iter = StratifiedKFold(
                                n_splits=n_folds, shuffle=True, random_state=self.config.random_seed
                            ).split(np.zeros(n_rows), valid_y.astype(int))
                        else:
                            cv_iter = KFold(
                                n_splits=n_folds, shuffle=True, random_state=self.config.random_seed
                            ).split(np.zeros(n_rows))
                        for tr_idx, va_idx in cv_iter:
                            f_gmean = float(np.nanmean(y_num[tr_idx])) if len(tr_idx) > 0 else gmean
                            f_df = pd.DataFrame({"cat": cat_arr[tr_idx], "y": y_num[tr_idx]})
                            f_stats = f_df.groupby("cat")["y"].agg(["count", "mean"])
                            f_smooth = (f_stats["count"] * f_stats["mean"] + smooth * f_gmean) / (
                                f_stats["count"] + smooth
                            )
                            f_map = {str(k): float(v) for k, v in f_smooth.items()}
                            oof_vals[va_idx] = [f_map.get(str(v), f_gmean) for v in cat_arr[va_idx]]
                    oof_target_enc[col] = oof_vals

        # Group aggregations
        for group_col, agg_cols in self.config.group_aggregations.items():
            if group_col in X.columns:
                valid_aggs = [
                    c
                    for c in agg_cols
                    if c in X.columns
                    and pd.api.types.is_numeric_dtype(X[c])
                    and not pd.api.types.is_bool_dtype(X[c])
                ]
                if valid_aggs:
                    agg_tbl = X.groupby(group_col, observed=True)[valid_aggs].agg(["mean", "std"])
                    agg_tbl.columns = [f"{group_col}_grp_{c}_{stat}" for c, stat in agg_tbl.columns]
                    for c in valid_aggs:
                        col_num = pd.to_numeric(X[c], errors="coerce").dropna()
                        g_mean = float(col_num.mean()) if len(col_num) > 0 else 0.0
                        g_std = float(col_num.std(ddof=1)) if len(col_num) > 1 else 0.0
                        self._state.group_agg_defaults[f"{group_col}_grp_{c}_mean"] = (
                            g_mean if np.isfinite(g_mean) else 0.0
                        )
                        self._state.group_agg_defaults[f"{group_col}_grp_{c}_std"] = (
                            g_std if np.isfinite(g_std) else 0.0
                        )
                    self._state.group_agg_tables[group_col] = agg_tbl.fillna(0.0)

        # Interaction / polynomial selection: pick highest-variance numeric features
        if (self.config.create_interactions or self.config.create_polynomials) and len(self._state.numeric_cols) >= 1:
            variances = X[self._state.numeric_cols].var(numeric_only=True).sort_values(ascending=False)
            self._state.interaction_cols = [str(c) for c in variances.index[: self.config.max_interaction_features]]

        # Skew correction
        if self.config.correct_skew and self._state.numeric_cols:
            for col in self._state.numeric_cols:
                vals = X[col].dropna().astype(float)
                if len(vals) >= 8 and abs(float(vals.skew())) >= self.config.skew_threshold:
                    self._state.skewed_cols.append(col)
            if self._state.skewed_cols:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                meds = {
                    col: (float(X[col].median()) if X[col].notna().any() else 0.0)
                    for col in self._state.skewed_cols
                }
                self._state.skewed_medians = meds
                mat = X[self._state.skewed_cols].copy()
                for col, med_val in meds.items():
                    mat[col] = pd.to_numeric(mat[col], errors="coerce").fillna(med_val)
                pt.fit(mat)
                self._state.power_transformer = pt

        # Build engineered training preview using leak-free OOF target encodings to fit medians & scaler cleanly
        self._is_fitted = True
        self._last_train_oof_target_enc: dict[str, np.ndarray] = oof_target_enc
        engineered_unscaled = self._transform_internal(
            X, apply_scaling=False, fit_medians=True, oof_target_override=oof_target_enc, fourier_offset=0
        )
        num_after = [
            str(c)
            for c in engineered_unscaled.select_dtypes(include=[np.number]).columns
            if c not in excluded
        ]
        self._state.scaled_cols = num_after
        if self.config.scale_numeric != "none" and num_after:
            scaler_map: dict[str, Any] = {
                "standard": StandardScaler(),
                "robust": RobustScaler(),
                "minmax": MinMaxScaler(),
            }
            scaler = scaler_map.get(self.config.scale_numeric, StandardScaler())
            mat = engineered_unscaled[num_after].replace([np.inf, -np.inf], np.nan)
            for c in num_after:
                if mat[c].isna().any():
                    mat[c] = mat[c].fillna(self._state.column_medians.get(c, 0.0))
            scaler.fit(mat)
            self._state.scaler = scaler

        self._state.output_columns = [str(c) for c in engineered_unscaled.columns]
        logger.info(
            "feature_pipeline_fitted",
            input_cols=X.shape[1],
            output_cols=len(self._state.output_columns),
        )
        return self

    def transform(self, X: pd.DataFrame, *, fourier_offset: int | None = None) -> pd.DataFrame:
        """Transform ``X`` using the fitted feature engineering pipeline (strictly Train-anchored statistics)."""
        if not self._is_fitted:
            raise FeatureEngineeringError("FeatureEngineeringPipeline must be fitted before transform().")
        eff_offset = self._state.train_row_count if fourier_offset is None else int(fourier_offset)
        return self._transform_internal(X, apply_scaling=True, fit_medians=False, fourier_offset=eff_offset)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
        """Fit pipeline on ``X`` (and optional ``y``) and return leak-free OOF-encoded training features."""
        self.fit(X, y)
        return self._transform_internal(
            X,
            apply_scaling=True,
            fit_medians=False,
            oof_target_override=getattr(self, "_last_train_oof_target_enc", None),
            fourier_offset=0,
        )

    def fit_transform_split(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series | np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Leakage-free split transformation: OOF target encoding on ``X_train``, full-train map on ``X_test``."""
        X_train_out = self.fit_transform(X_train, y_train)
        return X_train_out, self.transform(X_test)

    def _transform_internal(
        self,
        X: pd.DataFrame,
        *,
        apply_scaling: bool,
        fit_medians: bool = False,
        oof_target_override: dict[str, np.ndarray] | None = None,
        fourier_offset: int = 0,
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        excluded = set(self.config.exclude_columns)

        # Preserve excluded columns as-is when present
        for col in self.config.exclude_columns:
            if col in X.columns:
                out[col] = X[col]

        # Numeric columns (missing raw columns filled with fitted training medians so downstream schema never breaks)
        if self._state.numeric_cols:
            num_df = pd.DataFrame(index=X.index)
            for col in self._state.numeric_cols:
                if col in X.columns:
                    num_df[col] = pd.to_numeric(X[col], errors="coerce")
                else:
                    num_df[col] = self._state.raw_numeric_medians.get(col, 0.0)
            if self._state.power_transformer is not None and self._state.skewed_cols:
                present = [c for c in self._state.skewed_cols if c in num_df.columns]
                if len(present) == len(self._state.skewed_cols):
                    filled = num_df[present].copy()
                    for c in present:
                        filled[c] = filled[c].fillna(self._state.skewed_medians.get(c, 0.0))
                    num_df[present] = self._state.power_transformer.transform(filled)
            for col in num_df.columns:
                out[col] = num_df[col]

        # Datetime features
        if self.config.extract_datetime_features:
            for col in self._state.datetime_cols:
                dt = (
                    pd.to_datetime(X[col], errors="coerce")
                    if col in X.columns
                    else pd.Series(pd.NaT, index=X.index, dtype="datetime64[ns]")
                )
                out[f"{col}_year"] = dt.dt.year.fillna(0).astype(np.int32)
                out[f"{col}_month"] = dt.dt.month.fillna(1).astype(np.int16)
                out[f"{col}_day"] = dt.dt.day.fillna(1).astype(np.int16)
                out[f"{col}_dayofweek"] = dt.dt.day_of_week.fillna(0).astype(np.int16)
                out[f"{col}_dayofyear"] = dt.dt.day_of_year.fillna(1).astype(np.int16)
                out[f"{col}_quarter"] = dt.dt.quarter.fillna(1).astype(np.int16)
                out[f"{col}_is_weekend"] = (dt.dt.day_of_week >= 5).fillna(False).astype(np.int8)
                if self.config.add_cyclical_time:
                    out[f"{col}_month_sin"] = np.sin(2.0 * np.pi * out[f"{col}_month"] / 12.0)
                    out[f"{col}_month_cos"] = np.cos(2.0 * np.pi * out[f"{col}_month"] / 12.0)
                    out[f"{col}_dow_sin"] = np.sin(2.0 * np.pi * out[f"{col}_dayofweek"] / 7.0)
                    out[f"{col}_dow_cos"] = np.cos(2.0 * np.pi * out[f"{col}_dayofweek"] / 7.0)

        # Categorical features
        for col in self._state.categorical_cols:
            if col in X.columns:
                _s_raw = X[col]
                if isinstance(_s_raw.dtype, pd.CategoricalDtype) and "MISSING" not in _s_raw.cat.categories:
                    _s_raw = _s_raw.cat.add_categories(["MISSING"])
                s = _s_raw.fillna("MISSING").astype(str)
            else:
                s = pd.Series("MISSING", index=X.index, dtype=str)
            if col in self._state.onehot_categories:
                for cat, out_col_name in self._state.onehot_categories[col]:
                    out[out_col_name] = (s == cat).astype(np.float32)
            if col in self._state.freq_maps:
                fmap = self._state.freq_maps[col]
                out[f"{col}__freq"] = s.map(lambda v, _m=fmap: _m.get(v, 0.0)).astype(np.float32)
            if col in self._state.target_maps:
                if oof_target_override is not None and col in oof_target_override and len(oof_target_override[col]) == len(s):
                    out[f"{col}__target_enc"] = oof_target_override[col].astype(np.float32)
                else:
                    tmap = self._state.target_maps[col]
                    gmean = self._state.global_target_mean
                    out[f"{col}__target_enc"] = s.map(lambda v, _m=tmap, _g=gmean: _m.get(v, _g)).astype(np.float32)

        # Pairwise interactions & ratios (fill missing values with fitted training medians before interaction)
        if self.config.create_interactions and len(self._state.interaction_cols) >= 2:
            cols = [c for c in self._state.interaction_cols if c in out.columns]
            for i, col_a in enumerate(cols):
                med_a = self._state.raw_numeric_medians.get(col_a, 0.0)
                va = out[col_a].fillna(med_a).astype(float)
                for col_b in cols[i + 1 :]:
                    med_b = self._state.raw_numeric_medians.get(col_b, 0.0)
                    vb = out[col_b].fillna(med_b).astype(float)
                    out[f"{col_a}_x_{col_b}"] = va * vb
                    out[f"{col_a}_div_{col_b}"] = va / (np.abs(vb) + 1e-6)

        # Polynomial terms
        if self.config.create_polynomials and self._state.interaction_cols:
            for col in self._state.interaction_cols:
                if col in out.columns:
                    med_c = self._state.raw_numeric_medians.get(col, 0.0)
                    v = out[col].fillna(med_c).astype(float)
                    for deg in range(2, self.config.polynomial_degree + 1):
                        out[f"{col}_pow{deg}"] = v**deg

        # Group aggregations (unseen groups fall back to global training statistics, not 0.0)
        for group_col, agg_tbl in self._state.group_agg_tables.items():
            if group_col in X.columns:
                joined = X[[group_col]].merge(agg_tbl, left_on=group_col, right_index=True, how="left")
                for c in agg_tbl.columns:
                    default_val = self._state.group_agg_defaults.get(str(c), 0.0)
                    out[c] = joined[c].fillna(default_val).to_numpy(dtype=float)
            else:
                for c in agg_tbl.columns:
                    out[c] = self._state.group_agg_defaults.get(str(c), 0.0)

        # Time-series lag, rolling, and Fourier features
        ts_target = self.config.time_series_target
        if ts_target and (ts_target in X.columns or not fit_medians):
            for lag in self.config.lags:
                if lag <= 0:
                    raise ValueError(
                        f"Causal time-series lag must be > 0 to prevent future data leakage (got lag={lag})."
                    )
            if ts_target in X.columns:
                series = pd.to_numeric(X[ts_target], errors="coerce")
                grp_col = self.config.time_series_group_col
                if grp_col and grp_col in X.columns:
                    grouped = X.groupby(grp_col, observed=True)[ts_target]
                    for lag in self.config.lags:
                        out[f"{ts_target}_lag_{lag}"] = grouped.shift(lag)
                    for win in self.config.rolling_windows:
                        shifted = grouped.shift(1)
                        out[f"{ts_target}_roll_mean_{win}"] = shifted.groupby(X[grp_col], observed=True).transform(
                            lambda s, _w=win: s.rolling(_w, min_periods=1).mean()
                        )
                        out[f"{ts_target}_roll_std_{win}"] = shifted.groupby(X[grp_col], observed=True).transform(
                            lambda s, _w=win: s.rolling(_w, min_periods=1).std().fillna(0.0)
                        )
                    if self.config.lags:
                        out[f"{ts_target}_diff_1"] = grouped.shift(1) - grouped.shift(2)
                else:
                    for lag in self.config.lags:
                        out[f"{ts_target}_lag_{lag}"] = series.shift(lag)
                    for win in self.config.rolling_windows:
                        shifted = series.shift(1)
                        out[f"{ts_target}_roll_mean_{win}"] = shifted.rolling(win, min_periods=1).mean()
                        out[f"{ts_target}_roll_std_{win}"] = shifted.rolling(win, min_periods=1).std().fillna(0.0)
                        out[f"{ts_target}_roll_min_{win}"] = shifted.rolling(win, min_periods=1).min()
                        out[f"{ts_target}_roll_max_{win}"] = shifted.rolling(win, min_periods=1).max()
                    if self.config.lags:
                        out[f"{ts_target}_diff_1"] = series.shift(1).diff(1)

        if self.config.fourier_periods:
            ts_col = self.config.time_series_column
            if (
                ts_col
                and ts_col in X.columns
                and self._state.time_origin_ns is not None
                and self._state.time_step_ns is not None
                and self._state.time_step_ns > 0
            ):
                ts_vals = pd.to_datetime(X[ts_col], errors="coerce")
                ns_arr = ts_vals.astype("datetime64[ns]").astype("int64").to_numpy(dtype=float)
                fallback_idx = np.arange(fourier_offset, fourier_offset + len(X), dtype=float)
                t_idx = np.where(
                    ts_vals.notna().to_numpy(),
                    (ns_arr - self._state.time_origin_ns) / self._state.time_step_ns,
                    fallback_idx,
                )
            else:
                t_idx = np.arange(fourier_offset, fourier_offset + len(X), dtype=float)
            for period in self.config.fourier_periods:
                for k in range(1, self.config.fourier_order + 1):
                    angle = 2.0 * np.pi * k * t_idx / max(period, 1)
                    out[f"fourier_p{period}_sin_{k}"] = np.sin(angle)
                    out[f"fourier_p{period}_cos_{k}"] = np.cos(angle)

        # Ensure any expected non-excluded output column missing at transform time is populated with training median
        if not fit_medians and self._state.output_columns:
            for expected_col in self._state.output_columns:
                if expected_col not in excluded and expected_col not in out.columns:
                    out[expected_col] = self._state.column_medians.get(expected_col, 0.0)

        out = out.replace([np.inf, -np.inf], np.nan)
        for col in out.select_dtypes(include=[np.number]).columns:
            col_s = str(col)
            if col_s in excluded:
                continue
            if fit_medians:
                med = float(out[col].median()) if out[col].notna().any() else 0.0
                self._state.column_medians[col_s] = med
            else:
                med = self._state.column_medians.get(col_s, 0.0)
            if out[col].isna().any():
                out[col] = out[col].fillna(med)

        if apply_scaling and self._state.scaler is not None and self._state.scaled_cols:
            for c in self._state.scaled_cols:
                if c not in out.columns:
                    out[c] = self._state.column_medians.get(c, 0.0)
            out[self._state.scaled_cols] = self._state.scaler.transform(out[self._state.scaled_cols])

        if not fit_medians and self._state.output_columns:
            ordered = [c for c in self._state.output_columns if (c in out.columns)]
            extra_excl = [c for c in out.columns if c not in set(ordered)]
            out = out[ordered + extra_excl]

        return out
