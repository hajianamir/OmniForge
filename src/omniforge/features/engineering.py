"""Leakage-safe automated and manual feature engineering for tabular and time-series data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
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


@dataclass
class _FittedFeatureState:
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    datetime_cols: list[str] = field(default_factory=list)
    skewed_cols: list[str] = field(default_factory=list)
    skewed_medians: dict[str, float] = field(default_factory=dict)
    power_transformer: PowerTransformer | None = None
    scaler: Any = None
    scaled_cols: list[str] = field(default_factory=list)
    onehot_categories: dict[str, list[str]] = field(default_factory=dict)
    freq_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    target_maps: dict[str, dict[str, float]] = field(default_factory=dict)
    global_target_mean: float = 0.0
    interaction_cols: list[str] = field(default_factory=list)
    group_agg_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    column_medians: dict[str, float] = field(default_factory=dict)
    output_columns: list[str] = field(default_factory=list)


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
        excluded = set(self.config.exclude_columns)

        for col in X.columns:
            if col in excluded:
                continue
            s = X[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                self._state.datetime_cols.append(str(col))
            elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                self._state.numeric_cols.append(str(col))
            else:
                self._state.categorical_cols.append(str(col))

        y_num: np.ndarray | None = None
        if y is not None:
            y_series = pd.Series(y)
            if pd.api.types.is_numeric_dtype(y_series):
                y_num = y_series.to_numpy(dtype=float)
            else:
                y_num = pd.factorize(y_series)[0].astype(float)
            self._state.global_target_mean = float(np.nanmean(y_num)) if len(y_num) > 0 else 0.0

        # Categorical encodings (with out-of-fold target encoding cached for leak-free training transform)
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
                self._state.onehot_categories[col] = cats
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

                    # Out-of-fold Empirical Bayes smoothing (Micci-Barreca 2001) to prevent training self-leakage
                    n_rows = len(s)
                    n_folds = min(5, max(2, n_rows // 4))
                    oof_vals = np.full(n_rows, gmean, dtype=float)
                    if n_rows >= 4:
                        fold_ids = np.arange(n_rows) % n_folds
                        cat_arr = s.to_numpy()
                        for f_idx in range(n_folds):
                            tr_mask = fold_ids != f_idx
                            va_mask = fold_ids == f_idx
                            f_gmean = float(np.nanmean(y_num[tr_mask])) if np.any(tr_mask) else gmean
                            f_df = pd.DataFrame({"cat": cat_arr[tr_mask], "y": y_num[tr_mask]})
                            f_stats = f_df.groupby("cat")["y"].agg(["count", "mean"])
                            f_smooth = (f_stats["count"] * f_stats["mean"] + smooth * f_gmean) / (
                                f_stats["count"] + smooth
                            )
                            f_map = {str(k): float(v) for k, v in f_smooth.items()}
                            oof_vals[va_mask] = [f_map.get(str(v), f_gmean) for v in cat_arr[va_mask]]
                    oof_target_enc[col] = oof_vals

        # Group aggregations
        for group_col, agg_cols in self.config.group_aggregations.items():
            if group_col in X.columns:
                valid_aggs = [c for c in agg_cols if c in X.columns and pd.api.types.is_numeric_dtype(X[c])]
                if valid_aggs:
                    agg_tbl = X.groupby(group_col, observed=True)[valid_aggs].agg(["mean", "std"])
                    agg_tbl.columns = [f"{group_col}_grp_{c}_{stat}" for c, stat in agg_tbl.columns]
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
            X, apply_scaling=False, fit_medians=True, oof_target_override=oof_target_enc
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
            mat = engineered_unscaled[num_after].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            scaler.fit(mat)
            self._state.scaler = scaler

        self._state.output_columns = [str(c) for c in engineered_unscaled.columns]
        logger.info(
            "feature_pipeline_fitted",
            input_cols=X.shape[1],
            output_cols=len(self._state.output_columns),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform ``X`` using the fitted feature engineering pipeline (strictly Train-anchored statistics)."""
        if not self._is_fitted:
            raise FeatureEngineeringError("FeatureEngineeringPipeline must be fitted before transform().")
        return self._transform_internal(X, apply_scaling=True, fit_medians=False)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
        """Fit pipeline on ``X`` (and optional ``y``) and return leak-free OOF-encoded training features."""
        self.fit(X, y)
        return self._transform_internal(
            X, apply_scaling=True, fit_medians=False, oof_target_override=getattr(self, "_last_train_oof_target_enc", None)
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
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)

        # Preserve excluded columns as-is
        for col in self.config.exclude_columns:
            if col in X.columns:
                out[col] = X[col]

        # Numeric columns (with optional skew correction using fitted training medians)
        if self._state.numeric_cols:
            num_df = X[[c for c in self._state.numeric_cols if c in X.columns]].copy()
            for col in num_df.columns:
                num_df[col] = pd.to_numeric(num_df[col], errors="coerce")
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
                if col not in X.columns:
                    continue
                dt = pd.to_datetime(X[col], errors="coerce")
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
            if col not in X.columns:
                continue
            _s_raw = X[col]
            if isinstance(_s_raw.dtype, pd.CategoricalDtype) and "MISSING" not in _s_raw.cat.categories:
                _s_raw = _s_raw.cat.add_categories(["MISSING"])
            s = _s_raw.fillna("MISSING").astype(str)
            if col in self._state.onehot_categories:
                for cat in self._state.onehot_categories[col]:
                    safe_cat = "".join(ch if ch.isalnum() else "_" for ch in str(cat))[:32]
                    out[f"{col}__{safe_cat}"] = (s == cat).astype(np.float32)
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

        # Pairwise interactions & ratios
        if self.config.create_interactions and len(self._state.interaction_cols) >= 2:
            cols = [c for c in self._state.interaction_cols if c in out.columns]
            for i, col_a in enumerate(cols):
                for col_b in cols[i + 1 :]:
                    va = out[col_a].fillna(0.0).astype(float)
                    vb = out[col_b].fillna(0.0).astype(float)
                    out[f"{col_a}_x_{col_b}"] = va * vb
                    out[f"{col_a}_div_{col_b}"] = va / (np.abs(vb) + 1e-6)

        # Polynomial terms
        if self.config.create_polynomials and self._state.interaction_cols:
            for col in self._state.interaction_cols:
                if col in out.columns:
                    v = out[col].fillna(0.0).astype(float)
                    for deg in range(2, self.config.polynomial_degree + 1):
                        out[f"{col}_pow{deg}"] = v**deg

        # Group aggregations
        for group_col, agg_tbl in self._state.group_agg_tables.items():
            if group_col in X.columns:
                joined = X[[group_col]].merge(agg_tbl, left_on=group_col, right_index=True, how="left")
                for c in agg_tbl.columns:
                    out[c] = joined[c].fillna(0.0).to_numpy(dtype=float)

        # Time-series lag, rolling, and Fourier features
        ts_target = self.config.time_series_target
        if ts_target and ts_target in X.columns:
            for lag in self.config.lags:
                if lag <= 0:
                    raise ValueError(
                        f"Causal time-series lag must be > 0 to prevent future data leakage (got lag={lag})."
                    )
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
                    # Strictly causal lagged first difference: y_{t-1} - y_{t-2} (never uses current y_t)
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
                    # Strictly causal lagged first difference: y_{t-1} - y_{t-2} (never leaks current y_t)
                    out[f"{ts_target}_diff_1"] = series.shift(1).diff(1)

            if self.config.fourier_periods:
                t_idx = np.arange(len(X), dtype=float)
                for period in self.config.fourier_periods:
                    for k in range(1, self.config.fourier_order + 1):
                        angle = 2.0 * np.pi * k * t_idx / max(period, 1)
                        out[f"fourier_p{period}_sin_{k}"] = np.sin(angle)
                        out[f"fourier_p{period}_cos_{k}"] = np.cos(angle)

        out = out.replace([np.inf, -np.inf], np.nan)
        for col in out.select_dtypes(include=[np.number]).columns:
            col_s = str(col)
            if fit_medians:
                med = float(out[col].median()) if out[col].notna().any() else 0.0
                self._state.column_medians[col_s] = med
            else:
                med = self._state.column_medians.get(col_s, 0.0)
            if out[col].isna().any():
                out[col] = out[col].fillna(med)

        if apply_scaling and self._state.scaler is not None and self._state.scaled_cols:
            present_scaled = [c for c in self._state.scaled_cols if c in out.columns]
            if len(present_scaled) == len(self._state.scaled_cols):
                out[present_scaled] = self._state.scaler.transform(out[present_scaled])

        return out
