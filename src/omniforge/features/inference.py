"""Fitted workbench input preparation shared by training and prediction.

AutoML backends receive normalized raw columns and keep their own fitted encoder /
 imputer. Numeric-only workbench models receive a fitted, stable numeric encoding.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np
import pandas as pd


class WorkbenchInputPreprocessor:
    """Remember the input schema and, for numeric-only models, fitted encodings.

    Missing numeric values use training medians in ``numeric`` mode. Unseen
    categories become all-zero one-hot vectors or -1 for high-cardinality columns.
    ``raw`` mode leaves imputation and category encoding to the fitted model.
    """

    def __init__(self, *, mode: Literal["raw", "numeric"] = "raw") -> None:
        self.mode = mode
        self.input_columns: list[str] = []
        self.column_types: dict[str, str] = {}
        self.medians: dict[str, float] = {}
        self.categories: dict[str, list[str]] = {}
        self.output_columns: list[str] = []
        self._fitted = False

    @staticmethod
    def _missing_as_nan(series: pd.Series) -> pd.Series:
        result = series.astype(object)
        return result.where(result.notna() & (result != ""), np.nan)

    @classmethod
    def _number(cls, series: pd.Series, column: str) -> pd.Series:
        values = cls._missing_as_nan(series)
        numeric = pd.to_numeric(values, errors="coerce").astype(float)
        invalid = values.notna() & numeric.isna()
        if invalid.any():
            raise ValueError(f"Feature '{column}' must contain numeric values.")
        return numeric.replace([np.inf, -np.inf], np.nan)

    @classmethod
    def _datetime(cls, series: pd.Series, column: str) -> pd.Series:
        values = cls._missing_as_nan(series)
        # Numeric input for a datetime column is interpreted as Unix seconds;
        # ISO timestamps (including timezone offsets) follow the training path.
        is_number = values.map(
            lambda value: (
                isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
            ) or (isinstance(value, str) and bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value.strip())))
        )
        parsed = pd.to_datetime(values.where(~is_number), errors="coerce", utc=True, format="mixed")
        if is_number.any():
            seconds = pd.to_numeric(values.loc[is_number], errors="coerce")
            parsed.loc[is_number] = pd.to_datetime(seconds, errors="coerce", utc=True, unit="s")
        invalid = values.notna() & parsed.isna()
        if invalid.any():
            raise ValueError(f"Feature '{column}' must contain valid datetimes.")
        seconds = parsed.astype("datetime64[ns, UTC]").astype("int64").astype(float) / 1e9
        return seconds.where(parsed.notna(), np.nan)

    @classmethod
    def _category(cls, series: pd.Series) -> pd.Series:
        values = cls._missing_as_nan(series)
        return values.map(lambda value: str(value) if pd.notna(value) else np.nan).astype(object)

    def fit(self, frame: pd.DataFrame) -> WorkbenchInputPreprocessor:
        if not frame.columns.is_unique:
            raise ValueError("Feature column names must be unique.")
        self.input_columns = [str(column) for column in frame.columns]
        if list(frame.columns) != self.input_columns:
            raise ValueError("Feature column names must be strings.")
        self.column_types = {}
        self.medians = {}
        self.categories = {}
        for column in self.input_columns:
            series = frame[column]
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                kind = "datetime"
            elif pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
                kind = "numeric"
            else:
                kind = "categorical"
            self.column_types[column] = kind
            if self.mode == "numeric":
                if kind == "categorical":
                    values = self._category(series).fillna("__MISSING__")
                    # Preserve factorize order for large-cardinality columns.
                    self.categories[column] = list(dict.fromkeys(values.tolist()))
                else:
                    values = self._datetime(series, column) if kind == "datetime" else self._number(series, column)
                    self.medians[column] = float(values.median()) if values.notna().any() else 0.0
        self._fitted = True
        self.output_columns = list(self._transform(frame).columns)
        return self

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not frame.columns.is_unique:
            raise ValueError("Feature column names must be unique.")
        aligned = frame.reindex(columns=self.input_columns)
        normalized: dict[str, pd.Series] = {}
        for column in self.input_columns:
            kind = self.column_types[column]
            series = aligned[column]
            if kind == "datetime":
                normalized[column] = self._datetime(series, column)
            elif kind == "numeric":
                normalized[column] = self._number(series, column)
            else:
                normalized[column] = self._category(series)
        if self.mode == "raw":
            return pd.DataFrame(normalized, index=frame.index)

        result: dict[str, Any] = {}
        # Numeric and datetime features come first, just as in the original workbench preparation.
        for column in self.input_columns:
            if self.column_types[column] != "categorical":
                result[column] = normalized[column].fillna(self.medians[column])
        for column in self.input_columns:
            if self.column_types[column] != "categorical":
                continue
            values = normalized[column].fillna("__MISSING__")
            categories = self.categories[column]
            if len(categories) <= 20:
                for category in sorted(categories):
                    name = f"{column}_{category}"
                    if name in result:
                        raise ValueError(f"Encoded feature name '{name}' conflicts with another feature.")
                    result[name] = (values == category).astype(float)
            else:
                name = f"{column}_code"
                if name in result:
                    raise ValueError(f"Encoded feature name '{name}' conflicts with another feature.")
                mapping = {value: index for index, value in enumerate(categories)}
                result[name] = values.map(mapping).fillna(-1).astype(float)
        return pd.DataFrame(result, index=frame.index)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the same fitted schema to raw records in training order."""
        if not self._fitted:
            raise ValueError("Input preprocessor has not been fitted.")
        result = self._transform(frame)
        return result.reindex(columns=self.output_columns)

    def feature_sources(self) -> dict[str, str]:
        """Map each fitted output feature to its original input column."""
        if not self._fitted:
            raise ValueError("Input preprocessor has not been fitted.")
        sources = {}
        for column in self.input_columns:
            if self.mode == "raw" or self.column_types[column] != "categorical":
                sources[column] = column
            elif len(self.categories[column]) > 20:
                sources[f"{column}_code"] = column
            else:
                sources.update({f"{column}_{value}": column for value in self.categories[column]})
        return sources
