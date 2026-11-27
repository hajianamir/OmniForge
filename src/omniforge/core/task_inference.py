"""Shared supervised-task inference for tabular Workbench modelling and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.core.types import TaskType


def infer_supervised_task(target: pd.Series | np.ndarray) -> TaskType:
    """Categorical, boolean or at most ten finite integer levels imply classification."""
    observed = pd.Series(target).dropna().drop_duplicates()
    levels = len(observed)
    classification = not pd.api.types.is_numeric_dtype(observed) or pd.api.types.is_bool_dtype(observed)
    if not classification and levels <= 10 and not pd.api.types.is_complex_dtype(observed):
        numeric = observed.to_numpy(dtype=float)
        classification = bool(np.isfinite(numeric).all() and (numeric == np.trunc(numeric)).all())
    if classification:
        return TaskType.BINARY_CLASSIFICATION if levels <= 2 else TaskType.MULTICLASS_CLASSIFICATION
    return TaskType.REGRESSION
