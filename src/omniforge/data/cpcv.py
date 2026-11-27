"""Canonical combinatorial purging and embargo kernel for positional splits."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations

import numpy as np
import pandas as pd

from omniforge.core.exceptions import EvaluationError


def iter_cpcv_splits(
    X: np.ndarray | pd.DataFrame,
    *,
    n_splits: int,
    n_test_groups: int,
    purge_gap: int,
    post_buffer: int,
    eval_times: np.ndarray | pd.Series | None,
    include_empty: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Purge overlapping label intervals and exclude explicit positional buffers."""
    n = len(X)
    blocks = np.array_split(np.arange(n), n_splits)
    t0_num: np.ndarray | None = None
    t1_num: np.ndarray | None = None
    if eval_times is not None:
        if len(eval_times) != n:
            raise EvaluationError(f"eval_times length ({len(eval_times)}) must match X length ({n}).")
        ev_ser = pd.Series(eval_times)
        if pd.api.types.is_datetime64_any_dtype(ev_ser):
            t1_num = ev_ser.astype("int64").to_numpy(dtype=float)
            if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
                t0_num = X.index.asi8.astype(float)
            else:
                t0_num = t1_num.copy()
        else:
            t1_num = np.asarray(ev_ser, dtype=float)
            if isinstance(X, pd.DataFrame) and pd.api.types.is_numeric_dtype(X.index.dtype):
                t0_num = np.asarray(X.index, dtype=float)
            else:
                t0_num = np.arange(n, dtype=float)

    all_indices = np.arange(n, dtype=int)
    for test_combo in combinations(range(n_splits), n_test_groups):
        test_idx = np.concatenate([blocks[g] for g in test_combo])
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[test_idx] = False
        for g in test_combo:
            blk = blocks[g]
            if len(blk) == 0:
                continue
            b_start, b_end = int(blk[0]), int(blk[-1])
            keep_mask[max(0, b_start - purge_gap) : b_start] = False
            keep_mask[b_end + 1 : min(n, b_end + 1 + post_buffer)] = False
            if t0_num is not None and t1_num is not None:
                test_t0 = float(np.min(t0_num[blk]))
                test_t1 = float(np.max(t1_num[blk]))
                # Overlap condition: [t0_i, t1_i] intersects [test_t0, test_t1] <=> t0_i <= test_t1 and t1_i >= test_t0
                overlap = (t0_num <= test_t1) & (t1_num >= test_t0)
                keep_mask[overlap] = False
        train_idx = all_indices[keep_mask]
        if include_empty or (len(train_idx) > 0 and len(test_idx) > 0):
            yield train_idx, test_idx

