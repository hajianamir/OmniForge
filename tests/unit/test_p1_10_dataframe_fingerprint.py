"""P1-10 regression tests: full-content DataFrame fingerprint + TTL cache."""

from __future__ import annotations

import pandas as pd
import pytest

from omniforge.core.performance import DataFrameCache, compute_df_fingerprint


def _frame(rows: int = 200, cols: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        [list(range(i * cols, (i + 1) * cols)) for i in range(rows)],
        columns=[f"c{i}" for i in range(cols)],
    )


def test_fingerprint_stable_same_content() -> None:
    df = _frame()
    assert compute_df_fingerprint(df) == compute_df_fingerprint(df.copy())


def test_fingerprint_detects_change_outside_old_sample_window() -> None:
    """The old 64-row linspace sample missed edits in most of a big frame (rows ~65..n-2)."""
    df = _frame(rows=200, cols=4)
    fp = compute_df_fingerprint(df)
    edited = df.copy()
    edited.iloc[150, 2] = edited.iloc[150, 2] + 10_000
    assert compute_df_fingerprint(edited) != fp


def test_fingerprint_detects_single_cell_in_tail_row() -> None:
    df = _frame(rows=500, cols=3)
    fp = compute_df_fingerprint(df)
    edited = df.copy()
    edited.iloc[499, 0] = edited.iloc[499, 0] + 1
    assert compute_df_fingerprint(edited) != fp


def test_fingerprint_distinguishes_column_order_dtype_nulls_and_shape() -> None:
    df = _frame()
    fp = compute_df_fingerprint(df)
    assert compute_df_fingerprint(df[["c1", "c0", "c2", "c3"]]) != fp  # column order
    dtype_changed = df.copy()
    dtype_changed["c0"] = dtype_changed["c0"].astype("float64")
    assert compute_df_fingerprint(dtype_changed) != fp  # dtype lane
    nulled = df.copy()
    nulled.loc[0, "c0"] = None
    assert compute_df_fingerprint(nulled) != fp  # null lane
    assert compute_df_fingerprint(df.iloc[:100]) != fp  # shape lane


def test_fingerprint_object_columns_do_not_crash() -> None:
    df = pd.DataFrame({"a": ["x", "y", None], "b": [1, 2, 3]})
    fp = compute_df_fingerprint(df)
    assert isinstance(fp, str) and len(fp) == 32
    fp2 = compute_df_fingerprint(df.assign(a=["x", "y", "q"]))
    assert fp2 != fp


def test_fingerprint_deterministic_across_calls_and_instances() -> None:
    df = _frame(rows=64, cols=2)
    assert compute_df_fingerprint(df) == compute_df_fingerprint(pd.DataFrame(df.to_numpy(), columns=df.columns))


def test_dataframe_cache_hit_miss_and_params_namespacing() -> None:
    df = _frame(rows=50, cols=2)
    cache = DataFrameCache(max_entries=8)
    cache.set("eda", df, "expensive-result", params="k=5")
    assert cache.get("eda", df, params="k=5") == (True, "expensive-result")
    assert cache.get("eda", df, params="k=6") == (False, None)
    assert cache.get("other", df, params="k=5") == (False, None)
    edited = df.copy()
    edited.iloc[10, 0] += 1
    assert cache.get("eda", edited, params="k=5") == (False, None)


def test_dataframe_cache_ttl_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _frame(rows=20, cols=2)
    clock = {"t": 1000.0}
    cache = DataFrameCache(max_entries=8, ttl_seconds=30.0)
    monkeypatch.setattr(cache, "_now", lambda: clock["t"])
    cache.set("ns", df, "v")
    assert cache.get("ns", df) == (True, "v")
    clock["t"] += 31.0
    assert cache.get("ns", df) == (False, None)


def test_dataframe_cache_ttl_disabled_by_default() -> None:
    df = _frame(rows=10, cols=1)
    cache = DataFrameCache(max_entries=4)
    cache.set("ns", df, "v")
    assert cache.get("ns", df) == (True, "v")
