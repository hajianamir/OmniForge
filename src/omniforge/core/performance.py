"""High-performance computing, caching, stratified downsampling, WebGL chart compaction & server-side pagination utilities."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Generic, TypeVar

T = TypeVar("T")

import numpy as np  # noqa: E402  # deliberate: TypeVar shim must precede imports for py3.11 compat
import pandas as pd  # noqa: E402  # deliberate: TypeVar shim must precede imports for py3.11 compat


def parallel_n_jobs(cap: int = 8) -> int:
    """Return a safe optimal CPU worker count capped for responsive desktop/server execution."""
    cpus = os.cpu_count() or 2
    return max(1, min(int(cpus), int(cap)))


def compute_df_fingerprint(df: pd.DataFrame) -> str:
    """Compute a deterministic, collision-safe fingerprint for a DataFrame.

    Full-content scheme ("v2"): the hash covers shape, column names/order + dtypes, and the
    Pandas row hashes of *every* row (not a 64-row subsample), plus a per-column null-count
    summary. blake2b-128 over the full payload makes accidental collisions
    negligible for cache-key purposes. The fingerprint is NOT a security boundary and NOT
    stable across Pandas versions' internal hashing, so it is used only for in-process caching.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(b"omniforge-df-fingerprint-v2|")
    h.update(f"{df.shape[0]}x{df.shape[1]}".encode())
    h.update("|".join(f"{c}:{dt}" for c, dt in zip(df.columns, df.dtypes, strict=False)).encode())
    n = len(df)
    if n > 0:
        try:
            h.update("|".join(str(int(v)) for v in df.isna().sum(axis=0).tolist()).encode())
        except Exception:  # duplicate labels or exotic dtypes — the row-hash lane still catches content changes
            pass
        try:
            row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype=np.uint64)
            h.update(row_hashes.tobytes())
        except Exception:
            # Exceptional path (exotic/unhashable dtypes): fall back to a full CSV payload hash.
            h.update(df.to_csv(index=True).encode("utf-8", errors="ignore"))
    return h.hexdigest()


class DataFrameCache:
    """Thread-safe LRU memoization cache keyed by full-content DataFrame fingerprint and parameter signature.

    ``ttl_seconds`` is opt-in (default ``None`` preserves historical unbounded-entry semantics other
    than the correctness fix of hashing all rows). When set, entries older than the TTL are evicted
    lazily on access and eagerly on write.
    """

    def __init__(self, max_entries: int = 64, ttl_seconds: float | None = None) -> None:
        self._max_entries = max(4, int(max_entries))
        self._ttl_seconds = None if ttl_seconds is None else max(0.0, float(ttl_seconds))
        self._lock = threading.RLock()
        self._store: OrderedDict[tuple[str, str, str], tuple[float, Any]] = OrderedDict()
        self._now = time.monotonic

    def _fresh(self, ts: float) -> bool:
        if self._ttl_seconds is None:
            return True
        return (self._now() - ts) <= self._ttl_seconds

    def get(self, namespace: str, df: pd.DataFrame, params: str = "") -> tuple[bool, Any]:
        key = (namespace, compute_df_fingerprint(df), params)
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                ts, value = entry
                if self._fresh(ts):
                    self._store.move_to_end(key)
                    return True, value
                self._store.pop(key, None)
        return False, None

    def set(self, namespace: str, df: pd.DataFrame, value: Any, params: str = "") -> None:
        key = (namespace, compute_df_fingerprint(df), params)
        with self._lock:
            self._store[key] = (self._now(), value)
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


GLOBAL_DF_CACHE = DataFrameCache(max_entries=96)


def smart_sample_df(
    df: pd.DataFrame,
    max_rows: int = 10_000,
    *,
    stratify_col: str | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, bool]:
    """Deterministically sample a large DataFrame preserving class/group proportions when possible.

    Returns:
        Tuple ``(result_df, was_sampled)``.
    """
    n = len(df)
    if n <= max_rows or max_rows <= 0:
        return df, False

    rng = np.random.default_rng(seed)
    if stratify_col and stratify_col in df.columns:
        vc = df[stratify_col].value_counts(dropna=False)
        if 2 <= len(vc) <= 50:
            parts: list[pd.DataFrame] = []
            for val, cnt in vc.items():
                take = max(1, int(round((cnt / n) * max_rows)))
                sub = df[df[stratify_col].isna()] if pd.isna(val) else df[df[stratify_col] == val]
                if len(sub) <= take:
                    parts.append(sub)
                else:
                    idx = rng.choice(len(sub), size=take, replace=False)
                    parts.append(sub.iloc[np.sort(idx)])
            sampled = pd.concat(parts, axis=0)
            if len(sampled) > max_rows:
                sampled = sampled.iloc[:max_rows]
            return sampled, True

    idx = np.sort(rng.choice(n, size=max_rows, replace=False))
    return df.iloc[idx], True


def lttb_downsample(
    x: np.ndarray | list[Any],
    y: np.ndarray | list[float],
    threshold: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Largest-Triangle-Three-Buckets (LTTB) downsampling preserving visual extrema in O(N) time."""
    x_arr = np.asarray(x)
    y_arr = np.asarray(y, dtype=float)
    n = len(y_arr)
    if threshold >= n or threshold < 3 or n <= 2:
        return x_arr, y_arr

    x_num = np.arange(n, dtype=float)
    out_idx = np.empty(threshold, dtype=int)
    out_idx[0] = 0
    out_idx[-1] = n - 1
    bucket_size = (n - 2) / (threshold - 2)
    a = 0

    for i in range(threshold - 2):
        avg_start = int(np.floor((i + 1) * bucket_size)) + 1
        avg_end = min(int(np.floor((i + 2) * bucket_size)) + 1, n)
        avg_x = float(np.mean(x_num[avg_start:avg_end])) if avg_end > avg_start else x_num[min(avg_start, n - 1)]
        avg_y = float(np.nanmean(y_arr[avg_start:avg_end])) if avg_end > avg_start else float(y_arr[min(avg_start, n - 1)])

        range_offs = int(np.floor(i * bucket_size)) + 1
        range_to = min(int(np.floor((i + 1) * bucket_size)) + 1, n - 1)
        if range_to <= range_offs:
            out_idx[i + 1] = range_offs
            a = range_offs
            continue

        px = x_num[a]
        py = y_arr[a]
        xs = x_num[range_offs:range_to]
        ys = np.nan_to_num(y_arr[range_offs:range_to], nan=py)
        areas = np.abs((px - avg_x) * (ys - py) - (px - xs) * (avg_y - py))
        best_rel = int(np.argmax(areas))
        a = range_offs + best_rel
        out_idx[i + 1] = a

    return x_arr[out_idx], y_arr[out_idx]


def optimize_plotly_dict(fig_dict: dict[str, Any], *, max_svg_points: int = 2500) -> dict[str, Any]:
    """Upgrade heavy scatter/line traces to GPU WebGL (scattergl) and round high-precision floats for compact JSON."""
    data = fig_dict.get("data")
    if isinstance(data, list):
        for trace in data:
            if not isinstance(trace, dict):
                continue
            ttype = trace.get("type", "scatter")
            x_vals = trace.get("x")
            y_vals = trace.get("y")
            n_pts = len(x_vals) if isinstance(x_vals, (list, np.ndarray)) else (len(y_vals) if isinstance(y_vals, (list, np.ndarray)) else 0)
            if ttype == "scatter" and n_pts > max_svg_points:
                trace["type"] = "scattergl"
            for axis_key in ("x", "y", "z"):
                vals = trace.get(axis_key)
                if isinstance(vals, np.ndarray):
                    if np.issubdtype(vals.dtype, np.floating):
                        trace[axis_key] = np.round(vals, 5).tolist()
                    else:
                        trace[axis_key] = vals.tolist()
    return fig_dict


def serialize_cell_value(v: Any) -> Any:
    """Safely serialize any NumPy/Pandas scalar or container into compact JSON-safe primitive."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        if np.isnan(fv) or np.isinf(fv):
            return None
        return round(fv, 6)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, (list, dict, tuple, set)):
        return str(v)
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def paginate_dataframe(
    df: pd.DataFrame,
    *,
    page: int = 1,
    page_size: int = 50,
    sort_by: str | None = None,
    sort_dir: str = "asc",
    search: str | None = None,
    columns: list[str] | None = None,
) -> dict[str, Any]:
    """Fast server-side DataFrame search, sorting & pagination for instant UI rendering at 1M+ rows."""
    view = df[columns] if columns else df
    total_rows = int(len(view))

    if search and search.strip():
        q = search.strip().lower()
        # Vectorized case-insensitive match across stringifiable columns (fast cap on full scan)
        mask = pd.Series(False, index=view.index)
        for col in view.columns[:40]:
            try:
                mask = mask | view[col].astype(str).str.lower().str.contains(q, regex=False, na=False)
            except Exception:
                continue
        view = view[mask]

    filtered_rows = int(len(view))

    if sort_by and sort_by in view.columns and filtered_rows > 0:
        asc = str(sort_dir).lower() != "desc"
        try:
            view = view.sort_values(by=sort_by, ascending=asc, kind="mergesort", na_position="last")
        except Exception:
            pass

    p_size = max(1, min(int(page_size), 500))
    total_pages = max(1, int(np.ceil(filtered_rows / p_size)))
    cur_page = max(1, min(int(page), total_pages))
    start = (cur_page - 1) * p_size
    end = start + p_size
    page_df = view.iloc[start:end]

    col_meta = [
        {
            "name": str(c),
            "dtype": str(view[c].dtype),
            "is_numeric": bool(pd.api.types.is_numeric_dtype(view[c]) and not pd.api.types.is_bool_dtype(view[c])),
        }
        for c in view.columns
    ]

    records = [
        {str(k): serialize_cell_value(v) for k, v in row.items()}
        for row in page_df.to_dict(orient="records")
    ]

    return {
        "columns": col_meta,
        "rows": records,
        "page": cur_page,
        "page_size": p_size,
        "total_rows": total_rows,
        "filtered_rows": filtered_rows,
        "total_pages": total_pages,
    }


dataframe_fingerprint = compute_df_fingerprint


class TTLCache(Generic[T]):  # noqa: UP046  # deliberate: Generic[T] subclass for py3.11 compat
    """Lightweight thread-safe LRU + TTL cache supporting arbitrary keyed values."""

    def __init__(self, max_entries: int = 128, ttl_seconds: float = 300.0) -> None:
        self._max_entries = max(4, int(max_entries))
        self._ttl_seconds = float(ttl_seconds)
        self._lock = threading.RLock()
        self._store: OrderedDict[str, tuple[float, T]] = OrderedDict()

    def get(self, key: str) -> T | None:
        import time

        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, value = entry
            if now - ts > self._ttl_seconds:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: T) -> None:
        import time

        with self._lock:
            self._store[key] = (time.monotonic(), value)
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)


def downsample_table_for_plot(df: pd.DataFrame, max_rows: int = 15_000, random_seed: int = 42) -> pd.DataFrame:
    """Deterministically cap a DataFrame to ``max_rows`` for fast chart/report rendering."""
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=random_seed).sort_index()


def compact_scatter_payload(
    x: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    max_points: int = 5000,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Produce a compact scatter payload, omitting non-finite coordinate pairs rather than inventing zeros."""
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    n = min(len(x_arr), len(y_arr))
    x_arr, y_arr = x_arr[:n], y_arr[:n]
    g_arr = np.asarray(groups, dtype=object).ravel()[:n] if groups is not None else None

    indices = np.flatnonzero(np.isfinite(x_arr) & np.isfinite(y_arr))
    downsampled = len(indices) > max_points
    if downsampled:
        rng = np.random.default_rng(random_seed)
        indices = np.sort(rng.choice(indices, size=max_points, replace=False))
    selected_groups = [g_arr[i] if i < len(g_arr) else None for i in indices] if g_arr is not None else None

    return {
        "x": np.round(x_arr[indices], 4).tolist(),
        "y": np.round(y_arr[indices], 4).tolist(),
        "groups": selected_groups,
        "total_points": int(n),
        "rendered_points": int(len(indices)),
        "downsampled": bool(downsampled),
    }

