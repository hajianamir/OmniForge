"""General-purpose utilities for reproducibility, hashing, memory, and hardware inspection."""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import platform
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from omniforge.core.constants import DEFAULT_RANDOM_SEED


def set_global_seed(seed: int = DEFAULT_RANDOM_SEED) -> int:
    """Seed Python, NumPy, and optional deep learning frameworks for reproducibility.

    Args:
        seed: Non-negative integer seed.

    Returns:
        The seed value applied.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if is_package_available("torch"):
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return seed


def is_package_available(package_name: str) -> bool:
    """Check whether an optional third-party Python package or dotted submodule is importable.

    Args:
        package_name: Module import name (e.g. ``"torch"``, ``"autogluon.tabular"``).

    Returns:
        ``True`` if the package spec is found, ``False`` otherwise.
    """
    try:
        return importlib.util.find_spec(package_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def compute_bytes_checksum(data: bytes) -> str:
    """Compute a SHA-256 hex digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_file_checksum(file_path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Stream-hash a file on disk using SHA-256."""
    hasher = hashlib.sha256()
    path = Path(file_path)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dataframe_checksum(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 content checksum for a pandas DataFrame.

    Args:
        df: DataFrame to fingerprint.

    Returns:
        64-character SHA-256 hexadecimal digest covering schema, index, and values.
    """
    hasher = hashlib.sha256()
    schema_repr = "|".join(f"{col}:{dtype}" for col, dtype in zip(df.columns, df.dtypes, strict=False))
    hasher.update(schema_repr.encode("utf-8"))
    if len(df) > 0:
        try:
            row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype=np.uint64)
            hasher.update(row_hashes.tobytes())
        except TypeError:
            serialized = df.astype(str).to_csv(index=True).encode("utf-8")
            hasher.update(serialized)
    return hasher.hexdigest()


def optimize_dataframe_dtypes(
    df: pd.DataFrame,
    category_threshold: float = 0.5,
    downcast_floats: bool = False,
) -> pd.DataFrame:
    """Reduce DataFrame memory footprint safely without losing semantic fidelity.

    Args:
        df: Input DataFrame.
        category_threshold: Convert object columns with unique ratio strictly below
            this threshold (and at least 10 rows) to ``category`` dtype.
        downcast_floats: If ``True``, downcast ``float64`` to ``float32`` when range fits.

    Returns:
        Memory-optimized copy of ``df``.
    """
    optimized = df.copy()
    n_rows = len(optimized)

    for col in optimized.columns:
        series = optimized[col]
        if pd.api.types.is_integer_dtype(series) and not pd.api.types.is_bool_dtype(series):
            if series.isna().any():
                continue
            col_min = int(series.min())
            col_max = int(series.max())
            if col_min >= 0:
                if col_max <= np.iinfo(np.uint8).max:
                    optimized[col] = series.astype(np.uint8)
                elif col_max <= np.iinfo(np.uint16).max:
                    optimized[col] = series.astype(np.uint16)
                elif col_max <= np.iinfo(np.uint32).max:
                    optimized[col] = series.astype(np.uint32)
            else:
                if np.iinfo(np.int8).min <= col_min and col_max <= np.iinfo(np.int8).max:
                    optimized[col] = series.astype(np.int8)
                elif np.iinfo(np.int16).min <= col_min and col_max <= np.iinfo(np.int16).max:
                    optimized[col] = series.astype(np.int16)
                elif np.iinfo(np.int32).min <= col_min and col_max <= np.iinfo(np.int32).max:
                    optimized[col] = series.astype(np.int32)
        elif downcast_floats and pd.api.types.is_float_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                f32_max = float(np.finfo(np.float32).max)
                if float(non_null.abs().max()) < f32_max * 0.5:
                    optimized[col] = series.astype(np.float32)
        elif (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and n_rows >= 20:
            try:
                unique_ratio = series.nunique(dropna=True) / max(n_rows, 1)
                if 0 < unique_ratio < category_threshold:
                    optimized[col] = series.astype("category")
            except TypeError:
                # Unhashable elements (dicts/lists) stay as object dtype
                continue

    return optimized


def safe_json_default(obj: Any) -> Any:
    """JSON serialization fallback for NumPy, pandas, datetime, Path, and Pydantic types."""
    if obj is None or obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return to_json_safe(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return to_json_safe(obj.tolist())
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "model_dump"):
        return to_json_safe(obj.model_dump())
    if isinstance(obj, set):
        return [to_json_safe(v) for v in sorted(list(obj), key=str)]
    return str(obj)


def to_json_object(data: Any) -> dict[str, Any]:
    """``to_json_safe`` narrowed to a JSON object (empty dict for non-mapping input)."""
    safe = to_json_safe(data)
    return safe if isinstance(safe, dict) else {}


def to_json_list(data: Any) -> list[Any]:
    """``to_json_safe`` narrowed to a JSON array (empty list for non-sequence input)."""
    safe = to_json_safe(data)
    return safe if isinstance(safe, list) else []


def to_json_safe(data: Any) -> Any:
    """Deep-convert an arbitrary structure into strictly RFC-8259 JSON-compliant primitives (no NaN/Inf/NaT).

    The return value is JSON-native (dict/list/str/int/float/bool/None) but is
    deliberately annotated ``Any`` because the concrete shape depends on the
    input. Endpoints that must expose a stricter contract should funnel their
    payload through :func:`to_json_object` / :func:`to_json_list`.
    """
    if data is None or data is pd.NA or data is pd.NaT:
        return None
    if isinstance(data, bool):
        return data
    if isinstance(data, int):
        return data
    if isinstance(data, float):
        return None if (np.isnan(data) or np.isinf(data)) else data
    if isinstance(data, str):
        return data
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, (np.floating,)):
        val = float(data)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(data, (np.bool_,)):
        return bool(data)
    if isinstance(data, dict):
        return {str(k): to_json_safe(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_json_safe(v) for v in data]
    if isinstance(data, set):
        return [to_json_safe(v) for v in sorted(list(data), key=str)]
    if isinstance(data, np.ndarray):
        return [to_json_safe(v) for v in data.tolist()]
    if isinstance(data, pd.DataFrame):
        return to_json_safe(data.to_dict(orient="records"))
    if isinstance(data, pd.Series):
        return to_json_safe(data.tolist())
    if isinstance(data, (pd.Timestamp, datetime, date)):
        try:
            if pd.isna(data):
                return None
        except Exception:
            pass
        return data.isoformat()
    if isinstance(data, Path):
        return str(data)
    if hasattr(data, "model_dump"):
        return to_json_safe(data.model_dump())
    try:
        if pd.isna(data):
            return None
    except Exception:
        pass
    return str(data)


def ensure_directory(path: str | Path) -> Path:
    """Create directory (and parents) if missing and return its resolved Path."""
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """Context manager measuring elapsed wall-clock seconds."""
    state: dict[str, float] = {"elapsed_seconds": 0.0}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["elapsed_seconds"] = time.perf_counter() - start


def detect_hardware_profile() -> dict[str, Any]:
    """Inspect local CPU, RAM, OS, and GPU compute resources."""
    vm = psutil.virtual_memory()
    gpu_available = False
    gpu_name: str | None = None
    gpu_count = 0

    if is_package_available("torch"):
        import torch

        if torch.cuda.is_available():
            gpu_available = True
            gpu_count = int(torch.cuda.device_count())
            gpu_name = str(torch.cuda.get_device_name(0))

    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "cpu_physical_cores": psutil.cpu_count(logical=False) or 1,
        "cpu_logical_cores": psutil.cpu_count(logical=True) or 1,
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
    }


def infer_semantic_column_type(series: pd.Series) -> str:
    """Infer high-level semantic type of a pandas Series."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0 and non_null.nunique() <= 2 and set(non_null.unique()).issubset({0, 1}):
            return "binary_numeric"
        if pd.api.types.is_integer_dtype(series):
            return "integer"
        return "continuous"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "categorical"

    sample = series.dropna().astype(str).head(50)
    if len(sample) == 0:
        return "empty"
    avg_len = float(sample.str.len().mean())
    unique_ratio = series.nunique(dropna=True) / max(len(series.dropna()), 1)
    if avg_len > 40 and unique_ratio > 0.5:
        return "text"
    if unique_ratio > 0.95 and len(series) >= 20:
        return "identifier"
    return "categorical"


_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_spreadsheet_value(val: Any) -> Any:
    """Neutralize spreadsheet formula injection (CWE-1236) in CSV/XLSX string cells.

    Preserves legitimate signed numeric literals (e.g. ``"-42.5"``, ``"+3.14e-2"``) while
    prefixing a leading single quote onto strings starting with ``=``, ``+``, ``-``, ``@``,
    tab, or carriage-return characters.
    """
    if isinstance(val, str) and val:
        stripped = val.lstrip(" ")
        if val[0] in ("\t", "\r") or stripped.startswith(_FORMULA_PREFIXES):
            if stripped[0] in ("+", "-"):
                try:
                    num = float(stripped)
                    if math.isfinite(num):
                        return val
                except ValueError:
                    pass
            return f"'{val}"
    return val


def sanitize_spreadsheet_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare values and headers for spreadsheet exports without changing the source frame."""
    safe = frame.map(sanitize_spreadsheet_value, na_action="ignore")
    safe.columns = frame.columns.map(sanitize_spreadsheet_value)
    return safe
