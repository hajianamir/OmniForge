"""Latency & memory budget benchmark for OmniForge hot paths (audit P2 / §7.1).

Measures p50/p95/p99 wall-clock latency (perf_counter, in-process) for a fixed set of
representative hot paths and peak Python allocations (tracemalloc) where applicable.
Exits non-zero (and emits ``::error`` annotations) if any budget is exceeded.

Budgets are intentionally generous (roughly 5x observed p95 on a modern x86-64 developer
machine) to stay robust across shared CI runners; tighten them as measurements stabilize.
Override the multiplier with ``--budget-scale`` (e.g. ``--budget-scale 2`` on a quiet box).

Usage:
    python scripts/latency_budget_benchmark.py            # check budgets, human output
    python scripts/latency_budget_benchmark.py --json     # machine-readable report
    python scripts/latency_budget_benchmark.py --repeats 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

_rng = np.random.default_rng(42)
_DF_5K = pd.DataFrame(
    {
        "customer_id": np.arange(5000),
        "revenue": _rng.normal(1000.0, 250.0, 5000),
        "segment": _rng.choice(["a", "b", "c", "d"], 5000),
        "signup_days": _rng.integers(1, 2000, 5000),
        "flag": _rng.integers(0, 2, 5000),
    }
)
_DF_500 = _DF_5K.head(500).copy()
_PII_DF = pd.DataFrame(
    {
        "email": [f"user{i}@example.com" for i in range(1000)],
        "ssn": [f"123-45-{i:04d}" for i in range(1000)],
        "notes": ["nothing to see here"] * 1000,
    }
)

# --------------------------------------------------------------------------- #
# Benchmarked operations                                                       #
# --------------------------------------------------------------------------- #


def _bench_dataframe_fingerprint() -> float:
    from omniforge.core.performance import compute_df_fingerprint

    compute_df_fingerprint(_DF_5K)


def _bench_pii_scan() -> None:
    from omniforge.core.security import PIIRedactor

    PIIRedactor().scan_dataframe(_PII_DF, sample_size=200)


def _bench_pii_pseudonymize_hash() -> None:
    from omniforge.core.security import PIIRedactor

    PIIRedactor().anonymize_dataframe(_PII_DF, columns=["email"], strategy="hash", salt="benchmark-salt-" + "x" * 32)


def _bench_quality_profile() -> None:
    from omniforge.data.quality import DataQualityProfiler

    DataQualityProfiler().profile(_DF_500)


_DF_50 = _DF_5K.head(50).copy()
_CACHE = None


def _bench_ttl_cache_roundtrip() -> None:
    from omniforge.core.performance import DataFrameCache

    global _CACHE
    if _CACHE is None:
        _CACHE = DataFrameCache(max_entries=16, ttl_seconds=60.0)
    for i in range(10):
        _CACHE.set("bench", _DF_50, i, params=str(i))
    _CACHE.get("bench", _DF_50, "42")


_HEALTH_CLIENT = None


def _bench_health_endpoint() -> None:
    # App construction is one-time fixture cost, NOT per-request latency.
    global _HEALTH_CLIENT
    if _HEALTH_CLIENT is None:
        from fastapi.testclient import TestClient

        from omniforge.api.main import create_app

        _HEALTH_CLIENT = TestClient(create_app())
    _HEALTH_CLIENT.get("/health")


# Budgets in milliseconds (p95) / megabytes (peak alloc). See module docstring for sizing.
_BUDGETS: dict[str, dict[str, float]] = {
    "dataframe_fingerprint_5k": {"p95_ms": 250.0},
    "pii_scan_1000": {"p95_ms": 400.0},
    "pii_pseudonymize_hash_1000": {"p95_ms": 400.0},
    "quality_profile_500x5": {"p95_ms": 8000.0, "peak_mb": 300.0},
    "ttl_cache_50set_1get": {"p95_ms": 50.0},
    "health_endpoint_inprocess": {"p95_ms": 150.0},
}

_OPS: dict[str, Callable[[], Any]] = {
    "dataframe_fingerprint_5k": _bench_dataframe_fingerprint,
    "pii_scan_1000": _bench_pii_scan,
    "pii_pseudonymize_hash_1000": _bench_pii_pseudonymize_hash,
    "quality_profile_500x5": _bench_quality_profile,
    "ttl_cache_50set_1get": _bench_ttl_cache_roundtrip,
    "health_endpoint_inprocess": _bench_health_endpoint,
}


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def run(repeats: int, budget_scale: float) -> dict[str, Any]:
    report: dict[str, Any] = {"repeats": repeats, "budget_scale": budget_scale, "checks": {}}
    failures: list[str] = []
    for name, op in _OPS.items():
        # warmup (also surfaces import errors before timing)
        op()
        samples_ms: list[float] = []
        trace_mb = 0.0
        for i in range(repeats):
            if i == repeats - 1:  # trace allocations on the final iteration only
                tracemalloc.start()
            start = time.perf_counter()
            op()
            elapsed = (time.perf_counter() - start) * 1000.0
            if i == repeats - 1:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                trace_mb = peak / (1024 * 1024)
            samples_ms.append(elapsed)
        samples_ms.sort()
        p50 = _percentile(samples_ms, 0.50)
        p95 = _percentile(samples_ms, 0.95)
        p99 = _percentile(samples_ms, 0.99)
        budget = _BUDGETS[name]
        entry: dict[str, float] = {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99, "mean_ms": statistics.fmean(samples_ms)}
        if "peak_mb" in budget:
            entry["peak_mb"] = trace_mb
        budget_p95 = budget["p95_ms"] * budget_scale
        entry["p95_budget_ms"] = budget_p95
        if p95 > budget_p95:
            failures.append(f"{name}: p95 {p95:.1f}ms > budget {budget_p95:.1f}ms")
        if "peak_mb" in budget:
            entry["peak_mb_budget"] = budget["peak_mb"]
            if trace_mb > budget["peak_mb"]:
                failures.append(f"{name}: peak {trace_mb:.1f}MB > budget {budget['peak_mb']:.1f}MB")
        report["checks"][name] = entry
    report["failures"] = failures
    report["ok"] = not failures
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=25, help="timed iterations per hot path (default 25)")
    parser.add_argument("--budget-scale", type=float, default=1.0, help="multiply all p95 budgets (default 1.0)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON report")
    args = parser.parse_args()

    report = run(repeats=max(3, args.repeats), budget_scale=args.budget_scale)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"OmniForge latency/memory budgets (repeats={report['repeats']}, budget-scale={report['budget_scale']})")
        for name, entry in report["checks"].items():
            line = (
                f"  {name:28s} p50={entry['p50_ms']:8.1f}ms p95={entry['p95_ms']:8.1f}ms "
                f"p99={entry['p99_ms']:8.1f}ms (budget {entry['p95_budget_ms']:.0f}ms)"
            )
            if "peak_mb" in entry:
                line += f" | peak={entry['peak_mb']:.1f}MB (budget {entry['peak_mb_budget']:.0f}MB)"
            print(line)
        if report["ok"]:
            print("ALL LATENCY/MEMORY BUDGETS OK")
        else:
            for failure in report["failures"]:
                print(f"::error title=latency-budget::{failure}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
