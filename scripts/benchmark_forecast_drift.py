"""Compare fixed and trailing interval widths under held-out changes in a seasonal series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omniforge.core.utils import to_json_object
from omniforge.forecasting.engine import ForecastingEngine

SEEDS = (7, 29, 83)
HORIZON = 4
CALIBRATION_ORIGINS = 30
EVALUATION_ORIGINS = 36
HISTORY_ROWS = 400
SHIFT_POSITION = HISTORY_ROWS - HORIZON * EVALUATION_ORIGINS
VOLATILITY_RETURN_POSITION = SHIFT_POSITION + 8 * HORIZON
METRICS = ("coverage", "mean_interval_score", "mean_width")


def _pool(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate matched outcomes by their counts across seeds or origins."""
    used = [row for row in rows if row["paired_outcomes"]]
    n = sum(row["paired_outcomes"] for row in used)
    return {
        "paired_outcomes": n,
        **{
            method: {
                metric: sum(row[method][metric] * row["paired_outcomes"] for row in used) / n if n else None
                for metric in METRICS
            }
            for method in ("fixed", "rolling", "normalized")
        },
    }


def run_benchmark() -> dict[str, Any]:
    """Benchmark shared forecast paths for four regimes and three fixed seeds."""
    seasons = 100 + 5 * np.sin(2 * np.pi * np.arange(HISTORY_ROWS) / 7)
    results: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("stationary", "level_shift", "volatility_increase", "temporary_volatility")
    }
    for seed in SEEDS:
        noise = np.random.default_rng(seed).standard_normal(HISTORY_ROWS)
        stationary = seasons + 2 * noise
        level_shift = stationary.copy()
        level_shift[SHIFT_POSITION:] += 16
        increased_variance = stationary.copy()
        increased_variance[SHIFT_POSITION:] = seasons[SHIFT_POSITION:] + 7 * noise[SHIFT_POSITION:]
        temporary_variance = stationary.copy()
        temporary_variance[SHIFT_POSITION:VOLATILITY_RETURN_POSITION] = increased_variance[SHIFT_POSITION:VOLATILITY_RETURN_POSITION]

        for name, values in zip(
            results, (stationary, level_shift, increased_variance, temporary_variance), strict=True,
        ):
            report = ForecastingEngine(
                seasonal_period=7, include_deep=False, include_foundation=False, confidence_level=0.95,
                n_cv_splits=0, random_seed=42,
            ).forecast(
                pd.DataFrame({"y": values}), target_col="y", horizon=HORIZON,
                interval_calibration_origins=CALIBRATION_ORIGINS,
                interval_evaluation_origins=EVALUATION_ORIGINS,
                compare_rolling_intervals=True,
            )
            comparison = report.metadata["champion_interval_calibration"]["rolling_comparison"]
            if not comparison:
                raise RuntimeError(f"Rolling interval comparison unavailable for {name}, seed {seed}.")
            results[name].append({"seed": seed, "champion": report.champion_model, **comparison})

    return to_json_object({
        "seeds": SEEDS,
        "nominal_coverage": 0.95,
        "horizon": HORIZON,
        "calibration_origins": CALIBRATION_ORIGINS,
        "evaluation_origins": EVALUATION_ORIGINS,
        "change_position": SHIFT_POSITION,
        "temporary_volatility_return_position": VOLATILITY_RETURN_POSITION,
        "regimes": {
            name: {
                "overall": _pool(rows),
                "first_window": _pool([row["by_origin"][0] for row in rows]),
                "last_eight_windows": _pool([origin for row in rows for origin in row["by_origin"][-8:]]),
                "by_seed": rows,
            }
            for name, rows in results.items()
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Write the benchmark evidence as JSON.")
    args = parser.parse_args()
    args.out.write_text(json.dumps(run_benchmark(), indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
