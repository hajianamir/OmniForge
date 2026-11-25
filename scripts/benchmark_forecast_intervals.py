"""Measure native and calibrated intervals on fixed chronological evaluation windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omniforge.core.utils import to_json_object
from omniforge.forecasting.engine import ForecastingEngine

HORIZON = 4
CALIBRATION_ORIGINS = 30
EVALUATION_ORIGINS = 10


def run_benchmark() -> dict[str, Any]:
    """Evaluate three packaged public series and two seeded seasonal scenarios."""
    from statsmodels import datasets

    rng = np.random.default_rng(42)
    time = np.arange(240)
    seasonal = 100 + 5 * np.sin(2 * np.pi * time / 7) + rng.normal(0, 2, len(time))
    shifted = seasonal.copy()
    shifted[-EVALUATION_ORIGINS * HORIZON:] += 16
    cases = (
        ("weekly_co2", datasets.co2.load_pandas().data["co2"], 52),
        ("quarterly_real_gdp", datasets.macrodata.load_pandas().data["realgdp"], 4),
        ("annual_sunspots", datasets.sunspots.load_pandas().data["SUNACTIVITY"], 11),
        ("seeded_seasonal", seasonal, 7),
        ("seeded_level_shift", shifted, 7),
    )

    results = {}
    for name, values, seasonal_period in cases:
        series = pd.Series(values, name="target", dtype=float)
        report = ForecastingEngine(
            seasonal_period=seasonal_period, include_deep=False, include_foundation=False,
            n_cv_splits=0, confidence_level=0.95, random_seed=42,
        ).forecast(
            series.to_frame(), target_col="target", horizon=HORIZON,
            interval_calibration_origins=CALIBRATION_ORIGINS, interval_evaluation_origins=EVALUATION_ORIGINS,
        )
        results[name] = {
            "rows": len(series),
            "missing": int(series.isna().sum()),
            "seasonal_period": seasonal_period,
            "champion": report.champion_model,
            **report.metadata["champion_interval_calibration"],
        }
    return to_json_object({
        "horizon": HORIZON,
        "calibration_origins": CALIBRATION_ORIGINS,
        "evaluation_origins": EVALUATION_ORIGINS,
        "results": results,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Write the benchmark evidence as JSON.")
    args = parser.parse_args()
    args.out.write_text(json.dumps(run_benchmark(), indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
