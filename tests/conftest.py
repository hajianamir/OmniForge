"""Shared pytest fixtures for OmniForge test suites."""

from __future__ import annotations

import os

# The suite issues well over 300 requests/minute to the shared in-process app; the
# SlidingWindowRateLimitMiddleware would 429 later tests. Disable the limiter for tests
# (the limiter itself is unit-tested directly in tests/unit/test_request_limits.py).
os.environ.setdefault("OMNIFORGE_API__RATE_LIMIT_PER_MINUTE", "0")

from pathlib import Path

import pandas as pd
import pytest

from omniforge.core.utils import set_global_seed
from omniforge.data.synthetic import SyntheticDataGenerator


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    set_global_seed(42)


@pytest.fixture()
def classification_df() -> pd.DataFrame:
    return SyntheticDataGenerator(random_state=42).generate_classification_dataset(
        n_samples=220, n_features=6, missing_rate=0.05
    )


@pytest.fixture()
def regression_df() -> pd.DataFrame:
    return SyntheticDataGenerator(random_state=42).generate_regression_dataset(
        n_samples=200, n_features=6, missing_rate=0.04
    )


@pytest.fixture()
def timeseries_df() -> pd.DataFrame:
    return SyntheticDataGenerator(random_state=42).generate_time_series_dataset(n_periods=110)


@pytest.fixture()
def causal_df() -> pd.DataFrame:
    return SyntheticDataGenerator(random_state=42).generate_causal_dataset(n_samples=280, true_ate=3.2)


@pytest.fixture()
def workspace_tmp(tmp_path: Path) -> Path:
    return tmp_path
