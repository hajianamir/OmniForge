"""Focused contracts for the independent audit of main at 4385647."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import structlog
from statsmodels.stats.multitest import multipletests

from omniforge.causal import CausalInferenceEngine
from omniforge.core import logging as of_logging
from omniforge.core.exceptions import CausalInferenceError
from omniforge.eda.statistics import StatisticalAnalyzer


def test_holm_excludes_unevaluable_values_and_preserves_order() -> None:
    p = [0.3, np.nan, 0.01, 0.02, np.inf, 0.01, 0.9, -0.1, 0.0, 1.1, 1.0]
    observed = np.isfinite(p) & (np.asarray(p) >= 0) & (np.asarray(p) <= 1)
    actual = np.asarray(StatisticalAnalyzer.adjust_p_values(p, method="holm"))
    expected = multipletests(np.asarray(p)[observed], method="holm")[1]
    np.testing.assert_allclose(actual[observed], expected)
    assert np.isnan(actual[~observed]).all()


@pytest.mark.parametrize("method", ["aipw", "ipw", "psm", "t_learner", "s_learner", "x_learner"])
def test_binary_estimators_reject_third_arm_even_if_complete_case_filter_would_hide_it(method: str) -> None:
    # Both binary arms have complete cases; all third-arm outcomes are missing.
    df = pd.DataFrame({"t": np.tile([0, 1, 2], 20), "y": np.tile([0.0, 2.0, np.nan], 20)})
    with pytest.raises(CausalInferenceError, match="binary treatment coded 0"):
        CausalInferenceEngine().estimate_ate(df, treatment_col="t", outcome_col="y", confounders=[], method=method)


@pytest.mark.parametrize("other", [2, -1, 0.5, np.inf, "unknown"])
def test_aipw_rejects_observed_unsupported_levels(other: int | float | str) -> None:
    df = pd.DataFrame({"t": [0, 1, other] * 12, "y": np.arange(36)})
    with pytest.raises(CausalInferenceError, match="binary treatment coded 0"):
        CausalInferenceEngine().estimate_ate(df, treatment_col="t", outcome_col="y", confounders=[])


@pytest.mark.parametrize("lose_arm_on_filter", [False, True])
def test_aipw_requires_both_arms_in_complete_cases(lose_arm_on_filter: bool) -> None:
    df = pd.DataFrame({"t": [0] * 30, "y": np.arange(30)})
    if lose_arm_on_filter:
        df = pd.concat([df, pd.DataFrame({"t": [1, 1], "y": [np.nan, np.nan]})], ignore_index=True)
    with pytest.raises(CausalInferenceError, match="at least 2 treated .* and 2 control"):
        CausalInferenceEngine().estimate_ate(df, treatment_col="t", outcome_col="y", confounders=[])


def test_dml_retains_continuous_treatment_semantics() -> None:
    rng = np.random.default_rng(872)
    t = rng.uniform(-2, 2, 150)
    x = rng.normal(size=t.size)
    df = pd.DataFrame({"t": t, "x": x, "y": 2 * t + x + rng.normal(0, 0.05, t.size)})
    result = CausalInferenceEngine().estimate_ate(
        df, treatment_col="t", outcome_col="y", confounders=["x"], method="dml", run_refutations=False,
    )
    assert result.ate == pytest.approx(2.0, abs=0.16)
    assert result.details["se_method"] == "neyman_orthogonal_hc1"


@pytest.mark.parametrize(("structured", "log_format"), [(True, "json"), (False, "json"), (False, "console")])
def test_configured_loggers_redact_formatted_tracebacks(
    structured: bool, log_format: str, capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # Synthetic only. The token is present solely in the formatted exception and its cause.
    canary = "Bearer auditCanary_0123456789abcdefghijklmnopqrstuvwxyz"
    root = logging.getLogger("omniforge")
    log_file = tmp_path / "exceptions.log"
    original_structlog_config = structlog.get_config().copy()
    original_level = root.level
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "propagate", root.propagate)
    monkeypatch.setattr(of_logging, "_CONFIGURED", of_logging._CONFIGURED)
    monkeypatch.setattr(of_logging, "_HAS_STRUCTLOG", structured)
    try:
        of_logging.configure_logging(level="ERROR", log_format=log_format, log_file=log_file)
        try:
            try:
                raise ValueError(f"inner synthetic exception contains {canary}")
            except ValueError as exc:
                raise RuntimeError(f"outer synthetic exception contains {canary}") from exc
        except RuntimeError:
            if structured:
                of_logging.get_logger("test.traceback").error("synthetic_probe", exc_info=True)
            else:
                logging.getLogger("omniforge.test.traceback").exception("synthetic_probe")
        for output in (capsys.readouterr().out.strip(), log_file.read_text(encoding="utf-8").strip()):
            assert canary.split()[1] not in output
            traceback_text = json.loads(output)["exception"] if output.startswith("{") else output
            assert "RuntimeError: outer synthetic exception" in traceback_text
            assert "ValueError: inner synthetic exception" in traceback_text
            assert traceback_text.count("***REDACTED***") >= 2
    finally:
        for handler in root.handlers:
            handler.close()
        root.setLevel(original_level)
        structlog.configure(**original_structlog_config)
