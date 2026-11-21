# ruff: noqa: E402
# mypy: ignore-errors
from __future__ import annotations

import dataclasses

"""s03_eda router."""


from typing import Any

from fastapi import APIRouter, Header

from omniforge.eda.diagnostics import (
    DIAGNOSTIC_TEST_CATALOG,
    DOWNSTREAM_MODULE_CATALOG,
    StatisticalCriteriaConfig,
    UniversalDiagnosticsSuite,
)

router = APIRouter(tags=["workbench-s03"])


def _wb() -> Any:
    from omniforge.api import workbench as wb
    return wb


@router.get("/s03/catalog")
def s03_catalog() -> dict[str, Any]:
    return _wb().to_json_object(
        {
            "diagnostic_tests": DIAGNOSTIC_TEST_CATALOG,
            "downstream_modules": DOWNSTREAM_MODULE_CATALOG,
        }
    )


@router.post("/s03/run")
def s03_run_diagnostics(req: _wb().DiagnosticsReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    df = _wb().get_analysis_dataframe(state, _wb()._require_df(state))
    targets = _wb().get_selected_targets(state, df)
    criteria = StatisticalCriteriaConfig(
        alpha=req.alpha,
        confidence_level=req.confidence_level,
        multiple_testing_correction=req.fdr_method,  # type: ignore[arg-type]
        parametric_mode=req.parametric_mode,  # type: ignore[arg-type]
        normality_method=req.normality_method,  # type: ignore[arg-type]
        correlation_method=req.correlation_method,  # type: ignore[arg-type]
        outlier_method=req.outlier_method,  # type: ignore[arg-type]
        vif_threshold=req.vif_threshold,
        risk_tail_alpha=req.risk_tail_alpha,
        holdout_ratio=req.holdout_ratio,
    )
    suite = UniversalDiagnosticsSuite()
    modules: list[str] | None
    if req.mode == "auto":
        modules = list(suite.recommend_modules(df, target_cols=targets).keys())
    elif req.mode == "custom" and req.selected_modules:
        modules = list(req.selected_modules)
    else:
        modules = None  # Complete / Select All
    report = suite.run(df, target_cols=targets, selected_modules=modules, criteria=criteria)
    state["uap_diag_report"] = report
    state["uap_diag_criteria"] = criteria.model_dump() if hasattr(criteria, "model_dump") else dataclasses.asdict(criteria)
    state["uap_baseline_results"] = report.baseline_benchmarks
    state["uap_baseline_equations"] = report.baseline_equations
    state["uap_baseline_models"] = dict(getattr(suite, "last_baseline_models_", {}) or {})
    state["uap_risk_report"] = report.risk_assessment
    _wb().record_lineage(state, "Diagnostics", f"Ran Universal Diagnostics ({req.mode})", f"Tests={len(report.hypothesis_ledger)} | Targets={len(targets)}")
    rep_dict = report.model_dump() if hasattr(report, "model_dump") else dataclasses.asdict(report)
    rep_dict["total_tests"] = report.total_tests
    rep_dict["significant_tests"] = report.significant_tests
    return _wb()._cache_and_respond(state, sid, "s03_diagnostics", {"report": rep_dict})


# ---------------------------------------------------------------------------
# Studio 04 — Automated EDA & Expanded 17-Test Hypothesis Lab
# ---------------------------------------------------------------------------
