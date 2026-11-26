"""Studio 12 (Customizable Executive Reporting & Export Center) router."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any, cast

import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from omniforge.api.auth import require_permission
from omniforge.api.studios.schemas import ReportCompileReq, wb_cache_and_respond
from omniforge.core.performance import downsample_table_for_plot
from omniforge.core.utils import to_json_object
from omniforge.reporting import REPORT_PRESETS, REPORT_SECTIONS_CATALOG, ReportGenerator

router = APIRouter(tags=["workbench-s12"])


def _wb() -> Any:
    from omniforge.api import workbench as wb
    return wb


@router.get("/s12/catalog")
def s12_report_catalog(x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    state = _wb().get_session_state(_wb()._sid(x_session_id))
    has_df = isinstance(state.get("uap_df"), pd.DataFrame)
    avail_map = {
        "executive_summary": has_df,
        "analyses_performed": bool(state.get("uap_lineage")) or has_df,
        "data_exclusions": bool(
            state.get("uap_clean_report") or state.get("uap_exclusions") or state.get("uap_deleted_records_log")
        ),
        "intake_audit": bool(state.get("uap_intake_audit")),
        "quality_diagnostics": bool(state.get("uap_diag_report") or state.get("uap_quality")) or has_df,
        "clustering_anomalies": bool(
            state.get("uap_clustering_result") or state.get("uap_anomaly_meta") or state.get("uap_mining_result")
        ),
        "eda_hypothesis": bool(state.get("uap_eda_report") or state.get("uap_hypothesis_battery")) or has_df,
        "feature_engineering": bool(state.get("uap_fs_scores") or state.get("uap_s05_engineered")),
        "modeling_leaderboard": bool(
            state.get("uap_automl") or state.get("uap_active_model") or state.get("uap_dl_benchmark_df") is not None
        ),
        "mathematical_equations": bool(state.get("uap_baseline_equations") or state.get("uap_model_equations")),
        "future_predictions": bool(
            state.get("uap_supervisory_result")
            or state.get("uap_future_prediction_result")
            or state.get("uap_future_scores") is not None
        ),
        "responsible_ai_governance": bool(state.get("uap_shap_result") or state.get("uap_fairness_result")),
        "decision_intelligence": bool(
            state.get("uap_causal_result")
            or state.get("uap_forecast_report")
            or state.get("uap_forecast_result")
            or state.get("uap_opt_result")
            or state.get("uap_sim_result")
            or state.get("uap_risk_report")
        ),
        "recommendations": has_df,
        "dataset_appendix": has_df,
    }
    catalog = [
        {
            "key": spec.section_id,
            "title": spec.title,
            "category": spec.category,
            "description": spec.description,
            "default_on": getattr(spec, "default_enabled", True),
            "has_data": bool(avail_map.get(spec.section_id, False)),
        }
        for spec in REPORT_SECTIONS_CATALOG
    ]
    return to_json_object(
        {
            "sections": catalog,
            "presets": REPORT_PRESETS,
            "formats": ["pdf", "xlsx", "docx", "csv", "csv_bundle", "pptx", "html", "md", "json"],
        }
    )


def _compile_report_payload(state: dict[str, Any], req: ReportCompileReq) -> dict[str, Any]:
    df = _wb()._require_df(state)
    raw_df = state.get("uap_raw_df")
    quality = state.get("uap_quality")
    if quality is None:
        quality = _wb().DataQualityProfiler().profile(df)
        state["uap_quality"] = quality
    q_dict: dict[str, Any] = to_json_object(
        quality.model_dump() if hasattr(quality, "model_dump") else quality
    )
    targets = _wb().get_selected_targets(state, df)
    modeling = state.get("uap_automl")
    model_name = state.get("uap_active_model_name")
    metrics = state.get("uap_active_model_metrics") or {}
    metric_name = getattr(modeling, "primary_metric", None)
    if metric_name is None and "score" in metrics:
        metric_name = "score"
    model_summary = {
        "best_model_name": model_name,
        "primary_metric": metric_name or "Model Score",
        "best_score": metrics.get(metric_name),
        "lift_over_best_baseline": metrics.get("lift_over_baseline"),
        "metrics": metrics,
    } if model_name else None
    lineage = [
        {
            **entry,
            "studio": entry.get("studio") or entry.get("stage", ""),
            "analysis_name": entry.get("analysis_name") or entry.get("action", ""),
            "outcome_summary": entry.get("outcome_summary") or entry.get("details", ""),
            "inputs": entry.get("inputs") or f"{entry['rows']:,} rows × {entry['columns']:,} columns"
            if isinstance(entry.get("rows"), int) and isinstance(entry.get("columns"), int)
            else entry.get("inputs", ""),
        }
        for entry in state.get("uap_lineage") or [] if isinstance(entry, dict)
    ]
    gen = ReportGenerator()
    payload = gen.build_report(
        title=req.title,
        author=req.author,
        selected_sections=req.selected_sections,
        include_charts=req.include_charts,
        include_tables=req.include_tables,
        include_dataset_appendix=req.include_dataset_appendix,
        max_table_rows=req.max_table_rows,
        dataset_summary={
            "dataset_name": state.get("uap_dataset_name") or "Active Dataset",
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "target_columns": ", ".join(targets),
        },
        quality_report=q_dict,
        model_summary=model_summary,
        fairness_report=state.get("uap_fairness_result"),
        analyses_performed=lineage,
        exclusions=state.get("uap_exclusions"),
        df=downsample_table_for_plot(df, max_rows=15_000),
        raw_df=downsample_table_for_plot(raw_df, max_rows=15_000) if isinstance(raw_df, pd.DataFrame) else None,
        target_columns=targets,
        cleaning_report=state.get("uap_clean_report"),
        diagnostics_report=state.get("uap_diag_report"),
        clustering_result=state.get("uap_clustering_result"),
        anomaly_meta=state.get("uap_anomaly_meta"),
        eda_report=state.get("uap_eda_report"),
        fs_scores=state.get("uap_fs_scores"),
        fs_selected=state.get("uap_fs_selected_cols"),
        model_equations=state.get("uap_model_equations"),
        automl_result=state.get("uap_automl"),
        dl_benchmark_df=state.get("uap_dl_benchmark_df"),
        future_prediction_result=state.get("uap_future_prediction_result"),
        causal_result=state.get("uap_causal_result"),
        forecast_report=state.get("uap_forecast_report"),
    )

    # Enrich payload with Task 33 sections/tables when selected (Intake Audit, Data Mining, Hypothesis Battery, Supervisory Alarms)
    selected_set = set(req.selected_sections or [s.section_id for s in REPORT_SECTIONS_CATALOG])
    tables = payload.setdefault("tables", {})
    sections = payload.setdefault("sections", [])

    if req.include_tables and "data_exclusions" in selected_set:
        deleted = state.get("uap_deleted_records_log") or []
        if deleted:
            tables["Deleted Records Audit"] = [
                {key: entry.get(key) for key in ("timestamp", "stage", "reason", "rows_removed", "columns_removed")}
                for entry in deleted[:req.max_table_rows] if isinstance(entry, dict)
            ]

    if "intake_audit" in selected_set and isinstance(state.get("uap_intake_audit"), dict):
        ia = state["uap_intake_audit"]
        defects = ia.get("defects") or []
        pre_tests = ia.get("statistical_pretests") or []
        dc = ia.get("defect_counts") or {}
        ia_title = "Automatic Post-Ingestion Defect Audit & Statistical Pre-Screening"
        ia_body = (
            f"Automatic intake audit detected {dc.get('total', len(defects))} data defects "
            f"({dc.get('critical', 0)} Critical, {dc.get('high', 0)} High, {dc.get('medium', 0)} Medium) "
            f"with overall readiness score {ia.get('quality_score', 0):.1f}/100 (Grade {ia.get('quality_grade', 'N/A')} — {ia.get('overall_health', '')})."
        )
        sections.insert(
            min(2, len(sections)),
            {
                "section_id": "intake_audit",
                "heading": ia_title,
                "title": ia_title,
                "category": "Data",
                "body": ia_body,
                "narrative": ia_body,
                "bullets": [f"[{str(d.get('severity', '')).upper()}] {d.get('category', '')} ({d.get('column', '')}): {d.get('finding', '')}" for d in defects[:8]],
            },
        )
        if req.include_tables and defects:
            tables["Intake Defect Audit"] = defects[: req.max_table_rows]
        if req.include_tables and pre_tests:
            tables["Intake Statistical Pre-Tests"] = pre_tests[: req.max_table_rows]

    if "clustering_anomalies" in selected_set and isinstance(state.get("uap_mining_result"), dict):
        mr = state["uap_mining_result"]
        ar = (mr.get("association_rules") or {}).get("rules") or []
        sg = (mr.get("subgroups") or {}).get("subgroups") or []
        if req.include_tables and ar:
            tables["Association Rules (Apriori)"] = ar[: req.max_table_rows]
        if req.include_tables and sg:
            tables["Subgroup Discovery"] = sg[: req.max_table_rows]

    if (
        ("eda_hypothesis" in selected_set or "eda_visuals" in selected_set)
        and isinstance(state.get("uap_hypothesis_battery"), list)
        and state["uap_hypothesis_battery"]
    ):
        if req.include_tables:
            tables["Hypothesis Test Battery"] = state["uap_hypothesis_battery"][: req.max_table_rows]

    if "future_predictions" in selected_set and isinstance(state.get("uap_supervisory_result"), dict):
        sr = state["uap_supervisory_result"]
        kpis = sr.get("kpis") or {}
        alarms = sr.get("alarms") or []
        fp_title = "Live Prediction, Reality-vs-Baseline Comparison & Tolerance Alarm Audit"
        fp_body = (
            f"Supervisory evaluation of model '{kpis.get('model_label', 'Selected Model')}' vs baseline '{kpis.get('baseline_label', 'Baseline')}' "
            f"across {kpis.get('n_records', 0)} live observations resulted in status {sr.get('system_status')} "
            f"with {kpis.get('breach_count', 0)} tolerance breaches ({kpis.get('breach_rate_pct', 0):.2f}%) "
            f"and {kpis.get('compliance_rate_pct', 100):.2f}% compliance."
        )
        sections.append(
            {
                "section_id": "future_predictions",
                "heading": fp_title,
                "title": fp_title,
                "category": "Operations",
                "body": fp_body,
                "narrative": fp_body,
                "bullets": [
                    f"System Status: {sr.get('system_status')}",
                    f"Critical Breaches: {kpis.get('breach_count', 0)} | Warnings: {kpis.get('warning_count', 0)} | Compliance: {kpis.get('compliance_rate_pct', 100):.1f}%",
                    f"Model MAE: {kpis.get('mae', 'N/A')} | MAPE: {kpis.get('mape_pct', 'N/A')}% | Baseline MAE: {kpis.get('baseline_mae', 'N/A')}",
                ],
            }
        )
        if req.include_tables and alarms:
            tables["Supervisory Tolerance Alarms"] = alarms[: req.max_table_rows]
        if req.include_tables and isinstance(state.get("uap_supervisory_scored_df"), pd.DataFrame) and not state["uap_supervisory_scored_df"].empty:
            tables["Supervisory Scored Telemetry"] = state["uap_supervisory_scored_df"].head(req.max_table_rows).to_dict(orient="records")

    if ("decision_intelligence" in selected_set or "forecasting_causal" in selected_set) and state.get("uap_forecast_report") is not None:
        raw_fcr = state["uap_forecast_report"]
        fcr: dict[str, Any] = raw_fcr if isinstance(raw_fcr, dict) else to_json_object(raw_fcr)
        raw_cl = fcr.get("classical")
        cl_sum: dict[str, Any] = (
            raw_cl if isinstance(raw_cl, dict) else (to_json_object(raw_cl) if raw_cl is not None else fcr)
        )
        deep_fc = fcr.get("deep_forecast") or fcr.get("forecast") or []
        if req.include_tables and isinstance(deep_fc, list) and deep_fc:
            tables["Deep Time-Series Forecast"] = deep_fc[: req.max_table_rows]
        if not any(s.get("section_id") in ("decision_intelligence", "forecasting_causal") for s in sections):
            fc_title = "Time-Series Forecasting Tournament"
            fc_body = (
                f"Executed classical + deep ({str(fcr.get('deep_arch', 'nbeats')).upper()}) forecasting tournament "
                f"(champion: {cl_sum.get('champion_model', 'Ensemble')}, horizon={cl_sum.get('horizon', len(deep_fc))})."
            )
            sections.append(
                {
                    "section_id": "decision_intelligence",
                    "heading": fc_title,
                    "title": fc_title,
                    "category": "Decision Science",
                    "body": fc_body,
                    "narrative": fc_body,
                    "bullets": [
                        f"Classical Champion: {cl_sum.get('champion_model', 'N/A')}",
                        f"Deep Architecture: {str(fcr.get('deep_arch', 'N/A')).upper()}",
                    ],
                }
            )

    state["uap_report_payload"] = payload
    return cast(dict[str, Any], payload)


@router.post("/s12/compile", dependencies=[Depends(require_permission("report:generate"))])
def s12_compile_report(req: ReportCompileReq, x_session_id: str | None = Header(default="default")) -> dict[str, Any]:
    sid = _wb()._sid(x_session_id)
    state = _wb().get_session_state(sid)
    payload = _compile_report_payload(state, req)
    _wb().record_lineage(state, "Reporting", f"Compiled '{req.title}'", f"Sections={len(payload.get('sections', []))}")
    chart_previews = []
    for ch in (payload.get("charts") or [])[:12]:
        if isinstance(ch, dict):
            b64 = ch.get("png_base64")
            if not b64 and ch.get("png_bytes"):
                import base64 as _b64

                b64 = _b64.b64encode(ch["png_bytes"]).decode("ascii")
            if b64:
                chart_previews.append(
                    {
                        "title": ch.get("title", "Chart"),
                        "section": ch.get("section_id") or ch.get("section", "General"),
                        "caption": ch.get("caption", ""),
                        "image_data_url": f"data:image/png;base64,{b64}",
                    }
                )
    table_previews = {}
    for tname, trows in (payload.get("tables") or {}).items():
        if isinstance(trows, list) and trows and isinstance(trows[0], dict):
            table_previews[tname] = {
                "rows": len(trows),
                "columns": list(trows[0].keys()),
                "preview": trows[:15],
            }
        elif isinstance(trows, pd.DataFrame) and not trows.empty:
            table_previews[tname] = {
                "rows": int(len(trows)),
                "columns": [str(c) for c in trows.columns],
                "preview": trows.head(15).to_dict(orient="records"),
            }
    return wb_cache_and_respond(
        state,
        sid,
        "s12_report",
        {
            "title": payload.get("title"),
            "executive_summary": payload.get("executive_summary"),
            "sections": [
                {
                    "id": s.get("section_id") or s.get("id"),
                    "title": s.get("title") or s.get("heading"),
                    "category": s.get("category", "Report"),
                    "narrative": s.get("narrative") if s.get("narrative") is not None else s.get("body"),
                    "highlights": s.get("bullets") or s.get("highlights") or [],
                }
                for s in payload.get("sections", [])
            ],
            "table_names": list((payload.get("tables") or {}).keys()),
            "table_previews": table_previews,
            "n_charts": len(payload.get("charts") or []),
            "chart_previews": chart_previews,
        },
    )


@router.get("/s12/export/{fmt}", dependencies=[Depends(require_permission("report:generate"))])
def s12_export_report(fmt: str, x_session_id: str | None = Header(default="default")) -> StreamingResponse:
    state = _wb().get_session_state(_wb()._sid(x_session_id))
    payload = state.get("uap_report_payload")
    if payload is None:
        payload = _compile_report_payload(state, ReportCompileReq())
    ext_map = {
        "docx": (".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        "pdf": (".pdf", "application/pdf"),
        "xlsx": (".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        "pptx": (".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
        "html": (".html", "text/html"),
        "md": (".md", "text/markdown"),
        "json": (".json", "application/json"),
        "csv": (".csv", "text/csv"),
        "csv_bundle": (".zip", "application/zip"),
    }
    key = fmt.lower()
    if key not in ext_map:
        raise HTTPException(status_code=400, detail=f"Unsupported export format '{fmt}'. Supported: {sorted(ext_map.keys())}")
    ext, media = ext_map[key]
    with tempfile.TemporaryDirectory() as tmpd:
        out_path = Path(tmpd) / f"omniforge_report{ext}"
        exported = ReportGenerator().export(payload, out_path, fmt=cast(Any, key))
        data = exported.read_bytes()
    return StreamingResponse(
        io.BytesIO(data),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="omniforge_executive_report{ext}"'},
    )


# ---------------------------------------------------------------------------
# Studio 13 — Context-Aware AI Analytics Copilot
# ---------------------------------------------------------------------------
