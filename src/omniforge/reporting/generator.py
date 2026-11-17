"""Enterprise Report Generator orchestrating user-selectable sections, audit trails, exclusions, formulas, charts & multi-format export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omniforge.core.exceptions import ReportingError
from omniforge.core.logging import get_logger
from omniforge.reporting.charts import ReportChartBuilder
from omniforge.reporting.exporters import _TABLE_SECTION_MAP, ReportExporters
from omniforge.reporting.formulas import FormulaRulesCatalog
from omniforge.reporting.narrative import NarrativeGenerator
from omniforge.reporting.sections import (
    ALL_SECTION_IDS,
    ReportSelectionConfig,
    compute_raw_vs_processed_diff,
)

logger = get_logger(__name__)


def _compute_dataset_summary_stats(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute crisp column-level descriptive statistics table for numeric and categorical features."""
    rows: list[dict[str, Any]] = []
    for col in list(df.columns)[:40]:
        ser = df[col]
        miss_n = int(ser.isna().sum())
        miss_pct = round(miss_n / max(len(df), 1) * 100.0, 2)
        n_uniq = int(ser.nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(ser):
            clean = pd.to_numeric(ser, errors="coerce").dropna()
            q1 = float(clean.quantile(0.25)) if len(clean) > 0 else float("nan")
            q3 = float(clean.quantile(0.75)) if len(clean) > 0 else float("nan")
            iqr = q3 - q1 if len(clean) > 0 else 0.0
            iqr_out = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()) if iqr > 1e-12 else 0
            rows.append(
                {
                    "Column": str(col),
                    "Type": str(ser.dtype),
                    "Non-Null": int(len(clean)),
                    "Missing %": miss_pct,
                    "Unique": n_uniq,
                    "Mean": round(float(clean.mean()), 4) if len(clean) > 0 else "",
                    "Std": round(float(clean.std(ddof=1)), 4) if len(clean) > 1 else "",
                    "Min": round(float(clean.min()), 4) if len(clean) > 0 else "",
                    "Median": round(float(clean.median()), 4) if len(clean) > 0 else "",
                    "Max": round(float(clean.max()), 4) if len(clean) > 0 else "",
                    "IQR Outliers": iqr_out,
                }
            )
        else:
            top_val = str(ser.dropna().mode().iloc[0])[:30] if not ser.dropna().empty else ""
            rows.append(
                {
                    "Column": str(col),
                    "Type": str(ser.dtype),
                    "Non-Null": int(len(df) - miss_n),
                    "Missing %": miss_pct,
                    "Unique": n_uniq,
                    "Mean": "-",
                    "Std": "-",
                    "Min": "-",
                    "Median": f"Top: {top_val}",
                    "Max": "-",
                    "IQR Outliers": 0,
                }
            )
    return rows


def _compute_top_correlations(df: pd.DataFrame, max_pairs: int = 20) -> list[dict[str, Any]]:
    """Compute ranked strongest pairwise Pearson correlations."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return []
    if num_df.shape[1] > 25:
        variances = num_df.var(numeric_only=True).sort_values(ascending=False)
        num_df = num_df[list(variances.index[:25])]
    corr = num_df.corr().fillna(0.0)
    cols = list(corr.columns)
    pairs: list[dict[str, Any]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr.iloc[i, j])
            pairs.append(
                {
                    "Feature_A": str(cols[i]),
                    "Feature_B": str(cols[j]),
                    "Pearson_r": round(r, 4),
                    "Abs_r": round(abs(r), 4),
                    "Direction": "Positive (+)" if r >= 0 else "Negative (-)",
                    "Strength": (
                        "Very Strong (|r|>=0.8)"
                        if abs(r) >= 0.8
                        else ("Strong (|r|>=0.6)" if abs(r) >= 0.6 else ("Moderate (|r|>=0.35)" if abs(r) >= 0.35 else "Weak"))
                    ),
                }
            )
    pairs.sort(key=lambda x: x["Abs_r"], reverse=True)
    return pairs[:max_pairs]


class ReportGenerator:
    """Comprehensive customizable report builder supporting DOCX, PDF, XLSX, CSV (+ Bundle), HTML, PPTX, MD, and JSON."""

    def __init__(self, llm_agent: Any | None = None) -> None:
        self.narrative = NarrativeGenerator(llm_agent=llm_agent)
        self.exporters = ReportExporters()

    def build_report(
        self,
        *,
        title: str = "OmniForge Executive Report",
        domain: str = "general",
        author: str = "OmniForge Enterprise",
        selection: ReportSelectionConfig | None = None,
        selected_sections: list[str] | None = None,
        include_charts: bool = True,
        include_tables: bool = True,
        include_dataset_appendix: bool = False,
        max_table_rows: int = 100,
        dataset_summary: dict[str, Any] | None = None,
        quality_report: dict[str, Any] | None = None,
        model_summary: dict[str, Any] | None = None,
        fairness_report: dict[str, Any] | None = None,
        carbon_report: dict[str, Any] | None = None,
        tables: dict[str, list[dict[str, Any]]] | None = None,
        analyses_performed: list[dict[str, Any]] | None = None,
        exclusions: list[dict[str, Any]] | None = None,
        raw_vs_processed: dict[str, Any] | None = None,
        formulas_and_rules: list[dict[str, Any]] | None = None,
        charts: list[dict[str, Any]] | None = None,
        diagnostics_summary: dict[str, Any] | None = None,
        clustering_summary: dict[str, Any] | None = None,
        eda_summary: dict[str, Any] | None = None,
        feature_summary: dict[str, Any] | None = None,
        future_predictions_summary: dict[str, Any] | None = None,
        decision_summary: dict[str, Any] | None = None,
        # Optional live analytical objects for automatic chart/formula/diff enrichment
        df: pd.DataFrame | None = None,
        raw_df: pd.DataFrame | None = None,
        target_columns: list[str] | None = None,
        cleaning_report: dict[str, Any] | None = None,
        diagnostics_report: Any | None = None,
        clustering_result: Any | None = None,
        anomaly_meta: dict[str, Any] | None = None,
        eda_report: Any | None = None,
        feature_selection_method: str | None = None,
        fs_scores: dict[str, float] | None = None,
        fs_selected: list[str] | None = None,
        model_equations: dict[str, dict[str, dict[str, Any]]] | None = None,
        automl_result: Any | None = None,
        dl_benchmark_df: pd.DataFrame | None = None,
        future_prediction_result: Any | None = None,
        causal_result: Any | None = None,
        forecast_report: Any | None = None,
    ) -> dict[str, Any]:
        """Assemble a complete customizable report payload filtered strictly to user-selected sections."""
        if selection is not None:
            enabled_list = list(selection.selected_sections)
            include_charts = selection.include_charts
            include_tables = selection.include_tables
            include_dataset_appendix = selection.include_dataset_appendix
            max_table_rows = selection.max_table_rows
            author = selection.author
        elif selected_sections is not None:
            enabled_list = list(selected_sections)
        else:
            enabled_list = list(ALL_SECTION_IDS)
        enabled = set(enabled_list)

        # Normalize dataset summary automatically if df provided
        ds_sum = dict(dataset_summary or {})
        if df is not None and not ds_sum:
            ds_sum = {
                "dataset_name": domain,
                "n_rows": int(len(df)),
                "n_columns": int(df.shape[1]),
                "missing_cells": int(df.isna().sum().sum()),
                "target_columns": ", ".join(target_columns or []) or "N/A",
            }
        elif ds_sum:
            if "n_rows" not in ds_sum and "rows" in ds_sum:
                ds_sum["n_rows"] = ds_sum["rows"]
            if "n_columns" not in ds_sum and "columns" in ds_sum:
                ds_sum["n_columns"] = ds_sum["columns"]

        # Forensic RAW vs PROCESSED diff
        diff_dict = dict(raw_vs_processed or {})
        if not diff_dict and raw_df is not None and df is not None:
            diff_dict = compute_raw_vs_processed_diff(raw_df, df)

        # Build formulas & computational rules catalog automatically when section enabled
        formula_dicts: list[dict[str, Any]] = []
        if "mathematical_equations" in enabled:
            if formulas_and_rules is not None:
                formula_dicts = [dict(f) for f in formulas_and_rules]
            else:
                built_formulas = FormulaRulesCatalog.build_catalog(
                    cleaning_report=cleaning_report,
                    quality_report=quality_report,
                    diagnostics_report=diagnostics_report,
                    clustering_result=clustering_result,
                    anomaly_meta=anomaly_meta,
                    eda_report=eda_report,
                    feature_selection_method=feature_selection_method,
                    model_equations=model_equations,
                    automl_result=automl_result,
                    future_prediction_result=future_prediction_result,
                    fairness_report=fairness_report,
                    causal_result=causal_result,
                    forecast_report=forecast_report,
                    carbon_report=carbon_report,
                    include_core_methodology=True,
                )
                formula_dicts = [rec.model_dump() for rec in built_formulas]

        # Build assembled tables dictionary (filtered cleanly by enabled sections)
        assembled_tables: dict[str, list[dict[str, Any]]] = {}
        if include_tables:
            # 1. Analyses Performed Audit Log table
            if "analyses_performed" in enabled and analyses_performed:
                assembled_tables["Analyses_Performed_Audit_Log"] = [dict(a) for a in analyses_performed]

            # 2. Data Exclusions & Forensic Diff tables
            if "data_exclusions" in enabled:
                if exclusions:
                    assembled_tables["Excluded_Data_and_Reasons"] = [
                        {
                            "Stage": e.get("stage", ""),
                            "Exclusion Type": e.get("exclusion_type", ""),
                            "Affected Scope": e.get("affected_scope", ""),
                            "Count": e.get("count", 0),
                            "% Dataset": e.get("pct_of_dataset", 0.0),
                            "Exact Criterion / Rule": e.get("criterion_rule", ""),
                            "Reason": e.get("reason", ""),
                            "Sample IDs / Columns": ", ".join(str(x) for x in (e.get("sample_identifiers") or [])[:8]),
                        }
                        for e in exclusions
                    ]
                if diff_dict:
                    assembled_tables["Raw_vs_Processed_Forensic_Diff"] = [
                        {"Dimension": "Row Count", "Raw Ingested": diff_dict.get("raw_rows", 0), "Active Processed": diff_dict.get("processed_rows", 0), "Delta / Impact": f"-{diff_dict.get('rows_removed', 0):,} rows ({diff_dict.get('rows_removed_pct', 0.0):.2f}% excluded)"},
                        {"Dimension": "Column Count", "Raw Ingested": diff_dict.get("raw_columns", 0), "Active Processed": diff_dict.get("processed_columns", 0), "Delta / Impact": f"Dropped: {len(diff_dict.get('removed_columns') or [])} | Added: {len(diff_dict.get('added_columns') or [])}"},
                        {"Dimension": "Missing Cells", "Raw Ingested": diff_dict.get("raw_missing_cells", 0), "Active Processed": diff_dict.get("processed_missing_cells", 0), "Delta / Impact": f"Resolved: {diff_dict.get('missing_cells_resolved', 0):,}"},
                        {"Dimension": "Exact Duplicate Rows", "Raw Ingested": diff_dict.get("raw_duplicate_rows", 0), "Active Processed": diff_dict.get("processed_duplicate_rows", 0), "Delta / Impact": f"Removed: {max(0, int(diff_dict.get('raw_duplicate_rows', 0)) - int(diff_dict.get('processed_duplicate_rows', 0))):,}"},
                    ]
                    if diff_dict.get("numeric_range_shifts"):
                        assembled_tables["Numeric_Range_Shifts_After_Cleaning"] = list(diff_dict["numeric_range_shifts"])[:20]

            # 3. Quality 6D Breakdown table
            if "quality_diagnostics" in enabled and quality_report:
                dims = quality_report.get("dimensions") or {}
                if isinstance(dims, dict) and dims:
                    assembled_tables["Data_Quality_6D_Breakdown"] = [
                        {"Dimension": str(k).replace("_", " ").title(), "Score (0-100)": round(float(v), 2)}
                        for k, v in dims.items()
                    ]

            # 4. Top Correlations from live DataFrame if EDA enabled
            if "eda_hypothesis" in enabled and df is not None and not df.empty:
                top_corr = _compute_top_correlations(df)
                if top_corr:
                    assembled_tables["Top_Feature_Correlations"] = top_corr

            # 5. User/workspace supplied tables (strictly filtered by enabled sections when mapped)
            for tname, trows in (tables or {}).items():
                if not (trows and isinstance(trows, list) and isinstance(trows[0], dict)):
                    continue
                mapped_sec = _TABLE_SECTION_MAP.get(tname)
                if mapped_sec is not None and mapped_sec not in enabled:
                    continue
                if tname not in assembled_tables:
                    assembled_tables[tname] = trows

            # 6. Optional Dataset Summary Statistics Appendix
            if include_dataset_appendix and df is not None and not df.empty:
                assembled_tables["Dataset_Summary_Statistics"] = _compute_dataset_summary_stats(df)
                assembled_tables["Active_Dataset_Sample"] = df.head(20).to_dict(orient="records")

        # Render charts automatically if enabled
        chart_dicts: list[dict[str, Any]] = []
        if include_charts:
            if charts is not None:
                chart_dicts = [
                    dict(ch) for ch in charts if isinstance(ch, dict) and str(ch.get("section_id", "executive_summary")) in enabled
                ]
            else:
                leaderboard_rows = (
                    list(automl_result.leaderboard)
                    if automl_result is not None and getattr(automl_result, "leaderboard", None)
                    else (tables or {}).get("AutoML_Leaderboard")
                )
                risk_assess = getattr(diagnostics_report, "risk_assessment", None) if diagnostics_report is not None else None
                built_charts = ReportChartBuilder.build_all_charts(
                    selected_sections=enabled,
                    df=df,
                    raw_df=raw_df,
                    target_columns=target_columns,
                    quality_report=quality_report,
                    exclusions=exclusions,
                    raw_vs_processed=diff_dict,
                    clustering_result=clustering_result,
                    fs_scores=fs_scores,
                    fs_selected=fs_selected,
                    leaderboard=leaderboard_rows,
                    dl_benchmark_df=dl_benchmark_df,
                    model_equations=model_equations,
                    future_prediction_result=future_prediction_result,
                    risk_assessment=risk_assess,
                )
                chart_dicts = [c.model_dump() for c in built_charts]

        # Build executive & technical narrative
        nar = self.narrative.generate_executive_narrative(
            title=title,
            domain=domain,
            author=author,
            dataset_summary=ds_sum or None,
            quality_report=quality_report,
            model_summary=model_summary,
            fairness_report=fairness_report,
            carbon_report=carbon_report,
            selected_sections=enabled_list,
            analyses_performed=analyses_performed,
            exclusions=exclusions,
            raw_vs_processed=diff_dict or None,
            formulas_and_rules=formula_dicts,
            diagnostics_summary=diagnostics_summary,
            clustering_summary=clustering_summary,
            eda_summary=eda_summary,
            feature_summary=feature_summary,
            future_predictions_summary=future_predictions_summary,
            decision_summary=decision_summary,
        )

        # Build normalized KPIs dictionary
        kpis: dict[str, Any] = {}
        if ds_sum:
            kpis["Rows (Active)"] = f"{ds_sum['n_rows']:,}" if isinstance(ds_sum.get("n_rows"), int) else ds_sum.get("n_rows", "N/A")
            kpis["Columns (Active)"] = ds_sum.get("n_columns", "N/A")
            if ds_sum.get("processing_mode"):
                kpis["Dataset Provenance"] = ds_sum["processing_mode"]
        if diff_dict and diff_dict.get("rows_removed", 0) > 0:
            kpis["Excluded Rows"] = f"{diff_dict['rows_removed']:,} ({diff_dict.get('rows_removed_pct', 0.0):.2f}%)"
        if quality_report:
            kpis["Data Quality"] = f"{quality_report.get('overall_score', 0)}/100 ({quality_report.get('grade', '-')})"
        if analyses_performed:
            kpis["Analyses Audited"] = len(analyses_performed)
        if formula_dicts:
            kpis["Formulas & Equations"] = len(formula_dicts)
        if model_summary:
            kpis["Champion Model"] = (
                model_summary.get("best_model_name")
                or model_summary.get("champion_model")
                or model_summary.get("model_name", "N/A")
            )
            if model_summary.get("best_score") is not None:
                kpis[str(model_summary.get("primary_metric", "Model Score"))] = model_summary.get("best_score")
        if fairness_report:
            kpis["Fairness Gate"] = "PASSED" if fairness_report.get("passed", True) else "FLAGGED"
        if carbon_report:
            kpis["Carbon (gCO2eq)"] = round(float(carbon_report.get("emissions_g_co2eq", 0.0)), 6)
            kpis["Green AI Score"] = f"{carbon_report.get('green_score', 100)}/100"

        return {
            **nar,
            "kpis": kpis,
            "analyses_performed": list(analyses_performed or []) if "analyses_performed" in enabled else [],
            "exclusions": list(exclusions or []) if "data_exclusions" in enabled else [],
            "raw_vs_processed": diff_dict if "data_exclusions" in enabled else {},
            "formulas_and_rules": formula_dicts if "mathematical_equations" in enabled else [],
            "charts": chart_dicts,
            "tables": assembled_tables,
            "max_table_rows": max_table_rows,
            "include_charts": include_charts,
            "include_tables": include_tables,
        }

    def export(
        self,
        payload: dict[str, Any],
        output_path: str | Path,
        *,
        fmt: str | None = None,
    ) -> Path:
        """Export ``payload`` to the requested format (docx, pdf, xlsx, csv, csv_bundle, html, pptx, md, json)."""
        path = Path(output_path)
        target_fmt = (fmt or path.suffix.lstrip(".")).lower()
        dispatch = {
            "html": self.exporters.to_html,
            "md": self.exporters.to_markdown,
            "markdown": self.exporters.to_markdown,
            "json": self.exporters.to_json,
            "csv": self.exporters.to_csv,
            "csv_bundle": self.exporters.to_csv_bundle,
            "xlsx": self.exporters.to_excel,
            "excel": self.exporters.to_excel,
            "docx": self.exporters.to_docx,
            "doc": self.exporters.to_docx,
            "word": self.exporters.to_docx,
            "pptx": self.exporters.to_pptx,
            "pdf": self.exporters.to_pdf,
        }
        if target_fmt not in dispatch:
            raise ReportingError(
                f"Unsupported report format '{target_fmt}'.",
                suggestions=[f"Supported formats: {sorted(dispatch.keys())}"],
            )
        out = dispatch[target_fmt](payload, path)
        logger.info("report_exported", format=target_fmt, path=str(out))
        return out
