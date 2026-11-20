"""Enterprise Multi-Format Exporters: DOCX, PDF, XLSX, CSV (+ CSV Bundle), HTML, PPTX, MD, JSON, IPYNB, Docker."""

from __future__ import annotations

import base64
import csv
import io
import json
import re
import zipfile
from html import escape as html_escape
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

import nbformat
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from omniforge.core.exceptions import ReportingError
from omniforge.core.logging import get_logger
from omniforge.core.utils import safe_json_default

logger = get_logger(__name__)

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def sanitize_spreadsheet_value(val: Any) -> Any:
    """Neutralize spreadsheet formula injection (CWE-1236) in CSV/XLSX string cells."""
    if isinstance(val, str):
        stripped = val.lstrip(" \t\r\n")
        if stripped.startswith(_FORMULA_PREFIXES):
            return f"'{val}"
    return val

# Logical mapping from table names to report section IDs for contextual placement
_TABLE_SECTION_MAP: dict[str, str] = {
    "Analyses_Performed_Audit_Log": "analyses_performed",
    "Audit_Lineage": "analyses_performed",
    "Leakage_Audit_and_Split_Protocol": "quality_diagnostics",
    "Pipeline_CV_Summary": "modeling_leaderboard",
    "Excluded_Data_and_Reasons": "data_exclusions",
    "Raw_vs_Processed_Forensic_Diff": "data_exclusions",
    "Numeric_Range_Shifts_After_Cleaning": "data_exclusions",
    "Data_Quality_6D_Breakdown": "quality_diagnostics",
    "Column_Quality_Profile": "quality_diagnostics",
    "Data_Validation_Rules": "quality_diagnostics",
    "Clustering_Summary": "clustering_anomalies",
    "Cluster_Profiles": "clustering_anomalies",
    "Anomaly_Screening_Summary": "clustering_anomalies",
    "Statistical_Test_Ledger": "eda_hypothesis",
    "Top_Feature_Correlations": "eda_hypothesis",
    "Feature_Selection_Rankings": "feature_engineering",
    "AutoML_Leaderboard": "modeling_leaderboard",
    "Baseline_Benchmarks": "modeling_leaderboard",
    "Deep_Learning_vs_Baselines": "modeling_leaderboard",
    "Multi_Target_Summary": "modeling_leaderboard",
    "Model_Mathematical_Equations": "mathematical_equations",
    "Future_Target_Engine_Forecasts": "future_predictions",
    "Future_Target_Predictions": "future_predictions",
    "Risk_Assessment_Summary": "decision_intelligence",
    "Dataset_Summary_Statistics": "executive_summary",
    "Active_Dataset_Sample": "executive_summary",
}


def _clean_text(val: Any, max_len: int | None = None) -> str:
    """Convert any scalar/collection cleanly to display text."""
    if val is None:
        return ""
    if isinstance(val, float):
        if val != val:  # NaN
            return ""
        text = f"{val:.4f}" if abs(val) < 10_000 and val != int(val) else f"{val:,.2f}" if abs(val) >= 10_000 else str(val)
    elif isinstance(val, (list, tuple)):
        text = ", ".join(str(x) for x in val[:15])
    elif isinstance(val, dict):
        text = json.dumps(val, default=safe_json_default)
    else:
        text = str(val)
    text = text.replace("\r\n", "\n").strip()
    if max_len is not None and len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _decode_chart_png(chart: dict[str, Any]) -> bytes | None:
    """Decode base64 PNG payload from a chart artifact dict."""
    raw = chart.get("png_base64")
    if not raw or not isinstance(raw, str):
        return None
    try:
        return base64.b64decode(raw)
    except Exception:
        return None


def _safe_sheet_name(name: str, existing: set[str]) -> str:
    """Produce a valid, unique <=31-char Excel worksheet title."""
    cleaned = re.sub(r"[\\/*?:\[\]]+", "_", str(name)).strip("_ ") or "Sheet"
    base = cleaned[:31]
    candidate = base
    idx = 2
    while candidate in existing:
        suffix = f"_{idx}"
        candidate = f"{base[: 31 - len(suffix)]}{suffix}"
        idx += 1
    existing.add(candidate)
    return candidate


def _group_charts_by_section(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ch in payload.get("charts") or []:
        if isinstance(ch, dict):
            sec = str(ch.get("section_id") or "executive_summary")
            grouped.setdefault(sec, []).append(ch)
    return grouped


def _group_tables_by_section(payload: dict[str, Any]) -> tuple[dict[str, list[tuple[str, list[dict[str, Any]]]]], list[tuple[str, list[dict[str, Any]]]]]:
    grouped: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {}
    unmatched: list[tuple[str, list[dict[str, Any]]]] = []
    valid_section_ids = {str(s.get("section_id", "")) for s in (payload.get("sections") or []) if isinstance(s, dict)}
    for tname, rows in (payload.get("tables") or {}).items():
        if not (rows and isinstance(rows, list) and isinstance(rows[0], dict)):
            continue
        mapped_sec = _TABLE_SECTION_MAP.get(tname)
        if mapped_sec and mapped_sec in valid_section_ids:
            grouped.setdefault(mapped_sec, []).append((tname, rows))
        else:
            unmatched.append((tname, rows))
    return grouped, unmatched


class ReportExporters:
    """Exports comprehensive OmniForge reports with tables, graphs, audit logs, exclusions, and equations."""

    @staticmethod
    def to_markdown(payload: dict[str, Any], output_path: str | Path) -> Path:
        """Write complete structured report payload as formatted Markdown with tables and equations."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# {payload.get('title', 'OmniForge Report')}",
            (
                f"_Generated at: {payload.get('generated_at', '')} | Study: {payload.get('domain', 'general')} "
                f"| Author: {payload.get('author', 'OmniForge Enterprise')}_"
            ),
            "",
        ]
        if payload.get("kpis"):
            lines.append("## Key Performance Indicators")
            lines.append("| KPI | Value |")
            lines.append("|---|---|")
            for k, v in payload["kpis"].items():
                lines.append(f"| {k} | `{v}` |")
            lines.append("")

        for sec in payload.get("sections", []):
            lines.extend([f"## {sec.get('heading', 'Section')}", str(sec.get("body", "")), ""])
            for b in sec.get("bullets") or []:
                lines.append(f"- {b}")
            if sec.get("bullets"):
                lines.append("")

        if payload.get("formulas_and_rules"):
            lines.append("## Mathematical Relationships, Model Equations & Computational Rules")
            for f in payload["formulas_and_rules"]:
                lines.extend(
                    [
                        f"### [{f.get('category', 'Rule')}] {f.get('name', '')}",
                        f"- **Mathematical Expression:** `{f.get('mathematical_expression', '')}`",
                        f"- **Variables & Parameters:** {f.get('variables_and_parameters', '')}",
                        f"- **Computational Rule:** {f.get('computational_rule', '')}",
                        f"- **Detailed Explanation:** {f.get('detailed_explanation', '')}",
                        "",
                    ]
                )

        for tname, rows in (payload.get("tables") or {}).items():
            if rows and isinstance(rows, list) and isinstance(rows[0], dict):
                cols = list(rows[0].keys())[:12]
                lines.append(f"## Table: {tname.replace('_', ' ')}")
                lines.append("| " + " | ".join(str(c) for c in cols) + " |")
                lines.append("|" + "|".join("---" for _ in cols) + "|")
                for row in rows[:50]:
                    vals = [_clean_text(row.get(c, ""), 90).replace("|", "/").replace("\n", " ") for c in cols]
                    lines.append("| " + " | ".join(vals) + " |")
                lines.append("")

        if payload.get("recommendations"):
            lines.append("## Prioritized Recommendations")
            for idx, rec in enumerate(payload["recommendations"], start=1):
                lines.append(f"{idx}. {rec}")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    @staticmethod
    def to_html(payload: dict[str, Any], output_path: str | Path) -> Path:
        """Write a rich, self-contained HTML executive report with inline CSS, tables, equations, and embedded PNG charts."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        max_rows = int(payload.get("max_table_rows", 100))

        kpi_cards = "".join(
            f"<div class='kpi'><div class='lbl'>{html_escape(str(k))}</div><div class='val'>{html_escape(str(v))}</div></div>"
            for k, v in (payload.get("kpis") or {}).items()
        )

        charts_by_sec = _group_charts_by_section(payload)
        tables_by_sec, extra_tables = _group_tables_by_section(payload)

        def _render_table_html(tname: str, rows: list[dict[str, Any]]) -> str:
            headers = list(rows[0].keys())
            th = "".join(f"<th>{html_escape(str(h))}</th>" for h in headers)
            trs = "".join(
                "<tr>" + "".join(f"<td>{html_escape(_clean_text(row.get(h, ''), 220))}</td>" for h in headers) + "</tr>"
                for row in rows[:max_rows]
            )
            note = f"<div class='tbl-note'>Showing first {max_rows} of {len(rows):,} rows.</div>" if len(rows) > max_rows else ""
            return (
                f"<div class='tbl-wrap'><h3>{html_escape(tname.replace('_', ' '))}</h3>"
                f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>{note}</div>"
            )

        def _render_charts_html(ch_list: list[dict[str, Any]]) -> str:
            blocks = []
            for ch in ch_list:
                b64 = ch.get("png_base64")
                if not b64 or not isinstance(b64, str):
                    continue
                safe_b64 = "".join(c for c in b64 if c.isalnum() or c in "+/=")
                if not safe_b64:
                    continue
                blocks.append(
                    f"<figure class='chart-box'>"
                    f"<img src='data:image/png;base64,{safe_b64}' alt='{html_escape(str(ch.get('title', 'Chart')), quote=True)}' />"
                    f"<figcaption><strong>{html_escape(str(ch.get('title', '')), quote=True)}:</strong> {html_escape(str(ch.get('caption', '')), quote=True)}</figcaption>"
                    f"</figure>"
                )
            return "".join(blocks)

        sections_blocks: list[str] = []
        rendered_chart_ids: set[str] = set()
        for sec in payload.get("sections", []):
            sid = str(sec.get("section_id", ""))
            bullets_html = ""
            if sec.get("bullets"):
                bullets_html = "<ul>" + "".join(f"<li>{html_escape(str(b))}</li>" for b in sec["bullets"]) + "</ul>"

            sec_charts = charts_by_sec.get(sid, [])
            for ch in sec_charts:
                rendered_chart_ids.add(str(ch.get("chart_id", "")))
            charts_html = _render_charts_html(sec_charts)

            sec_tables_html = "".join(_render_table_html(tn, tr) for tn, tr in tables_by_sec.get(sid, []))

            formulas_html = ""
            if sid == "mathematical_equations" and payload.get("formulas_and_rules"):
                f_cards = []
                for f in payload["formulas_and_rules"]:
                    f_cards.append(
                        "<div class='formula-card'>"
                        f"<div class='f-hdr'><span class='f-cat'>{html_escape(str(f.get('category', '')))}</span> "
                        f"<strong>{html_escape(str(f.get('name', '')))}</strong></div>"
                        f"<pre class='f-eq'>{html_escape(str(f.get('mathematical_expression', '')))}</pre>"
                        f"<p><strong>Variables &amp; Parameters:</strong> {html_escape(str(f.get('variables_and_parameters', '')))}</p>"
                        f"<p><strong>Computational Rule:</strong> {html_escape(str(f.get('computational_rule', '')))}</p>"
                        f"<p><strong>Explanation:</strong> {html_escape(str(f.get('detailed_explanation', '')))}</p>"
                        "</div>"
                    )
                formulas_html = "".join(f_cards)

            sections_blocks.append(
                f"<section class='card'>"
                f"<h2>{html_escape(str(sec.get('heading', '')))}</h2>"
                f"<p>{html_escape(str(sec.get('body', '')))}</p>"
                f"{bullets_html}{formulas_html}{charts_html}{sec_tables_html}"
                f"</section>"
            )

        # Any remaining charts whose section wasn't rendered explicitly
        remaining_charts = [
            ch for ch in (payload.get("charts") or []) if isinstance(ch, dict) and str(ch.get("chart_id", "")) not in rendered_chart_ids
        ]
        if remaining_charts:
            sections_blocks.append(f"<section class='card'><h2>Additional Visual Analytics</h2>{_render_charts_html(remaining_charts)}</section>")

        if extra_tables:
            extra_html = "".join(_render_table_html(tn, tr) for tn, tr in extra_tables)
            sections_blocks.append(f"<section class='card'><h2>Additional Analytical Tables</h2>{extra_html}</section>")

        recs_block = ""
        if payload.get("recommendations"):
            recs_html = "".join(f"<li>{html_escape(str(r))}</li>" for r in payload["recommendations"])
            recs_block = f"<section class='card'><h2>Prioritized Action Plan &amp; Recommendations</h2><ol>{recs_html}</ol></section>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{html_escape(str(payload.get('title', 'OmniForge Executive Report')))}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 28px; line-height: 1.55; }}
  .container {{ max-width: 1180px; margin: 0 auto; }}
  header {{ background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); color: #f8fafc; padding: 28px 32px; border-radius: 12px; margin-bottom: 22px; }}
  header h1 {{ margin: 0; font-size: 1.9rem; }}
  .meta {{ color: #cbd5e1; font-size: 0.9rem; margin-top: 8px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 12px; margin-bottom: 22px; }}
  .kpi {{ background: #ffffff; border: 1px solid #cbd5e1; border-left: 4px solid #2563eb; border-radius: 8px; padding: 14px; box-shadow: 0 1px 2px rgba(15,23,42,0.05); }}
  .kpi .lbl {{ color: #475569; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; }}
  .kpi .val {{ font-size: 1.25rem; font-weight: 700; color: #0f172a; margin-top: 4px; word-break: break-word; }}
  .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 22px 26px; margin-bottom: 18px; box-shadow: 0 1px 3px rgba(15,23,42,0.05); }}
  h2 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin-top: 0; font-size: 1.32rem; }}
  h3 {{ color: #1e3a8a; font-size: 1.05rem; margin-top: 16px; margin-bottom: 8px; }}
  .tbl-wrap {{ overflow-x: auto; margin-top: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.84rem; }}
  th, td {{ padding: 7px 10px; border: 1px solid #cbd5e1; text-align: left; vertical-align: top; }}
  th {{ background: #0f172a; color: #f8fafc; font-weight: 600; }}
  tbody tr:nth-child(even) {{ background: #f8fafc; }}
  .tbl-note {{ font-size: 0.78rem; color: #64748b; margin-top: 4px; }}
  .chart-box {{ margin: 16px 0; text-align: center; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }}
  .chart-box img {{ max-width: 100%; height: auto; border-radius: 4px; }}
  .chart-box figcaption {{ font-size: 0.85rem; color: #334155; margin-top: 8px; text-align: left; }}
  .formula-card {{ background: #f8fafc; border: 1px solid #cbd5e1; border-left: 4px solid #0ea5e9; border-radius: 8px; padding: 14px 16px; margin: 12px 0; }}
  .f-hdr {{ margin-bottom: 8px; font-size: 0.98rem; }}
  .f-cat {{ display: inline-block; background: #0f172a; color: #f8fafc; font-size: 0.74rem; padding: 2px 8px; border-radius: 4px; margin-right: 6px; }}
  .f-eq {{ background: #0f172a; color: #e2e8f0; padding: 10px 12px; border-radius: 6px; font-family: Consolas, 'Courier New', monospace; font-size: 0.85rem; white-space: pre-wrap; word-break: break-word; margin: 8px 0; }}
  @media print {{ body {{ background: white; padding: 0; }} .card {{ box-shadow: none; break-inside: avoid; }} }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>{html_escape(str(payload.get('title', 'OmniForge Report')))}</h1>
    <div class="meta">OmniForge Enterprise | Study: {html_escape(str(payload.get('domain', 'general')))} | Author: {html_escape(str(payload.get('author', 'OmniForge Enterprise')))} | Generated: {html_escape(str(payload.get('generated_at', '')))}</div>
  </header>
  <div class="kpis">{kpi_cards}</div>
  {''.join(sections_blocks)}
  {recs_block}
</div>
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
        return path

    @staticmethod
    def to_json(payload: dict[str, Any], output_path: str | Path) -> Path:
        """Write full structured report payload as JSON (omitting giant raw base64 blobs by summarizing chart metadata)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        compact = dict(payload)
        if "charts" in compact and isinstance(compact["charts"], list):
            compact["charts"] = [
                {k: v for k, v in ch.items() if k != "png_base64"} | {"has_png_image": bool(ch.get("png_base64"))}
                for ch in compact["charts"]
                if isinstance(ch, dict)
            ]
        path.write_text(json.dumps(compact, indent=2, default=safe_json_default), encoding="utf-8")
        return path

    @staticmethod
    def to_csv(payload: dict[str, Any], output_path: str | Path) -> Path:
        """Export comprehensive, UTF-8-BOM unified CSV report covering all selected sections, exclusions, and equations."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        max_rows = int(payload.get("max_table_rows", 200))

        fieldnames = [
            "section_id",
            "section_title",
            "record_type",
            "item_name",
            "metric_or_rule",
            "value_or_expression",
            "explanation_or_reason",
            "extra_details",
        ]
        rows_out: list[dict[str, str]] = [
            {
                "section_id": "metadata",
                "section_title": "Report Metadata",
                "record_type": "metadata",
                "item_name": "Title",
                "metric_or_rule": "report_title",
                "value_or_expression": _clean_text(payload.get("title", "OmniForge Report")),
                "explanation_or_reason": f"Study: {payload.get('domain', 'general')} | Author: {payload.get('author', 'OmniForge Enterprise')}",
                "extra_details": f"Generated: {payload.get('generated_at', '')}",
            }
        ]

        for k, v in (payload.get("kpis") or {}).items():
            rows_out.append(
                {
                    "section_id": "executive_summary",
                    "section_title": "Executive KPIs",
                    "record_type": "kpi",
                    "item_name": str(k),
                    "metric_or_rule": "KPI",
                    "value_or_expression": _clean_text(v),
                    "explanation_or_reason": "",
                    "extra_details": "",
                }
            )

        for sec in payload.get("sections") or []:
            sid = str(sec.get("section_id", "section"))
            stitle = str(sec.get("heading", sid))
            rows_out.append(
                {
                    "section_id": sid,
                    "section_title": stitle,
                    "record_type": "narrative",
                    "item_name": stitle,
                    "metric_or_rule": "summary",
                    "value_or_expression": _clean_text(sec.get("body", "")),
                    "explanation_or_reason": " | ".join(_clean_text(b) for b in (sec.get("bullets") or [])),
                    "extra_details": "",
                }
            )

        for ap in payload.get("analyses_performed") or []:
            rows_out.append(
                {
                    "section_id": "analyses_performed",
                    "section_title": "Audit Log of Analyses Performed",
                    "record_type": "analysis_audit",
                    "item_name": f"{ap.get('studio', '')} — {ap.get('analysis_name', '')}",
                    "metric_or_rule": _clean_text(ap.get("criteria_or_config", "")),
                    "value_or_expression": _clean_text(ap.get("outcome_summary", "")),
                    "explanation_or_reason": f"Inputs: {_clean_text(ap.get('inputs', ''))} | Mode: {ap.get('dataset_mode', 'RAW')}",
                    "extra_details": f"Timestamp: {ap.get('timestamp', '')} | Status: {ap.get('status', 'Completed')}",
                }
            )

        for ev in payload.get("exclusions") or []:
            rows_out.append(
                {
                    "section_id": "data_exclusions",
                    "section_title": "Excluded Data & Reasons",
                    "record_type": "data_exclusion",
                    "item_name": f"{ev.get('stage', '')} — {ev.get('exclusion_type', '')}",
                    "metric_or_rule": _clean_text(ev.get("criterion_rule", "")),
                    "value_or_expression": f"{ev.get('affected_scope', '')} (count={ev.get('count', 0)}, pct={ev.get('pct_of_dataset', 0)}%)",
                    "explanation_or_reason": _clean_text(ev.get("reason", "")),
                    "extra_details": f"Samples: {', '.join(str(x) for x in (ev.get('sample_identifiers') or [])[:10])}",
                }
            )

        for f in payload.get("formulas_and_rules") or []:
            rows_out.append(
                {
                    "section_id": "mathematical_equations",
                    "section_title": "Mathematical Relationships, Equations & Computational Rules",
                    "record_type": "formula_or_equation",
                    "item_name": f"[{f.get('category', '')}] {f.get('name', '')}",
                    "metric_or_rule": _clean_text(f.get("computational_rule", "")),
                    "value_or_expression": _clean_text(f.get("mathematical_expression", "")),
                    "explanation_or_reason": _clean_text(f.get("detailed_explanation", "")),
                    "extra_details": f"Variables: {_clean_text(f.get('variables_and_parameters', ''))} | Scope: {f.get('scope', '')}",
                }
            )

        for tname, rows in (payload.get("tables") or {}).items():
            if not (rows and isinstance(rows, list) and isinstance(rows[0], dict)):
                continue
            sid = _TABLE_SECTION_MAP.get(tname, "tables")
            cols = list(rows[0].keys())
            for r_idx, row in enumerate(rows[:max_rows], start=1):
                first_col = cols[0]
                rest = {c: row.get(c, "") for c in cols[1:]}
                rows_out.append(
                    {
                        "section_id": sid,
                        "section_title": tname.replace("_", " "),
                        "record_type": "table_row",
                        "item_name": f"{tname}[{r_idx}]",
                        "metric_or_rule": str(first_col),
                        "value_or_expression": _clean_text(row.get(first_col, "")),
                        "explanation_or_reason": "; ".join(f"{k}={_clean_text(v, 80)}" for k, v in list(rest.items())[:8]),
                        "extra_details": json.dumps(row, default=safe_json_default) if len(cols) > 9 else "",
                    }
                )

        for idx, rec in enumerate(payload.get("recommendations") or [], start=1):
            rows_out.append(
                {
                    "section_id": "recommendations",
                    "section_title": "Prioritized Recommendations",
                    "record_type": "recommendation",
                    "item_name": f"Recommendation #{idx}",
                    "metric_or_rule": f"Priority {idx}",
                    "value_or_expression": _clean_text(rec),
                    "explanation_or_reason": "",
                    "extra_details": "",
                }
            )

        # UTF-8 with BOM so Microsoft Excel opens Unicode/symbols cleanly on Windows & macOS
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
        return path

    @staticmethod
    def to_csv_bundle(payload: dict[str, Any], output_zip_path: str | Path) -> Path:
        """Export a ZIP pack containing clean individual CSV files for every selected section & analytical table."""
        zip_path = Path(output_zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        def _dicts_to_csv_bytes(rows: list[dict[str, Any]]) -> bytes:
            buf = io.StringIO()
            if not rows:
                return b""
            cols: list[str] = []
            for r in rows:
                for k in r:
                    if k not in cols:
                        cols.append(k)
            w = csv.DictWriter(buf, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: _clean_text(r.get(c, "")) for c in cols})
            return buf.getvalue().encode("utf-8-sig")

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1. Master unified CSV
            tmp_master = zip_path.parent / "_tmp_master_report.csv"
            ReportExporters.to_csv(payload, tmp_master)
            zf.write(tmp_master, arcname="00_complete_unified_report.csv")
            tmp_master.unlink(missing_ok=True)

            # 2. KPIs
            if payload.get("kpis"):
                kpi_rows = [{"Metric": k, "Value": v} for k, v in payload["kpis"].items()]
                zf.writestr("01_executive_kpis.csv", _dicts_to_csv_bytes(kpi_rows))

            # 3. Analyses Performed
            if payload.get("analyses_performed"):
                zf.writestr("02_analyses_performed_audit.csv", _dicts_to_csv_bytes(payload["analyses_performed"]))

            # 4. Data Exclusions
            if payload.get("exclusions"):
                zf.writestr("03_excluded_data_and_reasons.csv", _dicts_to_csv_bytes(payload["exclusions"]))

            # 5. Mathematical Equations & Rules
            if payload.get("formulas_and_rules"):
                zf.writestr("04_mathematical_equations_and_rules.csv", _dicts_to_csv_bytes(payload["formulas_and_rules"]))

            # 6. Individual analytical tables
            for idx, (tname, rows) in enumerate((payload.get("tables") or {}).items(), start=5):
                if rows and isinstance(rows, list) and isinstance(rows[0], dict):
                    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", tname).strip("_").lower()
                    zf.writestr(f"{idx:02d}_{safe_name}.csv", _dicts_to_csv_bytes(rows))

        return zip_path

    @staticmethod
    def to_excel(payload: dict[str, Any], output_path: str | Path) -> Path:
        """Export richly styled multi-sheet Excel workbook (.xlsx) with tables, equations, exclusions, and embedded PNG charts."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        wb = openpyxl.Workbook()
        used_names: set[str] = set()

        # Reusable styles
        title_font = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
        hdr_font = Font(name="Calibri", size=10.5, bold=True, color="FFFFFF")
        bold_font = Font(name="Calibri", size=10, bold=True, color="0F172A")
        body_font = Font(name="Calibri", size=10, color="1E293B")
        mono_font = Font(name="Consolas", size=9.5, color="0F172A")
        navy_fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
        blue_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
        zebra_fill = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
        thin_border = Border(
            left=Side(style="thin", color="CBD5E1"),
            right=Side(style="thin", color="CBD5E1"),
            top=Side(style="thin", color="CBD5E1"),
            bottom=Side(style="thin", color="CBD5E1"),
        )
        wrap_align = Alignment(vertical="top", wrap_text=True)

        def _style_table_sheet(ws: Any, headers: list[str], rows: list[list[Any]], *, mono_cols: set[int] | None = None) -> None:
            ws.append(headers)
            for c_idx in range(1, len(headers) + 1):
                cell = ws.cell(row=1, column=c_idx)
                cell.font = hdr_font
                cell.fill = navy_fill
                cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
                cell.border = thin_border
            ws.row_dimensions[1].height = 24
            for r_idx, row_vals in enumerate(rows, start=2):
                ws.append(row_vals)
                fill = zebra_fill if r_idx % 2 == 0 else None
                for c_idx, _val in enumerate(row_vals, start=1):
                    cell = ws.cell(row=r_idx, column=c_idx)
                    cell.font = mono_font if (mono_cols and c_idx in mono_cols) else body_font
                    cell.alignment = wrap_align
                    cell.border = thin_border
                    if fill is not None:
                        cell.fill = fill
            ws.freeze_panes = "A2"
            if headers and rows:
                ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(rows) + 1}"
            # Auto-fit column widths sensibly (clamped 14..52)
            for c_idx in range(1, len(headers) + 1):
                max_len = len(str(headers[c_idx - 1]))
                for r_idx in range(2, min(len(rows) + 2, 120)):
                    v = ws.cell(row=r_idx, column=c_idx).value
                    if v is not None:
                        first_line_max = max((len(line) for line in str(v).splitlines()), default=0)
                        max_len = max(max_len, min(first_line_max, 52))
                ws.column_dimensions[get_column_letter(c_idx)].width = max(14, min(max_len + 4, 54))

        # Sheet 1: Executive Summary
        ws = wb.active
        ws.title = _safe_sheet_name("01_Executive_Summary", used_names)
        ws.merge_cells("A1:C1")
        ws["A1"] = str(payload.get("title", "OmniForge Report"))
        ws["A1"].font = title_font
        ws["A1"].fill = blue_fill
        ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
        ws.row_dimensions[1].height = 30

        meta_rows = [
            ("Study / Domain", str(payload.get("domain", "general")), ""),
            ("Author", str(payload.get("author", "OmniForge Enterprise")), ""),
            ("Generated At", str(payload.get("generated_at", "")), ""),
            ("Selected Sections", ", ".join(payload.get("selected_sections") or []), ""),
        ]
        for r in meta_rows:
            ws.append(list(r))
            ws.cell(row=ws.max_row, column=1).font = bold_font
            ws.cell(row=ws.max_row, column=2).alignment = wrap_align

        ws.append([])
        ws.append(["KPI Metric", "Value", "Notes"])
        kpi_hdr_row = ws.max_row
        for c in range(1, 4):
            ws.cell(row=kpi_hdr_row, column=c).font = hdr_font
            ws.cell(row=kpi_hdr_row, column=c).fill = navy_fill
        for k, v in (payload.get("kpis") or {}).items():
            ws.append([str(k), _clean_text(v), ""])
            for c in range(1, 4):
                ws.cell(row=ws.max_row, column=c).border = thin_border
                ws.cell(row=ws.max_row, column=c).alignment = wrap_align

        ws.append([])
        ws.append(["Section", "Executive Narrative", "Key Highlights"])
        sec_hdr_row = ws.max_row
        for c in range(1, 4):
            ws.cell(row=sec_hdr_row, column=c).font = hdr_font
            ws.cell(row=sec_hdr_row, column=c).fill = navy_fill
        for sec in payload.get("sections") or []:
            ws.append(
                [
                    str(sec.get("heading", "")),
                    str(sec.get("body", "")),
                    "\n".join(f"• {b}" for b in (sec.get("bullets") or [])),
                ]
            )
            for c in range(1, 4):
                ws.cell(row=ws.max_row, column=c).border = thin_border
                ws.cell(row=ws.max_row, column=c).alignment = wrap_align

        if payload.get("recommendations"):
            ws.append([])
            ws.append(["Priority", "Prioritized Recommendation", ""])
            rec_hdr = ws.max_row
            for c in range(1, 4):
                ws.cell(row=rec_hdr, column=c).font = hdr_font
                ws.cell(row=rec_hdr, column=c).fill = navy_fill
            for idx, rec in enumerate(payload["recommendations"], start=1):
                ws.append([idx, str(rec), ""])
                ws.cell(row=ws.max_row, column=2).alignment = wrap_align

        ws.column_dimensions["A"].width = 30
        ws.column_dimensions["B"].width = 62
        ws.column_dimensions["C"].width = 58

        # Sheet 2: Analyses Performed
        if payload.get("analyses_performed"):
            ws_ap = wb.create_sheet(_safe_sheet_name("02_Analyses_Performed", used_names))
            ap_cols = ["Timestamp", "Studio", "Analysis Name", "Dataset Mode", "Criteria / Config", "Inputs", "Outcome Summary", "Status"]
            ap_rows = [
                [
                    _clean_text(a.get("timestamp", "")),
                    _clean_text(a.get("studio", "")),
                    _clean_text(a.get("analysis_name", "")),
                    _clean_text(a.get("dataset_mode", "")),
                    _clean_text(a.get("criteria_or_config", "")),
                    _clean_text(a.get("inputs", "")),
                    _clean_text(a.get("outcome_summary", "")),
                    _clean_text(a.get("status", "")),
                ]
                for a in payload["analyses_performed"]
            ]
            _style_table_sheet(ws_ap, ap_cols, ap_rows)

        # Sheet 3: Excluded Data & Reasons
        if payload.get("exclusions"):
            ws_ex = wb.create_sheet(_safe_sheet_name("03_Excluded_Data_Reasons", used_names))
            ex_cols = ["Timestamp", "Stage", "Exclusion Type", "Affected Scope", "Count", "% of Dataset", "Exact Rule / Criterion", "Detailed Reason", "Sample IDs / Columns"]
            ex_rows = [
                [
                    _clean_text(e.get("timestamp", "")),
                    _clean_text(e.get("stage", "")),
                    _clean_text(e.get("exclusion_type", "")),
                    _clean_text(e.get("affected_scope", "")),
                    int(e.get("count", 0)),
                    float(e.get("pct_of_dataset", 0.0)),
                    _clean_text(e.get("criterion_rule", "")),
                    _clean_text(e.get("reason", "")),
                    ", ".join(str(x) for x in (e.get("sample_identifiers") or [])[:12]),
                ]
                for e in payload["exclusions"]
            ]
            _style_table_sheet(ws_ex, ex_cols, ex_rows, mono_cols={7})

        # Sheet 4: Mathematical Formulas, Model Equations & Computational Rules
        if payload.get("formulas_and_rules"):
            ws_eq = wb.create_sheet(_safe_sheet_name("04_Equations_and_Rules", used_names))
            eq_cols = ["Category", "Formula / Model Name", "Scope", "Mathematical Expression / Equation", "Variables & Parameters", "Computational Rule", "Detailed Explanation"]
            eq_rows = [
                [
                    _clean_text(f.get("category", "")),
                    _clean_text(f.get("name", "")),
                    _clean_text(f.get("scope", "")),
                    _clean_text(f.get("mathematical_expression", "")),
                    _clean_text(f.get("variables_and_parameters", "")),
                    _clean_text(f.get("computational_rule", "")),
                    _clean_text(f.get("detailed_explanation", "")),
                ]
                for f in payload["formulas_and_rules"]
            ]
            _style_table_sheet(ws_eq, eq_cols, eq_rows, mono_cols={4})

        # Remaining structured tables (avoiding duplicating already created sheets)
        skip_dup = {"Analyses_Performed_Audit_Log", "Excluded_Data_and_Reasons"}
        max_rows = int(payload.get("max_table_rows", 200))
        for tname, rows in (payload.get("tables") or {}).items():
            if tname in skip_dup and (
                (tname == "Analyses_Performed_Audit_Log" and payload.get("analyses_performed"))
                or (tname == "Excluded_Data_and_Reasons" and payload.get("exclusions"))
            ):
                continue
            if rows and isinstance(rows, list) and isinstance(rows[0], dict):
                sheet = wb.create_sheet(_safe_sheet_name(tname, used_names))
                cols = list(rows[0].keys())
                tbl_rows = [[_clean_text(r.get(c, "")) for c in cols] for r in rows[:max_rows]]
                _style_table_sheet(sheet, [str(c) for c in cols], tbl_rows)

        # Charts Gallery sheet with embedded PNG images
        charts = [ch for ch in (payload.get("charts") or []) if isinstance(ch, dict) and _decode_chart_png(ch)]
        if charts:
            try:
                from openpyxl.drawing.image import Image as XLImage

                ws_ch = wb.create_sheet(_safe_sheet_name("Charts_Gallery", used_names))
                ws_ch.column_dimensions["A"].width = 4
                ws_ch.column_dimensions["B"].width = 96
                cur_row = 1
                for idx, ch in enumerate(charts, start=1):
                    png_bytes = _decode_chart_png(ch)
                    if not png_bytes:
                        continue
                    ws_ch.cell(row=cur_row, column=2, value=f"Figure {idx}: {ch.get('title', 'Chart')}").font = Font(
                        name="Calibri", size=12, bold=True, color="0F172A"
                    )
                    cur_row += 1
                    ws_ch.cell(row=cur_row, column=2, value=str(ch.get("caption", ""))).font = Font(
                        name="Calibri", size=9.5, italic=True, color="475569"
                    )
                    ws_ch.cell(row=cur_row, column=2).alignment = wrap_align
                    cur_row += 1
                    xl_img = XLImage(io.BytesIO(png_bytes))
                    # Normalize display size to ~680px wide preserving aspect ratio
                    target_w = 680
                    if xl_img.width and xl_img.height:
                        ratio = target_w / float(xl_img.width)
                        xl_img.width = target_w
                        xl_img.height = int(xl_img.height * ratio)
                    ws_ch.add_image(xl_img, f"B{cur_row}")
                    cur_row += max(18, int((xl_img.height or 340) / 18) + 3)
            except Exception as exc:
                logger.warning("excel_chart_sheet_attachment_skipped", error=str(exc))

        wb.save(path)
        return path

    @staticmethod
    def to_docx(payload: dict[str, Any], output_path: str | Path) -> Path:
        """Export comprehensive, publication-grade Microsoft Word document (.docx) with styled tables, graphs, and equations."""
        try:
            from docx import Document
            from docx.enum.table import WD_TABLE_ALIGNMENT
            from docx.oxml import OxmlElement, parse_xml
            from docx.oxml.ns import nsdecls, qn
            from docx.shared import Inches, Pt, RGBColor
        except ImportError as exc:
            raise ReportingError("Install 'python-docx' to export Word reports.", cause=exc) from exc

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = Document()

        # Configure clean page margins
        for section in doc.sections:
            section.top_margin = Inches(0.75)
            section.bottom_margin = Inches(0.75)
            section.left_margin = Inches(0.75)
            section.right_margin = Inches(0.75)
            footer_p = section.footer.paragraphs[0]
            footer_p.text = f"OmniForge Enterprise  |  {payload.get('title', 'Analytical Report')}"
            footer_p.runs[0].font.size = Pt(8.5)
            footer_p.runs[0].font.color.rgb = RGBColor(100, 116, 139)

        def _shade_cell(cell: Any, fill_hex: str) -> None:
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_pr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

        def _set_repeat_header(row: Any) -> None:
            tr_pr = row._tr.get_or_add_trPr()
            tr_pr.append(OxmlElement("w:tblHeader"))

        def _add_styled_table(headers: list[str], rows: list[list[str]], *, max_cols: int = 8) -> None:
            use_cols = headers[:max_cols]
            tbl = doc.add_table(rows=1, cols=len(use_cols))
            tbl.style = "Table Grid"
            tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
            hdr_row = tbl.rows[0]
            _set_repeat_header(hdr_row)
            for idx, h in enumerate(use_cols):
                cell = hdr_row.cells[idx]
                cell.text = ""
                p = cell.paragraphs[0]
                run = p.add_run(str(h))
                run.bold = True
                run.font.size = Pt(8.5)
                run.font.color.rgb = RGBColor(255, 255, 255)
                _shade_cell(cell, "0F172A")

            for r_i, r_vals in enumerate(rows):
                row_cells = tbl.add_row().cells
                for c_i in range(len(use_cols)):
                    val = r_vals[c_i] if c_i < len(r_vals) else ""
                    row_cells[c_i].text = ""
                    p = row_cells[c_i].paragraphs[0]
                    run = p.add_run(val)
                    run.font.size = Pt(8.0)
                    run.font.color.rgb = RGBColor(30, 41, 59)
                    if r_i % 2 == 1:
                        _shade_cell(row_cells[c_i], "F1F5F9")
            if len(headers) > max_cols:
                note_p = doc.add_paragraph(
                    f"Note: Displaying first {max_cols} of {len(headers)} columns for print readability. Full wide table is included in Excel (.xlsx) and CSV exports."
                )
                note_p.runs[0].italic = True
                note_p.runs[0].font.size = Pt(8.0)
                note_p.runs[0].font.color.rgb = RGBColor(100, 116, 139)

        # Cover Banner Table
        banner = doc.add_table(rows=1, cols=1)
        banner.alignment = WD_TABLE_ALIGNMENT.CENTER
        b_cell = banner.rows[0].cells[0]
        _shade_cell(b_cell, "0F172A")
        bp = b_cell.paragraphs[0]
        r_title = bp.add_run(str(payload.get("title", "OmniForge Report")) + "\n")
        r_title.bold = True
        r_title.font.size = Pt(18)
        r_title.font.color.rgb = RGBColor(255, 255, 255)
        r_sub = bp.add_run(
            f"OmniForge Enterprise  |  Study: {payload.get('domain', 'general')}  |  "
            f"Author: {payload.get('author', 'OmniForge Enterprise')}  |  Generated: {payload.get('generated_at', '')}"
        )
        r_sub.font.size = Pt(9.5)
        r_sub.font.color.rgb = RGBColor(186, 230, 253)
        doc.add_paragraph()

        # KPI Scorecard Table
        if payload.get("kpis"):
            doc.add_heading("Executive Key Performance Indicators (KPIs)", level=1)
            kpi_rows = [[str(k), _clean_text(v, 140)] for k, v in payload["kpis"].items()]
            _add_styled_table(["Metric", "Value"], kpi_rows, max_cols=2)
            doc.add_paragraph()

        charts_by_sec = _group_charts_by_section(payload)
        tables_by_sec, extra_tables = _group_tables_by_section(payload)
        max_rows = min(int(payload.get("max_table_rows", 60)), 60)
        fig_counter = 1
        rendered_chart_ids: set[str] = set()

        def _embed_chart_docx(ch: dict[str, Any]) -> None:
            nonlocal fig_counter
            png = _decode_chart_png(ch)
            if not png:
                return
            w_in = min(float(ch.get("width_inches", 6.2)), 6.6)
            pic_p = doc.add_paragraph()
            pic_p.add_run().add_picture(io.BytesIO(png), width=Inches(w_in))
            cap = doc.add_paragraph()
            r1 = cap.add_run(f"Figure {fig_counter} — {ch.get('title', 'Chart')}: ")
            r1.bold = True
            r1.italic = True
            r1.font.size = Pt(8.5)
            r2 = cap.add_run(str(ch.get("caption", "")))
            r2.italic = True
            r2.font.size = Pt(8.5)
            r2.font.color.rgb = RGBColor(71, 85, 105)
            fig_counter += 1

        # Render each selected section with narrative, bullets, equations, charts, and tables
        for sec in payload.get("sections") or []:
            sid = str(sec.get("section_id", ""))
            doc.add_heading(str(sec.get("heading", "Section")), level=1)
            body_txt = str(sec.get("body", "")).strip()
            if body_txt:
                p = doc.add_paragraph(body_txt)
                p.runs[0].font.size = Pt(10)
            for b in sec.get("bullets") or []:
                bp_item = doc.add_paragraph(str(b), style="List Bullet")
                bp_item.runs[0].font.size = Pt(9.5)

            # Detailed Mathematical Formulas / Equations Callout Cards
            if sid == "mathematical_equations" and payload.get("formulas_and_rules"):
                for f in payload["formulas_and_rules"]:
                    ftbl = doc.add_table(rows=4, cols=1)
                    ftbl.style = "Table Grid"
                    ftbl.alignment = WD_TABLE_ALIGNMENT.CENTER
                    # Row 0: Header
                    c0 = ftbl.rows[0].cells[0]
                    _shade_cell(c0, "1E3A8A")
                    r0 = c0.paragraphs[0].add_run(f"[{f.get('category', 'Rule')}]  {f.get('name', '')}")
                    r0.bold = True
                    r0.font.size = Pt(9.5)
                    r0.font.color.rgb = RGBColor(255, 255, 255)
                    # Row 1: Monospace Formula
                    c1 = ftbl.rows[1].cells[0]
                    _shade_cell(c1, "F1F5F9")
                    r1 = c1.paragraphs[0].add_run(str(f.get("mathematical_expression", "")))
                    r1.font.name = "Courier New"
                    r1.font.size = Pt(8.5)
                    r1.font.color.rgb = RGBColor(15, 23, 42)
                    # Row 2: Variables & Rule
                    c2 = ftbl.rows[2].cells[0]
                    p2 = c2.paragraphs[0]
                    rv_lbl = p2.add_run("Variables & Parameters: ")
                    rv_lbl.bold = True
                    rv_lbl.font.size = Pt(8.5)
                    rv_txt = p2.add_run(_clean_text(f.get("variables_and_parameters", ""), 500) + "\n")
                    rv_txt.font.size = Pt(8.5)
                    rr_lbl = p2.add_run("Computational Rule: ")
                    rr_lbl.bold = True
                    rr_lbl.font.size = Pt(8.5)
                    rr_txt = p2.add_run(_clean_text(f.get("computational_rule", ""), 500))
                    rr_txt.font.size = Pt(8.5)
                    # Row 3: Explanation
                    c3 = ftbl.rows[3].cells[0]
                    p3 = c3.paragraphs[0]
                    re_lbl = p3.add_run("Detailed Explanation: ")
                    re_lbl.bold = True
                    re_lbl.font.size = Pt(8.5)
                    re_txt = p3.add_run(_clean_text(f.get("detailed_explanation", ""), 700))
                    re_txt.font.size = Pt(8.5)
                    doc.add_paragraph()

            # Embed matching section charts
            for ch in charts_by_sec.get(sid, []):
                rendered_chart_ids.add(str(ch.get("chart_id", "")))
                _embed_chart_docx(ch)

            # Embed matching section tables
            for tname, rows in tables_by_sec.get(sid, []):
                doc.add_heading(tname.replace("_", " "), level=2)
                cols = list(rows[0].keys())
                tbl_data = [[_clean_text(r.get(c, ""), 120) for c in cols] for r in rows[:max_rows]]
                _add_styled_table([str(c) for c in cols], tbl_data)
                doc.add_paragraph()

        # Any remaining charts or tables not tied to a rendered section
        for ch in payload.get("charts") or []:
            if isinstance(ch, dict) and str(ch.get("chart_id", "")) not in rendered_chart_ids:
                _embed_chart_docx(ch)

        for tname, rows in extra_tables:
            doc.add_heading(tname.replace("_", " "), level=2)
            cols = list(rows[0].keys())
            tbl_data = [[_clean_text(r.get(c, ""), 120) for c in cols] for r in rows[:max_rows]]
            _add_styled_table([str(c) for c in cols], tbl_data)
            doc.add_paragraph()

        if payload.get("recommendations"):
            doc.add_heading("Prioritized Action Plan & Recommendations", level=1)
            for rec in payload["recommendations"]:
                rp = doc.add_paragraph(str(rec), style="List Number")
                rp.runs[0].font.size = Pt(9.5)

        # Ensure qn referenced cleanly (no unused import lint)
        _ = qn
        doc.save(str(path))
        return path

    @staticmethod
    def to_pdf(payload: dict[str, Any], output_path: str | Path) -> Path:
        """Export comprehensive multi-page PDF report with tables, embedded high-res charts, exclusions, and equations."""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                Image as RLImage,
            )
            from reportlab.platypus import (
                KeepTogether,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError as exc:
            raise ReportingError("Install 'reportlab' to export PDF reports.", cause=exc) from exc

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        styles = getSampleStyleSheet()

        s_title = ParagraphStyle("UAPTitle", parent=styles["Title"], fontSize=17, leading=21, textColor=colors.white, alignment=0)
        s_sub = ParagraphStyle("UAPSub", parent=styles["Normal"], fontSize=8.8, leading=12, textColor=colors.HexColor("#BAE6FD"))
        s_h1 = ParagraphStyle("UAPH1", parent=styles["Heading1"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#0F172A"))
        s_h2 = ParagraphStyle("UAPH2", parent=styles["Heading2"], fontSize=10.5, leading=13.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1E3A8A"))
        s_body = ParagraphStyle("UAPBody", parent=styles["BodyText"], fontSize=9.2, leading=13, textColor=colors.HexColor("#1E293B"))
        s_bullet = ParagraphStyle("UAPBullet", parent=s_body, leftIndent=14, firstLineIndent=-8, spaceAfter=2)
        s_cell = ParagraphStyle("UAPCell", parent=styles["Normal"], fontSize=7.8, leading=10, textColor=colors.HexColor("#1E293B"))
        s_cell_hdr = ParagraphStyle("UAPCellHdr", parent=styles["Normal"], fontSize=8.0, leading=10, textColor=colors.white, fontName="Helvetica-Bold")
        s_mono = ParagraphStyle("UAPMono", parent=styles["Normal"], fontName="Courier", fontSize=8.0, leading=10.5, textColor=colors.HexColor("#0F172A"))
        s_cap = ParagraphStyle("UAPCap", parent=styles["Normal"], fontSize=8.0, leading=10.5, textColor=colors.HexColor("#475569"), fontName="Helvetica-Oblique")

        report_title = str(payload.get("title", "OmniForge Report"))

        def _draw_chrome(canvas: Any, doc_obj: Any) -> None:
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
            canvas.setLineWidth(0.6)
            canvas.line(0.65 * inch, 0.55 * inch, letter[0] - 0.65 * inch, 0.55 * inch)
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#64748B"))
            canvas.drawString(0.65 * inch, 0.38 * inch, f"OmniForge Enterprise  |  {report_title[:68]}")
            canvas.drawRightString(letter[0] - 0.65 * inch, 0.38 * inch, f"Page {doc_obj.page}")
            canvas.restoreState()

        doc = SimpleDocTemplate(
            str(path),
            pagesize=letter,
            leftMargin=0.65 * inch,
            rightMargin=0.65 * inch,
            topMargin=0.65 * inch,
            bottomMargin=0.72 * inch,
        )
        usable_w = letter[0] - 1.30 * inch  # 7.2 inches
        story: list[Any] = []

        # Executive Header Banner
        banner_tbl = Table(
            [
                [Paragraph(xml_escape(report_title), s_title)],
                [
                    Paragraph(
                        xml_escape(
                            f"Study: {payload.get('domain', 'general')}  |  Author: {payload.get('author', 'OmniForge Enterprise')}  |  "
                            f"Generated: {payload.get('generated_at', '')}"
                        ),
                        s_sub,
                    )
                ],
            ],
            colWidths=[usable_w],
        )
        banner_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 14),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                    ("TOPPADDING", (0, 0), (-1, 0), 12),
                    ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
                ]
            )
        )
        story.extend([banner_tbl, Spacer(1, 12)])

        def _make_pdf_table(headers: list[str], rows: list[list[str]], *, max_cols: int = 7) -> list[Any]:
            use_cols = headers[:max_cols]
            data: list[list[Any]] = [[Paragraph(xml_escape(str(h)), s_cell_hdr) for h in use_cols]]
            for r in rows:
                data.append([Paragraph(xml_escape(r[i] if i < len(r) else ""), s_cell) for i in range(len(use_cols))])
            col_w = usable_w / max(len(use_cols), 1)
            tbl = Table(data, colWidths=[col_w] * len(use_cols), repeatRows=1)
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            elems: list[Any] = [tbl]
            if len(headers) > max_cols:
                elems.append(
                    Paragraph(
                        xml_escape(
                            f"Note: Showing first {max_cols} of {len(headers)} columns for PDF page layout. "
                            "Complete wide tables are included in XLSX and CSV exports."
                        ),
                        s_cap,
                    )
                )
            elems.append(Spacer(1, 8))
            return elems

        # KPI Grid
        if payload.get("kpis"):
            story.append(Paragraph("Executive Key Performance Indicators (KPIs)", s_h1))
            kpi_data = [[str(k), _clean_text(v, 140)] for k, v in payload["kpis"].items()]
            story.extend(_make_pdf_table(["Metric", "Value"], kpi_data, max_cols=2))

        charts_by_sec = _group_charts_by_section(payload)
        tables_by_sec, extra_tables = _group_tables_by_section(payload)
        max_rows = min(int(payload.get("max_table_rows", 50)), 50)
        fig_num = 1
        rendered_chart_ids: set[str] = set()

        def _append_chart_pdf(ch: dict[str, Any]) -> None:
            nonlocal fig_num
            png = _decode_chart_png(ch)
            if not png:
                return
            try:
                from PIL import Image as PILImage

                with PILImage.open(io.BytesIO(png)) as pil_im:
                    px_w, px_h = pil_im.size
                aspect = float(px_h) / max(float(px_w), 1.0)
            except Exception:
                aspect = 0.52
            img_w = min(usable_w, float(ch.get("width_inches", 6.4)) * inch)
            img_h = min(img_w * aspect, 4.4 * inch)
            img_w = img_h / max(aspect, 0.1)
            rl_img = RLImage(io.BytesIO(png), width=img_w, height=img_h)
            cap = Paragraph(
                f"<b>Figure {fig_num} — {xml_escape(str(ch.get('title', 'Chart')))}:</b> {xml_escape(str(ch.get('caption', '')))}",
                s_cap,
            )
            story.append(KeepTogether([rl_img, Spacer(1, 4), cap, Spacer(1, 10)]))
            fig_num += 1

        # Render each selected section
        for sec in payload.get("sections") or []:
            sid = str(sec.get("section_id", ""))
            story.append(Paragraph(xml_escape(str(sec.get("heading", "Section"))), s_h1))
            if sec.get("body"):
                story.append(Paragraph(xml_escape(str(sec["body"])), s_body))
                story.append(Spacer(1, 4))
            for b in sec.get("bullets") or []:
                story.append(Paragraph(f"• {xml_escape(str(b))}", s_bullet))
            if sec.get("bullets"):
                story.append(Spacer(1, 6))

            # Mathematical Formulas & Model Equations Callout Cards
            if sid == "mathematical_equations" and payload.get("formulas_and_rules"):
                for f in payload["formulas_and_rules"]:
                    card = Table(
                        [
                            [Paragraph(xml_escape(f"[{f.get('category', 'Rule')}]  {f.get('name', '')}"), s_cell_hdr)],
                            [Paragraph(xml_escape(_clean_text(f.get("mathematical_expression", ""), 800)).replace("\n", "<br/>"), s_mono)],
                            [
                                Paragraph(
                                    f"<b>Variables &amp; Parameters:</b> {xml_escape(_clean_text(f.get('variables_and_parameters', ''), 500))}<br/>"
                                    f"<b>Computational Rule:</b> {xml_escape(_clean_text(f.get('computational_rule', ''), 500))}<br/>"
                                    f"<b>Detailed Explanation:</b> {xml_escape(_clean_text(f.get('detailed_explanation', ''), 700))}",
                                    s_cell,
                                )
                            ],
                        ],
                        colWidths=[usable_w],
                    )
                    card.setStyle(
                        TableStyle(
                            [
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F1F5F9")),
                                ("BACKGROUND", (0, 2), (-1, 2), colors.white),
                                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#94A3B8")),
                                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                                ("TOPPADDING", (0, 0), (-1, -1), 5),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                            ]
                        )
                    )
                    story.append(KeepTogether([card, Spacer(1, 8)]))

            # Section charts
            for ch in charts_by_sec.get(sid, []):
                rendered_chart_ids.add(str(ch.get("chart_id", "")))
                _append_chart_pdf(ch)

            # Section tables
            for tname, rows in tables_by_sec.get(sid, []):
                story.append(Paragraph(xml_escape(tname.replace("_", " ")), s_h2))
                cols = list(rows[0].keys())
                tbl_rows = [[_clean_text(r.get(c, ""), 110) for c in cols] for r in rows[:max_rows]]
                story.extend(_make_pdf_table([str(c) for c in cols], tbl_rows))

        # Unmatched charts & tables
        for ch in payload.get("charts") or []:
            if isinstance(ch, dict) and str(ch.get("chart_id", "")) not in rendered_chart_ids:
                _append_chart_pdf(ch)

        for tname, rows in extra_tables:
            story.append(Paragraph(xml_escape(tname.replace("_", " ")), s_h2))
            cols = list(rows[0].keys())
            tbl_rows = [[_clean_text(r.get(c, ""), 110) for c in cols] for r in rows[:max_rows]]
            story.extend(_make_pdf_table([str(c) for c in cols], tbl_rows))

        if payload.get("recommendations"):
            story.append(Paragraph("Prioritized Action Plan & Recommendations", s_h1))
            for idx, rec in enumerate(payload["recommendations"], start=1):
                story.append(Paragraph(f"{idx}. {xml_escape(str(rec))}", s_bullet))

        doc.build(story, onFirstPage=_draw_chrome, onLaterPages=_draw_chrome)
        return path

    @staticmethod
    def to_pptx(payload: dict[str, Any], output_path: str | Path) -> Path:
        """Export executive PowerPoint presentation (.pptx) with KPIs, narrative slides, equations, and embedded charts."""
        try:
            from pptx import Presentation
            from pptx.util import Inches, Pt
        except ImportError as exc:
            raise ReportingError("Install 'python-pptx' to export PowerPoint decks.", cause=exc) from exc

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        prs = Presentation()

        # Title slide
        slide0 = prs.slides.add_slide(prs.slide_layouts[0])
        slide0.shapes.title.text = str(payload.get("title", "OmniForge"))
        slide0.placeholders[1].text = (
            f"Enterprise Analytics Executive Briefing\nStudy: {payload.get('domain', 'general')} | {payload.get('generated_at', '')}"
        )

        # KPI slide
        if payload.get("kpis"):
            slide_kpi = prs.slides.add_slide(prs.slide_layouts[1])
            slide_kpi.shapes.title.text = "Key Performance Indicators"
            tf = slide_kpi.placeholders[1].text_frame
            tf.clear()
            for k, v in list(payload["kpis"].items())[:10]:
                p = tf.add_paragraph()
                p.text = f"{k}: {v}"
                p.font.size = Pt(16)

        # Narrative slides
        for sec in payload.get("sections") or []:
            sl = prs.slides.add_slide(prs.slide_layouts[1])
            sl.shapes.title.text = str(sec.get("heading", ""))[:80]
            tf = sl.placeholders[1].text_frame
            tf.clear()
            p0 = tf.paragraphs[0]
            p0.text = str(sec.get("body", ""))[:600]
            p0.font.size = Pt(13)
            for b in (sec.get("bullets") or [])[:5]:
                pb = tf.add_paragraph()
                pb.text = f"• {str(b)[:140]}"
                pb.font.size = Pt(12)

        # Chart slides
        for ch in (payload.get("charts") or [])[:8]:
            if not isinstance(ch, dict):
                continue
            png = _decode_chart_png(ch)
            if not png:
                continue
            sl_c = prs.slides.add_slide(prs.slide_layouts[5])
            if sl_c.shapes.title is not None:
                sl_c.shapes.title.text = str(ch.get("title", "Analytical Chart"))[:75]
            sl_c.shapes.add_picture(io.BytesIO(png), Inches(0.8), Inches(1.4), width=Inches(8.4))

        # Recommendations slide
        if payload.get("recommendations"):
            sl_r = prs.slides.add_slide(prs.slide_layouts[1])
            sl_r.shapes.title.text = "Prioritized Recommendations"
            tf = sl_r.placeholders[1].text_frame
            tf.clear()
            for rec in payload["recommendations"][:8]:
                p = tf.add_paragraph()
                p.text = str(rec)
                p.font.size = Pt(14)

        prs.save(str(path))
        return path

    @staticmethod
    def to_jupyter_notebook(
        output_path: str | Path,
        *,
        title: str = "OmniForge Reproducible Analysis Notebook",
        source_path: str = "data.csv",
        target_column: str = "target",
        domain: str = "finance",
    ) -> Path:
        """Export an executable, reproducible Jupyter Notebook (.ipynb) reproducing the full OmniForge workflow."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_markdown_cell(
                f"# {title}\nAuto-generated by OmniForge Enterprise (`domain={domain}`)."
            ),
            nbformat.v4.new_code_cell(
                "import omniforge\n"
                "from omniforge.data import DataIngestionEngine, DataCleaner, DataQualityProfiler\n"
                "from omniforge.eda import AutomatedEDA\n"
                "from omniforge.automl import AutoMLPipelineSearch\n"
                "from omniforge.evaluation import InterpretabilityEngine, ModelCardGenerator\n"
            ),
            nbformat.v4.new_code_cell(
                f"ingested = DataIngestionEngine().ingest({source_path!r})\n"
                "df_raw = ingested.data\n"
                "quality = DataQualityProfiler().profile(df_raw)\n"
                "print('Quality Score:', quality.overall_score, quality.grade)\n"
                "df_clean, clean_report = DataCleaner().fit_transform(df_raw)\n"
                "df_clean.head()\n"
            ),
            nbformat.v4.new_code_cell(
                f"eda_report = AutomatedEDA().run(df_clean, target_column={target_column!r})\n"
                "print(eda_report.narrative)\n"
            ),
            nbformat.v4.new_code_cell(
                f"X = df_clean.drop(columns=[{target_column!r}])\n"
                f"y = df_clean[{target_column!r}]\n"
                "automl_res = AutoMLPipelineSearch(time_budget_seconds=60).fit(X, y)\n"
                "print('Champion:', automl_res.best_model_name, automl_res.primary_metric, automl_res.best_score)\n"
                "print('Lift over best baseline:', automl_res.lift_over_best_baseline)\n"
            ),
            nbformat.v4.new_code_cell(
                "card = ModelCardGenerator.build_model_card(automl_res.best_model.metadata_)\n"
                "print(card.to_markdown())\n"
            ),
        ]
        nbformat.write(nb, str(path))
        return path

    @staticmethod
    def export_microservice_bundle(
        output_dir: str | Path,
        *,
        model_artifact_path: str | Path,
        service_name: str = "uap-model-service",
    ) -> Path:
        """Export a self-contained Docker + FastAPI microservice bundle for one-click deployment."""
        bundle_dir = Path(output_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        app_code = f'''"""Auto-exported OmniForge FastAPI Inference Microservice: {service_name}."""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any
from omniforge.models.base import BaseOmniForgeModel

app = FastAPI(title="{service_name}", version="2.0.0")
MODEL = BaseOmniForgeModel.load("{Path(model_artifact_path).name}")


class PredictRequest(BaseModel):
    records: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, str]:
    return {{"status": "ok", "model": MODEL.name}}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    import pandas as pd
    df = pd.DataFrame(req.records)
    preds = MODEL.predict(df).tolist()
    return {{"model": MODEL.name, "predictions": preds}}
'''
        dockerfile = """FROM python:3.12-slim
WORKDIR /service
COPY . /service
RUN pip install --no-cache-dir fastapi uvicorn pandas scikit-learn joblib pydantic omniforge
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
"""
        (bundle_dir / "app.py").write_text(app_code, encoding="utf-8")
        (bundle_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
        src_art = Path(model_artifact_path)
        if src_art.exists():
            import shutil

            shutil.copy2(src_art, bundle_dir / src_art.name)
        return bundle_dir
