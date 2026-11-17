"""Comprehensive verification of OmniForge Reporting: DOCX, PDF, XLSX, CSV (+ Bundle), section selection, charts, exclusions & equations."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from docx import Document

from omniforge.data.cleaning import CleaningConfig, DataCleaner
from omniforge.data.quality import DataQualityProfiler
from omniforge.evaluation.equations import ModelEquationExtractor
from omniforge.models.baselines import BaselineBenchmarkSuite
from omniforge.models.clustering import ClusteringEngine
from omniforge.reporting import (
    REPORT_PRESETS,
    REPORT_SECTIONS_CATALOG,
    ReportGenerator,
    ReportSelectionConfig,
)


def _build_rich_reporting_fixtures() -> dict[str, object]:
    """Create a realistic messy dataset, clean it, cluster it, fit baselines, and extract equations."""
    rng = np.random.default_rng(77)
    n = 160
    x1 = rng.normal(50.0, 10.0, n)
    x2 = rng.normal(20.0, 4.0, n)
    x3 = 0.7 * x1 + rng.normal(0.0, 3.0, n)
    y = 2.5 * x1 - 1.8 * x2 + 0.4 * x3 + rng.normal(0.0, 4.0, n)

    raw_df = pd.DataFrame(
        {
            "sensor_temp": x1,
            "pressure_bar": x2,
            "vibration_rms": x3,
            "plant_zone": rng.choice(["North", "South", "East"], size=n),
            "target_rul": y,
        }
    )
    # Inject realistic missing values, extreme outliers, and exact duplicates
    raw_df.loc[[3, 15, 28, 44], "sensor_temp"] = np.nan
    raw_df.loc[[5, 19], "pressure_bar"] = [999.5, -250.0]
    raw_df = pd.concat([raw_df, raw_df.iloc[:6]], ignore_index=True)

    cfg = CleaningConfig(
        numeric_imputation="median",
        categorical_imputation="mode",
        outlier_method="iqr",
        outlier_action="clip",
        drop_duplicates=True,
        exclude_columns=["target_rul"],
    )
    clean_df, clean_rep = DataCleaner(cfg).fit_transform(raw_df)
    quality = DataQualityProfiler().profile(clean_df).model_dump()

    clust_res = ClusteringEngine(
        algorithm="kmeans",
        n_clusters=3,
        random_seed=42,
    ).fit_predict_with_profile(
        clean_df,
        feature_cols=["sensor_temp", "pressure_bar", "vibration_rms"],
    )

    X = clean_df[["sensor_temp", "pressure_bar", "vibration_rms"]]
    y_ser = clean_df["target_rul"]
    bl_metrics, bl_models = BaselineBenchmarkSuite.run_regression_detailed(X, y_ser, X, y_ser)
    eq_map: dict[str, dict[str, object]] = {}
    for m_name, m_obj in bl_models.items():
        eq_map[m_name] = ModelEquationExtractor.extract(
            m_obj,
            X_reference=X,
            target_name="target_rul",
        ).model_dump()

    analyses = [
        {
            "timestamp": "2026-11-15 19:00:00 UTC",
            "studio": "Studio 01 · Cleaning",
            "analysis_name": "Data Cleaning (median imputation, IQR clipping, deduplication)",
            "dataset_mode": "PROCESSED",
            "criteria_or_config": "IQR k=1.5 | drop_duplicates=True",
            "inputs": "plant_telemetry",
            "outcome_summary": f"Removed {clean_rep.exact_duplicates_removed} duplicates and clipped extreme sensor values.",
            "status": "Completed",
        },
        {
            "timestamp": "2026-11-15 19:01:00 UTC",
            "studio": "Studio 02 · Clustering",
            "analysis_name": "K-Means Segmentation (K=3)",
            "dataset_mode": "PROCESSED",
            "criteria_or_config": "Standardized 3 numeric sensors",
            "inputs": "plant_telemetry",
            "outcome_summary": f"Silhouette = {clust_res.metrics.get('silhouette', 0.0):.3f}",
            "status": "Completed",
        },
        {
            "timestamp": "2026-11-15 19:02:00 UTC",
            "studio": "Studio 03 · Baselines",
            "analysis_name": "Baseline Model Benchmarks & Equation Extraction",
            "dataset_mode": "PROCESSED",
            "criteria_or_config": "Task=regression",
            "inputs": "target_rul",
            "outcome_summary": f"Evaluated {len(bl_models)} baseline models and extracted exact equations.",
            "status": "Completed",
        },
    ]

    exclusions = [
        {
            "timestamp": "2026-11-15 19:00:00 UTC",
            "stage": "Studio 01 · Data Cleaning",
            "exclusion_type": "Rows Removed (Exact Duplicates)",
            "affected_scope": f"{clean_rep.exact_duplicates_removed} duplicate rows",
            "count": int(clean_rep.exact_duplicates_removed),
            "pct_of_dataset": round(clean_rep.exact_duplicates_removed / len(raw_df) * 100.0, 2),
            "criterion_rule": "Exact match across all columns (keep='first')",
            "reason": "Prevent duplicate leakage across cross-validation splits.",
            "sample_identifiers": ["row_160", "row_161"],
        },
        {
            "timestamp": "2026-11-15 19:00:00 UTC",
            "stage": "Studio 01 · Data Cleaning",
            "exclusion_type": "Values Clipped (Outliers)",
            "affected_scope": "Extreme pressure_bar spikes (999.5, -250.0)",
            "count": int(sum(clean_rep.outliers_detected.values())),
            "pct_of_dataset": 1.25,
            "criterion_rule": "Tukey 1.5*IQR fence (x < Q1-1.5*IQR or x > Q3+1.5*IQR)",
            "reason": "Sensor saturation artifacts clipped to preserve physical validity.",
            "sample_identifiers": ["pressure_bar"],
        },
    ]

    leaderboard = [
        {"model": m_name, "score": float(m_vals.get("r2", 0.0)), "rmse": float(m_vals.get("rmse", 0.0)), "is_baseline": True}
        for m_name, m_vals in bl_metrics.items()
    ]

    return {
        "raw_df": raw_df,
        "clean_df": clean_df,
        "clean_rep": clean_rep.model_dump(),
        "quality": quality,
        "clust_res": clust_res,
        "eq_map": {"target_rul": eq_map},
        "analyses": analyses,
        "exclusions": exclusions,
        "leaderboard": leaderboard,
    }


def test_comprehensive_report_exports_docx_pdf_xlsx_csv_with_tables_charts_exclusions_and_equations(tmp_path: Path) -> None:
    """Verify full report exports cleanly to DOCX, PDF, XLSX, CSV, CSV-Bundle & HTML with real tables, embedded charts, exclusions, and formulas."""
    fx = _build_rich_reporting_fixtures()
    gen = ReportGenerator()

    selection = ReportSelectionConfig(
        selected_sections=list(REPORT_PRESETS["Full Comprehensive Audit Report (All Sections)"]),
        include_charts=True,
        include_tables=True,
        include_dataset_appendix=True,
        max_table_rows=50,
        author="OmniForge QA Verification",
    )

    payload = gen.build_report(
        title="Plant Telemetry Comprehensive Audit Report",
        domain="industrial_iot",
        selection=selection,
        quality_report=fx["quality"],
        model_summary={"best_model_name": "LinearBaseline", "primary_metric": "r2", "best_score": 0.96, "lift_over_best_baseline": 0.12},
        tables={"AutoML_Leaderboard": fx["leaderboard"]},
        analyses_performed=fx["analyses"],
        exclusions=fx["exclusions"],
        clustering_summary={
            "algorithm": "kmeans",
            "clusters_found": 3,
            "noise_ratio": 0.0,
            "silhouette": fx["clust_res"].metrics.get("silhouette", 0.0),
        },
        df=fx["clean_df"],
        raw_df=fx["raw_df"],
        target_columns=["target_rul"],
        cleaning_report=fx["clean_rep"],
        clustering_result=fx["clust_res"],
        model_equations=fx["eq_map"],
    )

    # Payload richness assertions
    assert len(payload["analyses_performed"]) == 3
    assert len(payload["exclusions"]) == 2
    assert payload["raw_vs_processed"]["rows_removed"] == 6
    assert len(payload["formulas_and_rules"]) >= 5
    assert any("Fitted Model Equation" in f["category"] for f in payload["formulas_and_rules"])
    assert len(payload["charts"]) >= 4
    assert "Excluded_Data_and_Reasons" in payload["tables"]
    assert "Dataset_Summary_Statistics" in payload["tables"]

    # 1. Word (.docx) — verify embedded PNG charts, styled tables, exclusions, and mathematical equations
    docx_path = gen.export(payload, tmp_path / "full_report.docx", fmt="docx")
    assert docx_path.exists() and docx_path.stat().st_size > 20_000
    doc = Document(str(docx_path))
    full_docx_text = "\n".join(p.text for p in doc.paragraphs)
    assert "Excluded Data, Filtering & Removals Log" in full_docx_text
    assert "Mathematical Relationships, Model Equations & Computational Rules" in full_docx_text
    assert len(doc.tables) >= 6
    assert len(doc.inline_shapes) >= 4  # REAL embedded Matplotlib PNG graphs!

    # 2. PDF (.pdf) — verify valid multi-page PDF with embedded images and text
    pdf_path = gen.export(payload, tmp_path / "full_report.pdf", fmt="pdf")
    assert pdf_path.exists() and pdf_path.stat().st_size > 20_000
    pdf_bytes = pdf_path.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-")
    assert b"/Subtype /Image" in pdf_bytes  # REAL embedded PNG charts inside PDF!

    # 3. Excel (.xlsx) — verify styled sheets + Charts_Gallery with embedded PNG images
    xlsx_path = gen.export(payload, tmp_path / "full_report.xlsx", fmt="xlsx")
    assert xlsx_path.exists() and xlsx_path.stat().st_size > 20_000
    wb = openpyxl.load_workbook(xlsx_path)
    expected_sheets = {
        "01_Executive_Summary",
        "02_Analyses_Performed",
        "03_Excluded_Data_Reasons",
        "04_Equations_and_Rules",
        "Charts_Gallery",
    }
    assert expected_sheets.issubset(set(wb.sheetnames))
    assert len(wb["Charts_Gallery"]._images) >= 4  # REAL embedded PNG charts inside Excel!
    assert wb["04_Equations_and_Rules"].max_row >= 6

    # 4. Unified CSV (.csv) — verify UTF-8-BOM CSV with all record types & explanations
    csv_path = gen.export(payload, tmp_path / "full_report.csv", fmt="csv")
    assert csv_path.exists() and csv_path.stat().st_size > 2_000
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rec_types = {r["record_type"] for r in rows}
    assert {"kpi", "narrative", "analysis_audit", "data_exclusion", "formula_or_equation", "table_row"}.issubset(rec_types)
    assert any("IQR" in r["value_or_expression"] or "IQR" in r["metric_or_rule"] for r in rows)

    # 5. Per-table CSV bundle (.zip)
    csv_zip = gen.export(payload, tmp_path / "csv_pack.zip", fmt="csv_bundle")
    assert csv_zip.exists()
    with zipfile.ZipFile(csv_zip, "r") as zf:
        names = set(zf.namelist())
        assert "00_complete_unified_report.csv" in names
        assert "02_analyses_performed_audit.csv" in names
        assert "03_excluded_data_and_reasons.csv" in names
        assert "04_mathematical_equations_and_rules.csv" in names

    # 6. HTML & PPTX & Markdown & JSON also export cleanly
    for ext in ("html", "pptx", "md", "json"):
        out_p = gen.export(payload, tmp_path / f"full_report.{ext}", fmt=ext)
        assert out_p.exists() and out_p.stat().st_size > 500
    assert "data:image/png;base64," in (tmp_path / "full_report.html").read_text(encoding="utf-8")


def test_custom_user_section_selection_strictly_filters_report_contents(tmp_path: Path) -> None:
    """Verify user-selected section checkboxes strictly filter narratives, exclusions, formulas, tables, and charts."""
    fx = _build_rich_reporting_fixtures()
    gen = ReportGenerator()

    # User selects ONLY Executive Summary + Data Exclusions, and disables charts
    custom_cfg = ReportSelectionConfig(
        selected_sections=["executive_summary", "data_exclusions"],
        include_charts=False,
        include_tables=True,
        include_dataset_appendix=False,
    )
    payload = gen.build_report(
        title="Focused Exclusions Audit Only",
        domain="industrial_iot",
        selection=custom_cfg,
        quality_report=fx["quality"],
        analyses_performed=fx["analyses"],
        exclusions=fx["exclusions"],
        df=fx["clean_df"],
        raw_df=fx["raw_df"],
        target_columns=["target_rul"],
        model_equations=fx["eq_map"],
    )

    assert len(REPORT_SECTIONS_CATALOG) == 14
    assert [s["section_id"] for s in payload["sections"]] == ["executive_summary", "data_exclusions"]
    assert payload["analyses_performed"] == []
    assert payload["formulas_and_rules"] == []
    assert payload["charts"] == []
    assert len(payload["exclusions"]) == 2
    assert "Excluded_Data_and_Reasons" in payload["tables"]
    assert "Dataset_Summary_Statistics" not in payload["tables"]

    # Confirm DOCX & CSV reflect the exact custom selection cleanly
    docx_p = gen.export(payload, tmp_path / "custom.docx", fmt="docx")
    doc = Document(str(docx_p))
    assert len(doc.inline_shapes) == 0
    txt = "\n".join(p.text for p in doc.paragraphs)
    assert "Excluded Data, Filtering & Removals Log" in txt
    assert "Mathematical Relationships, Model Equations" not in txt

    csv_p = gen.export(payload, tmp_path / "custom.csv", fmt="csv")
    with csv_p.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["record_type"] != "formula_or_equation" for r in rows)
    assert any(r["record_type"] == "data_exclusion" for r in rows)
