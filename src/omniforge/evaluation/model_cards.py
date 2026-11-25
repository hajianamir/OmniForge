"""Enterprise Model Cards & Data Cards generator aligned with EU AI Act and NIST AI RMF."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as _xml_escape

from pydantic import BaseModel, Field

from omniforge.core.exceptions import ReportingError
from omniforge.core.types import CarbonReport, DatasetMetadata, ModelMetadata
from omniforge.evaluation.fairness import FairnessReport


class ModelCard(BaseModel):
    """Structured Model Card capturing provenance, metrics, baselines, fairness, robustness, and carbon."""

    card_version: str = "1.0"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    model_details: ModelMetadata
    intended_use: str = "Enterprise decision support and predictive analytics."
    out_of_scope_uses: list[str] = Field(
        default_factory=lambda: [
            "Autonomous high-stakes decisions without human oversight.",
            "Deployment on out-of-distribution populations without recalibration.",
        ]
    )
    evaluation_metrics: dict[str, Any] = Field(default_factory=dict)
    baseline_comparison: dict[str, Any] = Field(default_factory=dict)
    feature_importance: dict[str, float] = Field(default_factory=dict)
    fairness_assessment: FairnessReport | None = None
    robustness_assessment: dict[str, Any] = Field(default_factory=dict)
    carbon_footprint: CarbonReport | None = None
    compliance_frameworks: list[str] = Field(default_factory=lambda: ["GDPR", "EU-AI-Act", "NIST-AI-RMF"])
    limitations_and_risks: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the Model Card as publication-grade Markdown documentation."""
        md = [
            f"# Model Card: {self.model_details.name}",
            f"- **Model ID**: `{self.model_details.model_id}`",
            f"- **Family / Framework**: `{self.model_details.model_family}` / `{self.model_details.framework}`",
            f"- **Task Type**: `{self.model_details.task_type.value}`",
            f"- **Version**: `{self.model_details.version}`",
            f"- **Generated At**: {self.generated_at}",
            f"- **Training Footprint**: {self.model_details.training_rows:,} rows × {self.model_details.training_features} features ({self.model_details.training_duration_seconds:.3f}s)",
            "",
            "## Intended Use & Governance",
            self.intended_use,
            "",
            f"**Compliance Profiles**: {', '.join(self.compliance_frameworks)}",
            "",
            "## Evaluation Performance",
        ]
        for k, v in self.evaluation_metrics.items():
            if not isinstance(v, (list, dict)):
                md.append(f"- **{k}**: `{v}`")

        if self.baseline_comparison:
            md.extend(["", "## Comparison Against Strong Baselines"])
            for bl_name, bl_scores in self.baseline_comparison.items():
                md.append(f"- **{bl_name}**: `{bl_scores}`")

        if self.feature_importance:
            md.extend(["", "## Top Predictive Drivers"])
            for feat, imp in list(self.feature_importance.items())[:10]:
                md.append(f"- `{feat}`: {imp:.4f}")

        if self.fairness_assessment is not None:
            fa = self.fairness_assessment
            md.extend(
                [
                    "",
                    "## Responsible AI — Fairness Audit",
                    f"- **Protected Attribute**: `{fa.protected_attribute}`",
                    f"- **Gate Status**: `{'PASSED' if fa.passed else 'REVIEW REQUIRED'}`",
                    f"- **Disparate Impact Ratio**: `{fa.disparate_impact_ratio}`",
                    f"- **Demographic Parity Diff**: `{fa.demographic_parity_diff}`",
                    f"- **Equalized Odds Diff**: `{fa.equalized_odds_diff}`",
                ]
            )

        if self.robustness_assessment:
            md.extend(
                [
                    "",
                    "## Robustness & Stress Testing",
                    f"- **Robustness Index (0–100)**: `{self.robustness_assessment.get('robustness_index_0_100', 'N/A')}`",
                    f"- **Clean Score**: `{self.robustness_assessment.get('clean_score', 'N/A')}`",
                    f"- **OOD Shift Score**: `{self.robustness_assessment.get('ood_shift_score', 'N/A')}`",
                ]
            )

        if self.carbon_footprint is not None:
            md.extend(["", "## Green AI — Carbon Footprint"])
            md.extend(f"- **{label}**: {value}" for label, value in self._carbon_rows())

        if self.limitations_and_risks:
            md.extend(["", "## Known Limitations & Risks"])
            for lim in self.limitations_and_risks:
                md.append(f"- {lim}")

        return "\n".join(md) + "\n"

    def _carbon_rows(self) -> list[tuple[str, str]]:
        cf = self.carbon_footprint
        if cf is None or cf.tracker_backend == "disabled":
            return [("Status", "Unmeasured (tracking disabled)")]
        energy_source = cf.metadata.get("energy_source", cf.tracker_backend)
        emissions_source = cf.metadata.get("emissions_source", cf.tracker_backend)
        return [
            ("Energy (kWh)", f"{cf.energy_kwh:.8f} ({energy_source})"),
            ("Emissions (gCO2eq)", f"{cf.emissions_g_co2eq:.6f} ({emissions_source})"),
            ("Green score", f"{cf.green_score}/100"),
        ]

    def save_markdown(self, path: str | Path) -> Path:
        """Write rendered Markdown Model Card to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")
        return target

    def to_pdf(self, path: str | Path) -> Path:
        """Render the Model Card as a publication-grade PDF (reportlab; ``omniforge[reporting]`` extra).

        Mirrors :meth:`to_markdown` section-for-section: banner, intended use & compliance,
        evaluation metrics, baseline comparison, top drivers, fairness/robustness/carbon
        audits (when present), and known limitations. Returns the written path.
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import LETTER
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        except ImportError as exc:
            raise ReportingError("Install 'reportlab' to export Model Card PDFs (pip install 'omniforge[reporting]').", cause=exc) from exc

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        styles = getSampleStyleSheet()
        usable_w = LETTER[0] - 1.3 * inch

        s_title = ParagraphStyle("CardTitle", parent=styles["Title"], fontSize=16, leading=20, textColor=colors.white, alignment=0)
        s_sub = ParagraphStyle("CardSub", parent=styles["Normal"], fontSize=8.6, leading=12, textColor=colors.HexColor("#BAE6FD"))
        s_h1 = ParagraphStyle("CardH1", parent=styles["Heading1"], fontSize=12.5, leading=15.5, spaceBefore=11, spaceAfter=5, textColor=colors.HexColor("#0F172A"))
        s_body = ParagraphStyle("CardBody", parent=styles["BodyText"], fontSize=9.0, leading=13, textColor=colors.HexColor("#1E293B"))
        s_bullet = ParagraphStyle("CardBullet", parent=s_body, leftIndent=13, firstLineIndent=-9, spaceAfter=2)
        s_cell = ParagraphStyle("CardCell", parent=styles["Normal"], fontSize=7.9, leading=10.5, textColor=colors.HexColor("#1E293B"))
        s_cell_hdr = ParagraphStyle("CardCellHdr", parent=styles["Normal"], fontSize=8.0, leading=10.5, textColor=colors.white, fontName="Helvetica-Bold")

        def esc(value: Any) -> str:
            return _xml_escape(str(value))

        def table(headers: list[str], rows: list[list[Any]], col_ratios: list[float] | None = None) -> Table:
            data = [[Paragraph(esc(h), s_cell_hdr) for h in headers]]
            for row in rows:
                data.append([Paragraph(esc(cell), s_cell) for cell in row])
            ratios = col_ratios or [1.0] * len(headers)
            total = sum(ratios)
            widths = [usable_w * r / total for r in ratios]
            tbl = Table(data, colWidths=widths, repeatRows=1)
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
            return tbl

        m = self.model_details
        subtitle = (
            f"ID: {m.model_id}  |  {m.model_family} / {m.framework}  |  task: {m.task_type.value}  |  "
            f"v{m.version}  |  generated: {self.generated_at}"
        )
        footprint = (
            f"Training footprint: {m.training_rows:,} rows x {m.training_features} features "
            f"({m.training_duration_seconds:.3f}s)"
        )

        def _chrome(canvas: Any, doc_obj: Any) -> None:
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
            canvas.setLineWidth(0.6)
            canvas.line(0.65 * inch, 0.55 * inch, LETTER[0] - 0.65 * inch, 0.55 * inch)
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#64748B"))
            canvas.drawString(0.65 * inch, 0.38 * inch, f"OmniForge Model Card  |  {m.name[:60]}")
            canvas.drawRightString(LETTER[0] - 0.65 * inch, 0.38 * inch, f"Page {doc_obj.page}")
            canvas.restoreState()

        story: list[Any] = []
        banner = Table(
            [
                [Paragraph(esc(f"Model Card: {m.name}"), s_title)],
                [Paragraph(esc(subtitle), s_sub)],
                [Paragraph(esc(footprint), s_sub)],
            ],
            colWidths=[usable_w],
        )
        banner.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, 0), 11),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
                    ("TOPPADDING", (0, 1), (-1, 1), 0),
                    ("BOTTOMPADDING", (0, 2), (-1, 2), 10),
                ]
            )
        )
        story.extend([banner, Spacer(1, 12)])

        story.append(Paragraph("Intended Use &amp; Governance", s_h1))
        story.append(Paragraph(esc(self.intended_use), s_body))
        if self.out_of_scope_uses:
            story.append(Paragraph("<b>Out-of-scope uses:</b>", s_body))
            for item in self.out_of_scope_uses:
                story.append(Paragraph(esc(f"• {item}"), s_bullet))
        story.append(Paragraph(f"<b>Compliance profiles:</b> {esc(', '.join(self.compliance_frameworks))}", s_body))

        scalar_metrics = {k: v for k, v in self.evaluation_metrics.items() if not isinstance(v, (list, dict))}
        if scalar_metrics:
            story.append(Paragraph("Evaluation Performance", s_h1))
            story.append(table(["Metric", "Value"], [[k, v] for k, v in scalar_metrics.items()], col_ratios=[2.2, 1.4]))

        if self.baseline_comparison:
            story.append(Paragraph("Comparison Against Strong Baselines", s_h1))
            story.append(table(["Baseline", "Scores"], [[k, v] for k, v in self.baseline_comparison.items()], col_ratios=[1.6, 3.4]))

        if self.feature_importance:
            story.append(Paragraph("Top Predictive Drivers", s_h1))
            top = list(self.feature_importance.items())[:10]
            story.append(table(["Feature", "Importance"], [[feat, f"{imp:.4f}"] for feat, imp in top], col_ratios=[3.0, 1.2]))

        if self.fairness_assessment is not None:
            fa = self.fairness_assessment
            story.append(Paragraph("Responsible AI — Fairness Audit", s_h1))
            story.append(
                table(
                    ["Check", "Result"],
                    [
                        ["Protected attribute", fa.protected_attribute],
                        ["Gate status", "PASSED" if fa.passed else "REVIEW REQUIRED"],
                        ["Disparate impact ratio", fa.disparate_impact_ratio],
                        ["Demographic parity diff", fa.demographic_parity_diff],
                        ["Equalized odds diff", fa.equalized_odds_diff],
                    ],
                    col_ratios=[2.2, 1.6],
                )
            )

        if self.robustness_assessment:
            story.append(Paragraph("Robustness &amp; Stress Testing", s_h1))
            robustness_rows = [[k, v] for k, v in self.robustness_assessment.items() if not isinstance(v, (list, dict))]
            if robustness_rows:
                story.append(table(["Measurement", "Value"], robustness_rows, col_ratios=[2.4, 1.4]))

        if self.carbon_footprint is not None:
            story.append(Paragraph("Green AI — Carbon Footprint", s_h1))
            story.append(table(["Measurement", "Value"], [list(row) for row in self._carbon_rows()], col_ratios=[2.2, 1.8]))

        if self.limitations_and_risks:
            story.append(Paragraph("Known Limitations &amp; Risks", s_h1))
            for lim in self.limitations_and_risks:
                story.append(Paragraph(esc(f"• {lim}"), s_bullet))

        doc = SimpleDocTemplate(
            str(target),
            pagesize=LETTER,
            leftMargin=0.65 * inch,
            rightMargin=0.65 * inch,
            topMargin=0.65 * inch,
            bottomMargin=0.72 * inch,
            title=f"Model Card: {m.name}",
            author="OmniForge Enterprise",
        )
        doc.build(story, onFirstPage=_chrome, onLaterPages=_chrome)
        return target


class DataCard(BaseModel):
    """Structured Data Card documenting lineage, schema, quality dimensions, and PII compliance."""

    card_version: str = "1.0"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    dataset_metadata: DatasetMetadata
    quality_scores: dict[str, float] = Field(default_factory=dict)
    compliance_summary: dict[str, Any] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render the Data Card as Markdown documentation."""
        m = self.dataset_metadata
        lines = [
            f"# Data Card: {m.name}",
            f"- **Dataset ID**: `{m.dataset_id}`",
            f"- **Source**: `{m.source}` (`{m.source_format}`)",
            f"- **Modality**: `{m.modality.value}`",
            f"- **Dimensions**: {m.n_rows:,} rows × {m.n_columns} columns",
            f"- **SHA-256 Checksum**: `{m.checksum_sha256}`",
            "",
            "## Quality Dimensions",
        ]
        for dim, score in self.quality_scores.items():
            lines.append(f"- **{dim.title()}**: `{score}/100`")
        lines.extend(["", "## Schema Summary"])
        for col in m.schema_columns[:30]:
            lines.append(
                f"- `{col.name}` ({col.dtype} / {col.inferred_semantic_type}): nulls={col.null_count} ({col.null_ratio*100:.1f}%), unique={col.unique_count}"
            )
        return "\n".join(lines) + "\n"


class ModelCardGenerator:
    """Factory assembling enterprise Model Cards and Data Cards from evaluation artifacts."""

    @staticmethod
    def build_model_card(
        metadata: ModelMetadata,
        *,
        evaluation_metrics: dict[str, Any] | None = None,
        baseline_comparison: dict[str, Any] | None = None,
        feature_importance: dict[str, float] | None = None,
        fairness_report: FairnessReport | None = None,
        robustness_report: dict[str, Any] | None = None,
        intended_use: str | None = None,
        compliance_frameworks: list[str] | None = None,
    ) -> ModelCard:
        """Construct a complete :class:`ModelCard`."""
        limitations: list[str] = []
        if fairness_report is not None and not fairness_report.passed:
            limitations.append("Fairness audit flagged subgroup disparity requiring mitigation prior to production.")
        if robustness_report and float(robustness_report.get("robustness_index_0_100", 100.0)) < 80.0:
            limitations.append("Model exhibited elevated sensitivity under perturbation/OOD stress testing.")
        if not limitations:
            limitations.append("Monitor feature and prediction drift continuously via OmniForge MLOps DriftDetector.")

        return ModelCard(
            model_details=metadata,
            intended_use=intended_use or "Enterprise predictive modeling and analytics decision intelligence.",
            evaluation_metrics=evaluation_metrics or metadata.metrics,
            baseline_comparison=baseline_comparison or metadata.baseline_comparison,
            feature_importance=feature_importance or {},
            fairness_assessment=fairness_report,
            robustness_assessment=robustness_report or {},
            carbon_footprint=metadata.carbon_report,
            compliance_frameworks=compliance_frameworks or ["GDPR", "SOX", "EU-AI-Act"],
            limitations_and_risks=limitations,
        )

    @staticmethod
    def build_data_card(
        metadata: DatasetMetadata,
        *,
        quality_scores: dict[str, float] | None = None,
        compliance_summary: dict[str, Any] | None = None,
    ) -> DataCard:
        """Construct a complete :class:`DataCard`."""
        return DataCard(
            dataset_metadata=metadata,
            quality_scores=quality_scores or {},
            compliance_summary=compliance_summary or {},
        )
