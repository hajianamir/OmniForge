"""Enterprise Model Cards & Data Cards generator aligned with EU AI Act and NIST AI RMF."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

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
            cf = self.carbon_footprint
            md.extend(
                [
                    "",
                    "## Green AI — Carbon Footprint",
                    f"- **Energy Consumed**: `{cf.energy_kwh:.8f} kWh`",
                    f"- **Emissions**: `{cf.emissions_g_co2eq:.6f} gCO2eq` ({cf.country_iso} grid)",
                    f"- **Green Score**: `{cf.green_score}/100`",
                ]
            )

        if self.limitations_and_risks:
            md.extend(["", "## Known Limitations & Risks"])
            for lim in self.limitations_and_risks:
                md.append(f"- {lim}")

        return "\n".join(md) + "\n"

    def save_markdown(self, path: str | Path) -> Path:
        """Write rendered Markdown Model Card to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_markdown(), encoding="utf-8")
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
