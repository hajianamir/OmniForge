"""PDF export for Model Cards (NIST AI RMF artifact requirement)."""

from __future__ import annotations

from pathlib import Path

import pytest

from omniforge.core.types import CarbonReport, ModelMetadata, TaskType
from omniforge.evaluation.fairness import FairnessReport
from omniforge.evaluation.model_cards import ModelCardGenerator


def _metadata() -> ModelMetadata:
    return ModelMetadata(
        name="demo_credit_risk",
        model_family="hist_gradient_boosting",
        task_type=TaskType.BINARY_CLASSIFICATION,
        version="2.1.0",
        framework="omniforge",
        hyperparameters={"max_iter": 120, "learning_rate": 0.08},
        feature_names=["age", "income", "credit_lines"],
        target_name="default_flag",
        training_rows=12345,
        training_features=3,
        training_duration_seconds=4.2,
        metrics={"f1_macro": 0.83, "roc_auc": 0.91},
        baseline_comparison={"majority_class": {"f1_macro": 0.41}, "logistic_l2": {"f1_macro": 0.74}},
        carbon_report=CarbonReport(
            operation_name="benchmark",
            energy_kwh=0.0012,
            emissions_g_co2eq=0.49,
            country_iso="USA",
            green_score=88.0,
        ),
    )


def _fairness_report() -> FairnessReport:

    fields = FairnessReport.model_fields
    kwargs = {}
    for name, field in fields.items():
        if name == "protected_attribute":
            kwargs[name] = "group_attr"
        elif name == "passed":
            kwargs[name] = False
        elif field.annotation in (float, "float"):
            kwargs[name] = 0.5
    return FairnessReport(**kwargs)


def test_model_card_pdf_renders_and_contains_all_sections(tmp_path: Path) -> None:
    card = ModelCardGenerator.build_model_card(
        _metadata(),
        evaluation_metrics={"f1_macro": 0.83, "roc_auc": 0.91, "nested": {"ignored": "scalars only"}},
        feature_importance={"income": 0.42, "credit_lines": 0.31, "age": 0.10},
        fairness_report=_fairness_report(),
        robustness_report={"robustness_index_0_100": 91.0, "clean_score": 0.83, "ood_shift_score": 0.77},
    )
    out = card.to_pdf(tmp_path / "nested" / "model_card.pdf")
    assert out.is_file()
    payload = out.read_bytes()
    assert payload.startswith(b"%PDF-")
    assert len(payload) > 2000  # real multi-section document, not an empty stub


def test_model_card_pdf_minimal(tmp_path: Path) -> None:
    card = ModelCardGenerator.build_model_card(_metadata())
    out = card.to_pdf(tmp_path / "minimal.pdf")
    assert out.read_bytes().startswith(b"%PDF-")


def test_model_card_pdf_out_of_scope_and_limitations(tmp_path: Path) -> None:
    card = ModelCardGenerator.build_model_card(
        _metadata(),
        intended_use="Internal credit-risk triage support.",
        compliance_frameworks=["GDPR", "EU-AI-Act", "NIST-AI-RMF"],
    )
    out = card.to_pdf(tmp_path / "full.pdf")
    assert out.read_bytes().startswith(b"%PDF-")


def test_markdown_and_pdf_parity_sections(tmp_path: Path) -> None:
    card = ModelCardGenerator.build_model_card(_metadata(), fairness_report=_fairness_report())
    md = card.to_markdown()
    out = card.to_pdf(tmp_path / "parity.pdf")
    assert "# Model Card:" in md
    assert out.read_bytes().startswith(b"%PDF-")


def test_reportlab_missing_raises_reporting_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import builtins

    card = ModelCardGenerator.build_model_card(_metadata())
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("reportlab"):
            raise ImportError("No module named 'reportlab'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    from omniforge.core.exceptions import ReportingError

    with pytest.raises(ReportingError):
        card.to_pdf(tmp_path / "should_fail.pdf")
