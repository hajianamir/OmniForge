"""Education & EdTech Analytics Domain Plugin: Student Retention, Learning Outcomes, Intervention Uplift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from omniforge.domains.plugin_sdk import BaseDomainPlugin, DomainMetadata


class EducationDomainPlugin(BaseDomainPlugin):
    """Enterprise Education & Learning Analytics plugin (FERPA / GDPR)."""

    metadata = DomainMetadata(
        name="education",
        display_name="Education & Learning Analytics",
        description="Student early-warning dropout risk, adaptive mastery forecasting, and tutoring causal impact.",
        compliance_frameworks=["FERPA", "GDPR", "EU-AI-Act"],
        kpis=["retention_rate", "mean_attendance_pct", "mean_assessment_score"],
        recommended_models=["logistic_regression", "gradient_boosting", "causal_inference"],
        default_target="at_risk",
        config_file="configs/domains/education.yaml",
    )

    def engineer_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"assignments_submitted", "assignments_total"}.issubset(out.columns):
            out["submission_rate"] = out["assignments_submitted"] / np.maximum(out["assignments_total"], 1.0)
        return out

    def compute_domain_kpis(self, df: pd.DataFrame, *, predictions: np.ndarray | None = None) -> dict[str, float]:
        kpis: dict[str, float] = {}
        if "at_risk" in df.columns:
            kpis["retention_rate"] = round(float(1.0 - df["at_risk"].mean()), 4)
        if "attendance_pct" in df.columns:
            kpis["mean_attendance_pct"] = round(float(df["attendance_pct"].mean()), 2)
        if "assessment_score" in df.columns:
            kpis["mean_assessment_score"] = round(float(df["assessment_score"].mean()), 2)
        return kpis

    def generate_sample_dataset(self, n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(random_state)
        attend = np.clip(rng.normal(82, 14, n_samples), 30, 100)
        lms_hrs = np.clip(rng.gamma(3.0, 2.5, n_samples), 0.5, 30)
        sub_tot = np.full(n_samples, 12)
        sub_done = np.clip(np.round(sub_tot * (attend / 100.0) + rng.normal(0, 1.2, n_samples)), 0, 12).astype(int)
        score = np.clip(0.45 * attend + 1.8 * lms_hrs + 2.0 * sub_done + rng.normal(5, 8, n_samples), 20, 100)
        at_risk = (score < 62.0).astype(int)
        return pd.DataFrame(
            {
                "attendance_pct": np.round(attend, 1),
                "lms_hours_weekly": np.round(lms_hrs, 2),
                "assignments_submitted": sub_done,
                "assignments_total": sub_tot,
                "assessment_score": np.round(score, 1),
                "at_risk": at_risk,
            }
        )


plugin = EducationDomainPlugin()
