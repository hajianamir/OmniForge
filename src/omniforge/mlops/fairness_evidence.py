"""Bind a holdout fairness assessment to the fitted model that produced it."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from typing import Any

import joblib
import pandas as pd
from pydantic import BaseModel, Field

from omniforge.core.exceptions import MLOpsError
from omniforge.core.performance import compute_df_fingerprint
from omniforge.evaluation.fairness import FairnessAssessmentStatus, FairnessReport
from omniforge.models.base import BaseOmniForgeModel


def model_fingerprint(model: Any) -> str:
    """Hash the fitted artifact state, excluding the checksum written by save()."""
    if isinstance(model, BaseOmniForgeModel):
        model = copy.copy(model)
        model.metadata_ = model.metadata_.model_copy(update={"artifact_checksum": ""})
    buf = io.BytesIO()
    try:
        joblib.dump(model, buf)
    except Exception as exc:
        raise MLOpsError("The evaluated model cannot be fingerprinted for registration.") from exc
    return hashlib.sha256(buf.getbuffer()).hexdigest()


def holdout_fingerprint(
    X: pd.DataFrame, y: pd.Series, sensitive: pd.Series | None = None, *, clusters: pd.Series | None = None,
) -> str:
    """Fingerprint ordered model inputs, labels, protected values, and cluster assignments."""
    if X.empty or not X.index.is_unique or not X.index.equals(y.index) or y.isna().any():
        raise MLOpsError("Fairness evidence requires aligned, uniquely indexed held-out rows with observed labels.")
    sections = [compute_df_fingerprint(X), compute_df_fingerprint(y.to_frame())]
    if sensitive is not None:
        if not X.index.equals(sensitive.index):
            raise MLOpsError("Protected values must align with the held-out rows.")
        sections.append(compute_df_fingerprint(sensitive.to_frame()))
    if clusters is not None:
        if sensitive is None or not X.index.equals(clusters.index):
            raise MLOpsError("Cluster assignments must align with the protected holdout rows.")
        sections.append(compute_df_fingerprint(clusters.to_frame()))
    return hashlib.sha256(json.dumps(sections).encode("utf-8")).hexdigest()


class FairnessEvidence(BaseModel):
    """Fairness evaluation recorded before a version number is assigned."""

    model_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    holdout_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    n_holdout_rows: int = Field(gt=0)
    report: FairnessReport

    @classmethod
    def capture(
        cls, model: Any, X: pd.DataFrame, y: pd.Series, sensitive: pd.Series, report: FairnessReport,
        *, clusters: pd.Series | None = None,
    ) -> FairnessEvidence:
        if (report.n_clusters is not None) != (clusters is not None) or (
            clusters is not None and report.cluster_attribute is not None and report.cluster_attribute != str(clusters.name)
        ):
            raise MLOpsError("Fairness assessment and its bound cluster assignments must agree.")
        return cls(
            model_fingerprint=model_fingerprint(model),
            holdout_fingerprint=holdout_fingerprint(X, y, sensitive, clusters=clusters),
            n_holdout_rows=len(X),
            report=report,
        )


class RegisteredFairnessEvidence(BaseModel):
    """Assessment attached to one immutable registry artifact."""

    model_name: str
    version: int = Field(gt=0)
    holdout_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    n_holdout_rows: int = Field(gt=0)
    protected_attribute: str
    passed: bool
    artifact_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    assessment_status: FairnessAssessmentStatus | None = None
    cluster_attribute: str | None = None
    n_clusters: int | None = None
    selection_interval_method: str | None = None
