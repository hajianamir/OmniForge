"""Pydantic v2 request and response schemas for the OmniForge FastAPI REST backend."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TokenRequest(BaseModel):
    username: str = "analyst"
    role: str = "analyst"
    tenant_id: str = "default"
    api_key: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    tenant_id: str


class DatasetPayload(BaseModel):
    records: list[dict[str, Any]]
    dataset_name: str = "api_dataset"


class AutoMLRequest(BaseModel):
    records: list[dict[str, Any]]
    target_column: str
    time_budget_seconds: float = Field(default=30.0, ge=5.0, le=600.0)
    register_as: str | None = None
    deploy_endpoint: str | None = None


class ForecastRequest(BaseModel):
    records: list[dict[str, Any]]
    target_column: str = "y"
    time_column: str | None = "timestamp"
    horizon: int = Field(default=14, ge=1, le=365)


class CausalRequest(BaseModel):
    records: list[dict[str, Any]]
    treatment_col: str = "treatment"
    outcome_col: str = "outcome"
    confounders: list[str]
    method: str = "aipw"


class PortfolioOptimizationRequest(BaseModel):
    expected_returns: list[float]
    covariance_matrix: list[list[float]]
    risk_aversion: float = 2.5
    max_weight: float = 1.0


class PredictRequest(BaseModel):
    records: list[dict[str, Any]]
    return_probabilities: bool = False


class DriftRequest(BaseModel):
    reference_records: list[dict[str, Any]]
    current_records: list[dict[str, Any]]


class CopilotChatRequest(BaseModel):
    message: str
    records: list[dict[str, Any]] | None = None
    provider: str = "local"


class PipelineExecuteRequest(BaseModel):
    pipeline_spec: dict[str, Any]
    records: list[dict[str, Any]] | None = None


class StageTransitionRequest(BaseModel):
    target_stage: str
