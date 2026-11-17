"""Core type definitions, enumerations, and structured metadata models for OmniForge."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    """Canonical analytics and machine learning task taxonomy."""

    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    RECOMMENDATION = "recommendation"
    SURVIVAL = "survival"
    CAUSAL_INFERENCE = "causal_inference"
    OPTIMIZATION = "optimization"
    SIMULATION = "simulation"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    GRAPH_ANALYTICS = "graph_analytics"
    MULTIMODAL = "multimodal"


class DataModality(StrEnum):
    """Supported data modalities across the platform."""

    TABULAR = "tabular"
    TIME_SERIES = "time_series"
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    GRAPH = "graph"
    GEOSPATIAL = "geospatial"
    SENSOR_IOT = "sensor_iot"
    GENOMIC = "genomic"
    MULTIMODAL = "multimodal"


class ModelStage(StrEnum):
    """Lifecycle stages for registered models in the MLOps Model Registry."""

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ValidationSeverity(StrEnum):
    """Severity levels for data validation expectations and compliance checks."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DriftStatus(StrEnum):
    """Drift assessment classification."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


class CarbonReport(BaseModel):
    """Structured carbon and energy footprint telemetry for a compute operation."""

    operation_name: str
    tracker_backend: str = "uap_hardware_estimator"
    duration_seconds: float = 0.0
    energy_kwh: float = 0.0
    emissions_kg_co2eq: float = 0.0
    emissions_g_co2eq: float = 0.0
    country_iso: str = "DEU"
    grid_intensity_g_per_kwh: float = 380.0
    pue: float = 1.25
    avg_cpu_utilization_pct: float = 0.0
    cpu_power_watts: float = 0.0
    ram_power_watts: float = 0.0
    gpu_power_watts: float = 0.0
    green_score: float = Field(default=100.0, ge=0.0, le=100.0)
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class ColumnSchema(BaseModel):
    """Schema metadata for a single dataset column."""

    name: str
    dtype: str
    inferred_semantic_type: str
    nullable: bool = True
    null_count: int = 0
    null_ratio: float = 0.0
    unique_count: int = 0
    sample_values: list[Any] = Field(default_factory=list)


class DatasetMetadata(BaseModel):
    """Rich lineage, schema, and integrity metadata for any ingested dataset."""

    dataset_id: str = Field(default_factory=lambda: f"ds_{uuid4().hex[:12]}")
    name: str = "unnamed_dataset"
    source: str = "in_memory"
    source_format: str = "dataframe"
    modality: DataModality = DataModality.TABULAR
    n_rows: int = 0
    n_columns: int = 0
    memory_usage_bytes: int = 0
    checksum_sha256: str = ""
    schema_columns: list[ColumnSchema] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    tenant_id: str = "default-tenant"
    version: str = "1.0.0"
    parent_dataset_id: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    lineage: list[dict[str, Any]] = Field(default_factory=list)


class ModelMetadata(BaseModel):
    """Structured metadata attached to every trained OmniForge model."""

    model_id: str = Field(default_factory=lambda: f"mdl_{uuid4().hex[:12]}")
    name: str
    model_family: str
    task_type: TaskType
    version: str = "1.0.0"
    framework: str = "uap"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    feature_names: list[str] = Field(default_factory=list)
    target_name: str | None = None
    classes: list[Any] = Field(default_factory=list)
    training_rows: int = 0
    training_features: int = 0
    training_duration_seconds: float = 0.0
    random_seed: int = 42
    dataset_checksum: str = ""
    artifact_checksum: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    baseline_comparison: dict[str, Any] = Field(default_factory=dict)
    carbon_report: CarbonReport | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    tags: dict[str, str] = Field(default_factory=dict)
    tenant_id: str = "default-tenant"


class AuditEvent(BaseModel):
    """Tamper-evident enterprise audit log entry."""

    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex[:12]}")
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    tenant_id: str = "default-tenant"
    actor: str = "system"
    role: str = "admin"
    action: str
    resource_type: str
    resource_id: str
    status: str = "success"
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    details: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str = ""
    event_hash: str = ""


class Principal(BaseModel):
    """Authenticated user or service principal for RBAC/ABAC enforcement."""

    subject: str
    roles: list[str] = Field(default_factory=lambda: ["analyst"])
    tenant_id: str = "default-tenant"
    attributes: dict[str, Any] = Field(default_factory=dict)
    scopes: list[str] = Field(default_factory=list)
