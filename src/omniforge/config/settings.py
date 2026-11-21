"""Pydantic Settings configuration architecture for the OmniForge."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsSettings(BaseModel):
    """Filesystem locations for workspace, artifacts, databases, and logs."""

    workspace_dir: Path = Path("./.omniforge_workspace")
    artifact_store_dir: Path = Path("./.omniforge_workspace/artifacts")
    cache_dir: Path = Path("./.omniforge_workspace/cache")
    feature_store_dir: Path = Path("./.omniforge_workspace/feature_store")
    tracking_db_path: Path = Path("./.omniforge_workspace/mlops_tracking.db")
    registry_db_path: Path = Path("./.omniforge_workspace/model_registry.db")
    audit_log_path: Path = Path("./.omniforge_workspace/audit.jsonl")


class LoggingSettings(BaseModel):
    """Structured logging configuration."""

    level: str = "INFO"
    format: str = "json"
    redact_secrets: bool = True
    include_trace_id: bool = True
    log_file: Path | None = None


class ComputeSettings(BaseModel):
    """Parallelism, memory, and distributed backend settings."""

    n_jobs: int = -1
    prefer_gpu: bool = True
    backend: str = "local"
    chunk_size: int = 100_000
    memory_limit_mb: int = 8192


class CarbonSettings(BaseModel):
    """Carbon footprint tracking settings."""

    enabled: bool = True
    country_iso: str = "DEU"
    region_intensity_g_per_kwh: float = 380.0
    pue: float = 1.25
    cpu_tdp_watts: float = 65.0
    gpu_tdp_watts: float = 250.0


class DataSettings(BaseModel):
    """Default data layer configuration."""

    default_encoding: str = "utf-8"
    optimize_dtypes: bool = True
    compute_checksums: bool = True
    missing_threshold_drop_col: float = 0.95
    outlier_method: str = "iqr"
    outlier_threshold: float = 1.5
    quality_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "completeness": 0.90,
            "uniqueness": 0.95,
            "validity": 0.95,
            "consistency": 0.95,
            "accuracy": 0.90,
            "timeliness": 0.85,
        }
    )


class ModelingSettings(BaseModel):
    """Default modeling and evaluation configuration."""

    always_compare_baselines: bool = True
    default_cv_folds: int = 5
    test_size: float = 0.2
    enable_conformal_uncertainty: bool = True
    enable_fairness_audit: bool = True
    enable_interpretability: bool = True
    default_classification_metric: str = "f1_weighted"
    default_regression_metric: str = "rmse"
    default_forecasting_metric: str = "smape"


class AutoMLSettings(BaseModel):
    """AutoML search budgets and strategy."""

    time_budget_seconds: int = 120
    max_trials: int = 30
    search_strategy: str = "bayesian"
    enable_ensembling: bool = True
    early_stopping_rounds: int = 10


class MLOpsSettings(BaseModel):
    """Experiment tracking, drift monitoring, and continuous training settings."""

    auto_track_experiments: bool = True
    drift_psi_warning: float = 0.1
    drift_psi_critical: float = 0.25
    drift_ks_alpha: float = 0.05
    retrain_on_critical_drift: bool = True


class SecuritySettings(BaseModel):
    """Security, encryption, RBAC, and compliance settings."""

    enable_rbac: bool = True
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    encryption_key: str = ""
    artifact_signing_key: str = ""
    api_key: str = ""
    auto_bootstrap_local_ui: bool = True
    encrypt_artifacts_at_rest: bool = False
    pii_detection_enabled: bool = True
    compliance_profiles: list[str] = Field(default_factory=lambda: ["GDPR", "SOX"])


class CopilotSettings(BaseModel):
    """AI Copilot LLM provider and agent settings."""

    provider: str = "local"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 2048
    max_tool_steps: int = 8
    memory_window: int = 20
    ollama_base_url: str = "http://localhost:11434"
    azure_endpoint: str | None = None


class APISettings(BaseModel):
    """FastAPI server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    cors_allow_credentials: bool = True
    rate_limit_per_minute: int = 300
    # Whole-request body cap (bytes). Enforced by RequestBodySizeLimitMiddleware before any
    # byte is read (Content-Length) and while streaming (chunked uploads). 256 MB default is
    # generous for local data-science workloads while still bounding memory exhaustion.
    max_request_body_bytes: int = 268_435_456

    @model_validator(mode="after")
    def _prevent_wildcard_credentials(self) -> APISettings:
        """Forbid wildcard CORS combined with ``allow_credentials=True`` (CWE-942)."""
        cleaned = [o.strip() for o in self.cors_origins if o and o.strip()]
        self.cors_origins = cleaned
        if "*" in cleaned and self.cors_allow_credentials:
            self.cors_allow_credentials = False
        return self


class UISettings(BaseModel):
    """OmniForge FastAPI + React enterprise workbench UI settings."""

    host: str = "0.0.0.0"
    port: int = 8501
    default_theme: str = "dark"
    locale: str = "en"


class OmniForgeSettings(BaseSettings):
    """Master typed settings object populated from defaults, YAML, and ``OMNIFORGE_*`` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="OMNIFORGE_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="allow",
    )

    environment: str = "development"
    debug: bool = False
    random_seed: int = 42
    default_tenant_id: str = "default-tenant"

    paths: PathsSettings = Field(default_factory=PathsSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    compute: ComputeSettings = Field(default_factory=ComputeSettings)
    carbon: CarbonSettings = Field(default_factory=CarbonSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    modeling: ModelingSettings = Field(default_factory=ModelingSettings)
    automl: AutoMLSettings = Field(default_factory=AutoMLSettings)
    mlops: MLOpsSettings = Field(default_factory=MLOpsSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    copilot: CopilotSettings = Field(default_factory=CopilotSettings)
    api: APISettings = Field(default_factory=APISettings)
    ui: UISettings = Field(default_factory=UISettings)
    domain_overrides: dict[str, Any] = Field(default_factory=dict)

    def ensure_workspace_dirs(self) -> None:
        """Create all configured workspace directories on disk."""
        for dir_path in (
            self.paths.workspace_dir,
            self.paths.artifact_store_dir,
            self.paths.cache_dir,
            self.paths.feature_store_dir,
            self.paths.tracking_db_path.parent,
            self.paths.registry_db_path.parent,
            self.paths.audit_log_path.parent,
        ):
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> OmniForgeSettings:
    """Return cached singleton :class:`OmniForgeSettings` instance."""
    return OmniForgeSettings()
