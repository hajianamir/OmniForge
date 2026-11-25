"""Enterprise exception hierarchy for the OmniForge.

Every exception raised across OmniForge inherits from :class:`OmniForgeError` and carries
structured diagnostic metadata including an error code, severity level,
remediation suggestions, trace identifier, timestamp, and contextual payload.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from omniforge.core.types import SerializableStrEnum


class ErrorSeverity(SerializableStrEnum):
    """Severity classification for structured platform errors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OmniForgeError(Exception):
    """Base exception for all OmniForge errors.

    Attributes:
        message: Human-readable description of the error.
        error_code: Stable machine-readable error identifier (e.g. ``OmniForge-DATA-001``).
        severity: Operational severity level.
        suggestions: Actionable remediation steps for users or operators.
        trace_id: Distributed correlation trace identifier.
        context: Additional structured metadata aiding root-cause analysis.
        timestamp: UTC ISO-8601 timestamp when the exception was instantiated.
        cause: Optional underlying exception that triggered this error.
    """

    default_error_code: str = "OmniForge-CORE-000"
    default_severity: ErrorSeverity = ErrorSeverity.HIGH

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        severity: ErrorSeverity | None = None,
        suggestions: list[str] | None = None,
        trace_id: str | None = None,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.default_error_code
        self.severity = severity or self.default_severity
        self.suggestions = list(suggestions) if suggestions else []
        self.trace_id = trace_id or str(uuid4())
        self.context = dict(context) if context else {}
        self.cause = cause
        self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self, *, include_internal: bool = True) -> dict[str, Any]:
        """Serialize the exception into an API- and log-safe dictionary.

        Args:
            include_internal: When ``False`` (used for external HTTP responses), omits
                ``context`` and ``cause`` so internal paths/driver messages never leak.

        Returns:
            Dictionary with structured error telemetry.
        """
        payload: dict[str, Any] = {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "severity": self.severity.value,
            "message": self.message,
            "suggestions": self.suggestions,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
        }
        if include_internal:
            payload["context"] = self.context
            if self.cause is not None:
                payload["cause"] = f"{type(self.cause).__name__}: {self.cause}"
        return payload

    def __str__(self) -> str:
        suggestion_str = (
            f" | Suggestions: {'; '.join(self.suggestions)}" if self.suggestions else ""
        )
        return (
            f"[{self.error_code}] ({self.severity.value.upper()}) {self.message} "
            f"[trace_id={self.trace_id}]{suggestion_str}"
        )


class ConfigurationError(OmniForgeError):
    """Raised when configuration loading, merging, or validation fails."""

    default_error_code = "OmniForge-CFG-001"
    default_severity = ErrorSeverity.HIGH


class SecurityError(OmniForgeError):
    """Raised on authentication, encryption, or compliance violations."""

    default_error_code = "OmniForge-SEC-001"
    default_severity = ErrorSeverity.CRITICAL


class AuthorizationError(SecurityError):
    """Raised when a principal lacks required RBAC/ABAC permissions."""

    default_error_code = "OmniForge-SEC-002"
    default_severity = ErrorSeverity.HIGH


class ComplianceError(SecurityError):
    """Raised when an operation violates active regulatory compliance rules."""

    default_error_code = "OmniForge-SEC-003"
    default_severity = ErrorSeverity.CRITICAL


class DataError(OmniForgeError):
    """Base exception for data layer failures."""

    default_error_code = "OmniForge-DATA-000"
    default_severity = ErrorSeverity.HIGH


class DataIngestionError(DataError):
    """Raised when data cannot be read, parsed, or fetched from a source."""

    default_error_code = "OmniForge-DATA-001"


class DataCleaningError(DataError):
    """Raised when cleaning or transformation steps fail."""

    default_error_code = "OmniForge-DATA-002"


class DataValidationError(DataError):
    """Raised when schema or expectation validation fails critically."""

    default_error_code = "OmniForge-DATA-003"
    default_severity = ErrorSeverity.MEDIUM


class FeatureEngineeringError(OmniForgeError):
    """Raised when feature generation, selection, or store lookup fails."""

    default_error_code = "OmniForge-FEAT-001"
    default_severity = ErrorSeverity.MEDIUM


class ModelError(OmniForgeError):
    """Base exception for modeling, training, and inference failures."""

    default_error_code = "OmniForge-MOD-000"
    default_severity = ErrorSeverity.HIGH


class ModelNotFittedError(ModelError):
    """Raised when prediction or inspection is attempted before fitting."""

    default_error_code = "OmniForge-MOD-001"
    default_severity = ErrorSeverity.MEDIUM


class ModelTrainingError(ModelError):
    """Raised when model fitting fails to converge or encounters invalid inputs."""

    default_error_code = "OmniForge-MOD-002"


class ModelSerializationError(ModelError):
    """Raised when saving or loading a model artifact fails."""

    default_error_code = "OmniForge-MOD-003"


class EvaluationError(OmniForgeError):
    """Raised during metric computation, explainability, or fairness checks."""

    default_error_code = "OmniForge-EVAL-001"
    default_severity = ErrorSeverity.MEDIUM


class AutoMLError(OmniForgeError):
    """Raised when hyperparameter search, pipeline search, or NAS fails."""

    default_error_code = "OmniForge-AML-001"


class ForecastingError(OmniForgeError):
    """Raised during time-series forecasting, reconciliation, or backtesting."""

    default_error_code = "OmniForge-FCST-001"


class CausalInferenceError(OmniForgeError):
    """Raised during causal identification, estimation, or refutation."""

    default_error_code = "OmniForge-CAUS-001"


class OptimizationError(OmniForgeError):
    """Raised when an optimization problem is infeasible, unbounded, or invalid."""

    default_error_code = "OmniForge-OPT-001"


class SimulationError(OmniForgeError):
    """Raised when Monte Carlo, DES, ABM, or Digital Twin execution fails."""

    default_error_code = "OmniForge-SIM-001"


class PipelineError(OmniForgeError):
    """Raised on DAG cycles, invalid step wiring, or pipeline execution failure."""

    default_error_code = "OmniForge-PIPE-001"


class MLOpsError(OmniForgeError):
    """Raised during experiment tracking, registry transitions, serving, or drift checks."""

    default_error_code = "OmniForge-MLOPS-001"


class CopilotError(OmniForgeError):
    """Raised when AI Copilot planning, tool dispatch, or LLM communication fails."""

    default_error_code = "OmniForge-COP-001"
    default_severity = ErrorSeverity.MEDIUM


class DomainPluginError(OmniForgeError):
    """Raised when loading, registering, or executing a domain plugin fails."""

    default_error_code = "OmniForge-DOM-001"
    default_severity = ErrorSeverity.MEDIUM


class ReportingError(OmniForgeError):
    """Raised when generating or exporting reports in any target format fails."""

    default_error_code = "OmniForge-REP-001"
    default_severity = ErrorSeverity.MEDIUM


class DependencyMissingError(OmniForgeError):
    """Raised when an explicitly requested optional backend is not installed."""

    default_error_code = "OmniForge-DEP-001"
    default_severity = ErrorSeverity.MEDIUM
