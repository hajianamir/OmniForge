"""Core pipeline step abstractions, shared execution context, and step-level audit records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class StepExecutionRecord(BaseModel):
    """Audit telemetry recorded for every executed step in a OmniForge pipeline."""

    step_id: str
    step_type: str
    status: str  # success, failed, cached, skipped
    started_at: str
    finished_at: str
    duration_seconds: float
    cache_hit: bool = False
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class PipelineContext:
    """Mutable artifact bus passed across steps during a pipeline run."""

    run_id: str = field(default_factory=lambda: f"run_{uuid4().hex[:12]}")
    tenant_id: str = "default-tenant"
    params: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    step_outputs: dict[str, Any] = field(default_factory=dict)
    records: list[StepExecutionRecord] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def set(self, key: str, value: Any) -> None:
        """Store an artifact in the pipeline context."""
        self.artifacts[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve an artifact from the pipeline context."""
        return self.artifacts.get(key, default)


class PipelineStep(ABC):
    """Abstract base class for reusable, declarative, auditable OmniForge pipeline steps."""

    step_type: str = "custom"

    def __init__(
        self,
        step_id: str,
        *,
        depends_on: list[str] | None = None,
        config: dict[str, Any] | None = None,
        cacheable: bool = False,
        retries: int = 0,
        implementation_version: str | None = None,
    ) -> None:
        self.step_id = step_id
        self.depends_on = list(depends_on or [])
        self.config = dict(config or {})
        self.cacheable = cacheable
        self.retries = max(0, int(retries))
        self.implementation_version = implementation_version

    @abstractmethod
    def run(self, context: PipelineContext) -> Any:
        """Execute step business logic against ``context`` and return primary output."""

    def to_spec(self) -> dict[str, Any]:
        """Serialize step definition for visual designer and YAML/JSON export."""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "depends_on": list(self.depends_on),
            "config": dict(self.config),
            "cacheable": self.cacheable,
            "retries": self.retries,
            "implementation_version": self.implementation_version,
        }


class FunctionStep(PipelineStep):
    """Adapter wrapping any Python callable ``fn(context, **config)`` as a :class:`PipelineStep`."""

    step_type = "function"

    def __init__(
        self,
        step_id: str,
        fn: Callable[..., Any],
        *,
        depends_on: list[str] | None = None,
        config: dict[str, Any] | None = None,
        cacheable: bool = False,
        retries: int = 0,
        step_type: str = "function",
        implementation_version: str | None = None,
    ) -> None:
        super().__init__(
            step_id=step_id,
            depends_on=depends_on,
            config=config,
            cacheable=cacheable,
            retries=retries,
            implementation_version=implementation_version,
        )
        self.fn = fn
        self.step_type = step_type

    def run(self, context: PipelineContext) -> Any:
        """Invoke wrapped callable with pipeline context and configured parameters."""
        return self.fn(context, **self.config)
