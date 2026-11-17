"""Auditable DAG Pipeline Executor with step-level caching, retries, carbon tracking, and lineage."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from omniforge.core.carbon import CarbonTracker
from omniforge.core.exceptions import PipelineError
from omniforge.core.logging import get_logger
from omniforge.core.types import CarbonReport
from omniforge.core.utils import safe_json_default
from omniforge.pipelines.base import PipelineContext, StepExecutionRecord
from omniforge.pipelines.dag import PipelineDAG

logger = get_logger(__name__)


@dataclass
class PipelineExecutionResult:
    """Complete execution result, artifact context, audit records, and carbon footprint."""

    pipeline_name: str
    run_id: str
    status: str  # success, failed
    context: PipelineContext
    step_records: list[StepExecutionRecord] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    carbon_report: CarbonReport | None = None


class PipelineExecutor:
    """Executes :class:`PipelineDAG` instances in topological order with caching, retries, and auditing."""

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        track_carbon: bool = True,
        fail_fast: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.track_carbon = track_carbon
        self.fail_fast = fail_fast

    def execute(
        self,
        dag: PipelineDAG,
        *,
        initial_artifacts: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        tenant_id: str = "default-tenant",
    ) -> PipelineExecutionResult:
        """Validate and execute all steps in ``dag``."""
        dag.validate()
        order = dag.topological_order()
        ctx = PipelineContext(
            tenant_id=tenant_id,
            params=dict(params or {}),
            artifacts=dict(initial_artifacts or {}),
        )
        t_total = time.perf_counter()
        overall_status = "success"

        with CarbonTracker(f"pipeline:{dag.name}", enabled=self.track_carbon) as tracker:
            for sid in order:
                step = dag.get_step(sid)
                started_at = datetime.now(UTC).isoformat()
                t0 = time.perf_counter()
                cache_path = self._cache_key_path(dag.name, step.step_id, step.config) if (step.cacheable and self.cache_dir) else None

                if cache_path is not None and cache_path.exists():
                    cached_out = joblib.load(cache_path)
                    ctx.step_outputs[sid] = cached_out
                    rec = StepExecutionRecord(
                        step_id=sid,
                        step_type=step.step_type,
                        status="cached",
                        started_at=started_at,
                        finished_at=datetime.now(UTC).isoformat(),
                        duration_seconds=round(time.perf_counter() - t0, 6),
                        cache_hit=True,
                    )
                    ctx.records.append(rec)
                    continue

                attempt = 0
                last_exc: Exception | None = None
                while attempt <= step.retries:
                    try:
                        output = step.run(ctx)
                        ctx.step_outputs[sid] = output
                        if cache_path is not None:
                            try:
                                joblib.dump(output, cache_path)
                            except Exception as exc:
                                logger.debug("pipeline_step_cache_write_skipped", step_id=sid, error=str(exc))
                        rec = StepExecutionRecord(
                            step_id=sid,
                            step_type=step.step_type,
                            status="success",
                            started_at=started_at,
                            finished_at=datetime.now(UTC).isoformat(),
                            duration_seconds=round(time.perf_counter() - t0, 6),
                            metadata={"attempts": attempt + 1},
                        )
                        ctx.records.append(rec)
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        attempt += 1

                if last_exc is not None:
                    overall_status = "failed"
                    rec = StepExecutionRecord(
                        step_id=sid,
                        step_type=step.step_type,
                        status="failed",
                        started_at=started_at,
                        finished_at=datetime.now(UTC).isoformat(),
                        duration_seconds=round(time.perf_counter() - t0, 6),
                        error=f"{type(last_exc).__name__}: {last_exc}",
                    )
                    ctx.records.append(rec)
                    logger.error("pipeline_step_failed", pipeline=dag.name, step=sid, error=str(last_exc))
                    if self.fail_fast:
                        raise PipelineError(
                            f"Pipeline '{dag.name}' failed at step '{sid}': {last_exc}",
                            context={"run_id": ctx.run_id, "step_id": sid},
                            cause=last_exc,
                        ) from last_exc

        total_s = round(time.perf_counter() - t_total, 6)
        logger.info("pipeline_executed", pipeline=dag.name, status=overall_status, duration_s=total_s)
        return PipelineExecutionResult(
            pipeline_name=dag.name,
            run_id=ctx.run_id,
            status=overall_status,
            context=ctx,
            step_records=ctx.records,
            total_duration_seconds=total_s,
            carbon_report=tracker.report,
        )

    def _cache_key_path(self, pipeline_name: str, step_id: str, config: dict[str, Any]) -> Path:
        assert self.cache_dir is not None
        digest = hashlib.sha256(
            json.dumps({"p": pipeline_name, "s": step_id, "c": config}, sort_keys=True, default=safe_json_default).encode("utf-8")
        ).hexdigest()[:16]
        return self.cache_dir / f"{pipeline_name}_{step_id}_{digest}.joblib"
