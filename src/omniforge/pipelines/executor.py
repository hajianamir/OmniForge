"""Auditable DAG Pipeline Executor with step-level caching, retries, carbon tracking, and lineage."""

from __future__ import annotations

import copy
import hashlib
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from omniforge.core.carbon import CarbonTracker
from omniforge.core.exceptions import PipelineError
from omniforge.core.logging import get_logger
from omniforge.core.security import ArtifactSigner, safe_joblib_load_bytes
from omniforge.core.types import CarbonReport
from omniforge.pipelines.base import PipelineContext, PipelineStep, StepExecutionRecord
from omniforge.pipelines.dag import PipelineDAG

logger = get_logger(__name__)
_CACHE_HMAC_CONTEXT = b"omniforge:pipeline-cache:v3:"


def _state_snapshot(ctx: PipelineContext) -> dict[str, dict[str, Any]]:
    return {"params": ctx.params, "artifacts": ctx.artifacts, "step_outputs": ctx.step_outputs}


def _apply_snapshot(ctx: PipelineContext, snapshot: dict[str, dict[str, Any]]) -> None:
    ctx.params = snapshot["params"]
    ctx.artifacts = snapshot["artifacts"]
    ctx.step_outputs = snapshot["step_outputs"]


def _load_snapshot(path: Path, step_id: str) -> tuple[dict[str, dict[str, Any]], str] | None:
    try:
        signature, _, payload = path.read_bytes().partition(b"\n")
        if not signature or not ArtifactSigner().verify_bytes(
            payload, signature.decode("ascii"), context=_CACHE_HMAC_CONTEXT + path.name.encode()
        ):
            logger.warning("pipeline_step_cache_signature_rejected", step_id=step_id)
            return None
        snapshot = safe_joblib_load_bytes(payload, filename=path.name)
        if (
            not isinstance(snapshot, dict)
            or set(snapshot) != {"params", "artifacts", "step_outputs"}
            or any(not isinstance(value, dict) for value in snapshot.values())
            or step_id not in snapshot["step_outputs"]
        ):
            return None
        return snapshot, hashlib.sha256(payload).hexdigest()
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("pipeline_step_cache_invalid_recomputing", step_id=step_id, error_type=type(exc).__name__)
        return None


def _publish_snapshot(path: Path, ctx: PipelineContext) -> str:
    payload = pickle.dumps(_state_snapshot(ctx), protocol=5)
    signature = ArtifactSigner().sign_bytes(payload, context=_CACHE_HMAC_CONTEXT + path.name.encode())
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as fh:
            temp_path = Path(fh.name)
            fh.write(signature.encode("ascii") + b"\n" + payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, path)
        return hashlib.sha256(payload).hexdigest()
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize pipeline execution metadata, step records, and carbon telemetry to JSON-safe dicts."""
        return {
            "pipeline_name": self.pipeline_name,
            "run_id": self.run_id,
            "status": self.status,
            "total_duration_seconds": round(float(self.total_duration_seconds), 4),
            "step_records": [r.model_dump() for r in self.step_records],
            "artifact_keys": sorted(str(k) for k in self.context.artifacts.keys()),
            "carbon_report": self.carbon_report.model_dump() if self.carbon_report is not None else None,
        }


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
        """Run the DAG using copyable artifacts; cacheable steps must be deterministic over the artifact bus."""
        dag.validate()
        order = dag.topological_order()
        try:
            initial_params, artifacts = copy.deepcopy((dict(params or {}), dict(initial_artifacts or {})))
        except Exception as exc:
            raise PipelineError(
                "Pipeline inputs could not be copied into a private execution context.",
                suggestions=["Supply copyable data and model artifacts instead of live resource handles."],
            ) from exc
        ctx = PipelineContext(tenant_id=tenant_id, params=initial_params, artifacts=artifacts)
        t_total = time.perf_counter()
        overall_status = "success"
        state_digest: str | None = None

        with CarbonTracker(f"pipeline:{dag.name}", enabled=self.track_carbon) as tracker:
            for sid in order:
                step = dag.get_step(sid)
                started_at = datetime.now(UTC).isoformat()
                t0 = time.perf_counter()
                cache_path = (
                    self._cache_key_path(dag.name, step, ctx, state_digest)
                    if step.cacheable and self.cache_dir else None
                )
                state_digest = None
                cached = _load_snapshot(cache_path, sid) if cache_path is not None else None
                if cached is not None:
                    snapshot, state_digest = cached
                    _apply_snapshot(ctx, snapshot)
                    ctx.records.append(
                        StepExecutionRecord(
                            step_id=sid,
                            step_type=step.step_type,
                            status="cached",
                            started_at=started_at,
                            finished_at=datetime.now(UTC).isoformat(),
                            duration_seconds=round(time.perf_counter() - t0, 6),
                            cache_hit=True,
                        )
                    )
                    continue

                last_exc: Exception | None = None
                for attempt in range(1, step.retries + 2):
                    try:
                        # Isolate attempts when execution may continue after a failure.
                        attempt_ctx = copy.deepcopy(ctx) if step.retries or not self.fail_fast else ctx
                        attempt_ctx.step_outputs[sid] = step.run(attempt_ctx)
                    except Exception as exc:
                        last_exc = exc
                        continue
                    _apply_snapshot(ctx, _state_snapshot(attempt_ctx))
                    if cache_path is not None:
                        try:
                            state_digest = _publish_snapshot(cache_path, ctx)
                        except Exception as exc:
                            logger.debug("pipeline_step_cache_write_skipped", step_id=sid, error_type=type(exc).__name__)
                    ctx.records.append(
                        StepExecutionRecord(
                            step_id=sid,
                            step_type=step.step_type,
                            status="success",
                            started_at=started_at,
                            finished_at=datetime.now(UTC).isoformat(),
                            duration_seconds=round(time.perf_counter() - t0, 6),
                            metadata={"attempts": attempt},
                        )
                    )
                    last_exc = None
                    break

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

    def _cache_key_path(
        self, pipeline_name: str, step: PipelineStep, ctx: PipelineContext, state_digest: str | None
    ) -> Path | None:
        assert self.cache_dir is not None
        base_dir = self.cache_dir.resolve()
        # Steps share the artifact bus, including outputs from earlier undeclared readers.
        # Cache identity covers the entire state that a step is allowed to read/write.
        try:
            if state_digest is None:
                state_digest = hashlib.sha256(pickle.dumps(_state_snapshot(ctx), protocol=5)).hexdigest()
            data = pickle.dumps((ctx.tenant_id, pipeline_name, step.to_spec(), state_digest), protocol=5)
        except Exception as exc:
            logger.debug("pipeline_step_uncacheable_input", step_id=step.step_id, error_type=type(exc).__name__)
            return None
        digest = hashlib.sha256(data).hexdigest()[:24]
        safe_pipe = re.sub(r"[^A-Za-z0-9._-]", "_", pipeline_name).strip("._-")[:64] or "pipeline"
        safe_step = re.sub(r"[^A-Za-z0-9._-]", "_", step.step_id).strip("._-")[:64] or "step"
        filename = f"{safe_pipe}_{safe_step}_{digest}.cache-v3"
        if ".." in filename or "/" in filename or "\\" in filename:
            raise PipelineError("Pipeline cache filename could not be safely resolved.")
        candidate = (base_dir / filename).resolve()
        if not candidate.is_relative_to(base_dir):
            raise PipelineError("Pipeline cache entry is outside the configured cache directory.")
        return candidate
