"""Auditable DAG Pipeline Executor with step-level caching, retries, carbon tracking, and lineage."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from omniforge.core.carbon import CarbonTracker
from omniforge.core.exceptions import PipelineError
from omniforge.core.logging import get_logger
from omniforge.core.performance import compute_df_fingerprint
from omniforge.core.security import ArtifactSigner, safe_joblib_load_bytes
from omniforge.core.types import CarbonReport
from omniforge.core.utils import safe_json_default
from omniforge.pipelines.base import PipelineContext, PipelineStep, StepExecutionRecord
from omniforge.pipelines.dag import PipelineDAG

logger = get_logger(__name__)
_CACHE_HMAC_CONTEXT = b"omniforge:pipeline-cache:v2"


def _fingerprint_value(val: Any) -> str:
    """Compute a deterministic content digest for DataFrames, arrays, JSON primitives, or Python objects."""
    if isinstance(val, pd.DataFrame):
        return f"df:{compute_df_fingerprint(val)}"
    if isinstance(val, pd.Series):
        return f"series:{compute_df_fingerprint(val.to_frame())}"
    if isinstance(val, np.ndarray):
        h = hashlib.sha256()
        h.update(f"{val.shape}:{val.dtype}:".encode())
        h.update(np.ascontiguousarray(val).tobytes())
        return f"ndarray:{h.hexdigest()[:24]}"
    try:
        encoded = json.dumps(val, sort_keys=True, default=safe_json_default).encode("utf-8")
        return f"json:{hashlib.sha256(encoded).hexdigest()[:24]}"
    except Exception:
        try:
            raw = pickle.dumps(val, protocol=5)
            return f"pkl:{hashlib.sha256(raw).hexdigest()[:24]}"
        except Exception:
            return f"repr:{hashlib.sha256(repr(val).encode('utf-8')).hexdigest()[:24]}"


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
        """Validate and execute all steps in ``dag``."""
        dag.validate()
        order = dag.topological_order()
        ctx = PipelineContext(
            tenant_id=tenant_id,
            params=dict(params or {}),
            artifacts=dict(initial_artifacts or {}),
        )
        initial_digests = {str(k): _fingerprint_value(v) for k, v in (initial_artifacts or {}).items()}
        step_digests: dict[str, str] = {}
        t_total = time.perf_counter()
        overall_status = "success"

        with CarbonTracker(f"pipeline:{dag.name}", enabled=self.track_carbon) as tracker:
            for sid in order:
                step = dag.get_step(sid)
                started_at = datetime.now(UTC).isoformat()
                t0 = time.perf_counter()
                cache_path = (
                    self._cache_key_path(
                        dag.name,
                        step,
                        tenant_id=tenant_id,
                        params=ctx.params,
                        initial_digests=initial_digests,
                        upstream_digests=step_digests,
                    )
                    if (step.cacheable and self.cache_dir)
                    else None
                )

                if cache_path is not None and cache_path.exists():
                    sig_path = cache_path.with_suffix(cache_path.suffix + ".hmac")
                    cached_bytes = cache_path.read_bytes()
                    expected_sig = sig_path.read_text(encoding="utf-8").strip() if sig_path.is_file() else ""
                    if expected_sig and ArtifactSigner().verify_bytes(
                        cached_bytes, expected_sig, context=_CACHE_HMAC_CONTEXT
                    ):
                        try:
                            cached_payload = safe_joblib_load_bytes(cached_bytes, filename=cache_path.name)
                            if isinstance(cached_payload, dict) and cached_payload.get("__omniforge_cache_v2__") is True:
                                cached_out = cached_payload.get("output")
                                for ak, av in (cached_payload.get("artifact_deltas") or {}).items():
                                    ctx.artifacts[str(ak)] = av
                            else:
                                cached_out = cached_payload
                            ctx.step_outputs[sid] = cached_out
                            step_digests[sid] = _fingerprint_value(cached_out)
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
                        except Exception as exc:
                            logger.warning("pipeline_step_cache_invalid_recomputing", step_id=sid, error=str(exc))
                    else:
                        logger.warning("pipeline_step_cache_signature_rejected", step_id=sid)

                attempt = 0
                last_exc: Exception | None = None
                while attempt <= step.retries:
                    try:
                        pre_keys = {k: id(v) for k, v in ctx.artifacts.items()}
                        output = step.run(ctx)
                        ctx.step_outputs[sid] = output
                        step_digests[sid] = _fingerprint_value(output)
                        if cache_path is not None:
                            try:
                                artifact_deltas = {
                                    k: v for k, v in ctx.artifacts.items() if k not in pre_keys or id(v) != pre_keys[k]
                                }
                                envelope = {
                                    "__omniforge_cache_v2__": True,
                                    "output": output,
                                    "artifact_deltas": artifact_deltas,
                                }
                                joblib.dump(envelope, cache_path)
                                sig = ArtifactSigner().sign_bytes(
                                    cache_path.read_bytes(), context=_CACHE_HMAC_CONTEXT
                                )
                                cache_path.with_suffix(cache_path.suffix + ".hmac").write_text(
                                    sig, encoding="utf-8"
                                )
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

    def _cache_key_path(
        self,
        pipeline_name: str,
        step: PipelineStep,
        *,
        tenant_id: str = "default-tenant",
        params: dict[str, Any] | None = None,
        initial_digests: dict[str, str] | None = None,
        upstream_digests: dict[str, str] | None = None,
    ) -> Path:
        assert self.cache_dir is not None
        base_dir = self.cache_dir.resolve()
        up_map = {dep: (upstream_digests or {}).get(dep, "") for dep in sorted(step.depends_on)}
        digest = hashlib.sha256(
            json.dumps(
                {
                    "tenant_id": tenant_id,
                    "pipeline": pipeline_name,
                    "step": step.step_id,
                    "step_type": step.step_type,
                    "config": step.config,
                    "params": params or {},
                    "initial_artifacts": initial_digests or {},
                    "upstream_outputs": up_map,
                },
                sort_keys=True,
                default=safe_json_default,
            ).encode("utf-8")
        ).hexdigest()[:24]
        safe_pipe = re.sub(r"[^A-Za-z0-9._-]", "_", pipeline_name).strip("._-")[:64] or "pipeline"
        safe_step = re.sub(r"[^A-Za-z0-9._-]", "_", step.step_id).strip("._-")[:64] or "step"
        filename = f"{safe_pipe}_{safe_step}_{digest}.joblib"
        if ".." in filename or "/" in filename or "\\" in filename:
            raise PipelineError(f"Unsafe cache filename rejected: {filename!r}")
        candidate = (base_dir / filename).resolve()
        if not candidate.is_relative_to(base_dir):
            raise PipelineError(f"Cache path escaped cache_dir confinement: {candidate}")
        return candidate
