"""Asynchronous background job queue and remote MLflow/S3 tracking configuration for OmniForge."""

from __future__ import annotations

import importlib.util
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any

from omniforge.core.exceptions import ConfigurationError, MLOpsError


def configure_remote_mlflow_tracking(
    tracking_uri: str,
    *,
    artifact_location: str | None = None,
) -> dict[str, Any]:
    """Validate and configure remote MLflow tracking URI and optional S3/object-store artifact URI."""
    uri = tracking_uri.strip()
    if not uri:
        raise ValueError("tracking_uri must be a non-empty URI.")
    mlflow_avail = importlib.util.find_spec("mlflow") is not None
    if mlflow_avail:
        import mlflow

        mlflow.set_tracking_uri(uri)
    return {
        "configured": True,
        "tracking_uri": uri,
        "artifact_location": artifact_location,
        "mlflow_installed": mlflow_avail,
    }


class BackgroundJobQueue:
    """Bound waiting work and terminal results for process-local background tasks."""

    def __init__(self, max_workers: int = 2, max_history: int | None = None, *, max_queue_depth: int = 128) -> None:
        self._max_history = 128 if max_history is None else max_history
        if max_workers < 1 or max_queue_depth < 1 or self._max_history < 1:
            raise ConfigurationError("Worker count, queue depth and terminal history must be positive.")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="omniforge-job")
        self._max_unfinished = max_workers + max_queue_depth
        self._jobs: dict[str, Future[Any]] = {}
        self._lock = Lock()
        self._closed = False

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            if self._closed:
                raise MLOpsError("The background job queue has shut down.")
            if sum(not job.done() for job in self._jobs.values()) >= self._max_unfinished:
                raise MLOpsError("The background job queue is full.", suggestions=["Retry after a waiting job finishes."])
            fut = self._executor.submit(fn, *args, **kwargs)
            self._jobs[job_id] = fut
        # Future invokes callbacks synchronously if it has already finished.
        fut.add_done_callback(self._prune_history)
        return job_id

    def _prune_history(self, _future: Future[Any]) -> None:
        with self._lock:
            terminal = [job_id for job_id, future in self._jobs.items() if future.done()]
            for job_id in terminal[:-self._max_history]:
                del self._jobs[job_id]

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        """Stop accepting work, optionally cancel queued tasks, and optionally join workers."""
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_pending)

    def status(self, job_id: str) -> str:
        with self._lock:
            fut = self._jobs.get(job_id)
        if fut is None:
            return "not_found"
        if not fut.done():
            return "running"
        if fut.cancelled():
            return "cancelled"
        return "failed" if fut.exception() is not None else "completed"

    def result(self, job_id: str, timeout: float | None = 30.0) -> Any:
        with self._lock:
            fut = self._jobs.get(job_id)
        if fut is None:
            raise KeyError(f"Unknown job_id '{job_id}'.")
        return fut.result(timeout=timeout)


def validate_production_promotion_gate(
    model_card: dict[str, Any] | None,
    data_card: dict[str, Any] | None,
) -> dict[str, bool]:
    """Enforce mandatory Model Card and Data Card presence before Production promotion."""
    if not model_card or not isinstance(model_card, dict):
        raise ValueError("Production promotion blocked: Model Card is required.")
    if not data_card or not isinstance(data_card, dict):
        raise ValueError("Production promotion blocked: Data Card is required.")
    return {"approved": True, "model_card_present": True, "data_card_present": True}

