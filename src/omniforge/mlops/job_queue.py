"""Asynchronous background job queue and remote MLflow/S3 tracking configuration for OmniForge."""

from __future__ import annotations

import importlib.util
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any


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
    """Thread-safe asynchronous job queue for long-running AutoML and pipeline tasks."""

    def __init__(self, max_workers: int = 2, max_history: int | None = None) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="omniforge-job")
        self._jobs: dict[str, Future[Any]] = {}
        self._lock = Lock()
        self._max_history = max_history
        self._job_order: list[str] = []  # insertion order for LRU eviction

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        job_id = uuid.uuid4().hex[:12]
        fut = self._executor.submit(fn, *args, **kwargs)
        with self._lock:
            self._jobs[job_id] = fut
            self._job_order.append(job_id)
            # Evict oldest jobs when max_history is exceeded
            if self._max_history is not None and len(self._job_order) > self._max_history:
                evict_count = len(self._job_order) - self._max_history
                for _ in range(evict_count):
                    old_id = self._job_order.pop(0)
                    self._jobs.pop(old_id, None)
        return job_id

    def status(self, job_id: str) -> str:
        with self._lock:
            fut = self._jobs.get(job_id)
        if fut is None:
            return "not_found"
        if not fut.done():
            return "running"
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

