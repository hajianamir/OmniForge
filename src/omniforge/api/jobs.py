"""Thread-Pool Background Job Manager for Non-Blocking Heavy Analytics & AI Tasks."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from omniforge.core.utils import to_json_object, to_json_safe


@dataclass
class JobRecord:
    """Lifecycle snapshot for a background analytics job."""

    job_id: str
    name: str
    status: str = "queued"  # queued | running | completed | failed
    progress: float = 0.0
    message: str = "Queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: str | None = None
    owner_key: str = "default"

    def to_dict(self) -> dict[str, Any]:
        duration = None
        if self.started_at is not None:
            end = self.finished_at if self.finished_at is not None else time.time()
            duration = round(end - self.started_at, 3)
        return to_json_object(
            {
                "job_id": self.job_id,
                "name": self.name,
                "status": self.status,
                "progress": round(float(self.progress), 3),
                "message": self.message,
                "duration_seconds": duration,
                "result": self.result if self.status == "completed" else None,
                "error": self.error,
            }
        )


class BackgroundJobManager:
    """In-process bounded thread-pool job manager for AutoML, Deep Learning & Reporting tasks."""

    def __init__(self, max_workers: int = 4, max_history: int = 128) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="omniforge-job")
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._max_history = max_history

    def submit(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        owner_key: str = "default",
        **kwargs: Any,
    ) -> JobRecord:
        """Schedule a callable onto the background thread pool and return its initial record."""
        job_id = uuid.uuid4().hex[:12]
        rec = JobRecord(job_id=job_id, name=name, owner_key=owner_key or "default")
        with self._lock:
            self._jobs[job_id] = rec
            if len(self._jobs) > self._max_history:
                oldest = sorted(self._jobs.values(), key=lambda j: j.created_at)[: len(self._jobs) - self._max_history]
                for old in oldest:
                    if old.status in ("completed", "failed"):
                        self._jobs.pop(old.job_id, None)

        def _runner() -> None:
            rec.status = "running"
            rec.started_at = time.time()
            rec.progress = 0.15
            rec.message = f"Running {name}..."
            try:
                out = fn(*args, **kwargs)
                rec.result = to_json_safe(out)
                rec.progress = 1.0
                rec.status = "completed"
                rec.message = f"Completed {name}"
            except Exception as exc:
                rec.status = "failed"
                rec.error = str(exc)
                rec.message = f"Failed: {exc}"
            finally:
                rec.finished_at = time.time()

        self._executor.submit(_runner)
        return rec

    def get(self, job_id: str, *, owner_key: str | None = None) -> JobRecord | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return None
            if owner_key is not None and rec.owner_key != owner_key:
                return None
            return rec

    def list_recent(self, limit: int = 20, *, owner_key: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            candidates = (
                [j for j in self._jobs.values() if j.owner_key == owner_key]
                if owner_key is not None
                else list(self._jobs.values())
            )
            ordered = sorted(candidates, key=lambda j: j.created_at, reverse=True)[:limit]
            return [j.to_dict() for j in ordered]


job_manager = BackgroundJobManager()
