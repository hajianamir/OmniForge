"""Bounded workbench jobs with SQLite-backed, owner-scoped status snapshots.

Callables still execute in the submitting process: SQLite persists their state,
not arbitrary Python closures. If a worker disappears, its unfinished jobs are
marked failed after its lease expires rather than being silently re-executed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omniforge.config.settings import get_settings
from omniforge.core.logging import get_logger
from omniforge.core.utils import to_json_object, to_json_safe

logger = get_logger(__name__)


class JobQueueFullError(RuntimeError):
    """The configured number of waiting jobs has been reached."""


class JobCancelledError(Exception):
    """A cooperative job observed cancellation before committing its result."""


class JobCancellationToken:
    """Cross-worker cancellation checks for an opted-in local callable."""

    def __init__(self, manager: BackgroundJobManager, job_id: str) -> None:
        self._manager = manager
        self._job_id = job_id

    def checkpoint(self) -> None:
        """Raise if cancellation was requested or the worker lease was lost."""
        self._manager._check_cancellation(self._job_id)

    def begin_commit(self) -> None:
        """Close the cancellation window atomically before publishing side effects."""
        self._manager._begin_commit(self._job_id)


@dataclass
class JobRecord:
    """Lifecycle snapshot for a background analytics job."""

    job_id: str
    name: str
    status: str = "queued"  # queued | running | cancelling | committing | completed | failed | cancelled
    cancellable: bool = False
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
                "can_cancel": self.status == "queued" or (self.status == "running" and self.cancellable),
                "progress": round(float(self.progress), 3),
                "message": self.message,
                "duration_seconds": duration,
                "result": self.result if self.status == "completed" else None,
                "error": self.error,
            }
        )


class BackgroundJobManager:
    """Execute local callables while sharing bounded queue state through SQLite.

    ``max_queue_depth`` limits waiting jobs across managers using the same DB.
    Queued jobs cancel immediately. Running callables must opt in and check a
    cancellation token; Python threads cannot be forcibly stopped safely. An
    in-memory DB is used when ``db_path`` is omitted, useful for isolated tests.
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_history: int = 128,
        *,
        max_queue_depth: int = 128,
        db_path: str | Path | None = None,
        worker_timeout_seconds: float = 120.0,
    ) -> None:
        if max_workers < 1 or max_history < 1 or max_queue_depth < 1 or worker_timeout_seconds <= 0:
            raise ValueError("Worker count, history, queue depth and worker timeout must be positive.")
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="omniforge-job")
        self._lock = threading.RLock()
        self._max_history = max_history
        self._max_queue_depth = max_queue_depth
        # Bound the executor's local waiting work items even if another process
        # cancels many queued jobs while all local worker threads are busy.
        self._local_pending = threading.BoundedSemaphore(max_queue_depth)
        self._worker_id = uuid.uuid4().hex
        self._worker_timeout = worker_timeout_seconds
        self._db_path = Path(db_path) if db_path is not None else None
        self._memory_db: sqlite3.Connection | None = None
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        if self._db_path is None:
            self._memory_db = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
            self._memory_db.row_factory = sqlite3.Row
        else:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        if self._db_path is not None:
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, name="omniforge-job-lease", daemon=True)
            self._heartbeat_thread.start()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Serialize a transaction locally; SQLite serializes writers across processes."""
        with self._lock:
            conn = self._memory_db
            if conn is None:
                assert self._db_path is not None
                conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
                conn.row_factory = sqlite3.Row
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                if conn is not self._memory_db:
                    conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS workbench_job_workers (
                    worker_id TEXT PRIMARY KEY, heartbeat REAL NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS workbench_jobs (
                    job_id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                    progress REAL NOT NULL, message TEXT NOT NULL, created_at REAL NOT NULL,
                    started_at REAL, finished_at REAL, result_json TEXT, error TEXT,
                    owner_key TEXT NOT NULL, worker_id TEXT NOT NULL,
                    cancellable INTEGER NOT NULL DEFAULT 0
                )"""
            )
            columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(workbench_jobs)")}
            if "cancellable" not in columns:
                conn.execute("ALTER TABLE workbench_jobs ADD COLUMN cancellable INTEGER NOT NULL DEFAULT 0")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workbench_jobs_owner ON workbench_jobs(owner_key, created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_workbench_jobs_status ON workbench_jobs(status)")
            conn.execute(
                "INSERT INTO workbench_job_workers (worker_id, heartbeat) VALUES (?, ?)",
                (self._worker_id, time.time()),
            )

    def _touch_worker(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """INSERT INTO workbench_job_workers (worker_id, heartbeat) VALUES (?, ?)
               ON CONFLICT(worker_id) DO UPDATE SET heartbeat = excluded.heartbeat""",
            (self._worker_id, time.time()),
        )

    def _heartbeat_loop(self) -> None:
        interval = max(0.1, min(10.0, self._worker_timeout / 3.0))
        while not self._stop_heartbeat.wait(interval):
            try:
                with self._connection() as conn:
                    self._touch_worker(conn)
            except sqlite3.Error as exc:
                logger.warning("job_worker_heartbeat_failed", error=str(exc))

    def _reap_expired_workers(self, conn: sqlite3.Connection) -> None:
        if self._db_path is None:
            return
        now = time.time()
        conn.execute(
            """UPDATE workbench_jobs SET status = 'failed', finished_at = ?,
                   message = 'Worker lost before job completed', error = 'Worker lease expired; job was not resumed'
               WHERE status IN ('queued', 'running', 'cancelling', 'committing') AND worker_id IN
                   (SELECT worker_id FROM workbench_job_workers WHERE heartbeat < ?)""",
            (now, now - self._worker_timeout),
        )
        conn.execute(
            """DELETE FROM workbench_job_workers WHERE heartbeat < ?
               AND worker_id NOT IN (SELECT worker_id FROM workbench_jobs WHERE status IN ('queued', 'running', 'cancelling', 'committing'))""",
            (now - self._worker_timeout,),
        )

    def _prune(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """DELETE FROM workbench_jobs WHERE job_id IN (
                   SELECT job_id FROM workbench_jobs WHERE status IN ('completed', 'failed', 'cancelled')
                   ORDER BY created_at DESC, job_id DESC LIMIT -1 OFFSET ?
               )""",
            (self._max_history,),
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"], name=row["name"], status=row["status"], cancellable=bool(row["cancellable"]), progress=row["progress"],
            message=row["message"], created_at=row["created_at"], started_at=row["started_at"],
            finished_at=row["finished_at"], result=json.loads(row["result_json"]) if row["result_json"] is not None else None,
            error=row["error"], owner_key=row["owner_key"],
        )

    def submit(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        owner_key: str = "default",
        cancellable: bool = False,
        **kwargs: Any,
    ) -> JobRecord:
        """Schedule work; opted-in callables receive a ``cancellation_token`` keyword."""
        if not self._local_pending.acquire(blocking=False):
            raise JobQueueFullError("The local job queue is full. Try again later.")
        rec = JobRecord(job_id=uuid.uuid4().hex, name=name, owner_key=owner_key or "default", cancellable=cancellable)
        try:
            with self._connection() as conn:
                self._touch_worker(conn)
                self._reap_expired_workers(conn)
                queued = int(conn.execute("SELECT COUNT(*) FROM workbench_jobs WHERE status = 'queued'").fetchone()[0])
                if queued >= self._max_queue_depth:
                    raise JobQueueFullError("The shared job queue is full. Try again later.")
                conn.execute(
                    """INSERT INTO workbench_jobs
                       (job_id, name, status, progress, message, created_at, owner_key, worker_id, cancellable)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (rec.job_id, rec.name, rec.status, rec.progress, rec.message, rec.created_at, rec.owner_key, self._worker_id, int(cancellable)),
                )
                self._prune(conn)
        except Exception:
            self._local_pending.release()
            raise

        def _runner() -> None:
            self._local_pending.release()
            with self._connection() as conn:
                claimed = conn.execute(
                    """UPDATE workbench_jobs SET status = 'running', started_at = ?, progress = 0.15, message = ?
                       WHERE job_id = ? AND worker_id = ? AND status = 'queued'""",
                    (time.time(), f"Running {name}...", rec.job_id, self._worker_id),
                ).rowcount
            if not claimed:
                return  # cancelled or expired before it started
            try:
                if cancellable:
                    token = JobCancellationToken(self, rec.job_id)
                    token.checkpoint()
                    out = fn(*args, cancellation_token=token, **kwargs)
                    # An opted-in callable can close this window before its side
                    # effects; otherwise close it before publishing its result.
                    token.begin_commit()
                else:
                    out = fn(*args, **kwargs)
                result_json = json.dumps(to_json_safe(out), allow_nan=False)
                status, message, error = "completed", f"Completed {name}", None
            except JobCancelledError:
                result_json, error = None, None
                status, message = "cancelled", "Cancelled at a safe checkpoint"
            except BaseException as exc:
                # ThreadPoolExecutor captures SystemExit/KeyboardInterrupt too; make
                # their jobs terminal rather than leaving an immortal running row.
                result_json = None
                error = str(exc) or type(exc).__name__
                status, message = "failed", f"Failed: {error}"
            with self._connection() as conn:
                conn.execute(
                    """UPDATE workbench_jobs SET status = ?, progress = ?, message = ?, result_json = ?, error = ?,
                       finished_at = ? WHERE job_id = ? AND worker_id = ? AND status IN ('running', 'cancelling', 'committing')""",
                    (status, 1.0 if status == "completed" else 0.15, message, result_json, error, time.time(), rec.job_id, self._worker_id),
                )
                self._prune(conn)

        try:
            self._executor.submit(_runner)
        except Exception as exc:
            self._local_pending.release()
            with self._connection() as conn:
                conn.execute(
                    "UPDATE workbench_jobs SET status = 'failed', message = ?, error = ?, finished_at = ? WHERE job_id = ?",
                    ("Failed to schedule job", str(exc), time.time(), rec.job_id),
                )
            raise
        return rec

    def _check_cancellation(self, job_id: str) -> None:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT status FROM workbench_jobs WHERE job_id = ? AND worker_id = ?", (job_id, self._worker_id)
            ).fetchone()
            if row is None or row["status"] not in ("running", "committing"):
                raise JobCancelledError("Job cancelled or worker lease lost.")

    def _begin_commit(self, job_id: str) -> None:
        with self._connection() as conn:
            updated = conn.execute(
                """UPDATE workbench_jobs SET status = 'committing', progress = 0.9, message = 'Publishing result'
                   WHERE job_id = ? AND worker_id = ? AND status = 'running'""", (job_id, self._worker_id)
            ).rowcount
            if not updated:
                row = conn.execute(
                    "SELECT status FROM workbench_jobs WHERE job_id = ? AND worker_id = ?", (job_id, self._worker_id)
                ).fetchone()
                if row is None or row["status"] != "committing":
                    raise JobCancelledError("Job cancelled or worker lease lost.")

    def get(self, job_id: str, *, owner_key: str | None = None) -> JobRecord | None:
        with self._connection() as conn:
            self._reap_expired_workers(conn)
            if owner_key is None:
                row = conn.execute("SELECT * FROM workbench_jobs WHERE job_id = ?", (job_id,)).fetchone()
            else:
                row = conn.execute("SELECT * FROM workbench_jobs WHERE job_id = ? AND owner_key = ?", (job_id, owner_key)).fetchone()
            return self._record(row) if row is not None else None

    def list_recent(self, limit: int = 20, *, owner_key: str | None = None) -> list[dict[str, Any]]:
        with self._connection() as conn:
            self._reap_expired_workers(conn)
            if owner_key is None:
                rows = conn.execute("SELECT * FROM workbench_jobs ORDER BY created_at DESC LIMIT ?", (max(0, limit),)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM workbench_jobs WHERE owner_key = ? ORDER BY created_at DESC LIMIT ?", (owner_key, max(0, limit))
                ).fetchall()
            return [self._record(row).to_dict() for row in rows]

    def cancel(self, job_id: str, *, owner_key: str | None = None) -> str | None:
        """Cancel waiting work or request cooperative running cancellation atomically."""
        with self._connection() as conn:
            self._reap_expired_workers(conn)
            if owner_key is None:
                row = conn.execute("SELECT status, cancellable FROM workbench_jobs WHERE job_id = ?", (job_id,)).fetchone()
            else:
                row = conn.execute("SELECT status, cancellable FROM workbench_jobs WHERE job_id = ? AND owner_key = ?", (job_id, owner_key)).fetchone()
            if row is None:
                return None
            status = str(row["status"])
            if status == "queued":
                conn.execute(
                    "UPDATE workbench_jobs SET status = 'cancelled', message = 'Cancelled before execution', finished_at = ? WHERE job_id = ?",
                    (time.time(), job_id),
                )
                self._prune(conn)
                return "cancelled"
            if status == "running" and bool(row["cancellable"]):
                conn.execute(
                    "UPDATE workbench_jobs SET status = 'cancelling', message = 'Cancellation requested; waiting for checkpoint' WHERE job_id = ?",
                    (job_id,),
                )
                return "cancelling"
            return status

    def shutdown(self) -> None:
        """Wait for local work, then release the worker lease and database handle."""
        self._executor.shutdown(wait=True)
        self._stop_heartbeat.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join()
        with self._connection() as conn:
            conn.execute("DELETE FROM workbench_job_workers WHERE worker_id = ?", (self._worker_id,))
        if self._memory_db is not None:
            self._memory_db.close()


class _ConfiguredJobManager:
    """Create the configured manager lazily and separately in each server process.

    This avoids inheriting a thread pool and heartbeat across pre-fork servers.
    """

    def __init__(self) -> None:
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._manager: BackgroundJobManager | None = None
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._after_fork)

    def _after_fork(self) -> None:
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._manager = None

    def _instance(self) -> BackgroundJobManager:
        if self._pid != os.getpid():
            self._after_fork()
        with self._lock:
            if self._manager is None:
                settings = get_settings()
                self._manager = BackgroundJobManager(
                    max_workers=settings.jobs.max_workers,
                    max_history=settings.jobs.max_history,
                    max_queue_depth=settings.jobs.max_queue_depth,
                    db_path=settings.jobs.db_path or (settings.paths.workspace_dir / "workbench_jobs.db"),
                    worker_timeout_seconds=settings.jobs.worker_timeout_seconds,
                )
            return self._manager

    def submit(
        self, name: str, fn: Callable[..., Any], *args: Any, owner_key: str = "default", cancellable: bool = False, **kwargs: Any
    ) -> JobRecord:
        return self._instance().submit(name, fn, *args, owner_key=owner_key, cancellable=cancellable, **kwargs)

    def get(self, job_id: str, *, owner_key: str | None = None) -> JobRecord | None:
        return self._instance().get(job_id, owner_key=owner_key)

    def list_recent(self, limit: int = 20, *, owner_key: str | None = None) -> list[dict[str, Any]]:
        return self._instance().list_recent(limit, owner_key=owner_key)

    def cancel(self, job_id: str, *, owner_key: str | None = None) -> str | None:
        return self._instance().cancel(job_id, owner_key=owner_key)


job_manager = _ConfiguredJobManager()
