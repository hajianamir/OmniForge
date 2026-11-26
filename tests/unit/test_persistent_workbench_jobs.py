"""Bounded, cancellable, owner-scoped workbench jobs survive manager restarts."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from omniforge.api import workbench as wb
from omniforge.api.auth import get_api_key
from omniforge.api.jobs import BackgroundJobManager, JobCancellationToken, JobQueueFullError
from omniforge.api.main import create_app
from omniforge.config.settings import OmniForgeSettings, get_settings
from omniforge.core.types import Principal


def _wait_for(manager: BackgroundJobManager, job_id: str, status: str) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        record = manager.get(job_id)
        if record is not None and record.status == status:
            return
        time.sleep(0.01)
    pytest.fail(f"job {job_id} did not become {status}")


def test_job_settings_can_be_configured_via_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OMNIFORGE_JOBS__MAX_WORKERS", "3")
    monkeypatch.setenv("OMNIFORGE_JOBS__MAX_QUEUE_DEPTH", "7")
    monkeypatch.setenv("OMNIFORGE_JOBS__MAX_HISTORY", "11")
    monkeypatch.setenv("OMNIFORGE_JOBS__WORKER_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("OMNIFORGE_JOBS__DB_PATH", str(tmp_path / "shared.db"))
    settings = OmniForgeSettings()
    assert (settings.jobs.max_workers, settings.jobs.max_queue_depth, settings.jobs.max_history) == (3, 7, 11)
    assert settings.jobs.worker_timeout_seconds == 30
    assert settings.jobs.db_path == tmp_path / "shared.db"


def test_shared_queue_limits_cancellation_visibility_and_restart(tmp_path: Path) -> None:
    db = tmp_path / "jobs.db"
    one = BackgroundJobManager(max_workers=1, max_queue_depth=1, db_path=db)
    two = BackgroundJobManager(max_workers=1, max_queue_depth=1, db_path=db)
    running = threading.Event()
    release = threading.Event()
    executed = threading.Event()

    def blocker() -> dict[str, int]:
        running.set()
        assert release.wait(timeout=8)
        return {"answer": 42}

    try:
        first = one.submit("running", blocker, owner_key="alice")
        assert running.wait(timeout=3)
        queued = one.submit("waiting", executed.set, owner_key="alice")
        assert two.get(first.job_id, owner_key="alice").status == "running"  # type: ignore[union-attr]
        assert two.get(queued.job_id, owner_key="bob") is None
        assert two.list_recent(owner_key="bob") == []
        with pytest.raises(JobQueueFullError):
            two.submit("too many", lambda: None, owner_key="alice")
        assert two.cancel(queued.job_id, owner_key="bob") is None
        assert two.cancel(first.job_id, owner_key="alice") == "running"
        assert two.cancel(queued.job_id, owner_key="alice") == "cancelled"
        assert two.cancel(queued.job_id, owner_key="alice") == "cancelled"
        release.set()
        _wait_for(two, first.job_id, "completed")
        assert not executed.is_set()
        assert two.get(first.job_id, owner_key="alice").result == {"answer": 42}  # type: ignore[union-attr]
    finally:
        release.set()
        one.shutdown()
        two.shutdown()

    restarted = BackgroundJobManager(db_path=db)
    try:
        assert restarted.get(first.job_id, owner_key="alice").result == {"answer": 42}  # type: ignore[union-attr]
        assert restarted.get(queued.job_id, owner_key="alice").status == "cancelled"  # type: ignore[union-attr]
    finally:
        restarted.shutdown()


def test_running_cooperative_cancellation_and_commit_race(tmp_path: Path) -> None:
    db = tmp_path / "jobs.db"
    worker = BackgroundJobManager(max_workers=1, db_path=db)
    other = BackgroundJobManager(max_workers=1, db_path=db)
    started, proceed, committed, release = (threading.Event() for _ in range(4))

    def work(*, cancellation_token: JobCancellationToken) -> str:
        started.set()
        assert proceed.wait(timeout=8)
        cancellation_token.checkpoint()
        cancellation_token.begin_commit()
        committed.set()
        assert release.wait(timeout=8)
        return "published"

    try:
        cancelled = worker.submit("cancel me", work, owner_key="alice", cancellable=True)
        assert started.wait(timeout=3)
        assert other.cancel(cancelled.job_id, owner_key="alice") == "cancelling"
        assert other.cancel(cancelled.job_id, owner_key="alice") == "cancelling"
        proceed.set()
        _wait_for(other, cancelled.job_id, "cancelled")
        assert not committed.is_set()

        started.clear()
        proceed.clear()
        committing = worker.submit("finish me", work, owner_key="alice", cancellable=True)
        assert started.wait(timeout=3)
        proceed.set()
        assert committed.wait(timeout=3)
        assert other.cancel(committing.job_id, owner_key="alice") == "committing"
        release.set()
        _wait_for(other, committing.job_id, "completed")
        assert other.get(committing.job_id).result == "published"  # type: ignore[union-attr]
    finally:
        proceed.set()
        release.set()
        worker.shutdown()
        other.shutdown()


def test_shared_capacity_is_atomic_for_simultaneous_submissions(tmp_path: Path) -> None:
    db = tmp_path / "jobs.db"
    one = BackgroundJobManager(max_workers=1, max_queue_depth=1, db_path=db)
    two = BackgroundJobManager(max_workers=1, max_queue_depth=1, db_path=db)
    started_a, started_b, release = threading.Event(), threading.Event(), threading.Event()

    def block(started: threading.Event) -> None:
        started.set()
        assert release.wait(timeout=8)

    outcomes: list[str] = []
    barrier = threading.Barrier(2)

    def submit(manager: BackgroundJobManager) -> None:
        barrier.wait(timeout=5)
        try:
            manager.submit("contender", lambda: None)
            outcomes.append("accepted")
        except JobQueueFullError:
            outcomes.append("full")

    try:
        one.submit("block-a", block, started_a)
        assert started_a.wait(timeout=3)
        two.submit("block-b", block, started_b)
        assert started_b.wait(timeout=3)
        threads = [threading.Thread(target=submit, args=(manager,)) for manager in (one, two)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert sorted(outcomes) == ["accepted", "full"]
    finally:
        release.set()
        one.shutdown()
        two.shutdown()


def test_thread_termination_is_recorded_as_failure(tmp_path: Path) -> None:
    manager = BackgroundJobManager(db_path=tmp_path / "jobs.db")

    def stops_thread() -> None:
        raise SystemExit("stopped")

    try:
        job = manager.submit("stop", stops_thread)
        _wait_for(manager, job.job_id, "failed")
        assert manager.get(job.job_id).error == "The job failed. Check the inputs and retry."  # type: ignore[union-attr]
    finally:
        manager.shutdown()


def test_lost_worker_is_marked_failed_without_reexecution(tmp_path: Path) -> None:
    db = tmp_path / "jobs.db"
    one = BackgroundJobManager(max_workers=1, db_path=db, worker_timeout_seconds=3)
    two = BackgroundJobManager(max_workers=1, db_path=db, worker_timeout_seconds=3)
    running = threading.Event()
    release = threading.Event()

    def blocker() -> str:
        running.set()
        assert release.wait(timeout=8)
        return "should-not-overwrite-expired-state"

    try:
        job = one.submit("long job", blocker, owner_key="alice")
        assert running.wait(timeout=3)
        # Simulate a dead worker: stop its heartbeat and expire its lease.
        one._stop_heartbeat.set()
        assert one._heartbeat_thread is not None
        one._heartbeat_thread.join(timeout=2)
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE workbench_job_workers SET heartbeat = 0 WHERE worker_id = ?", (one._worker_id,))
        _wait_for(two, job.job_id, "failed")
        assert "not resumed" in (two.get(job.job_id).error or "")  # type: ignore[union-attr]
        release.set()
    finally:
        release.set()
        one.shutdown()
        two.shutdown()

    check = BackgroundJobManager(db_path=db)
    try:
        assert check.get(job.job_id).status == "failed"  # type: ignore[union-attr]
    finally:
        check.shutdown()


def test_async_automl_returns_429_when_queue_is_full(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manager = BackgroundJobManager(max_workers=1, max_queue_depth=1, db_path=tmp_path / "jobs.db")
    monkeypatch.setattr(wb, "job_manager", manager)
    started, release = threading.Event(), threading.Event()

    def blocker() -> None:
        started.set()
        assert release.wait(timeout=8)

    try:
        manager.submit("blocking", blocker)
        assert started.wait(timeout=3)
        manager.submit("waiting", lambda: None)
        client = TestClient(create_app(), headers={"X-API-Key": get_api_key(), "X-Session-ID": "full-queue"})
        loaded = client.post("/api/workbench/s01/load-sample", json={"source_type": "domain", "name": "finance", "n_samples": 40})
        assert loaded.status_code == 200
        response = client.post("/api/workbench/s06/automl", json={"async_job": True})
        assert response.status_code == 429
        assert response.headers["Retry-After"] == "5"
    finally:
        release.set()
        manager.shutdown()


def test_cancel_api_is_owner_scoped_and_running_jobs_are_not_falsely_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manager = BackgroundJobManager(max_workers=1, db_path=tmp_path / "jobs.db")
    monkeypatch.setattr(wb, "job_manager", manager)
    owner_principal = Principal(subject="api_key_principal", roles=["admin"], tenant_id=get_settings().default_tenant_id)
    owner = wb.current_job_owner_key(owner_principal)
    started = threading.Event()
    release = threading.Event()

    def blocker() -> None:
        started.set()
        assert release.wait(timeout=8)

    try:
        running = manager.submit("blocking", blocker, owner_key=owner or "default")
        assert started.wait(timeout=3)
        waiting = manager.submit("waiting", lambda: None, owner_key=owner or "default")
        other = manager.submit("private", lambda: None, owner_key="someone-else")
        client = TestClient(create_app(), headers={"X-API-Key": get_api_key()})
        assert client.post(f"/api/workbench/jobs/{other.job_id}/cancel").status_code == 404
        assert client.post(f"/api/workbench/jobs/{running.job_id}/cancel").status_code == 409
        cancelled = client.post(f"/api/workbench/jobs/{waiting.job_id}/cancel")
        assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
        assert client.get(f"/api/workbench/jobs/{waiting.job_id}").json()["status"] == "cancelled"
    finally:
        release.set()
        manager.shutdown()


def test_fastapi_lifespan_drains_workbench_jobs_and_releases_worker_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omniforge.api import main
    from omniforge.api.jobs import _ConfiguredJobManager

    db_path = tmp_path / "lifespan.db"
    manager = BackgroundJobManager(max_workers=1, db_path=db_path)
    configured = _ConfiguredJobManager()
    configured._manager = manager
    monkeypatch.setattr(main, "job_manager", configured)
    owner = wb.current_job_owner_key(
        Principal(subject="api_key_principal", roles=["admin"], tenant_id=get_settings().default_tenant_id),
    )
    monkeypatch.setattr(wb, "job_manager", configured)
    release = threading.Event()
    started = threading.Event()
    draining = threading.Event()

    def work() -> dict[str, bool]:
        started.set()
        assert release.wait(timeout=5)
        return {"completed": True}

    def release_when_draining() -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            with manager._submission_lock:
                if not manager._accepting_jobs:
                    draining.set()
                    break
            time.sleep(0.01)
        release.set()

    responder = threading.Thread(target=release_when_draining)
    try:
        with TestClient(create_app(), headers={"X-API-Key": get_api_key()}) as outer:
            submitted = configured.submit("test shutdown", work, owner_key=owner or "default")
            assert started.wait(timeout=3)
            with TestClient(create_app(), headers={"X-API-Key": get_api_key()}) as inner:
                assert inner.get(f"/api/workbench/jobs/{submitted.job_id}").status_code == 200
            with sqlite3.connect(db_path) as conn:
                assert conn.execute("SELECT COUNT(*) FROM workbench_job_workers").fetchone() == (1,)
            assert outer.get(f"/api/workbench/jobs/{submitted.job_id}").status_code == 200
            responder.start()
        responder.join(timeout=3)
        assert draining.is_set()
        assert configured._manager is None
        with pytest.raises(JobQueueFullError, match="shutting down"):
            manager.submit("late job", lambda: None)
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM workbench_job_workers").fetchone() == (0,)
        persisted = manager.get(submitted.job_id, owner_key=owner)
        assert persisted is not None and persisted.result == {"completed": True}
    finally:
        release.set()
        manager.shutdown()
