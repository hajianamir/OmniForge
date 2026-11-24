"""Exercise the shared job database across real OS processes."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from omniforge.api.jobs import BackgroundJobManager


def _wait_for_file(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"worker exited before reporting its job: {stdout} {stderr}")
        time.sleep(0.02)
    pytest.fail("worker did not start in time")


def test_another_process_can_poll_and_cancel_an_owner_scoped_job(tmp_path: Path) -> None:
    db = tmp_path / "jobs.db"
    manager = BackgroundJobManager(max_workers=1, db_path=db)
    started, release, executed = threading.Event(), threading.Event(), threading.Event()

    def block() -> None:
        started.set()
        assert release.wait(timeout=15)

    try:
        manager.submit("block", block, owner_key="alice")
        assert started.wait(timeout=5)
        job = manager.submit("queued", executed.set, owner_key="alice")
        script = """
import json, sys
from omniforge.api.jobs import BackgroundJobManager
mgr = BackgroundJobManager(db_path=sys.argv[1])
rec = mgr.get(sys.argv[2], owner_key='alice')
print(json.dumps({'before': rec.status, 'hidden': mgr.get(sys.argv[2], owner_key='bob') is None,
                  'cancel': mgr.cancel(sys.argv[2], owner_key='alice')}), flush=True)
mgr.shutdown()
"""
        child = subprocess.run([sys.executable, "-c", script, str(db), job.job_id], capture_output=True, text=True, timeout=20, check=True)
        result = json.loads(child.stdout.strip().splitlines()[-1])
        assert result == {"before": "queued", "hidden": True, "cancel": "cancelled"}
        assert manager.get(job.job_id).status == "cancelled"  # type: ignore[union-attr]
    finally:
        release.set()
        manager.shutdown()
    assert not executed.is_set()


def test_killed_process_leaves_a_failed_not_immortal_job(tmp_path: Path) -> None:
    db, ready = tmp_path / "jobs.db", tmp_path / "ready.json"
    script = """
import json, sys, threading
from pathlib import Path
from omniforge.api.jobs import BackgroundJobManager
manager = BackgroundJobManager(max_workers=1, db_path=sys.argv[1], worker_timeout_seconds=1.5)
started = threading.Event()
def work():
    started.set()
    threading.Event().wait(60)
job = manager.submit('crash', work, owner_key='alice')
assert started.wait(5)
ready = Path(sys.argv[2])
tmp = ready.with_suffix('.tmp')
tmp.write_text(json.dumps({'job_id': job.job_id}), encoding='utf-8')
tmp.replace(ready)
threading.Event().wait(60)
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(db), str(ready)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        _wait_for_file(ready, process)
        job_id = json.loads(ready.read_text(encoding="utf-8"))["job_id"]
        process.kill()
        process.communicate(timeout=5)
        observer = BackgroundJobManager(db_path=db, worker_timeout_seconds=1.5)
        try:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                record = observer.get(job_id, owner_key="alice")
                if record is not None and record.status == "failed":
                    break
                time.sleep(0.05)
            else:
                pytest.fail("orphaned job was not marked failed after the worker lease expired")
            assert record is not None and "not resumed" in (record.error or "")
        finally:
            observer.shutdown()
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
