"""Regression coverage for bounded job history and batched experiment reads."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from omniforge.core.exceptions import MLOpsError
from omniforge.mlops import tracking
from omniforge.mlops.job_queue import BackgroundJobQueue


def test_sdk_job_capacity_never_evicts_unfinished_work() -> None:
    started, release = threading.Event(), threading.Event()

    def wait_for_release() -> str:
        started.set()
        assert release.wait(timeout=5)
        return "finished"

    queue = BackgroundJobQueue(max_workers=1, max_queue_depth=1, max_history=1)
    try:
        active = queue.submit(wait_for_release)
        assert started.wait(timeout=3)
        pending = queue.submit(int, 5)
        assert queue.status(active) == queue.status(pending) == "running"
        with pytest.raises(MLOpsError, match="queue is full"):
            queue.submit(int, 6)
        queue.shutdown(wait=False, cancel_pending=True)
        assert queue.status(pending) == "cancelled"
        assert queue.status(active) == "running"
        with pytest.raises(MLOpsError, match="shut down"):
            queue.submit(int, 7)
        release.set()
        queue.shutdown()
        assert queue.status(active) == "not_found"
        assert queue.status(pending) == "cancelled"
    finally:
        release.set()
        queue.shutdown()


def test_tracking_listing_preserves_records_with_one_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = tracking.ExperimentTracker(tmp_path / "runs.db", tmp_path / "artifacts")
    expected = []
    for experiment in ["a", "b", "a"]:
        with tracker.start_run(experiment, tags={"owner": "fixture"}) as run_id:
            tracker.log_params({"alpha": [1, 2]}, run_id)
            tracker.log_metrics({"rmse": 1.5}, run_id=run_id)
        expected.insert(0, tracker.get_run(run_id))

    connects = []
    old_connect = tracking.sqlite3.connect

    def track_open(*args, **kwargs):
        connects.append(1)
        return old_connect(*args, **kwargs)

    monkeypatch.setattr(tracking.sqlite3, "connect", track_open)
    assert tracker.list_runs() == expected
    assert len(connects) == 1
    connects.clear()
    assert tracker.list_runs("a", limit=1) == expected[:1]
    assert len(connects) == 1
