"""MLflow-compatible SQLite Experiment Tracking engine with runs, params, metrics, tags, and artifacts."""

from __future__ import annotations

import json
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from omniforge.core.exceptions import MLOpsError
from omniforge.core.logging import get_logger
from omniforge.core.utils import safe_json_default

logger = get_logger(__name__)
_RUN_COLUMNS = "run_id, experiment_name, run_name, status, started_at, ended_at, params_json, metrics_json, tags_json, artifacts_json"


@contextmanager
def _sqlite_conn(db_path: str | Path, **kwargs: Any) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection that is always committed/rolled back **and closed**.

    ``with sqlite3.connect(...)`` only wraps a transaction; the underlying handle
    stays open until garbage collection, which leaks file descriptors and emits
    ``ResourceWarning: unclosed database``.
    """
    conn = sqlite3.connect(db_path, **kwargs)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


class RunRecord(BaseModel):
    """Structured record of a tracked experiment run."""

    run_id: str
    experiment_name: str
    run_name: str
    status: str  # RUNNING, FINISHED, FAILED
    started_at: str
    ended_at: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)


class ExperimentTracker:
    """SQLite-persisted MLflow-compatible experiment tracking store."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        artifact_root: str | Path | None = None,
    ) -> None:
        from omniforge.config.settings import get_settings

        paths = get_settings().paths
        self.db_path = Path(db_path) if db_path is not None else paths.tracking_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root = Path(artifact_root) if artifact_root is not None else paths.artifact_store_dir
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._active_run_id: str | None = None
        self._init_db()

    def _init_db(self) -> None:
        with _sqlite_conn(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    run_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    params_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metric_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value REAL NOT NULL,
                    step INTEGER NOT NULL,
                    recorded_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_experiment_started ON runs(experiment_name, started_at DESC);
                """
            )
            conn.commit()

    def create_experiment(self, name: str, description: str = "") -> str:
        """Create an experiment if it does not already exist."""
        with _sqlite_conn(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO experiments (name, description, created_at) VALUES (?, ?, ?)",
                (name, description, datetime.now(UTC).isoformat()),
            )
            conn.commit()
        return name

    @contextmanager
    def start_run(
        self,
        experiment_name: str = "default",
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Iterator[str]:
        """Context manager starting, managing, and cleanly closing an experiment run."""
        self.create_experiment(experiment_name)
        run_id = f"exp_{uuid4().hex[:12]}"
        r_name = run_name or run_id
        now = datetime.now(UTC).isoformat()
        with _sqlite_conn(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, experiment_name, run_name, status, started_at, ended_at, params_json, metrics_json, tags_json, artifacts_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, experiment_name, r_name, "RUNNING", now, None, "{}", "{}", json.dumps(tags or {}), "[]"),
            )
            conn.commit()
        prev = self._active_run_id
        self._active_run_id = run_id
        try:
            yield run_id
            self.end_run(run_id, status="FINISHED")
        except Exception:
            self.end_run(run_id, status="FAILED")
            raise
        finally:
            self._active_run_id = prev

    def log_params(self, params: dict[str, Any], run_id: str | None = None) -> None:
        """Log a dictionary of hyperparameters to the active or specified run."""
        rid = self._resolve_run(run_id)
        rec = self.get_run(rid)
        merged = {**rec.params, **{str(k): v for k, v in params.items()}}
        with _sqlite_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE runs SET params_json = ? WHERE run_id = ?",
                (json.dumps(merged, default=safe_json_default), rid),
            )
            conn.commit()

    def log_metrics(self, metrics: dict[str, float], step: int = 0, run_id: str | None = None) -> None:
        """Log numeric evaluation metrics and record step-wise history."""
        rid = self._resolve_run(run_id)
        rec = self.get_run(rid)
        merged = dict(rec.metrics)
        now = datetime.now(UTC).isoformat()
        with _sqlite_conn(self.db_path) as conn:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    merged[str(k)] = float(v)
                    conn.execute(
                        "INSERT INTO metric_history (run_id, key, value, step, recorded_at) VALUES (?, ?, ?, ?, ?)",
                        (rid, str(k), float(v), int(step), now),
                    )
            conn.execute("UPDATE runs SET metrics_json = ? WHERE run_id = ?", (json.dumps(merged), rid))
            conn.commit()

    def log_artifact(self, source_path: str | Path, run_id: str | None = None) -> Path:
        """Copy an artifact file into the run's managed artifact store."""
        rid = self._resolve_run(run_id)
        rec = self.get_run(rid)
        src = Path(source_path)
        if not src.exists() or not src.is_file():
            raise MLOpsError(f"Artifact file does not exist: {src}")
        root_resolved = self.artifact_root.resolve()
        dest_dir = (self.artifact_root / rid).resolve()
        if not dest_dir.is_relative_to(root_resolved):
            raise MLOpsError(f"Run artifact directory escapes tracker root for run '{rid}'.")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = (dest_dir / src.name).resolve()
        if not dest.is_relative_to(dest_dir):
            raise MLOpsError(f"Artifact destination escapes run directory: {src.name!r}")
        shutil.copy2(src, dest)
        arts = list(dict.fromkeys([*rec.artifacts, str(dest)]))
        with _sqlite_conn(self.db_path) as conn:
            conn.execute("UPDATE runs SET artifacts_json = ? WHERE run_id = ?", (json.dumps(arts), rid))
            conn.commit()
        return dest

    def end_run(self, run_id: str, status: str = "FINISHED") -> None:
        """Finalize run status and end timestamp."""
        with _sqlite_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE runs SET status = ?, ended_at = ? WHERE run_id = ?",
                (status, datetime.now(UTC).isoformat(), run_id),
            )
            conn.commit()

    def get_run(self, run_id: str) -> RunRecord:
        """Fetch a tracked run record by ID."""
        with _sqlite_conn(self.db_path) as conn:
            row = conn.execute(
                f"SELECT {_RUN_COLUMNS} FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise MLOpsError(f"Experiment run '{run_id}' not found.")
        return self._parse_run(row)

    @staticmethod
    def _parse_run(row: tuple[Any, ...]) -> RunRecord:
        return RunRecord(
            run_id=row[0],
            experiment_name=row[1],
            run_name=row[2],
            status=row[3],
            started_at=row[4],
            ended_at=row[5],
            params=json.loads(row[6]),
            metrics=json.loads(row[7]),
            tags=json.loads(row[8]),
            artifacts=json.loads(row[9]),
        )

    def list_runs(self, experiment_name: str | None = None, limit: int = 50) -> list[RunRecord]:
        """List recent experiment runs, optionally filtered by experiment name."""
        with _sqlite_conn(self.db_path) as conn:
            if experiment_name:
                rows = conn.execute(
                    f"SELECT {_RUN_COLUMNS} FROM runs WHERE experiment_name = ? ORDER BY started_at DESC LIMIT ?",
                    (experiment_name, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {_RUN_COLUMNS} FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [self._parse_run(row) for row in rows]

    def _resolve_run(self, run_id: str | None) -> str:
        rid = run_id or self._active_run_id
        if not rid:
            raise MLOpsError("No active experiment run. Call start_run() or pass run_id.")
        return rid
