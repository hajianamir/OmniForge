"""Versioned Enterprise Model Registry with Lifecycle Stages (None/Staging/Production/Archived)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from omniforge.core.exceptions import MLOpsError
from omniforge.core.logging import get_logger
from omniforge.core.types import ModelStage
from omniforge.models.base import BaseOmniForgeModel

logger = get_logger(__name__)


@contextmanager
def _sqlite_conn(db_path: str | Path, **kwargs: Any) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection that is always committed/rolled back **and closed**.

    ``with sqlite3.connect(...)`` only wraps a transaction; the underlying handle
    stays open until garbage collection, which leaks file descriptors and emits
    ``ResourceWarning: unclosed database``.
    """
    conn = sqlite3.connect(db_path, **kwargs)
    try:
        with conn:  # transaction scope: commit on success, rollback on error
            yield conn
    finally:
        conn.close()


class RegisteredModelVersion(BaseModel):
    """Metadata for a single version of a registered model."""

    model_name: str
    version: int
    stage: ModelStage
    artifact_path: str
    checksum_sha256: str
    metrics: dict[str, float] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    description: str = ""
    registered_at: str
    updated_at: str


class ModelRegistry:
    """SQLite-backed Model Registry supporting semantic versioning, stage governance, and rollback."""

    def __init__(
        self,
        db_path: str | Path = "./.omniforge_workspace/model_registry.db",
        artifact_dir: str | Path = "./.omniforge_workspace/artifacts/registry",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with _sqlite_conn(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_versions (
                    model_name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    artifact_path TEXT NOT NULL,
                    checksum_sha256 TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    description TEXT NOT NULL,
                    registered_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (model_name, version)
                )
                """
            )
            conn.commit()

    def register_model(
        self,
        model: BaseOmniForgeModel,
        model_name: str,
        *,
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        description: str = "",
        stage: ModelStage = ModelStage.NONE,
        encrypt: bool = False,
    ) -> RegisteredModelVersion:
        """Persist a fitted OmniForge model as the next monotonically increasing version."""
        with _sqlite_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(version) FROM model_versions WHERE model_name = ?", (model_name,)
            ).fetchone()
            next_ver = int(row[0] or 0) + 1

        dest = self.artifact_dir / model_name / f"v{next_ver}" / "model.joblib"
        saved_path = model.save(dest, encrypt=encrypt)
        now = datetime.now(UTC).isoformat()
        merged_metrics = {
            **{k: float(v) for k, v in model.metadata_.metrics.items() if isinstance(v, (int, float))},
            **{k: float(v) for k, v in (metrics or {}).items() if isinstance(v, (int, float))},
        }
        rec = RegisteredModelVersion(
            model_name=model_name,
            version=next_ver,
            stage=stage,
            artifact_path=str(saved_path.resolve()),
            checksum_sha256=model.metadata_.artifact_checksum,
            metrics=merged_metrics,
            tags=tags or {},
            description=description,
            registered_at=now,
            updated_at=now,
        )

        with _sqlite_conn(self.db_path) as conn:
            if stage == ModelStage.PRODUCTION:
                conn.execute(
                    "UPDATE model_versions SET stage = ?, updated_at = ? WHERE model_name = ? AND stage = ?",
                    (ModelStage.ARCHIVED.value, now, model_name, ModelStage.PRODUCTION.value),
                )
            conn.execute(
                """
                INSERT INTO model_versions
                (model_name, version, stage, artifact_path, checksum_sha256, metrics_json, tags_json, description, registered_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.model_name,
                    rec.version,
                    rec.stage.value,
                    rec.artifact_path,
                    rec.checksum_sha256,
                    json.dumps(rec.metrics),
                    json.dumps(rec.tags),
                    rec.description,
                    rec.registered_at,
                    rec.updated_at,
                ),
            )
            conn.commit()
        logger.info("model_registered", name=model_name, version=next_ver, stage=stage.value)
        return rec

    def transition_stage(
        self,
        model_name: str,
        version: int,
        target_stage: ModelStage,
        *,
        archive_existing_production: bool = True,
    ) -> RegisteredModelVersion:
        """Promote or transition a registered model version across lifecycle stages."""
        now = datetime.now(UTC).isoformat()
        with _sqlite_conn(self.db_path) as conn:
            cur = conn.execute(
                "SELECT 1 FROM model_versions WHERE model_name = ? AND version = ?",
                (model_name, version),
            )
            if cur.fetchone() is None:
                raise MLOpsError(f"Model '{model_name}' version {version} not found in registry.")
            if target_stage == ModelStage.PRODUCTION and archive_existing_production:
                conn.execute(
                    "UPDATE model_versions SET stage = ?, updated_at = ? WHERE model_name = ? AND stage = ?",
                    (ModelStage.ARCHIVED.value, now, model_name, ModelStage.PRODUCTION.value),
                )
            conn.execute(
                "UPDATE model_versions SET stage = ?, updated_at = ? WHERE model_name = ? AND version = ?",
                (target_stage.value, now, model_name, version),
            )
            conn.commit()
        logger.info("model_stage_transitioned", name=model_name, version=version, stage=target_stage.value)
        return self.get_version(model_name, version)

    def get_version(self, model_name: str, version: int) -> RegisteredModelVersion:
        """Fetch metadata for a specific model version."""
        with _sqlite_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT model_name, version, stage, artifact_path, checksum_sha256, metrics_json, tags_json, description, registered_at, updated_at FROM model_versions WHERE model_name = ? AND version = ?",
                (model_name, version),
            ).fetchone()
        if row is None:
            raise MLOpsError(f"Model '{model_name}' v{version} not found.")
        return self._row_to_model(row)

    def load_model(
        self,
        model_name: str,
        *,
        version: int | None = None,
        stage: ModelStage | None = None,
    ) -> tuple[BaseOmniForgeModel, RegisteredModelVersion]:
        """Load a serialized :class:`BaseOmniForgeModel` by explicit version, lifecycle stage, or latest."""
        versions = self.list_versions(model_name)
        if not versions:
            raise MLOpsError(f"No versions registered for model '{model_name}'.")
        chosen: RegisteredModelVersion | None = None
        if version is not None:
            chosen = next((v for v in versions if v.version == version), None)
        elif stage is not None:
            matching = [v for v in versions if v.stage == stage]
            chosen = matching[-1] if matching else None
        else:
            chosen = versions[-1]
        if chosen is None:
            raise MLOpsError(f"No matching version found for '{model_name}' (version={version}, stage={stage}).")
        return BaseOmniForgeModel.load(chosen.artifact_path), chosen

    def list_versions(self, model_name: str | None = None) -> list[RegisteredModelVersion]:
        """List registered model versions, optionally scoped to one model name."""
        with _sqlite_conn(self.db_path) as conn:
            if model_name:
                rows = conn.execute(
                    "SELECT model_name, version, stage, artifact_path, checksum_sha256, metrics_json, tags_json, description, registered_at, updated_at FROM model_versions WHERE model_name = ? ORDER BY version ASC",
                    (model_name,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT model_name, version, stage, artifact_path, checksum_sha256, metrics_json, tags_json, description, registered_at, updated_at FROM model_versions ORDER BY model_name ASC, version ASC"
                ).fetchall()
        return [self._row_to_model(r) for r in rows]

    @staticmethod
    def _row_to_model(row: tuple[Any, ...]) -> RegisteredModelVersion:
        return RegisteredModelVersion(
            model_name=row[0],
            version=int(row[1]),
            stage=ModelStage(row[2]),
            artifact_path=row[3],
            checksum_sha256=row[4],
            metrics=json.loads(row[5]),
            tags=json.loads(row[6]),
            description=row[7],
            registered_at=row[8],
            updated_at=row[9],
        )
