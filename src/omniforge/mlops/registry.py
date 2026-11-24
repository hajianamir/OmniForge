"""Versioned Enterprise Model Registry with Lifecycle Stages (None/Staging/Production/Archived)."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from omniforge.core.exceptions import MLOpsError, SecurityError
from omniforge.core.logging import get_logger
from omniforge.core.types import ModelStage
from omniforge.models.base import BaseOmniForgeModel, SklearnEstimatorAdapter

logger = get_logger(__name__)

_SAFE_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VALID_RISK_TIERS: frozenset[str] = frozenset({"low", "medium", "high", "critical"})


def _audit_registry_event(action: str, *, model_name: str, status: str, details: dict[str, Any]) -> None:
    """Best-effort tamper-evident audit logging for registry governance gates & rollbacks."""
    try:
        from omniforge.config.settings import get_settings
        from omniforge.core.security import AuditLogger

        settings = get_settings()
        AuditLogger(log_path=settings.paths.audit_log_path).log_event(
            action=action,
            resource_type="model_registry",
            resource_id=model_name,
            actor="system",
            tenant_id=str(getattr(settings, "default_tenant_id", "default-tenant")),
            status=status,
            details=details,
        )
    except Exception as exc:
        logger.error("registry_governance_audit_failed", action=action, model=model_name, error=str(exc))


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
    """Metadata and NIST AI RMF governance record for a single version of a registered model."""

    model_name: str
    version: int
    stage: ModelStage
    artifact_path: str
    checksum_sha256: str
    metrics: dict[str, float | None] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)
    description: str = ""
    owner: str = ""
    intended_use: str = ""
    approved_domains: list[str] = Field(default_factory=list)
    risk_tier: Literal["low", "medium", "high", "critical"] = "medium"
    fairness_passed: bool | None = None
    encrypted_at_rest: bool = False
    registered_at: str
    updated_at: str


class ModelRegistry:
    """SQLite-backed Model Registry supporting semantic versioning, NIST AI RMF governance, fairness gates, and rollback."""

    _SELECT_COLS = (
        "model_name, version, stage, artifact_path, checksum_sha256, metrics_json, tags_json, "
        "description, owner, intended_use, approved_domains_json, risk_tier, fairness_passed, "
        "encrypted_at_rest, registered_at, updated_at"
    )

    def __init__(
        self,
        db_path: str | Path = "./.omniforge_workspace/model_registry.db",
        artifact_dir: str | Path = "./.omniforge_workspace/artifacts/registry",
        *,
        require_fairness_gate: bool | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        env_gate = os.getenv("OMNIFORGE_REQUIRE_FAIRNESS_FOR_PRODUCTION", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.require_fairness_gate = env_gate if require_fairness_gate is None else bool(require_fairness_gate)
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
                    owner TEXT NOT NULL DEFAULT '',
                    intended_use TEXT NOT NULL DEFAULT '',
                    approved_domains_json TEXT NOT NULL DEFAULT '[]',
                    risk_tier TEXT NOT NULL DEFAULT 'medium',
                    fairness_passed INTEGER DEFAULT NULL,
                    encrypted_at_rest INTEGER DEFAULT 0,
                    registered_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (model_name, version)
                )
                """
            )
            existing_cols = {
                str(r[1]) for r in conn.execute("PRAGMA table_info(model_versions)").fetchall()
            }
            migrations = [
                ("owner", "TEXT NOT NULL DEFAULT ''"),
                ("intended_use", "TEXT NOT NULL DEFAULT ''"),
                ("approved_domains_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("risk_tier", "TEXT NOT NULL DEFAULT 'medium'"),
                ("fairness_passed", "INTEGER DEFAULT NULL"),
                ("encrypted_at_rest", "INTEGER DEFAULT 0"),
            ]
            for col_name, col_ddl in migrations:
                if col_name not in existing_cols:
                    conn.execute(f"ALTER TABLE model_versions ADD COLUMN {col_name} {col_ddl}")
            conn.commit()

    def _enforce_fairness_gate(
        self,
        *,
        model_name: str,
        version: int,
        fairness_passed: bool | None,
        require_fairness_gate: bool | None,
        override_reason: str | None,
        tags: dict[str, str],
    ) -> dict[str, str]:
        """Enforce Responsible AI fairness promotion gate when transitioning to PRODUCTION (NIST AI RMF MANAGE 2.2)."""
        eff_require = self.require_fairness_gate if require_fairness_gate is None else bool(require_fairness_gate)
        failed_audit = fairness_passed is False
        missing_required_audit = eff_require and fairness_passed is None
        if not failed_audit and not missing_required_audit:
            return tags

        clean_override = (override_reason or "").strip()
        if not clean_override:
            reason_msg = (
                f"Model '{model_name}' v{version} failed its Responsible AI fairness audit (fairness_passed=False)"
                if failed_audit
                else f"Model '{model_name}' v{version} has no completed fairness audit (require_fairness_gate=True)"
            )
            _audit_registry_event(
                "registry:fairness_gate_blocked",
                model_name=model_name,
                status="blocked",
                details={"version": version, "fairness_passed": fairness_passed, "require_fairness_gate": eff_require},
            )
            raise MLOpsError(
                f"Production promotion blocked by Responsible AI fairness gate: {reason_msg}. "
                "Remediate group disparities or supply an explicit auditable override_reason."
            )

        updated_tags = dict(tags)
        updated_tags["fairness_gate_override"] = clean_override
        _audit_registry_event(
            "registry:fairness_gate_override",
            model_name=model_name,
            status="warning",
            details={
                "version": version,
                "fairness_passed": fairness_passed,
                "override_reason": clean_override,
            },
        )
        return updated_tags

    def register_model(
        self,
        model: BaseOmniForgeModel | Any,
        model_name: str,
        *,
        metrics: dict[str, float] | None = None,
        tags: dict[str, str] | None = None,
        description: str = "",
        owner: str = "",
        intended_use: str = "",
        approved_domains: list[str] | None = None,
        risk_tier: str = "medium",
        fairness_report: Any | None = None,
        fairness_passed: bool | None = None,
        require_fairness_gate: bool | None = None,
        override_reason: str | None = None,
        stage: ModelStage = ModelStage.NONE,
        encrypt: bool = False,
    ) -> RegisteredModelVersion:
        """Persist a fitted OmniForge (or wrapped sklearn) model with governance metadata and optional fairness gate."""
        clean_name = str(model_name or "").strip()
        if not _SAFE_MODEL_NAME_RE.match(clean_name) or ".." in clean_name:
            raise SecurityError(
                f"Invalid model_name {model_name!r}: must match ^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$ without '..'."
            )
        norm_tier = str(risk_tier or "medium").strip().lower()
        if norm_tier not in _VALID_RISK_TIERS:
            raise MLOpsError(f"Invalid risk_tier '{risk_tier}'. Choose one of: {sorted(_VALID_RISK_TIERS)}.")
        eff_fairness: bool | None = fairness_passed
        extra_fair_metrics: dict[str, float | None] = {}
        if fairness_report is not None:
            if hasattr(fairness_report, "passed"):
                eff_fairness = bool(fairness_report.passed)
                extra_fair_metrics["disparate_impact_ratio"] = float(fairness_report.disparate_impact_ratio)
                eod = fairness_report.equalized_odds_diff
                extra_fair_metrics["equalized_odds_diff"] = float(eod) if eod is not None else None
            elif isinstance(fairness_report, dict) and "passed" in fairness_report:
                eff_fairness = bool(fairness_report["passed"])

        model_name = clean_name
        if not isinstance(model, BaseOmniForgeModel):
            if not hasattr(model, "predict"):
                raise MLOpsError(f"Cannot register object of type {type(model).__name__} without a predict() method.")
            model = SklearnEstimatorAdapter(model, name=model_name)
        with _sqlite_conn(self.db_path) as conn:
            row = conn.execute(
                "SELECT MAX(version) FROM model_versions WHERE model_name = ?", (model_name,)
            ).fetchone()
            next_ver = int(row[0] or 0) + 1

        eff_tags = dict(tags or {})
        if stage == ModelStage.PRODUCTION:
            eff_tags = self._enforce_fairness_gate(
                model_name=model_name,
                version=next_ver,
                fairness_passed=eff_fairness,
                require_fairness_gate=require_fairness_gate,
                override_reason=override_reason,
                tags=eff_tags,
            )

        root_resolved = self.artifact_dir.resolve()
        dest = (self.artifact_dir / model_name / f"v{next_ver}" / "model.joblib").resolve()
        if not dest.is_relative_to(root_resolved):
            raise SecurityError(f"Artifact path escapes registry root for model {model_name!r}.")
        saved_path = model.save(dest, encrypt=encrypt)
        now = datetime.now(UTC).isoformat()
        merged_metrics = {
            **{k: float(v) for k, v in model.metadata_.metrics.items() if isinstance(v, (int, float))},
            **extra_fair_metrics,
            **{k: float(v) for k, v in (metrics or {}).items() if isinstance(v, (int, float))},
        }
        rec = RegisteredModelVersion(
            model_name=model_name,
            version=next_ver,
            stage=stage,
            artifact_path=str(saved_path.resolve()),
            checksum_sha256=model.metadata_.artifact_checksum,
            metrics=merged_metrics,
            tags=eff_tags,
            description=description,
            owner=str(owner).strip(),
            intended_use=str(intended_use).strip(),
            approved_domains=[str(d).strip() for d in (approved_domains or []) if str(d).strip()],
            risk_tier=norm_tier,  # type: ignore[arg-type]
            fairness_passed=eff_fairness,
            encrypted_at_rest=encrypt,
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
                (model_name, version, stage, artifact_path, checksum_sha256, metrics_json, tags_json,
                 description, owner, intended_use, approved_domains_json, risk_tier, fairness_passed,
                 encrypted_at_rest, registered_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    rec.owner,
                    rec.intended_use,
                    json.dumps(rec.approved_domains),
                    rec.risk_tier,
                    None if rec.fairness_passed is None else int(rec.fairness_passed),
                    int(rec.encrypted_at_rest),
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
        fairness_report: Any | None = None,
        require_fairness_gate: bool | None = None,
        override_reason: str | None = None,
    ) -> RegisteredModelVersion:
        """Promote or transition a registered model version across lifecycle stages with Responsible AI gate."""
        current = self.get_version(model_name, version)
        eff_fairness = current.fairness_passed
        if fairness_report is not None:
            if hasattr(fairness_report, "passed"):
                eff_fairness = bool(fairness_report.passed)
            elif isinstance(fairness_report, dict) and "passed" in fairness_report:
                eff_fairness = bool(fairness_report["passed"])

        eff_tags = dict(current.tags)
        if target_stage == ModelStage.PRODUCTION:
            eff_tags = self._enforce_fairness_gate(
                model_name=model_name,
                version=version,
                fairness_passed=eff_fairness,
                require_fairness_gate=require_fairness_gate,
                override_reason=override_reason,
                tags=eff_tags,
            )

        now = datetime.now(UTC).isoformat()
        with _sqlite_conn(self.db_path) as conn:
            if target_stage == ModelStage.PRODUCTION and archive_existing_production:
                conn.execute(
                    "UPDATE model_versions SET stage = ?, updated_at = ? WHERE model_name = ? AND stage = ?",
                    (ModelStage.ARCHIVED.value, now, model_name, ModelStage.PRODUCTION.value),
                )
            conn.execute(
                "UPDATE model_versions SET stage = ?, tags_json = ?, fairness_passed = ?, updated_at = ? "
                "WHERE model_name = ? AND version = ?",
                (
                    target_stage.value,
                    json.dumps(eff_tags),
                    None if eff_fairness is None else int(eff_fairness),
                    now,
                    model_name,
                    version,
                ),
            )
            conn.commit()
        logger.info("model_stage_transitioned", name=model_name, version=version, stage=target_stage.value)
        return self.get_version(model_name, version)

    def rollback_production(
        self,
        model_name: str,
        *,
        reason: str = "Automated or operator-initiated SLA degradation rollback",
        target_version: int | None = None,
    ) -> RegisteredModelVersion:
        """Atomically demote failing PRODUCTION version and restore the prior healthy version (NIST AI RMF MANAGE 2.4)."""
        versions = self.list_versions(model_name)
        if not versions:
            raise MLOpsError(f"No versions registered for model '{model_name}'.")
        prod_versions = [v for v in versions if v.stage == ModelStage.PRODUCTION]
        current_prod = prod_versions[-1] if prod_versions else None

        candidate: RegisteredModelVersion | None = None
        if target_version is not None:
            candidate = next((v for v in versions if v.version == target_version), None)
            if candidate is None:
                raise MLOpsError(f"Rollback target version {target_version} not found for '{model_name}'.")
        else:
            prior = [
                v
                for v in versions
                if (current_prod is None or v.version != current_prod.version)
                and v.fairness_passed is not False
            ]
            # Prefer previously deployed ARCHIVED or STAGING versions in reverse chronological order
            preferred = [v for v in prior if v.stage in (ModelStage.ARCHIVED, ModelStage.STAGING)]
            pool = preferred or prior
            candidate = pool[-1] if pool else None

        if candidate is None:
            raise MLOpsError(
                f"Cannot rollback model '{model_name}': no eligible prior version exists."
            )

        now = datetime.now(UTC).isoformat()
        cand_tags = dict(candidate.tags)
        cand_tags["rollback_restored_at"] = now
        cand_tags["rollback_reason"] = str(reason).strip() or "SLA rollback"

        with _sqlite_conn(self.db_path) as conn:
            conn.execute(
                "UPDATE model_versions SET stage = ?, updated_at = ? WHERE model_name = ? AND stage = ?",
                (ModelStage.ARCHIVED.value, now, model_name, ModelStage.PRODUCTION.value),
            )
            conn.execute(
                "UPDATE model_versions SET stage = ?, tags_json = ?, updated_at = ? WHERE model_name = ? AND version = ?",
                (ModelStage.PRODUCTION.value, json.dumps(cand_tags), now, model_name, candidate.version),
            )
            conn.commit()

        _audit_registry_event(
            "registry:production_rollback",
            model_name=model_name,
            status="rolled_back",
            details={
                "demoted_version": current_prod.version if current_prod else None,
                "restored_version": candidate.version,
                "reason": reason,
            },
        )
        logger.warning(
            "model_production_rolled_back",
            model=model_name,
            from_version=current_prod.version if current_prod else None,
            to_version=candidate.version,
            reason=reason,
        )
        return self.get_version(model_name, candidate.version)

    def get_version(self, model_name: str, version: int) -> RegisteredModelVersion:
        """Fetch metadata for a specific model version."""
        with _sqlite_conn(self.db_path) as conn:
            row = conn.execute(
                f"SELECT {self._SELECT_COLS} FROM model_versions WHERE model_name = ? AND version = ?",
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
        return BaseOmniForgeModel.load(chosen.artifact_path, encrypted=chosen.encrypted_at_rest), chosen

    def list_versions(self, model_name: str | None = None) -> list[RegisteredModelVersion]:
        """List registered model versions, optionally scoped to one model name."""
        with _sqlite_conn(self.db_path) as conn:
            if model_name:
                rows = conn.execute(
                    f"SELECT {self._SELECT_COLS} FROM model_versions WHERE model_name = ? ORDER BY version ASC",
                    (model_name,),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT {self._SELECT_COLS} FROM model_versions ORDER BY model_name ASC, version ASC"
                ).fetchall()
        return [self._row_to_model(r) for r in rows]

    @staticmethod
    def _row_to_model(row: tuple[Any, ...]) -> RegisteredModelVersion:
        raw_tier = str(row[11] or "medium").lower()
        tier_val = raw_tier if raw_tier in _VALID_RISK_TIERS else "medium"
        return RegisteredModelVersion(
            model_name=row[0],
            version=int(row[1]),
            stage=ModelStage(row[2]),
            artifact_path=row[3],
            checksum_sha256=row[4],
            metrics=json.loads(row[5]),
            tags=json.loads(row[6]),
            description=row[7],
            owner=str(row[8] or ""),
            intended_use=str(row[9] or ""),
            approved_domains=list(json.loads(row[10] or "[]")),
            risk_tier=tier_val,  # type: ignore[arg-type]
            fairness_passed=None if row[12] is None else bool(int(row[12])),
            encrypted_at_rest=bool(int(row[13])) if row[13] is not None else False,
            registered_at=row[14],
            updated_at=row[15],
        )
