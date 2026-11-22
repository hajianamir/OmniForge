"""Full Workspace Project Save & Reload Engine (.uap portable archive with SHA-256 integrity verification)."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pydantic import BaseModel, Field

from omniforge.core.exceptions import DataValidationError, SecurityError
from omniforge.core.logging import get_logger
from omniforge.core.security import ArtifactSigner, safe_joblib_load_bytes
from omniforge.version import __version__

logger = get_logger(__name__)

PROJECT_FILE_EXTENSION = ".omniforge"
LEGACY_PROJECT_EXTENSIONS: tuple[str, ...] = (".omniforge", ".uap")
MANIFEST_FILENAME = "manifest.json"
STATE_FILENAME = "session_state.joblib"
FORMAT_VERSION = "2.0"
_WORKSPACE_HMAC_CONTEXT = b"omniforge:workspace:v2"
_MAX_ARCHIVE_ENTRIES = 64
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024  # 4 MiB ceiling for manifest.json stream
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 512 * 1024 * 1024  # 512 MiB zip-bomb guard
_STREAM_READ_CHUNK_BYTES = 64 * 1024
_ALLOWED_ZIP_COMPRESS_TYPES: frozenset[int] = frozenset(
    {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
)

# Keys that are runtime-only, non-serializable clients, or UI upload widget transients
_EXCLUDED_EXACT_KEYS: frozenset[str] = frozenset(
    {
        "uap_copilot",
        "uap_copilot_key",
        "uap_serving_engine",
        "uap_upload_sig",
        "uap_uploader_nonce",
        "uap_project_uploader_nonce",
        "uap_loaded_project_sig",
        "uap_last_project_notice",
        "_uap_pending_restore_state",
        "_uap_pending_restore_manifest",
    }
)

# Prefixes of user-entered widget configurations & analytical inputs to preserve across sessions
_PERSISTED_KEY_PREFIXES: tuple[str, ...] = (
    "uap_",
    "crit_",
    "split_lock_",
    "automl_",
    "hub_",
    "rep_sec_",
    "rep_opt_",
    "dl_",
    "clust_studio_",
    "anom_studio_",
    "fs_",
    "future_",
    "pd_",
)

# Widget suffixes / patterns that must never be restored as saved session inputs (buttons, uploaders, forms)
_EXCLUDED_KEY_PATTERNS: tuple[str, ...] = (
    "_btn",
    "btn_",
    "_uploader",
    "file_uploader",
    "FormSubmitter:",
)


def _sanitize_project_filename(project_name: str) -> str:
    """Convert any human project name into a clean, filesystem-safe filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (project_name or "").strip()).strip("._-")
    return cleaned or f"omniforge_project_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"


def _should_persist_key(key: str, value: Any) -> bool:
    """Return True if a session_state entry represents workspace data, results, or user inputs."""
    if not isinstance(key, str) or key in _EXCLUDED_EXACT_KEYS or key.startswith("_"):
        return False
    if any(pat in key for pat in _EXCLUDED_KEY_PATTERNS):
        return False
    if not key.startswith(_PERSISTED_KEY_PREFIXES):
        return False
    # Never persist callables or transient uploaded file stream objects
    if callable(value):
        return False
    cls_name = type(value).__name__
    if "UploadedFile" in cls_name:
        return False
    return True


def _detect_completed_stages(state: Mapping[str, Any]) -> list[str]:
    """Derive human-readable completed CRISP-DM workflow stages from workspace state."""
    stages: list[str] = []
    if state.get("uap_df") is not None:
        stages.append("Data Ingested")
    if state.get("uap_quality") is not None:
        stages.append("6D Quality Profiled")
    if state.get("uap_clean_report") is not None:
        stages.append("Data Cleaned")
    if state.get("uap_validation_report") is not None:
        stages.append("Rules Validated")
    split_lock = state.get("uap_split_lock")
    if isinstance(split_lock, dict) and split_lock.get("active"):
        stages.append(f"Split Locked ({str(split_lock.get('strategy', 'auto')).upper()})")
    if state.get("uap_clustering_result") is not None:
        stages.append("Clustering Completed")
    if state.get("uap_anomaly_df") is not None:
        stages.append("Anomaly Screened")
    if state.get("uap_diag_report") is not None:
        stages.append("Statistical Diagnostics")
    if state.get("uap_baseline_results") is not None:
        stages.append("Baselines Benchmarked")
    if state.get("uap_risk_report") is not None:
        stages.append("Risk & Uncertainty Audited")
    if state.get("uap_eda_report") is not None:
        stages.append("Automated EDA")
    if state.get("uap_feature_report") is not None or state.get("uap_fs_selected_cols") is not None:
        stages.append("Features Engineered / Selected")
    if state.get("uap_active_model") is not None or state.get("uap_target_models"):
        stages.append("Models Trained")
    if state.get("uap_model_equations"):
        stages.append("Equations Extracted")
    if state.get("uap_future_prediction_result") is not None:
        stages.append("Future Predictions Scored")
    if state.get("uap_shap_result") is not None or state.get("uap_model_card") is not None:
        stages.append("Responsible AI / Governance")
    if any(state.get(k) is not None for k in ("uap_forecast_report", "uap_causal_result", "uap_opt_result", "uap_sim_result")):
        stages.append("Decision Intelligence")
    if state.get("uap_report_payload") is not None:
        stages.append("Executive Report Compiled")
    return stages


class WorkspaceManifest(BaseModel):
    """Human-readable metadata & integrity manifest embedded inside every ``.omniforge`` project archive."""

    format_version: str = FORMAT_VERSION
    uap_version: str = __version__
    project_name: str = "omniforge_project"
    notes: str = ""
    saved_at_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    active_studio: str = ""
    theme: str = "dark"
    dataset_name: str | None = None
    dataset_source: str | None = None
    dataset_shape: tuple[int, int] = (0, 0)
    raw_dataset_shape: tuple[int, int] = (0, 0)
    processing_mode: str = "RAW"
    target_columns: list[str] = Field(default_factory=list)
    split_lock_active: bool = False
    split_strategy: str | None = None
    trained_models: list[str] = Field(default_factory=list)
    completed_stages: list[str] = Field(default_factory=list)
    lineage_count: int = 0
    exclusion_count: int = 0
    copilot_turns: int = 0
    saved_keys: list[str] = Field(default_factory=list)
    skipped_keys: list[dict[str, str]] = Field(default_factory=list)
    payload_sha256: str = ""
    payload_hmac_sha256: str = ""
    payload_size_bytes: int = 0


class WorkspaceSnapshotSummary(BaseModel):
    """Summary entry for a locally stored ``.uap`` workspace project file."""

    path: str
    file_name: str
    file_size_bytes: int
    manifest: WorkspaceManifest


class WorkspaceSessionManager:
    """Save, inspect, validate, list, and reload complete OmniForge workspace sessions in portable ``.uap`` files.

    A ``.uap`` file is a compressed, integrity-verified archive containing:
        - ``manifest.json``: readable metadata, saved stage inventory, and SHA-256 payload checksum.
        - ``session_state.joblib``: full serialized state covering raw & processed datasets, split lock,
          cleaning/quality/validation reports, clustering & anomaly artifacts, statistical tests, EDA,
          feature pipelines, trained AutoML/Deep Learning/Baseline models, extracted equations, future
          predictions, decision-engine outputs, exclusions, lineage audit log, copilot history, and
          all user criteria/configuration inputs.
    """

    DEFAULT_PROJECTS_DIR = Path(".omniforge_workspace") / "projects"

    @classmethod
    def export_to_bytes(
        cls,
        session_state: Mapping[str, Any],
        *,
        project_name: str = "omniforge_project",
        notes: str = "",
    ) -> tuple[bytes, WorkspaceManifest]:
        """Serialize the entire workspace state into an integrity-verified ``.uap`` byte archive."""
        serializable: dict[str, Any] = {}
        skipped: list[dict[str, str]] = []

        for key in sorted(session_state.keys()):
            val = session_state[key]
            if not _should_persist_key(key, val):
                continue
            try:
                probe = io.BytesIO()
                joblib.dump(val, probe, compress=0)
                serializable[key] = val
            except Exception as exc:
                logger.warning("workspace_session_key_skipped", key=key, error=str(exc))
                skipped.append({"key": key, "reason": str(exc)[:200]})

        payload_buf = io.BytesIO()
        joblib.dump(serializable, payload_buf, compress=("zlib", 3))
        payload_bytes = payload_buf.getvalue()
        payload_sha = hashlib.sha256(payload_bytes).hexdigest()
        payload_hmac = ArtifactSigner().sign_bytes(payload_bytes, context=_WORKSPACE_HMAC_CONTEXT)

        df = serializable.get("uap_df")
        raw_df = serializable.get("uap_raw_df")
        ds_shape = (int(len(df)), int(len(df.columns))) if isinstance(df, pd.DataFrame) else (0, 0)
        raw_shape = (int(len(raw_df)), int(len(raw_df.columns))) if isinstance(raw_df, pd.DataFrame) else (0, 0)

        targets_raw = serializable.get("uap_targets") or []
        if not targets_raw and serializable.get("uap_target"):
            targets_raw = [str(serializable["uap_target"])]
        targets = [str(t) for t in targets_raw]

        split_lock = serializable.get("uap_split_lock")
        split_active = bool(isinstance(split_lock, dict) and split_lock.get("active"))
        split_strat = str(split_lock.get("strategy")) if split_active and isinstance(split_lock, dict) else None

        trained_models: list[str] = []
        active_m_name = serializable.get("uap_active_model_name")
        if active_m_name:
            trained_models.append(str(active_m_name))
        target_automls = serializable.get("uap_target_automls") or {}
        if isinstance(target_automls, dict):
            for t_name, a_res in target_automls.items():
                best_n = getattr(a_res, "best_model_name", None)
                if best_n:
                    label = f"{t_name}: {best_n}"
                    if label not in trained_models:
                        trained_models.append(label)

        lineage = serializable.get("uap_lineage") or []
        exclusions = serializable.get("uap_exclusions") or []
        copilot_hist = serializable.get("uap_copilot_history") or serializable.get("uap_chat_history") or []

        # Determine processing mode
        mutation_stages = {
            "Clean",
            "Cleaning",
            "Clustering",
            "Clustering Cleaning",
            "Anomaly Cleaning",
            "Anomaly Screening",
            "Features",
            "Feature Selection",
            "Embeddings",
            "Version Restore",
        }
        applied: list[str] = []
        for entry in lineage:
            if not isinstance(entry, dict):
                continue
            stg = str(entry.get("stage", ""))
            if stg in {"Ingest", "Reset"}:
                applied = []
            elif stg == "Undo" and applied:
                applied.pop()
            elif stg in mutation_stages:
                applied.append(stg)

        is_processed = bool(applied) or (raw_shape != (0, 0) and ds_shape != raw_shape)

        manifest = WorkspaceManifest(
            format_version=FORMAT_VERSION,
            uap_version=__version__,
            project_name=(project_name or "").strip() or str(serializable.get("uap_dataset_name") or "omniforge_project"),
            notes=(notes or "").strip(),
            active_studio=str(serializable.get("uap_selected_studio") or serializable.get("uap_active_studio") or ""),
            theme=str(serializable.get("uap_theme_radio") or serializable.get("uap_theme") or "dark").lower(),
            dataset_name=str(serializable["uap_dataset_name"]) if serializable.get("uap_dataset_name") else None,
            dataset_source=str(serializable["uap_dataset_source"]) if serializable.get("uap_dataset_source") else None,
            dataset_shape=ds_shape,
            raw_dataset_shape=raw_shape,
            processing_mode="PROCESSED" if is_processed else "RAW",
            target_columns=targets,
            split_lock_active=split_active,
            split_strategy=split_strat,
            trained_models=trained_models,
            completed_stages=_detect_completed_stages(serializable),
            lineage_count=len(lineage) if isinstance(lineage, list) else 0,
            exclusion_count=len(exclusions) if isinstance(exclusions, list) else 0,
            copilot_turns=len(copilot_hist) if isinstance(copilot_hist, list) else 0,
            saved_keys=sorted(serializable.keys()),
            skipped_keys=skipped,
            payload_sha256=payload_sha,
            payload_hmac_sha256=payload_hmac,
            payload_size_bytes=len(payload_bytes),
        )

        archive_buf = io.BytesIO()
        with zipfile.ZipFile(archive_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(MANIFEST_FILENAME, manifest.model_dump_json(indent=2).encode("utf-8"))
            zf.writestr(STATE_FILENAME, payload_bytes)

        logger.info(
            "workspace_project_exported",
            project=manifest.project_name,
            keys=len(manifest.saved_keys),
            skipped=len(manifest.skipped_keys),
            size_bytes=archive_buf.tell(),
        )
        return archive_buf.getvalue(), manifest

    @classmethod
    def save_to_file(
        cls,
        session_state: Mapping[str, Any],
        output_path: str | Path,
        *,
        project_name: str | None = None,
        notes: str = "",
    ) -> tuple[Path, WorkspaceManifest]:
        """Write a complete ``.uap`` workspace snapshot directly to ``output_path``."""
        dest = Path(output_path)
        if dest.suffix.lower() != PROJECT_FILE_EXTENSION:
            dest = dest.with_suffix(PROJECT_FILE_EXTENSION)
        dest.parent.mkdir(parents=True, exist_ok=True)
        resolved_name = project_name or dest.stem
        archive_bytes, manifest = cls.export_to_bytes(session_state, project_name=resolved_name, notes=notes)
        dest.write_bytes(archive_bytes)
        return dest, manifest

    @classmethod
    def save_local_project(
        cls,
        session_state: Mapping[str, Any],
        project_name: str,
        *,
        notes: str = "",
        projects_dir: str | Path | None = None,
    ) -> tuple[Path, WorkspaceManifest]:
        """Save a named workspace project snapshot under the local projects directory for 1-click resume."""
        target_dir = Path(projects_dir) if projects_dir is not None else cls.DEFAULT_PROJECTS_DIR
        stem = _sanitize_project_filename(project_name)
        return cls.save_to_file(
            session_state,
            target_dir / f"{stem}{PROJECT_FILE_EXTENSION}",
            project_name=project_name.strip() or stem,
            notes=notes,
        )

    @staticmethod
    def _read_zip_member_bounded(zf: zipfile.ZipFile, member_name: str, *, max_bytes: int) -> bytes:
        """Stream-decompress a ZIP member enforcing ``max_bytes`` on actual decompressed output (CWE-409)."""
        chunks: list[bytes] = []
        total = 0
        read_step = max(1, min(_STREAM_READ_CHUNK_BYTES, max_bytes + 1))
        with zf.open(member_name, mode="r") as stream:
            while True:
                chunk = stream.read(read_step)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise DataValidationError(
                        f"Invalid OmniForge project archive: decompressed stream for '{member_name}' "
                        f"exceeds safety limit ({max_bytes} bytes).",
                        suggestions=["Ensure the workspace archive is legitimate and within size limits."],
                    )
                chunks.append(chunk)
        return b"".join(chunks)

    @classmethod
    def _validate_archive_structure_and_manifest(
        cls,
        zf: zipfile.ZipFile,
        *,
        max_uncompressed_bytes: int = _MAX_ARCHIVE_UNCOMPRESSED_BYTES,
    ) -> tuple[WorkspaceManifest, int]:
        """Validate ZIP entry count, compression types, member hygiene, header sizes, and manifest JSON."""
        infos = zf.infolist()
        if len(infos) > _MAX_ARCHIVE_ENTRIES:
            raise DataValidationError(
                "Invalid OmniForge project archive: excessive ZIP entry count.",
                suggestions=["Only upload archives created by OmniForge Project Save."],
            )
        seen_names: set[str] = set()
        for info in infos:
            fname = info.filename
            if (
                not fname
                or "\x00" in fname
                or fname.startswith(("/", "\\"))
                or "\\" in fname
                or ".." in fname.split("/")
            ):
                raise DataValidationError(
                    f"Invalid OmniForge project archive: unsafe ZIP member path '{fname}'.",
                    suggestions=["Only upload archives created by OmniForge Project Save."],
                )
            if fname in seen_names:
                raise DataValidationError(
                    f"Invalid OmniForge project archive: duplicate ZIP member '{fname}'.",
                    suggestions=["Only upload unmodified archives created by OmniForge Project Save."],
                )
            if info.compress_type not in _ALLOWED_ZIP_COMPRESS_TYPES:
                raise DataValidationError(
                    f"Invalid OmniForge project archive: unsupported ZIP compression method ({info.compress_type}).",
                    suggestions=["Only upload standard DEFLATE/STORED archives created by OmniForge Project Save."],
                )
            seen_names.add(fname)

        total_uncompressed = sum(max(0, int(info.file_size)) for info in infos)
        if total_uncompressed > max_uncompressed_bytes:
            raise DataValidationError(
                "Invalid OmniForge project archive: uncompressed payload exceeds safety limit.",
                suggestions=["Ensure the workspace archive is legitimate and within size limits."],
            )
        if MANIFEST_FILENAME not in seen_names or STATE_FILENAME not in seen_names:
            raise DataValidationError(
                "Invalid OmniForge project archive: missing required manifest or state payload.",
                suggestions=["Ensure the uploaded file was saved by OmniForge Project Save (.omniforge)."],
            )
        manifest_cap = min(_MAX_MANIFEST_BYTES, max_uncompressed_bytes)
        manifest_bytes = cls._read_zip_member_bounded(zf, MANIFEST_FILENAME, max_bytes=manifest_cap)
        raw_manifest = json.loads(manifest_bytes.decode("utf-8"))
        return WorkspaceManifest.model_validate(raw_manifest), len(manifest_bytes)

    @classmethod
    def inspect_bytes(
        cls,
        data: bytes,
        *,
        max_uncompressed_bytes: int = _MAX_ARCHIVE_UNCOMPRESSED_BYTES,
    ) -> WorkspaceManifest:
        """Read and validate the manifest of a ``.omniforge`` project archive without restoring it."""
        if not data:
            raise DataValidationError(
                "Project file is empty.",
                suggestions=["Select a valid saved .omniforge workspace project file."],
            )
        if len(data) > max_uncompressed_bytes:
            raise DataValidationError(
                "Invalid OmniForge project archive: archive size exceeds safety limit.",
                suggestions=["Ensure the workspace archive is legitimate and within size limits."],
            )
        try:
            with zipfile.ZipFile(io.BytesIO(data), mode="r") as zf:
                manifest, _ = cls._validate_archive_structure_and_manifest(
                    zf,
                    max_uncompressed_bytes=max_uncompressed_bytes,
                )
                return manifest
        except DataValidationError:
            raise
        except Exception as exc:
            raise DataValidationError(
                f"Could not inspect OmniForge project file: {exc}",
                suggestions=["Verify the file is a valid, uncorrupted .omniforge project archive."],
                cause=exc,
            ) from exc

    @classmethod
    def inspect_file(cls, path: str | Path) -> WorkspaceManifest:
        """Read and validate the manifest of a ``.omniforge`` project file on disk."""
        src = Path(path)
        if not src.exists():
            raise DataValidationError(f"Workspace project file not found: {src}")
        if src.stat().st_size > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise DataValidationError(
                "Workspace project file exceeds maximum archive size limit.",
                suggestions=["Ensure the workspace archive is legitimate and within size limits."],
            )
        return cls.inspect_bytes(src.read_bytes())

    @classmethod
    def load_from_bytes(
        cls,
        data: bytes,
        *,
        require_signature: bool = True,
        max_uncompressed_bytes: int = _MAX_ARCHIVE_UNCOMPRESSED_BYTES,
    ) -> tuple[dict[str, Any], WorkspaceManifest]:
        """Verify SHA-256 + HMAC-SHA256 signature and safely deserialize a ``.omniforge`` archive."""
        if not data:
            raise DataValidationError(
                "Project file is empty.",
                suggestions=["Select a valid saved .omniforge workspace project file."],
            )
        if len(data) > max_uncompressed_bytes:
            raise DataValidationError(
                "Invalid OmniForge project archive: archive size exceeds safety limit.",
                suggestions=["Ensure the workspace archive is legitimate and within size limits."],
            )
        try:
            with zipfile.ZipFile(io.BytesIO(data), mode="r") as zf:
                manifest, manifest_len = cls._validate_archive_structure_and_manifest(
                    zf,
                    max_uncompressed_bytes=max_uncompressed_bytes,
                )
                remaining_budget = max_uncompressed_bytes - manifest_len
                if remaining_budget <= 0:
                    raise DataValidationError(
                        "Invalid OmniForge project archive: uncompressed payload exceeds safety limit.",
                        suggestions=["Ensure the workspace archive is legitimate and within size limits."],
                    )
                payload_bytes = cls._read_zip_member_bounded(
                    zf,
                    STATE_FILENAME,
                    max_bytes=remaining_budget,
                )
            actual_sha = hashlib.sha256(payload_bytes).hexdigest()
            if manifest.payload_sha256 and actual_sha != manifest.payload_sha256:
                raise DataValidationError(
                    "Workspace project integrity check failed (SHA-256 mismatch — file may be corrupted).",
                    context={"expected_sha256": manifest.payload_sha256, "actual_sha256": actual_sha},
                    suggestions=["Re-export the .omniforge project file or choose an intact backup snapshot."],
                )
            if require_signature:
                signer = ArtifactSigner()
                if not manifest.payload_hmac_sha256 or not signer.verify_bytes(
                    payload_bytes,
                    manifest.payload_hmac_sha256,
                    context=_WORKSPACE_HMAC_CONTEXT,
                ):
                    raise SecurityError(
                        "Workspace archive failed cryptographic HMAC-SHA256 signature verification. "
                        "Refusing to deserialize untrusted or externally modified project archive.",
                        suggestions=[
                            "Only load .omniforge project files signed by this OmniForge instance "
                            "or configure a shared OMNIFORGE_ARTIFACT_SIGNING_KEY across trusted instances."
                        ],
                    )
            loaded = safe_joblib_load_bytes(payload_bytes, filename=STATE_FILENAME)
            if not isinstance(loaded, dict):
                raise DataValidationError("Deserialized project payload is not a valid session state mapping.")
            logger.info(
                "workspace_project_loaded",
                project=manifest.project_name,
                keys=len(loaded),
                dataset=manifest.dataset_name,
            )
            return loaded, manifest
        except (DataValidationError, SecurityError):
            raise
        except Exception as exc:
            raise DataValidationError(
                f"Failed to restore OmniForge project archive: {exc}",
                suggestions=["Ensure the file matches this OmniForge Python environment and was not truncated."],
                cause=exc,
            ) from exc

    @classmethod
    def load_from_file(cls, path: str | Path) -> tuple[dict[str, Any], WorkspaceManifest]:
        """Verify and load a ``.uap`` workspace project file from disk."""
        src = Path(path)
        if not src.exists():
            raise DataValidationError(f"Workspace project file not found: {src}")
        if src.stat().st_size > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise DataValidationError(
                "Workspace project file exceeds maximum archive size limit.",
                suggestions=["Ensure the workspace archive is legitimate and within size limits."],
            )
        return cls.load_from_bytes(src.read_bytes())

    @classmethod
    def list_local_projects(cls, projects_dir: str | Path | None = None) -> list[WorkspaceSnapshotSummary]:
        """List all valid ``.omniforge`` (and legacy ``.uap``) project snapshots in ``projects_dir``, newest first."""
        target_dir = Path(projects_dir) if projects_dir is not None else cls.DEFAULT_PROJECTS_DIR
        if not target_dir.exists():
            return []
        summaries: list[WorkspaceSnapshotSummary] = []
        paths = [p for ext in LEGACY_PROJECT_EXTENSIONS for p in target_dir.glob(f"*{ext}")]
        for path in sorted(set(paths)):
            try:
                manifest = cls.inspect_file(path)
                summaries.append(
                    WorkspaceSnapshotSummary(
                        path=str(path),
                        file_name=path.name,
                        file_size_bytes=path.stat().st_size,
                        manifest=manifest,
                    )
                )
            except Exception as exc:
                logger.warning("local_project_manifest_unreadable", path=str(path), error=str(exc))
        summaries.sort(key=lambda s: s.manifest.saved_at_utc, reverse=True)
        return summaries

    @classmethod
    def delete_local_project(cls, path: str | Path) -> bool:
        """Delete a local ``.omniforge`` or legacy ``.uap`` project snapshot file if it exists."""
        target = Path(path)
        if (
            not target.is_symlink()
            and target.exists()
            and target.is_file()
            and target.suffix.lower() in LEGACY_PROJECT_EXTENSIONS
        ):
            target.unlink()
            return True
        return False
