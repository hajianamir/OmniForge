"""Content-addressed dataset versioning, lineage tracking, diffing, and DVC manifest export."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from omniforge.core.exceptions import DataError
from omniforge.core.logging import get_logger
from omniforge.core.utils import compute_dataframe_checksum

logger = get_logger(__name__)

_SAFE_DATASET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _validate_version_identifier(value: str, label: str) -> str:
    """Validate dataset names, version IDs, and tags against CWE-22 path traversal tokens."""
    cleaned = str(value).strip()
    if not cleaned or ".." in cleaned or not _SAFE_DATASET_ID_RE.fullmatch(cleaned):
        raise DataError(
            f"Invalid {label} '{value}': must match ^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$ without '..' or path separators."
        )
    return cleaned


class DatasetVersion(BaseModel):
    """Immutable snapshot descriptor for a versioned dataset."""

    dataset_name: str
    version_id: str
    tag: str | None = None
    checksum_sha256: str
    parent_version_id: str | None = None
    n_rows: int
    n_columns: int
    columns: list[str]
    dtypes: dict[str, str]
    storage_path: str
    message: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetVersionManager:
    """Git/DVC-style content-addressed versioning store for datasets."""

    def __init__(self, store_dir: str | Path = "./.omniforge_workspace/data_versions") -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.store_dir / "manifest.json"
        if not self._manifest_path.exists():
            self._manifest_path.write_text(json.dumps({"datasets": {}}, indent=2), encoding="utf-8")

    def _load_manifest(self) -> dict[str, Any]:
        return dict(json.loads(self._manifest_path.read_text(encoding="utf-8")))

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self._manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _resolve_confined_snapshot_path(self, raw_path: str | Path) -> Path:
        """Resolve a snapshot path and verify canonical containment inside ``self.store_dir`` (CWE-22)."""
        base_dir = self.store_dir.resolve()
        candidate = Path(raw_path).resolve()
        if candidate == base_dir or not candidate.is_relative_to(base_dir):
            raise DataError(
                f"Refusing snapshot path outside dataset version store (CWE-22 protection): '{raw_path}'."
            )
        return candidate

    def commit(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        *,
        message: str = "",
        tag: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetVersion:
        """Persist a versioned immutable Parquet/JSON snapshot of ``df``."""
        safe_ds = _validate_version_identifier(dataset_name, "dataset name")
        safe_tag = _validate_version_identifier(tag, "version tag") if tag is not None else None
        manifest = self._load_manifest()
        history: list[dict[str, Any]] = manifest["datasets"].setdefault(safe_ds, [])
        checksum = compute_dataframe_checksum(df)
        version_num = len(history) + 1
        version_id = f"v{version_num}_{checksum[:10]}"
        parent_id = history[-1]["version_id"] if history else None

        ds_dir = self._resolve_confined_snapshot_path(self.store_dir / safe_ds)
        ds_dir.mkdir(parents=True, exist_ok=True)
        try:
            artifact_path = self._resolve_confined_snapshot_path(ds_dir / f"{version_id}.parquet")
            df.to_parquet(artifact_path, index=False)
        except DataError:
            raise
        except Exception:
            artifact_path = self._resolve_confined_snapshot_path(ds_dir / f"{version_id}.json")
            df.to_json(artifact_path, orient="table")

        version = DatasetVersion(
            dataset_name=safe_ds,
            version_id=version_id,
            tag=safe_tag,
            checksum_sha256=checksum,
            parent_version_id=parent_id,
            n_rows=int(len(df)),
            n_columns=int(len(df.columns)),
            columns=[str(c) for c in df.columns],
            dtypes={str(c): str(dt) for c, dt in zip(df.columns, df.dtypes, strict=False)},
            storage_path=str(artifact_path),
            message=message,
            metadata=metadata or {},
        )
        history.append(version.model_dump())
        self._save_manifest(manifest)
        logger.info("dataset_version_committed", dataset=safe_ds, version_id=version_id)
        return version

    def checkout(
        self, dataset_name: str, version_or_tag: str | None = None
    ) -> tuple[pd.DataFrame, DatasetVersion]:
        """Load a specific version, tag, or latest snapshot of ``dataset_name``."""
        safe_ds = _validate_version_identifier(dataset_name, "dataset name")
        if version_or_tag is not None and version_or_tag != "latest":
            version_or_tag = _validate_version_identifier(version_or_tag, "version_or_tag")
        versions = self.list_versions(safe_ds)
        if not versions:
            raise DataError(f"No versions found for dataset '{safe_ds}'.")
        selected: DatasetVersion | None = None
        if version_or_tag is None or version_or_tag == "latest":
            selected = versions[-1]
        else:
            for v in reversed(versions):
                if v.version_id == version_or_tag or v.tag == version_or_tag:
                    selected = v
                    break
        if selected is None:
            raise DataError(
                f"Version or tag '{version_or_tag}' not found for dataset '{safe_ds}'."
            )
        path = self._resolve_confined_snapshot_path(selected.storage_path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".json":
            df = pd.read_json(path, orient="table")
        else:
            raise DataError(
                f"Refusing to load unsafe legacy dataset snapshot format '{path.suffix}' (CWE-502 protection). "
                "Re-commit the dataset version in Parquet or JSON format."
            )
        return df, selected

    def list_versions(self, dataset_name: str) -> list[DatasetVersion]:
        """List all recorded versions of ``dataset_name`` in chronological order."""
        safe_ds = _validate_version_identifier(dataset_name, "dataset name")
        manifest = self._load_manifest()
        return [DatasetVersion(**rec) for rec in manifest["datasets"].get(safe_ds, [])]

    def diff(
        self, dataset_name: str, version_a: str, version_b: str
    ) -> dict[str, Any]:
        """Compare schema, row counts, and numeric summary changes between two dataset versions."""
        df_a, ver_a = self.checkout(dataset_name, version_a)
        df_b, ver_b = self.checkout(dataset_name, version_b)
        cols_a, cols_b = set(ver_a.columns), set(ver_b.columns)

        dtype_changes = {
            c: {"from": ver_a.dtypes[c], "to": ver_b.dtypes[c]}
            for c in sorted(cols_a & cols_b)
            if ver_a.dtypes[c] != ver_b.dtypes[c]
        }
        return {
            "dataset_name": dataset_name,
            "version_a": ver_a.version_id,
            "version_b": ver_b.version_id,
            "row_delta": len(df_b) - len(df_a),
            "added_columns": sorted(cols_b - cols_a),
            "removed_columns": sorted(cols_a - cols_b),
            "dtype_changes": dtype_changes,
            "checksum_changed": ver_a.checksum_sha256 != ver_b.checksum_sha256,
        }

    def export_dvc_manifest(self, dataset_name: str, version_or_tag: str | None = None) -> dict[str, Any]:
        """Generate a DVC-compatible metadata dictionary for external lineage interoperability."""
        _, ver = self.checkout(dataset_name, version_or_tag)
        return {
            "outs": [
                {
                    "path": ver.storage_path,
                    "hash": "sha256",
                    "checksum": ver.checksum_sha256,
                    "n_rows": ver.n_rows,
                    "n_columns": ver.n_columns,
                }
            ],
            "meta": {
                "dataset_name": ver.dataset_name,
                "version_id": ver.version_id,
                "tag": ver.tag,
                "created_at": ver.created_at,
            },
        }
