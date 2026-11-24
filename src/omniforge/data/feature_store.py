"""Feast-compatible offline and online Feature Store backed by Parquet and SQLite."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from omniforge.core.exceptions import FeatureEngineeringError
from omniforge.core.logging import get_logger
from omniforge.core.utils import safe_json_default

logger = get_logger(__name__)

_SAFE_STORE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _validate_store_identifier(value: str, label: str) -> str:
    """Validate logical Feature Store identifiers against CWE-22 path traversal tokens."""
    cleaned = str(value).strip()
    if not cleaned or ".." in cleaned or not _SAFE_STORE_NAME_RE.fullmatch(cleaned):
        raise FeatureEngineeringError(
            f"Invalid {label} '{value}': must match ^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$ without '..' or path separators."
        )
    return cleaned


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


class FeatureEntity(BaseModel):
    """Entity join key definition (Feast-compatible)."""

    name: str
    join_key: str
    value_type: str = "STRING"
    description: str = ""


class FeatureViewSpec(BaseModel):
    """Feature view schema, TTL, entity bindings, and storage paths."""

    name: str
    entity_names: list[str]
    join_keys: list[str]
    timestamp_column: str | None = None
    feature_columns: list[str]
    ttl_seconds: int = 86400 * 30
    description: str = ""
    tags: dict[str, str] = Field(default_factory=dict)
    offline_path: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class LocalFeatureStore:
    """Production-ready Feast-compatible Feature Store with point-in-time joins and online serving."""

    def __init__(self, repo_path: str | Path = "./.omniforge_workspace/feature_store") -> None:
        self.repo_path = Path(repo_path)
        self.repo_path.mkdir(parents=True, exist_ok=True)
        self._offline_dir = self.repo_path / "offline"
        self._offline_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.repo_path / "registry.json"
        self._online_db = self.repo_path / "online_store.db"
        self._init_stores()

    def _init_stores(self) -> None:
        if not self._registry_path.exists():
            self._registry_path.write_text(
                json.dumps({"entities": {}, "feature_views": {}}, indent=2), encoding="utf-8"
            )
        with _sqlite_conn(self._online_db) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS online_features (
                    view_name TEXT NOT NULL,
                    entity_key TEXT NOT NULL,
                    event_timestamp TEXT,
                    features_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (view_name, entity_key)
                )
                """
            )
            conn.commit()

    def _load_registry(self) -> dict[str, Any]:
        return dict(json.loads(self._registry_path.read_text(encoding="utf-8")))

    def _save_registry(self, reg: dict[str, Any]) -> None:
        self._registry_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")

    def _resolve_confined_offline_path(self, raw_path: str | Path) -> Path:
        """Resolve an offline feature artifact path and enforce strict CWE-22 directory confinement."""
        base_dir = self._offline_dir.resolve()
        candidate = Path(raw_path).resolve()
        if candidate == base_dir or not candidate.is_relative_to(base_dir):
            raise FeatureEngineeringError(
                f"Refusing path outside feature store offline directory (CWE-22 protection): '{raw_path}'."
            )
        return candidate

    def register_entity(self, entity: FeatureEntity) -> FeatureEntity:
        """Register or update an entity definition."""
        safe_name = _validate_store_identifier(entity.name, "entity name")
        reg = self._load_registry()
        reg["entities"][safe_name] = entity.model_copy(update={"name": safe_name}).model_dump()
        self._save_registry(reg)
        return entity

    def register_feature_view(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        join_keys: list[str],
        timestamp_column: str | None = None,
        entity_names: list[str] | None = None,
        ttl_seconds: int = 86400 * 30,
        description: str = "",
        materialize_online: bool = True,
        tags: dict[str, str] | None = None,
    ) -> FeatureViewSpec:
        """Persist a feature table into the offline store and optionally materialize online."""
        safe_name = _validate_store_identifier(name, "feature view name")
        missing_keys = [k for k in join_keys if k not in df.columns]
        if missing_keys:
            raise FeatureEngineeringError(
                f"Join keys {missing_keys} missing from feature DataFrame for view '{safe_name}'."
            )
        if timestamp_column and timestamp_column not in df.columns:
            raise FeatureEngineeringError(
                f"Timestamp column '{timestamp_column}' missing from feature DataFrame."
            )

        excluded = set(join_keys) | ({timestamp_column} if timestamp_column else set())
        feature_cols = [str(c) for c in df.columns if c not in excluded]
        offline_path = self._resolve_confined_offline_path(self._offline_dir / f"{safe_name}.parquet")
        df.to_parquet(offline_path, index=False)

        spec = FeatureViewSpec(
            name=safe_name,
            entity_names=entity_names or join_keys,
            join_keys=join_keys,
            timestamp_column=timestamp_column,
            feature_columns=feature_cols,
            ttl_seconds=ttl_seconds,
            description=description,
            tags=tags or {},
            offline_path=str(offline_path),
        )
        reg = self._load_registry()
        reg["feature_views"][safe_name] = spec.model_dump()
        self._save_registry(reg)

        if materialize_online:
            self.materialize_to_online(safe_name)
        logger.info("feature_view_registered", view=safe_name, n_features=len(feature_cols), rows=len(df))
        return spec

    def materialize_to_online(self, view_name: str) -> int:
        """Materialize latest entity feature vectors from offline storage into SQLite online store."""
        safe_view = _validate_store_identifier(view_name, "feature view name")
        spec = self.get_feature_view(safe_view)
        safe_offline = self._resolve_confined_offline_path(spec.offline_path)
        df = pd.read_parquet(safe_offline)
        if spec.timestamp_column and spec.timestamp_column in df.columns:
            df = df.sort_values(spec.timestamp_column).drop_duplicates(
                subset=spec.join_keys, keep="last"
            )
        else:
            df = df.drop_duplicates(subset=spec.join_keys, keep="last")

        now_iso = datetime.now(UTC).isoformat()
        join_cols = list(spec.join_keys)
        feat_cols = list(spec.feature_columns)
        ts_col = spec.timestamp_column
        join_arrs = [df[k].astype(str).to_numpy() for k in join_cols]
        ts_arr = df[ts_col].astype(str).to_numpy() if ts_col and ts_col in df.columns else None
        feat_records = df[feat_cols].to_dict(orient="records") if feat_cols else [{} for _ in range(len(df))]
        rows_to_upsert: list[tuple[str, str, str, str, str]] = [
            (
                view_name,
                json.dumps([arr[idx] for arr in join_arrs], ensure_ascii=False),
                ts_arr[idx] if ts_arr is not None else now_iso,
                json.dumps(feat_records[idx], default=safe_json_default),
                now_iso,
            )
            for idx in range(len(df))
        ]

        with _sqlite_conn(self._online_db) as conn:
            conn.executemany(
                """
                INSERT INTO online_features (view_name, entity_key, event_timestamp, features_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(view_name, entity_key) DO UPDATE SET
                    event_timestamp = excluded.event_timestamp,
                    features_json = excluded.features_json,
                    updated_at = excluded.updated_at
                """,
                rows_to_upsert,
            )
            conn.commit()
        return len(rows_to_upsert)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_refs: list[str],
        *,
        entity_timestamp_col: str | None = None,
    ) -> pd.DataFrame:
        """Perform point-in-time-correct offline retrieval for training datasets.

        Args:
            entity_df: DataFrame of entity keys (and optional event timestamps).
            feature_refs: List of ``"view_name:feature_name"`` or ``"view_name:*"`` references.
            entity_timestamp_col: Timestamp column in ``entity_df`` for as-of point-in-time joins.
        """
        grouped: dict[str, list[str]] = {}
        for ref in feature_refs:
            if ":" not in ref:
                raise FeatureEngineeringError(
                    f"Invalid feature reference '{ref}'. Expected 'view_name:feature_name'."
                )
            view_name, feat_name = ref.split(":", 1)
            grouped.setdefault(view_name, []).append(feat_name)

        result = entity_df.copy()
        for view_name, feats in grouped.items():
            safe_view = _validate_store_identifier(view_name, "feature view name")
            spec = self.get_feature_view(safe_view)
            safe_offline = self._resolve_confined_offline_path(spec.offline_path)
            view_df = pd.read_parquet(safe_offline)
            selected_feats = spec.feature_columns if "*" in feats else [f for f in feats if f in view_df.columns]
            keep_cols = list(dict.fromkeys(spec.join_keys + ([spec.timestamp_column] if spec.timestamp_column else []) + selected_feats))
            view_sub = view_df[keep_cols].copy()

            if entity_timestamp_col and spec.timestamp_column and entity_timestamp_col in result.columns:
                left = result.copy()
                left["_uap_orig_order"] = np.arange(len(left))
                left[entity_timestamp_col] = pd.to_datetime(left[entity_timestamp_col])
                view_sub[spec.timestamp_column] = pd.to_datetime(view_sub[spec.timestamp_column])
                left = left.sort_values(entity_timestamp_col)
                view_sub = view_sub.sort_values(spec.timestamp_column)
                tol = pd.Timedelta(seconds=float(spec.ttl_seconds)) if getattr(spec, "ttl_seconds", 0) and spec.ttl_seconds > 0 else None
                merged = pd.merge_asof(
                    left,
                    view_sub,
                    left_on=entity_timestamp_col,
                    right_on=spec.timestamp_column,
                    by=spec.join_keys,
                    direction="backward",
                    tolerance=tol,
                    suffixes=("", f"__{view_name}"),
                )
                result = merged.sort_values("_uap_orig_order").drop(columns=["_uap_orig_order"]).reset_index(drop=True)
            else:
                dedup_view = view_sub.drop_duplicates(subset=spec.join_keys, keep="last")
                result = result.merge(dedup_view, on=spec.join_keys, how="left", suffixes=("", f"__{view_name}"))
        return result

    @staticmethod
    def _parse_utc_timestamp(ts_val: Any) -> pd.Timestamp | None:
        if ts_val is None or (isinstance(ts_val, float) and np.isnan(ts_val)):
            return None
        try:
            parsed = pd.Timestamp(ts_val)
            if pd.isna(parsed):
                return None
            return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
        except Exception:
            return None

    def get_online_features(
        self,
        view_name: str,
        entity_rows: list[dict[str, Any]],
        features: list[str] | None = None,
        *,
        enforce_ttl: bool = True,
        as_of: datetime | str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve low-latency feature vectors for real-time inference with Feast-compatible TTL freshness enforcement."""
        safe_view = _validate_store_identifier(view_name, "feature view name")
        spec = self.get_feature_view(safe_view)
        target_feats = features or spec.feature_columns
        ref_ts = self._parse_utc_timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
        output: list[dict[str, Any]] = []
        with _sqlite_conn(self._online_db) as conn:
            for row in entity_rows:
                key_str = json.dumps([str(row[k]) for k in spec.join_keys], ensure_ascii=False)
                cur = conn.execute(
                    "SELECT features_json, event_timestamp, updated_at FROM online_features WHERE view_name = ? AND entity_key = ?",
                    (safe_view, key_str),
                )
                rec = cur.fetchone()
                stored: dict[str, Any] = {}
                if rec:
                    expired = False
                    if enforce_ttl and spec.ttl_seconds > 0 and ref_ts is not None:
                        ev_ts = self._parse_utc_timestamp(rec[1]) or self._parse_utc_timestamp(rec[2])
                        if ev_ts is not None and (ref_ts - ev_ts).total_seconds() > float(spec.ttl_seconds):
                            expired = True
                    if not expired:
                        stored = json.loads(rec[0])
                item = {k: row[k] for k in spec.join_keys}
                for f in target_feats:
                    item[f] = stored.get(f)
                output.append(item)
        return output

    def freshness_report(self, *, as_of: datetime | str | None = None) -> dict[str, Any]:
        """Snapshot online TTL freshness without exposing entity keys or feature values.

        Counts describe rows currently in the online store, not a cumulative
        count of historical expirations or purged rows. Unknown timestamps are
        reported separately; TTL-disabled views never mark rows expired.
        """
        ref_ts = self._parse_utc_timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
        if ref_ts is None:
            raise FeatureEngineeringError("Invalid as_of timestamp for feature freshness report.")
        specs = self.list_feature_views()
        views: dict[str, dict[str, Any]] = {
            spec.name: {
                "view_name": spec.name,
                "ttl_seconds": spec.ttl_seconds,
                "feature_columns": list(spec.feature_columns),
                "online_rows": 0,
                "fresh_rows": 0,
                "expired_rows": 0,
                "unknown_timestamp_rows": 0,
            }
            for spec in specs
        }
        with _sqlite_conn(self._online_db) as conn:
            cursor = conn.execute("SELECT view_name, event_timestamp, updated_at FROM online_features")
            for view_name, event_raw, updated_raw in cursor:
                view = views.get(view_name)
                if view is None:
                    continue  # unregistered orphan rows are not exposed
                view["online_rows"] += 1
                if view["ttl_seconds"] <= 0:
                    view["fresh_rows"] += 1
                    continue
                timestamp = self._parse_utc_timestamp(event_raw) or self._parse_utc_timestamp(updated_raw)
                if timestamp is None:
                    view["unknown_timestamp_rows"] += 1
                elif (ref_ts - timestamp).total_seconds() > view["ttl_seconds"]:
                    view["expired_rows"] += 1
                else:
                    view["fresh_rows"] += 1
        view_list = sorted(views.values(), key=lambda view: view["view_name"])
        alerts = [
            {
                "view_name": view["view_name"],
                "expired_rows": view["expired_rows"],
                "message": f"{view['expired_rows']} online feature vector(s) in '{view['view_name']}' have expired.",
            }
            for view in view_list if view["expired_rows"] > 0
        ]
        return {
            "as_of": ref_ts.isoformat(),
            "total_views": len(view_list),
            "online_rows": sum(view["online_rows"] for view in view_list),
            "expired_rows": sum(view["expired_rows"] for view in view_list),
            "unknown_timestamp_rows": sum(view["unknown_timestamp_rows"] for view in view_list),
            "views": view_list,
            "alerts": alerts,
        }

    def purge_expired_features(
        self,
        view_name: str | None = None,
        *,
        as_of: datetime | str | None = None,
    ) -> dict[str, int]:
        """Enforce retention policy by purging expired records from offline Parquet and SQLite online stores."""
        ref_ts = self._parse_utc_timestamp(as_of) or pd.Timestamp.now(tz="UTC")
        views = [self.get_feature_view(view_name)] if view_name is not None else self.list_feature_views()
        offline_purged = 0
        online_purged = 0
        processed = 0

        for spec in views:
            if spec.ttl_seconds <= 0:
                continue
            processed += 1
            cutoff_utc = ref_ts - pd.Timedelta(seconds=float(spec.ttl_seconds))

            # 1. Purge expired rows in offline Parquet store when timestamp_column is configured
            safe_offline = self._resolve_confined_offline_path(spec.offline_path)
            if safe_offline.exists() and spec.timestamp_column:
                df = pd.read_parquet(safe_offline)
                if spec.timestamp_column in df.columns and len(df) > 0:
                    ts_ser = pd.to_datetime(df[spec.timestamp_column], utc=True, errors="coerce")
                    keep_mask = ts_ser.isna() | (ts_ser >= cutoff_utc)
                    removed = int((~keep_mask).sum())
                    if removed > 0:
                        df.loc[keep_mask].reset_index(drop=True).to_parquet(safe_offline, index=False)
                        offline_purged += removed

            # 2. Purge expired rows in SQLite online_features table
            with _sqlite_conn(self._online_db) as conn:
                rows = conn.execute(
                    "SELECT entity_key, event_timestamp, updated_at FROM online_features WHERE view_name = ?",
                    (spec.name,),
                ).fetchall()
                expired_keys: list[tuple[str, str]] = []
                for e_key, ev_raw, up_raw in rows:
                    row_ts = self._parse_utc_timestamp(ev_raw) or self._parse_utc_timestamp(up_raw)
                    if row_ts is not None and row_ts < cutoff_utc:
                        expired_keys.append((spec.name, str(e_key)))
                if expired_keys:
                    conn.executemany(
                        "DELETE FROM online_features WHERE view_name = ? AND entity_key = ?",
                        expired_keys,
                    )
                    conn.commit()
                    online_purged += len(expired_keys)

        return {
            "views_processed": processed,
            "offline_rows_purged": offline_purged,
            "online_rows_purged": online_purged,
        }

    def delete_entity_data(self, entity_filters: dict[str, Any]) -> dict[str, int]:
        """GDPR Art. 17 Right-to-Erasure: purge matching entity records across all offline and online feature views."""
        if not entity_filters:
            raise FeatureEngineeringError("entity_filters cannot be empty for delete_entity_data().")
        offline_deleted = 0
        online_deleted = 0

        for spec in self.list_feature_views():
            matched_keys = [k for k in spec.join_keys if k in entity_filters]
            if not matched_keys:
                continue

            safe_offline = self._resolve_confined_offline_path(spec.offline_path)
            if safe_offline.exists():
                df = pd.read_parquet(safe_offline)
                if len(df) > 0:
                    del_mask = pd.Series(True, index=df.index)
                    for k in matched_keys:
                        del_mask &= df[k].astype(str) == str(entity_filters[k])
                    cnt = int(del_mask.sum())
                    if cnt > 0:
                        df.loc[~del_mask].reset_index(drop=True).to_parquet(safe_offline, index=False)
                        offline_deleted += cnt

            key_positions = [(idx, str(entity_filters[k])) for idx, k in enumerate(spec.join_keys) if k in entity_filters]
            with _sqlite_conn(self._online_db) as conn:
                rows = conn.execute(
                    "SELECT entity_key FROM online_features WHERE view_name = ?",
                    (spec.name,),
                ).fetchall()
                to_delete: list[tuple[str, str]] = []
                for (ekey_raw,) in rows:
                    try:
                        parsed_key = json.loads(ekey_raw)
                    except Exception:
                        continue
                    if isinstance(parsed_key, list) and all(
                        pos < len(parsed_key) and str(parsed_key[pos]) == target_val
                        for pos, target_val in key_positions
                    ):
                        to_delete.append((spec.name, str(ekey_raw)))
                if to_delete:
                    conn.executemany(
                        "DELETE FROM online_features WHERE view_name = ? AND entity_key = ?",
                        to_delete,
                    )
                    conn.commit()
                    online_deleted += len(to_delete)

        return {"offline_rows_deleted": offline_deleted, "online_rows_deleted": online_deleted}

    def get_feature_view(self, name: str) -> FeatureViewSpec:
        """Retrieve a registered feature view specification."""
        safe_name = _validate_store_identifier(name, "feature view name")
        reg = self._load_registry()
        if safe_name not in reg["feature_views"]:
            raise FeatureEngineeringError(f"Feature view '{safe_name}' is not registered.")
        return FeatureViewSpec(**reg["feature_views"][safe_name])

    def list_feature_views(self) -> list[FeatureViewSpec]:
        """List all registered feature views."""
        reg = self._load_registry()
        return [FeatureViewSpec(**v) for v in reg["feature_views"].values()]

    def export_feast_yaml(self) -> Path:
        """Write a Feast-compatible ``feature_store.yaml`` configuration file."""
        cfg = {
            "project": "uap_feature_store",
            "registry": str(self._registry_path.resolve()),
            "provider": "local",
            "online_store": {"type": "sqlite", "path": str(self._online_db.resolve())},
            "entity_key_serialization_version": 3,
        }
        yaml_path = self.repo_path / "feature_store.yaml"
        yaml_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return yaml_path
