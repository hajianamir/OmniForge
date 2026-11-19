"""Feast-compatible offline and online Feature Store backed by Parquet and SQLite."""

from __future__ import annotations

import json
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

    def register_entity(self, entity: FeatureEntity) -> FeatureEntity:
        """Register or update an entity definition."""
        reg = self._load_registry()
        reg["entities"][entity.name] = entity.model_dump()
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
        missing_keys = [k for k in join_keys if k not in df.columns]
        if missing_keys:
            raise FeatureEngineeringError(
                f"Join keys {missing_keys} missing from feature DataFrame for view '{name}'."
            )
        if timestamp_column and timestamp_column not in df.columns:
            raise FeatureEngineeringError(
                f"Timestamp column '{timestamp_column}' missing from feature DataFrame."
            )

        excluded = set(join_keys) | ({timestamp_column} if timestamp_column else set())
        feature_cols = [str(c) for c in df.columns if c not in excluded]
        offline_path = self._offline_dir / f"{name}.parquet"
        df.to_parquet(offline_path, index=False)

        spec = FeatureViewSpec(
            name=name,
            entity_names=entity_names or join_keys,
            join_keys=join_keys,
            timestamp_column=timestamp_column,
            feature_columns=feature_cols,
            ttl_seconds=ttl_seconds,
            description=description,
            tags=tags or {},
            offline_path=str(offline_path.resolve()),
        )
        reg = self._load_registry()
        reg["feature_views"][name] = spec.model_dump()
        self._save_registry(reg)

        if materialize_online:
            self.materialize_to_online(name)
        logger.info("feature_view_registered", view=name, n_features=len(feature_cols), rows=len(df))
        return spec

    def materialize_to_online(self, view_name: str) -> int:
        """Materialize latest entity feature vectors from offline storage into SQLite online store."""
        spec = self.get_feature_view(view_name)
        df = pd.read_parquet(spec.offline_path)
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
                "|".join(arr[idx] for arr in join_arrs),
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
            spec = self.get_feature_view(view_name)
            view_df = pd.read_parquet(spec.offline_path)
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
                merged = pd.merge_asof(
                    left,
                    view_sub,
                    left_on=entity_timestamp_col,
                    right_on=spec.timestamp_column,
                    by=spec.join_keys,
                    direction="backward",
                    suffixes=("", f"__{view_name}"),
                )
                result = merged.sort_values("_uap_orig_order").drop(columns=["_uap_orig_order"]).reset_index(drop=True)
            else:
                dedup_view = view_sub.drop_duplicates(subset=spec.join_keys, keep="last")
                result = result.merge(dedup_view, on=spec.join_keys, how="left", suffixes=("", f"__{view_name}"))
        return result

    def get_online_features(
        self,
        view_name: str,
        entity_rows: list[dict[str, Any]],
        features: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve low-latency feature vectors for real-time inference requests."""
        spec = self.get_feature_view(view_name)
        target_feats = features or spec.feature_columns
        output: list[dict[str, Any]] = []
        with _sqlite_conn(self._online_db) as conn:
            for row in entity_rows:
                key_str = "|".join(str(row[k]) for k in spec.join_keys)
                cur = conn.execute(
                    "SELECT features_json FROM online_features WHERE view_name = ? AND entity_key = ?",
                    (view_name, key_str),
                )
                rec = cur.fetchone()
                stored = json.loads(rec[0]) if rec else {}
                item = {k: row[k] for k in spec.join_keys}
                for f in target_feats:
                    item[f] = stored.get(f)
                output.append(item)
        return output

    def get_feature_view(self, name: str) -> FeatureViewSpec:
        """Retrieve a registered feature view specification."""
        reg = self._load_registry()
        if name not in reg["feature_views"]:
            raise FeatureEngineeringError(f"Feature view '{name}' is not registered.")
        return FeatureViewSpec(**reg["feature_views"][name])

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
