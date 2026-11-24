"""TTL monitoring uses online-store semantics and respects tenant isolation."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from omniforge.api import workbench as wb
from omniforge.api.auth import get_api_key
from omniforge.api.main import create_app
from omniforge.config.settings import get_settings
from omniforge.core.exceptions import FeatureEngineeringError
from omniforge.core.types import Principal
from omniforge.data.feature_store import LocalFeatureStore


def test_freshness_report_matches_ttl_boundary_unknowns_and_purges(tmp_path: Path) -> None:
    store = LocalFeatureStore(tmp_path / "store")
    now = datetime(2025, 1, 1, 12, tzinfo=UTC)
    store.register_feature_view(
        "credit", pd.DataFrame({
            "id": ["expired", "boundary", "fresh", "unknown", "fallback"],
            "when": [now - timedelta(seconds=61), now - timedelta(seconds=60), now - timedelta(seconds=10), now, now],
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
        }), join_keys=["id"], timestamp_column="when", ttl_seconds=60,
    )
    store.register_feature_view(
        "no_ttl", pd.DataFrame({"id": ["old"], "when": [now - timedelta(days=10)], "score": [1.0]}),
        join_keys=["id"], timestamp_column="when", ttl_seconds=0,
    )
    with sqlite3.connect(store._online_db) as conn:
        conn.execute(
            "UPDATE online_features SET event_timestamp = ?, updated_at = ? WHERE view_name = ? AND entity_key = ?",
            ("bad", "also bad", "credit", '["unknown"]'),
        )
        conn.execute(
            "UPDATE online_features SET event_timestamp = ?, updated_at = ? WHERE view_name = ? AND entity_key = ?",
            ("bad", (now - timedelta(seconds=61)).isoformat(), "credit", '["fallback"]'),
        )
    report = store.freshness_report(as_of=now)
    views = {view["view_name"]: view for view in report["views"]}
    assert (report["online_rows"], report["expired_rows"], report["unknown_timestamp_rows"]) == (6, 2, 1)
    assert (views["credit"]["fresh_rows"], views["credit"]["expired_rows"]) == (2, 2)
    assert views["no_ttl"]["expired_rows"] == 0
    assert report["alerts"][0]["view_name"] == "credit"
    values = store.get_online_features("credit", [{"id": "expired"}, {"id": "boundary"}, {"id": "fallback"}], as_of=now)
    assert values[0]["score"] is None and values[1]["score"] == 2.0 and values[2]["score"] is None
    with pytest.raises(FeatureEngineeringError, match="Invalid as_of"):
        store.freshness_report(as_of="not-a-date")
    store.purge_expired_features(as_of=now)
    after = store.freshness_report(as_of=now)
    assert after["expired_rows"] == 0
    assert after["unknown_timestamp_rows"] == 1
    assert after["alerts"] == []


def test_freshness_endpoint_is_tenant_scoped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings.paths, "workspace_dir", tmp_path)
    monkeypatch.setattr(settings.paths, "feature_store_dir", tmp_path / "default-feature-store")
    tenant_a = f"fresh-a-{uuid.uuid4().hex}"
    tenant_b = f"fresh-b-{uuid.uuid4().hex}"
    a = Principal(subject="alice", roles=["admin"], tenant_id=tenant_a)
    store = wb.get_tenant_feature_store(a)
    store.register_feature_view(
        "private_a", pd.DataFrame({"id": ["1"], "when": [datetime(2000, 1, 1, tzinfo=UTC)], "score": [7]}),
        join_keys=["id"], timestamp_column="when", ttl_seconds=60,
    )
    client = TestClient(create_app())

    def token(subject: str, tenant: str) -> str:
        response = client.post(
            "/api/v1/auth/token", json={"username": subject, "role": "admin", "tenant_id": tenant},
            headers={"X-API-Key": get_api_key()},
        )
        assert response.status_code == 200, response.text
        return response.json()["access_token"]

    a_token, b_token = token("alice", tenant_a), token("bob", tenant_b)
    ra = client.get("/api/workbench/feature-store/freshness", headers={"Authorization": f"Bearer {a_token}"})
    rb = client.get("/api/workbench/feature-store/freshness", headers={"Authorization": f"Bearer {b_token}"})
    assert ra.status_code == 200 and ra.json()["expired_rows"] == 1
    assert ra.json()["alerts"][0]["view_name"] == "private_a"
    assert rb.status_code == 200 and rb.json()["total_views"] == 0
    assert "private_a" not in rb.text
