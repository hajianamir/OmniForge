"""BOLA (OWASP API1: Broken Object Level Authorization) session-isolation regression tests.

The workbench session store keys sessions by the client-supplied ``X-Session-ID`` header.
Before the namespace fix, ANY authenticated caller presenting the header value ``s``
landed in the SAME session store entry as every other caller — including callers from
different tenants — so one user could read/overwrite another user's in-flight dataset,
targets, exclusions and cached results merely by guessing a session id.

These tests pin the isolation contract:

1. Two different tenants presenting the SAME header value land in ISOLATED stores.
2. Two different subjects of the SAME tenant land in ISOLATED stores.
3. A caller re-presenting (tenant, subject, header) round-trips into ITS OWN store.
4. The namespaced key is an internal detail and is never echoed back to clients.
5. Unauthenticated requests are rejected outright.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from omniforge.api.auth import jwt_manager
from omniforge.api.main import app
from omniforge.api.workbench import _display_session_id, _sid
from omniforge.core.types import Principal

SHARED_SID = "shared-bola-regression-sid-7f3d"


def _token(subject: str, tenant: str) -> str:
    return jwt_manager.create_token(Principal(subject=subject, roles=["admin"], tenant_id=tenant))


def _hdr(token: str, sid: str = SHARED_SID) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "X-Session-Id": sid}


def _csv_bytes() -> io.BytesIO:
    df = pd.DataFrame(
        {
            "alpha_feature": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "beta_feature": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "bola_target": [0, 0, 0, 1, 1, 1, 0, 1],
        }
    )
    return io.BytesIO(df.to_csv(index=False).encode("utf-8"))


def _client() -> TestClient:
    return TestClient(app)


def test_same_header_value_is_isolated_across_tenants() -> None:
    token_a = _token("user-a", "tenant-alpha")
    token_b = _token("user-b", "tenant-beta")

    client_a = _client()
    r_up = client_a.post(
        "/api/workbench/s01/upload",
        headers=_hdr(token_a),
        files={"file": ("tenant_a_secret.csv", _csv_bytes(), "text/csv")},
    )
    assert r_up.status_code == 200, r_up.text

    # Tenant B presents the SAME header value: must NOT see tenant A's dataset.
    client_b = _client()
    r_state_b = client_b.get("/api/workbench/state", headers=_hdr(token_b))
    assert r_state_b.status_code == 200
    body_b = r_state_b.json()
    assert body_b["has_dataset"] is False or body_b["n_rows"] == 0
    assert "tenant_a_secret" not in str(body_b)

    # Tenant A round-trips into ITS OWN store.
    r_state_a = client_a.get("/api/workbench/state", headers=_hdr(token_a))
    body_a = r_state_a.json()
    assert body_a["has_dataset"] is True
    assert body_a["n_rows"] == 8
    # Displayed session id is the client-provided value, not the internal namespace.
    assert body_a["session_id"] == SHARED_SID


def test_same_header_value_is_isolated_across_subjects_within_tenant() -> None:
    token_alice = _token("alice", "tenant-shared")
    token_bob = _token("bob", "tenant-shared")
    sid = f"{SHARED_SID}-same-tenant"

    client_alice = _client()
    r_up = client_alice.post(
        "/api/workbench/s01/upload",
        headers=_hdr(token_alice, sid),
        files={"file": ("alice_private.csv", _csv_bytes(), "text/csv")},
    )
    assert r_up.status_code == 200, r_up.text

    client_bob = _client()
    r_state_bob = client_bob.get("/api/workbench/state", headers=_hdr(token_bob, sid))
    body_bob = r_state_bob.json()
    assert body_bob["has_dataset"] is False or body_bob["n_rows"] == 0

    r_state_alice = client_alice.get("/api/workbench/state", headers=_hdr(token_alice, sid))
    assert r_state_alice.json()["has_dataset"] is True


def test_unauthenticated_requests_are_rejected() -> None:
    client = _client()
    r = client.get("/api/workbench/state", headers={"X-Session-Id": SHARED_SID})
    assert r.status_code in (401, 403)
    r = client.post("/api/workbench/workspace/clear", headers={"X-Session-Id": SHARED_SID})
    assert r.status_code in (401, 403)


def test_sid_keys_differ_by_principal() -> None:
    pa = Principal(subject="alice", roles=["admin"], tenant_id="tenant-a")
    pb = Principal(subject="alice", roles=["admin"], tenant_id="tenant-b")
    pc = Principal(subject="bob", roles=["admin"], tenant_id="tenant-a")
    ka, kb, kc = _sid("x", pa), _sid("x", pb), _sid("x", pc)
    assert len({ka, kb, kc}) == 3
    assert ka.startswith("ws:")
    # Round-trip: the raw header value is recoverable for display purposes.
    assert _display_session_id(ka) == "x"
    assert _display_session_id("default") == "default"


@pytest.mark.parametrize(
    ("principal", "expect_prefix"),
    [(None, False)],
)
def test_sid_without_principal_keeps_legacy_key(principal, expect_prefix) -> None:
    key = _sid("legacy-sid-check", None)
    assert key == "legacy-sid-check"
