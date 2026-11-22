"""P1-12 regression tests: dedicated platform_admin role for cross-tenant access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omniforge.core.constants import RBAC_ROLE_PERMISSIONS
from omniforge.core.exceptions import AuthorizationError, SecurityError
from omniforge.core.security import AccessController, AuditLogger
from omniforge.core.types import Principal


def _controller() -> AccessController:
    return AccessController()


def _p(subject: str, roles: list[str], tenant: str) -> Principal:
    return Principal(subject=subject, roles=roles, tenant_id=tenant)


def test_platform_admin_registered_and_superset_of_admin() -> None:
    assert "platform_admin" in RBAC_ROLE_PERMISSIONS
    admin = RBAC_ROLE_PERMISSIONS["admin"]
    platform = RBAC_ROLE_PERMISSIONS["platform_admin"]
    assert admin <= platform  # platform_admin can do everything tenant admin can


def test_platform_admin_may_cross_tenants() -> None:
    c = _controller()
    pa = _p("ops", ["platform_admin"], "tenant-a")
    assert c.authorize(pa, "data:read", resource_tenant_id="tenant-b") is True


def test_tenant_admin_is_tenant_scoped() -> None:
    c = _controller()
    admin = _p("boss", ["admin"], "tenant-a")
    with pytest.raises(AuthorizationError, match="Cross-tenant"):
        c.authorize(admin, "data:read", resource_tenant_id="tenant-b")


def test_regular_roles_may_not_cross_tenants() -> None:
    c = _controller()
    for role in ("data_scientist", "mlops_engineer", "analyst", "viewer"):
        principal = _p("u", [role], "tenant-a")
        with pytest.raises(AuthorizationError, match="Cross-tenant"):
            c.authorize(principal, "data:read", resource_tenant_id="tenant-b")


def test_tenant_admin_same_tenant_ok() -> None:
    c = _controller()
    admin = _p("boss", ["admin"], "tenant-a")
    assert c.authorize(admin, "data:read", resource_tenant_id="tenant-a") is True


def test_platform_admin_still_needs_permission() -> None:
    c = _controller()
    pa = _p("ops", ["platform_admin"], "tenant-a")
    with pytest.raises(AuthorizationError, match="lacks required permission"):
        c.authorize(pa, "tenant:manage-everything", resource_tenant_id="tenant-b")


def test_tenant_isolation_disabled_allows_any_role() -> None:
    c = AccessController(enforce_tenant_isolation=False)
    analyst = _p("u", ["analyst"], "tenant-a")
    assert c.authorize(analyst, "data:read", resource_tenant_id="tenant-b") is True


def test_abac_attributes_still_enforced_for_platform_admin() -> None:
    c = _controller()
    pa = _p("ops", ["platform_admin"], "tenant-a")
    with pytest.raises(AuthorizationError, match="ABAC"):
        c.authorize(pa, "data:read", resource_tenant_id="tenant-b", required_attributes={"clearance": "top_secret"})


def _read_audit(log: Path) -> list[dict]:
    return [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_cross_tenant_allow_is_audited(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    audit = AuditLogger(log_path=log)
    c = AccessController(audit_logger=audit)
    pa = _p("ops", ["platform_admin"], "tenant-a")
    assert c.authorize(pa, "data:read", resource_tenant_id="tenant-b") is True
    events = [e for e in _read_audit(log) if e["action"] == "authorize:cross_tenant"]
    assert len(events) == 1
    assert events[0]["status"] == "success" and events[0]["resource_id"] == "tenant-b"
    assert events[0]["details"]["principal_tenant"] == "tenant-a"


def test_cross_tenant_deny_is_audited(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    audit = AuditLogger(log_path=log)
    c = AccessController(audit_logger=audit)
    admin = _p("boss", ["admin"], "tenant-a")
    with pytest.raises(AuthorizationError):
        c.authorize(admin, "data:read", resource_tenant_id="tenant-b")
    events = [e for e in _read_audit(log) if e["action"] == "authorize:cross_tenant"]
    assert len(events) == 1 and events[0]["status"] == "denied"


def test_same_tenant_access_writes_no_cross_tenant_audit(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    audit = AuditLogger(log_path=log)
    c = AccessController(audit_logger=audit)
    admin = _p("boss", ["admin"], "tenant-a")
    assert c.authorize(admin, "data:read", resource_tenant_id="tenant-a") is True
    assert log.exists() is False  # nothing written for ordinary same-tenant checks


def test_cross_tenant_allow_fails_closed_when_audit_write_fails(tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    audit = AuditLogger(log_path=log)
    c = AccessController(audit_logger=audit)

    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    audit.log_event = _boom  # type: ignore[method-assign]
    pa = _p("ops", ["platform_admin"], "tenant-a")
    with pytest.raises(SecurityError, match="fail-closed"):
        c.authorize(pa, "data:read", resource_tenant_id="tenant-b")


def test_platform_admin_permissions_resolve_via_get_permissions() -> None:
    c = _controller()
    pa = _p("ops", ["platform_admin"], "tenant-a")
    perms = c.get_permissions(pa)
    for required in ("security:manage", "audit:read", "model:promote", "data:delete"):
        assert required in perms
