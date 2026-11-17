"""JWT Bearer Authentication and RBAC/ABAC FastAPI dependencies backed by OmniForge Security primitives."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, Header, HTTPException, status

from omniforge.core.exceptions import AuthorizationError
from omniforge.core.security import AccessController, JWTManager
from omniforge.core.types import Principal

jwt_manager = JWTManager()
access_controller = AccessController()


def get_current_principal(
    authorization: str | None = Header(default=None),
) -> Principal:
    """Extract and validate a Bearer JWT token; falls back to dev admin Principal if omitted."""
    if not authorization:
        return Principal(subject="local_dev", roles=["admin"], tenant_id="default-tenant")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must use 'Bearer <token>' scheme.",
        )
    try:
        return jwt_manager.verify_token(parts[1])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired JWT token: {exc}",
        ) from exc


def require_permission(permission: str) -> Callable[..., Principal]:
    """FastAPI dependency enforcing RBAC/ABAC permission checks for protected endpoints."""

    def _checker(principal: Principal = Depends(get_current_principal)) -> Principal:
        try:
            access_controller.authorize(principal, permission)
        except AuthorizationError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message) from exc
        return principal

    return _checker
