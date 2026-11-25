"""JWT Bearer, API-Key, and CSRF-Protected Local Workbench Authentication & RBAC Dependencies."""

from __future__ import annotations

import contextvars
import hashlib
import hmac
import ipaddress
import os
import secrets
import time
from collections.abc import Callable

from fastapi import Depends, Header, HTTPException, Request, Response, status

from omniforge.config.settings import get_settings
from omniforge.core.exceptions import AuthorizationError
from omniforge.core.security import AccessController, JWTManager, _is_production_env, _resolve_or_persist_secret
from omniforge.core.types import Principal


def _build_jwt_manager() -> JWTManager:
    sec = get_settings().security
    return JWTManager(
        secret_key=sec.jwt_secret_key or None,
        algorithm=sec.jwt_algorithm,
        expiration_minutes=sec.jwt_expiration_minutes,
    )


jwt_manager = _build_jwt_manager()


def _build_access_controller() -> AccessController:
    """Wire the shared policy engine with the mandatory cross-tenant audit trail (P1-12).

    Constructed eagerly so the audit sink exists from app startup; cross-tenant decisions
    (only reachable with ``resource_tenant_id=``) are always recorded in the tamper-evident
    log, fail-closed. No audit records are written unless a cross-tenant decision is made.
    """
    from omniforge.config.settings import get_settings
    from omniforge.core.security import AuditLogger

    settings = get_settings()
    try:
        return AccessController(audit_logger=AuditLogger(log_path=settings.paths.audit_log_path))
    except Exception as exc:
        # Mandatory audit for cross-tenant decisions: refuse to boot unaudited rather than
        # degrade to silent cross-tenant access. Fix the audit path or directory permissions.
        raise RuntimeError(
            f"OmniForge could not initialize the mandatory audit trail at '{settings.paths.audit_log_path}'. "
            "Cross-tenant authorization is fail-closed by design; fix the path or permissions. "
            f"Cause: {exc}"
        ) from exc


access_controller = _build_access_controller()

# Request-scoped authenticated principal. Set once per request inside ``get_current_principal``.
# anyio propagates contextvars into the threadpool copies used for sync path operations, so the
# workbench session registry (``omniforge.api.workbench._sid``) can bind the caller's tenant and
# subject to the ``X-Session-ID`` without threading the principal through every call site.
_request_principal_ctx: contextvars.ContextVar[Principal | None] = contextvars.ContextVar(
    "omniforge_request_principal", default=None
)


def get_request_principal() -> Principal | None:
    """Return the authenticated principal for the current request context, if any."""
    return _request_principal_ctx.get(None)

WORKBENCH_AUTH_COOKIE = "omniforge_wb_auth"
WORKBENCH_CSRF_COOKIE = "omniforge_csrf"
_WORKBENCH_COOKIE_CONTEXT = b"omniforge:workbench-cookie:v1"
_STATE_CHANGING_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def get_api_key() -> str:
    """Resolve the active server API key from config/env or persistent 0600 local keyfile."""
    settings_key = get_settings().security.api_key or None
    raw = _resolve_or_persist_secret(
        key_name="api_key",
        explicit=settings_key,
        env_vars=("OMNIFORGE_API_KEY", "OMNIFORGE_SECURITY__API_KEY"),
        min_length=24,
    )
    return raw.decode("utf-8", errors="replace")


def verify_api_key(candidate: str | None) -> bool:
    """Verify an API key or workbench token in constant time."""
    if not candidate or not isinstance(candidate, str):
        return False
    expected = get_api_key()
    return hmac.compare_digest(candidate.strip(), expected)


_WORKBENCH_COOKIE_MAX_AGE = 8 * 60 * 60  # 8 hours


def _derive_workbench_cookie_token(*, session_id: str | None = None) -> str:
    """Derive an HMAC-authenticated cookie value bound to the server API key and a session.

    Returns ``{nonce}.{timestamp}.{hmac_hex}`` so verification is stateless:
    the nonce and timestamp are carried in the cookie value, and the HMAC covers
    all three fields bound to the server API key.
    """
    import time

    key_bytes = get_api_key().encode("utf-8")
    nonce = session_id or secrets.token_urlsafe(16)
    timestamp = str(int(time.time()))
    material = b"\x00".join(
        [_WORKBENCH_COOKIE_CONTEXT, nonce.encode("utf-8"), timestamp.encode("utf-8")]
    )
    mac = hmac.new(key_bytes, material, hashlib.sha256).hexdigest()
    return f"{nonce}.{timestamp}.{mac}"


def _verify_workbench_cookie_token(cookie_value: str) -> bool:
    """Statelessly verify a workbench auth cookie value.

    Parses ``{nonce}.{timestamp}.{hmac_hex}``, checks the timestamp is within
    ``_WORKBENCH_COOKIE_MAX_AGE``, and verifies the HMAC.
    """
    try:
        parts = cookie_value.split(".")
        if len(parts) != 3:
            return False
        nonce, timestamp_str, mac = parts
        ts = int(timestamp_str)
        now = int(time.time())
        if ts > now or now - ts > _WORKBENCH_COOKIE_MAX_AGE:
            return False
        key_bytes = get_api_key().encode("utf-8")
        material = b"\x00".join(
            [_WORKBENCH_COOKIE_CONTEXT, nonce.encode("utf-8"), timestamp_str.encode("utf-8")]
        )
        expected = hmac.new(key_bytes, material, hashlib.sha256).hexdigest()
        return hmac.compare_digest(mac, expected)
    except (ValueError, TypeError):
        return False


def issue_local_workbench_cookies(response: Response, *, secure: bool = False) -> None:
    """Attach HttpOnly SameSite=Strict auth + CSRF double-submit cookies for local SPA bootstrap.

    The auth cookie now includes a per-issuance session nonce, an issuance timestamp,
    and an 8-hour max_age, so it is not static and expires. The CSRF token is an
    independent double-submit value (not cryptographically bound to the auth issuance).
    """
    session_nonce = secrets.token_urlsafe(16)
    wb_token = _derive_workbench_cookie_token(session_id=session_nonce)
    csrf_token = secrets.token_urlsafe(32)
    response.set_cookie(
        key=WORKBENCH_AUTH_COOKIE,
        value=wb_token,
        httponly=True,
        secure=secure,
        samesite="strict",
        path="/",
        max_age=_WORKBENCH_COOKIE_MAX_AGE,
    )
    response.set_cookie(
        key=WORKBENCH_CSRF_COOKIE,
        value=csrf_token,
        httponly=False,
        secure=secure,
        samesite="strict",
        path="/",
        max_age=_WORKBENCH_COOKIE_MAX_AGE,
    )


def clear_local_workbench_cookies(response: Response, *, secure: bool = False) -> None:
    """Expire stale or invalid local workbench auth + CSRF cookies so browsers do not stay wedged."""
    response.delete_cookie(
        key=WORKBENCH_AUTH_COOKIE,
        path="/",
        secure=secure,
        httponly=True,
        samesite="strict",
    )
    response.delete_cookie(
        key=WORKBENCH_CSRF_COOKIE,
        path="/",
        secure=secure,
        httponly=False,
        samesite="strict",
    )


def has_valid_workbench_cookie(request: Request) -> bool:
    """Return True when ``request`` carries a currently valid ``omniforge_wb_auth`` cookie."""
    cookie_token = request.cookies.get(WORKBENCH_AUTH_COOKIE)
    if not cookie_token:
        return False
    return _verify_workbench_cookie_token(cookie_token)


_loopback_workbench_bootstrap_enabled: bool = False


def is_loopback_host(host: str | None) -> bool:
    """Return True strictly for loopback hostnames/IPs (never 0.0.0.0, ::, or LAN addresses)."""
    if not host:
        return False
    cleaned = host.strip().strip("[]").lower()
    if not cleaned:
        return False
    if cleaned == "localhost":
        return True
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


def set_loopback_workbench_bootstrap(enabled: bool) -> bool:
    """Enable or disable loopback-only cookie bootstrap for interactive Workbench UI servers."""
    global _loopback_workbench_bootstrap_enabled
    prev = _loopback_workbench_bootstrap_enabled
    _loopback_workbench_bootstrap_enabled = bool(enabled)
    return prev


def _is_loopback_bootstrap_disabled_by_env() -> bool:
    raw = (os.getenv("OMNIFORGE_DISABLE_LOOPBACK_UI_BOOTSTRAP") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def should_auto_bootstrap_local_ui(request: Request | None = None) -> bool:
    """Return True when non-production local UI cookie bootstrap is permitted for ``request``."""
    if _is_production_env():
        return False
    if bool(get_settings().security.auto_bootstrap_local_ui):
        return True
    if _loopback_workbench_bootstrap_enabled and not _is_loopback_bootstrap_disabled_by_env():
        if request is None:
            return False
        if not is_loopback_host(request.url.hostname):
            return False
        client_host = request.client.host if request.client is not None else None
        if client_host is not None and not is_loopback_host(client_host):
            return False
        return True
    return False


def _authenticate_from_cookie(request: Request) -> Principal | None:
    """Verify HttpOnly workbench auth cookie plus X-CSRF-Token double-submit on mutations."""
    cookie_token = request.cookies.get(WORKBENCH_AUTH_COOKIE)
    if not cookie_token:
        return None
    if not _verify_workbench_cookie_token(cookie_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired workbench session cookie.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if request.method.upper() in _STATE_CHANGING_METHODS:
        csrf_cookie = request.cookies.get(WORKBENCH_CSRF_COOKIE) or ""
        csrf_header = request.headers.get("x-csrf-token") or ""
        if not csrf_cookie or not csrf_header or not hmac.compare_digest(csrf_cookie, csrf_header):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="CSRF verification failed: missing or mismatched X-CSRF-Token header.",
            )
    return Principal(
        subject="local_workbench_user",
        roles=["admin"],
        tenant_id=get_settings().default_tenant_id,
    )


# NOTE: ``async def`` on purpose. Sync dependencies execute in a worker threadpool with a
# copied context, so a contextvar set there would be invisible to the request task and to the
# sync path operation's own context copy. As an async dependency this runs in the request task
# itself; contextvars set here are then visible to sync path operations (anyio copies the
# current context into the worker thread) and to async ones directly.
async def get_current_principal(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_workbench_token: str | None = Header(default=None, alias="X-Workbench-Token"),
) -> Principal:
    """Authenticate callers via Bearer JWT, X-API-Key, or CSRF-verified local workbench cookie.

    Never falls back to an unauthenticated admin principal. Missing or invalid credentials always
    result in HTTP 401 Unauthorized. The resolved principal is stashed in a request-scoped
    contextvar so downstream consumers (e.g. the workbench session registry) can enforce
    ownership binding without route-signature changes.
    """
    principal = _resolve_current_principal(request, authorization=authorization, x_api_key=x_api_key, x_workbench_token=x_workbench_token)
    _request_principal_ctx.set(principal)
    return principal


def _resolve_current_principal(
    request: Request,
    *,
    authorization: str | None,
    x_api_key: str | None,
    x_workbench_token: str | None,
) -> Principal:
    # 1. Bearer JWT authentication
    if authorization:
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header must use 'Bearer <token>' scheme.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            return jwt_manager.verify_token(parts[1])
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired JWT token.",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

    # 2. API Key / Workbench Token header authentication
    api_candidate = x_api_key or x_workbench_token
    if api_candidate:
        if verify_api_key(api_candidate):
            return Principal(
                subject="api_key_principal",
                roles=["admin"],
                tenant_id=get_settings().default_tenant_id,
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3. Same-origin local workbench HttpOnly cookie + CSRF verification
    cookie_principal = _authenticate_from_cookie(request)
    if cookie_principal is not None:
        return cookie_principal

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=(
            "Authentication required. Provide a valid Bearer JWT token or X-API-Key header "
            "(for local browser UI sessions, set OMNIFORGE_SECURITY__AUTO_BOOTSTRAP_LOCAL_UI=true)."
        ),
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_workbench_auth(
    principal: Principal = Depends(get_current_principal),
) -> Principal:
    """FastAPI dependency protecting all /api/workbench/* routes."""
    return principal


def authorize_principal(principal: Principal, permission: str) -> None:
    """Enforce an action permission when it depends on the request payload."""
    try:
        access_controller.authorize(principal, permission)
    except AuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=exc.message) from exc


def require_permission(permission: str) -> Callable[..., Principal]:
    """FastAPI dependency enforcing RBAC/ABAC permission checks for protected endpoints."""

    def _checker(principal: Principal = Depends(get_current_principal)) -> Principal:
        authorize_principal(principal, permission)
        return principal

    return _checker
