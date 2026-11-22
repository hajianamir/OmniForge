"""Structured enterprise logging with automatic secret redaction and trace correlation."""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from omniforge.core.constants import SENSITIVE_KEY_PATTERNS

try:
    import structlog

    _HAS_STRUCTLOG = True
except ImportError:
    structlog = None
    _HAS_STRUCTLOG = False

_TRACE_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "omniforge_trace_id", default=""
)
_TENANT_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "omniforge_tenant_id", default="default-tenant"
)

_SECRET_VALUE_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9\-._~+/]+=*"),
    re.compile(r"(?i)(sk-[A-Za-z0-9]{16,})"),
    re.compile(r"(?i)(ghp_[A-Za-z0-9]{20,})"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
)

_CONFIGURED = False


def get_trace_id() -> str:
    """Return the active trace identifier or generate one if unset."""
    current = _TRACE_ID_CTX.get()
    if not current:
        current = str(uuid4())
        _TRACE_ID_CTX.set(current)
    return current


def set_trace_id(trace_id: str | None = None) -> str:
    """Bind a trace identifier to the current execution context.

    Args:
        trace_id: Explicit trace ID or ``None`` to generate a fresh UUID4.

    Returns:
        The bound trace ID.
    """
    resolved = trace_id or str(uuid4())
    _TRACE_ID_CTX.set(resolved)
    return resolved


def set_tenant_context(tenant_id: str) -> None:
    """Bind a tenant identifier to the current execution context."""
    _TENANT_ID_CTX.set(tenant_id)


def get_tenant_context() -> str:
    """Return the active tenant identifier for the current context."""
    return _TENANT_ID_CTX.get()


def redact_sensitive_value(value: Any, key_name: str = "") -> Any:
    """Recursively redact secrets, tokens, and PII patterns from log payloads.

    Args:
        value: Arbitrary log field value.
        key_name: Key name associated with ``value``.

    Returns:
        Redacted copy safe for log persistence.
    """
    lowered_key = key_name.lower()
    if lowered_key and any(pat in lowered_key for pat in SENSITIVE_KEY_PATTERNS):
        return "***REDACTED***"

    if isinstance(value, dict):
        return {str(k): redact_sensitive_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        redacted_items = [redact_sensitive_value(item) for item in value]
        return tuple(redacted_items) if isinstance(value, tuple) else redacted_items
    if isinstance(value, str):
        sanitized = value
        for regex in _SECRET_VALUE_REGEXES:
            sanitized = regex.sub("***REDACTED***", sanitized)
        return sanitized
    return value


def _structlog_redact_processor(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that injects trace/tenant IDs and redacts secrets."""
    event_dict.setdefault("trace_id", get_trace_id())
    event_dict.setdefault("tenant_id", get_tenant_context())
    return cast(dict[str, Any], redact_sensitive_value(event_dict))


class RedactingJsonFormatter(logging.Formatter):
    """Standard-library JSON log formatter with redaction and trace correlation."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_sensitive_value(record.getMessage()),
            "trace_id": get_trace_id(),
            "tenant_id": get_tenant_context(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: str | Path | None = None,
    redact_secrets: bool = True,
) -> None:
    """Configure platform-wide structured logging once (idempotent).

    Args:
        level: Log level name (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
        log_format: ``json`` or ``console``.
        log_file: Optional file path for persisted logs.
        redact_secrets: Whether to apply automatic secret redaction.
    """
    global _CONFIGURED
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger("omniforge")
    root_logger.setLevel(numeric_level)
    root_logger.propagate = False
    root_logger.handlers.clear()

    use_structlog = _HAS_STRUCTLOG and structlog is not None
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)
    if use_structlog:
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
    elif log_format.lower() == "json":
        stream_handler.setFormatter(RedactingJsonFormatter())
    else:
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
    root_logger.addHandler(stream_handler)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter("%(message)s") if use_structlog else RedactingJsonFormatter()
        )
        root_logger.addHandler(file_handler)

    if use_structlog and structlog is not None:
        processors: list[Any] = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
        ]
        if redact_secrets:
            processors.append(_structlog_redact_processor)
        processors.extend(
            [
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
                if log_format.lower() == "json"
                else structlog.dev.ConsoleRenderer(),
            ]
        )
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=False,
        )

    _CONFIGURED = True


class OmniForgeLoggerAdapter:
    """Uniform structured logger facade compatible with both structlog and stdlib."""

    def __init__(self, name: str) -> None:
        self._name = name
        if not _CONFIGURED:
            configure_logging()
        self._std_logger = logging.getLogger(name)
        self._struct_logger = (
            structlog.get_logger(name) if _HAS_STRUCTLOG and structlog is not None else None
        )

    def _emit(self, level: str, event: str, **kwargs: Any) -> None:
        safe_kwargs = redact_sensitive_value(kwargs)
        safe_event = str(redact_sensitive_value(event))
        if self._struct_logger is not None:
            fn = getattr(self._struct_logger, level)
            fn(safe_event, **safe_kwargs)
        else:
            fn = getattr(self._std_logger, level)
            extra_str = f" | {json.dumps(safe_kwargs, default=str)}" if safe_kwargs else ""
            fn(f"{safe_event}{extra_str}")

    def debug(self, event: str, **kwargs: Any) -> None:
        """Log a debug event."""
        self._emit("debug", event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        """Log an informational event."""
        self._emit("info", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        """Log a warning event."""
        self._emit("warning", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        """Log an error event."""
        self._emit("error", event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        """Log a critical event."""
        self._emit("critical", event, **kwargs)


def get_logger(name: str = "omniforge") -> OmniForgeLoggerAdapter:
    """Return a structured logger bound under the canonical ``omniforge`` hierarchy."""
    cleaned = (name or "omniforge").strip()
    if cleaned == "omniforge" or cleaned.startswith("omniforge."):
        canonical = cleaned
    elif cleaned == "uap":
        canonical = "omniforge"
    elif cleaned.startswith("uap."):
        canonical = f"omniforge.{cleaned[4:]}"
    else:
        canonical = f"omniforge.{cleaned}"
    return OmniForgeLoggerAdapter(canonical)
