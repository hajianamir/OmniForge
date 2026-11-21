"""Unit tests for the ASGI hardening middleware (413 body cap, 429 sliding-window rate limit)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from omniforge.config.settings import APISettings
from omniforge.core.security import RequestBodySizeLimitMiddleware, SlidingWindowRateLimitMiddleware


def _dummy_app(body_collector: list[bytes], status: int = 200):
    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        body = b""
        more = True
        while more:
            message = await receive()
            more = message.get("more_body", False)
            body += message.get("body", b"")
        body_collector.append(body)
        payload = json.dumps({"len": len(body)}).encode()
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]})
        await send({"type": "http.response.body", "body": payload})

    return app


def _scope(body_len: int | None, *, client: str = "127.0.0.1", path: str = "/upload", chunked_no_length: bool = False) -> dict[str, Any]:
    headers = [(b"content-type", b"application/octet-stream")]
    if body_len is not None and not chunked_no_length:
        headers.append((b"content-length", str(body_len).encode()))
    if chunked_no_length:
        headers.append((b"transfer-encoding", b"chunked"))
    return {"type": "http", "path": path, "headers": headers, "client": (client, 1234)}


def _receive_factory(chunks: list[bytes]):
    remaining = list(chunks)

    async def receive() -> dict[str, Any]:
        if remaining:
            chunk = remaining.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(remaining)}
        return {"type": "http.request", "body": b"", "more_body": False}

    return receive


def _send_collector():
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    return sent, send


def _status(sent: list[dict[str, Any]]) -> int:
    return next(int(m["status"]) for m in sent if m["type"] == "http.response.start")


# ------------------------- Body size limit -------------------------

def test_small_body_passes() -> None:
    collected: list[bytes] = []
    app = RequestBodySizeLimitMiddleware(_dummy_app(collected), max_body_bytes=16)
    sent, send = _send_collector()
    asyncio.run(app(_scope(5), _receive_factory([b"12345"]), send))
    assert _status(sent) == 200
    assert collected == [b"12345"]


def test_content_length_precheck_rejects_before_reading() -> None:
    collected: list[bytes] = []
    app = RequestBodySizeLimitMiddleware(_dummy_app(collected), max_body_bytes=16)
    sent, send = _send_collector()

    async def refuse_receive() -> dict[str, Any]:  # must never be called on 413 fast path
        raise AssertionError("body must not be consumed when Content-Length exceeds the cap")

    asyncio.run(app(_scope(10_000), refuse_receive, send))
    assert _status(sent) == 413
    assert collected == []


def test_chunked_body_streaming_enforcement_rejects() -> None:
    collected: list[bytes] = []
    app = RequestBodySizeLimitMiddleware(_dummy_app(collected), max_body_bytes=16)
    sent, send = _send_collector()
    asyncio.run(
        app(
            _scope(None, chunked_no_length=True),
            _receive_factory([b"0123456789", b"abcdefghij"]),  # 20 bytes total > 16 cap
            send,
        )
    )
    assert _status(sent) == 413
    assert collected == []  # downstream handler never observed the payload


def test_under_cap_chunked_body_passes() -> None:
    collected: list[bytes] = []
    app = RequestBodySizeLimitMiddleware(_dummy_app(collected), max_body_bytes=16)
    sent, send = _send_collector()
    asyncio.run(app(_scope(None, chunked_no_length=True), _receive_factory([b"abc", b"def"]), send))
    assert _status(sent) == 200
    assert collected == [b"abcdef"]


def test_exact_cap_boundary_passes() -> None:
    collected: list[bytes] = []
    app = RequestBodySizeLimitMiddleware(_dummy_app(collected), max_body_bytes=4)
    sent, send = _send_collector()
    asyncio.run(app(_scope(4), _receive_factory([b"1234"]), send))
    assert _status(sent) == 200


# ------------------------- Rate limiting -------------------------

class _FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_rate_limited_app(limit: int, clock: _FakeClock):
    collected: list[bytes] = []
    limited = SlidingWindowRateLimitMiddleware(
        _dummy_app(collected),
        max_requests_per_minute=limit,
        window_seconds=60.0,
        clock=clock,
    )
    return limited, collected


def _call(app: Any, *, client: str = "127.0.0.1", path: str = "/x") -> list[dict[str, Any]]:
    sent, send = _send_collector()
    asyncio.run(app(_scope(0, client=client, path=path), _receive_factory([b""]), send))
    return sent


def test_requests_over_limit_get_429_with_retry_after() -> None:
    clock = _FakeClock()
    app, collected = _make_rate_limited_app(3, clock)
    for _ in range(3):
        sent = _call(app)
        assert _status(sent) == 200
    sent = _call(app)
    assert _status(sent) == 429
    headers = dict(sent[0]["headers"])
    assert b"retry-after" in headers
    assert len(collected) == 3  # 4th request never reached the app
    body = json.loads(sent[1]["body"])
    assert body["error_type"] == "TooManyRequests"


def test_window_slides_and_admits_again() -> None:
    clock = _FakeClock()
    app, _ = _make_rate_limited_app(2, clock)
    assert _status(_call(app)) == 200
    assert _status(_call(app)) == 200
    assert _status(_call(app)) == 429
    clock.advance(61.0)
    assert _status(_call(app)) == 200


def test_clients_are_isolated_by_ip() -> None:
    clock = _FakeClock()
    app, _ = _make_rate_limited_app(1, clock)
    assert _status(_call(app, client="10.0.0.1")) == 200
    assert _status(_call(app, client="10.0.0.2")) == 200  # different client not throttled
    assert _status(_call(app, client="10.0.0.1")) == 429  # original client throttled


def test_health_path_is_exempt() -> None:
    clock = _FakeClock()
    app, _ = _make_rate_limited_app(1, clock)
    assert _status(_call(app, path="/health")) == 200
    assert _status(_call(app, path="/api/v1/health")) == 200
    assert _status(_call(app, path="/health")) == 200  # exempt paths never throttle


def test_zero_disables_limiting() -> None:
    clock = _FakeClock()
    app, _ = _make_rate_limited_app(0, clock)
    for _ in range(50):
        assert _status(_call(app)) == 200


# ------------------------- Settings wiring -------------------------

def test_api_settings_defaults_are_secure_and_sane() -> None:
    cfg = APISettings()
    assert cfg.max_request_body_bytes == 268_435_456  # 256 MB
    assert cfg.rate_limit_per_minute == 300


def test_api_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIFORGE_API__RATE_LIMIT_PER_MINUTE", "12")
    monkeypatch.setenv("OMNIFORGE_API__MAX_REQUEST_BODY_BYTES", "1048576")
    from omniforge.config.settings import OmniForgeSettings

    cfg = OmniForgeSettings()
    assert cfg.api.rate_limit_per_minute == 12
    assert cfg.api.max_request_body_bytes == 1_048_576


def test_workbench_app_wires_both_middlewares() -> None:
    from omniforge.api.main import app

    names = [getattr(m.cls, "__name__", str(m.cls)) for m in app.user_middleware]
    assert "RequestBodySizeLimitMiddleware" in names
    assert "SlidingWindowRateLimitMiddleware" in names
