"""Shared/multi-process rate limiter: Redis backend, resilience, settings wiring."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from omniforge.core.rate_limit import (
    InMemorySlidingWindowLimiter,
    RateLimitDecision,
    RedisSlidingWindowLimiter,
    ResilientRateLimiter,
    build_rate_limiter,
)


class _FakeRedis:
    """Minimal async stand-in implementing ZSET sliding-window semantics for rate limiting."""

    def __init__(self) -> None:
        self.zsets: dict[str, list[tuple[float, str]]] = {}
        self.fail = False

    async def eval(self, script: str, numkeys: int, key: str, window_ms: int, limit: int, now_ms: int, member: str):
        if self.fail:
            raise ConnectionError("redis is down")
        window_ms = int(window_ms)
        limit = int(limit)
        now_ms = int(now_ms)
        entries = [(ts, m) for ts, m in self.zsets.get(key, []) if ts > now_ms - window_ms]
        self.zsets[key] = entries
        if len(entries) >= limit:
            oldest = min(ts for ts, _ in entries)
            retry = max(1, (int(oldest + window_ms - now_ms) + 999) // 1000)
            return [0, retry, 0]
        entries.append((float(now_ms), member))
        self.zsets[key] = entries
        return [1, 0, limit - len(entries)]

    async def aclose(self) -> None:
        pass


class _FakeSettings:
    def __init__(self, backend: str = "auto", url: str | None = None) -> None:
        self.rate_limit_backend = backend
        self.rate_limit_redis_url = url


def _hit(limiter: Any, key: str = "1.2.3.4") -> RateLimitDecision:
    return asyncio.run(limiter.hit(key))


def test_in_memory_limiter_matches_legacy_semantics() -> None:
    limiter = InMemorySlidingWindowLimiter(max_requests=2, window_seconds=60.0, clock=_ManualClock())
    assert _hit(limiter)["allowed"] is True
    assert _hit(limiter)["allowed"] is True
    third = _hit(limiter)
    assert third["allowed"] is False
    assert third["retry_after"] >= 1


class _ManualClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_redis_limiter_shared_window_across_instances() -> None:
    """Two limiter instances backed by the same Redis enforce ONE shared budget."""
    fake = _FakeRedis()
    clock = _ManualClock()
    a = RedisSlidingWindowLimiter("redis://fake", max_requests=3, window_seconds=60.0, clock=clock, client=fake)
    b = RedisSlidingWindowLimiter("redis://fake", max_requests=3, window_seconds=60.0, clock=clock, client=fake)
    assert _hit(a, "client")["allowed"] is True
    assert _hit(b, "client")["allowed"] is True   # counts against the shared budget
    assert _hit(a, "client")["allowed"] is True
    denied = _hit(b, "client")                   # 4th hit on the shared window
    assert denied["allowed"] is False
    assert denied["retry_after"] >= 1


def test_redis_limiter_tracks_remaining() -> None:
    fake = _FakeRedis()
    clock = _ManualClock()
    limiter = RedisSlidingWindowLimiter("redis://fake", max_requests=2, window_seconds=60.0, clock=clock, client=fake)
    assert _hit(limiter)["remaining"] == 1
    assert _hit(limiter)["remaining"] == 0


def test_redis_limiter_window_slides() -> None:
    fake = _FakeRedis()
    clock = _ManualClock()
    limiter = RedisSlidingWindowLimiter("redis://fake", max_requests=1, window_seconds=60.0, clock=clock, client=fake)
    assert _hit(limiter)["allowed"] is True
    assert _hit(limiter)["allowed"] is False
    clock.now += 61.0
    assert _hit(limiter)["allowed"] is True


def test_resilient_limiter_degrades_to_memory_on_backend_failure() -> None:
    fake = _FakeRedis()
    fake.fail = True
    clock = _ManualClock()
    primary = RedisSlidingWindowLimiter("redis://fake", max_requests=2, window_seconds=60.0, clock=clock, client=fake)
    fallback = InMemorySlidingWindowLimiter(max_requests=2, window_seconds=60.0, clock=clock)
    limiter = ResilientRateLimiter(primary, fallback)
    assert _hit(limiter)["allowed"] is True
    assert _hit(limiter)["allowed"] is True
    assert _hit(limiter)["allowed"] is False  # in-memory fallback still enforces the cap


def test_build_rate_limiter_memory_backend() -> None:
    limiter = build_rate_limiter(_FakeSettings("memory"), max_requests_per_minute=5)
    assert isinstance(limiter, InMemorySlidingWindowLimiter)


def test_build_rate_limiter_disabled_returns_none() -> None:
    assert build_rate_limiter(_FakeSettings("memory"), max_requests_per_minute=0) is None


def test_build_rate_limiter_unknown_backend_rejected() -> None:
    from omniforge.core.exceptions import DependencyMissingError

    with pytest.raises(DependencyMissingError):
        build_rate_limiter(_FakeSettings("bogus"), max_requests_per_minute=5)


def test_build_rate_limiter_redis_requires_url() -> None:
    from omniforge.core.exceptions import DependencyMissingError

    with pytest.raises(DependencyMissingError):
        build_rate_limiter(_FakeSettings("redis", None), max_requests_per_minute=5)


def test_build_rate_limiter_auto_without_url_uses_memory() -> None:
    limiter = build_rate_limiter(_FakeSettings("auto", None), max_requests_per_minute=5)
    assert isinstance(limiter, InMemorySlidingWindowLimiter)


def test_redis_missing_url_raises_dependency_missing() -> None:
    from omniforge.core.exceptions import DependencyMissingError

    with pytest.raises(DependencyMissingError):
        RedisSlidingWindowLimiter("", max_requests=1)


def test_cap_denies_new_clients_without_evicting_active_windows() -> None:
    clock = _ManualClock()
    limiter = InMemorySlidingWindowLimiter(
        max_requests=2, max_tracked_keys=2, window_seconds=60, clock=clock,
    )
    assert _hit(limiter, "oldest")["allowed"]
    clock.now += 1
    assert _hit(limiter, "newer")["allowed"]
    assert _hit(limiter, "newer")["remaining"] == 0
    for index in range(1000):
        assert _hit(limiter, f"other-{index}") == {"allowed": False, "retry_after": 59, "remaining": 0}
    assert len(limiter._hits) == 2
    assert not _hit(limiter, "newer")["allowed"]
    clock.now += 59
    assert _hit(limiter, "new-client")["allowed"]
    assert len(limiter._hits) == 2
    assert not _hit(limiter, "newer")["allowed"]
    clock.now += 1
    assert _hit(limiter, "newer")["allowed"]
