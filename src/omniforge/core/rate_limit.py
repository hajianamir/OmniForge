"""Sliding-window rate limiting with pluggable backends (in-memory and Redis).

The default in-memory backend is per-process: fine for single-worker deployments,
and honest about it — behind a multi-process/multi-replica deployment each worker
enforces the limit independently, so the *effective* ceiling is N x the configured
value. Set ``OMNIFORGE_API__RATE_LIMIT_BACKEND=redis`` with
``OMNIFORGE_API__RATE_LIMIT_REDIS_URL`` to share one atomic sliding window across
all processes (requires the optional ``redis`` package, ``pip install omniforge[redis]``).

The Redis backend implements the window as a single Lua script (ZSET-based), so
check-and-record is atomic across processes — no race between the count check and
the insert. It uses Redis's clock for every replica. On Redis errors the limiter degrades to a per-process in-memory window
and logs the failure (availability first), while a *missing* redis package with
``backend=redis`` is a boot-time configuration error (fail fast, not silent).

``X-Forwarded-For`` is deliberately NOT trusted for keying (spoofable); keying is by
the ASGI client host. Behind a single trusted proxy, either raise the ceiling via
``OMNIFORGE_API__RATE_LIMIT_PER_MINUTE`` or run the Redis backend so replicas
share one budget.
"""

from __future__ import annotations

import math
import time
import uuid
from collections import OrderedDict, deque
from typing import Any, Protocol, TypedDict

from omniforge.core.exceptions import ConfigurationError, DependencyMissingError
from omniforge.core.logging import get_logger

try:  # optional dependency — only needed when backend=redis
    import redis.asyncio as _aioredis
except ImportError:  # pragma: no cover - environment dependent
    _aioredis = None

logger = get_logger(__name__)

_LUA_SLIDING_WINDOW = """
local key = KEYS[1]
local window_ms = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local now_ms = tonumber(ARGV[3])
local member = ARGV[4]

if now_ms == nil then
    -- Redis 5/6 support effects replication for scripts that read server time.
    redis.replicate_commands()
    local server_time = redis.call('TIME')
    now_ms = tonumber(server_time[1]) * 1000 + math.floor(tonumber(server_time[2]) / 1000)
end
redis.call('ZREMRANGEBYSCORE', key, '-inf', now_ms - window_ms)
local count = redis.call('ZCARD', key)
if count >= limit then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_after = 1
    if oldest[2] then
        retry_after = math.max(1, math.ceil((tonumber(oldest[2]) + window_ms - now_ms) / 1000))
    end
    return {0, retry_after, 0}
end
redis.call('ZADD', key, now_ms, member)
redis.call('PEXPIRE', key, window_ms)
return {1, 0, limit - count - 1}
"""


class RateLimitDecision(TypedDict):
    allowed: bool
    retry_after: int
    remaining: int


class RateLimiter(Protocol):
    async def hit(self, key: str) -> RateLimitDecision: ...


class InMemorySlidingWindowLimiter:
    """Bound tracked clients without resetting active windows at capacity."""

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: float = 60.0,
        clock: Any = None,
        max_tracked_keys: int = 10_000,
    ) -> None:
        self.max_requests = int(max_requests)
        self.window_seconds = float(window_seconds)
        self._clock = clock or time.monotonic
        self._max_tracked_keys = int(max_tracked_keys)
        if self._max_tracked_keys < 1 or not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ConfigurationError("Rate limiter window and tracked-key capacity must be positive and finite.")
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    async def hit(self, key: str) -> RateLimitDecision:
        if self.max_requests <= 0:
            return RateLimitDecision(allowed=True, retry_after=0, remaining=0)
        now = self._clock()
        cutoff = now - self.window_seconds
        hits = self._hits.get(key)
        if hits is None:
            while len(self._hits) >= self._max_tracked_keys:
                oldest_key, oldest_hits = next(iter(self._hits.items()))
                if oldest_hits[-1] > cutoff:
                    retry_after = max(1, math.ceil(oldest_hits[-1] - cutoff))
                    return RateLimitDecision(allowed=False, retry_after=retry_after, remaining=0)
                del self._hits[oldest_key]
            hits = self._hits[key] = deque()
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self.max_requests:
            retry_after = max(1, math.ceil(hits[0] - cutoff))
            return RateLimitDecision(allowed=False, retry_after=retry_after, remaining=0)
        hits.append(now)
        self._hits.move_to_end(key)
        return RateLimitDecision(allowed=True, retry_after=0, remaining=self.max_requests - len(hits))


class RedisSlidingWindowLimiter:
    """Cross-process sliding window backed by Redis (atomic Lua script)."""

    def __init__(
        self,
        redis_url: str,
        *,
        max_requests: int,
        window_seconds: float = 60.0,
        key_prefix: str = "omniforge:ratelimit",
        clock: Any = None,
        client: Any = None,
    ) -> None:
        if not redis_url and client is None:
            raise DependencyMissingError(
                "OMNIFORGE_API__RATE_LIMIT_REDIS_URL is required when rate_limit_backend='redis'.",
                suggestions=["Set OMNIFORGE_API__RATE_LIMIT_REDIS_URL=redis://host:6379/0"],
            )
        self.max_requests = int(max_requests)
        self.window_seconds = float(window_seconds)
        self.key_prefix = key_prefix.rstrip(":")
        self._clock = clock
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            if _aioredis is None:
                raise DependencyMissingError(
                    "rate_limit_backend='redis' requires the optional 'redis' package.",
                    suggestions=["pip install 'omniforge[redis]'  (or pip install redis>=5)"],
                )
            self._client = _aioredis.from_url(redis_url, decode_responses=True)
            self._owns_client = True

    async def hit(self, key: str) -> RateLimitDecision:
        now_ms = "" if self._clock is None else str(int(self._clock() * 1000))
        window_ms = int(self.window_seconds * 1000)
        full_key = f"{self.key_prefix}:{key}"
        member = uuid.uuid4().hex
        result = await self._client.eval(
            _LUA_SLIDING_WINDOW, 1, full_key, window_ms, self.max_requests, now_ms, member
        )
        allowed = bool(int(result[0]))
        retry_after = int(result[1])
        remaining = int(result[2])
        return RateLimitDecision(allowed=allowed, retry_after=retry_after, remaining=remaining)

    async def aclose(self) -> None:
        if self._owns_client and hasattr(self._client, "aclose"):
            await self._client.aclose()


class ResilientRateLimiter:
    """Redis-first limiter that degrades to per-process in-memory on backend errors.

    A Redis outage must not take the API down: on any backend error we log and serve
    from a local in-memory window (single-process protection, honest degradation).
    Configuration errors (missing package/URL) raise at construction instead.
    """

    def __init__(self, primary: RateLimiter, fallback: InMemorySlidingWindowLimiter) -> None:
        self._primary = primary
        self._fallback = fallback
        self._degraded_logged = False

    async def hit(self, key: str) -> RateLimitDecision:
        try:
            decision = await self._primary.hit(key)
        except Exception as exc:
            if not self._degraded_logged:
                logger.error(
                    "rate_limiter_backend_degraded",
                    error_type=type(exc).__name__,
                    note="Redis rate-limit backend failing; serving per-process in-memory window until it recovers.",
                )
                self._degraded_logged = True
            return await self._fallback.hit(key)
        if self._degraded_logged:
            logger.info("rate_limiter_backend_recovered")
            self._degraded_logged = False
        return decision

    async def aclose(self) -> None:
        close_primary = getattr(self._primary, "aclose", None)
        if close_primary is not None:
            await close_primary()


def build_rate_limiter(
    settings: Any,
    *,
    max_requests_per_minute: int,
    window_seconds: float = 60.0,
    clock: Any = None,
) -> RateLimiter | None:
    """Construct the configured limiter, or ``None`` when limiting is disabled (0)."""
    if int(max_requests_per_minute) <= 0:
        return None
    backend = str(getattr(settings, "rate_limit_backend", "auto") or "auto").strip().lower()
    redis_url = getattr(settings, "rate_limit_redis_url", None)
    if backend not in {"auto", "memory", "redis"}:
        raise DependencyMissingError(
            f"Unknown OMNIFORGE_API__RATE_LIMIT_BACKEND={backend!r} (expected 'auto', 'memory' or 'redis').",
        )
    memory = InMemorySlidingWindowLimiter(
        max_requests=max_requests_per_minute, window_seconds=window_seconds, clock=clock
    )
    if backend == "memory":
        return memory
    if backend == "redis":
        primary = RedisSlidingWindowLimiter(
            str(redis_url or ""),
            max_requests=max_requests_per_minute,
            window_seconds=window_seconds,
            clock=clock,
        )
        return ResilientRateLimiter(primary, memory)
    # auto: use redis when a URL is configured and the package is importable, else memory.
    if redis_url and _aioredis is not None:
        try:
            primary = RedisSlidingWindowLimiter(
                str(redis_url),
                max_requests=max_requests_per_minute,
                window_seconds=window_seconds,
                clock=clock,
            )
            return ResilientRateLimiter(primary, memory)
        except Exception as exc:
            logger.warning("rate_limiter_auto_redis_unavailable", error_type=type(exc).__name__)
            return memory
    if redis_url and _aioredis is None:
        logger.warning(
            "rate_limiter_auto_redis_package_missing",
            note="OMNIFORGE_API__RATE_LIMIT_REDIS_URL set but 'redis' package not installed; using in-memory limiter.",
        )
    return memory
