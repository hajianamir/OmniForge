"""Exercise the real Lua limiter through independent Redis connections."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from uuid import uuid4

import pytest

from omniforge.core import rate_limit


@pytest.mark.integration
def test_live_redis_shares_one_window_despite_worker_clock_skew(monkeypatch: pytest.MonkeyPatch) -> None:
    url = os.environ.get("OMNIFORGE_TEST_REDIS_URL")
    if not url:
        pytest.skip("OMNIFORGE_TEST_REDIS_URL is required for live Redis verification")
    import redis.asyncio as redis

    prefix = f"omniforge:test:{uuid4().hex}"
    key = f"{prefix}:client"
    limit = 7
    window_seconds = 2
    monkeypatch.setattr(rate_limit, "time", SimpleNamespace(time=lambda: 10_000.0))
    worker_a = rate_limit.RedisSlidingWindowLimiter(
        url, max_requests=limit, key_prefix=prefix, window_seconds=window_seconds,
    )
    monkeypatch.setattr(rate_limit, "time", SimpleNamespace(time=lambda: 20_000.0))
    worker_b = rate_limit.RedisSlidingWindowLimiter(
        url, max_requests=limit, key_prefix=prefix, window_seconds=window_seconds,
    )

    async def verify() -> None:
        async with redis.from_url(url) as observer:
            try:
                seconds, microseconds = await observer.time()
                start_ms = seconds * 1000 + microseconds // 1000
                decisions = await asyncio.gather(
                    *(worker.hit("client") for worker in [worker_a, worker_b] * (limit * 2))
                )
                accepted = sorted(decision["remaining"] for decision in decisions if decision["allowed"])
                assert accepted == list(range(limit))
                scores = await observer.zrange(key, 0, -1, withscores=True)
                assert len(scores) == limit
                assert all(score >= start_ms for _, score in scores)
                assert 0 < await observer.pttl(key) <= window_seconds * 1000
                extra_hit = await worker_b.hit("client")
                assert not extra_hit["allowed"]
                assert 0 < extra_hit["retry_after"] <= window_seconds

                async with asyncio.timeout(window_seconds + 3):
                    while await observer.exists(key):
                        await asyncio.sleep(0.1)
                assert await worker_a.hit("client") == {
                    "allowed": True, "retry_after": 0, "remaining": limit - 1,
                }
            finally:
                await worker_a.aclose()
                await worker_b.aclose()
                await observer.delete(key)

    asyncio.run(verify())
