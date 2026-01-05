import time
import time
from collections import defaultdict
from typing import Any

import redis
from fastapi import Depends, HTTPException, Request, status

from sre_copilot.config import Settings, get_settings
from sre_copilot.utils.logger import get_logger

logger = get_logger(__name__)


class InMemoryRateLimiter:
    """
    Minimal token bucket per identifier (client IP by default).
    Not production-ready but good enough for local demos.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self.allowance: dict[str, float] = defaultdict(lambda: float(limit_per_minute))
        self.last_check: dict[str, float] = defaultdict(time.time)

    def _refill(self, key: str) -> None:
        current = time.time()
        time_passed = current - self.last_check[key]
        self.last_check[key] = current
        self.allowance[key] = min(self.limit, self.allowance[key] + time_passed * (self.limit / 60))

    def is_allowed(self, key: str) -> bool:
        self._refill(key)
        if self.allowance[key] < 1.0:
            return False
        self.allowance[key] -= 1.0
        return True


class RedisRateLimiter:
    """
    Fixed-window counter in Redis per identifier.
    Key = ratelimit:{identifier}:{epoch_minute}
    """

    def __init__(self, client: redis.Redis, limit_per_minute: int) -> None:
        self.client = client
        self.limit = limit_per_minute

    def is_allowed(self, key: str) -> bool:
        epoch_minute = int(time.time() // 60)
        redis_key = f"ratelimit:{key}:{epoch_minute}"
        count = self.client.incr(redis_key)
        if count == 1:
            self.client.expire(redis_key, 60)
        return count <= self.limit


def get_rate_limiter(settings: Settings):
    global _rate_limiter
    if "_rate_limiter" in globals():
        return globals()["_rate_limiter"]

    if settings.redis_url:
        try:
            client = redis.from_url(settings.redis_url)
            # quick ping to validate connectivity
            client.ping()
            globals()["_rate_limiter"] = RedisRateLimiter(client, settings.rate_limit_per_minute)
            logger.info("Using Redis rate limiter at %s", settings.redis_url)
            return globals()["_rate_limiter"]
        except Exception as exc:  # pragma: no cover - network side effect
            logger.warning("Redis unavailable, falling back to in-memory limiter: %s", exc)

    globals()["_rate_limiter"] = InMemoryRateLimiter(settings.rate_limit_per_minute)
    return globals()["_rate_limiter"]


async def enforce_rate_limit(
    request: Request, settings: Settings = Depends(get_settings)
) -> dict[str, Any]:
    limiter = get_rate_limiter(settings)
    client_ip = request.client.host if request.client else "unknown"
    if not limiter.is_allowed(client_ip):
        logger.warning("Rate limit exceeded for %s", client_ip)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests; please retry shortly.",
        )
    return {"client_ip": client_ip}
