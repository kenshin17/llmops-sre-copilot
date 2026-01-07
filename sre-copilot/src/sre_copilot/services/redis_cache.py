from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import redis

from sre_copilot.config import Settings
from sre_copilot.utils.logger import get_logger

logger = get_logger(__name__)


class RedisCache:
    """Minimal Redis-backed JSON cache for retrieval responses."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: redis.Redis | None = None
        if not settings.redis_url:
            return
        try:
            client = redis.from_url(settings.redis_url)
            client.ping()
            self.client = client
            logger.info("Redis cache enabled at %s", settings.redis_url)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Redis cache unavailable, continuing without cache: %s", exc)

    async def get(self, key: str) -> dict[str, Any] | None:
        if not self.client:
            return None
        try:
            data = await asyncio.to_thread(self.client.get, key)
            if not data:
                return None
            return json.loads(data)
        except Exception as exc:  # pragma: no cover
            logger.warning("Cache get failed for key=%s: %s", key, exc)
            return None

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int | None = None) -> None:
        if not self.client:
            return
        try:
            payload = json.dumps(value)
            ttl = ttl_seconds or self.settings.cache_ttl_seconds
            await asyncio.to_thread(self.client.setex, key, ttl, payload)
        except Exception as exc:  # pragma: no cover
            logger.warning("Cache set failed for key=%s: %s", key, exc)

    @staticmethod
    def cache_key(query: str, top_k: int | None = None) -> str:
        digest = hashlib.sha1(query.encode("utf-8")).hexdigest()
        if top_k is None:
            return f"retrieval:{digest}"
        return f"retrieval:k{top_k}:{digest}"
