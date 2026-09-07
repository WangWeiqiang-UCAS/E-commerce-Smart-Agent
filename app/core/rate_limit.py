"""Distributed, privacy-preserving fixed-window rate limiting for chat APIs."""
import hashlib
import logging
import time
from collections import defaultdict
from threading import Lock

from fastapi import HTTPException, status
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.core.config import settings

LOGGER = logging.getLogger(__name__)
_lock = Lock()
_local_hits: dict[str, list[float]] = defaultdict(list)
_WINDOW_SECONDS = 60
_redis_client: Redis | None = None
_redis_url: str | None = None
_INCREMENT_WITH_TTL = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return count
"""


def _reject() -> None:
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Rate limit exceeded. Please retry shortly.",
    )


def _rate_key(key: str) -> str:
    """Keep customer identifiers out of Redis key names and diagnostic output."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"agent:rate-limit:{digest}:{int(time.time() // _WINDOW_SECONDS)}"


def _local_check(key: str, limit: int) -> None:
    now = time.monotonic()
    with _lock:
        recent = [timestamp for timestamp in _local_hits.get(key, []) if now - timestamp < _WINDOW_SECONDS]
        if len(recent) >= limit:
            _local_hits[key] = recent
            _reject()
        recent.append(now)
        _local_hits[key] = recent


def _redis() -> Redis:
    global _redis_client, _redis_url
    redis_url = settings.REDIS_URL
    if _redis_client is None or _redis_url != redis_url:
        _redis_client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=settings.RATE_LIMIT_REDIS_TIMEOUT_SECONDS,
            socket_timeout=settings.RATE_LIMIT_REDIS_TIMEOUT_SECONDS,
        )
        _redis_url = redis_url
    return _redis_client


async def check_rate_limit(key: str) -> None:
    """Enforce a per-principal limit across replicas, with safe local fallback.

    The first increment in each Redis window receives a TTL slightly longer
    than the window. Redis failures must not turn a support outage into a
    checkout outage, so the existing per-process limiter remains a fallback.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return
    limit = max(settings.RATE_LIMIT_PER_MINUTE, 1)
    redis_key = _rate_key(key)
    try:
        count = await _redis().eval(_INCREMENT_WITH_TTL, 1, redis_key, _WINDOW_SECONDS + 1)
        if count > limit:
            _reject()
    except HTTPException:
        raise
    except RedisError:
        LOGGER.warning("Redis rate limiter unavailable; using local fallback")
        _local_check(key, limit)
