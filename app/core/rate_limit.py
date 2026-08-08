"""Fixed-window rate limiting for chat endpoints.

The in-process limiter is deterministic and easy to reason about for a single
replica. Multi-replica deployments should swap the implementation for a Redis
token bucket; the dependency signature stays the same.
"""
import time
from collections import defaultdict
from threading import Lock

from fastapi import HTTPException, status

from app.core.config import settings

_lock = Lock()
_hits: dict[str, list[float]] = defaultdict(list)
_WINDOW_SECONDS = 60.0


def check_rate_limit(key: str) -> None:
    if not settings.RATE_LIMIT_ENABLED:
        return
    limit = max(settings.RATE_LIMIT_PER_MINUTE, 1)
    now = time.monotonic()
    with _lock:
        recent = [timestamp for timestamp in _hits[key] if now - timestamp < _WINDOW_SECONDS]
        if len(recent) >= limit:
            _hits[key] = recent
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please retry shortly.",
            )
        recent.append(now)
        _hits[key] = recent
