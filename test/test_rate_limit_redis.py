import pytest
from fastapi import HTTPException
from redis.exceptions import ConnectionError

from app.core import rate_limit


class _FakeRedis:
    def __init__(self):
        self.counts = {}
        self.expiries = []

    async def eval(self, _script, _keys, key, seconds):
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] == 1:
            self.expiries.append((key, seconds))
        return self.counts[key]


@pytest.mark.asyncio
async def test_redis_limiter_uses_hashed_key_and_expiry(monkeypatch):
    from app.core.config import settings

    fake = _FakeRedis()
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 2)
    monkeypatch.setattr(rate_limit, "_redis", lambda: fake)

    await rate_limit.check_rate_limit("chat:customer-42")
    await rate_limit.check_rate_limit("chat:customer-42")

    key = next(iter(fake.counts))
    assert "customer-42" not in key
    assert fake.expiries == [(key, 61)]


@pytest.mark.asyncio
async def test_redis_failure_falls_back_to_local_limiter(monkeypatch):
    from app.core.config import settings

    class _UnavailableRedis:
        async def eval(self, *_args):
            raise ConnectionError("unavailable")

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 1)
    monkeypatch.setattr(rate_limit, "_redis", lambda: _UnavailableRedis())
    monkeypatch.setattr(rate_limit, "_local_hits", {})

    await rate_limit.check_rate_limit("fallback-customer")
    with pytest.raises(HTTPException) as error:
        await rate_limit.check_rate_limit("fallback-customer")
    assert error.value.status_code == 429
