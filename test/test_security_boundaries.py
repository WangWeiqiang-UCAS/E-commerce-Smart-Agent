"""Security regression tests for the prototype boundary."""
import os

import pytest
from fastapi import HTTPException


for key, value in {
    "PROJECT_NAME": "test",
    "API_V1_STR": "/api/v1",
    "POSTGRES_SERVER": "localhost",
    "POSTGRES_USER": "postgres",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_DB": "test",
    "POSTGRES_PORT": "5432",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "OPENAI_BASE_URL": "http://localhost",
    "OPENAI_API_KEY": "test",
    "SECRET_KEY": "test-secret",
}.items():
    os.environ.setdefault(key, value)

from app.core.config import settings
from app.core.security import create_access_token, get_admin_user_id, require_bff_secret


def test_normal_user_cannot_access_admin_dependency():
    token = create_access_token(user_id=10, is_admin=False)

    with pytest.raises(HTTPException) as error:
        get_admin_user_id(token)

    assert error.value.status_code == 403


def test_admin_token_is_accepted_by_admin_dependency():
    token = create_access_token(user_id=20, is_admin=True)

    assert get_admin_user_id(token) == 20


def test_bff_secret_blocks_direct_access_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "BFF_SHARED_SECRET", "shared-secret")
    with pytest.raises(HTTPException) as error:
        require_bff_secret(None)
    assert error.value.status_code == 403
    require_bff_secret("shared-secret")
