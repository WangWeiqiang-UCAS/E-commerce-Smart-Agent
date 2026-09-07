"""Deterministic non-secret settings for unit and contract tests."""
import os

for key, value in {
    "PROJECT_NAME": "test-agent",
    "API_V1_STR": "/api/v1",
    "POSTGRES_SERVER": "localhost",
    "POSTGRES_USER": "postgres",
    "POSTGRES_PASSWORD": "test-password",
    "POSTGRES_DB": "agent_test",
    "POSTGRES_PORT": "5432",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "OPENAI_BASE_URL": "http://localhost",
    "OPENAI_API_KEY": "test-key",
    "SECRET_KEY": "test-secret",
    "FIELD_ENCRYPTION_KEY": "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=",
}.items():
    os.environ.setdefault(key, value)
