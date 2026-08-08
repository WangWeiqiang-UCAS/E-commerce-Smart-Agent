import os

for key, value in {
    "PROJECT_NAME": "test", "API_V1_STR": "/api/v1", "POSTGRES_SERVER": "localhost",
    "POSTGRES_USER": "postgres", "POSTGRES_PASSWORD": "test", "POSTGRES_DB": "test",
    "POSTGRES_PORT": "5432", "REDIS_HOST": "localhost", "REDIS_PORT": "6379",
    "OPENAI_BASE_URL": "http://localhost", "OPENAI_API_KEY": "test", "SECRET_KEY": "test-secret",
}.items():
    os.environ.setdefault(key, value)

from app.events.commerce_consumer import project_event_card
from app.models.message import MessageType


def test_after_sale_event_projects_to_system_refund_card():
    card = project_event_card({"eventId": "event-1", "eventType": "after_sale.created", "schemaVersion": 1, "aggregateId": "42", "traceId": "trace-1", "occurredAt": "2026-08-08T00:00:00Z"})
    assert card is not None
    assert card.message_type == MessageType.REFUND_CARD
    assert card.content["after_sale_id"] == "42"


def test_unrelated_event_does_not_create_support_card():
    assert project_event_card({"eventId": "event-1", "eventType": "product.changed", "schemaVersion": 1, "aggregateId": "42", "occurredAt": "2026-08-08T00:00:00Z"}) is None
