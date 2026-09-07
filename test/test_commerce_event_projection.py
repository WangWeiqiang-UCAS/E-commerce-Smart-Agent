import os

for key, value in {
    "PROJECT_NAME": "test", "API_V1_STR": "/api/v1", "POSTGRES_SERVER": "localhost",
    "POSTGRES_USER": "postgres", "POSTGRES_PASSWORD": "test", "POSTGRES_DB": "test",
    "POSTGRES_PORT": "5432", "REDIS_HOST": "localhost", "REDIS_PORT": "6379",
    "OPENAI_BASE_URL": "http://localhost", "OPENAI_API_KEY": "test", "SECRET_KEY": "test-secret",
}.items():
    os.environ.setdefault(key, value)

import pytest

from app.events.commerce_consumer import project_event_card, validate_event
from app.models.message import MessageType


def test_after_sale_event_projects_to_system_refund_card():
    card = project_event_card({"eventId": "event-1", "eventType": "after_sale.created", "schemaVersion": 1, "aggregateId": "42", "traceId": "trace-1", "occurredAt": "2026-08-08T00:00:00Z"})
    assert card is not None
    assert card.message_type == MessageType.REFUND_CARD
    assert card.content["after_sale_id"] == "42"


def test_unrelated_event_does_not_create_support_card():
    assert project_event_card({"eventId": "event-1", "eventType": "product.changed", "schemaVersion": 1, "aggregateId": "42", "occurredAt": "2026-08-08T00:00:00Z"}) is None


def test_after_sale_status_event_projects_a_status_card():
    card = project_event_card({
        "eventId": "event-2", "eventType": "after_sale.status_changed", "schemaVersion": 1,
        "aggregateId": "42", "afterSaleId": "42", "status": 2, "occurredAt": "2026-08-08T00:00:00Z",
    })
    assert card is not None
    assert card.content["card_type"] == "after_sale_status_changed"
    assert card.content["after_sale_status"] == 2


def test_order_status_event_projects_a_system_order_card():
    card = project_event_card({
        "eventId": "event-3", "eventType": "order.status_changed", "schemaVersion": 1,
        "aggregateId": "9", "orderStatus": 4, "occurredAt": "2026-08-08T00:00:00Z",
    })
    assert card is not None
    assert card.message_type == MessageType.ORDER_CARD
    assert card.content["card_type"] == "order_status_changed"
    assert card.content["order_status"] == 4


def test_refund_completed_event_projects_a_read_only_payment_card():
    card = project_event_card({
        "eventId": "event-4", "eventType": "refund.completed", "schemaVersion": 1,
        "aggregateId": "42", "afterSaleId": "42", "refundId": "7", "refundAmount": "12.34",
        "occurredAt": "2026-08-08T00:00:00Z",
    })
    assert card is not None
    assert card.content["card_type"] == "refund_completed"
    assert card.content["refund_amount"] == "12.34"


def test_event_envelope_requires_versioned_identifiers_and_timestamp():
    payload = validate_event({
        "eventId": "7178b94b-2e6e-44d7-9c51-7613a4624599",
        "eventType": "after_sale.created",
        "schemaVersion": 1,
        "aggregateId": "42",
        "traceId": "e6d61973-ed11-4416-9a78-3f7a0f69088b",
        "occurredAt": "2026-08-08T00:00:00Z",
    })
    assert payload["eventId"] == "7178b94b-2e6e-44d7-9c51-7613a4624599"
    assert payload["traceId"] == "e6d61973-ed11-4416-9a78-3f7a0f69088b"


def test_status_event_requires_matching_after_sale_identifier_and_known_status():
    payload = validate_event({
        "eventId": "7178b94b-2e6e-44d7-9c51-7613a4624599",
        "eventType": "after_sale.status_changed", "schemaVersion": 1,
        "aggregateId": "42", "afterSaleId": "42", "status": 1,
        "occurredAt": "2026-08-08T00:00:00Z",
    })
    assert payload["status"] == 1


def test_order_status_event_requires_a_known_order_status():
    payload = validate_event({
        "eventId": "7178b94b-2e6e-44d7-9c51-7613a4624599",
        "eventType": "order.status_changed", "schemaVersion": 1,
        "aggregateId": "9", "orderStatus": 4, "occurredAt": "2026-08-08T00:00:00Z",
    })
    assert payload["orderStatus"] == 4


def test_refund_completed_event_requires_matching_after_sale_and_positive_amount():
    payload = validate_event({
        "eventId": "7178b94b-2e6e-44d7-9c51-7613a4624599", "eventType": "refund.completed", "schemaVersion": 1,
        "aggregateId": "42", "afterSaleId": "42", "refundId": "7", "refundAmount": "12.34",
        "occurredAt": "2026-08-08T00:00:00Z",
    })
    assert payload["refundAmount"] == "12.34"


@pytest.mark.parametrize("payload", [
    {"eventId": "not-a-uuid", "eventType": "after_sale.created", "schemaVersion": 1, "aggregateId": "42", "occurredAt": "2026-08-08T00:00:00Z"},
    {"eventId": "7178b94b-2e6e-44d7-9c51-7613a4624599", "eventType": "after_sale.created", "schemaVersion": 2, "aggregateId": "42", "occurredAt": "2026-08-08T00:00:00Z"},
    {"eventId": "7178b94b-2e6e-44d7-9c51-7613a4624599", "eventType": "after_sale.created", "schemaVersion": 1, "aggregateId": "zero", "occurredAt": "not-a-time"},
])
def test_event_envelope_rejects_invalid_or_unsupported_payloads(payload):
    with pytest.raises(Exception):
        validate_event(payload)
