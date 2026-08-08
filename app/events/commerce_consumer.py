"""RabbitMQ consumer for Java commerce domain events."""
import asyncio
import json
from datetime import datetime
from typing import Any

from kombu import Connection, Consumer, Exchange, Queue
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.inbox import InboxEvent
from app.models.message import MessageCard, MessageStatus, MessageType


REQUIRED_FIELDS = {"eventId", "eventType", "schemaVersion", "aggregateId", "occurredAt"}


def project_event_card(payload: dict[str, Any]) -> MessageCard | None:
    """Create a read-only support card for events useful to human operators."""
    if payload.get("eventType") != "after_sale.created":
        return None
    return MessageCard(
        thread_id=f"commerce:after_sale:{payload['aggregateId']}",
        message_type=MessageType.REFUND_CARD,
        status=MessageStatus.SENT,
        sender_type="system",
        content={
            "card_type": "after_sale_created",
            "event_id": payload["eventId"],
            "after_sale_id": payload["aggregateId"],
            "trace_id": payload.get("traceId"),
            "occurred_at": payload["occurredAt"],
        },
        meta_data={"event_type": payload["eventType"], "schema_version": payload["schemaVersion"]},
    )


async def record_event(payload: dict[str, Any]) -> bool:
    """Persist once. False means a duplicate delivery and is safe to acknowledge."""
    if not REQUIRED_FIELDS.issubset(payload):
        raise ValueError("Invalid commerce event envelope")
    occurred_at = datetime.fromisoformat(str(payload["occurredAt"]).replace("Z", "+00:00"))
    async with async_session_maker() as session:
        existing = await session.exec(select(InboxEvent).where(InboxEvent.event_id == str(payload["eventId"])))
        if existing.first():
            return False
        session.add(InboxEvent(
            event_id=str(payload["eventId"]), event_type=str(payload["eventType"]),
            schema_version=int(payload["schemaVersion"]), aggregate_id=str(payload["aggregateId"]),
            trace_id=payload.get("traceId"), payload=payload, occurred_at=occurred_at,
        ))
        card = project_event_card(payload)
        if card:
            session.add(card)
        try:
            await session.commit()
        except IntegrityError:
            await session.rollback()
            return False
    return True


def consume_forever() -> None:
    if not settings.RABBITMQ_URL:
        raise RuntimeError("RABBITMQ_URL is required for the commerce event consumer")
    exchange = Exchange("mall.agent.events", type="topic", durable=True)
    queue = Queue(settings.COMMERCE_EVENT_QUEUE, exchange=exchange, routing_key="#", durable=True)

    def on_message(body: Any, message) -> None:
        try:
            payload = body if isinstance(body, dict) else json.loads(body if isinstance(body, str) else body.decode("utf-8"))
            asyncio.run(record_event(payload))
        except (ValueError, json.JSONDecodeError):
            message.reject(requeue=False)
            return
        message.ack()

    with Connection(settings.RABBITMQ_URL) as connection:
        with Consumer(connection, queues=[queue], callbacks=[on_message], accept=["json", "text/plain"]):
            while True:
                connection.drain_events(timeout=None)
