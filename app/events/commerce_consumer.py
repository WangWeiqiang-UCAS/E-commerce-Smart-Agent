"""RabbitMQ consumer for Java commerce domain events."""
import asyncio
import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Annotated

from kombu import Connection, Consumer, Exchange, Queue
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, UUID4
from sqlalchemy.exc import IntegrityError
from sqlmodel import select

from app.core.config import settings
from app.core.database import async_session_maker
from app.models.inbox import InboxEvent
from app.models.message import MessageCard, MessageStatus, MessageType

LOGGER = logging.getLogger(__name__)



class CommerceEventEnvelope(BaseModel):
    """AsyncAPI envelope validation before an event can enter the Agent inbox."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    event_id: UUID4 = Field(validation_alias=AliasChoices("eventId", "event_id"), serialization_alias="eventId")
    event_type: Annotated[str, Field(min_length=1, max_length=128)] = Field(
        validation_alias=AliasChoices("eventType", "event_type"), serialization_alias="eventType"
    )
    schema_version: Annotated[int, Field(ge=1)] = Field(
        validation_alias=AliasChoices("schemaVersion", "schema_version"), serialization_alias="schemaVersion"
    )
    aggregate_id: Annotated[str, Field(pattern=r"^[1-9][0-9]*$")] = Field(
        validation_alias=AliasChoices("aggregateId", "aggregate_id"), serialization_alias="aggregateId"
    )
    trace_id: UUID4 | None = Field(default=None, validation_alias=AliasChoices("traceId", "trace_id"), serialization_alias="traceId")
    occurred_at: datetime = Field(validation_alias=AliasChoices("occurredAt", "occurred_at"), serialization_alias="occurredAt")


class AfterSaleStatusChangedEvent(CommerceEventEnvelope):
    after_sale_id: Annotated[str, Field(pattern=r"^[1-9][0-9]*$")] = Field(
        validation_alias=AliasChoices("afterSaleId", "after_sale_id"), serialization_alias="afterSaleId"
    )
    status: Annotated[int, Field(ge=1, le=3)]


class OrderStatusChangedEvent(CommerceEventEnvelope):
    order_status: Annotated[int, Field(ge=0, le=4)] = Field(
        validation_alias=AliasChoices("orderStatus", "order_status"), serialization_alias="orderStatus"
    )


class RefundCompletedEvent(CommerceEventEnvelope):
    after_sale_id: Annotated[str, Field(pattern=r"^[1-9][0-9]*$")] = Field(
        validation_alias=AliasChoices("afterSaleId", "after_sale_id"), serialization_alias="afterSaleId"
    )
    refund_id: Annotated[str, Field(pattern=r"^[1-9][0-9]*$")] = Field(
        validation_alias=AliasChoices("refundId", "refund_id"), serialization_alias="refundId"
    )
    refund_amount: Annotated[Decimal, Field(gt=0, max_digits=10, decimal_places=2)] = Field(
        validation_alias=AliasChoices("refundAmount", "refund_amount"), serialization_alias="refundAmount"
    )


def validate_event(payload: dict[str, Any]) -> dict[str, Any]:
    event_type = payload.get("eventType", payload.get("event_type"))
    if event_type == "after_sale.status_changed":
        event = AfterSaleStatusChangedEvent.model_validate(payload)
    elif event_type == "order.status_changed":
        event = OrderStatusChangedEvent.model_validate(payload)
    elif event_type == "refund.completed":
        event = RefundCompletedEvent.model_validate(payload)
    else:
        event = CommerceEventEnvelope.model_validate(payload)
    normalized = event.model_dump(by_alias=True, mode="json", exclude_none=True)
    if normalized["eventType"] not in {"after_sale.created", "after_sale.status_changed", "order.status_changed", "refund.completed"}:
        raise ValueError("Unsupported commerce event type")
    if normalized["schemaVersion"] != 1:
        raise ValueError(f"Unsupported {normalized['eventType']} schema version")
    if normalized["eventType"] == "after_sale.status_changed" and normalized["afterSaleId"] != normalized["aggregateId"]:
        raise ValueError("after-sale status event aggregate does not match afterSaleId")
    if normalized["eventType"] == "refund.completed" and normalized["afterSaleId"] != normalized["aggregateId"]:
        raise ValueError("refund completed event aggregate does not match afterSaleId")
    return normalized


def project_event_card(payload: dict[str, Any]) -> MessageCard | None:
    """Create a read-only support card for events useful to human operators."""
    event_type = payload.get("eventType")
    if event_type == "order.status_changed":
        return MessageCard(
            thread_id=f"commerce:order:{payload['aggregateId']}",
            message_type=MessageType.ORDER_CARD,
            status=MessageStatus.SENT,
            sender_type="system",
            content={
                "card_type": "order_status_changed",
                "event_id": payload["eventId"],
                "order_id": payload["aggregateId"],
                "order_status": payload["orderStatus"],
                "trace_id": payload.get("traceId"),
                "occurred_at": payload["occurredAt"],
            },
            meta_data={"event_type": payload["eventType"], "schema_version": payload["schemaVersion"]},
        )
    if event_type not in {"after_sale.created", "after_sale.status_changed", "refund.completed"}:
        return None
    return MessageCard(
        thread_id=f"commerce:after_sale:{payload['aggregateId']}",
        message_type=MessageType.REFUND_CARD,
        status=MessageStatus.SENT,
        sender_type="system",
        content={
            "card_type": "after_sale_created" if event_type == "after_sale.created"
            else "refund_completed" if event_type == "refund.completed" else "after_sale_status_changed",
            "event_id": payload["eventId"],
            "after_sale_id": payload["aggregateId"],
            "after_sale_status": payload.get("status"),
            "refund_id": payload.get("refundId"),
            "refund_amount": payload.get("refundAmount"),
            "trace_id": payload.get("traceId"),
            "occurred_at": payload["occurredAt"],
        },
        meta_data={"event_type": payload["eventType"], "schema_version": payload["schemaVersion"]},
    )


async def record_event(payload: dict[str, Any]) -> bool:
    """Persist once. False means a duplicate delivery and is safe to acknowledge."""
    payload = validate_event(payload)
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


def handle_message(body: Any, message: Any) -> None:
    """Acknowledge only after the event is safely in the Inbox transaction."""
    try:
        payload = body if isinstance(body, dict) else json.loads(body if isinstance(body, str) else body.decode("utf-8"))
        asyncio.run(record_event(payload))
    except (ValueError, json.JSONDecodeError):
        LOGGER.warning("Rejecting invalid commerce event to dead-letter queue")
        message.reject(requeue=False)
        return
    except Exception:
        # Do not acknowledge when persistence or projection failed. The
        # durable Inbox uniqueness key makes the replay safe.
        LOGGER.exception("Commerce event processing failed; requesting retry")
        message.reject(requeue=True)
        return
    message.ack()


def consume_forever() -> None:
    if not settings.RABBITMQ_URL:
        raise RuntimeError("RABBITMQ_URL is required for the commerce event consumer")
    exchange = Exchange("mall.agent.events", type="topic", durable=True)
    dead_letter_exchange = Exchange("mall.agent.events.dlx", type="direct", durable=True)
    dead_letter_queue = Queue(
        settings.COMMERCE_EVENT_DEAD_LETTER_QUEUE,
        exchange=dead_letter_exchange,
        routing_key="invalid",
        durable=True,
    )
    queue = Queue(
        settings.COMMERCE_EVENT_QUEUE,
        exchange=exchange,
        routing_key="#",
        durable=True,
        queue_arguments={
            "x-dead-letter-exchange": dead_letter_exchange.name,
            "x-dead-letter-routing-key": "invalid",
        },
    )

    with Connection(settings.RABBITMQ_URL) as connection:
        dead_letter_queue.declare(channel=connection.default_channel)
        with Consumer(connection, queues=[queue], callbacks=[handle_message], accept=["json", "text/plain"]):
            while True:
                connection.drain_events(timeout=None)
