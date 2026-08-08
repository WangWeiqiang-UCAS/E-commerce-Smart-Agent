from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column, DateTime, JSON, text
from sqlmodel import Field, SQLModel


class InboxEvent(SQLModel, table=True):
    """Deduplication and audit record for commerce domain events."""
    __tablename__ = "inbox_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: str = Field(index=True, unique=True, max_length=36)
    event_type: str = Field(index=True, max_length=128)
    schema_version: int
    aggregate_id: str = Field(index=True, max_length=64)
    trace_id: Optional[str] = Field(default=None, index=True, max_length=128)
    payload: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    occurred_at: datetime = Field(sa_column=Column(DateTime, nullable=False))
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_column=Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP")))
