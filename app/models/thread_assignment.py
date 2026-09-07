"""Conversation ownership belongs to the Agent support domain, not commerce."""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import Column, String, text
from sqlmodel import Field, SQLModel


class ThreadMode(str, Enum):
    AGENT = "agent"
    HUMAN = "human"


class ThreadAssignment(SQLModel, table=True):
    """Current human/Agent ownership for one persisted support conversation."""

    __tablename__ = "thread_assignments"

    id: Optional[int] = Field(default=None, primary_key=True)
    thread_id: str = Field(sa_column=Column(String(128), nullable=False, unique=True, index=True))
    mode: ThreadMode = Field(default=ThreadMode.AGENT, sa_column=Column(String(16), nullable=False))
    assigned_admin_id: Optional[int] = Field(default=None, index=True)
    reason: Optional[str] = Field(default=None, max_length=500)
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP"), "onupdate": text("CURRENT_TIMESTAMP")},
    )
