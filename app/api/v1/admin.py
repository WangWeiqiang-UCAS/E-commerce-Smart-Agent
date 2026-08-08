"""Administrator read-only conversation API.

Every endpoint is guarded by ``get_admin_user_id``, which requires an
``is_admin`` claim in a server-signed JWT. There are no approval or refund
transitions here: refund decisions belong to the Java after-sales state
machine and its audited action endpoints.
"""
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import desc, select

from app.core.database import async_session_maker
from app.core.security import get_admin_user_id
from app.models.message import MessageCard

router = APIRouter()


class ThreadSummary(BaseModel):
    thread_id: str
    message_count: int
    last_message_at: str
    last_sender_type: Optional[str] = None
    last_content: Optional[Dict[str, Any]] = None


class ThreadMessage(BaseModel):
    id: int
    thread_id: str
    message_type: str
    status: str
    sender_type: str
    sender_id: Optional[int]
    content: Dict[str, Any]
    created_at: str


def _message_type_value(message_type) -> str:
    return message_type.value if hasattr(message_type, "value") else str(message_type)


@router.get("/threads", response_model=list[ThreadSummary])
async def list_threads(admin_id: int = Depends(get_admin_user_id)):
    """Recent conversation threads and their latest message snippet."""
    async with async_session_maker() as session:
        rows = (await session.exec(
            select(
                MessageCard.thread_id,
                func.count(MessageCard.id),
                func.max(MessageCard.created_at),
            )
            .group_by(MessageCard.thread_id)
            .order_by(desc(func.max(MessageCard.created_at)))
            .limit(100)
        )).all()
        summaries = []
        for thread_id, count, last_at in rows:
            latest = (await session.exec(
                select(MessageCard)
                .where(MessageCard.thread_id == thread_id)
                .order_by(desc(MessageCard.created_at))
                .limit(1)
            )).first()
            summaries.append(ThreadSummary(
                thread_id=thread_id,
                message_count=int(count),
                last_message_at=last_at.isoformat(),
                last_sender_type=latest.sender_type if latest else None,
                last_content=latest.content if latest else None,
            ))
    return summaries


@router.get("/threads/{thread_id}", response_model=list[ThreadMessage])
async def get_thread_messages(thread_id: str, admin_id: int = Depends(get_admin_user_id)):
    """Full message history for one conversation thread."""
    async with async_session_maker() as session:
        rows = (await session.exec(
            select(MessageCard)
            .where(MessageCard.thread_id == thread_id)
            .order_by(MessageCard.created_at)
        )).all()
    return [
        ThreadMessage(
            id=row.id,
            thread_id=row.thread_id,
            message_type=_message_type_value(row.message_type),
            status=_message_type_value(row.status),
            sender_type=row.sender_type,
            sender_id=row.sender_id,
            content=row.content,
            created_at=row.created_at.isoformat(),
        )
        for row in rows
    ]
