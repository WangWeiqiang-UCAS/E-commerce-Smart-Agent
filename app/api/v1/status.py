"""Agent conversation status; commerce statuses are queried from Java via the BFF."""
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlmodel import desc, select

from app.core.database import async_session_maker
from app.core.security import get_current_user_id
from app.models.message import MessageCard

router = APIRouter()


class StatusResponse(BaseModel):
    thread_id: str
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: str


@router.get("/status/{thread_id}", response_model=StatusResponse)
async def get_thread_status(thread_id: str, current_user_id: int = Depends(get_current_user_id)):
    async with async_session_maker() as session:
        latest = (await session.exec(
            select(MessageCard)
            .where(MessageCard.thread_id == thread_id)
            .where((MessageCard.receiver_id == current_user_id) | (MessageCard.receiver_id.is_(None)))
            .order_by(desc(MessageCard.created_at))
            .limit(1)
        )).first()
    if latest:
        return StatusResponse(
            thread_id=thread_id,
            status="UPDATED",
            message=latest.content.get("message") if isinstance(latest.content, dict) else None,
            data=latest.content if isinstance(latest.content, dict) else {},
            timestamp=latest.created_at.isoformat(),
        )
    return StatusResponse(
        thread_id=thread_id,
        status="PROCESSING",
        message="The agent is processing your request.",
        data={},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
