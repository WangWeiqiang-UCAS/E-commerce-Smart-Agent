"""Administrator read-only conversation API.

Every endpoint is guarded by ``get_admin_user_id``, which requires an
``is_admin`` claim in a server-signed JWT. There are no approval or refund
transitions here: refund decisions belong to the Java after-sales state
machine and its audited action endpoints.
"""
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlmodel import desc, select

from app.core.database import async_session_maker
from app.core.field_encryption import decrypt_internal_note, encrypt_internal_note
from app.core.security import get_admin_user_id
from app.models.message import MessageCard, MessageStatus, MessageType
from app.models.support_operator import SupportOperator
from app.models.thread_assignment import ThreadAssignment, ThreadMode

router = APIRouter()


class ThreadSummary(BaseModel):
    thread_id: str
    message_count: int
    last_message_at: str
    last_sender_type: Optional[str] = None
    last_content: Optional[Dict[str, Any]] = None
    mode: str = ThreadMode.AGENT.value
    assigned_admin_id: Optional[int] = None


class ThreadMessage(BaseModel):
    id: int
    thread_id: str
    message_type: str
    status: str
    sender_type: str
    sender_id: Optional[int]
    content: Dict[str, Any]
    created_at: str


class TakeoverRequest(BaseModel):
    reason: Optional[str] = Field(default=None, max_length=500)


class AgentReplyRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class InternalNoteRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


class TransferRequest(BaseModel):
    target_admin_id: int = Field(..., gt=0)
    reason: Optional[str] = Field(default=None, max_length=500)


class AssignmentResponse(BaseModel):
    thread_id: str
    mode: str
    assigned_admin_id: Optional[int]
    updated_at: str


class SupportOperatorSummary(BaseModel):
    id: int
    username: str
    full_name: str


def _message_type_value(message_type) -> str:
    return message_type.value if hasattr(message_type, "value") else str(message_type)


def _thread_message(row: MessageCard) -> ThreadMessage:
    content = dict(row.content)
    if row.sender_type == "internal" and content.get("encryption") == "fernet-v1":
        content["message"] = decrypt_internal_note(str(content.get("ciphertext", "")))
        content.pop("ciphertext", None)
    return ThreadMessage(id=row.id, thread_id=row.thread_id, message_type=_message_type_value(row.message_type),
                         status=_message_type_value(row.status), sender_type=row.sender_type,
                         sender_id=row.sender_id, content=content, created_at=row.created_at.isoformat())


def _assignment_response(assignment: ThreadAssignment) -> AssignmentResponse:
    return AssignmentResponse(
        thread_id=assignment.thread_id,
        mode=_message_type_value(assignment.mode),
        assigned_admin_id=assignment.assigned_admin_id,
        updated_at=assignment.updated_at.isoformat(),
    )


async def _record_system_event(session, thread_id: str, admin_id: int, message: str, event: str) -> None:
    session.add(MessageCard(
        thread_id=thread_id,
        message_type=MessageType.SYSTEM,
        status=MessageStatus.SENT,
        content={"message": message, "event": event},
        sender_type="system",
        sender_id=admin_id,
    ))


async def _require_human_owner(session, thread_id: str, admin_id: int) -> ThreadAssignment:
    assignment = (await session.exec(
        select(ThreadAssignment).where(ThreadAssignment.thread_id == thread_id)
    )).first()
    if not assignment or assignment.mode != ThreadMode.HUMAN or assignment.assigned_admin_id != admin_id:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Take over this thread before changing its handling")
    return assignment


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
        assignment_rows = (await session.exec(select(ThreadAssignment))).all()
        assignments = {row.thread_id: row for row in assignment_rows}
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
                mode=_message_type_value(assignments[thread_id].mode) if thread_id in assignments else ThreadMode.AGENT.value,
                assigned_admin_id=assignments[thread_id].assigned_admin_id if thread_id in assignments else None,
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
        _thread_message(row)
        for row in rows
    ]


@router.get("/support-operators", response_model=list[SupportOperatorSummary])
async def list_active_support_operators(admin_id: int = Depends(get_admin_user_id)):
    """Active internal operators eligible to receive a human-owned thread."""
    async with async_session_maker() as session:
        operators = (await session.exec(
            select(SupportOperator)
            .where(SupportOperator.is_active.is_(True), SupportOperator.is_admin.is_(True))
            .order_by(SupportOperator.full_name, SupportOperator.id)
        )).all()
    return [SupportOperatorSummary(id=operator.id, username=operator.username, full_name=operator.full_name)
            for operator in operators]


@router.post("/threads/{thread_id}/takeover", response_model=AssignmentResponse)
async def takeover_thread(
    thread_id: str,
    request: TakeoverRequest,
    admin_id: int = Depends(get_admin_user_id),
):
    """Pause Agent replies and assign the support conversation to this admin."""
    async with async_session_maker() as session:
        assignment = (await session.exec(
            select(ThreadAssignment).where(ThreadAssignment.thread_id == thread_id)
        )).first()
        if assignment and assignment.mode == ThreadMode.HUMAN and assignment.assigned_admin_id != admin_id:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Thread is already assigned to another administrator")
        if not assignment:
            assignment = ThreadAssignment(thread_id=thread_id)
            session.add(assignment)
        assignment.mode = ThreadMode.HUMAN
        assignment.assigned_admin_id = admin_id
        assignment.reason = request.reason
        assignment.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await _record_system_event(session, thread_id, admin_id, "A human support agent has taken over this conversation.", "human_takeover")
        try:
            await session.commit()
        except IntegrityError as exc:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Thread was assigned concurrently; refresh and try again",
            ) from exc
        await session.refresh(assignment)
        return _assignment_response(assignment)


@router.post("/threads/{thread_id}/return-to-agent", response_model=AssignmentResponse)
async def return_thread_to_agent(thread_id: str, admin_id: int = Depends(get_admin_user_id)):
    """Return a conversation to the Agent after the current human owner releases it."""
    async with async_session_maker() as session:
        assignment = (await session.exec(
            select(ThreadAssignment).where(ThreadAssignment.thread_id == thread_id)
        )).first()
        if not assignment or assignment.mode != ThreadMode.HUMAN:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Thread is not currently human-owned")
        if assignment.assigned_admin_id != admin_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the assigned administrator can return this thread")
        assignment.mode = ThreadMode.AGENT
        assignment.assigned_admin_id = None
        assignment.reason = None
        assignment.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await _record_system_event(session, thread_id, admin_id, "This conversation has been returned to the Agent.", "returned_to_agent")
        await session.commit()
        await session.refresh(assignment)
        return _assignment_response(assignment)


@router.post("/threads/{thread_id}/messages", response_model=ThreadMessage)
async def send_human_reply(
    thread_id: str,
    request: AgentReplyRequest,
    admin_id: int = Depends(get_admin_user_id),
):
    """Send a support reply only while the caller owns the human handoff."""
    async with async_session_maker() as session:
        await _require_human_owner(session, thread_id, admin_id)
        card = MessageCard(
            thread_id=thread_id,
            message_type=MessageType.TEXT,
            status=MessageStatus.SENT,
            content={"message": request.message},
            sender_type="admin",
            sender_id=admin_id,
        )
        session.add(card)
        await session.commit()
        await session.refresh(card)
        return ThreadMessage(
            id=card.id,
            thread_id=card.thread_id,
            message_type=_message_type_value(card.message_type),
            status=_message_type_value(card.status),
            sender_type=card.sender_type,
            sender_id=card.sender_id,
            content=card.content,
            created_at=card.created_at.isoformat(),
        )


@router.post("/threads/{thread_id}/notes", response_model=ThreadMessage)
async def add_internal_note(
    thread_id: str,
    request: InternalNoteRequest,
    admin_id: int = Depends(get_admin_user_id),
):
    """Persist an internal-only note; customer history explicitly filters it out."""
    async with async_session_maker() as session:
        await _require_human_owner(session, thread_id, admin_id)
        card = MessageCard(
            thread_id=thread_id,
            message_type=MessageType.SYSTEM,
            status=MessageStatus.SENT,
            content={"ciphertext": encrypt_internal_note(request.message), "encryption": "fernet-v1", "visibility": "internal"},
            sender_type="internal",
            sender_id=admin_id,
        )
        session.add(card)
        await session.commit()
        await session.refresh(card)
        return _thread_message(card)


@router.post("/threads/{thread_id}/transfer", response_model=AssignmentResponse)
async def transfer_thread(
    thread_id: str,
    request: TransferRequest,
    admin_id: int = Depends(get_admin_user_id),
):
    """Atomically transfer a human-owned conversation and leave an audit event."""
    if request.target_admin_id == admin_id:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Choose a different administrator")
    async with async_session_maker() as session:
        assignment = await _require_human_owner(session, thread_id, admin_id)
        target = (await session.exec(select(SupportOperator).where(SupportOperator.id == request.target_admin_id))).first()
        if not target or not target.is_active or not target.is_admin:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target administrator is not available")
        assignment.assigned_admin_id = target.id
        assignment.reason = request.reason
        assignment.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
        await _record_system_event(session, thread_id, admin_id,
                                   "This conversation was transferred to another support administrator.",
                                   "human_transfer")
        await session.commit()
        await session.refresh(assignment)
        return _assignment_response(assignment)
