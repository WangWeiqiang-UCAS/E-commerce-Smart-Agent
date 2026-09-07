"""Chat endpoints: SSE streaming, interrupt-aware confirmation, and history."""
import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlmodel import desc, select

from app.api.v1.schemas import ChatRequest
from app.core.agent_context import reset_request_context, set_request_context
from app.core.database import async_session_maker
from app.core.rate_limit import check_rate_limit
from app.core.security import get_commerce_member_id, require_bff_secret
from app.graph.safety import redact_pii
from app.models.message import MessageCard, MessageStatus, MessageType
from app.models.thread_assignment import ThreadAssignment, ThreadMode

router = APIRouter()
LOGGER = logging.getLogger(__name__)


class ResumeRequest(BaseModel):
    """Resume payload for a paused refund confirmation."""

    thread_id: str = Field(..., min_length=1, max_length=128)
    decision: Literal["confirmed", "cancelled"]
    confirmed_at: Optional[str] = Field(default=None, description="ISO-8601 user confirmation time")
    approver_id: Optional[int] = Field(default=None, description="Human approver id when applicable")
    idempotency_key: Optional[str] = Field(default=None, max_length=64)


async def _persist_message(
    thread_id: str,
    message_type: MessageType,
    content: dict[str, Any],
    *,
    sender_type: str,
    sender_id: Optional[int] = None,
    receiver_id: Optional[int] = None,
) -> None:
    async with async_session_maker() as session:
        session.add(MessageCard(
            thread_id=thread_id,
            message_type=message_type,
            status=MessageStatus.SENT,
            content=content,
            sender_type=sender_type,
            sender_id=sender_id,
            receiver_id=receiver_id,
        ))
        await session.commit()


def _thread_key(user_id: int, thread_id: str) -> str:
    return f"{user_id}_{thread_id}"


async def _require_agent_owned_thread(thread_id: str) -> None:
    """Do not let the Agent answer while a human support owner is assigned."""
    async with async_session_maker() as session:
        assignment = (await session.exec(
            select(ThreadAssignment).where(ThreadAssignment.thread_id == thread_id)
        )).first()
    if assignment and assignment.mode == ThreadMode.HUMAN:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A human support agent is handling this conversation.",
        )


@router.post("/chat")
async def chat(
    http_request: Request,
    request: ChatRequest,
    current_user_id: int = Depends(get_commerce_member_id),
    _: None = Depends(require_bff_secret),
    commerce_access_token: str | None = Header(default=None, alias="X-Commerce-Access-Token"),
):
    """Chat endpoint: order, policy, guided selling, and confirmed after-sales."""
    await check_rate_limit(f"chat:{current_user_id}")
    from app.graph.workflow import app_graph

    if app_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not fully initialized. Please try again in a moment.",
        )

    thread_id = _thread_key(current_user_id, request.thread_id)
    await _require_agent_owned_thread(thread_id)
    safe_question = redact_pii(request.question)

    async def event_generator():
        from app.graph.workflow import app_graph

        context_tokens = set_request_context(commerce_access_token, http_request.state.trace_id)
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "question": safe_question,
            "user_id": current_user_id,
            "thread_id": thread_id,
            "history": [],
            "context": [],
            "order_data": None,
            "order_detail_loaded": False,
            "products": [],
            "product_detail": None,
            "after_sale_data": None,
            "handoff": None,
            "answer": "",
        }
        await _persist_message(
            thread_id,
            MessageType.TEXT,
            {"message": safe_question},
            sender_type="user",
            sender_id=current_user_id,
            receiver_id=current_user_id,
        )
        try:
            async for event in app_graph.astream_events(
                initial_state, config, version="v2"
            ):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    data = event.get("data")
                    if data and isinstance(data, dict):
                        chunk = data.get("chunk")
                        if chunk and chunk.content:
                            payload = json.dumps({"token": chunk.content}, ensure_ascii=False)
                            yield f"data: {payload}\n\n"

            # LangGraph surfaces interrupts through checkpoint state: when the
            # graph pauses, ``snapshot.next`` names the pending node and the
            # pending tasks carry the structured interrupt payload.
            snapshot = await app_graph.aget_state(config)
            answer = getattr(snapshot, "values", {}).get("answer")
            if answer:
                await _persist_message(
                    thread_id,
                    MessageType.TEXT,
                    {"message": answer},
                    sender_type="agent",
                    receiver_id=current_user_id,
                )
            if snapshot.next:
                interrupt_payload = None
                for task in snapshot.tasks:
                    for pending in task.interrupts:
                        interrupt_payload = pending.value
                        break
                    if interrupt_payload:
                        break
                if interrupt_payload:
                    await _persist_message(
                        thread_id,
                        MessageType.AUDIT_CARD,
                        {"message": "Awaiting explicit customer confirmation.", "interrupt": interrupt_payload},
                        sender_type="agent",
                        receiver_id=current_user_id,
                    )
                    yield f"data: {json.dumps({'interrupt': interrupt_payload}, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception:  # pragma: no cover - defensive stream termination
            LOGGER.exception("Chat stream failed", extra={"trace_id": http_request.state.trace_id})
            error_msg = json.dumps(
                {"error": "The support assistant is temporarily unavailable. Please try again."},
                ensure_ascii=False,
            )
            yield f"data: {error_msg}\n\n"
        finally:
            reset_request_context(context_tokens)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/chat/resume")
async def resume_chat(
    http_request: Request,
    request: ResumeRequest,
    current_user_id: int = Depends(get_commerce_member_id),
    _: None = Depends(require_bff_secret),
    commerce_access_token: str | None = Header(default=None, alias="X-Commerce-Access-Token"),
):
    """Resume a paused refund confirmation with the user's explicit decision."""
    await check_rate_limit(f"chat:{current_user_id}")
    from app.graph.workflow import app_graph

    if app_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not fully initialized. Please try again in a moment.",
        )
    context_tokens = set_request_context(commerce_access_token, http_request.state.trace_id)
    try:
        thread_id = _thread_key(current_user_id, request.thread_id)
        await _require_agent_owned_thread(thread_id)
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await app_graph.aget_state(config)
        if not snapshot.next:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No paused action exists for this thread.",
            )
        resume_payload = {
            "decision": request.decision,
            "confirmed_at": request.confirmed_at or datetime.now(timezone.utc).isoformat(),
            "approver_id": request.approver_id,
        }
        if request.idempotency_key:
            resume_payload["idempotency_key"] = request.idempotency_key
        result = await app_graph.ainvoke(Command(resume=resume_payload), config)
        answer = result.get("answer") or "The refund request was cancelled."
        handoff = result.get("handoff")
        content: dict[str, Any] = {"message": answer}
        if handoff:
            content["handoff"] = handoff
        await _persist_message(
            thread_id,
            MessageType.REFUND_CARD if handoff else MessageType.TEXT,
            content,
            sender_type="agent",
            receiver_id=current_user_id,
        )
        return {"answer": answer, "handoff": handoff}
    finally:
        reset_request_context(context_tokens)


@router.get("/chat/history/{thread_id}")
async def chat_history(
    thread_id: str,
    current_user_id: int = Depends(get_commerce_member_id),
    _: None = Depends(require_bff_secret),
):
    """Persisted message history for the current user's thread (refresh-safe)."""
    scoped_thread_id = _thread_key(current_user_id, thread_id)
    async with async_session_maker() as session:
        rows = (await session.exec(
            select(MessageCard)
            .where(MessageCard.thread_id == scoped_thread_id)
            .where(MessageCard.sender_type != "internal")
            .order_by(desc(MessageCard.created_at))
            .limit(200)
        )).all()
    return [
        {
            "id": row.id,
            "message_type": row.message_type.value if hasattr(row.message_type, "value") else str(row.message_type),
            "sender_type": row.sender_type,
            "content": row.content,
            "created_at": row.created_at.isoformat(),
        }
        for row in reversed(rows)
    ]


@router.get("/chat/handoff-status/{thread_id}")
async def handoff_status(
    thread_id: str,
    current_user_id: int = Depends(get_commerce_member_id),
    _: None = Depends(require_bff_secret),
):
    """Expose only whether this customer's conversation is human-handled."""
    scoped_thread_id = _thread_key(current_user_id, thread_id)
    async with async_session_maker() as session:
        assignment = (await session.exec(
            select(ThreadAssignment).where(ThreadAssignment.thread_id == scoped_thread_id)
        )).first()
    mode = assignment.mode.value if assignment and hasattr(assignment.mode, "value") else assignment.mode if assignment else ThreadMode.AGENT.value
    return {"mode": mode}
