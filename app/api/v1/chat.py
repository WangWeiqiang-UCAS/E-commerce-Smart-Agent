"""Chat endpoints: SSE streaming, interrupt-aware confirmation, and history."""
import json
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
from app.core.security import get_current_user_id, require_bff_secret
from app.models.message import MessageCard, MessageStatus, MessageType

router = APIRouter()


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


@router.post("/chat")
async def chat(
    http_request: Request,
    request: ChatRequest,
    current_user_id: int = Depends(get_current_user_id),
    _: None = Depends(require_bff_secret),
    commerce_access_token: str | None = Header(default=None, alias="X-Commerce-Access-Token"),
):
    """Chat endpoint: order, policy, guided selling, and confirmed after-sales."""
    check_rate_limit(f"chat:{current_user_id}")
    from app.graph.workflow import app_graph

    if app_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not fully initialized. Please try again in a moment.",
        )

    thread_id = _thread_key(current_user_id, request.thread_id)

    async def event_generator():
        from app.graph.workflow import app_graph

        context_tokens = set_request_context(commerce_access_token, http_request.state.trace_id)
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "question": request.question,
            "user_id": current_user_id,
            "thread_id": thread_id,
            "history": [],
            "context": [],
            "order_data": None,
            "products": [],
            "after_sale_data": None,
            "handoff": None,
            "answer": "",
        }
        await _persist_message(
            thread_id,
            MessageType.TEXT,
            {"message": request.question},
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
        except Exception as exc:  # pragma: no cover - defensive stream termination
            error_msg = json.dumps({"error": str(exc)}, ensure_ascii=False)
            yield f"data: {error_msg}\n\n"
        finally:
            reset_request_context(context_tokens)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/chat/resume")
async def resume_chat(
    http_request: Request,
    request: ResumeRequest,
    current_user_id: int = Depends(get_current_user_id),
    _: None = Depends(require_bff_secret),
    commerce_access_token: str | None = Header(default=None, alias="X-Commerce-Access-Token"),
):
    """Resume a paused refund confirmation with the user's explicit decision."""
    check_rate_limit(f"chat:{current_user_id}")
    from app.graph.workflow import app_graph

    if app_graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not fully initialized. Please try again in a moment.",
        )
    context_tokens = set_request_context(commerce_access_token, http_request.state.trace_id)
    try:
        thread_id = _thread_key(current_user_id, request.thread_id)
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
    current_user_id: int = Depends(get_current_user_id),
    _: None = Depends(require_bff_secret),
):
    """Persisted message history for the current user's thread (refresh-safe)."""
    scoped_thread_id = _thread_key(current_user_id, thread_id)
    async with async_session_maker() as session:
        rows = (await session.exec(
            select(MessageCard)
            .where(MessageCard.thread_id == scoped_thread_id)
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
