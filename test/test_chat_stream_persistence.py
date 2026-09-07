from types import SimpleNamespace

import pytest

from app.api.v1 import chat
from app.api.v1.schemas import ChatRequest


class _CompletedGraph:
    async def astream_events(self, *_args, **_kwargs):
        if False:
            yield None

    async def aget_state(self, _config):
        return SimpleNamespace(next=(), tasks=(), values={"answer": "Persisted agent answer."})


class _FailingGraph:
    async def astream_events(self, *_args, **_kwargs):
        raise RuntimeError("database password should never reach the browser")
        yield None


class _CapturingGraph(_CompletedGraph):
    def __init__(self):
        self.initial_state = None

    async def astream_events(self, initial_state, *_args, **_kwargs):
        self.initial_state = initial_state
        if False:
            yield None


@pytest.mark.asyncio
async def test_completed_chat_persists_agent_answer(monkeypatch):
    from app.graph import workflow

    records = []

    async def persist(*args, **kwargs):
        records.append((args, kwargs))

    async def allow_thread(_thread_id):
        return None

    monkeypatch.setattr(workflow, "app_graph", _CompletedGraph())
    monkeypatch.setattr(chat, "_persist_message", persist)
    monkeypatch.setattr(chat, "_require_agent_owned_thread", allow_thread)

    response = await chat.chat(
        SimpleNamespace(state=SimpleNamespace(trace_id="trace-1")),
        ChatRequest(question="Where is my order?", thread_id="thread-1"),
        7,
        None,
        None,
    )
    async for _ in response.body_iterator:
        pass

    assert len(records) == 2
    assert records[1][0][2] == {"message": "Persisted agent answer."}
    assert records[1][1]["sender_type"] == "agent"


@pytest.mark.asyncio
async def test_chat_stream_hides_internal_exception(monkeypatch):
    from app.graph import workflow

    async def persist(*_args, **_kwargs):
        return None

    async def allow_thread(_thread_id):
        return None

    monkeypatch.setattr(workflow, "app_graph", _FailingGraph())
    monkeypatch.setattr(chat, "_persist_message", persist)
    monkeypatch.setattr(chat, "_require_agent_owned_thread", allow_thread)

    response = await chat.chat(
        SimpleNamespace(state=SimpleNamespace(trace_id="trace-2")),
        ChatRequest(question="Help", thread_id="thread-2"),
        7,
        None,
        None,
    )
    body = "".join([chunk async for chunk in response.body_iterator])

    assert "temporarily unavailable" in body
    assert "database password" not in body


@pytest.mark.asyncio
async def test_chat_redacts_user_pii_before_state_and_message_persistence(monkeypatch):
    from app.graph import workflow

    records = []
    graph = _CapturingGraph()

    async def persist(*args, **kwargs):
        records.append((args, kwargs))

    async def allow_thread(_thread_id):
        return None

    monkeypatch.setattr(workflow, "app_graph", graph)
    monkeypatch.setattr(chat, "_persist_message", persist)
    monkeypatch.setattr(chat, "_require_agent_owned_thread", allow_thread)

    response = await chat.chat(
        SimpleNamespace(state=SimpleNamespace(trace_id="trace-3")),
        ChatRequest(question="Please call 13812345678 about order SN9", thread_id="thread-3"),
        7,
        None,
        None,
    )
    async for _ in response.body_iterator:
        pass

    assert "13812345678" not in graph.initial_state["question"]
    assert "[PHONE_REDACTED]" in graph.initial_state["question"]
    assert "13812345678" not in records[0][0][2]["message"]
