"""Ephemeral request context deliberately excluded from LangGraph checkpoints."""
from contextvars import ContextVar, Token

_commerce_access_token: ContextVar[str | None] = ContextVar("commerce_access_token", default=None)
_trace_id: ContextVar[str | None] = ContextVar("agent_trace_id", default=None)


def set_request_context(commerce_access_token: str | None, trace_id: str | None) -> tuple[Token, Token]:
    return _commerce_access_token.set(commerce_access_token), _trace_id.set(trace_id)


def reset_request_context(tokens: tuple[Token, Token]) -> None:
    _commerce_access_token.reset(tokens[0])
    _trace_id.reset(tokens[1])


def commerce_access_token() -> str | None:
    return _commerce_access_token.get()


def trace_id() -> str | None:
    return _trace_id.get()
