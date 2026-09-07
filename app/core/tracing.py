"""Small vendor-neutral trace correlation helpers."""
from uuid import UUID, uuid4
from time import perf_counter

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.metrics import record_http

TRACE_HEADER = "X-Trace-Id"


def valid_trace_id(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(UUID(value))
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid X-Trace-Id") from exc


class TraceIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = valid_trace_id(request.headers.get(TRACE_HEADER)) or str(uuid4())
        request.state.trace_id = trace_id
        started = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            route = request.scope.get("route")
            record_http(request.method, getattr(route, "path", "unmatched"), 500, perf_counter() - started)
            raise
        route = request.scope.get("route")
        record_http(request.method, getattr(route, "path", "unmatched"), response.status_code, perf_counter() - started)
        response.headers[TRACE_HEADER] = trace_id
        return response
