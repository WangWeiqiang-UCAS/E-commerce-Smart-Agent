# ADR 0005: Python 3.11+ and LangGraph interrupt for human confirmation

## Status

Accepted.

## Context

LangGraph `interrupt()` inside async nodes requires Python 3.11+ (contextvar
propagation across asyncio tasks). The prototype supported Python 3.10.

## Decision

Raise the Python floor to 3.11 (`pyproject.toml`, CI 3.12, Docker 3.12). The
after-sales subgraph uses `interrupt()` for explicit user confirmation and
resumes via `POST /api/v1/chat/resume` with a structured decision payload.
Tests use an in-memory checkpointer so interrupt/resume is exercised without
Redis.

## Consequences

Deployments must provide Python 3.11+. Deterministic unit tests cover both the
confirmed and cancelled paths.
