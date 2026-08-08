# ADR 0003: Write actions require preview, confirmation, and idempotency

## Status

Accepted.

## Context

target.md section 5 requires a preview-confirm-execute flow with
`action_token`, `idempotency_key`, preview hash, order version, trace ID, and
confirmation time.

## Decision

The Agent graph pauses refund intent at a LangGraph `interrupt`. Only a
confirmed resume produces a handoff payload; the BFF then calls Java
`/agent/commerce/actions/refund/preview` and `/confirm`. Java creates the
after-sales application and never refunds funds directly — payment refunds are
a later state-machine transition. Confirmation uses an SHA-256-hashed,
10-minute action token and a unique idempotency key.

## Consequences

The LLM cannot create or approve refunds. Duplicate confirmations with the same
idempotency key return the original result; replay of a consumed token fails.
