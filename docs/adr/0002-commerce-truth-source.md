# ADR 0002: Java commerce is the single source of truth

## Status

Accepted.

## Context

target.md section 4 assigns membership, catalogue, stock, orders, payments,
logistics, and after-sales to the Java commerce system. Python previously
mirrored orders and refunds.

## Decision

Python owns only conversation, message, knowledge, checkpoint, and evaluation
data. The Agent calls the allow-listed Java boundary
(`packages/api-contracts/agent-commerce.openapi.yaml`); it never connects to
the commerce database. Cross-system joins are done through APIs; asynchronous
synchronization uses events with producer outbox and consumer inbox.

## Consequences

No cross-database joins are permitted. Ownership checks for orders and
after-sales are derived from the authenticated Java principal, never from an
LLM-supplied user ID.
