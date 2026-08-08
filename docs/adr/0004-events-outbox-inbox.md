# ADR 0004: Domain events via outbox and inbox

## Status

Accepted.

## Context

target.md section 5 lists domain events (`product.changed`, `order.created`,
`after_sale.created`, etc.) with a strict envelope. RabbitMQ is the default
bus; Kafka only when existing infrastructure demands it.

## Decision

Java writes `oms_outbox_event` in the same transaction as the business change.
`OutboxPublisher` publishes to the durable `mall.agent.events` topic exchange.
Python consumes into `inbox_events` with a unique `event_id`; duplicates are
acknowledged without reprocessing. `after_sale.created` also projects a refund
card for human operators.

## Consequences

Message loss windows are bounded by the outbox retry loop. Consumers must never
assume at-most-once delivery; all projection logic is idempotent.
