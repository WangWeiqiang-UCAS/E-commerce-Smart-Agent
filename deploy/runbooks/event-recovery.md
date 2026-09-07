# Commerce event recovery runbook

Use this runbook when the Commerce outbox backlog, expired lease gauge, RabbitMQ
alerts, or Agent Inbox processing indicates a domain-event delivery incident.
It is safe to replay a Commerce outbox event: the Agent persists `event_id` in
`inbox_events` with a uniqueness constraint before acknowledging delivery.

## Triage

1. Record an incident owner and trace ID. Do not change orders, payments, or
   after-sales records as part of event recovery.
2. Inspect Java Prometheus metrics:
   `commerce_outbox_pending`,
   `commerce_outbox_oldest_unpublished_seconds`, and
   `commerce_outbox_expired_leases`. Also inspect
   `commerce_refund_pending` and `commerce_refund_failed`; the latter requires
   payment-channel remediation before it can be retried.
3. Confirm RabbitMQ broker health, the durable `mall.agent.events` exchange,
   and the Agent consumer process before attempting any replay.

## Inspect the durable source of truth

Run read-only queries through the approved Commerce MySQL connection:

```sql
SELECT id, event_id, event_type, aggregate_id, occurred_at, attempts,
       publish_locked_at, publish_lock_owner
FROM oms_outbox_event
WHERE published_at IS NULL
ORDER BY id
LIMIT 100;

SELECT event_id, event_type, aggregate_id, occurred_at
FROM inbox_events
ORDER BY received_at DESC
LIMIT 100;
```

Never delete an Inbox row to force a replay. A duplicate event is intentionally
acknowledged without recreating its projection.

## Recover

1. Restore broker connectivity or restart only the failed publisher/consumer.
   `OutboxPublisher` automatically releases failed leases and retries in ID
   order; a lease older than 60 seconds is claimable by another instance.
2. If a restored database contains rows with `published_at IS NULL`, leave
   those rows intact and start the publisher. This is the supported replay
   operation.
3. Verify `commerce_outbox_pending` and the oldest-age gauge return to their
   normal operating range, then sample the corresponding `inbox_events` rows
   by `event_id`.
4. For malformed payloads, inspect the `agent.runtime.events.dlq` queue. Fix
   the producer/schema mismatch first; only then republish the original event
   with the same `event_id`.

## Exit evidence

Record the outbox IDs, event IDs, trace IDs, metrics before/after, broker
status, and sampled Agent projections. Escalate if a financial state change is
missing from Commerce itself; event replay must never be used to repair the
Commerce system of record.

## Payment refund dispatch

`oms_payment_refund` is the durable source of truth for an approved and
received after-sales refund. Never invoke a payment provider from a shell or
manually mark an order as refunded. For a `RETRY` record, restore the channel
or credentials and let `RefundDispatcher` reuse its stored
`refund_request_no`; Alipay treats that request number idempotently. A
`FAILED` record has no configured adapter (for example WeChat before its
adapter is deployed) and needs an approved payment-operations procedure before
any state change. On success, verify both `status='SUCCEEDED'` and the
associated `refund.completed` outbox event.
