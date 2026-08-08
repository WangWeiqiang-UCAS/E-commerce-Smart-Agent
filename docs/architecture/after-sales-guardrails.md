# After-sales guardrails

The imported portal's original return-apply implementation copied client input
directly into its persistence model. It now validates the authenticated member's
ownership, checks the requested product and quantity against the order, and
derives order, product, and amount fields from the commerce database.

This hardening applies to the existing customer endpoint. It does **not** make
that endpoint an Agent tool. The Agent API remains read-only until a dedicated
after-sales action model is introduced:

1. Preview creates a short-lived action token and stores the preview hash.
2. Customer confirmation submits the action token, preview hash, confirmation
   time, trace ID, and idempotency key.
3. The commerce after-sales state machine creates an application; only its
   approved terminal transition can invoke payment refund processing.

The agent must never call `/returnApply/create` directly.

## Implemented agent action persistence

`document/sql/migrations/V20260808_01__agent_refund_actions.sql` adds the
`oms_agent_action` and `oms_outbox_event` tables. It must be applied through the
deployment migration process before enabling the two Agent endpoints. The action
service stores only a SHA-256 token hash, expires previews after ten minutes,
locks the action row during confirmation, validates the preview hash and trace
ID, and accepts the same idempotency key only as a retry of a completed action.

Confirmation creates an after-sales application only. It cannot call a payment
refund API; that remains a later state-machine transition.

The same transaction also writes an `after_sale.created` event to the outbox.
`OutboxPublisher` publishes committed rows to the durable `mall.agent.events`
RabbitMQ exchange; consumers must store and deduplicate `eventId` in an inbox.

For local development, the root Compose stack includes RabbitMQ and
`commerce_event_consumer`. Apply both the Java SQL migration and Agent Alembic
migration before sending events.
