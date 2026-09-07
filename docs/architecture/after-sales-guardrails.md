# After-sales guardrails

The imported portal's original return-apply implementation copied client input
directly into its persistence model. It now validates the authenticated member's
ownership, checks the requested product and quantity against the order, and
derives order, product, and amount fields from the commerce database.
Client-submitted evidence URLs are never persisted as `proof_pics`; only a
controlled object-storage upload may later link an attachment to a durable
after-sales application.

Evidence upload accepts at most three JPEG, PNG, or PDF files (5 MiB each).
The BFF forwards authenticated multipart data to Java; Java validates file
signatures, writes a private member-scoped object key to MinIO, and stores that
key in `oms_after_sale_attachment`. Download links are owner-scoped and expire
after five minutes. Neither browser code nor the Agent receives object-storage
credentials.

This hardening applies to the existing customer endpoint. It does **not** make
that endpoint an Agent tool. The only Agent write actions are dedicated
preview/confirm flows; after-sales uses the following model:

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

Applications require a paid, owned order. Pending and completed applications
consume the requested item quantity; rejected applications do not, so repeated
requests cannot exceed the purchased quantity.

Administrative processing uses the only valid transitions: `0` pending to `1`
returning, `1` returning to `2` completed, or `0` pending to `3` rejected.
Each transition has a database predicate on its prior status, and an approved
amount cannot exceed the application amount.

The `1` to `2` received-goods transition creates exactly one durable
`oms_payment_refund` record in the same transaction. `RefundDispatcher` is the
only component that calls a payment channel, and it reuses the stored merchant
refund request number on every retry. The initial adapter is Alipay; an
unsupported payment channel remains visibly failed for manual handling instead
of silently treating funds as refunded. A successful provider response updates
the refund record and emits `refund.completed` in the same transaction.

The same transaction also writes an `after_sale.created` event to the outbox.
Each successful administrative state transition writes an
`after_sale.status_changed` event in its transaction as well.
`OutboxPublisher` publishes committed rows to the durable `mall.agent.events`
RabbitMQ exchange; consumers must store and deduplicate `eventId` in an inbox.

For local development, the root Compose stack includes RabbitMQ and
`commerce_event_consumer`. Apply both the Java SQL migration and Agent Alembic
migration before sending events.
