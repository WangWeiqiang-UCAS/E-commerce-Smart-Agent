# Target architecture

The FastAPI application is an **agent prototype**, not the commerce system of
record. Do not add customer, catalogue, order, payment, inventory, or refund
domain tables to it.

| Domain | System of record | Agent access |
| --- | --- | --- |
| Customer, product, price, inventory, order, payment, after-sales | `apps/commerce-java` | Authenticated commerce API only |
| Conversation, message, knowledge, checkpoint, evaluation | `apps/agent-runtime-py` | Read/write |
| Internal support operators | `apps/agent-runtime-py` | Agent-desk authentication only |
| Customer chat and support workspace | TypeScript applications | BFF/API only |

## Migration guardrails

- Models cannot connect to databases. They use allow-listed, audited tools.
- State changes follow preview, explicit confirmation, then idempotent execution.
- Commerce validates ownership from the authenticated principal, never an LLM user ID.
- Customer cancellation routes validate the current member's ownership before
  scheduling or executing a cancellation; generic cancellation methods are
  reserved for trusted timeout consumers.
- Legacy browser-facing Portal cancellation endpoints are disabled. Customer
  cancellation is available only through the Agent Commerce preview/confirm
  action boundary.
- Bulk timeout cancellation is a background-only operation and is never
  invoked through a browser-facing portal endpoint.
- Agent-assisted order cancellation is a short-lived preview followed by an
  explicit customer confirmation with an idempotency key. Java rechecks the
  customer, pending-order status and previewed status snapshot before it makes
  the single state transition, then emits `order.status_changed` through the
  outbox.
- Java commerce requires `COMMERCE_JWT_SECRET` outside local development; the
  Compose and production profiles fail without it.
- The production Java profile reads database, cache, broker, Druid and payment
  credentials only from deployment environment variables; repository defaults
  are limited to local development profiles.
- The commerce after-sales state machine alone creates financial refund work;
  its durable dispatcher is the only component allowed to call a payment
  channel and uses the payment provider's idempotency key on retries.
- Payment-provider callbacks perform signature verification and an atomic
  pending-to-paid transition before consuming locked inventory. They bind the
  configured app ID, order number and amount before the state transition;
  browser-facing payment-success simulation endpoints do not mutate order state.
- Checkout reserves SKU inventory through a conditional update; payment consumes
  that reservation, while a single conditional pending-to-cancelled transition
  releases it exactly once for timeout or cancellation processing.
- Receipt confirmation uses the same conditional-transition rule: only a
  shipped owned order may become received; retries observe the resulting state
  without creating another transition or `order.status_changed` event.
- Hiding a terminal order is ownership-scoped and conditional. It never changes
  the order's business status, and concurrent or repeated hide requests are
  idempotent.
- Cross-system data is joined by APIs or events; producers use an outbox and consumers deduplicate with an inbox.
- The Agent RabbitMQ consumer acknowledges only committed Inbox writes; malformed
  events route to `agent.runtime.events.dlq`, while transient persistence
  failures are requeued for safe retry.
- The Agent database contains no commerce customer table or customer registration endpoint;
  its `support_operators` table is limited to internal desk access.
- Retrieved policy evidence is returned with its source, version, document ID
  and chunk ID; citations are appended deterministically when the model omits
  them.
- Every order, product, and after-sales read placed in a model prompt receives
  a deterministic allow-listed tool provenance marker in the final answer.
- Tracking requests invoke the authenticated Java logistics projection for the
  owned order; only carrier, tracking ID, and delivery timestamps enter Agent
  state or model context.
- Order questions locate an owned order through the bounded list projection,
  then fetch its current detail through the authenticated detail endpoint. A
  temporary detail failure may only fall back to the list projection, never to
  model-invented order state.
- An explicit product ID invokes the current Java product-detail projection;
  only bounded product and SKU identifiers, price, and stock fields enter the
  model context, never free-form catalogue attributes or descriptions.
- Customer chat requests are limited per authenticated principal through Redis
  with opaque hashed keys. A short Redis outage falls back to a process-local
  limit so the support channel remains available while preserving a bounded
  abuse control.

The initial Java-to-agent boundary is defined by
`packages/api-contracts/agent-commerce.openapi.yaml`.

The browser-facing boundary is `apps/agent-bff-ts`; the embeddable customer
component in `apps/customer-widget-ts` calls only that BFF, never Python or
commerce services directly.

`apps/agent-desk-ts` is the separate human-support workspace. It requires an
administrator session and keeps all decision operations server-side.

## Observability baseline

Every BFF, Agent, and Commerce request carries `X-Trace-Id`. The Agent exposes
dependency-free Prometheus-compatible metrics at `/metrics` for internal
scraping: bounded HTTP request counts/durations and allow-listed Commerce tool
counts/durations. Metrics deliberately never include user IDs, order numbers,
tokens, questions, or raw URL parameters as labels.

The Agent also creates OpenTelemetry spans for HTTP requests and allow-listed
Commerce read tools. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to the Collector HTTP
endpoint (for example `http://otel-collector:4318`); span attributes are
limited to service name, tool name, HTTP method/status and outcome, never a
customer identifier, order number, token, question or URL query.

Commerce exposes bounded Outbox recovery metrics through its internal Actuator
Prometheus endpoint: pending events, age of the oldest unpublished event, and
expired publish leases. Operators use these signals with
`deploy/runbooks/event-recovery.md`; replay preserves the original event ID and
is safe because the Agent Inbox deduplicates it before acknowledgement.

The BFF stores only the Java commerce token in an HttpOnly cookie for customer
sessions. Python resolves the Java-authenticated `memberId` through the
allow-listed `/agent/commerce/identity` projection and uses it solely to
namespace Agent conversation data. Browser code and LLM prompts never receive
the token. Support-operator authentication remains a separate internal
workspace concern, not a customer/member record.
