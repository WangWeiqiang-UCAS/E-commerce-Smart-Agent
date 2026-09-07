# Agent commerce boundary

The Agent and BFF interact with commerce only through
`/agent/commerce/**` in `apps/commerce-java` (`AgentCommerceController`).

## Read tools (LLM-visible)

| Tool | Method | Notes |
| --- | --- | --- |
| search_products | GET `/agent/commerce/products/search` | published products only, bounded params |
| get_product_detail | GET `/agent/commerce/products/{productId}` | includes SKU stock projection |
| get_user_orders | GET `/agent/commerce/orders` | member scoped by JWT |
| get_order_detail | GET `/agent/commerce/orders/{orderId}` | ownership enforced |
| get_logistics | GET `/agent/commerce/orders/{orderId}/logistics` | non-PII projection |
| get_after_sale_status | GET `/agent/commerce/after-sales/{afterSaleId}` | ownership enforced |

## Write actions (never LLM-direct)

POST `/agent/commerce/actions/refund/preview` then
POST `/agent/commerce/actions/refund/confirm`. Confirm requires `actionToken`,
`idempotencyKey`, `previewHash`, `agentTraceId`, and `confirmedAt`. It creates
an after-sales application and writes an `after_sale.created` outbox event.
Funds are refunded only by the Java after-sales state machine.
After the approved return is received, Java creates a durable payment refund
record; the channel dispatcher emits `refund.completed` only after the provider
accepts the idempotent refund request.

Pending-order cancellation follows the same boundary:

POST `/agent/commerce/actions/order-cancellation/preview` then
POST `/agent/commerce/actions/order-cancellation/confirm`. The confirmation
requires `actionToken`, `idempotencyKey`, `previewHash`, `agentTraceId`, and
`confirmedAt`. Java validates the authenticated owner and that the order is
still pending, performs one conditional cancellation, then publishes an
`order.status_changed` outbox event. Browser-facing portal cancellation routes
remain disabled.

## Customer evidence boundary

After an after-sales application is created, the BFF may call the authenticated
Portal-only evidence endpoints: `POST /returnApply/evidence/upload`,
`POST /returnApply/evidence/{afterSaleId}`, and
`GET /returnApply/evidence/{afterSaleId}/download`. These are not LLM tools.
They accept only controlled private object keys and issue short-lived download
URLs after an ownership check.

## Envelope

All events: `event_id`, `event_type`, `schema_version`, `occurred_at`,
`aggregate_id`, `trace_id`.
