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

## Envelope

All events: `event_id`, `event_type`, `schema_version`, `occurred_at`,
`aggregate_id`, `trace_id`.
