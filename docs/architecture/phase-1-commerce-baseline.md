# Phase 1 — commerce baseline

`apps/commerce-java` is the imported `macrozheng/mall` baseline at revision
`0504e86b1f1b6f1b8aa6a734d37a90fb67346be7` (Spring Boot 3.5 / Java 17).

The first safe agent boundary is implemented in `mall-portal`:

- `GET /agent/commerce/orders`
- `GET /agent/commerce/orders/{orderId}`
- `POST /agent/commerce/actions/refund/preview`
- `POST /agent/commerce/actions/refund/confirm`

Both derive membership from the authenticated Java commerce session. Order detail
now rejects records not belonging to that member. The Python runtime must use
this boundary, rather than the portal's general-purpose endpoints or its database.

The refund endpoints use a durable, ten-minute action token, preview hash,
authenticated ownership checks, trace ID validation, and an idempotency key.
Confirmation creates an after-sales application rather than refunding funds.
Apply `document/sql/migrations/V20260808_01__agent_refund_actions.sql` before
enabling them. The public contract is in
`packages/api-contracts/agent-commerce.openapi.yaml`.
