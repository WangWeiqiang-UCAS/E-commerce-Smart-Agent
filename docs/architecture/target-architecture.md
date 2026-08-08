# Target architecture

The FastAPI application is an **agent prototype**, not the commerce system of
record. Do not add customer, catalogue, order, payment, inventory, or refund
domain tables to it.

| Domain | System of record | Agent access |
| --- | --- | --- |
| Customer, product, price, inventory, order, payment, after-sales | `apps/commerce-java` | Authenticated commerce API only |
| Conversation, message, knowledge, checkpoint, evaluation | `apps/agent-runtime-py` | Read/write |
| Customer chat and support workspace | TypeScript applications | BFF/API only |

## Migration guardrails

- Models cannot connect to databases. They use allow-listed, audited tools.
- State changes follow preview, explicit confirmation, then idempotent execution.
- Commerce validates ownership from the authenticated principal, never an LLM user ID.
- The commerce after-sales state machine alone triggers financial refunds.
- Cross-system data is joined by APIs or events; producers use an outbox and consumers deduplicate with an inbox.

The initial Java-to-agent boundary is defined by
`packages/api-contracts/agent-commerce.openapi.yaml`.

The browser-facing boundary is `apps/agent-bff-ts`; the embeddable customer
component in `apps/customer-widget-ts` calls only that BFF, never Python or
commerce services directly.

`apps/agent-desk-ts` is the separate human-support workspace. It requires an
administrator session and keeps all decision operations server-side.

Until Java and Agent identity are consolidated behind a shared issuer, the BFF
is the only permitted token bridge: it authenticates to both systems and stores
each system's token in an independent HttpOnly cookie. Neither browser code nor
LLM prompts may receive those tokens.
