# Third-party upstreams

## macrozheng/mall

- Source: https://github.com/macrozheng/mall
- Imported revision: `0504e86b1f1b6f1b8aa6a734d37a90fb67346be7`
- Local location: `apps/commerce-java`
- License: Apache License 2.0; the upstream `LICENSE` is retained in that directory.

Local changes must be documented alongside upstream revision updates. The current
integration adds a narrow authenticated agent read boundary
(`AgentCommerceController`), an order ownership check, and the refund
preview/confirm action flow backed by `oms_agent_action` (token hash,
idempotency key, trace-ID binding) plus the `oms_outbox_event` outbox and
`OutboxPublisher`. The commerce after-sales state machine still owns payment
refunds.
