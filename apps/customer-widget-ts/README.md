# Customer widget

Embeddable React customer-support component. Host it under the same origin as
`agent-bff-ts`; it calls only BFF routes and never reads authentication tokens.
It loads persisted history on mount (refresh-safe), streams SSE tokens, and
renders the refund confirmation interrupt with explicit confirm/cancel.

For refunds, the host supplies the selected commerce order/product IDs. The
widget requests a preview, displays the amount and expiry, and sends a separate
explicit confirmation with the same trace ID. It cannot issue a payment refund.
