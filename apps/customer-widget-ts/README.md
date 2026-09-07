# Customer widget

Embeddable React customer-support component. Host it under the same origin as
`agent-bff-ts`; it calls only BFF routes and never reads authentication tokens.
It loads persisted history on mount (refresh-safe), streams SSE tokens, and
renders the refund confirmation interrupt with explicit confirm/cancel.

Product searches remain in the same authenticated BFF boundary. Customers can
open an item to fetch its current SKU price and stock through the BFF; the
widget never receives or stores a commerce access token.

For refunds, the host supplies the selected commerce order/product IDs. The
widget requests a preview, displays the amount and expiry, and sends a separate
explicit confirmation with the same trace ID. It cannot issue a payment refund.

For a cancellation, the host supplies `cancellationSelection` with the selected
order. The widget can only cancel a Java-validated pending order through a
separate preview and explicit confirmation; each confirmation carries a fresh
idempotency key that is retained for retries of that preview, and the browser
never calls the portal cancellation endpoints. Refund confirmation uses the
same retry-safe key lifecycle.
