# Agent workflow: router + deterministic subgraphs

## Graph

```
START → intent_router
         ├─ GUIDE   → guide_subgraph   (CommerceClient.search_products → generate)
         ├─ ORDER   → order_subgraph   (CommerceClient.get_orders → generate)
         ├─ POLICY  → policy_subgraph  (pgvector RAG → generate)
         ├─ REFUND  → after_sale_subgraph
         │            ├─ after-sale status → generate
         │            └─ refund intent → interrupt(user confirmation) → handoff → generate
         └─ OTHER   → fallback_subgraph (generate only)
```

Implementation: `app/graph/workflow.py`, `app/graph/nodes.py`.

## Write-intent guardrail

Refund requests never reach a write tool. `handle_after_sale` pauses with
LangGraph `interrupt()` and a structured payload
(`kind=refund_preview_confirmation`, order, item, quantity, reason). The chat
SSE stream emits the interrupt; the widget renders a confirm/cancel dialog and
`POST /api/v1/chat/resume` resumes the graph with a structured decision. Only a
confirmed resume produces the `handoff` payload that the BFF uses to drive the
Java preview/confirm endpoints.

## Safety properties

- Order ownership is derived from the Java authenticated principal.
- PII is redacted before any LLM call (`app/graph/safety.py`).
- Retrieved policy text is labeled as untrusted reference data and citations
  carry `source`, `version`, `chunk_id`, and `document_id`.
- Tool names/parameters/timeouts are bounded in `CommerceClient` and the Java
  controller; rate limits apply per user on chat and resume.
- Model failures fail closed to `OTHER` (fallback) and the commerce path is
  never blocked by Agent availability.

## Checkpoints

Production uses Redis checkpoints (`AsyncRedisSaver`). Tests compile with an
in-memory saver so interrupt/resume is covered without infrastructure
(`test/test_agent_workflow.py`).
