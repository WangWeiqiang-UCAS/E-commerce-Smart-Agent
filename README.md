# E-commerce Smart Agent

This repository is the production-oriented migration of a customer-support
Agent prototype into a commerce-integrated system. The Java commerce system is
the single source of truth for members, catalogue, stock, orders, payments,
logistics, and after-sales. The Python Agent runtime owns conversations,
knowledge, checkpoints, execution records, and event inbox projections. The
TypeScript applications own the browser boundaries.

## Architecture

```
商城用户 ──► 商城前台 (mall-app-web)         人工客服 ──► agent-desk-ts
商城用户 ──► customer-widget-ts ──────────► agent-bff-ts ──► agent-runtime-py (FastAPI + LangGraph)
                                                     │              │
                                                     ▼              ▼
                                        commerce-java (mall-portal)  PostgreSQL + pgvector
                                                     │
                                        MySQL + Redis + RabbitMQ (outbox → inbox)
```

Key principles (see `target.md`):

- The Agent never reads the commerce database; it calls the allow-listed
  boundary in `packages/api-contracts/agent-commerce.openapi.yaml`.
- Refund flow is preview → explicit confirmation → idempotent Java after-sales
  creation. Payment refunds are owned by the Java state machine.
- Refund intent pauses in the Agent graph at a LangGraph `interrupt`; the
  customer confirms in the widget before any handoff is produced.
- Domain events use a producer outbox (`oms_outbox_event`) and consumer inbox
  (`inbox_events`) with `event_id` deduplication.
- PII is redacted before LLM calls; retrieved policy text is untrusted input
  with `source/version/chunk_id` citations.

## Repository layout

```
apps/
  commerce-java/       # fork of macrozheng/mall (Spring Boot 3.5 / Java 17)
  agent-runtime-py/    # FastAPI + LangGraph (root `app/` package lives here)
  agent-bff-ts/        # Next.js BFF: chat, history, resume, refund, sessions
  customer-widget-ts/  # embeddable customer chat + refund confirmation
  agent-desk-ts/       # human-support workspace (admin-only threads API)
packages/
  api-contracts/       # OpenAPI boundary contract
  tool-schemas/        # JSON Schema for LLM-visible tools
  ui-components/       # shared card components (extract when shared)
deploy/                # compose overrides, helm notes, observability
evals/                 # datasets, graders, offline regression
docs/                  # architecture, ADRs, API, licenses
third_party/           # NOTICE, SBOM, upstream revisions
```

## Local development

Copy `.env.example` to `.env`, provide strong local credentials, set
`BFF_SHARED_SECRET` and `SECRET_KEY`, then start the stack:

```powershell
docker compose up --build
```

The `migrate` container applies Agent PostgreSQL migrations; MySQL
initialization includes the upstream `mall.sql` plus the agent action/outbox
schema. The commerce event consumer deduplicates `after_sale.created` events
into the inbox and projects support cards.

## Verification

```powershell
poetry install --with dev
poetry run pytest -q
npm run build --prefix apps/agent-bff-ts
npm run build --prefix apps/customer-widget-ts
npm run build --prefix apps/agent-desk-ts
```

CI (`.github/workflows/ci.yml`) runs Python tests, the Java portal module tests,
and the three TypeScript builds on every push.

## Phase status

- Phase 0 (freeze/fix prototype): complete — syntax fixed, admin auth sealed,
  Gradio/celery retired, Compose/Dockerfile/CI/.env.example in place.
- Phase 1 (Java commerce): baseline imported with the agent boundary, refund
  preview/confirm, and outbox implemented; upstream revision recorded.
- Phase 2 (Python Agent): dual writes removed; router + subgraphs; LangGraph
  interrupt/resume; PII redaction; rate limiting; versioned knowledge ETL.
- Phase 3 (TypeScript): BFF, widget, and desk built; SSE, history persistence,
  and refund confirmation wired end to end.
- Phase 4 (real integrations): scaffolding for outbox/inbox, Alipay/OSS
  adapters from upstream, logistics read API; production channel credentials
  and closed-loop after-sales flows remain external integration work.
- Phase 5 (production hardening): observability configs, SBOM/license records,
  and ADRs provided; load testing, backup drills, and gray release are
  operational tasks outside this repository.

See `docs/architecture/target-architecture.md` and
`docs/architecture/agent-workflow.md` for details.
