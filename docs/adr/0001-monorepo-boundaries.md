# ADR 0001: Monorepo with independent build chains

## Status

Accepted.

## Context

The prototype mixed commerce models, Agent logic, and a Gradio frontend in one
Python application. target.md section 8 specifies a monorepo with Java, Python,
and TypeScript build chains.

## Decision

Use the target.md layout:

- `apps/commerce-java` — Maven (upstream macrozheng/mall baseline).
- `app` — Poetry (FastAPI + LangGraph Agent runtime).
- `apps/agent-bff-ts`, `apps/customer-widget-ts`, `apps/agent-desk-ts` — npm.
- `packages/` for shared contracts and tool schemas; `deploy/`, `evals/`,
  `docs/`, `third_party/` at the root.

## Consequences

Each chain builds independently, enabling focused CI jobs. Cross-language
contracts are versioned in `packages/api-contracts`; breaking changes require an
OpenAPI diff review.
