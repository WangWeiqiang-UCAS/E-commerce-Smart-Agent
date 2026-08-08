# Helm

Reserved for Kubernetes packaging. Follow the target architecture guidance:
keep the commerce system a modular monolith until order volume, team size, or
independent scaling requirements justify splitting into microservices
(see `docs/adr/0002-commerce-truth-source.md`).

Planned charts:

- `agent-runtime` — FastAPI + LangGraph with Postgres/pgvector and Redis.
- `agent-bff` — Next.js BFF with the commerce + agent session bridge.
- `commerce-portal` — mall-portal module (same image as the modular monolith).

No chart is committed yet; this directory documents the intended layout.
