# Deployment

The repository is a monorepo with three independent build chains:

- `apps/commerce-java` — Maven (Spring Boot 3.5 / Java 17). The system of
  record for members, catalogue, stock, orders, payments, logistics, and
  after-sales.
- `app` — Poetry (Python 3.11+, FastAPI + LangGraph). Conversation, knowledge,
  checkpoints, execution records, and the event inbox.
- `apps/agent-bff-ts`, `apps/customer-widget-ts`, `apps/agent-desk-ts` — npm
  (Next.js / TypeScript). Browser boundaries.

## Local stack

The root `docker-compose.yaml` starts every service, including MySQL with the
upstream `mall.sql` plus the agent action/outbox migration, RabbitMQ, and the
Python event consumer:

```powershell
Copy-Item .env.example .env
# edit .env: SECRET_KEY, BFF_SHARED_SECRET, OPENAI_API_KEY
docker compose up --build
```

## Environment-specific overrides

- `deploy/compose/` — environment overrides layered on the root stack.
- `deploy/observability/` — OpenTelemetry Collector, Prometheus, and Grafana
  for traces, metrics, and logs.
- `deploy/helm/` — reserved for Kubernetes packaging once scaling needs are
  concrete (start with the modular monolith before microservices).
