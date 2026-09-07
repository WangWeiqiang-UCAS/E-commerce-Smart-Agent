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
upstream `mall.sql` plus the agent action/outbox/refund/evidence migrations,
RabbitMQ, private MinIO evidence storage, and the Python event consumer:

```powershell
Copy-Item .env.example .env
# edit .env: SECRET_KEY, BFF_SHARED_SECRET, OPENAI_API_KEY, and MinIO credentials
docker compose up --build
```

MinIO is intentionally not published on a host port. The Commerce service waits
for its S3 readiness probe before starting; evidence credentials stay within
the backend Compose network and must never be placed in browser configuration.

## Release safety

Before every production release, follow the [database backup, migration, and
restore runbook](runbooks/database-backup-restore.md). Compose initialization
scripts apply only to a new MySQL volume; existing environments require the
versioned Commerce migrations through the approved migration connection.

## Environment-specific overrides

- `deploy/compose/` — environment overrides layered on the root stack.
- `deploy/observability/` — OpenTelemetry Collector, Prometheus, and Grafana
  for traces, metrics, and logs.
- `deploy/helm/` — reserved for Kubernetes packaging once scaling needs are
  concrete (start with the modular monolith before microservices).
