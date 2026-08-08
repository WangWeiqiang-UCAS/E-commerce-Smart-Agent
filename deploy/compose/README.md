# Compose overrides

The root `docker-compose.yaml` is the canonical local stack. Environment
specific overlays live here and are applied with:

```powershell
docker compose -f docker-compose.yaml -f deploy/compose/observability.yaml up
```

Example overlays to add when needed:

- `observability.yaml` — add Prometheus / Grafana / OTel Collector sidecars
  (see `deploy/observability`).
- `prod.yaml` — resource limits, read-only mounts, log drivers, and non-default
  secrets; never commit real secrets.
- `agents-only.yaml` — Agent runtime + Postgres + Redis without the Java stack,
  useful for isolated evaluation runs.
