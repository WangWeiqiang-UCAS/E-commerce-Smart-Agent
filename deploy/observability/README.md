# Observability

Target: Langfuse OSS for LLM traces, prompts, datasets, and evaluation;
OpenTelemetry as the unified vendor-neutral signals layer across Java, Python,
and TypeScript; Prometheus + Grafana for metrics and dashboards.

Files in this directory are the baseline configuration:

- `otel-collector-config.yaml` — receives OTLP and forwards to a configured
  backend (Langfuse OSS or any OTLP endpoint).
- `prometheus.yml` — scrape config for the Agent runtime and Java actuator.
- `grafana-datasources.yaml` — Prometheus datasource provisioning.
- `docker-compose.observability.yaml` — standalone stack for local evaluation.

Start the application Compose stack first; it creates the internal
`ecommerce_internal` network. Then start this stack with:

```powershell
# In the application .env:
# OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
# OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://otel-collector:4318/v1/traces
# GRAFANA_ADMIN_PASSWORD=<long-random-password>
docker compose -f deploy/observability/docker-compose.observability.yaml up -d
```

`GRAFANA_ADMIN_PASSWORD` is required by Compose.
`OTEL_EXPORTER_OTLP_ENDPOINT` enables OTLP trace export from the Python Agent;
the Java Commerce profile uses the full `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`.
The BFF reads the same standard exporter variables and exports its server-side
route and `fetch` spans as the `agent-bff` service.
Leaving the Python value empty keeps local development telemetry in-process
without an external exporter.

Prometheus reaches the Agent runtime as `app:8000` and Java commerce as
`commerce_java:8085` on that internal network; neither scrape target needs to
be exposed through a public host port.

## Langfuse

Langfuse core is MIT licensed, but the `ee/` directory has a separate license.
When vendoring or extending it, keep the enterprise directory isolated and
record the upstream revision in `third_party/`.

## Trace correlation

Every Python request carries `X-Trace-Id` (`app/core/tracing.py`), forwarded by
the BFF and into Java commerce calls. The Java portal adds
`TraceIdFilter`/`TraceIdFilter`-style correlation for the agent boundary. Use
the same trace ID when evaluating dataset examples so chat, tool calls, and
commerce writes can be joined.
