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
