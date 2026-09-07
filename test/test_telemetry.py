import os
from pathlib import Path

for key, value in {
    "PROJECT_NAME": "test", "API_V1_STR": "/api/v1", "POSTGRES_SERVER": "localhost",
    "POSTGRES_USER": "postgres", "POSTGRES_PASSWORD": "test", "POSTGRES_DB": "test",
    "POSTGRES_PORT": "5432", "REDIS_HOST": "localhost", "REDIS_PORT": "6379",
    "OPENAI_BASE_URL": "http://localhost", "OPENAI_API_KEY": "test", "SECRET_KEY": "test-secret",
}.items():
    os.environ.setdefault(key, value)


def test_telemetry_uses_otlp_and_excludes_health_metrics_endpoints():
    from app import main
    from app.core import telemetry

    source = Path("app/core/telemetry.py").read_text(encoding="utf-8")
    assert "OTLPSpanExporter" in source
    assert 'excluded_urls="health,metrics"' in source
    assert "service.name" in source
    assert 'trace.get_tracer("agent-runtime.commerce")' in source
    assert main.app is not None
    assert telemetry._configured is True


def test_python_endpoint_accepts_collector_base_or_otlp_trace_path_once():
    source = Path("app/core/telemetry.py").read_text(encoding="utf-8")
    assert 'if not endpoint.endswith("/v1/traces"):' in source
    assert 'endpoint += "/v1/traces"' in source


def test_java_commerce_exports_traces_to_the_shared_otlp_collector():
    pom = Path("apps/commerce-java/mall-portal/pom.xml").read_text(encoding="utf-8")
    compose = Path("apps/commerce-java/mall-portal/src/main/resources/application-compose.yml").read_text(encoding="utf-8")
    prod = Path("apps/commerce-java/mall-portal/src/main/resources/application-prod.yml").read_text(encoding="utf-8")

    assert "spring-boot-starter-opentelemetry" in pom
    assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT:http://otel-collector:4318/v1/traces" in compose
    assert "endpoint: ${OTEL_EXPORTER_OTLP_TRACES_ENDPOINT}" in prod


def test_bff_registers_standard_otlp_instrumentation_and_receives_exporter_configuration():
    instrumentation = Path("apps/agent-bff-ts/src/instrumentation.ts").read_text(encoding="utf-8")
    compose = Path("docker-compose.yaml").read_text(encoding="utf-8")
    package = Path("apps/agent-bff-ts/package.json").read_text(encoding="utf-8")

    assert 'registerOTel({ serviceName: process.env.OTEL_SERVICE_NAME ?? "agent-bff" })' in instrumentation
    assert '"@vercel/otel"' in package
    assert "OTEL_EXPORTER_OTLP_ENDPOINT: ${OTEL_EXPORTER_OTLP_ENDPOINT:-}" in compose


def test_commerce_spans_use_allow_listed_tool_names_not_customer_identifiers():
    source = Path("app/services/commerce_client.py").read_text(encoding="utf-8")
    assert 'start_as_current_span("commerce.tool.read")' in source
    assert 'span.set_attribute("commerce.tool.name", tool_name)' in source
    assert 'span.set_attribute("commerce.outcome", outcome)' in source
    assert "order_id" not in source[source.index('start_as_current_span("commerce.tool.read")'):]
