"""Vendor-neutral OpenTelemetry bootstrap with safe, low-cardinality spans."""
from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.core.config import settings

_configured = False


def configure_telemetry(app) -> None:
    """Configure tracing once; no endpoint means local context propagation only."""
    global _configured
    if _configured:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": settings.OTEL_SERVICE_NAME}))
    if settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT.rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint += "/v1/traces"
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="health,metrics")
    _configured = True


def commerce_tracer():
    return trace.get_tracer("agent-runtime.commerce")
