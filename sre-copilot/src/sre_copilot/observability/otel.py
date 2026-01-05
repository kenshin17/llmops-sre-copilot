from __future__ import annotations

import uuid
from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from sre_copilot.config import Settings
from sre_copilot.utils.logger import get_logger

logger = get_logger(__name__)

_otel_initialized = False


def setup_otel(app: FastAPI, settings: Settings) -> None:
    """Configure OpenTelemetry exporters for traces and metrics (OTLP gRPC)."""
    global _otel_initialized
    if _otel_initialized or not settings.otel_enabled or not settings.otel_endpoint:
        if settings.otel_enabled:
            logger.info("OTEL initialization skipped (already initialized or missing endpoint).")
        return

    try:
        resource = Resource.create(
            {
                "service.name": settings.otel_service_name,
                "service.namespace": "sre-copilot",
            }
        )

        tracer_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint, insecure=True)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)

        if settings.otel_metrics_enabled:
            metric_exporter = OTLPMetricExporter(endpoint=settings.otel_endpoint, insecure=True)
            reader = PeriodicExportingMetricReader(metric_exporter)
            meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
            metrics.set_meter_provider(meter_provider)
        else:
            logger.info("OTEL metrics exporter disabled (otel_metrics_enabled=false).")

        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=tracer_provider,
            excluded_urls=settings.otel_excluded_urls,
        )
        HTTPXClientInstrumentor().instrument()
        RequestsInstrumentor().instrument()

        _add_request_context_middleware(app)

        _otel_initialized = True
        logger.info("OTEL initialized; exporting to %s", settings.otel_endpoint)
    except Exception as exc:  # pragma: no cover - network/exporter failures
        logger.warning("OTEL initialization failed: %s", exc)


def _add_request_context_middleware(app: FastAPI) -> None:
    """Add request/response hooks for request IDs and span decoration."""

    @app.middleware("http")
    async def _otel_request_context(request: Request, call_next: Callable) -> Response:  # type: ignore[override]
        request_id = request.headers.get("x-request-id") or request.headers.get("X-Request-Id") or uuid.uuid4().hex
        span = trace.get_current_span()
        if span:
            span.set_attribute("http.request_id", request_id)
            span.set_attribute("http.target", request.url.path)
            span.set_attribute("http.method", request.method)
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        if span:
            span.set_attribute("http.status_code", response.status_code)
        return response
