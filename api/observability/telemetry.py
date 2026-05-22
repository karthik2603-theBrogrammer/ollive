from __future__ import annotations

import logging
import threading

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from api.observability.trace_store import TRACE_STORE, InMemorySpanExporter

logger = logging.getLogger(__name__)

_tracer_provider: TracerProvider | None = None
_setup_lock = threading.Lock()


def _patch_openinference_text_accessor() -> None:
    """Avoid OpenInference calling TextAccessor as a lazy getter (LangChain deprecation)."""
    try:
        from langchain_core.messages.base import TextAccessor
        from openinference.instrumentation.config import TraceConfig
    except ImportError:
        return

    if getattr(TraceConfig, "_ollive_text_accessor_patch", False):
        return

    original_mask = TraceConfig.mask

    def patched_mask(self, key, value):
        if isinstance(value, TextAccessor):
            value = str(value)
        return original_mask(self, key, value)

    TraceConfig.mask = patched_mask  # type: ignore[method-assign]
    TraceConfig._ollive_text_accessor_patch = True


def setup_telemetry(
    *,
    service_name: str,
    enabled: bool = True,
    max_spans: int = 2000,
) -> TracerProvider | None:
    """Initialize OpenTelemetry tracing with in-memory span storage for the built-in UI."""
    global _tracer_provider

    if not enabled:
        logger.warning("OpenTelemetry observability disabled via config")
        return None

    with _setup_lock:
        if _tracer_provider is not None:
            return _tracer_provider

        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor

            TRACE_STORE._max_spans = max_spans

            resource = Resource.create(
                {
                    "service.name": service_name,
                    "service.namespace": "ollive",
                }
            )
            provider = TracerProvider(resource=resource)
            exporter = InMemorySpanExporter(TRACE_STORE)
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            trace.set_tracer_provider(provider)

            _patch_openinference_text_accessor()
            LangChainInstrumentor().instrument(tracer_provider=provider)

            _tracer_provider = provider
            logger.warning(
                "OpenTelemetry tracing enabled service=%s max_spans=%s",
                service_name,
                max_spans,
            )
            return _tracer_provider
        except Exception as exc:
            logger.warning("Failed to initialize OpenTelemetry observability: %s", exc)
            return None


def instrument_fastapi(app) -> None:
    if _tracer_provider is None:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception as exc:
        logger.warning("FastAPI instrumentation skipped: %s", exc)


def get_tracer(name: str = "ollive.api"):
    if _tracer_provider is None:
        return trace.get_tracer(name)
    return _tracer_provider.get_tracer(name)
