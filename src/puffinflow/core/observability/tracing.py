import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Optional

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode

    _OPENTELEMETRY_AVAILABLE = True
except ImportError:
    # Create mock classes for when OpenTelemetry is not available
    trace = None
    JaegerExporter = None
    OTLPSpanExporter = None
    Resource = None
    TracerProvider = None
    BatchSpanProcessor = None
    ConsoleSpanExporter = None
    Status = None
    StatusCode = None
    _OPENTELEMETRY_AVAILABLE = False

from .config import TracingConfig
from .interfaces import Span, SpanContext, SpanType, TracingProvider


class OpenTelemetrySpan(Span):
    """OpenTelemetry span implementation"""

    def __init__(self, otel_span: Any, span_context: SpanContext):
        self._span = otel_span
        self._context = span_context
        self._start_time = time.time()

        # Set workflow context attributes if OpenTelemetry is available
        if _OPENTELEMETRY_AVAILABLE and self._span:
            if span_context.workflow_id:
                self._span.set_attribute("workflow.id", span_context.workflow_id)
            if span_context.agent_name:
                self._span.set_attribute("agent.name", span_context.agent_name)
            if span_context.state_name:
                self._span.set_attribute("state.name", span_context.state_name)
            if span_context.user_id:
                self._span.set_attribute("user.id", span_context.user_id)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute"""
        if _OPENTELEMETRY_AVAILABLE and self._span and key and value is not None:
            if isinstance(value, (dict, list)):
                value = str(value)
            self._span.set_attribute(key, value)

    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """Set span status"""
        if _OPENTELEMETRY_AVAILABLE and self._span:
            if status.lower() in ["ok", "success"]:
                self._span.set_status(Status(StatusCode.OK, description))
            elif status.lower() in ["error", "failed"]:
                self._span.set_status(Status(StatusCode.ERROR, description))

    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """Add event to span"""
        if _OPENTELEMETRY_AVAILABLE and self._span:
            event_attrs = attributes or {}
            event_attrs = {k: v for k, v in event_attrs.items() if v is not None}
            self._span.add_event(name, event_attrs)

    def record_exception(self, exception: Exception) -> None:
        """Record exception in span"""
        if _OPENTELEMETRY_AVAILABLE and self._span:
            self._span.record_exception(exception)
            self.set_status("error", str(exception))

    def end(self) -> None:
        """End span"""
        duration = time.time() - self._start_time
        self.set_attribute("span.duration_ms", duration * 1000)
        if _OPENTELEMETRY_AVAILABLE and self._span:
            self._span.end()

    @property
    def context(self) -> SpanContext:
        """Get span context"""
        return self._context


class OpenTelemetryTracingProvider(TracingProvider):
    """OpenTelemetry tracing provider"""

    def __init__(self, config: TracingConfig):
        self.config = config
        self._current_context = threading.local()
        self._tracer: Any = None
        if _OPENTELEMETRY_AVAILABLE:
            self._setup_tracing()

    def _setup_tracing(self) -> None:
        """Setup OpenTelemetry tracing"""
        if not _OPENTELEMETRY_AVAILABLE:
            return

        resource = Resource.create(
            {
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
            }
        )

        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

        # Setup exporters
        processors = []

        if self.config.otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
            processors.append(BatchSpanProcessor(otlp_exporter))

        if self.config.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name=self.config.jaeger_endpoint.split(":")[0],
                agent_port=(
                    int(self.config.jaeger_endpoint.split(":")[1])
                    if ":" in self.config.jaeger_endpoint
                    else 6831
                ),
            )
            processors.append(BatchSpanProcessor(jaeger_exporter))

        if self.config.console_enabled:
            console_exporter = ConsoleSpanExporter()
            processors.append(BatchSpanProcessor(console_exporter))

        for processor in processors:
            provider.add_span_processor(processor)

        self._tracer = trace.get_tracer(
            instrumenting_module_name="puffinflow.observability",
            instrumenting_library_version="1.0.0",
        )

    def start_span(
        self,
        name: str,
        span_type: SpanType = SpanType.SYSTEM,
        parent: Optional[SpanContext] = None,
        **attributes: Any,
    ) -> Span:
        """Start new span"""
        # Create span context
        if parent:
            span_context = parent.child_context()
        else:
            current_span = self.get_current_span()
            if current_span:
                span_context = current_span.context.child_context()
            else:
                span_context = SpanContext()

        # Start OpenTelemetry span if available
        otel_span = None
        if _OPENTELEMETRY_AVAILABLE and self._tracer:
            otel_span = self._tracer.start_span(name)

        # Create wrapper
        span = OpenTelemetrySpan(otel_span, span_context)

        # Set additional attributes
        span.set_attribute("span.type", span_type.value)
        for key, value in attributes.items():
            span.set_attribute(key, value)

        self._set_current_span(span)
        return span

    def get_current_span(self) -> Optional[Span]:
        """Get current active span"""
        return getattr(self._current_context, "current_span", None)

    def _set_current_span(self, span: Optional[Span]) -> None:
        """Set current span in context"""
        self._current_context.current_span = span

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType = SpanType.SYSTEM,
        parent: Optional[SpanContext] = None,
        **attributes: Any,
    ) -> Iterator[Span]:
        """Context manager for spans"""
        span = self.start_span(name, span_type, parent, **attributes)
        try:
            yield span
            span.set_status("ok")
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()
            self._set_current_span(None)
