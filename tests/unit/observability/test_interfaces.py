from datetime import datetime
from typing import Optional
from unittest.mock import Mock

import pytest

from puffinflow.core.observability.interfaces import (
    AlertingProvider,
    AlertSeverity,
    EventProcessor,
    Metric,
    MetricsProvider,
    MetricType,
    ObservabilityEvent,
    Span,
    SpanContext,
    SpanType,
    TracingProvider,
)


class TestEnums:
    """Test enum classes"""

    def test_span_type_values(self):
        """Test SpanType enum values"""
        assert SpanType.WORKFLOW.value == "workflow"
        assert SpanType.STATE.value == "state"
        assert SpanType.RESOURCE.value == "resource"
        assert SpanType.BUSINESS.value == "business"
        assert SpanType.SYSTEM.value == "system"

    def test_metric_type_values(self):
        """Test MetricType enum values"""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"

    def test_alert_severity_values(self):
        """Test AlertSeverity enum values"""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestSpanContext:
    """Test SpanContext class"""

    def test_default_initialization(self):
        """Test default initialization generates UUIDs"""
        context = SpanContext()

        # Verify UUIDs are generated
        assert context.trace_id is not None
        assert context.span_id is not None
        assert len(context.trace_id) > 0
        assert len(context.span_id) > 0

        # Verify optional fields are None
        assert context.parent_span_id is None
        assert context.workflow_id is None
        assert context.agent_name is None
        assert context.state_name is None
        assert context.user_id is None
        assert context.session_id is None

    def test_custom_initialization(self):
        """Test initialization with custom values"""
        context = SpanContext(
            trace_id="custom-trace-id",
            span_id="custom-span-id",
            parent_span_id="parent-span-id",
            workflow_id="workflow-123",
            agent_name="test-agent",
            state_name="test-state",
            user_id="user-456",
            session_id="session-789",
        )

        assert context.trace_id == "custom-trace-id"
        assert context.span_id == "custom-span-id"
        assert context.parent_span_id == "parent-span-id"
        assert context.workflow_id == "workflow-123"
        assert context.agent_name == "test-agent"
        assert context.state_name == "test-state"
        assert context.user_id == "user-456"
        assert context.session_id == "session-789"

    def test_child_context(self):
        """Test creating child context"""
        parent = SpanContext(
            trace_id="parent-trace-id",
            span_id="parent-span-id",
            workflow_id="workflow-123",
            agent_name="test-agent",
            state_name="test-state",
            user_id="user-456",
            session_id="session-789",
        )

        child = parent.child_context()

        # Child should inherit trace_id and other context
        assert child.trace_id == "parent-trace-id"
        assert child.parent_span_id == "parent-span-id"
        assert child.workflow_id == "workflow-123"
        assert child.agent_name == "test-agent"
        assert child.state_name == "test-state"
        assert child.user_id == "user-456"
        assert child.session_id == "session-789"

        # Child should have new span_id
        assert child.span_id != "parent-span-id"
        assert len(child.span_id) > 0

    def test_multiple_child_contexts_unique(self):
        """Test that multiple child contexts have unique span IDs"""
        parent = SpanContext()

        child1 = parent.child_context()
        child2 = parent.child_context()

        assert child1.span_id != child2.span_id
        assert child1.trace_id == child2.trace_id == parent.trace_id
        assert child1.parent_span_id == child2.parent_span_id == parent.span_id


class TestObservabilityEvent:
    """Test ObservabilityEvent class"""

    def test_initialization(self):
        """Test event initialization"""
        timestamp = datetime.now()
        attributes = {"key1": "value1", "key2": 42}
        span_context = SpanContext()

        event = ObservabilityEvent(
            timestamp=timestamp,
            event_type="test_event",
            source="test_source",
            level="INFO",
            message="Test message",
            attributes=attributes,
            span_context=span_context,
        )

        assert event.timestamp == timestamp
        assert event.event_type == "test_event"
        assert event.source == "test_source"
        assert event.level == "INFO"
        assert event.message == "Test message"
        assert event.attributes == attributes
        assert event.span_context == span_context

    def test_default_attributes(self):
        """Test event with default attributes"""
        timestamp = datetime.now()

        event = ObservabilityEvent(
            timestamp=timestamp,
            event_type="test_event",
            source="test_source",
            level="INFO",
            message="Test message",
        )

        assert event.attributes == {}
        assert event.span_context is None


class TestAbstractInterfaces:
    """Test abstract interface classes"""

    def test_span_interface(self):
        """Test Span abstract interface"""
        # Verify it's abstract
        with pytest.raises(TypeError):
            Span()

        # Test concrete implementation
        class ConcreteSpan(Span):
            def __init__(self):
                self._context = SpanContext()

            def set_attribute(self, key: str, value):
                pass

            def set_status(self, status: str, description: Optional[str] = None):
                pass

            def add_event(self, name: str, attributes=None):
                pass

            def record_exception(self, exception: Exception):
                pass

            def end(self):
                pass

            @property
            def context(self):
                return self._context

        span = ConcreteSpan()
        assert isinstance(span, Span)
        assert isinstance(span.context, SpanContext)

    def test_tracing_provider_interface(self):
        """Test TracingProvider abstract interface"""
        # Verify it's abstract
        with pytest.raises(TypeError):
            TracingProvider()

        # Test concrete implementation
        class ConcreteTracingProvider(TracingProvider):
            def start_span(
                self, name: str, span_type=SpanType.SYSTEM, parent=None, **attributes
            ):
                return Mock(spec=Span)

            def get_current_span(self):
                return Mock(spec=Span)

        provider = ConcreteTracingProvider()
        assert isinstance(provider, TracingProvider)

        span = provider.start_span("test")
        assert span is not None

        current = provider.get_current_span()
        assert current is not None

    def test_metric_interface(self):
        """Test Metric abstract interface"""
        # Verify it's abstract
        with pytest.raises(TypeError):
            Metric()

        # Test concrete implementation
        class ConcreteMetric(Metric):
            def record(self, value: float, **labels):
                pass

        metric = ConcreteMetric()
        assert isinstance(metric, Metric)

        # Should not raise
        metric.record(1.0, label1="value1")

    def test_metrics_provider_interface(self):
        """Test MetricsProvider abstract interface"""
        # Verify it's abstract
        with pytest.raises(TypeError):
            MetricsProvider()

        # Test concrete implementation
        class ConcreteMetricsProvider(MetricsProvider):
            def counter(self, name: str, description: str = "", labels=None):
                return Mock(spec=Metric)

            def gauge(self, name: str, description: str = "", labels=None):
                return Mock(spec=Metric)

            def histogram(self, name: str, description: str = "", labels=None):
                return Mock(spec=Metric)

            def export_metrics(self):
                return "# Metrics data"

        provider = ConcreteMetricsProvider()
        assert isinstance(provider, MetricsProvider)

        counter = provider.counter("test_counter")
        assert counter is not None

        gauge = provider.gauge("test_gauge")
        assert gauge is not None

        histogram = provider.histogram("test_histogram")
        assert histogram is not None

        metrics_data = provider.export_metrics()
        assert metrics_data == "# Metrics data"

    def test_alerting_provider_interface(self):
        """Test AlertingProvider abstract interface"""
        # Verify it's abstract
        with pytest.raises(TypeError):
            AlertingProvider()

        # Test concrete implementation
        class ConcreteAlertingProvider(AlertingProvider):
            async def send_alert(self, message: str, severity, attributes=None):
                pass

        provider = ConcreteAlertingProvider()
        assert isinstance(provider, AlertingProvider)

    def test_event_processor_interface(self):
        """Test EventProcessor abstract interface"""
        # Verify it's abstract
        with pytest.raises(TypeError):
            EventProcessor()

        # Test concrete implementation
        class ConcreteEventProcessor(EventProcessor):
            async def process_event(self, event):
                pass

        processor = ConcreteEventProcessor()
        assert isinstance(processor, EventProcessor)


class TestInterfaceUsage:
    """Test interface usage patterns"""

    def test_span_context_in_event(self):
        """Test using SpanContext in ObservabilityEvent"""
        span_context = SpanContext(workflow_id="test-workflow", agent_name="test-agent")

        event = ObservabilityEvent(
            timestamp=datetime.now(),
            event_type="test",
            source="test",
            level="INFO",
            message="Test with context",
            span_context=span_context,
        )

        assert event.span_context.workflow_id == "test-workflow"
        assert event.span_context.agent_name == "test-agent"

    def test_span_type_usage(self):
        """Test SpanType enum usage"""
        # Test that enum values can be used in comparisons
        span_type = SpanType.WORKFLOW
        assert span_type == SpanType.WORKFLOW
        assert span_type != SpanType.STATE
        assert span_type.value == "workflow"

    def test_metric_type_usage(self):
        """Test MetricType enum usage"""
        metric_type = MetricType.COUNTER
        assert metric_type == MetricType.COUNTER
        assert metric_type != MetricType.GAUGE
        assert metric_type.value == "counter"

    def test_alert_severity_usage(self):
        """Test AlertSeverity enum usage"""
        severity = AlertSeverity.ERROR
        assert severity == AlertSeverity.ERROR
        assert severity != AlertSeverity.WARNING
        assert severity.value == "error"
