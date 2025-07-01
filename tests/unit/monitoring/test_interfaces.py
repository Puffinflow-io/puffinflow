import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional, List

from src.puffinflow.core.observability.interfaces import (
    SpanType,
    MetricType,
    AlertSeverity,
    SpanContext,
    ObservabilityEvent,
    Span,
    TracingProvider,
    Metric,
    MetricsProvider,
    AlertingProvider,
    EventProcessor
)


class TestEnums:
    """Test all enum classes"""

    def test_span_type_values(self):
        """Test SpanType enum values"""
        assert SpanType.WORKFLOW.value == "workflow"
        assert SpanType.STATE.value == "state"
        assert SpanType.RESOURCE.value == "resource"
        assert SpanType.BUSINESS.value == "business"
        assert SpanType.SYSTEM.value == "system"

        # Test all enum members are present
        expected_values = {"workflow", "state", "resource", "business", "system"}
        actual_values = {span_type.value for span_type in SpanType}
        assert actual_values == expected_values

    def test_metric_type_values(self):
        """Test MetricType enum values"""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"

        # Test all enum members are present
        expected_values = {"counter", "gauge", "histogram"}
        actual_values = {metric_type.value for metric_type in MetricType}
        assert actual_values == expected_values

    def test_alert_severity_values(self):
        """Test AlertSeverity enum values"""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"

        # Test all enum members are present
        expected_values = {"info", "warning", "error", "critical"}
        actual_values = {severity.value for severity in AlertSeverity}
        assert actual_values == expected_values


class TestSpanContext:
    """Test SpanContext dataclass"""

    def test_span_context_default_initialization(self):
        """Test SpanContext with default values"""
        context = SpanContext()

        # Check that UUIDs are generated
        assert context.trace_id is not None
        assert context.span_id is not None
        assert uuid.UUID(context.trace_id)  # Should not raise exception
        assert uuid.UUID(context.span_id)  # Should not raise exception

        # Check optional fields are None
        assert context.parent_span_id is None
        assert context.workflow_id is None
        assert context.agent_name is None
        assert context.state_name is None
        assert context.user_id is None
        assert context.session_id is None

    def test_span_context_custom_initialization(self):
        """Test SpanContext with custom values"""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        parent_span_id = str(uuid.uuid4())

        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            workflow_id="workflow_123",
            agent_name="test_agent",
            state_name="test_state",
            user_id="user_456",
            session_id="session_789"
        )

        assert context.trace_id == trace_id
        assert context.span_id == span_id
        assert context.parent_span_id == parent_span_id
        assert context.workflow_id == "workflow_123"
        assert context.agent_name == "test_agent"
        assert context.state_name == "test_state"
        assert context.user_id == "user_456"
        assert context.session_id == "session_789"

    def test_child_context_creation(self):
        """Test child_context method"""
        parent_context = SpanContext(
            workflow_id="workflow_123",
            agent_name="test_agent",
            state_name="test_state",
            user_id="user_456",
            session_id="session_789"
        )

        child_context = parent_context.child_context()

        # Child should inherit parent's trace_id and contextual info
        assert child_context.trace_id == parent_context.trace_id
        assert child_context.workflow_id == parent_context.workflow_id
        assert child_context.agent_name == parent_context.agent_name
        assert child_context.state_name == parent_context.state_name
        assert child_context.user_id == parent_context.user_id
        assert child_context.session_id == parent_context.session_id

        # Child should have different span_id and parent_span_id set to parent's span_id
        assert child_context.span_id != parent_context.span_id
        assert child_context.parent_span_id == parent_context.span_id

        # Verify span_id is valid UUID
        assert uuid.UUID(child_context.span_id)

    def test_multiple_child_contexts_are_unique(self):
        """Test that multiple child contexts have unique span IDs"""
        parent_context = SpanContext()

        child1 = parent_context.child_context()
        child2 = parent_context.child_context()

        assert child1.span_id != child2.span_id
        assert child1.trace_id == child2.trace_id == parent_context.trace_id
        assert child1.parent_span_id == child2.parent_span_id == parent_context.span_id


class TestObservabilityEvent:
    """Test ObservabilityEvent dataclass"""

    def test_observability_event_initialization(self):
        """Test ObservabilityEvent initialization"""
        timestamp = datetime.now()
        event = ObservabilityEvent(
            timestamp=timestamp,
            event_type="test_event",
            source="test_source",
            level="INFO",
            message="Test message"
        )

        assert event.timestamp == timestamp
        assert event.event_type == "test_event"
        assert event.source == "test_source"
        assert event.level == "INFO"
        assert event.message == "Test message"
        assert event.attributes == {}
        assert event.span_context is None

    def test_observability_event_with_attributes_and_context(self):
        """Test ObservabilityEvent with attributes and span context"""
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
            span_context=span_context
        )

        assert event.attributes == attributes
        assert event.span_context == span_context

    def test_observability_event_default_attributes(self):
        """Test that attributes defaults to empty dict"""
        timestamp = datetime.now()
        event = ObservabilityEvent(
            timestamp=timestamp,
            event_type="test_event",
            source="test_source",
            level="INFO",
            message="Test message"
        )

        assert event.attributes == {}
        assert isinstance(event.attributes, dict)


# Mock implementations for testing abstract classes
class MockSpan(Span):
    """Mock implementation of Span for testing"""

    def __init__(self, context: SpanContext):
        self._context = context
        self._attributes = {}
        self._status = None
        self._events = []
        self._exceptions = []
        self._ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        self._attributes[key] = value

    def set_status(self, status: str, description: str = None) -> None:
        self._status = (status, description)

    def add_event(self, name: str, attributes: Dict[str, Any] = None) -> None:
        self._events.append((name, attributes or {}))

    def record_exception(self, exception: Exception) -> None:
        self._exceptions.append(exception)

    def end(self) -> None:
        self._ended = True

    @property
    def context(self) -> SpanContext:
        return self._context


class MockTracingProvider(TracingProvider):
    """Mock implementation of TracingProvider for testing"""

    def __init__(self):
        self._current_span = None
        self._spans = []

    def start_span(self, name: str, span_type: SpanType = SpanType.SYSTEM,
                   parent: Optional[SpanContext] = None, **attributes) -> Span:
        if parent:
            context = parent.child_context()
        else:
            context = SpanContext()

        span = MockSpan(context)
        for key, value in attributes.items():
            span.set_attribute(key, value)

        self._current_span = span
        self._spans.append(span)
        return span

    def get_current_span(self) -> Optional[Span]:
        return self._current_span


class MockMetric(Metric):
    """Mock implementation of Metric for testing"""

    def __init__(self, name: str):
        self.name = name
        self.recordings = []

    def record(self, value: float, **labels) -> None:
        self.recordings.append((value, labels))


class MockMetricsProvider(MetricsProvider):
    """Mock implementation of MetricsProvider for testing"""

    def __init__(self):
        self.metrics = {}

    def counter(self, name: str, description: str = "", labels: List[str] = None) -> Metric:
        metric = MockMetric(name)
        self.metrics[name] = metric
        return metric

    def gauge(self, name: str, description: str = "", labels: List[str] = None) -> Metric:
        metric = MockMetric(name)
        self.metrics[name] = metric
        return metric

    def histogram(self, name: str, description: str = "", labels: List[str] = None) -> Metric:
        metric = MockMetric(name)
        self.metrics[name] = metric
        return metric

    def export_metrics(self) -> str:
        return "# Mock Prometheus metrics\n"


class MockAlertingProvider(AlertingProvider):
    """Mock implementation of AlertingProvider for testing"""

    def __init__(self):
        self.alerts = []

    async def send_alert(self, message: str, severity: AlertSeverity,
                         attributes: Dict[str, Any] = None) -> None:
        self.alerts.append((message, severity, attributes or {}))


class MockEventProcessor(EventProcessor):
    """Mock implementation of EventProcessor for testing"""

    def __init__(self):
        self.processed_events = []

    async def process_event(self, event: ObservabilityEvent) -> None:
        self.processed_events.append(event)


class TestAbstractInterfaces:
    """Test abstract interfaces using mock implementations"""

    def test_span_interface(self):
        """Test Span interface through mock implementation"""
        context = SpanContext()
        span = MockSpan(context)

        # Test set_attribute
        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 42)
        assert span._attributes == {"key1": "value1", "key2": 42}

        # Test set_status
        span.set_status("OK", "Success")
        assert span._status == ("OK", "Success")

        # Test add_event
        span.add_event("event1", {"attr": "value"})
        span.add_event("event2")
        assert len(span._events) == 2
        assert span._events[0] == ("event1", {"attr": "value"})
        assert span._events[1] == ("event2", {})

        # Test record_exception
        exception = ValueError("Test error")
        span.record_exception(exception)
        assert span._exceptions == [exception]

        # Test context property
        assert span.context == context

        # Test end
        assert not span._ended
        span.end()
        assert span._ended

    def test_tracing_provider_interface(self):
        """Test TracingProvider interface through mock implementation"""
        provider = MockTracingProvider()

        # Test start_span without parent
        span1 = provider.start_span("test_span", SpanType.WORKFLOW, attr1="value1")
        assert isinstance(span1, MockSpan)
        assert span1._attributes["attr1"] == "value1"

        # Test get_current_span
        current = provider.get_current_span()
        assert current == span1

        # Test start_span with parent
        span2 = provider.start_span("child_span", parent=span1.context)
        assert span2.context.parent_span_id == span1.context.span_id
        assert span2.context.trace_id == span1.context.trace_id

    def test_metric_interface(self):
        """Test Metric interface through mock implementation"""
        metric = MockMetric("test_metric")

        # Test record
        metric.record(1.5, label1="value1", label2="value2")
        metric.record(2.0)

        assert len(metric.recordings) == 2
        assert metric.recordings[0] == (1.5, {"label1": "value1", "label2": "value2"})
        assert metric.recordings[1] == (2.0, {})

    def test_metrics_provider_interface(self):
        """Test MetricsProvider interface through mock implementation"""
        provider = MockMetricsProvider()

        # Test counter creation
        counter = provider.counter("test_counter", "A test counter", ["label1"])
        assert counter.name == "test_counter"
        assert "test_counter" in provider.metrics

        # Test gauge creation
        gauge = provider.gauge("test_gauge", "A test gauge")
        assert gauge.name == "test_gauge"
        assert "test_gauge" in provider.metrics

        # Test histogram creation
        histogram = provider.histogram("test_histogram")
        assert histogram.name == "test_histogram"
        assert "test_histogram" in provider.metrics

        # Test export_metrics
        metrics_output = provider.export_metrics()
        assert isinstance(metrics_output, str)
        assert "Mock Prometheus metrics" in metrics_output

    @pytest.mark.asyncio
    async def test_alerting_provider_interface(self):
        """Test AlertingProvider interface through mock implementation"""
        provider = MockAlertingProvider()

        # Test send_alert without attributes
        await provider.send_alert("Test alert", AlertSeverity.WARNING)

        # Test send_alert with attributes
        await provider.send_alert(
            "Critical alert",
            AlertSeverity.CRITICAL,
            {"component": "database", "error_code": 500}
        )

        assert len(provider.alerts) == 2
        assert provider.alerts[0] == ("Test alert", AlertSeverity.WARNING, {})
        assert provider.alerts[1] == (
            "Critical alert",
            AlertSeverity.CRITICAL,
            {"component": "database", "error_code": 500}
        )

    @pytest.mark.asyncio
    async def test_event_processor_interface(self):
        """Test EventProcessor interface through mock implementation"""
        processor = MockEventProcessor()

        # Create test events
        event1 = ObservabilityEvent(
            timestamp=datetime.now(),
            event_type="test_event",
            source="test_source",
            level="INFO",
            message="Test message 1"
        )

        event2 = ObservabilityEvent(
            timestamp=datetime.now(),
            event_type="error_event",
            source="error_source",
            level="ERROR",
            message="Test message 2",
            attributes={"error_code": 404}
        )

        # Process events
        await processor.process_event(event1)
        await processor.process_event(event2)

        assert len(processor.processed_events) == 2
        assert processor.processed_events[0] == event1
        assert processor.processed_events[1] == event2


class TestAbstractClassInstantiation:
    """Test that abstract classes cannot be instantiated directly"""

    def test_abstract_classes_cannot_be_instantiated(self):
        """Test that abstract classes raise TypeError when instantiated"""
        with pytest.raises(TypeError):
            Span()

        with pytest.raises(TypeError):
            TracingProvider()

        with pytest.raises(TypeError):
            Metric()

        with pytest.raises(TypeError):
            MetricsProvider()

        with pytest.raises(TypeError):
            AlertingProvider()

        with pytest.raises(TypeError):
            EventProcessor()


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_span_context_with_invalid_uuids(self):
        """Test SpanContext behavior with invalid UUID strings"""
        # This should not raise an error during initialization
        # but would raise when trying to parse as UUID
        context = SpanContext(trace_id="invalid-uuid", span_id="also-invalid")
        assert context.trace_id == "invalid-uuid"
        assert context.span_id == "also-invalid"

    def test_observability_event_accepts_none_timestamp(self):
        """Test ObservabilityEvent accepts None timestamp (dataclass behavior)"""
        # Note: Python dataclasses don't enforce type checking at runtime
        # This test documents the actual behavior - None is accepted
        event = ObservabilityEvent(
            timestamp=None,
            event_type="test",
            source="test",
            level="INFO",
            message="test"
        )
        assert event.timestamp is None
        assert event.event_type == "test"

    def test_empty_attributes_in_observability_event(self):
        """Test ObservabilityEvent with explicitly empty attributes"""
        timestamp = datetime.now()
        event = ObservabilityEvent(
            timestamp=timestamp,
            event_type="test",
            source="test",
            level="INFO",
            message="test",
            attributes={}
        )
        assert event.attributes == {}

    @patch('uuid.uuid4')
    def test_span_context_uuid_generation(self, mock_uuid):
        """Test that UUIDs are properly generated using uuid.uuid4()"""
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="mocked-uuid")

        context = SpanContext()

        # Should be called twice (once for trace_id, once for span_id)
        assert mock_uuid.call_count == 2
        assert context.trace_id == "mocked-uuid"
        assert context.span_id == "mocked-uuid"

    def test_span_context_child_preserves_none_values(self):
        """Test that child context preserves None values from parent"""
        parent = SpanContext()
        # All optional fields should be None by default
        child = parent.child_context()

        assert child.workflow_id is None
        assert child.agent_name is None
        assert child.state_name is None
        assert child.user_id is None
        assert child.session_id is None

    def test_observability_event_with_complex_attributes(self):
        """Test ObservabilityEvent with complex nested attributes"""
        timestamp = datetime.now()
        complex_attributes = {
            "nested": {"key": "value", "number": 42},
            "list": [1, 2, 3],
            "boolean": True,
            "none_value": None
        }

        event = ObservabilityEvent(
            timestamp=timestamp,
            event_type="complex_test",
            source="test_source",
            level="DEBUG",
            message="Complex attributes test",
            attributes=complex_attributes
        )

        assert event.attributes == complex_attributes
        assert event.attributes["nested"]["key"] == "value"
        assert event.attributes["list"] == [1, 2, 3]
        assert event.attributes["boolean"] is True
        assert event.attributes["none_value"] is None