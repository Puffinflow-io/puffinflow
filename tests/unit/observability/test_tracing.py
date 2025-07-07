"""Tests for tracing functionality"""

from unittest.mock import Mock, call, patch

import pytest

from puffinflow.core.observability.config import TracingConfig
from puffinflow.core.observability.interfaces import SpanContext, SpanType
from puffinflow.core.observability.tracing import (
    OpenTelemetrySpan,
    OpenTelemetryTracingProvider,
)


class TestOpenTelemetrySpan:
    """Test OpenTelemetrySpan class"""

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    def test_span_creation(self):
        """Test OpenTelemetrySpan creation"""
        mock_otel_span = Mock()
        span_context = SpanContext(
            workflow_id="workflow-123",
            agent_name="test-agent",
            state_name="test-state",
            user_id="user-456",
        )

        span = OpenTelemetrySpan(mock_otel_span, span_context)

        assert span._span == mock_otel_span
        assert span._context == span_context

        # Check that workflow context attributes were set
        expected_calls = [
            call("workflow.id", "workflow-123"),
            call("agent.name", "test-agent"),
            call("state.name", "test-state"),
            call("user.id", "user-456"),
        ]
        mock_otel_span.set_attribute.assert_has_calls(expected_calls, any_order=True)

    def test_span_creation_minimal_context(self):
        """Test OpenTelemetrySpan creation with minimal context"""
        mock_otel_span = Mock()
        span_context = SpanContext()

        span = OpenTelemetrySpan(mock_otel_span, span_context)
        assert span._span == mock_otel_span
        assert span._context == span_context

        # Should not set any workflow attributes
        mock_otel_span.set_attribute.assert_not_called()

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    def test_set_attribute(self):
        """Test set_attribute method"""
        mock_otel_span = Mock()
        span_context = SpanContext()
        span = OpenTelemetrySpan(mock_otel_span, span_context)

        span.set_attribute("test.key", "test.value")
        mock_otel_span.set_attribute.assert_called_with("test.key", "test.value")

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    def test_set_attribute_with_dict_value(self):
        """Test set_attribute with dict value (should convert to string)"""
        mock_otel_span = Mock()
        span_context = SpanContext()
        span = OpenTelemetrySpan(mock_otel_span, span_context)

        test_dict = {"key": "value"}
        span.set_attribute("test.dict", test_dict)
        mock_otel_span.set_attribute.assert_called_with("test.dict", str(test_dict))

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    def test_set_attribute_with_list_value(self):
        """Test set_attribute with list value (should convert to string)"""
        mock_otel_span = Mock()
        span_context = SpanContext()
        span = OpenTelemetrySpan(mock_otel_span, span_context)

        test_list = ["item1", "item2"]
        span.set_attribute("test.list", test_list)
        mock_otel_span.set_attribute.assert_called_with("test.list", str(test_list))

    def test_set_attribute_none_key_or_value(self):
        """Test set_attribute with None key or value"""
        mock_otel_span = Mock()
        span_context = SpanContext()
        span = OpenTelemetrySpan(mock_otel_span, span_context)

        span.set_attribute(None, "value")
        span.set_attribute("key", None)
        span.set_attribute("", "value")

        # Should not call otel span set_attribute for None/empty keys or None values
        mock_otel_span.set_attribute.assert_not_called()

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    def test_set_status_ok(self):
        """Test set_status with OK status"""
        mock_otel_span = Mock()
        span_context = SpanContext()
        span = OpenTelemetrySpan(mock_otel_span, span_context)

        with patch("puffinflow.core.observability.tracing.Status") as mock_status:
            with patch(
                "puffinflow.core.observability.tracing.StatusCode"
            ) as mock_status_code:
                mock_status_code.OK = "OK"
                span.set_status("ok", "Success")
                mock_status.assert_called_once_with(mock_status_code.OK, "Success")

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    def test_set_status_error(self):
        """Test set_status with error status"""
        mock_otel_span = Mock()
        span_context = SpanContext()
        span = OpenTelemetrySpan(mock_otel_span, span_context)

        with patch("puffinflow.core.observability.tracing.Status") as mock_status:
            with patch(
                "puffinflow.core.observability.tracing.StatusCode"
            ) as mock_status_code:
                mock_status_code.ERROR = "ERROR"
                span.set_status("error", "Failed")
                mock_status.assert_called_once_with(mock_status_code.ERROR, "Failed")

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    def test_add_event(self):
        """Test add_event method"""
        mock_otel_span = Mock()
        span_context = SpanContext()
        span = OpenTelemetrySpan(mock_otel_span, span_context)

        attributes = {"key1": "value1", "key2": None, "key3": "value3"}
        span.add_event("test.event", attributes)

        # Should filter out None values
        expected_attrs = {"key1": "value1", "key3": "value3"}
        mock_otel_span.add_event.assert_called_once_with("test.event", expected_attrs)

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    def test_add_event_no_attributes(self):
        """Test add_event with no attributes"""
        mock_otel_span = Mock()
        span_context = SpanContext()
        span = OpenTelemetrySpan(mock_otel_span, span_context)

        span.add_event("test.event")
        mock_otel_span.add_event.assert_called_once_with("test.event", {})

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    def test_record_exception(self):
        """Test record_exception method"""
        mock_otel_span = Mock()
        span_context = SpanContext()
        span = OpenTelemetrySpan(mock_otel_span, span_context)

        exception = Exception("Test error")

        with patch.object(span, "set_status") as mock_set_status:
            span.record_exception(exception)
            mock_otel_span.record_exception.assert_called_once_with(exception)
            mock_set_status.assert_called_once_with("error", "Test error")

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    def test_end(self):
        """Test end method"""
        mock_otel_span = Mock()
        span_context = SpanContext()
        span = OpenTelemetrySpan(mock_otel_span, span_context)

        with patch.object(span, "set_attribute") as mock_set_attribute:
            span.end()
            mock_otel_span.end.assert_called_once()
            # Should set duration attribute
            mock_set_attribute.assert_called_once()
            call_args = mock_set_attribute.call_args[0]
            assert call_args[0] == "span.duration_ms"
            assert isinstance(call_args[1], float)

    def test_context_property(self):
        """Test context property"""
        mock_otel_span = Mock()
        span_context = SpanContext()
        span = OpenTelemetrySpan(mock_otel_span, span_context)

        assert span.context == span_context


class TestOpenTelemetryTracingProvider:
    """Test OpenTelemetryTracingProvider class"""

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    @patch("puffinflow.core.observability.tracing.trace")
    @patch("puffinflow.core.observability.tracing.TracerProvider")
    @patch("puffinflow.core.observability.tracing.Resource")
    def test_tracing_provider_creation(
        self, mock_resource, mock_tracer_provider, mock_trace
    ):
        """Test OpenTelemetryTracingProvider creation"""
        config = TracingConfig()

        mock_resource_instance = Mock()
        mock_resource.create.return_value = mock_resource_instance

        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance

        mock_tracer = Mock()
        mock_trace.get_tracer.return_value = mock_tracer

        provider = OpenTelemetryTracingProvider(config)

        # Check resource creation
        mock_resource.create.assert_called_once_with(
            {
                "service.name": config.service_name,
                "service.version": config.service_version,
            }
        )

        # Check tracer provider setup
        mock_tracer_provider.assert_called_once_with(resource=mock_resource_instance)
        mock_trace.set_tracer_provider.assert_called_once_with(mock_provider_instance)

        assert provider._tracer == mock_tracer

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    @patch("puffinflow.core.observability.tracing.trace")
    @patch("puffinflow.core.observability.tracing.TracerProvider")
    @patch("puffinflow.core.observability.tracing.Resource")
    @patch("puffinflow.core.observability.tracing.BatchSpanProcessor")
    @patch("puffinflow.core.observability.tracing.ConsoleSpanExporter")
    def test_console_exporter_setup(
        self,
        mock_console_exporter,
        mock_batch_processor,
        mock_resource,
        mock_tracer_provider,
        mock_trace,
    ):
        """Test console exporter setup"""
        config = TracingConfig(console_enabled=True)

        mock_resource_instance = Mock()
        mock_resource.create.return_value = mock_resource_instance

        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance

        mock_console_exporter_instance = Mock()
        mock_console_exporter.return_value = mock_console_exporter_instance

        mock_processor_instance = Mock()
        mock_batch_processor.return_value = mock_processor_instance

        OpenTelemetryTracingProvider(config)

        mock_console_exporter.assert_called_once()
        mock_batch_processor.assert_called_with(mock_console_exporter_instance)
        mock_provider_instance.add_span_processor.assert_called_with(
            mock_processor_instance
        )

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    @patch("puffinflow.core.observability.tracing.trace")
    @patch("puffinflow.core.observability.tracing.TracerProvider")
    @patch("puffinflow.core.observability.tracing.Resource")
    def test_start_span(self, mock_resource, mock_tracer_provider, mock_trace):
        """Test start_span method"""
        config = TracingConfig()

        mock_resource_instance = Mock()
        mock_resource.create.return_value = mock_resource_instance

        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance

        mock_tracer = Mock()
        mock_otel_span = Mock()
        mock_tracer.start_span.return_value = mock_otel_span
        mock_trace.get_tracer.return_value = mock_tracer

        provider = OpenTelemetryTracingProvider(config)

        span = provider.start_span("test.span", SpanType.BUSINESS, test_attr="value")

        assert isinstance(span, OpenTelemetrySpan)
        mock_tracer.start_span.assert_called_once_with("test.span")

        # Check that attributes were set
        span.set_attribute("span.type", SpanType.BUSINESS.value)
        span.set_attribute("test_attr", "value")

    @patch("puffinflow.core.observability.tracing.trace")
    @patch("puffinflow.core.observability.tracing.TracerProvider")
    @patch("puffinflow.core.observability.tracing.Resource")
    def test_get_current_span(self, mock_resource, mock_tracer_provider, mock_trace):
        """Test get_current_span method"""
        config = TracingConfig()
        provider = OpenTelemetryTracingProvider(config)

        # Initially should return None
        assert provider.get_current_span() is None

        # Set a span and test retrieval
        mock_span = Mock()
        provider._set_current_span(mock_span)
        assert provider.get_current_span() == mock_span

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    @patch("puffinflow.core.observability.tracing.trace")
    @patch("puffinflow.core.observability.tracing.TracerProvider")
    @patch("puffinflow.core.observability.tracing.Resource")
    def test_span_context_manager(
        self, mock_resource, mock_tracer_provider, mock_trace
    ):
        """Test span context manager"""
        config = TracingConfig()

        mock_resource_instance = Mock()
        mock_resource.create.return_value = mock_resource_instance

        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance

        mock_tracer = Mock()
        mock_otel_span = Mock()
        mock_tracer.start_span.return_value = mock_otel_span
        mock_trace.get_tracer.return_value = mock_tracer

        # Mock StatusCode and Status to have OK attribute
        with patch(
            "puffinflow.core.observability.tracing.StatusCode"
        ) as mock_status_code, patch(
            "puffinflow.core.observability.tracing.Status"
        ) as mock_status:
            mock_status_code.OK = "OK"
            mock_status.return_value = Mock()
            provider = OpenTelemetryTracingProvider(config)

            with provider.span("test.span", SpanType.SYSTEM) as span:
                assert isinstance(span, OpenTelemetrySpan)
                assert span._span == mock_otel_span

            # Span should be ended and status set on the underlying otel span
            mock_otel_span.end.assert_called_once()

    @patch("puffinflow.core.observability.tracing._OPENTELEMETRY_AVAILABLE", True)
    @patch("puffinflow.core.observability.tracing.trace")
    @patch("puffinflow.core.observability.tracing.TracerProvider")
    @patch("puffinflow.core.observability.tracing.Resource")
    def test_span_context_manager_with_exception(
        self, mock_resource, mock_tracer_provider, mock_trace
    ):
        """Test span context manager with exception"""
        config = TracingConfig()

        mock_resource_instance = Mock()
        mock_resource.create.return_value = mock_resource_instance

        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance

        mock_tracer = Mock()
        mock_otel_span = Mock()
        mock_tracer.start_span.return_value = mock_otel_span
        mock_trace.get_tracer.return_value = mock_tracer

        # Mock StatusCode and Status to have ERROR attribute
        with patch(
            "puffinflow.core.observability.tracing.StatusCode"
        ) as mock_status_code, patch(
            "puffinflow.core.observability.tracing.Status"
        ) as mock_status:
            mock_status_code.ERROR = "ERROR"
            mock_status.return_value = Mock()
            provider = OpenTelemetryTracingProvider(config)

            test_exception = Exception("Test error")

            with pytest.raises(Exception, match="Test error"):
                with provider.span("test.span"):
                    raise test_exception

            # Exception should be recorded and span ended on the underlying otel span
            mock_otel_span.record_exception.assert_called_once_with(test_exception)
            mock_otel_span.end.assert_called_once()

    @patch("puffinflow.core.observability.tracing.trace")
    @patch("puffinflow.core.observability.tracing.TracerProvider")
    @patch("puffinflow.core.observability.tracing.Resource")
    def test_start_span_with_parent(
        self, mock_resource, mock_tracer_provider, mock_trace
    ):
        """Test start_span with parent context"""
        config = TracingConfig()

        mock_resource_instance = Mock()
        mock_resource.create.return_value = mock_resource_instance

        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance

        mock_tracer = Mock()
        mock_otel_span = Mock()
        mock_tracer.start_span.return_value = mock_otel_span
        mock_trace.get_tracer.return_value = mock_tracer

        provider = OpenTelemetryTracingProvider(config)

        # Create parent context
        parent_context = SpanContext()

        span = provider.start_span("child.span", parent=parent_context)

        assert isinstance(span, OpenTelemetrySpan)
        # Child span should have parent's trace_id
        assert span.context.trace_id == parent_context.trace_id
        assert span.context.parent_span_id == parent_context.span_id

    def test_span_context_creation(self):
        """Test SpanContext creation and child context"""
        context = SpanContext()
        assert context.trace_id is not None
        assert context.span_id is not None
        assert context.parent_span_id is None

        # Test child context
        child = context.child_context()
        assert child.trace_id == context.trace_id
        assert child.parent_span_id == context.span_id
        assert child.span_id != context.span_id

    def test_span_context_with_attributes(self):
        """Test SpanContext with workflow attributes"""
        context = SpanContext(
            workflow_id="workflow-123",
            agent_name="test-agent",
            state_name="test-state",
            user_id="user-456",
            session_id="session-789",
        )
        assert context.workflow_id == "workflow-123"
        assert context.agent_name == "test-agent"
        assert context.state_name == "test-state"
        assert context.user_id == "user-456"
        assert context.session_id == "session-789"
