import unittest
from unittest.mock import Mock, MagicMock, patch, call
import time
import threading
from contextlib import contextmanager
from datetime import datetime
import json

import pytest


# Test the tracing module
class TestOpenTelemetrySpan(unittest.TestCase):
    """Test cases for OpenTelemetrySpan class"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_otel_span = Mock()
        self.mock_span_context = Mock()
        self.mock_span_context.workflow_id = "test-workflow-123"
        self.mock_span_context.agent_name = "test-agent"
        self.mock_span_context.state_name = "test-state"
        self.mock_span_context.user_id = "user-456"

        # Create the span instance
        with patch('time.time', return_value=1000.0):
            from src.puffinflow.core.observability.tracing import OpenTelemetrySpan
            self.span = OpenTelemetrySpan(self.mock_otel_span, self.mock_span_context)

    def test_init_sets_context_attributes(self):
        """Test that initialization sets workflow context attributes"""
        expected_calls = [
            call("workflow.id", "test-workflow-123"),
            call("agent.name", "test-agent"),
            call("state.name", "test-state"),
            call("user.id", "user-456")
        ]
        self.mock_otel_span.set_attribute.assert_has_calls(expected_calls, any_order=True)

    def test_init_with_none_attributes(self):
        """Test initialization with None context attributes"""
        mock_context = Mock()
        mock_context.workflow_id = None
        mock_context.agent_name = None
        mock_context.state_name = None
        mock_context.user_id = None

        mock_otel_span = Mock()
        with patch('time.time', return_value=1000.0):
            from src.puffinflow.core.observability.tracing import OpenTelemetrySpan
            span = OpenTelemetrySpan(mock_otel_span, mock_context)

        # Should not call set_attribute for None values
        mock_otel_span.set_attribute.assert_not_called()

    def test_set_attribute_with_valid_values(self):
        """Test setting attributes with various valid data types"""
        test_cases = [
            ("string_attr", "test_value"),
            ("int_attr", 42),
            ("float_attr", 3.14),
            ("bool_attr", True),
        ]

        for key, value in test_cases:
            with self.subTest(key=key, value=value):
                self.span.set_attribute(key, value)
                self.mock_otel_span.set_attribute.assert_called_with(key, value)

    def test_set_attribute_with_complex_types(self):
        """Test setting attributes with dict and list values"""
        test_cases = [
            ("dict_attr", {"key": "value"}, "{'key': 'value'}"),  # Fixed expected format
            ("list_attr", [1, 2, 3], '[1, 2, 3]'),
        ]

        for key, value, expected_str in test_cases:
            with self.subTest(key=key, value=value):
                self.span.set_attribute(key, value)
                self.mock_otel_span.set_attribute.assert_called_with(key, expected_str)

    def test_set_attribute_ignores_none_values(self):
        """Test that None values are ignored"""
        initial_call_count = self.mock_otel_span.set_attribute.call_count
        self.span.set_attribute("test_key", None)
        # Should not have made additional calls
        self.assertEqual(self.mock_otel_span.set_attribute.call_count, initial_call_count)

    def test_set_attribute_ignores_empty_key(self):
        """Test that empty keys are ignored"""
        initial_call_count = self.mock_otel_span.set_attribute.call_count
        self.span.set_attribute("", "test_value")
        self.span.set_attribute(None, "test_value")
        # Should not have made additional calls
        self.assertEqual(self.mock_otel_span.set_attribute.call_count, initial_call_count)

    def test_set_status_success(self):
        """Test setting success status"""
        with patch('src.puffinflow.core.observability.tracing.Status') as mock_status, \
                patch('src.puffinflow.core.observability.tracing.StatusCode') as mock_status_code:
            test_cases = ["ok", "success", "OK", "SUCCESS"]
            for status in test_cases:
                with self.subTest(status=status):
                    self.span.set_status(status, "Test description")
                    mock_status.assert_called_with(mock_status_code.OK, "Test description")
                    self.mock_otel_span.set_status.assert_called_with(mock_status.return_value)

    def test_set_status_error(self):
        """Test setting error status"""
        with patch('src.puffinflow.core.observability.tracing.Status') as mock_status, \
                patch('src.puffinflow.core.observability.tracing.StatusCode') as mock_status_code:
            test_cases = ["error", "failed", "ERROR", "FAILED"]
            for status in test_cases:
                with self.subTest(status=status):
                    self.span.set_status(status, "Error description")
                    mock_status.assert_called_with(mock_status_code.ERROR, "Error description")
                    self.mock_otel_span.set_status.assert_called_with(mock_status.return_value)

    def test_set_status_without_description(self):
        """Test setting status without description"""
        with patch('src.puffinflow.core.observability.tracing.Status') as mock_status, \
                patch('src.puffinflow.core.observability.tracing.StatusCode') as mock_status_code:
            self.span.set_status("ok")
            mock_status.assert_called_with(mock_status_code.OK, None)

    def test_add_event_with_attributes(self):
        """Test adding event with attributes"""
        attributes = {"key1": "value1", "key2": 42, "key3": None}
        expected_attrs = {"key1": "value1", "key2": 42}  # None values filtered out

        self.span.add_event("test_event", attributes)
        self.mock_otel_span.add_event.assert_called_with("test_event", expected_attrs)

    def test_add_event_without_attributes(self):
        """Test adding event without attributes"""
        self.span.add_event("test_event")
        self.mock_otel_span.add_event.assert_called_with("test_event", {})

    def test_record_exception(self):
        """Test recording exception"""
        test_exception = ValueError("Test error")

        with patch.object(self.span, 'set_status') as mock_set_status:
            self.span.record_exception(test_exception)

            self.mock_otel_span.record_exception.assert_called_with(test_exception)
            mock_set_status.assert_called_with("error", "Test error")

    def test_end_calculates_duration(self):
        """Test that end method calculates and sets duration"""
        # Start time was mocked as 1000.0 in setUp
        with patch('time.time', return_value=1002.5):  # 2.5 seconds later
            with patch.object(self.span, 'set_attribute') as mock_set_attr:
                self.span.end()

                mock_set_attr.assert_called_with("span.duration_ms", 2500.0)
                self.mock_otel_span.end.assert_called_once()

    def test_context_property(self):
        """Test context property returns span context"""
        self.assertEqual(self.span.context, self.mock_span_context)


class TestOpenTelemetryTracingProvider(unittest.TestCase):
    """Test cases for OpenTelemetryTracingProvider class"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock the config with proper string values
        self.mock_config = Mock()
        self.mock_config.service_name = "test-service"
        self.mock_config.service_version = "1.0.0"
        self.mock_config.otlp_endpoint = "http://localhost:4317"
        self.mock_config.jaeger_endpoint = "localhost:6831"
        self.mock_config.console_enabled = True

        # Mock SpanContext and SpanType
        self.mock_span_context = Mock()
        self.mock_span_context.child_context.return_value = Mock()

    @patch('src.puffinflow.core.observability.tracing.trace')
    @patch('src.puffinflow.core.observability.tracing.TracerProvider')
    @patch('src.puffinflow.core.observability.tracing.Resource')
    @patch('src.puffinflow.core.observability.tracing.BatchSpanProcessor')
    @patch('src.puffinflow.core.observability.tracing.OTLPSpanExporter')
    @patch('src.puffinflow.core.observability.tracing.JaegerExporter')
    @patch('src.puffinflow.core.observability.tracing.ConsoleSpanExporter')
    def test_setup_tracing(self, mock_console_exporter, mock_jaeger_exporter,
                           mock_otlp_exporter, mock_batch_processor, mock_resource,
                           mock_tracer_provider, mock_trace):
        """Test tracing setup with all exporters"""
        from src.puffinflow.core.observability.tracing import OpenTelemetryTracingProvider

        provider = OpenTelemetryTracingProvider(self.mock_config)

        # Verify resource creation
        mock_resource.create.assert_called_with({
            "service.name": "test-service",
            "service.version": "1.0.0"
        })

        # Verify tracer provider setup
        mock_tracer_provider.assert_called_with(resource=mock_resource.create.return_value)
        mock_trace.set_tracer_provider.assert_called_with(mock_tracer_provider.return_value)

        # Verify exporters are created
        mock_otlp_exporter.assert_called_with(endpoint="http://localhost:4317")
        mock_jaeger_exporter.assert_called_with(
            agent_host_name="localhost",
            agent_port=6831
        )
        mock_console_exporter.assert_called_once()

        # Verify processors are added
        self.assertEqual(mock_batch_processor.call_count, 3)  # One for each exporter
        self.assertEqual(mock_tracer_provider.return_value.add_span_processor.call_count, 3)

    @patch('src.puffinflow.core.observability.tracing.trace')
    @patch('src.puffinflow.core.observability.tracing.TracerProvider')
    @patch('src.puffinflow.core.observability.tracing.Resource')
    def test_setup_tracing_minimal_config(self, mock_resource, mock_tracer_provider, mock_trace):
        """Test tracing setup with minimal configuration"""
        minimal_config = Mock()
        minimal_config.service_name = "test-service"
        minimal_config.service_version = "1.0.0"
        minimal_config.otlp_endpoint = None
        minimal_config.jaeger_endpoint = None
        minimal_config.console_enabled = False

        from src.puffinflow.core.observability.tracing import OpenTelemetryTracingProvider
        provider = OpenTelemetryTracingProvider(minimal_config)

        # Should still create tracer provider but no processors
        mock_tracer_provider.assert_called_once()
        mock_tracer_provider.return_value.add_span_processor.assert_not_called()

    @patch('src.puffinflow.core.observability.tracing.trace')
    @patch('src.puffinflow.core.observability.tracing.TracerProvider')
    @patch('src.puffinflow.core.observability.tracing.Resource')
    @patch('src.puffinflow.core.observability.tracing.BatchSpanProcessor')
    @patch('src.puffinflow.core.observability.tracing.JaegerExporter')
    def test_jaeger_endpoint_parsing(self, mock_jaeger, mock_batch_processor, mock_resource, mock_tracer_provider,
                                     mock_trace):
        """Test Jaeger endpoint parsing for different formats"""
        test_cases = [
            ("localhost:6831", "localhost", 6831),
            ("jaeger-host", "jaeger-host", 6831),  # Default port
            ("192.168.1.100:14268", "192.168.1.100", 14268),
        ]

        for endpoint, expected_host, expected_port in test_cases:
            with self.subTest(endpoint=endpoint):
                config = Mock()
                config.service_name = "test"
                config.service_version = "1.0"
                config.otlp_endpoint = None
                config.jaeger_endpoint = endpoint
                config.console_enabled = False

                from src.puffinflow.core.observability.tracing import OpenTelemetryTracingProvider
                provider = OpenTelemetryTracingProvider(config)

                mock_jaeger.assert_called_with(
                    agent_host_name=expected_host,
                    agent_port=expected_port
                )

    @patch('src.puffinflow.core.observability.tracing.OpenTelemetrySpan')
    @patch('src.puffinflow.core.observability.tracing.SpanContext')
    @patch('src.puffinflow.core.observability.tracing.trace')
    def test_start_span_without_parent(self, mock_trace, mock_span_context_class, mock_span_class):
        """Test starting span without parent context"""
        from src.puffinflow.core.observability.tracing import OpenTelemetryTracingProvider, SpanType

        provider = OpenTelemetryTracingProvider(self.mock_config)
        provider._tracer = Mock()

        # No current span
        provider._current_context = Mock()
        provider._current_context.current_span = None

        span = provider.start_span("test_span", SpanType.SYSTEM, custom_attr="value")

        # Verify new span context is created
        mock_span_context_class.assert_called_once()

        # Verify tracer.start_span is called
        provider._tracer.start_span.assert_called_with("test_span")

        # Verify OpenTelemetrySpan is created
        mock_span_class.assert_called_with(
            provider._tracer.start_span.return_value,
            mock_span_context_class.return_value
        )

    @patch('src.puffinflow.core.observability.tracing.OpenTelemetrySpan')
    @patch('src.puffinflow.core.observability.tracing.trace')
    def test_start_span_with_parent(self, mock_trace, mock_span_class):
        """Test starting span with parent context"""
        # Create a proper mock enum value
        mock_user_span_type = Mock()
        mock_user_span_type.value = "user"

        # Mock SpanType.USER specifically for this test
        mock_span_type = Mock()
        mock_span_type.USER = mock_user_span_type

        with patch('src.puffinflow.core.observability.tracing.SpanType', mock_span_type):
            from src.puffinflow.core.observability.tracing import OpenTelemetryTracingProvider

            provider = OpenTelemetryTracingProvider(self.mock_config)
            provider._tracer = Mock()

            parent_context = Mock()
            child_context = Mock()
            parent_context.child_context.return_value = child_context

            span = provider.start_span("test_span", mock_span_type.USER, parent=parent_context)

            # Verify child context is used
            parent_context.child_context.assert_called_once()
            mock_span_class.assert_called_with(
                provider._tracer.start_span.return_value,
                child_context
            )

    @patch('src.puffinflow.core.observability.tracing.OpenTelemetrySpan')
    @patch('src.puffinflow.core.observability.tracing.trace')
    def test_start_span_with_current_span(self, mock_trace, mock_span_class):
        """Test starting span when current span exists"""
        from src.puffinflow.core.observability.tracing import OpenTelemetryTracingProvider, SpanType

        provider = OpenTelemetryTracingProvider(self.mock_config)
        provider._tracer = Mock()

        # Mock current span
        current_span = Mock()
        current_context = Mock()
        child_context = Mock()
        current_span.context = current_context
        current_context.child_context.return_value = child_context

        provider._current_context = Mock()
        provider._current_context.current_span = current_span

        span = provider.start_span("test_span")

        # Verify child context from current span is used
        current_context.child_context.assert_called_once()
        mock_span_class.assert_called_with(
            provider._tracer.start_span.return_value,
            child_context
        )

    @patch('src.puffinflow.core.observability.tracing.trace')
    def test_get_current_span(self, mock_trace):
        """Test getting current span"""
        from src.puffinflow.core.observability.tracing import OpenTelemetryTracingProvider

        provider = OpenTelemetryTracingProvider(self.mock_config)

        # Test when no current span
        provider._current_context = Mock()
        provider._current_context.current_span = None
        self.assertIsNone(provider.get_current_span())

        # Test when current span exists
        mock_span = Mock()
        provider._current_context.current_span = mock_span
        self.assertEqual(provider.get_current_span(), mock_span)

    @patch('src.puffinflow.core.observability.tracing.trace')
    def test_set_current_span(self, mock_trace):
        """Test setting current span"""
        from src.puffinflow.core.observability.tracing import OpenTelemetryTracingProvider

        provider = OpenTelemetryTracingProvider(self.mock_config)
        provider._current_context = Mock()

        mock_span = Mock()
        provider._set_current_span(mock_span)

        self.assertEqual(provider._current_context.current_span, mock_span)

    @patch('src.puffinflow.core.observability.tracing.trace')
    def test_span_context_manager_success(self, mock_trace):
        """Test span context manager with successful execution"""
        # Create a proper mock enum value
        mock_user_span_type = Mock()
        mock_user_span_type.value = "user"

        # Mock SpanType.USER specifically for this test
        mock_span_type = Mock()
        mock_span_type.USER = mock_user_span_type

        with patch('src.puffinflow.core.observability.tracing.SpanType', mock_span_type):
            from src.puffinflow.core.observability.tracing import OpenTelemetryTracingProvider

            provider = OpenTelemetryTracingProvider(self.mock_config)
            mock_span = Mock()

            with patch.object(provider, 'start_span', return_value=mock_span):
                with provider.span("test_span", mock_span_type.USER, test_attr="value") as span:
                    self.assertEqual(span, mock_span)
                    # Simulate some work
                    pass

                # Verify span lifecycle
                provider.start_span.assert_called_with("test_span", mock_span_type.USER, None, test_attr="value")
                mock_span.set_status.assert_called_with("ok")
                mock_span.end.assert_called_once()

    @patch('src.puffinflow.core.observability.tracing.trace')
    def test_span_context_manager_exception(self, mock_trace):
        """Test span context manager with exception"""
        from src.puffinflow.core.observability.tracing import OpenTelemetryTracingProvider

        provider = OpenTelemetryTracingProvider(self.mock_config)
        mock_span = Mock()
        test_exception = ValueError("Test error")

        with patch.object(provider, 'start_span', return_value=mock_span), \
                patch.object(provider, '_set_current_span'):
            with self.assertRaises(ValueError):
                with provider.span("test_span") as span:
                    raise test_exception

            # Verify exception handling
            mock_span.record_exception.assert_called_with(test_exception)
            mock_span.set_status.assert_not_called()  # Should not set OK status
            mock_span.end.assert_called_once()


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of the tracing provider"""

    @patch('src.puffinflow.core.observability.tracing.trace')
    def test_thread_local_context(self, mock_trace):
        """Test that span context is thread-local"""
        from src.puffinflow.core.observability.tracing import OpenTelemetryTracingProvider

        # Create a mock config that won't cause initialization issues
        mock_config = Mock()
        mock_config.service_name = "test"
        mock_config.service_version = "1.0"
        mock_config.otlp_endpoint = None
        mock_config.jaeger_endpoint = None
        mock_config.console_enabled = False

        provider = OpenTelemetryTracingProvider(mock_config)
        results = {}

        def worker(thread_id):
            mock_span = Mock()
            mock_span.name = f"span_{thread_id}"
            provider._set_current_span(mock_span)
            time.sleep(0.1)  # Allow other threads to run
            results[thread_id] = provider.get_current_span()

        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Each thread should have its own span
        self.assertEqual(len(results), 3)
        for thread_id, span in results.items():
            self.assertEqual(span.name, f"span_{thread_id}")


if __name__ == '__main__':
    # Create mock modules for the imports
    import sys
    from unittest.mock import MagicMock

    # Mock the interfaces module
    interfaces_mock = MagicMock()
    interfaces_mock.TracingProvider = MagicMock()
    interfaces_mock.Span = MagicMock()
    interfaces_mock.SpanContext = MagicMock()
    interfaces_mock.SpanType = MagicMock()
    interfaces_mock.SpanType.SYSTEM = "system"
    interfaces_mock.SpanType.USER = "user"
    # Make sure the enum values work properly
    interfaces_mock.SpanType.SYSTEM.value = "system"
    interfaces_mock.SpanType.USER.value = "user"
    sys.modules['src.puffinflow.core.observability.interfaces'] = interfaces_mock

    # Mock the config module
    config_mock = MagicMock()
    config_mock.TracingConfig = MagicMock()
    sys.modules['src.puffinflow.core.observability.config'] = config_mock

    # Run the tests
    unittest.main()