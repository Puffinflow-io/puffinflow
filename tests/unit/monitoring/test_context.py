import pytest
import time
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from contextlib import nullcontext
from typing import Dict, Any

from src.puffinflow.core.observability.context import ObservableContext
from src.puffinflow.core.observability.interfaces import SpanType, ObservabilityEvent, SpanContext
from src.puffinflow.core.observability.core import ObservabilityManager
from src.puffinflow.core.agent.context import Context


class TestObservableContext:
    """Comprehensive test suite for ObservableContext"""

    @pytest.fixture
    def shared_state(self):
        """Fixture providing shared state for context"""
        return {
            "workflow_id": "test-workflow-123",
            "agent_name": "test-agent",
            "current_state": "processing"
        }

    @pytest.fixture
    def mock_observability(self):
        """Fixture providing mocked observability manager"""
        mock_obs = Mock(spec=ObservabilityManager)
        mock_obs.tracing = Mock()
        mock_obs.metrics = Mock()
        mock_obs.events = Mock()
        # Make process_event an async mock
        mock_obs.events.process_event = AsyncMock()
        return mock_obs

    @pytest.fixture
    def mock_span(self):
        """Fixture providing mocked span"""
        span = Mock(spec=SpanContext)
        return span

    @pytest.fixture
    def context_with_observability(self, shared_state, mock_observability):
        """Fixture providing ObservableContext with observability"""
        return ObservableContext(shared_state, mock_observability)

    @pytest.fixture
    def context_without_observability(self, shared_state):
        """Fixture providing ObservableContext without observability"""
        return ObservableContext(shared_state, None)

    def test_init_with_observability(self, shared_state, mock_observability):
        """Test initialization with observability manager"""
        context = ObservableContext(shared_state, mock_observability)

        assert context._observability == mock_observability
        # Check that parent Context was initialized by verifying we can access shared state
        assert context.get_variable("workflow_id") == "test-workflow-123"

    def test_init_without_observability(self, shared_state):
        """Test initialization without observability manager"""
        context = ObservableContext(shared_state, None)

        assert context._observability is None

    def test_init_default_observability(self, shared_state):
        """Test initialization with default observability parameter"""
        context = ObservableContext(shared_state)

        assert context._observability is None

    @patch('src.puffinflow.core.observability.context.Context.get_variable')
    def test_trace_with_observability_and_tracing(self, mock_get_variable,
                                                  context_with_observability,
                                                  mock_span):
        """Test trace context manager with valid observability and tracing"""
        # Setup mocks
        mock_get_variable.side_effect = lambda key, default=None: {
            "workflow_id": "test-workflow-123",
            "agent_name": "test-agent",
            "current_state": "processing"
        }.get(key, default)

        context_with_observability._observability.tracing.span.return_value.__enter__ = Mock(return_value=mock_span)
        context_with_observability._observability.tracing.span.return_value.__exit__ = Mock(return_value=None)

        # Test trace execution
        with context_with_observability.trace("test_operation", custom_attr="value") as span:
            assert span == mock_span

        # Verify span was created with correct parameters
        context_with_observability._observability.tracing.span.assert_called_once_with(
            "test_operation",
            SpanType.BUSINESS,
            workflow_id="test-workflow-123",
            agent_name="test-agent",
            state_name="processing",
            custom_attr="value"
        )

    def test_trace_without_observability(self, context_without_observability):
        """Test trace context manager without observability"""
        with context_without_observability.trace("test_operation") as span:
            assert span is None

    def test_trace_without_tracing(self, context_with_observability):
        """Test trace context manager when tracing is None"""
        context_with_observability._observability.tracing = None

        with context_with_observability.trace("test_operation") as span:
            assert span is None

    def test_trace_with_tracing_falsy(self, context_with_observability):
        """Test trace context manager when tracing is falsy"""
        context_with_observability._observability.tracing = False

        with context_with_observability.trace("test_operation") as span:
            assert span is None

    @patch('src.puffinflow.core.observability.context.Context.get_variable')
    def test_trace_with_missing_context_variables(self, mock_get_variable,
                                                  context_with_observability,
                                                  mock_span):
        """Test trace when some context variables are missing"""
        # Setup mock to return None for some variables
        mock_get_variable.side_effect = lambda key, default=None: {
            "workflow_id": "test-workflow-123",
            "agent_name": None,
            "current_state": None
        }.get(key, default)

        context_with_observability._observability.tracing.span.return_value.__enter__ = Mock(return_value=mock_span)
        context_with_observability._observability.tracing.span.return_value.__exit__ = Mock(return_value=None)

        with context_with_observability.trace("test_operation"):
            pass

        # Verify span was still created with available attributes
        context_with_observability._observability.tracing.span.assert_called_once_with(
            "test_operation",
            SpanType.BUSINESS,
            workflow_id="test-workflow-123",
            agent_name=None,
            state_name=None
        )

    @patch('src.puffinflow.core.observability.context.Context.get_variable')
    def test_metric_with_observability_and_metrics(self, mock_get_variable,
                                                   context_with_observability):
        """Test metric recording with valid observability and metrics"""
        # Setup mocks
        mock_get_variable.side_effect = lambda key, default="unknown": {
            "workflow_id": "test-workflow-123",
            "agent_name": "test-agent"
        }.get(key, default)

        mock_histogram = Mock()
        context_with_observability._observability.histogram.return_value = mock_histogram

        # Test metric recording
        context_with_observability.metric("test_metric", 1.5, custom_label="value")

        # Verify histogram creation and recording
        context_with_observability._observability.histogram.assert_called_once_with(
            "test_metric",
            labels=["workflow_id", "agent_name", "custom_label"]
        )
        mock_histogram.record.assert_called_once_with(
            1.5,
            workflow_id="test-workflow-123",
            agent_name="test-agent",
            custom_label="value"
        )

    def test_metric_without_observability(self, context_without_observability):
        """Test metric recording without observability"""
        # Should not raise exception
        context_without_observability.metric("test_metric", 1.5)

    def test_metric_without_metrics(self, context_with_observability):
        """Test metric recording when metrics is None"""
        context_with_observability._observability.metrics = None

        # Should not raise exception
        context_with_observability.metric("test_metric", 1.5)

    def test_metric_with_metrics_falsy(self, context_with_observability):
        """Test metric recording when metrics is falsy"""
        context_with_observability._observability.metrics = False

        # Should not raise exception
        context_with_observability.metric("test_metric", 1.5)

    @patch('src.puffinflow.core.observability.context.Context.get_variable')
    def test_metric_with_no_histogram_returned(self, mock_get_variable,
                                               context_with_observability):
        """Test metric recording when histogram returns None"""
        mock_get_variable.side_effect = lambda key, default="unknown": default
        context_with_observability._observability.histogram.return_value = None

        # Should not raise exception
        context_with_observability.metric("test_metric", 1.5)

    @patch('src.puffinflow.core.observability.context.Context.get_variable')
    def test_metric_with_default_labels(self, mock_get_variable,
                                        context_with_observability):
        """Test metric recording with default label values"""
        # Return None for get_variable calls
        mock_get_variable.return_value = None

        mock_histogram = Mock()
        context_with_observability._observability.histogram.return_value = mock_histogram

        context_with_observability.metric("test_metric", 2.0)

        # The actual code uses None when get_variable returns None
        mock_histogram.record.assert_called_once_with(
            2.0,
            workflow_id=None,
            agent_name=None
        )

    @patch('time.time', return_value=1234567890.0)
    @patch('src.puffinflow.core.observability.context.Context.get_variable')
    @patch('asyncio.get_event_loop')
    def test_log_with_observability_and_events(self, mock_get_loop,
                                               mock_get_variable,
                                               mock_time,
                                               context_with_observability):
        """Test logging with valid observability and events"""
        # Setup mocks
        mock_get_variable.side_effect = lambda key, default=None: {
            "workflow_id": "test-workflow-123",
            "agent_name": "test-agent",
            "current_state": "processing"
        }.get(key, default)

        mock_loop = Mock()
        mock_get_loop.return_value = mock_loop
        mock_task = Mock()
        mock_loop.create_task.return_value = mock_task

        # Test logging
        context_with_observability.log("info", "Test message", extra_attr="value")

        # Verify event processing
        context_with_observability._observability.events.process_event.assert_called_once()

        # Get the event that was passed
        call_args = context_with_observability._observability.events.process_event.call_args[0]
        event = call_args[0]

        assert isinstance(event, ObservabilityEvent)
        assert event.timestamp == 1234567890.0
        assert event.event_type == "log"
        assert event.source == "context"
        assert event.level == "INFO"
        assert event.message == "Test message"
        assert event.attributes["workflow_id"] == "test-workflow-123"
        assert event.attributes["agent_name"] == "test-agent"
        assert event.attributes["state_name"] == "processing"
        assert event.attributes["extra_attr"] == "value"

        # Verify task creation
        mock_loop.create_task.assert_called_once()

    def test_log_without_observability(self, context_without_observability):
        """Test logging without observability"""
        # Should not raise exception
        context_without_observability.log("info", "Test message")

    def test_log_without_events(self, context_with_observability):
        """Test logging when events is None"""
        context_with_observability._observability.events = None

        # Should not raise exception
        context_with_observability.log("info", "Test message")

    def test_log_with_events_falsy(self, context_with_observability):
        """Test logging when events is falsy"""
        context_with_observability._observability.events = False

        # Should not raise exception
        context_with_observability.log("info", "Test message")

    @patch('time.time', return_value=1234567890.0)
    @patch('src.puffinflow.core.observability.context.Context.get_variable')
    @patch('asyncio.get_event_loop', side_effect=RuntimeError("No running event loop"))
    def test_log_without_event_loop(self, mock_get_loop,
                                    mock_get_variable,
                                    mock_time,
                                    context_with_observability):
        """Test logging when no event loop is running"""
        mock_get_variable.return_value = None

        # Should not raise exception even when no event loop
        context_with_observability.log("error", "Test error")

        # When RuntimeError is caught, process_event should NOT be called
        context_with_observability._observability.events.process_event.assert_not_called()

    @patch('time.time', return_value=1234567890.0)
    @patch('src.puffinflow.core.observability.context.Context.get_variable')
    @patch('asyncio.get_event_loop')
    def test_log_level_conversion(self, mock_get_loop,
                                  mock_get_variable,
                                  mock_time,
                                  context_with_observability):
        """Test that log levels are converted to uppercase"""
        mock_get_variable.return_value = None

        mock_loop = Mock()
        mock_get_loop.return_value = mock_loop

        context_with_observability.log("debug", "Debug message")

        call_args = context_with_observability._observability.events.process_event.call_args[0]
        event = call_args[0]
        assert event.level == "DEBUG"

    @patch('time.time', return_value=1234567890.0)
    @patch('src.puffinflow.core.observability.context.Context.get_variable')
    @patch('asyncio.get_event_loop')
    def test_log_with_missing_context_variables(self, mock_get_loop,
                                                mock_get_variable,
                                                mock_time,
                                                context_with_observability):
        """Test logging when context variables are missing"""
        mock_get_variable.return_value = None

        mock_loop = Mock()
        mock_get_loop.return_value = mock_loop

        context_with_observability.log("warning", "Warning message")

        call_args = context_with_observability._observability.events.process_event.call_args[0]
        event = call_args[0]

        assert event.attributes["workflow_id"] is None
        assert event.attributes["agent_name"] is None
        assert event.attributes["state_name"] is None

    def test_inheritance_from_context(self, shared_state):
        """Test that ObservableContext properly inherits from Context"""
        context = ObservableContext(shared_state)
        assert isinstance(context, Context)

    @patch('src.puffinflow.core.observability.context.Context.__init__')
    def test_super_init_called(self, mock_super_init, shared_state, mock_observability):
        """Test that parent __init__ is called correctly"""
        ObservableContext(shared_state, mock_observability)
        mock_super_init.assert_called_once_with(shared_state)


# Integration tests
class TestObservableContextIntegration:
    """Integration tests for ObservableContext"""

    @pytest.fixture
    def real_context(self):
        """Fixture providing a more realistic context setup"""
        shared_state = {
            "workflow_id": "integration-test-workflow",
            "agent_name": "integration-agent",
            "current_state": "active"
        }
        return ObservableContext(shared_state)

    def test_multiple_operations_without_observability(self, real_context):
        """Test multiple operations work correctly without observability"""
        # All operations should work without raising exceptions
        with real_context.trace("operation1"):
            real_context.metric("test_metric", 1.0)
            real_context.log("info", "Test message")

        with real_context.trace("operation2", custom="value"):
            real_context.metric("another_metric", 2.0, label="test")
            real_context.log("debug", "Debug message", extra="data")

    def test_context_manager_exception_handling(self, real_context):
        """Test that trace context manager handles exceptions properly"""
        try:
            with real_context.trace("failing_operation"):
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception

        # Context should still be usable after exception
        with real_context.trace("recovery_operation"):
            real_context.log("info", "Recovered successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])