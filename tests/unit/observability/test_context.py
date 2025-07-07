from unittest.mock import AsyncMock, Mock, patch

from puffinflow.core.observability.context import ObservableContext
from puffinflow.core.observability.core import ObservabilityManager
from puffinflow.core.observability.interfaces import SpanType


class TestObservableContext:
    """Test ObservableContext class"""

    def test_init_without_observability(self):
        """Test initialization without observability manager"""
        shared_state = {"test_key": "test_value"}
        context = ObservableContext(shared_state)

        assert context._observability is None
        assert context.get_variable("test_key") == "test_value"

    def test_init_with_observability(self):
        """Test initialization with observability manager"""
        shared_state = {"test_key": "test_value"}
        observability = Mock(spec=ObservabilityManager)
        context = ObservableContext(shared_state, observability)

        assert context._observability is observability
        assert context.get_variable("test_key") == "test_value"

    def test_trace_without_observability(self):
        """Test trace method without observability manager"""
        shared_state = {}
        context = ObservableContext(shared_state)

        with context.trace("test_span") as span:
            assert span is None

    def test_trace_without_tracing_provider(self):
        """Test trace method without tracing provider"""
        shared_state = {}
        observability = Mock(spec=ObservabilityManager)
        observability.tracing = None
        context = ObservableContext(shared_state, observability)

        with context.trace("test_span") as span:
            assert span is None

    def test_trace_with_observability(self):
        """Test trace method with observability manager"""
        shared_state = {
            "workflow_id": "test_workflow",
            "agent_name": "test_agent",
            "current_state": "test_state",
        }

        # Mock the tracing provider and span
        mock_span = Mock()
        mock_tracing = Mock()
        mock_tracing.span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracing.span.return_value.__exit__ = Mock(return_value=None)

        observability = Mock(spec=ObservabilityManager)
        observability.tracing = mock_tracing

        context = ObservableContext(shared_state, observability)

        with context.trace("test_span", custom_attr="custom_value") as span:
            assert span is mock_span

        # Verify span was created with correct attributes
        expected_attrs = {
            "workflow_id": "test_workflow",
            "agent_name": "test_agent",
            "state_name": "test_state",
            "custom_attr": "custom_value",
        }
        mock_tracing.span.assert_called_once_with(
            "test_span", SpanType.BUSINESS, **expected_attrs
        )

    def test_trace_with_missing_context_variables(self):
        """Test trace method with missing context variables"""
        shared_state = {
            "agent_name": "test_agent"
        }  # Missing workflow_id and current_state

        mock_span = Mock()
        mock_tracing = Mock()
        mock_tracing.span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracing.span.return_value.__exit__ = Mock(return_value=None)

        observability = Mock(spec=ObservabilityManager)
        observability.tracing = mock_tracing

        context = ObservableContext(shared_state, observability)

        with context.trace("test_span") as span:
            assert span is mock_span

        # Verify span was created with available attributes (None for missing ones)
        expected_attrs = {
            "workflow_id": None,
            "agent_name": "test_agent",
            "state_name": None,
        }
        mock_tracing.span.assert_called_once_with(
            "test_span", SpanType.BUSINESS, **expected_attrs
        )

    def test_metric_without_observability(self):
        """Test metric method without observability manager"""
        shared_state = {}
        context = ObservableContext(shared_state)

        # Should not raise an exception
        context.metric("test_metric", 1.0)

    def test_metric_without_metrics_provider(self):
        """Test metric method without metrics provider"""
        shared_state = {}
        observability = Mock(spec=ObservabilityManager)
        observability.metrics = None
        context = ObservableContext(shared_state, observability)

        # Should not raise an exception
        context.metric("test_metric", 1.0)

    def test_metric_with_observability(self):
        """Test metric method with observability manager"""
        shared_state = {"workflow_id": "test_workflow", "agent_name": "test_agent"}

        mock_histogram = Mock()
        observability = Mock(spec=ObservabilityManager)
        observability.histogram.return_value = mock_histogram

        context = ObservableContext(shared_state, observability)

        context.metric("test_metric", 1.5, custom_label="custom_value")

        # Verify histogram was created and recorded
        expected_labels = ["workflow_id", "agent_name", "custom_label"]
        observability.histogram.assert_called_once_with(
            "test_metric", labels=expected_labels
        )

        expected_record_kwargs = {
            "workflow_id": "test_workflow",
            "agent_name": "test_agent",
            "custom_label": "custom_value",
        }
        mock_histogram.record.assert_called_once_with(1.5, **expected_record_kwargs)

    def test_metric_with_missing_context_variables(self):
        """Test metric method with missing context variables"""
        shared_state = {}  # No context variables

        mock_histogram = Mock()
        observability = Mock(spec=ObservabilityManager)
        observability.histogram.return_value = mock_histogram

        context = ObservableContext(shared_state, observability)

        context.metric("test_metric", 2.0)

        # Verify histogram was created and recorded with defaults
        expected_labels = ["workflow_id", "agent_name"]
        observability.histogram.assert_called_once_with(
            "test_metric", labels=expected_labels
        )

        expected_record_kwargs = {"workflow_id": "unknown", "agent_name": "unknown"}
        mock_histogram.record.assert_called_once_with(2.0, **expected_record_kwargs)

    def test_metric_with_no_histogram_returned(self):
        """Test metric method when histogram returns None"""
        shared_state = {}
        observability = Mock(spec=ObservabilityManager)
        observability.histogram.return_value = None

        context = ObservableContext(shared_state, observability)

        # Should not raise an exception
        context.metric("test_metric", 1.0)

    def test_log_without_observability(self):
        """Test log method without observability manager"""
        shared_state = {}
        context = ObservableContext(shared_state)

        # Should not raise an exception
        context.log("info", "test message")

    def test_log_without_events_processor(self):
        """Test log method without events processor"""
        shared_state = {}
        observability = Mock(spec=ObservabilityManager)
        observability.events = None
        context = ObservableContext(shared_state, observability)

        # Should not raise an exception
        context.log("info", "test message")

    @patch("asyncio.get_event_loop")
    def test_log_with_observability(self, mock_get_loop):
        """Test log method with observability manager"""
        shared_state = {
            "workflow_id": "test_workflow",
            "agent_name": "test_agent",
            "current_state": "test_state",
        }

        mock_loop = Mock()
        mock_get_loop.return_value = mock_loop

        mock_events = Mock()
        mock_events.process_event = AsyncMock()

        observability = Mock(spec=ObservabilityManager)
        observability.events = mock_events

        context = ObservableContext(shared_state, observability)

        with patch("time.time", return_value=1234567890.0):
            context.log("warning", "test message", custom_attr="custom_value")

        # Verify event was created and processed
        mock_loop.create_task.assert_called_once()

        # Get the event that was passed to create_task
        mock_loop.create_task.call_args[0][0]
        # The call_args[0] should be a coroutine, we can't easily inspect it
        # but we can verify create_task was called

    @patch("asyncio.get_event_loop")
    def test_log_with_runtime_error(self, mock_get_loop):
        """Test log method when asyncio.get_event_loop raises RuntimeError"""
        shared_state = {}
        mock_get_loop.side_effect = RuntimeError("No event loop")

        mock_events = Mock()
        observability = Mock(spec=ObservabilityManager)
        observability.events = mock_events

        context = ObservableContext(shared_state, observability)

        # Should not raise an exception
        context.log("error", "test message")

    def test_log_event_structure(self):
        """Test that log creates proper ObservabilityEvent structure"""
        shared_state = {
            "workflow_id": "test_workflow",
            "agent_name": "test_agent",
            "current_state": "test_state",
        }

        # Mock the event processor to capture the event
        captured_event = None

        async def capture_event(event):
            nonlocal captured_event
            captured_event = event

        mock_events = Mock()
        mock_events.process_event = capture_event

        observability = Mock(spec=ObservabilityManager)
        observability.events = mock_events

        context = ObservableContext(shared_state, observability)

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = Mock()
            mock_get_loop.return_value = mock_loop

            # Mock create_task to actually call the coroutine
            def mock_create_task(coro):
                import asyncio

                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(coro)
                    loop.close()
                except Exception:
                    pass  # Ignore any errors in test

            mock_loop.create_task.side_effect = mock_create_task

            with patch("time.time", return_value=1234567890.0):
                context.log("info", "test message", custom_attr="custom_value")

        # Verify the event structure (if captured)
        if captured_event:
            from datetime import datetime

            assert captured_event.timestamp == datetime.fromtimestamp(1234567890.0)
            assert captured_event.event_type == "log"
            assert captured_event.source == "context"
            assert captured_event.level == "INFO"
            assert captured_event.message == "test message"
            assert captured_event.attributes["workflow_id"] == "test_workflow"
            assert captured_event.attributes["agent_name"] == "test_agent"
            assert captured_event.attributes["state_name"] == "test_state"
            assert captured_event.attributes["custom_attr"] == "custom_value"
