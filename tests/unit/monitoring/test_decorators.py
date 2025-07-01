import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock, call
from typing import Any

from src.puffinflow.core.observability.decorators import observe, trace_state
from src.puffinflow.core.observability.interfaces import SpanType


class TestObserveDecorator:
    """Test cases for the observe decorator"""

    @pytest.fixture
    def mock_observability(self):
        """Create a mock observability object"""
        observability = Mock()
        observability.tracing = Mock()
        span = Mock()
        span.__enter__ = Mock(return_value=span)
        span.__exit__ = Mock(return_value=None)
        observability.tracing.span.return_value = span
        return observability, span

    @pytest.fixture
    def mock_observability_disabled(self):
        """Create a mock observability object with tracing disabled"""
        observability = Mock()
        observability.tracing = None
        return observability

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    def test_observe_sync_function_with_tracing(self, mock_get_obs, mock_observability):
        """Test observe decorator with sync function when tracing is enabled"""
        observability, span = mock_observability
        mock_get_obs.return_value = observability

        @observe(name="test_function", span_type=SpanType.BUSINESS, custom_attr="value")
        def sync_function(x, y):
            return x + y

        result = sync_function(1, 2)

        assert result == 3
        observability.tracing.span.assert_called_once_with(
            "test_function",
            SpanType.BUSINESS,
            custom_attr="value"
        )
        span.set_status.assert_called_once_with("ok")

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    def test_observe_sync_function_without_tracing(self, mock_get_obs, mock_observability_disabled):
        """Test observe decorator with sync function when tracing is disabled"""
        mock_get_obs.return_value = mock_observability_disabled

        @observe()
        def sync_function(x, y):
            return x + y

        result = sync_function(1, 2)

        assert result == 3

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    @pytest.mark.asyncio
    async def test_observe_async_function_with_tracing(self, mock_get_obs, mock_observability):
        """Test observe decorator with async function when tracing is enabled"""
        observability, span = mock_observability
        mock_get_obs.return_value = observability

        @observe(name="async_test", span_type=SpanType.SYSTEM)
        async def async_function(x, y):
            await asyncio.sleep(0.01)  # Small delay to test timing
            return x * y

        result = await async_function(3, 4)

        assert result == 12
        observability.tracing.span.assert_called_once_with(
            "async_test",
            SpanType.SYSTEM,
            function="async_function"
        )
        span.set_status.assert_called_once_with("ok")
        # Check that duration was set
        duration_calls = [call for call in span.set_attribute.call_args_list
                          if call[0][0] == "function.duration"]
        assert len(duration_calls) == 1
        assert duration_calls[0][0][1] > 0  # Duration should be positive

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    @pytest.mark.asyncio
    async def test_observe_async_function_without_tracing(self, mock_get_obs, mock_observability_disabled):
        """Test observe decorator with async function when tracing is disabled"""
        mock_get_obs.return_value = mock_observability_disabled

        @observe()
        async def async_function(x, y):
            return x * y

        result = await async_function(3, 4)

        assert result == 12

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    def test_observe_sync_function_with_exception(self, mock_get_obs, mock_observability):
        """Test observe decorator with sync function that raises an exception"""
        observability, span = mock_observability
        mock_get_obs.return_value = observability

        @observe()
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

        span.record_exception.assert_called_once()
        span.set_status.assert_not_called()

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    @pytest.mark.asyncio
    async def test_observe_async_function_with_exception(self, mock_get_obs, mock_observability):
        """Test observe decorator with async function that raises an exception"""
        observability, span = mock_observability
        mock_get_obs.return_value = observability

        @observe()
        async def failing_async_function():
            raise RuntimeError("Async test error")

        with pytest.raises(RuntimeError, match="Async test error"):
            await failing_async_function()

        span.record_exception.assert_called_once()
        span.set_status.assert_not_called()

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    def test_observe_default_name_generation(self, mock_get_obs, mock_observability):
        """Test that default name is generated from module and function name"""
        observability, span = mock_observability
        mock_get_obs.return_value = observability

        @observe()
        def test_function():
            return "test"

        test_function()

        # Should use module.function_name format
        # Note: sync functions don't pass the 'function' parameter to span
        expected_name = f"{test_function.__module__}.test_function"
        observability.tracing.span.assert_called_once_with(
            expected_name,
            SpanType.BUSINESS
        )

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    def test_observe_with_span_none(self, mock_get_obs):
        """Test observe decorator when span returns None"""
        observability = Mock()
        observability.tracing = Mock()

        # Create a proper context manager mock that returns None
        span_context_manager = MagicMock()
        span_context_manager.__enter__.return_value = None
        span_context_manager.__exit__.return_value = None
        observability.tracing.span.return_value = span_context_manager

        mock_get_obs.return_value = observability

        @observe()
        def test_function():
            return "test"

        result = test_function()

        assert result == "test"
        # Should not call set_status or set_attribute on None span

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    def test_observe_preserves_function_metadata(self, mock_get_obs, mock_observability_disabled):
        """Test that the decorator preserves function metadata"""
        mock_get_obs.return_value = mock_observability_disabled

        @observe()
        def original_function(x: int, y: int) -> int:
            """A test function"""
            return x + y

        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ == "A test function"
        assert original_function.__annotations__ == {'x': int, 'y': int, 'return': int}


class TestTraceStateDecorator:
    """Test cases for the trace_state decorator"""

    @pytest.fixture
    def mock_context(self):
        """Create a mock context object"""
        context = Mock()
        context.get_variable.side_effect = lambda key: {
            "workflow_id": "test-workflow-123",
            "agent_name": "test-agent"
        }.get(key)
        return context

    @pytest.fixture
    def mock_observability(self):
        """Create a mock observability object"""
        observability = Mock()
        observability.tracing = Mock()
        span = Mock()
        span.__enter__ = Mock(return_value=span)
        span.__exit__ = Mock(return_value=None)
        observability.tracing.span.return_value = span
        return observability, span

    @pytest.fixture
    def mock_observability_disabled(self):
        """Create a mock observability object with tracing disabled"""
        observability = Mock()
        observability.tracing = None
        return observability

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    @pytest.mark.asyncio
    async def test_trace_state_with_tracing(self, mock_get_obs, mock_context, mock_observability):
        """Test trace_state decorator when tracing is enabled"""
        observability, span = mock_observability
        mock_get_obs.return_value = observability

        @trace_state(span_type=SpanType.STATE, custom_attr="test_value")
        async def test_state(context, param1, param2="default"):
            return f"processed {param1} {param2}"

        result = await test_state(mock_context, "arg1", param2="arg2")

        assert result == "processed arg1 arg2"

        expected_attrs = {
            "state.name": "test_state",
            "workflow.id": "test-workflow-123",
            "agent.name": "test-agent",
            "custom_attr": "test_value"
        }

        observability.tracing.span.assert_called_once_with(
            "state.test_state",
            SpanType.STATE,
            **expected_attrs
        )
        span.set_status.assert_called_once_with("ok")

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    @pytest.mark.asyncio
    async def test_trace_state_without_tracing(self, mock_get_obs, mock_context, mock_observability_disabled):
        """Test trace_state decorator when tracing is disabled"""
        mock_get_obs.return_value = mock_observability_disabled

        @trace_state()
        async def test_state(context, param):
            return f"processed {param}"

        result = await test_state(mock_context, "test")

        assert result == "processed test"

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    @pytest.mark.asyncio
    async def test_trace_state_with_exception(self, mock_get_obs, mock_context, mock_observability):
        """Test trace_state decorator when state function raises an exception"""
        observability, span = mock_observability
        mock_get_obs.return_value = observability

        @trace_state()
        async def failing_state(context):
            raise ValueError("State processing failed")

        with pytest.raises(ValueError, match="State processing failed"):
            await failing_state(mock_context)

        span.record_exception.assert_called_once()
        span.set_status.assert_not_called()

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    @pytest.mark.asyncio
    async def test_trace_state_with_missing_context_variables(self, mock_get_obs, mock_observability):
        """Test trace_state decorator when context variables are missing"""
        observability, span = mock_observability
        mock_get_obs.return_value = observability

        context = Mock()
        context.get_variable.return_value = None

        @trace_state()
        async def test_state(context):
            return "processed"

        result = await test_state(context)

        assert result == "processed"

        expected_attrs = {
            "state.name": "test_state",
            "workflow.id": None,
            "agent.name": None
        }

        observability.tracing.span.assert_called_once_with(
            "state.test_state",
            SpanType.STATE,
            **expected_attrs
        )

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    @pytest.mark.asyncio
    async def test_trace_state_with_span_none(self, mock_get_obs, mock_context):
        """Test trace_state decorator when span returns None"""
        observability = Mock()
        observability.tracing = Mock()

        # Create a proper context manager mock that returns None
        span_context_manager = MagicMock()
        span_context_manager.__enter__.return_value = None
        span_context_manager.__exit__.return_value = None
        observability.tracing.span.return_value = span_context_manager

        mock_get_obs.return_value = observability

        @trace_state()
        async def test_state(context):
            return "processed"

        result = await test_state(mock_context)

        assert result == "processed"

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    @pytest.mark.asyncio
    async def test_trace_state_preserves_function_metadata(self, mock_get_obs, mock_observability_disabled):
        """Test that trace_state decorator preserves function metadata"""
        mock_get_obs.return_value = mock_observability_disabled

        @trace_state()
        async def original_state_function(context: Any, param: str) -> str:
            """A test state function"""
            return f"processed {param}"

        assert original_state_function.__name__ == "original_state_function"
        assert original_state_function.__doc__ == "A test state function"
        assert original_state_function.__annotations__ == {
            'context': Any,
            'param': str,
            'return': str
        }


class TestDecoratorIntegration:
    """Integration tests for decorator behavior"""

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    def test_multiple_decorators_on_same_function(self, mock_get_obs):
        """Test behavior when multiple decorators are applied"""
        observability = Mock()
        observability.tracing = None
        mock_get_obs.return_value = observability

        @observe(name="outer")
        @observe(name="inner")
        def decorated_function(x):
            return x * 2

        result = decorated_function(5)
        assert result == 10

    @patch('src.puffinflow.core.observability.decorators.get_observability')
    @pytest.mark.asyncio
    async def test_timing_accuracy(self, mock_get_obs):
        """Test that timing measurements are reasonably accurate"""
        observability = Mock()
        observability.tracing = Mock()
        span = Mock()
        span.__enter__ = Mock(return_value=span)
        span.__exit__ = Mock(return_value=None)
        observability.tracing.span.return_value = span
        mock_get_obs.return_value = observability

        @observe()
        async def timed_function():
            await asyncio.sleep(0.1)  # 100ms delay
            return "done"

        start = time.time()
        result = await timed_function()
        end = time.time()
        actual_duration = end - start

        assert result == "done"

        # Check that duration was recorded
        duration_calls = [call for call in span.set_attribute.call_args_list
                          if call[0][0] == "function.duration"]
        assert len(duration_calls) == 1
        recorded_duration = duration_calls[0][0][1]

        # Duration should be close to actual (within 50ms tolerance)
        assert abs(recorded_duration - actual_duration) < 0.05


if __name__ == "__main__":
    pytest.main([__file__])