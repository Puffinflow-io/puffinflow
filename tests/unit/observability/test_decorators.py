"""Tests for observability decorators"""

from unittest.mock import Mock, patch

import pytest

from puffinflow.core.observability.decorators import observe, trace_state
from puffinflow.core.observability.interfaces import SpanType


class TestObservabilityDecorators:
    """Test observability decorators"""

    @pytest.mark.asyncio
    async def test_observe_decorator_async_function(self):
        """Test observe decorator on async function"""
        mock_observability = Mock()
        mock_tracing = Mock()
        mock_span = Mock()
        mock_observability.tracing = mock_tracing

        # Create a proper context manager mock
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracing.span.return_value = mock_context_manager

        with patch(
            "puffinflow.core.observability.decorators.get_observability",
            return_value=mock_observability,
        ):

            @observe("custom.operation", SpanType.BUSINESS, custom_attr="value")
            async def test_function(arg1, arg2):
                return f"{arg1}-{arg2}"

            result = await test_function("hello", "world")

            assert result == "hello-world"
            mock_tracing.span.assert_called_once_with(
                "custom.operation",
                SpanType.BUSINESS,
                function="test_function",
                custom_attr="value",
            )
            mock_span.set_status.assert_called_once_with("ok")

    @pytest.mark.asyncio
    async def test_observe_decorator_async_function_exception(self):
        """Test observe decorator on async function with exception"""
        mock_observability = Mock()
        mock_tracing = Mock()
        mock_span = Mock()
        mock_observability.tracing = mock_tracing

        # Create a proper context manager mock
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracing.span.return_value = mock_context_manager

        test_exception = Exception("Test error")

        with patch(
            "puffinflow.core.observability.decorators.get_observability",
            return_value=mock_observability,
        ):

            @observe()
            async def test_function():
                raise test_exception

            with pytest.raises(Exception, match="Test error"):
                await test_function()

            mock_span.record_exception.assert_called_once_with(test_exception)

    def test_observe_decorator_sync_function(self):
        """Test observe decorator on sync function"""
        mock_observability = Mock()
        mock_tracing = Mock()
        mock_span = Mock()
        mock_observability.tracing = mock_tracing

        # Create a proper context manager mock
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracing.span.return_value = mock_context_manager

        with patch(
            "puffinflow.core.observability.decorators.get_observability",
            return_value=mock_observability,
        ):

            @observe("custom.sync.operation")
            def test_function(arg1):
                return f"result-{arg1}"

            result = test_function("test")

            assert result == "result-test"
            mock_tracing.span.assert_called_once_with(
                "custom.sync.operation", SpanType.BUSINESS
            )
            mock_span.set_status.assert_called_once_with("ok")

    def test_observe_decorator_sync_function_exception(self):
        """Test observe decorator on sync function with exception"""
        mock_observability = Mock()
        mock_tracing = Mock()
        mock_span = Mock()
        mock_observability.tracing = mock_tracing

        # Create a proper context manager mock
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracing.span.return_value = mock_context_manager

        test_exception = Exception("Sync error")

        with patch(
            "puffinflow.core.observability.decorators.get_observability",
            return_value=mock_observability,
        ):

            @observe()
            def test_function():
                raise test_exception

            with pytest.raises(Exception, match="Sync error"):
                test_function()

            mock_span.record_exception.assert_called_once_with(test_exception)

    @pytest.mark.asyncio
    async def test_observe_decorator_no_tracing(self):
        """Test observe decorator without tracing"""
        mock_observability = Mock()
        mock_observability.tracing = None

        with patch(
            "puffinflow.core.observability.decorators.get_observability",
            return_value=mock_observability,
        ):

            @observe()
            async def test_function():
                return "result"

            result = await test_function()
            assert result == "result"

    def test_observe_decorator_default_name(self):
        """Test observe decorator with default operation name"""
        mock_observability = Mock()
        mock_tracing = Mock()
        mock_span = Mock()
        mock_observability.tracing = mock_tracing

        # Create a proper context manager mock
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracing.span.return_value = mock_context_manager

        with patch(
            "puffinflow.core.observability.decorators.get_observability",
            return_value=mock_observability,
        ):

            @observe()
            def test_function():
                return "result"

            test_function()

            # Should use module.function_name as operation name
            expected_name = f"{test_function.__module__}.{test_function.__name__}"
            mock_tracing.span.assert_called_once()
            call_args = mock_tracing.span.call_args[0]
            assert call_args[0] == expected_name

    @pytest.mark.asyncio
    async def test_trace_state_decorator(self):
        """Test trace_state decorator"""
        mock_observability = Mock()
        mock_tracing = Mock()
        mock_span = Mock()
        mock_observability.tracing = mock_tracing

        # Create a proper context manager mock
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracing.span.return_value = mock_context_manager

        mock_context = Mock()
        mock_context.get_variable.side_effect = lambda key: {
            "workflow_id": "wf-123",
            "agent_name": "test-agent",
        }.get(key)

        with patch(
            "puffinflow.core.observability.decorators.get_observability",
            return_value=mock_observability,
        ):

            @trace_state(SpanType.STATE, custom_attr="value")
            async def test_state(context):
                return "state-result"

            result = await test_state(mock_context)

            assert result == "state-result"
            mock_tracing.span.assert_called_once_with(
                "state.test_state",
                SpanType.STATE,
                **{
                    "state.name": "test_state",
                    "workflow.id": "wf-123",
                    "agent.name": "test-agent",
                    "custom_attr": "value",
                },
            )
            mock_span.set_status.assert_called_once_with("ok")

    @pytest.mark.asyncio
    async def test_trace_state_decorator_exception(self):
        """Test trace_state decorator with exception"""
        mock_observability = Mock()
        mock_tracing = Mock()
        mock_span = Mock()
        mock_observability.tracing = mock_tracing

        # Create a proper context manager mock
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracing.span.return_value = mock_context_manager

        mock_context = Mock()
        mock_context.get_variable.return_value = None

        test_exception = Exception("State error")

        with patch(
            "puffinflow.core.observability.decorators.get_observability",
            return_value=mock_observability,
        ):

            @trace_state()
            async def test_state(context):
                raise test_exception

            with pytest.raises(Exception, match="State error"):
                await test_state(mock_context)

            mock_span.record_exception.assert_called_once_with(test_exception)

    @pytest.mark.asyncio
    async def test_trace_state_decorator_no_tracing(self):
        """Test trace_state decorator without tracing"""
        mock_observability = Mock()
        mock_observability.tracing = None

        mock_context = Mock()

        with patch(
            "puffinflow.core.observability.decorators.get_observability",
            return_value=mock_observability,
        ):

            @trace_state()
            async def test_state(context):
                return "result"

            result = await test_state(mock_context)
            assert result == "result"

    def test_observe_decorator_preserves_function_metadata(self):
        """Test that observe decorator preserves function metadata"""
        mock_observability = Mock()
        mock_observability.tracing = None

        with patch(
            "puffinflow.core.observability.decorators.get_observability",
            return_value=mock_observability,
        ):

            @observe()
            def test_function():
                """Test function docstring"""
                return "result"

            assert test_function.__name__ == "test_function"
            assert test_function.__doc__ == "Test function docstring"

    @pytest.mark.asyncio
    async def test_trace_state_decorator_preserves_function_metadata(self):
        """Test that trace_state decorator preserves function metadata"""
        mock_observability = Mock()
        mock_observability.tracing = None

        Mock()

        with patch(
            "puffinflow.core.observability.decorators.get_observability",
            return_value=mock_observability,
        ):

            @trace_state()
            async def test_state(context):
                """Test state function docstring"""
                return "result"

            assert test_state.__name__ == "test_state"
            assert test_state.__doc__ == "Test state function docstring"

    def test_observe_decorator_with_kwargs(self):
        """Test observe decorator with keyword arguments"""
        mock_observability = Mock()
        mock_tracing = Mock()
        mock_span = Mock()
        mock_observability.tracing = mock_tracing

        # Create a proper context manager mock
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracing.span.return_value = mock_context_manager

        with patch(
            "puffinflow.core.observability.decorators.get_observability",
            return_value=mock_observability,
        ):

            @observe("test.operation")
            def test_function(arg1, arg2=None, **kwargs):
                return f"{arg1}-{arg2}-{kwargs.get('extra', 'none')}"

            result = test_function("hello", arg2="world", extra="test")

            assert result == "hello-world-test"
            mock_span.set_status.assert_called_once_with("ok")

    @pytest.mark.asyncio
    async def test_trace_state_with_additional_args(self):
        """Test trace_state decorator with additional arguments"""
        mock_observability = Mock()
        mock_tracing = Mock()
        mock_span = Mock()
        mock_observability.tracing = mock_tracing

        # Create a proper context manager mock
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracing.span.return_value = mock_context_manager

        mock_context = Mock()
        mock_context.get_variable.return_value = None

        with patch(
            "puffinflow.core.observability.decorators.get_observability",
            return_value=mock_observability,
        ):

            @trace_state()
            async def test_state(context, additional_arg, **kwargs):
                return f"result-{additional_arg}-{kwargs.get('extra', 'none')}"

            result = await test_state(mock_context, "test", extra="value")

            assert result == "result-test-value"
            mock_span.set_status.assert_called_once_with("ok")
