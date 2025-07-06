"""Tests for ObservableAgent"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from puffinflow.core.observability.agent import ObservableAgent
from puffinflow.core.observability.context import ObservableContext
from puffinflow.core.observability.interfaces import SpanType


class TestObservableAgent:
    """Test ObservableAgent class"""

    def test_observable_agent_creation(self):
        """Test ObservableAgent creation"""
        mock_observability = Mock()

        with patch(
            "puffinflow.core.observability.agent.Agent.__init__"
        ) as mock_super_init:
            agent = ObservableAgent("test-agent", observability=mock_observability)

            assert agent._observability == mock_observability
            assert hasattr(agent, "workflow_id")
            mock_super_init.assert_called_once()

    def test_observable_agent_workflow_id_generation(self):
        """Test ObservableAgent workflow_id generation"""
        with patch("puffinflow.core.observability.agent.Agent.__init__"):
            with patch("time.time", return_value=1234567890):
                agent = ObservableAgent("test-agent")
                assert agent.workflow_id == "workflow_1234567890"

    def test_observable_agent_custom_workflow_id(self):
        """Test ObservableAgent with custom workflow_id"""
        with patch("puffinflow.core.observability.agent.Agent.__init__"):
            agent = ObservableAgent("test-agent", workflow_id="custom-workflow-123")
            assert agent.workflow_id == "custom-workflow-123"

    def test_observable_agent_metrics_setup(self):
        """Test ObservableAgent metrics setup"""
        mock_observability = Mock()
        mock_metrics = Mock()
        mock_histogram = Mock()
        mock_observability.metrics = mock_metrics
        mock_metrics.histogram.return_value = mock_histogram

        with patch("puffinflow.core.observability.agent.Agent.__init__"):
            agent = ObservableAgent("test-agent", observability=mock_observability)

            # Check workflow duration metric
            mock_metrics.histogram.assert_any_call(
                "workflow_duration_seconds",
                "Workflow execution duration",
                ["agent_name", "status"],
            )

            # Check state duration metric
            mock_metrics.histogram.assert_any_call(
                "state_execution_duration_seconds",
                "State execution duration",
                ["agent_name", "state_name", "status"],
            )

            assert agent.workflow_duration == mock_histogram
            assert agent.state_duration == mock_histogram

    def test_observable_agent_no_metrics(self):
        """Test ObservableAgent without metrics"""
        mock_observability = Mock()
        mock_observability.metrics = None

        with patch("puffinflow.core.observability.agent.Agent.__init__"):
            agent = ObservableAgent("test-agent", observability=mock_observability)

            # Should not have metrics attributes
            assert not hasattr(agent, "workflow_duration")
            assert not hasattr(agent, "state_duration")

    def test_create_context(self):
        """Test _create_context method"""
        mock_observability = Mock()
        shared_state = {"key": "value"}

        with patch("puffinflow.core.observability.agent.Agent.__init__"):
            agent = ObservableAgent(
                "test-agent", observability=mock_observability, workflow_id="wf-123"
            )
            agent.name = "test-agent"  # Set name since we're mocking parent __init__

            context = agent._create_context(shared_state)

            assert isinstance(context, ObservableContext)
            assert context._observability == mock_observability
            assert context.get_variable("agent_name") == "test-agent"
            assert context.get_variable("workflow_id") == "wf-123"

    @pytest.mark.asyncio
    async def test_run_with_tracing(self):
        """Test run method with tracing"""
        mock_observability = Mock()
        mock_tracing = Mock()
        mock_span = Mock()
        mock_observability.tracing = mock_tracing

        # Create a proper context manager mock
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracing.span.return_value = mock_context_manager

        mock_histogram = Mock()
        mock_observability.metrics = Mock()
        mock_observability.metrics.histogram.return_value = mock_histogram

        with patch("puffinflow.core.observability.agent.Agent.__init__"):
            with patch(
                "puffinflow.core.observability.agent.Agent.run",
                new_callable=AsyncMock,
            ) as mock_super_run:
                agent = ObservableAgent(
                    "test-agent", observability=mock_observability, workflow_id="wf-123"
                )
                agent.name = (
                    "test-agent"  # Set name since we're mocking parent __init__
                )
                agent.workflow_duration = mock_histogram

                await agent.run(timeout=30.0)

                mock_super_run.assert_called_once_with(30.0)
                mock_tracing.span.assert_called_once_with(
                    "workflow.test-agent",
                    SpanType.WORKFLOW,
                    agent_name="test-agent",
                    workflow_id="wf-123",
                )
                mock_span.set_status.assert_called_once_with("ok")
                mock_histogram.record.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_tracing_exception(self):
        """Test run method with tracing and exception"""
        mock_observability = Mock()
        mock_tracing = Mock()
        mock_span = Mock()
        mock_observability.tracing = mock_tracing

        # Create a proper context manager mock
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracing.span.return_value = mock_context_manager

        mock_histogram = Mock()
        mock_observability.metrics = Mock()
        mock_observability.metrics.histogram.return_value = mock_histogram

        test_exception = Exception("Test error")

        with patch("puffinflow.core.observability.agent.Agent.__init__"):
            with patch(
                "puffinflow.core.observability.agent.Agent.run",
                new_callable=AsyncMock,
                side_effect=test_exception,
            ):
                agent = ObservableAgent("test-agent", observability=mock_observability)
                agent.name = (
                    "test-agent"  # Set name since we're mocking parent __init__
                )
                agent.workflow_duration = mock_histogram

                with pytest.raises(Exception, match="Test error"):
                    await agent.run()

                mock_span.record_exception.assert_called_once_with(test_exception)
                # Should record error metric
                mock_histogram.record.assert_called_once()
                call_args = mock_histogram.record.call_args[1]
                assert call_args["status"] == "error"

    @pytest.mark.asyncio
    async def test_run_without_tracing(self):
        """Test run method without tracing"""
        mock_observability = Mock()
        mock_observability.tracing = None

        with patch("puffinflow.core.observability.agent.Agent.__init__"):
            with patch(
                "puffinflow.core.observability.agent.Agent.run",
                new_callable=AsyncMock,
            ) as mock_super_run:
                agent = ObservableAgent("test-agent", observability=mock_observability)

                await agent.run()

                mock_super_run.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_run_state_with_tracing(self):
        """Test run_state method with tracing"""
        mock_observability = Mock()
        mock_tracing = Mock()
        mock_span = Mock()
        mock_observability.tracing = mock_tracing

        # Create a proper context manager mock
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracing.span.return_value = mock_context_manager

        mock_histogram = Mock()
        mock_observability.metrics = Mock()
        mock_observability.metrics.histogram.return_value = mock_histogram

        mock_state_func = AsyncMock()

        with patch("puffinflow.core.observability.agent.Agent.__init__"):
            agent = ObservableAgent("test-agent", observability=mock_observability)
            agent.name = "test-agent"  # Set name since we're mocking parent __init__
            agent.state_duration = mock_histogram
            agent.states = {"test_state": mock_state_func}
            agent.shared_state = {"key": "value"}

            with patch.object(agent, "_create_context") as mock_create_context:
                mock_context = Mock()
                mock_create_context.return_value = mock_context

                await agent.run_state("test_state")

                mock_tracing.span.assert_called_once_with(
                    "state.test_state",
                    SpanType.STATE,
                    agent_name="test-agent",
                    state_name="test_state",
                )
                mock_create_context.assert_called_once_with(agent.shared_state)
                mock_context.set_variable.assert_called_once_with(
                    "current_state", "test_state"
                )
                mock_state_func.assert_called_once_with(mock_context)
                mock_span.set_status.assert_called_once_with("ok")
                mock_histogram.record.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_state_with_tracing_exception(self):
        """Test run_state method with tracing and exception"""
        mock_observability = Mock()
        mock_tracing = Mock()
        mock_span = Mock()
        mock_observability.tracing = mock_tracing

        # Create a proper context manager mock
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_span)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_tracing.span.return_value = mock_context_manager

        mock_histogram = Mock()
        mock_observability.metrics = Mock()
        mock_observability.metrics.histogram.return_value = mock_histogram

        test_exception = Exception("State error")
        mock_state_func = AsyncMock(side_effect=test_exception)

        with patch("puffinflow.core.observability.agent.Agent.__init__"):
            agent = ObservableAgent("test-agent", observability=mock_observability)
            agent.name = "test-agent"  # Set name since we're mocking parent __init__
            agent.state_duration = mock_histogram
            agent.states = {"test_state": mock_state_func}
            agent.shared_state = {}

            with patch.object(agent, "_create_context", return_value=Mock()):
                with pytest.raises(Exception, match="State error"):
                    await agent.run_state("test_state")

                mock_span.record_exception.assert_called_once_with(test_exception)
                # Should record error metric
                mock_histogram.record.assert_called_once()
                call_args = mock_histogram.record.call_args[1]
                assert call_args["status"] == "error"

    @pytest.mark.asyncio
    async def test_run_state_without_tracing(self):
        """Test run_state method without tracing"""
        mock_observability = Mock()
        mock_observability.tracing = None

        with patch("puffinflow.core.observability.agent.Agent.__init__"):
            with patch(
                "puffinflow.core.observability.agent.Agent.run_state",
                new_callable=AsyncMock,
            ) as mock_super_run_state:
                agent = ObservableAgent("test-agent", observability=mock_observability)

                await agent.run_state("test_state")

                mock_super_run_state.assert_called_once_with("test_state")
