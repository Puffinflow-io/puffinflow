# tests/test_observability_agent.py
import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.puffinflow.core.observability.agent import ObservableAgent
from src.puffinflow.core.observability.core import ObservabilityManager
from src.puffinflow.core.observability.context import ObservableContext
from src.puffinflow.core.observability.interfaces import SpanType, SpanContext


class TestObservableAgent:
    """Test suite for ObservableAgent"""

    @pytest.fixture
    def mock_observability(self):
        """Create mock observability manager"""
        observability = Mock(spec=ObservabilityManager)

        # Mock tracing
        mock_tracing = Mock()
        mock_span = Mock()
        mock_span.set_attribute = Mock()
        mock_span.set_status = Mock()
        mock_span.record_exception = Mock()
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=None)

        mock_tracing.span = Mock(return_value=mock_span)
        observability.tracing = mock_tracing

        # Mock metrics
        mock_metrics = Mock()
        mock_histogram = Mock()
        mock_histogram.record = Mock()
        mock_metrics.histogram = Mock(return_value=mock_histogram)
        observability.metrics = mock_metrics

        return observability

    @pytest.fixture
    def mock_shared_state(self):
        """Create mock shared state"""
        return {
            "test_key": "test_value",
            "workflow_data": {"step": 1}
        }

    @pytest.fixture
    def test_states(self):
        """Create test state functions"""

        async def test_state_1(context):
            context.set_variable("state_1_executed", True)
            return "state_1_result"

        async def test_state_2(context):
            context.set_variable("state_2_executed", True)
            return "state_2_result"

        async def failing_state(context):
            raise ValueError("Test error in state")

        async def slow_state(context):
            await asyncio.sleep(0.1)
            context.set_variable("slow_state_executed", True)
            return "slow_result"

        return {
            "test_state_1": test_state_1,
            "test_state_2": test_state_2,
            "failing_state": failing_state,
            "slow_state": slow_state
        }

    def test_init_with_observability(self, mock_observability):
        """Test agent initialization with observability"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability,
            workflow_id="test_workflow_123"
        )

        assert agent.name == "test_agent"
        assert agent.workflow_id == "test_workflow_123"
        assert agent._observability == mock_observability

        # Check that metrics were initialized
        mock_observability.metrics.histogram.assert_any_call(
            "workflow_duration_seconds",
            "Workflow execution duration",
            ["agent_name", "status"]
        )
        mock_observability.metrics.histogram.assert_any_call(
            "state_execution_duration_seconds",
            "State execution duration",
            ["agent_name", "state_name", "status"]
        )

    def test_init_without_observability(self):
        """Test agent initialization without observability"""
        agent = ObservableAgent(name="test_agent")

        assert agent.name == "test_agent"
        assert agent._observability is None
        assert "workflow_" in agent.workflow_id
        assert not hasattr(agent, 'workflow_duration')

    def test_init_auto_workflow_id(self):
        """Test automatic workflow ID generation"""
        with patch('time.time', return_value=1234567890):
            agent = ObservableAgent(name="test_agent")
            assert agent.workflow_id == "workflow_1234567890"

    def test_init_with_custom_kwargs(self, mock_observability):
        """Test initialization with custom kwargs"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability,
            workflow_id="custom_workflow",
            max_concurrent=15,  # Valid Agent parameter
            enable_circuit_breaker=False  # Valid Agent parameter
        )

        assert agent.name == "test_agent"
        assert agent.workflow_id == "custom_workflow"
        assert agent.max_concurrent == 15
        assert agent.enable_circuit_breaker == False

    def test_create_context_with_observability(self, mock_observability, mock_shared_state):
        """Test context creation with observability"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability,
            workflow_id="test_workflow"
        )

        context = agent._create_context(mock_shared_state)

        assert isinstance(context, ObservableContext)
        assert context._observability == mock_observability
        assert context.get_variable("agent_name") == "test_agent"
        assert context.get_variable("workflow_id") == "test_workflow"

        # Check shared state is preserved
        assert context.get_variable("test_key") == "test_value"

    def test_create_context_without_observability(self, mock_shared_state):
        """Test context creation without observability"""
        agent = ObservableAgent(name="test_agent")

        context = agent._create_context(mock_shared_state)

        assert isinstance(context, ObservableContext)
        assert context._observability is None
        assert context.get_variable("agent_name") == "test_agent"

    def test_create_context_preserves_state(self, mock_observability):
        """Test that context creation preserves shared state"""
        shared_state = {
            "existing_key": "existing_value",
            "nested": {"data": [1, 2, 3]}
        }

        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability
        )

        context = agent._create_context(shared_state)

        # Original state should be preserved
        assert context.get_variable("existing_key") == "existing_value"
        assert context.get_variable("nested") == {"data": [1, 2, 3]}

        # Agent-specific variables should be added
        assert context.get_variable("agent_name") == "test_agent"
        assert context.get_variable("workflow_id") is not None

    @pytest.mark.asyncio
    async def test_run_with_observability_success(self, mock_observability):
        """Test successful workflow run with observability"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability,
            workflow_id="test_workflow"
        )

        # Mock the parent run method
        with patch.object(agent.__class__.__bases__[0], 'run', new_callable=AsyncMock) as mock_parent_run:
            await agent.run(timeout=30)

            # Verify tracing was called
            mock_observability.tracing.span.assert_called_once_with(
                "workflow.test_agent",
                SpanType.WORKFLOW,
                agent_name="test_agent",
                workflow_id="test_workflow"
            )

            # Verify parent run was called
            mock_parent_run.assert_called_once_with(30)

            # Verify span was configured correctly
            span = mock_observability.tracing.span.return_value.__enter__.return_value
            span.set_attribute.assert_called()
            span.set_status.assert_called_with("ok")

            # Verify metrics were recorded
            agent.workflow_duration.record.assert_called_once()
            call_args = agent.workflow_duration.record.call_args
            assert call_args[1]["agent_name"] == "test_agent"
            assert call_args[1]["status"] == "success"

    @pytest.mark.asyncio
    async def test_run_with_observability_failure(self, mock_observability):
        """Test workflow run failure with observability"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability,
            workflow_id="test_workflow"
        )

        test_error = RuntimeError("Workflow failed")

        # Mock the parent run method to raise an exception
        with patch.object(agent.__class__.__bases__[0], 'run', new_callable=AsyncMock) as mock_parent_run:
            mock_parent_run.side_effect = test_error

            with pytest.raises(RuntimeError, match="Workflow failed"):
                await agent.run(timeout=30)

            # Verify span recorded the exception
            span = mock_observability.tracing.span.return_value.__enter__.return_value
            span.record_exception.assert_called_once_with(test_error)

            # Verify error metrics were recorded
            agent.workflow_duration.record.assert_called_once()
            call_args = agent.workflow_duration.record.call_args
            assert call_args[1]["status"] == "error"

    @pytest.mark.asyncio
    async def test_run_without_observability(self):
        """Test workflow run without observability"""
        agent = ObservableAgent(name="test_agent")

        # Mock the parent run method
        with patch.object(agent.__class__.__bases__[0], 'run', new_callable=AsyncMock) as mock_parent_run:
            await agent.run(timeout=30)

            # Verify parent run was called
            mock_parent_run.assert_called_once_with(30)

    @pytest.mark.asyncio
    async def test_run_with_no_timeout(self, mock_observability):
        """Test workflow run without timeout parameter"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability
        )

        with patch.object(agent.__class__.__bases__[0], 'run', new_callable=AsyncMock) as mock_parent_run:
            await agent.run()

            mock_parent_run.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_run_records_duration_metrics(self, mock_observability):
        """Test that run method records duration metrics"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability
        )

        # Mock time to control duration measurement
        with patch('time.time', side_effect=[1000.0, 1005.5]):  # 5.5 second duration
            with patch.object(agent.__class__.__bases__[0], 'run', new_callable=AsyncMock):
                await agent.run()

                # Verify duration was recorded
                agent.workflow_duration.record.assert_called_once()
                duration_recorded = agent.workflow_duration.record.call_args[0][0]
                assert duration_recorded == 5.5

    @pytest.mark.asyncio
    async def test_run_state_with_observability_success(self, mock_observability, test_states):
        """Test successful state execution with observability"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability,
            workflow_id="test_workflow"
        )
        agent.states = test_states
        agent.shared_state = {}

        await agent.run_state("test_state_1")

        # Verify tracing was called
        mock_observability.tracing.span.assert_called_once_with(
            "state.test_state_1",
            SpanType.STATE,
            agent_name="test_agent",
            state_name="test_state_1"
        )

        # Verify span was configured correctly
        span = mock_observability.tracing.span.return_value.__enter__.return_value
        span.set_attribute.assert_called()
        span.set_status.assert_called_with("ok")

        # Verify metrics were recorded
        agent.state_duration.record.assert_called_once()
        call_args = agent.state_duration.record.call_args
        assert call_args[1]["agent_name"] == "test_agent"
        assert call_args[1]["state_name"] == "test_state_1"
        assert call_args[1]["status"] == "success"

        # Verify state was actually executed
        assert agent.shared_state.get("state_1_executed") is True

    @pytest.mark.asyncio
    async def test_run_state_with_observability_failure(self, mock_observability, test_states):
        """Test state execution failure with observability"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability,
            workflow_id="test_workflow"
        )
        agent.states = test_states
        agent.shared_state = {}

        with pytest.raises(ValueError, match="Test error in state"):
            await agent.run_state("failing_state")

        # Verify span recorded the exception
        span = mock_observability.tracing.span.return_value.__enter__.return_value
        span.record_exception.assert_called_once()

        # Verify error metrics were recorded
        agent.state_duration.record.assert_called_once()
        call_args = agent.state_duration.record.call_args
        assert call_args[1]["status"] == "error"

    @pytest.mark.asyncio
    async def test_run_state_without_observability(self, test_states):
        """Test state execution without observability"""
        agent = ObservableAgent(name="test_agent")
        agent.states = test_states
        agent.shared_state = {}

        # Mock the parent run_state method
        with patch.object(agent.__class__.__bases__[0], 'run_state', new_callable=AsyncMock) as mock_parent_run:
            await agent.run_state("test_state_1")

            # Verify parent run_state was called
            mock_parent_run.assert_called_once_with("test_state_1")

    @pytest.mark.asyncio
    async def test_run_state_context_setup(self, mock_observability, test_states):
        """Test that run_state properly sets up context"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability,
            workflow_id="test_workflow"
        )
        agent.states = test_states
        agent.shared_state = {"existing_data": "preserved"}

        # Create a state that checks context
        async def context_checking_state(context):
            assert context.get_variable("agent_name") == "test_agent"
            assert context.get_variable("workflow_id") == "test_workflow"
            assert context.get_variable("current_state") == "context_checking_state"
            assert context.get_variable("existing_data") == "preserved"
            return "context_ok"

        agent.states["context_checking_state"] = context_checking_state

        await agent.run_state("context_checking_state")

        # If we get here without assertion errors, context was set up correctly

    @pytest.mark.asyncio
    async def test_run_state_duration_measurement(self, mock_observability, test_states):
        """Test that state duration is measured correctly"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability
        )
        agent.states = test_states
        agent.shared_state = {}

        # Mock time to control duration measurement
        with patch('time.time', side_effect=[2000.0, 2000.1]):  # 0.1 second duration
            await agent.run_state("test_state_1")

            # Verify duration was recorded
            agent.state_duration.record.assert_called_once()
            duration_recorded = agent.state_duration.record.call_args[0][0]
            import pytest
            assert duration_recorded == pytest.approx(0.1, rel=1e-9)

    @pytest.mark.asyncio
    async def test_run_state_span_attributes(self, mock_observability, test_states):
        """Test that state span gets correct attributes"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability,
            workflow_id="test_workflow"
        )
        agent.states = test_states
        agent.shared_state = {}

        await agent.run_state("test_state_1")

        # Check span was created with correct attributes
        mock_observability.tracing.span.assert_called_once_with(
            "state.test_state_1",
            SpanType.STATE,
            agent_name="test_agent",
            state_name="test_state_1"
        )

        # Check span attributes were set
        span = mock_observability.tracing.span.return_value.__enter__.return_value
        span.set_attribute.assert_called()
        span.set_status.assert_called_with("ok")

    @pytest.mark.asyncio
    async def test_run_state_missing_state(self, mock_observability):
        """Test run_state with non-existent state"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability
        )
        agent.states = {}
        agent.shared_state = {}

        with pytest.raises(KeyError):
            await agent.run_state("non_existent_state")

        # Verify span still recorded the exception
        span = mock_observability.tracing.span.return_value.__enter__.return_value
        span.record_exception.assert_called_once()

    def test_workflow_id_persistence(self, mock_observability):
        """Test that workflow ID persists across operations"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability,
            workflow_id="persistent_workflow"
        )

        # Check initial workflow ID
        assert agent.workflow_id == "persistent_workflow"

        # Create context and check workflow ID is preserved
        context = agent._create_context({})
        assert context.get_variable("workflow_id") == "persistent_workflow"

    def test_metrics_initialization_conditions(self):
        """Test metrics initialization under different conditions"""
        # With observability but no metrics
        mock_obs_no_metrics = Mock(spec=ObservabilityManager)
        mock_obs_no_metrics.metrics = None

        agent = ObservableAgent(
            name="test_agent",
            observability=mock_obs_no_metrics
        )

        assert not hasattr(agent, 'workflow_duration')
        assert not hasattr(agent, 'state_duration')

    @pytest.mark.asyncio
    async def test_multiple_state_executions(self, mock_observability, test_states):
        """Test multiple state executions maintain separate spans"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability
        )
        agent.states = test_states
        agent.shared_state = {}

        # Execute multiple states
        await agent.run_state("test_state_1")
        await agent.run_state("test_state_2")

        # Verify tracing was called for each state
        assert mock_observability.tracing.span.call_count == 2

        # Verify both states were executed
        assert agent.shared_state.get("state_1_executed") is True
        assert agent.shared_state.get("state_2_executed") is True

    @pytest.mark.asyncio
    async def test_concurrent_state_execution(self, mock_observability, test_states):
        """Test concurrent state execution (if supported)"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability
        )
        agent.states = test_states
        agent.shared_state = {}

        # Note: This test assumes states can be run concurrently
        # Adjust based on actual Agent implementation
        tasks = [
            agent.run_state("test_state_1"),
            agent.run_state("test_state_2")
        ]

        await asyncio.gather(*tasks)

        # Verify both spans were created
        assert mock_observability.tracing.span.call_count == 2

        # Verify both metrics were recorded
        assert agent.state_duration.record.call_count == 2

    def test_agent_name_validation(self, mock_observability):
        """Test agent name validation and usage"""
        agent = ObservableAgent(
            name="special-agent_123",
            observability=mock_observability
        )

        assert agent.name == "special-agent_123"

        # Check name propagates to context
        context = agent._create_context({})
        assert context.get_variable("agent_name") == "special-agent_123"

    @pytest.mark.asyncio
    async def test_exception_handling_preserves_metrics(self, mock_observability):
        """Test that exceptions don't prevent metrics recording"""
        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability
        )

        test_error = RuntimeError("Test error")

        with patch.object(agent.__class__.__bases__[0], 'run', new_callable=AsyncMock) as mock_parent_run:
            mock_parent_run.side_effect = test_error

            with pytest.raises(RuntimeError):
                await agent.run()

            # Metrics should still be recorded even after exception
            agent.workflow_duration.record.assert_called_once()

    @pytest.mark.asyncio
    async def test_observability_manager_integration(self):
        """Test integration with real ObservabilityManager"""
        from src.puffinflow.core.observability.config import ObservabilityConfig
        from src.puffinflow.core.observability.core import ObservabilityManager

        # Create real observability manager with disabled features for testing
        config = ObservabilityConfig()
        config.tracing.enabled = False
        config.metrics.enabled = False
        config.alerting.enabled = False
        config.events.enabled = False

        observability = ObservabilityManager(config)
        await observability.initialize()

        agent = ObservableAgent(
            name="integration_test_agent",
            observability=observability
        )

        # Should work without errors even with disabled observability
        with patch.object(agent.__class__.__bases__[0], 'run', new_callable=AsyncMock):
            await agent.run()

        await observability.shutdown()

    def test_memory_usage_considerations(self, mock_observability):
        """Test that agent doesn't hold unnecessary references"""
        import weakref

        agent = ObservableAgent(
            name="test_agent",
            observability=mock_observability
        )

        # Create weak reference to test cleanup
        weak_ref = weakref.ref(agent)

        # Basic operations shouldn't create circular references
        context = agent._create_context({})

        # Clean up
        del agent
        del context

        # This test mainly ensures no obvious memory leaks
        # In practice, you'd use memory profiling tools


# Additional test fixtures and utilities

@pytest.fixture
def sample_workflow_config():
    """Sample workflow configuration for testing"""
    return {
        "name": "test_workflow",
        "states": ["init", "process", "finalize"],
        "transitions": {
            "init": "process",
            "process": "finalize"
        }
    }


class MockAgent:
    """Mock agent for testing inheritance behavior"""

    def __init__(self, name, **kwargs):
        self.name = name
        self.shared_state = {}
        self.states = {}

    async def run(self, timeout=None):
        """Mock run implementation"""
        pass

    async def run_state(self, state_name):
        """Mock run_state implementation"""
        if state_name in self.states:
            await self.states[state_name](self.shared_state)


@pytest.mark.integration
class TestObservableAgentIntegration:
    """Integration tests for ObservableAgent"""

    @pytest.mark.asyncio
    async def test_full_workflow_observability(self):
        """Test complete workflow with real observability components"""
        from src.puffinflow.core.observability.config import ObservabilityConfig

        # Create minimal working config
        config = ObservabilityConfig()
        config.tracing.console_enabled = True
        config.metrics.enabled = True
        config.alerting.enabled = False
        config.events.enabled = True

        from src.puffinflow.core.observability.core import setup_observability

        observability = await setup_observability(config)

        # Define simple workflow
        async def init_state(context):
            context.set_variable("initialized", True)
            return "init_complete"

        async def process_state(context):
            context.set_variable("processed", True)
            return "process_complete"

        # Create observable agent
        agent = ObservableAgent(
            name="integration_agent",
            observability=observability,
            workflow_id="integration_test"
        )

        agent.states = {
            "init": init_state,
            "process": process_state
        }
        agent.shared_state = {}

        # Execute states
        await agent.run_state("init")
        await agent.run_state("process")

        # Verify state execution
        assert agent.shared_state.get("initialized") is True
        assert agent.shared_state.get("processed") is True

        # Cleanup
        await observability.shutdown()

    @pytest.mark.asyncio
    async def test_error_propagation_with_observability(self):
        """Test that errors propagate correctly through observability layers"""
        from src.puffinflow.core.observability.core import ObservabilityManager
        from src.puffinflow.core.observability.config import ObservabilityConfig

        config = ObservabilityConfig()
        observability = ObservabilityManager(config)
        await observability.initialize()

        async def error_state(context):
            raise ValueError("Intentional test error")

        agent = ObservableAgent(
            name="error_test_agent",
            observability=observability
        )
        agent.states = {"error_state": error_state}
        agent.shared_state = {}

        # Error should still propagate
        with pytest.raises(ValueError, match="Intentional test error"):
            await agent.run_state("error_state")

        await observability.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])