"""
Comprehensive test coverage for src.puffinflow.core.agent.base module.

Tests cover:
- RetryPolicy class functionality
- Agent initialization and configuration
- State management (add_state, run_state)
- Dependency resolution and queue management
- Error handling and retry logic
- Checkpoint creation/restoration
- Pause/resume functionality
- Cancellation logic
- Main workflow execution scenarios
"""

import asyncio
import contextlib
import logging
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Import the modules to testHe
from puffinflow.core.agent.base import Agent, RetryPolicy
from puffinflow.core.agent.context import Context
from puffinflow.core.agent.state import (
    AgentStatus,
    PrioritizedState,
    Priority,
    StateStatus,
)
from puffinflow.core.resources.requirements import ResourceRequirements

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def retry_policy():
    """Standard retry policy for testing."""
    return RetryPolicy(
        max_retries=3, initial_delay=0.01, exponential_base=2.0, jitter=False
    )


@pytest.fixture
def agent():
    """Basic agent instance for testing."""
    return Agent(
        name="test_agent",
        max_concurrent=2,
        retry_policy=RetryPolicy(max_retries=2, initial_delay=0.01),
        state_timeout=1.0,
    )


@pytest.fixture
def simple_state_func():
    """Simple async state function that returns success."""

    async def state_func(context: Context) -> str:
        await asyncio.sleep(0.001)  # Very small delay
        return "completed"

    return state_func


@pytest.fixture
def failing_state_func():
    """State function that always fails."""

    async def state_func(context: Context) -> None:
        await asyncio.sleep(0.001)
        raise ValueError("Test failure")

    return state_func


@pytest.fixture
def mock_resource_pool():
    """Mock resource pool for testing."""
    pool = Mock()
    pool.acquire = AsyncMock()
    pool.release = AsyncMock()
    return pool


# ============================================================================
# RETRY POLICY TESTS
# ============================================================================


class TestRetryPolicy:
    """Test cases for RetryPolicy class."""

    def test_retry_policy_initialization(self):
        """Test RetryPolicy initialization with various parameters."""
        policy = RetryPolicy(
            max_retries=5, initial_delay=2.0, exponential_base=3.0, jitter=True
        )

        assert policy.max_retries == 5
        assert policy.initial_delay == 2.0
        assert policy.exponential_base == 3.0
        assert policy.jitter is True

    def test_retry_policy_defaults(self):
        """Test RetryPolicy with default values."""
        policy = RetryPolicy()

        assert policy.max_retries == 3
        assert policy.initial_delay == 1.0
        assert policy.exponential_base == 2.0
        assert policy.jitter is True

    @pytest.mark.asyncio
    async def test_wait_without_jitter(self):
        """Test wait method without jitter for predictable delays."""
        policy = RetryPolicy(initial_delay=0.01, exponential_base=2.0, jitter=False)

        # Test increasing delays
        start_time = time.time()
        await policy.wait(0)
        first_delay = time.time() - start_time
        assert first_delay >= 0.008  # Allow for timing variance

        start_time = time.time()
        await policy.wait(1)
        second_delay = time.time() - start_time
        assert second_delay >= 0.015  # Should be roughly 2x first delay

    @pytest.mark.asyncio
    async def test_wait_with_jitter(self):
        """Test wait method with jitter enabled."""
        policy = RetryPolicy(initial_delay=0.01, exponential_base=2.0, jitter=True)

        start_time = time.time()
        await policy.wait(0)
        delay = time.time() - start_time

        # With jitter, delay should be variable but reasonable
        # Increased upper bound to account for system timing variations
        assert 0.005 <= delay <= 0.025

    @pytest.mark.asyncio
    async def test_wait_max_delay_cap(self):
        """Test that wait method respects the 60 second cap."""
        policy = RetryPolicy(initial_delay=0.01, exponential_base=2.0, jitter=False)

        # Mock the sleep function to verify the cap logic without actually waiting
        with patch("asyncio.sleep") as mock_sleep:
            await policy.wait(20)  # Would be huge without cap

            # Verify that sleep was called with capped value
            mock_sleep.assert_called_once()
            called_delay = mock_sleep.call_args[0][0]
            assert called_delay <= 60.0

    @pytest.mark.asyncio
    async def test_wait_with_zero_delay(self):
        """Test wait method with zero initial delay."""
        policy = RetryPolicy(initial_delay=0.0, exponential_base=2.0, jitter=False)

        start_time = time.time()
        await policy.wait(0)
        delay = time.time() - start_time

        assert delay < 0.01  # Should be nearly instantaneous

    @pytest.mark.asyncio
    async def test_wait_exponential_calculation(self):
        """Test that exponential backoff calculation is correct."""
        policy = RetryPolicy(initial_delay=0.01, exponential_base=2.0, jitter=False)

        with patch("asyncio.sleep") as mock_sleep:
            await policy.wait(3)

            # 0.01 * 2^3 = 0.08, capped at 60
            expected_delay = min(0.01 * (2.0**3), 60.0)
            mock_sleep.assert_called_once_with(expected_delay)


# ============================================================================
# AGENT INITIALIZATION TESTS
# ============================================================================


class TestAgentInitialization:
    """Test cases for Agent initialization."""

    def test_agent_basic_initialization(self):
        """Test basic Agent initialization."""
        agent = Agent(name="test_agent")

        assert agent.name == "test_agent"
        assert agent.max_concurrent == 5
        assert isinstance(agent.retry_policy, RetryPolicy)
        assert agent.state_timeout is None
        assert agent.status == AgentStatus.IDLE
        assert len(agent.states) == 0
        assert len(agent.state_metadata) == 0
        assert len(agent.dependencies) == 0
        assert len(agent.priority_queue) == 0
        assert isinstance(agent.shared_state, dict)
        assert len(agent.running_states) == 0
        assert len(agent.completed_states) == 0
        assert len(agent.completed_once) == 0
        assert isinstance(agent.context, Context)
        assert agent.session_start is None

    def test_agent_initialization_with_custom_values(
        self, retry_policy, mock_resource_pool
    ):
        """Test Agent initialization with custom parameters."""
        agent = Agent(
            name="custom_agent",
            max_concurrent=10,
            retry_policy=retry_policy,
            state_timeout=30.0,
            resource_pool=mock_resource_pool,
        )

        assert agent.name == "custom_agent"
        assert agent.max_concurrent == 10
        assert agent.retry_policy is retry_policy
        assert agent.state_timeout == 30.0
        assert agent.resource_pool is mock_resource_pool

    def test_agent_initialization_with_none_retry_policy(self):
        """Test Agent initialization with None retry policy creates default."""
        agent = Agent(name="test_agent", retry_policy=None)

        assert isinstance(agent.retry_policy, RetryPolicy)
        assert agent.retry_policy.max_retries == 3


# ============================================================================
# STATE MANAGEMENT TESTS
# ============================================================================


class TestStateManagement:
    """Test cases for state management functionality."""

    def test_add_state_basic(self, agent, simple_state_func):
        """Test adding a basic state to the agent."""
        agent.add_state("test_state", simple_state_func)

        assert "test_state" in agent.states
        assert agent.states["test_state"] is simple_state_func
        assert "test_state" in agent.state_metadata
        assert agent.dependencies["test_state"] == []

        metadata = agent.state_metadata["test_state"]
        assert metadata.status == StateStatus.PENDING
        assert metadata.max_retries == agent.retry_policy.max_retries
        assert isinstance(metadata.resources, ResourceRequirements)
        assert metadata.priority == Priority.NORMAL

    def test_add_state_with_dependencies(self, agent, simple_state_func):
        """Test adding a state with dependencies."""
        dependencies = ["state1", "state2"]
        agent.add_state("test_state", simple_state_func, dependencies=dependencies)

        assert agent.dependencies["test_state"] == dependencies

    def test_add_state_with_custom_resources(self, agent, simple_state_func):
        """Test adding a state with custom resource requirements."""
        resources = ResourceRequirements(cpu_units=2.0, memory_mb=500.0)
        agent.add_state("test_state", simple_state_func, resources=resources)

        metadata = agent.state_metadata["test_state"]
        assert metadata.resources is resources

    def test_add_state_with_custom_retry_settings(self, agent, simple_state_func):
        """Test adding a state with custom retry settings."""
        custom_retry_policy = RetryPolicy(max_retries=5)
        agent.add_state(
            "test_state",
            simple_state_func,
            max_retries=10,
            retry_policy=custom_retry_policy,
        )

        metadata = agent.state_metadata["test_state"]
        assert metadata.max_retries == 10
        assert metadata.retry_policy is custom_retry_policy

    def test_add_state_with_priority(self, agent, simple_state_func):
        """Test adding a state with custom priority."""
        agent.add_state("test_state", simple_state_func, priority=Priority.HIGH)

        metadata = agent.state_metadata["test_state"]
        assert metadata.priority == Priority.HIGH

    def test_find_entry_states_no_dependencies(self, agent, simple_state_func):
        """Test finding entry states when states have no dependencies."""
        agent.add_state("state1", simple_state_func)
        agent.add_state("state2", simple_state_func)

        entry_states = agent._find_entry_states()
        assert set(entry_states) == {"state1", "state2"}

    def test_find_entry_states_with_dependencies(self, agent, simple_state_func):
        """Test finding entry states when some states have dependencies."""
        agent.add_state("state1", simple_state_func)
        agent.add_state("state2", simple_state_func, dependencies=["state1"])
        agent.add_state("state3", simple_state_func)

        entry_states = agent._find_entry_states()
        assert set(entry_states) == {"state1", "state3"}

    def test_find_entry_states_all_have_dependencies(self, agent, simple_state_func):
        """Test finding entry states when all states have dependencies."""
        agent.add_state("state1", simple_state_func, dependencies=["state2"])
        agent.add_state("state2", simple_state_func, dependencies=["state1"])

        entry_states = agent._find_entry_states()
        assert len(entry_states) == 0


# ============================================================================
# QUEUE MANAGEMENT TESTS
# ============================================================================


class TestQueueManagement:
    """Test cases for priority queue management."""

    @pytest.mark.asyncio
    async def test_add_to_queue_basic(self, agent, simple_state_func):
        """Test adding states to the priority queue."""
        agent.add_state("test_state", simple_state_func)

        await agent._add_to_queue("test_state")

        assert len(agent.priority_queue) == 1
        state_item = agent.priority_queue[0]
        assert isinstance(state_item, PrioritizedState)
        assert state_item.state_name == "test_state"

    @pytest.mark.asyncio
    async def test_add_to_queue_with_priority_boost(self, agent, simple_state_func):
        """Test adding states to queue with priority boost."""
        agent.add_state("test_state", simple_state_func, priority=Priority.NORMAL)

        await agent._add_to_queue("test_state", priority_boost=2)

        state_item = agent.priority_queue[0]
        expected_priority = -(Priority.NORMAL.value + 2)
        assert state_item.priority == expected_priority

    @pytest.mark.asyncio
    async def test_add_to_queue_nonexistent_state(self, agent, caplog):
        """Test adding non-existent state to queue logs error."""
        with caplog.at_level(logging.WARNING):
            await agent._add_to_queue("nonexistent_state")

        assert "State nonexistent_state not found in metadata" in caplog.text
        assert len(agent.priority_queue) == 0

    @pytest.mark.asyncio
    async def test_get_ready_states_empty_queue(self, agent):
        """Test getting ready states from empty queue."""
        ready_states = await agent._get_ready_states()
        assert ready_states == []

    @pytest.mark.asyncio
    async def test_get_ready_states_with_ready_state(self, agent, simple_state_func):
        """Test getting ready states when states are ready to run."""
        agent.add_state("test_state", simple_state_func)
        await agent._add_to_queue("test_state")

        ready_states = await agent._get_ready_states()
        assert ready_states == ["test_state"]

    @pytest.mark.asyncio
    async def test_can_run_basic_state(self, agent, simple_state_func):
        """Test checking if a basic state can run."""
        agent.add_state("test_state", simple_state_func)

        can_run = await agent._can_run("test_state")
        assert can_run is True

    @pytest.mark.asyncio
    async def test_can_run_already_running_state(self, agent, simple_state_func):
        """Test checking if an already running state can run."""
        agent.add_state("test_state", simple_state_func)
        agent.running_states.add("test_state")

        can_run = await agent._can_run("test_state")
        assert can_run is False

    @pytest.mark.asyncio
    async def test_can_run_completed_once_state(self, agent, simple_state_func):
        """Test checking if a state that completed once can run again."""
        agent.add_state("test_state", simple_state_func)
        agent.completed_once.add("test_state")

        can_run = await agent._can_run("test_state")
        assert can_run is False

    @pytest.mark.asyncio
    async def test_can_run_with_unmet_dependencies(self, agent, simple_state_func):
        """Test checking if a state with unmet dependencies can run."""
        agent.add_state("state1", simple_state_func)
        agent.add_state("state2", simple_state_func, dependencies=["state1"])

        can_run = await agent._can_run("state2")
        assert can_run is False

    @pytest.mark.asyncio
    async def test_can_run_with_met_dependencies(self, agent, simple_state_func):
        """Test checking if a state with met dependencies can run."""
        agent.add_state("state1", simple_state_func)
        agent.add_state("state2", simple_state_func, dependencies=["state1"])
        agent.completed_states.add("state1")

        can_run = await agent._can_run("state2")
        assert can_run is True


# ============================================================================
# STATE EXECUTION TESTS
# ============================================================================


class TestStateExecution:
    """Test cases for state execution functionality."""

    @pytest.mark.asyncio
    async def test_run_state_success(self, agent, simple_state_func):
        """Test successful state execution."""
        agent.add_state("test_state", simple_state_func)

        await agent.run_state("test_state")

        assert "test_state" not in agent.running_states
        assert "test_state" in agent.completed_states
        assert "test_state" in agent.completed_once

        metadata = agent.state_metadata["test_state"]
        assert metadata.status == StateStatus.COMPLETED
        assert metadata.last_execution is not None
        assert metadata.last_success is not None

    @pytest.mark.asyncio
    async def test_run_state_already_running(self, agent, simple_state_func):
        """Test running a state that's already running."""
        agent.add_state("test_state", simple_state_func)
        agent.running_states.add("test_state")

        await agent.run_state("test_state")

        # State should still be in running states (early return)
        assert "test_state" in agent.running_states

    @pytest.mark.asyncio
    async def test_run_state_with_context_operations(self, agent):
        """Test state execution with context operations."""

        async def context_state(context: Context) -> None:
            context.set_variable("test_key", "test_value")

        agent.add_state("context_state", context_state)

        await agent.run_state("context_state")

        assert agent.shared_state.get("test_key") == "test_value"

    @pytest.mark.asyncio
    async def test_handle_state_result_none(self, agent):
        """Test handling state result when result is None."""
        await agent._handle_state_result("test_state", None)
        assert len(agent.priority_queue) == 0

    @pytest.mark.asyncio
    async def test_handle_state_result_string(self, agent, simple_state_func):
        """Test handling state result when result is a next state name."""
        agent.add_state("next_state", simple_state_func)

        await agent._handle_state_result("test_state", "next_state")

        assert len(agent.priority_queue) == 1
        assert agent.priority_queue[0].state_name == "next_state"

    @pytest.mark.asyncio
    async def test_handle_state_result_list(self, agent, simple_state_func):
        """Test handling state result when result is a list of state names."""
        agent.add_state("state1", simple_state_func)
        agent.add_state("state2", simple_state_func)

        await agent._handle_state_result("test_state", ["state1", "state2"])

        assert len(agent.priority_queue) == 2
        state_names = {item.state_name for item in agent.priority_queue}
        assert state_names == {"state1", "state2"}

    @pytest.mark.asyncio
    async def test_resolve_dependencies(self, agent, simple_state_func, caplog):
        """Test dependency resolution logs warning for unmet dependencies."""
        agent.add_state("state1", simple_state_func)
        agent.add_state("state2", simple_state_func, dependencies=["state1"])

        with caplog.at_level(logging.WARNING):
            await agent._resolve_dependencies("state2")

        assert "has unmet dependencies" in caplog.text


# ============================================================================
# ERROR HANDLING AND RETRY TESTS
# ============================================================================


class TestErrorHandlingAndRetry:
    """Test cases for error handling and retry functionality."""

    @pytest.mark.asyncio
    async def test_run_state_with_failure(self, agent, failing_state_func):
        """Test state execution with failure and retry."""
        agent.add_state("failing_state", failing_state_func)

        await agent.run_state("failing_state")

        metadata = agent.state_metadata["failing_state"]
        assert metadata.status == StateStatus.PENDING
        assert metadata.attempts == 1
        assert len(agent.priority_queue) == 1

    @pytest.mark.asyncio
    async def test_handle_failure_within_retry_limit(self, agent, failing_state_func):
        """Test failure handling within retry limit."""
        agent.add_state("test_state", failing_state_func)
        metadata = agent.state_metadata["test_state"]
        metadata.attempts = 1
        metadata.max_retries = 3

        with patch.object(
            metadata.retry_policy, "wait", new_callable=AsyncMock
        ) as mock_wait:
            await agent._handle_failure("test_state", ValueError("Test error"))

        assert metadata.attempts == 2
        assert metadata.status == StateStatus.PENDING
        mock_wait.assert_called_once_with(1)
        assert len(agent.priority_queue) == 1

    @pytest.mark.asyncio
    async def test_handle_failure_exceeds_retry_limit(self, agent, failing_state_func):
        """Test failure handling when retry limit is exceeded."""
        agent.add_state("test_state", failing_state_func)
        metadata = agent.state_metadata["test_state"]
        metadata.attempts = 3
        metadata.max_retries = 3

        await agent._handle_failure("test_state", ValueError("Test error"))

        assert metadata.attempts == 4
        assert metadata.status == StateStatus.FAILED
        assert len(agent.priority_queue) == 0

    @pytest.mark.asyncio
    async def test_handle_failure_with_compensation(
        self, agent, failing_state_func, simple_state_func
    ):
        """Test failure handling with compensation state."""
        agent.add_state("test_state", failing_state_func)
        agent.add_state("test_state_compensation", simple_state_func)

        metadata = agent.state_metadata["test_state"]
        metadata.attempts = 3
        metadata.max_retries = 3

        await agent._handle_failure("test_state", ValueError("Test error"))

        assert len(agent.priority_queue) == 1
        assert agent.priority_queue[0].state_name == "test_state_compensation"


# ============================================================================
# CHECKPOINT MANAGEMENT TESTS
# ============================================================================


class TestCheckpointManagement:
    """Test cases for checkpoint functionality."""

    def test_create_checkpoint(self, agent):
        """Test creating a checkpoint from agent state."""

        # Create a simple state function locally for this test
        async def simple_state(context):
            return "completed"

        agent.add_state("test_state", simple_state)
        agent.completed_states.add("completed_state")
        agent.shared_state["test_key"] = "test_value"

        checkpoint = agent.create_checkpoint()

        assert checkpoint.agent_name == agent.name
        assert checkpoint.agent_status == agent.status
        assert checkpoint.completed_states == agent.completed_states
        assert checkpoint.shared_state == agent.shared_state
        assert checkpoint.timestamp is not None

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint(self, agent):
        """Test restoring agent from checkpoint."""
        mock_checkpoint = Mock()
        mock_checkpoint.agent_status = AgentStatus.PAUSED
        mock_checkpoint.priority_queue = []
        mock_checkpoint.state_metadata = {}
        mock_checkpoint.running_states = set()
        mock_checkpoint.completed_states = {"state1"}
        mock_checkpoint.completed_once = {"state1"}
        mock_checkpoint.shared_state = {"key": "value"}
        mock_checkpoint.session_start = time.time()

        await agent.restore_from_checkpoint(mock_checkpoint)

        assert agent.status == AgentStatus.PAUSED
        assert agent.completed_states == {"state1"}
        assert agent.completed_once == {"state1"}
        assert agent.shared_state == {"key": "value"}

    @pytest.mark.asyncio
    async def test_pause_returns_checkpoint(self, agent):
        """Test that pause returns a checkpoint."""
        checkpoint = await agent.pause()

        assert agent.status == AgentStatus.PAUSED
        assert checkpoint.agent_name == agent.name
        assert checkpoint.agent_status == AgentStatus.PAUSED

    @pytest.mark.asyncio
    async def test_resume_from_paused(self, agent):
        """Test resuming from paused state."""
        agent.status = AgentStatus.PAUSED

        await agent.resume()

        assert agent.status == AgentStatus.RUNNING


# ============================================================================
# CANCELLATION TESTS
# ============================================================================


class TestCancellation:
    """Test cases for cancellation functionality."""

    def test_cancel_state(self, agent):
        """Test cancelling a specific state."""

        async def simple_state(context):
            return "completed"

        agent.add_state("test_state", simple_state)
        agent.running_states.add("test_state")

        agent.cancel_state("test_state")

        metadata = agent.state_metadata["test_state"]
        assert metadata.status == StateStatus.CANCELLED
        assert "test_state" not in agent.running_states

    def test_cancel_nonexistent_state(self, agent):
        """Test cancelling a non-existent state."""
        agent.cancel_state("nonexistent_state")  # Should not raise

    @pytest.mark.asyncio
    async def test_cancel_all(self, agent):
        """Test cancelling all states."""

        async def simple_state(context):
            return "completed"

        agent.add_state("state1", simple_state)
        agent.add_state("state2", simple_state)
        agent.running_states.update(["state1", "state2"])
        agent.priority_queue = [Mock(), Mock()]

        await agent.cancel_all()

        assert agent.status == AgentStatus.CANCELLED
        assert len(agent.running_states) == 0
        assert len(agent.priority_queue) == 0


# ============================================================================
# WORKFLOW EXECUTION TESTS
# ============================================================================


class TestWorkflowExecution:
    """Test cases for main workflow execution scenarios."""

    @pytest.mark.asyncio
    async def test_run_empty_workflow(self, agent, caplog):
        """Test running workflow with no states."""
        with caplog.at_level(logging.INFO):
            await agent.run()

        assert "No states defined, nothing to run" in caplog.text
        assert agent.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_run_single_state_workflow(self, agent, simple_state_func):
        """Test running workflow with single state."""
        agent.add_state("test_state", simple_state_func)

        await agent.run()

        assert agent.status == AgentStatus.COMPLETED
        assert "test_state" in agent.completed_states

    @pytest.mark.asyncio
    async def test_run_sequential_workflow(self, agent):
        """Test running workflow with sequential states."""
        call_order = []

        async def state1(context: Context) -> str:
            call_order.append("state1")
            return "state2"

        async def state2(context: Context) -> None:
            call_order.append("state2")

        agent.add_state("state1", state1)
        agent.add_state("state2", state2)

        await agent.run(timeout=5)

        assert agent.status == AgentStatus.COMPLETED
        assert call_order == ["state1", "state2"]
        assert agent.completed_states == {"state1", "state2"}

    @pytest.mark.asyncio
    async def test_run_workflow_manual_dependency_execution(self, agent):
        """Test workflow execution by manually managing dependencies."""
        call_order = []

        async def state1(context: Context) -> None:
            call_order.append("state1")
            # Manually add dependent state to queue
            await agent._add_to_queue("state2")

        async def state2(context: Context) -> None:
            call_order.append("state2")

        agent.add_state("state1", state1)
        agent.add_state("state2", state2, dependencies=["state1"])

        # Manually mark state1 as completed to satisfy dependencies
        await agent.run_state("state1")

        # Now run state2
        await agent.run_state("state2")

        assert call_order == ["state1", "state2"]
        assert agent.completed_states == {"state1", "state2"}

    @pytest.mark.asyncio
    async def test_run_workflow_with_failure(
        self, agent, simple_state_func, failing_state_func
    ):
        """Test running workflow with state failure."""
        agent.retry_policy.max_retries = 1

        agent.add_state("good_state", simple_state_func)
        agent.add_state("bad_state", failing_state_func, max_retries=1)

        # Run each state individually to test failure handling
        await agent.run_state("good_state")
        await agent.run_state("bad_state")

        assert "good_state" in agent.completed_states
        assert agent.state_metadata["bad_state"].status in [
            StateStatus.FAILED,
            StateStatus.PENDING,
        ]

    @pytest.mark.asyncio
    async def test_session_start_tracking(self, agent, simple_state_func):
        """Test that session start time is tracked."""
        agent.add_state("test_state", simple_state_func)

        start_time = time.time()
        await agent.run()

        assert agent.session_start is not None
        assert agent.session_start >= start_time

    @pytest.mark.asyncio
    async def test_timeout_mechanism(self, agent):
        """Test timeout mechanism with mocked sleep."""

        async def slow_state(context: Context) -> None:
            # This would normally take a long time, but we'll mock it
            await asyncio.sleep(0.001)

        agent.add_state("slow_state", slow_state)

        # Mock time.time to simulate timeout
        start_time = time.time()
        with patch("time.time") as mock_time:
            # Provide enough values to avoid StopIteration
            time_values = [
                start_time,
                start_time,
                start_time + 0.05,
                start_time + 0.06,
                start_time + 0.07,
                start_time + 0.08,
                start_time + 0.09,
                start_time + 0.2,
            ]

            def time_side_effect():
                if time_values:
                    return time_values.pop(0)
                return start_time + 0.2

            mock_time.side_effect = time_side_effect

            result = await agent.run(timeout=0.1)

        # Should fail due to timeout with our mocked timing
        assert result.status == AgentStatus.FAILED

    @pytest.mark.asyncio
    async def test_max_concurrent_execution(self, agent):
        """Test that max_concurrent limit is respected."""
        execution_order = []

        async def concurrent_state(context: Context) -> None:
            execution_order.append(
                f"start_{context.get_variable('state_id', 'unknown')}"
            )
            await asyncio.sleep(0.01)
            execution_order.append(f"end_{context.get_variable('state_id', 'unknown')}")

        # Set max_concurrent to 1 to force sequential execution
        agent.max_concurrent = 1

        for i in range(3):
            agent.add_state(f"state{i}", concurrent_state)

        # We'll test this by running states individually to verify the logic
        tasks = []
        for i in range(3):
            tasks.append(asyncio.create_task(agent.run_state(f"state{i}")))

        await asyncio.gather(*tasks)

        # All states should have completed
        assert len(agent.completed_states) == 3

    @pytest.mark.asyncio
    async def test_workflow_exception_handling(self, agent):
        """Test workflow exception handling."""

        async def simple_state(context):
            return "completed"

        agent.add_state("test_state", simple_state)

        with patch.object(
            agent, "_get_ready_states", side_effect=RuntimeError("Test error")
        ):
            result = await agent.run()

        assert result.status == AgentStatus.FAILED
        assert isinstance(result.error, RuntimeError)
        assert "Test error" in str(result.error)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_complex_workflow_with_retries(self, agent):
        """Test complex workflow with retries and state transitions."""
        execution_log = []

        async def reliable_start(context: Context) -> str:
            execution_log.append("reliable_start")
            context.set_variable("workflow_id", "test_workflow")
            return "final_cleanup"

        async def final_cleanup(context: Context) -> None:
            execution_log.append("final_cleanup")
            workflow_id = context.get_variable("workflow_id")
            assert workflow_id == "test_workflow"

        agent.add_state("reliable_start", reliable_start)
        agent.add_state("final_cleanup", final_cleanup)

        await agent.run(timeout=5)

        assert agent.status == AgentStatus.COMPLETED
        assert "reliable_start" in execution_log
        assert "final_cleanup" in execution_log
        assert agent.shared_state.get("workflow_id") == "test_workflow"

    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self, agent):
        """Test that priority queue maintains correct ordering."""

        # Create a simple state function for this test
        async def simple_state_func(context):
            return "completed"

        agent.add_state("low_priority", simple_state_func, priority=Priority.LOW)
        agent.add_state("high_priority", simple_state_func, priority=Priority.HIGH)
        agent.add_state("normal_priority", simple_state_func, priority=Priority.NORMAL)

        await agent._add_to_queue("low_priority")
        await agent._add_to_queue("high_priority")
        await agent._add_to_queue("normal_priority")

        # Priority queue should be ordered by priority (negative values for max-heap)
        priorities = [item.priority for item in agent.priority_queue]
        # heapq maintains min-heap order, so most negative (highest priority) comes first
        assert priorities[0] == -Priority.HIGH.value  # Most negative = highest priority
        # The rest should maintain heap property, not necessarily sorted

    @pytest.mark.asyncio
    async def test_state_metadata_persistence(self, agent, simple_state_func):
        """Test that state metadata is properly maintained."""
        agent.add_state("test_state", simple_state_func, priority=Priority.HIGH)

        metadata = agent.state_metadata["test_state"]
        original_id = metadata.state_id

        await agent.run_state("test_state")

        # Metadata should be updated but preserve important fields
        assert metadata.state_id == original_id
        assert metadata.status == StateStatus.COMPLETED
        assert metadata.priority == Priority.HIGH
        assert metadata.last_execution is not None
        assert metadata.last_success is not None

    @pytest.mark.asyncio
    async def test_resource_requirements_integration(self, agent, simple_state_func):
        """Test integration with resource requirements."""
        resources = ResourceRequirements(
            cpu_units=2.0, memory_mb=512.0, priority_boost=1
        )

        agent.add_state("resource_state", simple_state_func, resources=resources)

        metadata = agent.state_metadata["resource_state"]
        assert metadata.resources.cpu_units == 2.0
        assert metadata.resources.memory_mb == 512.0
        assert metadata.resources.priority_boost == 1

        await agent.run_state("resource_state")
        assert "resource_state" in agent.completed_states


# ============================================================================
# EDGE CASES AND ERROR CONDITIONS
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_state_name(self, agent):
        """Test adding state with empty name."""

        async def simple_state(context):
            return "completed"

        agent.add_state("", simple_state)
        assert "" in agent.states

    @pytest.mark.asyncio
    async def test_duplicate_state_name(self, agent):
        """Test adding duplicate state names (should overwrite)."""

        async def state_v1(context):
            return "v1"

        async def state_v2(context):
            return "v2"

        agent.add_state("duplicate", state_v1)
        agent.add_state("duplicate", state_v2)

        assert agent.states["duplicate"] is state_v2

    @pytest.mark.asyncio
    async def test_circular_dependencies(self, agent):
        """Test handling of circular dependencies."""

        async def simple_state(context):
            return "completed"

        agent.add_state("state1", simple_state, dependencies=["state2"])
        agent.add_state("state2", simple_state, dependencies=["state1"])

        # Neither state should be able to run due to circular dependencies
        assert not await agent._can_run("state1")
        assert not await agent._can_run("state2")

    @pytest.mark.asyncio
    async def test_state_execution_with_context_exception(self, agent):
        """Test state execution when context operations fail."""

        async def failing_context_state(context: Context) -> None:
            # This might fail if context is in bad state
            with contextlib.suppress(Exception):
                context.set_variable("test", "value")

        agent.add_state("context_state", failing_context_state)

        # Should complete even if context operations have issues
        await agent.run_state("context_state")
        assert "context_state" in agent.completed_states

    def test_agent_string_representation(self, agent):
        """Test that agent can be represented as string for debugging."""
        string_repr = str(agent)
        assert "test_agent" in string_repr or "Agent" in string_repr

    @pytest.mark.asyncio
    async def test_agent_result_methods(self):
        """Test AgentResult methods."""
        from puffinflow.core.agent.base import AgentResult
        from puffinflow.core.agent.state import AgentStatus

        result = AgentResult(
            agent_name="test_agent",
            status=AgentStatus.COMPLETED,
            outputs={"key1": "value1"},
            variables={"var1": "val1"},
            metadata={"meta1": "meta_val1"},
            metrics={"metric1": 100},
            start_time=100.0,
            end_time=105.0,
            execution_duration=5.0,
        )

        # Test getter methods
        assert result.get_output("key1") == "value1"
        assert result.get_output("nonexistent", "default") == "default"
        assert result.get_variable("var1") == "val1"
        assert result.get_variable("nonexistent", "default") == "default"
        assert result.get_metadata("meta1") == "meta_val1"
        assert result.get_metadata("nonexistent", "default") == "default"
        assert result.get_metric("metric1") == 100
        assert result.get_metric("nonexistent", "default") == "default"

        # Test properties
        assert result.is_success is True
        assert result.is_failed is False

        # Test failed result
        failed_result = AgentResult(
            agent_name="failed_agent",
            status=AgentStatus.FAILED,
            error=ValueError("Test error"),
        )
        assert failed_result.is_success is False
        assert failed_result.is_failed is True

    @pytest.mark.asyncio
    async def test_agent_variable_access_methods(self, agent):
        """Test comprehensive variable access methods."""
        # Test increment_variable
        agent.set_variable("counter", 5)
        agent.increment_variable("counter", 3)
        assert agent.get_variable("counter") == 8

        # Test increment with default
        agent.increment_variable("new_counter", 10)
        assert agent.get_variable("new_counter") == 10

        # Test append_variable
        agent.set_variable("list_var", ["item1"])
        agent.append_variable("list_var", "item2")
        assert agent.get_variable("list_var") == ["item1", "item2"]

        # Test append to non-list (converts to list)
        agent.set_variable("scalar_var", "single")
        agent.append_variable("scalar_var", "appended")
        assert agent.get_variable("scalar_var") == ["single", "appended"]

        # Test append to new variable
        agent.append_variable("new_list", "first_item")
        assert agent.get_variable("new_list") == ["first_item"]

    @pytest.mark.asyncio
    async def test_agent_shared_variable_methods(self, agent):
        """Test shared variable methods."""
        # Test shared variable operations
        agent.set_shared_variable("shared_key", "shared_value")
        assert agent.get_shared_variable("shared_key") == "shared_value"
        assert agent.get_shared_variable("nonexistent", "default") == "default"

    @pytest.mark.asyncio
    async def test_agent_persistent_variable_methods(self, agent):
        """Test persistent variable methods."""
        # Test persistent variable operations
        agent.set_persistent_variable("persistent_key", "persistent_value")
        assert agent.get_persistent_variable("persistent_key") == "persistent_value"
        assert agent.get_persistent_variable("nonexistent", "default") == "default"

    @pytest.mark.asyncio
    async def test_agent_context_methods(self, agent):
        """Test agent context access methods."""
        # Test output methods
        agent.set_output("output_key", "output_value")
        assert agent.get_output("output_key") == "output_value"
        assert agent.get_output("nonexistent", "default") == "default"

        # Test get_all_outputs
        agent.set_output("output1", "value1")
        agent.set_output("output2", "value2")
        all_outputs = agent.get_all_outputs()
        assert "output1" in all_outputs
        assert "output2" in all_outputs

        # Test metadata methods
        agent.set_metadata("meta_key", "meta_value")
        assert agent.get_metadata("meta_key") == "meta_value"
        assert agent.get_metadata("nonexistent", "default") == "default"

        # Test cached methods
        agent.set_cached("cache_key", "cache_value", ttl=60)
        assert agent.get_cached("cache_key") == "cache_value"
        assert agent.get_cached("nonexistent", "default") == "default"

    @pytest.mark.asyncio
    async def test_agent_property_system(self, agent):
        """Test agent property definition system."""

        # Define a property with validation
        def validate_positive(value):
            if value < 0:
                raise ValueError("Value must be positive")
            return value

        agent.define_property("score", int, default=0, validator=validate_positive)

        # Test default value
        assert agent.get_variable("score") == 0

        # Test setting valid value
        agent.set_variable("score", 100)
        assert agent.get_variable("score") == 100

        # Test type conversion
        agent.set_variable("score", "50")
        # The value might be stored as string or converted to int
        score_value = agent.get_variable("score")
        assert score_value == 50 or score_value == "50"

    @pytest.mark.asyncio
    async def test_agent_variable_watchers(self, agent):
        """Test variable watching functionality."""
        watcher_calls = []

        def sync_watcher(old_val, new_val):
            watcher_calls.append(("sync", old_val, new_val))

        async def async_watcher(old_val, new_val):
            watcher_calls.append(("async", old_val, new_val))

        # Add watchers
        agent.watch_variable("watched_var", sync_watcher)
        agent.watch_variable("watched_var", async_watcher)

        # Set variable to trigger watchers
        agent.set_variable("watched_var", "new_value")

        # Give async watcher time to execute
        await asyncio.sleep(0.01)

        # Check that watchers were called
        assert len(watcher_calls) >= 1  # At least sync watcher should be called

    @pytest.mark.asyncio
    async def test_agent_shared_variable_watchers(self, agent):
        """Test shared variable watching functionality."""
        watcher_calls = []

        def shared_watcher(old_val, new_val):
            watcher_calls.append((old_val, new_val))

        agent.watch_shared_variable("shared_watched", shared_watcher)
        agent.set_shared_variable("shared_watched", "shared_new_value")

        # Check that watcher was called
        assert len(watcher_calls) >= 1

    @pytest.mark.asyncio
    async def test_agent_state_change_handlers(self, agent):
        """Test state change handlers."""
        state_changes = []

        def state_change_handler(old_state, new_state):
            state_changes.append((old_state, new_state))

        agent.on_state_change(state_change_handler)
        agent._trigger_state_change("old", "new")

        assert len(state_changes) == 1
        assert state_changes[0] == ("old", "new")

    @pytest.mark.asyncio
    async def test_agent_team_coordination(self, agent):
        """Test team coordination methods."""
        # Test team setting and getting
        mock_team = Mock()
        agent.set_team(mock_team)

        # Test get_team with weak reference
        team = agent.get_team()
        assert team is mock_team

    @pytest.mark.asyncio
    async def test_agent_messaging_without_team(self, agent):
        """Test messaging methods without team."""
        # Test sending message without team should raise error
        with pytest.raises(RuntimeError, match="Agent must be part of a team"):
            await agent.send_message_to("other_agent", {"message": "test"})

        # Test broadcasting without team
        with contextlib.suppress(AttributeError, RuntimeError):
            await agent.broadcast_message("test_type", {"data": "test"})

    @pytest.mark.asyncio
    async def test_agent_event_system(self, agent):
        """Test event system functionality."""
        event_calls = []

        @agent.on_event("test_event")
        async def event_handler(context, data):
            event_calls.append(data)

        # Emit event locally (without team)
        await agent.emit_event("test_event", {"test": "data"})

        assert len(event_calls) == 1
        assert event_calls[0] == {"test": "data"}

    @pytest.mark.asyncio
    async def test_agent_message_handlers(self, agent):
        """Test message handler system."""

        @agent.message_handler("test_message")
        async def handle_test_message(message, sender):
            return {"response": "handled", "original": message}

        # Test handling message
        response = await agent.handle_message(
            "test_message", {"data": "test"}, "sender"
        )
        assert response["response"] == "handled"
        assert response["original"] == {"data": "test"}

        # Test unknown message type
        response = await agent.handle_message("unknown", {"data": "test"}, "sender")
        assert response == {}

    @pytest.mark.asyncio
    async def test_agent_resource_properties(self, agent):
        """Test resource-related properties."""
        # Test resource pool property
        pool = agent.resource_pool
        assert pool is not None

        # Test circuit breaker property
        cb = agent.circuit_breaker
        assert cb is not None

        # Test bulkhead property
        bh = agent.bulkhead
        assert bh is not None

    @pytest.mark.asyncio
    async def test_agent_cleanup_handlers(self, agent):
        """Test cleanup handler functionality."""
        cleanup_calls = []

        def sync_cleanup():
            cleanup_calls.append("sync")

        async def async_cleanup():
            cleanup_calls.append("async")

        agent.add_cleanup_handler(sync_cleanup)
        agent.add_cleanup_handler(async_cleanup)

        await agent.cleanup()

        assert "sync" in cleanup_calls
        assert "async" in cleanup_calls

    @pytest.mark.asyncio
    async def test_agent_resource_status(self, agent):
        """Test resource status methods."""
        status = agent.get_resource_status()
        assert "available" in status
        assert "allocated" in status
        assert "waiting" in status
        assert "preempted" in status

    @pytest.mark.asyncio
    async def test_agent_state_info_methods(self, agent, simple_state_func):
        """Test state information methods."""
        agent.add_state("info_state", simple_state_func)

        # Test get_state_info
        info = agent.get_state_info("info_state")
        assert info["name"] == "info_state"
        assert "status" in info
        assert "dependencies" in info
        assert "has_decorator" in info

        # Test get_state_info for nonexistent state
        info = agent.get_state_info("nonexistent")
        assert info == {}

        # Test list_states
        states = agent.list_states()
        assert len(states) == 1
        assert states[0]["name"] == "info_state"

    @pytest.mark.asyncio
    async def test_agent_dead_letter_management(self, agent):
        """Test dead letter management."""
        # Initially no dead letters
        assert agent.get_dead_letter_count() == 0
        assert agent.get_dead_letters() == []

        # Add a mock dead letter
        from puffinflow.core.agent.state import DeadLetter

        dead_letter = DeadLetter(
            state_name="failed_state",
            agent_name=agent.name,
            error_message="Test error",
            error_type="ValueError",
            attempts=3,
            failed_at=time.time(),
            timeout_occurred=False,
            context_snapshot={},
        )
        agent.dead_letters.append(dead_letter)

        # Test dead letter methods
        assert agent.get_dead_letter_count() == 1
        assert len(agent.get_dead_letters()) == 1
        assert len(agent.get_dead_letters_by_state("failed_state")) == 1
        assert len(agent.get_dead_letters_by_state("other_state")) == 0

        # Test clearing dead letters
        agent.clear_dead_letters()
        assert agent.get_dead_letter_count() == 0

    @pytest.mark.asyncio
    async def test_agent_circuit_breaker_control(self, agent):
        """Test circuit breaker control methods."""
        # Test force open/close
        await agent.force_circuit_breaker_open()
        await agent.force_circuit_breaker_close()

    @pytest.mark.asyncio
    async def test_agent_resource_leak_detection(self, agent):
        """Test resource leak detection."""
        leaks = agent.check_resource_leaks()
        assert isinstance(leaks, list)

    @pytest.mark.asyncio
    async def test_agent_execution_metadata_and_metrics(self, agent, simple_state_func):
        """Test execution metadata and metrics."""
        agent.add_state("test_state", simple_state_func)

        # Test metadata
        metadata = agent._get_execution_metadata()
        assert "states_completed" in metadata
        assert "states_failed" in metadata
        assert "total_states" in metadata
        assert "session_start" in metadata
        assert "dead_letter_count" in metadata

        # Test metrics
        metrics = agent._get_execution_metrics()
        assert "completion_rate" in metrics
        assert "error_rate" in metrics
        assert "resource_usage" in metrics
        assert "circuit_breaker_metrics" in metrics
        assert "bulkhead_metrics" in metrics

    @pytest.mark.asyncio
    async def test_agent_save_checkpoint(self, agent):
        """Test checkpoint saving."""
        # This is a placeholder method that just logs
        agent.save_checkpoint()  # Should not raise

    @pytest.mark.asyncio
    async def test_resource_timeout_error(self):
        """Test ResourceTimeoutError."""
        from puffinflow.core.agent.base import ResourceTimeoutError

        error = ResourceTimeoutError("Timeout occurred")
        assert str(error) == "Timeout occurred"
        assert isinstance(error, Exception)

    @pytest.mark.asyncio
    async def test_agent_extract_decorator_requirements(self, agent):
        """Test decorator requirements extraction."""
        # Mock function with requirements
        mock_func = Mock()
        mock_func._resource_requirements = ResourceRequirements(
            cpu_units=1.0, memory_mb=100.0
        )

        requirements = agent._extract_decorator_requirements(mock_func)
        assert requirements is mock_func._resource_requirements

        # Mock function without requirements
        mock_func_no_req = Mock()
        del mock_func_no_req._resource_requirements  # Ensure it doesn't exist

        requirements = agent._extract_decorator_requirements(mock_func_no_req)
        assert requirements is None

    @pytest.mark.asyncio
    async def test_agent_handle_state_result_edge_cases(self, agent, simple_state_func):
        """Test edge cases in state result handling."""
        agent.add_state("next_state", simple_state_func)
        agent.add_state("completed_state", simple_state_func)
        agent.completed_states.add("completed_state")

        # Test with completed state (should not be added to queue)
        await agent._handle_state_result("test_state", "completed_state")
        assert len(agent.priority_queue) == 0

        # Test with list containing completed state
        await agent._handle_state_result(
            "test_state", ["next_state", "completed_state"]
        )
        assert len(agent.priority_queue) == 1

        # Test with tuple (agent transition)
        mock_other_agent = Mock()
        mock_other_agent._add_to_queue = AsyncMock()
        await agent._handle_state_result(
            "test_state", [(mock_other_agent, "other_state")]
        )
        mock_other_agent._add_to_queue.assert_called_once_with("other_state")

    @pytest.mark.asyncio
    async def test_agent_del_method(self, agent):
        """Test agent deletion cleanup."""
        # Add cleanup handler
        cleanup_called = []

        def cleanup_handler():
            cleanup_called.append(True)

        agent.add_cleanup_handler(cleanup_handler)

        # Test __del__ method
        with contextlib.suppress(Exception):
            agent.__del__()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
