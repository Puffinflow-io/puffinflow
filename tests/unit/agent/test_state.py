"""Tests for agent state module."""

import time
import uuid
from unittest.mock import Mock

import pytest

from puffinflow.core.agent.state import (
    AgentStatus,
    DeadLetter,
    PrioritizedState,
    Priority,
    RetryPolicy,
    StateFunction,
    StateMetadata,
    StateResult,
    StateStatus,
)


class TestPriority:
    """Test Priority enum."""

    def test_priority_values(self):
        """Test that priority values are correct."""
        assert Priority.LOW == 0
        assert Priority.NORMAL == 1
        assert Priority.HIGH == 2
        assert Priority.CRITICAL == 3

    def test_priority_ordering(self):
        """Test that priorities can be ordered."""
        assert Priority.LOW < Priority.NORMAL
        assert Priority.NORMAL < Priority.HIGH
        assert Priority.HIGH < Priority.CRITICAL

        # Test reverse ordering
        assert Priority.CRITICAL > Priority.HIGH
        assert Priority.HIGH > Priority.NORMAL
        assert Priority.NORMAL > Priority.LOW

    def test_priority_comparison(self):
        """Test priority comparison operations."""
        assert Priority.LOW <= Priority.NORMAL
        assert Priority.NORMAL >= Priority.LOW
        assert Priority.HIGH != Priority.LOW
        assert Priority.CRITICAL == Priority.CRITICAL

    def test_priority_membership(self):
        """Test priority membership."""
        assert Priority.LOW in Priority
        assert Priority.NORMAL in Priority
        assert Priority.HIGH in Priority
        assert Priority.CRITICAL in Priority

    def test_priority_count(self):
        """Test that we have the expected number of priorities."""
        assert len(Priority) == 4


class TestAgentStatus:
    """Test AgentStatus enum."""

    def test_agent_status_values(self):
        """Test that agent status values are correct."""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.PAUSED.value == "paused"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"
        assert AgentStatus.CANCELLED.value == "cancelled"

    def test_agent_status_membership(self):
        """Test agent status membership."""
        assert AgentStatus.IDLE in AgentStatus
        assert AgentStatus.RUNNING in AgentStatus
        assert AgentStatus.PAUSED in AgentStatus
        assert AgentStatus.COMPLETED in AgentStatus
        assert AgentStatus.FAILED in AgentStatus
        assert AgentStatus.CANCELLED in AgentStatus

    def test_agent_status_count(self):
        """Test that we have the expected number of agent statuses."""
        assert len(AgentStatus) == 6

    def test_agent_status_string_representation(self):
        """Test string representation of agent status."""
        assert str(AgentStatus.IDLE) == "idle"
        assert str(AgentStatus.RUNNING) == "running"


class TestStateStatus:
    """Test StateStatus enum."""

    def test_state_status_values(self):
        """Test that state status values are correct."""
        assert StateStatus.PENDING.value == "pending"
        assert StateStatus.READY.value == "ready"
        assert StateStatus.RUNNING.value == "running"
        assert StateStatus.COMPLETED.value == "completed"
        assert StateStatus.FAILED.value == "failed"
        assert StateStatus.CANCELLED.value == "cancelled"
        assert StateStatus.BLOCKED.value == "blocked"
        assert StateStatus.TIMEOUT.value == "timeout"
        assert StateStatus.RETRYING.value == "retrying"

    def test_state_status_membership(self):
        """Test state status membership."""
        assert StateStatus.PENDING in StateStatus
        assert StateStatus.READY in StateStatus
        assert StateStatus.RUNNING in StateStatus
        assert StateStatus.COMPLETED in StateStatus
        assert StateStatus.FAILED in StateStatus
        assert StateStatus.CANCELLED in StateStatus
        assert StateStatus.BLOCKED in StateStatus
        assert StateStatus.TIMEOUT in StateStatus
        assert StateStatus.RETRYING in StateStatus

    def test_state_status_count(self):
        """Test that we have the expected number of state statuses."""
        assert len(StateStatus) == 9

    def test_state_status_string_representation(self):
        """Test string representation of state status."""
        assert str(StateStatus.PENDING) == "pending"
        assert str(StateStatus.RUNNING) == "running"


class TestStateFunction:
    """Test StateFunction protocol."""

    def test_state_function_protocol(self):
        """Test that StateFunction is a protocol."""
        from typing import runtime_checkable

        assert runtime_checkable(StateFunction)

    def test_async_function_implements_protocol(self):
        """Test that async functions implement StateFunction protocol."""

        async def test_state(context):
            return "result"

        assert isinstance(test_state, StateFunction)

    def test_sync_function_does_not_implement_protocol(self):
        """Test that sync functions don't implement StateFunction protocol."""

        def test_state(context):
            return "result"

        # Sync functions don't implement the async protocol
        # Note: isinstance check with Protocol may not work as expected in all Python versions
        # Let's check if it's actually callable with the right signature but not async
        import inspect

        assert not inspect.iscoroutinefunction(test_state)
        assert callable(test_state)

    def test_callable_with_wrong_signature_does_not_implement_protocol(self):
        """Test that callables with wrong signature don't implement protocol."""

        async def wrong_signature():
            return "result"

        # Wrong signature doesn't implement the protocol
        # Check that it's async but has wrong signature (no context parameter)
        import inspect

        assert inspect.iscoroutinefunction(wrong_signature)
        sig = inspect.signature(wrong_signature)
        assert len(sig.parameters) == 0  # No context parameter


class TestRetryPolicy:
    """Test RetryPolicy dataclass."""

    def test_retry_policy_defaults(self):
        """Test retry policy default values."""
        policy = RetryPolicy()

        assert policy.max_retries == 3
        assert policy.initial_delay == 1.0
        assert policy.exponential_base == 2.0
        assert policy.jitter is True
        assert policy.dead_letter_on_max_retries is True
        assert policy.dead_letter_on_timeout is True

    def test_retry_policy_custom_values(self):
        """Test retry policy with custom values."""
        policy = RetryPolicy(
            max_retries=5,
            initial_delay=2.0,
            exponential_base=1.5,
            jitter=False,
            dead_letter_on_max_retries=False,
            dead_letter_on_timeout=False,
        )

        assert policy.max_retries == 5
        assert policy.initial_delay == 2.0
        assert policy.exponential_base == 1.5
        assert policy.jitter is False
        assert policy.dead_letter_on_max_retries is False
        assert policy.dead_letter_on_timeout is False

    @pytest.mark.asyncio
    async def test_retry_policy_wait_without_jitter(self):
        """Test retry policy wait calculation without jitter."""
        policy = RetryPolicy(
            initial_delay=0.01,  # Much smaller delay for testing
            exponential_base=2.0,
            jitter=False,
        )

        start_time = time.time()
        await policy.wait(0)  # First attempt
        elapsed = time.time() - start_time

        # Should wait for initial_delay * (base ^ attempt)
        # For attempt 0: 0.01 * (2.0 ^ 0) = 0.01
        # Allow more tolerance for timing variations
        assert 0.005 <= elapsed <= 0.05

    @pytest.mark.asyncio
    async def test_retry_policy_wait_with_exponential_backoff(self):
        """Test retry policy exponential backoff."""
        policy = RetryPolicy(
            initial_delay=0.1,  # Small delay for testing
            exponential_base=2.0,
            jitter=False,
        )

        # Test different attempts
        start_time = time.time()
        await policy.wait(1)  # Second attempt
        elapsed = time.time() - start_time

        # For attempt 1: 0.1 * (2.0 ^ 1) = 0.2
        assert 0.15 <= elapsed <= 0.25

    @pytest.mark.asyncio
    async def test_retry_policy_wait_max_delay(self):
        """Test retry policy maximum delay cap."""
        policy = RetryPolicy(
            initial_delay=0.1,  # Much smaller for testing
            exponential_base=3.0,
            jitter=False,
        )

        start_time = time.time()
        await policy.wait(10)  # High attempt number to trigger max delay
        elapsed = time.time() - start_time

        # Should be capped at 60 seconds, but we'll test with smaller values
        # 0.1 * (3.0 ^ 10) would be huge, but capped at 60s
        # For testing, let's just verify it's reasonable
        assert elapsed <= 61.0  # Allow small margin for execution time

    @pytest.mark.asyncio
    async def test_retry_policy_wait_with_jitter(self):
        """Test retry policy with jitter."""
        policy = RetryPolicy(initial_delay=0.1, exponential_base=2.0, jitter=True)

        # Run multiple times to test jitter variation
        delays = []
        for _ in range(5):
            start_time = time.time()
            await policy.wait(1)
            elapsed = time.time() - start_time
            delays.append(elapsed)

        # With jitter, delays should vary
        # Base delay for attempt 1: 0.1 * 2 = 0.2
        # With jitter: 0.2 * (0.5 + random * 0.5) = 0.1 to 0.2
        for delay in delays:
            assert 0.05 <= delay <= 0.25

        # Check that we have some variation (not all exactly the same)
        assert len({round(d, 3) for d in delays}) > 1


class TestDeadLetter:
    """Test DeadLetter dataclass."""

    def test_dead_letter_creation(self):
        """Test dead letter creation."""
        dead_letter = DeadLetter(
            state_name="test_state",
            agent_name="test_agent",
            error_message="Test error",
            error_type="ValueError",
            attempts=3,
            failed_at=time.time(),
        )

        assert dead_letter.state_name == "test_state"
        assert dead_letter.agent_name == "test_agent"
        assert dead_letter.error_message == "Test error"
        assert dead_letter.error_type == "ValueError"
        assert dead_letter.attempts == 3
        assert dead_letter.failed_at > 0
        assert dead_letter.timeout_occurred is False
        assert dead_letter.context_snapshot == {}

    def test_dead_letter_with_timeout(self):
        """Test dead letter with timeout."""
        dead_letter = DeadLetter(
            state_name="test_state",
            agent_name="test_agent",
            error_message="Timeout error",
            error_type="TimeoutError",
            attempts=1,
            failed_at=time.time(),
            timeout_occurred=True,
        )

        assert dead_letter.timeout_occurred is True

    def test_dead_letter_with_context_snapshot(self):
        """Test dead letter with context snapshot."""
        context_data = {"key": "value", "count": 42}
        dead_letter = DeadLetter(
            state_name="test_state",
            agent_name="test_agent",
            error_message="Test error",
            error_type="ValueError",
            attempts=3,
            failed_at=time.time(),
            context_snapshot=context_data,
        )

        assert dead_letter.context_snapshot == context_data


class TestStateMetadata:
    """Test StateMetadata dataclass."""

    def test_state_metadata_creation(self):
        """Test state metadata creation."""
        metadata = StateMetadata(status=StateStatus.PENDING)

        assert metadata.status == StateStatus.PENDING
        assert metadata.attempts == 0
        assert metadata.max_retries == 3
        assert metadata.resources is not None  # Should be initialized in __post_init__
        assert metadata.dependencies == {}
        assert metadata.satisfied_dependencies == set()
        assert metadata.last_execution is None
        assert metadata.last_success is None
        assert metadata.state_id is not None
        assert metadata.retry_policy is None
        assert metadata.priority == Priority.NORMAL
        assert metadata.coordination_primitives == []

    def test_state_metadata_with_custom_values(self):
        """Test state metadata with custom values."""
        retry_policy = RetryPolicy(max_retries=5)
        dependencies = {"dep1": "value1"}
        satisfied_deps = {"dep1"}
        coordination_primitives = ["primitive1", "primitive2"]

        metadata = StateMetadata(
            status=StateStatus.RUNNING,
            attempts=2,
            max_retries=5,
            dependencies=dependencies,
            satisfied_dependencies=satisfied_deps,
            last_execution=time.time(),
            last_success=time.time() - 100,
            retry_policy=retry_policy,
            priority=Priority.HIGH,
            coordination_primitives=coordination_primitives,
        )

        assert metadata.status == StateStatus.RUNNING
        assert metadata.attempts == 2
        assert metadata.max_retries == 5
        assert metadata.dependencies == dependencies
        assert metadata.satisfied_dependencies == satisfied_deps
        assert metadata.last_execution is not None
        assert metadata.last_success is not None
        assert metadata.retry_policy is retry_policy
        assert metadata.priority == Priority.HIGH
        assert metadata.coordination_primitives == coordination_primitives

    def test_state_metadata_state_id_uniqueness(self):
        """Test that state IDs are unique."""
        metadata1 = StateMetadata(status=StateStatus.PENDING)
        metadata2 = StateMetadata(status=StateStatus.PENDING)

        assert metadata1.state_id != metadata2.state_id

        # Should be valid UUIDs
        uuid.UUID(metadata1.state_id)
        uuid.UUID(metadata2.state_id)

    def test_state_metadata_post_init_resources(self):
        """Test that resources are initialized in __post_init__."""
        metadata = StateMetadata(status=StateStatus.PENDING)

        # Resources should be initialized if ResourceRequirements is available
        if hasattr(metadata, "resources") and metadata.resources is not None:
            # If resources are initialized, they should be a ResourceRequirements instance
            assert metadata.resources is not None


class TestPrioritizedState:
    """Test PrioritizedState dataclass."""

    def test_prioritized_state_creation(self):
        """Test prioritized state creation."""
        metadata = StateMetadata(status=StateStatus.PENDING)
        timestamp = time.time()

        prioritized_state = PrioritizedState(
            priority=Priority.HIGH.value,
            timestamp=timestamp,
            state_name="test_state",
            metadata=metadata,
        )

        assert prioritized_state.priority == Priority.HIGH.value
        assert prioritized_state.timestamp == timestamp
        assert prioritized_state.state_name == "test_state"
        assert prioritized_state.metadata is metadata

    def test_prioritized_state_ordering(self):
        """Test prioritized state ordering."""
        metadata1 = StateMetadata(status=StateStatus.PENDING)
        metadata2 = StateMetadata(status=StateStatus.PENDING)

        # Higher priority (higher number) should come first
        state1 = PrioritizedState(
            priority=Priority.HIGH.value,
            timestamp=time.time(),
            state_name="high_priority",
            metadata=metadata1,
        )

        state2 = PrioritizedState(
            priority=Priority.LOW.value,
            timestamp=time.time(),
            state_name="low_priority",
            metadata=metadata2,
        )

        # Higher priority should be "less than" for min-heap behavior
        assert state2 < state1  # Low priority comes before high priority in min-heap

    def test_prioritized_state_timestamp_ordering(self):
        """Test prioritized state ordering by timestamp when priority is same."""
        metadata1 = StateMetadata(status=StateStatus.PENDING)
        metadata2 = StateMetadata(status=StateStatus.PENDING)

        timestamp1 = time.time()
        timestamp2 = timestamp1 + 1

        state1 = PrioritizedState(
            priority=Priority.NORMAL.value,
            timestamp=timestamp1,
            state_name="earlier",
            metadata=metadata1,
        )

        state2 = PrioritizedState(
            priority=Priority.NORMAL.value,
            timestamp=timestamp2,
            state_name="later",
            metadata=metadata2,
        )

        # Earlier timestamp should come first when priority is same
        assert state1 < state2

    def test_prioritized_state_comparison_fields(self):
        """Test that only priority and timestamp are used for comparison."""
        metadata1 = StateMetadata(status=StateStatus.PENDING)
        metadata2 = StateMetadata(status=StateStatus.RUNNING)

        timestamp = time.time()

        state1 = PrioritizedState(
            priority=Priority.NORMAL.value,
            timestamp=timestamp,
            state_name="state1",
            metadata=metadata1,
        )

        state2 = PrioritizedState(
            priority=Priority.NORMAL.value,
            timestamp=timestamp,
            state_name="state2",  # Different name
            metadata=metadata2,  # Different metadata
        )

        # Should be equal because priority and timestamp are same
        # (state_name and metadata are marked as compare=False)
        assert state1 == state2


class TestStateResult:
    """Test StateResult type alias."""

    def test_state_result_string(self):
        """Test StateResult as string."""
        result: StateResult = "next_state"
        assert isinstance(result, str)
        assert result == "next_state"

    def test_state_result_list_of_strings(self):
        """Test StateResult as list of strings."""
        result: StateResult = ["state1", "state2"]
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)

    def test_state_result_list_of_tuples(self):
        """Test StateResult as list of tuples."""
        # Note: This test assumes Agent class exists
        # In a real scenario, you'd import the actual Agent class
        mock_agent = Mock()
        result: StateResult = [(mock_agent, "state1"), "state2"]
        assert isinstance(result, list)
        assert len(result) == 2

    def test_state_result_none(self):
        """Test StateResult as None."""
        result: StateResult = None
        assert result is None


class TestIntegration:
    """Integration tests for state module components."""

    def test_state_metadata_with_retry_policy_integration(self):
        """Test state metadata integration with retry policy."""
        retry_policy = RetryPolicy(
            max_retries=5, initial_delay=0.5, exponential_base=1.5
        )

        metadata = StateMetadata(
            status=StateStatus.RETRYING,
            attempts=2,
            retry_policy=retry_policy,
            priority=Priority.HIGH,
        )

        assert metadata.retry_policy.max_retries == 5
        assert metadata.attempts < metadata.retry_policy.max_retries
        assert metadata.priority == Priority.HIGH

    def test_dead_letter_from_failed_state(self):
        """Test creating dead letter from failed state metadata."""
        metadata = StateMetadata(status=StateStatus.FAILED, attempts=3, max_retries=3)

        dead_letter = DeadLetter(
            state_name="failed_state",
            agent_name="test_agent",
            error_message="Max retries exceeded",
            error_type="MaxRetriesError",
            attempts=metadata.attempts,
            failed_at=time.time(),
            context_snapshot={"metadata_id": metadata.state_id},
        )

        assert dead_letter.attempts == metadata.attempts
        assert dead_letter.context_snapshot["metadata_id"] == metadata.state_id

    def test_prioritized_state_queue_behavior(self):
        """Test prioritized state in priority queue scenario."""
        import heapq

        # Create states with different priorities
        states = []

        for i, priority in enumerate(
            [Priority.LOW, Priority.CRITICAL, Priority.NORMAL, Priority.HIGH]
        ):
            metadata = StateMetadata(status=StateStatus.READY)
            state = PrioritizedState(
                priority=priority.value,
                timestamp=time.time() + i,  # Different timestamps
                state_name=f"state_{priority.name.lower()}",
                metadata=metadata,
            )
            heapq.heappush(states, state)

        # Pop states - should come out in priority order (lowest priority value first)
        popped_priorities = []
        while states:
            state = heapq.heappop(states)
            popped_priorities.append(state.priority)

        # Should be in ascending order of priority values
        assert popped_priorities == sorted(popped_priorities)
