# tests/conftest.py
"""
Global test configuration and fixtures for Puffinflow tests.
"""

import asyncio
import time
from collections.abc import Generator
from unittest.mock import AsyncMock, Mock

import pytest

# Try to import the modules, with fallbacks for missing dependencies
try:
    from puffinflow.core.agent.base import Agent, RetryPolicy
    from puffinflow.core.agent.context import Context
    from puffinflow.core.agent.state import (
        AgentStatus,
        PrioritizedState,
        Priority,
        StateMetadata,
        StateStatus,
    )
    from puffinflow.core.resources.pool import ResourcePool
    from puffinflow.core.resources.requirements import ResourceRequirements
except ImportError:
    # Fallback if modules don't exist yet
    Agent = Mock
    RetryPolicy = Mock
    StateStatus = Mock
    AgentStatus = Mock
    StateMetadata = Mock
    PrioritizedState = Mock
    Priority = Mock
    Context = Mock
    ResourceRequirements = Mock
    ResourcePool = Mock


@pytest.fixture(scope="function")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create a fresh event loop for each test function."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        yield loop
    finally:
        loop.close()
        # Clean up any remaining tasks
        asyncio.set_event_loop(None)


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock()
    agent.name = "test_agent"
    agent.max_concurrent = 5
    agent.retry_policy = Mock()
    agent.state_timeout = None
    agent.resource_pool = None

    # State management
    agent.states = {}
    agent.state_metadata = {}
    agent.dependencies = {}
    agent.priority_queue = []
    agent.shared_state = {}

    # Execution tracking
    agent._running_states = set()
    agent.completed_states = set()
    agent.completed_once = set()
    agent.status = Mock()
    agent.session_start = None

    # Context
    agent.context = Mock()

    return agent


@pytest.fixture
def retry_policy():
    """Create a retry policy for testing."""
    if RetryPolicy == Mock:
        policy = Mock()
        policy.max_retries = 3
        policy.initial_delay = 0.01
        policy.exponential_base = 2.0
        policy.jitter = False
        policy.wait = AsyncMock()
        return policy
    else:
        return RetryPolicy(
            max_retries=3, initial_delay=0.01, exponential_base=2.0, jitter=False
        )


@pytest.fixture
def sample_context():
    """Create a sample context for testing."""
    if Context == Mock:
        context = Mock()
        context.shared_state = {}
        context.set_variable = Mock()
        context.get_variable = Mock(return_value=None)
        context.get_variable_keys = Mock(return_value=set())
        return context
    else:
        return Context({})


@pytest.fixture
def resource_requirements():
    """Create sample resource requirements."""
    if ResourceRequirements == Mock:
        req = Mock()
        req.cpu_units = 1.0
        req.memory_mb = 100.0
        req.io_weight = 1.0
        req.network_weight = 1.0
        req.gpu_units = 0.0
        req.priority_boost = 0
        req.timeout = None
        return req
    else:
        return ResourceRequirements(
            cpu_units=1.0,
            memory_mb=100.0,
            io_weight=1.0,
            network_weight=1.0,
            gpu_units=0.0,
        )


@pytest.fixture
def mock_resource_pool():
    """Create a mock resource pool."""
    if ResourcePool == Mock:
        pool = Mock()
        pool.acquire = AsyncMock(return_value=True)
        pool.release = AsyncMock()
        pool.available = {
            "CPU": 4.0,
            "MEMORY": 1024.0,
            "IO": 100.0,
            "NETWORK": 100.0,
            "GPU": 0.0,
        }
        return pool
    else:
        return ResourcePool(
            total_cpu=4.0,
            total_memory=1024.0,
            total_io=100.0,
            total_network=100.0,
            total_gpu=0.0,
        )


@pytest.fixture
async def simple_state_func():
    """Simple async state function for testing."""

    async def state_func(context) -> str:
        await asyncio.sleep(0.001)
        return "completed"

    return state_func


@pytest.fixture
async def failing_state_func():
    """State function that always fails."""

    async def state_func(context) -> None:
        await asyncio.sleep(0.001)
        raise ValueError("Test failure")

    return state_func


@pytest.fixture
def sample_state_metadata():
    """Create sample state metadata."""
    if StateMetadata == Mock:
        metadata = Mock()
        metadata.status = Mock()
        metadata.attempts = 0
        metadata.max_retries = 3
        metadata.resources = Mock()
        metadata.dependencies = {}
        metadata.satisfied_dependencies = set()
        metadata.last_execution = None
        metadata.last_success = None
        metadata.state_id = "test-state-id"
        metadata.retry_policy = None
        metadata.priority = Mock()
        return metadata
    else:
        return StateMetadata(
            status=StateStatus.PENDING if StateStatus != Mock else Mock(),
            attempts=0,
            max_retries=3,
            resources=(
                ResourceRequirements() if ResourceRequirements != Mock else Mock()
            ),
            dependencies={},
            satisfied_dependencies=set(),
            last_execution=None,
            last_success=None,
            state_id="test-state-id",
            retry_policy=None,
            priority=Priority.NORMAL if Priority != Mock else Mock(),
        )


@pytest.fixture
def sample_agent():
    """Create a real agent instance for testing."""
    if Agent == Mock:
        # Return mock if real Agent not available
        return mock_agent()
    else:
        return Agent(
            name="test_agent",
            max_concurrent=2,
            retry_policy=RetryPolicy(max_retries=2, initial_delay=0.01),
            state_timeout=1.0,
        )


# Test markers for async tests
pytestmark = pytest.mark.asyncio


# Helper functions for tests
def create_mock_with_async_methods(**kwargs):
    """Create a mock with async methods."""
    mock = Mock(**kwargs)
    # Make commonly used methods async
    mock.run = AsyncMock()
    mock.run_state = AsyncMock()
    mock.pause = AsyncMock()
    mock.resume = AsyncMock()
    mock.cancel_all = AsyncMock()
    return mock


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment for each test."""
    # This runs before each test
    pass


@pytest.fixture
def mock_time():
    """Mock time for testing."""

    mock_time_value = 1234567890.0

    def mock_time_func():
        return mock_time_value

    # Patch time.time
    import unittest.mock

    with unittest.mock.patch("time.time", side_effect=mock_time_func):
        yield mock_time_func


# Collection of test data
@pytest.fixture
def test_data():
    """Common test data."""
    return {
        "workflow_data": {
            "workflow_id": "test-workflow-123",
            "user_id": "test-user",
            "timestamp": time.time(),
            "data": {"key": "value"},
        },
        "state_results": {
            "success": "completed",
            "failure": None,
            "next_states": ["state2", "state3"],
        },
    }


# Performance testing helpers
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {"iterations": 100, "timeout": 5.0, "warmup": 10}
