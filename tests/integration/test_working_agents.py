"""Fresh integration tests using proven working agent patterns."""

import asyncio

import pytest

from puffinflow import Agent, Context, state
from puffinflow.core.resources.pool import ResourcePool


class SimpleWorkingAgent(Agent):
    """Simple working agent with proven pattern."""

    def __init__(self, name: str):
        super().__init__(name)
        self.add_state("start", self.start)
        self.initial_state = "start"

    @state(cpu=1.0, memory=256.0)
    async def start(self, context: Context):
        await asyncio.sleep(0.1)
        context.set_output("result", "success")
        context.set_output("agent_name", self.name)
        return None


class ResourceWorkingAgent(Agent):
    """Resource-intensive agent with proven pattern."""

    def __init__(self, name: str, cpu_req: float = 1.5, memory_req: float = 256.0):
        super().__init__(name)
        self.set_variable("cpu_req", cpu_req)
        self.set_variable("memory_req", memory_req)
        self.add_state("start", self.start)
        self.initial_state = "start"

    @state(cpu=1.5, memory=256.0)
    async def start(self, context: Context):
        cpu_req = self.get_variable("cpu_req", 1.5)
        memory_req = self.get_variable("memory_req", 256.0)

        await asyncio.sleep(0.1)

        context.set_output("cpu_used", cpu_req)
        context.set_output("memory_used", memory_req)
        context.set_output("success", True)
        return None


@pytest.mark.asyncio
async def test_simple_working_agent():
    """Test simple agent execution."""
    agent = SimpleWorkingAgent("simple-test")
    result = await agent.run()

    # Check status
    status = (
        result.status.name if hasattr(result.status, "name") else str(result.status)
    )
    assert status.upper() in ["COMPLETED", "SUCCESS"], f"Expected success, got {status}"

    # Check outputs
    assert result.get_output("result") == "success"
    assert result.get_output("agent_name") == "simple-test"


@pytest.mark.asyncio
async def test_resource_working_agent():
    """Test resource-intensive agent execution."""
    # Create resource pool
    resource_pool = ResourcePool(
        total_cpu=4.0, total_memory=1024.0, total_io=100.0, total_network=100.0
    )

    # Create agent
    agent = ResourceWorkingAgent("resource-test", cpu_req=1.5, memory_req=256.0)
    agent.resource_pool = resource_pool

    # Run agent
    result = await agent.run()

    # Check status
    status = (
        result.status.name if hasattr(result.status, "name") else str(result.status)
    )
    assert status.upper() in ["COMPLETED", "SUCCESS"], f"Expected success, got {status}"

    # Check outputs
    assert result.get_output("cpu_used") == 1.5
    assert result.get_output("memory_used") == 256.0
    assert result.get_output("success") is True


@pytest.mark.asyncio
async def test_multiple_working_agents():
    """Test multiple agents execution."""
    agents = [SimpleWorkingAgent(f"agent-{i}") for i in range(3)]

    # Run all agents
    results = await asyncio.gather(*[agent.run() for agent in agents])

    # Check all results
    assert len(results) == 3

    for i, result in enumerate(results):
        status = (
            result.status.name if hasattr(result.status, "name") else str(result.status)
        )
        assert status.upper() in [
            "COMPLETED",
            "SUCCESS",
        ], f"Agent {i} failed with status {status}"
        assert result.get_output("result") == "success"
        assert result.get_output("agent_name") == f"agent-{i}"


@pytest.mark.asyncio
async def test_resource_contention_working():
    """Test resource contention with working agents."""
    # Create limited resource pool
    resource_pool = ResourcePool(
        total_cpu=3.0, total_memory=768.0, total_io=100.0, total_network=100.0
    )

    # Create multiple resource agents
    agents = [
        ResourceWorkingAgent(f"resource-agent-{i}", cpu_req=1.0, memory_req=200.0)
        for i in range(4)
    ]

    # Set resource pool for all agents
    for agent in agents:
        agent.resource_pool = resource_pool

    # Run agents concurrently
    results = await asyncio.gather(
        *[agent.run() for agent in agents], return_exceptions=True
    )

    # Check results
    successful_results = [r for r in results if not isinstance(r, Exception)]

    # At least some should succeed
    assert (
        len(successful_results) >= 2
    ), f"Expected at least 2 successes, got {len(successful_results)}"

    # Check successful results
    for result in successful_results:
        if hasattr(result, "status"):
            status = (
                result.status.name
                if hasattr(result.status, "name")
                else str(result.status)
            )
            assert status.upper() in [
                "COMPLETED",
                "SUCCESS",
            ], f"Expected success, got {status}"
            assert result.get_output("success") is True
