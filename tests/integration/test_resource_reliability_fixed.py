"""Integration tests for resource management and reliability patterns.

Tests the interaction between resource allocation, circuit breakers, bulkheads,
and other reliability patterns working together.
"""

import pytest
import asyncio
import time
import random
from unittest.mock import AsyncMock, Mock, patch

from puffinflow import (
    Agent, Context, ResourcePool, ResourceRequirements,
    CircuitBreaker, CircuitBreakerConfig, Bulkhead, BulkheadConfig,
    ResourceLeakDetector, state
)


class ResourceIntensiveAgent(Agent):
    """Agent that requires significant resources."""
    
    def __init__(self, name: str, cpu_req: float = 1.5, memory_req: float = 256.0):
        super().__init__(name)
        self.set_variable("cpu_req", cpu_req)
        self.set_variable("memory_req", memory_req)
        self.add_state('start', self.start)
        self.initial_state = 'start'

    @state(cpu=1.5, memory=256.0)
    async def start(self, context: Context):
        cpu_req = self.get_variable("cpu_req", 1.5)
        memory_req = self.get_variable("memory_req", 256.0)
        
        # Simulate resource-intensive work
        start_time = time.time()
        await asyncio.sleep(0.2)
        duration = time.time() - start_time
        
        context.set_output("cpu_used", cpu_req)
        context.set_output("memory_used", memory_req)
        context.set_output("execution_time", duration)
        context.set_metric("resource_efficiency", 0.85)
        
        return None


class UnreliableAgent(Agent):
    """Agent that fails intermittently."""
    
    def __init__(self, name: str, failure_rate: float = 0.3):
        super().__init__(name)
        self.set_variable("failure_rate", failure_rate)
        self.add_state('start', self.start)
        self.initial_state = 'start'

    @state(cpu=0.5, memory=128.0)
    async def start(self, context: Context):
        failure_rate = self.get_variable("failure_rate", 0.3)
        
        # Simulate unreliable behavior
        await asyncio.sleep(0.1)
        
        if random.random() < failure_rate:
            context.set_output("status", "failed")
            context.set_metric("reliability", 0.0)
            raise Exception("Agent failed randomly")
        else:
            context.set_output("status", "success")
            context.set_output("message", "Completed successfully")
            context.set_metric("reliability", 1.0)
            return None


class SlowAgent(Agent):
    """Agent that takes a long time to execute."""
    
    def __init__(self, name: str, delay: float = 1.0):
        super().__init__(name)
        self.set_variable("delay", delay)
        self.add_state('start', self.start)
        self.initial_state = 'start'

    @state(cpu=0.2, memory=64.0)
    async def start(self, context: Context):
        delay = self.get_variable("delay", 1.0)
        
        start_time = time.time()
        await asyncio.sleep(delay)
        duration = time.time() - start_time
        
        context.set_output("execution_time", duration)
        context.set_output("delay_requested", delay)
        context.set_metric("performance", 1.0 / duration)
        
        return None


@pytest.mark.asyncio
async def test_resource_pool_allocation():
    """Test resource pool allocation across multiple agents."""
    # Create a resource pool
    resource_pool = ResourcePool(
        total_cpu=4.0,
        total_memory=1024.0,
        total_io=100.0,
        total_network=100.0
    )
    
    # Create agent
    agent = ResourceIntensiveAgent("test-agent", cpu_req=1.5, memory_req=256.0)
    agent.resource_pool = resource_pool
    
    # Run the agent
    result = await agent.run()
    
    # Check the result
    status = result.status.name if hasattr(result.status, 'name') else str(result.status)
    assert status.upper() in ["COMPLETED", "SUCCESS"], f"Expected COMPLETED or SUCCESS, got {status}"
    assert "cpu_used" in result.outputs, f"cpu_used not in outputs: {result.outputs}"
    assert "memory_used" in result.outputs, f"memory_used not in outputs: {result.outputs}"
    assert result.outputs["cpu_used"] == 1.5, f"Expected cpu_used=1.5, got {result.outputs.get('cpu_used')}"
    assert result.outputs["memory_used"] == 256.0, f"Expected memory_used=256.0, got {result.outputs.get('memory_used')}"


@pytest.mark.asyncio
async def test_resource_contention():
    """Test behavior when resources are over-allocated."""
    # Create a small resource pool
    resource_pool = ResourcePool(
        total_cpu=3.0,
        total_memory=768.0,
        total_io=100.0,
        total_network=100.0
    )
    
    # Create agents that together may exceed available resources
    agents = [
        ResourceIntensiveAgent(f"contention-agent-{i}", cpu_req=1.5, memory_req=256.0)
        for i in range(3)
    ]
    
    # Set resource pool for all agents
    for agent in agents:
        agent.resource_pool = resource_pool
    
    # Run all agents concurrently
    tasks = [agent.run() for agent in agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Some agents should complete successfully
    successful_results = [r for r in results if not isinstance(r, Exception)]
    
    # At least some should succeed
    assert len(successful_results) >= 1, f"Expected at least 1 success, got {len(successful_results)}"
    
    # Verify successful agents used resources correctly
    for result in successful_results:
        if hasattr(result, 'outputs'):
            assert "cpu_used" in result.outputs
            assert "memory_used" in result.outputs


@pytest.mark.asyncio
async def test_circuit_breaker_integration():
    """Test circuit breaker with agents."""
    # Create circuit breaker configuration
    cb_config = CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout=0.5,
        name="cb-test"
    )
    
    circuit_breaker = CircuitBreaker(cb_config)
    
    # Create unreliable agent that always fails
    unreliable_agent = UnreliableAgent("cb-test-agent", failure_rate=1.0)
    
    # Test circuit breaker behavior
    exceptions = []
    
    # Run multiple attempts to trigger circuit breaker
    for i in range(3):
        try:
            async with circuit_breaker.protect():
                await unreliable_agent.run()
        except Exception as e:
            exceptions.append(e)
        
        await asyncio.sleep(0.1)
    
    # Should have captured exceptions
    assert len(exceptions) >= 2, f"Expected at least 2 exceptions, got {len(exceptions)}"
    
    # Circuit breaker should have tracked failures
    assert circuit_breaker._failure_count >= 2


@pytest.mark.asyncio
async def test_combined_reliability_patterns():
    """Test multiple reliability patterns working together."""
    # Create circuit breaker
    cb_config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=0.3,
        name="combined-test"
    )
    circuit_breaker = CircuitBreaker(cb_config)
    
    # Create bulkhead
    bulkhead = Bulkhead(BulkheadConfig(
        name="combined",
        max_concurrent=2,
        max_queue_size=1,
        timeout=2.0
    ))
    
    # Create resource pool
    resource_pool = ResourcePool(
        total_cpu=4.0,
        total_memory=1024.0,
        total_io=100.0,
        total_network=100.0
    )
    
    # Create agents with different characteristics
    agents = [
        ResourceIntensiveAgent("resource-heavy", cpu_req=1.5, memory_req=256.0),
        SlowAgent("slow-agent", delay=0.3),
        ResourceIntensiveAgent("resource-heavy-2", cpu_req=1.0, memory_req=128.0),
    ]
    
    # Set resource pool for resource-intensive agents
    for agent in agents:
        if isinstance(agent, ResourceIntensiveAgent):
            agent.resource_pool = resource_pool
    
    # Combined execution with protection patterns
    async def protected_execution(agent):
        async with bulkhead.isolate():
            async with circuit_breaker.protect():
                return await agent.run()
    
    # Execute agents with combined protection
    tasks = [protected_execution(agent) for agent in agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Analyze results
    successful_results = [r for r in results if not isinstance(r, Exception)]
    
    # Some agents should succeed
    assert len(successful_results) >= 1, f"Expected at least 1 success, got {len(successful_results)}"
    
    # Verify successful executions
    for result in successful_results:
        if hasattr(result, 'status'):
            status = result.status.name if hasattr(result.status, 'name') else str(result.status)
            assert status.upper() in ["COMPLETED", "SUCCESS"]


@pytest.mark.asyncio
async def test_metrics_collection_with_failures():
    """Test that metrics are collected even when agents fail."""
    
    class MetricsAgent(Agent):
        def __init__(self, name: str, should_fail: bool = False):
            super().__init__(name)
            self.set_variable("should_fail", should_fail)
            self.add_state('start', self.start)
            self.initial_state = 'start'
        
        @state(cpu=1.0, memory=256.0)
        async def start(self, context: Context):
            # Always set some metrics
            context.set_metric("execution_start", time.time())
            context.set_metric("agent_name", self.name)
            
            await asyncio.sleep(0.1)
            
            context.set_metric("execution_duration", 0.1)
            context.set_output("metrics_collected", True)
            
            if self.get_variable("should_fail", False):
                context.set_metric("failure_reason", "intentional")
                raise RuntimeError("Intentional failure for metrics test")
            
            context.set_metric("success", True)
            return None
    
    # Create agents - some that succeed, some that fail
    agents = [
        MetricsAgent("metrics-success-1", should_fail=False),
        MetricsAgent("metrics-failure-1", should_fail=True),
        MetricsAgent("metrics-success-2", should_fail=False),
    ]
    
    # Run agents and collect results
    results = []
    for agent in agents:
        result = await agent.run()
        results.append(result)
    
    # Verify results
    assert len(results) == 3
    
    # Count based on actual execution status
    success_count = 0
    failure_count = 0
    
    for result in results:
        status = result.status.name if hasattr(result.status, 'name') else str(result.status)
        if status.upper() in ["COMPLETED", "SUCCESS"]:
            success_count += 1
            assert "metrics_collected" in result.outputs
            assert result.get_output("metrics_collected") is True
        else:
            failure_count += 1
    
    assert success_count == 2, f"Expected 2 successes, got {success_count}"
    assert failure_count == 1, f"Expected 1 failure, got {failure_count}"