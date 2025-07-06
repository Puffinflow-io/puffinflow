"""
Comprehensive unit tests for ResourcePool class.

Tests cover:
- Basic allocation and release
- Quotas and quota enforcement
- Preemption functionality
- Concurrent access and thread safety
- Timeout handling
- Statistics tracking
- Error conditions and edge cases
- Resource validation
- Historical tracking
"""

import asyncio
import contextlib
import time
from unittest.mock import patch

import pytest

# Import the classes under test
from puffinflow.core.resources.pool import (
    ResourceOverflowError,
    ResourcePool,
    ResourceQuotaExceededError,
)
from puffinflow.core.resources.requirements import (
    ResourceRequirements,
    ResourceType,
)


class TestResourcePoolBasic:
    """Test basic ResourcePool functionality."""

    def test_initialization_default(self):
        """Test ResourcePool initialization with default values."""
        pool = ResourcePool()

        # Check default resource limits
        assert pool.resources[ResourceType.CPU] == 4.0
        assert pool.resources[ResourceType.MEMORY] == 1024.0
        assert pool.resources[ResourceType.IO] == 100.0
        assert pool.resources[ResourceType.NETWORK] == 100.0
        assert pool.resources[ResourceType.GPU] == 0.0

        # Check available resources equal total initially
        assert pool.available == pool.resources

        # Check empty allocations initially
        assert pool._allocations == {}
        assert pool._waiting_states == set()
        assert pool._preempted_states == set()

    def test_initialization_custom(self):
        """Test ResourcePool initialization with custom values."""
        pool = ResourcePool(
            total_cpu=8.0,
            total_memory=2048.0,
            total_io=200.0,
            total_network=150.0,
            total_gpu=2.0,
            enable_preemption=True,
            enable_quotas=True,
        )

        assert pool.resources[ResourceType.CPU] == 8.0
        assert pool.resources[ResourceType.MEMORY] == 2048.0
        assert pool.resources[ResourceType.IO] == 200.0
        assert pool.resources[ResourceType.NETWORK] == 150.0
        assert pool.resources[ResourceType.GPU] == 2.0
        assert pool.enable_preemption is True
        assert pool.enable_quotas is True


class TestResourceAllocation:
    """Test resource allocation and release."""

    @pytest.fixture
    def pool(self):
        """Create a test resource pool."""
        return ResourcePool(
            total_cpu=4.0,
            total_memory=1024.0,
            total_io=100.0,
            total_network=100.0,
            total_gpu=0.0,
        )

    @pytest.fixture
    def basic_requirements(self):
        """Create basic resource requirements."""
        return ResourceRequirements(
            cpu_units=1.0,
            memory_mb=256.0,
            io_weight=10.0,
            network_weight=10.0,
            gpu_units=0.0,
        )

    @pytest.mark.asyncio
    async def test_successful_allocation(self, pool, basic_requirements):
        """Test successful resource allocation."""
        success = await pool.acquire("test_state", basic_requirements)

        assert success is True
        assert "test_state" in pool._allocations
        assert pool.available[ResourceType.CPU] == 3.0  # 4.0 - 1.0
        assert pool.available[ResourceType.MEMORY] == 768.0  # 1024.0 - 256.0
        assert pool.available[ResourceType.IO] == 90.0  # 100.0 - 10.0
        assert pool.available[ResourceType.NETWORK] == 90.0  # 100.0 - 10.0
        assert pool.available[ResourceType.GPU] == 0.0  # No GPU requested

    @pytest.mark.asyncio
    async def test_multiple_allocations(self, pool):
        """Test multiple resource allocations."""
        req1 = ResourceRequirements(cpu_units=1.0, memory_mb=256.0)
        req2 = ResourceRequirements(cpu_units=2.0, memory_mb=512.0)

        success1 = await pool.acquire("state1", req1)
        success2 = await pool.acquire("state2", req2)

        assert success1 is True
        assert success2 is True
        assert len(pool._allocations) == 2
        assert pool.available[ResourceType.CPU] == 1.0  # 4.0 - 1.0 - 2.0
        assert pool.available[ResourceType.MEMORY] == 256.0  # 1024.0 - 256.0 - 512.0

    @pytest.mark.asyncio
    async def test_allocation_exceeds_available(self, pool):
        """Test allocation that exceeds available resources."""
        # First allocation uses most resources
        req1 = ResourceRequirements(cpu_units=3.0, memory_mb=800.0)
        success1 = await pool.acquire("state1", req1, timeout=0.1)
        assert success1 is True

        # Second allocation should fail due to insufficient resources
        req2 = ResourceRequirements(cpu_units=2.0, memory_mb=300.0)
        success2 = await pool.acquire("state2", req2, timeout=0.1)
        assert success2 is False

    @pytest.mark.asyncio
    async def test_resource_release(self, pool, basic_requirements):
        """Test resource release."""
        # Allocate resources
        success = await pool.acquire("test_state", basic_requirements)
        assert success is True

        pool.available[ResourceType.CPU]
        pool.available[ResourceType.MEMORY]

        # Release resources
        await pool.release("test_state")

        assert "test_state" not in pool._allocations
        assert pool.available[ResourceType.CPU] == pool.resources[ResourceType.CPU]
        assert (
            pool.available[ResourceType.MEMORY] == pool.resources[ResourceType.MEMORY]
        )

    @pytest.mark.asyncio
    async def test_release_nonexistent_state(self, pool):
        """Test releasing resources for non-existent state."""
        # Should not raise exception
        await pool.release("nonexistent_state")
        assert len(pool._allocations) == 0


class TestResourceValidation:
    """Test resource requirement validation."""

    @pytest.fixture
    def pool(self):
        return ResourcePool()

    @pytest.mark.asyncio
    async def test_negative_requirements(self, pool):
        """Test validation of negative resource requirements."""
        invalid_req = ResourceRequirements(cpu_units=-1.0)

        with pytest.raises(ValueError, match="Negative resource requirement"):
            await pool.acquire("test_state", invalid_req)

    @pytest.mark.asyncio
    async def test_requirements_exceed_total(self, pool):
        """Test validation when requirements exceed total resources."""
        excessive_req = ResourceRequirements(cpu_units=10.0)  # Pool only has 4.0 CPU

        with pytest.raises(ResourceOverflowError, match="exceeds total available"):
            await pool.acquire("test_state", excessive_req)


class TestResourceQuotas:
    """Test resource quota functionality."""

    @pytest.fixture
    def pool_with_quotas(self):
        return ResourcePool(enable_quotas=True)

    @pytest.mark.asyncio
    async def test_set_quota(self, pool_with_quotas):
        """Test setting resource quotas."""
        await pool_with_quotas.set_quota("test_state", ResourceType.CPU, 2.0)

        assert pool_with_quotas._quotas["test_state"][ResourceType.CPU] == 2.0

    @pytest.mark.asyncio
    async def test_quota_disabled_raises_error(self):
        """Test that setting quotas raises error when disabled."""
        pool = ResourcePool(enable_quotas=False)

        with pytest.raises(RuntimeError, match="Quotas are not enabled"):
            await pool.set_quota("test_state", ResourceType.CPU, 2.0)

    @pytest.mark.asyncio
    async def test_quota_enforcement(self, pool_with_quotas):
        """Test quota enforcement during allocation."""
        await pool_with_quotas.set_quota("test_state", ResourceType.CPU, 2.0)

        # First allocation within quota should succeed
        req1 = ResourceRequirements(cpu_units=1.5)
        success1 = await pool_with_quotas.acquire("test_state", req1)
        assert success1 is True

        # Second allocation exceeding quota should fail
        req2 = ResourceRequirements(cpu_units=1.0)  # Would total 2.5 > 2.0 quota
        with pytest.raises(ResourceQuotaExceededError):
            await pool_with_quotas.acquire("test_state", req2)

    @pytest.mark.asyncio
    async def test_no_quota_set_allows_allocation(self, pool_with_quotas):
        """Test allocation when no quota is set for a state."""
        req = ResourceRequirements(cpu_units=3.0)  # Large allocation
        success = await pool_with_quotas.acquire("test_state", req)
        assert success is True


class TestPreemption:
    """Test resource preemption functionality."""

    @pytest.fixture
    def pool_with_preemption(self):
        return ResourcePool(total_cpu=4.0, total_memory=1024.0, enable_preemption=True)

    @pytest.mark.asyncio
    async def test_preemption_enabled(self, pool_with_preemption):
        """Test that preemption is properly enabled."""
        assert pool_with_preemption.enable_preemption is True

    @pytest.mark.asyncio
    async def test_preemption_frees_resources(self, pool_with_preemption):
        """Test that preemption frees up resources."""
        # First state allocates most resources
        req1 = ResourceRequirements(cpu_units=3.0, memory_mb=800.0)
        success1 = await pool_with_preemption.acquire("state1", req1)
        assert success1 is True

        # Second state tries to allocate with preemption
        req2 = ResourceRequirements(cpu_units=2.0, memory_mb=600.0)
        success2 = await pool_with_preemption.acquire(
            "state2", req2, allow_preemption=True
        )

        # Should succeed by preempting state1
        assert success2 is True
        assert "state1" in pool_with_preemption._preempted_states
        assert "state1" not in pool_with_preemption._allocations
        assert "state2" in pool_with_preemption._allocations

    @pytest.mark.asyncio
    async def test_preemption_disabled_by_default(self):
        """Test that preemption is disabled by default."""
        pool = ResourcePool()
        assert pool.enable_preemption is False

        # Fill up resources
        req1 = ResourceRequirements(cpu_units=4.0)
        await pool.acquire("state1", req1)

        # Should fail without preemption
        req2 = ResourceRequirements(cpu_units=1.0)
        success = await pool.acquire("state2", req2, timeout=0.1, allow_preemption=True)
        assert success is False


class TestConcurrency:
    """Test concurrent access and thread safety."""

    @pytest.fixture
    def pool(self):
        return ResourcePool(total_cpu=10.0, total_memory=2048.0)

    @pytest.mark.asyncio
    async def test_concurrent_allocations(self, pool):
        """Test multiple concurrent allocations."""

        async def allocate_resource(state_name: str, cpu: float):
            req = ResourceRequirements(cpu_units=cpu, memory_mb=200.0)
            return await pool.acquire(state_name, req, timeout=1.0)

        # Create multiple concurrent allocation tasks
        tasks = [
            allocate_resource(f"state{i}", 1.0)
            for i in range(8)  # 8 states requiring 1 CPU each, pool has 10
        ]

        results = await asyncio.gather(*tasks)

        # Should have successful allocations up to resource limit
        successful = sum(results)
        assert successful <= 10  # Can't exceed total CPU
        assert successful >= 8  # Should accommodate all requests
        assert len(pool._allocations) == successful

    @pytest.mark.asyncio
    async def test_concurrent_allocation_and_release(self, pool):
        """Test concurrent allocation and release operations."""

        async def alloc_and_release(state_name: str):
            req = ResourceRequirements(cpu_units=2.0, memory_mb=400.0)
            success = await pool.acquire(state_name, req, timeout=1.0)
            if success:
                await asyncio.sleep(0.1)  # Hold resources briefly
                await pool.release(state_name)
                return True
            return False

        # Run multiple alloc/release cycles concurrently
        tasks = [alloc_and_release(f"state{i}") for i in range(6)]
        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert all(results)
        assert len(pool._allocations) == 0  # All released


class TestTimeout:
    """Test timeout functionality."""

    @pytest.fixture
    def pool(self):
        return ResourcePool(total_cpu=2.0)

    @pytest.mark.asyncio
    async def test_allocation_timeout(self, pool):
        """Test allocation timeout when resources unavailable."""
        # Fill up resources
        req1 = ResourceRequirements(cpu_units=2.0)
        success1 = await pool.acquire("state1", req1)
        assert success1 is True

        # Try to allocate more with timeout
        req2 = ResourceRequirements(cpu_units=1.0)
        start_time = time.time()
        success2 = await pool.acquire("state2", req2, timeout=0.5)
        elapsed = time.time() - start_time

        assert success2 is False
        assert 0.4 <= elapsed <= 0.6  # Should timeout around 0.5 seconds

    @pytest.mark.asyncio
    async def test_allocation_without_timeout(self, pool):
        """Test allocation waiting indefinitely without timeout."""
        # Fill up resources
        req1 = ResourceRequirements(cpu_units=2.0)
        await pool.acquire("state1", req1)

        # Start allocation without timeout
        req2 = ResourceRequirements(cpu_units=1.0)
        alloc_task = asyncio.create_task(pool.acquire("state2", req2))

        # Wait a bit, should still be waiting
        await asyncio.sleep(0.1)
        assert not alloc_task.done()

        # Release resources
        await pool.release("state1")

        # Now allocation should complete
        success = await alloc_task
        assert success is True


class TestStatistics:
    """Test usage statistics tracking."""

    @pytest.fixture
    def pool(self):
        return ResourcePool()

    @pytest.mark.asyncio
    async def test_usage_stats_initialization(self, pool):
        """Test initial usage statistics."""
        stats = pool.get_usage_stats()

        for resource_type in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            assert resource_type in stats
            stat = stats[resource_type]
            assert stat.peak_usage == 0.0
            assert stat.current_usage == 0.0
            assert stat.total_allocations == 0
            assert stat.failed_allocations == 0
            assert stat.last_allocation_time is None
            assert stat.total_wait_time == 0.0

    @pytest.mark.asyncio
    async def test_stats_updated_on_allocation(self, pool):
        """Test statistics are updated on allocation."""
        req = ResourceRequirements(cpu_units=2.0, memory_mb=512.0)
        success = await pool.acquire("test_state", req)
        assert success is True

        stats = pool.get_usage_stats()

        # CPU stats should be updated
        cpu_stats = stats[ResourceType.CPU]
        assert cpu_stats.total_allocations == 1
        assert cpu_stats.current_usage == 2.0
        assert cpu_stats.peak_usage == 2.0
        assert cpu_stats.last_allocation_time is not None
        assert cpu_stats.total_wait_time >= 0.0

        # Memory stats should be updated
        memory_stats = stats[ResourceType.MEMORY]
        assert memory_stats.total_allocations == 1
        assert memory_stats.current_usage == 512.0
        assert memory_stats.peak_usage == 512.0

    @pytest.mark.asyncio
    async def test_failed_allocation_stats(self, pool):
        """Test failed allocation statistics."""
        # Fill up CPU
        req1 = ResourceRequirements(cpu_units=4.0)
        await pool.acquire("state1", req1)

        # Try to allocate more CPU (should fail)
        req2 = ResourceRequirements(cpu_units=1.0)
        success = await pool.acquire("state2", req2, timeout=0.1)
        assert success is False

        stats = pool.get_usage_stats()
        cpu_stats = stats[ResourceType.CPU]
        assert cpu_stats.failed_allocations == 1


class TestInformationMethods:
    """Test information/query methods."""

    @pytest.fixture
    def pool(self):
        return ResourcePool()

    @pytest.mark.asyncio
    async def test_get_state_allocations(self, pool):
        """Test getting current state allocations."""
        req1 = ResourceRequirements(cpu_units=1.0, memory_mb=256.0)
        req2 = ResourceRequirements(cpu_units=2.0, memory_mb=512.0)

        await pool.acquire("state1", req1)
        await pool.acquire("state2", req2)

        allocations = pool.get_state_allocations()

        assert len(allocations) == 2
        assert "state1" in allocations
        assert "state2" in allocations
        assert allocations["state1"][ResourceType.CPU] == 1.0
        assert allocations["state1"][ResourceType.MEMORY] == 256.0
        assert allocations["state2"][ResourceType.CPU] == 2.0
        assert allocations["state2"][ResourceType.MEMORY] == 512.0

    @pytest.mark.asyncio
    async def test_get_waiting_states(self, pool):
        """Test getting states waiting for resources."""
        # Fill up resources
        req1 = ResourceRequirements(cpu_units=4.0)
        await pool.acquire("state1", req1)

        # Start allocation that will wait
        req2 = ResourceRequirements(cpu_units=1.0)
        task = asyncio.create_task(pool.acquire("state2", req2))

        # Give it time to start waiting
        await asyncio.sleep(0.1)

        waiting = pool.get_waiting_states()
        assert "state2" in waiting

        # Clean up
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_get_preempted_states(self):
        """Test getting preempted states."""
        pool = ResourcePool(total_cpu=2.0, enable_preemption=True)

        # Allocate resources
        req1 = ResourceRequirements(cpu_units=2.0)
        await pool.acquire("state1", req1)

        # Preempt with new allocation
        req2 = ResourceRequirements(cpu_units=1.5)
        await pool.acquire("state2", req2, allow_preemption=True)

        preempted = pool.get_preempted_states()
        assert "state1" in preempted


class TestHistoricalTracking:
    """Test historical resource usage tracking."""

    @pytest.fixture
    def pool(self):
        return ResourcePool()

    @pytest.mark.asyncio
    async def test_usage_history_recorded(self, pool):
        """Test that usage history is recorded."""
        # Initially no history
        assert len(pool._usage_history) == 0

        # Make allocation
        req = ResourceRequirements(cpu_units=1.0, memory_mb=256.0)
        await pool.acquire("test_state", req)

        # Should have history entry
        assert len(pool._usage_history) == 1
        timestamp, available = pool._usage_history[0]
        assert isinstance(timestamp, float)
        assert timestamp > 0
        assert ResourceType.CPU in available
        assert ResourceType.MEMORY in available

    @pytest.mark.asyncio
    async def test_history_cleanup(self, pool):
        """Test cleanup of old history entries."""
        # Mock time to control history retention
        with patch("time.time") as mock_time:
            # Start at time 0
            mock_time.return_value = 0.0

            # Make allocation
            req = ResourceRequirements(cpu_units=1.0)
            await pool.acquire("state1", req)

            # Move time forward beyond retention period
            mock_time.return_value = pool._history_retention + 100

            # Make another allocation (triggers cleanup)
            await pool.acquire("state2", req)

            # Old entry should be cleaned up
            assert len(pool._usage_history) == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def pool(self):
        return ResourcePool()

    @pytest.mark.asyncio
    async def test_zero_resource_allocation(self, pool):
        """Test allocation with zero resources."""
        req = ResourceRequirements(cpu_units=0.0, memory_mb=0.0)
        success = await pool.acquire("test_state", req)
        assert success is True

        # Should still track the allocation
        assert "test_state" in pool._allocations

        # Available resources shouldn't change
        assert pool.available[ResourceType.CPU] == pool.resources[ResourceType.CPU]
        assert (
            pool.available[ResourceType.MEMORY] == pool.resources[ResourceType.MEMORY]
        )

    @pytest.mark.asyncio
    async def test_exact_resource_match(self, pool):
        """Test allocation that exactly matches available resources."""
        req = ResourceRequirements(
            cpu_units=4.0,  # Exactly the total CPU
            memory_mb=1024.0,  # Exactly the total memory
        )
        success = await pool.acquire("test_state", req)
        assert success is True

        # All resources should be allocated
        assert pool.available[ResourceType.CPU] == 0.0
        assert pool.available[ResourceType.MEMORY] == 0.0

    @pytest.mark.asyncio
    async def test_double_allocation_same_state(self, pool):
        """Test double allocation for the same state."""
        req1 = ResourceRequirements(cpu_units=1.0, memory_mb=256.0)
        req2 = ResourceRequirements(cpu_units=2.0, memory_mb=512.0)

        # First allocation
        success1 = await pool.acquire("test_state", req1)
        assert success1 is True

        # Second allocation (should replace first)
        success2 = await pool.acquire("test_state", req2)
        assert success2 is True

        # Should have new allocation amounts
        allocations = pool.get_state_allocations()
        assert allocations["test_state"][ResourceType.CPU] == 2.0
        assert allocations["test_state"][ResourceType.MEMORY] == 512.0

    @pytest.mark.asyncio
    async def test_partial_resource_types(self, pool):
        """Test allocation with partial resource types."""
        # Only allocate CPU and memory, not IO/network/GPU
        req = ResourceRequirements(
            cpu_units=2.0,
            memory_mb=512.0,
            resource_types=ResourceType.CPU | ResourceType.MEMORY,
        )

        success = await pool.acquire("test_state", req)
        assert success is True

        # Should only allocate specified resource types
        allocations = pool.get_state_allocations()
        state_alloc = allocations["test_state"]

        assert ResourceType.CPU in state_alloc
        assert ResourceType.MEMORY in state_alloc
        # Other resource types may or may not be present but should be 0 if present
        for rt in [ResourceType.IO, ResourceType.NETWORK, ResourceType.GPU]:
            if rt in state_alloc:
                assert state_alloc[rt] == 0.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
