"""
Comprehensive unit tests for src.puffinflow.core.resources.allocation module
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from puffinflow.core.resources.allocation import (
    AllocationMetrics,
    AllocationRequest,
    AllocationResult,
    AllocationStrategy,
    BestFitAllocator,
    FairShareAllocator,
    FirstFitAllocator,
    PriorityAllocator,
    WeightedAllocator,
    WorstFitAllocator,
    create_allocator,
    get_resource_amount,
)
from puffinflow.core.resources.requirements import (
    ResourceRequirements,
    ResourceType,
)


class TestAllocationStrategy:
    """Test AllocationStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert AllocationStrategy.FIRST_FIT.value == "first_fit"
        assert AllocationStrategy.BEST_FIT.value == "best_fit"
        assert AllocationStrategy.WORST_FIT.value == "worst_fit"
        assert AllocationStrategy.PRIORITY.value == "priority"
        assert AllocationStrategy.FAIR_SHARE.value == "fair_share"
        assert AllocationStrategy.WEIGHTED.value == "weighted"


class TestAllocationRequest:
    """Test AllocationRequest dataclass."""

    def test_creation(self):
        """Test creating an allocation request."""
        requirements = ResourceRequirements(cpu_units=2.0, memory_mb=512.0)
        request = AllocationRequest(
            request_id="test-1",
            requester_id="agent-1",
            requirements=requirements,
            priority=5,
        )

        assert request.request_id == "test-1"
        assert request.requester_id == "agent-1"
        assert request.requirements == requirements
        assert request.priority == 5
        assert request.weight == 1.0
        assert isinstance(request.timestamp, datetime)
        assert request.deadline is None

    def test_ordering(self):
        """Test priority-based ordering."""
        req1 = AllocationRequest("1", "agent-1", ResourceRequirements(), priority=1)
        req2 = AllocationRequest("2", "agent-2", ResourceRequirements(), priority=5)
        req3 = AllocationRequest("3", "agent-3", ResourceRequirements(), priority=3)

        # Higher priority should be "less than" (for heap)
        assert req2 < req1  # priority 5 < priority 1
        assert req3 < req1  # priority 3 < priority 1
        assert req2 < req3  # priority 5 < priority 3

    def test_with_deadline(self):
        """Test request with deadline."""
        deadline = datetime.utcnow() + timedelta(minutes=30)
        request = AllocationRequest(
            "test-1", "agent-1", ResourceRequirements(), deadline=deadline
        )

        assert request.deadline == deadline

    def test_request_defaults(self):
        """Test request default values."""
        requirements = ResourceRequirements()
        request = AllocationRequest(
            request_id="req_123", requester_id="agent_1", requirements=requirements
        )

        assert request.priority == 0
        assert request.weight == 1.0
        assert request.metadata == {}
        assert request.deadline is None

    def test_request_comparison(self):
        """Test request comparison for priority queue."""
        requirements = ResourceRequirements()

        req1 = AllocationRequest("req1", "agent1", requirements, priority=1)
        req2 = AllocationRequest("req2", "agent2", requirements, priority=2)

        # Higher priority should be "less than" for min-heap behavior
        assert req2 < req1  # req2 has higher priority


class TestAllocationResult:
    """Test AllocationResult dataclass."""

    def test_successful_result(self):
        """Test successful allocation result."""
        allocated = {ResourceType.CPU: 2.0, ResourceType.MEMORY: 512.0}
        result = AllocationResult(
            request_id="test-1", success=True, allocated=allocated, allocation_time=0.5
        )

        assert result.request_id == "test-1"
        assert result.success is True
        assert result.allocated == allocated
        assert result.reason is None
        assert result.allocation_time == 0.5

    def test_failed_result(self):
        """Test failed allocation result."""
        result = AllocationResult(
            request_id="test-1", success=False, reason="Insufficient resources"
        )

        assert result.request_id == "test-1"
        assert result.success is False
        assert result.allocated == {}
        assert result.reason == "Insufficient resources"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        allocated = {ResourceType.CPU: 2.0, ResourceType.MEMORY: 512.0}
        result = AllocationResult(
            request_id="test-1", success=True, allocated=allocated, allocation_time=0.5
        )

        result_dict = result.to_dict()

        assert result_dict["request_id"] == "test-1"
        assert result_dict["success"] is True
        assert result_dict["allocated"]["CPU"] == 2.0
        assert result_dict["allocated"]["MEMORY"] == 512.0
        assert result_dict["allocation_time"] == 0.5
        assert "timestamp" in result_dict

    def test_result_to_dict(self):
        """Test result conversion to dictionary."""
        allocated = {ResourceType.CPU: 2.0}
        result = AllocationResult(
            request_id="req_123", success=True, allocated=allocated, allocation_time=0.5
        )

        result_dict = result.to_dict()

        assert result_dict["request_id"] == "req_123"
        assert result_dict["success"] is True
        assert result_dict["allocated"]["CPU"] == 2.0
        assert result_dict["allocation_time"] == 0.5


class TestAllocationMetrics:
    """Test AllocationMetrics class."""

    def test_initial_state(self):
        """Test initial metrics state."""
        metrics = AllocationMetrics()

        assert metrics.total_requests == 0
        assert metrics.successful_allocations == 0
        assert metrics.failed_allocations == 0
        assert metrics.total_allocation_time == 0.0
        assert len(metrics.resource_utilization) == 0
        assert len(metrics.queue_lengths) == 0
        assert len(metrics.wait_times) == 0

    def test_record_successful_allocation(self):
        """Test recording successful allocation."""
        metrics = AllocationMetrics()
        allocated = {ResourceType.CPU: 2.0, ResourceType.MEMORY: 512.0}
        result = AllocationResult(
            request_id="test-1", success=True, allocated=allocated, allocation_time=0.5
        )

        metrics.record_allocation(result, wait_time=1.0)

        assert metrics.total_requests == 1
        assert metrics.successful_allocations == 1
        assert metrics.failed_allocations == 0
        assert metrics.total_allocation_time == 0.5
        assert metrics.resource_utilization[ResourceType.CPU] == 2.0
        assert metrics.resource_utilization[ResourceType.MEMORY] == 512.0
        assert metrics.wait_times == [1.0]

    def test_record_failed_allocation(self):
        """Test recording failed allocation."""
        metrics = AllocationMetrics()
        result = AllocationResult(
            request_id="test-1", success=False, reason="Insufficient resources"
        )

        metrics.record_allocation(result)

        assert metrics.total_requests == 1
        assert metrics.successful_allocations == 0
        assert metrics.failed_allocations == 1
        assert metrics.total_allocation_time == 0.0

    def test_get_stats(self):
        """Test getting statistics."""
        metrics = AllocationMetrics()

        # Record some allocations
        for i in range(10):
            success = i < 8  # 80% success rate
            allocated = {ResourceType.CPU: 1.0} if success else {}
            result = AllocationResult(
                request_id=f"test-{i}",
                success=success,
                allocated=allocated,
                allocation_time=0.1 if success else None,
            )
            metrics.record_allocation(result, wait_time=0.5)

        stats = metrics.get_stats()

        assert stats["total_requests"] == 10
        assert stats["successful_allocations"] == 8
        assert stats["failed_allocations"] == 2
        assert stats["success_rate"] == 0.8
        # Use approximate equality for floating point comparison
        assert abs(stats["avg_allocation_time"] - 0.1) < 1e-10
        assert stats["avg_wait_time"] == 0.5
        assert stats["resource_utilization"][ResourceType.CPU] == 8.0

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = AllocationMetrics()
        assert metrics.total_requests == 0
        assert metrics.successful_allocations == 0
        assert metrics.failed_allocations == 0
        assert metrics.total_allocation_time == 0.0

    def test_get_allocation_stats(self):
        """Test getting allocation statistics."""
        metrics = AllocationMetrics()
        # Record some allocations
        success_result = AllocationResult(
            "req1", True, {ResourceType.CPU: 1.0}, allocation_time=0.3
        )
        fail_result = AllocationResult("req2", False)

        metrics.record_allocation(success_result, wait_time=0.5)
        metrics.record_allocation(fail_result)

        stats = metrics.get_stats()

        assert stats["total_requests"] == 2
        assert stats["successful_allocations"] == 1
        assert stats["failed_allocations"] == 1
        assert stats["success_rate"] == 0.5
        assert stats["avg_allocation_time"] == 0.3
        assert stats["avg_wait_time"] == 0.5


class MockResourcePool:
    """Mock resource pool for testing."""

    def __init__(self, resources=None, available=None):
        # Initialize all resource types to avoid KeyError
        default_resources = {
            ResourceType.CPU: 8.0,
            ResourceType.MEMORY: 1024.0,
            ResourceType.IO: 100.0,
            ResourceType.NETWORK: 100.0,
            ResourceType.GPU: 2.0,
        }
        self.resources = resources or default_resources
        self.available = available or self.resources.copy()

        # Ensure all resource types are present in available
        for resource_type in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            if resource_type not in self.available:
                self.available[resource_type] = self.resources.get(resource_type, 0.0)

        self._allocations = {}

    async def acquire(
        self,
        state_name: str,
        requirements: ResourceRequirements,
        timeout=None,
        allow_preemption=False,
    ):
        """Mock acquire method."""
        # Check if resources are available using the fixed helper function
        for resource_type in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            if resource_type not in requirements.resource_types:
                continue

            required = get_resource_amount(requirements, resource_type)

            if self.available.get(resource_type, 0.0) < required:
                return False

        # Allocate resources
        allocated = {}
        for resource_type in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            if resource_type not in requirements.resource_types:
                continue

            required = get_resource_amount(requirements, resource_type)
            if required > 0:
                self.available[resource_type] -= required
                allocated[resource_type] = required

        if allocated:  # Only store if we actually allocated something
            self._allocations[state_name] = allocated
        return True

    async def release(self, state_name: str):
        """Mock release method."""
        if state_name in self._allocations:
            for resource_type, amount in self._allocations[state_name].items():
                self.available[resource_type] += amount
            del self._allocations[state_name]


class TestFirstFitAllocator:
    """Test FirstFitAllocator."""

    @pytest.fixture
    def allocator(self):
        """Create allocator with mock resource pool."""
        resource_pool = MockResourcePool()
        return FirstFitAllocator(resource_pool)

    @pytest.mark.asyncio
    async def test_successful_allocation(self, allocator):
        """Test successful resource allocation."""
        requirements = ResourceRequirements(cpu_units=2.0, memory_mb=256.0)
        request = AllocationRequest("test-1", "agent-1", requirements)

        result = await allocator.allocate(request)

        assert result.success is True
        assert result.request_id == "test-1"
        assert ResourceType.CPU in result.allocated
        assert ResourceType.MEMORY in result.allocated
        assert result.allocated[ResourceType.CPU] == 2.0
        assert result.allocated[ResourceType.MEMORY] == 256.0
        assert result.allocation_time is not None

    @pytest.mark.asyncio
    async def test_failed_allocation(self, allocator):
        """Test failed allocation due to insufficient resources."""
        requirements = ResourceRequirements(cpu_units=20.0)  # More than available
        request = AllocationRequest("test-1", "agent-1", requirements)

        result = await allocator.allocate(request)

        assert result.success is False
        assert result.request_id == "test-1"
        assert result.reason == "Insufficient resources"
        assert result.allocation_time is not None

    def test_get_allocation_order(self, allocator):
        """Test FIFO ordering."""
        time1 = datetime.utcnow()
        time2 = time1 + timedelta(seconds=1)
        time3 = time1 + timedelta(seconds=2)

        req1 = AllocationRequest("1", "agent-1", ResourceRequirements())
        req1.timestamp = time2
        req2 = AllocationRequest("2", "agent-2", ResourceRequirements())
        req2.timestamp = time1
        req3 = AllocationRequest("3", "agent-3", ResourceRequirements())
        req3.timestamp = time3

        requests = [req1, req2, req3]
        ordered = allocator.get_allocation_order(requests)

        # Should be ordered by timestamp (FIFO)
        assert ordered[0].request_id == "2"  # earliest
        assert ordered[1].request_id == "1"
        assert ordered[2].request_id == "3"  # latest

    def test_can_allocate(self, allocator):
        """Test can_allocate method."""
        # Should be able to allocate within limits
        requirements1 = ResourceRequirements(cpu_units=4.0, memory_mb=512.0)
        assert allocator.can_allocate(requirements1) is True

        # Should not be able to allocate beyond limits
        requirements2 = ResourceRequirements(cpu_units=20.0)
        assert allocator.can_allocate(requirements2) is False

    def test_allocator_initialization(self):
        """Test allocator initialization."""
        pool = MockResourcePool()
        allocator = FirstFitAllocator(pool)
        assert allocator.resource_pool == pool
        assert isinstance(allocator.metrics, AllocationMetrics)

    @pytest.mark.asyncio
    async def test_successful_allocation_with_mock(self):
        """Test successful allocation with mock."""
        pool = Mock()
        pool.acquire = AsyncMock(return_value=True)
        allocator = FirstFitAllocator(pool)

        requirements = ResourceRequirements(cpu_units=2.0, memory_mb=1024.0)
        request = AllocationRequest("req_123", "agent_1", requirements)

        with patch("time.time", side_effect=[0.0, 0.5]):  # Mock timing
            result = await allocator.allocate(request)

        assert result.success is True
        assert result.request_id == "req_123"
        assert result.allocation_time == 0.5
        pool.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_allocation_with_mock(self):
        """Test failed allocation with mock."""
        pool = Mock()
        pool.acquire = AsyncMock(return_value=False)
        allocator = FirstFitAllocator(pool)

        requirements = ResourceRequirements(cpu_units=100.0)
        request = AllocationRequest("req_123", "agent_1", requirements)

        result = await allocator.allocate(request)

        assert result.success is False
        assert result.reason == "Insufficient resources"

    def test_get_allocation_order_fifo(self):
        """Test allocation order (FIFO)."""
        pool = MockResourcePool()
        allocator = FirstFitAllocator(pool)
        requirements = ResourceRequirements()

        req1 = AllocationRequest("req1", "agent1", requirements)
        req2 = AllocationRequest("req2", "agent2", requirements)
        req3 = AllocationRequest("req3", "agent3", requirements)

        # Manually set timestamps to control order
        req1.timestamp = datetime(2023, 1, 1, 10, 0, 0)
        req2.timestamp = datetime(2023, 1, 1, 10, 0, 1)
        req3.timestamp = datetime(2023, 1, 1, 10, 0, 2)

        requests = [req3, req1, req2]  # Out of order
        ordered = allocator.get_allocation_order(requests)

        assert ordered == [req1, req2, req3]  # Should be in timestamp order


class TestBestFitAllocator:
    """Test BestFitAllocator."""

    @pytest.fixture
    def allocator(self):
        """Create allocator with mock resource pool."""
        resource_pool = MockResourcePool()
        return BestFitAllocator(resource_pool)

    @pytest.mark.asyncio
    async def test_allocation(self, allocator):
        """Test basic allocation."""
        requirements = ResourceRequirements(cpu_units=2.0, memory_mb=256.0)
        request = AllocationRequest("test-1", "agent-1", requirements)

        result = await allocator.allocate(request)

        assert result.success is True
        assert result.request_id == "test-1"
        assert result.allocated[ResourceType.CPU] == 2.0
        assert result.allocated[ResourceType.MEMORY] == 256.0

    def test_calculate_waste(self, allocator):
        """Test waste calculation."""
        requirements = ResourceRequirements(cpu_units=2.0, memory_mb=256.0)
        waste = allocator._calculate_waste(requirements)

        # CPU: 8.0 - 2.0 = 6.0
        # MEMORY: 1024.0 - 256.0 = 768.0
        # IO: 100.0 - 1.0 = 99.0 (io_weight default is 1.0)
        # NETWORK: 100.0 - 1.0 = 99.0 (network_weight default is 1.0)
        # GPU: 2.0 - 0.0 = 2.0 (gpu_units default is 0.0)
        expected_waste = 6.0 + 768.0 + 99.0 + 99.0 + 2.0  # = 974.0
        assert waste == expected_waste

    def test_get_allocation_order(self, allocator):
        """Test ordering by waste (best fit first)."""
        # Small request (less waste)
        req1 = AllocationRequest(
            "1", "agent-1", ResourceRequirements(cpu_units=1.0, memory_mb=100.0)
        )
        # Large request (more waste)
        req2 = AllocationRequest(
            "2", "agent-2", ResourceRequirements(cpu_units=4.0, memory_mb=500.0)
        )

        requests = [req2, req1]  # Large first
        ordered = allocator.get_allocation_order(requests)

        # req1 waste: (8-1) + (1024-100) + (100-1) + (100-1) + (2-0) = 1031
        # req2 waste: (8-4) + (1024-500) + (100-1) + (100-1) + (2-0) = 728
        # req2 has less waste, so should come first
        assert ordered[0].request_id == "2"  # Less waste (better fit)
        assert ordered[1].request_id == "1"  # More waste


class TestWorstFitAllocator:
    """Test WorstFitAllocator."""

    @pytest.fixture
    def allocator(self):
        """Create allocator with mock resource pool."""
        resource_pool = MockResourcePool()
        return WorstFitAllocator(resource_pool)

    def test_calculate_remaining(self, allocator):
        """Test remaining space calculation."""
        requirements = ResourceRequirements(cpu_units=2.0, memory_mb=256.0)
        remaining = allocator._calculate_remaining(requirements)

        expected_remaining = (
            (8.0 - 2.0) + (1024.0 - 256.0) + (100.0 - 1.0) + (100.0 - 1.0) + (2.0 - 0.0)
        )
        # = 6.0 + 768.0 + 99.0 + 99.0 + 2.0 = 974.0
        assert remaining == expected_remaining

    def test_get_allocation_order(self, allocator):
        """Test ordering by remaining space (worst fit first)."""
        # Small request (more remaining space)
        req1 = AllocationRequest(
            "1", "agent-1", ResourceRequirements(cpu_units=1.0, memory_mb=100.0)
        )
        # Large request (less remaining space)
        req2 = AllocationRequest(
            "2", "agent-2", ResourceRequirements(cpu_units=4.0, memory_mb=500.0)
        )

        requests = [req2, req1]
        ordered = allocator.get_allocation_order(requests)

        # req1 remaining: 1031 (more remaining)
        # req2 remaining: 728 (less remaining)
        assert ordered[0].request_id == "1"  # More remaining
        assert ordered[1].request_id == "2"  # Less remaining


class TestPriorityAllocator:
    """Test PriorityAllocator."""

    @pytest.fixture
    def allocator(self):
        """Create allocator with mock resource pool."""
        resource_pool = MockResourcePool()
        return PriorityAllocator(resource_pool)

    @pytest.mark.asyncio
    async def test_priority_allocation(self, allocator):
        """Test priority-based allocation."""
        requirements = ResourceRequirements(cpu_units=1.0, memory_mb=100.0)
        request = AllocationRequest("test-1", "agent-1", requirements, priority=5)

        result = await allocator.allocate(request)

        assert result.success is True
        assert result.request_id == "test-1"

    @pytest.mark.asyncio
    async def test_insufficient_resources(self, allocator):
        """Test handling when resources are insufficient."""
        requirements = ResourceRequirements(cpu_units=20.0)  # Too much
        request = AllocationRequest("test-1", "agent-1", requirements, priority=5)

        result = await allocator.allocate(request)

        assert result.success is False
        assert "Queued for resources" in result.reason

    def test_get_allocation_order(self, allocator):
        """Test ordering by priority."""
        req1 = AllocationRequest("1", "agent-1", ResourceRequirements(), priority=1)
        req2 = AllocationRequest("2", "agent-2", ResourceRequirements(), priority=5)
        req3 = AllocationRequest("3", "agent-3", ResourceRequirements(), priority=3)

        requests = [req1, req2, req3]
        ordered = allocator.get_allocation_order(requests)

        # Should be ordered by priority (highest first)
        assert ordered[0].request_id == "2"  # priority 5
        assert ordered[1].request_id == "3"  # priority 3
        assert ordered[2].request_id == "1"  # priority 1

    def test_allocator_initialization(self):
        """Test allocator initialization."""
        pool = MockResourcePool()
        allocator = PriorityAllocator(pool)
        assert hasattr(allocator, "_priority_queue")
        assert allocator._priority_queue == []

    def test_get_allocation_order_by_priority(self):
        """Test allocation order by priority."""
        pool = MockResourcePool()
        allocator = PriorityAllocator(pool)
        requirements = ResourceRequirements()

        req1 = AllocationRequest("req1", "agent1", requirements, priority=1)
        req2 = AllocationRequest("req2", "agent2", requirements, priority=5)
        req3 = AllocationRequest("req3", "agent3", requirements, priority=3)

        requests = [req1, req2, req3]
        ordered = allocator.get_allocation_order(requests)

        # Should be ordered by priority (highest first)
        assert ordered[0].priority == 5
        assert ordered[1].priority == 3
        assert ordered[2].priority == 1


class TestFairShareAllocator:
    """Test FairShareAllocator."""

    @pytest.fixture
    def allocator(self):
        """Create allocator with mock resource pool."""
        resource_pool = MockResourcePool()
        return FairShareAllocator(resource_pool)

    @pytest.mark.asyncio
    async def test_within_fair_share(self, allocator):
        """Test allocation within fair share."""
        requirements = ResourceRequirements(cpu_units=1.0, memory_mb=100.0)
        request = AllocationRequest("test-1", "agent-1", requirements)

        result = await allocator.allocate(request)

        assert result.success is True
        assert result.request_id == "test-1"

        # Check usage tracking
        assert allocator._usage_history["agent-1"] > 0
        assert allocator._allocation_counts["agent-1"] == 1

    @pytest.mark.asyncio
    async def test_exceeds_fair_share(self, allocator):
        """Test allocation that exceeds fair share."""
        # First allocate to establish usage
        requirements1 = ResourceRequirements(cpu_units=3.0, memory_mb=300.0)
        request1 = AllocationRequest("test-1", "agent-1", requirements1)
        result1 = await allocator.allocate(request1)
        assert result1.success is True

        # Try to allocate amount that would exceed fair share
        # Total resources = 8+1024+100+100+2 = 1234, fair share for 1 agent = 1234
        # After first allocation: agent-1 used 3+300+1+1+0 = 305
        # Try to allocate more than remaining fair share: 1234 - 305 = 929
        requirements2 = ResourceRequirements(
            cpu_units=5.0, memory_mb=1000.0
        )  # Total: 5+1000+1+1+0 = 1007
        # Total usage would be 305 + 1007 = 1312 > 1234 (exceeds fair share)
        request2 = AllocationRequest("test-2", "agent-1", requirements2)

        result = await allocator.allocate(request2)

        assert result.success is False
        assert "Exceeds fair share" in result.reason

    def test_calculate_fair_share(self, allocator):
        """Test fair share calculation."""
        # With no requesters in history
        fair_share = allocator._calculate_fair_share("agent-1")
        total_resources = sum(allocator.resource_pool.resources.values())
        assert fair_share == total_resources  # Default to 1 requester

        # Add two requesters to the history
        allocator._usage_history["agent-1"] = 100.0
        allocator._usage_history["agent-2"] = 50.0
        fair_share = allocator._calculate_fair_share("agent-1")
        # Now there are 2 requesters in the history
        assert fair_share == total_resources / 2

    def test_calculate_resource_total(self, allocator):
        """Test resource total calculation."""
        requirements = ResourceRequirements(cpu_units=2.0, memory_mb=256.0)
        total = allocator._calculate_resource_total(requirements)

        # cpu_units (2.0) + memory_mb (256.0) + io_weight (1.0) + network_weight (1.0) + gpu_units (0.0)
        expected_total = 2.0 + 256.0 + 1.0 + 1.0 + 0.0  # = 260.0
        assert total == expected_total

    def test_get_allocation_order(self, allocator):
        """Test ordering by usage history."""
        # Set up usage history
        allocator._usage_history["agent-1"] = 100.0
        allocator._usage_history["agent-2"] = 50.0

        req1 = AllocationRequest("1", "agent-1", ResourceRequirements())
        req2 = AllocationRequest("2", "agent-2", ResourceRequirements())

        requests = [req1, req2]
        ordered = allocator.get_allocation_order(requests)

        # Should order by least used first
        assert ordered[0].request_id == "2"  # agent-2 used less
        assert ordered[1].request_id == "1"  # agent-1 used more

    def test_reset_usage_history(self, allocator):
        """Test resetting usage history."""
        allocator._usage_history["agent-1"] = 100.0
        allocator._allocation_counts["agent-1"] = 5

        allocator.reset_usage_history()

        assert len(allocator._usage_history) == 0
        assert len(allocator._allocation_counts) == 0

    def test_allocator_initialization(self):
        """Test allocator initialization."""
        pool = MockResourcePool()
        pool.resources = {ResourceType.CPU: 10.0, ResourceType.MEMORY: 2048.0}
        allocator = FairShareAllocator(pool)
        assert hasattr(allocator, "_usage_history")
        assert hasattr(allocator, "_allocation_counts")

    def test_get_allocation_order_by_usage(self):
        """Test allocation order by usage history."""
        pool = MockResourcePool()
        allocator = FairShareAllocator(pool)
        requirements = ResourceRequirements()

        req1 = AllocationRequest("req1", "agent1", requirements)
        req2 = AllocationRequest("req2", "agent2", requirements)

        # Set usage history
        allocator._usage_history["agent1"] = 5.0
        allocator._usage_history["agent2"] = 2.0

        requests = [req1, req2]
        ordered = allocator.get_allocation_order(requests)

        # Agent with less usage should come first
        assert ordered[0].requester_id == "agent2"
        assert ordered[1].requester_id == "agent1"


class TestWeightedAllocator:
    """Test WeightedAllocator."""

    @pytest.fixture
    def allocator(self):
        """Create allocator with mock resource pool."""
        resource_pool = MockResourcePool()
        return WeightedAllocator(resource_pool)

    def test_set_weight(self, allocator):
        """Test setting requester weight."""
        allocator.set_weight("agent-1", 2.5)
        assert allocator._weights["agent-1"] == 2.5

    @pytest.mark.asyncio
    async def test_weighted_allocation(self, allocator):
        """Test allocation with weights."""
        allocator.set_weight("agent-1", 2.0)

        requirements = ResourceRequirements(cpu_units=1.0, memory_mb=100.0)
        request = AllocationRequest("test-1", "agent-1", requirements, priority=3)

        with patch.object(PriorityAllocator, "allocate") as mock_allocate:
            mock_allocate.return_value = AllocationResult("test-1", True)

            await allocator.allocate(request)

            # Check that PriorityAllocator was called with weighted request
            mock_allocate.assert_called_once()
            called_request = mock_allocate.call_args[0][0]
            assert called_request.priority == 6  # 3 * 2.0

    def test_get_allocation_order(self, allocator):
        """Test ordering by weighted priority."""
        allocator.set_weight("agent-1", 2.0)
        allocator.set_weight("agent-2", 0.5)

        req1 = AllocationRequest("1", "agent-1", ResourceRequirements(), priority=3)
        req2 = AllocationRequest("2", "agent-2", ResourceRequirements(), priority=5)

        requests = [req1, req2]
        ordered = allocator.get_allocation_order(requests)

        # agent-1: priority 3 * weight 2.0 = 6.0
        # agent-2: priority 5 * weight 0.5 = 2.5
        # Higher weighted priority should come first
        assert ordered[0].request_id == "1"  # Higher weighted priority
        assert ordered[1].request_id == "2"  # Lower weighted priority


class TestResourceAllocatorBase:
    """Test ResourceAllocator base class functionality."""

    @pytest.fixture
    def allocator(self):
        """Create a concrete allocator for testing."""
        resource_pool = MockResourcePool()
        return FirstFitAllocator(resource_pool)

    @pytest.mark.asyncio
    async def test_allocate_batch(self, allocator):
        """Test batch allocation."""
        requests = [
            AllocationRequest("1", "agent-1", ResourceRequirements(cpu_units=1.0)),
            AllocationRequest("2", "agent-2", ResourceRequirements(cpu_units=1.0)),
            AllocationRequest(
                "3", "agent-3", ResourceRequirements(cpu_units=20.0)
            ),  # Too much
        ]

        results = await allocator.allocate_batch(requests)

        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is True
        assert results[2].success is False  # Insufficient resources

        # Check metrics were updated
        assert allocator.metrics.total_requests == 3
        assert allocator.metrics.successful_allocations == 2
        assert allocator.metrics.failed_allocations == 1

    def test_can_allocate_sufficient_resources(self):
        """Test can_allocate with sufficient resources."""
        pool = MockResourcePool()
        pool.available = {
            ResourceType.CPU: 4.0,
            ResourceType.MEMORY: 2048.0,
            ResourceType.IO: 100.0,
            ResourceType.NETWORK: 100.0,
            ResourceType.GPU: 2.0,
        }
        allocator = FirstFitAllocator(pool)
        requirements = ResourceRequirements(cpu_units=2.0, memory_mb=1024.0)

        assert allocator.can_allocate(requirements) is True

    def test_can_allocate_insufficient_resources(self):
        """Test can_allocate with insufficient resources."""
        pool = MockResourcePool()
        pool.available = {ResourceType.CPU: 4.0, ResourceType.MEMORY: 2048.0}
        allocator = FirstFitAllocator(pool)
        requirements = ResourceRequirements(cpu_units=8.0, memory_mb=1024.0)

        assert allocator.can_allocate(requirements) is False

    @pytest.mark.asyncio
    async def test_allocate_batch_mock(self):
        """Test batch allocation with mock."""
        pool = Mock()
        pool.acquire = AsyncMock(return_value=True)
        allocator = FirstFitAllocator(pool)

        requirements = ResourceRequirements(cpu_units=1.0)
        requests = [
            AllocationRequest("req1", "agent1", requirements),
            AllocationRequest("req2", "agent2", requirements),
        ]

        with patch("time.time", return_value=0.0):
            results = await allocator.allocate_batch(requests)

        assert len(results) == 2
        assert all(result.success for result in results)
        assert allocator.metrics.total_requests == 2


class TestCreateAllocator:
    """Test allocator factory function."""

    def test_create_first_fit(self):
        """Test creating first-fit allocator."""
        resource_pool = MockResourcePool()
        allocator = create_allocator(AllocationStrategy.FIRST_FIT, resource_pool)
        assert isinstance(allocator, FirstFitAllocator)

    def test_create_best_fit(self):
        """Test creating best-fit allocator."""
        resource_pool = MockResourcePool()
        allocator = create_allocator(AllocationStrategy.BEST_FIT, resource_pool)
        assert isinstance(allocator, BestFitAllocator)

    def test_create_worst_fit(self):
        """Test creating worst-fit allocator."""
        resource_pool = MockResourcePool()
        allocator = create_allocator(AllocationStrategy.WORST_FIT, resource_pool)
        assert isinstance(allocator, WorstFitAllocator)

    def test_create_priority(self):
        """Test creating priority allocator."""
        resource_pool = MockResourcePool()
        allocator = create_allocator(AllocationStrategy.PRIORITY, resource_pool)
        assert isinstance(allocator, PriorityAllocator)

    def test_create_fair_share(self):
        """Test creating fair-share allocator."""
        resource_pool = MockResourcePool()
        allocator = create_allocator(AllocationStrategy.FAIR_SHARE, resource_pool)
        assert isinstance(allocator, FairShareAllocator)

    def test_create_weighted(self):
        """Test creating weighted allocator."""
        resource_pool = MockResourcePool()
        allocator = create_allocator(AllocationStrategy.WEIGHTED, resource_pool)
        assert isinstance(allocator, WeightedAllocator)

    def test_create_default(self):
        """Test creating with unknown strategy defaults to first-fit."""
        resource_pool = MockResourcePool()
        # Pass an invalid strategy
        allocator = create_allocator("invalid_strategy", resource_pool)
        assert isinstance(allocator, FirstFitAllocator)

    def test_create_first_fit_allocator(self):
        """Test creating first-fit allocator."""
        pool = MockResourcePool()
        allocator = create_allocator(AllocationStrategy.FIRST_FIT, pool)
        assert isinstance(allocator, FirstFitAllocator)

    def test_create_best_fit_allocator(self):
        """Test creating best-fit allocator."""
        pool = MockResourcePool()
        allocator = create_allocator(AllocationStrategy.BEST_FIT, pool)
        assert isinstance(allocator, BestFitAllocator)

    def test_create_priority_allocator(self):
        """Test creating priority allocator."""
        pool = MockResourcePool()
        allocator = create_allocator(AllocationStrategy.PRIORITY, pool)
        assert isinstance(allocator, PriorityAllocator)

    def test_create_fair_share_allocator(self):
        """Test creating fair-share allocator."""
        pool = MockResourcePool()
        allocator = create_allocator(AllocationStrategy.FAIR_SHARE, pool)
        assert isinstance(allocator, FairShareAllocator)

    def test_create_default_allocator(self):
        """Test creating allocator with unknown strategy defaults to first-fit."""
        pool = MockResourcePool()
        # Create a mock strategy that doesn't exist
        unknown_strategy = "unknown_strategy"
        allocator = create_allocator(unknown_strategy, pool)
        assert isinstance(allocator, FirstFitAllocator)


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_resource_contention(self):
        """Test multiple allocators competing for resources."""
        # Use default resource pool but limit CPU availability
        resource_pool = MockResourcePool()
        # Set CPU to exactly 4.0 to test contention
        resource_pool.available[ResourceType.CPU] = 4.0
        resource_pool.resources[ResourceType.CPU] = 4.0

        allocator = FirstFitAllocator(resource_pool)

        # Create competing requests
        requests = [
            AllocationRequest("1", "agent-1", ResourceRequirements(cpu_units=2.0)),
            AllocationRequest("2", "agent-2", ResourceRequirements(cpu_units=2.0)),
            AllocationRequest(
                "3", "agent-3", ResourceRequirements(cpu_units=2.0)
            ),  # Won't fit
        ]

        results = await allocator.allocate_batch(requests)

        # First two should succeed, third should fail
        assert results[0].success is True
        assert results[1].success is True
        assert results[2].success is False

        # Check resource pool state
        assert resource_pool.available[ResourceType.CPU] == 0.0

    @pytest.mark.asyncio
    async def test_allocation_and_release_cycle(self):
        """Test complete allocation and release cycle."""
        resource_pool = MockResourcePool()
        allocator = FirstFitAllocator(resource_pool)

        # Initial available resources
        initial_cpu = resource_pool.available[ResourceType.CPU]

        # Allocate resources
        requirements = ResourceRequirements(cpu_units=2.0)
        request = AllocationRequest("test-1", "agent-1", requirements)

        result = await allocator.allocate(request)
        assert result.success is True

        # Check resources were consumed
        assert resource_pool.available[ResourceType.CPU] == initial_cpu - 2.0

        # Release resources
        await resource_pool.release("test-1")

        # Check resources were returned
        assert resource_pool.available[ResourceType.CPU] == initial_cpu

    @pytest.mark.asyncio
    async def test_multiple_strategies_same_pool(self):
        """Test different strategies with same resource pool."""
        resource_pool = MockResourcePool()

        # Create different allocators
        first_fit = FirstFitAllocator(resource_pool)
        best_fit = BestFitAllocator(resource_pool)
        priority = PriorityAllocator(resource_pool)

        # All should be able to allocate from same pool
        requirements = ResourceRequirements(cpu_units=1.0)

        result1 = await first_fit.allocate(
            AllocationRequest("1", "agent-1", requirements)
        )
        result2 = await best_fit.allocate(
            AllocationRequest("2", "agent-2", requirements)
        )
        result3 = await priority.allocate(
            AllocationRequest("3", "agent-3", requirements, priority=5)
        )

        assert result1.success is True
        assert result2.success is True
        assert result3.success is True


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_resources(self):
        """Test with zero resource requirements."""
        resource_pool = MockResourcePool()
        allocator = FirstFitAllocator(resource_pool)

        requirements = ResourceRequirements(cpu_units=0.0, memory_mb=0.0)
        assert allocator.can_allocate(requirements) is True

    def test_empty_resource_pool(self):
        """Test with empty resource pool."""
        resource_pool = MockResourcePool(
            resources=dict.fromkeys(
                [
                    ResourceType.CPU,
                    ResourceType.MEMORY,
                    ResourceType.IO,
                    ResourceType.NETWORK,
                    ResourceType.GPU,
                ],
                0.0,
            ),
            available=dict.fromkeys(
                [
                    ResourceType.CPU,
                    ResourceType.MEMORY,
                    ResourceType.IO,
                    ResourceType.NETWORK,
                    ResourceType.GPU,
                ],
                0.0,
            ),
        )
        allocator = FirstFitAllocator(resource_pool)

        requirements = ResourceRequirements(cpu_units=1.0)
        assert allocator.can_allocate(requirements) is False

    @pytest.mark.asyncio
    async def test_concurrent_allocations(self):
        """Test concurrent allocation requests."""
        resource_pool = MockResourcePool()
        allocator = FirstFitAllocator(resource_pool)

        # Create concurrent requests
        async def allocate_request(request_id):
            requirements = ResourceRequirements(cpu_units=1.0)
            request = AllocationRequest(request_id, f"agent-{request_id}", requirements)
            return await allocator.allocate(request)

        # Run multiple allocations concurrently
        tasks = [allocate_request(f"test-{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)

        # All should succeed (we have enough resources)
        for result in results:
            assert result.success is True

    def test_invalid_resource_type(self):
        """Test handling of invalid resource types."""
        resource_pool = MockResourcePool()
        allocator = FirstFitAllocator(resource_pool)

        # Create requirements with only NONE type
        requirements = ResourceRequirements()
        requirements.resource_types = ResourceType.NONE

        # Should be able to "allocate" nothing
        assert allocator.can_allocate(requirements) is True


class TestGetResourceAmount:
    """Test the get_resource_amount helper function."""

    def test_cpu_units(self):
        """Test getting CPU units."""
        requirements = ResourceRequirements(cpu_units=2.0)
        amount = get_resource_amount(requirements, ResourceType.CPU)
        assert amount == 2.0

    def test_memory_mb(self):
        """Test getting memory amount."""
        requirements = ResourceRequirements(memory_mb=512.0)
        amount = get_resource_amount(requirements, ResourceType.MEMORY)
        assert amount == 512.0

    def test_io_weight(self):
        """Test getting IO weight."""
        requirements = ResourceRequirements(io_weight=2.5)
        amount = get_resource_amount(requirements, ResourceType.IO)
        assert amount == 2.5

    def test_network_weight(self):
        """Test getting network weight."""
        requirements = ResourceRequirements(network_weight=3.0)
        amount = get_resource_amount(requirements, ResourceType.NETWORK)
        assert amount == 3.0

    def test_gpu_units(self):
        """Test getting GPU units."""
        requirements = ResourceRequirements(gpu_units=1.0)
        amount = get_resource_amount(requirements, ResourceType.GPU)
        assert amount == 1.0

    def test_none_resource_type(self):
        """Test getting NONE resource type."""
        requirements = ResourceRequirements(cpu_units=2.0)
        amount = get_resource_amount(requirements, ResourceType.NONE)
        assert amount == 0.0

    def test_excluded_resource_type(self):
        """Test getting excluded resource type."""
        requirements = ResourceRequirements(cpu_units=2.0)
        requirements.resource_types = ResourceType.MEMORY  # Only memory
        amount = get_resource_amount(requirements, ResourceType.CPU)
        assert amount == 0.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
