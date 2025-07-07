"""Resource allocation strategies for PuffinFlow resource management.

This module provides various allocation strategies for distributing computational
resources across agent states, including first-fit, best-fit, priority-based,
and fair-share allocation algorithms.
"""

import heapq
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import structlog

from .pool import ResourcePool

# Import resource management components from the canonical source
from .requirements import (
    ResourceRequirements,
    ResourceType,  # Use the canonical mapping from requirements.py
    get_resource_amount,
)

logger = structlog.get_logger(__name__)


class AllocationStrategy(Enum):
    """Enumeration of available resource allocation strategies.

    Each strategy implements a different approach to distributing limited
    computational resources among competing agent states.
    """

    FIRST_FIT = "first_fit"  # Allocate to first available slot
    BEST_FIT = "best_fit"  # Minimize resource waste
    WORST_FIT = "worst_fit"  # Maximize remaining free space
    PRIORITY = "priority"  # Allocate based on state priority
    FAIR_SHARE = "fair_share"  # Ensure equitable resource distribution
    ROUND_ROBIN = "round_robin"  # Rotate allocations cyclically
    WEIGHTED = "weighted"  # Weight allocations by importance


@dataclass
class AllocationRequest:
    """Represents a request for computational resource allocation.

    Encapsulates all information needed to process a resource allocation
    request, including resource requirements, priority, and metadata.
    """

    request_id: str  # Unique identifier for this request
    requester_id: str  # ID of the requesting agent/state
    requirements: ResourceRequirements  # Detailed resource requirements
    priority: int = 0  # Request priority (higher = more important)
    weight: float = 1.0  # Relative importance weight
    metadata: dict[str, Any] = field(
        default_factory=dict
    )  # Additional request metadata
    timestamp: datetime = field(
        default_factory=datetime.utcnow
    )  # When request was created
    deadline: Optional[datetime] = None  # Optional deadline for allocation

    def __lt__(self, other: "AllocationRequest") -> bool:
        """Define ordering for priority queue operations.

        Higher priority values are considered "less than" for max-heap behavior.
        """
        return self.priority > other.priority


@dataclass
class AllocationResult:
    """Contains the outcome of a resource allocation attempt.

    Provides detailed information about whether allocation succeeded,
    what resources were allocated, and performance metrics.
    """

    request_id: str  # ID of the original request
    success: bool  # Whether allocation succeeded
    allocated: dict[ResourceType, float] = field(
        default_factory=dict
    )  # Resources actually allocated
    reason: Optional[str] = None  # Reason for failure (if applicable)
    timestamp: datetime = field(
        default_factory=datetime.utcnow
    )  # When allocation was processed
    allocation_time: Optional[float] = None  # Time taken to process allocation

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary format for serialization."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "allocated": {rt.name: amount for rt, amount in self.allocated.items()},
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "allocation_time": self.allocation_time,
        }


class AllocationMetrics:
    """Tracks and aggregates allocation performance metrics.

    Maintains statistics about allocation success rates, timing,
    resource utilization, and queue behavior.
    """

    def __init__(self) -> None:
        """Initialize metrics tracking."""
        self.total_requests = 0  # Total number of allocation requests
        self.successful_allocations = 0  # Number of successful allocations
        self.failed_allocations = 0  # Number of failed allocations
        self.total_allocation_time = 0.0  # Cumulative time spent on allocations
        self.resource_utilization: dict[ResourceType, float] = defaultdict(
            float
        )  # Resource usage by type
        self.queue_lengths: list[int] = []  # Historical queue length snapshots
        self.wait_times: list[float] = []  # Request wait times

    def record_allocation(
        self, result: AllocationResult, wait_time: float = 0.0
    ) -> None:
        """Record metrics for a completed allocation attempt.

        Args:
            result: The allocation result to record
            wait_time: How long the request waited in queue
        """
        self.total_requests += 1

        if result.success:
            self.successful_allocations += 1
            # Track resource utilization for successful allocations
            for rt, amount in result.allocated.items():
                self.resource_utilization[rt] += amount
        else:
            self.failed_allocations += 1

        # Record timing metrics
        if result.allocation_time:
            self.total_allocation_time += result.allocation_time

        if wait_time > 0:
            self.wait_times.append(wait_time)

    def get_stats(self) -> dict[str, Any]:
        """Calculate and return comprehensive allocation statistics."""
        success_rate = (
            self.successful_allocations / self.total_requests
            if self.total_requests > 0
            else 0
        )

        avg_allocation_time = (
            self.total_allocation_time / self.successful_allocations
            if self.successful_allocations > 0
            else 0
        )

        avg_wait_time = (
            sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0
        )

        return {
            "total_requests": self.total_requests,
            "successful_allocations": self.successful_allocations,
            "failed_allocations": self.failed_allocations,
            "success_rate": success_rate,
            "avg_allocation_time": avg_allocation_time,
            "avg_wait_time": avg_wait_time,
            "resource_utilization": dict(self.resource_utilization),
        }


class ResourceAllocator(ABC):
    """Abstract base class for all resource allocation strategies.

    Defines the common interface and shared functionality for different
    allocation algorithms. Subclasses implement specific allocation logic.
    """

    def __init__(self, resource_pool: ResourcePool):
        """Initialize allocator with a resource pool.

        Args:
            resource_pool: The pool of available computational resources
        """
        self.resource_pool = resource_pool
        self.metrics = AllocationMetrics()
        self._pending_requests: list[AllocationRequest] = []

    @abstractmethod
    async def allocate(self, request: AllocationRequest) -> AllocationResult:
        """Allocate resources for a single request.

        Args:
            request: The resource allocation request

        Returns:
            Result indicating success/failure and allocated resources
        """
        pass

    @abstractmethod
    def get_allocation_order(
        self, requests: list[AllocationRequest]
    ) -> list[AllocationRequest]:
        """Determine the order for processing multiple allocation requests.

        Args:
            requests: List of pending allocation requests

        Returns:
            Requests ordered according to the allocation strategy
        """
        pass

    async def allocate_batch(
        self, requests: list[AllocationRequest]
    ) -> list[AllocationResult]:
        """Allocate resources for multiple requests in optimal order.

        Args:
            requests: List of allocation requests to process

        Returns:
            List of allocation results in processing order
        """
        ordered_requests = self.get_allocation_order(requests)
        results = []

        for request in ordered_requests:
            result = await self.allocate(request)
            results.append(result)
            self.metrics.record_allocation(result)

        return results

    def can_allocate(self, requirements: ResourceRequirements) -> bool:
        """Check if the given resource requirements can be satisfied.

        Args:
            requirements: Resource requirements to check

        Returns:
            True if resources are available, False otherwise
        """
        # Check each resource type that is requested
        for resource_type in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            # Use bitwise AND to check if this resource type is requested
            if requirements.resource_types & resource_type:
                required = get_resource_amount(requirements, resource_type)
                available = self.resource_pool.available.get(resource_type, 0.0)

                if required > available:
                    return False

        return True


class FirstFitAllocator(ResourceAllocator):
    """First-fit allocation strategy implementation.

    Allocates resources to the first request that can be satisfied,
    without considering optimization. Simple and fast but may lead
    to resource fragmentation.
    """

    async def allocate(self, request: AllocationRequest) -> AllocationResult:
        """Allocate resources using first-fit strategy.

        Attempts to immediately satisfy the request with available resources.
        """
        start_time = time.time()

        # Attempt non-blocking resource acquisition
        success = await self.resource_pool.acquire(
            request.request_id,
            request.requirements,
            timeout=0,  # Non-blocking
        )

        if success:
            # Build dictionary of actually allocated resources
            allocated = {}
            for resource_type in [
                ResourceType.CPU,
                ResourceType.MEMORY,
                ResourceType.IO,
                ResourceType.NETWORK,
                ResourceType.GPU,
            ]:
                # Check if this resource type was requested using bitwise AND
                if request.requirements.resource_types & resource_type:
                    amount = get_resource_amount(request.requirements, resource_type)
                    if amount > 0:
                        allocated[resource_type] = amount

            return AllocationResult(
                request_id=request.request_id,
                success=True,
                allocated=allocated,
                allocation_time=time.time() - start_time,
            )
        else:
            return AllocationResult(
                request_id=request.request_id,
                success=False,
                reason="Insufficient resources",
                allocation_time=time.time() - start_time,
            )

    def get_allocation_order(
        self, requests: list[AllocationRequest]
    ) -> list[AllocationRequest]:
        """Order requests by arrival time (FIFO - First In, First Out)."""
        return sorted(requests, key=lambda r: r.timestamp)


class BestFitAllocator(ResourceAllocator):
    """Best-fit allocation strategy implementation.

    Chooses allocations that minimize resource waste by finding the
    allocation that leaves the smallest amount of unused resources.
    More complex than first-fit but can improve resource utilization.
    """

    async def allocate(self, request: AllocationRequest) -> AllocationResult:
        """Allocate resources using best-fit strategy."""
        start_time = time.time()

        # Calculate potential resource waste for this allocation
        waste = self._calculate_waste(request.requirements)

        # Attempt resource acquisition
        success = await self.resource_pool.acquire(
            request.request_id, request.requirements, timeout=0
        )

        if success:
            # Build dictionary of allocated resources
            allocated = {}
            for resource_type in [
                ResourceType.CPU,
                ResourceType.MEMORY,
                ResourceType.IO,
                ResourceType.NETWORK,
                ResourceType.GPU,
            ]:
                # Check if this resource type was requested
                if request.requirements.resource_types & resource_type:
                    amount = get_resource_amount(request.requirements, resource_type)
                    if amount > 0:
                        allocated[resource_type] = amount

            logger.debug(
                "best_fit_allocation", request_id=request.request_id, waste=waste
            )

            return AllocationResult(
                request_id=request.request_id,
                success=True,
                allocated=allocated,
                allocation_time=time.time() - start_time,
            )
        else:
            return AllocationResult(
                request_id=request.request_id,
                success=False,
                reason="Insufficient resources",
                allocation_time=time.time() - start_time,
            )

    def _calculate_waste(self, requirements: ResourceRequirements) -> float:
        """Calculate the amount of resource waste this allocation would cause.

        Args:
            requirements: Resource requirements to evaluate

        Returns:
            Total amount of wasted resources (lower is better)
        """
        total_waste = 0.0

        for resource_type in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            # Only calculate waste for requested resource types
            if requirements.resource_types & resource_type:
                required = get_resource_amount(requirements, resource_type)
                available = self.resource_pool.available.get(resource_type, 0.0)

                if available >= required:
                    # Waste is the unused portion after allocation
                    waste = available - required
                    total_waste += waste

        return total_waste

    def get_allocation_order(
        self, requests: list[AllocationRequest]
    ) -> list[AllocationRequest]:
        """Order requests by waste (ascending) - least waste first."""
        # Calculate waste for each request and sort by it
        requests_with_waste = [
            (self._calculate_waste(req.requirements), req) for req in requests
        ]

        # Sort by waste amount (ascending - least waste first)
        requests_with_waste.sort(key=lambda x: x[0])

        return [req for _, req in requests_with_waste]


class WorstFitAllocator(ResourceAllocator):
    """Worst-fit allocation strategy implementation.

    Chooses allocations that maximize remaining free space, which can
    help accommodate future large requests but may lead to more fragmentation.
    """

    async def allocate(self, request: AllocationRequest) -> AllocationResult:
        """Allocate resources using worst-fit strategy.

        Uses the same allocation mechanism as first-fit but with different ordering.
        """
        # Delegate to first-fit allocator for actual allocation
        first_fit = FirstFitAllocator(self.resource_pool)
        return await first_fit.allocate(request)

    def get_allocation_order(
        self, requests: list[AllocationRequest]
    ) -> list[AllocationRequest]:
        """Order requests by remaining space (descending) - most remaining space first."""
        # Calculate remaining space for each request and sort by it
        requests_with_remaining = [
            (self._calculate_remaining(req.requirements), req) for req in requests
        ]

        # Sort by remaining space (descending - most remaining first)
        requests_with_remaining.sort(key=lambda x: x[0], reverse=True)

        return [req for _, req in requests_with_remaining]

    def _calculate_remaining(self, requirements: ResourceRequirements) -> float:
        """Calculate remaining resources after this allocation.

        Args:
            requirements: Resource requirements to evaluate

        Returns:
            Total amount of resources that would remain free
        """
        total_remaining = 0.0

        for resource_type in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            # Only calculate for requested resource types
            if requirements.resource_types & resource_type:
                required = get_resource_amount(requirements, resource_type)
                available = self.resource_pool.available.get(resource_type, 0.0)

                if available >= required:
                    remaining = available - required
                    total_remaining += remaining

        return total_remaining


class PriorityAllocator(ResourceAllocator):
    """Priority-based allocation strategy implementation.

    Maintains a priority queue and always processes the highest-priority
    requests first. Essential for systems with critical vs. non-critical workloads.
    """

    def __init__(self, resource_pool: ResourcePool):
        """Initialize priority allocator with a priority queue."""
        super().__init__(resource_pool)
        self._priority_queue: list[AllocationRequest] = []

    async def allocate(self, request: AllocationRequest) -> AllocationResult:
        """Allocate resources based on priority using a priority queue."""
        start_time = time.time()

        # Add request to priority queue (heapq maintains min-heap, so we negate priority)
        heapq.heappush(self._priority_queue, request)

        # Process queue starting with highest priority requests
        processed = []
        while self._priority_queue:
            next_request = heapq.heappop(self._priority_queue)

            # Check if we can allocate to this request
            if self.can_allocate(next_request.requirements):
                success = await self.resource_pool.acquire(
                    next_request.request_id, next_request.requirements, timeout=0
                )

                if success:
                    processed.append(next_request)

                    # If this was our original request, return success
                    if next_request.request_id == request.request_id:
                        # Build allocated resources dictionary
                        allocated = {}
                        for resource_type in [
                            ResourceType.CPU,
                            ResourceType.MEMORY,
                            ResourceType.IO,
                            ResourceType.NETWORK,
                            ResourceType.GPU,
                        ]:
                            if request.requirements.resource_types & resource_type:
                                amount = get_resource_amount(
                                    request.requirements, resource_type
                                )
                                if amount > 0:
                                    allocated[resource_type] = amount

                        return AllocationResult(
                            request_id=request.request_id,
                            success=True,
                            allocated=allocated,
                            allocation_time=time.time() - start_time,
                        )
                else:
                    # Couldn't acquire resources, put back in queue
                    heapq.heappush(self._priority_queue, next_request)
                    break
            else:
                # Can't allocate to this request, put it back and stop
                heapq.heappush(self._priority_queue, next_request)
                break

        # Request couldn't be processed immediately
        return AllocationResult(
            request_id=request.request_id,
            success=False,
            reason="Queued for resources",
            allocation_time=time.time() - start_time,
        )

    def get_allocation_order(
        self, requests: list[AllocationRequest]
    ) -> list[AllocationRequest]:
        """Order requests by priority (highest first)."""
        return sorted(requests, key=lambda r: r.priority, reverse=True)


class FairShareAllocator(ResourceAllocator):
    """Fair-share allocation strategy implementation.

    Ensures equitable resource distribution among different requesters
    by tracking usage history and enforcing fair share limits.
    """

    def __init__(self, resource_pool: ResourcePool):
        """Initialize fair-share allocator with usage tracking."""
        super().__init__(resource_pool)
        self._usage_history: dict[str, float] = defaultdict(
            float
        )  # Total resources used by each requester
        self._allocation_counts: dict[str, int] = defaultdict(
            int
        )  # Number of allocations per requester

    async def allocate(self, request: AllocationRequest) -> AllocationResult:
        """Allocate resources with fair-share constraints."""
        start_time = time.time()

        # Calculate fair share limit for this requester
        fair_share = self._calculate_fair_share(request.requester_id)

        # Check if request would exceed fair share
        current_usage = self._usage_history[request.requester_id]
        requested_total = self._calculate_resource_total(request.requirements)

        if current_usage + requested_total > fair_share:
            return AllocationResult(
                request_id=request.request_id,
                success=False,
                reason=f"Exceeds fair share (current: {current_usage}, "
                f"limit: {fair_share})",
                allocation_time=time.time() - start_time,
            )

        # Attempt allocation within fair share limits
        success = await self.resource_pool.acquire(
            request.request_id, request.requirements, timeout=0
        )

        if success:
            # Update usage tracking
            self._usage_history[request.requester_id] += requested_total
            self._allocation_counts[request.requester_id] += 1

            # Build allocated resources dictionary
            allocated = {}
            for resource_type in [
                ResourceType.CPU,
                ResourceType.MEMORY,
                ResourceType.IO,
                ResourceType.NETWORK,
                ResourceType.GPU,
            ]:
                if request.requirements.resource_types & resource_type:
                    amount = get_resource_amount(request.requirements, resource_type)
                    if amount > 0:
                        allocated[resource_type] = amount

            return AllocationResult(
                request_id=request.request_id,
                success=True,
                allocated=allocated,
                allocation_time=time.time() - start_time,
            )
        else:
            return AllocationResult(
                request_id=request.request_id,
                success=False,
                reason="Insufficient resources",
                allocation_time=time.time() - start_time,
            )

    def _calculate_fair_share(self, requester_id: str) -> float:
        """Calculate fair share limit for a specific requester.

        Args:
            requester_id: ID of the requester

        Returns:
            Fair share resource limit
        """
        # Simple fair share: divide total resources equally among all requesters
        total_requesters = len(self._usage_history) or 1
        total_resources = sum(self.resource_pool.resources.values())

        return total_resources / total_requesters

    def _calculate_resource_total(self, requirements: ResourceRequirements) -> float:
        """Calculate total resource units requested across all resource types.

        Args:
            requirements: Resource requirements to sum

        Returns:
            Total resource units requested
        """
        total = 0.0

        for resource_type in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            if requirements.resource_types & resource_type:
                total += get_resource_amount(requirements, resource_type)

        return total

    def get_allocation_order(
        self, requests: list[AllocationRequest]
    ) -> list[AllocationRequest]:
        """Order requests by usage history (least used requesters first)."""
        return sorted(requests, key=lambda r: self._usage_history[r.requester_id])

    def reset_usage_history(self) -> None:
        """Reset usage history for a new allocation period."""
        self._usage_history.clear()
        self._allocation_counts.clear()


class WeightedAllocator(ResourceAllocator):
    """Weighted allocation strategy implementation.

    Combines priority and weight factors to determine allocation order,
    allowing for fine-grained control over resource distribution.
    """

    def __init__(self, resource_pool: ResourcePool):
        """Initialize weighted allocator."""
        super().__init__(resource_pool)
        self._weights: dict[str, float] = {}  # Requester-specific weights

    def set_weight(self, requester_id: str, weight: float) -> None:
        """Set allocation weight for a specific requester.

        Args:
            requester_id: ID of the requester
            weight: Weight factor (higher = more important)
        """
        self._weights[requester_id] = weight

    async def allocate(self, request: AllocationRequest) -> AllocationResult:
        """Allocate resources based on weighted priority."""
        time.time()

        # Get weight for this requester (use request weight as fallback)
        weight = self._weights.get(request.requester_id, request.weight)

        # Calculate weighted priority
        weighted_priority = request.priority * weight

        # Create modified request with weighted priority
        weighted_request = AllocationRequest(
            request_id=request.request_id,
            requester_id=request.requester_id,
            requirements=request.requirements,
            priority=int(weighted_priority),
            weight=weight,
            metadata=request.metadata,
            timestamp=request.timestamp,
            deadline=request.deadline,
        )

        # Use priority allocator with the weighted priority
        priority_allocator = PriorityAllocator(self.resource_pool)
        return await priority_allocator.allocate(weighted_request)

    def get_allocation_order(
        self, requests: list[AllocationRequest]
    ) -> list[AllocationRequest]:
        """Order requests by weighted priority (highest weighted priority first)."""
        weighted_requests = []

        for req in requests:
            weight = self._weights.get(req.requester_id, req.weight)
            weighted_priority = req.priority * weight
            weighted_requests.append((weighted_priority, req))

        # Sort by weighted priority (descending - highest first)
        weighted_requests.sort(key=lambda x: x[0], reverse=True)

        return [req for _, req in weighted_requests]


def create_allocator(
    strategy: AllocationStrategy, resource_pool: ResourcePool
) -> ResourceAllocator:
    """Factory function for creating resource allocators.

    Args:
        strategy: The allocation strategy to use
        resource_pool: The pool of available resources

    Returns:
        Configured allocator instance

    Raises:
        ValueError: If strategy is not recognized
    """
    allocators = {
        AllocationStrategy.FIRST_FIT: FirstFitAllocator,
        AllocationStrategy.BEST_FIT: BestFitAllocator,
        AllocationStrategy.WORST_FIT: WorstFitAllocator,
        AllocationStrategy.PRIORITY: PriorityAllocator,
        AllocationStrategy.FAIR_SHARE: FairShareAllocator,
        AllocationStrategy.WEIGHTED: WeightedAllocator,
    }

    allocator_class = allocators.get(strategy)
    if allocator_class is None:
        # Default to first-fit for unknown strategies
        allocator_class = FirstFitAllocator

    return allocator_class(resource_pool)  # type: ignore[abstract]
