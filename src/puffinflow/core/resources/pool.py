"""Resource pool implementation with advanced features.

Provides a comprehensive resource management system with leak detection,
quota enforcement, preemption capabilities, and detailed usage tracking.
Supports CPU, memory, I/O, network, and GPU resources.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

# Import from the canonical source to ensure consistent enum instances
from .requirements import (
    ResourceRequirements,
    ResourceType,
    get_resource_amount,
    safe_check_resource_type,
)

# Import leak detector with fallback
try:
    from ..reliability.leak_detector import leak_detector
except ImportError:
    # Mock leak detector if not available
    class MockLeakDetector:
        def track_allocation(
            self, state_name: Any, agent_name: Any, resources: Any
        ) -> None:
            pass

        def track_release(self, state_name: Any, agent_name: Any) -> None:
            pass

        def detect_leaks(self) -> list[Any]:
            return []

        def get_metrics(self) -> dict[str, Any]:
            return {"leak_detection": "mock"}

    leak_detector = MockLeakDetector()  # type: ignore

logger = logging.getLogger(__name__)


class ResourceAllocationError(Exception):
    """Base class for resource allocation errors."""

    pass


class ResourceOverflowError(ResourceAllocationError):
    """Raised when resource allocation would exceed system limits."""

    pass


class ResourceQuotaExceededError(ResourceAllocationError):
    """Raised when a state/agent exceeds its assigned resource quota."""

    pass


@dataclass
class ResourceUsageStats:
    """Statistics container for tracking resource usage patterns."""

    peak_usage: float = 0.0
    current_usage: float = 0.0
    total_allocations: int = 0
    failed_allocations: int = 0
    last_allocation_time: Optional[float] = None
    total_wait_time: float = 0.0


class ResourcePool:
    """Advanced resource management system with comprehensive features."""

    def __init__(
        self,
        total_cpu: float = 4.0,
        total_memory: float = 1024.0,
        total_io: float = 100.0,
        total_network: float = 100.0,
        total_gpu: float = 0.0,
        enable_quotas: bool = False,
        enable_preemption: bool = False,
        enable_leak_detection: bool = True,
    ):
        """Initialize resource pool with specified capacities and features."""
        # Resource capacity limits
        self.resources = {
            ResourceType.CPU: total_cpu,
            ResourceType.MEMORY: total_memory,
            ResourceType.IO: total_io,
            ResourceType.NETWORK: total_network,
            ResourceType.GPU: total_gpu,
        }

        # Currently available resources
        self.available = self.resources.copy()

        # Synchronization primitives
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

        # Resource allocation tracking
        self._allocations: dict[str, dict[ResourceType, float]] = {}
        self._allocation_times: dict[str, float] = {}

        # Usage statistics
        self._usage_stats = {
            rt: ResourceUsageStats()
            for rt in ResourceType
            if rt != ResourceType.NONE and rt != ResourceType.ALL
        }

        # Feature flags
        self.enable_quotas = enable_quotas
        self._quotas: dict[str, dict[ResourceType, float]] = {}

        self.enable_preemption = enable_preemption
        self._preempted_states: set[str] = set()

        # Historical data
        self._allocation_history: dict[ResourceType, list[tuple]] = defaultdict(list)
        self._usage_history: list[tuple] = []
        self._history_retention = 3600

        # Queue management
        self._waiting_states: set[str] = set()

        # Leak detection
        self.enable_leak_detection = enable_leak_detection
        self._agent_names: dict[str, str] = {}

    async def set_quota(
        self, state_name: str, resource_type: ResourceType, limit: float
    ) -> None:
        """Set resource quota for a specific state."""
        if not self.enable_quotas:
            raise RuntimeError("Quotas are not enabled for this resource pool")

        async with self._lock:
            if state_name not in self._quotas:
                self._quotas[state_name] = {}
            self._quotas[state_name][resource_type] = limit

    def _check_quota(self, state_name: str, requirements: ResourceRequirements) -> bool:
        """Check if allocation would exceed assigned quota."""
        if not self.enable_quotas:
            return True

        current_usage = self._allocations.get(state_name, {})

        for resource_type in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            # Only check quotas for resources that are actually requested
            if safe_check_resource_type(requirements, resource_type):
                quota = self._quotas.get(state_name, {}).get(resource_type)
                if quota is None:
                    continue

                required = get_resource_amount(requirements, resource_type)
                current = current_usage.get(resource_type, 0.0)

                if current + required > quota:
                    logger.warning(
                        f"Quota exceeded for {state_name}: {resource_type.name} "
                        f"(current: {current}, required: {required}, quota: {quota})"
                    )
                    return False

        return True

    async def acquire(
        self,
        state_name: str,
        requirements: ResourceRequirements,
        timeout: Optional[float] = None,
        allow_preemption: bool = False,
        agent_name: Optional[str] = None,
    ) -> bool:
        """Acquire resources for a state with advanced features."""
        start_time = time.time()

        # Store agent name for leak detection
        if agent_name and self.enable_leak_detection:
            self._agent_names[state_name] = agent_name

        try:
            # Validate and fix requirements if needed
            requirements = self._validate_and_fix_requirements(requirements)

            async with self._condition:
                # Check if requirements exceed total available resources
                self._validate_requirements_against_total(requirements)

                # Check quota constraints
                if not self._check_quota(state_name, requirements):
                    raise ResourceQuotaExceededError(f"Quota exceeded for {state_name}")

                # Wait for resources to become available
                while not self._can_allocate(requirements):
                    self._waiting_states.add(state_name)

                    # Try preemption if enabled
                    if (
                        allow_preemption
                        and self.enable_preemption
                        and self._try_preemption(state_name, requirements)
                    ):
                        break

                    # Wait with timeout
                    if timeout:
                        remaining_time = timeout - (time.time() - start_time)
                        if remaining_time <= 0:
                            self._waiting_states.discard(state_name)
                            self._update_stats_failure(requirements)
                            return False

                        try:
                            await asyncio.wait_for(
                                self._condition.wait(), timeout=remaining_time
                            )
                        except asyncio.TimeoutError:
                            self._waiting_states.discard(state_name)
                            self._update_stats_failure(requirements)
                            return False
                    else:
                        await self._condition.wait()

                # Remove from waiting states
                self._waiting_states.discard(state_name)

                # Perform allocation
                self._allocate(state_name, requirements)

                # Track for leak detection
                if self.enable_leak_detection:
                    agent = self._agent_names.get(state_name, "unknown")
                    resource_dict = self._build_resource_dict(requirements)
                    leak_detector.track_allocation(state_name, agent, resource_dict)

                # Update statistics
                self._update_stats(state_name, requirements, start_time)

                return True

        except Exception as e:
            self._update_stats_failure(requirements)
            logger.error(f"Error acquiring resources for {state_name}: {e}")
            raise

    def _validate_and_fix_requirements(
        self, requirements: ResourceRequirements
    ) -> ResourceRequirements:
        """Validate and fix resource requirements if needed."""
        try:
            # Check for negative resource values
            resource_values = {
                "cpu_units": getattr(requirements, "cpu_units", 0.0),
                "memory_mb": getattr(requirements, "memory_mb", 0.0),
                "io_weight": getattr(requirements, "io_weight", 0.0),
                "network_weight": getattr(requirements, "network_weight", 0.0),
                "gpu_units": getattr(requirements, "gpu_units", 0.0),
            }

            for attr_name, value in resource_values.items():
                if value < 0:
                    raise ValueError(
                        f"Negative resource requirement: {attr_name}={value}"
                    )

            # Test bitwise operations
            requirements.resource_types & ResourceType.CPU
            logger.debug(f"Requirements validation passed: {requirements}")

            return requirements

        except ValueError:
            # Re-raise ValueError for negative resource requirements
            raise
        except Exception as e:
            logger.error(f"Error validating requirements: {e}")
            # Create a safe fallback
            fallback = ResourceRequirements(
                cpu_units=getattr(requirements, "cpu_units", 1.0),
                memory_mb=getattr(requirements, "memory_mb", 100.0),
                io_weight=getattr(requirements, "io_weight", 1.0),
                network_weight=getattr(requirements, "network_weight", 1.0),
                gpu_units=getattr(requirements, "gpu_units", 0.0),
                resource_types=ResourceType.ALL,
            )
            logger.info(f"Using fallback requirements: {fallback}")
            return fallback

    def _validate_requirements_against_total(
        self, requirements: ResourceRequirements
    ) -> None:
        """Validate that requirements don't exceed total available resources."""
        for resource_type in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            if safe_check_resource_type(requirements, resource_type):
                required = get_resource_amount(requirements, resource_type)
                total_available = self.resources.get(resource_type, 0.0)

                if required > total_available:
                    raise ResourceOverflowError(
                        f"Required {resource_type.name} ({required}) exceeds total available ({total_available})"
                    )

    def _build_resource_dict(
        self, requirements: ResourceRequirements
    ) -> dict[str, float]:
        """Build resource dictionary for leak detection."""
        resource_dict = {}

        for resource_type in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            if safe_check_resource_type(requirements, resource_type):
                amount = get_resource_amount(requirements, resource_type)
                if amount > 0 and resource_type.name:
                    resource_dict[resource_type.name.lower()] = amount

        return resource_dict

    def _can_allocate(self, requirements: ResourceRequirements) -> bool:
        """Check if resources can be allocated immediately."""
        try:
            logger.debug(f"Checking allocation for: {requirements}")

            for resource_type in [
                ResourceType.CPU,
                ResourceType.MEMORY,
                ResourceType.IO,
                ResourceType.NETWORK,
                ResourceType.GPU,
            ]:
                # Use safe check for resource type
                if safe_check_resource_type(requirements, resource_type):
                    required = get_resource_amount(requirements, resource_type)
                    available = self.available.get(resource_type, 0.0)

                    logger.debug(
                        f"Resource {resource_type.name}: required={required}, available={available}"
                    )

                    if required > available:
                        logger.debug(
                            f"Cannot allocate - insufficient {resource_type.name}"
                        )
                        return False

            return True

        except Exception as e:
            logger.error(f"Error in _can_allocate: {e}")
            logger.error(f"Requirements: {requirements}")
            logger.error(f"Requirements type: {type(requirements)}")
            # In case of error, assume we can't allocate safely
            return False

    def _allocate(self, state_name: str, requirements: ResourceRequirements) -> None:
        """Perform the actual resource allocation."""
        try:
            if state_name not in self._allocations:
                self._allocations[state_name] = {}

            # Record allocation timestamp
            self._allocation_times[state_name] = time.time()

            # Allocate each requested resource type
            for resource_type in [
                ResourceType.CPU,
                ResourceType.MEMORY,
                ResourceType.IO,
                ResourceType.NETWORK,
                ResourceType.GPU,
            ]:
                if safe_check_resource_type(requirements, resource_type):
                    amount = get_resource_amount(requirements, resource_type)
                    if amount > 0:
                        self._allocations[state_name][resource_type] = amount
                        self.available[resource_type] -= amount

                        logger.debug(
                            f"Allocated {amount} {resource_type.name} to {state_name}"
                        )

        except Exception as e:
            logger.error(f"Error in _allocate: {e}")
            raise

    def _try_preemption(
        self, state_name: str, requirements: ResourceRequirements
    ) -> bool:
        """Attempt to preempt lower-priority states."""
        if not self.enable_preemption:
            return False

        try:
            # Find candidates for preemption
            candidates = []
            for allocated_state, resources in self._allocations.items():
                if allocated_state != state_name:
                    total_resources = sum(resources.values())
                    candidates.append((allocated_state, total_resources))

            if not candidates:
                return False

            # Sort by resource usage (preempt largest first)
            candidates.sort(key=lambda x: x[1], reverse=True)

            # Simulate preemption
            would_free = {
                rt: 0.0
                for rt in ResourceType
                if rt != ResourceType.NONE and rt != ResourceType.ALL
            }
            preempt_list = []

            for candidate_state, _ in candidates:
                candidate_resources = self._allocations[candidate_state]
                for rt, amount in candidate_resources.items():
                    would_free[rt] += amount  # type: ignore
                preempt_list.append(candidate_state)

                # Check if preemption would free enough resources
                could_satisfy = True
                for resource_type in [
                    ResourceType.CPU,
                    ResourceType.MEMORY,
                    ResourceType.IO,
                    ResourceType.NETWORK,
                    ResourceType.GPU,
                ]:
                    if safe_check_resource_type(requirements, resource_type):
                        required = get_resource_amount(requirements, resource_type)
                        available_after = (
                            self.available[resource_type] + would_free[resource_type]  # type: ignore
                        )
                        if required > available_after:
                            could_satisfy = False
                            break

                if could_satisfy:
                    # Perform actual preemption
                    for preempt_state in preempt_list:
                        self._preempt_state(preempt_state)
                    return True

            return False

        except Exception as e:
            logger.error(f"Error in preemption: {e}")
            return False

    def _preempt_state(self, state_name: str) -> None:
        """Forcibly preempt a state."""
        try:
            if state_name in self._allocations:
                # Return resources to pool
                for resource_type, amount in self._allocations[state_name].items():
                    self.available[resource_type] += amount

                # Track preemption
                self._preempted_states.add(state_name)
                del self._allocations[state_name]

                # Remove from leak detection
                if self.enable_leak_detection:
                    agent = self._agent_names.get(state_name, "unknown")
                    leak_detector.track_release(state_name, agent)

                logger.warning(f"Preempted state {state_name}")

        except Exception as e:
            logger.error(f"Error preempting state {state_name}: {e}")

    async def release(self, state_name: str) -> None:
        """Release all resources held by a state."""
        try:
            async with self._condition:
                if state_name in self._allocations:
                    # Return resources to pool
                    for resource_type, amount in self._allocations[state_name].items():
                        self.available[resource_type] += amount
                        logger.debug(
                            f"Released {amount} {resource_type.name} from {state_name}"
                        )

                    # Clean up tracking
                    del self._allocations[state_name]
                    if state_name in self._allocation_times:
                        del self._allocation_times[state_name]

                    # Update leak detection
                    if self.enable_leak_detection:
                        agent = self._agent_names.get(state_name, "unknown")
                        leak_detector.track_release(state_name, agent)
                        if state_name in self._agent_names:
                            del self._agent_names[state_name]

                    # Notify waiting states
                    self._condition.notify_all()

        except Exception as e:
            logger.error(f"Error releasing resources for {state_name}: {e}")

    def _update_stats(
        self, state_name: str, requirements: ResourceRequirements, start_time: float
    ) -> None:
        """Update usage statistics after successful allocation."""
        try:
            wait_time = time.time() - start_time
            current_time = time.time()

            # Add to usage history
            self._usage_history.append((current_time, self.available.copy()))

            # Update stats for each resource type
            for resource_type in [
                ResourceType.CPU,
                ResourceType.MEMORY,
                ResourceType.IO,
                ResourceType.NETWORK,
                ResourceType.GPU,
            ]:
                if safe_check_resource_type(requirements, resource_type):
                    amount = get_resource_amount(requirements, resource_type)
                    if amount <= 0:
                        continue

                    stats = self._usage_stats[resource_type]  # type: ignore
                    stats.total_allocations += 1
                    stats.total_wait_time += wait_time
                    stats.last_allocation_time = current_time

                    # Calculate current usage
                    current_usage = sum(
                        alloc.get(resource_type, 0.0)
                        for alloc in self._allocations.values()
                    )
                    stats.current_usage = current_usage
                    stats.peak_usage = max(stats.peak_usage, current_usage)

                    # Record historical data
                    self._allocation_history[resource_type].append(
                        (current_time, current_usage)
                    )

            # Clean up old history
            cutoff = current_time - self._history_retention
            self._usage_history = [
                (t, usage) for t, usage in self._usage_history if t >= cutoff
            ]

            for resource_type in self._allocation_history:
                self._allocation_history[resource_type] = [
                    (t, usage)
                    for t, usage in self._allocation_history[resource_type]
                    if t >= cutoff
                ]

        except Exception as e:
            logger.error(f"Error updating stats: {e}")

    def _update_stats_failure(self, requirements: ResourceRequirements) -> None:
        """Update statistics for failed allocations."""
        try:
            for resource_type in [
                ResourceType.CPU,
                ResourceType.MEMORY,
                ResourceType.IO,
                ResourceType.NETWORK,
                ResourceType.GPU,
            ]:
                if safe_check_resource_type(requirements, resource_type):
                    amount = get_resource_amount(requirements, resource_type)
                    if amount > 0:
                        self._usage_stats[resource_type].failed_allocations += 1  # type: ignore
        except Exception as e:
            logger.error(f"Error updating failure stats: {e}")

    # Information methods
    def get_usage_stats(self) -> dict[ResourceType, ResourceUsageStats]:
        """Get usage statistics for all resource types."""
        return self._usage_stats.copy()  # type: ignore

    def get_state_allocations(self) -> dict[str, dict[ResourceType, float]]:
        """Get current allocations by state."""
        return self._allocations.copy()

    def get_waiting_states(self) -> set[str]:
        """Get states waiting for resources."""
        return self._waiting_states.copy()

    def get_preempted_states(self) -> set[str]:
        """Get states that were preempted."""
        return self._preempted_states.copy()

    def check_leaks(self) -> list[Any]:
        """Check for resource leaks."""
        if not self.enable_leak_detection:
            return []
        try:
            return leak_detector.detect_leaks()
        except Exception as e:
            logger.error(f"Error checking leaks: {e}")
            return []

    def get_leak_metrics(self) -> dict[str, Any]:
        """Get leak detection metrics."""
        if not self.enable_leak_detection:
            return {"leak_detection": "disabled"}
        try:
            return leak_detector.get_metrics()
        except Exception as e:
            logger.error(f"Error getting leak metrics: {e}")
            return {"leak_detection": "error", "error": str(e)}

    async def force_release(self, state_name: str) -> None:
        """Force release resources from a state."""
        logger.warning(f"Force releasing resources for state {state_name}")
        await self.release(state_name)
