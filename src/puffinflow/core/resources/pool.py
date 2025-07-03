"""Resource pool implementation with leak detection"""
import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Any, Set, List
from enum import Flag, auto

from .requirements import ResourceRequirements, ResourceType, get_resource_amount
from ..reliability.leak_detector import leak_detector

logger = logging.getLogger(__name__)

RESOURCE_ATTRIBUTE_MAPPING = {
    ResourceType.CPU: 'cpu_units',
    ResourceType.MEMORY: 'memory_mb',
    ResourceType.IO: 'io_weight',
    ResourceType.NETWORK: 'network_weight',
    ResourceType.GPU: 'gpu_units'
}

class ResourceAllocationError(Exception):
    """Base class for resource allocation errors."""
    pass

class ResourceOverflowError(ResourceAllocationError):
    """Raised when resource allocation would exceed limits."""
    pass

class ResourceQuotaExceededError(ResourceAllocationError):
    """Raised when state/agent exceeds its resource quota."""
    pass

@dataclass
class ResourceUsageStats:
    """Statistics for resource usage."""
    peak_usage: float = 0.0
    current_usage: float = 0.0
    total_allocations: int = 0
    failed_allocations: int = 0
    last_allocation_time: Optional[float] = None
    total_wait_time: float = 0.0

class ResourcePool:
    """Advanced resource management system with leak detection."""

    def __init__(
        self,
        total_cpu: float = 4.0,
        total_memory: float = 1024.0,
        total_io: float = 100.0,
        total_network: float = 100.0,
        total_gpu: float = 0.0,  # Changed default from 1.0 to 0.0
        enable_quotas: bool = False,
        enable_preemption: bool = False,
        enable_leak_detection: bool = True  # NEW
    ):
        # Resource limits
        self.resources = {
            ResourceType.CPU: total_cpu,
            ResourceType.MEMORY: total_memory,
            ResourceType.IO: total_io,
            ResourceType.NETWORK: total_network,
            ResourceType.GPU: total_gpu
        }

        # Available resources
        self.available = self.resources.copy()

        # Core synchronization
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

        # Allocation tracking
        self._allocations: Dict[str, Dict[ResourceType, float]] = {}
        self._allocation_times: Dict[str, float] = {}  # NEW: Track allocation times

        # Usage statistics
        self._usage_stats = {
            rt: ResourceUsageStats() for rt in ResourceType if rt != ResourceType.NONE
        }

        # Quotas (if enabled)
        self.enable_quotas = enable_quotas
        self._enable_quotas = enable_quotas  # Add private attribute for tests
        self._quotas: Dict[str, Dict[ResourceType, float]] = {}

        # Preemption (if enabled)
        self.enable_preemption = enable_preemption
        self._enable_preemption = enable_preemption  # Add private attribute for tests
        self._preempted_states: Set[str] = set()

        # Historical tracking
        self._allocation_history: Dict[ResourceType, List[tuple]] = defaultdict(list)
        self._usage_history: List[tuple] = []  # Add for tests
        self._history_retention = 3600  # 1 hour

        # Waiting states tracking (for tests)
        self._waiting_states: Set[str] = set()

        # NEW: Leak detection
        self.enable_leak_detection = enable_leak_detection
        self._agent_names: Dict[str, str] = {}  # state_name -> agent_name mapping

    async def set_quota(self, state_name: str, resource_type: ResourceType, limit: float) -> None:
        """Set resource quota for a state."""
        if not self.enable_quotas:
            raise RuntimeError("Quotas are not enabled for this resource pool")

        async with self._lock:
            if state_name not in self._quotas:
                self._quotas[state_name] = {}
            self._quotas[state_name][resource_type] = limit

    def _check_quota(self, state_name: str, requirements: ResourceRequirements) -> bool:
        """Check if allocation would exceed quota."""
        if not self.enable_quotas:
            return True

        current_usage = self._allocations.get(state_name, {})

        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.IO,
                             ResourceType.NETWORK, ResourceType.GPU]:
            # Only check quotas for resources that are specified in resource_types
            if resource_type not in requirements.resource_types:
                continue
                
            quota = self._quotas.get(state_name, {}).get(resource_type)
            if quota is None:
                continue  # No quota set

            required = get_resource_amount(requirements, resource_type)
            current = current_usage.get(resource_type, 0.0)

            if current + required > quota:
                return False

        return True

    async def acquire(
        self,
        state_name: str,
        requirements: ResourceRequirements,
        timeout: Optional[float] = None,
        allow_preemption: bool = False,
        agent_name: Optional[str] = None  # NEW: Track agent name for leak detection
    ) -> bool:
        """Acquire resources with advanced features and leak detection."""
        start_time = time.time()

        # Store agent name for leak detection
        if agent_name and self.enable_leak_detection:
            self._agent_names[state_name] = agent_name

        try:
            # Validate requirements
            self._validate_requirements(requirements)

            async with self._condition:
                # Check quotas
                if not self._check_quota(state_name, requirements):
                    raise ResourceQuotaExceededError(f"Quota exceeded for {state_name}")

                # Try allocation
                while not self._can_allocate(requirements):
                    # Add to waiting states
                    self._waiting_states.add(state_name)
                    
                    # Handle preemption
                    if allow_preemption and self.enable_preemption:
                        if self._try_preemption(state_name, requirements):
                            break

                    # Wait for resources to become available
                    if timeout:
                        remaining_time = timeout - (time.time() - start_time)
                        if remaining_time <= 0:
                            self._waiting_states.discard(state_name)
                            self._update_stats_failure(requirements)
                            return False

                        try:
                            await asyncio.wait_for(self._condition.wait(), timeout=remaining_time)
                        except asyncio.TimeoutError:
                            self._waiting_states.discard(state_name)
                            self._update_stats_failure(requirements)
                            return False
                    else:
                        await self._condition.wait()

                # Remove from waiting states when allocation succeeds
                self._waiting_states.discard(state_name)

                # Allocate resources
                self._allocate(state_name, requirements)

                # NEW: Track allocation for leak detection
                if self.enable_leak_detection:
                    agent = self._agent_names.get(state_name, "unknown")
                    resource_dict = {
                        rt.name.lower(): get_resource_amount(requirements, rt)
                        for rt in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.IO,
                                  ResourceType.NETWORK, ResourceType.GPU]
                        if rt in requirements.resource_types and get_resource_amount(requirements, rt) > 0
                    }
                    leak_detector.track_allocation(state_name, agent, resource_dict)

                # Update statistics
                self._update_stats(state_name, requirements, start_time)

                return True

        except Exception as e:
            self._update_stats_failure(requirements)
            raise

    def _validate_requirements(self, requirements: ResourceRequirements) -> None:
        """Validate resource requirements."""
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.IO,
                             ResourceType.NETWORK, ResourceType.GPU]:
            # Only validate resources that are specified in resource_types
            if resource_type not in requirements.resource_types:
                continue
                
            amount = get_resource_amount(requirements, resource_type)
            if amount < 0:
                raise ValueError(f"Negative resource requirement: {resource_type.name}={amount}")

            total_available = self.resources.get(resource_type, 0)
            if amount > total_available:
                raise ResourceOverflowError(
                    f"Requested {resource_type.name}={amount} exceeds total available={total_available}"
                )

    def _can_allocate(self, requirements: ResourceRequirements) -> bool:
        """Check if resources can be allocated."""
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.IO,
                             ResourceType.NETWORK, ResourceType.GPU]:
            # Only check resources that are specified in resource_types
            if resource_type not in requirements.resource_types:
                continue
                
            required = get_resource_amount(requirements, resource_type)
            available = self.available.get(resource_type, 0.0)
            if required > available:
                return False
        return True

    def _allocate(self, state_name: str, requirements: ResourceRequirements) -> None:
        """Allocate resources to a state."""
        if state_name not in self._allocations:
            self._allocations[state_name] = {}

        self._allocation_times[state_name] = time.time()  # NEW: Track allocation time

        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.IO,
                             ResourceType.NETWORK, ResourceType.GPU]:
            # Only allocate resources that are specified in resource_types
            if resource_type not in requirements.resource_types:
                continue
                
            amount = get_resource_amount(requirements, resource_type)
            if amount > 0:
                self._allocations[state_name][resource_type] = amount
                self.available[resource_type] -= amount

    def _try_preemption(self, state_name: str, requirements: ResourceRequirements) -> bool:
        """Attempt to preempt lower priority states."""
        if not self.enable_preemption:
            return False

        # Find potential states to preempt
        candidates = []
        for allocated_state, resources in self._allocations.items():
            if allocated_state != state_name:
                total_resources = sum(resources.values())
                candidates.append((allocated_state, total_resources))

        if not candidates:
            return False

        # Sort by total resource usage (preempt largest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Check if preempting would free enough resources
        would_free = {rt: 0.0 for rt in ResourceType if rt != ResourceType.NONE}
        preempt_list = []

        for candidate_state, _ in candidates:
            candidate_resources = self._allocations[candidate_state]
            for rt, amount in candidate_resources.items():
                would_free[rt] += amount
            preempt_list.append(candidate_state)

            # Check if we now have enough
            could_satisfy = True
            for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.IO,
                                 ResourceType.NETWORK, ResourceType.GPU]:
                # Only check resources that are specified in resource_types
                if resource_type not in requirements.resource_types:
                    continue
                    
                required = get_resource_amount(requirements, resource_type)
                available_after_preemption = self.available[resource_type] + would_free[resource_type]
                if required > available_after_preemption:
                    could_satisfy = False
                    break

            if could_satisfy:
                # Preempt states
                for preempt_state in preempt_list:
                    self._preempt_state(preempt_state)
                return True

        return False

    def _preempt_state(self, state_name: str) -> None:
        """Preempt a state and return its resources."""
        if state_name in self._allocations:
            # Return resources
            for resource_type, amount in self._allocations[state_name].items():
                self.available[resource_type] += amount

            # Track preemption
            self._preempted_states.add(state_name)
            del self._allocations[state_name]

            # NEW: Remove from leak detection
            if self.enable_leak_detection:
                agent = self._agent_names.get(state_name, "unknown")
                leak_detector.track_release(state_name, agent)

            logger.warning(f"Preempted state {state_name}")

    async def release(self, state_name: str) -> None:
        """Release resources held by a state."""
        async with self._condition:
            if state_name in self._allocations:
                # Return resources
                for resource_type, amount in self._allocations[state_name].items():
                    self.available[resource_type] += amount

                # Clean up tracking
                del self._allocations[state_name]
                if state_name in self._allocation_times:
                    del self._allocation_times[state_name]

                # NEW: Track release for leak detection
                if self.enable_leak_detection:
                    agent = self._agent_names.get(state_name, "unknown")
                    leak_detector.track_release(state_name, agent)
                    if state_name in self._agent_names:
                        del self._agent_names[state_name]

                # Notify waiting states
                self._condition.notify_all()

    def _update_stats(self, state_name: str, requirements: ResourceRequirements, start_time: float) -> None:
        """Update usage statistics."""
        wait_time = time.time() - start_time
        current_time = time.time()

        # Add to usage history
        available_resources = self.available.copy()
        self._usage_history.append((current_time, available_resources))

        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.IO,
                             ResourceType.NETWORK, ResourceType.GPU]:
            # Only update stats for resources that are specified in resource_types
            if resource_type not in requirements.resource_types:
                continue
                
            amount = get_resource_amount(requirements, resource_type)
            if amount <= 0:
                continue

            stats = self._usage_stats[resource_type]
            stats.total_allocations += 1
            stats.total_wait_time += wait_time
            stats.last_allocation_time = current_time

            # Calculate current usage across all allocations
            current_usage = sum(
                alloc.get(resource_type, 0.0) for alloc in self._allocations.values()
            )
            stats.current_usage = current_usage
            stats.peak_usage = max(stats.peak_usage, current_usage)

            # Record historical data point
            self._allocation_history[resource_type].append((current_time, current_usage))

            # Cleanup old history
            cutoff = current_time - self._history_retention
            self._allocation_history[resource_type] = [
                (t, usage) for t, usage in self._allocation_history[resource_type] if t >= cutoff
            ]

        # Cleanup old usage history
        cutoff = current_time - self._history_retention
        self._usage_history = [
            (t, usage) for t, usage in self._usage_history if t >= cutoff
        ]

    def _update_stats_failure(self, requirements: ResourceRequirements) -> None:
        """Update stats for failed allocation"""
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.IO,
                             ResourceType.NETWORK, ResourceType.GPU]:
            # Only update failure stats for resources that are specified in resource_types
            if resource_type not in requirements.resource_types:
                continue
                
            amount = get_resource_amount(requirements, resource_type)
            if amount > 0:
                self._usage_stats[resource_type].failed_allocations += 1

    def get_usage_stats(self) -> Dict[ResourceType, ResourceUsageStats]:
        """Get current usage statistics."""
        return self._usage_stats.copy()

    def get_state_allocations(self) -> Dict[str, Dict[ResourceType, float]]:
        """Get current resource allocations by state."""
        return self._allocations.copy()

    def get_waiting_states(self) -> Set[str]:
        """Get states waiting for resources."""
        return self._waiting_states.copy()

    def get_preempted_states(self) -> Set[str]:
        """Get states that were preempted."""
        return self._preempted_states.copy()

    # Leak detection methods
    def check_leaks(self) -> List[Any]:
        """Check for resource leaks"""
        if not self.enable_leak_detection:
            return []
        return leak_detector.detect_leaks()

    def get_leak_metrics(self) -> Dict[str, Any]:
        """Get leak detection metrics"""
        if not self.enable_leak_detection:
            return {"leak_detection": "disabled"}
        return leak_detector.get_metrics()

    async def force_release(self, state_name: str):
        """Force release resources (for leak cleanup)"""
        logger.warning(f"Force releasing resources for state {state_name}")
        await self.release(state_name)