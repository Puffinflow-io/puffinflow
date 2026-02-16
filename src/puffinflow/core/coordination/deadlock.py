"""
Deadlock detection for workflow execution.

This module provides comprehensive deadlock detection capabilities including:
- Dependency graph cycle detection
- Resource wait-for graph analysis
- Configurable resolution strategies
- Performance monitoring and metrics
- Memory management and cleanup
- Thread-safe operations
"""

import asyncio
import logging
import time
import uuid
import weakref
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from typing import Any, Callable, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)


class DeadlockResolutionStrategy(Enum):
    """Strategies for resolving deadlocks"""

    RAISE_EXCEPTION = auto()
    KILL_YOUNGEST = auto()
    KILL_OLDEST = auto()
    KILL_LOWEST_PRIORITY = auto()
    PREEMPT_RESOURCES = auto()
    ROLLBACK_TRANSACTION = auto()
    LOG_ONLY = auto()
    CUSTOM_CALLBACK = auto()


class DeadlockError(Exception):
    """Raised when a deadlock is detected"""

    def __init__(
        self,
        cycle: list[str],
        detection_id: Optional[str] = None,
        message: str = "Deadlock detected",
    ):
        self.cycle = cycle
        self.detection_id = detection_id or str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc)
        super().__init__(f"{message}: {' -> '.join(cycle)} (ID: {self.detection_id})")


@dataclass
class ResourceNode:
    """Node in resource wait graph with enhanced metadata"""

    resource_id: str
    resource_type: str
    holders: set[str] = field(default_factory=set)
    waiters: set[str] = field(default_factory=set)
    acquired_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    max_holders: int = 1  # For semaphore-like resources
    priority: int = 0

    def is_free(self) -> bool:
        """Check if resource has available capacity"""
        return len(self.holders) < self.max_holders

    def can_acquire(self, count: int = 1) -> bool:
        """Check if resource can be acquired by count holders"""
        return len(self.holders) + count <= self.max_holders

    def age_seconds(self) -> float:
        """Get age of resource in seconds"""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    def idle_time_seconds(self) -> float:
        """Get idle time since last access"""
        return (datetime.now(timezone.utc) - self.last_accessed).total_seconds()

    def update_access(self) -> None:
        """Update last access time"""
        self.last_accessed = datetime.now(timezone.utc)


@dataclass
class ProcessNode:
    """Node representing a process/state in wait graph with enhanced tracking"""

    process_id: str
    process_name: str
    holding: set[str] = field(default_factory=set)
    waiting_for: set[str] = field(default_factory=set)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    blocked_at: Optional[datetime] = None
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 0
    timeout: Optional[float] = None  # Timeout in seconds
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_blocked(self) -> bool:
        """Check if process is blocked"""
        return len(self.waiting_for) > 0

    def is_timed_out(self) -> bool:
        """Check if process has timed out"""
        if not self.timeout or not self.blocked_at:
            return False
        return (
            datetime.now(timezone.utc) - self.blocked_at
        ).total_seconds() > self.timeout

    def age_seconds(self) -> float:
        """Get age of process in seconds"""
        return (datetime.now(timezone.utc) - self.started_at).total_seconds()

    def blocked_duration_seconds(self) -> float:
        """Get how long process has been blocked"""
        if self.blocked_at:
            return (datetime.now(timezone.utc) - self.blocked_at).total_seconds()
        return 0.0

    def idle_time_seconds(self) -> float:
        """Get idle time since last activity"""
        return (datetime.now(timezone.utc) - self.last_activity).total_seconds()

    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)


@dataclass
class CycleDetectionResult:
    """Enhanced result of cycle detection with performance metrics"""

    has_cycle: bool
    cycles: list[list[str]] = field(default_factory=list)
    detection_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    graph_size: int = 0
    edge_count: int = 0
    detection_duration_ms: float = 0.0
    algorithm_used: str = "dfs"

    def get_shortest_cycle(self) -> Optional[list[str]]:
        """Get the shortest detected cycle"""
        if not self.cycles:
            return None
        return min(self.cycles, key=len)

    def get_longest_cycle(self) -> Optional[list[str]]:
        """Get the longest detected cycle"""
        if not self.cycles:
            return None
        return max(self.cycles, key=len)

    def get_critical_cycle(self) -> Optional[list[str]]:
        """Get the most critical cycle (shortest with highest priority nodes)"""
        if not self.cycles:
            return None
        # For now, return shortest. Can be enhanced with priority logic
        return self.get_shortest_cycle()


class NodeCleanupStrategy:
    """Strategy for node cleanup with different policies"""

    @staticmethod
    def lru_cleanup(
        nodes: dict[str, Any], metadata: dict[str, dict], count: int
    ) -> list[str]:
        """Least Recently Used cleanup"""
        sorted_nodes = sorted(
            nodes.keys(),
            key=lambda n: metadata.get(n, {}).get(
                "last_access", datetime.min.replace(tzinfo=timezone.utc)
            ),
        )
        return sorted_nodes[:count]

    @staticmethod
    def age_based_cleanup(
        nodes: dict[str, Any], metadata: dict[str, dict], count: int
    ) -> list[str]:
        """Age-based cleanup (oldest first)"""
        sorted_nodes = sorted(
            nodes.keys(),
            key=lambda n: metadata.get(n, {}).get(
                "created_at", datetime.max.replace(tzinfo=timezone.utc)
            ),
        )
        return sorted_nodes[:count]

    @staticmethod
    def usage_based_cleanup(
        nodes: dict[str, Any], metadata: dict[str, dict], count: int
    ) -> list[str]:
        """Usage-based cleanup (least used first)"""
        sorted_nodes = sorted(
            nodes.keys(),
            key=lambda n: metadata.get(n, {}).get("access_count", float("inf")),
        )
        return sorted_nodes[:count]


class DependencyGraph:
    """Enhanced thread-safe graph for tracking dependencies and detecting cycles"""

    def __init__(
        self,
        max_nodes: int = 10000,
        cleanup_threshold: float = 0.8,
        cache_ttl: float = 5.0,
        enable_metrics: bool = True,
        prevent_cycles: bool = False,
    ):
        self.nodes: dict[str, set[str]] = {}
        self.reverse_edges: dict[str, set[str]] = {}
        self.node_metadata: dict[str, Any] = {}  # Store metadata directly as provided

        # Configuration
        self.max_nodes = max_nodes
        self.cleanup_threshold = cleanup_threshold
        self.cache_ttl = cache_ttl
        self.enable_metrics = enable_metrics
        self.prevent_cycles = prevent_cycles  # Option to prevent cycle creation

        # Thread safety
        self._lock = asyncio.Lock()
        self._operation_count = 0

        # Caching
        self._cycle_cache: dict[str, CycleDetectionResult] = {}
        self._topology_cache: Optional[tuple[list[str], str, float]] = None

        # Metrics
        self._metrics: dict[str, Union[int, float]] = {
            "operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cleanups_performed": 0,
            "nodes_cleaned": 0,
            "avg_detection_time_ms": 0.0,
        }

    async def add_dependency(
        self, node: str, depends_on: str, metadata: Optional[dict[str, Any]] = None
    ) -> bool:
        """Add a dependency edge with metadata and validation"""
        if not node or not depends_on:
            raise ValueError("Node and dependency names cannot be empty")

        # Allow self-loops for testing compatibility
        # if node == depends_on:
        #     raise ValueError("Node cannot depend on itself")

        async with self._lock:
            self._operation_count += 1

            # Check capacity and cleanup if needed
            if len(self.nodes) >= self.max_nodes * self.cleanup_threshold:
                await self._cleanup_old_nodes_internal()

            # Store metadata directly as provided by user
            if metadata is not None:
                self.node_metadata[node] = metadata

            # Check if this would create a cycle (optional)
            if self.prevent_cycles and self._would_create_cycle_sync(node, depends_on):
                return False

            # Add the dependency
            if node not in self.nodes:
                self.nodes[node] = set()
            if depends_on not in self.reverse_edges:
                self.reverse_edges[depends_on] = set()

            self.nodes[node].add(depends_on)
            self.reverse_edges[depends_on].add(node)

            # Invalidate caches
            self._invalidate_caches()

            if self.enable_metrics:
                self._metrics["operations"] += 1

            return True

    def _would_create_cycle_sync(self, from_node: str, to_node: str) -> bool:
        """Synchronous cycle check for use during dependency addition"""
        # Simple DFS to check if to_node can reach from_node
        visited = set()

        def dfs(node: str) -> bool:
            if node == from_node:
                return True
            if node in visited:
                return False
            visited.add(node)

            return any(dfs(neighbor) for neighbor in self.nodes.get(node, []))

        return dfs(to_node)

    async def remove_dependency(self, node: str, depends_on: str) -> bool:
        """Remove a dependency edge"""
        async with self._lock:
            return await self._remove_dependency_internal(node, depends_on)

    async def _remove_dependency_internal(self, node: str, depends_on: str) -> bool:
        """Internal method to remove dependency without acquiring lock"""
        if node not in self.nodes or depends_on not in self.nodes[node]:
            return False

        self.nodes[node].discard(depends_on)
        if not self.nodes[node]:
            del self.nodes[node]
            self.node_metadata.pop(node, None)

        if depends_on in self.reverse_edges:
            self.reverse_edges[depends_on].discard(node)
            if not self.reverse_edges[depends_on]:
                del self.reverse_edges[depends_on]

        self._invalidate_caches()
        return True

    async def remove_node(self, node: str) -> bool:
        """Remove a node and all its edges"""
        async with self._lock:
            return await self._remove_node_internal(node)

    async def _remove_node_internal(self, node: str) -> bool:
        """Internal method to remove node without acquiring lock"""
        removed = False

        # Remove outgoing edges
        if node in self.nodes:
            for dep in list(self.nodes[node]):
                if dep in self.reverse_edges:
                    self.reverse_edges[dep].discard(node)
                    if not self.reverse_edges[dep]:
                        del self.reverse_edges[dep]
            del self.nodes[node]
            removed = True

        # Remove incoming edges
        if node in self.reverse_edges:
            for dependent in list(self.reverse_edges[node]):
                if dependent in self.nodes:
                    self.nodes[dependent].discard(node)
                    if not self.nodes[dependent]:
                        del self.nodes[dependent]
                        self.node_metadata.pop(dependent, None)
            del self.reverse_edges[node]
            removed = True

        # Remove metadata
        if node in self.node_metadata:
            del self.node_metadata[node]
            removed = True

        if removed:
            self._invalidate_caches()

        return removed

    async def _cleanup_old_nodes_internal(self) -> int:
        """Internal cleanup method without acquiring lock"""
        if len(self.nodes) < self.max_nodes * self.cleanup_threshold:
            return 0

        target_size = int(self.max_nodes * 0.6)  # Clean to 60% capacity
        nodes_to_remove_count = len(self.nodes) - target_size

        if nodes_to_remove_count <= 0:
            return 0

        # Simple cleanup - remove oldest nodes
        nodes_to_remove = list(self.nodes.keys())[:nodes_to_remove_count]

        cleaned_count = 0
        for node in nodes_to_remove:
            if await self._remove_node_internal(node):
                cleaned_count += 1

        if self.enable_metrics:
            self._metrics["cleanups_performed"] += 1
            self._metrics["nodes_cleaned"] += cleaned_count

        logger.info(f"Cleaned up {cleaned_count} nodes from dependency graph")
        return cleaned_count

    def find_cycles(self, use_cache: bool = True) -> CycleDetectionResult:
        """Find all cycles in the graph using optimized DFS with proper cycle detection"""
        start_time = time.perf_counter()

        # Check cache first
        if use_cache:
            cache_key = self._get_graph_hash()
            cached_result = self._cycle_cache.get(cache_key)
            if cached_result and self._is_cache_valid(cached_result):
                if self.enable_metrics:
                    self._metrics["cache_hits"] += 1
                return cached_result

        if self.enable_metrics:
            self._metrics["cache_misses"] += 1

        # Perform cycle detection using proper DFS for directed graphs
        cycles = []
        visited = set()
        rec_stack = set()  # Recursion stack to track current path

        def dfs_detect_cycles(node: str, path: list[str]) -> None:
            # If node is in recursion stack, we found a cycle
            if node in rec_stack:
                # Find the cycle in the current path
                try:
                    cycle_start = path.index(node)
                    cycle = [*path[cycle_start:], node]
                    cycles.append(cycle)
                except ValueError:
                    # Fallback if node not found in path
                    cycles.append([node])
                return

            # If already visited but not in current path, skip
            if node in visited:
                return

            # Mark as visited and add to recursion stack
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            # Visit all neighbors
            for neighbor in self.nodes.get(node, []):
                dfs_detect_cycles(neighbor, path)

            # Backtrack: remove from recursion stack and path
            rec_stack.remove(node)
            path.pop()

        # Check all nodes to handle disconnected components
        for node in list(self.nodes.keys()):
            if node not in visited:
                dfs_detect_cycles(node, [])

        detection_duration = (time.perf_counter() - start_time) * 1000

        # Count edges
        edge_count = sum(len(deps) for deps in self.nodes.values())

        result = CycleDetectionResult(
            has_cycle=len(cycles) > 0,
            cycles=cycles,
            graph_size=len(self.nodes),
            edge_count=edge_count,
            detection_duration_ms=detection_duration,
            algorithm_used="dfs",
        )

        # Cache the result
        if use_cache:
            cache_key = self._get_graph_hash()
            self._cycle_cache[cache_key] = result

        # Update metrics
        if self.enable_metrics:
            alpha = 0.1
            self._metrics["avg_detection_time_ms"] = (
                alpha * detection_duration
                + (1 - alpha) * self._metrics["avg_detection_time_ms"]
            )

        return result

    def topological_sort(self) -> Optional[list[str]]:
        """Perform topological sort if no cycles exist"""
        # Check cache first
        if self._topology_cache:
            result, graph_hash, timestamp = self._topology_cache
            if (
                self._get_graph_hash() == graph_hash
                and time.time() - timestamp < self.cache_ttl
            ):
                return result

        # Check for cycles first - if cycles exist, no topological sort possible
        cycle_result = self.find_cycles()
        if cycle_result.has_cycle:
            return None

        # Get all unique nodes in the graph
        all_nodes = set(self.nodes.keys())
        for deps in self.nodes.values():
            all_nodes.update(deps)

        if not all_nodes:
            return []

        # Kahn's algorithm
        in_degree = dict.fromkeys(all_nodes, 0)

        # Calculate in-degrees
        for node in self.nodes:
            for dep in self.nodes[node]:
                in_degree[dep] += 1

        # Find nodes with no incoming edges
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # Remove edges from this node
            for neighbor in self.nodes.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Validate result - if not all nodes processed, there was a cycle
        if len(result) != len(all_nodes):
            return None

        # Cache the result
        self._topology_cache = (result, self._get_graph_hash(), time.time())
        return result

    def _get_graph_hash(self) -> str:
        """Get a hash representing the current graph state"""
        edge_count = sum(len(deps) for deps in self.nodes.values())
        return f"{len(self.nodes)}:{edge_count}:{self._operation_count}"

    def _is_cache_valid(self, result: CycleDetectionResult) -> bool:
        """Check if cached result is still valid"""
        return (time.time() - result.detection_time.timestamp()) < self.cache_ttl

    def _invalidate_caches(self) -> None:
        """Invalidate all caches"""
        self._cycle_cache.clear()
        self._topology_cache = None

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics"""
        return {
            **self._metrics,
            "node_count": len(self.nodes),
            "edge_count": sum(len(deps) for deps in self.nodes.values()),
            "cache_size": len(self._cycle_cache),
            "operation_count": self._operation_count,
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check"""
        async with self._lock:
            return {
                "status": "healthy",
                "node_count": len(self.nodes),
                "memory_usage_percent": len(self.nodes) / self.max_nodes * 100,
                "cache_hit_rate": (
                    (
                        self._metrics["cache_hits"]
                        / max(
                            1,
                            self._metrics["cache_hits"] + self._metrics["cache_misses"],
                        )
                    )
                    * 100
                    if self.enable_metrics
                    else 0
                ),
                "last_cleanup": self._metrics.get("last_cleanup"),
                "needs_cleanup": len(self.nodes)
                >= self.max_nodes * self.cleanup_threshold,
            }


class ResourceWaitGraph:
    """Enhanced wait-for graph for resource-based deadlock detection"""

    def __init__(
        self,
        max_resources: int = 5000,
        max_processes: int = 5000,
        cleanup_interval: float = 300.0,
        enable_timeouts: bool = True,
    ):
        self.resources: dict[str, ResourceNode] = {}
        self.processes: dict[str, ProcessNode] = {}

        # Configuration
        self.max_resources = max_resources
        self.max_processes = max_processes
        self.cleanup_interval = cleanup_interval
        self.enable_timeouts = enable_timeouts

        # Thread safety
        self._lock = asyncio.Lock()

        # Caching and optimization
        self._wait_graph_cache: Optional[DependencyGraph] = None
        self._cache_invalidated = True
        self._last_cleanup = datetime.now(timezone.utc)

        # Metrics
        self._metrics: dict[str, int] = {
            "resource_acquisitions": 0,
            "resource_releases": 0,
            "deadlock_detections": 0,
            "timeouts": 0,
            "preemptions": 0,
        }

    async def add_resource(
        self,
        resource_id: str,
        resource_type: str = "generic",
        max_holders: int = 1,
        priority: int = 0,
    ) -> bool:
        """Add a resource to the graph with configuration"""
        if not resource_id:
            raise ValueError("Resource ID cannot be empty")

        async with self._lock:
            if len(self.resources) >= self.max_resources:
                await self._cleanup_old_resources_internal()

            if resource_id not in self.resources:
                self.resources[resource_id] = ResourceNode(
                    resource_id=resource_id,
                    resource_type=resource_type,
                    max_holders=max_holders,
                    priority=priority,
                )
                self._cache_invalidated = True
                return True
            return False

    async def add_process(
        self,
        process_id: str,
        process_name: str = "",
        priority: int = 0,
        timeout: Optional[float] = None,
    ) -> bool:
        """Add a process to the graph with configuration"""
        if not process_id:
            raise ValueError("Process ID cannot be empty")

        async with self._lock:
            if len(self.processes) >= self.max_processes:
                await self._cleanup_old_processes_internal()

            if process_id not in self.processes:
                self.processes[process_id] = ProcessNode(
                    process_id=process_id,
                    process_name=process_name or process_id,
                    priority=priority,
                    timeout=timeout,
                )
                self._cache_invalidated = True
                return True
            return False

    async def acquire_resource(
        self,
        process_id: str,
        resource_id: str,
        count: int = 1,
        timeout: Optional[float] = None,
    ) -> bool:
        """Process attempts to acquire a resource with optional timeout"""
        async with self._lock:
            return await self._acquire_resource_internal(
                process_id, resource_id, count, timeout
            )

    async def _acquire_resource_internal(
        self,
        process_id: str,
        resource_id: str,
        count: int = 1,
        timeout: Optional[float] = None,
    ) -> bool:
        """Internal method to acquire resource without acquiring lock"""
        if count <= 0:
            raise ValueError("Count must be positive")

        # Ensure resource and process exist
        if resource_id not in self.resources:
            self.resources[resource_id] = ResourceNode(
                resource_id=resource_id, resource_type="generic"
            )
            self._cache_invalidated = True

        if process_id not in self.processes:
            self.processes[process_id] = ProcessNode(
                process_id=process_id, process_name=process_id, timeout=timeout
            )
            self._cache_invalidated = True

        resource = self.resources[resource_id]
        process = self.processes[process_id]

        # Check if resource can be acquired
        if resource.can_acquire(count) and process_id not in resource.waiters:
            # Successful acquisition - use simple process ID for holders
            resource.holders.add(process_id)

            resource.acquired_at = datetime.now(timezone.utc)
            resource.access_count += 1
            resource.update_access()

            process.holding.add(resource_id)
            process.waiting_for.discard(resource_id)
            process.update_activity()

            # Clear blocked status if not waiting for anything
            if not process.waiting_for:
                process.blocked_at = None

            self._cache_invalidated = True
            self._metrics["resource_acquisitions"] += 1
            return True
        else:
            # Must wait
            resource.waiters.add(process_id)
            process.waiting_for.add(resource_id)
            if process.blocked_at is None:
                process.blocked_at = datetime.now(timezone.utc)

            self._cache_invalidated = True
            return False

    async def release_resource(
        self, process_id: str, resource_id: str, count: int = 1
    ) -> bool:
        """Process releases a resource"""
        if count <= 0:
            raise ValueError("Count must be positive")

        async with self._lock:
            if resource_id not in self.resources or process_id not in self.processes:
                return False

            resource = self.resources[resource_id]
            process = self.processes[process_id]

            # Release the resource
            resource.holders.discard(process_id)
            process.holding.discard(resource_id)
            process.update_activity()
            resource.update_access()

            # Try to wake up waiters (using internal method)
            await self._process_waiters_internal(resource_id)

            self._cache_invalidated = True
            self._metrics["resource_releases"] += 1
            return True

    async def _process_waiters_internal(self, resource_id: str) -> None:
        """Process waiting list for a resource (internal method)"""
        if resource_id not in self.resources:
            return

        resource = self.resources[resource_id]

        # Sort waiters by priority and wait time
        if resource.waiters:
            sorted_waiters = sorted(
                resource.waiters,
                key=lambda pid: (
                    -self.processes.get(pid, ProcessNode("", "")).priority,
                    self.processes.get(pid, ProcessNode("", "")).blocked_at
                    or datetime.max.replace(tzinfo=timezone.utc),
                ),
            )

            # Try to satisfy waiters using internal method
            for waiter_id in list(sorted_waiters):
                if resource.can_acquire(1):
                    resource.waiters.remove(waiter_id)
                    # Use internal method to avoid lock acquisition
                    await self._acquire_resource_internal(waiter_id, resource_id)
                else:
                    break

    async def detect_deadlock(self) -> CycleDetectionResult:
        """Detect deadlocks using wait-for graph analysis"""
        async with self._lock:
            # Check for timeouts first
            if self.enable_timeouts:
                await self._handle_timeouts_internal()

            # Build or reuse wait-for graph
            if self._cache_invalidated or self._wait_graph_cache is None:
                self._wait_graph_cache = DependencyGraph()

                # Add edges: if P1 waits for resource held by P2, add edge P1 -> P2
                for resource in self.resources.values():
                    for waiter in resource.waiters:
                        for holder in resource.holders:
                            if waiter != holder:
                                await self._wait_graph_cache.add_dependency(
                                    waiter, holder
                                )

                self._cache_invalidated = False

            # Find cycles
            result = self._wait_graph_cache.find_cycles()
            self._metrics["deadlock_detections"] += 1
            return result

    async def _handle_timeouts_internal(self) -> None:
        """Handle process timeouts (internal method)"""
        timed_out_processes = []

        for process in self.processes.values():
            if process.is_timed_out():
                timed_out_processes.append(process.process_id)

        for process_id in timed_out_processes:
            await self._timeout_process_internal(process_id)
            self._metrics["timeouts"] += 1

    async def _timeout_process_internal(self, process_id: str) -> None:
        """Handle process timeout (internal method)"""
        if process_id not in self.processes:
            return

        process = self.processes[process_id]

        # Remove from all waiting lists
        for resource_id in list(process.waiting_for):
            if resource_id in self.resources:
                self.resources[resource_id].waiters.discard(process_id)

        process.waiting_for.clear()
        process.blocked_at = None

        logger.warning(f"Process {process_id} timed out after waiting")

    async def _cleanup_old_resources_internal(self) -> int:
        """Clean up old unused resources (internal method)"""
        now = datetime.now(timezone.utc)
        cleanup_threshold = timedelta(seconds=self.cleanup_interval)

        old_resources = [
            rid
            for rid, resource in self.resources.items()
            if (
                resource.is_free()
                and len(resource.waiters) == 0
                and now - resource.last_accessed > cleanup_threshold
            )
        ]

        cleaned_count = 0
        for rid in old_resources[: len(self.resources) // 4]:  # Remove 25%
            del self.resources[rid]
            cleaned_count += 1

        self._last_cleanup = now
        return cleaned_count

    async def _cleanup_old_processes_internal(self) -> int:
        """Clean up old inactive processes (internal method)"""
        now = datetime.now(timezone.utc)
        cleanup_threshold = timedelta(seconds=self.cleanup_interval)

        old_processes = [
            pid
            for pid, process in self.processes.items()
            if (
                len(process.holding) == 0
                and len(process.waiting_for) == 0
                and now - process.last_activity > cleanup_threshold
            )
        ]

        cleaned_count = 0
        for pid in old_processes[: len(self.processes) // 4]:  # Remove 25%
            del self.processes[pid]
            cleaned_count += 1

        return cleaned_count

    def get_blocked_processes(self) -> list[ProcessNode]:
        """Get all currently blocked processes"""
        return [proc for proc in self.processes.values() if proc.is_blocked()]

    def get_resource_holders(self, resource_id: str) -> set[str]:
        """Get processes holding a resource"""
        if resource_id in self.resources:
            return self.resources[resource_id].holders.copy()
        return set()

    def get_resource_waiters(self, resource_id: str) -> set[str]:
        """Get processes waiting for a resource"""
        if resource_id in self.resources:
            return self.resources[resource_id].waiters.copy()
        return set()

    def get_resource_stats(self) -> dict[str, Any]:
        """Get comprehensive resource statistics"""
        total_resources = len(self.resources)
        free_resources = sum(1 for r in self.resources.values() if r.is_free())
        total_holders = sum(len(r.holders) for r in self.resources.values())
        total_waiters = sum(len(r.waiters) for r in self.resources.values())

        return {
            "total_resources": total_resources,
            "free_resources": free_resources,
            "utilized_resources": total_resources - free_resources,
            "total_holders": total_holders,
            "total_waiters": total_waiters,
            "average_utilization": (total_resources - free_resources)
            / max(1, total_resources),
            "blocked_processes": len(self.get_blocked_processes()),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics"""
        return {
            **self._metrics,
            **self.get_resource_stats(),
            "total_processes": len(self.processes),
        }


class DeadlockDetector:
    """Production-grade deadlock detection with comprehensive monitoring and resolution"""

    def __init__(
        self,
        agent: Any,
        detection_interval: float = 1.0,
        max_cycles: int = 100,
        resolution_strategy: DeadlockResolutionStrategy = DeadlockResolutionStrategy.LOG_ONLY,
        enable_metrics: bool = True,
        enable_health_monitoring: bool = True,
        max_resolution_attempts: int = 3,
    ):
        self.agent = weakref.proxy(agent) if agent else None
        self.detection_interval = detection_interval
        self.max_cycles = max_cycles
        self.resolution_strategy = resolution_strategy
        self.enable_metrics = enable_metrics
        self.enable_health_monitoring = enable_health_monitoring
        self.max_resolution_attempts = max_resolution_attempts

        # Core components
        self._dependency_graph = DependencyGraph(enable_metrics=enable_metrics)
        self._resource_graph = ResourceWaitGraph()

        # Control and synchronization
        self._lock = asyncio.Lock()
        self._detection_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # State tracking
        self._cycle_count = 0
        self._last_cycle: Optional[list[str]] = None
        self._detection_history: deque = deque(maxlen=1000)
        self._resolution_history: deque = deque(maxlen=100)

        # Metrics and monitoring
        self._metrics: dict[str, Union[int, float]] = {
            "total_detections": 0,
            "deadlocks_found": 0,
            "deadlocks_resolved": 0,
            "detection_errors": 0,
            "resolution_failures": 0,
            "avg_detection_time_ms": 0.0,
            "uptime_seconds": 0.0,
            "last_error": "",  # type: ignore
        }

        # Callbacks and extensibility
        self._resolution_callbacks: list[Callable[[list[str]], bool]] = []
        self._notification_callbacks: list[Callable[[str, dict[str, Any]], None]] = []

        # Health monitoring
        self._health_status = "initializing"
        self._last_successful_detection = datetime.now(timezone.utc)
        self._start_time = datetime.now(timezone.utc)

    async def start(self) -> bool:
        """Start deadlock detection with comprehensive initialization"""
        try:
            async with self._lock:
                if self._detection_task and not self._detection_task.done():
                    logger.warning("Deadlock detector already running")
                    return False

                self._shutdown_event.clear()
                self._health_status = "starting"

                # Start detection task
                self._detection_task = asyncio.create_task(self._detection_loop())

                # Start health monitoring if enabled
                if self.enable_health_monitoring:
                    self._health_task = asyncio.create_task(
                        self._health_monitoring_loop()
                    )

                self._health_status = "running"
                self._start_time = datetime.now(timezone.utc)

                logger.info(
                    f"Deadlock detector started with strategy: {self.resolution_strategy.name}"
                )
                await self._notify(
                    "deadlock_detector_started",
                    {"strategy": self.resolution_strategy.name},
                )

                return True

        except Exception as e:
            self._health_status = "error"
            self._metrics["last_error"] = str(e)  # type: ignore
            logger.error(f"Failed to start deadlock detector: {e}")
            return False

    async def stop(self, timeout: float = 10.0) -> bool:
        """Stop deadlock detection gracefully"""
        try:
            async with self._lock:
                self._health_status = "stopping"
                self._shutdown_event.set()

                # Cancel tasks
                tasks_to_cancel = []
                if self._detection_task:
                    tasks_to_cancel.append(self._detection_task)
                if self._health_task:
                    tasks_to_cancel.append(self._health_task)

                if tasks_to_cancel:
                    for task in tasks_to_cancel:
                        task.cancel()

                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*tasks_to_cancel, return_exceptions=True),
                            timeout=timeout,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Some tasks did not stop gracefully within timeout"
                        )

                self._detection_task = None
                self._health_task = None
                self._health_status = "stopped"

                logger.info("Deadlock detector stopped")
                await self._notify("deadlock_detector_stopped", {})
                return True

        except Exception as e:
            self._health_status = "error"
            logger.error(f"Error stopping deadlock detector: {e}")
            return False

    async def _detection_loop(self) -> None:
        """Main detection loop with comprehensive error handling"""
        consecutive_errors = 0
        max_consecutive_errors = 5

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Wait for next detection cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=self.detection_interval
                    )
                    if self._shutdown_event.is_set():
                        break

                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue with detection

                detection_start = time.perf_counter()

                try:
                    # Perform detection
                    await self._perform_detection_cycle()

                    # Update metrics
                    detection_duration = (time.perf_counter() - detection_start) * 1000
                    self._update_detection_metrics(detection_duration)

                    # Reset error counter on successful detection
                    consecutive_errors = 0
                    self._last_successful_detection = datetime.now(timezone.utc)

                except Exception as detection_error:
                    consecutive_errors += 1
                    self._metrics["detection_errors"] += 1
                    self._metrics["last_error"] = str(detection_error)  # type: ignore

                    logger.error(f"Detection cycle error: {detection_error}")

                    # Implement exponential backoff on errors
                    if consecutive_errors >= max_consecutive_errors:
                        logger.critical(
                            f"Too many consecutive errors ({consecutive_errors}), stopping detection"
                        )
                        self._health_status = "error"
                        break

                    # Exponential backoff with jitter
                    error_delay = min(
                        self.detection_interval
                        * (2**consecutive_errors)
                        * (0.5 + 0.5 * time.time() % 1),
                        60.0,
                    )
                    await asyncio.sleep(error_delay)

        except asyncio.CancelledError:
            logger.info("Detection loop cancelled")
        except Exception as e:
            logger.critical(f"Unexpected error in detection loop: {e}")
            self._health_status = "error"
            self._metrics["last_error"] = str(e)  # type: ignore

    async def _perform_detection_cycle(self) -> None:
        """Perform a single detection cycle"""
        self._metrics["total_detections"] += 1

        # Check state dependencies
        state_result = self._dependency_graph.find_cycles()
        if state_result.has_cycle:
            await self._handle_deadlock_detection(state_result, "dependency_graph")

        # Check resource wait graph
        resource_result = await self._resource_graph.detect_deadlock()
        if resource_result.has_cycle:
            await self._handle_deadlock_detection(resource_result, "resource_graph")

        # Keep detection history
        self._detection_history.append(
            {
                "timestamp": datetime.now(timezone.utc),
                "state_cycles": len(state_result.cycles),
                "resource_cycles": len(resource_result.cycles),
                "total_cycles": len(state_result.cycles) + len(resource_result.cycles),
            }
        )

    async def _handle_deadlock_detection(
        self, result: CycleDetectionResult, source: str
    ) -> None:
        """Handle detected deadlock with enhanced resolution logic"""
        self._cycle_count += 1
        self._last_cycle = result.get_critical_cycle()
        self._metrics["deadlocks_found"] += 1

        detection_id = str(uuid.uuid4())

        logger.error(
            f"Deadlock detected from {source} (ID: {detection_id}): "
            f"cycle_count={self._cycle_count}, "
            f"cycle={self._last_cycle}, "
            f"total_cycles={len(result.cycles)}"
        )

        # Notify callbacks
        await self._notify(
            "deadlock_detected",
            {
                "detection_id": detection_id,
                "source": source,
                "cycle": self._last_cycle,
                "total_cycles": len(result.cycles),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        # Attempt resolution
        resolution_attempts = 0
        resolved = False

        while resolution_attempts < self.max_resolution_attempts and not resolved:
            resolution_attempts += 1

            try:
                # Try custom callbacks first
                for callback in self._resolution_callbacks:
                    try:
                        if self._last_cycle and await self._run_callback_safely(
                            callback, self._last_cycle
                        ):
                            resolved = True
                            self._metrics["deadlocks_resolved"] += 1
                            logger.info(
                                f"Deadlock {detection_id} resolved by custom callback (attempt {resolution_attempts})"
                            )
                            break
                    except Exception as e:
                        logger.error(f"Resolution callback failed: {e}")

                # Apply configured strategy if not resolved
                if not resolved and self._last_cycle:
                    resolved = await self._apply_resolution_strategy(
                        self._last_cycle, detection_id
                    )

                if resolved:
                    break

            except Exception as e:
                logger.error(f"Resolution attempt {resolution_attempts} failed: {e}")

            # Wait before retry
            if resolution_attempts < self.max_resolution_attempts:
                await asyncio.sleep(0.1 * resolution_attempts)  # Progressive delay

        # Record resolution outcome
        self._resolution_history.append(
            {
                "detection_id": detection_id,
                "cycle": self._last_cycle,
                "resolved": resolved,
                "attempts": resolution_attempts,
                "strategy": self.resolution_strategy.name,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        if not resolved:
            self._metrics["resolution_failures"] += 1

            # Raise exception if strategy requires it
            if self.resolution_strategy == DeadlockResolutionStrategy.RAISE_EXCEPTION:
                if self._last_cycle:
                    raise DeadlockError(self._last_cycle, detection_id)
                else:
                    raise DeadlockError([], detection_id)

    # Add the missing method alias for backward compatibility
    async def _handle_deadlock(
        self, result: CycleDetectionResult, source: str = "test"
    ) -> None:
        """Handle detected deadlock (alias for backward compatibility)"""
        return await self._handle_deadlock_detection(result, source)

    async def _run_callback_safely(self, callback: Callable, cycle: list[str]) -> bool:
        """Run callback safely with timeout"""
        try:
            if asyncio.iscoroutinefunction(callback):
                return await asyncio.wait_for(callback(cycle), timeout=5.0)
            else:
                # Run sync callback in thread pool
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, callback, cycle)
        except asyncio.TimeoutError:
            logger.warning("Resolution callback timed out")
            return False

    async def _apply_resolution_strategy(
        self, cycle: list[str], detection_id: str
    ) -> bool:
        """Apply the configured resolution strategy"""
        try:
            if self.resolution_strategy == DeadlockResolutionStrategy.LOG_ONLY:
                return True  # Just log, consider resolved

            elif self.resolution_strategy == DeadlockResolutionStrategy.KILL_YOUNGEST:
                return await self._kill_youngest_process(cycle, detection_id)

            elif self.resolution_strategy == DeadlockResolutionStrategy.KILL_OLDEST:
                return await self._kill_oldest_process(cycle, detection_id)

            elif (
                self.resolution_strategy
                == DeadlockResolutionStrategy.KILL_LOWEST_PRIORITY
            ):
                return await self._kill_lowest_priority_process(cycle, detection_id)

            elif (
                self.resolution_strategy == DeadlockResolutionStrategy.PREEMPT_RESOURCES
            ):
                return await self._preempt_resources(cycle, detection_id)

            return False

        except Exception as e:
            logger.error(
                f"Resolution strategy {self.resolution_strategy.name} failed: {e}"
            )
            return False

    async def _kill_youngest_process(self, cycle: list[str], detection_id: str) -> bool:
        """Kill the youngest process in the cycle"""
        try:
            valid_processes = [
                pid for pid in cycle if pid in self._resource_graph.processes
            ]
            if not valid_processes:
                return False

            youngest_process = min(
                valid_processes,
                key=lambda pid: self._resource_graph.processes[pid].age_seconds(),
            )

            await self._terminate_process(
                youngest_process, f"deadlock_resolution_{detection_id}"
            )
            logger.info(
                f"Killed youngest process {youngest_process} to resolve deadlock {detection_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to kill youngest process: {e}")
            return False

    async def _kill_oldest_process(self, cycle: list[str], detection_id: str) -> bool:
        """Kill the oldest process in the cycle"""
        try:
            valid_processes = [
                pid for pid in cycle if pid in self._resource_graph.processes
            ]
            if not valid_processes:
                return False

            oldest_process = max(
                valid_processes,
                key=lambda pid: self._resource_graph.processes[pid].age_seconds(),
            )

            await self._terminate_process(
                oldest_process, f"deadlock_resolution_{detection_id}"
            )
            logger.info(
                f"Killed oldest process {oldest_process} to resolve deadlock {detection_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to kill oldest process: {e}")
            return False

    async def _kill_lowest_priority_process(
        self, cycle: list[str], detection_id: str
    ) -> bool:
        """Kill the lowest priority process in the cycle"""
        try:
            valid_processes = [
                pid for pid in cycle if pid in self._resource_graph.processes
            ]
            if not valid_processes:
                return False

            lowest_priority_process = min(
                valid_processes,
                key=lambda pid: self._resource_graph.processes[pid].priority,
            )

            await self._terminate_process(
                lowest_priority_process, f"deadlock_resolution_{detection_id}"
            )
            logger.info(
                f"Killed lowest priority process {lowest_priority_process} to resolve deadlock {detection_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to kill lowest priority process: {e}")
            return False

    async def _preempt_resources(self, cycle: list[str], detection_id: str) -> bool:
        """Preempt resources from processes in the cycle"""
        try:
            valid_processes = [
                pid for pid in cycle if pid in self._resource_graph.processes
            ]
            if not valid_processes:
                return False

            # Find process with most resources to preempt from
            victim_process = max(
                valid_processes,
                key=lambda pid: len(self._resource_graph.processes[pid].holding),
            )

            process = self._resource_graph.processes[victim_process]
            resources_to_preempt = list(process.holding)

            # Release all resources held by victim process
            for resource_id in resources_to_preempt:
                await self._resource_graph.release_resource(victim_process, resource_id)

            logger.info(
                f"Preempted {len(resources_to_preempt)} resources from process "
                f"{victim_process} to resolve deadlock {detection_id}"
            )
            self._resource_graph._metrics["preemptions"] += 1
            return True

        except Exception as e:
            logger.error(f"Failed to preempt resources: {e}")
            return False

    async def _terminate_process(self, process_id: str, reason: str) -> None:
        """Terminate a process and clean up its resources"""
        try:
            if process_id in self._resource_graph.processes:
                process = self._resource_graph.processes[process_id]

                # Release all held resources
                for resource_id in list(process.holding):
                    await self._resource_graph.release_resource(process_id, resource_id)

                # Remove from waiting lists
                for resource_id in list(process.waiting_for):
                    if resource_id in self._resource_graph.resources:
                        self._resource_graph.resources[resource_id].waiters.discard(
                            process_id
                        )

                # Remove process
                del self._resource_graph.processes[process_id]

                # Remove from dependency graph
                await self._dependency_graph.remove_node(process_id)

                logger.info(f"Terminated process {process_id}, reason: {reason}")

        except Exception as e:
            logger.error(f"Failed to terminate process {process_id}: {e}")

    async def _health_monitoring_loop(self) -> None:
        """Health monitoring loop"""
        try:
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=30.0)
                    if self._shutdown_event.is_set():
                        break
                except asyncio.TimeoutError:
                    pass

                # Perform health checks
                await self._perform_health_checks()

        except asyncio.CancelledError:
            logger.info("Health monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")

    async def _perform_health_checks(self) -> None:
        """Perform comprehensive health checks"""
        try:
            now = datetime.now(timezone.utc)

            # Check if detection is stuck
            time_since_last_detection = (
                now - self._last_successful_detection
            ).total_seconds()
            if (
                time_since_last_detection > self.detection_interval * 10
            ):  # 10x normal interval
                self._health_status = "degraded"
                logger.warning(
                    f"No successful detection in {time_since_last_detection:.1f} seconds"
                )

            # Check graph health
            dep_health = await self._dependency_graph.health_check()
            resource_health = self._resource_graph.get_metrics()

            # Update uptime
            self._metrics["uptime_seconds"] = (now - self._start_time).total_seconds()

            # Log health status periodically
            if int(time.time()) % 300 == 0:  # Every 5 minutes
                logger.info(
                    f"Health check: status={self._health_status}, "
                    f"dep_nodes={dep_health['node_count']}, "
                    f"resources={resource_health['total_resources']}, "
                    f"blocked_processes={resource_health['blocked_processes']}"
                )

        except Exception as e:
            logger.error(f"Health check failed: {e}")

    async def _notify(self, event: str, data: dict[str, Any]) -> None:
        """Send notifications to registered callbacks"""
        for callback in self._notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, data)
                else:
                    callback(event, data)
            except Exception as e:
                logger.error(f"Notification callback failed for event {event}: {e}")

    def _update_detection_metrics(self, duration_ms: float) -> None:
        """Update detection performance metrics"""
        if self.enable_metrics:
            # Exponential moving average
            alpha = 0.1
            self._metrics["avg_detection_time_ms"] = (
                alpha * duration_ms
                + (1 - alpha) * self._metrics["avg_detection_time_ms"]
            )

    # Public API methods

    def add_resolution_callback(self, callback: Callable[[list[str]], bool]) -> None:
        """Add a callback for custom deadlock resolution"""
        self._resolution_callbacks.append(callback)

    def add_notification_callback(
        self, callback: Callable[[str, dict[str, Any]], None]
    ) -> None:
        """Add a callback for event notifications"""
        self._notification_callbacks.append(callback)

    async def add_dependency(
        self, from_state: str, to_state: str, metadata: Optional[dict[str, Any]] = None
    ) -> bool:
        """Add a dependency between states"""
        return await self._dependency_graph.add_dependency(
            from_state, to_state, metadata
        )

    async def remove_dependency(self, from_state: str, to_state: str) -> bool:
        """Remove a dependency between states"""
        return await self._dependency_graph.remove_dependency(from_state, to_state)

    async def acquire_resource(
        self,
        process_id: str,
        resource_id: str,
        process_name: Optional[str] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
    ) -> bool:
        """Process attempts to acquire a resource"""
        if process_name:
            await self._resource_graph.add_process(
                process_id, process_name, priority, timeout
            )

        success = await self._resource_graph.acquire_resource(
            process_id, resource_id, timeout=timeout
        )

        # Immediate deadlock check after failed acquisition
        if not success:
            try:
                result = await self._resource_graph.detect_deadlock()
                if result.has_cycle:
                    await self._handle_deadlock_detection(result, "immediate_check")
            except Exception as e:
                logger.error(f"Error during immediate deadlock check: {e}")

        return success

    async def release_resource(self, process_id: str, resource_id: str) -> bool:
        """Process releases a resource"""
        return await self._resource_graph.release_resource(process_id, resource_id)

    def get_comprehensive_status(self) -> dict[str, Any]:
        """Get comprehensive detector status"""
        return {
            # Basic status
            "active": bool(self._detection_task and not self._detection_task.done()),
            "health_status": self._health_status,
            "cycle_count": self._cycle_count,
            "last_cycle": self._last_cycle,
            # Missing fields that tests expect
            "graph_size": len(self._dependency_graph.nodes),
            "resource_count": len(self._resource_graph.resources),
            "process_count": len(self._resource_graph.processes),
            "blocked_processes": len(self._resource_graph.get_blocked_processes()),
            # Configuration
            "detection_interval": self.detection_interval,
            "resolution_strategy": self.resolution_strategy.name,
            "max_resolution_attempts": self.max_resolution_attempts,
            # Graph statistics
            "dependency_graph": self._dependency_graph.get_metrics(),
            "resource_graph": self._resource_graph.get_metrics(),
            # Performance metrics
            "metrics": self._metrics.copy(),
            # Recent activity
            "recent_detections": len(
                [
                    h
                    for h in self._detection_history
                    if (datetime.now(timezone.utc) - h["timestamp"]).total_seconds()
                    < 300
                ]
            ),
            "recent_resolutions": len(
                [
                    r
                    for r in self._resolution_history
                    if (datetime.now(timezone.utc) - r["timestamp"]).total_seconds()
                    < 300
                ]
            ),
            # Health indicators
            "last_successful_detection": self._last_successful_detection.isoformat(),
            "time_since_last_detection": (
                datetime.now(timezone.utc) - self._last_successful_detection
            ).total_seconds(),
        }

    async def force_detection(self) -> CycleDetectionResult:
        """Force an immediate deadlock detection"""
        try:
            # Check both graphs
            state_result = self._dependency_graph.find_cycles()
            resource_result = await self._resource_graph.detect_deadlock()

            # Return combined result
            all_cycles = state_result.cycles + resource_result.cycles

            return CycleDetectionResult(
                has_cycle=len(all_cycles) > 0,
                cycles=all_cycles,
                graph_size=state_result.graph_size + resource_result.graph_size,
                detection_duration_ms=max(
                    state_result.detection_duration_ms,
                    resource_result.detection_duration_ms,
                ),
                algorithm_used="combined",
            )

        except Exception as e:
            logger.error(f"Force detection failed: {e}")
            raise

    async def export_state(self) -> dict[str, Any]:
        """Export current state for debugging/analysis"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": self.get_comprehensive_status(),
            "dependency_graph": self._dependency_graph.get_metrics(),
            "resource_graph": {
                "processes": {
                    pid: {
                        "name": proc.process_name,
                        "holding": list(proc.holding),
                        "waiting_for": list(proc.waiting_for),
                        "priority": proc.priority,
                        "blocked_duration": proc.blocked_duration_seconds(),
                    }
                    for pid, proc in self._resource_graph.processes.items()
                },
                "resources": {
                    rid: {
                        "type": res.resource_type,
                        "holders": list(res.holders),
                        "waiters": list(res.waiters),
                        "access_count": res.access_count,
                    }
                    for rid, res in self._resource_graph.resources.items()
                },
            },
            "detection_history": list(self._detection_history)[
                -10:
            ],  # Last 10 detections
            "resolution_history": list(self._resolution_history)[
                -10:
            ],  # Last 10 resolutions
        }

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive detector status (alias for backward compatibility)"""
        return self.get_comprehensive_status()

    def get_dependency_graph(self) -> dict[str, set[str]]:
        """Get current dependency graph"""
        return dict(self._dependency_graph.nodes)

    def get_wait_graph(self) -> dict[str, dict[str, Any]]:
        """Get current wait-for graph with enhanced information"""
        graph = {}

        for process_id, process in self._resource_graph.processes.items():
            graph[process_id] = {
                "name": process.process_name,
                "holding": list(process.holding),
                "waiting_for": list(process.waiting_for),
                "blocked": process.is_blocked(),
                "blocked_duration_seconds": process.blocked_duration_seconds(),
                "age_seconds": process.age_seconds(),
                "priority": process.priority,
                "last_activity": process.last_activity.isoformat(),
            }

        return graph

    def find_potential_deadlocks(self) -> list[tuple[str, str]]:
        """Find potential deadlock situations before they occur"""
        potential = []

        # Check for circular wait conditions
        for p1_id, p1 in self._resource_graph.processes.items():
            for p2_id, p2 in self._resource_graph.processes.items():
                if p1_id == p2_id:
                    continue

                # Check if P1 holds what P2 wants and vice versa
                p1_holds_p2_wants = bool(p1.holding & p2.waiting_for)
                p2_holds_p1_wants = bool(p2.holding & p1.waiting_for)

                if p1_holds_p2_wants and p2_holds_p1_wants:
                    potential.append((p1_id, p2_id))

        return potential

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive metrics"""
        return {
            **self._metrics,
            "detection_history_length": len(self._detection_history),
            "active_processes": len(self._resource_graph.processes),
            "active_resources": len(self._resource_graph.resources),
            "blocked_processes": len(self._resource_graph.get_blocked_processes()),
        }

    # Context manager support
    async def __aenter__(self) -> "DeadlockDetector":
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Async context manager exit"""
        await self.stop()
