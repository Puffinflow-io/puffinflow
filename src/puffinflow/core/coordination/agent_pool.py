"""Agent pool with dynamic scaling capabilities."""

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from ..agent.base import Agent, AgentResult

logger = logging.getLogger(__name__)


class ScalingPolicy(Enum):
    """Scaling policy options."""

    MANUAL = "manual"
    AUTO_CPU = "auto_cpu"
    AUTO_QUEUE = "auto_queue"
    AUTO_LATENCY = "auto_latency"
    CUSTOM = "custom"


@dataclass
class WorkItem:
    """Work item for agent processing."""

    id: str
    data: Any
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    assigned_at: Optional[float] = None
    completed_at: Optional[float] = None
    agent_name: Optional[str] = None
    retries: int = 0
    max_retries: int = 3

    @property
    def processing_time(self) -> Optional[float]:
        """Get processing time if completed."""
        if self.assigned_at and self.completed_at:
            return self.completed_at - self.assigned_at
        return None

    @property
    def wait_time(self) -> Optional[float]:
        """Get time spent waiting in queue."""
        if self.assigned_at:
            return self.assigned_at - self.created_at
        return time.time() - self.created_at


@dataclass
class CompletedWork:
    """Completed work result."""

    work_item: WorkItem
    agent: Agent
    result: AgentResult
    success: bool


class WorkQueue:
    """Priority queue for work items."""

    def __init__(self, max_size: Optional[int] = None):
        self._queue: deque[WorkItem] = deque()
        self._max_size = max_size
        self._priority_queue: list[tuple[int, WorkItem]] = []
        self._use_priority = False

    def add_work(self, work_item: WorkItem) -> bool:
        """Add work item to queue."""
        if self._max_size and len(self._queue) >= self._max_size:
            return False

        if work_item.priority > 0:
            self._use_priority = True
            import heapq

            heapq.heappush(self._priority_queue, (-work_item.priority, work_item))
        else:
            self._queue.append(work_item)

        return True

    def get_work(self) -> Optional[WorkItem]:
        """Get next work item."""
        # Try priority queue first
        if self._priority_queue:
            import heapq

            priority_tuple = heapq.heappop(self._priority_queue)
            _, work_item = priority_tuple
            work_item.assigned_at = time.time()
            return work_item

        # Then regular queue
        if self._queue:
            work_item = self._queue.popleft()
            work_item.assigned_at = time.time()
            return work_item

        return None

    def size(self) -> int:
        """Get queue size."""
        return len(self._queue) + len(self._priority_queue)

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0 and len(self._priority_queue) == 0

    def clear(self) -> None:
        """Clear all work items."""
        self._queue.clear()
        self._priority_queue.clear()


class AgentPool:
    """Agent pool with dynamic scaling."""

    def __init__(
        self,
        agent_factory: Callable[[int], Agent],
        min_size: int = 1,
        max_size: int = 10,
        scaling_policy: ScalingPolicy = ScalingPolicy.AUTO_QUEUE,
        scale_up_threshold: float = 2.0,
        scale_down_threshold: float = 0.5,
        scale_check_interval: float = 10.0,
    ):
        self.agent_factory = agent_factory
        self.min_size = min_size
        self.max_size = max_size
        self.scaling_policy = scaling_policy
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_check_interval = scale_check_interval

        self._agents: list[Agent] = []
        self._active_agents: set[str] = set()
        self._idle_agents: set[str] = set()
        self._agent_tasks: dict[str, asyncio.Task] = {}
        self._scaling_task: Optional[asyncio.Task] = None
        self._metrics: dict[str, Any] = {
            "total_processed": 0,
            "total_errors": 0,
            "avg_processing_time": 0.0,
            "current_queue_size": 0,
            "scale_events": [],
        }

        # Initialize minimum agents
        for i in range(min_size):
            agent = agent_factory(i)
            self._agents.append(agent)
            self._idle_agents.add(agent.name)

    def auto_scale(self) -> "PoolContext":
        """Get auto-scaling context manager."""
        return PoolContext(self)

    async def scale_up(self, count: int = 1) -> int:
        """Scale up by adding agents."""
        added = 0
        current_size = len(self._agents)

        for _i in range(count):
            if current_size + added >= self.max_size:
                break

            agent = self.agent_factory(current_size + added)
            self._agents.append(agent)
            self._idle_agents.add(agent.name)
            added += 1

            logger.info(f"Scaled up: added agent {agent.name}")

        if added > 0:
            self._metrics["scale_events"].append(
                {
                    "type": "scale_up",
                    "count": added,
                    "timestamp": time.time(),
                    "total_agents": len(self._agents),
                }
            )

        return added

    async def scale_down(self, count: int = 1) -> int:
        """Scale down by removing idle agents."""
        removed = 0
        current_size = len(self._agents)

        # Don't go below minimum
        max_removable = max(0, current_size - self.min_size)
        count = min(count, max_removable)

        # Remove idle agents first
        idle_agents = list(self._idle_agents)
        for i in range(min(count, len(idle_agents))):
            agent_name = idle_agents[i]

            # Find and remove agent
            agent_to_remove = None
            for agent in self._agents:
                if agent.name == agent_name:
                    agent_to_remove = agent
                    break

            if agent_to_remove:
                self._agents.remove(agent_to_remove)
                self._idle_agents.discard(agent_name)
                removed += 1

                logger.info(f"Scaled down: removed agent {agent_name}")

        if removed > 0:
            self._metrics["scale_events"].append(
                {
                    "type": "scale_down",
                    "count": removed,
                    "timestamp": time.time(),
                    "total_agents": len(self._agents),
                }
            )

        return removed

    async def _auto_scaling_loop(self, work_queue: WorkQueue) -> None:
        """Auto-scaling monitoring loop."""
        while True:
            try:
                await asyncio.sleep(self.scale_check_interval)

                queue_size = work_queue.size()
                active_count = len(self._active_agents)
                idle_count = len(self._idle_agents)

                self._metrics["current_queue_size"] = queue_size

                # Auto-scaling logic based on policy
                if self.scaling_policy == ScalingPolicy.AUTO_QUEUE:
                    # Scale based on queue size vs active agents
                    if queue_size > active_count * self.scale_up_threshold:
                        await self.scale_up()
                    elif (
                        queue_size < active_count * self.scale_down_threshold
                        and idle_count > 0
                    ):
                        await self.scale_down()

                elif self.scaling_policy == ScalingPolicy.AUTO_CPU:
                    # Scale based on CPU usage with adaptive thresholds and hysteresis
                    try:
                        import psutil

                        # Get CPU usage over a short interval for more accurate measurement
                        cpu_percent = psutil.cpu_percent(interval=1.0)

                        # Get system load average to understand CPU pressure
                        try:
                            load_avg = psutil.getloadavg()[0]  # 1-minute load average
                            cpu_count = psutil.cpu_count()
                            load_per_cpu = (
                                load_avg / cpu_count
                                if cpu_count is not None and cpu_count > 0
                                else load_avg
                            )
                        except (AttributeError, OSError):
                            # getloadavg not available on Windows
                            load_per_cpu = cpu_percent / 100.0

                        # Adaptive thresholds based on agent pool state
                        base_scale_up_cpu = 75.0
                        base_scale_down_cpu = 25.0

                        # Adjust thresholds based on queue pressure
                        queue_pressure = queue_size / max(active_count, 1)
                        if queue_pressure > 2.0:
                            # High queue pressure - lower CPU threshold for scaling up
                            scale_up_cpu = max(base_scale_up_cpu - 15, 60.0)
                        else:
                            scale_up_cpu = base_scale_up_cpu

                        if queue_pressure < 0.5:
                            # Low queue pressure - higher CPU threshold for scaling down
                            scale_down_cpu = min(base_scale_down_cpu + 15, 40.0)
                        else:
                            scale_down_cpu = base_scale_down_cpu

                        # Hysteresis: track recent scaling decisions to prevent flapping
                        current_time = time.time()
                        if not hasattr(self, "_last_cpu_scale_time"):
                            self._last_cpu_scale_time = 0.0
                        if not hasattr(self, "_cpu_scale_cooldown"):
                            self._cpu_scale_cooldown = 30.0  # 30 second cooldown

                        time_since_last_scale = current_time - self._last_cpu_scale_time

                        # Scale up conditions: high CPU AND (queue backlog OR high load average)
                        should_scale_up = (
                            cpu_percent > scale_up_cpu
                            and (queue_size > 0 or load_per_cpu > 0.8)
                            and active_count < self.max_size
                            and time_since_last_scale > self._cpu_scale_cooldown
                        )

                        # Scale down conditions: low CPU AND low load AND idle agents available
                        should_scale_down = (
                            cpu_percent < scale_down_cpu
                            and load_per_cpu < 0.3
                            and idle_count > 0
                            and active_count > self.min_size
                            and time_since_last_scale > self._cpu_scale_cooldown
                        )

                        if should_scale_up:
                            logger.info(
                                f"CPU-based scale up: CPU={cpu_percent:.1f}%, "
                                f"load_per_cpu={load_per_cpu:.2f}, queue_size={queue_size}"
                            )
                            await self.scale_up()
                            self._last_cpu_scale_time = current_time
                        elif should_scale_down:
                            logger.info(
                                f"CPU-based scale down: CPU={cpu_percent:.1f}%, "
                                f"load_per_cpu={load_per_cpu:.2f}, idle_count={idle_count}"
                            )
                            await self.scale_down()
                            self._last_cpu_scale_time = current_time

                        # Update metrics for monitoring
                        self._metrics.update(
                            {
                                "cpu_percent": cpu_percent,
                                "load_per_cpu": load_per_cpu,
                                "scale_up_cpu_threshold": scale_up_cpu,
                                "scale_down_cpu_threshold": scale_down_cpu,
                                "queue_pressure": queue_pressure,
                            }
                        )

                    except ImportError:
                        logger.warning("psutil not available for CPU-based scaling")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")

    def process_queue(self, work_queue: WorkQueue) -> "WorkProcessor":
        """Process work queue with pool."""
        return WorkProcessor(self, work_queue)

    def get_metrics(self) -> dict[str, Any]:
        """Get pool metrics."""
        return {
            **self._metrics,
            "total_agents": len(self._agents),
            "active_agents": len(self._active_agents),
            "idle_agents": len(self._idle_agents),
            "min_size": self.min_size,
            "max_size": self.max_size,
            "scaling_policy": self.scaling_policy.value,
        }

    @property
    def active_agents(self) -> int:
        """Get count of active agents."""
        return len(self._active_agents)


class PoolContext:
    """Context manager for pool operations."""

    def __init__(self, pool: AgentPool):
        self.pool = pool

    async def __aenter__(self) -> "AgentPool":
        """Enter context - start auto-scaling if enabled."""
        if self.pool.scaling_policy != ScalingPolicy.MANUAL:
            # Auto-scaling will be started when processing begins
            pass
        return self.pool

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context - cleanup."""
        if self.pool._scaling_task:
            self.pool._scaling_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.pool._scaling_task

        # Cancel all agent tasks
        for task in self.pool._agent_tasks.values():
            if not task.done():
                task.cancel()


class WorkProcessor:
    """Processes work items using agent pool."""

    def __init__(self, pool: AgentPool, work_queue: WorkQueue):
        self.pool = pool
        self.work_queue = work_queue
        self._completed_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._worker_tasks: list[asyncio.Task] = []

    async def __aiter__(self) -> "AsyncGenerator[CompletedWork, None]":
        """Async iterator for completed work."""
        if not self._running:
            await self._start_processing()

        while self._running or not self._completed_queue.empty():
            try:
                completed_work = await asyncio.wait_for(
                    self._completed_queue.get(), timeout=1.0
                )
                yield completed_work
            except asyncio.TimeoutError:
                if self.work_queue.is_empty() and len(self.pool._active_agents) == 0:
                    break

    async def _start_processing(self) -> None:
        """Start processing work items."""
        self._running = True

        # Start auto-scaling if enabled
        if self.pool.scaling_policy != ScalingPolicy.MANUAL:
            self.pool._scaling_task = asyncio.create_task(
                self.pool._auto_scaling_loop(self.work_queue)
            )

        # Start worker tasks for each agent
        for agent in self.pool._agents:
            task = asyncio.create_task(self._worker_loop(agent))
            self._worker_tasks.append(task)
            self.pool._agent_tasks[agent.name] = task

    async def _worker_loop(self, agent: Agent) -> None:
        """Worker loop for individual agent."""
        while self._running:
            try:
                # Get work item
                work_item = self.work_queue.get_work()
                if not work_item:
                    # No work available, mark as idle
                    self.pool._active_agents.discard(agent.name)
                    self.pool._idle_agents.add(agent.name)
                    await asyncio.sleep(0.1)
                    continue

                # Mark as active
                self.pool._active_agents.add(agent.name)
                self.pool._idle_agents.discard(agent.name)
                work_item.agent_name = agent.name

                # Process work item
                try:
                    # Set work data in agent
                    agent.set_variable("work_item", work_item.data)
                    agent.set_variable("work_id", work_item.id)

                    # Run agent
                    result = await agent.run()
                    work_item.completed_at = time.time()

                    # Create completed work
                    completed_work = CompletedWork(
                        work_item=work_item,
                        agent=agent,
                        result=result,
                        success=result.is_success,
                    )

                    # Update metrics
                    self.pool._metrics["total_processed"] += 1
                    if work_item.processing_time:
                        # Update average processing time
                        current_avg = self.pool._metrics["avg_processing_time"]
                        total_processed = self.pool._metrics["total_processed"]
                        new_avg = (
                            (current_avg * (total_processed - 1))
                            + work_item.processing_time
                        ) / total_processed
                        self.pool._metrics["avg_processing_time"] = new_avg

                    # Queue completed work
                    await self._completed_queue.put(completed_work)

                except Exception as e:
                    logger.error(f"Error processing work item {work_item.id}: {e}")

                    # Handle retry
                    work_item.retries += 1
                    if work_item.retries < work_item.max_retries:
                        # Re-queue for retry
                        self.work_queue.add_work(work_item)
                    else:
                        # Max retries reached
                        self.pool._metrics["total_errors"] += 1

                        from ..agent import AgentStatus

                        error_result = AgentResult(
                            agent_name=agent.name, status=AgentStatus.FAILED, error=e
                        )

                        completed_work = CompletedWork(
                            work_item=work_item,
                            agent=agent,
                            result=error_result,
                            success=False,
                        )

                        await self._completed_queue.put(completed_work)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker loop for {agent.name}: {e}")
                await asyncio.sleep(1)

        # Mark as idle when shutting down
        self.pool._active_agents.discard(agent.name)
        self.pool._idle_agents.add(agent.name)

    async def stop(self) -> None:
        """Stop processing."""
        self._running = False

        # Cancel all worker tasks
        for task in self._worker_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)


class DynamicProcessingPool:
    """High-level dynamic processing pool."""

    def __init__(
        self,
        agent_factory: Callable[[int], Agent],
        min_agents: int = 2,
        max_agents: int = 10,
    ):
        self.pool = AgentPool(
            agent_factory=agent_factory, min_size=min_agents, max_size=max_agents
        )
        self.work_queue = WorkQueue()
        self.results: list[CompletedWork] = []

    async def process_workload(
        self, work_items: list[WorkItem]
    ) -> list[dict[str, Any]]:
        """Process a complete workload."""
        # Add all work to queue
        for item in work_items:
            self.work_queue.add_work(item)

        results = []

        # Process with auto-scaling
        async with self.pool.auto_scale() as pool:
            work_processor = pool.process_queue(self.work_queue)
            async for completed_work in work_processor:
                result_dict = {
                    "work_item_id": completed_work.work_item.id,
                    "agent_name": completed_work.agent.name,
                    "success": completed_work.success,
                    "processing_time": completed_work.work_item.processing_time,
                    "wait_time": completed_work.work_item.wait_time,
                    "result": (
                        completed_work.result.outputs
                        if completed_work.success
                        else None
                    ),
                    "error": (
                        str(completed_work.result.error)
                        if completed_work.result.error
                        else None
                    ),
                }

                results.append(result_dict)
                self.results.append(completed_work)

        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get processing statistics."""
        if not self.results:
            return {}

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        processing_times = [
            r.work_item.processing_time
            for r in successful
            if r.work_item.processing_time
        ]

        wait_times = [
            r.work_item.wait_time for r in self.results if r.work_item.wait_time
        ]

        return {
            "total_processed": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.results) * 100,
            "avg_processing_time": (
                sum(processing_times) / len(processing_times) if processing_times else 0
            ),
            "avg_wait_time": sum(wait_times) / len(wait_times) if wait_times else 0,
            "pool_metrics": self.pool.get_metrics(),
        }
