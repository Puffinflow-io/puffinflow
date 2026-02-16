"""Coordination system with comprehensive monitoring and control."""

import asyncio
import contextlib
import inspect
import logging
import time
import uuid
import weakref
from collections.abc import AsyncGenerator, Awaitable
from dataclasses import asdict, dataclass
from typing import Any, Optional, Protocol

from .deadlock import DeadlockDetector
from .primitives import (
    CoordinationPrimitive,
    PrimitiveType,
)
from .rate_limiter import RateLimiter, RateLimitStrategy

logger = logging.getLogger(__name__)


class AgentProtocol(Protocol):
    """Protocol for agent objects that can be coordinated."""

    name: str
    state_metadata: dict[str, Any]

    def _add_to_queue(
        self, state_name: str, priority_boost: int = 0
    ) -> Awaitable[None]:
        ...

    async def run_state(self, state_name: str) -> None:
        ...


@dataclass
class CoordinationConfig:
    """Configuration for coordination system."""

    detection_interval: float = 1.0
    cleanup_interval: float = 60.0
    max_coordination_timeout: float = 30.0
    enable_metrics: bool = True
    enable_deadlock_detection: bool = True
    max_retry_attempts: int = 3
    backoff_multiplier: float = 1.5


class CoordinationError(Exception):
    """Base exception for coordination errors."""

    pass


class CoordinationTimeout(CoordinationError):
    """Raised when coordination times out."""

    pass


class AgentCoordinator:
    """Enhanced agent coordination system with comprehensive monitoring and control."""

    def __init__(
        self, agent: AgentProtocol, config: Optional[CoordinationConfig] = None
    ):
        """Initialize the coordination system.

        Args:
            agent: The agent to coordinate
            config: Configuration for the coordination system
        """
        self.agent = weakref.proxy(agent)
        self.config = config or CoordinationConfig()
        self.instance_id = str(uuid.uuid4())

        # Components
        self.rate_limiters: dict[str, RateLimiter] = {}
        self.primitives: dict[str, CoordinationPrimitive] = {}

        # Initialize deadlock detector if enabled
        self.deadlock_detector: Optional[DeadlockDetector] = None
        if self.config.enable_deadlock_detection:
            self.deadlock_detector = DeadlockDetector(
                agent, detection_interval=self.config.detection_interval
            )

        # State management
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutting_down = False
        self._start_time: Optional[float] = None
        self._coordination_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
            "timeout_requests": 0,
        }

        # Thread safety
        self._state_lock = asyncio.Lock()

        logger.info(
            f"coordinator_initialized: instance_id={self.instance_id}, "
            f"agent_name={agent.name}, detection_interval={self.config.detection_interval}, "
            f"cleanup_interval={self.config.cleanup_interval}, "
            f"deadlock_detection={self.config.enable_deadlock_detection}"
        )

    async def start(self) -> None:
        """Start the coordination system."""
        async with self._state_lock:
            if self._cleanup_task is not None:
                logger.warning(
                    f"coordinator_already_started: instance_id={self.instance_id}"
                )
                return

            self._start_time = time.time()
            self._shutting_down = False

            try:
                # Start deadlock detector
                if self.deadlock_detector:
                    await self.deadlock_detector.start()

                # Start cleanup task
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

                logger.info(f"coordinator_started: instance_id={self.instance_id}")

            except Exception as e:
                logger.error(
                    f"coordinator_start_failed: instance_id={self.instance_id}, error={e!s}"
                )
                await self._emergency_cleanup()
                raise CoordinationError(f"Failed to start coordinator: {e}") from e

    async def stop(self) -> None:
        """Stop the coordination system gracefully."""
        async with self._state_lock:
            if self._shutting_down:
                return

            self._shutting_down = True
            logger.info(f"coordinator_stopping: instance_id={self.instance_id}")

            try:
                # Stop deadlock detector
                if self.deadlock_detector:
                    await self.deadlock_detector.stop()

                # Cancel and wait for cleanup task
                if self._cleanup_task and not self._cleanup_task.done():
                    self._cleanup_task.cancel()
                    try:
                        await asyncio.wait_for(self._cleanup_task, timeout=5.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        logger.warning(
                            f"cleanup_task_forced_termination: instance_id={self.instance_id}"
                        )

                # Release all coordination resources
                await self._release_all_resources()

                # Log final statistics
                uptime = time.time() - (self._start_time or 0)
                logger.info(
                    f"coordinator_stopped: instance_id={self.instance_id}, "
                    f"uptime={uptime:.2f}, total_requests={self._coordination_stats['total_requests']}, "
                    f"successful_requests={self._coordination_stats['successful_requests']}, "
                    f"failed_requests={self._coordination_stats['failed_requests']}"
                )

            except Exception as e:
                logger.error(
                    f"coordinator_stop_error: instance_id={self.instance_id}, error={e!s}"
                )

    async def _emergency_cleanup(self) -> None:
        """Emergency cleanup in case of startup failure."""
        try:
            if self.deadlock_detector:
                await self.deadlock_detector.stop()
        except Exception as e:
            logger.error(
                f"emergency_cleanup_failed: instance_id={self.instance_id}, error={e!s}"
            )

    async def _release_all_resources(self) -> None:
        """Release all coordination resources."""
        released_count = 0
        for primitive in self.primitives.values():
            try:
                # Release all acquisitions for this coordinator instance
                caller_prefix = f"{self.instance_id}:"
                for owner in list(primitive._owners):
                    if owner.startswith(caller_prefix):
                        await primitive.release(owner)
                        released_count += 1
            except Exception as e:
                logger.error(
                    f"resource_release_error: primitive={primitive.name}, error={e!s}"
                )

        if released_count > 0:
            logger.info(
                f"released_all_resources: instance_id={self.instance_id}, count={released_count}"
            )

    def add_rate_limiter(
        self,
        name: str,
        max_rate: float,
        strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
        **kwargs: Any,
    ) -> None:
        """Add a rate limiter.

        Args:
            name: Name of the rate limiter
            max_rate: Maximum rate (requests per second)
            strategy: Rate limiting strategy
            **kwargs: Additional arguments for the rate limiter
        """
        if name in self.rate_limiters:
            logger.warning(f"rate_limiter_already_exists: name={name}")
            return

        self.rate_limiters[name] = RateLimiter(
            max_rate=max_rate, strategy=strategy, **kwargs
        )

        logger.info(
            f"rate_limiter_added: name={name}, max_rate={max_rate}, strategy={strategy.name}"
        )

    def create_primitive(
        self, name: str, primitive_type: PrimitiveType, **kwargs: Any
    ) -> None:
        """Create a coordination primitive.

        Args:
            name: Name of the primitive
            primitive_type: Type of coordination primitive
            **kwargs: Additional arguments for the primitive
        """
        if name in self.primitives:
            logger.warning(f"primitive_already_exists: name={name}")
            return

        self.primitives[name] = CoordinationPrimitive(
            name=name, type=primitive_type, **kwargs
        )

        logger.info(
            f"primitive_created: name={name}, type={primitive_type.name}, "
            f"config={kwargs}"
        )

    async def coordinate_state_execution(
        self, state_name: str, timeout: Optional[float] = None
    ) -> bool:
        """Coordinate state execution with rate limiting and resource management.

        Args:
            state_name: Name of the state to coordinate
            timeout: Optional timeout for coordination

        Returns:
            True if coordination successful, False otherwise
        """
        coordination_id = str(uuid.uuid4())
        start_time = time.time()
        timeout = timeout or self.config.max_coordination_timeout

        self._coordination_stats["total_requests"] += 1

        logger.debug(
            f"coordination_request: state={state_name}, "
            f"coordination_id={coordination_id}, timeout={timeout}"
        )

        try:
            # Check rate limits
            if state_name in self.rate_limiters and not await asyncio.wait_for(
                self.rate_limiters[state_name].acquire(), timeout=timeout
            ):
                self._coordination_stats["rate_limited_requests"] += 1
                await self._log_coordination_failure(
                    state_name, coordination_id, "rate_limit_exceeded"
                )
                return False

            # Check coordination primitives
            caller_id = f"{self.instance_id}:{state_name}:{coordination_id}"
            acquired_primitives = []

            try:
                for primitive_name, primitive in self.primitives.items():
                    remaining_timeout = timeout - (time.time() - start_time)
                    if remaining_timeout <= 0:
                        raise asyncio.TimeoutError("Coordination timeout")

                    if not await asyncio.wait_for(
                        primitive.acquire(caller_id, timeout=remaining_timeout),
                        timeout=remaining_timeout,
                    ):
                        await self._log_coordination_failure(
                            state_name,
                            coordination_id,
                            f"primitive_blocked:{primitive_name}",
                        )
                        return False

                    acquired_primitives.append((primitive_name, primitive))

                # All coordination successful
                self._coordination_stats["successful_requests"] += 1
                duration = time.time() - start_time

                logger.debug(
                    f"coordination_successful: state={state_name}, "
                    f"coordination_id={coordination_id}, duration={duration:.3f}, "
                    f"acquired_primitives={[name for name, _ in acquired_primitives]}"
                )

                return True

            except asyncio.TimeoutError:
                self._coordination_stats["timeout_requests"] += 1
                # Release any acquired primitives
                for primitive_name, primitive in acquired_primitives:
                    try:
                        await primitive.release(caller_id)
                    except Exception as release_error:
                        logger.error(
                            f"primitive_release_error: primitive={primitive_name}, "
                            f"caller_id={caller_id}, error={release_error!s}"
                        )

                await self._log_coordination_failure(
                    state_name, coordination_id, "timeout"
                )
                return False

        except Exception as e:
            self._coordination_stats["failed_requests"] += 1
            await self._log_coordination_failure(
                state_name, coordination_id, f"exception:{e!s}"
            )
            return False

    async def _log_coordination_failure(
        self, state_name: str, coordination_id: str, reason: str
    ) -> None:
        """Log coordination failure with monitoring integration."""
        logger.warning(
            f"coordination_failed: state={state_name}, "
            f"coordination_id={coordination_id}, reason={reason}"
        )

        if hasattr(self.agent, "_monitor"):
            try:
                self.agent._monitor.logger.warning(
                    f"coordination_failed: state={state_name}, "
                    f"coordination_id={coordination_id}, reason={reason}"
                )
            except Exception as e:
                logger.error(f"monitor_logging_error: {e!s}")

    async def release_coordination(
        self, state_name: str, coordination_id: Optional[str] = None
    ) -> None:
        """Release coordination resources for a state.

        Args:
            state_name: Name of the state
            coordination_id: Optional specific coordination ID
        """
        if coordination_id:
            caller_id = f"{self.instance_id}:{state_name}:{coordination_id}"
        else:
            # Release all coordinations for this state
            caller_prefix = f"{self.instance_id}:{state_name}:"

        released_count = 0

        for primitive_name, primitive in self.primitives.items():
            try:
                if coordination_id:
                    await primitive.release(caller_id)
                    released_count += 1
                else:
                    # Release all matching coordination IDs
                    for owner in list(primitive._owners):
                        if owner.startswith(caller_prefix):
                            await primitive.release(owner)
                            released_count += 1
            except Exception as e:
                logger.error(
                    f"coordination_release_error: primitive={primitive_name}, "
                    f"state={state_name}, error={e!s}"
                )

        logger.debug(
            f"coordination_released: state={state_name}, "
            f"coordination_id={coordination_id}, released_count={released_count}"
        )

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive coordinator status."""
        uptime = time.time() - (self._start_time or 0) if self._start_time else 0

        return {
            "instance_id": self.instance_id,
            "agent_name": self.agent.name,
            "uptime": uptime,
            "shutting_down": self._shutting_down,
            "config": asdict(self.config),
            "stats": self._coordination_stats.copy(),
            "rate_limiters": {
                name: limiter.get_stats()
                for name, limiter in self.rate_limiters.items()
            },
            "primitives": {
                name: primitive.get_state()
                for name, primitive in self.primitives.items()
            },
            "deadlock_detector": (
                self.deadlock_detector.get_status() if self.deadlock_detector else None
            ),
        }

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop for maintenance tasks."""
        logger.info(f"cleanup_loop_started: instance_id={self.instance_id}")

        while not self._shutting_down:
            try:
                cleanup_start = time.time()

                # Clean up expired primitive acquisitions
                cleanup_count = 0
                for primitive in self.primitives.values():
                    try:
                        async with primitive._lock:
                            before_count = len(primitive._owners)
                            primitive._cleanup_expired()
                            after_count = len(primitive._owners)
                            cleanup_count += before_count - after_count
                    except Exception as e:
                        logger.error(
                            f"primitive_cleanup_error: primitive={primitive.name}, error={e!s}"
                        )

                cleanup_duration = time.time() - cleanup_start

                if cleanup_count > 0 or cleanup_duration > 1.0:
                    logger.debug(
                        f"cleanup_cycle_completed: instance_id={self.instance_id}, "
                        f"cleaned_acquisitions={cleanup_count}, duration={cleanup_duration:.3f}"
                    )

                await asyncio.sleep(self.config.cleanup_interval)

            except asyncio.CancelledError:
                logger.info(f"cleanup_loop_cancelled: instance_id={self.instance_id}")
                break
            except Exception as e:
                logger.error(
                    f"cleanup_loop_error: instance_id={self.instance_id}, error={e!s}"
                )
                # Continue the loop even on errors
                await asyncio.sleep(1.0)

        logger.info(f"cleanup_loop_stopped: instance_id={self.instance_id}")


def enhance_agent(
    agent: AgentProtocol, config: Optional[CoordinationConfig] = None
) -> AgentProtocol:
    """Add production coordination to an agent with proper method binding.

    Args:
        agent: The agent to enhance
        config: Optional coordination configuration

    Returns:
        The enhanced agent
    """
    # Add coordinator
    coordinator = AgentCoordinator(agent, config)
    agent._coordinator = coordinator  # type: ignore

    # Store original methods
    original_run_state = agent.run_state
    original_cleanup = getattr(agent, "_cleanup", None)

    # Handle startup coordination
    async def start_coordinator() -> None:
        await coordinator.start()

    if hasattr(agent, "_startup_tasks"):
        agent._startup_tasks.append(start_coordinator())
    else:
        # Only create task if there's a running event loop
        try:
            asyncio.get_running_loop()
            task = asyncio.create_task(start_coordinator())
            # Store task reference to prevent garbage collection
            if not hasattr(agent, "_coordination_tasks"):
                agent._coordination_tasks = set()  # type: ignore
            agent._coordination_tasks.add(task)  # type: ignore
            task.add_done_callback(lambda t: agent._coordination_tasks.discard(t))  # type: ignore
        except RuntimeError:
            # No running event loop, coordinator will be started manually
            logger.info(f"no_event_loop_for_auto_start: agent={agent.name}")

    # Enhanced run_state with proper binding
    async def enhanced_run_state(state_name: str) -> None:
        """Enhanced state execution with coordination and monitoring."""
        attempt_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Check coordination and rate limits
            if not await agent._coordinator.coordinate_state_execution(state_name):  # type: ignore
                if hasattr(agent, "_monitor"):
                    agent._monitor.logger.warning(
                        f"coordination_failed: state={state_name}, attempt={attempt_id}"
                    )

                # Requeue with backoff if agent supports it
                if (
                    hasattr(agent, "_add_to_queue")
                    and hasattr(agent, "state_metadata")
                    and state_name in agent.state_metadata
                ):
                    await agent._add_to_queue(
                        state_name,
                        priority_boost=-1,
                    )
                return

            # Log execution start
            if hasattr(agent, "_monitor"):
                metadata = {}
                if (
                    hasattr(agent, "state_metadata")
                    and state_name in agent.state_metadata
                ):
                    state_meta = agent.state_metadata[state_name]
                    metadata = {
                        "resources": (
                            asdict(state_meta.resources)
                            if hasattr(state_meta, "resources")
                            else {}
                        ),
                        "dependencies": len(getattr(state_meta, "dependencies", [])),
                        "attempts": getattr(state_meta, "attempts", 0),
                    }

                agent._monitor.logger.info(
                    f"state_execution_started: state={state_name}, "
                    f"attempt={attempt_id}, metadata={metadata}"
                )

            # Execute original state with monitoring span
            async with agent._execution_span(state_name, attempt_id):  # type: ignore
                await original_run_state(state_name)

            # Record success metrics
            if hasattr(agent, "_monitor"):
                duration = time.time() - start_time
                await agent._monitor.record_metric(
                    "state_duration",
                    duration,
                    {"state": state_name, "status": "success"},
                )
                await agent._monitor.record_metric(
                    "state_success", 1, {"state": state_name}
                )

        except Exception as e:
            # Handle failure with monitoring
            if hasattr(agent, "_monitor"):
                duration = time.time() - start_time
                agent._monitor.logger.error(
                    f"state_execution_failed: state={state_name}, "
                    f"attempt={attempt_id}, error={e!s}, duration={duration:.3f}"
                )
                await agent._monitor.record_metric(
                    "state_duration", duration, {"state": state_name, "status": "error"}
                )
                await agent._monitor.record_metric(
                    "state_error",
                    1,
                    {"state": state_name, "error_type": type(e).__name__},
                )
            raise

        finally:
            # Always release coordination
            await agent._coordinator.release_coordination(state_name, attempt_id)  # type: ignore

    # Bind the enhanced method to the agent
    agent.run_state = enhanced_run_state  # type: ignore

    # Add execution span context manager
    @contextlib.asynccontextmanager
    async def _execution_span(
        state_name: str, attempt_id: str
    ) -> AsyncGenerator[None, None]:
        """Create execution span for monitoring."""
        if hasattr(agent, "_monitor"):
            try:
                async with agent._monitor.monitor_operation(
                    "state_execution",
                    {"state": state_name, "attempt": attempt_id, "agent": agent.name},
                ) as span:
                    yield span
            except Exception as e:
                logger.error(f"monitor_span_error: {e!s}")
                yield None
        else:
            yield None

    agent._execution_span = _execution_span  # type: ignore

    # Enhanced cleanup
    async def enhanced_cleanup() -> None:
        """Enhanced cleanup with coordination system shutdown."""
        try:
            # Stop coordinator first
            await agent._coordinator.stop()  # type: ignore

            # Run original cleanup if it exists
            if original_cleanup:
                if inspect.iscoroutinefunction(original_cleanup):
                    await original_cleanup()
                else:
                    original_cleanup()

        except Exception as e:
            if hasattr(agent, "_monitor"):
                agent._monitor.logger.error(f"cleanup_error: error={e!s}")
            logger.error(f"enhanced_cleanup_error: agent={agent.name}, error={e!s}")
            raise

    agent._cleanup = enhanced_cleanup  # type: ignore

    # Add utility methods with proper binding
    def add_utility_methods() -> None:
        async def get_coordination_status() -> dict[str, Any]:
            """Get coordination system status."""
            return agent._coordinator.get_status()  # type: ignore

        async def reset_coordination() -> None:
            """Reset coordination system."""
            old_config = agent._coordinator.config  # type: ignore
            await agent._coordinator.stop()  # type: ignore
            agent._coordinator = AgentCoordinator(agent, old_config)  # type: ignore
            await agent._coordinator.start()  # type: ignore

        def add_state_rate_limit(
            state_name: str,
            max_rate: float,
            strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
            **kwargs: Any,
        ) -> None:
            """Add rate limit for specific state."""
            agent._coordinator.add_rate_limiter(  # type: ignore
                state_name, max_rate, strategy, **kwargs
            )

        def add_state_coordination(
            state_name: str, primitive_type: PrimitiveType, **kwargs: Any
        ) -> None:
            """Add coordination primitive for specific state."""
            agent._coordinator.create_primitive(  # type: ignore
                f"state_{state_name}", primitive_type, **kwargs
            )

        # Bind methods to agent
        agent.get_coordination_status = get_coordination_status  # type: ignore
        agent.reset_coordination = reset_coordination  # type: ignore
        agent.add_state_rate_limit = add_state_rate_limit  # type: ignore
        agent.add_state_coordination = add_state_coordination  # type: ignore

    add_utility_methods()

    logger.info(
        f"agent_enhanced: agent_name={agent.name}, "
        f"coordinator_id={coordinator.instance_id}"
    )

    return agent


def create_coordinated_agent(
    name: str, config: Optional[CoordinationConfig] = None, **agent_kwargs: Any
) -> Any:
    """Create an agent with coordination enabled.

    Args:
        name: Name of the agent
        config: Optional coordination configuration
        **agent_kwargs: Additional arguments for agent creation

    Returns:
        Enhanced agent with coordination
    """
    from puffinflow.core.agent.base import Agent

    agent = Agent(name, **agent_kwargs)
    return enhance_agent(agent, config)
