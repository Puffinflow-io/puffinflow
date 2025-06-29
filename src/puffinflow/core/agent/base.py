"""Agent base class with enhanced reliability patterns."""
import asyncio
import heapq
import logging
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Any

from .state import (
    AgentStatus, StateStatus, StateMetadata, PrioritizedState, Priority,
    RetryPolicy, StateResult, DeadLetter
)
from .context import Context
from .checkpoint import AgentCheckpoint
from ..resources.pool import ResourcePool
from ..resources.requirements import ResourceRequirements
from ..reliability.circuit_breaker import circuit_registry, CircuitBreakerConfig, CircuitBreakerError
from ..reliability.bulkhead import bulkhead_registry, BulkheadConfig, BulkheadFullError

logger = logging.getLogger(__name__)

class ResourceTimeoutError(Exception):
    """Raised when resource acquisition times out"""
    pass

class Agent:
    """Enhanced Agent with circuit breaker, bulkhead, and dead letter handling."""

    def __init__(
        self,
        name: str,
        retry_policy: Optional[RetryPolicy] = None,
        max_concurrent: int = 10,
        resource_pool: Optional[ResourcePool] = None,
        enable_circuit_breaker: bool = True,  # NEW
        enable_bulkhead: bool = True,         # NEW
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,  # NEW
        bulkhead_config: Optional[BulkheadConfig] = None  # NEW
    ):
        self.name = name
        self.max_concurrent = max_concurrent

        # Use provided resource pool or create default one
        self.resource_pool = resource_pool or ResourcePool()

        # Core state management
        self.states: Dict[str, Callable] = {}
        self.state_metadata: Dict[str, StateMetadata] = {}
        self.dependencies: Dict[str, List[str]] = {}
        self.priority_queue: List[PrioritizedState] = []

        # Execution tracking
        self.running_states: Set[str] = set()
        self.completed_states: Set[str] = set()
        self.completed_once: Set[str] = set()
        self.status = AgentStatus.IDLE
        self.retry_policy = retry_policy or RetryPolicy()

        # Context for state execution
        self.shared_state: Dict[str, Any] = {}

        # Session tracking
        self.session_start: Optional[float] = None

        # Dead letter tracking
        self.dead_letters: List[DeadLetter] = []
        self._max_dead_letters = 1000

        # NEW: Reliability patterns
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_bulkhead = enable_bulkhead

        if enable_circuit_breaker:
            cb_config = circuit_breaker_config or CircuitBreakerConfig(name=f"{name}_circuit_breaker")
            self.circuit_breaker = circuit_registry.get_or_create(f"{name}_cb", cb_config)

        if enable_bulkhead:
            bh_config = bulkhead_config or BulkheadConfig(name=f"{name}_bulkhead", max_concurrent=max_concurrent)
            self.bulkhead = bulkhead_registry.get_or_create(f"{name}_bh", bh_config)

    def add_state(
        self,
        name: str,
        func: Callable,
        resources: Optional[ResourceRequirements] = None,
        dependencies: Optional[List[str]] = None,
        priority: Optional[Priority] = None,
        retry_policy: Optional[RetryPolicy] = None
    ) -> None:
        """Add a state to the agent - automatically extracts resource requirements from decorator."""

        # Extract configuration from decorator if present
        decorator_requirements = self._extract_decorator_requirements(func)

        # Use explicit resources if provided, otherwise use decorator requirements
        final_requirements = resources or decorator_requirements

        # Extract priority from decorator if not explicitly provided
        final_priority = priority
        if priority is None and hasattr(func, '_priority'):
            final_priority = func._priority
        elif priority is None:
            final_priority = Priority.NORMAL

        # Store the function
        self.states[name] = func

        # Create resources if still None
        if final_requirements is None:
            final_requirements = ResourceRequirements()

        # Set priority on resources
        final_requirements.priority = final_priority

        metadata = StateMetadata(
            status=StateStatus.PENDING,
            resources=final_requirements,
            priority=final_priority,
            retry_policy=retry_policy or self.retry_policy
        )

        self.state_metadata[name] = metadata

        # Set up dependencies
        if dependencies:
            self.dependencies[name] = dependencies

    def _extract_decorator_requirements(self, func: Callable) -> Optional[ResourceRequirements]:
        """Extract resource requirements from decorator."""
        try:
            from .decorators.inspection import get_state_requirements
            return get_state_requirements(func)
        except (ImportError, AttributeError):
            return None

    def create_checkpoint(self):
        """Create a checkpoint of current agent state"""
        return AgentCheckpoint.create_from_agent(self)

    async def restore_from_checkpoint(self, checkpoint) -> None:
        """Restore agent from checkpoint"""
        self.status = checkpoint.agent_status
        self.priority_queue = checkpoint.priority_queue[:]
        self.state_metadata = checkpoint.state_metadata.copy()
        self.running_states = checkpoint.running_states.copy()
        self.completed_states = checkpoint.completed_states.copy()
        self.completed_once = checkpoint.completed_once.copy()
        self.shared_state = checkpoint.shared_state.copy()
        self.session_start = checkpoint.session_start

    async def pause(self):
        """Pause the agent and return checkpoint"""
        self.status = AgentStatus.PAUSED
        return self.create_checkpoint()

    async def resume(self) -> None:
        """Resume the agent"""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.RUNNING

    async def run(self, timeout: Optional[float] = None) -> None:
        """Run workflow starting from entry point states"""
        self.status = AgentStatus.RUNNING
        self.session_start = time.time()

        try:
            # Find entry point states (states without dependencies)
            entry_states = self._find_entry_states()

            if not entry_states:
                if self.states:
                    entry_states = [next(iter(self.states.keys()))]
                else:
                    logger.warning("No states defined in agent")
                    return

            # Add entry states to queue
            for state_name in entry_states:
                await self._add_to_queue(state_name)

            # Main execution loop
            start_time = time.time()
            while self.priority_queue or self.running_states:
                # Check timeout
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning(f"Workflow timeout after {timeout} seconds")
                    break

                # Get ready states
                ready_states = await self._get_ready_states()

                if not ready_states:
                    if self.running_states:
                        await asyncio.sleep(0.1)
                        continue
                    else:
                        break

                # Process ready states (up to max_concurrent)
                tasks = []
                for state_name in ready_states[:self.max_concurrent]:
                    if len(self.running_states) < self.max_concurrent:
                        task = asyncio.create_task(self.run_state(state_name))
                        tasks.append(task)

                if tasks:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

                await asyncio.sleep(0.01)

            # Mark as completed if we finished normally
            if self.status == AgentStatus.RUNNING:
                self.status = AgentStatus.COMPLETED

        except Exception as e:
            self.status = AgentStatus.FAILED
            logger.error(f"Agent {self.name} failed: {e}")
            raise

    def _find_entry_states(self) -> List[str]:
        """Find states that can be entry points (no dependencies)"""
        entry_states = []
        for state_name in self.states:
            deps = self.dependencies.get(state_name, [])
            if not deps:
                entry_states.append(state_name)
        return entry_states

    async def _add_to_queue(self, state_name: str, priority_boost: int = 0) -> None:
        """Add state to execution queue"""
        if state_name not in self.state_metadata:
            logger.error(f"Unknown state: {state_name}")
            return

        metadata = self.state_metadata[state_name]
        priority_value = metadata.priority.value + priority_boost

        prioritized_state = PrioritizedState(
            priority=-priority_value,
            timestamp=time.time(),
            state_name=state_name,
            metadata=metadata
        )

        heapq.heappush(self.priority_queue, prioritized_state)

    async def _get_ready_states(self) -> List[str]:
        """Get states that are ready to run"""
        ready_states = []
        temp_queue = []

        while self.priority_queue:
            state = heapq.heappop(self.priority_queue)
            if await self._can_run(state.state_name):
                ready_states.append(state.state_name)
            else:
                temp_queue.append(state)

        # Put non-ready states back in queue
        for state in temp_queue:
            heapq.heappush(self.priority_queue, state)

        return ready_states

    async def run_state(self, state_name: str) -> None:
        """Execute a single state with circuit breaker, bulkhead, and timeout handling."""
        if state_name in self.running_states:
            return

        if not await self._can_run(state_name):
            return

        self.running_states.add(state_name)
        start_time = time.time()

        try:
            # NEW: Apply bulkhead isolation
            if self.enable_bulkhead:
                async with self.bulkhead.isolate():
                    await self._execute_state_with_circuit_breaker(state_name, start_time)
            else:
                await self._execute_state_with_circuit_breaker(state_name, start_time)

        except BulkheadFullError as e:
            logger.warning(f"Bulkhead full for state {state_name}: {e}")
            # Requeue with delay
            await asyncio.sleep(1.0)
            await self._add_to_queue(state_name)

        except Exception as error:
            await self._handle_failure(state_name, error, is_timeout=False)

        finally:
            self.running_states.discard(state_name)

    async def _execute_state_with_circuit_breaker(self, state_name: str, start_time: float):
        """Execute state with circuit breaker protection"""
        try:
            # NEW: Apply circuit breaker protection
            if self.enable_circuit_breaker:
                async with self.circuit_breaker.protect():
                    await self._execute_state_core(state_name, start_time)
            else:
                await self._execute_state_core(state_name, start_time)

        except CircuitBreakerError as e:
            logger.warning(f"Circuit breaker open for state {state_name}: {e}")
            # Don't retry immediately - circuit breaker will handle recovery
            raise

    async def _execute_state_core(self, state_name: str, start_time: float):
        """Core state execution logic"""
        await self._resolve_dependencies(state_name)

        metadata = self.state_metadata[state_name]

        # Get timeout from resources or default
        state_timeout = None
        if metadata.resources and hasattr(metadata.resources, 'timeout'):
            state_timeout = metadata.resources.timeout

        # Acquire resources (pass agent name for leak detection)
        resource_acquired = await self.resource_pool.acquire(
            state_name, metadata.resources, timeout=30.0, agent_name=self.name
        )

        if not resource_acquired:
            raise ResourceTimeoutError(f"Could not acquire resources for {state_name}")

        try:
            # Execute the state function
            context = Context(self.shared_state)

            # Execute with timeout if specified
            if state_timeout:
                result = await asyncio.wait_for(
                    self.states[state_name](context),
                    timeout=state_timeout
                )
            else:
                result = await self.states[state_name](context)

            duration = time.time() - start_time
            logger.info(f"State {state_name} completed in {duration:.2f}s")

            # Update metadata on success
            metadata.attempts = 0
            metadata.last_execution = time.time()
            metadata.last_success = time.time()

            # Track completion
            self.completed_states.add(state_name)
            self.completed_once.add(state_name)

            # Handle transitions/next states
            await self._handle_state_result(state_name, result)

        finally:
            # Always release resources
            await self.resource_pool.release(state_name)

    async def _handle_state_result(self, state_name: str, result: Any) -> None:
        """Handle the result of state execution"""
        if result is None:
            return

        if isinstance(result, str):
            await self._add_to_queue(result)
        elif isinstance(result, (list, tuple)):
            for next_state in result:
                if isinstance(next_state, str):
                    await self._add_to_queue(next_state)

    async def _handle_failure(self, state_name: str, error: Exception,
                            is_timeout: bool = False) -> None:
        """Enhanced failure handling with integrated dead letter"""
        logger.error(f"State {state_name} failed: {error}")

        metadata = self.state_metadata[state_name]
        metadata.attempts += 1

        # Check if we've exceeded max retries
        if metadata.attempts >= metadata.max_retries:
            retry_policy = metadata.retry_policy or self.retry_policy
            should_dead_letter = (
                retry_policy and retry_policy.dead_letter_on_max_retries
            ) or (
                is_timeout and retry_policy and retry_policy.dead_letter_on_timeout
            )

            if should_dead_letter:
                await self._move_to_dead_letter(state_name, error, metadata, is_timeout)
                return
            else:
                logger.error(f"State {state_name} failed permanently after {metadata.attempts} attempts")
                return

        # Continue with retry
        logger.info(f"Retrying state {state_name}, attempt {metadata.attempts}/{metadata.max_retries}")

        if metadata.retry_policy:
            await metadata.retry_policy.wait(metadata.attempts - 1)
        elif self.retry_policy:
            await self.retry_policy.wait(metadata.attempts - 1)

        # Re-queue for retry
        await self._add_to_queue(state_name, priority_boost=1)

    async def _move_to_dead_letter(self, state_name: str, error: Exception,
                                  metadata: StateMetadata, is_timeout: bool = False) -> None:
        """Move failed state to dead letter"""
        dead_letter = DeadLetter(
            state_name=state_name,
            agent_name=self.name,
            error_message=str(error),
            error_type=type(error).__name__,
            attempts=metadata.attempts,
            failed_at=time.time(),
            timeout_occurred=is_timeout,
            context_snapshot=dict(self.shared_state)
        )

        self.dead_letters.append(dead_letter)

        if len(self.dead_letters) > self._max_dead_letters:
            self.dead_letters = self.dead_letters[-self._max_dead_letters:]

        logger.error(
            f"State {state_name} moved to dead letter queue after {metadata.attempts} attempts. "
            f"Error: {error}. Timeout: {is_timeout}"
        )

    async def _resolve_dependencies(self, state_name: str) -> None:
        """Resolve state dependencies"""
        deps = self.dependencies.get(state_name, [])
        for dep in deps:
            if dep not in self.completed_once:
                logger.warning(f"Unmet dependency {dep} for state {state_name}")

    async def _can_run(self, state_name: str) -> bool:
        """Check if state can run (dependencies satisfied)"""
        if state_name in self.completed_once:
            return False

        deps = self.dependencies.get(state_name, [])
        for dep in deps:
            if dep not in self.completed_once:
                return False

        return True

    def cancel_state(self, state_name: str) -> None:
        """Cancel a specific state"""
        if state_name in self.running_states:
            self.running_states.discard(state_name)
            logger.info(f"Cancelled state: {state_name}")

    async def cancel_all(self) -> None:
        """Cancel all running and pending states"""
        self.running_states.clear()
        self.priority_queue.clear()
        self.status = AgentStatus.CANCELLED
        logger.info("All states cancelled")

    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status including reliability metrics."""
        status = {
            "resource_pool": self.resource_pool.get_usage_stats(),
            "running_states": len(self.running_states),
            "queued_states": len(self.priority_queue),
            "completed_states": len(self.completed_states),
            "agent_status": self.status.value
        }

        # NEW: Add reliability metrics
        if self.enable_circuit_breaker:
            status["circuit_breaker"] = self.circuit_breaker.get_metrics()

        if self.enable_bulkhead:
            status["bulkhead"] = self.bulkhead.get_metrics()

        # Resource leak detection
        status["resource_leaks"] = self.resource_pool.get_leak_metrics()

        return status

    def get_state_info(self, state_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific state."""
        if state_name not in self.states:
            return {"error": "State not found"}

        metadata = self.state_metadata.get(state_name)

        # Check if state has decorator info
        has_decorator = False
        try:
            from .decorators.inspection import is_puffinflow_state
            has_decorator = is_puffinflow_state(self.states[state_name])
        except ImportError:
            pass

        return {
            "name": state_name,
            "status": metadata.status.value if metadata else "unknown",
            "attempts": metadata.attempts if metadata else 0,
            "priority": metadata.priority.value if metadata else "unknown",
            "has_decorator": has_decorator,
            "dependencies": self.dependencies.get(state_name, []),
            "in_running": state_name in self.running_states,
            "completed": state_name in self.completed_states
        }

    def list_states(self) -> List[Dict[str, Any]]:
        """List all states with their basic information."""
        result = []
        for name in self.states:
            try:
                from .decorators.inspection import is_puffinflow_state
                has_decorator = is_puffinflow_state(self.states[name])
            except ImportError:
                has_decorator = False

            result.append({
                "name": name,
                "has_decorator": has_decorator,
                "dependencies": self.dependencies.get(name, [])
            })
        return result

    # Dead letter methods
    def get_dead_letters(self) -> List[DeadLetter]:
        """Get all dead letters for inspection"""
        return self.dead_letters.copy()

    def clear_dead_letters(self) -> None:
        """Clear dead letter queue"""
        count = len(self.dead_letters)
        self.dead_letters.clear()
        logger.info(f"Cleared {count} dead letters")

    def get_dead_letter_count(self) -> int:
        """Get count of dead letters"""
        return len(self.dead_letters)

    def get_dead_letters_by_state(self, state_name: str) -> List[DeadLetter]:
        """Get dead letters for specific state"""
        return [dl for dl in self.dead_letters if dl.state_name == state_name]

    # NEW: Reliability control methods
    async def force_circuit_breaker_open(self):
        """Manually open circuit breaker"""
        if self.enable_circuit_breaker:
            await self.circuit_breaker.force_open()

    async def force_circuit_breaker_close(self):
        """Manually close circuit breaker"""
        if self.enable_circuit_breaker:
            await self.circuit_breaker.force_close()

    def check_resource_leaks(self):
        """Check for resource leaks"""
        return self.resource_pool.check_leaks()