"""Enhanced Agent with direct access and coordination features."""

import asyncio
import json
import logging
import pickle
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, Union

from .checkpoint import AgentCheckpoint
from .context import Context
from .state import (
    AgentStatus,
    DeadLetter,
    PrioritizedState,
    Priority,
    RetryPolicy,
    StateMetadata,
    StateResult,
    StateStatus,
)

# Import scheduling components
try:
    from .scheduling.builder import ScheduleBuilder
    from .scheduling.scheduler import GlobalScheduler, ScheduledAgent

    _SCHEDULING_AVAILABLE = True
except ImportError:
    _SCHEDULING_AVAILABLE = False

# Import ResourceRequirements conditionally
try:
    from ..resources.requirements import ResourceRequirements

    _ResourceRequirements: Optional[type] = ResourceRequirements
except ImportError:
    _ResourceRequirements = None

# Import these conditionally to avoid circular imports
if TYPE_CHECKING:
    from ..coordination.agent_team import AgentTeam
    from ..coordination.primitives import CoordinationPrimitive
    from ..reliability.bulkhead import Bulkhead, BulkheadConfig
    from ..reliability.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
    from ..resources.pool import ResourcePool

logger = logging.getLogger(__name__)


# Checkpoint persistence interfaces
class CheckpointStorage(Protocol):
    """Protocol for checkpoint storage backends."""

    async def save_checkpoint(
        self, agent_name: str, checkpoint: AgentCheckpoint
    ) -> str:
        """Save checkpoint and return checkpoint ID."""
        ...

    async def load_checkpoint(
        self, agent_name: str, checkpoint_id: Optional[str] = None
    ) -> Optional[AgentCheckpoint]:
        """Load checkpoint by ID or latest for agent."""
        ...

    async def list_checkpoints(self, agent_name: str) -> list[str]:
        """List available checkpoint IDs for agent."""
        ...

    async def delete_checkpoint(self, agent_name: str, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        ...


class FileCheckpointStorage:
    """File-based checkpoint storage."""

    def __init__(self, base_path: str = "./checkpoints", format: str = "pickle"):
        """
        Initialize file storage.

        Args:
            base_path: Directory to store checkpoint files
            format: Storage format ('pickle' or 'json')
        """
        self.base_path = Path(base_path)
        self.format = format.lower()
        if self.format not in ("pickle", "json"):
            raise ValueError(f"Unsupported format: {format}. Use 'pickle' or 'json'")

        # Create directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, agent_name: str, checkpoint_id: str) -> Path:
        """Get file path for checkpoint."""
        agent_dir = self.base_path / agent_name
        agent_dir.mkdir(exist_ok=True)
        ext = "pkl" if self.format == "pickle" else "json"
        return agent_dir / f"{checkpoint_id}.{ext}"

    async def save_checkpoint(
        self, agent_name: str, checkpoint: AgentCheckpoint
    ) -> str:
        """Save checkpoint to file."""
        checkpoint_id = f"checkpoint_{int(checkpoint.timestamp)}"
        file_path = self._get_checkpoint_path(agent_name, checkpoint_id)

        try:
            if self.format == "pickle":
                with file_path.open("wb") as f:
                    pickle.dump(checkpoint, f)
            else:  # json
                # Convert checkpoint to JSON-serializable format
                checkpoint_data = {
                    "timestamp": checkpoint.timestamp,
                    "agent_name": checkpoint.agent_name,
                    "agent_status": checkpoint.agent_status.value,
                    "priority_queue": [
                        {
                            "priority": ps.priority,
                            "timestamp": ps.timestamp,
                            "state_name": ps.state_name,
                            "metadata": {
                                "status": ps.metadata.status.value,
                                "attempts": ps.metadata.attempts,
                                "max_retries": ps.metadata.max_retries,
                                "last_execution": ps.metadata.last_execution,
                                "last_success": ps.metadata.last_success,
                                "state_id": ps.metadata.state_id,
                                "priority": ps.metadata.priority.value,
                            },
                        }
                        for ps in checkpoint.priority_queue
                    ],
                    "state_metadata": {
                        k: {
                            "status": v.status.value,
                            "attempts": v.attempts,
                            "max_retries": v.max_retries,
                            "last_execution": v.last_execution,
                            "last_success": v.last_success,
                            "state_id": v.state_id,
                            "priority": v.priority.value,
                        }
                        for k, v in checkpoint.state_metadata.items()
                    },
                    "running_states": list(checkpoint.running_states),
                    "completed_states": list(checkpoint.completed_states),
                    "completed_once": list(checkpoint.completed_once),
                    "shared_state": checkpoint.shared_state,
                    "session_start": checkpoint.session_start,
                }

                with file_path.open("w") as f:
                    json.dump(checkpoint_data, f, indent=2, default=str)

            logger.info(f"Checkpoint saved to {file_path}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint to {file_path}: {e}")
            raise

    async def load_checkpoint(
        self, agent_name: str, checkpoint_id: Optional[str] = None
    ) -> Optional[AgentCheckpoint]:
        """Load checkpoint from file."""
        if checkpoint_id is None:
            # Load latest checkpoint
            checkpoints = await self.list_checkpoints(agent_name)
            if not checkpoints:
                return None
            checkpoint_id = checkpoints[-1]  # Latest by timestamp

        file_path = self._get_checkpoint_path(agent_name, checkpoint_id)

        if not file_path.exists():
            logger.warning(f"Checkpoint file not found: {file_path}")
            return None

        try:
            if self.format == "pickle":
                with file_path.open("rb") as f:
                    checkpoint: AgentCheckpoint = pickle.load(f)
                    return checkpoint
            else:  # json
                with file_path.open("r") as f:
                    data = json.load(f)

                # Reconstruct checkpoint from JSON data
                from .state import (
                    AgentStatus,
                    PrioritizedState,
                    Priority,
                    StateMetadata,
                    StateStatus,
                )

                checkpoint = AgentCheckpoint(
                    timestamp=data["timestamp"],
                    agent_name=data["agent_name"],
                    agent_status=AgentStatus(data["agent_status"]),
                    priority_queue=[
                        PrioritizedState(
                            priority=ps["priority"],
                            timestamp=ps["timestamp"],
                            state_name=ps["state_name"],
                            metadata=StateMetadata(
                                status=StateStatus(ps["metadata"]["status"]),
                                attempts=ps["metadata"]["attempts"],
                                max_retries=ps["metadata"]["max_retries"],
                                last_execution=ps["metadata"]["last_execution"],
                                last_success=ps["metadata"]["last_success"],
                                state_id=ps["metadata"]["state_id"],
                                priority=Priority(ps["metadata"]["priority"]),
                            ),
                        )
                        for ps in data["priority_queue"]
                    ],
                    state_metadata={
                        k: StateMetadata(
                            status=StateStatus(v["status"]),
                            attempts=v["attempts"],
                            max_retries=v["max_retries"],
                            last_execution=v["last_execution"],
                            last_success=v["last_success"],
                            state_id=v["state_id"],
                            priority=Priority(v["priority"]),
                        )
                        for k, v in data["state_metadata"].items()
                    },
                    running_states=set(data["running_states"]),
                    completed_states=set(data["completed_states"]),
                    completed_once=set(data["completed_once"]),
                    shared_state=data["shared_state"],
                    session_start=data["session_start"],
                )

                return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint from {file_path}: {e}")
            return None

    async def list_checkpoints(self, agent_name: str) -> list[str]:
        """List available checkpoint files."""
        agent_dir = self.base_path / agent_name
        if not agent_dir.exists():
            return []

        ext = "pkl" if self.format == "pickle" else "json"
        checkpoint_files = [f.stem for f in agent_dir.glob(f"*.{ext}") if f.is_file()]

        # Sort by timestamp (extract from filename)
        checkpoint_files.sort(
            key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0
        )
        return checkpoint_files

    async def delete_checkpoint(self, agent_name: str, checkpoint_id: str) -> bool:
        """Delete checkpoint file."""
        file_path = self._get_checkpoint_path(agent_name, checkpoint_id)

        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted checkpoint: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {file_path}: {e}")
            return False


class MemoryCheckpointStorage:
    """In-memory checkpoint storage for testing."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, dict[str, AgentCheckpoint]] = {}

    async def save_checkpoint(
        self, agent_name: str, checkpoint: AgentCheckpoint
    ) -> str:
        """Save checkpoint to memory."""
        checkpoint_id = f"checkpoint_{int(checkpoint.timestamp)}"

        if agent_name not in self._checkpoints:
            self._checkpoints[agent_name] = {}

        # Deep copy to prevent modifications
        import copy

        self._checkpoints[agent_name][checkpoint_id] = copy.deepcopy(checkpoint)

        logger.info(f"Checkpoint saved to memory: {agent_name}/{checkpoint_id}")
        return checkpoint_id

    async def load_checkpoint(
        self, agent_name: str, checkpoint_id: Optional[str] = None
    ) -> Optional[AgentCheckpoint]:
        """Load checkpoint from memory."""
        if agent_name not in self._checkpoints:
            return None

        agent_checkpoints = self._checkpoints[agent_name]

        if checkpoint_id is None:
            # Get latest checkpoint
            if not agent_checkpoints:
                return None
            latest_id = max(
                agent_checkpoints.keys(), key=lambda x: int(x.split("_")[-1])
            )
            checkpoint_id = latest_id

        checkpoint = agent_checkpoints.get(checkpoint_id)
        if checkpoint:
            # Return deep copy to prevent modifications
            import copy

            return copy.deepcopy(checkpoint)
        return None

    async def list_checkpoints(self, agent_name: str) -> list[str]:
        """List checkpoints in memory."""
        if agent_name not in self._checkpoints:
            return []

        checkpoints = list(self._checkpoints[agent_name].keys())
        checkpoints.sort(
            key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0
        )
        return checkpoints

    async def delete_checkpoint(self, agent_name: str, checkpoint_id: str) -> bool:
        """Delete checkpoint from memory."""
        if (
            agent_name in self._checkpoints
            and checkpoint_id in self._checkpoints[agent_name]
        ):
            del self._checkpoints[agent_name][checkpoint_id]
            logger.info(f"Deleted checkpoint from memory: {agent_name}/{checkpoint_id}")
            return True
        return False


@dataclass
class AgentResult:
    """Rich result container for agent execution."""

    agent_name: str
    status: AgentStatus
    outputs: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_duration: Optional[float] = None

    def get_output(self, key: str, default: Any = None) -> Any:
        """Get output value."""
        return self.outputs.get(key, default)

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get variable value."""
        return self.variables.get(key, default)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self.metadata.get(key, default)

    def get_metric(self, key: str, default: Any = None) -> Any:
        """Get metric value."""
        return self.metrics.get(key, default)

    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.status == AgentStatus.COMPLETED and self.error is None

    @property
    def is_failed(self) -> bool:
        """Check if execution failed."""
        return self.status == AgentStatus.FAILED or self.error is not None


class ResourceTimeoutError(Exception):
    """Raised when resource acquisition times out."""

    pass


class Agent:
    """Enhanced Agent with direct variable access and coordination features."""

    def __init__(
        self,
        name: str,
        resource_pool: Optional["ResourcePool"] = None,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker_config: Optional["CircuitBreakerConfig"] = None,
        bulkhead_config: Optional["BulkheadConfig"] = None,
        max_concurrent: int = 5,
        enable_dead_letter: bool = True,
        state_timeout: Optional[float] = None,
        checkpoint_storage: Optional[CheckpointStorage] = None,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.states: dict[str, Callable] = {}
        self.state_metadata: dict[str, StateMetadata] = {}
        self.dependencies: dict[str, list[str]] = {}
        self.status = AgentStatus.IDLE
        self.shared_state: dict[str, Any] = {}
        self.priority_queue: list[PrioritizedState] = []
        self.running_states: set[str] = set()
        self.completed_states: set[str] = set()
        self.completed_once: set[str] = set()
        self.dead_letters: list[DeadLetter] = []
        self.session_start: Optional[float] = None

        # Configuration
        self.max_concurrent = max_concurrent
        self.state_timeout = state_timeout
        self.checkpoint_storage = checkpoint_storage or MemoryCheckpointStorage()

        # Enhanced features
        self._context: Optional[Context] = None
        self._variable_watchers: dict[str, list[Callable]] = {}
        self._shared_variable_watchers: dict[str, list[Callable]] = {}
        self._agent_variables: dict[str, Any] = {}
        self._persistent_variables: dict[str, Any] = {}
        self._property_definitions: dict[str, dict] = {}
        self._team: Optional[weakref.ReferenceType] = None
        self._message_handlers: dict[str, Callable] = {}
        self._event_handlers: dict[str, list[Callable]] = {}
        self._state_change_handlers: list[Callable] = []

        # Resource and reliability components - lazy initialization
        self._resource_pool = resource_pool
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._bulkhead: Optional[Bulkhead] = None
        self._circuit_breaker_config = circuit_breaker_config
        self._bulkhead_config = bulkhead_config

        self.retry_policy = retry_policy or RetryPolicy()
        self.enable_dead_letter = enable_dead_letter
        self._cleanup_handlers: list[Callable] = []

        # Create context
        self.context = self._create_context(self.shared_state)

    @property
    def resource_pool(self) -> "ResourcePool":
        """Get or create resource pool."""
        if self._resource_pool is None:
            from ..resources.pool import ResourcePool

            self._resource_pool = ResourcePool()
        return self._resource_pool

    @resource_pool.setter
    def resource_pool(self, value: "ResourcePool") -> None:
        """Set resource pool."""
        self._resource_pool = value

    @property
    def circuit_breaker(self) -> "CircuitBreaker":
        """Get or create circuit breaker."""
        if self._circuit_breaker is None:
            from ..reliability.circuit_breaker import (
                CircuitBreaker,
                CircuitBreakerConfig,
            )

            if self._circuit_breaker_config:
                config = self._circuit_breaker_config
            else:
                config = CircuitBreakerConfig(name=f"{self.name}_circuit_breaker")

            self._circuit_breaker = CircuitBreaker(config)
        return self._circuit_breaker

    @property
    def bulkhead(self) -> "Bulkhead":
        """Get or create bulkhead."""
        if self._bulkhead is None:
            from ..reliability.bulkhead import Bulkhead, BulkheadConfig

            if self._bulkhead_config:
                config = self._bulkhead_config
            else:
                config = BulkheadConfig(
                    name=f"{self.name}_bulkhead", max_concurrent=self.max_concurrent
                )

            self._bulkhead = Bulkhead(config)
        return self._bulkhead

    # Direct variable access methods
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get variable directly from agent context or internal storage."""
        if self._context:
            return self._context.get_variable(key, default)
        return self._agent_variables.get(key, default)

    def set_variable(self, key: str, value: Any) -> None:
        """Set variable directly on agent context or internal storage."""
        if self._context:
            old_value = self._context.get_variable(key)
            self._context.set_variable(key, value)
            self._trigger_variable_watchers(key, old_value, value)
        else:
            old_value = self._agent_variables.get(key)
            self._agent_variables[key] = value
            self._trigger_variable_watchers(key, old_value, value)

    def increment_variable(self, key: str, amount: Union[int, float] = 1) -> None:
        """Increment a numeric variable."""
        current = self.get_variable(key, 0)
        self.set_variable(key, current + amount)

    def append_variable(self, key: str, value: Any) -> None:
        """Append to a list variable."""
        current = self.get_variable(key, [])
        if not isinstance(current, list):
            current = [current]
        current.append(value)
        self.set_variable(key, current)

    def get_shared_variable(self, key: str, default: Any = None) -> Any:
        """Get shared variable accessible to all agents."""
        return self.shared_state.get(key, default)

    def set_shared_variable(self, key: str, value: Any) -> None:
        """Set shared variable accessible to all agents."""
        old_value = self.shared_state.get(key)
        self.shared_state[key] = value
        self._trigger_shared_variable_watchers(key, old_value, value)

    def get_agent_variable(self, key: str, default: Any = None) -> Any:
        """Get agent-specific variable (not shared)."""
        return self._agent_variables.get(key, default)

    def set_agent_variable(self, key: str, value: Any) -> None:
        """Set agent-specific variable (not shared)."""
        old_value = self._agent_variables.get(key)
        self._agent_variables[key] = value
        self._trigger_variable_watchers(key, old_value, value)

    def get_persistent_variable(self, key: str, default: Any = None) -> Any:
        """Get persistent variable that survives restarts."""
        return self._persistent_variables.get(key, default)

    def set_persistent_variable(self, key: str, value: Any) -> None:
        """Set persistent variable that survives restarts."""
        self._persistent_variables[key] = value

    # Context content access methods
    def get_output(self, key: str, default: Any = None) -> Any:
        """Get output value from context."""
        if self._context:
            return self._context.get_output(key, default)
        return default

    def set_output(self, key: str, value: Any) -> None:
        """Set output value in context."""
        if self._context:
            self._context.set_output(key, value)

    def get_all_outputs(self) -> dict[str, Any]:
        """Get all output values."""
        if self._context:
            output_keys = self._context.get_output_keys()
            return {key: self._context.get_output(key) for key in output_keys}
        return {}

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        if self._context and hasattr(self._context, "get_metadata"):
            return self._context.get_metadata(key, default)
        return default

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        if self._context and hasattr(self._context, "set_metadata"):
            self._context.set_metadata(key, value)

    def get_cached(self, key: str, default: Any = None) -> Any:
        """Get cached value."""
        if self._context:
            return self._context.get_cached(key, default)
        return default

    def set_cached(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with optional TTL."""
        if self._context:
            self._context.set_cached(key, value, ttl)

    # Property system
    def define_property(
        self,
        name: str,
        prop_type: type,
        default: Any = None,
        validator: Optional[Callable] = None,
    ) -> None:
        """Define a typed property with validation."""
        self._property_definitions[name] = {
            "type": prop_type,
            "default": default,
            "validator": validator,
        }

        # Set default value if not already set
        if name not in self._agent_variables:
            self.set_variable(name, default)

        # Create property accessor
        def getter(obj: Any) -> Any:
            return obj.get_variable(name, default)

        def setter(obj: Any, value: Any) -> None:
            if validator:
                value = validator(value)
            if not isinstance(value, prop_type) and value is not None:
                try:
                    value = prop_type(value)
                except (ValueError, TypeError) as e:
                    raise TypeError(f"Cannot convert {value} to {prop_type}") from e
            obj.set_variable(name, value)

        setattr(self.__class__, name, property(getter, setter))

    # Variable watching
    def watch_variable(self, key: str, handler: Callable) -> None:
        """Watch for changes to a variable."""
        if key not in self._variable_watchers:
            self._variable_watchers[key] = []
        self._variable_watchers[key].append(handler)

    def watch_shared_variable(self, key: str, handler: Callable) -> None:
        """Watch for changes to a shared variable."""
        if key not in self._shared_variable_watchers:
            self._shared_variable_watchers[key] = []
        self._shared_variable_watchers[key].append(handler)

    def _trigger_variable_watchers(
        self, key: str, old_value: Any, new_value: Any
    ) -> None:
        """Trigger watchers for variable changes."""
        if key in self._variable_watchers and old_value != new_value:
            for handler in self._variable_watchers[key]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        task = asyncio.create_task(handler(old_value, new_value))
                        # Store task reference to prevent it from being garbage collected
                        if not hasattr(self, "_background_tasks"):
                            self._background_tasks = set()
                        self._background_tasks.add(task)
                        task.add_done_callback(
                            lambda t: self._background_tasks.discard(t)
                        )
                    else:
                        handler(old_value, new_value)
                except Exception as e:
                    logger.error(f"Error in variable watcher for {key}: {e}")

    def _trigger_shared_variable_watchers(
        self, key: str, old_value: Any, new_value: Any
    ) -> None:
        """Trigger watchers for shared variable changes."""
        if key in self._shared_variable_watchers and old_value != new_value:
            for handler in self._shared_variable_watchers[key]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        task = asyncio.create_task(handler(old_value, new_value))
                        # Store task reference to prevent it from being garbage collected
                        if not hasattr(self, "_background_tasks"):
                            self._background_tasks = set()
                        self._background_tasks.add(task)
                        task.add_done_callback(
                            lambda t: self._background_tasks.discard(t)
                        )
                    else:
                        handler(old_value, new_value)
                except Exception as e:
                    logger.error(f"Error in shared variable watcher for {key}: {e}")

    # State change events
    def on_state_change(self, handler: Callable) -> None:
        """Register handler for state changes."""
        self._state_change_handlers.append(handler)

    def _trigger_state_change(self, old_state: Any, new_state: Any) -> None:
        """Trigger state change handlers."""
        for handler in self._state_change_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    task = asyncio.create_task(handler(old_state, new_state))
                    # Store task reference to prevent it from being garbage collected
                    if not hasattr(self, "_background_tasks"):
                        self._background_tasks = set()
                    self._background_tasks.add(task)
                    task.add_done_callback(lambda t: self._background_tasks.discard(t))
                else:
                    handler(old_state, new_state)
            except Exception as e:
                logger.error(f"Error in state change handler: {e}")

    # Team coordination
    def set_team(self, team: "AgentTeam") -> None:
        """Set the team this agent belongs to."""
        self._team = weakref.ref(team)

    def get_team(self) -> Optional["AgentTeam"]:
        """Get the team this agent belongs to."""
        if self._team:
            return self._team()
        return None

    # Messaging
    def message_handler(self, message_type: str) -> Callable:
        """Decorator for message handlers."""

        def decorator(func: Callable) -> Callable:
            self._message_handlers[message_type] = func
            return func

        return decorator

    async def send_message_to(
        self, agent_name: str, message: dict[str, Any]
    ) -> dict[str, Any]:
        """Send message to another agent."""
        team = self.get_team()
        if team:
            return await team.send_message(self.name, agent_name, message)
        raise RuntimeError("Agent must be part of a team to send messages")

    async def reply_to(self, sender_agent: str, message: dict[str, Any]) -> None:
        """Reply to a message from another agent."""
        await self.send_message_to(sender_agent, message)

    async def broadcast_message(self, message_type: str, data: dict[str, Any]) -> None:
        """Broadcast message to all agents in team."""
        team = self.get_team()
        if team:
            await team.broadcast_message(self.name, message_type, data)

    async def handle_message(
        self, message_type: str, message: dict[str, Any], sender: str
    ) -> dict[str, Any]:
        """Handle incoming message."""
        if message_type in self._message_handlers:
            result = await self._message_handlers[message_type](message, sender)
            return dict(result) if result else {}
        return {}

    # Event system
    def on_event(self, event_type: str) -> Callable:
        """Decorator for event handlers."""

        def decorator(func: Callable) -> Callable:
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            self._event_handlers[event_type].append(func)
            return func

        return decorator

    async def emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event."""
        team = self.get_team()
        if team:
            await team.emit_event(self.name, event_type, data)

        # Handle local events
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(self._context, data)
                    else:
                        handler(self._context, data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")

    # Variable synchronization
    async def sync_variables_with(
        self, agent_name: str, variable_names: list[str]
    ) -> None:
        """Sync specific variables with another agent."""
        team = self.get_team()
        if team:
            other_agent = team.get_agent(agent_name)
            if other_agent:
                for var_name in variable_names:
                    value = self.get_variable(var_name)
                    other_agent.set_variable(var_name, value)

    async def wait_for_agent_variable(
        self,
        agent_name: str,
        variable_name: str,
        expected_value: Any,
        timeout: Optional[float] = None,
    ) -> bool:
        """Wait for another agent's variable to reach a specific value."""
        team = self.get_team()
        if not team:
            return False

        other_agent = team.get_agent(agent_name)
        if not other_agent:
            return False

        start_time = time.time()
        while True:
            current_value = other_agent.get_variable(variable_name)
            if current_value == expected_value:
                return True

            if timeout and (time.time() - start_time) > timeout:
                return False

            await asyncio.sleep(0.1)

    def get_synced_variable(
        self, agent_name: str, variable_name: str, default: Any = None
    ) -> Any:
        """Get a variable from another agent."""
        team = self.get_team()
        if team:
            other_agent = team.get_agent(agent_name)
            if other_agent:
                return other_agent.get_variable(variable_name, default)
        return default

    # Context creation
    def _create_context(self, shared_state: dict[str, Any]) -> Context:
        """Create enhanced context with agent variables."""
        context = Context(shared_state)

        # Copy agent variables to context
        for key, value in self._agent_variables.items():
            context.set_variable(key, value)

        self._context = context
        return context

    # State management (keeping existing methods)
    def add_state(
        self,
        name: str,
        func: Callable,
        dependencies: Optional[list[str]] = None,
        resources: Optional[
            Any
        ] = None,  # Using Any since ResourceRequirements may not be available
        priority: Optional[Priority] = None,
        retry_policy: Optional[RetryPolicy] = None,
        coordination_primitives: Optional[list["CoordinationPrimitive"]] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Add a state to the agent."""
        self.states[name] = func
        self.dependencies[name] = dependencies or []

        # Extract decorator requirements if available
        decorator_requirements = self._extract_decorator_requirements(func)
        final_requirements = resources or decorator_requirements

        # Get priority from function if not provided
        final_priority = priority
        if hasattr(func, "_priority"):
            final_priority = func._priority
        elif final_priority is None:
            final_priority = Priority.NORMAL

        # Ensure final_priority is not None
        if final_priority is None:
            final_priority = Priority.NORMAL

        if final_requirements is None and _ResourceRequirements is not None:
            final_requirements = _ResourceRequirements()

        # Use max_retries parameter or fall back to retry_policy or agent default
        final_max_retries = max_retries or (
            retry_policy.max_retries if retry_policy else self.retry_policy.max_retries
        )

        # Create state metadata
        metadata = StateMetadata(
            status=StateStatus.PENDING,
            priority=final_priority,
            resources=final_requirements,
            retry_policy=retry_policy or self.retry_policy,
            coordination_primitives=coordination_primitives or [],
            max_retries=final_max_retries,
        )

        self.state_metadata[name] = metadata

    def _extract_decorator_requirements(
        self, func: Callable
    ) -> Optional[Any]:  # Using Any since ResourceRequirements may not be available
        """Extract resource requirements from decorator metadata."""
        if hasattr(func, "_resource_requirements"):
            requirements = func._resource_requirements
            if _ResourceRequirements is not None and isinstance(
                requirements, _ResourceRequirements
            ):
                return requirements
        return None

    # Checkpointing
    def create_checkpoint(self) -> AgentCheckpoint:
        """Create a checkpoint of current agent state."""
        return AgentCheckpoint.create_from_agent(self)

    async def restore_from_checkpoint(self, checkpoint: AgentCheckpoint) -> None:
        """Restore agent from checkpoint."""
        self.status = checkpoint.agent_status
        self.priority_queue = checkpoint.priority_queue.copy()
        self.state_metadata = checkpoint.state_metadata.copy()
        self.running_states = checkpoint.running_states.copy()
        self.completed_states = checkpoint.completed_states.copy()
        self.completed_once = checkpoint.completed_once.copy()
        self.shared_state = checkpoint.shared_state.copy()
        self.session_start = checkpoint.session_start

    async def save_checkpoint(self) -> str:
        """Save current state as checkpoint with persistent storage."""
        checkpoint = self.create_checkpoint()

        try:
            checkpoint_id = await self.checkpoint_storage.save_checkpoint(
                agent_name=self.name, checkpoint=checkpoint
            )
            logger.info(
                f"Checkpoint saved for agent {self.name} with ID: {checkpoint_id}"
            )
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to save checkpoint for agent {self.name}: {e}")
            raise

    async def load_checkpoint(self, checkpoint_id: Optional[str] = None) -> bool:
        """
        Load agent state from a checkpoint.

        Args:
            checkpoint_id: Specific checkpoint ID to load, or None for latest

        Returns:
            True if checkpoint was loaded successfully, False otherwise
        """
        try:
            checkpoint = await self.checkpoint_storage.load_checkpoint(
                agent_name=self.name, checkpoint_id=checkpoint_id
            )

            if checkpoint is None:
                logger.warning(f"No checkpoint found for agent {self.name}")
                return False

            await self.restore_from_checkpoint(checkpoint)
            logger.info(
                f"Agent {self.name} restored from checkpoint {checkpoint_id or 'latest'}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint for agent {self.name}: {e}")
            return False

    async def list_checkpoints(self) -> list[str]:
        """List available checkpoint IDs for this agent."""
        try:
            return await self.checkpoint_storage.list_checkpoints(self.name)
        except Exception as e:
            logger.error(f"Failed to list checkpoints for agent {self.name}: {e}")
            return []

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        try:
            success = await self.checkpoint_storage.delete_checkpoint(
                self.name, checkpoint_id
            )
            if success:
                logger.info(f"Deleted checkpoint {checkpoint_id} for agent {self.name}")
            return success
        except Exception as e:
            logger.error(
                f"Failed to delete checkpoint {checkpoint_id} for agent {self.name}: {e}"
            )
            return False

    # Execution control
    async def pause(self) -> AgentCheckpoint:
        """Pause agent execution and return checkpoint."""
        self.status = AgentStatus.PAUSED
        return self.create_checkpoint()

    async def resume(self) -> None:
        """Resume agent execution."""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.RUNNING

    # Find entry states
    def _find_entry_states(self) -> list[str]:
        """Find states with no dependencies."""
        entry_states = []
        for state_name in self.states:
            deps = self.dependencies.get(state_name, [])
            if not deps:
                entry_states.append(state_name)
        return entry_states

    async def _add_to_queue(self, state_name: str, priority_boost: int = 0) -> None:
        """Add state to priority queue."""
        if state_name not in self.state_metadata:
            logger.error(f"State {state_name} not found in metadata")
            return

        metadata = self.state_metadata[state_name]
        priority_value = metadata.priority.value + priority_boost

        prioritized_state = PrioritizedState(
            priority=-priority_value,  # Negative for max-heap behavior
            timestamp=time.time(),
            state_name=state_name,
            metadata=metadata,
        )

        import heapq

        heapq.heappush(self.priority_queue, prioritized_state)

    async def _get_ready_states(self) -> list[str]:
        """Get states that are ready to run."""
        ready_states = []
        temp_queue = []

        import heapq

        while self.priority_queue:
            state = heapq.heappop(self.priority_queue)
            if await self._can_run(state.state_name):
                ready_states.append(state.state_name)
            # If it can't run, only put it back if it hasn't completed.
            elif state.state_name not in self.completed_once:
                temp_queue.append(state)

        # Put non-ready states back
        for state in temp_queue:
            heapq.heappush(self.priority_queue, state)

        return ready_states

    async def run_state(self, state_name: str) -> None:
        """Execute a single state."""
        if state_name in self.running_states:
            return

        self.running_states.add(state_name)
        start_time = time.time()

        try:
            await self._execute_state_with_circuit_breaker(state_name, start_time)
        finally:
            self.running_states.discard(state_name)

    async def _execute_state_with_circuit_breaker(
        self, state_name: str, start_time: float
    ) -> None:
        """Execute state with circuit breaker protection."""
        try:
            async with self.circuit_breaker.protect():
                await self._execute_state_core(state_name, start_time)
        except Exception as e:
            await self._handle_state_failure(state_name, e, start_time)

    async def _execute_state_core(self, state_name: str, start_time: float) -> None:
        """Core state execution logic."""
        metadata = self.state_metadata[state_name]

        # Get timeout from resources or default
        state_timeout = None
        if metadata.resources and hasattr(metadata.resources, "timeout"):
            state_timeout = metadata.resources.timeout

        # Acquire resources (pass agent name for leak detection)
        resources = metadata.resources
        if resources is None and _ResourceRequirements is not None:
            resources = _ResourceRequirements()

        # Only try to acquire resources if we have a valid ResourceRequirements object
        if resources is not None:
            resource_acquired = await self.resource_pool.acquire(
                state_name, resources, timeout=state_timeout, agent_name=self.name
            )

            if not resource_acquired:
                raise ResourceTimeoutError(
                    f"Failed to acquire resources for state {state_name}"
                )

        try:
            # Execute the state function
            context = self._create_context(self.shared_state)

            # Execute with timeout if specified
            if state_timeout:
                result = await asyncio.wait_for(
                    self.states[state_name](context), timeout=state_timeout
                )
            else:
                result = await self.states[state_name](context)

            time.time() - start_time

            # Update metadata on success
            metadata.status = StateStatus.COMPLETED
            metadata.last_execution = time.time()
            metadata.last_success = time.time()
            metadata.attempts += 1

            # Track completion
            self.completed_states.add(state_name)
            self.completed_once.add(state_name)

            # Handle transitions/next states
            await self._handle_state_result(state_name, result)

            # Update shared state from context
            self.shared_state.update(context.shared_state)

        finally:
            # Always release resources if they were acquired
            if resources is not None:
                await self.resource_pool.release(state_name)

    async def _handle_state_result(self, state_name: str, result: StateResult) -> None:
        """Handle the result of state execution."""
        if result is None:
            return

        if isinstance(result, str):
            # Single next state
            if result in self.states and result not in self.completed_states:
                await self._add_to_queue(result)
        elif isinstance(result, list):
            # Multiple next states
            for next_state in result:
                if (
                    isinstance(next_state, str)
                    and next_state in self.states
                    and next_state not in self.completed_states
                ):
                    await self._add_to_queue(next_state)
                elif isinstance(next_state, tuple):
                    # Handle agent transition: (agent, state)
                    agent, state = next_state
                    if hasattr(agent, "add_to_queue"):
                        await agent._add_to_queue(state)

    async def _handle_state_failure(
        self, state_name: str, error: Exception, start_time: float
    ) -> None:
        """Handle state execution failure."""
        metadata = self.state_metadata[state_name]
        metadata.attempts += 1

        # Check if we've exceeded max retries
        if metadata.attempts >= metadata.max_retries:
            metadata.status = StateStatus.FAILED

            # Check for compensation state
            compensation_state = f"{state_name}_compensation"
            if compensation_state in self.states:
                await self._add_to_queue(compensation_state)

            # Determine if this should go to dead letter queue
            retry_policy = metadata.retry_policy or self.retry_policy
            should_dead_letter = (
                self.enable_dead_letter and retry_policy.dead_letter_on_max_retries
            )

            if should_dead_letter:
                dead_letter = DeadLetter(
                    state_name=state_name,
                    agent_name=self.name,
                    error_message=str(error),
                    error_type=type(error).__name__,
                    attempts=metadata.attempts,
                    failed_at=time.time(),
                    timeout_occurred=isinstance(error, asyncio.TimeoutError),
                    context_snapshot=dict(self.shared_state),
                )
                self.dead_letters.append(dead_letter)
        else:
            # Retry the state
            metadata.status = StateStatus.PENDING
            if metadata.retry_policy:
                await metadata.retry_policy.wait(metadata.attempts - 1)
            await self._add_to_queue(state_name)

    # Add alias for backward compatibility
    async def _handle_failure(
        self, state_name: str, error: Exception, start_time: Optional[float] = None
    ) -> None:
        """Handle state execution failure (alias for backward compatibility)."""
        if start_time is None:
            start_time = time.time()
        await self._handle_state_failure(state_name, error, start_time)

    async def _resolve_dependencies(self, state_name: str) -> None:
        """Resolve dependencies for a state."""
        deps = self.dependencies.get(state_name, [])
        unmet_deps = [dep for dep in deps if dep not in self.completed_states]

        if unmet_deps:
            logger.warning(f"State {state_name} has unmet dependencies: {unmet_deps}")

    async def _can_run(self, state_name: str) -> bool:
        """Check if a state can run."""
        if state_name in self.running_states:
            return False

        if state_name in self.completed_once:
            return False

        # Check dependencies
        deps = self.dependencies.get(state_name, [])
        return all(dep in self.completed_states for dep in deps)

    # State control
    def cancel_state(self, state_name: str) -> None:
        """Cancel a running or queued state."""
        # Remove from queue
        import heapq

        self.priority_queue = [
            s for s in self.priority_queue if s.state_name != state_name
        ]
        heapq.heapify(self.priority_queue)

        # Remove from running states
        self.running_states.discard(state_name)

        # Update metadata
        if state_name in self.state_metadata:
            self.state_metadata[state_name].status = StateStatus.CANCELLED

    async def cancel_all(self) -> None:
        """Cancel all running and queued states."""
        self.priority_queue.clear()
        self.running_states.clear()
        self.status = AgentStatus.CANCELLED

    # Information methods
    def get_resource_status(self) -> dict[str, Any]:
        """Get current resource status."""
        status = {
            "available": dict(self.resource_pool.available),
            "allocated": self.resource_pool.get_state_allocations(),
            "waiting": list(self.resource_pool.get_waiting_states()),
            "preempted": list(self.resource_pool.get_preempted_states()),
        }
        return status

    def get_state_info(self, state_name: str) -> dict[str, Any]:
        """Get information about a specific state."""
        if state_name not in self.states:
            return {}

        metadata = self.state_metadata.get(state_name)
        has_decorator = False
        try:
            from .decorators.inspection import is_puffinflow_state

            has_decorator = is_puffinflow_state(self.states[state_name])
        except ImportError:
            pass

        return {
            "name": state_name,
            "status": metadata.status if metadata else "unknown",
            "dependencies": self.dependencies.get(state_name, []),
            "has_decorator": has_decorator,
            "in_queue": any(s.state_name == state_name for s in self.priority_queue),
            "running": state_name in self.running_states,
            "completed": state_name in self.completed_states,
        }

    def list_states(self) -> list[dict[str, Any]]:
        """List all states with their information."""
        result = []
        for name in self.states:
            try:
                from .decorators.inspection import is_puffinflow_state

                has_decorator = is_puffinflow_state(self.states[name])
            except ImportError:
                has_decorator = False

            metadata = self.state_metadata.get(name)
            status = metadata.status if metadata is not None else "unknown"

            result.append(
                {
                    "name": name,
                    "has_decorator": has_decorator,
                    "dependencies": self.dependencies.get(name, []),
                    "status": status,
                }
            )
        return result

    # Dead letter management
    def get_dead_letters(self) -> list[DeadLetter]:
        """Get all dead letters."""
        return self.dead_letters.copy()

    def clear_dead_letters(self) -> None:
        """Clear all dead letters."""
        count = len(self.dead_letters)
        self.dead_letters.clear()
        logger.info(f"Cleared {count} dead letters for agent {self.name}")

    def get_dead_letter_count(self) -> int:
        """Get count of dead letters."""
        return len(self.dead_letters)

    def get_dead_letters_by_state(self, state_name: str) -> list[DeadLetter]:
        """Get dead letters for a specific state."""
        return [dl for dl in self.dead_letters if dl.state_name == state_name]

    # Circuit breaker control
    async def force_circuit_breaker_open(self) -> None:
        """Force circuit breaker to open state."""
        await self.circuit_breaker.force_open()

    async def force_circuit_breaker_close(self) -> None:
        """Force circuit breaker to close state."""
        await self.circuit_breaker.force_close()

    # Resource leak detection
    def check_resource_leaks(self) -> list[Any]:
        """Check for resource leaks."""
        return self.resource_pool.check_leaks()

    # Cleanup
    def add_cleanup_handler(self, handler: Callable) -> None:
        """Add cleanup handler."""
        self._cleanup_handlers.append(handler)

    async def cleanup(self) -> None:
        """Cleanup resources and handlers."""
        for handler in self._cleanup_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                logger.error(f"Error in cleanup handler: {e}")

    def _get_execution_metadata(self) -> dict[str, Any]:
        """Get execution metadata."""
        return {
            "states_completed": list(self.completed_states),
            "states_failed": [dl.state_name for dl in self.dead_letters],
            "total_states": len(self.states),
            "session_start": self.session_start,
            "dead_letter_count": len(self.dead_letters),
        }

    def _get_execution_metrics(self) -> dict[str, Any]:
        """Get execution metrics."""
        return {
            "completion_rate": len(self.completed_states) / max(1, len(self.states)),
            "error_rate": len(self.dead_letters) / max(1, len(self.states)),
            "resource_usage": self.get_resource_status(),
            "circuit_breaker_metrics": self.circuit_breaker.get_metrics(),
            "bulkhead_metrics": self.bulkhead.get_metrics(),
        }

    # Scheduling methods
    def schedule(self, when: str, **inputs: Any) -> "ScheduledAgent":
        """Schedule this agent to run at specified times with given inputs.

        Args:
            when: Schedule string (natural language or cron expression)
                Examples: "daily", "hourly", "every 5 minutes", "0 9 * * 1-5"
            **inputs: Input parameters with optional magic prefixes:
                - secret:value - Store as secret
                - const:value - Store as constant
                - cache:TTL:value - Store as cached with TTL
                - typed:value - Store as typed variable
                - output:value - Pre-set as output
                - value (no prefix) - Store as regular variable

        Returns:
            ScheduledAgent instance for managing the scheduled execution

        Raises:
            ImportError: If scheduling module is not available
            SchedulingError: If scheduling fails

        Examples:
            # Basic scheduling
            agent.schedule("daily at 09:00", source="database")

            # With magic prefixes
            agent.schedule(
                "every 30 minutes",
                api_key="secret:sk-1234567890abcdef",
                pool_size="const:10",
                config="cache:3600:{'timeout': 30}",
                source="warehouse"
            )
        """
        if not _SCHEDULING_AVAILABLE:
            raise ImportError(
                "Scheduling module not available. Install required dependencies."
            )

        scheduler = GlobalScheduler.get_instance_sync()
        return scheduler.schedule_agent(self, when, **inputs)

    def every(self, interval: str) -> "ScheduleBuilder":
        """Start fluent API for scheduling with intervals.

        Args:
            interval: Interval string like "5 minutes", "2 hours", "daily"

        Returns:
            ScheduleBuilder for chaining

        Examples:
            agent.every("5 minutes").with_inputs(source="api").run()
            agent.every("daily").with_secrets(api_key="sk-123").run()
        """
        if not _SCHEDULING_AVAILABLE:
            raise ImportError(
                "Scheduling module not available. Install required dependencies."
            )

        # Handle "every X" format - avoid double "every"
        if not interval.startswith("every "):
            interval = f"every {interval}"
        else:
            # If it already starts with "every", don't add another
            pass

        return ScheduleBuilder(self, interval)

    def daily(self, time_str: Optional[str] = None) -> "ScheduleBuilder":
        """Start fluent API for daily scheduling.

        Args:
            time_str: Optional time like "09:00" or "2pm"

        Returns:
            ScheduleBuilder for chaining

        Examples:
            agent.daily().with_inputs(batch_size=1000).run()
            agent.daily("09:00").with_secrets(db_pass="secret123").run()
        """
        if not _SCHEDULING_AVAILABLE:
            raise ImportError(
                "Scheduling module not available. Install required dependencies."
            )

        schedule_str = f"daily at {time_str}" if time_str else "daily"

        return ScheduleBuilder(self, schedule_str)

    def hourly(self, minute: Optional[int] = None) -> "ScheduleBuilder":
        """Start fluent API for hourly scheduling.

        Args:
            minute: Optional minute of the hour (0-59)

        Returns:
            ScheduleBuilder for chaining

        Examples:
            agent.hourly().with_inputs(check_status=True).run()
            agent.hourly(30).with_constants(timeout=60).run()
        """
        if not _SCHEDULING_AVAILABLE:
            raise ImportError(
                "Scheduling module not available. Install required dependencies."
            )

        schedule_str = f"every hour at {minute}" if minute is not None else "hourly"

        return ScheduleBuilder(self, schedule_str)

    # Main execution
    async def run(self, timeout: Optional[float] = None) -> AgentResult:
        """Run the agent workflow with enhanced result tracking."""
        start_time = time.time()
        self.status = AgentStatus.RUNNING

        if self.session_start is None:
            self.session_start = start_time

        try:
            # Check if we have any states
            if not self.states:
                logger.info("No states defined, nothing to run")
                self.status = AgentStatus.IDLE
                return AgentResult(
                    agent_name=self.name,
                    status=self.status,
                    start_time=start_time,
                    end_time=time.time(),
                    execution_duration=time.time() - start_time,
                )

            # Create context with current shared state
            self._create_context(self.shared_state)

            # Find entry states (states with no dependencies)
            entry_states = self._find_entry_states()
            if not entry_states:
                # If no entry states found, use all states (they will be filtered by dependencies during execution)
                entry_states = list(self.states.keys())

            # Add entry states to queue
            for state_name in entry_states:
                await self._add_to_queue(state_name)

            # Main execution loop
            while self.status == AgentStatus.RUNNING:
                if timeout and (time.time() - start_time) > timeout:
                    logger.warning(
                        f"Agent {self.name} timed out after {timeout} seconds."
                    )
                    self.status = AgentStatus.FAILED
                    break

                # Stop if there's nothing left to do
                if not self.priority_queue and not self.running_states:
                    break

                ready_states = await self._get_ready_states()

                if ready_states:
                    tasks = []
                    for state_name in ready_states[: self.max_concurrent]:
                        task = asyncio.create_task(self.run_state(state_name))
                        tasks.append(task)

                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)

                elif self.priority_queue and not self.running_states:
                    # States are in queue but none can run, and nothing is running
                    # This indicates a deadlock or unmeetable dependencies
                    logger.warning(
                        f"Deadlock in agent {self.name}: States in queue but none can run."
                    )
                    self.status = AgentStatus.FAILED
                    break
                else:
                    # No states ready, but some are running. Wait for them.
                    await asyncio.sleep(0.01)  # Short sleep to yield control

            # Determine final status
            if self.status == AgentStatus.RUNNING:
                has_failed_states = any(
                    s.status == StateStatus.FAILED for s in self.state_metadata.values()
                )
                if has_failed_states:
                    self.status = AgentStatus.FAILED
                else:
                    self.status = AgentStatus.COMPLETED

            end_time = time.time()

            # Create result
            result = AgentResult(
                agent_name=self.name,
                status=self.status,
                outputs=self.get_all_outputs(),
                variables={**self._agent_variables, **self.shared_state},
                metadata=self._get_execution_metadata(),
                metrics=self._get_execution_metrics(),
                start_time=start_time,
                end_time=end_time,
                execution_duration=end_time - start_time,
            )

            return result

        except Exception as e:
            self.status = AgentStatus.FAILED
            end_time = time.time()

            return AgentResult(
                agent_name=self.name,
                status=self.status,
                error=e,
                start_time=start_time,
                end_time=end_time,
                execution_duration=end_time - start_time,
            )

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if self._cleanup_handlers:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create cleanup task but don't store reference as object is being destroyed
                    task = loop.create_task(self.cleanup())
                    task.add_done_callback(lambda t: None)  # Prevent warnings
            except Exception:
                pass
