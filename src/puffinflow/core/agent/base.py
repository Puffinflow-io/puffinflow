"""Agent with direct access and coordination features."""

import asyncio
import contextlib
import heapq
import json
import logging
import pickle
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Protocol,
    Union,
)

from .checkpoint import AgentCheckpoint
from .checkpoint_serializer import CheckpointSerializer
from .command import Command, Send
from .context import Context
from .reducers import ReducerRegistry
from .state import (
    AgentStatus,
    DeadLetter,
    ExecutionMode,
    PrioritizedState,
    Priority,
    RetryPolicy,
    StateMetadata,
    StateResult,
    StateStatus,
)
from .streaming import StreamManager, StreamMode

# Import StateMachineCore and AgentCore (Rust or Python fallback)
try:
    from ._core import StateMachineCore

    _HAS_CORE = True
except Exception:
    _HAS_CORE = False

try:
    from ._core import _HAS_AGENT_CORE, AgentCore
except Exception:
    AgentCore = None
    _HAS_AGENT_CORE = False

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
    """File-based checkpoint storage with pluggable serialization."""

    def __init__(self, base_path: str = "./checkpoints", format: str = "json"):
        """
        Initialize file storage.

        Args:
            base_path: Directory to store checkpoint files
            format: Storage format ('json', 'msgpack', or 'pickle')
        """
        self.base_path = Path(base_path)
        self.format = format.lower()
        if self.format not in ("pickle", "json", "msgpack"):
            raise ValueError(
                f"Unsupported format: {format}. Use 'json', 'msgpack', or 'pickle'"
            )

        if self.format == "pickle":
            import warnings

            warnings.warn(
                "Pickle checkpoint format is deprecated and will be removed in a "
                "future version. Use format='json' (default) or format='msgpack' "
                "for portable, secure serialization.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Create serializer
        self._serializer: Optional[CheckpointSerializer] = None
        if self.format == "json":
            from .checkpoint_serializer import JsonCheckpointSerializer

            self._serializer = JsonCheckpointSerializer()
        elif self.format == "msgpack":
            from .checkpoint_serializer import MsgpackCheckpointSerializer

            self._serializer = MsgpackCheckpointSerializer()

        # Create directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, agent_name: str, checkpoint_id: str) -> Path:
        """Get file path for checkpoint."""
        agent_dir = self.base_path / agent_name
        agent_dir.mkdir(exist_ok=True)
        ext_map = {"pickle": "pkl", "json": "json", "msgpack": "msgpack"}
        ext = ext_map[self.format]
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
            else:
                # Use pluggable serializer (JSON or MsgPack)
                data = self._serializer.serialize(checkpoint)
                mode = "wb" if self.format == "msgpack" else "w"
                with file_path.open(mode) as f:
                    if self.format == "msgpack":
                        f.write(data)
                    else:
                        f.write(data.decode("utf-8"))

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
            else:
                # Use pluggable serializer
                mode = "rb" if self.format == "msgpack" else "r"
                with file_path.open(mode) as f:
                    raw = f.read()
                if isinstance(raw, str):
                    raw = raw.encode("utf-8")
                return self._serializer.deserialize(raw)

        except Exception as e:
            logger.error(f"Failed to load checkpoint from {file_path}: {e}")
            return None

    async def list_checkpoints(self, agent_name: str) -> list[str]:
        """List available checkpoint files."""
        agent_dir = self.base_path / agent_name
        if not agent_dir.exists():
            return []

        ext_map = {"pickle": "pkl", "json": "json", "msgpack": "msgpack"}
        ext = ext_map[self.format]
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

    async def cleanup_checkpoints(self, agent_name: str) -> int:
        """Remove old checkpoints (keep only the most recent 10)."""
        checkpoints = await self.list_checkpoints(agent_name)
        removed = 0
        if len(checkpoints) > 10:
            for cid in checkpoints[:-10]:
                if await self.delete_checkpoint(agent_name, cid):
                    removed += 1
        return removed


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

    async def cleanup_checkpoints(self, agent_name: str) -> int:
        """Remove old checkpoints (keep only the most recent 10)."""
        checkpoints = await self.list_checkpoints(agent_name)
        removed = 0
        if len(checkpoints) > 10:
            for cid in checkpoints[:-10]:
                if await self.delete_checkpoint(agent_name, cid):
                    removed += 1
        return removed


@dataclass
class AgentResult:
    """Rich result container for agent execution."""

    agent_name: str
    status: AgentStatus
    outputs: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    metadata: Optional[dict[str, Any]] = None
    metrics: Optional[dict[str, Any]] = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    execution_duration: Optional[float] = None
    _final_context: Optional[Context] = field(default=None, repr=False)

    def get_output(self, key: str, default: Any = None) -> Any:
        """Get output value."""
        return self.outputs.get(key, default)

    def get_variable(self, key: str, default: Any = None) -> Any:
        """
        Get variable value from any context storage type.

        This method is neutral and will look for the key across all storage types:
        - Regular variables
        - Typed variables
        - Constants
        - Secrets
        - Cached data
        - Validated data
        - Outputs
        """
        # First check regular variables
        if key in self.variables:
            return self.variables[key]

        # If we have final context, check other storage types
        if self._final_context:
            # Check constants
            const_value = self._final_context.get_constant(key)
            if const_value is not None:
                return const_value

            # Check secrets
            secret_value = self._final_context.get_secret(key)
            if secret_value is not None:
                return secret_value

            # Check cached data
            cached_value = self._final_context.get_cached(key)
            if cached_value is not None:
                return cached_value

            # Check outputs
            output_value = self._final_context.get_output(key)
            if output_value is not None:
                return output_value

            # Check if it's validated data (we can't know the type, so we'll check
            # if the key exists in shared_state)
            if (
                hasattr(self._final_context, "shared_state")
                and key in self._final_context.shared_state
            ):
                return self._final_context.shared_state[key]

        # Check outputs directly
        if key in self.outputs:
            return self.outputs[key]

        return default

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        if self.metadata is None:
            return default
        return self.metadata.get(key, default)

    def get_metric(self, key: str, default: Any = None) -> Any:
        """Get metric value."""
        if self.metrics is None:
            return default
        return self.metrics.get(key, default)

    # Advanced context methods
    def get_typed_variable(self, key: str, expected: Optional[type] = None) -> Any:
        """Get a typed variable with optional type checking."""
        if self._final_context:
            if expected is not None:
                return self._final_context.get_typed_variable(key, expected)
            else:
                # Return the raw value if no type checking is needed
                return self._final_context.get_variable(key)
        return self.variables.get(key)

    def get_validated_data(self, key: str, expected: type) -> Any:
        """Get validated Pydantic data."""
        if self._final_context:
            return self._final_context.get_validated_data(key, expected)
        return None

    def get_constant(self, key: str, default: Any = None) -> Any:
        """Get a constant value."""
        if self._final_context:
            return self._final_context.get_constant(key, default)
        return default

    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret value."""
        if self._final_context:
            return self._final_context.get_secret(key)
        return None

    def get_cached(self, key: str, default: Any = None) -> Any:
        """Get a cached value."""
        if self._final_context:
            return self._final_context.get_cached(key, default)
        return default

    def has_variable(self, key: str) -> bool:
        """Check if variable exists."""
        return key in self.variables

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


class _StateMetadataProxy:
    """Single-state metadata proxy. Reads from AgentCore on demand."""

    __slots__ = ("_core", "_name", "_extras", "_overrides", "_default_retry_policy")

    def __init__(
        self, core, name: str, extras: Optional[dict] = None, default_retry_policy=None
    ):
        self._core = core
        self._name = name
        self._extras = extras  # ref to agent._state_extras.get(name)
        self._overrides: Optional[dict] = None
        self._default_retry_policy = default_retry_policy

    def _set_override(self, key: str, value) -> None:
        if self._overrides is None:
            self._overrides = {}
        self._overrides[key] = value

    @property
    def status(self):
        if self._overrides and "status" in self._overrides:
            return self._overrides["status"]
        s = self._core.get_state_status(self._name)
        return StateStatus(s) if s != "pending" else StateStatus.PENDING

    @status.setter
    def status(self, value):
        self._set_override("status", value)

    @property
    def attempts(self) -> int:
        if self._overrides and "attempts" in self._overrides:
            return self._overrides["attempts"]
        return self._core.get_state_attempts(self._name)

    @attempts.setter
    def attempts(self, value):
        self._set_override("attempts", value)

    @property
    def max_retries(self) -> int:
        if self._overrides and "max_retries" in self._overrides:
            return self._overrides["max_retries"]
        return self._core.get_state_max_retries(self._name)

    @max_retries.setter
    def max_retries(self, value):
        self._set_override("max_retries", value)

    @property
    def priority(self):
        v = self._core.get_state_priority(self._name)
        try:
            return Priority(v)
        except ValueError:
            return Priority.NORMAL

    @property
    def last_execution(self) -> Optional[float]:
        if self._overrides and "last_execution" in self._overrides:
            return self._overrides["last_execution"]
        return self._core.get_state_last_execution(self._name)

    @last_execution.setter
    def last_execution(self, value):
        self._set_override("last_execution", value)

    @property
    def last_success(self) -> Optional[float]:
        if self._overrides and "last_success" in self._overrides:
            return self._overrides["last_success"]
        return self._core.get_state_last_success(self._name)

    @last_success.setter
    def last_success(self, value):
        self._set_override("last_success", value)

    @property
    def resources(self):
        if self._extras:
            return self._extras.get("resources")
        return None

    @property
    def state_id(self) -> str:
        return ""

    @property
    def retry_policy(self):
        if self._extras:
            rp = self._extras.get("retry_policy")
            if rp is not None:
                return rp
        return self._default_retry_policy

    @property
    def coordination_primitives(self) -> list:
        if self._extras:
            return self._extras.get("coordination_primitives", [])
        return []

    @property
    def dependencies(self) -> dict:
        return {}

    @property
    def satisfied_dependencies(self) -> set:
        return set()


class _StateMetadataProxyDict:
    """Dict-like proxy backed by AgentCore. No StateMetadata objects created."""

    __slots__ = ("_core", "_cache", "_extras", "_default_retry_policy")

    def __init__(self, core, extras: Optional[dict] = None, default_retry_policy=None):
        self._core = core
        self._cache: dict[str, _StateMetadataProxy] = {}
        self._extras = extras or {}
        self._default_retry_policy = default_retry_policy

    def __getitem__(self, name: str) -> _StateMetadataProxy:
        proxy = self._cache.get(name)
        if proxy is None:
            ex = self._extras.get(name) if self._extras else None
            proxy = _StateMetadataProxy(
                self._core, name, ex, self._default_retry_policy
            )
            self._cache[name] = proxy
        return proxy

    def __contains__(self, name: str) -> bool:
        return (
            name in self._core._name_to_idx
            if hasattr(self._core, "_name_to_idx")
            else name in self._core.get_all_state_names()
        )

    def __iter__(self):
        return iter(self._core.get_all_state_names())

    def __len__(self):
        return self._core.num_states()

    def get(self, name: str, default=None):
        try:
            names = self._core.get_all_state_names()
        except Exception:
            return default
        if name in names:
            return self[name]
        return default

    def values(self):
        for name in self._core.get_all_state_names():
            yield self[name]

    def items(self):
        for name in self._core.get_all_state_names():
            yield name, self[name]

    def keys(self):
        return iter(self._core.get_all_state_names())

    def __deepcopy__(self, memo):
        """Materialize as a plain dict of StateMetadata for checkpoint compatibility."""
        from copy import deepcopy

        result = {}
        for name in self._core.get_all_state_names():
            proxy = self[name]
            sm = StateMetadata(
                status=proxy.status,
                priority=proxy.priority,
                max_retries=proxy.max_retries,
                attempts=proxy.attempts,
                last_execution=proxy.last_execution,
                last_success=proxy.last_success,
                resources=proxy.resources,
                retry_policy=proxy.retry_policy,
                coordination_primitives=proxy.coordination_primitives,
            )
            result[name] = deepcopy(sm, memo)
        return result

    def __copy__(self):
        return self.__deepcopy__({})


class Agent:
    """Enhanced Agent with direct variable access and coordination features."""

    # Shared default RetryPolicy — avoids per-agent allocation
    _DEFAULT_RETRY_POLICY = RetryPolicy()

    # Auto-discovered @state-decorated methods — populated by __init_subclass__
    _puffinflow_auto_states: tuple[tuple[str, str], ...] = ()
    # Pre-computed sorted order with extracted metadata (state_name, method_name, deps, priority, resources)
    _puffinflow_sorted_states: Optional[
        tuple[tuple[str, str, list[str], Optional[Priority], Optional[Any]], ...]
    ] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Walk MRO (reversed so derived classes override base) and collect
        # methods marked with @state decorator (_puffinflow_state == True).
        seen: dict[str, str] = {}  # state_name -> method_name
        for klass in reversed(cls.__mro__):
            for attr_name, attr_value in vars(klass).items():
                if callable(attr_value) and getattr(
                    attr_value, "_puffinflow_state", False
                ):
                    state_name = getattr(attr_value, "_state_name", attr_name)
                    seen[state_name] = attr_name
        cls._puffinflow_auto_states = tuple(seen.items())

        # Pre-compute sorted order and extract metadata at class definition time
        if seen:
            cls._puffinflow_sorted_states = cls._compute_sorted_states(seen)

    @classmethod
    def _compute_sorted_states(
        cls, seen: dict[str, str]
    ) -> tuple[tuple[str, str, list[str], Optional[Priority], Optional[Any]], ...]:
        """Pre-compute topological sort and extract metadata at class definition time."""
        # Extract deps from decorator metadata
        deps_map: dict[str, list[str]] = {}
        for state_name, method_name in seen.items():
            # Walk MRO to find the actual function
            func = None
            for klass in cls.__mro__:
                if method_name in vars(klass):
                    func = vars(klass)[method_name]
                    break
            if func is None:
                deps_map[state_name] = []
                continue
            config = getattr(func, "_state_config", None)
            deps = config.get("depends_on", []) if isinstance(config, dict) else []
            deps_map[state_name] = deps

        # Check if any states have dependencies — if not, skip sort
        has_deps = any(deps_map.values())
        if has_deps:
            # Kahn's algorithm
            pending_set = set(seen.keys())
            in_degree: dict[str, int] = {sn: 0 for sn in pending_set}
            for sn, deps in deps_map.items():
                for d in deps:
                    if d in pending_set:
                        in_degree[sn] += 1

            queue = sorted(sn for sn, deg in in_degree.items() if deg == 0)
            sorted_names: list[str] = []
            while queue:
                node = queue.pop(0)
                sorted_names.append(node)
                for sn, deps in deps_map.items():
                    if node in deps and sn in in_degree:
                        in_degree[sn] -= 1
                        if in_degree[sn] == 0:
                            queue.append(sn)
                            queue.sort()
            # Add remaining (circular/missing deps — let add_state catch it)
            for sn in pending_set - set(sorted_names):
                sorted_names.append(sn)
        else:
            # No dependencies — use insertion order (deterministic)
            sorted_names = list(seen.keys())

        # Extract metadata for each state
        result: list[tuple[str, str, list[str], Optional[Priority], Optional[Any]]] = []
        for state_name in sorted_names:
            method_name = seen[state_name]
            func = None
            for klass in cls.__mro__:
                if method_name in vars(klass):
                    func = vars(klass)[method_name]
                    break
            if func is None:
                result.append((state_name, method_name, [], None, None))
                continue
            deps = deps_map.get(state_name, [])
            priority = getattr(func, "_priority", None)
            resources = getattr(func, "_resource_requirements", None)
            result.append((state_name, method_name, deps, priority, resources))

        return tuple(result)

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

        # AgentCore owns: deps, dependents, metadata, queue, bitsets, validation
        self._core = AgentCore(name) if _HAS_AGENT_CORE else None
        self._sm_proxy: Optional[_StateMetadataProxyDict] = None
        self._state_extras: Optional[dict[str, dict]] = None  # resources, coord_prims

        # Fallback state_metadata dict (only used when _core is None)
        self._fallback_state_metadata: dict[str, StateMetadata] = {}

        # When _core is available, skip allocating tracking containers that duplicate Rust state
        _has_core = self._core is not None
        # Dependencies always populated (needed by slow path even when _core exists)
        self._dependencies_dict: dict[str, list[str]] = {}
        self._dependents_dict: dict[str, list[str]] = {}
        self._deps_snapshot: Optional[
            int
        ] = None  # id() of deps dict values at last add_state
        self._use_fast_path: bool = False
        self._validated: bool = False
        self.status = AgentStatus.IDLE
        self._shared_state: Optional[dict[str, Any]] = None
        # Tracking containers: lazy when _core is available
        self._priority_queue: Optional[list[PrioritizedState]] = (
            None if _has_core else []
        )
        self._running_states_set: Optional[set[str]] = None if _has_core else set()
        self._completed_states_set: Optional[set[str]] = None if _has_core else set()
        self._completed_once_set: Optional[set[str]] = None if _has_core else set()
        self.session_start: Optional[float] = None

        # Configuration
        self.max_concurrent = max_concurrent
        self.state_timeout = state_timeout
        self._checkpoint_storage = checkpoint_storage  # Lazy — see property

        # Enhanced features — None sentinels, created on first use
        self._context: Optional[Context] = None
        self._variable_watchers: Optional[dict[str, list[Callable]]] = None
        self._shared_variable_watchers: Optional[dict[str, list[Callable]]] = None
        self._agent_variables: Optional[dict[str, Any]] = None
        self._persistent_variables: Optional[dict[str, Any]] = None
        self._property_definitions: Optional[dict[str, dict]] = None
        self._team: Optional[weakref.ReferenceType] = None
        self._message_handlers: Optional[dict[str, Callable]] = None
        self._event_handlers: Optional[dict[str, list[Callable]]] = None
        self._state_change_handlers: Optional[list[Callable]] = None
        self.dead_letters: Optional[list[DeadLetter]] = None

        # Resource and reliability components - lazy initialization
        self._resource_pool = resource_pool
        self._circuit_breaker: Optional["CircuitBreaker"] = None
        self._bulkhead: Optional["Bulkhead"] = None
        self._circuit_breaker_config = circuit_breaker_config
        self._bulkhead_config = bulkhead_config

        self.retry_policy = retry_policy or Agent._DEFAULT_RETRY_POLICY
        self.enable_dead_letter = enable_dead_letter
        self._cleanup_handlers: Optional[list[Callable]] = None
        self._auto_discovered: bool = False

        # New feature attributes — zero overhead when unused
        self._reducers: Optional[ReducerRegistry] = None
        self._store: Optional[Any] = kwargs.get("store", None)
        self._stream_manager: Optional[StreamManager] = None

        # Durable execution attributes
        self._drain_protocol: Optional[Any] = None  # DrainProtocol
        self._pending_events: Optional[dict[str, asyncio.Event]] = None
        self._event_results: Optional[dict[str, Any]] = None

    # --- Property wrappers for lazy/core-backed containers ---

    @property
    def dependencies(self) -> dict[str, list[str]]:
        if self._dependencies_dict is None:
            self._dependencies_dict = {}
        return self._dependencies_dict

    @dependencies.setter
    def dependencies(self, value: dict[str, list[str]]) -> None:
        self._dependencies_dict = value

    @property
    def _dependents(self) -> dict[str, list[str]]:
        if self._dependents_dict is None:
            self._dependents_dict = {}
        return self._dependents_dict

    @_dependents.setter
    def _dependents(self, value: dict[str, list[str]]) -> None:
        self._dependents_dict = value

    @property
    def shared_state(self) -> dict[str, Any]:
        if self._shared_state is None:
            self._shared_state = {}
        return self._shared_state

    @shared_state.setter
    def shared_state(self, value: dict[str, Any]) -> None:
        self._shared_state = value

    @property
    def priority_queue(self) -> list[PrioritizedState]:
        if self._priority_queue is None:
            self._priority_queue = []
        return self._priority_queue

    @priority_queue.setter
    def priority_queue(self, value: list[PrioritizedState]) -> None:
        self._priority_queue = value

    @property
    def running_states(self) -> set[str]:
        if self._running_states_set is None:
            self._running_states_set = set()
        return self._running_states_set

    @running_states.setter
    def running_states(self, value: set[str]) -> None:
        self._running_states_set = value

    @property
    def completed_states(self) -> set[str]:
        if self._completed_states_set is not None:
            return self._completed_states_set
        if self._core is not None:
            self._completed_states_set = set(self._core.get_completed_states())
        else:
            self._completed_states_set = set()
        return self._completed_states_set

    @completed_states.setter
    def completed_states(self, value: set[str]) -> None:
        self._completed_states_set = value

    @property
    def completed_once(self) -> set[str]:
        if self._completed_once_set is not None:
            return self._completed_once_set
        if self._core is not None:
            self._completed_once_set = set(self._core.get_completed_once())
        else:
            self._completed_once_set = set()
        return self._completed_once_set

    @completed_once.setter
    def completed_once(self, value: set[str]) -> None:
        self._completed_once_set = value

    @property
    def state_metadata(self):
        """State metadata dict — proxy backed by AgentCore when available."""
        if self._core is not None:
            if self._sm_proxy is None:
                self._sm_proxy = _StateMetadataProxyDict(
                    self._core, self._state_extras, self.retry_policy
                )
            return self._sm_proxy
        return self._fallback_state_metadata

    @state_metadata.setter
    def state_metadata(self, value):
        """Setter for backward compat (checkpoint restore)."""
        self._fallback_state_metadata = value

    @property
    def checkpoint_storage(self):
        """Lazy checkpoint storage — only created if accessed."""
        if self._checkpoint_storage is None:
            self._checkpoint_storage = MemoryCheckpointStorage()
        return self._checkpoint_storage

    @checkpoint_storage.setter
    def checkpoint_storage(self, value) -> None:
        self._checkpoint_storage = value

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
        if self._agent_variables:
            return self._agent_variables.get(key, default)
        return default

    def set_variable(self, key: str, value: Any) -> None:
        """Set variable directly on agent context or internal storage."""
        if self._context:
            old_value = self._context.get_variable(key)
            self._context.set_variable(key, value)
            self._trigger_variable_watchers(key, old_value, value)
        else:
            if self._agent_variables is None:
                self._agent_variables = {}
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
        if self._agent_variables is None:
            return default
        return self._agent_variables.get(key, default)

    def set_agent_variable(self, key: str, value: Any) -> None:
        """Set agent-specific variable (not shared)."""
        if self._agent_variables is None:
            self._agent_variables = {}
        old_value = self._agent_variables.get(key)
        self._agent_variables[key] = value
        self._trigger_variable_watchers(key, old_value, value)

    def get_persistent_variable(self, key: str, default: Any = None) -> Any:
        """Get persistent variable that survives restarts."""
        if self._persistent_variables is None:
            return default
        return self._persistent_variables.get(key, default)

    def set_persistent_variable(self, key: str, value: Any) -> None:
        """Set persistent variable that survives restarts."""
        if self._persistent_variables is None:
            self._persistent_variables = {}
        self._persistent_variables[key] = value

    # Context content access methods
    def _ensure_context(self) -> "Context":
        """Lazily create context if it doesn't exist yet."""
        if self._context is None:
            self._create_context(self.shared_state)
        assert self._context is not None
        return self._context

    def get_output(self, key: str, default: Any = None) -> Any:
        """Get output value from context."""
        if self._context:
            return self._context.get_output(key, default)
        return default

    def set_output(self, key: str, value: Any) -> None:
        """Set output value in context."""
        self._ensure_context().set_output(key, value)

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
        self._ensure_context().set_metadata(key, value)

    def get_cached(self, key: str, default: Any = None) -> Any:
        """Get cached value."""
        if self._context:
            return self._context.get_cached(key, default)
        return default

    def set_cached(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with optional TTL."""
        self._ensure_context().set_cached(key, value, ttl)

    # Property system
    def define_property(
        self,
        name: str,
        prop_type: type,
        default: Any = None,
        validator: Optional[Callable] = None,
    ) -> None:
        """Define a typed property with validation."""
        if self._property_definitions is None:
            self._property_definitions = {}
        self._property_definitions[name] = {
            "type": prop_type,
            "default": default,
            "validator": validator,
        }

        # Set default value if not already set
        if self._agent_variables is None or name not in self._agent_variables:
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
        if self._variable_watchers is None:
            self._variable_watchers = {}
        if key not in self._variable_watchers:
            self._variable_watchers[key] = []
        self._variable_watchers[key].append(handler)

    def watch_shared_variable(self, key: str, handler: Callable) -> None:
        """Watch for changes to a shared variable."""
        if self._shared_variable_watchers is None:
            self._shared_variable_watchers = {}
        if key not in self._shared_variable_watchers:
            self._shared_variable_watchers[key] = []
        self._shared_variable_watchers[key].append(handler)

    def _trigger_variable_watchers(
        self, key: str, old_value: Any, new_value: Any
    ) -> None:
        """Trigger watchers for variable changes."""
        if (
            self._variable_watchers
            and key in self._variable_watchers
            and old_value != new_value
        ):
            for handler in self._variable_watchers[key]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        task = asyncio.create_task(handler(old_value, new_value))
                        # Store task reference to prevent garbage collection
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
        if (
            self._shared_variable_watchers
            and key in self._shared_variable_watchers
            and old_value != new_value
        ):
            for handler in self._shared_variable_watchers[key]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        task = asyncio.create_task(handler(old_value, new_value))
                        # Store task reference to prevent garbage collection
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
        if self._state_change_handlers is None:
            self._state_change_handlers = []
        self._state_change_handlers.append(handler)

    def _trigger_state_change(self, old_state: Any, new_state: Any) -> None:
        """Trigger state change handlers."""
        if not self._state_change_handlers:
            return
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
            if self._message_handlers is None:
                self._message_handlers = {}
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
        if self._message_handlers and message_type in self._message_handlers:
            result = await self._message_handlers[message_type](message, sender)
            return dict(result) if result else {}
        return {}

    # Event system
    def on_event(self, event_type: str) -> Callable:
        """Decorator for event handlers."""

        def decorator(func: Callable) -> Callable:
            if self._event_handlers is None:
                self._event_handlers = {}
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
        if self._event_handlers and event_type in self._event_handlers:
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
        if self._agent_variables:
            for key, value in self._agent_variables.items():
                context.set_variable(key, value)

        # Propagate new feature objects to context
        if self._store is not None:
            context._store = self._store
        if self._stream_manager is not None:
            context._stream = self._stream_manager
        if self._reducers is not None:
            context._reducers = self._reducers

        self._context = context
        return context

    # --- New feature methods ---

    def add_reducer(self, key: str, reducer: Callable) -> None:
        """Register a reducer for the given shared-state key."""
        if self._reducers is None:
            self._reducers = ReducerRegistry()
        self._reducers.register(key, reducer)

    def _apply_command(self, cmd: Command, ctx: Any) -> None:
        """Apply a Command's update dict to context shared_state, using reducers."""
        if cmd.update:
            for key, value in cmd.update.items():
                if self._reducers is not None and self._reducers.has(key):
                    existing = ctx.shared_state.get(key)
                    ctx.shared_state[key] = self._reducers.apply(key, existing, value)
                else:
                    ctx.shared_state[key] = value

    async def _execute_send_branches(self, sends: list[Send], ctx: Any) -> None:
        """Execute Send branches in parallel with isolated contexts."""
        tasks = []
        for send in sends:
            if send.state not in self.states:
                continue
            # Create an isolated context copy with Send payload merged
            child_shared = dict(ctx.shared_state)
            child_shared.update(send.payload)
            child_ctx = Context(child_shared)
            # Propagate store, stream, reducers to child context
            child_ctx._store = getattr(ctx, "_store", None)
            child_ctx._stream = getattr(ctx, "_stream", None)
            child_ctx._reducers = self._reducers

            async def _run_send(_state_name: str, _child_ctx: Context) -> Any:
                return await self.states[_state_name](_child_ctx)

            tasks.append(asyncio.create_task(_run_send(send.state, child_ctx)))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Apply Command results from branches using reducers
            for _, res in enumerate(results):
                if isinstance(res, Exception):
                    continue
                if isinstance(res, Command):
                    self._apply_command(res, ctx)

    def add_subgraph(
        self,
        name: str,
        child_agent: "Agent",
        input_map: Optional[dict[str, str]] = None,
        output_map: Optional[dict[str, str]] = None,
        dependencies: Optional[list[str]] = None,
        priority: Optional[Any] = None,
    ) -> None:
        """Add a child agent as a subgraph state."""
        from .subgraph import StateMapping, make_subgraph_state

        state_map = StateMapping(
            inputs=input_map or {},
            outputs=output_map or {},
        )
        state_fn = make_subgraph_state(
            child_agent,
            state_map,
            stream_manager=self._stream_manager,
            store=self._store,
        )
        self.add_state(name, state_fn, dependencies=dependencies or [])

    async def stream(
        self,
        mode: StreamMode = StreamMode.EVENTS,
        **run_kwargs: Any,
    ):
        """Async generator that yields StreamEvents during agent execution.

        Usage::

            async for event in agent.stream():
                print(event.event_type, event.data)
        """
        sm = StreamManager(mode=mode)
        self._stream_manager = sm

        async def _run_and_close() -> None:
            try:
                await self.run(**run_kwargs)
            finally:
                sm.close()
                self._stream_manager = None

        task = asyncio.create_task(_run_and_close())
        try:
            async for event in sm:
                yield event
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    def _apply_initial_context(self, initial_context: dict[str, Any]) -> None:
        """Apply initial context data with support for different data types."""
        if not self._context:
            raise RuntimeError(
                "Context must be created before applying initial context"
            )

        # Check if it's a structured context (has known section keys)
        structured_keys = {
            "variables",
            "typed_variables",
            "constants",
            "secrets",
            "validated_data",
            "cached",
            "outputs",
            "metadata",
        }

        is_structured = any(key in structured_keys for key in initial_context)

        if is_structured:
            # Handle structured format
            if "variables" in initial_context:
                for key, value in initial_context["variables"].items():
                    self._context.set_variable(key, value)

            if "typed_variables" in initial_context:
                for key, value in initial_context["typed_variables"].items():
                    self._context.set_typed_variable(key, value)

            if "constants" in initial_context:
                for key, value in initial_context["constants"].items():
                    self._context.set_constant(key, value)

            if "secrets" in initial_context:
                for key, value in initial_context["secrets"].items():
                    if not isinstance(value, str):
                        raise TypeError(f"Secret '{key}' must be a string")
                    self._context.set_secret(key, value)

            if "validated_data" in initial_context:
                for key, value in initial_context["validated_data"].items():
                    self._context.set_validated_data(key, value)

            if "cached" in initial_context:
                for key, cache_config in initial_context["cached"].items():
                    if isinstance(cache_config, dict) and "value" in cache_config:
                        value = cache_config["value"]
                        ttl = cache_config.get("ttl", 300)  # Default 5 minutes
                        self._context.set_cached(key, value, ttl)
                    else:
                        # Simple value, use default TTL
                        self._context.set_cached(key, cache_config)

            if "outputs" in initial_context:
                for key, value in initial_context["outputs"].items():
                    self._context.set_output(key, value)

            if "metadata" in initial_context:
                for key, value in initial_context["metadata"].items():
                    self._context.set_metadata(key, value)
        else:
            # Handle simple format - treat all as regular variables
            for key, value in initial_context.items():
                self._context.set_variable(key, value)

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
        entry_point: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        """Add a state to the agent.

        Args:
            entry_point: Controls whether this state is selected as an entry
                point in PARALLEL mode. ``True`` forces it as an entry point,
                ``False`` excludes it (only reachable via return-value routing
                or dependency auto-trigger). ``None`` (default) infers from
                dependencies: states with no dependencies are entry points.
        """
        # Validate state name
        if not name or not isinstance(name, str):
            raise ValueError("State name must be a non-empty string")

        if name in self.states:
            raise ValueError(f"State '{name}' already exists in agent '{self.name}'")

        # Validate dependencies exist
        dependencies = dependencies or []
        for dep in dependencies:
            if dep not in self.states:
                raise ValueError(
                    f"Dependency '{dep}' for state '{name}' does not exist. "
                    f"Add dependency states before states that depend on them."
                )

        self.states[name] = func

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

        # Use max_retries parameter or fall back to retry_policy or agent default
        final_max_retries = max_retries or (
            retry_policy.max_retries if retry_policy else self.retry_policy.max_retries
        )

        # Resolve retry config
        rp = retry_policy or self.retry_policy
        retry_delay = rp.initial_delay if rp else 1.0
        retry_base = rp.exponential_base if rp else 2.0
        retry_jitter = rp.jitter if rp else True

        if self._core is not None:
            # Register in AgentCore — no StateMetadata dataclass, no uuid
            self._core.add_state(
                name,
                final_priority.value,
                final_max_retries,
                dependencies,
                retry_delay,
                retry_base,
                retry_jitter,
            )
            # Invalidate proxy cache
            self._sm_proxy = None

            # Store extras when non-None (resources, coordination_primitives, retry_policy)
            if (
                final_requirements is not None
                or coordination_primitives
                or retry_policy
            ):
                if self._state_extras is None:
                    self._state_extras = {}
                extras: dict[str, Any] = {}
                if final_requirements is not None:
                    extras["resources"] = final_requirements
                if coordination_primitives:
                    extras["coordination_primitives"] = coordination_primitives
                if retry_policy:
                    extras["retry_policy"] = retry_policy
                self._state_extras[name] = extras
        else:
            # Fallback: create StateMetadata dataclass
            metadata = StateMetadata(
                status=StateStatus.PENDING,
                priority=final_priority,
                resources=final_requirements,
                retry_policy=retry_policy or self.retry_policy,
                coordination_primitives=coordination_primitives or [],
                max_retries=final_max_retries,
            )
            self._fallback_state_metadata[name] = metadata

        # Keep dep tracking for backward compat (slow path, checkpoint, etc.)
        self.dependencies[name] = dependencies
        # Snapshot: sum of id()s of all dep lists — changes if any value is reassigned
        self._deps_snapshot = sum(id(v) for v in self._dependencies_dict.values())
        for dep in dependencies:
            if dep not in self._dependents:
                self._dependents[dep] = []
            self._dependents[dep].append(name)

        # Track explicit entry_point overrides
        if entry_point is not None:
            if not hasattr(self, "_entry_point_overrides"):
                self._entry_point_overrides: dict[str, bool] = {}
            self._entry_point_overrides[name] = entry_point

    def _auto_discover_states(self) -> None:
        """Register @state-decorated methods that weren't manually added via add_state().

        Called once at the start of run(). Uses pre-computed sorted order from
        __init_subclass__ to avoid re-running topological sort per instance.
        """
        if self._auto_discovered:
            return
        self._auto_discovered = True

        sorted_states = self._puffinflow_sorted_states
        if sorted_states is None:
            return

        # Use pre-computed sorted order with extracted metadata
        _states = self.states
        _any_added = False
        for state_name, method_name, deps, priority, resources in sorted_states:
            if state_name in _states:
                continue
            bound_method = getattr(self, method_name)
            self.add_state(
                state_name,
                bound_method,
                dependencies=deps if deps else None,
                resources=resources,
                priority=priority,
            )
            _any_added = True

        # Compute _deps_snapshot once after all states added (instead of per add_state)
        if _any_added:
            self._deps_snapshot = sum(
                id(v) for v in self._dependencies_dict.values()
            )

    def _validate_workflow_configuration(self, execution_mode: ExecutionMode) -> None:
        """Validate the overall workflow configuration before execution."""
        self._auto_discover_states()
        if not self.states:
            raise ValueError(
                "No states defined. Agent must have at least one state to run."
            )

        # Check for circular dependencies
        self._check_circular_dependencies()

        # Validate execution mode configuration
        if execution_mode == ExecutionMode.SEQUENTIAL:
            self._validate_sequential_mode()
        elif execution_mode == ExecutionMode.PARALLEL:
            self._validate_parallel_mode()

    def _check_circular_dependencies(self) -> None:
        """Check for circular dependencies in the state graph."""

        def has_cycle(state: str, visited: set, rec_stack: set) -> bool:
            visited.add(state)
            rec_stack.add(state)

            # Follow dependencies from this state
            for dep in self.dependencies.get(state, []):
                if dep not in self.states:
                    # Skip non-existent dependencies
                    continue
                if dep not in visited:
                    if has_cycle(dep, visited, rec_stack):
                        return True
                elif dep in rec_stack:
                    # Found a back edge - cycle detected
                    return True

            rec_stack.remove(state)
            return False

        visited: set[str] = set()
        for state in self.states:
            if state not in visited and has_cycle(state, visited, set()):
                raise ValueError(
                    "Circular dependency detected in workflow. "
                    "State dependencies form a cycle, which would prevent execution."
                )

    def _validate_sequential_mode(self) -> None:
        """Validate configuration for sequential execution mode."""
        entry_states = self._find_entry_states()

        if not entry_states:
            raise ValueError(
                "Sequential execution mode requires at least one state without dependencies "
                "to serve as an entry point. All states have dependencies, creating a deadlock."
            )

        # Check if all states are reachable
        reachable = set()
        to_visit = entry_states.copy()

        while to_visit:
            current = to_visit.pop(0)
            if current in reachable:
                continue
            reachable.add(current)

            # Find states that depend on current state
            for state, deps in self.dependencies.items():
                if current in deps and state not in reachable:
                    to_visit.append(state)

        unreachable = set(self.states.keys()) - reachable
        if unreachable:
            logger.warning(
                f"States {unreachable} are unreachable in sequential mode. "
                f"They have dependencies that will never be satisfied or lack proper transitions."
            )

    def _validate_parallel_mode(self) -> None:
        """Validate configuration for parallel execution mode."""
        entry_states = self._find_entry_states()

        if not entry_states:
            logger.warning(
                "No entry states found for parallel mode. All states have dependencies. "
                "This may prevent execution unless states return proper transitions."
            )

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
                f"Agent {self.name} restored from checkpoint "
                f"{checkpoint_id or 'latest'}"
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
                f"Failed to delete checkpoint {checkpoint_id} for agent "
                f"{self.name}: {e}"
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
        """Find states that should be entry points.

        A state is an entry point if:
        - It has ``entry_point=True`` explicitly set, OR
        - It has no dependencies AND no ``entry_point=False`` override.
        """
        entry_states = []
        state_names = list(self.states.keys())  # Preserve order
        overrides = getattr(self, "_entry_point_overrides", {})

        for state_name in state_names:
            override = overrides.get(state_name)
            if override is True:
                # Explicitly marked as entry point
                entry_states.append(state_name)
            elif override is False:
                # Explicitly excluded from entry points
                continue
            else:
                # Default: entry point if no dependencies
                deps = self.dependencies.get(state_name, [])
                if not deps:
                    entry_states.append(state_name)

        return entry_states

    def _find_entry_states_by_mode(self, execution_mode: ExecutionMode) -> list[str]:
        """Find entry states based on execution mode."""
        if execution_mode == ExecutionMode.PARALLEL:
            # PARALLEL mode: All entry-point states run initially
            entry_states = self._find_entry_states()
            if not entry_states:
                # If no entry states found, use all states for backward compatibility
                entry_states = list(self.states.keys())
            return entry_states

        # execution_mode == ExecutionMode.SEQUENTIAL
        # SEQUENTIAL mode: Only the first entry-point state runs initially
        true_entry_states = self._find_entry_states()
        if true_entry_states:
            # Return only the first entry state to enforce sequential execution
            return [true_entry_states[0]]
        else:
            # If no true entry states, return the first state
            state_names = list(self.states.keys())
            return [state_names[0]] if state_names else []

    async def _check_dependent_states(self, completed_state: str) -> None:
        """Check for states that depend on the completed state and queue them
        if ready."""
        for state_name in self._dependents.get(completed_state, []):
            # Skip if state is already completed or running
            if state_name in self.completed_once or state_name in self.running_states:
                continue

            # Skip if state is already in queue
            if any(ps.state_name == state_name for ps in self.priority_queue):
                continue

            # Check if all dependencies are now satisfied
            if self._can_run(state_name):
                await self._add_to_queue(state_name)

    def _check_dependent_states_fast(self, completed_state: str) -> None:
        """Fast sync version — uses reverse dep index and sync queue add."""
        for state_name in self._dependents.get(completed_state, []):
            if state_name in self.completed_once or state_name in self.running_states:
                continue
            if any(ps.state_name == state_name for ps in self.priority_queue):
                continue
            if self._can_run(state_name):
                self._add_to_queue_fast(state_name)

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

        heapq.heappush(self.priority_queue, prioritized_state)

    def _add_to_queue_fast(self, state_name: str, priority_boost: int = 0) -> None:
        """Sync version of _add_to_queue — no async overhead."""
        metadata = self.state_metadata[state_name]
        ps = PrioritizedState(
            priority=-(metadata.priority.value + priority_boost),
            timestamp=time.time(),
            state_name=state_name,
            metadata=metadata,
        )
        heapq.heappush(self.priority_queue, ps)

    def _get_ready_states(self) -> list[str]:
        """Get states that are ready to run."""
        ready = []
        remaining = []

        while self.priority_queue:
            state = heapq.heappop(self.priority_queue)
            if self._can_run(state.state_name):
                ready.append(state.state_name)
            elif state.state_name not in self.completed_once:
                remaining.append(state)

        # Put non-ready states back
        for s in remaining:
            heapq.heappush(self.priority_queue, s)

        return ready

    async def _execute_state_fast(self, state_name: str) -> None:
        """Fast execution path — no circuit breaker, no resource pool.
        Reuses self._context instead of creating a new one each step."""
        self.running_states.add(state_name)
        try:
            if self._stream_manager is not None:
                self._stream_manager.emit_node_start(state_name)

            result = await self.states[state_name](self._context)

            if self._stream_manager is not None:
                self._stream_manager.emit_node_complete(state_name, result)

            metadata = self.state_metadata[state_name]
            metadata.status = StateStatus.COMPLETED
            now = time.monotonic()
            metadata.last_execution = now
            metadata.last_success = now
            metadata.attempts += 1

            self.completed_states.add(state_name)
            self.completed_once.add(state_name)
            self._check_dependent_states_fast(state_name)
            self._handle_state_result_fast(state_name, result)
        except Exception as e:
            await self._handle_state_failure(state_name, e, 0.0)
        finally:
            self.running_states.discard(state_name)

    def _handle_state_result_fast(self, state_name: str, result) -> None:
        """Sync version of _handle_state_result — no async overhead."""
        if result is None:
            return
        # Command pattern: apply update then route via goto
        if isinstance(result, Command):
            if result.update and self._context is not None:
                self._apply_command(result, self._context)
            result = result.goto
            if result is None:
                return
        if isinstance(result, str):
            if result in self.states and result not in self.completed_states:
                self._add_to_queue_fast(result)
        elif isinstance(result, list):
            for ns in result:
                if (
                    isinstance(ns, str)
                    and ns in self.states
                    and ns not in self.completed_states
                ):
                    self._add_to_queue_fast(ns)

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

            # Streaming: emit node_start
            if self._stream_manager is not None:
                self._stream_manager.emit_node_start(state_name)

            # Execute with timeout if specified
            if state_timeout:
                result = await asyncio.wait_for(
                    self.states[state_name](context), timeout=state_timeout
                )
            else:
                result = await self.states[state_name](context)

            # Streaming: emit node_complete
            if self._stream_manager is not None:
                self._stream_manager.emit_node_complete(state_name, result)

            # Update metadata on success
            metadata.status = StateStatus.COMPLETED
            metadata.last_execution = time.time()
            metadata.last_success = time.time()
            metadata.attempts += 1

            # Track completion
            self.completed_states.add(state_name)
            self.completed_once.add(state_name)

            # Check for dependent states that can now run
            await self._check_dependent_states(state_name)

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

        # Command pattern: apply update then route via goto
        if isinstance(result, Command):
            if result.update and self._context is not None:
                self._apply_command(result, self._context)
            goto = result.goto
            if goto is None:
                return
            result = goto  # type: ignore[assignment]

        if isinstance(result, str):
            # Single next state
            if result in self.states and result not in self.completed_states:
                await self._add_to_queue(result)
        elif isinstance(result, list):
            # Check for Send instances
            sends = [r for r in result if isinstance(r, Send)]
            if sends:
                await self._execute_send_branches(sends, self._context)
                # Also process non-Send items in the list
                for next_state in result:
                    if (
                        isinstance(next_state, str)
                        and next_state in self.states
                        and next_state not in self.completed_states
                    ):
                        await self._add_to_queue(next_state)
                return

            # Multiple next states (no Send)
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
                if self.dead_letters is None:
                    self.dead_letters = []
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

    def _can_run(self, state_name: str) -> bool:
        """Check if a state can run (sync — only does set/dict lookups)."""
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
        if self.dead_letters is None:
            return []
        return self.dead_letters.copy()

    def clear_dead_letters(self) -> None:
        """Clear all dead letters."""
        count = len(self.dead_letters) if self.dead_letters else 0
        if self.dead_letters:
            self.dead_letters.clear()
        logger.info(f"Cleared {count} dead letters for agent {self.name}")

    def get_dead_letter_count(self) -> int:
        """Get count of dead letters."""
        return len(self.dead_letters) if self.dead_letters else 0

    def get_dead_letters_by_state(self, state_name: str) -> list[DeadLetter]:
        """Get dead letters for a specific state."""
        if self.dead_letters is None:
            return []
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
        if self._cleanup_handlers is None:
            self._cleanup_handlers = []
        self._cleanup_handlers.append(handler)

    async def cleanup(self) -> None:
        """Cleanup resources and handlers."""
        if not self._cleanup_handlers:
            return
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
        dl = self.dead_letters or []
        return {
            "states_completed": list(self.completed_states),
            "states_failed": [d.state_name for d in dl],
            "total_states": len(self.states),
            "session_start": self.session_start,
            "dead_letter_count": len(dl),
        }

    def _get_execution_metrics(self) -> dict[str, Any]:
        """Get execution metrics. Does NOT trigger lazy creation of
        resource_pool / circuit_breaker / bulkhead."""
        dl = self.dead_letters or []
        metrics: dict[str, Any] = {
            "completion_rate": len(self.completed_states) / max(1, len(self.states)),
            "error_rate": len(dl) / max(1, len(self.states)),
        }
        # Only include if already initialized (don't trigger lazy creation)
        if self._resource_pool is not None:
            metrics["resource_usage"] = self.get_resource_status()
        if self._circuit_breaker is not None:
            metrics["circuit_breaker_metrics"] = self._circuit_breaker.get_metrics()
        if self._bulkhead is not None:
            metrics["bulkhead_metrics"] = self._bulkhead.get_metrics()
        return metrics

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

    # --- Durable execution helpers ---

    async def wait_for_event(
        self, event_name: str, timeout: Optional[float] = None
    ) -> Any:
        """Pause execution until a named event fires.

        Used in temporal-style workflows where one agent waits for
        an external signal before proceeding.

        Args:
            event_name: Name of the event to wait for.
            timeout: Optional timeout in seconds.

        Returns:
            Data passed to ``fire_event``, or None on timeout.
        """
        if self._pending_events is None:
            self._pending_events = {}
        if self._event_results is None:
            self._event_results = {}

        event = asyncio.Event()
        self._pending_events[event_name] = event

        try:
            if timeout:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            else:
                await event.wait()
            return self._event_results.get(event_name)
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending_events.pop(event_name, None)

    def fire_event(self, event_name: str, data: Any = None) -> None:
        """Fire a named event, unblocking any ``wait_for_event`` calls.

        Args:
            event_name: Name of the event to fire.
            data: Optional data to pass to the waiter.
        """
        if self._event_results is None:
            self._event_results = {}
        self._event_results[event_name] = data

        if self._pending_events and event_name in self._pending_events:
            self._pending_events[event_name].set()

    async def resume_from(
        self,
        checkpoint_id: str,
        timeout: Optional[float] = None,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
    ) -> "AgentResult":
        """Resume agent execution from a specific checkpoint.

        Loads the checkpoint, restores agent state, and continues the run.

        Args:
            checkpoint_id: ID of the checkpoint to resume from.
            timeout: Optional timeout for the resumed run.
            execution_mode: Execution mode for the resumed run.

        Returns:
            AgentResult from the resumed execution.
        """
        checkpoint = await self.checkpoint_storage.load_checkpoint(
            self.name, checkpoint_id
        )
        if checkpoint is None:
            raise ValueError(
                f"Checkpoint {checkpoint_id} not found for agent {self.name}"
            )

        # Restore agent state from checkpoint
        self._restore_from_checkpoint(checkpoint)

        # Emit restoration event
        if self._stream_manager is not None:
            from .streaming import StreamEvent

            self._stream_manager.emit(
                StreamEvent(
                    event_type="checkpoint_restored",
                    data={
                        "checkpoint_id": checkpoint_id,
                        "agent_name": self.name,
                    },
                )
            )

        return await self.run(
            timeout=timeout,
            execution_mode=execution_mode,
            durable=True,
        )

    def _restore_from_checkpoint(self, checkpoint: AgentCheckpoint) -> None:
        """Restore agent state from a checkpoint."""
        from copy import deepcopy

        self.status = checkpoint.agent_status
        self.priority_queue = deepcopy(checkpoint.priority_queue)
        self.state_metadata = deepcopy(checkpoint.state_metadata)
        self.running_states = set(checkpoint.running_states)
        self.completed_states = set(checkpoint.completed_states)
        self.completed_once = set(checkpoint.completed_once)
        self.shared_state = deepcopy(checkpoint.shared_state)
        self.session_start = checkpoint.session_start

    # Main execution
    async def run(
        self,
        timeout: Optional[float] = None,
        initial_context: Optional[dict[str, Any]] = None,
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL,
        durable: bool = False,
        checkpoint_granularity: str = "per-state",
    ) -> "AgentResult":
        """
        Run the agent workflow with enhanced result tracking.

        Args:
            timeout: Optional timeout in seconds
            initial_context: Initial context data. Can be:
                - Simple dict: {"key": value} - sets as regular variables
                - Structured dict with type hints:
                  {
                      "variables": {"key": value},
                      "typed_variables": {"key": value},
                      "constants": {"key": value},
                      "secrets": {"key": "secret_value"},
                      "validated_data": {"key": pydantic_model_instance},
                      "cached": {"key": {"value": data, "ttl": 300}},
                      "outputs": {"key": value}
                  }
            execution_mode: Controls how entry states are determined:
                - SEQUENTIAL (default): Only the first state runs initially, flow controlled
                  by return values
                - PARALLEL: All states without dependencies run as
                  entry points
            durable: When True, enables automatic checkpointing for crash
                recovery. On entry, attempts to resume from the latest
                checkpoint. After each state completes, saves a checkpoint.
            checkpoint_granularity: Controls when checkpoints are saved during
                durable execution. Options: "per-state" (after each state),
                "on-error" (only on failure).
        """
        start_time = time.time()
        self._auto_discover_states()

        # Durable execution: try to resume from latest checkpoint
        if durable and self._checkpoint_storage is not None:
            try:
                # Check for crash marker (running marker from previous run)
                latest = await self.checkpoint_storage.load_checkpoint(self.name)
                if latest is not None and latest.agent_status == AgentStatus.RUNNING:
                    logger.info(
                        "Agent %s: detected crash — resuming from checkpoint",
                        self.name,
                    )
                    self._restore_from_checkpoint(latest)
                    if self._stream_manager is not None:
                        from .streaming import StreamEvent

                        self._stream_manager.emit(
                            StreamEvent(
                                event_type="checkpoint_restored",
                                data={
                                    "agent_name": self.name,
                                    "resumed_from_crash": True,
                                },
                            )
                        )
            except Exception as exc:
                logger.warning(
                    "Agent %s: failed to load checkpoint for resume: %s",
                    self.name,
                    exc,
                )

        self.status = AgentStatus.RUNNING

        if self.session_start is None:
            self.session_start = start_time

        # Durable: save a "running" marker checkpoint
        if durable and self._checkpoint_storage is not None:
            try:
                marker = AgentCheckpoint.create_from_agent(self)
                await self.checkpoint_storage.save_checkpoint(self.name, marker)
            except Exception as exc:
                logger.warning("Failed to save running marker: %s", exc)

        try:
            # Determine if we can use the AgentCore fast path
            _core = self._core
            _extras = self._state_extras
            _fast = (
                _core is not None
                and self._circuit_breaker is None
                and self._resource_pool is None
                and "_find_entry_states_by_mode" not in self.__dict__
                and not (_extras and any("resources" in v for v in _extras.values()))
            )
            if _fast:
                # ============================================================
                # AgentCore fast path — persistent core, validation cached,
                # sync fast path for non-suspending coroutines
                # ============================================================
                assert _core is not None
                # Validate only if deps were externally mutated after add_state
                _snap = self._deps_snapshot
                if _snap is not None and _snap != sum(
                    id(v) for v in self._dependencies_dict.values()
                ):
                    self._validate_workflow_configuration(execution_mode)

                mode_str = execution_mode.value
                entry_states = _core.prepare_run(mode_str)

                # Apply entry_point overrides — filter out states
                # explicitly excluded via entry_point=False
                _ep_overrides = getattr(self, "_entry_point_overrides", None)
                if _ep_overrides:
                    entry_states = [
                        s for s in entry_states
                        if _ep_overrides.get(s) is not False
                    ]

                # Lazy context
                self._create_context(self.shared_state)
                if initial_context:
                    self._apply_initial_context(initial_context)

                for sn in entry_states:
                    _core.add_to_queue(sn, 0)

                _ctx = self._context
                _states = self.states
                _timeout: Optional[float] = timeout

                # Pre-compute durable/drain guards as locals — zero cost when unused
                _drain = self._drain_protocol
                _ckpt = self._checkpoint_storage
                _durable_pstate = durable and checkpoint_granularity == "per-state" and _ckpt is not None
                _durable_onerr = durable and checkpoint_granularity == "on-error" and _ckpt is not None
                _durable_final = durable and _ckpt is not None
                _has_extras = _durable_pstate or _durable_onerr or _drain is not None

                while self.status == AgentStatus.RUNNING:
                    if _timeout is not None and (time.time() - start_time) > _timeout:
                        logger.warning(
                            f"Agent {self.name} timed out after {timeout} seconds."
                        )
                        self.status = AgentStatus.FAILED
                        break

                    # Use batched get_next_ready_state (single state, no list alloc)
                    sn = _core.get_next_ready_state()

                    if sn is not None:
                        try:
                            # Streaming: emit node_start
                            if self._stream_manager is not None:
                                self._stream_manager.emit_node_start(sn)

                            result = await _states[sn](_ctx)

                            # Command pattern: apply updates before passing to core
                            _goto = result
                            if isinstance(result, Command):
                                if result.update:
                                    self._apply_command(result, _ctx)
                                _goto = result.goto
                            elif isinstance(result, list) and any(
                                isinstance(r, Send) for r in result
                            ):
                                sends = [r for r in result if isinstance(r, Send)]
                                await self._execute_send_branches(sends, _ctx)
                                _goto = [
                                    r for r in result if isinstance(r, str)
                                ] or None

                            # Streaming: emit node_complete
                            if self._stream_manager is not None:
                                self._stream_manager.emit_node_complete(sn, _goto)

                            # Batched: mark_completed + handle_result + is_done
                            if _core.execute_step(sn, _goto):
                                if _durable_final:
                                    try:
                                        cp = AgentCheckpoint.create_from_agent(self)
                                        await self.checkpoint_storage.save_checkpoint(
                                            self.name, cp
                                        )
                                    except Exception:
                                        pass
                                if _timeout is not None:
                                    continue
                                break

                            if _has_extras:
                                if _durable_pstate:
                                    try:
                                        cp = AgentCheckpoint.create_from_agent(self)
                                        await self.checkpoint_storage.save_checkpoint(
                                            self.name, cp
                                        )
                                    except Exception:
                                        pass
                                if _drain is not None and _drain.is_draining:
                                    await self.save_checkpoint()
                                    self.status = AgentStatus.PAUSED
                                    break
                        except Exception as e:
                            _core.mark_failed(sn)
                            if _durable_onerr:
                                try:
                                    cp = AgentCheckpoint.create_from_agent(self)
                                    await self.checkpoint_storage.save_checkpoint(
                                        self.name, cp
                                    )
                                except Exception:
                                    pass
                            # Retry handling via core
                            if _core.should_retry(sn):
                                delay = _core.get_retry_delay(
                                    sn,
                                    _core.get_state_attempts(sn),
                                )
                                if delay > 0:
                                    await asyncio.sleep(delay)
                                _core.add_to_queue(sn, 0)
                            else:
                                # Max retries exceeded
                                if self.enable_dead_letter:
                                    dl = DeadLetter(
                                        state_name=sn,
                                        agent_name=self.name,
                                        error_message=str(e),
                                        error_type=type(e).__name__,
                                        attempts=_core.get_state_attempts(sn),
                                        failed_at=time.time(),
                                        timeout_occurred=isinstance(
                                            e, asyncio.TimeoutError
                                        ),
                                        context_snapshot=dict(self.shared_state),
                                    )
                                    if self.dead_letters is None:
                                        self.dead_letters = []
                                    self.dead_letters.append(dl)
                                if _core.is_done():
                                    break
                    elif _core.has_queued():
                        # States in queue but none can run — check for parallel ready
                        ready = _core.get_ready_states()
                        if ready:

                            async def _run_with_agent_core(_sn: str, _cr, _ct) -> None:
                                _cr.mark_running(_sn)
                                try:
                                    res = await _states[_sn](_ct)
                                    _cr.execute_step(_sn, res)
                                except Exception:
                                    _cr.mark_failed(_sn)
                                    if _cr.should_retry(_sn):
                                        _cr.add_to_queue(_sn, 0)

                            tasks = [
                                asyncio.create_task(
                                    _run_with_agent_core(sn, _core, _ctx)
                                )
                                for sn in ready[: self.max_concurrent]
                            ]
                            await asyncio.gather(*tasks, return_exceptions=True)
                        else:
                            logger.warning(
                                f"Deadlock in agent {self.name}: States in queue "
                                f"but none can run."
                            )
                            self.status = AgentStatus.FAILED
                            break
                    elif _core.is_done():
                        break
                    else:
                        await asyncio.sleep(0.01)

                self._use_fast_path = True

            else:
                # ============================================================
                # Original execution path (no AgentCore, or has extras)
                # ============================================================

                # Ensure Python-side tracking sets are initialized
                # (they may be None when _core is available but fast path is disabled)
                if self._completed_states_set is None:
                    self._completed_states_set = set()
                if self._completed_once_set is None:
                    self._completed_once_set = set()
                if self._running_states_set is None:
                    self._running_states_set = set()
                if self._priority_queue is None:
                    self._priority_queue = []

                # Validate workflow configuration — only once per agent lifetime
                if not self._validated:
                    self._validate_workflow_configuration(execution_mode)
                    self._validated = True

                # Detect if we can use the (non-core) fast path
                self._use_fast_path = (
                    self._circuit_breaker is None
                    and self._resource_pool is None
                    and not any(
                        self.state_metadata[s].resources is not None
                        for s in self.states
                    )
                )

                # Create context
                self._create_context(self.shared_state)
                if initial_context:
                    self._apply_initial_context(initial_context)

                entry_states = self._find_entry_states_by_mode(execution_mode)

                # Try to use StateMachineCore on the fast path
                _use_sm_core = (
                    _HAS_CORE
                    and self._use_fast_path
                    and not hasattr(type(self), "_execute_state_fast_override")
                )

                if _use_sm_core:
                    _state_configs = []
                    for sn in self.states:
                        meta = self.state_metadata[sn]
                        deps = self.dependencies.get(sn, [])
                        _state_configs.append(
                            (sn, meta.priority.value, meta.max_retries, deps)
                        )
                    _sm_core = StateMachineCore(_state_configs)

                    for sn in entry_states:
                        _sm_core.add_to_queue(sn, 0)

                    while self.status == AgentStatus.RUNNING:
                        if timeout and (time.time() - start_time) > timeout:
                            logger.warning(
                                f"Agent {self.name} timed out after "
                                f"{timeout} seconds."
                            )
                            self.status = AgentStatus.FAILED
                            break

                        if _sm_core.is_done():
                            break

                        ready = _sm_core.get_ready_states()

                        if ready:
                            if len(ready) == 1:
                                sn = ready[0]
                                _sm_core.mark_running(sn)
                                try:
                                    result = await self.states[sn](self._context)
                                    _sm_core.mark_completed(sn)
                                    _sm_core.handle_result(sn, result)
                                except Exception as e:
                                    _sm_core.mark_failed(sn)
                                    await self._handle_state_failure(sn, e, 0.0)
                            else:

                                async def _run_with_core(_sn: str, _core_ref) -> None:
                                    _core_ref.mark_running(_sn)
                                    try:
                                        res = await self.states[_sn](self._context)
                                        _core_ref.mark_completed(_sn)
                                        _core_ref.handle_result(_sn, res)
                                    except Exception as exc:
                                        _core_ref.mark_failed(_sn)
                                        await self._handle_state_failure(_sn, exc, 0.0)

                                tasks = [
                                    asyncio.create_task(_run_with_core(sn, _sm_core))
                                    for sn in ready[: self.max_concurrent]
                                ]
                                await asyncio.gather(*tasks, return_exceptions=True)
                        elif _sm_core.has_queued():
                            logger.warning(
                                f"Deadlock in agent {self.name}: States in "
                                f"queue but none can run."
                            )
                            self.status = AgentStatus.FAILED
                            break
                        else:
                            await asyncio.sleep(0.01)

                    # Sync core state back
                    self.completed_states = set(_sm_core.get_completed_states())
                    self.completed_once = set(_sm_core.get_completed_once())
                    for sn in self.states:
                        status_str = _sm_core.get_state_status(sn)
                        with contextlib.suppress(ValueError, AttributeError):
                            self.state_metadata[sn].status = StateStatus(status_str)

                else:
                    # Pure Python path
                    if self._use_fast_path:
                        for state_name in entry_states:
                            self._add_to_queue_fast(state_name)
                    else:
                        for state_name in entry_states:
                            await self._add_to_queue(state_name)

                    _s_drain = self._drain_protocol
                    _s_ckpt = self._checkpoint_storage
                    _s_durable_ps = durable and checkpoint_granularity == "per-state" and _s_ckpt is not None
                    _s_has_extras = _s_durable_ps or _s_drain is not None

                    while self.status == AgentStatus.RUNNING:
                        if timeout and (time.time() - start_time) > timeout:
                            logger.warning(
                                f"Agent {self.name} timed out after "
                                f"{timeout} seconds."
                            )
                            self.status = AgentStatus.FAILED
                            break

                        if _s_has_extras and _s_drain is not None and _s_drain.is_draining:
                            if durable and _s_ckpt is not None:
                                try:
                                    cp = AgentCheckpoint.create_from_agent(self)
                                    await self.checkpoint_storage.save_checkpoint(
                                        self.name, cp
                                    )
                                except Exception:
                                    pass
                            self.status = AgentStatus.PAUSED
                            break

                        if not self.priority_queue and not self.running_states:
                            break

                        ready_states = self._get_ready_states()

                        if ready_states:
                            if len(ready_states) == 1:
                                if self._use_fast_path:
                                    await self._execute_state_fast(ready_states[0])
                                else:
                                    await self.run_state(ready_states[0])
                            else:
                                tasks = [
                                    asyncio.create_task(
                                        self._execute_state_fast(s)
                                        if self._use_fast_path
                                        else self.run_state(s)
                                    )
                                    for s in ready_states[: self.max_concurrent]
                                ]
                                await asyncio.gather(*tasks, return_exceptions=True)

                            if _s_durable_ps:
                                try:
                                    cp = AgentCheckpoint.create_from_agent(self)
                                    await self.checkpoint_storage.save_checkpoint(
                                        self.name, cp
                                    )
                                except Exception:
                                    pass
                        elif self.priority_queue and not self.running_states:
                            logger.warning(
                                f"Deadlock in agent {self.name}: States in "
                                f"queue but none can run."
                            )
                            self.status = AgentStatus.FAILED
                            break
                        else:
                            await asyncio.sleep(0.01)

            # Determine final status
            if self.status == AgentStatus.RUNNING:
                if not self.states:
                    self.status = AgentStatus.IDLE
                else:
                    has_failed_states = any(
                        s.status == StateStatus.FAILED
                        for s in self.state_metadata.values()
                    )
                    if has_failed_states:
                        self.status = AgentStatus.FAILED
                    else:
                        self.status = AgentStatus.COMPLETED

            # Durable: save final checkpoint with completed/failed status
            if durable and self._checkpoint_storage is not None:
                try:
                    final_cp = AgentCheckpoint.create_from_agent(self)
                    await self.checkpoint_storage.save_checkpoint(self.name, final_cp)
                except Exception:
                    pass

            end_time = time.time()

            # Create result — lightweight on fast path
            if self._use_fast_path:
                _outputs = (
                    self._context._outputs.copy()
                    if self._context and self._context._outputs
                    else {}
                )
                result = AgentResult(
                    agent_name=self.name,
                    status=self.status,
                    outputs=_outputs,
                    variables=self._shared_state or {},
                    start_time=start_time,
                    end_time=end_time,
                    execution_duration=end_time - start_time,
                    _final_context=self._context,
                )
            else:
                agent_vars = self._agent_variables or {}
                result = AgentResult(
                    agent_name=self.name,
                    status=self.status,
                    outputs=self.get_all_outputs(),
                    variables={**agent_vars, **self.shared_state},
                    metadata=self._get_execution_metadata(),
                    metrics=self._get_execution_metrics(),
                    start_time=start_time,
                    end_time=end_time,
                    execution_duration=end_time - start_time,
                    _final_context=self._context,
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
        if self._cleanup_handlers is not None and self._cleanup_handlers:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create cleanup task but don't store reference as object is
                    # being destroyed
                    task = loop.create_task(self.cleanup())
                    task.add_done_callback(lambda t: None)  # Prevent warnings
            except Exception:
                pass
