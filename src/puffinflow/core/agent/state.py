"""State management types and enums."""
import uuid
import time
import random
import asyncio
from enum import IntEnum, Enum
from dataclasses import dataclass, field
from typing import Union, List, Tuple, Optional, Dict, Any, Callable, TYPE_CHECKING
from typing_extensions import runtime_checkable, Protocol

if TYPE_CHECKING:
    from .context import Context
    from ..resources.requirements import ResourceRequirements

try:
    from ..resources.requirements import ResourceRequirements
except ImportError:
    ResourceRequirements = None

# Type definitions
StateResult = Union[str, List[Union[str, Tuple["Agent", str]]], None]

class Priority(IntEnum):
    """Priority levels for state execution."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

class AgentStatus(str, Enum):
    """Agent execution status."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StateStatus(str, Enum):
    """State execution status."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"
    RETRYING = "retrying"

@runtime_checkable
class StateFunction(Protocol):
    """Protocol for state functions."""
    async def __call__(self, context: "Context") -> StateResult: ...

@dataclass
class RetryPolicy:
    max_retries: int = 3
    initial_delay: float = 1.0
    exponential_base: float = 2.0
    jitter: bool = True
    # Dead letter handling
    dead_letter_on_max_retries: bool = True
    dead_letter_on_timeout: bool = True

    async def wait(self, attempt: int) -> None:
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            60.0  # Max 60 seconds
        )
        if self.jitter:
            delay *= (0.5 + random.random() * 0.5)
        await asyncio.sleep(delay)

# Dead letter data structure
@dataclass
class DeadLetter:
    state_name: str
    agent_name: str
    error_message: str
    error_type: str
    attempts: int
    failed_at: float
    timeout_occurred: bool = False
    context_snapshot: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StateMetadata:
    """Metadata for state execution."""
    status: StateStatus
    attempts: int = 0
    max_retries: int = 3
    resources: Optional["ResourceRequirements"] = None
    dependencies: Dict[str, Any] = field(default_factory=dict)
    satisfied_dependencies: set = field(default_factory=set)
    last_execution: Optional[float] = None
    last_success: Optional[float] = None
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    retry_policy: Optional[RetryPolicy] = None
    priority: Priority = Priority.NORMAL

    def __post_init__(self):
        """Initialize resources if not provided."""
        if self.resources is None and ResourceRequirements is not None:
            self.resources = ResourceRequirements()

@dataclass(order=True)
class PrioritizedState:
    """State with priority for queue management."""
    priority: int
    timestamp: float
    state_name: str = field(compare=False)
    metadata: StateMetadata = field(compare=False)