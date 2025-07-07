"""State management types and enums."""

import asyncio
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Any, Optional, Union, runtime_checkable

from typing_extensions import Protocol

if TYPE_CHECKING:
    from ..resources.requirements import ResourceRequirements
    from .context import Context

try:
    from ..resources.requirements import ResourceRequirements
except ImportError:
    ResourceRequirements = None  # type: ignore

# Type definitions
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Agent

StateResult = Union[str, list[Union[str, tuple["Agent", str]]], None]


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

    def __str__(self) -> str:
        return self.value


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

    def __str__(self) -> str:
        return self.value


@runtime_checkable
class StateFunction(Protocol):
    """Protocol for state functions."""

    async def __call__(self, context: "Context") -> StateResult:
        ...


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
            self.initial_delay * (self.exponential_base**attempt),
            60.0,  # Max 60 seconds
        )
        if self.jitter:
            delay *= 0.5 + random.random() * 0.5
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
    context_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateMetadata:
    """Metadata for state execution."""

    status: StateStatus
    attempts: int = 0
    max_retries: int = 3
    resources: Optional["ResourceRequirements"] = None
    dependencies: dict[str, Any] = field(default_factory=dict)
    satisfied_dependencies: set = field(default_factory=set)
    last_execution: Optional[float] = None
    last_success: Optional[float] = None
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    retry_policy: Optional[RetryPolicy] = None
    priority: Priority = Priority.NORMAL
    coordination_primitives: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
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
