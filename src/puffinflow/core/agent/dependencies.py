"""Dependency management types."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .base import Agent


class DependencyType(Enum):
    """Types of dependencies between states."""

    REQUIRED = "required"  # Must complete before state can run
    OPTIONAL = "optional"  # Will wait if running, otherwise skips
    PARALLEL = "parallel"  # Can run in parallel with dependency
    SEQUENTIAL = "sequential"  # Must run after dependency completes
    CONDITIONAL = "conditional"  # Depends on condition function
    TIMEOUT = "timeout"  # Wait for max time then continue
    XOR = "xor"  # Only one dependency needs to be satisfied
    AND = "and"  # All dependencies must be satisfied
    OR = "or"  # At least one dependency must be satisfied

    def __str__(self) -> str:
        return f"DependencyType.{self.name}"

    def __repr__(self) -> str:
        return f"DependencyType.{self.name}"


class DependencyLifecycle(Enum):
    """Lifecycle management for dependencies."""

    ONCE = "once"  # Dependency only needs to be satisfied once
    ALWAYS = "always"  # Dependency must be satisfied every time
    SESSION = "session"  # Dependency valid for current run() execution
    TEMPORARY = "temporary"  # Dependency expires after specified time
    PERIODIC = "periodic"  # Must be re-satisfied after specified interval

    def __str__(self) -> str:
        return f"DependencyLifecycle.{self.name}"

    def __repr__(self) -> str:
        return f"DependencyLifecycle.{self.name}"


@dataclass
class DependencyConfig:
    """Configuration for state dependencies."""

    type: DependencyType
    lifecycle: DependencyLifecycle = DependencyLifecycle.ALWAYS
    condition: Optional[Callable[["Agent"], bool]] = None
    expiry: Optional[float] = None
    interval: Optional[float] = None
    timeout: Optional[float] = None
    retry_policy: Optional[dict[str, Any]] = None
