"""Workflow Orchestrator Core Engine."""

__version__ = "0.1.0"

from .agent.base import Agent
from .agent.context import Context
from .agent.state import Priority, StateStatus
from .resources.pool import ResourcePool
from .resources.requirements import (
    ResourceRequirements,
    ResourceType,
)

# Import reliability components
try:
    from .reliability.bulkhead import Bulkhead, BulkheadConfig
    from .reliability.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
    )
    from .reliability.leak_detector import ResourceLeakDetector
except ImportError:
    # Create mock classes if reliability module is not available
    from typing import Any

    class MockCircuitBreaker:
        def __init__(self, config: Any) -> None:
            pass

    class MockCircuitBreakerConfig:
        def __init__(self, **kwargs: Any) -> None:
            pass

    class MockBulkhead:
        def __init__(self, config: Any) -> None:
            pass

    class MockBulkheadConfig:
        def __init__(self, **kwargs: Any) -> None:
            pass

    class MockResourceLeakDetector:
        def __init__(self, **kwargs: Any) -> None:
            pass

    # Assign to actual names to avoid redefinition
    CircuitBreaker = MockCircuitBreaker  # type: ignore[assignment,misc]
    CircuitBreakerConfig = MockCircuitBreakerConfig  # type: ignore[assignment,misc]
    Bulkhead = MockBulkhead  # type: ignore[assignment,misc]
    BulkheadConfig = MockBulkheadConfig  # type: ignore[assignment,misc]
    ResourceLeakDetector = MockResourceLeakDetector  # type: ignore[assignment,misc]


# Import state decorator if available
try:
    from .agent.decorators.flexible import state
except ImportError:
    # Create a simple mock state decorator
    from typing import Any, Callable

    def state(**kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:  # type: ignore[misc]
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            # Create ResourceRequirements from kwargs
            requirements = ResourceRequirements()
            if "cpu" in kwargs:
                requirements.cpu_units = kwargs["cpu"]
            if "memory" in kwargs:
                requirements.memory_mb = kwargs["memory"]
            if "io" in kwargs:
                requirements.io_weight = kwargs["io"]
            if "network" in kwargs:
                requirements.network_weight = kwargs["network"]
            if "gpu" in kwargs:
                requirements.gpu_units = kwargs["gpu"]

            # Determine resource_types from non-zero values
            resource_types = ResourceType.NONE
            if requirements.cpu_units > 0:
                resource_types |= ResourceType.CPU
            if requirements.memory_mb > 0:
                resource_types |= ResourceType.MEMORY
            if requirements.io_weight > 0:
                resource_types |= ResourceType.IO
            if requirements.network_weight > 0:
                resource_types |= ResourceType.NETWORK
            if requirements.gpu_units > 0:
                resource_types |= ResourceType.GPU

            requirements.resource_types = resource_types
            func._resource_requirements = requirements  # type: ignore
            return func

        return decorator


__all__ = [
    "Agent",
    "Bulkhead",
    "BulkheadConfig",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "Context",
    "Priority",
    "ResourceLeakDetector",
    "ResourcePool",
    "ResourceRequirements",
    "ResourceType",
    "StateStatus",
    "state",
]
