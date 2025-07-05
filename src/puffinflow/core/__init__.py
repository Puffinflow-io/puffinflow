"""Workflow Orchestrator Core Engine."""

__version__ = "0.1.0"

from src.puffinflow.core.agent.base import Agent
from src.puffinflow.core.agent.context import Context
from src.puffinflow.core.agent.state import Priority, StateStatus
from src.puffinflow.core.resources.pool import ResourcePool
from src.puffinflow.core.resources.requirements import (
    ResourceRequirements,
    ResourceType,
)

# Import reliability components
try:
    from src.puffinflow.core.reliability.bulkhead import Bulkhead, BulkheadConfig
    from src.puffinflow.core.reliability.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
    )
    from src.puffinflow.core.reliability.leak_detector import ResourceLeakDetector
except ImportError:
    # Create mock classes if reliability module is not available
    class CircuitBreaker:
        def __init__(self, config): pass
    class CircuitBreakerConfig:
        def __init__(self, **kwargs): pass
    class Bulkhead:
        def __init__(self, config): pass
    class BulkheadConfig:
        def __init__(self, **kwargs): pass
    class ResourceLeakDetector:
        def __init__(self, **kwargs): pass

# Import state decorator if available
try:
    from src.puffinflow.core.agent.decorators.flexible import state
except ImportError:
    # Create a simple mock state decorator
    def state(**kwargs):
        def decorator(func):
            # Create ResourceRequirements from kwargs
            requirements = ResourceRequirements()
            if 'cpu' in kwargs:
                requirements.cpu_units = kwargs['cpu']
            if 'memory' in kwargs:
                requirements.memory_mb = kwargs['memory']
            if 'io' in kwargs:
                requirements.io_weight = kwargs['io']
            if 'network' in kwargs:
                requirements.network_weight = kwargs['network']
            if 'gpu' in kwargs:
                requirements.gpu_units = kwargs['gpu']

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
            func._resource_requirements = requirements
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
