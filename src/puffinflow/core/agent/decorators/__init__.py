"""Enhanced state decorators with flexible configuration."""

# Import flexible decorator as the main state decorator
# Import builder pattern
from .builder import (
    StateBuilder,
    build_state,
    cpu_state,
    exclusive_state,
    gpu_state,
    memory_state,
)
from .builder import concurrent_state as builder_concurrent_state
from .builder import critical_state as builder_critical_state
from .builder import high_priority_state as builder_high_priority_state
from .flexible import (
    PROFILES,
    FlexibleStateDecorator,
    StateProfile,
    batch_state,
    concurrent_state,
    cpu_intensive,
    create_custom_decorator,
    critical_state,
    get_profile,
    gpu_accelerated,
    io_intensive,
    list_profiles,
    memory_intensive,
    minimal_state,
    network_intensive,
    quick_state,
    state,
    synchronized_state,
)

# Import inspection utilities
from .inspection import (
    compare_states,
    get_state_config,
    get_state_coordination,
    get_state_rate_limit,
    get_state_requirements,
    get_state_summary,
    is_puffinflow_state,
    list_state_metadata,
)

__all__ = [
    "PROFILES",
    "FlexibleStateDecorator",
    # Builder pattern
    "StateBuilder",
    "StateProfile",
    "batch_state",
    "build_state",
    "builder_concurrent_state",
    "builder_critical_state",
    "builder_high_priority_state",
    "compare_states",
    "concurrent_state",
    "cpu_intensive",
    "cpu_state",
    "create_custom_decorator",
    "critical_state",
    "exclusive_state",
    # Profile management
    "get_profile",
    "get_state_config",
    "get_state_coordination",
    "get_state_rate_limit",
    "get_state_requirements",
    "get_state_summary",
    "gpu_accelerated",
    "gpu_state",
    "io_intensive",
    # Inspection utilities
    "is_puffinflow_state",
    "list_profiles",
    "list_state_metadata",
    "memory_intensive",
    "memory_state",
    # Profile-based decorators
    "minimal_state",
    "network_intensive",
    "quick_state",
    # Main decorator
    "state",
    "synchronized_state",
]
