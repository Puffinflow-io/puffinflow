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
    # Main decorator
    "state",
    "FlexibleStateDecorator",

    # Profile-based decorators
    "minimal_state",
    "cpu_intensive",
    "memory_intensive",
    "io_intensive",
    "gpu_accelerated",
    "network_intensive",
    "quick_state",
    "batch_state",
    "critical_state",
    "concurrent_state",
    "synchronized_state",

    # Profile management
    "get_profile",
    "list_profiles",
    "create_custom_decorator",
    "StateProfile",
    "PROFILES",

    # Builder pattern
    "StateBuilder",
    "build_state",
    "cpu_state",
    "memory_state",
    "gpu_state",
    "exclusive_state",
    "builder_concurrent_state",
    "builder_high_priority_state",
    "builder_critical_state",

    # Inspection utilities
    "is_puffinflow_state",
    "get_state_config",
    "get_state_requirements",
    "get_state_rate_limit",
    "get_state_coordination",
    "list_state_metadata",
    "compare_states",
    "get_state_summary"
]
