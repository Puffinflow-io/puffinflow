"""Enhanced state decorators with flexible configuration."""

# Import flexible decorator as the main state decorator
from src.puffinflow.core.agent.decorators.flexible import (
    state,
    FlexibleStateDecorator,
    minimal_state,
    cpu_intensive,
    memory_intensive,
    io_intensive,
    gpu_accelerated,
    network_intensive,
    quick_state,
    batch_state,
    critical_state,
    concurrent_state,
    synchronized_state,
    get_profile,
    list_profiles,
    create_custom_decorator,
    StateProfile,
    PROFILES
)

# Import builder pattern
from src.puffinflow.core.agent.decorators.builder import (
    StateBuilder,
    build_state,
    cpu_state,
    memory_state,
    gpu_state,
    exclusive_state,
    concurrent_state as builder_concurrent_state,
    high_priority_state as builder_high_priority_state,
    critical_state as builder_critical_state
)

# Import inspection utilities
from src.puffinflow.core.agent.decorators.inspection import (
    is_puffinflow_state,
    get_state_config,
    get_state_requirements,
    get_state_rate_limit,
    get_state_coordination,
    list_state_metadata,
    compare_states,
    get_state_summary
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