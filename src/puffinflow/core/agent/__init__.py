"""Agent module for workflow orchestrator with flexible decorators."""

# Core agent functionality
from src.puffinflow.core.agent.base import Agent, RetryPolicy
from src.puffinflow.core.agent.context import Context, TypedContextData, StateType
from src.puffinflow.core.agent.state import (
    Priority,
    AgentStatus,
    StateStatus,
    StateResult,
    StateFunction,
    StateMetadata,
    PrioritizedState
)
from src.puffinflow.core.agent.dependencies import (
    DependencyType,
    DependencyLifecycle,
    DependencyConfig
)
from src.puffinflow.core.agent.checkpoint import AgentCheckpoint

# Import decorator functionality
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
    # Core classes
    "Agent",
    "Context",
    "RetryPolicy",

    # State types
    "Priority",
    "AgentStatus",
    "StateStatus",
    "StateResult",
    "StateFunction",
    "StateMetadata",
    "PrioritizedState",

    # Context types
    "TypedContextData",
    "StateType",

    # Dependencies
    "DependencyType",
    "DependencyLifecycle",
    "DependencyConfig",

    # Checkpoint
    "AgentCheckpoint",

    # Main Flexible Decorator
    "state",
    "FlexibleStateDecorator",

    # Profile-based Decorators
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

    # Profile Management
    "StateProfile",
    "PROFILES",
    "get_profile",
    "list_profiles",
    "create_custom_decorator",

    # Builder Pattern
    "StateBuilder",
    "build_state",
    "cpu_state",
    "memory_state",
    "gpu_state",
    "exclusive_state",
    "builder_concurrent_state",
    "builder_high_priority_state",
    "builder_critical_state",

    # Inspection and Utilities
    "is_puffinflow_state",
    "get_state_config",
    "get_state_requirements",
    "get_state_rate_limit",
    "get_state_coordination",
    "list_state_metadata",
    "compare_states",
    "get_state_summary",
]


# Helper functions for common use cases
def create_team_decorator(team_name: str, **defaults):
    """Create a team-specific decorator with common defaults."""
    team_defaults = {
        'tags': {'team': team_name},
        **defaults
    }
    return create_custom_decorator(**team_defaults)


def create_environment_decorator(env: str, **defaults):
    """Create an environment-specific decorator."""
    env_defaults = {
        'tags': {'environment': env},
        **defaults
    }
    return create_custom_decorator(**env_defaults)


def create_service_decorator(service_name: str, **defaults):
    """Create a service-specific decorator."""
    service_defaults = {
        'tags': {'service': service_name},
        **defaults
    }
    return create_custom_decorator(**service_defaults)


# Add convenience functions to __all__
__all__.extend([
    "create_team_decorator",
    "create_environment_decorator",
    "create_service_decorator",
])


# Example usage documentation
"""
Usage Examples:

1. Basic Usage:
   @state
   async def simple_state(context):
       pass

2. Profile-based:
   @cpu_intensive
   async def heavy_computation(context):
       pass
   
   @state('gpu_accelerated')
   async def ml_training(context):
       pass

3. Flexible Parameters:
   @state(cpu=2.0, memory=512.0, mutex=True, priority='high')
   async def configured_state(context):
       pass

4. Builder Pattern:
   @build_state().cpu(4.0).memory(2048.0).mutex().high_priority()
   async def builder_state(context):
       pass

5. Custom Team Decorator:
   ml_team = create_team_decorator('ml', cpu=2.0, memory=1024.0)
   
   @ml_team(gpu=1.0)
   async def ml_state(context):
       pass

6. Environment-specific:
   prod = create_environment_decorator('production', max_retries=5)
   
   @prod('critical')
   async def prod_critical_state(context):
       pass

7. Inspection:
   print(get_state_summary(my_state))
   print(list_state_metadata(my_state))

8. Profile Management:
   # List available profiles
   print(list_profiles())
   
   # Get specific profile
   profile = get_profile('cpu_intensive')
   
   # Create custom profile
   state.register_profile('my_custom', cpu=8.0, memory=4096.0, priority='high')

9. Comparison:
   differences = compare_states(state1, state2)
   print(differences)

Available Profiles:
- minimal: Lightweight operations (cpu=0.1, memory=50MB)
- standard: Default configuration (cpu=1.0, memory=100MB)
- cpu_intensive: CPU-heavy work (cpu=4.0, memory=1024MB)
- memory_intensive: Memory-heavy work (cpu=2.0, memory=4096MB)
- io_intensive: I/O operations (cpu=1.0, memory=256MB, io=10.0)
- gpu_accelerated: GPU workloads (cpu=2.0, memory=2048MB, gpu=1.0)
- network_intensive: Network operations (cpu=1.0, memory=512MB, network=10.0)
- quick: Fast operations (cpu=0.5, memory=50MB, timeout=30s, rate_limit=100/s)
- batch: Batch processing (cpu=2.0, memory=1024MB, low priority, timeout=30min)
- critical: Critical operations (cpu=2.0, memory=512MB, critical priority, mutex)
- concurrent: Limited parallelism (cpu=1.0, memory=256MB, semaphore:5)
- synchronized: Barrier sync (cpu=1.0, memory=200MB, barrier:3)

For detailed documentation, see the decorators module.
"""