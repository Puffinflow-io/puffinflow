"""Agent module with enhanced coordination features."""

from typing import Any, Callable

from .base import Agent, AgentResult, ResourceTimeoutError
from .checkpoint import AgentCheckpoint
from .context import Context, StateType
from .dependencies import DependencyConfig, DependencyLifecycle, DependencyType
from .state import (
    AgentStatus,
    DeadLetter,
    PrioritizedState,
    Priority,
    RetryPolicy,
    StateMetadata,
    StateResult,
    StateStatus,
)

# Decorators
try:
    from .decorators.builder import (
        StateBuilder,
        build_state,
        cpu_state,
        exclusive_state,
        external_service_state,
        fault_tolerant_state,
        gpu_state,
        high_priority_state,
        isolated_state,
        memory_state,
        production_state,
        protected_state,
    )
    from .decorators.flexible import (
        FlexibleStateDecorator,
        StateProfile,
        batch_state,
        concurrent_state,
        cpu_intensive,
        create_custom_decorator,
        critical_state,
        external_service,
        fault_tolerant,
        get_profile,
        gpu_accelerated,
        high_availability,
        io_intensive,
        list_profiles,
        memory_intensive,
        minimal_state,
        network_intensive,
        quick_state,
        state,
    )

    # synchronized_state,  # Temporarily disabled
    from .decorators.inspection import (
        compare_states,
        get_state_config,
        get_state_coordination,
        get_state_rate_limit,
        get_state_requirements,
        get_state_summary,
        is_puffinflow_state,
        list_state_metadata,
    )
except ImportError:
    # Decorators not available
    pass

# Scheduling components
try:
    from .scheduling import (
        GlobalScheduler,
        InputType,
        InvalidInputTypeError,
        InvalidScheduleError,
        ScheduleBuilder,
        ScheduledAgent,
        ScheduledInput,
        ScheduleParser,
        SchedulingError,
        parse_magic_prefix,
        parse_schedule_string,
    )

    _SCHEDULING_AVAILABLE = True
except ImportError:
    # Scheduling not available
    _SCHEDULING_AVAILABLE = False


# Team decorators for convenience
def create_team_decorator(
    team_name: str, **defaults: Any
) -> Callable[[Callable], Callable]:
    """Create a decorator for team-specific agents."""
    try:
        from .decorators.flexible import create_custom_decorator

        team_defaults = {
            "tags": {"team": team_name},
            "description": f"Agent for {team_name} team",
            **defaults,
        }
        return create_custom_decorator(**team_defaults)
    except ImportError:
        return lambda func: func


def create_environment_decorator(
    env: str, **defaults: Any
) -> Callable[[Callable], Callable]:
    """Create a decorator for environment-specific agents."""
    try:
        from .decorators.flexible import create_custom_decorator

        env_defaults = {
            "tags": {"environment": env},
            "description": f"Agent for {env} environment",
            **defaults,
        }
        return create_custom_decorator(**env_defaults)
    except ImportError:
        return lambda func: func


def create_service_decorator(
    service_name: str, **defaults: Any
) -> Callable[[Callable], Callable]:
    """Create a decorator for service-specific agents."""
    try:
        from .decorators.flexible import create_custom_decorator

        service_defaults = {
            "tags": {"service": service_name},
            "description": f"Agent for {service_name} service",
            **defaults,
        }
        return create_custom_decorator(**service_defaults)
    except ImportError:
        return lambda func: func


def create_reliable_team_decorator(
    team_name: str, **defaults: Any
) -> Callable[[Callable], Callable]:
    """Create a decorator for reliable team agents."""
    try:
        from .decorators.flexible import create_custom_decorator

        reliable_defaults = {
            "tags": {"team": team_name, "reliability": "high"},
            "circuit_breaker": True,
            "bulkhead": True,
            "retries": 5,
            **defaults,
        }
        return create_custom_decorator(**reliable_defaults)
    except ImportError:
        return lambda func: func


def create_external_team_decorator(
    team_name: str, **defaults: Any
) -> Callable[[Callable], Callable]:
    """Create a decorator for external service team agents."""
    try:
        from .decorators.flexible import create_custom_decorator

        external_defaults = {
            "tags": {"team": team_name, "type": "external"},
            "circuit_breaker": True,
            "timeout": 30.0,
            "retries": 3,
            **defaults,
        }
        return create_custom_decorator(**external_defaults)
    except ImportError:
        return lambda func: func


__all__ = [
    # Core classes
    "Agent",
    "AgentCheckpoint",
    "AgentResult",
    "AgentStatus",
    "Context",
    "DeadLetter",
    "DependencyConfig",
    "DependencyLifecycle",
    # Dependencies
    "DependencyType",
    "FlexibleStateDecorator",
    # Scheduling (if available)
    "GlobalScheduler",
    "InputType",
    "InvalidInputTypeError",
    "InvalidScheduleError",
    "PrioritizedState",
    # State management
    "Priority",
    "ResourceTimeoutError",
    "RetryPolicy",
    "ScheduleBuilder",
    "ScheduleParser",
    "ScheduledAgent",
    "ScheduledInput",
    "SchedulingError",
    # Decorators (if available)
    "StateBuilder",
    "StateMetadata",
    "StateProfile",
    "StateResult",
    "StateStatus",
    "StateType",
    "batch_state",
    "build_state",
    "compare_states",
    "concurrent_state",
    "cpu_intensive",
    "cpu_state",
    "create_custom_decorator",
    "create_environment_decorator",
    "create_external_team_decorator",
    "create_reliable_team_decorator",
    "create_service_decorator",
    # Team decorators
    "create_team_decorator",
    "critical_state",
    "exclusive_state",
    "external_service",
    "external_service_state",
    "fault_tolerant",
    "fault_tolerant_state",
    "get_profile",
    "get_state_config",
    "get_state_coordination",
    "get_state_rate_limit",
    "get_state_requirements",
    "get_state_summary",
    "gpu_accelerated",
    "gpu_state",
    "high_availability",
    "high_priority_state",
    "io_intensive",
    # Inspection utilities
    "is_puffinflow_state",
    "isolated_state",
    "list_profiles",
    "list_state_metadata",
    "memory_intensive",
    "memory_state",
    "minimal_state",
    "network_intensive",
    "parse_magic_prefix",
    "parse_schedule_string",
    "production_state",
    "protected_state",
    "quick_state",
    # Flexible decorators
    "state",
]

__doc__ = """
Enhanced Agent Module for PuffinFlow

This module provides the core Agent class with enhanced features for:
- Direct variable access and manipulation
- Rich context management with multiple content types
- Team coordination and messaging
- Event-driven communication
- State management and checkpointing
- Resource management and reliability patterns

Key Classes:
- Agent: Enhanced agent with direct variable access
- Context: Rich context with outputs, metadata, metrics, caching
- AgentResult: Comprehensive result container
- AgentCheckpoint: State persistence and recovery

Decorators (if available):
- @state: Flexible state decorator with profiles
- @cpu_intensive, @memory_intensive, etc.: Predefined profiles
- @build_state(): Fluent builder pattern

Team Features:
- AgentTeam: Multi-agent coordination
- Messaging between agents
- Event bus for loose coupling
- Shared variables and context

Example:
    from puffinflow import Agent, state

    class DataProcessor(Agent):
        @state(cpu=2.0, memory=512.0)
        async def process_data(self, context):
            # Direct variable access
            batch_size = self.get_variable("batch_size", 1000)

            # Process data
            result = await self.process(batch_size)

            # Set outputs and metrics
            context.set_output("processed_count", len(result))
            context.set_metric("processing_time", time.time())

            return "completed"

    # Create and run agent
    processor = DataProcessor("processor")
    processor.set_variable("batch_size", 2000)

    result = await processor.run()
    print(f"Processed: {result.get_output('processed_count')}")
"""
