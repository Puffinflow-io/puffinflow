"""Agent module with coordination features."""

from typing import Any, Callable

_LAZY_IMPORTS = {
    # Core classes
    "Agent": (".base", "Agent"),
    "AgentResult": (".base", "AgentResult"),
    "ResourceTimeoutError": (".base", "ResourceTimeoutError"),
    "AgentCheckpoint": (".checkpoint", "AgentCheckpoint"),
    "Command": (".command", "Command"),
    "Send": (".command", "Send"),
    "Context": (".context", "Context"),
    "StateType": (".context", "StateType"),
    "DependencyConfig": (".dependencies", "DependencyConfig"),
    "DependencyLifecycle": (".dependencies", "DependencyLifecycle"),
    "DependencyType": (".dependencies", "DependencyType"),
    "ReducerRegistry": (".reducers", "ReducerRegistry"),
    "add_reducer": (".reducers", "add_reducer"),
    "append_reducer": (".reducers", "append_reducer"),
    "replace_reducer": (".reducers", "replace_reducer"),
    "AgentStatus": (".state", "AgentStatus"),
    "DeadLetter": (".state", "DeadLetter"),
    "ExecutionMode": (".state", "ExecutionMode"),
    "PrioritizedState": (".state", "PrioritizedState"),
    "Priority": (".state", "Priority"),
    "RetryPolicy": (".state", "RetryPolicy"),
    "StateMetadata": (".state", "StateMetadata"),
    "StateResult": (".state", "StateResult"),
    "StateStatus": (".state", "StateStatus"),
    "StreamEvent": (".streaming", "StreamEvent"),
    "StreamManager": (".streaming", "StreamManager"),
    "StreamMode": (".streaming", "StreamMode"),
    "StateMapping": (".subgraph", "StateMapping"),
    # Decorators - builder
    "StateBuilder": (".decorators.builder", "StateBuilder"),
    "build_state": (".decorators.builder", "build_state"),
    "cpu_state": (".decorators.builder", "cpu_state"),
    "exclusive_state": (".decorators.builder", "exclusive_state"),
    "external_service_state": (".decorators.builder", "external_service_state"),
    "fault_tolerant_state": (".decorators.builder", "fault_tolerant_state"),
    "gpu_state": (".decorators.builder", "gpu_state"),
    "high_priority_state": (".decorators.builder", "high_priority_state"),
    "isolated_state": (".decorators.builder", "isolated_state"),
    "memory_state": (".decorators.builder", "memory_state"),
    "production_state": (".decorators.builder", "production_state"),
    "protected_state": (".decorators.builder", "protected_state"),
    # Decorators - flexible
    "FlexibleStateDecorator": (".decorators.flexible", "FlexibleStateDecorator"),
    "StateProfile": (".decorators.flexible", "StateProfile"),
    "batch_state": (".decorators.flexible", "batch_state"),
    "concurrent_state": (".decorators.flexible", "concurrent_state"),
    "cpu_intensive": (".decorators.flexible", "cpu_intensive"),
    "create_custom_decorator": (".decorators.flexible", "create_custom_decorator"),
    "critical_state": (".decorators.flexible", "critical_state"),
    "external_service": (".decorators.flexible", "external_service"),
    "fault_tolerant": (".decorators.flexible", "fault_tolerant"),
    "get_profile": (".decorators.flexible", "get_profile"),
    "gpu_accelerated": (".decorators.flexible", "gpu_accelerated"),
    "high_availability": (".decorators.flexible", "high_availability"),
    "io_intensive": (".decorators.flexible", "io_intensive"),
    "list_profiles": (".decorators.flexible", "list_profiles"),
    "memory_intensive": (".decorators.flexible", "memory_intensive"),
    "minimal_state": (".decorators.flexible", "minimal_state"),
    "network_intensive": (".decorators.flexible", "network_intensive"),
    "quick_state": (".decorators.flexible", "quick_state"),
    "state": (".decorators.flexible", "state"),
    # Decorators - inspection
    "compare_states": (".decorators.inspection", "compare_states"),
    "get_state_config": (".decorators.inspection", "get_state_config"),
    "get_state_coordination": (".decorators.inspection", "get_state_coordination"),
    "get_state_rate_limit": (".decorators.inspection", "get_state_rate_limit"),
    "get_state_requirements": (".decorators.inspection", "get_state_requirements"),
    "get_state_summary": (".decorators.inspection", "get_state_summary"),
    "is_puffinflow_state": (".decorators.inspection", "is_puffinflow_state"),
    "list_state_metadata": (".decorators.inspection", "list_state_metadata"),
    # Scheduling
    "GlobalScheduler": (".scheduling.scheduler", "GlobalScheduler"),
    "InputType": (".scheduling.inputs", "InputType"),
    "InvalidInputTypeError": (".scheduling.exceptions", "InvalidInputTypeError"),
    "InvalidScheduleError": (".scheduling.exceptions", "InvalidScheduleError"),
    "ScheduleBuilder": (".scheduling.builder", "ScheduleBuilder"),
    "ScheduledAgent": (".scheduling.scheduler", "ScheduledAgent"),
    "ScheduledInput": (".scheduling.inputs", "ScheduledInput"),
    "ScheduleParser": (".scheduling.parser", "ScheduleParser"),
    "SchedulingError": (".scheduling.exceptions", "SchedulingError"),
    "parse_magic_prefix": (".scheduling.inputs", "parse_magic_prefix"),
    "parse_schedule_string": (".scheduling.parser", "parse_schedule_string"),
}

_SUBMODULES = {"decorators", "scheduling"}


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
    # Command / Send
    "Command",
    "Context",
    "DeadLetter",
    "DependencyConfig",
    "DependencyLifecycle",
    # Dependencies
    "DependencyType",
    "ExecutionMode",
    "FlexibleStateDecorator",
    # Scheduling (if available)
    "GlobalScheduler",
    "InputType",
    "InvalidInputTypeError",
    "InvalidScheduleError",
    "PrioritizedState",
    # State management
    "Priority",
    # Reducers
    "ReducerRegistry",
    "ResourceTimeoutError",
    "RetryPolicy",
    "ScheduleBuilder",
    "ScheduleParser",
    "ScheduledAgent",
    "ScheduledInput",
    "SchedulingError",
    "Send",
    # Decorators (if available)
    "StateBuilder",
    # Subgraph
    "StateMapping",
    "StateMetadata",
    "StateProfile",
    "StateResult",
    "StateStatus",
    "StateType",
    # Streaming
    "StreamEvent",
    "StreamManager",
    "StreamMode",
    "add_reducer",
    "append_reducer",
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
    "replace_reducer",
    # Flexible decorators
    "state",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path, __package__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    if name in _SUBMODULES:
        import importlib

        mod = importlib.import_module(f".{name}", __package__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__doc__ = """
Agent Module for PuffinFlow

This module provides the core Agent class with features for:
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
