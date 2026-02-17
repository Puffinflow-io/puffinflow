"""
PuffinFlow - Workflow Orchestration Framework.
"""

import importlib
from typing import Any

# Import version from setuptools-scm generated file
try:
    from .version import __version__
except ImportError:
    __version__ = "unknown"

__author__ = "Mohamed Ahmed"
__email__ = "mohamed.ahmed.4894@gmail.com"

# ---------------------------------------------------------------------------
# Lazy import system — only resolve names on first access
# ---------------------------------------------------------------------------

_LAZY_IMPORTS = {
    # Core agent functionality
    "Agent": (".core.agent.base", "Agent"),
    "AgentCheckpoint": (".core.agent.checkpoint", "AgentCheckpoint"),
    "AgentResult": (".core.agent.base", "AgentResult"),
    "AgentStatus": (".core.agent.state", "AgentStatus"),
    "Context": (".core.agent.context", "Context"),
    "ExecutionMode": (".core.agent.state", "ExecutionMode"),
    "Priority": (".core.agent.state", "Priority"),
    "StateBuilder": (".core.agent.decorators.builder", "StateBuilder"),
    "StateResult": (".core.agent.state", "StateResult"),
    "StateStatus": (".core.agent.state", "StateStatus"),
    "build_state": (".core.agent.decorators.builder", "build_state"),
    "cpu_intensive": (".core.agent.decorators.flexible", "cpu_intensive"),
    "critical_state": (".core.agent.decorators.flexible", "critical_state"),
    "gpu_accelerated": (".core.agent.decorators.flexible", "gpu_accelerated"),
    "io_intensive": (".core.agent.decorators.flexible", "io_intensive"),
    "memory_intensive": (".core.agent.decorators.flexible", "memory_intensive"),
    "network_intensive": (".core.agent.decorators.flexible", "network_intensive"),
    "state": (".core.agent.decorators.flexible", "state"),
    # Configuration
    "Features": (".core.config", "Features"),
    "Settings": (".core.config", "Settings"),
    "get_features": (".core.config", "get_features"),
    "get_settings": (".core.config", "get_settings"),
    # Coordination
    "AgentGroup": (".core.coordination.agent_group", "AgentGroup"),
    "AgentOrchestrator": (".core.coordination.agent_group", "AgentOrchestrator"),
    "AgentPool": (".core.coordination.agent_pool", "AgentPool"),
    "Agents": (".core.coordination.fluent_api", "Agents"),
    "AgentTeam": (".core.coordination.agent_team", "AgentTeam"),
    "DynamicProcessingPool": (".core.coordination.agent_pool", "DynamicProcessingPool"),
    "EventBus": (".core.coordination.agent_team", "EventBus"),
    "ParallelAgentGroup": (".core.coordination.agent_group", "ParallelAgentGroup"),
    "TeamResult": (".core.coordination.agent_team", "TeamResult"),
    "WorkItem": (".core.coordination.agent_pool", "WorkItem"),
    "WorkQueue": (".core.coordination.agent_pool", "WorkQueue"),
    "create_pipeline": (".core.coordination.fluent_api", "create_pipeline"),
    "create_team": (".core.coordination.agent_team", "create_team"),
    "run_agents_parallel": (".core.coordination.agent_team", "run_agents_parallel"),
    "run_agents_sequential": (".core.coordination.agent_team", "run_agents_sequential"),
    # Reliability
    "Bulkhead": (".core.reliability.bulkhead", "Bulkhead"),
    "BulkheadConfig": (".core.reliability.bulkhead", "BulkheadConfig"),
    "CircuitBreaker": (".core.reliability.circuit_breaker", "CircuitBreaker"),
    "CircuitBreakerConfig": (
        ".core.reliability.circuit_breaker",
        "CircuitBreakerConfig",
    ),
    "ResourceLeakDetector": (".core.reliability.leak_detector", "ResourceLeakDetector"),
    # Resources
    "AllocationStrategy": (".core.resources.allocation", "AllocationStrategy"),
    "QuotaManager": (".core.resources.quotas", "QuotaManager"),
    "ResourcePool": (".core.resources.pool", "ResourcePool"),
    "ResourceRequirements": (".core.resources.requirements", "ResourceRequirements"),
    "ResourceType": (".core.resources.requirements", "ResourceType"),
    # Command / Send
    "Command": (".core.agent.command", "Command"),
    "Send": (".core.agent.command", "Send"),
    # Reducers
    "ReducerRegistry": (".core.agent.reducers", "ReducerRegistry"),
    "add_reducer": (".core.agent.reducers", "add_reducer"),
    "append_reducer": (".core.agent.reducers", "append_reducer"),
    "replace_reducer": (".core.agent.reducers", "replace_reducer"),
    # Streaming
    "StreamEvent": (".core.agent.streaming", "StreamEvent"),
    "StreamMode": (".core.agent.streaming", "StreamMode"),
    # Store
    "BaseStore": (".core.store.base", "BaseStore"),
    "Item": (".core.store.base", "Item"),
    "MemoryStore": (".core.store.base", "MemoryStore"),
    # Subgraph
    "StateMapping": (".core.agent.subgraph", "StateMapping"),
    # Scheduling
    "GlobalScheduler": (".core.agent.scheduling.scheduler", "GlobalScheduler"),
    "ScheduledAgent": (".core.agent.scheduling.scheduler", "ScheduledAgent"),
    "ScheduleBuilder": (".core.agent.scheduling.builder", "ScheduleBuilder"),
    # Checkpoint serialization
    "CheckpointSerializer": (
        ".core.agent.checkpoint_serializer",
        "CheckpointSerializer",
    ),
    "JsonCheckpointSerializer": (
        ".core.agent.checkpoint_serializer",
        "JsonCheckpointSerializer",
    ),
    "MsgpackCheckpointSerializer": (
        ".core.agent.checkpoint_serializer",
        "MsgpackCheckpointSerializer",
    ),
    # Checkpoint backends
    "RedisCheckpointStorage": (
        ".core.agent.checkpoint_backends",
        "RedisCheckpointStorage",
    ),
    "PostgresCheckpointStorage": (
        ".core.agent.checkpoint_backends",
        "PostgresCheckpointStorage",
    ),
    "S3CheckpointStorage": (
        ".core.agent.checkpoint_backends",
        "S3CheckpointStorage",
    ),
    "RetentionPolicy": (".core.agent.checkpoint_backends", "RetentionPolicy"),
    # Drain protocol
    "DrainProtocol": (".core.agent.drain", "DrainProtocol"),
    "DrainResult": (".core.agent.drain", "DrainResult"),
    # Streaming endpoints
    "SSEStreamEndpoint": (
        ".core.agent.streaming_endpoint",
        "SSEStreamEndpoint",
    ),
    "WebSocketStreamEndpoint": (
        ".core.agent.streaming_endpoint",
        "WebSocketStreamEndpoint",
    ),
}

__all__ = [
    "Agent",
    "AgentCheckpoint",
    "AgentGroup",
    "AgentOrchestrator",
    "AgentPool",
    "AgentResult",
    "AgentStatus",
    "AgentTeam",
    "Agents",
    "AllocationStrategy",
    # New features
    "BaseStore",
    "Bulkhead",
    "BulkheadConfig",
    # Checkpoint serialization
    "CheckpointSerializer",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "Command",
    "Context",
    # Drain protocol
    "DrainProtocol",
    "DrainResult",
    "DynamicProcessingPool",
    "EventBus",
    "ExecutionMode",
    "Features",
    # Scheduling
    "GlobalScheduler",
    "Item",
    "JsonCheckpointSerializer",
    "MemoryStore",
    "MsgpackCheckpointSerializer",
    "ParallelAgentGroup",
    "PostgresCheckpointStorage",
    "Priority",
    "QuotaManager",
    # Checkpoint backends
    "RedisCheckpointStorage",
    "ReducerRegistry",
    "ResourceLeakDetector",
    "ResourcePool",
    "ResourceRequirements",
    "ResourceType",
    "RetentionPolicy",
    "S3CheckpointStorage",
    # Streaming endpoints
    "SSEStreamEndpoint",
    "ScheduleBuilder",
    "ScheduledAgent",
    "Send",
    "Settings",
    "StateBuilder",
    "StateMapping",
    "StateResult",
    "StateStatus",
    "StreamEvent",
    "StreamMode",
    "TeamResult",
    "WebSocketStreamEndpoint",
    "WorkItem",
    "WorkQueue",
    "add_reducer",
    "append_reducer",
    "build_state",
    "cpu_intensive",
    "create_pipeline",
    "create_team",
    "critical_state",
    "get_features",
    "get_settings",
    "gpu_accelerated",
    "io_intensive",
    "memory_intensive",
    "network_intensive",
    "replace_reducer",
    "run_agents_parallel",
    "run_agents_sequential",
    "state",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path, __package__)
        val = getattr(mod, attr)
        globals()[name] = val  # Cache for subsequent access
        return val
    raise AttributeError(f"module 'puffinflow' has no attribute {name}")


def get_version() -> str:
    """Get PuffinFlow version."""
    return __version__


def get_info() -> dict[str, str]:
    """Get PuffinFlow package information."""
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "Workflow orchestration framework with advanced resource "
        "management and observability",
    }
