"""
PuffinFlow - Workflow Orchestration Framework.
"""

__version__ = "0.1.dev12+g0f61537.d20250701"
__author__ = "Mohamed Ahmed"
__email__ = "mohamed.ahmed.4894@gmail.com"

# Core agent functionality
from .core.agent import (
    Agent,
    AgentCheckpoint,
    AgentResult,
    AgentStatus,
    Context,
    Priority,
    StateBuilder,
    StateResult,
    StateStatus,
    build_state,
    cpu_intensive,
    critical_state,
    gpu_accelerated,
    io_intensive,
    memory_intensive,
    network_intensive,
    state,
)

# Configuration
from .core.config import Features, Settings, get_features, get_settings

# Enhanced coordination
from .core.coordination import (
    AgentGroup,
    AgentOrchestrator,
    AgentPool,
    Agents,
    AgentTeam,
    DynamicProcessingPool,
    EventBus,
    ParallelAgentGroup,
    TeamResult,
    WorkItem,
    WorkQueue,
    create_pipeline,
    create_team,
    run_agents_parallel,
    run_agents_sequential,
)

# Reliability patterns
from .core.reliability import (
    Bulkhead,
    BulkheadConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    ResourceLeakDetector,
)

# Resource management
from .core.resources import (
    AllocationStrategy,
    QuotaManager,
    ResourcePool,
    ResourceRequirements,
    ResourceType,
)

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
    "Bulkhead",
    "BulkheadConfig",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "Context",
    "DynamicProcessingPool",
    "EventBus",
    "Features",
    "ParallelAgentGroup",
    "Priority",
    "QuotaManager",
    "ResourceLeakDetector",
    "ResourcePool",
    "ResourceRequirements",
    "ResourceType",
    "Settings",
    "StateBuilder",
    "StateResult",
    "StateStatus",
    "TeamResult",
    "WorkItem",
    "WorkQueue",
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
    "run_agents_parallel",
    "run_agents_sequential",
    "state",
]


def get_version() -> str:
    """Get PuffinFlow version."""
    return __version__


def get_info() -> dict[str, str]:
    """Get PuffinFlow package information."""
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "Workflow orchestration framework with advanced resource management and observability",
    }
