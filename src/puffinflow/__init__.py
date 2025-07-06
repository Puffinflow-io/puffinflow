"""PuffinFlow -  Workflow Orchestration Framework."""

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
    # Core
    "Agent",
    "AgentResult",
    "Context",
    "AgentCheckpoint",
    "Priority",
    "AgentStatus",
    "StateStatus",
    "StateResult",

    # Decorators
    "state",
    "cpu_intensive",
    "memory_intensive",
    "io_intensive",
    "gpu_accelerated",
    "network_intensive",
    "critical_state",
    "build_state",
    "StateBuilder",

    # Coordination
    "AgentTeam",
    "TeamResult",
    "AgentGroup",
    "ParallelAgentGroup",
    "AgentOrchestrator",
    "Agents",
    "run_agents_parallel",
    "run_agents_sequential",
    "AgentPool",
    "WorkQueue",
    "WorkItem",
    "DynamicProcessingPool",
    "EventBus",
    "create_team",
    "create_pipeline",

    # Resources
    "ResourceRequirements",
    "ResourceType",
    "ResourcePool",
    "QuotaManager",
    "AllocationStrategy",

    # Reliability
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "Bulkhead",
    "BulkheadConfig",
    "ResourceLeakDetector",

    # Configuration
    "Settings",
    "get_settings",
    "Features",
    "get_features",
]

def get_version():
    """Get PuffinFlow version."""
    return __version__

def get_info():
    """Get PuffinFlow package information."""
    return {
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "Workflow orchestration framework with advanced resource management and observability"
    }

# For backwards compatibility
from .core.agent.base import Agent as BaseAgent
