"""Enhanced coordination module for multi-agent workflows."""

from .agent_team import AgentTeam, TeamResult, EventBus, Message, Event, create_team, run_agents_parallel, run_agents_sequential
from .agent_group import (
    AgentGroup, ParallelAgentGroup, AgentOrchestrator, GroupResult,
    OrchestrationExecution, OrchestrationResult, ExecutionStrategy, StageConfig
)
from .fluent_api import (
    Agents, ConditionalAgents, PipelineAgents, FluentResult,
    run_parallel_agents, run_sequential_agents, collect_agent_outputs,
    get_best_agent, create_pipeline, create_agent_team
)
from .agent_pool import (
    AgentPool, WorkQueue, WorkItem, CompletedWork, DynamicProcessingPool,
    ScalingPolicy, PoolContext, WorkProcessor
)

# Import existing coordination components
try:
    from .coordinator import AgentCoordinator, CoordinationConfig, enhance_agent
    from .deadlock import DeadlockDetector, DeadlockResolutionStrategy
    from .primitives import (
        CoordinationPrimitive, Mutex, Semaphore, Barrier, Lease, Lock, Quota,
        PrimitiveType, create_primitive
    )
    from .rate_limiter import (
        RateLimiter, TokenBucket, LeakyBucket, SlidingWindow, FixedWindow,
        AdaptiveRateLimiter, CompositeRateLimiter, RateLimitStrategy
    )
except ImportError:
    # Some coordination components may not be available
    pass

__all__ = [
    # Team coordination
    "AgentTeam",
    "TeamResult",
    "EventBus",
    "Message",
    "Event",
    "create_team",
    "run_agents_parallel",
    "run_agents_sequential",

    # Group coordination
    "AgentGroup",
    "ParallelAgentGroup",
    "AgentOrchestrator",
    "GroupResult",
    "OrchestrationExecution",
    "OrchestrationResult",
    "ExecutionStrategy",
    "StageConfig",

    # Fluent APIs
    "Agents",
    "ConditionalAgents",
    "PipelineAgents",
    "FluentResult",
    "run_parallel_agents",
    "run_sequential_agents",
    "collect_agent_outputs",
    "get_best_agent",
    "create_pipeline",
    "create_agent_team",

    # Agent pools
    "AgentPool",
    "WorkQueue",
    "WorkItem",
    "CompletedWork",
    "DynamicProcessingPool",
    "ScalingPolicy",
    "PoolContext",
    "WorkProcessor",

    # Existing coordination (if available)
    "AgentCoordinator",
    "CoordinationConfig",
    "enhance_agent",
    "DeadlockDetector",
    "DeadlockResolutionStrategy",
    "CoordinationPrimitive",
    "Mutex",
    "Semaphore",
    "Barrier",
    "Lease",
    "Lock",
    "Quota",
    "PrimitiveType",
    "create_primitive",
    "RateLimiter",
    "TokenBucket",
    "LeakyBucket",
    "SlidingWindow",
    "FixedWindow",
    "AdaptiveRateLimiter",
    "CompositeRateLimiter",
    "RateLimitStrategy",
]