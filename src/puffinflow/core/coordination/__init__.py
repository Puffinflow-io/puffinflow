"""Enhanced coordination module for multi-agent workflows."""

from .agent_group import (
    AgentGroup,
    AgentOrchestrator,
    ExecutionStrategy,
    GroupResult,
    OrchestrationExecution,
    OrchestrationResult,
    ParallelAgentGroup,
    StageConfig,
)
from .agent_pool import (
    AgentPool,
    CompletedWork,
    DynamicProcessingPool,
    PoolContext,
    ScalingPolicy,
    WorkItem,
    WorkProcessor,
    WorkQueue,
)
from .agent_team import (
    AgentTeam,
    Event,
    EventBus,
    Message,
    TeamResult,
    create_team,
    run_agents_parallel,
    run_agents_sequential,
)
from .fluent_api import (
    Agents,
    ConditionalAgents,
    FluentResult,
    PipelineAgents,
    collect_agent_outputs,
    create_agent_team,
    create_pipeline,
    get_best_agent,
    run_parallel_agents,
    run_sequential_agents,
)

# Import existing coordination components
try:
    from .coordinator import AgentCoordinator, CoordinationConfig, enhance_agent
    from .deadlock import DeadlockDetector, DeadlockResolutionStrategy
    from .primitives import (
        Barrier,
        CoordinationPrimitive,
        Lease,
        Lock,
        Mutex,
        PrimitiveType,
        Quota,
        Semaphore,
        create_primitive,
    )
    from .rate_limiter import (
        AdaptiveRateLimiter,
        CompositeRateLimiter,
        FixedWindow,
        LeakyBucket,
        RateLimiter,
        RateLimitStrategy,
        SlidingWindow,
        TokenBucket,
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
