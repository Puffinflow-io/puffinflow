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
    "AdaptiveRateLimiter",
    # Existing coordination (if available)
    "AgentCoordinator",
    # Group coordination
    "AgentGroup",
    "AgentOrchestrator",
    # Agent pools
    "AgentPool",
    # Team coordination
    "AgentTeam",
    # Fluent APIs
    "Agents",
    "Barrier",
    "CompletedWork",
    "CompositeRateLimiter",
    "ConditionalAgents",
    "CoordinationConfig",
    "CoordinationPrimitive",
    "DeadlockDetector",
    "DeadlockResolutionStrategy",
    "DynamicProcessingPool",
    "Event",
    "EventBus",
    "ExecutionStrategy",
    "FixedWindow",
    "FluentResult",
    "GroupResult",
    "LeakyBucket",
    "Lease",
    "Lock",
    "Message",
    "Mutex",
    "OrchestrationExecution",
    "OrchestrationResult",
    "ParallelAgentGroup",
    "PipelineAgents",
    "PoolContext",
    "PrimitiveType",
    "Quota",
    "RateLimitStrategy",
    "RateLimiter",
    "ScalingPolicy",
    "Semaphore",
    "SlidingWindow",
    "StageConfig",
    "TeamResult",
    "TokenBucket",
    "WorkItem",
    "WorkProcessor",
    "WorkQueue",
    "collect_agent_outputs",
    "create_agent_team",
    "create_pipeline",
    "create_primitive",
    "create_team",
    "enhance_agent",
    "get_best_agent",
    "run_agents_parallel",
    "run_agents_sequential",
    "run_parallel_agents",
    "run_sequential_agents",
]
