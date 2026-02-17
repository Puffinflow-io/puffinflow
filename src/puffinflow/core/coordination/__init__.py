"""Coordination module for multi-agent workflows."""

from typing import Any as _Any

_LAZY_IMPORTS = {
    # Agent group
    "AgentGroup": (".agent_group", "AgentGroup"),
    "AgentOrchestrator": (".agent_group", "AgentOrchestrator"),
    "ExecutionStrategy": (".agent_group", "ExecutionStrategy"),
    "GroupResult": (".agent_group", "GroupResult"),
    "OrchestrationExecution": (".agent_group", "OrchestrationExecution"),
    "OrchestrationResult": (".agent_group", "OrchestrationResult"),
    "ParallelAgentGroup": (".agent_group", "ParallelAgentGroup"),
    "StageConfig": (".agent_group", "StageConfig"),
    # Agent pool
    "AgentPool": (".agent_pool", "AgentPool"),
    "CompletedWork": (".agent_pool", "CompletedWork"),
    "DynamicProcessingPool": (".agent_pool", "DynamicProcessingPool"),
    "PoolContext": (".agent_pool", "PoolContext"),
    "ScalingPolicy": (".agent_pool", "ScalingPolicy"),
    "WorkItem": (".agent_pool", "WorkItem"),
    "WorkProcessor": (".agent_pool", "WorkProcessor"),
    "WorkQueue": (".agent_pool", "WorkQueue"),
    # Agent team
    "AgentTeam": (".agent_team", "AgentTeam"),
    "Event": (".agent_team", "Event"),
    "EventBus": (".agent_team", "EventBus"),
    "Message": (".agent_team", "Message"),
    "TeamResult": (".agent_team", "TeamResult"),
    "create_team": (".agent_team", "create_team"),
    "run_agents_parallel": (".agent_team", "run_agents_parallel"),
    "run_agents_sequential": (".agent_team", "run_agents_sequential"),
    # Fluent API
    "Agents": (".fluent_api", "Agents"),
    "ConditionalAgents": (".fluent_api", "ConditionalAgents"),
    "FluentResult": (".fluent_api", "FluentResult"),
    "PipelineAgents": (".fluent_api", "PipelineAgents"),
    "collect_agent_outputs": (".fluent_api", "collect_agent_outputs"),
    "create_agent_team": (".fluent_api", "create_agent_team"),
    "create_pipeline": (".fluent_api", "create_pipeline"),
    "get_best_agent": (".fluent_api", "get_best_agent"),
    "run_parallel_agents": (".fluent_api", "run_parallel_agents"),
    "run_sequential_agents": (".fluent_api", "run_sequential_agents"),
    # Coordinator
    "AgentCoordinator": (".coordinator", "AgentCoordinator"),
    "CoordinationConfig": (".coordinator", "CoordinationConfig"),
    "enhance_agent": (".coordinator", "enhance_agent"),
    # Deadlock
    "DeadlockDetector": (".deadlock", "DeadlockDetector"),
    "DeadlockResolutionStrategy": (".deadlock", "DeadlockResolutionStrategy"),
    # Primitives
    "Barrier": (".primitives", "Barrier"),
    "CoordinationPrimitive": (".primitives", "CoordinationPrimitive"),
    "Lease": (".primitives", "Lease"),
    "Lock": (".primitives", "Lock"),
    "Mutex": (".primitives", "Mutex"),
    "PrimitiveType": (".primitives", "PrimitiveType"),
    "Quota": (".primitives", "Quota"),
    "Semaphore": (".primitives", "Semaphore"),
    "create_primitive": (".primitives", "create_primitive"),
    # Rate limiter
    "AdaptiveRateLimiter": (".rate_limiter", "AdaptiveRateLimiter"),
    "CompositeRateLimiter": (".rate_limiter", "CompositeRateLimiter"),
    "FixedWindow": (".rate_limiter", "FixedWindow"),
    "LeakyBucket": (".rate_limiter", "LeakyBucket"),
    "RateLimiter": (".rate_limiter", "RateLimiter"),
    "RateLimitStrategy": (".rate_limiter", "RateLimitStrategy"),
    "SlidingWindow": (".rate_limiter", "SlidingWindow"),
    "TokenBucket": (".rate_limiter", "TokenBucket"),
}

__all__ = [
    "AdaptiveRateLimiter",
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


def __getattr__(name: str) -> _Any:
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path, __package__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
