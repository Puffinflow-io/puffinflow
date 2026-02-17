"""Workflow Orchestrator Core Engine."""

# Version is managed by the parent package

_LAZY_IMPORTS = {
    "Agent": (".agent.base", "Agent"),
    "Context": (".agent.context", "Context"),
    "state": (".agent.decorators.flexible", "state"),
    "Priority": (".agent.state", "Priority"),
    "StateStatus": (".agent.state", "StateStatus"),
    "Bulkhead": (".reliability.bulkhead", "Bulkhead"),
    "BulkheadConfig": (".reliability.bulkhead", "BulkheadConfig"),
    "CircuitBreaker": (".reliability.circuit_breaker", "CircuitBreaker"),
    "CircuitBreakerConfig": (".reliability.circuit_breaker", "CircuitBreakerConfig"),
    "ResourceLeakDetector": (".reliability.leak_detector", "ResourceLeakDetector"),
    "ResourcePool": (".resources.pool", "ResourcePool"),
    "ResourceRequirements": (".resources.requirements", "ResourceRequirements"),
    "ResourceType": (".resources.requirements", "ResourceType"),
}

__all__ = [
    "Agent",
    "Bulkhead",
    "BulkheadConfig",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "Context",
    "Priority",
    "ResourceLeakDetector",
    "ResourcePool",
    "ResourceRequirements",
    "ResourceType",
    "StateStatus",
    "state",
]


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path, __package__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
