"""Reliability patterns for production workflows."""

from typing import Any

_LAZY_IMPORTS = {
    "Bulkhead": (".bulkhead", "Bulkhead"),
    "BulkheadConfig": (".bulkhead", "BulkheadConfig"),
    "BulkheadFullError": (".bulkhead", "BulkheadFullError"),
    "CircuitBreaker": (".circuit_breaker", "CircuitBreaker"),
    "CircuitBreakerConfig": (".circuit_breaker", "CircuitBreakerConfig"),
    "CircuitBreakerError": (".circuit_breaker", "CircuitBreakerError"),
    "CircuitState": (".circuit_breaker", "CircuitState"),
    "ResourceLeak": (".leak_detector", "ResourceLeak"),
    "ResourceLeakDetector": (".leak_detector", "ResourceLeakDetector"),
}

_SUBMODULES = {"bulkhead", "circuit_breaker", "leak_detector"}

__all__ = [
    "Bulkhead",
    "BulkheadConfig",
    "BulkheadFullError",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitState",
    "ResourceLeak",
    "ResourceLeakDetector",
    "bulkhead",
    "circuit_breaker",
    "leak_detector",
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
