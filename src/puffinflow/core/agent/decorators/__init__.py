"""Enhanced state decorators with flexible configuration."""

from typing import Any

_LAZY_IMPORTS = {
    # Builder
    "StateBuilder": (".builder", "StateBuilder"),
    "build_state": (".builder", "build_state"),
    "cpu_state": (".builder", "cpu_state"),
    "exclusive_state": (".builder", "exclusive_state"),
    "gpu_state": (".builder", "gpu_state"),
    "memory_state": (".builder", "memory_state"),
    "builder_concurrent_state": (".builder", "concurrent_state"),
    "builder_critical_state": (".builder", "critical_state"),
    "builder_high_priority_state": (".builder", "high_priority_state"),
    # Flexible
    "PROFILES": (".flexible", "PROFILES"),
    "FlexibleStateDecorator": (".flexible", "FlexibleStateDecorator"),
    "StateProfile": (".flexible", "StateProfile"),
    "batch_state": (".flexible", "batch_state"),
    "concurrent_state": (".flexible", "concurrent_state"),
    "cpu_intensive": (".flexible", "cpu_intensive"),
    "create_custom_decorator": (".flexible", "create_custom_decorator"),
    "critical_state": (".flexible", "critical_state"),
    "get_profile": (".flexible", "get_profile"),
    "gpu_accelerated": (".flexible", "gpu_accelerated"),
    "io_intensive": (".flexible", "io_intensive"),
    "list_profiles": (".flexible", "list_profiles"),
    "memory_intensive": (".flexible", "memory_intensive"),
    "minimal_state": (".flexible", "minimal_state"),
    "network_intensive": (".flexible", "network_intensive"),
    "quick_state": (".flexible", "quick_state"),
    "state": (".flexible", "state"),
    "synchronized_state": (".flexible", "synchronized_state"),
    # Inspection
    "compare_states": (".inspection", "compare_states"),
    "get_state_config": (".inspection", "get_state_config"),
    "get_state_coordination": (".inspection", "get_state_coordination"),
    "get_state_rate_limit": (".inspection", "get_state_rate_limit"),
    "get_state_requirements": (".inspection", "get_state_requirements"),
    "get_state_summary": (".inspection", "get_state_summary"),
    "is_puffinflow_state": (".inspection", "is_puffinflow_state"),
    "list_state_metadata": (".inspection", "list_state_metadata"),
}

__all__ = [
    "PROFILES",
    "FlexibleStateDecorator",
    # Builder pattern
    "StateBuilder",
    "StateProfile",
    "batch_state",
    "build_state",
    "builder_concurrent_state",
    "builder_critical_state",
    "builder_high_priority_state",
    "compare_states",
    "concurrent_state",
    "cpu_intensive",
    "cpu_state",
    "create_custom_decorator",
    "critical_state",
    "exclusive_state",
    # Profile management
    "get_profile",
    "get_state_config",
    "get_state_coordination",
    "get_state_rate_limit",
    "get_state_requirements",
    "get_state_summary",
    "gpu_accelerated",
    "gpu_state",
    "io_intensive",
    # Inspection utilities
    "is_puffinflow_state",
    "list_profiles",
    "list_state_metadata",
    "memory_intensive",
    "memory_state",
    # Profile-based decorators
    "minimal_state",
    "network_intensive",
    "quick_state",
    # Main decorator
    "state",
    "synchronized_state",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path, __package__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
