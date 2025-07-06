"""Utilities for inspecting decorated states."""

from typing import Any, Callable, Optional

from ...coordination.rate_limiter import RateLimitStrategy
from ...resources.requirements import ResourceRequirements
from ..state import Priority


def is_puffinflow_state(func: Callable) -> bool:
    """Check if a function is a PuffinFlow state."""
    return getattr(func, "_puffinflow_state", False)


def get_state_config(func: Callable) -> Optional[dict[str, Any]]:
    """Get the configuration of a PuffinFlow state."""
    return getattr(func, "_state_config", None)


def get_state_requirements(func: Callable) -> Optional[ResourceRequirements]:
    """Get the resource requirements of a PuffinFlow state."""
    return getattr(func, "_resource_requirements", None)


def get_state_rate_limit(func: Callable) -> Optional[dict[str, Any]]:
    """Get the rate limiting configuration of a PuffinFlow state."""
    if not hasattr(func, "_rate_limit"):
        return None

    strategy = getattr(func, "_rate_strategy", RateLimitStrategy.TOKEN_BUCKET)

    return {
        "rate": func._rate_limit,
        "burst": getattr(func, "_burst_limit", None),
        "strategy": strategy,
    }


def get_state_coordination(func: Callable) -> Optional[dict[str, Any]]:
    """Get the coordination configuration of a PuffinFlow state."""
    primitive = getattr(func, "_coordination_primitive", None)
    if primitive is None:
        return None

    return {"type": primitive, "config": getattr(func, "_coordination_config", {})}


def list_state_metadata(func: Callable) -> dict[str, Any]:
    """Get all metadata for a PuffinFlow state."""
    if not is_puffinflow_state(func):
        return {}

    # Get description, falling back to docstring if not set
    description = getattr(func, "_state_description", "")
    if not description and func.__doc__:
        description = func.__doc__.strip()

    # If still no description, use fallback
    if not description:
        description = f"State: {func.__name__}"

    return {
        "name": getattr(func, "_state_name", func.__name__),
        "description": description,
        "tags": getattr(func, "_state_tags", {}),
        "priority": getattr(func, "_priority", Priority.NORMAL),
        "requirements": get_state_requirements(func),
        "rate_limit": get_state_rate_limit(func),
        "coordination": get_state_coordination(func),
        "dependencies": getattr(func, "_dependency_configs", {}),
        "preemptible": getattr(func, "_preemptible", False),
        "checkpoint_interval": getattr(func, "_checkpoint_interval", None),
        "cleanup_on_failure": getattr(func, "_cleanup_on_failure", True),
    }


def compare_states(func1: Callable, func2: Callable) -> dict[str, Any]:
    """Compare two state configurations."""
    config1 = get_state_config(func1) or {}
    config2 = get_state_config(func2) or {}

    differences = {}
    all_keys = set(config1.keys()) | set(config2.keys())

    for key in all_keys:
        val1 = config1.get(key)
        val2 = config2.get(key)
        if val1 != val2:
            differences[key] = {"func1": val1, "func2": val2}

    return differences


def get_state_summary(func: Callable) -> str:
    """Get a human-readable summary of state configuration."""
    if not is_puffinflow_state(func):
        return f"{func.__name__}: Not a PuffinFlow state"

    config = get_state_config(func)
    if config is None or not isinstance(config, dict):
        return f"{func.__name__}: No configuration found"

    summary_parts = [f"{func.__name__}:"]

    # Resources
    resources = []
    cpu = config.get("cpu", 0)
    if cpu is not None and cpu > 0:
        resources.append(f"CPU={cpu}")
    memory = config.get("memory", 0)
    if memory is not None and memory > 0:
        resources.append(f"Memory={memory}MB")
    gpu = config.get("gpu", 0)
    if gpu is not None and gpu > 0:
        resources.append(f"GPU={gpu}")

    if resources:
        summary_parts.append(f"  Resources: {', '.join(resources)}")

    # Priority
    priority = config.get("priority")
    if priority and priority != Priority.NORMAL:
        summary_parts.append(f"  Priority: {priority.name}")

    # Coordination
    coord_info = []
    if config.get("mutex"):
        coord_info.append("Mutex")
    if config.get("semaphore"):
        coord_info.append(f"Semaphore({config['semaphore']})")
    if config.get("barrier"):
        coord_info.append(f"Barrier({config['barrier']})")
    if config.get("rate_limit"):
        coord_info.append(f"RateLimit({config['rate_limit']}/s)")

    if coord_info:
        summary_parts.append(f"  Coordination: {', '.join(coord_info)}")

    # Dependencies
    deps = config.get("depends_on")
    if deps:
        summary_parts.append(f"  Dependencies: {', '.join(deps)}")

    return "\n".join(summary_parts)
