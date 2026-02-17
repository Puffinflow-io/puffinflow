"""Resource management module for workflow orchestrator."""

from typing import Any as _Any

_LAZY_IMPORTS = {
    # Allocation
    "AllocationRequest": (".allocation", "AllocationRequest"),
    "AllocationResult": (".allocation", "AllocationResult"),
    "AllocationStrategy": (".allocation", "AllocationStrategy"),
    "BestFitAllocator": (".allocation", "BestFitAllocator"),
    "FairShareAllocator": (".allocation", "FairShareAllocator"),
    "FirstFitAllocator": (".allocation", "FirstFitAllocator"),
    "PriorityAllocator": (".allocation", "PriorityAllocator"),
    "ResourceAllocator": (".allocation", "ResourceAllocator"),
    "WorstFitAllocator": (".allocation", "WorstFitAllocator"),
    # Pool
    "ResourceAllocationError": (".pool", "ResourceAllocationError"),
    "ResourceOverflowError": (".pool", "ResourceOverflowError"),
    "ResourcePool": (".pool", "ResourcePool"),
    "ResourceQuotaExceededError": (".pool", "ResourceQuotaExceededError"),
    "ResourceUsageStats": (".pool", "ResourceUsageStats"),
    # Quotas
    "QuotaExceededError": (".quotas", "QuotaExceededError"),
    "QuotaLimit": (".quotas", "QuotaLimit"),
    "QuotaManager": (".quotas", "QuotaManager"),
    "QuotaMetrics": (".quotas", "QuotaMetrics"),
    "QuotaPolicy": (".quotas", "QuotaPolicy"),
    "QuotaScope": (".quotas", "QuotaScope"),
    # Requirements
    "ResourceRequirements": (".requirements", "ResourceRequirements"),
    "ResourceType": (".requirements", "ResourceType"),
}

_SUBMODULES = {"allocation", "pool", "quotas", "requirements"}

__all__ = [
    "AllocationRequest",
    "AllocationResult",
    # Allocation
    "AllocationStrategy",
    "BestFitAllocator",
    "FairShareAllocator",
    "FirstFitAllocator",
    "PriorityAllocator",
    "QuotaExceededError",
    "QuotaLimit",
    # Quotas
    "QuotaManager",
    "QuotaMetrics",
    "QuotaPolicy",
    "QuotaScope",
    "ResourceAllocationError",
    "ResourceAllocator",
    "ResourceOverflowError",
    # Pool
    "ResourcePool",
    "ResourceQuotaExceededError",
    "ResourceRequirements",
    # Requirements
    "ResourceType",
    "ResourceUsageStats",
    "WorstFitAllocator",
    "allocation",
    # Submodules
    "pool",
    "quotas",
    "requirements",
]


def __getattr__(name: str) -> _Any:
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
