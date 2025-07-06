"""Resource management module for workflow orchestrator."""

# Import submodules for import path tests
from . import allocation, pool, quotas, requirements
from .allocation import (
    AllocationRequest,
    AllocationResult,
    AllocationStrategy,
    BestFitAllocator,
    FairShareAllocator,
    FirstFitAllocator,
    PriorityAllocator,
    ResourceAllocator,
    WorstFitAllocator,
)
from .pool import (
    ResourceAllocationError,
    ResourceOverflowError,
    ResourcePool,
    ResourceQuotaExceededError,
    ResourceUsageStats,
)
from .quotas import (
    QuotaExceededError,
    QuotaLimit,
    QuotaManager,
    QuotaMetrics,
    QuotaPolicy,
    QuotaScope,
)
from .requirements import (
    ResourceRequirements,
    ResourceType,
)

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

# Clean up module namespace
import sys as _sys

_current_module = _sys.modules[__name__]
for _attr_name in dir(_current_module):
    if not _attr_name.startswith("_") and _attr_name not in __all__:
        delattr(_current_module, _attr_name)
del _sys, _current_module, _attr_name
