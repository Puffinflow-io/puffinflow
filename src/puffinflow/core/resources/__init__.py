"""Resource management module for workflow orchestrator."""

# Import submodules for import path tests
from src.puffinflow.core.resources import allocation, pool, quotas, requirements
from src.puffinflow.core.resources.allocation import (
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
from src.puffinflow.core.resources.pool import (
    ResourceAllocationError,
    ResourceOverflowError,
    ResourcePool,
    ResourceQuotaExceededError,
    ResourceUsageStats,
)
from src.puffinflow.core.resources.quotas import (
    QuotaExceededError,
    QuotaLimit,
    QuotaManager,
    QuotaMetrics,
    QuotaPolicy,
    QuotaScope,
)
from src.puffinflow.core.resources.requirements import (
    ResourceRequirements,
    ResourceType,
)

__all__ = [
    # Pool
    "ResourcePool",
    "ResourceAllocationError",
    "ResourceOverflowError",
    "ResourceQuotaExceededError",
    "ResourceUsageStats",

    # Requirements
    "ResourceType",
    "ResourceRequirements",

    # Quotas
    "QuotaManager",
    "QuotaPolicy",
    "QuotaLimit",
    "QuotaScope",
    "QuotaExceededError",
    "QuotaMetrics",

    # Allocation
    "AllocationStrategy",
    "AllocationRequest",
    "AllocationResult",
    "FirstFitAllocator",
    "BestFitAllocator",
    "WorstFitAllocator",
    "PriorityAllocator",
    "FairShareAllocator",
    "ResourceAllocator",

    # Submodules
    "pool",
    "requirements",
    "quotas",
    "allocation",
]

# Clean up module namespace
import sys as _sys

_current_module = _sys.modules[__name__]
for _attr_name in dir(_current_module):
    if not _attr_name.startswith('_') and _attr_name not in __all__:
        delattr(_current_module, _attr_name)
del _sys, _current_module, _attr_name
