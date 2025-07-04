"""
Resource requirements and types for PuffinFlow resource management.
"""

from dataclasses import dataclass
from enum import Flag
from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent.state import Priority


class ResourceType(Flag):
    """Resource type flags for specifying required resources."""

    NONE = 0
    CPU = 1
    MEMORY = 2
    IO = 4
    NETWORK = 8
    GPU = 16

    # Convenience combination for all resource types
    ALL = CPU | MEMORY | IO | NETWORK | GPU


@dataclass
class ResourceRequirements:
    """
    Specifies resource requirements for agent states.

    This class defines the computational resources needed by an agent state,
    including CPU, memory, I/O, network, and GPU resources, along with
    priority and timeout specifications.
    """

    cpu_units: float = 1.0
    memory_mb: float = 100.0
    io_weight: float = 1.0
    network_weight: float = 1.0
    gpu_units: float = 0.0
    priority_boost: int = 0
    timeout: Optional[float] = None
    resource_types: ResourceType = ResourceType.ALL

    @property
    def priority(self) -> "Priority":
        """Get priority level based on priority_boost."""
        from ..agent.state import Priority
        if self.priority_boost <= 0:
            return Priority.LOW
        elif self.priority_boost == 1:
            return Priority.NORMAL
        elif self.priority_boost == 2:
            return Priority.HIGH
        else:  # priority_boost >= 3
            return Priority.CRITICAL

    @priority.setter
    def priority(self, value: Union["Priority", int]) -> None:
        """Set priority level, updating priority_boost accordingly."""
        from ..agent.state import Priority
        if isinstance(value, Priority):
            self.priority_boost = value.value
        elif isinstance(value, int):
            self.priority_boost = value
        else:
            raise TypeError(f"Priority must be Priority enum or int, got {type(value)}")


# Resource attribute mapping for get_resource_amount function
RESOURCE_ATTRIBUTE_MAPPING = {
    ResourceType.CPU: 'cpu_units',
    ResourceType.MEMORY: 'memory_mb',
    ResourceType.IO: 'io_weight',
    ResourceType.NETWORK: 'network_weight',
    ResourceType.GPU: 'gpu_units'
}


def get_resource_amount(requirements: ResourceRequirements, resource_type: ResourceType) -> float:
    """
    Get the amount of a specific resource type from requirements.

    Args:
        requirements: The ResourceRequirements object
        resource_type: The ResourceType to get the amount for

    Returns:
        The amount of the specified resource type

    Raises:
        ValueError: If resource_type is not a single resource type
    """
    if resource_type == ResourceType.NONE:
        return 0.0

    if resource_type == ResourceType.ALL:
        # Return sum of all resource amounts
        total = 0.0
        for rt, attr in RESOURCE_ATTRIBUTE_MAPPING.items():
            total += getattr(requirements, attr, 0.0)
        return total

    # Check if it's a single resource type (power of 2, excluding NONE)
    if resource_type.value > 0 and (resource_type.value & (resource_type.value - 1)) == 0:
        # Single resource type
        if resource_type in RESOURCE_ATTRIBUTE_MAPPING:
            attr_name = RESOURCE_ATTRIBUTE_MAPPING[resource_type]
            return getattr(requirements, attr_name, 0.0)

    # Handle combined resource types by summing individual types
    total = 0.0
    for rt, attr in RESOURCE_ATTRIBUTE_MAPPING.items():
        if rt in resource_type:
            total += getattr(requirements, attr, 0.0)
    return total