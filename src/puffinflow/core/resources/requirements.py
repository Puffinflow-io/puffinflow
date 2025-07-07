"""
Resource requirements and types for PuffinFlow resource management.
"""

import logging
from dataclasses import dataclass
from enum import Flag
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from ..agent.state import Priority

logger = logging.getLogger(__name__)


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

    def __post_init__(self) -> None:
        """Ensure resource_types is always a valid ResourceType enum."""
        try:
            # Validate that it supports bitwise operations
            try:
                test_result = self.resource_types & ResourceType.CPU
                logger.debug(
                    f"Bitwise test successful: {self.resource_types} & CPU = {test_result}"
                )
            except Exception as e:
                logger.warning(f"Bitwise operation failed: {e}")
                self._auto_determine_resource_types()

        except Exception as e:
            logger.error(f"Error in ResourceRequirements.__post_init__: {e}")
            # Fallback to a safe default
            object.__setattr__(self, "resource_types", ResourceType.ALL)

    def _auto_determine_resource_types(self) -> None:
        """Auto-determine resource_types from individual resource amounts."""
        resource_types = ResourceType.NONE

        if getattr(self, "cpu_units", 0) > 0:
            resource_types |= ResourceType.CPU
        if getattr(self, "memory_mb", 0) > 0:
            resource_types |= ResourceType.MEMORY
        if getattr(self, "io_weight", 0) > 0:
            resource_types |= ResourceType.IO
        if getattr(self, "network_weight", 0) > 0:
            resource_types |= ResourceType.NETWORK
        if getattr(self, "gpu_units", 0) > 0:
            resource_types |= ResourceType.GPU

        # If no specific resources are requested, default to ALL
        if resource_types == ResourceType.NONE:
            resource_types = ResourceType.ALL

        # Use object.__setattr__ for dataclass
        object.__setattr__(self, "resource_types", resource_types)
        logger.info(f"Auto-determined resource_types: {resource_types}")

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
    ResourceType.CPU: "cpu_units",
    ResourceType.MEMORY: "memory_mb",
    ResourceType.IO: "io_weight",
    ResourceType.NETWORK: "network_weight",
    ResourceType.GPU: "gpu_units",
}


def safe_check_resource_type(
    requirements: ResourceRequirements, resource_type: ResourceType
) -> bool:
    """
    Safely check if a resource type is requested in requirements.

    Args:
        requirements: The ResourceRequirements object
        resource_type: The ResourceType to check

    Returns:
        True if the resource type is requested, False otherwise
    """
    try:
        # First try the normal bitwise operation
        return bool(requirements.resource_types & resource_type)
    except TypeError as e:
        logger.warning(
            f"Bitwise operation failed: {e}. Falling back to value comparison."
        )
        try:
            # Fallback to value-based comparison
            return bool(requirements.resource_types.value & resource_type.value)
        except Exception as e2:
            logger.error(
                f"Fallback comparison also failed: {e2}. Assuming resource is requested."
            )
            # If all else fails, check if the individual resource amount is > 0
            # Direct attribute access to avoid circular dependency
            if resource_type in RESOURCE_ATTRIBUTE_MAPPING:
                attr_name = RESOURCE_ATTRIBUTE_MAPPING[resource_type]
                return getattr(requirements, attr_name, 0.0) > 0
            return True


def get_resource_amount(
    requirements: ResourceRequirements, resource_type: ResourceType
) -> float:
    """
    Get the amount of a specific resource type from requirements.

    Args:
        requirements: The ResourceRequirements object
        resource_type: The ResourceType to get the amount for

    Returns:
        The amount of the specified resource type, or 0.0 if the resource type is not enabled

    Raises:
        ValueError: If resource_type is not a single resource type
    """
    if resource_type == ResourceType.NONE:
        return 0.0

    # Check if the resource type is enabled in the requirements
    if not safe_check_resource_type(requirements, resource_type):
        return 0.0

    if resource_type == ResourceType.ALL:
        # Return sum of all resource amounts
        total = 0.0
        for rt, attr in RESOURCE_ATTRIBUTE_MAPPING.items():
            if safe_check_resource_type(requirements, rt):
                total += getattr(requirements, attr, 0.0)
        return total

    # Check if it's a single resource type (power of 2, excluding NONE)
    if (
        resource_type.value > 0
        and (resource_type.value & (resource_type.value - 1)) == 0
        and resource_type in RESOURCE_ATTRIBUTE_MAPPING
    ):
        attr_name = RESOURCE_ATTRIBUTE_MAPPING[resource_type]
        return getattr(requirements, attr_name, 0.0)

    # Handle combined resource types by summing individual types
    total = 0.0
    for rt, attr in RESOURCE_ATTRIBUTE_MAPPING.items():
        try:
            if (resource_type.value & rt.value) != 0 and safe_check_resource_type(
                requirements, rt
            ):
                total += getattr(requirements, attr, 0.0)
        except TypeError:
            # Fallback if 'in' operation fails
            if (resource_type.value & rt.value) != 0 and safe_check_resource_type(
                requirements, rt
            ):
                total += getattr(requirements, attr, 0.0)

    return total
