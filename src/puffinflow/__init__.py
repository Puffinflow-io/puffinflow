"""
PuffinFlow - A comprehensive workflow orchestration framework.

PuffinFlow provides a powerful, async-first workflow orchestration system with
advanced resource management, monitoring, and coordination primitives.
"""

__version__ = "0.1.0"
__author__ = "Mohamed Ahmed"
__email__ = "mohamed.ahmed.4894@gmail.com"

# Core imports for convenience
from .core.agent import Agent, Context, state
from .core.resources import ResourcePool, ResourceType
from .core.coordination import AgentCoordinator

# Type annotations
from .core.agent.state import Priority, StateStatus

__all__ = [
    # Core classes
    "Agent",
    "Context",
    "ResourcePool",
    "AgentCoordinator",
    # Decorators
    "state",
    # Enums
    "Priority",
    "StateStatus",
    "ResourceType",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]

# Package metadata for runtime access
def get_version():
    """Get the current version of PuffinFlow."""
    return __version__

def get_info():
    """Get package information."""
    return {
        "name": "puffinflow",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "A comprehensive workflow orchestration framework",
    }