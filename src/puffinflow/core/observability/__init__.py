"""PuffinFlow Observability System"""

# Import submodules for import path tests
# Clean up indirect imports that might leak from submodules

from . import agent, config, context, core, decorators
from .agent import ObservableAgent
from .config import ObservabilityConfig
from .context import ObservableContext
from .core import ObservabilityManager, get_observability, setup_observability
from .decorators import observe, trace_state

# Clean up imports - just comment out since variables don't exist
# with contextlib.suppress(NameError):
#     del interfaces, tracing, metrics, alerting, events

__all__ = [
    "ObservabilityManager",
    "get_observability",
    "setup_observability",
    "ObservabilityConfig",
    "observe",
    "trace_state",
    "ObservableContext",
    "ObservableAgent",
    "core",
    "config",
    "decorators",
    "context",
    "agent",
]
