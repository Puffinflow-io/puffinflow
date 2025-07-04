"""PuffinFlow Observability System"""

from .core import ObservabilityManager, get_observability, setup_observability
from .config import ObservabilityConfig
from .decorators import observe, trace_state
from .context import ObservableContext
from .agent import ObservableAgent

# Import submodules for import path tests
from . import core, config, decorators, context, agent

# Clean up indirect imports that might leak from submodules
try:
    del interfaces, tracing, metrics, alerting, events
except NameError:
    pass  # Some modules might not be imported depending on the import order

__all__ = [
    'ObservabilityManager',
    'get_observability',
    'setup_observability',
    'ObservabilityConfig',
    'observe',
    'trace_state',
    'ObservableContext',
    'ObservableAgent',
    'core',
    'config',
    'decorators',
    'context',
    'agent'
]