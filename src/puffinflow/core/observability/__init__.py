"""PuffinFlow Observability System"""

# Import submodules for import path tests
from . import agent, config, context, core, decorators
from .agent import ObservableAgent
from .config import ObservabilityConfig
from .context import ObservableContext
from .core import ObservabilityManager, get_observability, setup_observability
from .decorators import observe, trace_state

# Clean up indirect imports that might leak from submodules
try:
    del interfaces, tracing, metrics, alerting, events
except NameError:
    pass  # Some modules might not be imported depending on the import order

__all__ = [
    'ObservabilityConfig',
    'ObservabilityManager',
    'ObservableAgent',
    'ObservableContext',
    'agent',
    'config',
    'context',
    'core',
    'decorators',
    'get_observability',
    'observe',
    'setup_observability',
    'trace_state'
]
