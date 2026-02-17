"""Import shim for StateMachineCore and AgentCore.

Tries to import the Rust extension first, falls back to pure Python.
"""

try:
    from puffinflow._rust_core import StateMachineCore  # type: ignore[import-not-found]
except ImportError:
    from ._fallback_core import StateMachineCore

try:
    from puffinflow._rust_core import AgentCore

    _HAS_AGENT_CORE = True
except ImportError:
    from ._fallback_agent_core import FallbackAgentCore as AgentCore

    _HAS_AGENT_CORE = True  # fallback is always available

__all__ = ["_HAS_AGENT_CORE", "AgentCore", "StateMachineCore"]
