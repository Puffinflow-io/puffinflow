"""Subgraph composition for modular agent pipelines."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from .base import Agent
    from .streaming import StreamManager


@dataclass
class StateMapping:
    """Defines how parent state maps to/from a child agent's state."""

    inputs: dict[str, str] = field(default_factory=dict)  # parent_key -> child_key
    outputs: dict[str, str] = field(default_factory=dict)  # child_key -> parent_key


def make_subgraph_state(
    child_agent: "Agent",
    state_map: StateMapping,
    stream_manager: Optional["StreamManager"] = None,
    store: Any = None,
) -> Callable:
    """Return an async state function that runs child_agent.run() internally.

    The function:
    1. Copies parent context vars via input_map → child shared_state
    2. Runs child.run()
    3. Maps child outputs back to parent context via output_map
    """

    async def _subgraph_state(ctx: Any) -> None:
        from .streaming import StreamEvent

        # Build child initial context from parent variables via input map
        child_initial: dict[str, Any] = {}
        for parent_key, child_key in state_map.inputs.items():
            val = ctx.get_variable(parent_key)
            if val is not None:
                child_initial[child_key] = val

        # Propagate store to child if available
        if store is not None and hasattr(child_agent, "_store"):
            child_agent._store = store

        # Propagate stream manager — prefix child events with subgraph name
        child_stream = None
        if stream_manager is not None:
            from .streaming import StreamManager as _SM

            child_stream = _SM(mode=stream_manager.mode)
            child_agent._stream_manager = child_stream

        # Run the child agent
        child_result = await child_agent.run(initial_context=child_initial or None)

        # Forward child stream events with prefix
        if child_stream is not None and stream_manager is not None:
            prefix = child_agent.name
            # Drain any remaining events
            child_stream.close()
            while not child_stream._queue.empty():
                evt = child_stream._queue.get_nowait()
                if evt is not None:
                    prefixed = StreamEvent(
                        event_type=evt.event_type,
                        state_name=f"{prefix}.{evt.state_name}"
                        if evt.state_name
                        else prefix,
                        data=evt.data,
                        timestamp=evt.timestamp,
                    )
                    stream_manager.emit(prefixed)

        # Map child outputs back to parent context via output map
        if child_result is not None:
            child_vars = child_result.variables or {}
            child_outputs = child_result.outputs or {}
            # Merge both variables and outputs from child
            all_child = {**child_vars, **child_outputs}

            for child_key, parent_key in state_map.outputs.items():
                if child_key in all_child:
                    ctx.set_variable(parent_key, all_child[child_key])

        return None

    _subgraph_state.__qualname__ = f"subgraph_{child_agent.name}"
    return _subgraph_state
