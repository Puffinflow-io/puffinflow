"""Command pattern for unified state updates and routing."""

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class Command:
    """Unified return type that combines state updates and routing.

    Instead of splitting logic between context.set_*() calls and return values,
    Command lets a state function return both data writes and routing decisions
    in a single value.

    Usage::

        @state
        async def decide(self, ctx):
            result = do_work()
            return Command(update={"result": result}, goto="next_step")
    """

    update: dict[str, Any] = field(default_factory=dict)
    goto: Union[str, list[str], None] = None


@dataclass
class Send:
    """Dynamic fan-out dispatch to a target state with a custom payload.

    Send dispatches different payloads to dynamically-created branches of the
    same state — enabling true map-reduce patterns.

    Usage::

        @state
        async def scatter(self, ctx):
            items = ctx.get_variable("items")
            return [Send("process", {"item": it}) for it in items]
    """

    state: str
    payload: dict[str, Any] = field(default_factory=dict)
