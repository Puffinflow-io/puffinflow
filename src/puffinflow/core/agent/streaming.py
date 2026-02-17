"""Streaming support for real-time event delivery."""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class StreamMode(str, Enum):
    """Stream filtering modes."""

    UPDATES = "updates"  # state outputs after each step
    EVENTS = "events"  # all events including custom
    DEBUG = "debug"  # everything


@dataclass(frozen=True)
class StreamEvent:
    """A single event emitted during agent execution."""

    event_type: str  # "node_start", "node_complete", "token", "custom", etc.
    data: dict[str, Any] = field(default_factory=dict)
    state_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """Format as Server-Sent Events string."""
        payload = {"data": self.data, "state_name": self.state_name}
        return f"event: {self.event_type}\ndata: {json.dumps(payload, default=str)}\n\n"

    def to_json(self) -> str:
        """Format as JSON string."""
        return json.dumps(
            {
                "event_type": self.event_type,
                "data": self.data,
                "state_name": self.state_name,
                "timestamp": self.timestamp,
            },
            default=str,
        )


class StreamManager:
    """Manages an async queue of StreamEvents for real-time delivery."""

    __slots__ = ("_queue", "_mode", "_closed")

    def __init__(self, mode: StreamMode = StreamMode.EVENTS) -> None:
        self._queue: asyncio.Queue[Optional[StreamEvent]] = asyncio.Queue()
        self._mode = mode
        self._closed = False

    @property
    def mode(self) -> StreamMode:
        return self._mode

    def emit(self, event: StreamEvent) -> None:
        """Non-blocking put of an event onto the queue."""
        if self._closed:
            return
        # Filter by mode
        if self._mode == StreamMode.UPDATES and event.event_type not in (
            "node_complete",
            "custom",
        ):
            return
        self._queue.put_nowait(event)

    def emit_node_start(self, state_name: str) -> None:
        """Emit a node_start event."""
        self.emit(
            StreamEvent(
                event_type="node_start",
                state_name=state_name,
                data={"state": state_name},
            )
        )

    def emit_node_complete(self, state_name: str, result: Any = None) -> None:
        """Emit a node_complete event."""
        self.emit(
            StreamEvent(
                event_type="node_complete",
                state_name=state_name,
                data={"state": state_name, "result": result},
            )
        )

    def emit_token(self, state_name: str, token: str) -> None:
        """Emit a token event (for LLM streaming)."""
        self.emit(
            StreamEvent(
                event_type="token",
                state_name=state_name,
                data={"token": token},
            )
        )

    def emit_custom(self, state_name: str, name: str, data: Any) -> None:
        """Emit a custom event."""
        self.emit(
            StreamEvent(
                event_type="custom",
                state_name=state_name,
                data={"name": name, "payload": data},
            )
        )

    def close(self) -> None:
        """Signal end of stream with a sentinel None."""
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(None)

    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """Yield events until the stream is closed."""
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event
