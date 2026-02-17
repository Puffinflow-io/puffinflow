"""WebSocket/SSE streaming endpoint helpers (ASGI-compatible)."""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .base import Agent

from .streaming import StreamEvent, StreamManager, StreamMode

logger = logging.getLogger(__name__)


class SSEStreamEndpoint:
    """Server-Sent Events endpoint for agent monitoring.

    ASGI-compatible — can be mounted directly in any ASGI app
    (Starlette, FastAPI, etc.).
    """

    def __init__(
        self,
        agent: "Agent",
        mode: StreamMode = StreamMode.EVENTS,
    ) -> None:
        self.agent = agent
        self.mode = mode

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        """ASGI interface for SSE streaming."""
        if scope["type"] != "http":
            return

        # Send SSE headers
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    [b"content-type", b"text/event-stream"],
                    [b"cache-control", b"no-cache"],
                    [b"connection", b"keep-alive"],
                    [b"x-accel-buffering", b"no"],
                ],
            }
        )

        # Get or create stream manager
        stream = self.agent._stream_manager
        if stream is None:
            stream = StreamManager(self.mode)
            self.agent._stream_manager = stream

        # Stream events
        try:
            async for event in stream:
                sse_data = event.to_sse().encode("utf-8")
                await send(
                    {
                        "type": "http.response.body",
                        "body": sse_data,
                        "more_body": True,
                    }
                )
        except (asyncio.CancelledError, ConnectionError):
            pass
        finally:
            await send(
                {
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False,
                }
            )

    def as_starlette_route(self) -> Any:
        """Return a Starlette Route for this endpoint."""
        try:
            from starlette.routing import Route  # type: ignore[import-not-found]

            return Route("/stream", endpoint=self)
        except ImportError:
            raise ImportError(
                "starlette is required for as_starlette_route(). "
                "Install with: pip install starlette"
            )

    def as_fastapi_route(self) -> Any:
        """Return a FastAPI APIRoute for this endpoint."""
        try:
            from fastapi.routing import APIRoute  # type: ignore[import-not-found]

            return APIRoute("/stream", endpoint=self)
        except ImportError:
            raise ImportError(
                "fastapi is required for as_fastapi_route(). "
                "Install with: pip install fastapi"
            )


class WebSocketStreamEndpoint:
    """WebSocket endpoint for bidirectional agent monitoring.

    ASGI-compatible — accepts WebSocket connections and streams
    agent events as JSON messages. Also accepts control messages
    (pause, resume, cancel).
    """

    def __init__(
        self,
        agent: "Agent",
        mode: StreamMode = StreamMode.EVENTS,
    ) -> None:
        self.agent = agent
        self.mode = mode

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
    ) -> None:
        """ASGI WebSocket interface."""
        if scope["type"] != "websocket":
            return

        # Accept WebSocket connection
        await send({"type": "websocket.accept"})

        # Get or create stream manager
        stream = self.agent._stream_manager
        if stream is None:
            stream = StreamManager(self.mode)
            self.agent._stream_manager = stream

        # Run send and receive concurrently
        send_task = asyncio.create_task(self._send_events(stream, send))
        receive_task = asyncio.create_task(self._receive_controls(receive))

        try:
            done, pending = await asyncio.wait(
                {send_task, receive_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        except (asyncio.CancelledError, ConnectionError):
            pass
        finally:
            await send({"type": "websocket.close", "code": 1000})

    async def _send_events(self, stream: StreamManager, send: Any) -> None:
        """Stream events to WebSocket client."""
        try:
            async for event in stream:
                message = event.to_json()
                await send(
                    {
                        "type": "websocket.send",
                        "text": message,
                    }
                )
        except (asyncio.CancelledError, ConnectionError):
            pass

    async def _receive_controls(self, receive: Any) -> None:
        """Receive control messages from WebSocket client."""
        from .state import AgentStatus

        try:
            while True:
                message = await receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message["type"] == "websocket.receive":
                    text = message.get("text", "")
                    try:
                        control = json.loads(text)
                        action = control.get("action")
                        if action == "pause":
                            self.agent.status = AgentStatus.PAUSED
                            logger.info("Agent %s paused via WebSocket", self.agent.name)
                        elif action == "resume":
                            self.agent.status = AgentStatus.RUNNING
                            logger.info(
                                "Agent %s resumed via WebSocket", self.agent.name
                            )
                        elif action == "cancel":
                            self.agent.status = AgentStatus.FAILED
                            logger.info(
                                "Agent %s cancelled via WebSocket", self.agent.name
                            )
                    except (json.JSONDecodeError, KeyError):
                        logger.warning("Invalid WebSocket control message: %s", text)
        except (asyncio.CancelledError, ConnectionError):
            pass


def sse_response(
    agent: "Agent", mode: StreamMode = StreamMode.EVENTS
) -> Any:  # Returns StreamingResponse
    """Return a Starlette StreamingResponse for SSE.

    Usage with Starlette/FastAPI:
        @app.get("/stream")
        async def stream():
            return sse_response(my_agent)
    """
    try:
        from starlette.responses import StreamingResponse  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "starlette is required for sse_response(). "
            "Install with: pip install starlette"
        )

    stream = agent._stream_manager
    if stream is None:
        stream = StreamManager(mode)
        agent._stream_manager = stream

    async def _generate() -> Any:
        async for event in stream:
            yield event.to_sse()

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def websocket_handler(
    agent: "Agent",
    websocket: Any,
    mode: StreamMode = StreamMode.EVENTS,
) -> None:
    """FastAPI WebSocket handler for agent streaming.

    Usage with FastAPI:
        @app.websocket("/ws")
        async def ws(websocket: WebSocket):
            await websocket.accept()
            await websocket_handler(my_agent, websocket)
    """
    stream = agent._stream_manager
    if stream is None:
        stream = StreamManager(mode)
        agent._stream_manager = stream

    async def _send_events() -> None:
        async for event in stream:
            await websocket.send_text(event.to_json())

    async def _receive_controls() -> None:
        from .state import AgentStatus

        try:
            while True:
                text = await websocket.receive_text()
                try:
                    control = json.loads(text)
                    action = control.get("action")
                    if action == "pause":
                        agent.status = AgentStatus.PAUSED
                    elif action == "resume":
                        agent.status = AgentStatus.RUNNING
                    elif action == "cancel":
                        agent.status = AgentStatus.FAILED
                except (json.JSONDecodeError, KeyError):
                    pass
        except Exception:
            pass

    send_task = asyncio.create_task(_send_events())
    receive_task = asyncio.create_task(_receive_controls())

    try:
        done, pending = await asyncio.wait(
            {send_task, receive_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    except (asyncio.CancelledError, ConnectionError):
        pass
