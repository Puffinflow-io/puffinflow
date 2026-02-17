"""Tests for SSE/WebSocket streaming endpoints and StreamEvent formatting."""

import json

from puffinflow.core.agent.streaming import StreamEvent, StreamMode


class TestStreamEventToSse:
    """Test StreamEvent.to_sse() formatting."""

    def test_basic_sse_format(self):
        event = StreamEvent(
            event_type="node_start",
            data={"state": "step_one"},
            state_name="step_one",
            timestamp=1000.0,
        )
        sse = event.to_sse()
        assert sse.startswith("event: node_start\n")
        assert "data: " in sse
        assert sse.endswith("\n\n")

    def test_sse_contains_valid_json_data(self):
        event = StreamEvent(
            event_type="node_complete",
            data={"result": "done"},
            state_name="step_two",
            timestamp=1000.0,
        )
        sse = event.to_sse()
        # Extract the data line
        lines = sse.strip().split("\n")
        data_line = next(line for line in lines if line.startswith("data: "))
        payload = json.loads(data_line[6:])  # Skip "data: "
        assert payload["data"]["result"] == "done"
        assert payload["state_name"] == "step_two"

    def test_sse_with_empty_data(self):
        event = StreamEvent(event_type="custom", timestamp=1000.0)
        sse = event.to_sse()
        assert "event: custom" in sse


class TestStreamEventToJson:
    """Test StreamEvent.to_json() formatting."""

    def test_returns_valid_json(self):
        event = StreamEvent(
            event_type="node_start",
            data={"state": "step_one"},
            state_name="step_one",
            timestamp=1000.0,
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "node_start"
        assert parsed["data"]["state"] == "step_one"
        assert parsed["state_name"] == "step_one"
        assert parsed["timestamp"] == 1000.0

    def test_json_with_none_state_name(self):
        event = StreamEvent(
            event_type="token",
            data={"token": "hello"},
            timestamp=1000.0,
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["state_name"] is None

    def test_json_with_complex_data(self):
        event = StreamEvent(
            event_type="custom",
            data={"nested": {"key": [1, 2, 3]}},
            state_name="step",
            timestamp=1000.0,
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["data"]["nested"]["key"] == [1, 2, 3]


class TestSSEStreamEndpoint:
    """Test SSEStreamEndpoint ASGI compatibility."""

    def test_import_and_create(self):
        from unittest.mock import MagicMock

        from puffinflow.core.agent.streaming_endpoint import SSEStreamEndpoint

        agent = MagicMock()
        endpoint = SSEStreamEndpoint(agent, mode=StreamMode.EVENTS)
        assert endpoint.agent is agent
        assert endpoint.mode == StreamMode.EVENTS

    def test_default_mode(self):
        from unittest.mock import MagicMock

        from puffinflow.core.agent.streaming_endpoint import SSEStreamEndpoint

        agent = MagicMock()
        endpoint = SSEStreamEndpoint(agent)
        assert endpoint.mode == StreamMode.EVENTS


class TestWebSocketStreamEndpoint:
    """Test WebSocketStreamEndpoint ASGI compatibility."""

    def test_import_and_create(self):
        from unittest.mock import MagicMock

        from puffinflow.core.agent.streaming_endpoint import WebSocketStreamEndpoint

        agent = MagicMock()
        endpoint = WebSocketStreamEndpoint(agent, mode=StreamMode.DEBUG)
        assert endpoint.agent is agent
        assert endpoint.mode == StreamMode.DEBUG


class TestConvenienceFunctions:
    """Test sse_response and websocket_handler convenience functions."""

    def test_sse_response_requires_starlette(self):
        from unittest.mock import MagicMock

        from puffinflow.core.agent.streaming_endpoint import sse_response

        agent = MagicMock()
        agent._stream_manager = None

        # Test that it either works (starlette installed) or raises ImportError
        try:
            response = sse_response(agent)
            # If starlette is installed, we get a response
            assert response is not None
        except ImportError:
            pass  # Expected if starlette is not installed
