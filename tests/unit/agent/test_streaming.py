"""Unit tests for the streaming feature of Agent.

Tests cover:
- node_start / node_complete events emitted during agent.stream()
- Backward compatibility of agent.run() without streaming
- Token emission via ctx.emit_token()
- Custom event emission via ctx.emit_event()
- StreamMode.UPDATES filtering behaviour
- Streaming with parallel state execution
- Proper stream closure on state errors
- Command return generating node_complete events
- Zero overhead when no stream manager is attached
"""

import asyncio

import pytest

from puffinflow.core.agent import Agent, AgentStatus, Command
from puffinflow.core.agent.streaming import StreamEvent, StreamManager, StreamMode
from puffinflow.core.agent.state import ExecutionMode, RetryPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_linear_agent(name: str = "test-stream"):
    """Create a simple two-state sequential agent: step_a -> step_b -> end."""
    agent = Agent(name)

    async def step_a(ctx):
        ctx.set_variable("visited_a", True)
        return "step_b"

    async def step_b(ctx):
        ctx.set_variable("visited_b", True)
        return None  # terminal

    agent.add_state("step_a", step_a)
    agent.add_state("step_b", step_b)
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStreaming:
    """Tests for the streaming feature."""

    async def test_node_events(self):
        """agent.stream() yields node_start and node_complete for each state."""
        agent = _make_linear_agent()

        events: list[StreamEvent] = []
        async for event in agent.stream(mode=StreamMode.EVENTS):
            events.append(event)

        event_types = [e.event_type for e in events]

        # Each state should produce a node_start followed by a node_complete.
        assert "node_start" in event_types
        assert "node_complete" in event_types

        # Specifically, both step_a and step_b should appear.
        node_start_states = [
            e.state_name for e in events if e.event_type == "node_start"
        ]
        node_complete_states = [
            e.state_name for e in events if e.event_type == "node_complete"
        ]
        assert "step_a" in node_start_states
        assert "step_b" in node_start_states
        assert "step_a" in node_complete_states
        assert "step_b" in node_complete_states

    async def test_run_still_works(self):
        """agent.run() works normally without streaming (backward compat)."""
        agent = _make_linear_agent("compat-agent")

        result = await agent.run()

        assert result.status == AgentStatus.COMPLETED
        assert result.get_variable("visited_a") is True
        assert result.get_variable("visited_b") is True
        # No stream manager should be lingering after a plain run.
        assert agent._stream_manager is None

    async def test_token_emit(self):
        """ctx.emit_token('tok') in a state emits a token event through stream."""
        agent = Agent("token-agent")

        async def emitter(ctx):
            ctx.emit_token("tok")
            return None

        agent.add_state("emitter", emitter)

        events: list[StreamEvent] = []
        async for event in agent.stream(mode=StreamMode.EVENTS):
            events.append(event)

        token_events = [e for e in events if e.event_type == "token"]
        assert len(token_events) >= 1
        assert token_events[0].data["token"] == "tok"

    async def test_custom_events(self):
        """ctx.emit_event('progress', {'pct': 50}) emits a custom event."""
        agent = Agent("custom-event-agent")

        async def reporter(ctx):
            ctx.emit_event("progress", {"pct": 50})
            return None

        agent.add_state("reporter", reporter)

        events: list[StreamEvent] = []
        async for event in agent.stream(mode=StreamMode.EVENTS):
            events.append(event)

        custom_events = [e for e in events if e.event_type == "custom"]
        assert len(custom_events) >= 1
        assert custom_events[0].data["name"] == "progress"
        assert custom_events[0].data["payload"] == {"pct": 50}

    async def test_stream_mode_updates(self):
        """StreamMode.UPDATES only yields node_complete and custom events."""
        agent = Agent("updates-mode-agent")

        async def state_with_custom(ctx):
            ctx.emit_token("ignored-token")
            ctx.emit_event("kept", {"x": 1})
            return None

        agent.add_state("state_with_custom", state_with_custom)

        events: list[StreamEvent] = []
        async for event in agent.stream(mode=StreamMode.UPDATES):
            events.append(event)

        event_types = {e.event_type for e in events}
        # node_start and token events should be filtered out.
        assert "node_start" not in event_types
        assert "token" not in event_types
        # node_complete and custom events should pass through.
        assert "node_complete" in event_types
        assert "custom" in event_types

    async def test_parallel_states(self):
        """Streaming works with parallel state execution."""
        agent = Agent("parallel-stream-agent")
        execution_log: list[str] = []

        async def branch_a(ctx):
            execution_log.append("branch_a")
            ctx.set_variable("a_done", True)
            return None

        async def branch_b(ctx):
            execution_log.append("branch_b")
            ctx.set_variable("b_done", True)
            return None

        agent.add_state("branch_a", branch_a)
        agent.add_state("branch_b", branch_b)

        events: list[StreamEvent] = []
        async for event in agent.stream(
            mode=StreamMode.EVENTS,
            execution_mode=ExecutionMode.PARALLEL,
        ):
            events.append(event)

        node_complete_states = [
            e.state_name for e in events if e.event_type == "node_complete"
        ]
        assert "branch_a" in node_complete_states
        assert "branch_b" in node_complete_states

    async def test_error_in_state(self):
        """Stream still closes properly if a state raises."""
        agent = Agent(
            "error-stream-agent",
            retry_policy=RetryPolicy(max_retries=0, initial_delay=0.0),
        )

        async def failing_state(ctx):
            raise RuntimeError("boom")

        agent.add_state("failing_state", failing_state)

        events: list[StreamEvent] = []
        # The stream should terminate without hanging even though the state
        # raised an exception.  We just collect whatever events come out.
        async for event in agent.stream(mode=StreamMode.EVENTS):
            events.append(event)

        # The stream manager should have been cleaned up.
        assert agent._stream_manager is None

    async def test_command_events(self):
        """Command return also generates a node_complete event."""
        agent = Agent("command-agent")

        async def cmd_state(ctx):
            return Command(update={"answer": 42}, goto="final")

        async def final_state(ctx):
            return None

        agent.add_state("cmd_state", cmd_state)
        agent.add_state("final_state", final_state, dependencies=["cmd_state"])

        events: list[StreamEvent] = []
        async for event in agent.stream(mode=StreamMode.EVENTS):
            events.append(event)

        # cmd_state should have a node_complete event emitted.
        cmd_completes = [
            e
            for e in events
            if e.event_type == "node_complete" and e.state_name == "cmd_state"
        ]
        assert len(cmd_completes) >= 1

    async def test_no_overhead_check(self):
        """When no stream manager, agent._stream_manager is None (zero overhead)."""
        agent = Agent("no-overhead-agent")

        async def noop(ctx):
            return None

        agent.add_state("noop", noop)

        # Before any run, stream manager should be None.
        assert agent._stream_manager is None

        # After a plain run(), stream manager should still be None.
        await agent.run()
        assert agent._stream_manager is None
