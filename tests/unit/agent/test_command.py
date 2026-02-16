"""Tests for the Command pattern in agent state routing.

Tests cover:
- Command with update only (no goto) acts as terminal
- Command with goto as a string routes to the next state
- Command with goto as a list routes to multiple states
- update dict is applied before routing happens
- Backward compatibility: returning a plain string still works
- Empty Command() is equivalent to returning None
- Mixed returns: one state returns Command, another returns string
"""

import pytest

from puffinflow.core.agent import Agent, AgentStatus, Command, Send


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(name: str = "cmd_test") -> Agent:
    """Create a minimal Agent with no retry overhead."""
    return Agent(name=name, max_concurrent=4)


# ---------------------------------------------------------------------------
# 1. Command with update but no goto (None) acts as terminal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_command_update_only():
    """Returning Command(update={...}) without goto should terminate the agent
    after applying the update to shared_state."""
    agent = _make_agent()

    async def start(ctx):
        return Command(update={"result": 42})

    agent.add_state("start", start)

    result = await agent.run()

    assert result.status == AgentStatus.COMPLETED
    assert result.variables.get("result") == 42


# ---------------------------------------------------------------------------
# 2. Command with goto as a string routes to next state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_command_goto_string():
    """Command(goto='next') should route execution to the named state."""
    agent = _make_agent()

    async def start(ctx):
        return Command(update={"step": 1}, goto="finish")

    async def finish(ctx):
        ctx.set_variable("step", 2)
        return None

    agent.add_state("start", start)
    agent.add_state("finish", finish)

    result = await agent.run()

    assert result.status == AgentStatus.COMPLETED
    assert result.variables.get("step") == 2


# ---------------------------------------------------------------------------
# 3. Command with goto as list routes to multiple states
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_command_goto_list():
    """Command(goto=['a', 'b']) should enqueue both target states."""
    agent = _make_agent()

    async def start(ctx):
        return Command(update={"started": True}, goto=["branch_a", "branch_b"])

    async def branch_a(ctx):
        ctx.set_variable("a_done", True)
        return None

    async def branch_b(ctx):
        ctx.set_variable("b_done", True)
        return None

    agent.add_state("start", start)
    agent.add_state("branch_a", branch_a)
    agent.add_state("branch_b", branch_b)

    result = await agent.run()

    assert result.status == AgentStatus.COMPLETED
    assert result.variables.get("started") is True
    assert result.variables.get("a_done") is True
    assert result.variables.get("b_done") is True


# ---------------------------------------------------------------------------
# 4. Verify update is applied before routing happens
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_command_update_before_routing():
    """The update dict must be written to shared_state before the goto state
    executes, so the next state can read the updated value."""
    agent = _make_agent()

    async def writer(ctx):
        return Command(update={"message": "hello"}, goto="reader")

    async def reader(ctx):
        # The value set by the Command update should already be visible
        value = ctx.get_variable("message")
        ctx.set_variable("read_value", value)
        return None

    agent.add_state("writer", writer)
    agent.add_state("reader", reader)

    result = await agent.run()

    assert result.status == AgentStatus.COMPLETED
    assert result.variables.get("read_value") == "hello"


# ---------------------------------------------------------------------------
# 5. Backward compatibility: returning a plain string still works
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backward_compat_string_return():
    """Returning a plain string from a state function should still route to
    the named next state, preserving pre-Command behaviour."""
    agent = _make_agent()

    async def first(ctx):
        ctx.set_variable("visited_first", True)
        return "second"

    async def second(ctx):
        ctx.set_variable("visited_second", True)
        return None

    agent.add_state("first", first)
    agent.add_state("second", second)

    result = await agent.run()

    assert result.status == AgentStatus.COMPLETED
    assert result.variables.get("visited_first") is True
    assert result.variables.get("visited_second") is True


# ---------------------------------------------------------------------------
# 6. Empty Command() with no args is equivalent to returning None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_command():
    """Command() with default arguments (empty update, goto=None) should
    behave identically to returning None -- the agent terminates normally."""
    agent = _make_agent()

    async def only_state(ctx):
        ctx.set_variable("ran", True)
        return Command()

    agent.add_state("only_state", only_state)

    result = await agent.run()

    assert result.status == AgentStatus.COMPLETED
    assert result.variables.get("ran") is True


# ---------------------------------------------------------------------------
# 7. Mixed returns: one state returns Command, another returns string
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_returns():
    """An agent where one state returns a Command and another returns a plain
    string should execute the full chain correctly."""
    agent = _make_agent()

    async def step_one(ctx):
        # Use the Command pattern to set data and route
        return Command(update={"count": 1}, goto="step_two")

    async def step_two(ctx):
        # Use the legacy string-return pattern to route
        current = ctx.get_variable("count")
        ctx.set_variable("count", current + 1)
        return "step_three"

    async def step_three(ctx):
        current = ctx.get_variable("count")
        ctx.set_variable("count", current + 1)
        return None

    agent.add_state("step_one", step_one)
    agent.add_state("step_two", step_two)
    agent.add_state("step_three", step_three)

    result = await agent.run()

    assert result.status == AgentStatus.COMPLETED
    assert result.variables.get("count") == 3
