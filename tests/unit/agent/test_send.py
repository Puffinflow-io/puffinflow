"""Tests for the Send API -- dynamic fan-out dispatch to target states.

Covers:
- Single and multiple Send branches
- Dynamic branch count driven by runtime context
- Reducer-based result collection from parallel branches
- Context isolation between Send branches
- Command integration (updates from branches applied to parent)
- Error handling in branches (return_exceptions semantics)
- Fan-in continuation after Send branches complete
"""

import asyncio

import pytest

from puffinflow.core.agent import Agent, AgentStatus, Command, Send
from puffinflow.core.agent.reducers import add_reducer

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# 1. Single Send -- one branch dispatched and completed
# ---------------------------------------------------------------------------


async def test_single_send():
    """Returning [Send('target', payload)] dispatches one branch that runs to completion."""
    agent = Agent("single-send")
    executed = []

    async def scatter(ctx):
        return [Send("target", {"x": 1})]

    async def target(ctx):
        executed.append(ctx.get_variable("x"))

    agent.add_state("scatter", scatter)
    agent.add_state("target", target)

    result = await agent.run()

    assert result.status == AgentStatus.COMPLETED
    assert executed == [1]


# ---------------------------------------------------------------------------
# 2. Multiple Sends -- all branches run in parallel
# ---------------------------------------------------------------------------


async def test_multiple_sends():
    """Multiple Send objects are dispatched and all run in parallel."""
    agent = Agent("multi-send")
    collected = []

    async def scatter(ctx):
        return [Send("target", {"x": i}) for i in range(3)]

    async def target(ctx):
        val = ctx.get_variable("x")
        collected.append(val)

    agent.add_state("scatter", scatter)
    agent.add_state("target", target)

    result = await agent.run()

    assert result.status == AgentStatus.COMPLETED
    assert sorted(collected) == [0, 1, 2]


# ---------------------------------------------------------------------------
# 3. Dynamic count -- number of Sends determined from a context variable
# ---------------------------------------------------------------------------


async def test_dynamic_count():
    """The number of Send branches is determined at runtime from a context variable."""
    agent = Agent("dynamic-send")
    collected = []

    async def scatter(ctx):
        items = ctx.get_variable("items")
        return [Send("process", {"item": it}) for it in items]

    async def process(ctx):
        collected.append(ctx.get_variable("item"))

    agent.add_state("scatter", scatter)
    agent.add_state("process", process)

    result = await agent.run(initial_context={"items": ["a", "b", "c", "d"]})

    assert result.status == AgentStatus.COMPLETED
    assert sorted(collected) == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# 4. Reducer collection -- branches return Command(update=...) merged via reducer
# ---------------------------------------------------------------------------


async def test_reducer_collection():
    """Send branches return Command(update={...}) and a registered reducer collects all results."""
    agent = Agent("reducer-send")
    agent.add_reducer("results", add_reducer)

    async def scatter(ctx):
        ctx.set_variable("results", [])
        return [Send("worker", {"val": i * 10}) for i in range(4)]

    async def worker(ctx):
        val = ctx.get_variable("val")
        return Command(update={"results": [val]})

    agent.add_state("scatter", scatter)
    agent.add_state("worker", worker)

    result = await agent.run()

    assert result.status == AgentStatus.COMPLETED
    results = result.get_variable("results")
    assert sorted(results) == [0, 10, 20, 30]


# ---------------------------------------------------------------------------
# 5. Isolated contexts -- each branch gets its own copy, no cross-talk
# ---------------------------------------------------------------------------


async def test_isolated_contexts():
    """Each Send branch receives an isolated context; mutations do not leak between branches."""
    agent = Agent("isolated-send")
    agent.add_reducer("results", add_reducer)

    async def scatter(ctx):
        ctx.set_variable("shared_marker", "original")
        ctx.set_variable("results", [])
        return [Send("worker", {"branch_id": i}) for i in range(3)]

    async def worker(ctx):
        branch_id = ctx.get_variable("branch_id")
        # Overwrite the shared marker -- must not affect other branches
        ctx.set_variable("shared_marker", f"modified_by_{branch_id}")
        # Small yield so other branches can interleave
        await asyncio.sleep(0.001)
        # Read it back -- should still be our own write
        marker = ctx.get_variable("shared_marker")
        return Command(update={"results": [marker]})

    agent.add_state("scatter", scatter)
    agent.add_state("worker", worker)

    result = await agent.run()

    assert result.status == AgentStatus.COMPLETED
    results = sorted(result.get_variable("results"))
    # Each branch should see only its own modification
    assert results == ["modified_by_0", "modified_by_1", "modified_by_2"]


# ---------------------------------------------------------------------------
# 6. Send with Command -- target returns Command whose updates apply to parent
# ---------------------------------------------------------------------------


async def test_send_with_command():
    """A Send target state that returns a Command has its updates applied to the parent context."""
    agent = Agent("send-command")

    async def scatter(ctx):
        return [Send("worker", {"label": "hello"})]

    async def worker(ctx):
        label = ctx.get_variable("label")
        return Command(update={"computed": f"{label}_world"})

    agent.add_state("scatter", scatter)
    agent.add_state("worker", worker)

    result = await agent.run()

    assert result.status == AgentStatus.COMPLETED
    assert result.get_variable("computed") == "hello_world"


# ---------------------------------------------------------------------------
# 7. Branch failure -- one branch raises, others still complete
# ---------------------------------------------------------------------------


async def test_branch_failure():
    """When one Send branch raises an exception the remaining branches still complete."""
    agent = Agent("failure-send")
    agent.add_reducer("results", add_reducer)

    async def scatter(ctx):
        ctx.set_variable("results", [])
        return [Send("worker", {"val": i}) for i in range(3)]

    async def worker(ctx):
        val = ctx.get_variable("val")
        if val == 1:
            raise RuntimeError("branch 1 failed")
        return Command(update={"results": [val]})

    agent.add_state("scatter", scatter)
    agent.add_state("worker", worker)

    result = await agent.run()

    # The agent itself should still complete (gather with return_exceptions)
    assert result.status == AgentStatus.COMPLETED
    # Only the non-failing branches contribute results
    collected = sorted(result.get_variable("results"))
    assert collected == [0, 2]


# ---------------------------------------------------------------------------
# 8. Fan-in after Sends -- routing continues to a subsequent state
# ---------------------------------------------------------------------------


async def test_fan_in_after_sends():
    """After Send branches complete, the flow continues to the next routed state."""
    agent = Agent("fanin-send")
    agent.add_reducer("results", add_reducer)
    execution_log = []

    async def scatter(ctx):
        ctx.set_variable("results", [])
        # Return both Send objects and a string routing target
        return [
            Send("worker", {"val": 1}),
            Send("worker", {"val": 2}),
            "merge",
        ]

    async def worker(ctx):
        val = ctx.get_variable("val")
        return Command(update={"results": [val]})

    async def merge(ctx):
        execution_log.append("merge_ran")
        collected = ctx.get_variable("results")
        ctx.set_variable("final_sum", sum(collected))

    agent.add_state("scatter", scatter)
    agent.add_state("worker", worker)
    agent.add_state("merge", merge)

    result = await agent.run()

    assert result.status == AgentStatus.COMPLETED
    assert "merge_ran" in execution_log
    assert result.get_variable("final_sum") == 3
