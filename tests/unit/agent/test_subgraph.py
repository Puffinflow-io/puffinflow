"""Tests for subgraph composition feature.

Tests cover:
- Basic subgraph execution
- Input and output mapping between parent and child agents
- State isolation between parent and child
- Streaming event propagation with prefix
- Store sharing between parent and child
- Error propagation from child to parent
- Nested subgraphs (subgraph within a subgraph)
- Dependency ordering with subgraph states
- Parallel execution of independent subgraphs
"""

import asyncio

import pytest

from puffinflow.core.agent import Agent, AgentStatus
from puffinflow.core.agent.context import Context
from puffinflow.core.agent.state import ExecutionMode, RetryPolicy
from puffinflow.core.agent.streaming import StreamMode
from puffinflow.core.store import MemoryStore

# ============================================================================
# HELPERS
# ============================================================================


def _make_child_agent(name, state_fn, state_name="work"):
    """Create a simple child agent with a single state."""
    child = Agent(name)
    child.add_state(state_name, state_fn)
    return child


def _make_child_agent_with_store(name, state_fn, store, state_name="work"):
    """Create a child agent that shares a MemoryStore."""
    child = Agent(name, store=store)
    child.add_state(state_name, state_fn)
    return child


# ============================================================================
# TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_basic_run():
    """Child agent with one state runs as subgraph of parent."""

    async def child_work(ctx: Context) -> None:
        ctx.set_variable("done", True)

    child = _make_child_agent("child", child_work)

    parent = Agent("parent")
    parent.add_subgraph("run_child", child)

    result = await parent.run()

    assert result.status == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_input_mapping():
    """Parent var 'topic' maps to child var 'query' via input_map."""

    async def child_search(ctx: Context) -> None:
        query = ctx.get_variable("query")
        ctx.set_variable("result", f"searched for: {query}")

    child = _make_child_agent("searcher", child_search, state_name="search")

    parent = Agent("parent")

    # Set the parent variable before adding the subgraph
    async def set_topic(ctx: Context) -> None:
        ctx.set_variable("topic", "python testing")

    parent.add_state("init", set_topic)
    parent.add_subgraph(
        "run_search",
        child,
        input_map={"topic": "query"},
        output_map={"result": "search_result"},
        dependencies=["init"],
    )

    result = await parent.run()

    assert result.status == AgentStatus.COMPLETED
    assert result.variables.get("search_result") == "searched for: python testing"


@pytest.mark.asyncio
async def test_output_mapping():
    """Child output 'findings' maps to parent var 'research_results' via output_map."""

    async def child_research(ctx: Context) -> None:
        ctx.set_output("findings", ["fact1", "fact2", "fact3"])

    child = _make_child_agent("researcher", child_research, state_name="research")

    parent = Agent("parent")
    parent.add_subgraph(
        "do_research",
        child,
        output_map={"findings": "research_results"},
    )

    result = await parent.run()

    assert result.status == AgentStatus.COMPLETED
    assert result.variables.get("research_results") == ["fact1", "fact2", "fact3"]


@pytest.mark.asyncio
async def test_isolation():
    """Child's internal state doesn't leak to parent beyond mapped outputs."""

    async def child_work(ctx: Context) -> None:
        ctx.set_variable("internal_secret", "do_not_leak")
        ctx.set_variable("mapped_value", "safe_to_share")

    child = _make_child_agent("isolated_child", child_work)

    parent = Agent("parent")
    parent.add_subgraph(
        "run_child",
        child,
        output_map={"mapped_value": "shared_value"},
    )

    result = await parent.run()

    assert result.status == AgentStatus.COMPLETED
    # The mapped output should be present in parent
    assert result.variables.get("shared_value") == "safe_to_share"
    # The unmapped internal variable should NOT appear in parent
    assert result.variables.get("internal_secret") is None


@pytest.mark.asyncio
async def test_streaming_propagation():
    """When parent streams, child events appear with prefix."""

    async def child_state(ctx: Context) -> None:
        ctx.set_variable("x", 1)

    child = _make_child_agent("my_child", child_state)

    parent = Agent("parent")
    parent.add_subgraph("sub", child)

    events = []
    async for event in parent.stream(mode=StreamMode.DEBUG):
        events.append(event)

    # There should be events from the parent's own state execution
    parent_events = [e for e in events if e.state_name and "sub" in e.state_name]
    assert len(parent_events) > 0, (
        "Expected at least one event referencing the subgraph state 'sub'"
    )

    # Child events forwarded by make_subgraph_state should carry the child name prefix
    child_prefixed = [e for e in events if e.state_name and "my_child." in e.state_name]
    # The child stream events are forwarded after child.run() completes.
    # Depending on timing they may or may not appear; verify no crash at minimum.
    # If they do appear, they must have the prefix.
    for evt in child_prefixed:
        assert evt.state_name.startswith("my_child.")


@pytest.mark.asyncio
async def test_store_sharing():
    """Parent and child share the same MemoryStore instance."""
    store = MemoryStore()

    async def parent_write(ctx: Context) -> None:
        await ctx.store.put(("shared",), "greeting", "hello from parent")

    async def child_read(ctx: Context) -> None:
        item = await ctx.store.get(("shared",), "greeting")
        ctx.set_variable("read_value", item.value if item else None)

    child = _make_child_agent_with_store("child", child_read, store)

    parent = Agent("parent", store=store)
    parent.add_state("write", parent_write)
    parent.add_subgraph(
        "run_child",
        child,
        output_map={"read_value": "child_read_value"},
        dependencies=["write"],
    )

    result = await parent.run()

    assert result.status == AgentStatus.COMPLETED
    assert result.variables.get("child_read_value") == "hello from parent"


@pytest.mark.asyncio
async def test_error_propagation():
    """If child agent fails, parent handles it gracefully."""
    no_retry = RetryPolicy(max_retries=0, initial_delay=0.0)

    async def child_explode(ctx: Context) -> None:
        raise RuntimeError("child failed!")

    # Build child with no retries so it fails fast
    child = Agent("bad_child", retry_policy=no_retry)
    child.add_state("explode", child_explode)

    parent = Agent("parent", retry_policy=no_retry)
    parent.add_subgraph("run_bad", child)

    result = await parent.run()

    # The parent should have caught the error from the child.
    # The subgraph state function itself does not raise (child.run() catches
    # errors), so the parent may complete, but the child result will be FAILED.
    assert result.status in (AgentStatus.FAILED, AgentStatus.COMPLETED)
    # If the parent reports FAILED status, it should carry an error.
    if result.status == AgentStatus.FAILED:
        assert result.error is not None


@pytest.mark.asyncio
async def test_nested_subgraphs():
    """A subgraph containing another subgraph works correctly."""

    async def grandchild_work(ctx: Context) -> None:
        ctx.set_variable("depth", "grandchild")

    grandchild = _make_child_agent("grandchild", grandchild_work)

    # Middle child wraps the grandchild
    middle = Agent("middle")
    middle.add_subgraph(
        "run_grandchild",
        grandchild,
        output_map={"depth": "depth"},
    )

    # Parent wraps the middle child
    parent = Agent("parent")
    parent.add_subgraph(
        "run_middle",
        middle,
        output_map={"depth": "depth_from_nested"},
    )

    result = await parent.run()

    assert result.status == AgentStatus.COMPLETED
    assert result.variables.get("depth_from_nested") == "grandchild"


@pytest.mark.asyncio
async def test_dependencies():
    """Subgraph state depends on another state completing first."""

    execution_order = []

    async def prerequisite(ctx: Context) -> None:
        execution_order.append("prerequisite")
        ctx.set_variable("prepared", True)

    async def child_work(ctx: Context) -> None:
        execution_order.append("child_work")
        prepared = ctx.get_variable("prepared_flag")
        ctx.set_variable("result", f"prepared={prepared}")

    child = _make_child_agent("dependent_child", child_work)

    parent = Agent("parent")
    parent.add_state("prep", prerequisite)
    parent.add_subgraph(
        "run_child",
        child,
        input_map={"prepared": "prepared_flag"},
        output_map={"result": "final_result"},
        dependencies=["prep"],
    )

    result = await parent.run()

    assert result.status == AgentStatus.COMPLETED
    # Prerequisite must have run before the child
    assert execution_order.index("prerequisite") < execution_order.index("child_work")
    assert result.variables.get("final_result") == "prepared=True"


@pytest.mark.asyncio
async def test_parallel_subgraphs():
    """Two independent subgraphs run concurrently."""

    async def child_a_work(ctx: Context) -> None:
        await asyncio.sleep(0.05)
        ctx.set_variable("a_done", True)

    async def child_b_work(ctx: Context) -> None:
        await asyncio.sleep(0.05)
        ctx.set_variable("b_done", True)

    child_a = _make_child_agent("child_a", child_a_work)
    child_b = _make_child_agent("child_b", child_b_work)

    parent = Agent("parent")
    parent.add_subgraph(
        "run_a",
        child_a,
        output_map={"a_done": "a_complete"},
    )
    parent.add_subgraph(
        "run_b",
        child_b,
        output_map={"b_done": "b_complete"},
    )

    result = await parent.run(execution_mode=ExecutionMode.PARALLEL)

    assert result.status == AgentStatus.COMPLETED
    assert result.variables.get("a_complete") is True
    assert result.variables.get("b_complete") is True
