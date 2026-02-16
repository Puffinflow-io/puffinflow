"""
Test coverage for the State Reducers feature.

Tests cover:
- Built-in reducer functions (add_reducer, append_reducer, replace_reducer)
- ReducerRegistry class
- Agent.add_reducer() integration with Command.update
- Parallel state merging with reducers
- Context.set_reduced() reducer-aware writes
- Default last-write-wins behavior without reducers
"""


import pytest

from puffinflow.core.agent.base import Agent
from puffinflow.core.agent.command import Command
from puffinflow.core.agent.context import Context
from puffinflow.core.agent.reducers import (
    ReducerRegistry,
    add_reducer,
    append_reducer,
    replace_reducer,
)
from puffinflow.core.agent.state import AgentStatus, ExecutionMode

# ============================================================================
# UNIT TESTS FOR BUILT-IN REDUCERS
# ============================================================================


class TestBuiltInReducers:
    """Unit tests for the standalone reducer functions."""

    def test_add_reducer_lists(self):
        """add_reducer concatenates two lists."""
        existing = [1, 2, 3]
        new = [4, 5, 6]
        result = add_reducer(existing, new)
        assert result == [1, 2, 3, 4, 5, 6]

    def test_add_reducer_numbers(self):
        """add_reducer adds two numbers together."""
        assert add_reducer(10, 5) == 15
        assert add_reducer(3.5, 1.5) == 5.0
        assert add_reducer(10, 2.5) == 12.5

    def test_replace_reducer(self):
        """replace_reducer always returns the new value (last-write-wins)."""
        assert replace_reducer("old", "new") == "new"
        assert replace_reducer([1, 2], [3, 4]) == [3, 4]
        assert replace_reducer(100, 200) == 200
        assert replace_reducer({"a": 1}, {"b": 2}) == {"b": 2}

    def test_append_reducer(self):
        """append_reducer appends the new value to a list."""
        # Append a scalar to an existing list
        result = append_reducer([1, 2, 3], 4)
        assert result == [1, 2, 3, 4]

        # Append a list to an existing list (concatenation)
        result = append_reducer([1, 2], [3, 4])
        assert result == [1, 2, 3, 4]

        # Append to a non-list existing value wraps it first
        result = append_reducer("existing", "new")
        assert result == ["existing", "new"]

        # Append when existing is None starts fresh
        result = append_reducer(None, "first")
        assert result == ["first"]

    def test_add_reducer_dict_merge(self):
        """add_reducer merges two dicts (new keys override existing)."""
        existing = {"a": 1, "b": 2}
        new = {"b": 99, "c": 3}
        result = add_reducer(existing, new)
        assert result == {"a": 1, "b": 99, "c": 3}
        # Original dict must not be mutated
        assert existing == {"a": 1, "b": 2}


# ============================================================================
# CUSTOM REDUCER TEST
# ============================================================================


class TestCustomReducer:
    """Test using a custom lambda reducer."""

    def test_custom_reducer(self):
        """A custom lambda reducer is applied correctly via the registry."""
        registry = ReducerRegistry()
        # Custom reducer: multiply existing by new
        registry.register("score", lambda existing, new: existing * new)

        result = registry.apply("score", 5, 3)
        assert result == 15

        # Key without a reducer falls back to replacement
        result = registry.apply("other_key", "old", "new")
        assert result == "new"


# ============================================================================
# AGENT INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
class TestAgentReducerIntegration:
    """Tests for Agent.add_reducer() and Command-based reducer application."""

    async def test_command_with_reducer(self):
        """Command.update uses the registered reducer when the agent runs."""
        agent = Agent("cmd-reducer-test")

        async def init_state(context):
            context.shared_state["items"] = [1, 2]
            return "accumulate"

        async def accumulate_state(context):
            # Return a Command whose update triggers the add_reducer
            return Command(update={"items": [3, 4]}, goto=None)

        agent.add_state("init", init_state)
        agent.add_state("accumulate", accumulate_state, dependencies=["init"])
        agent.add_reducer("items", add_reducer)

        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)

        assert result.status == AgentStatus.COMPLETED
        # The reducer should have concatenated [1,2] + [3,4]
        assert result.get_variable("items") == [1, 2, 3, 4]

    async def test_parallel_states_with_reducer(self):
        """Two parallel states write the same key; the reducer merges correctly."""
        agent = Agent("parallel-reducer-test")

        async def state_a(context):
            return Command(update={"total": 10})

        async def state_b(context):
            return Command(update={"total": 20})

        async def gather(context):
            # Just read the merged value, nothing to return
            pass

        agent.add_state("state_a", state_a)
        agent.add_state("state_b", state_b)
        agent.add_state("gather", gather, dependencies=["state_a", "state_b"])
        agent.add_reducer("total", add_reducer)

        result = await agent.run(execution_mode=ExecutionMode.PARALLEL)

        assert result.status == AgentStatus.COMPLETED
        # add_reducer should sum the two numbers: 10 + 20 = 30
        # (the first write goes through as-is because existing is None/0,
        #  so the reducer fallback replaces; the second adds)
        total = result.get_variable("total")
        # Both states ran; final value depends on ordering:
        #   First write: reducer(None, 10) -> fallback -> 10
        #   Second write: reducer(10, 20) -> 30
        assert total == 30

    async def test_no_reducer_default(self):
        """Without a reducer, last-write-wins (direct replacement)."""
        agent = Agent("no-reducer-test")

        async def state_a(context):
            return Command(update={"value": "first"}, goto="state_b")

        async def state_b(context):
            return Command(update={"value": "second"}, goto=None)

        agent.add_state("state_a", state_a)
        agent.add_state("state_b", state_b, dependencies=["state_a"])

        # No reducer registered for "value"
        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)

        assert result.status == AgentStatus.COMPLETED
        # Last write wins
        assert result.get_variable("value") == "second"

    async def test_context_set_reduced(self):
        """Context.set_reduced() applies the reducer when one is registered."""
        registry = ReducerRegistry()
        registry.register("scores", add_reducer)

        ctx = Context(shared_state={"scores": [10, 20]})
        ctx._reducers = registry

        ctx.set_reduced("scores", [30, 40])
        assert ctx.shared_state["scores"] == [10, 20, 30, 40]

        # Without a reducer for a different key, set_reduced does a direct write
        ctx.set_reduced("other", "hello")
        assert ctx.shared_state["other"] == "hello"
