"""Tests for class-level @state auto-discovery via __init_subclass__."""

import pytest

from puffinflow.core.agent.base import Agent
from puffinflow.core.agent.decorators.flexible import cpu_intensive, state
from puffinflow.core.agent.state import AgentStatus


# ---------------------------------------------------------------------------
# Helper agent classes defined at module level so __init_subclass__ runs
# ---------------------------------------------------------------------------


class TwoStateAgent(Agent):
    """Simple agent with two auto-discovered states, no deps."""

    @state
    async def first(self, ctx):
        return "second"

    @state
    async def second(self, ctx):
        return None


class DepsAgent(Agent):
    """Agent with depends_on between states."""

    @state
    async def validate(self, ctx):
        return "charge"

    @state(depends_on=["validate"])
    async def charge(self, ctx):
        return None


class ManualAgent(Agent):
    """Agent that uses only manual add_state (no decorators)."""

    def __init__(self, name: str):
        super().__init__(name)

        async def step_a(ctx):
            return "step_b"

        async def step_b(ctx):
            return None

        self.add_state("step_a", step_a)
        self.add_state("step_b", step_b)


class MixedAgent(Agent):
    """Some states manual, some auto-discovered."""

    def __init__(self, name: str):
        super().__init__(name)

        async def manual_start(ctx):
            return "auto_end"

        self.add_state("manual_start", manual_start)

    @state
    async def auto_end(self, ctx):
        return None


class ManualOverrideAgent(Agent):
    """Manual registration of a decorated method prevents double-registration."""

    def __init__(self, name: str):
        super().__init__(name)
        self.add_state("do_work", self.do_work)

    @state
    async def do_work(self, ctx):
        return None


class ParentAgent(Agent):
    """Base agent with one state."""

    @state
    async def parent_state(self, ctx):
        ctx.set_variable("parent_ran", True)
        return None


class ChildInheritsAgent(ParentAgent):
    """Child that inherits parent state and adds its own."""

    @state
    async def child_state(self, ctx):
        return "parent_state"


class ChildOverridesAgent(ParentAgent):
    """Child that overrides parent state."""

    @state
    async def parent_state(self, ctx):
        ctx.set_variable("child_override_ran", True)
        return None


class ProfileAgent(Agent):
    """Agent using profile decorators like @cpu_intensive."""

    @cpu_intensive(cpu=4.0, memory=1024.0)
    async def heavy_work(self, ctx):
        return None


class EmptySubclass(Agent):
    """Subclass with no states at all."""

    pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAutoDiscoverySimple:
    """Basic auto-discovery of @state-decorated methods."""

    @pytest.mark.asyncio
    async def test_two_states_discovered(self):
        agent = TwoStateAgent("test")
        await agent.run()
        assert agent.status == AgentStatus.COMPLETED
        assert "first" in agent.states
        assert "second" in agent.states

    @pytest.mark.asyncio
    async def test_class_attribute_populated(self):
        """__init_subclass__ populates _puffinflow_auto_states."""
        names = {sn for sn, _ in TwoStateAgent._puffinflow_auto_states}
        assert names == {"first", "second"}


class TestAutoDiscoveryWithDeps:
    """Auto-discovery respects depends_on and topological ordering."""

    @pytest.mark.asyncio
    async def test_deps_agent_runs(self):
        agent = DepsAgent("deps")
        await agent.run()
        assert agent.status == AgentStatus.COMPLETED
        assert "validate" in agent.states
        assert "charge" in agent.states

    @pytest.mark.asyncio
    async def test_deps_registered_correctly(self):
        agent = DepsAgent("deps")
        # Trigger auto-discovery
        agent._auto_discover_states()
        assert agent.dependencies.get("charge") == ["validate"]


class TestBackwardCompat:
    """Manual add_state() still works on bare Agent."""

    @pytest.mark.asyncio
    async def test_manual_agent_runs(self):
        agent = ManualAgent("manual")
        await agent.run()
        assert agent.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_bare_agent_no_auto_states(self):
        agent = Agent("bare")
        assert Agent._puffinflow_auto_states == ()


class TestMixed:
    """Some states manual, some auto-discovered."""

    @pytest.mark.asyncio
    async def test_mixed_agent_runs(self):
        agent = MixedAgent("mixed")
        await agent.run()
        assert agent.status == AgentStatus.COMPLETED
        assert "manual_start" in agent.states
        assert "auto_end" in agent.states


class TestManualPreventsDoubleRegistration:
    """If a state is manually registered in __init__, auto-discovery skips it."""

    @pytest.mark.asyncio
    async def test_no_double_registration(self):
        agent = ManualOverrideAgent("override")
        await agent.run()
        assert agent.status == AgentStatus.COMPLETED
        assert "do_work" in agent.states


class TestInheritance:
    """Inheritance of @state methods across class hierarchies."""

    @pytest.mark.asyncio
    async def test_child_inherits_parent_states(self):
        agent = ChildInheritsAgent("child")
        await agent.run()
        assert agent.status == AgentStatus.COMPLETED
        assert "parent_state" in agent.states
        assert "child_state" in agent.states

    @pytest.mark.asyncio
    async def test_child_overrides_parent_state(self):
        agent = ChildOverridesAgent("override")
        await agent.run()
        assert agent.status == AgentStatus.COMPLETED
        assert agent._context.get_variable("child_override_ran") is True

    @pytest.mark.asyncio
    async def test_child_adds_new_states(self):
        names = {sn for sn, _ in ChildInheritsAgent._puffinflow_auto_states}
        assert "parent_state" in names
        assert "child_state" in names


class TestProfileDecorators:
    """Profile decorators like @cpu_intensive are auto-discovered."""

    @pytest.mark.asyncio
    async def test_profile_agent_discovered(self):
        agent = ProfileAgent("profile")
        agent._auto_discover_states()
        assert "heavy_work" in agent.states

    @pytest.mark.asyncio
    async def test_profile_agent_runs(self):
        agent = ProfileAgent("profile")
        await agent.run()
        assert agent.status == AgentStatus.COMPLETED


class TestEmptySubclass:
    """Edge case: subclass with no states."""

    @pytest.mark.asyncio
    async def test_empty_subclass_no_states(self):
        assert EmptySubclass._puffinflow_auto_states == ()

    @pytest.mark.asyncio
    async def test_empty_subclass_run_fails(self):
        agent = EmptySubclass("empty")
        result = await agent.run()
        assert agent.status == AgentStatus.FAILED
        assert isinstance(result.error, ValueError)
        assert "No states defined" in str(result.error)
