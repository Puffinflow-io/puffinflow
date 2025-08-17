"""Test agent workflow validation."""

import pytest

from puffinflow import Agent
from puffinflow.core.agent.state import ExecutionMode


class TestAgentValidation:
    """Test agent validation logic."""

    def test_empty_state_name_raises_error(self):
        """Test that empty state name raises ValueError."""
        agent = Agent("test-agent")

        with pytest.raises(ValueError, match="State name must be a non-empty string"):
            agent.add_state("", lambda ctx: None)

        with pytest.raises(ValueError, match="State name must be a non-empty string"):
            agent.add_state(None, lambda ctx: None)

    def test_duplicate_state_name_raises_error(self):
        """Test that duplicate state names raise ValueError."""
        agent = Agent("test-agent")

        async def dummy_state(ctx):
            pass

        agent.add_state("state1", dummy_state)

        with pytest.raises(ValueError, match="State 'state1' already exists"):
            agent.add_state("state1", dummy_state)

    def test_nonexistent_dependency_raises_error(self):
        """Test that referencing nonexistent dependencies raises ValueError."""
        agent = Agent("test-agent")

        async def state_with_deps(ctx):
            pass

        with pytest.raises(
            ValueError,
            match="Dependency 'nonexistent' for state 'dependent' does not exist",
        ):
            agent.add_state("dependent", state_with_deps, dependencies=["nonexistent"])

    def test_valid_dependency_order_works(self):
        """Test that adding states in correct dependency order works."""
        agent = Agent("test-agent")

        async def state1(ctx):
            return "state2"

        async def state2(ctx):
            pass

        # This should work - state1 exists before state2 references it
        agent.add_state("state1", state1)
        agent.add_state("state2", state2, dependencies=["state1"])

        # Should not raise any errors
        assert "state1" in agent.states
        assert "state2" in agent.states
        assert agent.dependencies["state2"] == ["state1"]

    @pytest.mark.asyncio
    async def test_circular_dependencies_raise_error(self):
        """Test that circular dependencies are detected and return error."""
        agent = Agent("test-agent")

        async def state1(ctx):
            pass

        async def state2(ctx):
            pass

        async def state3(ctx):
            pass

        # Create valid chain first
        agent.add_state("state1", state1)
        agent.add_state("state2", state2, dependencies=["state1"])
        agent.add_state("state3", state3, dependencies=["state2"])

        # Manually create circular dependency after states are added
        agent.dependencies["state1"] = ["state3"]

        result = await agent.run()
        assert result.status.value == "failed"
        assert isinstance(result.error, ValueError)
        assert "Circular dependency detected" in str(result.error)

    @pytest.mark.asyncio
    async def test_sequential_mode_requires_entry_state(self):
        """Test that sequential mode requires at least one state without dependencies."""
        agent = Agent("test-agent")

        async def state1(ctx):
            pass

        async def state2(ctx):
            pass

        async def state3(ctx):
            pass

        # Create all states first
        agent.add_state("state1", state1)
        agent.add_state("state2", state2)
        agent.add_state("state3", state3)

        # Now manually create a dependency chain where no state is an entry point
        # All states depend on each other in a line, but add an extra dependency
        # that makes the "entry" state also have a dependency
        agent.dependencies["state1"] = ["state2"]  # state1 depends on state2
        agent.dependencies["state2"] = ["state3"]  # state2 depends on state3
        agent.dependencies["state3"] = ["state1"]  # This creates a cycle!

        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)
        assert result.status.value == "failed"
        assert isinstance(result.error, ValueError)
        # This will actually be caught as a circular dependency, which is correct
        assert "Circular dependency detected" in str(result.error)

    @pytest.mark.asyncio
    async def test_sequential_mode_no_entry_states_linear_chain(self):
        """Test sequential mode with no entry states in a linear dependency chain."""
        agent = Agent("test-agent")

        async def state1(ctx):
            pass

        async def state2(ctx):
            pass

        async def state3(ctx):
            pass

        # Create valid linear chain but manually break it to have no entry states
        agent.add_state("state1", state1)
        agent.add_state("state2", state2, dependencies=["state1"])
        agent.add_state("state3", state3, dependencies=["state2"])

        # Now manually create a dependency that makes state1 also depend on something
        # This way we have: state1 -> [unknown] <- state2 <- state3
        # and no state is an entry point
        agent.dependencies["state1"] = ["unknown_state"]

        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)
        assert result.status.value == "failed"
        assert isinstance(result.error, ValueError)
        assert (
            "Sequential execution mode requires at least one state without dependencies"
            in str(result.error)
        )

    @pytest.mark.asyncio
    async def test_sequential_mode_with_valid_entry_state(self):
        """Test that sequential mode works with valid entry state."""
        agent = Agent("test-agent")

        async def entry_state(ctx):
            ctx.set_variable("executed", True)
            return "dependent_state"

        async def dependent_state(ctx):
            ctx.set_variable("dependent_executed", True)

        agent.add_state("entry_state", entry_state)
        agent.add_state(
            "dependent_state", dependent_state, dependencies=["entry_state"]
        )

        # Should execute successfully
        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)
        assert result.get_variable("executed") is True
        assert result.get_variable("dependent_executed") is True

    @pytest.mark.asyncio
    async def test_parallel_mode_with_no_entry_states_warns(self):
        """Test that parallel mode warns when no entry states exist."""
        agent = Agent("test-agent")

        async def state1(ctx):
            pass

        async def state2(ctx):
            pass

        # Create states first, then manually create mutual dependencies
        agent.add_state("state1", state1)
        agent.add_state("state2", state2)

        # Manually create mutual dependencies after adding states
        agent.dependencies["state1"] = ["state2"]
        agent.dependencies["state2"] = ["state1"]

        # Should detect deadlock and fail
        result = await agent.run(execution_mode=ExecutionMode.PARALLEL)
        # Should complete but with no states executed due to unresolvable dependencies
        assert result.status.value == "failed"  # Deadlock detection

    @pytest.mark.asyncio
    async def test_complex_valid_workflow(self):
        """Test a complex but valid workflow with multiple dependencies."""
        agent = Agent("test-agent")

        async def init_state(ctx):
            ctx.set_variable("initialized", True)
            return ["fetch_data", "setup_config"]

        async def fetch_data(ctx):
            ctx.set_variable("data_fetched", True)
            return "process_data"

        async def setup_config(ctx):
            ctx.set_variable("config_ready", True)
            return "process_data"

        async def process_data(ctx):
            ctx.set_variable("processing_complete", True)
            return "finalize"

        async def finalize(ctx):
            ctx.set_variable("finalized", True)

        # Build workflow: init -> [fetch_data, setup_config] -> process_data -> finalize
        agent.add_state("init_state", init_state)
        agent.add_state("fetch_data", fetch_data, dependencies=["init_state"])
        agent.add_state("setup_config", setup_config, dependencies=["init_state"])
        agent.add_state(
            "process_data", process_data, dependencies=["fetch_data", "setup_config"]
        )
        agent.add_state("finalize", finalize, dependencies=["process_data"])

        # Should execute successfully
        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)
        assert result.get_variable("initialized") is True
        assert result.get_variable("data_fetched") is True
        assert result.get_variable("config_ready") is True
        assert result.get_variable("processing_complete") is True
        assert result.get_variable("finalized") is True

    @pytest.mark.asyncio
    async def test_self_dependency_raises_error(self):
        """Test that a state depending on itself returns error."""
        agent = Agent("test-agent")

        async def self_dependent_state(ctx):
            pass

        agent.add_state("self_state", self_dependent_state)

        # Manually create self-dependency (simulating edge case)
        agent.dependencies["self_state"] = ["self_state"]

        result = await agent.run()
        assert result.status.value == "failed"
        assert isinstance(result.error, ValueError)
        assert "Circular dependency detected" in str(result.error)

    @pytest.mark.asyncio
    async def test_no_states_returns_error(self):
        """Test that running agent with no states returns error status."""
        agent = Agent("test-agent")

        # Agent with no states should return error in result
        result = await agent.run()
        assert result.status.value == "failed"
        assert isinstance(result.error, ValueError)
        assert "No states defined" in str(result.error)
