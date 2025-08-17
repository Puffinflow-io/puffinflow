"""Test execution modes for agent workflows."""

import asyncio

import pytest

from puffinflow import Agent, ExecutionMode


@pytest.mark.asyncio
class TestExecutionModes:
    """Test different execution modes for agent workflows."""

    async def test_parallel_execution_mode_default(self):
        """Test that PARALLEL mode is the default and all entry states run."""
        agent = Agent("parallel-test")
        execution_log = []

        async def state_a(context):
            execution_log.append("state_a")
            context.set_variable("result_a", "A executed")

        async def state_b(context):
            execution_log.append("state_b")
            context.set_variable("result_b", "B executed")

        async def state_c(context):
            execution_log.append("state_c")
            context.set_variable("result_c", "C executed")

        # Add states without dependencies - all should be entry points
        agent.add_state("state_a", state_a)
        agent.add_state("state_b", state_b)
        agent.add_state("state_c", state_c)

        # Run with default (parallel) mode
        result = await agent.run(execution_mode=ExecutionMode.PARALLEL)

        # All states should have executed
        assert "state_a" in execution_log
        assert "state_b" in execution_log
        assert "state_c" in execution_log
        assert result.get_variable("result_a") == "A executed"
        assert result.get_variable("result_b") == "B executed"
        assert result.get_variable("result_c") == "C executed"

    async def test_parallel_execution_mode_explicit(self):
        """Test PARALLEL mode explicitly set."""
        agent = Agent("parallel-explicit-test")
        execution_log = []

        async def state_a(context):
            execution_log.append("state_a")
            context.set_variable("result_a", "A executed")

        async def state_b(context):
            execution_log.append("state_b")
            context.set_variable("result_b", "B executed")

        agent.add_state("state_a", state_a)
        agent.add_state("state_b", state_b)

        # Run with explicit PARALLEL mode
        result = await agent.run(execution_mode=ExecutionMode.PARALLEL)

        # Both states should have executed
        assert len(execution_log) == 2
        assert "state_a" in execution_log
        assert "state_b" in execution_log
        assert result.get_variable("result_a") == "A executed"
        assert result.get_variable("result_b") == "B executed"

    async def test_sequential_execution_mode(self):
        """Test SEQUENTIAL mode only runs first entry state."""
        agent = Agent("sequential-test")
        execution_log = []

        async def state_a(context):
            execution_log.append("state_a")
            context.set_variable("result_a", "A executed")
            # Don't return anything - should end here

        async def state_b(context):
            execution_log.append("state_b")
            context.set_variable("result_b", "B executed")

        async def state_c(context):
            execution_log.append("state_c")
            context.set_variable("result_c", "C executed")

        # Add states without dependencies
        agent.add_state("state_a", state_a)
        agent.add_state("state_b", state_b)
        agent.add_state("state_c", state_c)

        # Run with SEQUENTIAL mode
        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)

        # Only the first state should have executed
        assert execution_log == ["state_a"]
        assert result.get_variable("result_a") == "A executed"
        assert result.get_variable("result_b") is None
        assert result.get_variable("result_c") is None

    async def test_sequential_with_dynamic_flow_control(self):
        """Test SEQUENTIAL mode with dynamic flow control via return values."""
        agent = Agent("sequential-dynamic-test")
        execution_log = []

        async def router(context):
            execution_log.append("router")
            user_type = context.get_variable("user_type", "premium")
            context.set_variable("user_type", user_type)

            if user_type == "premium":
                return "premium_flow"
            else:
                return "basic_flow"

        async def premium_flow(context):
            execution_log.append("premium_flow")
            context.set_variable("features", ["advanced_analytics", "priority_support"])
            return "send_welcome"

        async def basic_flow(context):
            execution_log.append("basic_flow")
            context.set_variable("features", ["basic_analytics"])
            return "send_welcome"

        async def send_welcome(context):
            execution_log.append("send_welcome")
            user_type = context.get_variable("user_type")
            context.get_variable("features")
            context.set_variable("welcome_message", f"Welcome {user_type} user!")

        # Add states - router is first, others should only run via return values
        agent.add_state("router", router)
        agent.add_state("premium_flow", premium_flow)
        agent.add_state("basic_flow", basic_flow)
        agent.add_state("send_welcome", send_welcome)

        # Run with SEQUENTIAL mode
        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)

        # Should follow: router -> premium_flow -> send_welcome
        assert execution_log == ["router", "premium_flow", "send_welcome"]
        assert result.get_variable("user_type") == "premium"
        assert result.get_variable("features") == [
            "advanced_analytics",
            "priority_support",
        ]
        assert "premium" in result.get_variable("welcome_message")

    async def test_sequential_with_dependencies(self):
        """Test SEQUENTIAL mode respects dependencies."""
        agent = Agent("sequential-deps-test")
        execution_log = []

        async def prepare_data(context):
            execution_log.append("prepare_data")
            context.set_variable("data", [1, 2, 3])

        async def process_data(context):
            execution_log.append("process_data")
            data = context.get_variable("data")
            result = sum(data)
            context.set_variable("result", result)

        async def independent_task(context):
            execution_log.append("independent_task")
            context.set_variable("independent", "done")

        # Add states with dependencies
        agent.add_state("prepare_data", prepare_data)
        agent.add_state("process_data", process_data, dependencies=["prepare_data"])
        agent.add_state("independent_task", independent_task)

        # Run with SEQUENTIAL mode
        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)

        # prepare_data is first entry state, should trigger process_data
        # independent_task should not run in sequential mode
        assert "prepare_data" in execution_log
        assert "process_data" in execution_log
        assert "independent_task" not in execution_log
        assert result.get_variable("result") == 6

    async def test_parallel_execution_with_dependencies(self):
        """Test PARALLEL mode with dependencies works correctly."""
        agent = Agent("parallel-deps-test")
        execution_log = []

        async def fetch_user_data(context):
            execution_log.append("fetch_user_data")
            await asyncio.sleep(0.1)  # Simulate async work
            context.set_variable("user_count", 1250)

        async def fetch_sales_data(context):
            execution_log.append("fetch_sales_data")
            await asyncio.sleep(0.1)  # Simulate async work
            context.set_variable("revenue", 45000)

        async def generate_report(context):
            execution_log.append("generate_report")
            users = context.get_variable("user_count")
            revenue = context.get_variable("revenue")
            rpu = revenue / users
            context.set_variable("revenue_per_user", rpu)

        # Add states with dependencies
        agent.add_state("fetch_user_data", fetch_user_data)
        agent.add_state("fetch_sales_data", fetch_sales_data)
        agent.add_state(
            "generate_report",
            generate_report,
            dependencies=["fetch_user_data", "fetch_sales_data"],
        )

        # Run with PARALLEL mode
        result = await agent.run(execution_mode=ExecutionMode.PARALLEL)

        # Both fetch states should run in parallel, then generate_report
        assert "fetch_user_data" in execution_log
        assert "fetch_sales_data" in execution_log
        assert "generate_report" in execution_log
        assert result.get_variable("revenue_per_user") == 36.0

    async def test_sequential_mode_with_parallel_return(self):
        """Test SEQUENTIAL mode can still trigger parallel execution via return values."""
        agent = Agent("sequential-parallel-test")
        execution_log = []

        async def initiate_parallel(context):
            execution_log.append("initiate_parallel")
            context.set_variable("order_id", "ORD-123")
            # Return multiple states for parallel execution
            return ["send_confirmation", "update_inventory", "charge_payment"]

        async def send_confirmation(context):
            execution_log.append("send_confirmation")
            order_id = context.get_variable("order_id")
            context.set_variable("confirmation", f"Confirmed {order_id}")

        async def update_inventory(context):
            execution_log.append("update_inventory")
            context.set_variable("inventory", "Updated")

        async def charge_payment(context):
            execution_log.append("charge_payment")
            order_id = context.get_variable("order_id")
            context.set_variable("payment", f"Charged {order_id}")

        agent.add_state("initiate_parallel", initiate_parallel)
        agent.add_state("send_confirmation", send_confirmation)
        agent.add_state("update_inventory", update_inventory)
        agent.add_state("charge_payment", charge_payment)

        # Run with SEQUENTIAL mode
        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)

        # Should start with initiate_parallel, then run the three parallel states
        assert "initiate_parallel" in execution_log
        assert "send_confirmation" in execution_log
        assert "update_inventory" in execution_log
        assert "charge_payment" in execution_log
        assert result.get_variable("order_id") == "ORD-123"
        assert "ORD-123" in result.get_variable("confirmation")
        assert result.get_variable("inventory") == "Updated"

    async def test_sequential_mode_empty_states(self):
        """Test SEQUENTIAL mode with no states."""
        agent = Agent("empty-test")

        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)

        assert result.status.value == "failed"
        assert isinstance(result.error, ValueError)
        assert "No states defined" in str(result.error)

    async def test_sequential_mode_single_state(self):
        """Test SEQUENTIAL mode with single state."""
        agent = Agent("single-state-test")
        execution_log = []

        async def single_state(context):
            execution_log.append("single_state")
            context.set_variable("result", "done")

        agent.add_state("single_state", single_state)

        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)

        assert execution_log == ["single_state"]
        assert result.get_variable("result") == "done"

    async def test_backwards_compatibility(self):
        """Test that existing code without execution_mode parameter still works."""
        agent = Agent("backwards-compat-test")
        execution_log = []

        async def state_a(context):
            execution_log.append("state_a")
            context.set_variable("result", "A executed")

        async def state_b(context):
            execution_log.append("state_b")
            context.set_variable("result", "B executed")

        agent.add_state("state_a", state_a)
        agent.add_state("state_b", state_b)

        # Run without execution_mode parameter (should default to SEQUENTIAL)
        await agent.run()

        # Only first state should execute (sequential behavior)
        assert len(execution_log) == 1
        assert "state_a" in execution_log
