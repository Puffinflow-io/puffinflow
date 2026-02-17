"""Test examples from the best practices documentation."""

import pytest

from puffinflow import Agent, ExecutionMode, Priority, state


@pytest.mark.asyncio
class TestBestPracticesExamples:
    """Test examples from best-practices.ts documentation."""

    async def test_agent_pattern_with_multiple_states(self):
        """Test creating an agent with multiple states as a reusable pattern."""
        agent = Agent("order-handler")

        @state
        async def validate_order(context):
            order = context.get_variable("order")
            if order["total"] <= 0:
                context.set_variable("error", "Invalid order total")
                return None
            context.set_variable("validated", True)
            return "process_payment"

        @state
        async def process_payment(context):
            order = context.get_variable("order")
            context.set_variable("payment_status", "charged")
            context.set_variable("amount_charged", order["total"])
            return "confirm_order"

        @state
        async def confirm_order(context):
            context.set_variable("order_status", "confirmed")
            return None

        agent.add_state("validate_order", validate_order)
        agent.add_state("process_payment", process_payment)
        agent.add_state("confirm_order", confirm_order)

        result = await agent.run(initial_context={"order": {"id": 42, "total": 99.99}})

        assert result.get_variable("validated") is True
        assert result.get_variable("payment_status") == "charged"
        assert result.get_variable("amount_charged") == 99.99
        assert result.get_variable("order_status") == "confirmed"

    async def test_context_data_lifecycle(self):
        """Test setting a variable in one state, reading it in the next, and verifying after run."""
        agent = Agent("lifecycle-test")

        async def producer(context):
            context.set_variable("message", "hello from producer")
            context.set_variable("counter", 10)
            return "consumer"

        async def consumer(context):
            msg = context.get_variable("message")
            counter = context.get_variable("counter")
            context.set_variable("result", f"{msg} | count={counter}")
            return None

        agent.add_state("producer", producer)
        agent.add_state("consumer", consumer)

        result = await agent.run()

        # Verify data flows correctly between states
        assert result.get_variable("message") == "hello from producer"
        assert result.get_variable("counter") == 10
        assert result.get_variable("result") == "hello from producer | count=10"

    async def test_resource_annotated_states(self):
        """Test states annotated with resource requirements."""
        agent = Agent("resource-demo")

        @state(cpu=2.0, memory=512)
        async def cpu_heavy_task(context):
            # Simulate CPU-intensive work
            total = sum(range(1000))
            context.set_variable("cpu_result", total)
            return "io_task"

        @state(cpu=0.5, memory=256, timeout=10.0)
        async def io_task(context):
            context.set_variable("io_result", "data loaded")
            return "finalize"

        @state(cpu=1.0, memory=128, max_retries=2, priority=Priority.HIGH)
        async def finalize(context):
            cpu_res = context.get_variable("cpu_result")
            io_res = context.get_variable("io_result")
            context.set_variable("summary", f"cpu={cpu_res}, io={io_res}")
            return None

        agent.add_state("cpu_heavy_task", cpu_heavy_task)
        agent.add_state("io_task", io_task)
        agent.add_state("finalize", finalize)

        result = await agent.run()

        assert result.get_variable("cpu_result") == 499500
        assert result.get_variable("io_result") == "data loaded"
        assert result.get_variable("summary") == "cpu=499500, io=data loaded"

    async def test_initial_context_pattern(self):
        """Test passing initial context to an agent run."""
        agent = Agent("configurable-agent")

        @state
        async def greet(context):
            name = context.get_variable("user_name")
            lang = context.get_variable("language", "en")
            if lang == "es":
                context.set_variable("greeting", f"Hola, {name}!")
            else:
                context.set_variable("greeting", f"Hello, {name}!")
            return None

        agent.add_state("greet", greet)

        # Run with English
        result_en = await agent.run(
            initial_context={"user_name": "Alice", "language": "en"}
        )
        assert result_en.get_variable("greeting") == "Hello, Alice!"

        # Create a fresh agent for Spanish run
        agent_es = Agent("configurable-agent-es")
        agent_es.add_state("greet", greet)
        result_es = await agent_es.run(
            initial_context={"user_name": "Bob", "language": "es"}
        )
        assert result_es.get_variable("greeting") == "Hola, Bob!"

    async def test_parallel_independent_states(self):
        """Test running independent states in parallel with dependencies."""
        agent = Agent("parallel-best-practice")

        async def load_config(context):
            context.set_variable("config", {"env": "test"})

        async def load_data(context):
            context.set_variable("data", [1, 2, 3])

        async def combine(context):
            cfg = context.get_variable("config")
            data = context.get_variable("data")
            context.set_variable("output", f"env={cfg['env']}, items={len(data)}")

        agent.add_state("load_config", load_config)
        agent.add_state("load_data", load_data)
        agent.add_state("combine", combine, dependencies=["load_config", "load_data"])

        result = await agent.run(execution_mode=ExecutionMode.PARALLEL)

        assert result.get_variable("config") == {"env": "test"}
        assert result.get_variable("data") == [1, 2, 3]
        assert result.get_variable("output") == "env=test, items=3"
