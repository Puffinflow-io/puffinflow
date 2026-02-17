"""Test examples from the API reference documentation."""

import pytest

from puffinflow import Agent, ExecutionMode, Priority, state


@pytest.mark.asyncio
class TestApiReferenceExamples:
    """Test examples from api-reference.ts documentation."""

    async def test_agent_creation_and_basic_run(self):
        """Test basic Agent creation and run."""
        agent = Agent("hello-agent")

        @state
        async def hello(context):
            context.set_variable("message", "Hello, world!")
            return None

        agent.add_state("hello", hello)
        result = await agent.run()

        assert result.get_variable("message") == "Hello, world!"

    async def test_agent_run_with_initial_context(self):
        """Test Agent.run() with initial_context parameter."""
        agent = Agent("ctx-agent")

        @state
        async def greet(context):
            name = context.get_variable("name")
            context.set_variable("greeting", f"Hi, {name}!")
            return None

        agent.add_state("greet", greet)
        result = await agent.run(initial_context={"name": "Alice"})

        assert result.get_variable("greeting") == "Hi, Alice!"

    async def test_context_set_get_variable(self):
        """Test Context.set_variable() and Context.get_variable()."""
        agent = Agent("variable-test")

        async def writer(context):
            context.set_variable("count", 42)
            context.set_variable("items", ["a", "b", "c"])
            return "reader"

        async def reader(context):
            count = context.get_variable("count")
            items = context.get_variable("items")
            missing = context.get_variable("missing_key", "default_val")
            context.set_variable("read_count", count)
            context.set_variable("read_items", items)
            context.set_variable("read_missing", missing)
            return None

        agent.add_state("writer", writer)
        agent.add_state("reader", reader)
        result = await agent.run()

        assert result.get_variable("read_count") == 42
        assert result.get_variable("read_items") == ["a", "b", "c"]
        assert result.get_variable("read_missing") == "default_val"

    async def test_context_typed_variable(self):
        """Test Context.set_typed_variable() for type-checked storage."""
        agent = Agent("typed-var-test")

        async def init_typed(context):
            context.set_typed_variable("score", 95.5)
            context.set_typed_variable("active", True)
            return "update_typed"

        async def update_typed(context):
            # Valid update (same type)
            context.set_typed_variable("score", 98.0)

            # Invalid update (wrong type) should raise
            type_error_caught = False
            try:
                context.set_typed_variable("score", "not a float")
            except (TypeError, ValueError):
                type_error_caught = True

            context.set_variable("type_error_caught", type_error_caught)

            score = context.get_typed_variable("score")
            context.set_variable("final_score", score)
            return None

        agent.add_state("init_typed", init_typed)
        agent.add_state("update_typed", update_typed)
        result = await agent.run()

        assert result.get_variable("final_score") == 98.0
        assert result.get_variable("type_error_caught") is True

    async def test_context_set_get_output(self):
        """Test Context.set_output() and Context.get_output()."""
        agent = Agent("output-test")

        async def produce(context):
            context.set_output("result_id", "res-001")
            context.set_output("total", 250)
            return "consume"

        async def consume(context):
            result_id = context.get_output("result_id")
            total = context.get_output("total")
            context.set_variable("summary", f"id={result_id}, total={total}")
            return None

        agent.add_state("produce", produce)
        agent.add_state("consume", consume)
        result = await agent.run()

        assert result.get_output("result_id") == "res-001"
        assert result.get_output("total") == 250
        assert result.get_variable("summary") == "id=res-001, total=250"

    async def test_state_decorator_with_resource_params(self):
        """Test @state decorator with cpu, memory, max_retries params."""
        agent = Agent("decorated-resource-test")

        @state(cpu=2.0, memory=1024, max_retries=3, timeout=30.0)
        async def heavy_task(context):
            context.set_variable("heavy_done", True)
            return "light_task"

        @state(cpu=0.5, memory=128, priority=Priority.LOW)
        async def light_task(context):
            context.set_variable("light_done", True)
            return None

        agent.add_state("heavy_task", heavy_task)
        agent.add_state("light_task", light_task)
        result = await agent.run()

        assert result.get_variable("heavy_done") is True
        assert result.get_variable("light_done") is True

    async def test_execution_mode_sequential(self):
        """Test ExecutionMode.SEQUENTIAL runs states in order."""
        agent = Agent("seq-mode-test")
        execution_log = []

        @state
        async def step_a(context):
            execution_log.append("a")
            context.set_variable("a_done", True)
            return "step_b"

        @state
        async def step_b(context):
            execution_log.append("b")
            context.set_variable("b_done", True)
            return "step_c"

        @state
        async def step_c(context):
            execution_log.append("c")
            context.set_variable("c_done", True)
            return None

        agent.add_state("step_a", step_a)
        agent.add_state("step_b", step_b)
        agent.add_state("step_c", step_c)

        result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)

        assert result.get_variable("a_done") is True
        assert result.get_variable("b_done") is True
        assert result.get_variable("c_done") is True
        assert execution_log == ["a", "b", "c"]

    async def test_execution_mode_parallel(self):
        """Test ExecutionMode.PARALLEL runs independent states concurrently."""
        agent = Agent("par-mode-test")

        async def fetch_a(context):
            context.set_variable("a_data", "alpha")

        async def fetch_b(context):
            context.set_variable("b_data", "beta")

        async def merge(context):
            a = context.get_variable("a_data")
            b = context.get_variable("b_data")
            context.set_variable("merged", f"{a}+{b}")

        agent.add_state("fetch_a", fetch_a)
        agent.add_state("fetch_b", fetch_b)
        agent.add_state("merge", merge, dependencies=["fetch_a", "fetch_b"])

        result = await agent.run(execution_mode=ExecutionMode.PARALLEL)

        assert result.get_variable("a_data") == "alpha"
        assert result.get_variable("b_data") == "beta"
        assert result.get_variable("merged") == "alpha+beta"

    async def test_state_returning_none_ends_workflow(self):
        """Test that returning None from a state ends the workflow."""
        agent = Agent("end-test")

        @state
        async def only_state(context):
            context.set_variable("ran", True)
            return None  # Ends the workflow

        agent.add_state("only_state", only_state)
        result = await agent.run()

        assert result.get_variable("ran") is True

    async def test_state_decorator_bare_vs_parens(self):
        """Test that @state and @state() both work."""
        agent = Agent("decorator-test")

        @state
        async def bare_decorator(context):
            context.set_variable("bare", True)
            return "parens_decorator"

        @state()
        async def parens_decorator(context):
            context.set_variable("parens", True)
            return None

        agent.add_state("bare_decorator", bare_decorator)
        agent.add_state("parens_decorator", parens_decorator)
        result = await agent.run()

        assert result.get_variable("bare") is True
        assert result.get_variable("parens") is True

    async def test_agent_set_variable_before_run(self):
        """Test Agent.set_variable() before calling run()."""
        agent = Agent("pre-set-test")

        @state
        async def use_preset(context):
            val = context.get_variable("preset_key")
            context.set_variable("received", val)
            return None

        agent.add_state("use_preset", use_preset)
        agent.set_variable("preset_key", "preset_value")
        result = await agent.run()

        assert result.get_variable("received") == "preset_value"
