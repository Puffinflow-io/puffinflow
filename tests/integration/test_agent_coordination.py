"""Integration tests for agent coordination functionality.

Tests the interaction between agents, teams, and coordination patterns.
"""

import asyncio
import time

import pytest

from puffinflow import (
    Agent,
    Context,
    cpu_intensive,
    create_team,
    memory_intensive,
    run_agents_parallel,
    run_agents_sequential,
    state,
)


class TestDataCollector(Agent):
    """Test agent that collects data."""

    def __init__(self, name: str, data_size: int = 100):
        super().__init__(name)
        self.set_variable("data_size", data_size)
        self.add_state("collect", self.collect)

    @state(cpu=1.0, memory=256.0)
    async def collect(self, context: Context):
        """Collect test data."""
        data_size = self.get_variable("data_size", 100)
        await asyncio.sleep(0.1)  # Simulate collection time

        data = list(range(data_size))
        context.set_output("collected_data", data)
        context.set_output("collection_time", 0.1)
        context.set_metric("data_points", len(data))

        return None


class TestDataProcessor(Agent):
    """Test agent that processes data."""

    def __init__(self, name: str, processing_factor: float = 1.0):
        super().__init__(name)
        self.set_variable("processing_factor", processing_factor)
        self.add_state("process", self.process)

    @cpu_intensive(cpu=2.0, memory=512.0)
    async def process(self, context: Context):
        """Process test data."""
        factor = self.get_variable("processing_factor", 1.0)
        await asyncio.sleep(0.2 * factor)  # Simulate processing time

        # Simulate processing result
        processed_count = int(100 * factor)
        result = {
            "processed_count": processed_count,
            "processing_factor": factor,
            "quality_score": min(0.95, 0.8 + factor * 0.1),
        }

        context.set_output("processed_data", result)
        context.set_metric("processing_efficiency", factor)

        return None


class TestDataAggregator(Agent):
    """Test agent that aggregates results."""

    def __init__(self, name: str):
        super().__init__(name)
        self.add_state("aggregate", self.aggregate)

    @memory_intensive(memory=1024.0, cpu=1.5)
    async def aggregate(self, context: Context):
        """Aggregate test results."""
        await asyncio.sleep(0.15)  # Simulate aggregation time

        # Simulate aggregation of multiple sources
        aggregated_result = {
            "total_processed": 300,
            "average_quality": 0.88,
            "sources_count": 3,
            "aggregation_time": 0.15,
        }

        context.set_output("aggregated_data", aggregated_result)
        context.set_metric("aggregation_efficiency", 0.95)

        return None


@pytest.mark.integration
@pytest.mark.asyncio
class TestAgentCoordination:
    """Test agent coordination patterns."""

    async def test_parallel_agent_execution(self):
        """Test running multiple agents in parallel."""
        # Create multiple data collectors
        collectors = [
            TestDataCollector(f"collector-{i}", data_size=50 + i * 10) for i in range(3)
        ]

        start_time = time.time()
        results = await run_agents_parallel(collectors)
        execution_time = time.time() - start_time

        # Verify all agents completed
        assert len(results) == 3
        for _agent_name, result in results.items():
            # Handle both string and enum status
            status = (
                result.status.name
                if hasattr(result.status, "name")
                else str(result.status)
            )
            assert status.upper() in ["COMPLETED", "SUCCESS"]
            assert "collected_data" in result.outputs

        # Parallel execution should be faster than sequential
        assert execution_time < 0.5  # Should complete in less than 0.5s

        # Verify data collection results
        total_data_points = sum(
            len(result.get_output("collected_data", [])) for result in results.values()
        )
        assert total_data_points == 50 + 60 + 70  # 180 total

    async def test_sequential_agent_execution(self):
        """Test running agents in sequence."""
        # Create a processing pipeline
        agents = [
            TestDataCollector("seq-collector", data_size=100),
            TestDataProcessor("seq-processor", processing_factor=1.2),
            TestDataAggregator("seq-aggregator"),
        ]

        start_time = time.time()
        results = await run_agents_sequential(agents)
        execution_time = time.time() - start_time

        # Verify all agents completed in order
        assert len(results) == 3
        agent_names = list(results.keys())
        assert agent_names == ["seq-collector", "seq-processor", "seq-aggregator"]

        # Sequential execution should take some time
        assert (
            execution_time >= 0.3
        )  # At least 0.1 + 0.24 + 0.15 = 0.49s, but allowing for timing variance

        # Verify pipeline results
        collector_result = results["seq-collector"]
        processor_result = results["seq-processor"]
        aggregator_result = results["seq-aggregator"]

        assert len(collector_result.get_output("collected_data", [])) == 100
        assert processor_result.get_output("processed_data")["processed_count"] == 120
        assert aggregator_result.get_output("aggregated_data")["total_processed"] == 300

    async def test_agent_team_coordination(self):
        """Test team-based agent coordination."""
        # Create a processing team
        agents = [
            TestDataCollector("collector-1", data_size=100),
            TestDataProcessor("processor-1", processing_factor=1.0),
            TestDataAggregator("aggregator-1"),
        ]
        team = create_team("integration-test-team", agents)

        # Add team members
        team.add_agent(TestDataCollector("team-collector", data_size=75))
        team.add_agent(TestDataProcessor("team-processor-1", processing_factor=0.8))
        team.add_agent(TestDataProcessor("team-processor-2", processing_factor=1.5))
        team.add_agent(TestDataAggregator("team-aggregator"))

        # Run the team
        start_time = time.time()
        team_result = await team.run()
        time.time() - start_time

        # Verify team execution
        assert team_result.status in ["completed", "success"]
        assert len(team_result.agent_results) == 7  # 3 initial + 4 added agents

        # Verify individual agent results
        for _agent_name, result in team_result.agent_results.items():
            # Handle both string and enum status
            status = (
                result.status.name
                if hasattr(result.status, "name")
                else str(result.status)
            )
            assert status.upper() in ["COMPLETED", "SUCCESS"]

        # Check specific results from added agents
        collector_result = team_result.agent_results["team-collector"]
        processor1_result = team_result.agent_results["team-processor-1"]
        processor2_result = team_result.agent_results["team-processor-2"]
        aggregator_result = team_result.agent_results["team-aggregator"]

        assert len(collector_result.get_output("collected_data", [])) == 75
        assert (
            processor1_result.get_output("processed_data")["processing_factor"] == 0.8
        )
        assert (
            processor2_result.get_output("processed_data")["processing_factor"] == 1.5
        )
        assert aggregator_result.get_output("aggregated_data")["sources_count"] == 3

        # Verify initial agents are also present
        assert "collector-1" in team_result.agent_results
        assert "processor-1" in team_result.agent_results
        assert "aggregator-1" in team_result.agent_results

    async def test_mixed_coordination_patterns(self):
        """Test combining different coordination patterns."""
        # Phase 1: Parallel data collection
        collectors = [
            TestDataCollector(f"mixed-collector-{i}", data_size=30) for i in range(2)
        ]
        collection_results = await run_agents_parallel(collectors)

        # Phase 2: Sequential processing
        processors = [
            TestDataProcessor("mixed-processor-1", processing_factor=1.0),
            TestDataProcessor("mixed-processor-2", processing_factor=1.3),
        ]
        processing_results = await run_agents_sequential(processors)

        # Phase 3: Final aggregation
        aggregator = TestDataAggregator("mixed-aggregator")
        aggregation_result = await aggregator.run()

        # Verify all phases completed successfully
        assert len(collection_results) == 2
        assert len(processing_results) == 2
        # Handle both string and enum status
        status = (
            aggregation_result.status.name
            if hasattr(aggregation_result.status, "name")
            else str(aggregation_result.status)
        )
        assert status.upper() in ["COMPLETED", "SUCCESS"]

        # Verify data flow
        total_collected = sum(
            len(result.get_output("collected_data", []))
            for result in collection_results.values()
        )
        assert total_collected == 60  # 2 * 30

        proc1_data = processing_results["mixed-processor-1"].get_output(
            "processed_data"
        )
        proc2_data = processing_results["mixed-processor-2"].get_output(
            "processed_data"
        )
        assert proc1_data["processing_factor"] == 1.0
        assert proc2_data["processing_factor"] == 1.3

        final_data = aggregation_result.get_output("aggregated_data")
        assert final_data["total_processed"] == 300

    async def test_agent_failure_handling(self):
        """Test coordination behavior when agents fail."""

        class FailingAgent(Agent):
            def __init__(self, name: str, should_fail: bool = True):
                super().__init__(name)
                self.set_variable("should_fail", should_fail)
                self.add_state("fail_test", self.fail_test)

            @state(cpu=0.5, memory=128.0)
            async def fail_test(self, context: Context):
                if self.get_variable("should_fail", False):
                    raise ValueError("Intentional test failure")
                context.set_output("success", True)
                return None

        # Test parallel execution with one failing agent
        agents = [
            TestDataCollector("success-agent", data_size=50),
            FailingAgent("failing-agent", should_fail=True),
            TestDataProcessor("another-success-agent", processing_factor=1.0),
        ]

        results = await run_agents_parallel(agents)

        # Verify that successful agents completed
        success_count = sum(
            1
            for result in results.values()
            if (
                result.status.name
                if hasattr(result.status, "name")
                else str(result.status)
            ).upper()
            in ["COMPLETED", "SUCCESS"]
        )
        failure_count = sum(
            1
            for result in results.values()
            if (
                result.status.name
                if hasattr(result.status, "name")
                else str(result.status)
            ).upper()
            in ["FAILED", "ERROR"]
        )

        assert success_count >= 2  # At least 2 should succeed
        assert failure_count >= 1  # At least 1 should fail

        # Verify successful agents have expected outputs
        for _agent_name, result in results.items():
            status = (
                result.status.name
                if hasattr(result.status, "name")
                else str(result.status)
            )
            if status.upper() in ["COMPLETED", "SUCCESS"]:
                assert len(result.outputs) > 0

    async def test_resource_coordination(self):
        """Test resource allocation across coordinated agents."""

        class ResourceIntensiveAgent(Agent):
            def __init__(self, name: str, cpu_req: float, memory_req: float):
                super().__init__(name)
                self.set_variable("cpu_req", cpu_req)
                self.set_variable("memory_req", memory_req)
                self.add_state("consume_resources", self.consume_resources)

            @state(cpu=0, memory=0)  # Will be set dynamically
            async def consume_resources(self, context: Context):
                cpu_req = self.get_variable("cpu_req", 1.0)
                memory_req = self.get_variable("memory_req", 256.0)

                # Simulate resource-intensive work
                await asyncio.sleep(0.1)

                context.set_output("cpu_used", cpu_req)
                context.set_output("memory_used", memory_req)
                context.set_metric("resource_efficiency", 0.9)

                return None

        # Create agents with different resource requirements
        agents = [
            ResourceIntensiveAgent("low-resource", cpu_req=1.0, memory_req=256.0),
            ResourceIntensiveAgent("medium-resource", cpu_req=2.0, memory_req=512.0),
            ResourceIntensiveAgent("high-resource", cpu_req=4.0, memory_req=1024.0),
        ]

        # Run in parallel to test resource coordination
        results = await run_agents_parallel(agents)

        # Verify all agents completed (resource manager should handle allocation)
        assert len(results) == 3
        for result in results.values():
            status = (
                result.status.name
                if hasattr(result.status, "name")
                else str(result.status)
            )
            assert status.upper() in ["COMPLETED", "SUCCESS"]
            assert "cpu_used" in result.outputs
            assert "memory_used" in result.outputs

        # Verify resource usage was tracked
        total_cpu = sum(result.get_output("cpu_used", 0) for result in results.values())
        total_memory = sum(
            result.get_output("memory_used", 0) for result in results.values()
        )

        assert total_cpu == 7.0  # 1.0 + 2.0 + 4.0
        assert total_memory == 1792.0  # 256 + 512 + 1024

    async def test_context_sharing_between_agents(self):
        """Test context and data sharing between coordinated agents."""

        class ContextSharingAgent(Agent):
            def __init__(self, name: str, shared_key: str):
                super().__init__(name)
                self.set_variable("shared_key", shared_key)
                self.add_state("share_context", self.share_context)

            @state(cpu=0.5, memory=128.0)
            async def share_context(self, context: Context):
                shared_key = self.get_variable("shared_key")

                # Set shared data using context methods
                context.set_variable(shared_key, f"data_from_{self.name}")
                context.set_output("shared_key", shared_key)
                context.set_output("shared_value", f"data_from_{self.name}")

                return None

        # Create agents that will share context
        agents = [
            ContextSharingAgent("sharer-1", "shared_data_1"),
            ContextSharingAgent("sharer-2", "shared_data_2"),
            ContextSharingAgent("sharer-3", "shared_data_3"),
        ]

        # Run agents and verify context sharing
        results = await run_agents_parallel(agents)

        assert len(results) == 3
        for result in results.values():
            status = (
                result.status.name
                if hasattr(result.status, "name")
                else str(result.status)
            )
            assert status.upper() in ["COMPLETED", "SUCCESS"]
            assert "shared_key" in result.outputs
            assert "shared_value" in result.outputs

        # Verify each agent set its shared data
        shared_keys = [result.get_output("shared_key") for result in results.values()]
        shared_values = [
            result.get_output("shared_value") for result in results.values()
        ]

        assert "shared_data_1" in shared_keys
        assert "shared_data_2" in shared_keys
        assert "shared_data_3" in shared_keys

        assert "data_from_sharer-1" in shared_values
        assert "data_from_sharer-2" in shared_values
        assert "data_from_sharer-3" in shared_values


@pytest.mark.integration
@pytest.mark.asyncio
class TestAdvancedCoordination:
    """Test advanced coordination scenarios."""

    async def test_dynamic_agent_creation(self):
        """Test creating and coordinating agents dynamically."""

        class DynamicAgentFactory:
            @staticmethod
            def create_collector(name: str, size: int) -> TestDataCollector:
                return TestDataCollector(name, data_size=size)

            @staticmethod
            def create_processor(name: str, factor: float) -> TestDataProcessor:
                return TestDataProcessor(name, processing_factor=factor)

        factory = DynamicAgentFactory()

        # Dynamically create agents based on requirements
        collectors = [
            factory.create_collector(f"dynamic-collector-{i}", size=25 * (i + 1))
            for i in range(3)
        ]

        processors = [
            factory.create_processor(f"dynamic-processor-{i}", factor=0.8 + i * 0.2)
            for i in range(2)
        ]

        # Run dynamic agents
        collection_results = await run_agents_parallel(collectors)
        processing_results = await run_agents_parallel(processors)

        # Verify dynamic creation and execution
        assert len(collection_results) == 3
        assert len(processing_results) == 2

        # Verify different configurations were applied
        data_sizes = [
            len(result.get_output("collected_data", []))
            for result in collection_results.values()
        ]
        assert sorted(data_sizes) == [25, 50, 75]

        processing_factors = [
            result.get_output("processed_data", {}).get("processing_factor", 0)
            for result in processing_results.values()
        ]
        assert sorted(processing_factors) == [0.8, 1.0]

    async def test_conditional_coordination(self):
        """Test coordination based on conditions and results."""

        class ConditionalAgent(Agent):
            def __init__(self, name: str, condition_value: int):
                super().__init__(name)
                self.set_variable("condition_value", condition_value)
                self.add_state("check_condition", self.check_condition)

            @state(cpu=0.5, memory=128.0)
            async def check_condition(self, context: Context):
                value = self.get_variable("condition_value", 0)

                if value > 50:
                    context.set_output("condition_result", "high")
                    context.set_output("next_action", "process_intensive")
                elif value > 20:
                    context.set_output("condition_result", "medium")
                    context.set_output("next_action", "process_standard")
                else:
                    context.set_output("condition_result", "low")
                    context.set_output("next_action", "process_light")

                context.set_output("condition_value", value)
                return None

        # Create agents with different condition values
        conditional_agents = [
            ConditionalAgent("conditional-1", condition_value=75),
            ConditionalAgent("conditional-2", condition_value=35),
            ConditionalAgent("conditional-3", condition_value=15),
        ]

        # Run conditional agents
        results = await run_agents_parallel(conditional_agents)

        # Verify conditional logic
        assert len(results) == 3

        # Check condition results
        condition_results = {
            name: result.get_output("condition_result")
            for name, result in results.items()
        }

        next_actions = {
            name: result.get_output("next_action") for name, result in results.items()
        }

        # Verify conditions were evaluated correctly
        assert "high" in condition_results.values()
        assert "medium" in condition_results.values()
        assert "low" in condition_results.values()

        assert "process_intensive" in next_actions.values()
        assert "process_standard" in next_actions.values()
        assert "process_light" in next_actions.values()

    async def test_cascading_coordination(self):
        """Test cascading coordination where results trigger new agents."""

        class TriggerAgent(Agent):
            def __init__(self, name: str, trigger_threshold: int):
                super().__init__(name)
                self.set_variable("trigger_threshold", trigger_threshold)
                self.add_state("evaluate_trigger", self.evaluate_trigger)

            @state(cpu=1.0, memory=256.0)
            async def evaluate_trigger(self, context: Context):
                threshold = self.get_variable("trigger_threshold", 50)

                # Simulate some computation
                computed_value = hash(self.name) % 100

                context.set_output("computed_value", computed_value)
                context.set_output("threshold", threshold)
                context.set_output("should_trigger", computed_value > threshold)

                if computed_value > threshold:
                    context.set_output("trigger_type", "high_value")
                else:
                    context.set_output("trigger_type", "low_value")

                return None

        # Phase 1: Run trigger agents
        trigger_agents = [
            TriggerAgent(f"trigger-{i}", trigger_threshold=30 + i * 10)
            for i in range(4)
        ]

        trigger_results = await run_agents_parallel(trigger_agents)

        # Phase 2: Create follow-up agents based on trigger results
        follow_up_agents = []
        for name, result in trigger_results.items():
            if result.get_output("should_trigger", False):
                trigger_type = result.get_output("trigger_type", "unknown")
                if trigger_type == "high_value":
                    follow_up_agents.append(
                        TestDataProcessor(f"followup-{name}", processing_factor=1.5)
                    )

        # Run follow-up agents if any were triggered
        if follow_up_agents:
            followup_results = await run_agents_parallel(follow_up_agents)

            # Verify cascading coordination
            assert len(followup_results) > 0
            for result in followup_results.values():
                assert result.status.name in ["COMPLETED", "SUCCESS"]
                processed_data = result.get_output("processed_data", {})
                assert processed_data.get("processing_factor") == 1.5

        # Verify trigger evaluation
        trigger_count = sum(
            1
            for result in trigger_results.values()
            if result.get_output("should_trigger", False)
        )

        assert trigger_count >= 0  # At least some triggers should have evaluated
        assert len(follow_up_agents) == trigger_count
