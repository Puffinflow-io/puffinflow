"""
Tests for the fluent API coordination module.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from puffinflow.core.agent.base import Agent
from puffinflow.core.coordination.fluent_api import (
    Agents,
    ConditionalAgents,
    FluentResult,
    PipelineAgents,
    collect_agent_outputs,
    create_agent_team,
    create_pipeline,
    get_best_agent,
    run_parallel_agents,
    run_sequential_agents,
)


class TestAgents:
    """Test the Agents fluent API class."""

    def test_agents_creation(self):
        """Test creating an Agents instance."""
        agents = Agents()
        assert agents is not None
        assert hasattr(agents, "_agents")
        assert hasattr(agents, "_stages")

    def test_add_agent(self):
        """Test adding a single agent."""
        agents = Agents()
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "test_agent"

        result = agents.add(mock_agent)
        assert result is agents  # Should return self for chaining
        assert mock_agent in agents._agents

    def test_add_many_agents(self):
        """Test adding multiple agents."""
        agents = Agents()
        mock_agents = [Mock(spec=Agent) for _ in range(3)]
        for i, agent in enumerate(mock_agents):
            agent.name = f"agent_{i}"

        result = agents.add_many(mock_agents)
        assert result is agents  # Should return self for chaining
        assert len(agents._agents) == 3

    def test_add_parallel(self):
        """Test adding agents for parallel execution."""
        agents = Agents()
        mock_agents = [Mock(spec=Agent) for _ in range(2)]
        for i, agent in enumerate(mock_agents):
            agent.name = f"parallel_agent_{i}"

        result = agents.add_parallel(mock_agents)
        assert result is agents
        assert len(agents._stages) == 1
        assert agents._stages[0]["type"] == "parallel"

    def test_then_add(self):
        """Test sequential agent addition."""
        agents = Agents()
        mock_agent1 = Mock(spec=Agent)
        mock_agent1.name = "agent_1"
        mock_agent2 = Mock(spec=Agent)
        mock_agent2.name = "agent_2"

        result = agents.add(mock_agent1).then_add(mock_agent2)
        assert result is agents
        assert len(agents._agents) == 2

    def test_add_stage(self):
        """Test adding named stage."""
        agents = Agents()
        mock_agents = [Mock(spec=Agent) for _ in range(2)]
        for i, agent in enumerate(mock_agents):
            agent.name = f"stage_agent_{i}"

        result = agents.add_stage("test_stage", mock_agents)
        assert result is agents
        assert len(agents._stages) == 1
        assert agents._stages[0]["name"] == "test_stage"

    def test_set_variable_for_all(self):
        """Test setting variable for all agents."""
        agents = Agents()
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "test_agent"
        mock_agent.set_variable = Mock()

        agents.add(mock_agent)
        result = agents.set_variable_for_all("key", "value")

        assert result is agents
        mock_agent.set_variable.assert_called_with("key", "value")

    def test_pipe_variable(self):
        """Test variable piping configuration."""
        agents = Agents()
        result = agents.pipe_variable("output_key", "input_key")
        assert result is agents
        assert agents._pipeline_mode is True

    def test_fan_out(self):
        """Test fan-out configuration."""
        agents = Agents()
        result = agents.fan_out("source_stage", "target_stage")
        assert result is agents
        assert agents._fan_out_config is not None

    def test_aggregate_with(self):
        """Test aggregator configuration."""
        agents = Agents()
        mock_aggregator = Mock(spec=Agent)
        mock_aggregator.name = "aggregator"

        result = agents.aggregate_with(mock_aggregator)
        assert result is agents
        assert agents._aggregator is mock_aggregator

    def test_if_output(self):
        """Test conditional execution setup."""
        agents = Agents()
        result = agents.if_output("agent_name", "key", "expected_value")
        assert isinstance(result, ConditionalAgents)

    @pytest.mark.asyncio
    async def test_run_parallel(self):
        """Test parallel execution."""
        agents = Agents()
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "test_agent"

        # Mock the team creation and execution
        with patch(
            "puffinflow.core.coordination.fluent_api.AgentTeam"
        ) as mock_team_class:
            mock_team = Mock()
            mock_team_class.return_value = mock_team
            mock_team.add_agents = Mock(return_value=mock_team)
            mock_team.set_global_variable = Mock()
            mock_team.run_parallel = AsyncMock(return_value=Mock())

            agents.add(mock_agent)
            result = await agents.run_parallel()

            assert isinstance(result, FluentResult)
            mock_team.add_agents.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_sequential(self):
        """Test sequential execution."""
        agents = Agents()
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "test_agent"

        # Mock the team creation and execution
        with patch(
            "puffinflow.core.coordination.fluent_api.AgentTeam"
        ) as mock_team_class:
            mock_team = Mock()
            mock_team_class.return_value = mock_team
            mock_team.add_agents = Mock(return_value=mock_team)
            mock_team.set_global_variable = Mock()
            mock_team.run_sequential = AsyncMock(return_value=Mock())

            agents.add(mock_agent)
            result = await agents.run_sequential()

            assert isinstance(result, FluentResult)

    def test_pipeline(self):
        """Test pipeline creation."""
        agents = Agents()
        mock_agent1 = Mock(spec=Agent)
        mock_agent2 = Mock(spec=Agent)

        result = agents.pipeline(mock_agent1, mock_agent2)
        assert isinstance(result, PipelineAgents)


class TestConditionalAgents:
    """Test conditional agents functionality."""

    def test_conditional_creation(self):
        """Test creating conditional agents."""
        parent = Agents()
        conditional = ConditionalAgents(parent, "agent_name", "key", "value")
        assert conditional.parent is parent
        assert conditional.condition_agent == "agent_name"
        assert conditional.condition_key == "key"
        assert conditional.expected_value == "value"

    def test_then_add(self):
        """Test adding agent to then branch."""
        parent = Agents()
        conditional = ConditionalAgents(parent, "agent_name", "key", "value")
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "then_agent"

        result = conditional.then_add(mock_agent)
        assert result is conditional
        assert mock_agent in conditional.then_agents

    def test_else_branch(self):
        """Test switching to else branch."""
        parent = Agents()
        conditional = ConditionalAgents(parent, "agent_name", "key", "value")

        result = conditional.else_()
        assert result is conditional
        assert conditional._in_else is True

    def test_endif(self):
        """Test ending conditional and returning to parent."""
        parent = Agents()
        conditional = ConditionalAgents(parent, "agent_name", "key", "value")

        result = conditional.endif()
        assert result is parent
        assert len(parent._conditions) == 1


class TestPipelineAgents:
    """Test pipeline agents functionality."""

    def test_pipeline_creation(self):
        """Test creating pipeline agents."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]
        for i, agent in enumerate(mock_agents):
            agent.name = f"pipeline_agent_{i}"

        pipeline = PipelineAgents(mock_agents)
        assert pipeline.agents == mock_agents
        assert len(pipeline._pipe_configs) == 0

    def test_pipe_output(self):
        """Test configuring output piping."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]
        pipeline = PipelineAgents(mock_agents)

        result = pipeline.pipe_output("agent1", "output_key", "agent2", "input_key")
        assert result is pipeline
        assert len(pipeline._pipe_configs) == 1
        assert pipeline._pipe_configs[0]["from"] == "agent1"

    @pytest.mark.asyncio
    async def test_pipeline_run(self):
        """Test pipeline execution."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]
        for i, agent in enumerate(mock_agents):
            agent.name = f"pipeline_agent_{i}"
            agent.set_variable = Mock()
            agent.run = AsyncMock(return_value=Mock())

        pipeline = PipelineAgents(mock_agents)

        # Mock TeamResult class and FluentResult creation
        with patch(
            "puffinflow.core.coordination.fluent_api.TeamResult"
        ) as mock_team_result_class:
            with patch(
                "puffinflow.core.coordination.fluent_api.FluentResult"
            ) as mock_fluent_result_class:
                # Create mock instances
                mock_team_result_instance = Mock()
                mock_team_result_class.return_value = mock_team_result_instance

                mock_fluent_result_instance = Mock()
                mock_fluent_result_class.return_value = mock_fluent_result_instance

                result = await pipeline.run()
                assert result is mock_fluent_result_instance

                # Verify all agents were run
                for agent in mock_agents:
                    agent.run.assert_called_once()


class TestFluentResult:
    """Test fluent result wrapper."""

    def test_fluent_result_creation(self):
        """Test creating fluent result."""
        mock_team_result = Mock()
        result = FluentResult(mock_team_result)
        assert result._team_result is mock_team_result

    def test_attribute_delegation(self):
        """Test attribute delegation to underlying result."""
        mock_team_result = Mock()
        mock_team_result.test_attr = "test_value"

        result = FluentResult(mock_team_result)
        assert result.test_attr == "test_value"

    def test_get_all_outputs(self):
        """Test getting all outputs."""
        mock_team_result = Mock()
        mock_team_result.get_all_outputs = Mock(return_value=["output1", "output2"])

        result = FluentResult(mock_team_result)
        outputs = result.get_all_outputs("key")
        assert outputs == ["output1", "output2"]

    def test_get_best_by(self):
        """Test getting best agent by metric."""
        mock_team_result = Mock()
        mock_agent_result = Mock()
        mock_team_result.get_best_by = Mock(return_value=mock_agent_result)

        result = FluentResult(mock_team_result)
        best = result.get_best_by("metric", True)
        assert best is mock_agent_result

    def test_filter_successful(self):
        """Test filtering successful results."""
        mock_team_result = Mock()
        mock_agent_result1 = Mock()
        mock_agent_result1.is_success = True
        mock_agent_result2 = Mock()
        mock_agent_result2.is_success = False

        mock_team_result.agent_results = {
            "agent1": mock_agent_result1,
            "agent2": mock_agent_result2,
        }

        result = FluentResult(mock_team_result)
        successful = result.filter_successful()
        assert len(successful) == 1
        assert successful[0] is mock_agent_result1


class TestHelperFunctions:
    """Test helper functions."""

    @pytest.mark.asyncio
    async def test_run_parallel_agents(self):
        """Test parallel agents helper function."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]

        with patch(
            "puffinflow.core.coordination.fluent_api.Agents"
        ) as mock_agents_class:
            mock_agents_instance = Mock()
            mock_agents_class.return_value = mock_agents_instance
            mock_agents_instance.add_many = Mock(return_value=mock_agents_instance)

            # Create a mock FluentResult
            mock_fluent_result = Mock()
            mock_agents_instance.run_parallel = AsyncMock(
                return_value=mock_fluent_result
            )

            result = await run_parallel_agents(*mock_agents)
            assert result is mock_fluent_result

    @pytest.mark.asyncio
    async def test_run_sequential_agents(self):
        """Test sequential agents helper function."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]

        with patch(
            "puffinflow.core.coordination.fluent_api.Agents"
        ) as mock_agents_class:
            mock_agents_instance = Mock()
            mock_agents_class.return_value = mock_agents_instance
            mock_agents_instance.add_many = Mock(return_value=mock_agents_instance)

            # Create a mock FluentResult
            mock_fluent_result = Mock()
            mock_agents_instance.run_sequential = AsyncMock(
                return_value=mock_fluent_result
            )

            result = await run_sequential_agents(*mock_agents)
            assert result is mock_fluent_result

    @pytest.mark.asyncio
    async def test_collect_agent_outputs(self):
        """Test collecting agent outputs helper function."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]

        with patch(
            "puffinflow.core.coordination.fluent_api.Agents"
        ) as mock_agents_class:
            mock_agents_instance = Mock()
            mock_agents_class.return_value = mock_agents_instance
            mock_agents_instance.add_many = Mock(return_value=mock_agents_instance)
            mock_agents_instance.collect_outputs = AsyncMock(
                return_value=["output1", "output2"]
            )

            result = await collect_agent_outputs(mock_agents, "output_key")
            assert result == ["output1", "output2"]

    @pytest.mark.asyncio
    async def test_get_best_agent(self):
        """Test getting best agent helper function."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]
        mock_best_result = Mock()

        with patch(
            "puffinflow.core.coordination.fluent_api.Agents"
        ) as mock_agents_class:
            mock_agents_instance = Mock()
            mock_agents_class.return_value = mock_agents_instance
            mock_agents_instance.add_many = Mock(return_value=mock_agents_instance)
            mock_agents_instance.get_best_by = AsyncMock(return_value=mock_best_result)

            result = await get_best_agent(mock_agents, "metric", True)
            assert result is mock_best_result

    def test_create_pipeline(self):
        """Test creating pipeline helper function."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]

        result = create_pipeline(*mock_agents)
        assert isinstance(result, PipelineAgents)
        assert result.agents == list(mock_agents)

    def test_create_agent_team(self):
        """Test creating agent team helper function."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]

        with patch(
            "puffinflow.core.coordination.fluent_api.AgentTeam"
        ) as mock_team_class:
            mock_team = Mock()
            mock_team_class.return_value = mock_team
            mock_team.add_agents = Mock(return_value=mock_team)

            result = create_agent_team("test_team", mock_agents)
            assert result is mock_team
            mock_team.add_agents.assert_called_with(mock_agents)


class TestAgentsAdvanced:
    """Test advanced Agents functionality."""

    @pytest.mark.asyncio
    async def test_run_with_coordination(self):
        """Test running with coordination using orchestrator."""
        agents = Agents()
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "test_agent"

        agents.add(mock_agent)
        agents.add_stage("stage1", [mock_agent])
        agents.set_variable_for_all("global_var", "global_value")

        with patch(
            "puffinflow.core.coordination.fluent_api.AgentOrchestrator"
        ) as mock_orchestrator_class:
            mock_orchestrator = Mock()
            mock_orchestrator_class.return_value = mock_orchestrator
            mock_orchestrator.add_agents = Mock(return_value=mock_orchestrator)
            mock_orchestrator.add_stage = Mock(return_value=mock_orchestrator)
            mock_orchestrator.set_global_variable = Mock()
            mock_orchestrator.run = AsyncMock(return_value=Mock())

            result = await agents.run_with_coordination()

            assert isinstance(result, FluentResult)
            mock_orchestrator.add_agents.assert_called_once()
            mock_orchestrator.add_stage.assert_called()
            mock_orchestrator.set_global_variable.assert_called_with(
                "global_var", "global_value"
            )

    @pytest.mark.asyncio
    async def test_collect_all(self):
        """Test collect_all method."""
        agents = Agents()
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "test_agent"

        with patch.object(agents, "run_parallel") as mock_run_parallel:
            mock_result = Mock()
            mock_run_parallel.return_value = mock_result

            result = await agents.collect_all()

            assert result is mock_result
            mock_run_parallel.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_collect_outputs(self):
        """Test collect_outputs method."""
        agents = Agents()
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "test_agent"

        with patch.object(agents, "run_parallel") as mock_run_parallel:
            mock_result = Mock()
            mock_result.get_all_outputs = Mock(return_value=["output1", "output2"])
            mock_run_parallel.return_value = mock_result

            outputs = await agents.collect_outputs("test_key")

            assert outputs == ["output1", "output2"]
            mock_result.get_all_outputs.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_wait_for_all(self):
        """Test wait_for_all method."""
        agents = Agents()

        with patch.object(agents, "run_parallel") as mock_run_parallel:
            mock_result = Mock()
            mock_run_parallel.return_value = mock_result

            result = await agents.wait_for_all(timeout=30.0)

            assert result is mock_result
            mock_run_parallel.assert_called_once_with(30.0)

    @pytest.mark.asyncio
    async def test_get_best_by(self):
        """Test get_best_by method."""
        agents = Agents()

        with patch.object(agents, "run_parallel") as mock_run_parallel:
            mock_result = Mock()
            mock_agent_result = Mock()
            mock_result.get_best_by = Mock(return_value=mock_agent_result)
            mock_run_parallel.return_value = mock_result

            best = await agents.get_best_by("score", maximize=False)

            assert best is mock_agent_result
            mock_result.get_best_by.assert_called_once_with("score", False)

    @pytest.mark.asyncio
    async def test_aggregate_stage_results(self):
        """Test aggregate_stage_results method."""
        agents = Agents()

        with patch.object(agents, "run_with_coordination") as mock_run_coordination:
            mock_result = Mock()
            mock_stage_result = Mock()
            mock_stage_result.get_all_outputs = Mock(
                return_value=["result1", "result2"]
            )
            mock_stage_result.success_rate = 0.8
            mock_stage_result.agent_results = {"agent1": Mock(), "agent2": Mock()}

            mock_result.stage_results = {"test_stage": mock_stage_result}
            mock_run_coordination.return_value = mock_result

            aggregated = await agents.aggregate_stage_results("test_stage")

            assert "outputs" in aggregated
            assert "metrics" in aggregated
            assert aggregated["outputs"] == ["result1", "result2"]
            assert aggregated["metrics"]["success_rate"] == 0.8
            assert aggregated["metrics"]["agent_count"] == 2

    @pytest.mark.asyncio
    async def test_aggregate_stage_results_no_stage(self):
        """Test aggregate_stage_results with non-existent stage."""
        agents = Agents()

        with patch.object(agents, "run_with_coordination") as mock_run_coordination:
            mock_result = Mock()
            mock_result.stage_results = {}
            mock_run_coordination.return_value = mock_result

            aggregated = await agents.aggregate_stage_results("nonexistent")

            assert aggregated == {}

    @pytest.mark.asyncio
    async def test_get_final_result_with_stages(self):
        """Test get_final_result with stages."""
        agents = Agents()
        agents.add_stage("stage1", [Mock()])

        with patch.object(agents, "run_with_coordination") as mock_run_coordination:
            mock_result = Mock()
            mock_result.get_final_result = Mock(return_value={"final": "result"})
            mock_run_coordination.return_value = mock_result

            final = await agents.get_final_result()

            assert final == {"final": "result"}

    @pytest.mark.asyncio
    async def test_get_final_result_without_stages(self):
        """Test get_final_result without stages."""
        agents = Agents()

        with patch.object(agents, "run_parallel") as mock_run_parallel:
            mock_result = Mock()
            mock_result.agent_results = {"agent1": Mock()}
            mock_result.success_rate = 0.9
            mock_result.total_duration = 10.5
            mock_run_parallel.return_value = mock_result

            final = await agents.get_final_result()

            assert "agent_results" in final
            assert "success_rate" in final
            assert "total_duration" in final
            assert final["success_rate"] == 0.9
            assert final["total_duration"] == 10.5

    @pytest.mark.asyncio
    async def test_get_aggregated_result_with_aggregator(self):
        """Test get_aggregated_result with aggregator."""
        agents = Agents()
        mock_aggregator = Mock(spec=Agent)
        mock_aggregator.set_variable = Mock()
        mock_aggregator.run = AsyncMock()

        # Create mock aggregator result
        mock_aggregator_result = Mock()
        mock_aggregator_result.outputs = {"aggregated": "result"}
        mock_aggregator_result.execution_duration = 2.0
        mock_aggregator.run.return_value = mock_aggregator_result

        agents.aggregate_with(mock_aggregator)

        with patch.object(agents, "run_parallel") as mock_run_parallel:
            mock_main_result = Mock()
            mock_agent_result = Mock()
            mock_agent_result.outputs = {"output1": "value1", "output2": "value2"}
            mock_main_result.agent_results = {"agent1": mock_agent_result}
            mock_main_result.total_duration = 5.0
            mock_run_parallel.return_value = mock_main_result

            result = await agents.get_aggregated_result()

            assert "main_results" in result
            assert "aggregated_result" in result
            assert "total_duration" in result
            assert result["aggregated_result"] == {"aggregated": "result"}
            assert result["total_duration"] == 7.0  # 5.0 + 2.0

            # Verify aggregator was called with input data
            mock_aggregator.set_variable.assert_called_once()
            call_args = mock_aggregator.set_variable.call_args
            assert call_args[0][0] == "input_data"
            assert "value1" in call_args[0][1]
            assert "value2" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_get_aggregated_result_without_aggregator(self):
        """Test get_aggregated_result without aggregator."""
        agents = Agents()

        with patch.object(agents, "get_final_result") as mock_get_final:
            mock_final_result = {"final": "result"}
            mock_get_final.return_value = mock_final_result

            result = await agents.get_aggregated_result()

            assert result == {"final": "result"}

    @pytest.mark.asyncio
    async def test_run_conditional(self):
        """Test run_conditional method."""
        agents = Agents()

        with patch.object(agents, "run_sequential") as mock_run_sequential:
            mock_result = Mock()
            mock_run_sequential.return_value = mock_result

            result = await agents.run_conditional()

            assert result is mock_result

    def test_then_add_with_existing_stage_agents(self):
        """Test then_add when there are existing stage agents."""
        agents = Agents()
        mock_agent1 = Mock(spec=Agent)
        mock_agent1.name = "agent1"
        mock_agent2 = Mock(spec=Agent)
        mock_agent2.name = "agent2"

        # Add first agent, then use then_add
        agents.add(mock_agent1)
        agents.then_add(mock_agent2)

        # Should create a stage for the first agent
        assert len(agents._stages) == 1
        assert agents._stages[0]["type"] == "sequential"
        assert agents._stages[0]["agents"] == [mock_agent1]

        # Current stage agents should now contain only the second agent
        assert agents._current_stage_agents == [mock_agent2]


class TestConditionalAgentsAdvanced:
    """Test advanced ConditionalAgents functionality."""

    def test_then_add_in_else_branch(self):
        """Test adding agent in else branch."""
        parent = Agents()
        conditional = ConditionalAgents(parent, "agent_name", "key", "value")
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "else_agent"

        # Switch to else branch and add agent
        conditional.else_().then_add(mock_agent)

        assert mock_agent in conditional.else_agents
        assert mock_agent not in conditional.then_agents

    def test_conditional_full_workflow(self):
        """Test complete conditional workflow."""
        parent = Agents()
        conditional = ConditionalAgents(
            parent, "condition_agent", "output_key", "expected"
        )

        mock_then_agent = Mock(spec=Agent)
        mock_then_agent.name = "then_agent"
        mock_else_agent = Mock(spec=Agent)
        mock_else_agent.name = "else_agent"

        # Build conditional
        result = (
            conditional.then_add(mock_then_agent)
            .else_()
            .then_add(mock_else_agent)
            .endif()
        )

        assert result is parent
        assert len(parent._conditions) == 1

        condition = parent._conditions[0]
        assert condition["agent"] == "condition_agent"
        assert condition["key"] == "output_key"
        assert condition["expected"] == "expected"
        assert condition["then"] == [mock_then_agent]
        assert condition["else"] == [mock_else_agent]


class TestPipelineAgentsAdvanced:
    """Test advanced PipelineAgents functionality."""

    @pytest.mark.asyncio
    async def test_pipeline_run_with_piping(self):
        """Test pipeline execution with output piping."""
        mock_agent1 = Mock(spec=Agent)
        mock_agent1.name = "agent1"
        mock_agent1.set_variable = Mock()

        mock_agent2 = Mock(spec=Agent)
        mock_agent2.name = "agent2"
        mock_agent2.set_variable = Mock()

        # Create mock results
        mock_result1 = Mock()
        mock_result1.get_output = Mock(return_value="piped_value")
        mock_result2 = Mock()

        mock_agent1.run = AsyncMock(return_value=mock_result1)
        mock_agent2.run = AsyncMock(return_value=mock_result2)

        pipeline = PipelineAgents([mock_agent1, mock_agent2])
        pipeline.pipe_output("agent1", "output_key", "agent2", "input_key")

        with patch(
            "puffinflow.core.coordination.fluent_api.TeamResult"
        ) as mock_team_result_class:
            with patch(
                "puffinflow.core.coordination.fluent_api.FluentResult"
            ) as mock_fluent_result_class:
                mock_team_result_instance = Mock()
                mock_team_result_class.return_value = mock_team_result_instance

                mock_fluent_result_instance = Mock()
                mock_fluent_result_class.return_value = mock_fluent_result_instance

                await pipeline.run()

                # Verify piping occurred
                mock_agent2.set_variable.assert_called_once_with(
                    "input_key", "piped_value"
                )

                # Verify both agents ran
                mock_agent1.run.assert_called_once()
                mock_agent2.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_pipeline_run_with_none_output(self):
        """Test pipeline execution when output is None."""
        mock_agent1 = Mock(spec=Agent)
        mock_agent1.name = "agent1"

        mock_agent2 = Mock(spec=Agent)
        mock_agent2.name = "agent2"
        mock_agent2.set_variable = Mock()

        # Create mock result with None output
        mock_result1 = Mock()
        mock_result1.get_output = Mock(return_value=None)
        mock_result2 = Mock()

        mock_agent1.run = AsyncMock(return_value=mock_result1)
        mock_agent2.run = AsyncMock(return_value=mock_result2)

        pipeline = PipelineAgents([mock_agent1, mock_agent2])
        pipeline.pipe_output("agent1", "output_key", "agent2", "input_key")

        with patch("puffinflow.core.coordination.fluent_api.TeamResult"):
            with patch("puffinflow.core.coordination.fluent_api.FluentResult"):
                await pipeline.run()

                # Should not set variable when output is None
                mock_agent2.set_variable.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_run_single_agent(self):
        """Test pipeline execution with single agent."""
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "single_agent"
        mock_agent.run = AsyncMock(return_value=Mock())

        pipeline = PipelineAgents([mock_agent])

        with patch("puffinflow.core.coordination.fluent_api.TeamResult"):
            with patch("puffinflow.core.coordination.fluent_api.FluentResult"):
                await pipeline.run()

                mock_agent.run.assert_called_once()

    def test_get_final_output(self):
        """Test get_final_output method."""
        pipeline = PipelineAgents([Mock()])

        # This method is not implemented, should return None
        result = pipeline.get_final_output()
        assert result is None


class TestFluentResultAdvanced:
    """Test advanced FluentResult functionality."""

    def test_fluent_result_with_non_team_result(self):
        """Test FluentResult with non-TeamResult object."""
        mock_other_result = Mock()
        mock_other_result.some_attr = "test_value"

        result = FluentResult(mock_other_result)

        # Should still delegate to the object
        assert result.some_attr == "test_value"

    def test_get_all_outputs_no_method(self):
        """Test get_all_outputs when underlying result doesn't have the method."""
        mock_result = Mock()
        # Remove get_all_outputs method if it exists
        if hasattr(mock_result, "get_all_outputs"):
            delattr(mock_result, "get_all_outputs")

        result = FluentResult(mock_result)
        outputs = result.get_all_outputs("key")

        assert outputs == []

    def test_get_best_by_no_method(self):
        """Test get_best_by when underlying result doesn't have the method."""
        mock_result = Mock()
        # Remove get_best_by method if it exists
        if hasattr(mock_result, "get_best_by"):
            delattr(mock_result, "get_best_by")

        result = FluentResult(mock_result)
        best = result.get_best_by("metric")

        assert best is None

    def test_filter_successful_no_agent_results(self):
        """Test filter_successful when no agent_results attribute."""
        mock_result = Mock()
        # Remove agent_results attribute completely
        if hasattr(mock_result, "agent_results"):
            delattr(mock_result, "agent_results")

        result = FluentResult(mock_result)
        successful = result.filter_successful()

        assert successful == []

    def test_group_by_status(self):
        """Test group_by_status method."""
        mock_result1 = Mock()
        mock_result1.status = Mock()
        mock_result1.status.value = "completed"

        mock_result2 = Mock()
        mock_result2.status = Mock()
        mock_result2.status.value = "failed"

        mock_result3 = Mock()
        mock_result3.status = "running"  # String status

        mock_team_result = Mock()
        mock_team_result.agent_results = {
            "agent1": mock_result1,
            "agent2": mock_result2,
            "agent3": mock_result3,
        }

        result = FluentResult(mock_team_result)
        groups = result.group_by_status()

        assert "completed" in groups
        assert "failed" in groups
        assert "running" in groups
        assert len(groups["completed"]) == 1
        assert len(groups["failed"]) == 1
        assert len(groups["running"]) == 1

    def test_group_by_status_no_agent_results(self):
        """Test group_by_status when no agent_results."""
        mock_result = Mock()
        # Remove agent_results attribute completely
        if hasattr(mock_result, "agent_results"):
            delattr(mock_result, "agent_results")

        result = FluentResult(mock_result)
        groups = result.group_by_status()

        assert groups == {}


class TestHelperFunctionsAdvanced:
    """Test advanced helper functions."""

    @pytest.mark.asyncio
    async def test_run_parallel_agents_with_timeout(self):
        """Test run_parallel_agents with timeout."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]

        with patch(
            "puffinflow.core.coordination.fluent_api.Agents"
        ) as mock_agents_class:
            mock_agents_instance = Mock()
            mock_agents_class.return_value = mock_agents_instance
            mock_agents_instance.add_many = Mock(return_value=mock_agents_instance)
            mock_agents_instance.run_parallel = AsyncMock(return_value=Mock())

            await run_parallel_agents(*mock_agents, timeout=60.0)

            mock_agents_instance.run_parallel.assert_called_once_with(60.0)

    @pytest.mark.asyncio
    async def test_get_best_agent_minimize(self):
        """Test get_best_agent with minimize=False."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]

        with patch(
            "puffinflow.core.coordination.fluent_api.Agents"
        ) as mock_agents_class:
            mock_agents_instance = Mock()
            mock_agents_class.return_value = mock_agents_instance
            mock_agents_instance.add_many = Mock(return_value=mock_agents_instance)
            mock_agents_instance.get_best_by = AsyncMock(return_value=Mock())

            await get_best_agent(mock_agents, "error_rate", maximize=False)

            mock_agents_instance.get_best_by.assert_called_once_with(
                "error_rate", False
            )

    def test_create_agent_team_return_value(self):
        """Test create_agent_team return value."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]

        with patch(
            "puffinflow.core.coordination.fluent_api.AgentTeam"
        ) as mock_team_class:
            mock_team = Mock()
            mock_team_class.return_value = mock_team
            mock_team.add_agents = Mock(return_value="team_with_agents")

            result = create_agent_team("test_team", mock_agents)

            # Should return the result of add_agents, not the team itself
            assert result == "team_with_agents"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
