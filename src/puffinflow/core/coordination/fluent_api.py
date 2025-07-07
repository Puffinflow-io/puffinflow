"""Fluent APIs for intuitive agent coordination."""

from collections import defaultdict
from typing import Any, Optional, Union

from ..agent.base import Agent, AgentResult
from .agent_group import AgentOrchestrator, ExecutionStrategy
from .agent_team import AgentTeam, TeamResult


class Agents:
    """Fluent API for agent coordination."""

    def __init__(self) -> None:
        self._agents: list[Agent] = []
        self._stages: list[dict] = []
        self._conditions: list[dict] = []
        self._current_stage_agents: list[Agent] = []
        self._pipeline_mode = False
        self._fan_out_config: Optional[dict] = None
        self._aggregator: Optional[Agent] = None
        self._variables: dict[str, Any] = {}

    def add(self, agent: Agent) -> "Agents":
        """Add a single agent."""
        self._agents.append(agent)
        self._current_stage_agents.append(agent)
        return self

    def add_many(self, agents: list[Agent]) -> "Agents":
        """Add multiple agents."""
        self._agents.extend(agents)
        self._current_stage_agents.extend(agents)
        return self

    def add_parallel(self, agents: list[Agent]) -> "Agents":
        """Add agents to run in parallel."""
        self._stages.append({"type": "parallel", "agents": agents.copy()})
        self._agents.extend(agents)
        return self

    def then_add(self, agent: Agent) -> "Agents":
        """Add agent for sequential execution."""
        if self._current_stage_agents:
            self._stages.append(
                {"type": "sequential", "agents": self._current_stage_agents.copy()}
            )
            self._current_stage_agents.clear()

        self._current_stage_agents.append(agent)
        self._agents.append(agent)
        return self

    def add_stage(self, name: str, agents: list[Agent]) -> "Agents":
        """Add named stage."""
        self._stages.append({"type": "stage", "name": name, "agents": agents.copy()})
        self._agents.extend(agents)
        return self

    def set_variable_for_all(self, key: str, value: Any) -> "Agents":
        """Set variable for all agents."""
        self._variables[key] = value
        for agent in self._agents:
            agent.set_variable(key, value)
        return self

    def pipe_variable(self, output_key: str, input_key: str) -> "Agents":
        """Pipe output from one stage to input of next."""
        self._pipeline_mode = True
        # This would be implemented in the execution phase
        return self

    def fan_out(self, source_stage: str, target_stage: str) -> "Agents":
        """Configure fan-out from one stage to another."""
        self._fan_out_config = {"source": source_stage, "target": target_stage}
        return self

    def aggregate_with(self, aggregator: Agent) -> "Agents":
        """Set aggregator agent."""
        self._aggregator = aggregator
        return self

    def if_output(
        self, agent_name: str, key: str, expected_value: Any
    ) -> "ConditionalAgents":
        """Start conditional execution."""
        return ConditionalAgents(self, agent_name, key, expected_value)

    # Execution methods
    async def run_parallel(self, timeout: Optional[float] = None) -> "FluentResult":
        """Run all agents in parallel."""
        team = AgentTeam("fluent_parallel")
        team.add_agents(self._agents)

        # Set variables
        for key, value in self._variables.items():
            team.set_global_variable(key, value)

        result = await team.run_parallel(timeout)
        return FluentResult(result)

    async def run_sequential(self) -> "FluentResult":
        """Run agents sequentially."""
        team = AgentTeam("fluent_sequential")
        team.add_agents(self._agents)

        # Set variables
        for key, value in self._variables.items():
            team.set_global_variable(key, value)

        result = await team.run_sequential()
        return FluentResult(result)

    async def run_with_coordination(self) -> "FluentResult":
        """Run with advanced coordination."""
        orchestrator = AgentOrchestrator("fluent_orchestration")
        orchestrator.add_agents(self._agents)

        # Add stages
        for i, stage_config in enumerate(self._stages):
            stage_name = stage_config.get("name", f"stage_{i}")
            agents = stage_config["agents"]
            stage_type = stage_config["type"]

            strategy = ExecutionStrategy.PARALLEL
            if stage_type == "sequential":
                strategy = ExecutionStrategy.SEQUENTIAL

            orchestrator.add_stage(stage_name, agents, strategy)

        # Set variables
        for key, value in self._variables.items():
            orchestrator.set_global_variable(key, value)

        result = await orchestrator.run()
        return FluentResult(result)

    async def collect_all(self) -> "FluentResult":
        """Run and collect all results."""
        return await self.run_parallel()

    async def collect_outputs(self, key: str) -> list[Any]:
        """Collect specific output from all agents."""
        result = await self.run_parallel()
        return result.get_all_outputs(key)

    async def wait_for_all(self, timeout: Optional[float] = None) -> "FluentResult":
        """Wait for all agents to complete."""
        return await self.run_parallel(timeout)

    async def get_best_by(self, metric: str, maximize: bool = True) -> AgentResult:
        """Get agent with best metric."""
        result = await self.run_parallel()
        best_result = result.get_best_by(metric, maximize)
        if best_result is None:
            raise ValueError(f"No agent found with metric '{metric}'")
        return best_result

    async def aggregate_stage_results(self, stage_name: str) -> dict[str, Any]:
        """Aggregate results from a specific stage."""
        result = await self.run_with_coordination()

        if hasattr(result, "stage_results"):
            stage_result = result.stage_results.get(stage_name)
            if stage_result:
                return {
                    "outputs": stage_result.get_all_outputs("result"),
                    "metrics": {
                        "success_rate": stage_result.success_rate,
                        "agent_count": len(stage_result.agent_results),
                    },
                }

        return {}

    async def get_final_result(self) -> dict[str, Any]:
        """Get final aggregated result."""
        if self._stages:
            result = await self.run_with_coordination()
            if hasattr(result, "get_final_result"):
                final_result = result.get_final_result()
                if isinstance(final_result, dict):
                    return final_result
                else:
                    return {"result": final_result}

        team_result = await self.run_parallel()
        return {
            "agent_results": team_result.agent_results,
            "success_rate": team_result.success_rate,
            "total_duration": team_result.total_duration,
        }

    async def get_aggregated_result(self) -> dict[str, Any]:
        """Get result aggregated by aggregator agent."""
        if self._aggregator:
            # Run main agents first
            main_result = await self.run_parallel()

            # Collect outputs for aggregator
            all_outputs = []
            for agent_result in main_result.agent_results.values():
                all_outputs.extend(agent_result.outputs.values())

            # Run aggregator
            self._aggregator.set_variable("input_data", all_outputs)
            aggregator_result = await self._aggregator.run()

            return {
                "main_results": main_result.agent_results,
                "aggregated_result": aggregator_result.outputs,
                "total_duration": main_result.total_duration
                + aggregator_result.execution_duration,
            }

        return await self.get_final_result()

    # Pipeline methods
    def pipeline(self, *agents: Agent) -> "PipelineAgents":
        """Create pipeline of agents."""
        return PipelineAgents(list(agents))

    # Conditional execution
    async def run_conditional(self) -> "FluentResult":
        """Run with conditional logic."""
        # Execute conditions and build execution plan
        for _condition in self._conditions:
            # Evaluate condition and add appropriate agents
            pass

        return await self.run_sequential()


class ConditionalAgents:
    """Conditional execution builder."""

    def __init__(self, parent: Agents, agent_name: str, key: str, expected_value: Any):
        self.parent = parent
        self.condition_agent = agent_name
        self.condition_key = key
        self.expected_value = expected_value
        self.then_agents: list[Agent] = []
        self.else_agents: list[Agent] = []
        self._in_else = False

    def then_add(self, agent: Agent) -> "ConditionalAgents":
        """Add agent for 'then' branch."""
        if not self._in_else:
            self.then_agents.append(agent)
        else:
            self.else_agents.append(agent)
        return self

    def else_(self) -> "ConditionalAgents":
        """Switch to 'else' branch."""
        self._in_else = True
        return self

    def endif(self) -> Agents:
        """End conditional and return to main builder."""
        # Store conditional configuration
        self.parent._conditions.append(
            {
                "agent": self.condition_agent,
                "key": self.condition_key,
                "expected": self.expected_value,
                "then": self.then_agents,
                "else": self.else_agents,
            }
        )
        return self.parent


class PipelineAgents:
    """Pipeline execution builder."""

    def __init__(self, agents: list[Agent]):
        self.agents = agents
        self._pipe_configs: list[dict] = []

    def pipe_output(
        self, from_agent: str, output_key: str, to_agent: str, input_key: str
    ) -> "PipelineAgents":
        """Configure output piping between agents."""
        self._pipe_configs.append(
            {
                "from": from_agent,
                "output_key": output_key,
                "to": to_agent,
                "input_key": input_key,
            }
        )
        return self

    async def run(self) -> "FluentResult":
        """Run pipeline."""
        results: dict[str, Any] = {}

        for i, agent in enumerate(self.agents):
            # Set up inputs from previous agent if configured
            if i > 0:
                prev_agent = self.agents[i - 1]
                prev_result = results.get(prev_agent.name)

                if prev_result:
                    # Find pipe configuration
                    for pipe_config in self._pipe_configs:
                        if (
                            pipe_config["from"] == prev_agent.name
                            and pipe_config["to"] == agent.name
                        ):
                            output_value = prev_result.get_output(
                                pipe_config["output_key"]
                            )
                            if output_value is not None:
                                agent.set_variable(
                                    pipe_config["input_key"], output_value
                                )

            # Run agent
            result = await agent.run()
            results[agent.name] = result

        # Create team result
        team_result = TeamResult(
            team_name="pipeline", status="completed", agent_results=results
        )

        return FluentResult(team_result)

    def get_final_output(self) -> Any:
        """Get output from last agent."""
        # This would be implemented after execution
        pass


class FluentResult:
    """Enhanced result wrapper with fluent methods."""

    def __init__(self, result: Union[TeamResult, Any]):
        if isinstance(result, TeamResult):
            self._team_result = result
        else:
            # Handle other result types
            self._team_result = result

    def __getattr__(self, name: str) -> Any:
        """Delegate to underlying result."""
        return getattr(self._team_result, name)

    def get_all_outputs(self, key: str) -> list[Any]:
        """Get specific output from all agents."""
        if hasattr(self._team_result, "get_all_outputs"):
            return self._team_result.get_all_outputs(key)
        return []

    def get_best_by(self, metric: str, maximize: bool = True) -> Optional[AgentResult]:
        """Get best agent by metric."""
        if hasattr(self._team_result, "get_best_by"):
            return self._team_result.get_best_by(metric, maximize)
        return None

    def filter_successful(self) -> list[AgentResult]:
        """Get only successful agent results."""
        if hasattr(self._team_result, "agent_results"):
            return [
                result
                for result in self._team_result.agent_results.values()
                if result.is_success
            ]
        return []

    def group_by_status(self) -> dict[str, list[AgentResult]]:
        """Group results by status."""
        groups = defaultdict(list)
        if hasattr(self._team_result, "agent_results"):
            for result in self._team_result.agent_results.values():
                status = (
                    result.status.value
                    if hasattr(result.status, "value")
                    else str(result.status)
                )
                groups[status].append(result)
        return dict(groups)


# Helper functions for one-liner patterns
async def run_parallel_agents(
    *agents: Agent, timeout: Optional[float] = None
) -> FluentResult:
    """One-liner for parallel execution."""
    return await Agents().add_many(list(agents)).run_parallel(timeout)


async def run_sequential_agents(*agents: Agent) -> FluentResult:
    """One-liner for sequential execution."""
    return await Agents().add_many(list(agents)).run_sequential()


async def collect_agent_outputs(agents: list[Agent], output_key: str) -> list[Any]:
    """One-liner to collect specific output from agents."""
    return await Agents().add_many(agents).collect_outputs(output_key)


async def get_best_agent(
    agents: list[Agent], metric: str, maximize: bool = True
) -> AgentResult:
    """One-liner to get best performing agent."""
    return await Agents().add_many(agents).get_best_by(metric, maximize)


# Pipeline helpers
def create_pipeline(*agents: Agent) -> PipelineAgents:
    """Create agent pipeline."""
    return PipelineAgents(list(agents))


def create_agent_team(name: str, agents: list[Agent]) -> AgentTeam:
    """Create agent team."""
    team = AgentTeam(name)
    return team.add_agents(agents)
