"""Agent groups for advanced coordination patterns."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..agent.base import Agent, AgentResult
from .agent_team import AgentTeam, TeamResult

if TYPE_CHECKING:
    import asyncio

logger = logging.getLogger(__name__)


class ExecutionStrategy:
    """Execution strategy constants."""

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    PIPELINE = "pipeline"
    FAN_OUT = "fan_out"
    FAN_IN = "fan_in"
    CONDITIONAL = "conditional"


@dataclass
class StageConfig:
    """Configuration for execution stage."""

    name: str
    agents: list[str]
    strategy: str = ExecutionStrategy.PARALLEL
    depends_on: Optional[list[str]] = None
    condition: Optional[Callable] = None
    timeout: Optional[float] = None

    def __post_init__(self) -> None:
        if self.depends_on is None:
            self.depends_on = []


class AgentGroup:
    """Simple agent group for basic parallel execution."""

    def __init__(self, agents: list[Agent]):
        self.agents = {agent.name: agent for agent in agents}
        self._results: dict[str, AgentResult] = {}

    async def run_parallel(self, timeout: Optional[float] = None) -> "GroupResult":
        """Run all agents in parallel."""
        team = AgentTeam("group_execution")
        team.add_agents(list(self.agents.values()))

        result = await team.run_parallel(timeout)
        return GroupResult(result)

    async def collect_all(self) -> "GroupResult":
        """Run and collect all results."""
        return await self.run_parallel()


class GroupResult:
    """Wrapper for team results with additional methods."""

    def __init__(self, team_result: TeamResult):
        self._team_result = team_result

    def __getattr__(self, name: str) -> Any:
        """Delegate to team result."""
        return getattr(self._team_result, name)

    @property
    def agents(self) -> list[AgentResult]:
        """Get list of agent results."""
        return list(self._team_result.agent_results.values())


class ParallelAgentGroup(AgentGroup):
    """Enhanced parallel agent group."""

    def __init__(self, name: str):
        self.name = name
        self._agents: list[Agent] = []

    def add_agent(self, agent: Agent) -> "ParallelAgentGroup":
        """Add agent to group."""
        self._agents.append(agent)
        return self

    async def run_and_collect(self, timeout: Optional[float] = None) -> GroupResult:
        """Run all agents and collect results."""
        return await AgentGroup(self._agents).run_parallel(timeout)


class AgentOrchestrator:
    """Advanced agent orchestrator with multiple execution strategies."""

    def __init__(self, name: str):
        self.name = name
        self._agents: dict[str, Agent] = {}
        self._stages: list[StageConfig] = []
        self._global_variables: dict[str, Any] = {}
        self._stage_results: dict[str, TeamResult] = {}
        self._execution_context: dict[str, Any] = {}

    def add_agent(self, agent: Agent) -> "AgentOrchestrator":
        """Add agent to orchestrator."""
        self._agents[agent.name] = agent
        return self

    def add_agents(self, agents: list[Agent]) -> "AgentOrchestrator":
        """Add multiple agents."""
        for agent in agents:
            self.add_agent(agent)
        return self

    def add_stage(
        self,
        name: str,
        agents: list[Agent],
        strategy: str = ExecutionStrategy.PARALLEL,
        depends_on: Optional[list[str]] = None,
        condition: Optional[Callable] = None,
        timeout: Optional[float] = None,
    ) -> "AgentOrchestrator":
        """Add execution stage."""
        # Add agents if not already added
        for agent in agents:
            if agent.name not in self._agents:
                self.add_agent(agent)

        stage = StageConfig(
            name=name,
            agents=[agent.name for agent in agents],
            strategy=strategy,
            depends_on=depends_on or [],
            condition=condition,
            timeout=timeout,
        )

        self._stages.append(stage)
        return self

    def set_global_variable(self, key: str, value: Any) -> None:
        """Set global variable for all agents."""
        self._global_variables[key] = value
        for agent in self._agents.values():
            agent.set_shared_variable(key, value)

    def get_global_variable(self, key: str, default: Any = None) -> Any:
        """Get global variable."""
        return self._global_variables.get(key, default)

    def run_with_monitoring(self) -> "OrchestrationExecution":
        """Run with monitoring capability."""
        return OrchestrationExecution(self)

    async def run(self) -> "OrchestrationResult":
        """Run the complete orchestration."""
        async with self.run_with_monitoring() as execution:
            result = await execution.wait_for_completion()
            return result


class OrchestrationExecution:
    """Context manager for orchestration execution with monitoring."""

    def __init__(self, orchestrator: AgentOrchestrator):
        self.orchestrator = orchestrator
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._stage_results: dict[str, TeamResult] = {}
        self._completed_stages: set[str] = set()
        self._running_stages: dict[str, asyncio.Task] = {}
        self._stage_timings: dict[str, tuple] = {}

    async def __aenter__(self) -> "OrchestrationExecution":
        """Start execution."""
        import time

        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """End execution."""
        import time

        self.end_time = time.time()

        # Cancel any running stages
        for task in self._running_stages.values():
            if not task.done():
                task.cancel()

    async def wait_for_stage(self, stage_name: str) -> TeamResult:
        """Wait for specific stage to complete."""
        if stage_name in self._completed_stages:
            return self._stage_results[stage_name]

        # Find stage config
        stage_config = None
        for stage in self.orchestrator._stages:
            if stage.name == stage_name:
                stage_config = stage
                break

        if not stage_config:
            raise ValueError(f"Stage {stage_name} not found")

        # Check dependencies
        for dep in stage_config.depends_on or []:
            if dep not in self._completed_stages:
                await self.wait_for_stage(dep)

        # Check condition
        if stage_config.condition and not stage_config.condition():
            # Create empty result for skipped stage
            result = TeamResult(
                team_name=f"{self.orchestrator.name}_{stage_name}", status="skipped"
            )
            self._stage_results[stage_name] = result
            self._completed_stages.add(stage_name)
            return result

        # Execute stage
        return await self._execute_stage(stage_config)

    async def _execute_stage(self, stage_config: StageConfig) -> TeamResult:
        """Execute a single stage."""
        import time

        stage_start = time.time()

        # Get agents for this stage
        stage_agents = [
            self.orchestrator._agents[name]
            for name in stage_config.agents
            if name in self.orchestrator._agents
        ]

        if not stage_agents:
            raise ValueError(f"No valid agents found for stage {stage_config.name}")

        # Create team for stage
        team = AgentTeam(f"{self.orchestrator.name}_{stage_config.name}")
        team.add_agents(stage_agents)

        # Set global variables
        for key, value in self.orchestrator._global_variables.items():
            team.set_global_variable(key, value)

        # Execute based on strategy
        if stage_config.strategy == ExecutionStrategy.PARALLEL:
            result = await team.run_parallel(stage_config.timeout)
        elif stage_config.strategy == ExecutionStrategy.SEQUENTIAL:
            result = await team.run_sequential()
        else:
            # Default to parallel
            result = await team.run_parallel(stage_config.timeout)

        stage_end = time.time()

        # Store results
        self._stage_results[stage_config.name] = result
        self._completed_stages.add(stage_config.name)
        self._stage_timings[stage_config.name] = (stage_start, stage_end)

        return result

    def set_stage_input(self, stage_name: str, key: str, value: Any) -> None:
        """Set input for a stage."""
        # Set variable for all agents in the stage
        stage_config = None
        for stage in self.orchestrator._stages:
            if stage.name == stage_name:
                stage_config = stage
                break

        if stage_config:
            for agent_name in stage_config.agents:
                agent = self.orchestrator._agents.get(agent_name)
                if agent:
                    agent.set_variable(key, value)

    def is_stage_complete(self, stage_name: str) -> bool:
        """Check if stage is complete."""
        return stage_name in self._completed_stages

    def get_stage_results(self, stage_name: str) -> Optional[TeamResult]:
        """Get results for a stage."""
        return self._stage_results.get(stage_name)

    async def get_final_results(self) -> "OrchestrationResult":
        """Get final orchestration results."""
        # Wait for all stages
        for stage in self.orchestrator._stages:
            if stage.name not in self._completed_stages:
                await self.wait_for_stage(stage.name)

        return OrchestrationResult(
            orchestrator_name=self.orchestrator.name,
            stage_results=self._stage_results,
            stage_timings=self._stage_timings,
            total_duration=self.total_duration,
        )

    async def wait_for_completion(self) -> "OrchestrationResult":
        """Wait for complete orchestration."""
        return await self.get_final_results()

    @property
    def total_duration(self) -> Optional[float]:
        """Get total execution duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def get_stage_timings(self) -> dict[str, float]:
        """Get timing for each stage."""
        return {
            stage: end - start for stage, (start, end) in self._stage_timings.items()
        }


@dataclass
class OrchestrationResult:
    """Result of orchestration execution."""

    orchestrator_name: str
    stage_results: dict[str, TeamResult]
    stage_timings: dict[str, tuple]
    total_duration: Optional[float]

    def get_stage_result(self, stage_name: str) -> Optional[TeamResult]:
        """Get result for specific stage."""
        return self.stage_results.get(stage_name)

    def get_final_result(self) -> dict[str, Any]:
        """Get final aggregated result."""
        # Combine results from all stages
        all_outputs = {}
        all_variables = {}

        for stage_name, stage_result in self.stage_results.items():
            # Collect outputs from all agents in stage
            for agent_name, agent_result in stage_result.agent_results.items():
                all_outputs[f"{stage_name}_{agent_name}"] = agent_result.outputs
                all_variables[f"{stage_name}_{agent_name}"] = agent_result.variables

        return {
            "outputs": all_outputs,
            "variables": all_variables,
            "stage_count": len(self.stage_results),
            "total_duration": self.total_duration,
            "stage_durations": {
                stage: end - start for stage, (start, end) in self.stage_timings.items()
            },
        }
