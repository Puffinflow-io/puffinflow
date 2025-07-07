"""Agent team coordination with messaging and event systems."""

import asyncio
import contextlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..agent.base import Agent, AgentResult
from ..agent.state import AgentStatus

logger = logging.getLogger(__name__)


@dataclass
class TeamResult:
    """Result container for team execution."""

    team_name: str
    status: str
    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    error: Optional[Exception] = None

    def get_agent_result(self, agent_name: str) -> Optional[AgentResult]:
        """Get result for specific agent."""
        return self.agent_results.get(agent_name)

    def get_all_outputs(self, key: str) -> list[Any]:
        """Get specific output from all agents."""
        return [
            result.get_output(key)
            for result in self.agent_results.values()
            if result.get_output(key) is not None
        ]

    def get_all_variables(self, key: str) -> list[Any]:
        """Get specific variable from all agents."""
        return [
            result.get_variable(key)
            for result in self.agent_results.values()
            if result.get_variable(key) is not None
        ]

    def get_best_by(self, metric: str, maximize: bool = True) -> Optional[AgentResult]:
        """Get agent with best metric value."""
        valid_results = [
            result
            for result in self.agent_results.values()
            if result.get_output(metric) is not None
            or result.get_metric(metric) is not None
        ]

        if not valid_results:
            return None

        def get_value(result: Any) -> Any:
            return result.get_output(metric) or result.get_metric(metric) or 0

        return (
            max(valid_results, key=get_value)
            if maximize
            else min(valid_results, key=get_value)
        )

    def average(self, key: str) -> float:
        """Get average of a numeric output/metric across agents."""
        values = []
        for result in self.agent_results.values():
            value = result.get_output(key) or result.get_metric(key)
            if isinstance(value, (int, float)):
                values.append(value)

        return sum(values) / len(values) if values else 0.0

    def sum(self, key: str) -> float:
        """Get sum of a numeric output/metric across agents."""
        total = 0.0
        for result in self.agent_results.values():
            value = result.get_output(key) or result.get_metric(key)
            if isinstance(value, (int, float)):
                total += value
        return total

    def count_successful(self) -> int:
        """Count successful agent executions."""
        return sum(1 for result in self.agent_results.values() if result.is_success)

    def count_failed(self) -> int:
        """Count failed agent executions."""
        return sum(1 for result in self.agent_results.values() if result.is_failed)

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        total = len(self.agent_results)
        return (self.count_successful() / total * 100) if total > 0 else 0.0


@dataclass
class Message:
    """Message between agents."""

    sender: str
    recipient: str
    message_type: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None


@dataclass
class Event:
    """Event emitted by agents."""

    source: str
    event_type: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: f"event_{int(time.time() * 1000)}")


class EventBus:
    """Event bus for agent communication."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = defaultdict(list)
        self._event_history: list[Event] = []
        self._max_history = 1000

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to event type."""
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from event type."""
        if event_type in self._handlers:
            with contextlib.suppress(ValueError):
                self._handlers[event_type].remove(handler)

    async def emit(self, event: Event) -> None:
        """Emit an event to all subscribers."""
        self._event_history.append(event)

        # Trim history if needed
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history :]

        # Notify handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type}: {e}")

    def get_events(
        self, event_type: Optional[str] = None, source: Optional[str] = None
    ) -> list[Event]:
        """Get events by type and/or source."""
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if source:
            events = [e for e in events if e.source == source]

        return events


class AgentTeam:
    """Enhanced agent team with coordination features."""

    def __init__(self, name: str):
        self.name = name
        self._agents: dict[str, Agent] = {}
        self._shared_context: dict[str, Any] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._event_bus = EventBus()
        self._running = False
        self._results: dict[str, AgentResult] = {}
        self._execution_order: list[str] = []
        self._parallel_groups: list[list[str]] = []
        self._dependencies: dict[str, set[str]] = defaultdict(set)

    def add_agent(self, agent: Agent) -> "AgentTeam":
        """Add agent to team."""
        self._agents[agent.name] = agent
        agent.set_team(self)

        # Share context
        agent.shared_state.update(self._shared_context)

        return self

    def add_agents(self, agents: list[Agent]) -> "AgentTeam":
        """Add multiple agents to team."""
        for agent in agents:
            self.add_agent(agent)
        return self

    def get_agent(self, name: str) -> Optional[Agent]:
        """Get agent by name."""
        return self._agents.get(name)

    def remove_agent(self, name: str) -> bool:
        """Remove agent from team."""
        if name in self._agents:
            del self._agents[name]
            return True
        return False

    def with_shared_context(
        self, context: Optional[dict[str, Any]] = None
    ) -> "AgentTeam":
        """Set shared context for all agents."""
        if context:
            self._shared_context.update(context)

            # Update all agent contexts
            for agent in self._agents.values():
                agent.shared_state.update(context)

        return self

    def set_global_variable(self, key: str, value: Any) -> None:
        """Set variable for all agents."""
        self._shared_context[key] = value
        for agent in self._agents.values():
            agent.set_shared_variable(key, value)

    def get_global_variable(self, key: str, default: Any = None) -> Any:
        """Get global variable."""
        return self._shared_context.get(key, default)

    def set_variable_for_all(self, key: str, value: Any) -> "AgentTeam":
        """Set variable for all agents (fluent)."""
        for agent in self._agents.values():
            agent.set_variable(key, value)
        return self

    # Messaging system
    async def send_message(
        self, sender: str, recipient: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Send message between agents."""
        message = Message(
            sender=sender,
            recipient=recipient,
            message_type=data.get("message_type", "generic"),
            data=data,
        )

        recipient_agent = self._agents.get(recipient)
        if recipient_agent:
            return await recipient_agent.handle_message(
                message.message_type, message.data, sender
            )

        return {}

    async def broadcast_message(
        self, sender: str, message_type: str, data: dict[str, Any]
    ) -> None:
        """Broadcast message to all agents except sender."""
        for agent_name, agent in self._agents.items():
            if agent_name != sender:
                try:
                    await agent.handle_message(message_type, data, sender)
                except Exception as e:
                    logger.error(f"Error broadcasting to {agent_name}: {e}")

    # Event system
    async def emit_event(
        self, source: str, event_type: str, data: dict[str, Any]
    ) -> None:
        """Emit event to event bus."""
        event = Event(source=source, event_type=event_type, data=data)
        await self._event_bus.emit(event)

    def subscribe_to_events(self, event_type: str, handler: Callable) -> None:
        """Subscribe to events."""
        self._event_bus.subscribe(event_type, handler)

    # Execution methods
    async def run(
        self, mode: str = "parallel", timeout: Optional[float] = None
    ) -> TeamResult:
        """Run the team with specified mode.

        Args:
            mode: Execution mode - "parallel", "sequential", or "dependencies"
            timeout: Optional timeout for execution

        Returns:
            TeamResult with execution results
        """
        if mode == "parallel":
            return await self.run_parallel(timeout)
        elif mode == "sequential":
            return await self.run_sequential()
        elif mode == "dependencies":
            return await self.run_with_dependencies()
        else:
            raise ValueError(f"Unknown execution mode: {mode}")

    async def run_parallel(self, timeout: Optional[float] = None) -> TeamResult:
        """Run all agents in parallel."""
        start_time = time.time()

        try:
            # Create tasks for all agents
            tasks = {
                agent.name: asyncio.create_task(agent.run())
                for agent in self._agents.values()
            }

            # Wait for completion with optional timeout
            if timeout:
                done, pending = await asyncio.wait(
                    tasks.values(), timeout=timeout, return_when=asyncio.ALL_COMPLETED
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
            else:
                await asyncio.gather(*tasks.values(), return_exceptions=True)

            # Collect results
            results = {}
            for agent_name, task in tasks.items():
                if task.done():
                    try:
                        result = task.result()
                        results[agent_name] = result
                    except Exception as e:
                        # Create error result
                        results[agent_name] = AgentResult(
                            agent_name=agent_name,
                            status=AgentStatus.FAILED,
                            error=e,
                            start_time=start_time,
                            end_time=time.time(),
                        )

            end_time = time.time()

            return TeamResult(
                team_name=self.name,
                status="completed",
                agent_results=results,
                start_time=start_time,
                end_time=end_time,
                total_duration=end_time - start_time,
            )

        except Exception as e:
            end_time = time.time()
            return TeamResult(
                team_name=self.name,
                status="failed",
                error=e,
                start_time=start_time,
                end_time=end_time,
                total_duration=end_time - start_time,
            )

    async def run_sequential(
        self, agent_order: Optional[list[str]] = None
    ) -> TeamResult:
        """Run agents sequentially."""
        start_time = time.time()
        results = {}

        try:
            order = agent_order or list(self._agents.keys())

            for agent_name in order:
                agent = self._agents.get(agent_name)
                if agent:
                    result = await agent.run()
                    results[agent_name] = result

                    # Stop on first failure if needed
                    if not result.is_success:
                        logger.warning(f"Agent {agent_name} failed, continuing...")

            end_time = time.time()

            return TeamResult(
                team_name=self.name,
                status="completed",
                agent_results=results,
                start_time=start_time,
                end_time=end_time,
                total_duration=end_time - start_time,
            )

        except Exception as e:
            end_time = time.time()
            return TeamResult(
                team_name=self.name,
                status="failed",
                agent_results=results,
                error=e,
                start_time=start_time,
                end_time=end_time,
                total_duration=end_time - start_time,
            )

    async def run_parallel_and_collect(self) -> TeamResult:
        """Run in parallel and collect all results."""
        return await self.run_parallel()

    async def run_with_dependencies(self) -> TeamResult:
        """Run agents respecting dependencies."""
        start_time = time.time()
        results = {}
        completed: set[str] = set()
        running: dict[str, Any] = {}

        try:
            while len(completed) < len(self._agents):
                # Find agents ready to run
                ready = []
                for agent_name, _agent in self._agents.items():
                    if (
                        agent_name not in completed
                        and agent_name not in running
                        and self._dependencies[agent_name].issubset(completed)
                    ):
                        ready.append(agent_name)

                if not ready:
                    # Wait for running agents
                    if running:
                        done_tasks = await asyncio.wait(
                            running.values(), return_when=asyncio.FIRST_COMPLETED
                        )

                        # Process completed agents
                        for task in done_tasks[0]:
                            for agent_name, agent_task in list(running.items()):
                                if agent_task == task:
                                    result = await task
                                    results[agent_name] = result
                                    completed.add(agent_name)
                                    del running[agent_name]
                                    break
                    else:
                        break  # No agents ready and none running
                else:
                    # Start ready agents
                    for agent_name in ready:
                        agent = self._agents[agent_name]
                        task = asyncio.create_task(agent.run())
                        running[agent_name] = task

            # Wait for remaining agents
            if running:
                remaining_results = await asyncio.gather(*running.values())
                for i, (agent_name, _) in enumerate(running.items()):
                    results[agent_name] = remaining_results[i]
                    completed.add(agent_name)

            end_time = time.time()

            return TeamResult(
                team_name=self.name,
                status="completed",
                agent_results=results,
                start_time=start_time,
                end_time=end_time,
                total_duration=end_time - start_time,
            )

        except Exception as e:
            end_time = time.time()
            return TeamResult(
                team_name=self.name,
                status="failed",
                agent_results=results,
                error=e,
                start_time=start_time,
                end_time=end_time,
                total_duration=end_time - start_time,
            )

    # Dependency management
    def add_dependency(self, agent: str, depends_on: str) -> "AgentTeam":
        """Add dependency between agents."""
        self._dependencies[agent].add(depends_on)
        return self

    def set_execution_order(self, order: list[str]) -> "AgentTeam":
        """Set execution order for sequential runs."""
        self._execution_order = order
        return self

    # Monitoring and control
    def get_status(self) -> dict[str, Any]:
        """Get team status."""
        agent_statuses = {
            name: (
                agent.status.value
                if hasattr(agent.status, "value")
                else str(agent.status)
            )
            for name, agent in self._agents.items()
        }

        return {
            "team_name": self.name,
            "agent_count": len(self._agents),
            "agent_statuses": agent_statuses,
            "shared_variables": len(self._shared_context),
            "running": self._running,
        }

    async def pause_all(self) -> None:
        """Pause all agents."""
        for agent in self._agents.values():
            await agent.pause()

    async def resume_all(self) -> None:
        """Resume all agents."""
        for agent in self._agents.values():
            await agent.resume()

    async def cancel_all(self) -> None:
        """Cancel all agents."""
        for agent in self._agents.values():
            await agent.cancel_all()

    # Context manager support
    async def __aenter__(self) -> "AgentTeam":
        """Async context manager entry."""
        self._running = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        self._running = False
        await self.cancel_all()


# Helper functions for easy team creation
def create_team(name: str, agents: list[Agent]) -> AgentTeam:
    """Create a team with agents."""
    team = AgentTeam(name)
    team.add_agents(agents)
    return team


async def run_agents_parallel(
    agents: list[Agent], timeout: Optional[float] = None
) -> dict[str, AgentResult]:
    """Run agents in parallel and return results."""
    team = create_team("parallel_execution", agents)
    result = await team.run_parallel(timeout)
    return result.agent_results


async def run_agents_sequential(agents: list[Agent]) -> dict[str, AgentResult]:
    """Run agents sequentially and return results."""
    team = create_team("sequential_execution", agents)
    result = await team.run_sequential()
    return result.agent_results
