"""
Tests for the agent team coordination module.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from puffinflow.core.agent.base import Agent
from puffinflow.core.coordination.agent_team import (
    AgentTeam,
    Event,
    EventBus,
    Message,
    TeamResult,
    create_team,
    run_agents_parallel,
    run_agents_sequential,
)


class TestTeamResult:
    """Test team result functionality."""

    def test_team_result_creation(self):
        """Test creating a team result."""
        result = TeamResult(team_name="test_team", status="completed")
        assert result.team_name == "test_team"
        assert result.status == "completed"
        assert isinstance(result.agent_results, dict)

    def test_get_agent_result(self):
        """Test getting specific agent result."""
        mock_agent_result = Mock()
        result = TeamResult(
            team_name="test_team",
            status="completed",
            agent_results={"agent1": mock_agent_result},
        )

        assert result.get_agent_result("agent1") is mock_agent_result
        assert result.get_agent_result("nonexistent") is None

    def test_get_all_outputs(self):
        """Test getting all outputs for a key."""
        mock_result1 = Mock()
        mock_result1.get_output.return_value = "output1"
        mock_result2 = Mock()
        mock_result2.get_output.return_value = "output2"

        result = TeamResult(
            team_name="test_team",
            status="completed",
            agent_results={"agent1": mock_result1, "agent2": mock_result2},
        )

        outputs = result.get_all_outputs("test_key")
        assert outputs == ["output1", "output2"]

    def test_get_best_by(self):
        """Test getting best agent by metric."""
        mock_result1 = Mock()
        mock_result1.get_output.return_value = 5
        mock_result1.get_metric.return_value = None
        mock_result2 = Mock()
        mock_result2.get_output.return_value = 10
        mock_result2.get_metric.return_value = None

        result = TeamResult(
            team_name="test_team",
            status="completed",
            agent_results={"agent1": mock_result1, "agent2": mock_result2},
        )

        best = result.get_best_by("test_metric", maximize=True)
        assert best is mock_result2

    def test_average(self):
        """Test calculating average of numeric values."""
        mock_result1 = Mock()
        mock_result1.get_output.return_value = 5
        mock_result1.get_metric.return_value = None
        mock_result2 = Mock()
        mock_result2.get_output.return_value = 15
        mock_result2.get_metric.return_value = None

        result = TeamResult(
            team_name="test_team",
            status="completed",
            agent_results={"agent1": mock_result1, "agent2": mock_result2},
        )

        avg = result.average("test_key")
        assert avg == 10.0

    def test_sum(self):
        """Test calculating sum of numeric values."""
        mock_result1 = Mock()
        mock_result1.get_output.return_value = 5
        mock_result1.get_metric.return_value = None
        mock_result2 = Mock()
        mock_result2.get_output.return_value = 15
        mock_result2.get_metric.return_value = None

        result = TeamResult(
            team_name="test_team",
            status="completed",
            agent_results={"agent1": mock_result1, "agent2": mock_result2},
        )

        total = result.sum("test_key")
        assert total == 20.0

    def test_count_successful(self):
        """Test counting successful executions."""
        mock_result1 = Mock()
        mock_result1.is_success = True
        mock_result2 = Mock()
        mock_result2.is_success = False

        result = TeamResult(
            team_name="test_team",
            status="completed",
            agent_results={"agent1": mock_result1, "agent2": mock_result2},
        )

        assert result.count_successful() == 1

    def test_count_failed(self):
        """Test counting failed executions."""
        mock_result1 = Mock()
        mock_result1.is_failed = True
        mock_result2 = Mock()
        mock_result2.is_failed = False

        result = TeamResult(
            team_name="test_team",
            status="completed",
            agent_results={"agent1": mock_result1, "agent2": mock_result2},
        )

        assert result.count_failed() == 1

    def test_success_rate(self):
        """Test calculating success rate."""
        mock_result1 = Mock()
        mock_result1.is_success = True
        mock_result2 = Mock()
        mock_result2.is_success = False

        result = TeamResult(
            team_name="test_team",
            status="completed",
            agent_results={"agent1": mock_result1, "agent2": mock_result2},
        )

        assert result.success_rate == 50.0


class TestMessage:
    """Test message functionality."""

    def test_message_creation(self):
        """Test creating a message."""
        message = Message(
            sender="agent1",
            recipient="agent2",
            message_type="test",
            data={"content": "hello"},
        )

        assert message.sender == "agent1"
        assert message.recipient == "agent2"
        assert message.message_type == "test"
        assert message.data == {"content": "hello"}
        assert message.timestamp is not None


class TestEvent:
    """Test event functionality."""

    def test_event_creation(self):
        """Test creating an event."""
        event = Event(
            source="agent1", event_type="status_change", data={"status": "completed"}
        )

        assert event.source == "agent1"
        assert event.event_type == "status_change"
        assert event.data == {"status": "completed"}
        assert event.timestamp is not None
        assert event.event_id is not None


class TestEventBus:
    """Test event bus functionality."""

    def test_event_bus_creation(self):
        """Test creating an event bus."""
        bus = EventBus()
        assert isinstance(bus._handlers, dict)
        assert isinstance(bus._event_history, list)

    def test_subscribe(self):
        """Test subscribing to events."""
        bus = EventBus()
        handler = Mock()

        bus.subscribe("test_event", handler)
        assert handler in bus._handlers["test_event"]

    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        bus = EventBus()
        handler = Mock()

        bus.subscribe("test_event", handler)
        bus.unsubscribe("test_event", handler)
        assert handler not in bus._handlers["test_event"]

    @pytest.mark.asyncio
    async def test_emit_event(self):
        """Test emitting an event."""
        bus = EventBus()
        handler = Mock()
        bus.subscribe("test_event", handler)

        event = Event(
            source="test_source", event_type="test_event", data={"test": "data"}
        )

        await bus.emit(event)

        assert event in bus._event_history
        handler.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_emit_async_handler(self):
        """Test emitting event with async handler."""
        bus = EventBus()
        handler = AsyncMock()
        bus.subscribe("test_event", handler)

        event = Event(
            source="test_source", event_type="test_event", data={"test": "data"}
        )

        await bus.emit(event)
        handler.assert_called_once_with(event)

    def test_get_events(self):
        """Test getting events from history."""
        bus = EventBus()
        event1 = Event(source="agent1", event_type="type1", data={})
        event2 = Event(source="agent2", event_type="type2", data={})

        bus._event_history = [event1, event2]

        # Get all events
        all_events = bus.get_events()
        assert len(all_events) == 2

        # Get events by type
        type1_events = bus.get_events(event_type="type1")
        assert len(type1_events) == 1
        assert type1_events[0] is event1

        # Get events by source
        agent1_events = bus.get_events(source="agent1")
        assert len(agent1_events) == 1
        assert agent1_events[0] is event1


class TestAgentTeam:
    """Test agent team functionality."""

    def test_agent_team_creation(self):
        """Test creating an agent team."""
        team = AgentTeam("test_team")
        assert team.name == "test_team"
        assert isinstance(team._agents, dict)
        assert isinstance(team._shared_context, dict)

    def test_add_agent(self):
        """Test adding an agent to the team."""
        team = AgentTeam("test_team")
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "agent1"
        mock_agent.set_team = Mock()
        mock_agent.shared_state = {}

        result = team.add_agent(mock_agent)

        assert result is team  # Should return self for chaining
        assert "agent1" in team._agents
        assert team._agents["agent1"] is mock_agent
        mock_agent.set_team.assert_called_once_with(team)

    def test_add_agents(self):
        """Test adding multiple agents."""
        team = AgentTeam("test_team")
        mock_agents = []
        for i in range(3):
            agent = Mock(spec=Agent)
            agent.name = f"agent{i}"
            agent.set_team = Mock()
            agent.shared_state = {}
            mock_agents.append(agent)

        result = team.add_agents(mock_agents)

        assert result is team
        assert len(team._agents) == 3

    def test_get_agent(self):
        """Test getting an agent by name."""
        team = AgentTeam("test_team")
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "agent1"
        mock_agent.set_team = Mock()
        mock_agent.shared_state = {}

        team.add_agent(mock_agent)

        retrieved = team.get_agent("agent1")
        assert retrieved is mock_agent

        not_found = team.get_agent("nonexistent")
        assert not_found is None

    def test_remove_agent(self):
        """Test removing an agent from the team."""
        team = AgentTeam("test_team")
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "agent1"
        mock_agent.set_team = Mock()
        mock_agent.shared_state = {}

        team.add_agent(mock_agent)

        result = team.remove_agent("agent1")
        assert result is True
        assert "agent1" not in team._agents

        result = team.remove_agent("nonexistent")
        assert result is False

    def test_with_shared_context(self):
        """Test setting shared context."""
        team = AgentTeam("test_team")
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "agent1"
        mock_agent.set_team = Mock()
        mock_agent.shared_state = {}

        team.add_agent(mock_agent)

        context = {"key": "value"}
        result = team.with_shared_context(context)

        assert result is team
        assert team._shared_context["key"] == "value"

    def test_set_global_variable(self):
        """Test setting global variable."""
        team = AgentTeam("test_team")
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "agent1"
        mock_agent.set_team = Mock()
        mock_agent.shared_state = {}
        mock_agent.set_shared_variable = Mock()

        team.add_agent(mock_agent)
        team.set_global_variable("test_key", "test_value")

        assert team._shared_context["test_key"] == "test_value"
        mock_agent.set_shared_variable.assert_called_with("test_key", "test_value")

    def test_get_global_variable(self):
        """Test getting global variable."""
        team = AgentTeam("test_team")
        team._shared_context["test_key"] = "test_value"

        value = team.get_global_variable("test_key")
        assert value == "test_value"

        default_value = team.get_global_variable("nonexistent", "default")
        assert default_value == "default"

    def test_set_variable_for_all(self):
        """Test setting variable for all agents."""
        team = AgentTeam("test_team")
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "agent1"
        mock_agent.set_team = Mock()
        mock_agent.shared_state = {}
        mock_agent.set_variable = Mock()

        team.add_agent(mock_agent)
        result = team.set_variable_for_all("key", "value")

        assert result is team
        mock_agent.set_variable.assert_called_with("key", "value")

    @pytest.mark.asyncio
    async def test_send_message(self):
        """Test sending message between agents."""
        team = AgentTeam("test_team")

        sender_agent = Mock(spec=Agent)
        sender_agent.name = "sender"
        sender_agent.set_team = Mock()
        sender_agent.shared_state = {}

        recipient_agent = Mock(spec=Agent)
        recipient_agent.name = "recipient"
        recipient_agent.set_team = Mock()
        recipient_agent.shared_state = {}
        recipient_agent.handle_message = AsyncMock(return_value={"response": "ok"})

        team.add_agent(sender_agent)
        team.add_agent(recipient_agent)

        response = await team.send_message(
            "sender", "recipient", {"message_type": "test", "content": "hello"}
        )

        assert response == {"response": "ok"}
        recipient_agent.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_message(self):
        """Test broadcasting message to all agents."""
        team = AgentTeam("test_team")

        agents = []
        for i in range(3):
            agent = Mock(spec=Agent)
            agent.name = f"agent{i}"
            agent.set_team = Mock()
            agent.shared_state = {}
            agent.handle_message = AsyncMock()
            agents.append(agent)
            team.add_agent(agent)

        await team.broadcast_message("agent0", "test_type", {"content": "broadcast"})

        # Sender should not receive the message
        agents[0].handle_message.assert_not_called()

        # Other agents should receive the message
        agents[1].handle_message.assert_called_once()
        agents[2].handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_event(self):
        """Test emitting event to event bus."""
        team = AgentTeam("test_team")

        with patch.object(team._event_bus, "emit") as mock_emit:
            await team.emit_event("source", "event_type", {"data": "test"})
            mock_emit.assert_called_once()

    def test_subscribe_to_events(self):
        """Test subscribing to events."""
        team = AgentTeam("test_team")
        handler = Mock()

        team.subscribe_to_events("test_event", handler)
        assert handler in team._event_bus._handlers["test_event"]

    @pytest.mark.asyncio
    async def test_run_parallel(self):
        """Test running agents in parallel."""
        team = AgentTeam("test_team")

        agents = []
        for i in range(2):
            agent = Mock(spec=Agent)
            agent.name = f"agent{i}"
            agent.set_team = Mock()
            agent.shared_state = {}
            agent.run = AsyncMock()
            # Mock successful result
            mock_result = Mock()
            mock_result.is_success = True
            agent.run.return_value = mock_result
            agents.append(agent)
            team.add_agent(agent)

        result = await team.run_parallel()

        assert isinstance(result, TeamResult)
        assert result.team_name == "test_team"
        assert result.status == "completed"
        assert len(result.agent_results) == 2

        # Verify all agents were run
        for agent in agents:
            agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_sequential(self):
        """Test running agents sequentially."""
        team = AgentTeam("test_team")

        agents = []
        for i in range(2):
            agent = Mock(spec=Agent)
            agent.name = f"agent{i}"
            agent.set_team = Mock()
            agent.shared_state = {}
            agent.run = AsyncMock()
            # Mock successful result
            mock_result = Mock()
            mock_result.is_success = True
            agent.run.return_value = mock_result
            agents.append(agent)
            team.add_agent(agent)

        result = await team.run_sequential()

        assert isinstance(result, TeamResult)
        assert result.team_name == "test_team"
        assert result.status == "completed"
        assert len(result.agent_results) == 2

    def test_add_dependency(self):
        """Test adding dependency between agents."""
        team = AgentTeam("test_team")

        result = team.add_dependency("agent2", "agent1")
        assert result is team
        assert "agent1" in team._dependencies["agent2"]

    def test_set_execution_order(self):
        """Test setting execution order."""
        team = AgentTeam("test_team")
        order = ["agent1", "agent2", "agent3"]

        result = team.set_execution_order(order)
        assert result is team
        assert team._execution_order == order

    def test_get_status(self):
        """Test getting team status."""
        team = AgentTeam("test_team")
        mock_agent = Mock(spec=Agent)
        mock_agent.name = "agent1"
        mock_agent.set_team = Mock()
        mock_agent.shared_state = {}
        mock_agent.status = Mock()
        mock_agent.status.value = "running"

        team.add_agent(mock_agent)

        status = team.get_status()
        assert isinstance(status, dict)
        assert status["team_name"] == "test_team"
        assert status["agent_count"] == 1
        assert "agent_statuses" in status

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test team as async context manager."""
        team = AgentTeam("test_team")

        async with team as t:
            assert t is team
            assert team._running is True

        assert team._running is False


class TestHelperFunctions:
    """Test helper functions."""

    def test_create_team(self):
        """Test creating a team with helper function."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]
        for i, agent in enumerate(mock_agents):
            agent.name = f"agent{i}"
            agent.set_team = Mock()
            agent.shared_state = {}

        team = create_team("helper_team", mock_agents)

        assert isinstance(team, AgentTeam)
        assert team.name == "helper_team"
        assert len(team._agents) == 2

    @pytest.mark.asyncio
    async def test_run_agents_parallel(self):
        """Test running agents in parallel with helper function."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]
        for i, agent in enumerate(mock_agents):
            agent.name = f"agent{i}"
            agent.set_team = Mock()
            agent.shared_state = {}
            agent.run = AsyncMock()
            mock_result = Mock()
            mock_result.is_success = True
            agent.run.return_value = mock_result

        results = await run_agents_parallel(mock_agents)

        assert isinstance(results, dict)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_run_agents_sequential(self):
        """Test running agents sequentially with helper function."""
        mock_agents = [Mock(spec=Agent) for _ in range(2)]
        for i, agent in enumerate(mock_agents):
            agent.name = f"agent{i}"
            agent.set_team = Mock()
            agent.shared_state = {}
            agent.run = AsyncMock()
            mock_result = Mock()
            mock_result.is_success = True
            agent.run.return_value = mock_result

        results = await run_agents_sequential(mock_agents)

        assert isinstance(results, dict)
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
