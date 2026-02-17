"""Tests for the graceful shutdown drain protocol."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from puffinflow.core.agent.drain import DrainProtocol, DrainResult


class TestDrainResult:
    """Test DrainResult dataclass."""

    def test_defaults(self):
        result = DrainResult()
        assert result.agents_drained == []
        assert result.agents_timed_out == []
        assert result.checkpoints_saved == []
        assert result.duration == 0.0

    def test_with_values(self):
        result = DrainResult(
            agents_drained=["agent1"],
            agents_timed_out=["agent2"],
            checkpoints_saved=["cp1"],
            duration=1.5,
        )
        assert result.agents_drained == ["agent1"]
        assert result.agents_timed_out == ["agent2"]
        assert result.checkpoints_saved == ["cp1"]
        assert result.duration == 1.5


class TestDrainProtocol:
    """Test DrainProtocol lifecycle."""

    def test_init_defaults(self):
        dp = DrainProtocol()
        assert dp._timeout == 30.0
        assert dp.is_draining is False

    def test_init_custom_timeout(self):
        dp = DrainProtocol(timeout=10.0)
        assert dp._timeout == 10.0

    def test_register_and_unregister(self):
        dp = DrainProtocol()
        agent = MagicMock()
        agent.__hash__ = MagicMock(return_value=id(agent))
        agent.__eq__ = MagicMock(side_effect=lambda o: o is agent)

        dp.register(agent)
        assert agent in dp._active_agents

        dp.unregister(agent)
        assert agent not in dp._active_agents

    def test_is_draining_property(self):
        dp = DrainProtocol()
        assert dp.is_draining is False
        dp._draining = True
        assert dp.is_draining is True

    @pytest.mark.asyncio
    async def test_drain_no_agents(self):
        dp = DrainProtocol()
        result = await dp.drain()
        assert isinstance(result, DrainResult)
        assert dp.is_draining is True
        assert result.duration >= 0

    @pytest.mark.asyncio
    async def test_drain_with_idle_agent(self):
        dp = DrainProtocol(timeout=1.0)

        agent = MagicMock()
        agent.name = "test-agent"
        agent.__hash__ = MagicMock(return_value=id(agent))
        agent.__eq__ = MagicMock(side_effect=lambda o: o is agent)
        agent.running_states = set()  # Not running anything
        agent._checkpoint_storage = None  # No checkpoint storage

        dp.register(agent)
        result = await dp.drain()

        assert "test-agent" in result.agents_drained
        assert result.agents_timed_out == []

    @pytest.mark.asyncio
    async def test_drain_with_checkpoint_storage(self):
        dp = DrainProtocol(timeout=1.0)

        mock_storage = AsyncMock()
        mock_storage.save_checkpoint = AsyncMock(return_value="checkpoint_123")

        agent = MagicMock()
        agent.name = "test-agent"
        agent.__hash__ = MagicMock(return_value=id(agent))
        agent.__eq__ = MagicMock(side_effect=lambda o: o is agent)
        agent.running_states = set()
        agent._checkpoint_storage = mock_storage
        agent.checkpoint_storage = mock_storage

        dp.register(agent)

        with patch(
            "puffinflow.core.agent.checkpoint.AgentCheckpoint"
        ) as mock_cp_cls:
            mock_cp = MagicMock()
            mock_cp_cls.create_from_agent.return_value = mock_cp

            result = await dp.drain()

        assert "test-agent" in result.agents_drained
        assert "checkpoint_123" in result.checkpoints_saved

    @pytest.mark.asyncio
    async def test_drain_already_draining(self):
        dp = DrainProtocol()
        dp._draining = True
        result = await dp.drain()
        assert result.agents_drained == []
        assert result.agents_timed_out == []

    def test_reset(self):
        dp = DrainProtocol()
        dp._draining = True
        dp._drain_event.set()
        dp.reset()
        assert dp.is_draining is False
        assert not dp._drain_event.is_set()


class TestDrainProtocolSignalHandlers:
    """Test signal handler installation."""

    def test_install_signal_handlers_does_not_raise(self):
        dp = DrainProtocol()
        # Should not raise on any platform
        # On Windows, uses signal.signal; on Unix, uses loop.add_signal_handler
        # Just verify it doesn't crash
        try:
            dp.install_signal_handlers()
        except (RuntimeError, NotImplementedError):
            # May fail if no running event loop
            pass
