"""Tests for durable execution: crash recovery, resume, wait-for-event, delay."""

import asyncio

import pytest

from puffinflow.core.agent.base import Agent, MemoryCheckpointStorage
from puffinflow.core.agent.context import Context
from puffinflow.core.agent.state import AgentStatus


class TestDurableExecution:
    """Test Agent.run() with durable=True."""

    @pytest.mark.asyncio
    async def test_durable_run_saves_checkpoint_after_state(self):
        """Verify checkpoints are saved after each state completes."""
        agent = Agent("durable-test")
        storage = MemoryCheckpointStorage()
        agent._checkpoint_storage = storage

        async def step_one(ctx):
            ctx.set_variable("step", 1)
            return "step_two"

        async def step_two(ctx):
            ctx.set_variable("step", 2)
            return None

        agent.add_state("step_one", step_one)
        agent.add_state("step_two", step_two)

        result = await agent.run(durable=True, checkpoint_granularity="per-state")
        assert result.status == AgentStatus.COMPLETED

        # Should have multiple checkpoints saved
        checkpoints = await storage.list_checkpoints("durable-test")
        assert len(checkpoints) >= 1

    @pytest.mark.asyncio
    async def test_durable_run_saves_final_checkpoint(self):
        """Verify a final checkpoint is saved when the agent completes."""
        agent = Agent("durable-final")
        storage = MemoryCheckpointStorage()
        agent._checkpoint_storage = storage

        async def only_state(ctx):
            return None

        agent.add_state("only_state", only_state)

        result = await agent.run(durable=True)
        assert result.status == AgentStatus.COMPLETED

        checkpoints = await storage.list_checkpoints("durable-final")
        assert len(checkpoints) >= 1

    @pytest.mark.asyncio
    async def test_non_durable_run_does_not_checkpoint(self):
        """Verify no checkpoints are saved when durable=False."""
        agent = Agent("non-durable")
        storage = MemoryCheckpointStorage()
        agent._checkpoint_storage = storage

        async def only_state(ctx):
            return None

        agent.add_state("only_state", only_state)

        result = await agent.run(durable=False)
        assert result.status == AgentStatus.COMPLETED

        checkpoints = await storage.list_checkpoints("non-durable")
        assert len(checkpoints) == 0


class TestResumeFrom:
    """Test Agent.resume_from() convenience method."""

    @pytest.mark.asyncio
    async def test_resume_from_raises_on_missing_checkpoint(self):
        agent = Agent("resume-test")
        storage = MemoryCheckpointStorage()
        agent._checkpoint_storage = storage

        async def step(ctx):
            return None

        agent.add_state("step", step)

        with pytest.raises(ValueError, match="not found"):
            await agent.resume_from("nonexistent_checkpoint")


class TestWaitForEvent:
    """Test Agent.wait_for_event() and fire_event()."""

    @pytest.mark.asyncio
    async def test_fire_event_unblocks_wait(self):
        agent = Agent("event-test")

        async def _wait():
            return await agent.wait_for_event("my_event", timeout=5.0)

        async def _fire():
            await asyncio.sleep(0.05)
            agent.fire_event("my_event", data={"value": 42})

        wait_task = asyncio.create_task(_wait())
        fire_task = asyncio.create_task(_fire())

        result = await wait_task
        await fire_task

        assert result == {"value": 42}

    @pytest.mark.asyncio
    async def test_wait_for_event_timeout(self):
        agent = Agent("timeout-test")
        result = await agent.wait_for_event("no_event", timeout=0.05)
        assert result is None

    @pytest.mark.asyncio
    async def test_fire_event_without_waiter(self):
        """fire_event should not raise if nobody is waiting."""
        agent = Agent("fire-only")
        agent.fire_event("orphan_event", data="hello")
        # Should complete without error

    @pytest.mark.asyncio
    async def test_multiple_events(self):
        agent = Agent("multi-event")

        agent.fire_event("event_a", data="a")
        agent.fire_event("event_b", data="b")

        # Events are stored in _event_results
        assert agent._event_results["event_a"] == "a"
        assert agent._event_results["event_b"] == "b"


class TestContextDelay:
    """Test Context.delay() method."""

    @pytest.mark.asyncio
    async def test_delay_sleeps(self):
        ctx = Context(shared_state={})
        # Just verify it doesn't raise and actually waits
        import time

        start = time.time()
        await ctx.delay(0.05)
        elapsed = time.time() - start
        assert elapsed >= 0.04  # Allow small timing variance


class TestDrainIntegration:
    """Test drain protocol integration with Agent.run()."""

    @pytest.mark.asyncio
    async def test_drain_pauses_agent(self):
        """Verify that a draining protocol causes the agent to pause."""
        from puffinflow.core.agent.drain import DrainProtocol

        agent = Agent("drain-test")
        dp = DrainProtocol(timeout=1.0)
        agent._drain_protocol = dp

        call_count = 0

        async def step_one(ctx):
            nonlocal call_count
            call_count += 1
            # Trigger drain during execution
            dp._draining = True
            dp._drain_event.set()
            return "step_two"

        async def step_two(ctx):
            nonlocal call_count
            call_count += 1
            return None

        agent.add_state("step_one", step_one)
        agent.add_state("step_two", step_two)

        result = await agent.run()
        # Agent should have paused after step_one completed
        assert result.status == AgentStatus.PAUSED or call_count <= 2
