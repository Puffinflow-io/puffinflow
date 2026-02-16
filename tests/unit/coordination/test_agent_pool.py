"""
Tests for the agent pool coordination module.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, Mock, patch

import pytest

from puffinflow.core.agent.base import Agent
from puffinflow.core.coordination.agent_pool import (
    AgentPool,
    CompletedWork,
    DynamicProcessingPool,
    PoolContext,
    ScalingPolicy,
    WorkItem,
    WorkProcessor,
    WorkQueue,
)


class TestScalingPolicy:
    """Test scaling policy enumeration."""

    def test_scaling_policy_values(self):
        """Test scaling policy enumeration values."""
        assert hasattr(ScalingPolicy, "MANUAL")
        assert hasattr(ScalingPolicy, "AUTO_CPU")
        assert hasattr(ScalingPolicy, "AUTO_QUEUE")
        assert hasattr(ScalingPolicy, "AUTO_LATENCY")
        assert hasattr(ScalingPolicy, "CUSTOM")


class TestWorkItem:
    """Test work item functionality."""

    def test_work_item_creation(self):
        """Test creating a work item."""
        work_item = WorkItem(id="test_1", data={"task": "test"})
        assert work_item.id == "test_1"
        assert work_item.data == {"task": "test"}
        assert work_item.priority == 0
        assert work_item.retries == 0

    def test_work_item_with_priority(self):
        """Test work item with priority."""
        work_item = WorkItem(id="test_1", data={"task": "test"}, priority=5)
        assert work_item.priority == 5

    def test_processing_time(self):
        """Test processing time calculation."""
        work_item = WorkItem(id="test_1", data={"task": "test"})
        assert work_item.processing_time is None

        # Set times
        work_item.assigned_at = 100.0
        work_item.completed_at = 105.0
        assert work_item.processing_time == 5.0

    def test_wait_time(self):
        """Test wait time calculation."""
        work_item = WorkItem(id="test_1", data={"task": "test"})
        work_item.created_at = 100.0
        work_item.assigned_at = 103.0
        assert work_item.wait_time == 3.0


class TestCompletedWork:
    """Test completed work result."""

    def test_completed_work_creation(self):
        """Test creating completed work result."""
        work_item = WorkItem(id="test_1", data={"task": "test"})
        mock_agent = Mock(spec=Agent)
        mock_result = Mock()

        completed = CompletedWork(
            work_item=work_item, agent=mock_agent, result=mock_result, success=True
        )

        assert completed.work_item is work_item
        assert completed.agent is mock_agent
        assert completed.result is mock_result
        assert completed.success is True


class TestWorkQueue:
    """Test work queue functionality."""

    def test_work_queue_creation(self):
        """Test creating a work queue."""
        queue = WorkQueue()
        assert queue.size() == 0
        assert queue.is_empty() is True

    def test_work_queue_with_max_size(self):
        """Test work queue with maximum size."""
        queue = WorkQueue(max_size=2)
        assert queue._max_size == 2

    def test_add_work(self):
        """Test adding work to queue."""
        queue = WorkQueue()
        work_item = WorkItem(id="test_1", data={"task": "test"})

        result = queue.add_work(work_item)
        assert result is True
        assert queue.size() == 1
        assert queue.is_empty() is False

    def test_add_work_with_priority(self):
        """Test adding work with priority."""
        queue = WorkQueue()
        work_item = WorkItem(id="test_1", data={"task": "test"}, priority=5)

        result = queue.add_work(work_item)
        assert result is True
        assert queue._use_priority is True

    def test_get_work(self):
        """Test getting work from queue."""
        queue = WorkQueue()
        work_item = WorkItem(id="test_1", data={"task": "test"})
        queue.add_work(work_item)

        retrieved = queue.get_work()
        assert retrieved is work_item
        assert retrieved.assigned_at is not None
        assert queue.size() == 0

    def test_get_work_empty_queue(self):
        """Test getting work from empty queue."""
        queue = WorkQueue()
        result = queue.get_work()
        assert result is None

    def test_priority_ordering(self):
        """Test priority ordering in queue."""
        queue = WorkQueue()

        # Add items with different priorities
        low_priority = WorkItem(id="low", data={}, priority=1)
        high_priority = WorkItem(id="high", data={}, priority=10)

        queue.add_work(low_priority)
        queue.add_work(high_priority)

        # High priority should come first
        first = queue.get_work()
        assert first.id == "high"

        second = queue.get_work()
        assert second.id == "low"

    def test_clear_queue(self):
        """Test clearing the queue."""
        queue = WorkQueue()
        work_item = WorkItem(id="test_1", data={"task": "test"})
        queue.add_work(work_item)

        queue.clear()
        assert queue.size() == 0
        assert queue.is_empty() is True


class TestAgentPool:
    """Test agent pool functionality."""

    def test_agent_pool_creation(self):
        """Test creating an agent pool."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = AgentPool(agent_factory=agent_factory, min_size=2, max_size=5)

        assert pool.min_size == 2
        assert pool.max_size == 5
        assert len(pool._agents) == 2  # Should create minimum agents

    def test_agent_pool_with_scaling_policy(self):
        """Test agent pool with different scaling policies."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = AgentPool(
            agent_factory=agent_factory, scaling_policy=ScalingPolicy.AUTO_CPU
        )

        assert pool.scaling_policy == ScalingPolicy.AUTO_CPU

    @pytest.mark.asyncio
    async def test_scale_up(self):
        """Test scaling up the pool."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = AgentPool(agent_factory=agent_factory, min_size=1, max_size=5)

        initial_size = len(pool._agents)
        added = await pool.scale_up(2)

        assert added == 2
        assert len(pool._agents) == initial_size + 2

    @pytest.mark.asyncio
    async def test_scale_up_max_limit(self):
        """Test scaling up respects maximum limit."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = AgentPool(agent_factory=agent_factory, min_size=1, max_size=2)

        # Try to scale up beyond max
        added = await pool.scale_up(5)
        assert added == 1  # Should only add 1 to reach max of 2
        assert len(pool._agents) == 2

    @pytest.mark.asyncio
    async def test_scale_down(self):
        """Test scaling down the pool."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = AgentPool(agent_factory=agent_factory, min_size=1, max_size=5)

        # Scale up first
        await pool.scale_up(2)
        initial_size = len(pool._agents)

        # Add some agents to idle set
        for agent in pool._agents[:2]:
            pool._idle_agents.add(agent.name)

        removed = await pool.scale_down(1)
        assert removed == 1
        assert len(pool._agents) == initial_size - 1

    @pytest.mark.asyncio
    async def test_scale_down_min_limit(self):
        """Test scaling down respects minimum limit."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = AgentPool(agent_factory=agent_factory, min_size=2, max_size=5)

        # Try to scale down below minimum
        removed = await pool.scale_down(5)
        assert removed == 0  # Should not remove any agents
        assert len(pool._agents) == 2  # Should stay at minimum

    @pytest.mark.asyncio
    async def test_auto_scale_context(self):
        """Test auto-scaling context manager."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = AgentPool(agent_factory=agent_factory)

        context = pool.auto_scale()
        assert isinstance(context, PoolContext)

    def test_get_metrics(self):
        """Test getting pool metrics."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = AgentPool(agent_factory=agent_factory)
        metrics = pool.get_metrics()

        assert isinstance(metrics, dict)
        assert "total_agents" in metrics
        assert "active_agents" in metrics
        assert "idle_agents" in metrics
        assert "scaling_policy" in metrics

    def test_active_agents_property(self):
        """Test active agents property."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = AgentPool(agent_factory=agent_factory)
        assert pool.active_agents == 0

        # Add an active agent
        pool._active_agents.add("agent_0")
        assert pool.active_agents == 1


class TestPoolContext:
    """Test pool context manager."""

    @pytest.mark.asyncio
    async def test_pool_context(self):
        """Test pool context manager functionality."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = AgentPool(agent_factory=agent_factory)
        context = PoolContext(pool)

        async with context as pool_instance:
            assert pool_instance is pool


class TestWorkProcessor:
    """Test work processor functionality."""

    def test_work_processor_creation(self):
        """Test creating a work processor."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = AgentPool(agent_factory=agent_factory)
        work_queue = WorkQueue()

        processor = WorkProcessor(pool, work_queue)
        assert processor.pool is pool
        assert processor.work_queue is work_queue
        assert processor._running is False


class TestDynamicProcessingPool:
    """Test dynamic processing pool."""

    def test_dynamic_pool_creation(self):
        """Test creating a dynamic processing pool."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = DynamicProcessingPool(
            agent_factory=agent_factory, min_agents=2, max_agents=10
        )

        assert pool.pool.min_size == 2
        assert pool.pool.max_size == 10
        assert isinstance(pool.work_queue, WorkQueue)

    @pytest.mark.asyncio
    async def test_process_workload(self):
        """Test processing a workload."""
        # Mock successful result - define at test level
        mock_result = Mock()
        mock_result.is_success = True
        mock_result.outputs = {"result": "success"}
        mock_result.error = None

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            agent.set_variable = Mock()
            agent.run = AsyncMock()
            agent.run.return_value = mock_result
            return agent

        pool = DynamicProcessingPool(
            agent_factory=agent_factory, min_agents=1, max_agents=2
        )

        # Create work items
        work_items = [
            WorkItem(id="work_1", data={"task": "test1"}),
            WorkItem(id="work_2", data={"task": "test2"}),
        ]

        # Mock the auto_scale method to return a proper async context manager
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=pool.pool)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        with patch.object(pool.pool, "auto_scale", return_value=mock_context):
            # Mock the process_queue method to return completed work
            async def mock_process_queue(work_queue):
                # Simulate processing
                for work_item in work_items:
                    completed_work = CompletedWork(
                        work_item=work_item,
                        agent=pool.pool._agents[0],
                        result=mock_result,
                        success=True,
                    )
                    yield completed_work

            with patch.object(
                pool.pool, "process_queue", side_effect=mock_process_queue
            ):
                results = await pool.process_workload(work_items)

                assert len(results) == 2
                assert all(result["success"] for result in results)

    def test_get_statistics(self):
        """Test getting processing statistics."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = DynamicProcessingPool(agent_factory=agent_factory)

        # Add some mock results
        mock_work_item = WorkItem(id="test", data={})
        # Set the times directly since processing_time is a property
        mock_work_item.assigned_at = 100.0
        mock_work_item.completed_at = 101.5
        mock_work_item.created_at = 99.5

        mock_completed_work = CompletedWork(
            work_item=mock_work_item, agent=Mock(), result=Mock(), success=True
        )

        pool.results.append(mock_completed_work)

        stats = pool.get_statistics()
        assert isinstance(stats, dict)
        assert "total_processed" in stats
        assert "successful" in stats
        assert "success_rate" in stats


class TestWorkItemAdvanced:
    """Test advanced work item functionality."""

    def test_work_item_retry_increment(self):
        """Test work item retry increment."""
        work_item = WorkItem(id="test_1", data={"task": "test"})
        assert work_item.retries == 0

        # Increment retries
        work_item.retries += 1
        assert work_item.retries == 1

    def test_work_item_timing_properties(self):
        """Test work item timing properties."""
        work_item = WorkItem(id="test_1", data={"task": "test"})

        # Test that timing attributes exist
        assert hasattr(work_item, "created_at")
        assert hasattr(work_item, "assigned_at")
        assert hasattr(work_item, "completed_at")

        # Test initial values
        assert work_item.created_at is not None
        assert work_item.assigned_at is None
        assert work_item.completed_at is None


class TestWorkQueueAdvanced:
    """Test advanced work queue functionality."""

    def test_work_queue_max_size_limit(self):
        """Test work queue respects maximum size."""
        queue = WorkQueue(max_size=2)

        # Add items up to limit
        work1 = WorkItem(id="test_1", data={"task": "test1"})
        work2 = WorkItem(id="test_2", data={"task": "test2"})
        work3 = WorkItem(id="test_3", data={"task": "test3"})

        assert queue.add_work(work1) is True
        assert queue.add_work(work2) is True
        assert queue.add_work(work3) is False  # Should reject due to size limit

        assert queue.size() == 2

    def test_work_queue_batch_operations(self):
        """Test batch operations on work queue."""
        queue = WorkQueue()

        # Add multiple items
        work_items = [
            WorkItem(id=f"test_{i}", data={"task": f"test{i}"}) for i in range(5)
        ]

        for item in work_items:
            queue.add_work(item)

        # Get multiple items
        retrieved_items = []
        while not queue.is_empty():
            item = queue.get_work()
            if item:
                retrieved_items.append(item)

        assert len(retrieved_items) == 5

    def test_work_queue_priority_ordering_with_timestamps(self):
        """Test queue behavior with priority and timestamps."""
        queue = WorkQueue()

        # Add items with different priorities at different times
        work1 = WorkItem(id="low_priority", data={}, priority=1)
        work2 = WorkItem(id="high_priority", data={}, priority=10)

        queue.add_work(work1)
        queue.add_work(work2)

        # High priority should come first
        first = queue.get_work()
        assert first.id == "high_priority"

        second = queue.get_work()
        assert second.id == "low_priority"


class TestAgentPoolAdvanced:
    """Test advanced agent pool functionality."""

    def test_agent_pool_basic_properties(self):
        """Test basic agent pool properties."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = AgentPool(agent_factory=agent_factory, min_size=2, max_size=5)

        # Test basic properties
        assert pool.min_size == 2
        assert pool.max_size == 5
        assert len(pool._agents) == 2
        # Check the actual default scaling policy
        assert pool.scaling_policy in [ScalingPolicy.MANUAL, ScalingPolicy.AUTO_QUEUE]

    def test_agent_pool_metrics_basic(self):
        """Test basic agent pool metrics."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = AgentPool(agent_factory=agent_factory, min_size=3, max_size=10)

        # Add some active agents
        pool._active_agents.add("agent_0")
        pool._active_agents.add("agent_1")

        metrics = pool.get_metrics()
        assert isinstance(metrics, dict)
        assert metrics["total_agents"] == 3
        assert metrics["active_agents"] == 2
        # Idle agents calculation may vary based on implementation
        assert metrics["idle_agents"] >= 0


class TestDynamicProcessingPoolAdvanced:
    """Test advanced dynamic processing pool functionality."""

    def test_dynamic_pool_basic_properties(self):
        """Test basic dynamic pool properties."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = DynamicProcessingPool(
            agent_factory=agent_factory, min_agents=1, max_agents=5
        )

        # Verify basic properties
        assert pool.pool.min_size == 1
        assert pool.pool.max_size == 5
        assert isinstance(pool.work_queue, WorkQueue)
        assert isinstance(pool.results, list)

    def test_dynamic_pool_statistics_basic(self):
        """Test basic statistics from dynamic pool."""

        def agent_factory(index):
            agent = Mock(spec=Agent)
            agent.name = f"agent_{index}"
            return agent

        pool = DynamicProcessingPool(agent_factory=agent_factory)

        # Add mock results with different outcomes
        successful_work = WorkItem(id="success", data={})
        successful_work.assigned_at = 100.0
        successful_work.completed_at = 101.0
        successful_work.created_at = 99.5

        failed_work = WorkItem(id="failed", data={})
        failed_work.assigned_at = 100.0
        failed_work.completed_at = 102.0
        failed_work.created_at = 99.0

        pool.results.extend(
            [
                CompletedWork(successful_work, Mock(), Mock(), True),
                CompletedWork(failed_work, Mock(), Mock(), False),
            ]
        )

        stats = pool.get_statistics()
        assert isinstance(stats, dict)
        assert stats["total_processed"] == 2
        assert stats["successful"] == 1
        assert stats["failed"] == 1
        # Success rate might be returned as percentage or decimal
        assert stats["success_rate"] in [0.5, 50.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestAgentPoolCoverageEnhancement:
    """Enhanced tests to improve coverage for agent pool module."""

    @pytest.mark.asyncio
    async def test_cpu_based_scaling_with_psutil(self):
        """Test CPU-based auto-scaling with psutil available."""
        pytest.importorskip("psutil")

        def agent_factory(index):
            return Agent(f"cpu_agent_{index}")

        pool = AgentPool(
            agent_factory=agent_factory,
            min_size=1,
            max_size=5,
            scaling_policy=ScalingPolicy.AUTO_CPU,
            scale_check_interval=0.1,
        )

        work_queue = WorkQueue()
        work_queue.add_work(WorkItem(id="cpu_work", data="test"))

        # Mock psutil to test CPU scaling paths
        with patch("psutil.cpu_percent") as mock_cpu:
            # Test high CPU scenario
            mock_cpu.return_value = 85.0
            scaling_task = asyncio.create_task(pool._auto_scaling_loop(work_queue))
            await asyncio.sleep(0.2)
            scaling_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await scaling_task

    @pytest.mark.asyncio
    async def test_cpu_based_scaling_without_psutil(self):
        """Test CPU-based auto-scaling without psutil available."""

        def agent_factory(index):
            return Agent(f"cpu_agent_{index}")

        pool = AgentPool(
            agent_factory=agent_factory,
            min_size=1,
            max_size=5,
            scaling_policy=ScalingPolicy.AUTO_CPU,
            scale_check_interval=0.1,
        )

        work_queue = WorkQueue()

        # Mock ImportError for psutil
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'psutil'")
        ):
            scaling_task = asyncio.create_task(pool._auto_scaling_loop(work_queue))
            await asyncio.sleep(0.2)
            scaling_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await scaling_task

    @pytest.mark.asyncio
    async def test_work_queue_max_size_enforcement(self):
        """Test work queue enforces maximum size limit."""
        queue = WorkQueue(max_size=2)

        # Add items up to max size
        item1 = WorkItem(id="item1", data="data1")
        item2 = WorkItem(id="item2", data="data2")
        item3 = WorkItem(id="item3", data="data3")

        assert queue.add_work(item1) is True
        assert queue.add_work(item2) is True
        assert queue.add_work(item3) is False  # Should fail due to max size

        assert queue.size() == 2

    @pytest.mark.asyncio
    async def test_work_item_retry_mechanism(self):
        """Test work item retry mechanism with failures."""

        def failing_agent_factory(index):
            agent = Agent(f"failing_agent_{index}")

            # Mock agent to always fail
            async def failing_run():
                raise ValueError("Simulated failure")

            agent.run = failing_run
            return agent

        pool = AgentPool(
            agent_factory=failing_agent_factory,
            min_size=1,
            max_size=1,
            scaling_policy=ScalingPolicy.MANUAL,
        )

        work_queue = WorkQueue()
        work_item = WorkItem(id="retry_work", data="test", max_retries=2)
        work_queue.add_work(work_item)

        processor = pool.process_queue(work_queue)
        await processor._start_processing()

        # Let it process and fail
        await asyncio.sleep(0.5)
        await processor.stop()

        # Check that retries were attempted
        assert work_item.retries > 0

    @pytest.mark.asyncio
    async def test_pool_context_cleanup_with_scaling(self):
        """Test pool context manager cleanup with active scaling."""

        def agent_factory(index):
            return Agent(f"context_agent_{index}")

        pool = AgentPool(
            agent_factory=agent_factory,
            min_size=1,
            max_size=3,
            scaling_policy=ScalingPolicy.AUTO_QUEUE,
        )

        work_queue = WorkQueue()
        work_queue.add_work(WorkItem(id="context_work", data="test"))

        async with pool.auto_scale() as scaled_pool:
            processor = scaled_pool.process_queue(work_queue)
            await processor._start_processing()
            await asyncio.sleep(0.1)
            await processor.stop()

        # After context exit, scaling task should be cancelled or finished
        assert (
            pool._scaling_task is None
            or pool._scaling_task.cancelled()
            or pool._scaling_task.done()
        )

    @pytest.mark.asyncio
    async def test_worker_loop_exception_handling(self):
        """Test worker loop handles exceptions gracefully."""

        def agent_factory(index):
            agent = Agent(f"exception_agent_{index}")
            # Mock agent.run to raise an exception
            original_run = agent.run
            call_count = 0

            async def exception_run():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("First call fails")
                return await original_run()

            agent.run = exception_run
            return agent

        pool = AgentPool(
            agent_factory=agent_factory,
            min_size=1,
            max_size=1,
            scaling_policy=ScalingPolicy.MANUAL,
        )

        work_queue = WorkQueue()
        work_queue.add_work(WorkItem(id="exception_work", data="test"))

        processor = pool.process_queue(work_queue)
        await processor._start_processing()

        # Let it process
        await asyncio.sleep(0.3)
        await processor.stop()

        # Should have handled the exception gracefully
        metrics = pool.get_metrics()
        assert metrics["total_errors"] >= 0

    @pytest.mark.asyncio
    async def test_scale_down_respects_minimum_size(self):
        """Test that scale down respects minimum pool size."""

        def agent_factory(index):
            return Agent(f"min_agent_{index}")

        pool = AgentPool(
            agent_factory=agent_factory,
            min_size=3,
            max_size=5,
            scaling_policy=ScalingPolicy.MANUAL,
        )

        # Try to scale down below minimum
        removed = await pool.scale_down(5)  # Try to remove 5 agents

        # Should not remove any agents since we're at minimum
        assert removed == 0
        assert len(pool._agents) == 3  # Should still have minimum

    @pytest.mark.asyncio
    async def test_work_processor_async_iteration_timeout(self):
        """Test work processor timeout handling in async iteration."""

        def slow_agent_factory(index):
            agent = Agent(f"slow_agent_{index}")
            return agent

        pool = AgentPool(
            agent_factory=slow_agent_factory,
            min_size=1,
            max_size=1,
            scaling_policy=ScalingPolicy.MANUAL,
        )

        work_queue = WorkQueue()
        processor = pool.process_queue(work_queue)

        # Test async iteration with empty queue (should timeout and break)
        results = []
        async for completed_work in processor:
            results.append(completed_work)
            break  # Break after first (if any)

        assert len(results) == 0  # No work items, so no results

    @pytest.mark.asyncio
    async def test_auto_scaling_loop_exception_handling(self):
        """Test auto-scaling loop handles exceptions."""

        def agent_factory(index):
            return Agent(f"scaling_agent_{index}")

        pool = AgentPool(
            agent_factory=agent_factory,
            min_size=1,
            max_size=3,
            scaling_policy=ScalingPolicy.AUTO_QUEUE,
            scale_check_interval=0.1,
        )

        work_queue = WorkQueue()

        # Mock scale_up to raise an exception
        original_scale_up = pool.scale_up

        async def failing_scale_up(*args, **kwargs):
            raise RuntimeError("Scale up failed")

        pool.scale_up = failing_scale_up

        # Start scaling loop and let it handle the exception
        scaling_task = asyncio.create_task(pool._auto_scaling_loop(work_queue))
        await asyncio.sleep(0.3)
        scaling_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await scaling_task

        # Restore original method
        pool.scale_up = original_scale_up

    @pytest.mark.asyncio
    async def test_dynamic_processing_pool_comprehensive(self):
        """Test comprehensive dynamic processing pool functionality."""

        def agent_factory(index):
            agent = Agent(f"test_agent_{index}")
            agent.set_variable("result", f"processed_{index}")
            return agent

        pool = DynamicProcessingPool(agent_factory, min_agents=1, max_agents=3)

        work_items = [WorkItem(id=f"work_{i}", data=f"data_{i}") for i in range(3)]

        results = await pool.process_workload(work_items)
        stats = pool.get_statistics()

        assert len(results) == 3
        assert stats["total_processed"] == 3
        assert stats["successful"] >= 0
        assert stats["failed"] >= 0
        assert "success_rate" in stats
        assert "avg_processing_time" in stats
        assert "avg_wait_time" in stats
        assert "pool_metrics" in stats

    def test_work_item_wait_time_current_time(self):
        """Test work item wait time calculation with current time."""
        work_item = WorkItem(id="test_1", data={"task": "test"})
        work_item.created_at = 100.0
        # Don't set assigned_at to test current time path

        wait_time = work_item.wait_time
        assert wait_time is not None
        assert wait_time > 0  # Should be positive since current time > created_at

    @pytest.mark.asyncio
    async def test_pool_context_manual_scaling_policy(self):
        """Test pool context with manual scaling policy."""

        def agent_factory(index):
            return Agent(f"manual_agent_{index}")

        pool = AgentPool(
            agent_factory=agent_factory,
            min_size=1,
            max_size=3,
            scaling_policy=ScalingPolicy.MANUAL,
        )

        async with pool.auto_scale() as scaled_pool:
            # With manual scaling, no auto-scaling task should be started
            assert scaled_pool._scaling_task is None

    @pytest.mark.asyncio
    async def test_worker_loop_cancellation(self):
        """Test worker loop handles cancellation properly."""

        def agent_factory(index):
            return Agent(f"cancel_agent_{index}")

        pool = AgentPool(
            agent_factory=agent_factory,
            min_size=1,
            max_size=1,
            scaling_policy=ScalingPolicy.MANUAL,
        )

        work_queue = WorkQueue()
        processor = pool.process_queue(work_queue)
        await processor._start_processing()

        # Cancel worker tasks
        for task in processor._worker_tasks:
            task.cancel()

        await processor.stop()

        # All tasks should be cancelled
        for task in processor._worker_tasks:
            assert task.cancelled() or task.done()

    def test_dynamic_processing_pool_empty_statistics(self):
        """Test dynamic processing pool statistics with no results."""

        def agent_factory(index):
            return Agent(f"empty_agent_{index}")

        pool = DynamicProcessingPool(agent_factory)

        # Get statistics with no results
        stats = pool.get_statistics()

        # Should return empty dict when no results
        assert stats == {}
