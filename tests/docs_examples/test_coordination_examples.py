"""Test examples from the coordination documentation."""

import asyncio

import pytest

from puffinflow.core.coordination import Barrier, Mutex, Semaphore


@pytest.mark.asyncio
class TestCoordinationExamples:
    """Test examples from coordination.ts documentation."""

    async def test_semaphore_basic_usage(self):
        """Test Semaphore basic usage with acquire/release."""
        sem = Semaphore("api-limiter", max_count=2)

        # Acquire two permits
        assert await sem.acquire("caller-1") is True
        assert await sem.acquire("caller-2") is True
        assert sem.available_permits == 0

        # Third caller should be rejected (no waiting, immediate return)
        assert await sem.acquire("caller-3") is False

        # Release one permit and retry
        await sem.release("caller-1")
        assert sem.available_permits == 1
        assert await sem.acquire("caller-3") is True

        # Cleanup
        await sem.release("caller-2")
        await sem.release("caller-3")

    async def test_mutex_exclusive_access(self):
        """Test Mutex ensures only one holder at a time."""
        mutex = Mutex("resource-lock")

        # First caller acquires
        assert await mutex.acquire("writer-1") is True

        # Second caller is blocked
        assert await mutex.acquire("writer-2") is False

        # Release and second caller can now acquire
        await mutex.release("writer-1")
        assert await mutex.acquire("writer-2") is True

        await mutex.release("writer-2")

    async def test_mutex_async_context_manager(self):
        """Test Mutex as an async context manager."""
        mutex = Mutex("ctx-lock")
        results = []

        async def critical_section(label: str):
            async with mutex:
                results.append(f"{label}-enter")
                await asyncio.sleep(0.01)
                results.append(f"{label}-exit")

        # Run two tasks; only one should be in the critical section at a time
        await critical_section("a")
        await critical_section("b")

        assert results == ["a-enter", "a-exit", "b-enter", "b-exit"]

    async def test_barrier_synchronization(self):
        """Test Barrier synchronization with multiple parties."""
        barrier = Barrier("sync-point", parties=3, timeout=5.0)
        arrival_order = []
        after_barrier = []

        async def worker(worker_id: str, delay: float):
            await asyncio.sleep(delay)
            arrival_order.append(worker_id)
            await barrier.wait(caller_id=worker_id)
            after_barrier.append(worker_id)

        # Launch three workers with different delays
        await asyncio.gather(
            worker("fast", 0.0),
            worker("medium", 0.01),
            worker("slow", 0.02),
        )

        # All three should have arrived
        assert len(arrival_order) == 3
        # All three should have passed the barrier
        assert len(after_barrier) == 3
        assert set(after_barrier) == {"fast", "medium", "slow"}

    async def test_semaphore_concurrent_limit(self):
        """Test Semaphore limits concurrent access to a resource."""
        sem = Semaphore("pool", max_count=2)
        active = []
        max_concurrent = 0

        async def task(task_id: str):
            nonlocal max_concurrent
            acquired = await sem.acquire(task_id)
            if acquired:
                active.append(task_id)
                max_concurrent = max(max_concurrent, len(active))
                await asyncio.sleep(0.01)
                active.remove(task_id)
                await sem.release(task_id)

        # Run 4 tasks; at most 2 should be active simultaneously
        # Since acquire is non-blocking, we run in two waves
        await asyncio.gather(task("t1"), task("t2"))
        await asyncio.gather(task("t3"), task("t4"))

        assert max_concurrent <= 2
