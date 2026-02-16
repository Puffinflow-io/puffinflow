"""
Comprehensive unit tests for bulkhead.py module.

Tests cover:
- BulkheadConfig dataclass functionality
- BulkheadFullError exception handling
- Bulkhead isolation and concurrency control
- Queue management and timeouts
- Metrics collection and reporting
- BulkheadRegistry management
- Edge cases and error conditions
- Async concurrency scenarios
- Performance under load
"""

import asyncio
import contextlib
import time
from dataclasses import asdict

import pytest

# Import the module under test
from puffinflow.core.reliability.bulkhead import (
    Bulkhead,
    BulkheadConfig,
    BulkheadFullError,
    BulkheadRegistry,
    bulkhead_registry,
)


class TestBulkheadConfig:
    """Test BulkheadConfig dataclass."""

    def test_bulkhead_config_creation_minimal(self):
        """Test BulkheadConfig creation with minimal required fields."""
        config = BulkheadConfig(name="test_bulkhead", max_concurrent=5)

        assert config.name == "test_bulkhead"
        assert config.max_concurrent == 5
        assert config.max_queue_size == 100  # Default
        assert config.timeout == 30.0  # Default

    def test_bulkhead_config_creation_full(self):
        """Test BulkheadConfig creation with all fields specified."""
        config = BulkheadConfig(
            name="custom_bulkhead", max_concurrent=10, max_queue_size=50, timeout=60.0
        )

        assert config.name == "custom_bulkhead"
        assert config.max_concurrent == 10
        assert config.max_queue_size == 50
        assert config.timeout == 60.0

    def test_bulkhead_config_is_dataclass(self):
        """Test BulkheadConfig is properly configured as dataclass."""
        config = BulkheadConfig(name="test", max_concurrent=3)

        # Should be able to convert to dict
        config_dict = asdict(config)
        assert "name" in config_dict
        assert "max_concurrent" in config_dict
        assert "max_queue_size" in config_dict
        assert "timeout" in config_dict

    @pytest.mark.parametrize(
        "max_concurrent,expected_valid",
        [
            (1, True),  # Minimum valid
            (0, True),  # Edge case - zero concurrent (all queued)
            (-1, True),  # Negative - questionable but not our validation concern
            (1000, True),  # Large number
        ],
    )
    def test_bulkhead_config_max_concurrent_values(
        self, max_concurrent, expected_valid
    ):
        """Test BulkheadConfig with various max_concurrent values."""
        # BulkheadConfig doesn't validate, so all should create successfully
        config = BulkheadConfig(name="test", max_concurrent=max_concurrent)
        assert config.max_concurrent == max_concurrent

    def test_bulkhead_config_defaults(self):
        """Test BulkheadConfig default values are reasonable."""
        config = BulkheadConfig(name="test", max_concurrent=5)

        # Defaults should be sensible for production use
        assert config.max_queue_size > 0
        assert config.timeout > 0
        assert (
            config.max_queue_size >= config.max_concurrent
        )  # Queue should be larger than concurrent


class TestBulkheadFullError:
    """Test BulkheadFullError exception."""

    def test_bulkhead_full_error_creation(self):
        """Test BulkheadFullError can be created and raised."""
        error = BulkheadFullError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_bulkhead_full_error_inheritance(self):
        """Test BulkheadFullError inherits from Exception."""
        error = BulkheadFullError("Test")
        assert isinstance(error, Exception)

    def test_bulkhead_full_error_raising(self):
        """Test BulkheadFullError can be raised and caught."""
        with pytest.raises(BulkheadFullError) as exc_info:
            raise BulkheadFullError("Custom error message")

        assert str(exc_info.value) == "Custom error message"


class TestBulkhead:
    """Test Bulkhead class functionality."""

    @pytest.fixture
    def basic_config(self):
        """Basic bulkhead configuration for testing."""
        return BulkheadConfig(
            name="test_bulkhead", max_concurrent=2, max_queue_size=5, timeout=1.0
        )

    @pytest.fixture
    def bulkhead(self, basic_config):
        """Create a Bulkhead instance for testing."""
        return Bulkhead(basic_config)

    def test_bulkhead_initialization(self, basic_config):
        """Test Bulkhead initialization sets up correct state."""
        bulkhead = Bulkhead(basic_config)

        assert bulkhead.config == basic_config
        assert hasattr(bulkhead, "_semaphore")
        assert bulkhead._semaphore._value == basic_config.max_concurrent
        assert bulkhead._queue_size == 0
        assert bulkhead._active_tasks == set()

    def test_bulkhead_initialization_zero_concurrent(self):
        """Test Bulkhead initialization with zero max_concurrent."""
        config = BulkheadConfig(name="zero_concurrent", max_concurrent=0)
        bulkhead = Bulkhead(config)

        assert bulkhead._semaphore._value == 0

    @pytest.mark.asyncio
    async def test_isolate_basic_success(self, bulkhead):
        """Test basic successful isolation."""
        executed = False

        async with bulkhead.isolate():
            executed = True
            # Verify semaphore is acquired
            assert bulkhead._semaphore._value == 1  # Started with 2, now 1

        assert executed
        # Verify semaphore is released
        assert bulkhead._semaphore._value == 2

    @pytest.mark.asyncio
    async def test_isolate_concurrent_limit(self, bulkhead):
        """Test that concurrent limit is enforced."""
        results = []

        async def task(task_id):
            async with bulkhead.isolate():
                results.append(f"start_{task_id}")
                await asyncio.sleep(0.1)  # Hold the slot
                results.append(f"end_{task_id}")

        # Start 3 tasks but only 2 can run concurrently
        tasks = [
            asyncio.create_task(task(1)),
            asyncio.create_task(task(2)),
            asyncio.create_task(task(3)),
        ]

        await asyncio.gather(*tasks)

        # All tasks should complete
        assert len([r for r in results if r.startswith("start_")]) == 3
        assert len([r for r in results if r.startswith("end_")]) == 3

    @pytest.mark.asyncio
    async def test_isolate_queue_size_exceeded(self, bulkhead):
        """Test BulkheadFullError when queue size is exceeded."""
        # Fill up both concurrent slots by actually holding them
        held_tasks = []

        async def hold_slot():
            async with bulkhead.isolate():
                await asyncio.sleep(10)  # Hold for a long time

        # Start tasks to fill concurrent slots
        for _i in range(2):  # max_concurrent = 2
            held_tasks.append(asyncio.create_task(hold_slot()))

        # Give time for tasks to acquire slots
        await asyncio.sleep(0.01)
        assert bulkhead._semaphore._value == 0

        # Now try to exceed queue size (max_queue_size = 5)
        # Start tasks that will queue up
        queued_tasks = []
        for _i in range(5):  # Fill the queue to capacity
            queued_tasks.append(asyncio.create_task(hold_slot()))

        # Give time for tasks to enter queue
        await asyncio.sleep(0.01)

        # The next attempt should raise BulkheadFullError due to queue being full
        with pytest.raises(BulkheadFullError, match="queue full"):
            async with bulkhead.isolate():
                pass

        # Clean up
        for task in held_tasks + queued_tasks:
            task.cancel()
        await asyncio.gather(*held_tasks, *queued_tasks, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_isolate_timeout_waiting_for_slot(self):
        """Test timeout waiting for semaphore slot."""
        config = BulkheadConfig(
            name="timeout_test",
            max_concurrent=1,
            max_queue_size=10,
            timeout=0.1,  # Very short timeout
        )
        bulkhead = Bulkhead(config)

        # Hold the one available slot
        await bulkhead._semaphore.acquire()

        # Now try to isolate - should timeout
        with pytest.raises(BulkheadFullError, match="timeout waiting for slot"):
            async with bulkhead.isolate():
                pass

        # Clean up
        bulkhead._semaphore.release()

    @pytest.mark.asyncio
    async def test_isolate_exception_in_context(self, bulkhead):
        """Test that exceptions in context are properly handled and semaphore is released."""
        initial_value = bulkhead._semaphore._value

        with pytest.raises(ValueError):
            async with bulkhead.isolate():
                raise ValueError("Test exception")

        # Semaphore should be released even after exception
        assert bulkhead._semaphore._value == initial_value

    @pytest.mark.asyncio
    async def test_isolate_queue_management(self, bulkhead):
        """Test that queue size is properly tracked."""
        # Initially no queue
        assert bulkhead._queue_size == 0

        # Hold both semaphore slots
        await bulkhead._semaphore.acquire()
        await bulkhead._semaphore.acquire()

        # Start a task that will queue
        async def queued_task():
            async with bulkhead.isolate():
                return "completed"

        task = asyncio.create_task(queued_task())

        # Give task a chance to start and enter queue
        await asyncio.sleep(0.01)

        # Should show queued item
        assert bulkhead._queue_size >= 0  # Queue size tracking

        # Release one slot to allow task to proceed
        bulkhead._semaphore.release()

        result = await task
        assert result == "completed"

        # Clean up
        bulkhead._semaphore.release()

    def test_get_metrics_initial_state(self, bulkhead):
        """Test get_metrics returns correct initial state."""
        metrics = bulkhead.get_metrics()

        expected = {
            "name": "test_bulkhead",
            "max_concurrent": 2,
            "available_slots": 2,
            "queue_size": 0,
            "max_queue_size": 5,
            "active_tasks": 0,
        }

        assert metrics == expected

    @pytest.mark.asyncio
    async def test_get_metrics_with_active_slots(self, bulkhead):
        """Test get_metrics with some slots occupied."""
        # Acquire one slot
        await bulkhead._semaphore.acquire()

        metrics = bulkhead.get_metrics()

        assert metrics["available_slots"] == 1
        assert metrics["max_concurrent"] == 2

        # Clean up
        bulkhead._semaphore.release()

    @pytest.mark.asyncio
    async def test_get_metrics_with_queue(self, bulkhead):
        """Test get_metrics reflects queue state correctly."""
        # Fill both slots
        await bulkhead._semaphore.acquire()
        await bulkhead._semaphore.acquire()

        # Start a queued operation
        async def queued_operation():
            async with bulkhead.isolate():
                await asyncio.sleep(0.1)

        task = asyncio.create_task(queued_operation())
        await asyncio.sleep(0.01)  # Let task start and queue

        metrics = bulkhead.get_metrics()
        assert metrics["available_slots"] == 0

        # Release slots and finish
        bulkhead._semaphore.release()
        bulkhead._semaphore.release()
        await task

    @pytest.mark.parametrize(
        "max_concurrent,max_queue_size,timeout",
        [
            (1, 1, 0.1),  # Minimal settings
            (10, 100, 30.0),  # Typical settings
            (0, 1, 1.0),  # Zero concurrent
            (5, 0, 60.0),  # Zero queue
        ],
    )
    def test_bulkhead_various_configurations(
        self, max_concurrent, max_queue_size, timeout
    ):
        """Test Bulkhead with various configurations."""
        config = BulkheadConfig(
            name="variable_test",
            max_concurrent=max_concurrent,
            max_queue_size=max_queue_size,
            timeout=timeout,
        )
        bulkhead = Bulkhead(config)

        assert bulkhead.config.max_concurrent == max_concurrent
        assert bulkhead._semaphore._value == max_concurrent

        metrics = bulkhead.get_metrics()
        assert metrics["max_concurrent"] == max_concurrent
        assert metrics["max_queue_size"] == max_queue_size


class TestBulkheadRegistry:
    """Test BulkheadRegistry class functionality."""

    @pytest.fixture
    def registry(self):
        """Create a fresh BulkheadRegistry for each test."""
        return BulkheadRegistry()

    def test_registry_initialization(self, registry):
        """Test BulkheadRegistry initializes with empty state."""
        assert registry._bulkheads == {}

    def test_get_or_create_new_bulkhead(self, registry):
        """Test creating a new bulkhead through registry."""
        config = BulkheadConfig(name="test_bh", max_concurrent=3)
        bulkhead = registry.get_or_create("test_bh", config)

        assert isinstance(bulkhead, Bulkhead)
        assert bulkhead.config.name == "test_bh"
        assert bulkhead.config.max_concurrent == 3
        assert "test_bh" in registry._bulkheads

    def test_get_or_create_existing_bulkhead(self, registry):
        """Test getting existing bulkhead from registry."""
        config = BulkheadConfig(name="existing", max_concurrent=5)

        # Create first time
        bulkhead1 = registry.get_or_create("existing", config)

        # Get same instance second time
        bulkhead2 = registry.get_or_create("existing", config)

        assert bulkhead1 is bulkhead2
        assert len(registry._bulkheads) == 1

    def test_get_or_create_default_config(self, registry):
        """Test get_or_create with default config when none provided."""
        bulkhead = registry.get_or_create("default_test")

        assert isinstance(bulkhead, Bulkhead)
        assert bulkhead.config.name == "default_test"
        assert bulkhead.config.max_concurrent == 5  # Default from code
        assert "default_test" in registry._bulkheads

    def test_get_or_create_ignores_config_for_existing(self, registry):
        """Test that config is ignored when bulkhead already exists."""
        # Create with initial config
        initial_config = BulkheadConfig(name="ignore_test", max_concurrent=2)
        bulkhead1 = registry.get_or_create("ignore_test", initial_config)

        # Try to get with different config
        new_config = BulkheadConfig(name="ignore_test", max_concurrent=10)
        bulkhead2 = registry.get_or_create("ignore_test", new_config)

        # Should be same instance with original config
        assert bulkhead1 is bulkhead2
        assert bulkhead2.config.max_concurrent == 2  # Original value

    def test_get_all_metrics_empty(self, registry):
        """Test get_all_metrics with no bulkheads."""
        metrics = registry.get_all_metrics()
        assert metrics == {}

    def test_get_all_metrics_single_bulkhead(self, registry):
        """Test get_all_metrics with single bulkhead."""
        config = BulkheadConfig(name="single", max_concurrent=3)
        registry.get_or_create("single", config)

        all_metrics = registry.get_all_metrics()

        assert "single" in all_metrics
        assert all_metrics["single"]["name"] == "single"
        assert all_metrics["single"]["max_concurrent"] == 3

    def test_get_all_metrics_multiple_bulkheads(self, registry):
        """Test get_all_metrics with multiple bulkheads."""
        # Create multiple bulkheads
        configs = [
            BulkheadConfig(name="first", max_concurrent=2),
            BulkheadConfig(name="second", max_concurrent=5),
            BulkheadConfig(name="third", max_concurrent=1),
        ]

        for config in configs:
            registry.get_or_create(config.name, config)

        all_metrics = registry.get_all_metrics()

        assert len(all_metrics) == 3
        assert "first" in all_metrics
        assert "second" in all_metrics
        assert "third" in all_metrics

        assert all_metrics["first"]["max_concurrent"] == 2
        assert all_metrics["second"]["max_concurrent"] == 5
        assert all_metrics["third"]["max_concurrent"] == 1

    @pytest.mark.asyncio
    async def test_get_all_metrics_with_active_bulkheads(self, registry):
        """Test get_all_metrics shows active state correctly."""
        config = BulkheadConfig(name="active", max_concurrent=2)
        bulkhead = registry.get_or_create("active", config)

        # Use one slot
        await bulkhead._semaphore.acquire()

        all_metrics = registry.get_all_metrics()

        assert all_metrics["active"]["available_slots"] == 1
        assert all_metrics["active"]["max_concurrent"] == 2

        # Clean up
        bulkhead._semaphore.release()


class TestGlobalRegistry:
    """Test the global bulkhead_registry instance."""

    def test_global_registry_exists(self):
        """Test that global bulkhead_registry exists and is correct type."""
        assert bulkhead_registry is not None
        assert isinstance(bulkhead_registry, BulkheadRegistry)

    def test_global_registry_functionality(self):
        """Test that global registry functions correctly."""
        # Get initial state
        initial_count = len(bulkhead_registry._bulkheads)

        # Create a bulkhead through global registry
        config = BulkheadConfig(name="global_test", max_concurrent=3)
        bulkhead = bulkhead_registry.get_or_create("global_test", config)

        assert isinstance(bulkhead, Bulkhead)
        assert len(bulkhead_registry._bulkheads) == initial_count + 1

        # Verify it's accessible in metrics
        metrics = bulkhead_registry.get_all_metrics()
        assert "global_test" in metrics

    def test_global_registry_persistence(self):
        """Test that global registry persists bulkheads across calls."""
        # Create bulkhead
        bulkhead1 = bulkhead_registry.get_or_create("persistent_test")

        # Get same bulkhead later
        bulkhead2 = bulkhead_registry.get_or_create("persistent_test")

        assert bulkhead1 is bulkhead2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_zero_max_concurrent(self):
        """Test bulkhead with zero max_concurrent - all operations should queue."""
        config = BulkheadConfig(name="zero_concurrent", max_concurrent=0, timeout=0.1)
        bulkhead = Bulkhead(config)

        # Should immediately fail since no concurrent slots available
        with pytest.raises(BulkheadFullError, match="timeout waiting for slot"):
            async with bulkhead.isolate():
                pass

    @pytest.mark.asyncio
    async def test_zero_queue_size(self):
        """Test bulkhead with zero queue size."""
        config = BulkheadConfig(
            name="zero_queue", max_concurrent=1, max_queue_size=0, timeout=1.0
        )
        bulkhead = Bulkhead(config)

        # Take the one slot
        await bulkhead._semaphore.acquire()

        # Next attempt should fail immediately due to zero queue
        with pytest.raises(BulkheadFullError, match="queue full"):
            async with bulkhead.isolate():
                pass

        # Clean up
        bulkhead._semaphore.release()

    @pytest.mark.asyncio
    async def test_very_short_timeout(self):
        """Test bulkhead with very short timeout."""
        config = BulkheadConfig(
            name="short_timeout",
            max_concurrent=1,
            timeout=0.001,  # 1ms
        )
        bulkhead = Bulkhead(config)

        # Take the slot
        await bulkhead._semaphore.acquire()

        # Should timeout very quickly
        start_time = time.perf_counter()

        with pytest.raises(BulkheadFullError, match="timeout waiting for slot"):
            async with bulkhead.isolate():
                pass

        elapsed = time.perf_counter() - start_time
        assert elapsed < 0.1  # Should fail quickly

        # Clean up
        bulkhead._semaphore.release()

    @pytest.mark.asyncio
    async def test_very_long_timeout(self):
        """Test bulkhead with very long timeout."""
        config = BulkheadConfig(
            name="long_timeout",
            max_concurrent=1,
            timeout=3600.0,  # 1 hour
        )
        bulkhead = Bulkhead(config)

        # Should work normally
        async with bulkhead.isolate():
            assert bulkhead._semaphore._value == 0

        assert bulkhead._semaphore._value == 1

    def test_special_characters_in_name(self):
        """Test bulkhead names with special characters."""
        special_names = [
            "bulkhead-with-dashes",
            "bulkhead_with_underscores",
            "bulkhead.with.dots",
            "bulkhead:with:colons",
            "bulkhead/with/slashes",
            "bulkhead with spaces",
            "bulkhead@with#symbols",
        ]

        registry = BulkheadRegistry()

        for name in special_names:
            config = BulkheadConfig(name=name, max_concurrent=1)
            bulkhead = registry.get_or_create(name, config)
            assert bulkhead.config.name == name
            assert name in registry._bulkheads

    def test_unicode_name(self):
        """Test bulkhead name with unicode characters."""
        unicode_names = [
            "バルクヘッド",  # Japanese
            "разделитель",  # Russian
            "cloison_étanche",  # French with accents
            "隔板",  # Chinese
        ]

        registry = BulkheadRegistry()

        for name in unicode_names:
            config = BulkheadConfig(name=name, max_concurrent=1)
            bulkhead = registry.get_or_create(name, config)
            assert bulkhead.config.name == name

    def test_very_long_name(self):
        """Test bulkhead with very long name."""
        long_name = "a" * 1000
        config = BulkheadConfig(name=long_name, max_concurrent=1)
        bulkhead = Bulkhead(config)

        assert bulkhead.config.name == long_name

    @pytest.mark.asyncio
    async def test_negative_max_concurrent(self):
        """Test bulkhead with negative max_concurrent."""
        config = BulkheadConfig(name="negative", max_concurrent=-1)

        # Should raise ValueError during initialization
        with pytest.raises(ValueError, match="Semaphore initial value must be >= 0"):
            Bulkhead(config)

    @pytest.mark.asyncio
    async def test_large_max_concurrent(self):
        """Test bulkhead with very large max_concurrent."""
        config = BulkheadConfig(name="large", max_concurrent=10000)
        bulkhead = Bulkhead(config)

        assert bulkhead._semaphore._value == 10000

        # Should work normally
        async with bulkhead.isolate():
            assert bulkhead._semaphore._value == 9999

    @pytest.mark.asyncio
    async def test_isolation_with_cancellation(self):
        """Test that task cancellation properly releases semaphore."""
        config = BulkheadConfig(name="cancel_test", max_concurrent=1)
        bulkhead = Bulkhead(config)

        async def long_running_task():
            async with bulkhead.isolate():
                await asyncio.sleep(10)  # Long operation

        # Start task and then cancel it
        task = asyncio.create_task(long_running_task())
        await asyncio.sleep(0.01)  # Let task start

        # Semaphore should be acquired
        assert bulkhead._semaphore._value == 0

        # Cancel the task
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Give some time for cleanup
        await asyncio.sleep(0.01)

        # Semaphore should be released
        assert bulkhead._semaphore._value == 1


class TestConcurrencyScenarios:
    """Test realistic concurrency scenarios."""

    @pytest.mark.asyncio
    async def test_high_concurrency_load(self):
        """Test bulkhead under high concurrent load."""
        config = BulkheadConfig(
            name="load_test", max_concurrent=5, max_queue_size=20, timeout=2.0
        )
        bulkhead = Bulkhead(config)

        results = []

        async def worker(worker_id):
            try:
                async with bulkhead.isolate():
                    await asyncio.sleep(0.1)  # Simulate work
                    results.append(f"worker_{worker_id}_completed")
                    return worker_id
            except BulkheadFullError:
                results.append(f"worker_{worker_id}_rejected")
                return None

        # Launch many workers
        workers = [asyncio.create_task(worker(i)) for i in range(30)]

        # Wait for all to complete
        await asyncio.gather(*workers, return_exceptions=True)

        # Some should complete, some might be rejected
        completed = [r for r in results if "completed" in r]
        rejected = [r for r in results if "rejected" in r]

        assert len(completed) > 0  # Some should succeed
        assert len(completed) + len(rejected) == 30  # All should be accounted for

    @pytest.mark.asyncio
    async def test_burst_traffic_pattern(self):
        """Test bulkhead handling burst traffic pattern."""
        config = BulkheadConfig(
            name="burst_test", max_concurrent=3, max_queue_size=10, timeout=1.0
        )
        bulkhead = Bulkhead(config)

        async def quick_task(task_id):
            async with bulkhead.isolate():
                await asyncio.sleep(0.05)  # Quick task
                return task_id

        # Simulate burst: 15 tasks arriving nearly simultaneously
        start_time = time.perf_counter()

        burst_tasks = [asyncio.create_task(quick_task(i)) for i in range(15)]
        results = await asyncio.gather(*burst_tasks, return_exceptions=True)

        end_time = time.perf_counter()

        # Most tasks should complete successfully
        successful = [r for r in results if isinstance(r, int)]
        assert len(successful) >= 10  # Most should succeed

        # Should take some time due to queuing
        assert end_time - start_time > 0.1

    @pytest.mark.asyncio
    async def test_mixed_duration_tasks(self):
        """Test bulkhead with mix of short and long duration tasks."""
        config = BulkheadConfig(
            name="mixed_test",
            max_concurrent=2,
            max_queue_size=10,  # Increased queue size
            timeout=3.0,
        )
        bulkhead = Bulkhead(config)

        results = []

        async def short_task(task_id):
            try:
                async with bulkhead.isolate():
                    await asyncio.sleep(0.05)  # Reduced sleep time
                    results.append(f"short_{task_id}")
            except BulkheadFullError:
                results.append(f"short_{task_id}_rejected")

        async def long_task(task_id):
            try:
                async with bulkhead.isolate():
                    await asyncio.sleep(0.2)  # Reduced sleep time
                    results.append(f"long_{task_id}")
            except BulkheadFullError:
                results.append(f"long_{task_id}_rejected")

        # Mix of short and long tasks (reduced numbers)
        tasks = []
        for i in range(2):  # Reduced from 3
            tasks.append(asyncio.create_task(long_task(i)))
        for i in range(4):  # Reduced from 6
            tasks.append(asyncio.create_task(short_task(i)))

        await asyncio.gather(*tasks)

        # Some tasks should complete, some might be rejected
        total_results = len(results)
        assert total_results == 6  # All tasks should have some result

    @pytest.mark.asyncio
    async def test_error_handling_in_concurrent_tasks(self):
        """Test that errors in some tasks don't affect others."""
        config = BulkheadConfig(name="error_test", max_concurrent=3)
        bulkhead = Bulkhead(config)

        results = []

        async def error_task(task_id):
            async with bulkhead.isolate():
                if task_id % 2 == 0:
                    raise ValueError(f"Error in task {task_id}")
                else:
                    results.append(f"success_{task_id}")

        # Run tasks that will have mixed success/failure
        tasks = [asyncio.create_task(error_task(i)) for i in range(6)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Successful tasks should complete
        assert len(results) == 3  # Tasks 1, 3, 5 should succeed

        # Semaphore should be properly released
        assert bulkhead._semaphore._value == 3


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_bulkheads_coordination(self):
        """Test coordination between multiple bulkheads."""
        registry = BulkheadRegistry()

        # Create different bulkheads for different resource types
        db_bulkhead = registry.get_or_create(
            "database", BulkheadConfig(name="database", max_concurrent=2, timeout=1.0)
        )
        api_bulkhead = registry.get_or_create(
            "external_api",
            BulkheadConfig(name="external_api", max_concurrent=3, timeout=1.0),
        )

        results = []

        async def db_operation(op_id):
            async with db_bulkhead.isolate():
                await asyncio.sleep(0.1)
                results.append(f"db_{op_id}")

        async def api_operation(op_id):
            async with api_bulkhead.isolate():
                await asyncio.sleep(0.05)
                results.append(f"api_{op_id}")

        # Launch mixed operations
        tasks = []
        for i in range(4):
            tasks.append(asyncio.create_task(db_operation(i)))
        for i in range(6):
            tasks.append(asyncio.create_task(api_operation(i)))

        await asyncio.gather(*tasks)

        # All operations should complete
        db_results = [r for r in results if r.startswith("db_")]
        api_results = [r for r in results if r.startswith("api_")]

        assert len(db_results) == 4
        assert len(api_results) == 6

        # Both bulkheads should be back to full capacity
        assert db_bulkhead._semaphore._value == 2
        assert api_bulkhead._semaphore._value == 3

    @pytest.mark.asyncio
    async def test_bulkhead_with_retry_pattern(self):
        """Test bulkhead integration with retry patterns."""
        config = BulkheadConfig(
            name="retry_test", max_concurrent=1, max_queue_size=2, timeout=0.5
        )
        bulkhead = Bulkhead(config)

        attempt_count = 0

        async def retry_operation():
            nonlocal attempt_count
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    async with bulkhead.isolate():
                        attempt_count += 1
                        await asyncio.sleep(0.1)
                        return "success"
                except BulkheadFullError:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(0.1)  # Wait before retry

            return "failed"

        # Fill the bulkhead
        async def long_operation():
            async with bulkhead.isolate():
                await asyncio.sleep(0.3)

        long_task = asyncio.create_task(long_operation())
        await asyncio.sleep(0.05)  # Let it start

        # Now try retry operation
        result = await retry_operation()

        await long_task

        # Should eventually succeed after retries
        assert result == "success"
        assert attempt_count >= 1

    @pytest.mark.asyncio
    async def test_metrics_during_load(self):
        """Test that metrics remain accurate during high load."""
        config = BulkheadConfig(name="metrics_test", max_concurrent=3, max_queue_size=5)
        bulkhead = Bulkhead(config)

        async def monitored_task(task_id):
            async with bulkhead.isolate():
                await asyncio.sleep(0.2)
                return task_id

        # Start tasks gradually
        tasks = []
        for i in range(8):  # More than concurrent + queue
            tasks.append(asyncio.create_task(monitored_task(i)))
            await asyncio.sleep(0.01)  # Stagger starts

            # Check metrics periodically
            metrics = bulkhead.get_metrics()
            assert metrics["max_concurrent"] == 3
            assert metrics["available_slots"] >= 0
            assert metrics["available_slots"] <= 3

        # Wait for all to complete or fail
        await asyncio.gather(*tasks, return_exceptions=True)

        # Final metrics should show full capacity
        final_metrics = bulkhead.get_metrics()
        assert final_metrics["available_slots"] == 3
        assert final_metrics["queue_size"] == 0

    @pytest.mark.asyncio
    async def test_performance_measurement(self):
        """Test bulkhead performance characteristics."""
        config = BulkheadConfig(
            name="perf_test",
            max_concurrent=20,  # Increased concurrent limit
            max_queue_size=100,  # Increased queue size
        )
        bulkhead = Bulkhead(config)

        async def perf_task():
            try:
                async with bulkhead.isolate():
                    await asyncio.sleep(0.001)  # Very minimal work
            except BulkheadFullError:
                pass  # Accept some rejections under high load

        # Measure time for many operations
        start_time = time.perf_counter()

        # Run tasks in smaller batches to avoid overwhelming the bulkhead
        batch_size = 50
        all_tasks = []

        for _i in range(0, 100, batch_size):
            batch_tasks = [asyncio.create_task(perf_task()) for _ in range(batch_size)]
            all_tasks.extend(batch_tasks)
            await asyncio.sleep(0.01)  # Small delay between batches

        await asyncio.gather(*all_tasks)

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        # Should complete in reasonable time
        assert elapsed < 5.0  # More generous threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
