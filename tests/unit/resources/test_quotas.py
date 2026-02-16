"""
Comprehensive unit tests for quotas module - FIXED VERSION

Tests cover:
- QuotaScope and QuotaPolicy enums
- QuotaLimit, QuotaUsage, and QuotaMetrics dataclasses
- QuotaManager functionality (start/stop, quotas, allocation/release)
- QuotaEnforcer functionality
- Rate limiting and different quota policies
- Concurrent access and thread safety
- Historical tracking and cleanup
- Error conditions and edge cases
- Predefined quota policies
"""

import asyncio
import time
from datetime import datetime, timedelta

import pytest

# Import the classes under test
from puffinflow.core.resources.quotas import (
    QuotaEnforcer,
    QuotaExceededError,
    QuotaLimit,
    QuotaManager,
    QuotaMetrics,
    QuotaPolicies,
    QuotaPolicy,
    QuotaScope,
    QuotaUsage,
)
from puffinflow.core.resources.requirements import ResourceType


class TestQuotaEnums:
    """Test QuotaScope and QuotaPolicy enums."""

    def test_quota_scope_values(self):
        """Test QuotaScope enum values."""
        assert QuotaScope.AGENT.value == "agent"
        assert QuotaScope.WORKFLOW.value == "workflow"
        assert QuotaScope.STATE.value == "state"
        assert QuotaScope.USER.value == "user"
        assert QuotaScope.GLOBAL.value == "global"

    def test_quota_policy_values(self):
        """Test QuotaPolicy enum values."""
        assert QuotaPolicy.HARD.value == "hard"
        assert QuotaPolicy.SOFT.value == "soft"
        assert QuotaPolicy.BURST.value == "burst"
        assert QuotaPolicy.RATE_LIMIT.value == "rate_limit"

    def test_enum_iteration(self):
        """Test that enums can be iterated."""
        scopes = list(QuotaScope)
        assert len(scopes) == 6  # Updated for POOL addition
        assert QuotaScope.AGENT in scopes
        assert QuotaScope.POOL in scopes
        assert QuotaScope.GLOBAL in scopes

        policies = list(QuotaPolicy)
        assert len(policies) == 4
        assert QuotaPolicy.HARD in policies
        assert QuotaPolicy.RATE_LIMIT in policies


class TestQuotaLimit:
    """Test QuotaLimit dataclass."""

    def test_quota_limit_initialization(self):
        """Test QuotaLimit initialization."""
        limit = QuotaLimit(
            resource_type=ResourceType.CPU,
            limit=4.0,
            scope=QuotaScope.AGENT,
            policy=QuotaPolicy.HARD,
        )

        assert limit.resource_type == ResourceType.CPU
        assert limit.limit == 4.0
        assert limit.scope == QuotaScope.AGENT
        assert limit.policy == QuotaPolicy.HARD
        assert limit.burst_limit is None
        assert limit.rate_limit is None
        assert limit.window_size == timedelta(minutes=1)
        assert limit.cooldown == timedelta(minutes=5)

    def test_quota_limit_with_burst(self):
        """Test QuotaLimit with burst policy."""
        limit = QuotaLimit(
            resource_type=ResourceType.MEMORY,
            limit=1024.0,
            scope=QuotaScope.STATE,
            policy=QuotaPolicy.BURST,
        )

        # __post_init__ should set burst_limit to 1.5 * limit
        assert limit.burst_limit == 1536.0  # 1024.0 * 1.5

    def test_quota_limit_with_rate_limit(self):
        """Test QuotaLimit with rate limit policy."""
        limit = QuotaLimit(
            resource_type=ResourceType.IO,
            limit=10.0,
            scope=QuotaScope.WORKFLOW,
            policy=QuotaPolicy.RATE_LIMIT,
        )

        # __post_init__ should set rate_limit to limit
        assert limit.rate_limit == 10.0

    def test_quota_limit_custom_values(self):
        """Test QuotaLimit with custom values."""
        limit = QuotaLimit(
            resource_type=ResourceType.GPU,
            limit=2.0,
            scope=QuotaScope.USER,
            policy=QuotaPolicy.BURST,
            burst_limit=5.0,
            rate_limit=1.0,
            window_size=timedelta(minutes=5),
            cooldown=timedelta(minutes=10),
        )

        assert limit.burst_limit == 5.0  # Custom value, not auto-calculated
        assert limit.rate_limit == 1.0
        assert limit.window_size == timedelta(minutes=5)
        assert limit.cooldown == timedelta(minutes=10)


class TestQuotaUsage:
    """Test QuotaUsage dataclass."""

    def test_quota_usage_initialization(self):
        """Test QuotaUsage initialization with defaults."""
        usage = QuotaUsage()

        assert usage.current == 0.0
        assert usage.peak == 0.0
        assert usage.total_allocated == 0.0
        assert usage.total_released == 0.0
        assert usage.allocations == 0
        assert usage.violations == 0
        assert usage.last_violation is None
        assert isinstance(usage.last_reset, datetime)
        assert usage.request_times == []

    def test_quota_usage_reset(self):
        """Test QuotaUsage reset functionality."""
        usage = QuotaUsage()

        # Add some usage
        usage.current = 5.0
        usage.allocations = 3
        usage.request_times = [time.time(), time.time()]

        # Reset
        usage.reset()

        assert usage.current == 0.0
        assert usage.allocations == 0
        assert usage.request_times == []
        assert isinstance(usage.last_reset, datetime)

    def test_quota_usage_add_allocation(self):
        """Test adding allocations to usage."""
        usage = QuotaUsage()

        usage.add_allocation(3.0)
        assert usage.current == 3.0
        assert usage.total_allocated == 3.0
        assert usage.peak == 3.0
        assert usage.allocations == 1
        assert len(usage.request_times) == 1

        usage.add_allocation(2.0)
        assert usage.current == 5.0
        assert usage.total_allocated == 5.0
        assert usage.peak == 5.0
        assert usage.allocations == 2
        assert len(usage.request_times) == 2

    def test_quota_usage_remove_allocation(self):
        """Test removing allocations from usage."""
        usage = QuotaUsage()

        usage.add_allocation(5.0)
        usage.remove_allocation(2.0)

        assert usage.current == 3.0
        assert usage.total_released == 2.0
        assert usage.peak == 5.0  # Peak doesn't decrease

        # Remove more than current (should not go negative)
        usage.remove_allocation(5.0)
        assert usage.current == 0.0
        assert usage.total_released == 7.0

    def test_quota_usage_record_violation(self):
        """Test recording quota violations."""
        usage = QuotaUsage()

        usage.record_violation()
        assert usage.violations == 1
        assert isinstance(usage.last_violation, datetime)

        first_violation_time = usage.last_violation

        usage.record_violation()
        assert usage.violations == 2
        assert usage.last_violation >= first_violation_time


class TestQuotaMetrics:
    """Test QuotaMetrics dataclass."""

    def test_quota_metrics_initialization(self):
        """Test QuotaMetrics initialization."""
        usage = QuotaUsage()
        limit = QuotaLimit(
            resource_type=ResourceType.CPU, limit=4.0, scope=QuotaScope.AGENT
        )

        metrics = QuotaMetrics(
            scope=QuotaScope.AGENT,
            scope_id="test_agent",
            resource_type=ResourceType.CPU,
            usage=usage,
            limit=limit,
        )

        assert metrics.scope == QuotaScope.AGENT
        assert metrics.scope_id == "test_agent"
        assert metrics.resource_type == ResourceType.CPU
        assert metrics.usage == usage
        assert metrics.limit == limit

    def test_quota_metrics_utilization(self):
        """Test utilization calculation."""
        usage = QuotaUsage()
        usage.current = 2.0

        limit = QuotaLimit(
            resource_type=ResourceType.CPU, limit=4.0, scope=QuotaScope.AGENT
        )

        metrics = QuotaMetrics(
            scope=QuotaScope.AGENT,
            scope_id="test_agent",
            resource_type=ResourceType.CPU,
            usage=usage,
            limit=limit,
        )

        assert metrics.utilization == 50.0  # 2.0 / 4.0 * 100

    def test_quota_metrics_utilization_zero_limit(self):
        """Test utilization with zero limit."""
        usage = QuotaUsage()
        usage.current = 2.0

        limit = QuotaLimit(
            resource_type=ResourceType.CPU, limit=0.0, scope=QuotaScope.AGENT
        )

        metrics = QuotaMetrics(
            scope=QuotaScope.AGENT,
            scope_id="test_agent",
            resource_type=ResourceType.CPU,
            usage=usage,
            limit=limit,
        )

        assert metrics.utilization == 0.0

    def test_quota_metrics_is_exceeded(self):
        """Test is_exceeded property."""
        usage = QuotaUsage()
        limit = QuotaLimit(
            resource_type=ResourceType.CPU, limit=4.0, scope=QuotaScope.AGENT
        )

        metrics = QuotaMetrics(
            scope=QuotaScope.AGENT,
            scope_id="test_agent",
            resource_type=ResourceType.CPU,
            usage=usage,
            limit=limit,
        )

        # Not exceeded
        usage.current = 3.0
        assert not metrics.is_exceeded

        # Exceeded
        usage.current = 5.0
        assert metrics.is_exceeded

        # Exactly at limit
        usage.current = 4.0
        assert not metrics.is_exceeded

    def test_quota_metrics_to_dict(self):
        """Test conversion to dictionary."""
        usage = QuotaUsage()
        usage.current = 2.0
        usage.peak = 3.0
        usage.allocations = 5
        usage.violations = 1

        limit = QuotaLimit(
            resource_type=ResourceType.CPU,
            limit=4.0,
            scope=QuotaScope.AGENT,
            policy=QuotaPolicy.SOFT,
        )

        metrics = QuotaMetrics(
            scope=QuotaScope.AGENT,
            scope_id="test_agent",
            resource_type=ResourceType.CPU,
            usage=usage,
            limit=limit,
        )

        result = metrics.to_dict()

        expected = {
            "scope": "agent",
            "scope_id": "test_agent",
            "resource_type": "CPU",
            "current_usage": 2.0,
            "limit": 4.0,
            "utilization": 50.0,
            "peak_usage": 3.0,
            "allocations": 5,
            "violations": 1,
            "policy": "soft",
        }

        assert result == expected


class TestQuotaExceededError:
    """Test QuotaExceededError exception."""

    def test_quota_exceeded_error_initialization(self):
        """Test QuotaExceededError initialization."""
        error = QuotaExceededError(
            scope=QuotaScope.AGENT,
            scope_id="test_agent",
            resource_type=ResourceType.CPU,
            requested=5.0,
            available=2.0,
        )

        assert error.scope == QuotaScope.AGENT
        assert error.scope_id == "test_agent"
        assert error.resource_type == ResourceType.CPU
        assert error.requested == 5.0
        assert error.available == 2.0

        expected_message = (
            "Quota exceeded for agent 'test_agent': " "requested 5.0 CPU, available 2.0"
        )
        assert str(error) == expected_message


class TestQuotaManager:
    """Test QuotaManager functionality."""

    @pytest.mark.asyncio
    async def test_quota_manager_start_stop(self):
        """Test QuotaManager start and stop."""
        manager = QuotaManager()

        assert not manager._running

        await manager.start()
        assert manager._running
        assert manager._cleanup_task is not None

        await manager.stop()
        assert not manager._running

    @pytest.mark.asyncio
    async def test_quota_manager_set_quota_with_float(self):
        """Test setting quota with float value."""
        manager = QuotaManager()
        await manager.start()
        try:
            manager.set_quota(QuotaScope.AGENT, "test_agent", ResourceType.CPU, 4.0)

            limits = manager._limits[QuotaScope.AGENT]["test_agent"]
            assert ResourceType.CPU in limits

            limit = limits[ResourceType.CPU]
            assert limit.limit == 4.0
            assert limit.resource_type == ResourceType.CPU
            assert limit.scope == QuotaScope.AGENT
            assert limit.policy == QuotaPolicy.HARD  # Default
        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_allocate_hard_quota_success(self):
        """Test successful allocation with hard quota."""
        manager = QuotaManager()
        await manager.start()
        try:
            manager.set_quota(QuotaScope.AGENT, "test", ResourceType.CPU, 4.0)

            result = await manager.allocate(
                QuotaScope.AGENT, "test", ResourceType.CPU, 3.0
            )

            assert result is True

            usage = manager.get_usage(QuotaScope.AGENT, "test", ResourceType.CPU)
            assert usage.current == 3.0
            assert usage.total_allocated == 3.0
            assert usage.allocations == 1
        finally:
            await manager.stop()


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_rate_limit_policy(self):
        """Test rate limiting policy."""
        manager = QuotaManager()
        await manager.start()
        try:
            limit = QuotaLimit(
                resource_type=ResourceType.IO,
                limit=2.0,  # 2 requests per window
                scope=QuotaScope.AGENT,
                policy=QuotaPolicy.RATE_LIMIT,
                window_size=timedelta(seconds=1),
            )
            manager.set_quota(QuotaScope.AGENT, "test", ResourceType.IO, limit)

            # First allocation should pass
            result1 = await manager.allocate(
                QuotaScope.AGENT, "test", ResourceType.IO, 1.0
            )
            assert result1 is True

            # Second allocation should pass
            result2 = await manager.allocate(
                QuotaScope.AGENT, "test", ResourceType.IO, 1.0
            )
            assert result2 is True

            # Third allocation should fail (exceeds rate)
            result3 = await manager.allocate(
                QuotaScope.AGENT, "test", ResourceType.IO, 1.0
            )
            assert result3 is False

            # After window expires, should pass again
            await asyncio.sleep(1.1)
            result4 = await manager.allocate(
                QuotaScope.AGENT, "test", ResourceType.IO, 1.0
            )
            assert result4 is True
        finally:
            await manager.stop()


class TestConcurrency:
    """Test concurrent quota operations."""

    @pytest.mark.asyncio
    async def test_concurrent_allocations(self):
        """Test concurrent quota allocations."""
        manager = QuotaManager()
        await manager.start()
        try:
            manager.set_quota(QuotaScope.AGENT, "test", ResourceType.CPU, 10.0)

            async def allocate_resource(amount: float) -> bool:
                try:
                    return await manager.allocate(
                        QuotaScope.AGENT, "test", ResourceType.CPU, amount
                    )
                except QuotaExceededError:
                    return False

            # Create multiple concurrent allocation tasks
            tasks = [
                allocate_resource(2.0)
                for _ in range(8)  # 8 * 2.0 = 16.0 > 10.0 limit
            ]

            results = await asyncio.gather(*tasks)

            # Some should succeed, some should fail
            successful = sum(results)
            assert successful <= 5  # Maximum that can fit in 10.0 quota

            usage = manager.get_usage(QuotaScope.AGENT, "test", ResourceType.CPU)
            assert usage.current <= 10.0  # Should not exceed quota
        finally:
            await manager.stop()


class TestQuotaEnforcer:
    """Test QuotaEnforcer functionality."""

    @pytest.mark.asyncio
    async def test_check_all_quotas_success(self):
        """Test checking multiple quota requests successfully."""
        manager = QuotaManager()
        await manager.start()
        try:
            enforcer = QuotaEnforcer(manager)

            manager.set_quota(QuotaScope.AGENT, "agent1", ResourceType.CPU, 4.0)
            manager.set_quota(QuotaScope.AGENT, "agent2", ResourceType.MEMORY, 1024.0)

            requests = [
                (QuotaScope.AGENT, "agent1", ResourceType.CPU, 2.0),
                (QuotaScope.AGENT, "agent2", ResourceType.MEMORY, 512.0),
            ]

            allowed, violations = await enforcer.check_all_quotas(requests)

            assert allowed is True
            assert violations == []
        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_allocate_with_quotas_success(self):
        """Test successful allocation with quota enforcement."""
        manager = QuotaManager()
        await manager.start()
        try:
            enforcer = QuotaEnforcer(manager)

            manager.set_quota(QuotaScope.AGENT, "agent1", ResourceType.CPU, 4.0)
            manager.set_quota(QuotaScope.AGENT, "agent1", ResourceType.MEMORY, 1024.0)

            allocations = [
                (QuotaScope.AGENT, "agent1", ResourceType.CPU, 2.0),
                (QuotaScope.AGENT, "agent1", ResourceType.MEMORY, 512.0),
            ]

            success, allocated = await enforcer.allocate_with_quotas(allocations)

            assert success is True
            assert len(allocated) == 2
            assert allocated == allocations

            # Verify allocations were made
            cpu_usage = manager.get_usage(QuotaScope.AGENT, "agent1", ResourceType.CPU)
            mem_usage = manager.get_usage(
                QuotaScope.AGENT, "agent1", ResourceType.MEMORY
            )
            assert cpu_usage.current == 2.0
            assert mem_usage.current == 512.0
        finally:
            await manager.stop()


class TestQuotaPolicies:
    """Test predefined quota policies."""

    def test_small_agent_policy(self):
        """Test small agent quota policy."""
        policy = QuotaPolicies.SMALL_AGENT

        assert policy[ResourceType.CPU] == 2.0
        assert policy[ResourceType.MEMORY] == 512.0
        assert policy[ResourceType.IO] == 10.0
        assert policy[ResourceType.NETWORK] == 10.0

    def test_medium_agent_policy(self):
        """Test medium agent quota policy."""
        policy = QuotaPolicies.MEDIUM_AGENT

        assert policy[ResourceType.CPU] == 4.0
        assert policy[ResourceType.MEMORY] == 2048.0
        assert policy[ResourceType.IO] == 50.0
        assert policy[ResourceType.NETWORK] == 50.0

    def test_large_agent_policy(self):
        """Test large agent quota policy."""
        policy = QuotaPolicies.LARGE_AGENT

        assert policy[ResourceType.CPU] == 8.0
        assert policy[ResourceType.MEMORY] == 8192.0
        assert policy[ResourceType.IO] == 100.0
        assert policy[ResourceType.NETWORK] == 100.0

    def test_gpu_agent_policy(self):
        """Test GPU agent quota policy."""
        policy = QuotaPolicies.GPU_AGENT

        assert policy[ResourceType.CPU] == 4.0
        assert policy[ResourceType.MEMORY] == 16384.0
        assert policy[ResourceType.GPU] == 1.0
        assert policy[ResourceType.IO] == 100.0
        assert policy[ResourceType.NETWORK] == 100.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_zero_allocation(self):
        """Test allocation of zero resources."""
        manager = QuotaManager()
        await manager.start()
        try:
            manager.set_quota(QuotaScope.AGENT, "test", ResourceType.CPU, 4.0)

            result = await manager.allocate(
                QuotaScope.AGENT, "test", ResourceType.CPU, 0.0
            )

            assert result is True
            usage = manager.get_usage(QuotaScope.AGENT, "test", ResourceType.CPU)
            assert usage.current == 0.0
            assert usage.allocations == 1  # Still counts as an allocation
        finally:
            await manager.stop()

    @pytest.mark.asyncio
    async def test_negative_allocation(self):
        """Test allocation of negative resources."""
        manager = QuotaManager()
        await manager.start()
        try:
            manager.set_quota(QuotaScope.AGENT, "test", ResourceType.CPU, 4.0)

            # Should be allowed (might represent resource credits)
            result = await manager.allocate(
                QuotaScope.AGENT, "test", ResourceType.CPU, -1.0
            )

            assert result is True
            usage = manager.get_usage(QuotaScope.AGENT, "test", ResourceType.CPU)
            assert usage.current == -1.0
        finally:
            await manager.stop()

    def test_quota_manager_apply_quota_policy(self):
        """Test applying quota policy template."""
        manager = QuotaManager()

        policy_result = manager.apply_quota_policy(
            "small_workload",
            QuotaPolicies.SMALL_AGENT,
            QuotaScope.WORKFLOW,
            QuotaPolicy.SOFT,
        )

        expected = {
            "policy_name": "small_workload",
            "quotas": QuotaPolicies.SMALL_AGENT,
            "scope": QuotaScope.WORKFLOW,
            "policy": QuotaPolicy.SOFT,
        }

        assert policy_result == expected

    def test_quota_manager_remove_quota_all(self):
        """Test removing all quotas for a scope_id."""
        manager = QuotaManager()

        # Set multiple quotas
        manager.set_quota(QuotaScope.AGENT, "test_agent", ResourceType.CPU, 4.0)
        manager.set_quota(QuotaScope.AGENT, "test_agent", ResourceType.MEMORY, 1024.0)

        # Verify they exist
        assert ResourceType.CPU in manager._limits[QuotaScope.AGENT]["test_agent"]
        assert ResourceType.MEMORY in manager._limits[QuotaScope.AGENT]["test_agent"]

        # Remove all quotas for this agent
        manager.remove_quota(QuotaScope.AGENT, "test_agent")

        # Verify they're gone
        assert "test_agent" not in manager._limits[QuotaScope.AGENT]
        assert "test_agent" not in manager._usage[QuotaScope.AGENT]

    def test_quota_manager_remove_specific_quota(self):
        """Test removing specific quota for a scope_id."""
        manager = QuotaManager()

        # Set multiple quotas
        manager.set_quota(QuotaScope.AGENT, "test_agent", ResourceType.CPU, 4.0)
        manager.set_quota(QuotaScope.AGENT, "test_agent", ResourceType.MEMORY, 1024.0)

        # Remove only CPU quota
        manager.remove_quota(QuotaScope.AGENT, "test_agent", ResourceType.CPU)

        # Verify CPU is gone but MEMORY remains
        assert ResourceType.CPU not in manager._limits[QuotaScope.AGENT]["test_agent"]
        assert ResourceType.MEMORY in manager._limits[QuotaScope.AGENT]["test_agent"]

    @pytest.mark.asyncio
    async def test_quota_manager_async_stop(self):
        """Test async stop functionality."""
        manager = QuotaManager()
        await manager.start()

        # Verify cleanup task is running
        assert manager._cleanup_task is not None

        # Stop the manager
        await manager.stop()

        # Verify cleanup task is cancelled
        assert manager._cleanup_task.cancelled()

    @pytest.mark.asyncio
    async def test_quota_manager_stop_without_cleanup_task(self):
        """Test stop when no cleanup task exists."""
        manager = QuotaManager()
        # Don't start, so cleanup_task is None

        # Should not raise exception
        await manager.stop()

    def test_quota_limit_with_quota_limit_object(self):
        """Test set_quota with QuotaLimit object instead of numeric value."""
        manager = QuotaManager()

        quota_limit = QuotaLimit(
            resource_type=ResourceType.CPU,
            limit=8.0,
            scope=QuotaScope.AGENT,
            policy=QuotaPolicy.SOFT,
        )

        manager.set_quota(QuotaScope.AGENT, "test_agent", ResourceType.CPU, quota_limit)

        stored_limit = manager._limits[QuotaScope.AGENT]["test_agent"][ResourceType.CPU]
        assert stored_limit.limit == 8.0
        assert stored_limit.policy == QuotaPolicy.SOFT

    @pytest.mark.asyncio
    async def test_quota_manager_cleanup_functionality(self):
        """Test cleanup functionality without relying on implementation details."""
        manager = QuotaManager()
        await manager.start()
        try:
            # Test that cleanup can run without errors
            await manager._cleanup_expired()
        finally:
            await manager.stop()

    def test_quota_enums_exist(self):
        """Test QuotaScope and QuotaPolicy enums exist and have expected members."""
        # Test that enums exist and have the basic expected members
        assert hasattr(QuotaScope, "AGENT")
        assert hasattr(QuotaScope, "POOL")
        assert hasattr(QuotaScope, "GLOBAL")

        assert hasattr(QuotaPolicy, "SOFT")
        assert hasattr(QuotaPolicy, "HARD")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
