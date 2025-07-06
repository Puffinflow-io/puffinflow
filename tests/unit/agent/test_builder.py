"""Tests for the StateBuilder pattern and builder functions."""

from unittest.mock import Mock, patch

import pytest

from puffinflow.core.agent.decorators.builder import (
    StateBuilder,
    build_state,
    concurrent_state,
    cpu_state,
    critical_state,
    exclusive_state,
    external_service_state,
    fault_tolerant_state,
    gpu_state,
    high_priority_state,
    isolated_state,
    memory_state,
    production_state,
    protected_state,
)
from puffinflow.core.agent.state import Priority


class TestStateBuilder:
    """Test StateBuilder class functionality."""

    def test_builder_initialization(self):
        """Test StateBuilder initialization."""
        builder = StateBuilder()
        assert builder._config == {}

    def test_cpu_configuration(self):
        """Test CPU configuration."""
        builder = StateBuilder()
        result = builder.cpu(4.0)
        assert result is builder  # Should return self for chaining
        assert builder._config["cpu"] == 4.0

    def test_memory_configuration(self):
        """Test memory configuration."""
        builder = StateBuilder()
        result = builder.memory(2048.0)
        assert result is builder
        assert builder._config["memory"] == 2048.0

    def test_gpu_configuration(self):
        """Test GPU configuration."""
        builder = StateBuilder()
        result = builder.gpu(2.0)
        assert result is builder
        assert builder._config["gpu"] == 2.0

    def test_io_configuration(self):
        """Test I/O configuration."""
        builder = StateBuilder()
        result = builder.io(5.0)
        assert result is builder
        assert builder._config["io"] == 5.0

    def test_network_configuration(self):
        """Test network configuration."""
        builder = StateBuilder()
        result = builder.network(3.0)
        assert result is builder
        assert builder._config["network"] == 3.0

    def test_resources_multiple(self):
        """Test setting multiple resources at once."""
        builder = StateBuilder()
        result = builder.resources(cpu=2.0, memory=1024.0, gpu=1.0, io=2.0, network=1.5)
        assert result is builder
        assert builder._config["cpu"] == 2.0
        assert builder._config["memory"] == 1024.0
        assert builder._config["gpu"] == 1.0
        assert builder._config["io"] == 2.0
        assert builder._config["network"] == 1.5

    def test_resources_partial(self):
        """Test setting only some resources."""
        builder = StateBuilder()
        result = builder.resources(cpu=4.0, memory=2048.0)
        assert result is builder
        assert builder._config["cpu"] == 4.0
        assert builder._config["memory"] == 2048.0
        assert "gpu" not in builder._config

    def test_priority_enum(self):
        """Test priority configuration with enum."""
        builder = StateBuilder()
        result = builder.priority(Priority.HIGH)
        assert result is builder
        assert builder._config["priority"] == Priority.HIGH

    def test_priority_int(self):
        """Test priority configuration with int."""
        builder = StateBuilder()
        result = builder.priority(2)
        assert result is builder
        assert builder._config["priority"] == 2

    def test_priority_string(self):
        """Test priority configuration with string."""
        builder = StateBuilder()
        result = builder.priority("high")
        assert result is builder
        assert builder._config["priority"] == "high"

    def test_high_priority(self):
        """Test high priority shortcut."""
        builder = StateBuilder()
        result = builder.high_priority()
        assert result is builder
        assert builder._config["priority"] == Priority.HIGH

    def test_critical_priority(self):
        """Test critical priority shortcut."""
        builder = StateBuilder()
        result = builder.critical_priority()
        assert result is builder
        assert builder._config["priority"] == Priority.CRITICAL

    def test_low_priority(self):
        """Test low priority shortcut."""
        builder = StateBuilder()
        result = builder.low_priority()
        assert result is builder
        assert builder._config["priority"] == Priority.LOW

    def test_timeout_configuration(self):
        """Test timeout configuration."""
        builder = StateBuilder()
        result = builder.timeout(120.0)
        assert result is builder
        assert builder._config["timeout"] == 120.0

    def test_rate_limit_basic(self):
        """Test basic rate limiting."""
        builder = StateBuilder()
        result = builder.rate_limit(10.0)
        assert result is builder
        assert builder._config["rate_limit"] == 10.0

    def test_rate_limit_with_burst(self):
        """Test rate limiting with burst."""
        builder = StateBuilder()
        result = builder.rate_limit(10.0, 20)
        assert result is builder
        assert builder._config["rate_limit"] == 10.0
        assert builder._config["burst_limit"] == 20

    def test_throttle_alias(self):
        """Test throttle as alias for rate_limit."""
        builder = StateBuilder()
        result = builder.throttle(5.0)
        assert result is builder
        assert builder._config["rate_limit"] == 5.0

    def test_mutex_configuration(self):
        """Test mutex configuration."""
        builder = StateBuilder()
        result = builder.mutex()
        assert result is builder
        assert builder._config["mutex"]

    def test_exclusive_alias(self):
        """Test exclusive as alias for mutex."""
        builder = StateBuilder()
        result = builder.exclusive()
        assert result is builder
        assert builder._config["mutex"]

    def test_semaphore_configuration(self):
        """Test semaphore configuration."""
        builder = StateBuilder()
        result = builder.semaphore(5)
        assert result is builder
        assert builder._config["semaphore"] == 5

    def test_concurrent_alias(self):
        """Test concurrent as alias for semaphore."""
        builder = StateBuilder()
        result = builder.concurrent(10)
        assert result is builder
        assert builder._config["semaphore"] == 10

    def test_barrier_configuration(self):
        """Test barrier configuration."""
        builder = StateBuilder()
        result = builder.barrier(3)
        assert result is builder
        assert builder._config["barrier"] == 3

    def test_synchronized_alias(self):
        """Test synchronized as alias for barrier."""
        builder = StateBuilder()
        result = builder.synchronized(4)
        assert result is builder
        assert builder._config["barrier"] == 4

    def test_lease_configuration(self):
        """Test lease configuration."""
        builder = StateBuilder()
        result = builder.lease(60.0)
        assert result is builder
        assert builder._config["lease"] == 60.0

    def test_quota_configuration(self):
        """Test quota configuration."""
        builder = StateBuilder()
        result = builder.quota(100.0)
        assert result is builder
        assert builder._config["quota"] == 100.0

    def test_depends_on_single(self):
        """Test single dependency."""
        builder = StateBuilder()
        result = builder.depends_on("init_state")
        assert result is builder
        assert builder._config["depends_on"] == ["init_state"]

    def test_depends_on_multiple(self):
        """Test multiple dependencies."""
        builder = StateBuilder()
        result = builder.depends_on("init_state", "setup_state", "validate_state")
        assert result is builder
        assert builder._config["depends_on"] == [
            "init_state",
            "setup_state",
            "validate_state",
        ]

    def test_after_alias(self):
        """Test after as alias for depends_on."""
        builder = StateBuilder()
        result = builder.after("prev_state")
        assert result is builder
        assert builder._config["depends_on"] == ["prev_state"]

    def test_retry_configuration_full(self):
        """Test full retry configuration."""
        builder = StateBuilder()
        result = builder.retry(
            max_retries=5,
            initial_delay=2.0,
            exponential_base=3.0,
            jitter=False,
            dead_letter=False,
            circuit_breaker=True,
        )
        assert result is builder
        assert builder._config["retry_config"]["max_retries"] == 5
        assert builder._config["retry_config"]["initial_delay"] == 2.0
        assert builder._config["retry_config"]["exponential_base"] == 3.0
        assert not builder._config["retry_config"]["jitter"]
        assert not builder._config["retry_config"]["dead_letter_on_max_retries"]
        assert not builder._config["retry_config"]["dead_letter_on_timeout"]
        assert builder._config["circuit_breaker"]

    def test_retry_configuration_minimal(self):
        """Test minimal retry configuration."""
        builder = StateBuilder()
        result = builder.retry(3)
        assert result is builder
        assert builder._config["retry_config"]["max_retries"] == 3
        assert builder._config["retry_config"]["initial_delay"] == 1.0  # default
        assert builder._config["retry_config"]["exponential_base"] == 2.0  # default
        assert builder._config["retry_config"]["jitter"]  # default

    def test_retries_simple(self):
        """Test simple retries configuration."""
        builder = StateBuilder()
        result = builder.retries(7)
        assert result is builder
        assert builder._config["max_retries"] == 7

    def test_no_retry(self):
        """Test disabling retries."""
        builder = StateBuilder()
        result = builder.no_retry()
        assert result is builder
        assert builder._config["max_retries"] == 0

    def test_enable_dead_letter(self):
        """Test enabling dead letter queue."""
        builder = StateBuilder()
        result = builder.enable_dead_letter()
        assert result is builder
        assert builder._config["dead_letter"]

    def test_disable_dead_letter(self):
        """Test disabling dead letter queue."""
        builder = StateBuilder()
        result = builder.disable_dead_letter()
        assert result is builder
        assert builder._config["no_dead_letter"]

    def test_no_dlq_alias(self):
        """Test no_dlq as alias for disable_dead_letter."""
        builder = StateBuilder()
        result = builder.no_dlq()
        assert result is builder
        assert builder._config["no_dead_letter"]

    def test_with_dead_letter(self):
        """Test configuring dead letter behavior."""
        builder = StateBuilder()
        result = builder.with_dead_letter(False)
        assert result is builder
        assert not builder._config["dead_letter"]

        result = builder.with_dead_letter(True)
        assert result is builder
        assert builder._config["dead_letter"]

    def test_circuit_breaker_full_config(self):
        """Test full circuit breaker configuration."""
        builder = StateBuilder()
        result = builder.circuit_breaker(
            enabled=True,
            failure_threshold=10,
            recovery_timeout=120.0,
            success_threshold=5,
        )
        assert result is builder
        assert builder._config["circuit_breaker"]
        assert builder._config["circuit_breaker_config"]["failure_threshold"] == 10
        assert builder._config["circuit_breaker_config"]["recovery_timeout"] == 120.0
        assert builder._config["circuit_breaker_config"]["success_threshold"] == 5

    def test_circuit_breaker_disabled(self):
        """Test disabling circuit breaker."""
        builder = StateBuilder()
        result = builder.circuit_breaker(False)
        assert result is builder
        assert not builder._config["circuit_breaker"]

    def test_with_circuit_breaker(self):
        """Test circuit breaker with custom config."""
        builder = StateBuilder()
        result = builder.with_circuit_breaker(
            failure_threshold=3, recovery_timeout=30.0
        )
        assert result is builder
        assert builder._config["circuit_breaker"]
        assert builder._config["circuit_breaker_config"]["failure_threshold"] == 3
        assert builder._config["circuit_breaker_config"]["recovery_timeout"] == 30.0

    def test_protected(self):
        """Test protected shortcut."""
        builder = StateBuilder()
        result = builder.protected(failure_threshold=5, recovery_timeout=45.0)
        assert result is builder
        assert builder._config["circuit_breaker"]
        assert builder._config["circuit_breaker_config"]["failure_threshold"] == 5
        assert builder._config["circuit_breaker_config"]["recovery_timeout"] == 45.0

    def test_fragile(self):
        """Test fragile shortcut."""
        builder = StateBuilder()
        result = builder.fragile(failure_threshold=1, recovery_timeout=180.0)
        assert result is builder
        assert builder._config["circuit_breaker"]
        assert builder._config["circuit_breaker_config"]["failure_threshold"] == 1
        assert builder._config["circuit_breaker_config"]["recovery_timeout"] == 180.0

    def test_bulkhead_full_config(self):
        """Test full bulkhead configuration."""
        builder = StateBuilder()
        result = builder.bulkhead(
            enabled=True, max_concurrent=10, max_queue_size=200, timeout=60.0
        )
        assert result is builder
        assert builder._config["bulkhead"]
        assert builder._config["bulkhead_config"]["max_concurrent"] == 10
        assert builder._config["bulkhead_config"]["max_queue_size"] == 200
        assert builder._config["bulkhead_config"]["timeout"] == 60.0

    def test_bulkhead_disabled(self):
        """Test disabling bulkhead."""
        builder = StateBuilder()
        result = builder.bulkhead(False)
        assert result is builder
        assert not builder._config["bulkhead"]

    def test_with_bulkhead(self):
        """Test bulkhead with custom config."""
        builder = StateBuilder()
        result = builder.with_bulkhead(max_concurrent=15, timeout=45.0)
        assert result is builder
        assert builder._config["bulkhead"]
        assert builder._config["bulkhead_config"]["max_concurrent"] == 15
        assert builder._config["bulkhead_config"]["timeout"] == 45.0

    def test_isolated(self):
        """Test isolated shortcut."""
        builder = StateBuilder()
        result = builder.isolated(max_concurrent=2)
        assert result is builder
        assert builder._config["bulkhead"]
        assert builder._config["bulkhead_config"]["max_concurrent"] == 2

    def test_single_threaded(self):
        """Test single threaded shortcut."""
        builder = StateBuilder()
        result = builder.single_threaded()
        assert result is builder
        assert builder._config["bulkhead"]
        assert builder._config["bulkhead_config"]["max_concurrent"] == 1

    def test_highly_concurrent(self):
        """Test highly concurrent shortcut."""
        builder = StateBuilder()
        result = builder.highly_concurrent(max_concurrent=50)
        assert result is builder
        assert builder._config["bulkhead"]
        assert builder._config["bulkhead_config"]["max_concurrent"] == 50

    def test_leak_detection(self):
        """Test leak detection configuration."""
        builder = StateBuilder()
        result = builder.leak_detection(True)
        assert result is builder
        assert builder._config["leak_detection"]

        result = builder.leak_detection(False)
        assert result is builder
        assert not builder._config["leak_detection"]

    def test_no_leak_detection(self):
        """Test disabling leak detection."""
        builder = StateBuilder()
        result = builder.no_leak_detection()
        assert result is builder
        assert not builder._config["leak_detection"]

    def test_fault_tolerant(self):
        """Test fault tolerant configuration."""
        builder = StateBuilder()
        result = builder.fault_tolerant(
            circuit_breaker=True, bulkhead=True, max_concurrent=5, failure_threshold=2
        )
        assert result is builder
        assert builder._config["circuit_breaker"]
        assert builder._config["circuit_breaker_config"]["failure_threshold"] == 2
        assert builder._config["bulkhead"]
        assert builder._config["bulkhead_config"]["max_concurrent"] == 5
        assert builder._config["retry_config"]["max_retries"] == 5
        assert builder._config["retry_config"]["dead_letter_on_max_retries"]

    def test_fault_tolerant_partial(self):
        """Test fault tolerant with partial configuration."""
        builder = StateBuilder()
        result = builder.fault_tolerant(circuit_breaker=False, bulkhead=True)
        assert result is builder
        assert (
            "circuit_breaker" not in builder._config
            or not builder._config["circuit_breaker"]
        )
        assert builder._config["bulkhead"]

    def test_production_ready(self):
        """Test production ready configuration."""
        builder = StateBuilder()
        result = builder.production_ready()
        assert result is builder
        assert builder._config["circuit_breaker"]
        assert builder._config["circuit_breaker_config"]["failure_threshold"] == 5
        assert builder._config["circuit_breaker_config"]["recovery_timeout"] == 60.0
        assert builder._config["bulkhead"]
        assert builder._config["bulkhead_config"]["max_concurrent"] == 5
        assert builder._config["retry_config"]["max_retries"] == 3
        assert builder._config["leak_detection"]

    def test_external_call(self):
        """Test external call configuration."""
        builder = StateBuilder()
        result = builder.external_call(timeout=45.0)
        assert result is builder
        assert builder._config["circuit_breaker"]
        assert builder._config["circuit_breaker_config"]["failure_threshold"] == 2
        assert builder._config["circuit_breaker_config"]["recovery_timeout"] == 30.0
        assert builder._config["bulkhead"]
        assert builder._config["bulkhead_config"]["max_concurrent"] == 10
        assert builder._config["timeout"] == 45.0
        assert builder._config["retry_config"]["max_retries"] == 3

    def test_tag_single(self):
        """Test adding single tag."""
        builder = StateBuilder()
        result = builder.tag("environment", "production")
        assert result is builder
        assert builder._config["tags"]["environment"] == "production"

    def test_tags_multiple(self):
        """Test adding multiple tags."""
        builder = StateBuilder()
        result = builder.tags(team="backend", env="prod", version="1.0")
        assert result is builder
        assert builder._config["tags"]["team"] == "backend"
        assert builder._config["tags"]["env"] == "prod"
        assert builder._config["tags"]["version"] == "1.0"

    def test_description(self):
        """Test setting description."""
        builder = StateBuilder()
        result = builder.description("A test state for processing data")
        assert result is builder
        assert builder._config["description"] == "A test state for processing data"

    def test_describe_alias(self):
        """Test describe as alias for description."""
        builder = StateBuilder()
        result = builder.describe("Another test description")
        assert result is builder
        assert builder._config["description"] == "Another test description"

    def test_profile(self):
        """Test setting profile."""
        builder = StateBuilder()
        result = builder.profile("cpu_intensive")
        assert result is builder
        assert builder._config["profile"] == "cpu_intensive"

    def test_like_alias(self):
        """Test like as alias for profile."""
        builder = StateBuilder()
        result = builder.like("memory_intensive")
        assert result is builder
        assert builder._config["profile"] == "memory_intensive"

    def test_preemptible(self):
        """Test preemptible configuration."""
        builder = StateBuilder()
        result = builder.preemptible(True)
        assert result is builder
        assert builder._config["preemptible"]

        result = builder.preemptible(False)
        assert result is builder
        assert not builder._config["preemptible"]

    def test_checkpoint_every(self):
        """Test checkpoint interval."""
        builder = StateBuilder()
        result = builder.checkpoint_every(30.0)
        assert result is builder
        assert builder._config["checkpoint_interval"] == 30.0

    def test_cleanup_on_failure(self):
        """Test cleanup on failure configuration."""
        builder = StateBuilder()
        result = builder.cleanup_on_failure(True)
        assert result is builder
        assert builder._config["cleanup_on_failure"]

        result = builder.cleanup_on_failure(False)
        assert result is builder
        assert not builder._config["cleanup_on_failure"]

    def test_build_method(self):
        """Test build method returns copy of config."""
        builder = StateBuilder()
        builder.cpu(2.0).memory(1024.0).mutex()

        config = builder.build()
        assert config["cpu"] == 2.0
        assert config["memory"] == 1024.0
        assert config["mutex"]

        # Should be a copy, not the same object
        assert config is not builder._config

    def test_call_method_as_decorator(self):
        """Test using builder as decorator."""
        builder = StateBuilder().cpu(2.0).memory(1024.0)

        with patch("puffinflow.core.agent.decorators.flexible.state") as mock_state:
            mock_decorator = Mock()
            mock_state.return_value = mock_decorator

            def test_func():
                return None

            result = builder(test_func)

            mock_state.assert_called_once_with(config=builder._config)
            # The result should be the decorated function, not the decorator itself
            assert result is mock_decorator.return_value

    def test_decorator_method(self):
        """Test decorator method."""
        builder = StateBuilder().cpu(4.0)

        with patch("puffinflow.core.agent.decorators.flexible.state") as mock_state:
            mock_decorator = Mock()
            mock_state.return_value = mock_decorator

            result = builder.decorator()

            mock_state.assert_called_once_with(config=builder._config)
            assert result is mock_decorator

    def test_chaining_complex(self):
        """Test complex method chaining."""
        builder = StateBuilder()
        result = (
            builder.cpu(4.0)
            .memory(2048.0)
            .gpu(1.0)
            .high_priority()
            .mutex()
            .rate_limit(10.0, 20)
            .depends_on("init", "setup")
            .retries(3)
            .circuit_breaker(True, failure_threshold=5)
            .bulkhead(True, max_concurrent=3)
            .tags(team="ml", env="prod")
            .description("Complex ML processing state")
            .timeout(300.0)
        )

        assert result is builder
        config = builder.build()

        # Verify all configurations
        assert config["cpu"] == 4.0
        assert config["memory"] == 2048.0
        assert config["gpu"] == 1.0
        assert config["priority"] == Priority.HIGH
        assert config["mutex"]
        assert config["rate_limit"] == 10.0
        assert config["burst_limit"] == 20
        assert config["depends_on"] == ["init", "setup"]
        assert config["max_retries"] == 3
        assert config["circuit_breaker"]
        assert config["circuit_breaker_config"]["failure_threshold"] == 5
        assert config["bulkhead"]
        assert config["bulkhead_config"]["max_concurrent"] == 3
        assert config["tags"]["team"] == "ml"
        assert config["tags"]["env"] == "prod"
        assert config["description"] == "Complex ML processing state"
        assert config["timeout"] == 300.0


class TestBuilderFunctions:
    """Test builder convenience functions."""

    def test_build_state(self):
        """Test build_state function."""
        builder = build_state()
        assert isinstance(builder, StateBuilder)
        assert builder._config == {}

    def test_cpu_state(self):
        """Test cpu_state function."""
        builder = cpu_state(8.0)
        assert isinstance(builder, StateBuilder)
        assert builder._config["cpu"] == 8.0

    def test_memory_state(self):
        """Test memory_state function."""
        builder = memory_state(4096.0)
        assert isinstance(builder, StateBuilder)
        assert builder._config["memory"] == 4096.0

    def test_gpu_state(self):
        """Test gpu_state function."""
        builder = gpu_state(2.0)
        assert isinstance(builder, StateBuilder)
        assert builder._config["gpu"] == 2.0

    def test_exclusive_state(self):
        """Test exclusive_state function."""
        builder = exclusive_state()
        assert isinstance(builder, StateBuilder)
        assert builder._config["mutex"]

    def test_concurrent_state(self):
        """Test concurrent_state function."""
        builder = concurrent_state(5)
        assert isinstance(builder, StateBuilder)
        assert builder._config["semaphore"] == 5

    def test_high_priority_state(self):
        """Test high_priority_state function."""
        builder = high_priority_state()
        assert isinstance(builder, StateBuilder)
        assert builder._config["priority"] == Priority.HIGH

    def test_critical_state(self):
        """Test critical_state function."""
        builder = critical_state()
        assert isinstance(builder, StateBuilder)
        assert builder._config["priority"] == Priority.CRITICAL

    def test_fault_tolerant_state(self):
        """Test fault_tolerant_state function."""
        builder = fault_tolerant_state()
        assert isinstance(builder, StateBuilder)
        # Should have fault tolerant configuration
        assert builder._config["circuit_breaker"]
        assert builder._config["bulkhead"]
        assert builder._config["retry_config"]["max_retries"] == 5

    def test_external_service_state(self):
        """Test external_service_state function."""
        builder = external_service_state(timeout=60.0)
        assert isinstance(builder, StateBuilder)
        # Should have external service configuration
        assert builder._config["circuit_breaker"]
        assert builder._config["bulkhead"]
        assert builder._config["timeout"] == 60.0

    def test_production_state(self):
        """Test production_state function."""
        builder = production_state()
        assert isinstance(builder, StateBuilder)
        # Should have production-ready configuration
        assert builder._config["circuit_breaker"]
        assert builder._config["bulkhead"]
        assert builder._config["leak_detection"]

    def test_protected_state(self):
        """Test protected_state function."""
        builder = protected_state(failure_threshold=2)
        assert isinstance(builder, StateBuilder)
        assert builder._config["circuit_breaker"]
        assert builder._config["circuit_breaker_config"]["failure_threshold"] == 2

    def test_isolated_state(self):
        """Test isolated_state function."""
        builder = isolated_state(max_concurrent=1)
        assert isinstance(builder, StateBuilder)
        assert builder._config["bulkhead"]
        assert builder._config["bulkhead_config"]["max_concurrent"] == 1


class TestBuilderChaining:
    """Test builder method chaining scenarios."""

    def test_chaining_returns_self(self):
        """Test that all methods return self for chaining."""
        builder = StateBuilder()

        # Test that each method returns the same builder instance
        assert builder.cpu(1.0) is builder
        assert builder.memory(512.0) is builder
        assert builder.gpu(0.5) is builder
        assert builder.priority(Priority.HIGH) is builder
        assert builder.mutex() is builder
        assert builder.rate_limit(5.0) is builder
        assert builder.depends_on("test") is builder
        assert builder.retries(3) is builder
        assert builder.circuit_breaker() is builder
        assert builder.bulkhead() is builder
        assert builder.tag("key", "value") is builder
        assert builder.description("test") is builder

    def test_convenience_function_chaining(self):
        """Test chaining from convenience functions."""
        # Test that convenience functions return chainable builders
        result = cpu_state(4.0).memory(2048.0).high_priority()
        assert isinstance(result, StateBuilder)
        config = result.build()
        assert config["cpu"] == 4.0
        assert config["memory"] == 2048.0
        assert config["priority"] == Priority.HIGH

        result = memory_state(8192.0).gpu(2.0).mutex()
        assert isinstance(result, StateBuilder)
        config = result.build()
        assert config["memory"] == 8192.0
        assert config["gpu"] == 2.0
        assert config["mutex"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
