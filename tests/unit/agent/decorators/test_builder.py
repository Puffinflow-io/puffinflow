"""Tests for state builder module."""

from unittest.mock import Mock, patch

# Import the modules we're testing
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
    """Test StateBuilder class."""

    def test_builder_creation(self):
        """Test creating a StateBuilder."""
        builder = StateBuilder()
        assert builder._config == {}

    def test_cpu_configuration(self):
        """Test CPU configuration."""
        builder = StateBuilder()
        result = builder.cpu(2.0)

        assert result is builder  # Fluent interface
        assert builder._config["cpu"] == 2.0

    def test_memory_configuration(self):
        """Test memory configuration."""
        builder = StateBuilder()
        result = builder.memory(512.0)

        assert result is builder
        assert builder._config["memory"] == 512.0

    def test_gpu_configuration(self):
        """Test GPU configuration."""
        builder = StateBuilder()
        result = builder.gpu(1.0)

        assert result is builder
        assert builder._config["gpu"] == 1.0

    def test_io_configuration(self):
        """Test I/O configuration."""
        builder = StateBuilder()
        result = builder.io(2.0)

        assert result is builder
        assert builder._config["io"] == 2.0

    def test_network_configuration(self):
        """Test network configuration."""
        builder = StateBuilder()
        result = builder.network(3.0)

        assert result is builder
        assert builder._config["network"] == 3.0

    def test_resources_method_all_params(self):
        """Test resources method with all parameters."""
        builder = StateBuilder()
        result = builder.resources(cpu=2.0, memory=512.0, gpu=1.0, io=1.5, network=2.5)

        assert result is builder
        assert builder._config["cpu"] == 2.0
        assert builder._config["memory"] == 512.0
        assert builder._config["gpu"] == 1.0
        assert builder._config["io"] == 1.5
        assert builder._config["network"] == 2.5

    def test_resources_method_partial_params(self):
        """Test resources method with partial parameters."""
        builder = StateBuilder()
        result = builder.resources(cpu=2.0, memory=512.0)

        assert result is builder
        assert builder._config["cpu"] == 2.0
        assert builder._config["memory"] == 512.0
        assert "gpu" not in builder._config
        assert "io" not in builder._config
        assert "network" not in builder._config

    def test_resources_method_none_values(self):
        """Test resources method ignores None values."""
        builder = StateBuilder()
        result = builder.resources(cpu=2.0, memory=None, gpu=1.0)

        assert result is builder
        assert builder._config["cpu"] == 2.0
        assert builder._config["gpu"] == 1.0
        assert "memory" not in builder._config


class TestPriorityAndTiming:
    """Test priority and timing configuration methods."""

    def test_priority_with_enum(self):
        """Test priority configuration with Priority enum."""
        builder = StateBuilder()
        result = builder.priority(Priority.HIGH)

        assert result is builder
        assert builder._config["priority"] == Priority.HIGH

    def test_priority_with_int(self):
        """Test priority configuration with integer."""
        builder = StateBuilder()
        result = builder.priority(3)

        assert result is builder
        assert builder._config["priority"] == 3

    def test_priority_with_string(self):
        """Test priority configuration with string."""
        builder = StateBuilder()
        result = builder.priority("high")

        assert result is builder
        assert builder._config["priority"] == "high"

    def test_high_priority_shortcut(self):
        """Test high_priority shortcut method."""
        builder = StateBuilder()
        result = builder.high_priority()

        assert result is builder
        assert builder._config["priority"] == Priority.HIGH

    def test_critical_priority_shortcut(self):
        """Test critical_priority shortcut method."""
        builder = StateBuilder()
        result = builder.critical_priority()

        assert result is builder
        assert builder._config["priority"] == Priority.CRITICAL

    def test_low_priority_shortcut(self):
        """Test low_priority shortcut method."""
        builder = StateBuilder()
        result = builder.low_priority()

        assert result is builder
        assert builder._config["priority"] == Priority.LOW

    def test_timeout_configuration(self):
        """Test timeout configuration."""
        builder = StateBuilder()
        result = builder.timeout(60.0)

        assert result is builder
        assert builder._config["timeout"] == 60.0


class TestRateLimiting:
    """Test rate limiting configuration methods."""

    def test_rate_limit_basic(self):
        """Test basic rate limit configuration."""
        builder = StateBuilder()
        result = builder.rate_limit(10.0)

        assert result is builder
        assert builder._config["rate_limit"] == 10.0
        assert "burst_limit" not in builder._config

    def test_rate_limit_with_burst(self):
        """Test rate limit with burst configuration."""
        builder = StateBuilder()
        result = builder.rate_limit(10.0, burst=20)

        assert result is builder
        assert builder._config["rate_limit"] == 10.0
        assert builder._config["burst_limit"] == 20

    def test_throttle_alias(self):
        """Test throttle as alias for rate_limit."""
        builder = StateBuilder()
        result = builder.throttle(5.0)

        assert result is builder
        assert builder._config["rate_limit"] == 5.0


class TestCoordination:
    """Test coordination configuration methods."""

    def test_mutex_configuration(self):
        """Test mutex configuration."""
        builder = StateBuilder()
        result = builder.mutex()

        assert result is builder
        assert builder._config["mutex"] is True

    def test_exclusive_alias(self):
        """Test exclusive as alias for mutex."""
        builder = StateBuilder()
        result = builder.exclusive()

        assert result is builder
        assert builder._config["mutex"] is True

    def test_semaphore_configuration(self):
        """Test semaphore configuration."""
        builder = StateBuilder()
        result = builder.semaphore(5)

        assert result is builder
        assert builder._config["semaphore"] == 5

    def test_concurrent_alias(self):
        """Test concurrent as alias for semaphore."""
        builder = StateBuilder()
        result = builder.concurrent(3)

        assert result is builder
        assert builder._config["semaphore"] == 3

    def test_barrier_configuration(self):
        """Test barrier configuration."""
        builder = StateBuilder()
        result = builder.barrier(4)

        assert result is builder
        assert builder._config["barrier"] == 4

    def test_synchronized_alias(self):
        """Test synchronized as alias for barrier."""
        builder = StateBuilder()
        result = builder.synchronized(2)

        assert result is builder
        assert builder._config["barrier"] == 2

    def test_lease_configuration(self):
        """Test lease configuration."""
        builder = StateBuilder()
        result = builder.lease(30.0)

        assert result is builder
        assert builder._config["lease"] == 30.0

    def test_quota_configuration(self):
        """Test quota configuration."""
        builder = StateBuilder()
        result = builder.quota(100.0)

        assert result is builder
        assert builder._config["quota"] == 100.0


class TestDependencies:
    """Test dependency configuration methods."""

    def test_depends_on_single(self):
        """Test depends_on with single dependency."""
        builder = StateBuilder()
        result = builder.depends_on("state1")

        assert result is builder
        assert builder._config["depends_on"] == ["state1"]

    def test_depends_on_multiple(self):
        """Test depends_on with multiple dependencies."""
        builder = StateBuilder()
        result = builder.depends_on("state1", "state2", "state3")

        assert result is builder
        assert builder._config["depends_on"] == ["state1", "state2", "state3"]

    def test_after_alias(self):
        """Test after as alias for depends_on."""
        builder = StateBuilder()
        result = builder.after("state1", "state2")

        assert result is builder
        assert builder._config["depends_on"] == ["state1", "state2"]


class TestRetryConfiguration:
    """Test retry configuration methods."""

    def test_retry_basic(self):
        """Test basic retry configuration."""
        builder = StateBuilder()
        result = builder.retry(3)

        assert result is builder
        retry_config = builder._config["retry_config"]
        assert retry_config["max_retries"] == 3
        assert retry_config["initial_delay"] == 1.0  # default
        assert retry_config["exponential_base"] == 2.0  # default
        assert retry_config["jitter"] is True  # default
        assert retry_config["dead_letter_on_max_retries"] is True  # default
        assert retry_config["dead_letter_on_timeout"] is True  # default

    def test_retry_full_config(self):
        """Test retry with full configuration."""
        builder = StateBuilder()
        result = builder.retry(
            max_retries=5,
            initial_delay=2.0,
            exponential_base=1.5,
            jitter=False,
            dead_letter=False,
            circuit_breaker=True,
        )

        assert result is builder
        retry_config = builder._config["retry_config"]
        assert retry_config["max_retries"] == 5
        assert retry_config["initial_delay"] == 2.0
        assert retry_config["exponential_base"] == 1.5
        assert retry_config["jitter"] is False
        assert retry_config["dead_letter_on_max_retries"] is False
        assert retry_config["dead_letter_on_timeout"] is False
        assert builder._config["circuit_breaker"] is True

    def test_retries_simple(self):
        """Test simple retries method."""
        builder = StateBuilder()
        result = builder.retries(5)

        assert result is builder
        assert builder._config["max_retries"] == 5

    def test_no_retry(self):
        """Test no_retry method."""
        builder = StateBuilder()
        result = builder.no_retry()

        assert result is builder
        assert builder._config["max_retries"] == 0


class TestDeadLetterConfiguration:
    """Test dead letter configuration methods."""

    def test_enable_dead_letter(self):
        """Test enable_dead_letter method."""
        builder = StateBuilder()
        result = builder.enable_dead_letter()

        assert result is builder
        assert builder._config["dead_letter"] is True

    def test_disable_dead_letter(self):
        """Test disable_dead_letter method."""
        builder = StateBuilder()
        result = builder.disable_dead_letter()

        assert result is builder
        assert builder._config["no_dead_letter"] is True

    def test_no_dlq_alias(self):
        """Test no_dlq as alias for disable_dead_letter."""
        builder = StateBuilder()
        result = builder.no_dlq()

        assert result is builder
        assert builder._config["no_dead_letter"] is True

    def test_with_dead_letter_enabled(self):
        """Test with_dead_letter with enabled=True."""
        builder = StateBuilder()
        result = builder.with_dead_letter(True)

        assert result is builder
        assert builder._config["dead_letter"] is True

    def test_with_dead_letter_disabled(self):
        """Test with_dead_letter with enabled=False."""
        builder = StateBuilder()
        result = builder.with_dead_letter(False)

        assert result is builder
        assert builder._config["dead_letter"] is False

    def test_with_dead_letter_default(self):
        """Test with_dead_letter with default parameter."""
        builder = StateBuilder()
        result = builder.with_dead_letter()

        assert result is builder
        assert builder._config["dead_letter"] is True


class TestCircuitBreakerConfiguration:
    """Test circuit breaker configuration methods."""

    def test_circuit_breaker_basic(self):
        """Test basic circuit breaker configuration."""
        builder = StateBuilder()
        result = builder.circuit_breaker()

        assert result is builder
        assert builder._config["circuit_breaker"] is True
        cb_config = builder._config["circuit_breaker_config"]
        assert cb_config["failure_threshold"] == 5  # default
        assert cb_config["recovery_timeout"] == 60.0  # default
        assert cb_config["success_threshold"] == 3  # default

    def test_circuit_breaker_disabled(self):
        """Test circuit breaker disabled."""
        builder = StateBuilder()
        result = builder.circuit_breaker(enabled=False)

        assert result is builder
        assert builder._config["circuit_breaker"] is False

    def test_circuit_breaker_custom_config(self):
        """Test circuit breaker with custom configuration."""
        builder = StateBuilder()
        result = builder.circuit_breaker(
            enabled=True,
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
        )

        assert result is builder
        assert builder._config["circuit_breaker"] is True
        cb_config = builder._config["circuit_breaker_config"]
        assert cb_config["failure_threshold"] == 3
        assert cb_config["recovery_timeout"] == 30.0
        assert cb_config["success_threshold"] == 2

    def test_with_circuit_breaker(self):
        """Test with_circuit_breaker method."""
        builder = StateBuilder()
        custom_config = {"failure_threshold": 2, "recovery_timeout": 45.0}
        result = builder.with_circuit_breaker(**custom_config)

        assert result is builder
        assert builder._config["circuit_breaker"] is True
        assert builder._config["circuit_breaker_config"] == custom_config

    def test_protected_shortcut(self):
        """Test protected shortcut method."""
        builder = StateBuilder()
        result = builder.protected(failure_threshold=2, recovery_timeout=20.0)

        assert result is builder
        assert builder._config["circuit_breaker"] is True
        cb_config = builder._config["circuit_breaker_config"]
        assert cb_config["failure_threshold"] == 2
        assert cb_config["recovery_timeout"] == 20.0

    def test_fragile_shortcut(self):
        """Test fragile shortcut method."""
        builder = StateBuilder()
        result = builder.fragile(failure_threshold=1, recovery_timeout=180.0)

        assert result is builder
        assert builder._config["circuit_breaker"] is True
        cb_config = builder._config["circuit_breaker_config"]
        assert cb_config["failure_threshold"] == 1
        assert cb_config["recovery_timeout"] == 180.0


class TestBulkheadConfiguration:
    """Test bulkhead configuration methods."""

    def test_bulkhead_basic(self):
        """Test basic bulkhead configuration."""
        builder = StateBuilder()
        result = builder.bulkhead()

        assert result is builder
        assert builder._config["bulkhead"] is True
        bh_config = builder._config["bulkhead_config"]
        assert bh_config["max_concurrent"] == 5  # default
        assert bh_config["max_queue_size"] == 100  # default
        assert bh_config["timeout"] == 30.0  # default

    def test_bulkhead_disabled(self):
        """Test bulkhead disabled."""
        builder = StateBuilder()
        result = builder.bulkhead(enabled=False)

        assert result is builder
        assert builder._config["bulkhead"] is False

    def test_bulkhead_custom_config(self):
        """Test bulkhead with custom configuration."""
        builder = StateBuilder()
        result = builder.bulkhead(
            enabled=True, max_concurrent=3, max_queue_size=50, timeout=15.0
        )

        assert result is builder
        assert builder._config["bulkhead"] is True
        bh_config = builder._config["bulkhead_config"]
        assert bh_config["max_concurrent"] == 3
        assert bh_config["max_queue_size"] == 50
        assert bh_config["timeout"] == 15.0

    def test_with_bulkhead(self):
        """Test with_bulkhead method."""
        builder = StateBuilder()
        custom_config = {"max_concurrent": 2, "timeout": 10.0}
        result = builder.with_bulkhead(**custom_config)

        assert result is builder
        assert builder._config["bulkhead"] is True
        assert builder._config["bulkhead_config"] == custom_config

    def test_isolated_shortcut(self):
        """Test isolated shortcut method."""
        builder = StateBuilder()
        result = builder.isolated(max_concurrent=2)

        assert result is builder
        assert builder._config["bulkhead"] is True
        bh_config = builder._config["bulkhead_config"]
        assert bh_config["max_concurrent"] == 2

    def test_single_threaded_shortcut(self):
        """Test single_threaded shortcut method."""
        builder = StateBuilder()
        result = builder.single_threaded()

        assert result is builder
        assert builder._config["bulkhead"] is True
        bh_config = builder._config["bulkhead_config"]
        assert bh_config["max_concurrent"] == 1

    def test_highly_concurrent_shortcut(self):
        """Test highly_concurrent shortcut method."""
        builder = StateBuilder()
        result = builder.highly_concurrent(max_concurrent=50)

        assert result is builder
        assert builder._config["bulkhead"] is True
        bh_config = builder._config["bulkhead_config"]
        assert bh_config["max_concurrent"] == 50


class TestLeakDetectionConfiguration:
    """Test leak detection configuration methods."""

    def test_leak_detection_enabled(self):
        """Test leak detection enabled."""
        builder = StateBuilder()
        result = builder.leak_detection(True)

        assert result is builder
        assert builder._config["leak_detection"] is True

    def test_leak_detection_disabled(self):
        """Test leak detection disabled."""
        builder = StateBuilder()
        result = builder.leak_detection(False)

        assert result is builder
        assert builder._config["leak_detection"] is False

    def test_leak_detection_default(self):
        """Test leak detection with default parameter."""
        builder = StateBuilder()
        result = builder.leak_detection()

        assert result is builder
        assert builder._config["leak_detection"] is True

    def test_no_leak_detection(self):
        """Test no_leak_detection method."""
        builder = StateBuilder()
        result = builder.no_leak_detection()

        assert result is builder
        assert builder._config["leak_detection"] is False


class TestCombinedReliabilityMethods:
    """Test combined reliability configuration methods."""

    def test_fault_tolerant_basic(self):
        """Test basic fault_tolerant configuration."""
        builder = StateBuilder()
        result = builder.fault_tolerant()

        assert result is builder
        # Should enable circuit breaker
        assert builder._config["circuit_breaker"] is True
        cb_config = builder._config["circuit_breaker_config"]
        assert cb_config["failure_threshold"] == 3  # default

        # Should enable bulkhead
        assert builder._config["bulkhead"] is True
        bh_config = builder._config["bulkhead_config"]
        assert bh_config["max_concurrent"] == 3  # default

        # Should configure retry
        retry_config = builder._config["retry_config"]
        assert retry_config["max_retries"] == 5
        assert retry_config["dead_letter_on_max_retries"] is True

    def test_fault_tolerant_custom(self):
        """Test fault_tolerant with custom parameters."""
        builder = StateBuilder()
        result = builder.fault_tolerant(
            circuit_breaker=False, bulkhead=True, max_concurrent=2, failure_threshold=5
        )

        assert result is builder
        assert builder._config.get("circuit_breaker") is None  # Not enabled
        assert builder._config["bulkhead"] is True
        bh_config = builder._config["bulkhead_config"]
        assert bh_config["max_concurrent"] == 2

    def test_production_ready(self):
        """Test production_ready configuration."""
        builder = StateBuilder()
        result = builder.production_ready()

        assert result is builder
        # Should enable circuit breaker
        assert builder._config["circuit_breaker"] is True
        cb_config = builder._config["circuit_breaker_config"]
        assert cb_config["failure_threshold"] == 5
        assert cb_config["recovery_timeout"] == 60.0

        # Should enable bulkhead
        assert builder._config["bulkhead"] is True
        bh_config = builder._config["bulkhead_config"]
        assert bh_config["max_concurrent"] == 5

        # Should configure retry
        retry_config = builder._config["retry_config"]
        assert retry_config["max_retries"] == 3
        assert retry_config["dead_letter_on_max_retries"] is True

        # Should enable leak detection
        assert builder._config["leak_detection"] is True

    def test_external_call(self):
        """Test external_call configuration."""
        builder = StateBuilder()
        result = builder.external_call(timeout=45.0)

        assert result is builder
        # Should enable circuit breaker with external service settings
        assert builder._config["circuit_breaker"] is True
        cb_config = builder._config["circuit_breaker_config"]
        assert cb_config["failure_threshold"] == 2  # Lower for external calls
        assert cb_config["recovery_timeout"] == 30.0

        # Should enable bulkhead with higher concurrency
        assert builder._config["bulkhead"] is True
        bh_config = builder._config["bulkhead_config"]
        assert bh_config["max_concurrent"] == 10

        # Should set timeout
        assert builder._config["timeout"] == 45.0

        # Should configure retry
        retry_config = builder._config["retry_config"]
        assert retry_config["max_retries"] == 3
        assert retry_config["dead_letter_on_max_retries"] is True


class TestMetadata:
    """Test metadata configuration methods."""

    def test_tag_single(self):
        """Test adding a single tag."""
        builder = StateBuilder()
        result = builder.tag("env", "production")

        assert result is builder
        assert builder._config["tags"] == {"env": "production"}

    def test_tag_multiple_calls(self):
        """Test adding multiple tags with separate calls."""
        builder = StateBuilder()
        result = builder.tag("env", "production").tag("version", "1.0")

        assert result is builder
        assert builder._config["tags"] == {"env": "production", "version": "1.0"}

    def test_tags_multiple(self):
        """Test adding multiple tags at once."""
        builder = StateBuilder()
        result = builder.tags(env="production", version="1.0", team="backend")

        assert result is builder
        assert builder._config["tags"] == {
            "env": "production",
            "version": "1.0",
            "team": "backend",
        }

    def test_tags_merge_with_existing(self):
        """Test that tags merge with existing tags."""
        builder = StateBuilder()
        builder.tag("existing", "value")
        result = builder.tags(new="value", another="tag")

        assert result is builder
        assert builder._config["tags"] == {
            "existing": "value",
            "new": "value",
            "another": "tag",
        }

    def test_description(self):
        """Test description configuration."""
        builder = StateBuilder()
        result = builder.description("Test state description")

        assert result is builder
        assert builder._config["description"] == "Test state description"

    def test_describe_alias(self):
        """Test describe as alias for description."""
        builder = StateBuilder()
        result = builder.describe("Test description")

        assert result is builder
        assert builder._config["description"] == "Test description"


class TestProfileApplication:
    """Test profile application methods."""

    def test_profile(self):
        """Test profile application."""
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


class TestAdvancedOptions:
    """Test advanced configuration options."""

    def test_preemptible_default(self):
        """Test preemptible with default value."""
        builder = StateBuilder()
        result = builder.preemptible()

        assert result is builder
        assert builder._config["preemptible"] is True

    def test_preemptible_explicit(self):
        """Test preemptible with explicit value."""
        builder = StateBuilder()
        result = builder.preemptible(False)

        assert result is builder
        assert builder._config["preemptible"] is False

    def test_checkpoint_every(self):
        """Test checkpoint_every configuration."""
        builder = StateBuilder()
        result = builder.checkpoint_every(30.0)

        assert result is builder
        assert builder._config["checkpoint_interval"] == 30.0

    def test_cleanup_on_failure_default(self):
        """Test cleanup_on_failure with default value."""
        builder = StateBuilder()
        result = builder.cleanup_on_failure()

        assert result is builder
        assert builder._config["cleanup_on_failure"] is True

    def test_cleanup_on_failure_explicit(self):
        """Test cleanup_on_failure with explicit value."""
        builder = StateBuilder()
        result = builder.cleanup_on_failure(False)

        assert result is builder
        assert builder._config["cleanup_on_failure"] is False


class TestBuildMethods:
    """Test build and decorator methods."""

    def test_build(self):
        """Test build method returns copy of config."""
        builder = StateBuilder()
        builder.cpu(2.0).memory(512.0).high_priority()

        config = builder.build()

        assert isinstance(config, dict)
        assert config == {"cpu": 2.0, "memory": 512.0, "priority": Priority.HIGH}

        # Should be a copy, not the same object
        assert config is not builder._config

    @patch("puffinflow.core.agent.decorators.flexible.state")
    def test_call_as_decorator(self, mock_state):
        """Test using builder as decorator."""
        builder = StateBuilder()
        builder.cpu(2.0)

        mock_decorator = Mock()
        mock_state.return_value = mock_decorator

        def test_func():
            return "test"

        builder(test_func)

        # Should call state with config
        mock_state.assert_called_once_with(config=builder._config)
        mock_decorator.assert_called_once_with(test_func)

    @patch("puffinflow.core.agent.decorators.flexible.state")
    def test_decorator_method(self, mock_state):
        """Test decorator method."""
        builder = StateBuilder()
        builder.memory(256.0)

        mock_decorator = Mock()
        mock_state.return_value = mock_decorator

        result = builder.decorator()

        # Should call state with config and return the decorator
        mock_state.assert_called_once_with(config=builder._config)
        assert result == mock_decorator


class TestFluentInterface:
    """Test fluent interface chaining."""

    def test_method_chaining(self):
        """Test that methods can be chained together."""
        builder = StateBuilder()

        result = (
            builder.cpu(2.0)
            .memory(512.0)
            .high_priority()
            .timeout(60.0)
            .mutex()
            .retries(3)
            .tag("env", "test")
            .description("Chained configuration")
        )

        assert result is builder
        config = builder._config
        assert config["cpu"] == 2.0
        assert config["memory"] == 512.0
        assert config["priority"] == Priority.HIGH
        assert config["timeout"] == 60.0
        assert config["mutex"] is True
        assert config["max_retries"] == 3
        assert config["tags"] == {"env": "test"}
        assert config["description"] == "Chained configuration"

    def test_complex_chaining(self):
        """Test complex method chaining with reliability patterns."""
        builder = StateBuilder()

        result = (
            builder.resources(cpu=4.0, memory=1024.0, gpu=1.0)
            .critical_priority()
            .timeout(300.0)
            .depends_on("state1", "state2")
            .circuit_breaker(failure_threshold=3, recovery_timeout=45.0)
            .bulkhead(max_concurrent=2, max_queue_size=50)
            .retry(5, initial_delay=2.0, dead_letter=True)
            .tags(workload="gpu", importance="critical")
            .leak_detection(True)
        )

        assert result is builder
        config = builder._config

        # Verify all configurations
        assert config["cpu"] == 4.0
        assert config["memory"] == 1024.0
        assert config["gpu"] == 1.0
        assert config["priority"] == Priority.CRITICAL
        assert config["timeout"] == 300.0
        assert config["depends_on"] == ["state1", "state2"]
        assert config["circuit_breaker"] is True
        assert config["bulkhead"] is True
        assert config["leak_detection"] is True

        # Verify nested configurations
        cb_config = config["circuit_breaker_config"]
        assert cb_config["failure_threshold"] == 3
        assert cb_config["recovery_timeout"] == 45.0

        bh_config = config["bulkhead_config"]
        assert bh_config["max_concurrent"] == 2
        assert bh_config["max_queue_size"] == 50

        retry_config = config["retry_config"]
        assert retry_config["max_retries"] == 5
        assert retry_config["initial_delay"] == 2.0
        assert retry_config["dead_letter_on_max_retries"] is True

        assert config["tags"] == {"workload": "gpu", "importance": "critical"}


class TestConvenienceFunctions:
    """Test convenience builder functions."""

    def test_build_state(self):
        """Test build_state function."""
        builder = build_state()
        assert isinstance(builder, StateBuilder)
        assert builder._config == {}

    def test_cpu_state(self):
        """Test cpu_state convenience function."""
        builder = cpu_state(4.0)
        assert isinstance(builder, StateBuilder)
        assert builder._config["cpu"] == 4.0

    def test_memory_state(self):
        """Test memory_state convenience function."""
        builder = memory_state(1024.0)
        assert isinstance(builder, StateBuilder)
        assert builder._config["memory"] == 1024.0

    def test_gpu_state(self):
        """Test gpu_state convenience function."""
        builder = gpu_state(2.0)
        assert isinstance(builder, StateBuilder)
        assert builder._config["gpu"] == 2.0

    def test_exclusive_state(self):
        """Test exclusive_state convenience function."""
        builder = exclusive_state()
        assert isinstance(builder, StateBuilder)
        assert builder._config["mutex"] is True

    def test_concurrent_state(self):
        """Test concurrent_state convenience function."""
        builder = concurrent_state(5)
        assert isinstance(builder, StateBuilder)
        assert builder._config["semaphore"] == 5

    def test_high_priority_state(self):
        """Test high_priority_state convenience function."""
        builder = high_priority_state()
        assert isinstance(builder, StateBuilder)
        assert builder._config["priority"] == Priority.HIGH

    def test_critical_state(self):
        """Test critical_state convenience function."""
        builder = critical_state()
        assert isinstance(builder, StateBuilder)
        assert builder._config["priority"] == Priority.CRITICAL

    def test_fault_tolerant_state(self):
        """Test fault_tolerant_state convenience function."""
        builder = fault_tolerant_state()
        assert isinstance(builder, StateBuilder)
        assert builder._config["circuit_breaker"] is True
        assert builder._config["bulkhead"] is True
        assert "retry_config" in builder._config

    def test_external_service_state(self):
        """Test external_service_state convenience function."""
        builder = external_service_state(timeout=45.0)
        assert isinstance(builder, StateBuilder)
        assert builder._config["circuit_breaker"] is True
        assert builder._config["bulkhead"] is True
        assert builder._config["timeout"] == 45.0

    def test_production_state(self):
        """Test production_state convenience function."""
        builder = production_state()
        assert isinstance(builder, StateBuilder)
        assert builder._config["circuit_breaker"] is True
        assert builder._config["bulkhead"] is True
        assert builder._config["leak_detection"] is True
        assert "retry_config" in builder._config

    def test_protected_state(self):
        """Test protected_state convenience function."""
        builder = protected_state(failure_threshold=2)
        assert isinstance(builder, StateBuilder)
        assert builder._config["circuit_breaker"] is True
        cb_config = builder._config["circuit_breaker_config"]
        assert cb_config["failure_threshold"] == 2

    def test_isolated_state(self):
        """Test isolated_state convenience function."""
        builder = isolated_state(max_concurrent=1)
        assert isinstance(builder, StateBuilder)
        assert builder._config["bulkhead"] is True
        bh_config = builder._config["bulkhead_config"]
        assert bh_config["max_concurrent"] == 1


class TestBuilderIntegration:
    """Test builder integration with actual decoration."""

    @patch("puffinflow.core.agent.decorators.flexible.state")
    def test_builder_as_decorator_integration(self, mock_state):
        """Test using builder as decorator in realistic scenario."""
        # Setup mock
        mock_decorated_func = Mock()
        mock_decorator = Mock(return_value=mock_decorated_func)
        mock_state.return_value = mock_decorator

        # Create builder with complex configuration
        builder = (
            StateBuilder()
            .cpu(2.0)
            .memory(512.0)
            .high_priority()
            .timeout(60.0)
            .circuit_breaker(failure_threshold=3)
            .bulkhead(max_concurrent=5)
            .retry(3, dead_letter=True)
            .tags(env="production", service="api")
        )

        # Use as decorator
        @builder
        def test_function():
            """Test function."""
            return "result"

        # Verify state was called with correct config
        expected_config = {
            "cpu": 2.0,
            "memory": 512.0,
            "priority": Priority.HIGH,
            "timeout": 60.0,
            "circuit_breaker": True,
            "circuit_breaker_config": {
                "failure_threshold": 3,
                "recovery_timeout": 60.0,
                "success_threshold": 3,
            },
            "bulkhead": True,
            "bulkhead_config": {
                "max_concurrent": 5,
                "max_queue_size": 100,
                "timeout": 30.0,
            },
            "retry_config": {
                "max_retries": 3,
                "initial_delay": 1.0,
                "exponential_base": 2.0,
                "jitter": True,
                "dead_letter_on_max_retries": True,
                "dead_letter_on_timeout": True,
            },
            "tags": {"env": "production", "service": "api"},
        }

        mock_state.assert_called_once_with(config=expected_config)
        mock_decorator.assert_called_once()
        assert test_function == mock_decorated_func

    def test_builder_chaining_preserves_all_config(self):
        """Test that complex chaining preserves all configuration."""
        builder = (
            StateBuilder()
            .resources(cpu=4.0, memory=2048.0, gpu=1.0, io=2.0, network=3.0)
            .critical_priority()
            .timeout(300.0)
            .rate_limit(10.0, burst=20)
            .depends_on("state1", "state2", "state3")
            .mutex()
            .retry(5, initial_delay=2.0, exponential_base=1.5, jitter=False)
            .circuit_breaker(failure_threshold=2, recovery_timeout=30.0)
            .bulkhead(max_concurrent=3, max_queue_size=50, timeout=15.0)
            .leak_detection(False)
            .tags(workload="gpu", team="ml", priority="critical")
            .description("Complex GPU processing state")
            .preemptible(False)
            .checkpoint_every(60.0)
            .cleanup_on_failure(True)
        )

        config = builder.build()

        # Verify all configurations are preserved
        assert config["cpu"] == 4.0
        assert config["memory"] == 2048.0
        assert config["gpu"] == 1.0
        assert config["io"] == 2.0
        assert config["network"] == 3.0
        assert config["priority"] == Priority.CRITICAL
        assert config["timeout"] == 300.0
        assert config["rate_limit"] == 10.0
        assert config["burst_limit"] == 20
        assert config["depends_on"] == ["state1", "state2", "state3"]
        assert config["mutex"] is True
        assert config["circuit_breaker"] is True
        assert config["bulkhead"] is True
        assert config["leak_detection"] is False
        assert config["tags"] == {
            "workload": "gpu",
            "team": "ml",
            "priority": "critical",
        }
        assert config["description"] == "Complex GPU processing state"
        assert config["preemptible"] is False
        assert config["checkpoint_interval"] == 60.0
        assert config["cleanup_on_failure"] is True

        # Verify nested configurations
        retry_config = config["retry_config"]
        assert retry_config["max_retries"] == 5
        assert retry_config["initial_delay"] == 2.0
        assert retry_config["exponential_base"] == 1.5
        assert retry_config["jitter"] is False

        cb_config = config["circuit_breaker_config"]
        assert cb_config["failure_threshold"] == 2
        assert cb_config["recovery_timeout"] == 30.0

        bh_config = config["bulkhead_config"]
        assert bh_config["max_concurrent"] == 3
        assert bh_config["max_queue_size"] == 50
        assert bh_config["timeout"] == 15.0


class TestBuilderEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_builder_build(self):
        """Test building empty builder."""
        builder = StateBuilder()
        config = builder.build()
        assert config == {}

    def test_overriding_configuration(self):
        """Test that later configuration overrides earlier."""
        builder = StateBuilder()

        # Set CPU twice
        builder.cpu(1.0).cpu(2.0)
        assert builder._config["cpu"] == 2.0

        # Set priority twice
        builder.high_priority().low_priority()
        assert builder._config["priority"] == Priority.LOW

    def test_tags_accumulation(self):
        """Test that tags accumulate correctly."""
        builder = StateBuilder()

        # Add tags in different ways
        builder.tag("first", "value1")
        builder.tag("second", "value2")
        builder.tags(third="value3", fourth="value4")

        expected_tags = {
            "first": "value1",
            "second": "value2",
            "third": "value3",
            "fourth": "value4",
        }
        assert builder._config["tags"] == expected_tags

    def test_tags_overriding(self):
        """Test that tags can be overridden."""
        builder = StateBuilder()

        builder.tag("key", "original")
        builder.tag("key", "updated")

        assert builder._config["tags"]["key"] == "updated"

    def test_multiple_coordination_methods(self):
        """Test setting multiple coordination methods."""
        builder = StateBuilder()

        # Set multiple coordination types
        builder.mutex().semaphore(5).barrier(3)

        # All should be set (though only one would be used in practice)
        assert builder._config["mutex"] is True
        assert builder._config["semaphore"] == 5
        assert builder._config["barrier"] == 3

    def test_reliability_patterns_interaction(self):
        """Test interaction between reliability patterns."""
        builder = StateBuilder()

        # Configure overlapping reliability patterns
        builder.fault_tolerant(circuit_breaker=True, bulkhead=True)
        builder.production_ready()  # Should override some settings

        config = builder.build()

        # Later configuration should win
        assert config["circuit_breaker"] is True
        assert config["bulkhead"] is True
        assert config["leak_detection"] is True

        # Should have retry config from production_ready (last call)
        retry_config = config["retry_config"]
        assert retry_config["max_retries"] == 3  # From production_ready
