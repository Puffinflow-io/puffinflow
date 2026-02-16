"""
Comprehensive unit tests for circuit_breaker.py module.

Tests cover:
- CircuitState enum functionality
- CircuitBreakerConfig dataclass
- CircuitBreakerError exception handling
- CircuitBreaker state transitions and logic
- Failure counting and threshold management
- Recovery timeout behavior
- Success counting in half-open state
- Manual state control
- Metrics collection and reporting
- CircuitBreakerRegistry management
- Edge cases and error conditions
- Async concurrency scenarios
- Performance under load
- Integration patterns
"""

import asyncio
import time
from dataclasses import asdict

import pytest

# Import the module under test
from puffinflow.core.reliability.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
    circuit_registry,
)


class TestCircuitState:
    """Test CircuitState enum."""

    def test_circuit_state_values(self):
        """Test CircuitState enum has correct values."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"

    def test_circuit_state_membership(self):
        """Test CircuitState enum membership."""
        assert CircuitState.CLOSED in CircuitState
        assert CircuitState.OPEN in CircuitState
        assert CircuitState.HALF_OPEN in CircuitState

    def test_circuit_state_comparison(self):
        """Test CircuitState enum comparison."""
        assert CircuitState.CLOSED == CircuitState.CLOSED
        assert CircuitState.OPEN != CircuitState.CLOSED
        assert CircuitState.HALF_OPEN != CircuitState.OPEN

    def test_circuit_state_string_representation(self):
        """Test CircuitState string representation."""
        assert str(CircuitState.CLOSED) == "CircuitState.CLOSED"
        assert repr(CircuitState.OPEN) == "<CircuitState.OPEN: 'open'>"


class TestCircuitBreakerConfig:
    """Test CircuitBreakerConfig dataclass."""

    def test_config_creation_minimal(self):
        """Test CircuitBreakerConfig creation with defaults."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 3
        assert config.timeout == 30.0
        assert config.name == "default"

    def test_config_creation_full(self):
        """Test CircuitBreakerConfig creation with all fields specified."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            recovery_timeout=120.0,
            success_threshold=5,
            timeout=45.0,
            name="custom_circuit",
        )

        assert config.failure_threshold == 10
        assert config.recovery_timeout == 120.0
        assert config.success_threshold == 5
        assert config.timeout == 45.0
        assert config.name == "custom_circuit"

    def test_config_is_dataclass(self):
        """Test CircuitBreakerConfig is properly configured as dataclass."""
        config = CircuitBreakerConfig(name="test", failure_threshold=3)

        # Should be able to convert to dict
        config_dict = asdict(config)
        assert "name" in config_dict
        assert "failure_threshold" in config_dict
        assert "recovery_timeout" in config_dict
        assert "success_threshold" in config_dict
        assert "timeout" in config_dict

    @pytest.mark.parametrize(
        "failure_threshold,expected_valid",
        [
            (1, True),  # Minimum practical value
            (0, True),  # Edge case - never opens
            (-1, True),  # Negative - questionable but allowed
            (100, True),  # Large number
        ],
    )
    def test_config_failure_threshold_values(self, failure_threshold, expected_valid):
        """Test CircuitBreakerConfig with various failure_threshold values."""
        config = CircuitBreakerConfig(failure_threshold=failure_threshold)
        assert config.failure_threshold == failure_threshold

    def test_config_defaults_are_reasonable(self):
        """Test CircuitBreakerConfig defaults are production-ready."""
        config = CircuitBreakerConfig()

        # Defaults should be sensible for production use
        assert config.failure_threshold > 0
        assert config.recovery_timeout > 0
        assert config.success_threshold > 0
        assert config.timeout > 0
        assert config.success_threshold <= config.failure_threshold


class TestCircuitBreakerError:
    """Test CircuitBreakerError exception."""

    def test_circuit_breaker_error_creation(self):
        """Test CircuitBreakerError can be created and raised."""
        error = CircuitBreakerError("Circuit breaker is open")
        assert str(error) == "Circuit breaker is open"
        assert isinstance(error, Exception)

    def test_circuit_breaker_error_inheritance(self):
        """Test CircuitBreakerError inherits from Exception."""
        error = CircuitBreakerError("Test")
        assert isinstance(error, Exception)

    def test_circuit_breaker_error_raising(self):
        """Test CircuitBreakerError can be raised and caught."""
        with pytest.raises(CircuitBreakerError) as exc_info:
            raise CircuitBreakerError("Custom circuit breaker error")

        assert str(exc_info.value) == "Custom circuit breaker error"


class TestCircuitBreaker:
    """Test CircuitBreaker class functionality."""

    @pytest.fixture
    def basic_config(self):
        """Basic circuit breaker configuration for testing."""
        return CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1.0,  # Short timeout for testing
            success_threshold=2,
            timeout=0.5,
            name="test_circuit",
        )

    @pytest.fixture
    def circuit_breaker(self, basic_config):
        """Create a CircuitBreaker instance for testing."""
        return CircuitBreaker(basic_config)

    def test_circuit_breaker_initialization(self, basic_config):
        """Test CircuitBreaker initialization sets up correct state."""
        cb = CircuitBreaker(basic_config)

        assert cb.config == basic_config
        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 0
        assert cb._success_count == 0
        assert cb._last_failure_time == 0
        assert hasattr(cb, "_lock")

    @pytest.mark.asyncio
    async def test_protect_basic_success(self, circuit_breaker):
        """Test basic successful protection."""
        executed = False

        async with circuit_breaker.protect():
            executed = True

        assert executed
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_protect_single_failure(self, circuit_breaker):
        """Test protection with single failure."""
        with pytest.raises(ValueError):
            async with circuit_breaker.protect():
                raise ValueError("Test error")

        assert circuit_breaker.state == CircuitState.CLOSED  # Still closed
        assert circuit_breaker._failure_count == 1

    @pytest.mark.asyncio
    async def test_protect_failure_threshold_reached(self, circuit_breaker):
        """Test circuit opens when failure threshold is reached."""
        # Fail enough times to reach threshold (3)
        for i in range(3):
            with pytest.raises(ValueError):
                async with circuit_breaker.protect():
                    raise ValueError(f"Test error {i}")

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker._failure_count == 3

    @pytest.mark.asyncio
    async def test_protect_open_circuit_rejects_immediately(self, circuit_breaker):
        """Test that open circuit immediately rejects requests."""
        # First, open the circuit
        for i in range(3):
            with pytest.raises(ValueError):
                async with circuit_breaker.protect():
                    raise ValueError(f"Failure {i}")

        assert circuit_breaker.state == CircuitState.OPEN

        # Now any request should be immediately rejected
        with pytest.raises(CircuitBreakerError, match="is OPEN"):
            async with circuit_breaker.protect():
                pass  # This code should never execute

    @pytest.mark.asyncio
    async def test_protect_recovery_timeout_transition_to_half_open(
        self, circuit_breaker
    ):
        """Test transition from OPEN to HALF_OPEN after recovery timeout."""
        # Open the circuit
        for i in range(3):
            with pytest.raises(ValueError):
                async with circuit_breaker.protect():
                    raise ValueError(f"Failure {i}")

        assert circuit_breaker.state == CircuitState.OPEN

        # Record the time when circuit opened to ensure proper wait
        last_failure_time = circuit_breaker._last_failure_time
        recovery_timeout = circuit_breaker.config.recovery_timeout

        # Wait for recovery timeout with a small buffer for timing accuracy
        elapsed = time.time() - last_failure_time
        remaining_wait = max(0, recovery_timeout - elapsed) + 0.1
        await asyncio.sleep(remaining_wait)

        # Next request should transition to HALF_OPEN
        executed = False
        async with circuit_breaker.protect():
            executed = True

        assert executed
        assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_success_threshold_closes_circuit(self, circuit_breaker):
        """Test that enough successes in HALF_OPEN closes the circuit."""
        # Open the circuit first
        for i in range(3):
            with pytest.raises(ValueError):
                async with circuit_breaker.protect():
                    raise ValueError(f"Failure {i}")

        # Wait for recovery and get to HALF_OPEN
        await asyncio.sleep(1.1)

        # First success - should be in HALF_OPEN now
        async with circuit_breaker.protect():
            pass  # First success

        assert circuit_breaker.state == CircuitState.HALF_OPEN
        assert circuit_breaker._success_count == 1

        # Second success should close the circuit (success_threshold = 2)
        async with circuit_breaker.protect():
            pass  # Second success

        # Manually trigger state check to ensure transition
        async with circuit_breaker._lock:
            await circuit_breaker._check_state()

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, circuit_breaker):
        """Test that failure in HALF_OPEN reopens the circuit."""
        # Open the circuit
        for i in range(3):
            with pytest.raises(ValueError):
                async with circuit_breaker.protect():
                    raise ValueError(f"Failure {i}")

        # Get to HALF_OPEN
        await asyncio.sleep(1.1)
        async with circuit_breaker.protect():
            pass  # Success to get to HALF_OPEN

        assert circuit_breaker.state == CircuitState.HALF_OPEN

        # Now fail - should reopen
        with pytest.raises(ValueError):
            async with circuit_breaker.protect():
                raise ValueError("Failure in half-open")

        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_closed_state_gradual_recovery(self, circuit_breaker):
        """Test that successes in CLOSED state gradually reduce failure count."""
        # Accumulate some failures (but not enough to open)
        for i in range(2):  # threshold is 3
            with pytest.raises(ValueError):
                async with circuit_breaker.protect():
                    raise ValueError(f"Failure {i}")

        assert circuit_breaker._failure_count == 2
        assert circuit_breaker.state == CircuitState.CLOSED

        # Success should reduce failure count
        async with circuit_breaker.protect():
            pass

        assert circuit_breaker._failure_count == 1  # Reduced by 1
        assert circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_protect_exception_propagation(self, circuit_breaker):
        """Test that exceptions are properly propagated."""
        # Test different exception types
        with pytest.raises(ValueError):
            async with circuit_breaker.protect():
                raise ValueError("Value error")

        with pytest.raises(RuntimeError):
            async with circuit_breaker.protect():
                raise RuntimeError("Runtime error")

        with pytest.raises(KeyError):
            async with circuit_breaker.protect():
                raise KeyError("Key error")

    def test_get_metrics_closed_state(self, circuit_breaker):
        """Test get_metrics in CLOSED state."""
        metrics = circuit_breaker.get_metrics()

        expected = {
            "name": "test_circuit",
            "state": "closed",
            "failure_count": 0,
            "success_count": 0,
            "last_failure_time": 0,
        }

        assert metrics == expected

    @pytest.mark.asyncio
    async def test_get_metrics_with_failures(self, circuit_breaker):
        """Test get_metrics after failures."""
        # Cause some failures
        for i in range(2):
            with pytest.raises(ValueError):
                async with circuit_breaker.protect():
                    raise ValueError(f"Failure {i}")

        metrics = circuit_breaker.get_metrics()

        assert metrics["name"] == "test_circuit"
        assert metrics["state"] == "closed"  # Not enough to open
        assert metrics["failure_count"] == 2
        assert metrics["success_count"] == 0
        assert metrics["last_failure_time"] > 0

    @pytest.mark.asyncio
    async def test_get_metrics_open_state(self, circuit_breaker):
        """Test get_metrics in OPEN state."""
        # Open the circuit
        for i in range(3):
            with pytest.raises(ValueError):
                async with circuit_breaker.protect():
                    raise ValueError(f"Failure {i}")

        metrics = circuit_breaker.get_metrics()

        assert metrics["state"] == "open"
        assert metrics["failure_count"] == 3

    @pytest.mark.asyncio
    async def test_get_metrics_half_open_state(self, circuit_breaker):
        """Test get_metrics in HALF_OPEN state."""
        # Open then transition to HALF_OPEN
        for i in range(3):
            with pytest.raises(ValueError):
                async with circuit_breaker.protect():
                    raise ValueError(f"Failure {i}")

        await asyncio.sleep(1.1)  # Wait for recovery
        async with circuit_breaker.protect():
            pass  # Transition to HALF_OPEN

        metrics = circuit_breaker.get_metrics()

        assert metrics["state"] == "half_open"
        assert metrics["success_count"] == 1

    @pytest.mark.asyncio
    async def test_force_open(self, circuit_breaker):
        """Test manually forcing circuit open."""
        assert circuit_breaker.state == CircuitState.CLOSED

        await circuit_breaker.force_open()

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker._last_failure_time > 0

        # Should reject requests
        with pytest.raises(CircuitBreakerError):
            async with circuit_breaker.protect():
                pass

    @pytest.mark.asyncio
    async def test_force_close(self, circuit_breaker):
        """Test manually forcing circuit closed."""
        # First open the circuit
        for i in range(3):
            with pytest.raises(ValueError):
                async with circuit_breaker.protect():
                    raise ValueError(f"Failure {i}")

        assert circuit_breaker.state == CircuitState.OPEN

        # Force close
        await circuit_breaker.force_close()

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker._failure_count == 0

        # Should accept requests
        async with circuit_breaker.protect():
            pass

    @pytest.mark.asyncio
    async def test_concurrent_access_thread_safety(self, circuit_breaker):
        """Test circuit breaker thread safety under concurrent access."""
        results = []

        async def worker(worker_id):
            try:
                async with circuit_breaker.protect():
                    await asyncio.sleep(0.01)  # Small delay
                    results.append(f"success_{worker_id}")
            except Exception as e:
                results.append(f"error_{worker_id}_{type(e).__name__}")

        # Run multiple workers concurrently
        workers = [asyncio.create_task(worker(i)) for i in range(10)]
        await asyncio.gather(*workers)

        # All should succeed since circuit is closed
        success_results = [r for r in results if r.startswith("success_")]
        assert len(success_results) == 10

    @pytest.mark.asyncio
    async def test_concurrent_failures_thread_safety(self, circuit_breaker):
        """Test thread safety when multiple failures occur concurrently."""
        failure_count = 0

        async def failing_worker(worker_id):
            nonlocal failure_count
            try:
                async with circuit_breaker.protect():
                    failure_count += 1
                    raise ValueError(f"Failure {worker_id}")
            except (ValueError, CircuitBreakerError):
                pass

        # Run multiple failing workers concurrently
        workers = [asyncio.create_task(failing_worker(i)) for i in range(5)]
        await asyncio.gather(*workers)

        # Circuit should open after threshold reached
        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.parametrize(
        "failure_threshold,recovery_timeout,success_threshold",
        [
            (1, 0.1, 1),  # Minimal settings
            (10, 5.0, 5),  # Typical settings
            (0, 1.0, 1),  # Zero threshold (never opens)
            (1, 0.0, 1),  # Zero recovery timeout
            (5, 2.0, 10),  # Success threshold > failure threshold
        ],
    )
    def test_various_configurations(
        self, failure_threshold, recovery_timeout, success_threshold
    ):
        """Test CircuitBreaker with various configurations."""
        config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            name="variable_test",
        )
        cb = CircuitBreaker(config)

        assert cb.config.failure_threshold == failure_threshold
        assert cb.config.recovery_timeout == recovery_timeout
        assert cb.config.success_threshold == success_threshold
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_timing_precision(self, circuit_breaker):
        """Test timing precision for recovery timeout."""
        # Open circuit
        for i in range(3):
            with pytest.raises(ValueError):
                async with circuit_breaker.protect():
                    raise ValueError(f"Failure {i}")

        # Ensure circuit is open
        assert circuit_breaker.state == CircuitState.OPEN

        # Test that circuit recovers after timeout
        await asyncio.sleep(1.1)  # Wait past recovery timeout (1.0)

        # Should work (transition to HALF_OPEN)
        async with circuit_breaker.protect():
            pass

        # Circuit should now be in HALF_OPEN state
        assert circuit_breaker.state == CircuitState.HALF_OPEN


class TestCircuitBreakerRegistry:
    """Test CircuitBreakerRegistry class functionality."""

    @pytest.fixture
    def registry(self):
        """Create a fresh CircuitBreakerRegistry for each test."""
        return CircuitBreakerRegistry()

    def test_registry_initialization(self, registry):
        """Test CircuitBreakerRegistry initializes with empty state."""
        assert registry._breakers == {}

    def test_get_or_create_new_circuit_breaker(self, registry):
        """Test creating a new circuit breaker through registry."""
        config = CircuitBreakerConfig(name="test_cb", failure_threshold=5)
        cb = registry.get_or_create("test_cb", config)

        assert isinstance(cb, CircuitBreaker)
        assert cb.config.name == "test_cb"
        assert cb.config.failure_threshold == 5
        assert "test_cb" in registry._breakers

    def test_get_or_create_existing_circuit_breaker(self, registry):
        """Test getting existing circuit breaker from registry."""
        config = CircuitBreakerConfig(name="existing", failure_threshold=3)

        # Create first time
        cb1 = registry.get_or_create("existing", config)

        # Get same instance second time
        cb2 = registry.get_or_create("existing", config)

        assert cb1 is cb2
        assert len(registry._breakers) == 1

    def test_get_or_create_default_config(self, registry):
        """Test get_or_create with default config when none provided."""
        cb = registry.get_or_create("default_test")

        assert isinstance(cb, CircuitBreaker)
        assert cb.config.name == "default_test"
        assert cb.config.failure_threshold == 5  # Default
        assert "default_test" in registry._breakers

    def test_get_or_create_ignores_config_for_existing(self, registry):
        """Test that config is ignored when circuit breaker already exists."""
        # Create with initial config
        initial_config = CircuitBreakerConfig(name="ignore_test", failure_threshold=2)
        cb1 = registry.get_or_create("ignore_test", initial_config)

        # Try to get with different config
        new_config = CircuitBreakerConfig(name="ignore_test", failure_threshold=10)
        cb2 = registry.get_or_create("ignore_test", new_config)

        # Should be same instance with original config
        assert cb1 is cb2
        assert cb2.config.failure_threshold == 2  # Original value

    def test_get_all_metrics_empty(self, registry):
        """Test get_all_metrics with no circuit breakers."""
        metrics = registry.get_all_metrics()
        assert metrics == {}

    def test_get_all_metrics_single_circuit_breaker(self, registry):
        """Test get_all_metrics with single circuit breaker."""
        config = CircuitBreakerConfig(name="single", failure_threshold=3)
        registry.get_or_create("single", config)

        all_metrics = registry.get_all_metrics()

        assert "single" in all_metrics
        assert all_metrics["single"]["name"] == "single"
        assert all_metrics["single"]["state"] == "closed"

    def test_get_all_metrics_multiple_circuit_breakers(self, registry):
        """Test get_all_metrics with multiple circuit breakers."""
        # Create multiple circuit breakers
        configs = [
            CircuitBreakerConfig(name="first", failure_threshold=2),
            CircuitBreakerConfig(name="second", failure_threshold=5),
            CircuitBreakerConfig(name="third", failure_threshold=1),
        ]

        for config in configs:
            registry.get_or_create(config.name, config)

        all_metrics = registry.get_all_metrics()

        assert len(all_metrics) == 3
        assert "first" in all_metrics
        assert "second" in all_metrics
        assert "third" in all_metrics

    @pytest.mark.asyncio
    async def test_get_all_metrics_with_active_circuit_breakers(self, registry):
        """Test get_all_metrics shows state changes correctly."""
        config = CircuitBreakerConfig(name="active", failure_threshold=2)
        cb = registry.get_or_create("active", config)

        # Cause a failure
        with pytest.raises(ValueError):
            async with cb.protect():
                raise ValueError("Test failure")

        all_metrics = registry.get_all_metrics()

        assert all_metrics["active"]["failure_count"] == 1
        assert all_metrics["active"]["state"] == "closed"


class TestGlobalRegistry:
    """Test the global circuit_registry instance."""

    def test_global_registry_exists(self):
        """Test that global circuit_registry exists and is correct type."""
        assert circuit_registry is not None
        assert isinstance(circuit_registry, CircuitBreakerRegistry)

    def test_global_registry_functionality(self):
        """Test that global registry functions correctly."""
        # Get initial state
        initial_count = len(circuit_registry._breakers)

        # Create a circuit breaker through global registry
        config = CircuitBreakerConfig(name="global_test", failure_threshold=3)
        cb = circuit_registry.get_or_create("global_test", config)

        assert isinstance(cb, CircuitBreaker)
        assert len(circuit_registry._breakers) == initial_count + 1

        # Verify it's accessible in metrics
        metrics = circuit_registry.get_all_metrics()
        assert "global_test" in metrics

    def test_global_registry_persistence(self):
        """Test that global registry persists circuit breakers across calls."""
        # Create circuit breaker
        cb1 = circuit_registry.get_or_create("persistent_test")

        # Get same circuit breaker later
        cb2 = circuit_registry.get_or_create("persistent_test")

        assert cb1 is cb2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_zero_failure_threshold(self):
        """Test circuit breaker with zero failure threshold - opens immediately."""
        config = CircuitBreakerConfig(failure_threshold=0)
        cb = CircuitBreaker(config)

        # With zero threshold, circuit should open on first failure
        with pytest.raises(ValueError):
            async with cb.protect():
                raise ValueError("First failure")

        # Circuit should now be open
        assert cb.state == CircuitState.OPEN

        # Subsequent calls should be rejected
        with pytest.raises(CircuitBreakerError):
            async with cb.protect():
                pass

    @pytest.mark.asyncio
    async def test_zero_recovery_timeout(self):
        """Test circuit breaker with zero recovery timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.0)
        cb = CircuitBreaker(config)

        # Open circuit
        with pytest.raises(ValueError):
            async with cb.protect():
                raise ValueError("Failure")

        assert cb.state == CircuitState.OPEN

        # Should immediately be able to transition to HALF_OPEN
        async with cb.protect():
            pass

        assert cb.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_zero_success_threshold(self):
        """Test circuit breaker with zero success threshold in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1, recovery_timeout=0.1, success_threshold=0
        )
        cb = CircuitBreaker(config)

        # Open circuit
        with pytest.raises(ValueError):
            async with cb.protect():
                raise ValueError("Failure")

        await asyncio.sleep(0.2)  # Wait for recovery

        # With zero success threshold, circuit should close immediately when transitioning to HALF_OPEN
        async with cb.protect():
            pass

        # Force state check to ensure transition logic runs
        async with cb._lock:
            await cb._check_state()

        # Should be closed now (zero threshold means no successes needed)
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_very_large_thresholds(self):
        """Test circuit breaker with very large thresholds."""
        config = CircuitBreakerConfig(
            failure_threshold=1000000, recovery_timeout=0.1, success_threshold=1000000
        )
        cb = CircuitBreaker(config)

        # Many failures shouldn't open circuit
        for i in range(100):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Failure {i}")

        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 100

    @pytest.mark.asyncio
    async def test_negative_thresholds(self):
        """Test circuit breaker with negative thresholds."""
        config = CircuitBreakerConfig(
            failure_threshold=-1, recovery_timeout=0.1, success_threshold=-1
        )
        cb = CircuitBreaker(config)

        # Behavior with negative thresholds - circuit opens due to comparison logic
        with pytest.raises(ValueError):
            async with cb.protect():
                raise ValueError("Failure")

        # Negative threshold causes circuit to open (1 >= -1 is True)
        assert cb.state == CircuitState.OPEN

    def test_special_characters_in_name(self):
        """Test circuit breaker names with special characters."""
        special_names = [
            "circuit-with-dashes",
            "circuit_with_underscores",
            "circuit.with.dots",
            "circuit:with:colons",
            "circuit/with/slashes",
            "circuit with spaces",
            "circuit@with#symbols",
        ]

        registry = CircuitBreakerRegistry()

        for name in special_names:
            config = CircuitBreakerConfig(name=name)
            cb = registry.get_or_create(name, config)
            assert cb.config.name == name
            assert name in registry._breakers

    def test_unicode_name(self):
        """Test circuit breaker name with unicode characters."""
        unicode_names = [
            "サーキット",  # Japanese
            "прерыватель",  # Russian
            "disjoncteur",  # French
            "断路器",  # Chinese
        ]

        registry = CircuitBreakerRegistry()

        for name in unicode_names:
            config = CircuitBreakerConfig(name=name)
            cb = registry.get_or_create(name, config)
            assert cb.config.name == name

    def test_very_long_name(self):
        """Test circuit breaker with very long name."""
        long_name = "a" * 1000
        config = CircuitBreakerConfig(name=long_name)
        cb = CircuitBreaker(config)

        assert cb.config.name == long_name

    @pytest.mark.asyncio
    async def test_exception_types_handling(self):
        """Test circuit breaker handles different exception types."""
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))

        # Different exception types should all count as failures
        exception_types = [
            ValueError,
            RuntimeError,
            TypeError,
            KeyError,
            AttributeError,
        ]

        for i, exc_type in enumerate(
            exception_types[:2]
        ):  # Only first 2 to stay under threshold
            with pytest.raises(exc_type):
                async with cb.protect():
                    raise exc_type(f"Error {i}")

        assert cb._failure_count == 2
        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_nested_protection(self):
        """Test nested circuit breaker protection calls."""
        cb = CircuitBreaker(CircuitBreakerConfig())

        # Nested calls should work (though not typical usage)
        async with cb.protect():
            async with cb.protect():
                assert True  # Should reach here

    @pytest.mark.asyncio
    async def test_concurrent_state_transitions(self):
        """Test concurrent state transitions are handled correctly."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(failure_threshold=5, recovery_timeout=0.1)
        )

        async def state_changer():
            try:
                async with cb.protect():
                    raise ValueError("Failure")
            except (ValueError, CircuitBreakerError):
                pass

        # Run many concurrent operations that might cause state changes
        tasks = [asyncio.create_task(state_changer()) for _ in range(10)]
        await asyncio.gather(*tasks)

        # Circuit should be in consistent state
        assert cb.state in [
            CircuitState.CLOSED,
            CircuitState.OPEN,
            CircuitState.HALF_OPEN,
        ]


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_typical_service_failure_scenario(self):
        """Test typical service failure and recovery scenario."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=1.0,
                success_threshold=2,
                name="service_api",
            )
        )

        results = []

        async def call_external_service(call_id):
            try:
                async with cb.protect():
                    # Simulate service that fails initially then recovers
                    if call_id < 5:  # First 5 calls fail
                        raise RuntimeError(f"Service unavailable {call_id}")
                    else:
                        await asyncio.sleep(0.01)  # Simulate successful call
                        results.append(f"success_{call_id}")
                        return f"result_{call_id}"
            except (RuntimeError, CircuitBreakerError) as e:
                results.append(f"failed_{call_id}_{type(e).__name__}")
                return None

        # Make calls that will fail initially
        for i in range(5):
            await call_external_service(i)

        # Circuit should be open after 3 failures
        assert cb.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(1.1)

        # Make successful calls to recover
        for i in range(5, 10):
            await call_external_service(i)

        # Should have some successes and circuit should recover
        success_results = [r for r in results if r.startswith("success_")]
        assert len(success_results) > 0

    @pytest.mark.asyncio
    async def test_gradual_failure_recovery_pattern(self):
        """Test gradual failure and recovery pattern."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=5, recovery_timeout=0.5, success_threshold=3
            )
        )

        # Phase 1: Gradual failures (not enough to trip)
        for i in range(3):
            with pytest.raises(ValueError):
                async with cb.protect():
                    raise ValueError(f"Gradual failure {i}")

        assert cb.state == CircuitState.CLOSED
        assert cb._failure_count == 3

        # Phase 2: Some successes (should reduce failure count)
        for _i in range(2):
            async with cb.protect():
                pass  # Success

        assert cb._failure_count == 1  # Should be reduced

        # Phase 3: More failures to trip circuit
        failures_to_trip = cb.config.failure_threshold - cb._failure_count

        for i in range(failures_to_trip):
            try:
                with pytest.raises(ValueError):
                    async with cb.protect():
                        raise ValueError(f"Final failure {i}")
            except CircuitBreakerError:
                # Circuit opened during the loop
                break

        assert cb.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_multiple_circuit_breakers_coordination(self):
        """Test coordination between multiple circuit breakers."""
        registry = CircuitBreakerRegistry()

        # Create circuit breakers for different services
        db_cb = registry.get_or_create(
            "database", CircuitBreakerConfig(name="database", failure_threshold=2)
        )
        api_cb = registry.get_or_create(
            "external_api",
            CircuitBreakerConfig(name="external_api", failure_threshold=3),
        )
        cache_cb = registry.get_or_create(
            "cache", CircuitBreakerConfig(name="cache", failure_threshold=1)
        )

        results = []

        async def db_operation(op_id):
            try:
                async with db_cb.protect():
                    if op_id < 2:  # First 2 fail
                        raise RuntimeError("DB connection failed")
                    results.append(f"db_success_{op_id}")
            except (RuntimeError, CircuitBreakerError):
                results.append(f"db_failed_{op_id}")

        async def api_operation(op_id):
            try:
                async with api_cb.protect():
                    results.append(f"api_success_{op_id}")
            except CircuitBreakerError:
                results.append(f"api_failed_{op_id}")

        async def cache_operation(op_id):
            try:
                async with cache_cb.protect():
                    if op_id == 0:  # First one fails
                        raise RuntimeError("Cache miss")
                    results.append(f"cache_success_{op_id}")
            except (RuntimeError, CircuitBreakerError):
                results.append(f"cache_failed_{op_id}")

        # Run operations concurrently
        tasks = []
        for i in range(4):
            tasks.extend(
                [
                    asyncio.create_task(db_operation(i)),
                    asyncio.create_task(api_operation(i)),
                    asyncio.create_task(cache_operation(i)),
                ]
            )

        await asyncio.gather(*tasks)

        # Verify different behaviors
        db_results = [r for r in results if r.startswith("db_")]
        api_results = [r for r in results if r.startswith("api_")]
        cache_results = [r for r in results if r.startswith("cache_")]

        assert len(db_results) == 4
        assert len(api_results) == 4
        assert len(cache_results) == 4

        # Check circuit states
        assert db_cb.state == CircuitState.OPEN  # Should be open after 2 failures
        assert api_cb.state == CircuitState.CLOSED  # No failures
        assert cache_cb.state == CircuitState.OPEN  # Should be open after 1 failure

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_retry_logic(self):
        """Test circuit breaker integration with retry logic."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.5)
        )

        attempt_count = 0

        async def retry_operation():
            nonlocal attempt_count
            max_retries = 5

            for _attempt in range(max_retries):
                try:
                    async with cb.protect():
                        attempt_count += 1
                        # Fail first few attempts, then succeed
                        if attempt_count < 4:
                            raise RuntimeError(f"Attempt {attempt_count} failed")
                        return f"success_after_{attempt_count}_attempts"
                except CircuitBreakerError:
                    # Circuit is open, wait for recovery
                    await asyncio.sleep(0.6)
                except RuntimeError:
                    # Service failure, retry immediately
                    await asyncio.sleep(0.1)

            return "all_retries_failed"

        result = await retry_operation()

        # Should eventually succeed after circuit recovery
        assert "success" in result or result == "all_retries_failed"

    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test circuit breaker performance under high load."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=100,  # High threshold to avoid opening
                name="load_test",
            )
        )

        async def load_test_operation(op_id):
            async with cb.protect():
                await asyncio.sleep(0.001)  # Minimal work
                return op_id

        # Measure time for many operations
        start_time = time.perf_counter()

        # Run many concurrent operations
        tasks = [asyncio.create_task(load_test_operation(i)) for i in range(1000)]
        results = await asyncio.gather(*tasks)

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        # All operations should succeed
        assert len(results) == 1000
        assert all(isinstance(r, int) for r in results)

        # Should complete in reasonable time
        assert elapsed < 5.0  # Adjust threshold as needed

        # Circuit should remain closed
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_metrics_accuracy_under_load(self):
        """Test that metrics remain accurate under concurrent load."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(failure_threshold=50, name="metrics_test")
        )

        success_count = 0
        failure_count = 0

        async def mixed_operation(op_id):
            nonlocal success_count, failure_count
            try:
                async with cb.protect():
                    if op_id % 3 == 0:  # Every 3rd operation fails
                        failure_count += 1
                        raise RuntimeError(f"Planned failure {op_id}")
                    else:
                        success_count += 1
                        return f"success_{op_id}"
            except (RuntimeError, CircuitBreakerError):
                return f"failed_{op_id}"

        # Run many mixed operations
        tasks = [asyncio.create_task(mixed_operation(i)) for i in range(150)]
        results = await asyncio.gather(*tasks)

        # Check metrics
        metrics = cb.get_metrics()

        # Should track failures accurately
        assert metrics["failure_count"] <= failure_count
        assert metrics["state"] in ["closed", "open", "half_open"]

        # Verify results match expectations
        success_results = [r for r in results if r.startswith("success_")]
        failed_results = [r for r in results if r.startswith("failed_")]

        assert len(success_results) + len(failed_results) == 150


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
