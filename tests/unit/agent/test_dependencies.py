"""Tests for agent dependencies module."""

from unittest.mock import Mock

from puffinflow.core.agent.dependencies import (
    DependencyConfig,
    DependencyLifecycle,
    DependencyType,
)


class TestDependencyType:
    """Test DependencyType enum."""

    def test_dependency_type_values(self):
        """Test that all dependency types have correct values."""
        assert DependencyType.REQUIRED.value == "required"
        assert DependencyType.OPTIONAL.value == "optional"
        assert DependencyType.PARALLEL.value == "parallel"
        assert DependencyType.SEQUENTIAL.value == "sequential"
        assert DependencyType.CONDITIONAL.value == "conditional"
        assert DependencyType.TIMEOUT.value == "timeout"
        assert DependencyType.XOR.value == "xor"
        assert DependencyType.AND.value == "and"
        assert DependencyType.OR.value == "or"

    def test_dependency_type_membership(self):
        """Test dependency type membership."""
        assert DependencyType.REQUIRED in DependencyType
        assert DependencyType.OPTIONAL in DependencyType
        assert DependencyType.PARALLEL in DependencyType
        assert DependencyType.SEQUENTIAL in DependencyType
        assert DependencyType.CONDITIONAL in DependencyType
        assert DependencyType.TIMEOUT in DependencyType
        assert DependencyType.XOR in DependencyType
        assert DependencyType.AND in DependencyType
        assert DependencyType.OR in DependencyType

    def test_dependency_type_count(self):
        """Test that we have the expected number of dependency types."""
        assert len(DependencyType) == 9

    def test_dependency_type_string_representation(self):
        """Test string representation of dependency types."""
        assert str(DependencyType.REQUIRED) == "DependencyType.REQUIRED"
        assert repr(DependencyType.REQUIRED) == "DependencyType.REQUIRED"


class TestDependencyLifecycle:
    """Test DependencyLifecycle enum."""

    def test_dependency_lifecycle_values(self):
        """Test that all lifecycle types have correct values."""
        assert DependencyLifecycle.ONCE.value == "once"
        assert DependencyLifecycle.ALWAYS.value == "always"
        assert DependencyLifecycle.SESSION.value == "session"
        assert DependencyLifecycle.TEMPORARY.value == "temporary"
        assert DependencyLifecycle.PERIODIC.value == "periodic"

    def test_dependency_lifecycle_membership(self):
        """Test dependency lifecycle membership."""
        assert DependencyLifecycle.ONCE in DependencyLifecycle
        assert DependencyLifecycle.ALWAYS in DependencyLifecycle
        assert DependencyLifecycle.SESSION in DependencyLifecycle
        assert DependencyLifecycle.TEMPORARY in DependencyLifecycle
        assert DependencyLifecycle.PERIODIC in DependencyLifecycle

    def test_dependency_lifecycle_count(self):
        """Test that we have the expected number of lifecycle types."""
        assert len(DependencyLifecycle) == 5

    def test_dependency_lifecycle_string_representation(self):
        """Test string representation of lifecycle types."""
        assert str(DependencyLifecycle.ONCE) == "DependencyLifecycle.ONCE"
        assert repr(DependencyLifecycle.ONCE) == "DependencyLifecycle.ONCE"


class TestDependencyConfig:
    """Test DependencyConfig dataclass."""

    def test_dependency_config_creation(self):
        """Test basic dependency config creation."""
        config = DependencyConfig(type=DependencyType.REQUIRED)

        assert config.type == DependencyType.REQUIRED
        assert config.lifecycle == DependencyLifecycle.ALWAYS
        assert config.condition is None
        assert config.expiry is None
        assert config.interval is None
        assert config.timeout is None
        assert config.retry_policy is None

    def test_dependency_config_with_all_fields(self):
        """Test dependency config with all fields specified."""
        condition_func = Mock()
        retry_policy = {"max_retries": 3, "delay": 1.0}

        config = DependencyConfig(
            type=DependencyType.CONDITIONAL,
            lifecycle=DependencyLifecycle.TEMPORARY,
            condition=condition_func,
            expiry=300.0,
            interval=60.0,
            timeout=30.0,
            retry_policy=retry_policy,
        )

        assert config.type == DependencyType.CONDITIONAL
        assert config.lifecycle == DependencyLifecycle.TEMPORARY
        assert config.condition is condition_func
        assert config.expiry == 300.0
        assert config.interval == 60.0
        assert config.timeout == 30.0
        assert config.retry_policy == retry_policy

    def test_dependency_config_defaults(self):
        """Test that dependency config has correct defaults."""
        config = DependencyConfig(type=DependencyType.OPTIONAL)

        # Only type is required, all others should have defaults
        assert config.type == DependencyType.OPTIONAL
        assert config.lifecycle == DependencyLifecycle.ALWAYS
        assert config.condition is None
        assert config.expiry is None
        assert config.interval is None
        assert config.timeout is None
        assert config.retry_policy is None

    def test_dependency_config_immutable_after_creation(self):
        """Test that dependency config fields can be modified after creation."""
        config = DependencyConfig(type=DependencyType.REQUIRED)

        # Dataclass fields should be mutable by default
        config.timeout = 60.0
        assert config.timeout == 60.0

        config.expiry = 300.0
        assert config.expiry == 300.0

    def test_dependency_config_with_condition_function(self):
        """Test dependency config with condition function."""

        def test_condition(agent):
            return agent.status == "ready"

        config = DependencyConfig(
            type=DependencyType.CONDITIONAL, condition=test_condition
        )

        assert config.condition is test_condition
        assert callable(config.condition)

    def test_dependency_config_with_async_condition(self):
        """Test dependency config with async condition function."""

        async def async_condition(agent):
            return await agent.check_status()

        config = DependencyConfig(
            type=DependencyType.CONDITIONAL, condition=async_condition
        )

        assert config.condition is async_condition
        assert callable(config.condition)

    def test_dependency_config_equality(self):
        """Test dependency config equality comparison."""
        config1 = DependencyConfig(
            type=DependencyType.REQUIRED,
            lifecycle=DependencyLifecycle.ONCE,
            timeout=30.0,
        )

        config2 = DependencyConfig(
            type=DependencyType.REQUIRED,
            lifecycle=DependencyLifecycle.ONCE,
            timeout=30.0,
        )

        config3 = DependencyConfig(
            type=DependencyType.OPTIONAL,
            lifecycle=DependencyLifecycle.ONCE,
            timeout=30.0,
        )

        assert config1 == config2
        assert config1 != config3

    def test_dependency_config_repr(self):
        """Test dependency config string representation."""
        config = DependencyConfig(
            type=DependencyType.REQUIRED, lifecycle=DependencyLifecycle.ONCE
        )

        repr_str = repr(config)
        assert "DependencyConfig" in repr_str
        assert "REQUIRED" in repr_str
        assert "ONCE" in repr_str


class TestDependencyConfigValidation:
    """Test dependency config validation scenarios."""

    def test_required_dependency_config(self):
        """Test required dependency configuration."""
        config = DependencyConfig(type=DependencyType.REQUIRED)

        assert config.type == DependencyType.REQUIRED
        # Required dependencies typically don't need conditions
        assert config.condition is None

    def test_conditional_dependency_config(self):
        """Test conditional dependency configuration."""

        def condition(agent):
            return True

        config = DependencyConfig(type=DependencyType.CONDITIONAL, condition=condition)

        assert config.type == DependencyType.CONDITIONAL
        assert config.condition is condition

    def test_timeout_dependency_config(self):
        """Test timeout dependency configuration."""
        config = DependencyConfig(type=DependencyType.TIMEOUT, timeout=30.0)

        assert config.type == DependencyType.TIMEOUT
        assert config.timeout == 30.0

    def test_periodic_dependency_config(self):
        """Test periodic dependency configuration."""
        config = DependencyConfig(
            type=DependencyType.REQUIRED,
            lifecycle=DependencyLifecycle.PERIODIC,
            interval=60.0,
        )

        assert config.lifecycle == DependencyLifecycle.PERIODIC
        assert config.interval == 60.0

    def test_temporary_dependency_config(self):
        """Test temporary dependency configuration."""
        config = DependencyConfig(
            type=DependencyType.REQUIRED,
            lifecycle=DependencyLifecycle.TEMPORARY,
            expiry=300.0,
        )

        assert config.lifecycle == DependencyLifecycle.TEMPORARY
        assert config.expiry == 300.0


class TestDependencyConfigUseCases:
    """Test dependency config for common use cases."""

    def test_database_connection_dependency(self):
        """Test configuration for database connection dependency."""
        config = DependencyConfig(
            type=DependencyType.REQUIRED,
            lifecycle=DependencyLifecycle.SESSION,
            timeout=10.0,
            retry_policy={"max_retries": 3, "delay": 1.0},
        )

        assert config.type == DependencyType.REQUIRED
        assert config.lifecycle == DependencyLifecycle.SESSION
        assert config.timeout == 10.0
        assert config.retry_policy is not None

    def test_cache_warming_dependency(self):
        """Test configuration for cache warming dependency."""
        config = DependencyConfig(
            type=DependencyType.OPTIONAL,
            lifecycle=DependencyLifecycle.ONCE,
            timeout=5.0,
        )

        assert config.type == DependencyType.OPTIONAL
        assert config.lifecycle == DependencyLifecycle.ONCE
        assert config.timeout == 5.0

    def test_health_check_dependency(self):
        """Test configuration for health check dependency."""

        def health_check(agent):
            return agent.is_healthy()

        config = DependencyConfig(
            type=DependencyType.CONDITIONAL,
            lifecycle=DependencyLifecycle.PERIODIC,
            condition=health_check,
            interval=30.0,
        )

        assert config.type == DependencyType.CONDITIONAL
        assert config.lifecycle == DependencyLifecycle.PERIODIC
        assert config.condition is health_check
        assert config.interval == 30.0

    def test_resource_availability_dependency(self):
        """Test configuration for resource availability dependency."""

        def resource_available(agent):
            return agent.resources.available()

        config = DependencyConfig(
            type=DependencyType.CONDITIONAL,
            lifecycle=DependencyLifecycle.ALWAYS,
            condition=resource_available,
            timeout=60.0,
        )

        assert config.type == DependencyType.CONDITIONAL
        assert config.lifecycle == DependencyLifecycle.ALWAYS
        assert config.condition is resource_available
        assert config.timeout == 60.0


class TestDependencyTypes:
    """Test different dependency type behaviors."""

    def test_xor_dependency_logic(self):
        """Test XOR dependency type logic."""
        config = DependencyConfig(type=DependencyType.XOR)

        # XOR means only one dependency needs to be satisfied
        assert config.type == DependencyType.XOR

    def test_and_dependency_logic(self):
        """Test AND dependency type logic."""
        config = DependencyConfig(type=DependencyType.AND)

        # AND means all dependencies must be satisfied
        assert config.type == DependencyType.AND

    def test_or_dependency_logic(self):
        """Test OR dependency type logic."""
        config = DependencyConfig(type=DependencyType.OR)

        # OR means at least one dependency must be satisfied
        assert config.type == DependencyType.OR

    def test_parallel_dependency_logic(self):
        """Test parallel dependency type logic."""
        config = DependencyConfig(type=DependencyType.PARALLEL)

        # Parallel means can run alongside dependency
        assert config.type == DependencyType.PARALLEL

    def test_sequential_dependency_logic(self):
        """Test sequential dependency type logic."""
        config = DependencyConfig(type=DependencyType.SEQUENTIAL)

        # Sequential means must run after dependency completes
        assert config.type == DependencyType.SEQUENTIAL
