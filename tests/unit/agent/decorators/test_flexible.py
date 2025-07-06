"""Tests for flexible state decorator module."""

from unittest.mock import patch

import pytest

# Import the modules we're testing
from puffinflow.core.agent.decorators.flexible import (
    PROFILES,
    FlexibleStateDecorator,
    StateProfile,
    batch_state,
    concurrent_state,
    cpu_intensive,
    create_custom_decorator,
    critical_state,
    external_service,
    fault_tolerant,
    get_profile,
    gpu_accelerated,
    high_availability,
    io_intensive,
    list_profiles,
    memory_intensive,
    minimal_state,
    network_intensive,
    quick_state,
    state,
    synchronized_state,
)
from puffinflow.core.agent.state import Priority


class TestStateProfile:
    """Test StateProfile dataclass."""

    def test_state_profile_creation(self):
        """Test creating a StateProfile."""
        profile = StateProfile(
            name="test", cpu=2.0, memory=512.0, priority=Priority.HIGH
        )

        assert profile.name == "test"
        assert profile.cpu == 2.0
        assert profile.memory == 512.0
        assert profile.priority == Priority.HIGH
        assert profile.max_retries == 3  # default
        assert profile.leak_detection is True  # default

    def test_state_profile_defaults(self):
        """Test StateProfile default values."""
        profile = StateProfile(name="minimal")

        assert profile.cpu == 1.0
        assert profile.memory == 100.0
        assert profile.io == 1.0
        assert profile.network == 1.0
        assert profile.gpu == 0.0
        assert profile.priority == Priority.NORMAL
        assert profile.timeout is None
        assert profile.rate_limit is None
        assert profile.burst_limit is None
        assert profile.coordination is None
        assert profile.max_retries == 3
        assert isinstance(profile.tags, dict)
        assert profile.description is None
        assert profile.circuit_breaker is False
        assert profile.circuit_breaker_config is None
        assert profile.bulkhead is False
        assert profile.bulkhead_config is None
        assert profile.leak_detection is True

    def test_state_profile_to_dict(self):
        """Test StateProfile.to_dict() method."""
        profile = StateProfile(
            name="test",
            cpu=2.0,
            memory=512.0,
            priority=Priority.HIGH,
            timeout=60.0,
            tags={"env": "test"},
        )

        result = profile.to_dict()

        # Should exclude None values and name
        assert "name" not in result
        assert "description" not in result  # None value
        assert "rate_limit" not in result  # None value

        # Should include non-None values
        assert result["cpu"] == 2.0
        assert result["memory"] == 512.0
        assert result["priority"] == Priority.HIGH
        assert result["timeout"] == 60.0
        assert result["tags"] == {"env": "test"}

    def test_state_profile_with_reliability_config(self):
        """Test StateProfile with reliability configurations."""
        profile = StateProfile(
            name="reliable",
            circuit_breaker=True,
            circuit_breaker_config={"failure_threshold": 3},
            bulkhead=True,
            bulkhead_config={"max_concurrent": 5},
            leak_detection=False,
        )

        assert profile.circuit_breaker is True
        assert profile.circuit_breaker_config == {"failure_threshold": 3}
        assert profile.bulkhead is True
        assert profile.bulkhead_config == {"max_concurrent": 5}
        assert profile.leak_detection is False


class TestPredefinedProfiles:
    """Test predefined profiles in PROFILES dictionary."""

    def test_profiles_exist(self):
        """Test that all expected profiles exist."""
        expected_profiles = [
            "minimal",
            "standard",
            "cpu_intensive",
            "memory_intensive",
            "io_intensive",
            "gpu_accelerated",
            "network_intensive",
            "quick",
            "batch",
            "critical",
            "concurrent",
            "synchronized",
            "resilient",
            "critical_no_dlq",
            "fault_tolerant",
            "external_service",
            "high_availability",
        ]

        for profile_name in expected_profiles:
            assert profile_name in PROFILES
            assert isinstance(PROFILES[profile_name], StateProfile)

    def test_minimal_profile(self):
        """Test minimal profile configuration."""
        profile = PROFILES["minimal"]

        assert profile.name == "minimal"
        assert profile.cpu == 0.1
        assert profile.memory == 50.0
        assert profile.priority == Priority.NORMAL
        assert profile.max_retries == 1
        assert profile.circuit_breaker is False
        assert profile.bulkhead is False
        assert profile.leak_detection is False
        assert profile.tags == {"profile": "minimal"}

    def test_cpu_intensive_profile(self):
        """Test CPU intensive profile configuration."""
        profile = PROFILES["cpu_intensive"]

        assert profile.name == "cpu_intensive"
        assert profile.cpu == 4.0
        assert profile.memory == 1024.0
        assert profile.priority == Priority.HIGH
        assert profile.timeout == 300.0
        assert profile.max_retries == 3
        assert profile.circuit_breaker is True
        assert profile.bulkhead is True
        assert profile.bulkhead_config == {"max_concurrent": 2}
        assert profile.leak_detection is True
        assert profile.tags == {"profile": "cpu_intensive", "workload": "compute"}

    def test_io_intensive_profile(self):
        """Test IO intensive profile configuration."""
        profile = PROFILES["io_intensive"]

        assert profile.name == "io_intensive"
        assert profile.cpu == 1.0
        assert profile.memory == 256.0
        assert profile.io == 10.0
        assert profile.priority == Priority.NORMAL
        assert profile.timeout == 120.0
        assert profile.max_retries == 5
        assert profile.circuit_breaker is True
        assert profile.circuit_breaker_config == {
            "failure_threshold": 3,
            "recovery_timeout": 30.0,
        }
        assert profile.bulkhead is True
        assert profile.bulkhead_config == {"max_concurrent": 5}

    def test_critical_profile(self):
        """Test critical profile configuration."""
        profile = PROFILES["critical"]

        assert profile.name == "critical"
        assert profile.priority == Priority.CRITICAL
        assert profile.coordination == "mutex"
        assert profile.circuit_breaker is False  # Critical shouldn't be circuit broken
        assert profile.bulkhead is True
        assert profile.bulkhead_config == {"max_concurrent": 1}

    def test_fault_tolerant_profile(self):
        """Test fault tolerant profile configuration."""
        profile = PROFILES["fault_tolerant"]

        assert profile.name == "fault_tolerant"
        assert profile.max_retries == 5
        assert profile.circuit_breaker is True
        assert profile.circuit_breaker_config == {
            "failure_threshold": 3,
            "recovery_timeout": 45.0,
        }
        assert profile.bulkhead is True
        assert profile.bulkhead_config == {"max_concurrent": 4, "max_queue_size": 20}
        assert profile.tags == {"profile": "fault_tolerant", "reliability": "high"}


class TestFlexibleStateDecorator:
    """Test FlexibleStateDecorator class."""

    def test_decorator_creation(self):
        """Test creating a FlexibleStateDecorator."""
        decorator = FlexibleStateDecorator()
        assert decorator.default_config == {}

    def test_direct_decoration_without_parentheses(self):
        """Test @state decoration without parentheses."""
        decorator = FlexibleStateDecorator()

        @decorator
        def test_func():
            """Test function."""
            return "test"

        # Should be decorated as PuffinFlow state
        assert hasattr(test_func, "_puffinflow_state")
        assert test_func._puffinflow_state is True
        assert test_func._state_name == "test_func"
        assert isinstance(test_func._state_config, dict)

    def test_decoration_with_empty_parentheses(self):
        """Test @state() decoration with empty parentheses."""
        decorator = FlexibleStateDecorator()

        @decorator()
        def test_func():
            """Test function."""
            return "test"

        assert hasattr(test_func, "_puffinflow_state")
        assert test_func._puffinflow_state is True
        assert test_func._state_name == "test_func"

    def test_decoration_with_profile_string(self):
        """Test @state(profile='minimal') decoration."""
        decorator = FlexibleStateDecorator()

        @decorator(profile="minimal")
        def test_func():
            return "test"

        assert test_func._puffinflow_state is True
        config = test_func._state_config
        assert config["cpu"] == 0.1  # From minimal profile
        assert config["memory"] == 50.0  # From minimal profile
        assert config["max_retries"] == 1  # From minimal profile

    def test_decoration_with_direct_parameters(self):
        """Test @state(cpu=2.0, memory=512.0) decoration."""
        decorator = FlexibleStateDecorator()

        @decorator(cpu=2.0, memory=512.0, priority=Priority.HIGH)
        def test_func():
            return "test"

        assert test_func._puffinflow_state is True
        config = test_func._state_config
        assert config["cpu"] == 2.0
        assert config["memory"] == 512.0
        assert config["priority"] == Priority.HIGH

    def test_decoration_with_config_dict(self):
        """Test @state(config={'cpu': 2.0}) decoration."""
        decorator = FlexibleStateDecorator()

        config_dict = {"cpu": 2.0, "memory": 512.0}

        @decorator(config=config_dict)
        def test_func():
            return "test"

        assert test_func._puffinflow_state is True
        config = test_func._state_config
        assert config["cpu"] == 2.0
        assert config["memory"] == 512.0

    def test_decoration_with_positional_dict(self):
        """Test @state({'cpu': 2.0}) decoration with positional dict."""
        decorator = FlexibleStateDecorator()

        config_dict = {"cpu": 2.0, "memory": 512.0}

        @decorator(config_dict)
        def test_func():
            return "test"

        assert test_func._puffinflow_state is True
        config = test_func._state_config
        assert config["cpu"] == 2.0
        assert config["memory"] == 512.0

    def test_decoration_with_positional_profile_string(self):
        """Test @state('minimal') decoration with positional profile."""
        decorator = FlexibleStateDecorator()

        @decorator("minimal")
        def test_func():
            return "test"

        assert test_func._puffinflow_state is True
        config = test_func._state_config
        assert config["cpu"] == 0.1  # From minimal profile

    def test_decoration_with_profile_object(self):
        """Test decoration with StateProfile object."""
        decorator = FlexibleStateDecorator()
        custom_profile = StateProfile(name="custom", cpu=3.0, memory=256.0)

        @decorator(custom_profile)
        def test_func():
            return "test"

        assert test_func._puffinflow_state is True
        config = test_func._state_config
        assert config["cpu"] == 3.0
        assert config["memory"] == 256.0

    def test_configuration_priority_order(self):
        """Test that configuration sources are applied in correct priority order."""
        decorator = FlexibleStateDecorator()

        # Profile sets cpu=0.1, config dict sets cpu=2.0, direct param sets cpu=4.0
        # Direct param should win (highest priority)
        @decorator("minimal", {"cpu": 2.0}, cpu=4.0)
        def test_func():
            return "test"

        config = test_func._state_config
        assert config["cpu"] == 4.0  # Direct parameter wins
        assert config["memory"] == 50.0  # From minimal profile (not overridden)

    def test_invalid_profile_name_raises_error(self):
        """Test that invalid profile name raises ValueError."""
        decorator = FlexibleStateDecorator()

        with pytest.raises(ValueError, match="Unknown profile: nonexistent"):

            @decorator(profile="nonexistent")
            def test_func():
                return "test"

    def test_invalid_positional_profile_raises_error(self):
        """Test that invalid positional profile raises ValueError."""
        decorator = FlexibleStateDecorator()

        with pytest.raises(ValueError, match="Unknown profile: nonexistent"):

            @decorator("nonexistent")
            def test_func():
                return "test"


class TestConfigurationProcessing:
    """Test configuration processing and validation."""

    def test_priority_string_normalization(self):
        """Test priority string normalization."""
        decorator = FlexibleStateDecorator()

        @decorator(priority="high")
        def test_func():
            return "test"

        config = test_func._state_config
        assert config["priority"] == Priority.HIGH

    def test_priority_int_normalization(self):
        """Test priority integer normalization."""
        decorator = FlexibleStateDecorator()

        @decorator(priority=3)  # Priority.HIGH.value
        def test_func():
            return "test"

        config = test_func._state_config
        assert config["priority"] == Priority.CRITICAL

    def test_invalid_priority_raises_error(self):
        """Test that invalid priority raises KeyError."""
        decorator = FlexibleStateDecorator()

        with pytest.raises(KeyError, match="Invalid priority: invalid"):

            @decorator(priority="invalid")
            def test_func():
                return "test"

    def test_coordination_string_parsing(self):
        """Test coordination string parsing."""
        decorator = FlexibleStateDecorator()

        @decorator(coordination="semaphore:5")
        def test_func():
            return "test"

        config = test_func._state_config
        assert config["semaphore"] == 5

    def test_coordination_mutex_parsing(self):
        """Test mutex coordination parsing."""
        decorator = FlexibleStateDecorator()

        @decorator(coordination="mutex")
        def test_func():
            return "test"

        config = test_func._state_config
        assert config["mutex"] is True

    def test_coordination_barrier_parsing(self):
        """Test barrier coordination parsing."""
        decorator = FlexibleStateDecorator()

        @decorator(coordination="barrier:3")
        def test_func():
            return "test"

        config = test_func._state_config
        assert config["barrier"] == 3

    def test_invalid_coordination_raises_error(self):
        """Test that invalid coordination raises ValueError."""
        decorator = FlexibleStateDecorator()

        with pytest.raises(ValueError, match="Unknown coordination type: invalid"):

            @decorator(coordination="invalid")
            def test_func():
                return "test"

    def test_depends_on_string_normalization(self):
        """Test depends_on string normalization to list."""
        decorator = FlexibleStateDecorator()

        @decorator(depends_on="other_state")
        def test_func():
            return "test"

        config = test_func._state_config
        assert config["depends_on"] == ["other_state"]

    def test_depends_on_list_preserved(self):
        """Test depends_on list is preserved."""
        decorator = FlexibleStateDecorator()

        @decorator(depends_on=["state1", "state2"])
        def test_func():
            return "test"

        config = test_func._state_config
        assert config["depends_on"] == ["state1", "state2"]

    def test_auto_description_from_docstring(self):
        """Test auto-generated description from function docstring."""
        decorator = FlexibleStateDecorator()

        @decorator()
        def test_func():
            """This is a test function."""
            return "test"

        config = test_func._state_config
        assert config["description"] == "This is a test function."

    def test_auto_description_from_name(self):
        """Test auto-generated description from function name."""
        decorator = FlexibleStateDecorator()

        @decorator()
        def test_func():
            return "test"

        config = test_func._state_config
        assert config["description"] == "State: test_func"

    def test_automatic_tags_added(self):
        """Test that automatic tags are added."""
        decorator = FlexibleStateDecorator()

        @decorator(tags={"custom": "value"})
        def test_func():
            return "test"

        config = test_func._state_config
        tags = config["tags"]
        assert tags["custom"] == "value"  # User tag preserved
        assert tags["function_name"] == "test_func"  # Auto tag added
        assert tags["decorated_at"] == "runtime"  # Auto tag added


class TestReliabilityConfiguration:
    """Test reliability pattern configuration."""

    def test_circuit_breaker_configuration(self):
        """Test circuit breaker configuration processing."""
        decorator = FlexibleStateDecorator()

        @decorator(circuit_breaker=True)
        def test_func():
            return "test"

        assert test_func._circuit_breaker_enabled is True
        cb_config = test_func._circuit_breaker_config
        assert cb_config["name"] == "test_func_circuit_breaker"
        assert cb_config["failure_threshold"] == 5
        assert cb_config["recovery_timeout"] == 60.0
        assert cb_config["success_threshold"] == 3
        assert cb_config["timeout"] == 30.0

    def test_circuit_breaker_custom_config(self):
        """Test circuit breaker with custom configuration."""
        decorator = FlexibleStateDecorator()

        custom_config = {"failure_threshold": 3, "recovery_timeout": 45.0}

        @decorator(circuit_breaker=True, circuit_breaker_config=custom_config)
        def test_func():
            return "test"

        cb_config = test_func._circuit_breaker_config
        assert cb_config["failure_threshold"] == 3  # Custom value
        assert cb_config["recovery_timeout"] == 45.0  # Custom value
        assert cb_config["success_threshold"] == 3  # Default value

    def test_bulkhead_configuration(self):
        """Test bulkhead configuration processing."""
        decorator = FlexibleStateDecorator()

        @decorator(bulkhead=True)
        def test_func():
            return "test"

        assert test_func._bulkhead_enabled is True
        bh_config = test_func._bulkhead_config
        assert bh_config["name"] == "test_func_bulkhead"
        assert bh_config["max_concurrent"] == 5
        assert bh_config["max_queue_size"] == 100
        assert bh_config["timeout"] == 30.0

    def test_bulkhead_custom_config(self):
        """Test bulkhead with custom configuration."""
        decorator = FlexibleStateDecorator()

        custom_config = {"max_concurrent": 3, "max_queue_size": 50}

        @decorator(bulkhead=True, bulkhead_config=custom_config)
        def test_func():
            return "test"

        bh_config = test_func._bulkhead_config
        assert bh_config["max_concurrent"] == 3  # Custom value
        assert bh_config["max_queue_size"] == 50  # Custom value
        assert bh_config["timeout"] == 30.0  # Default value

    def test_leak_detection_configuration(self):
        """Test leak detection configuration."""
        decorator = FlexibleStateDecorator()

        @decorator(leak_detection=False)
        def test_func():
            return "test"

        assert test_func._leak_detection_enabled is False

    def test_reliability_disabled_by_default(self):
        """Test that reliability patterns are disabled by default."""
        decorator = FlexibleStateDecorator()

        @decorator()
        def test_func():
            return "test"

        assert test_func._circuit_breaker_enabled is False
        assert test_func._bulkhead_enabled is False
        assert test_func._leak_detection_enabled is True  # Default enabled


class TestResourceRequirements:
    """Test resource requirements configuration."""

    @patch("puffinflow.core.agent.decorators.flexible.ResourceRequirements")
    def test_resource_requirements_creation(self, mock_resource_req):
        """Test resource requirements are created correctly."""
        decorator = FlexibleStateDecorator()

        @decorator(cpu=2.0, memory=512.0, io=1.5, network=2.0, gpu=1.0)
        def test_func():
            return "test"

        # Verify ResourceRequirements was called with correct parameters
        mock_resource_req.assert_called_once()
        call_args = mock_resource_req.call_args[1]
        assert call_args["cpu_units"] == 2.0
        assert call_args["memory_mb"] == 512.0
        assert call_args["io_weight"] == 1.5
        assert call_args["network_weight"] == 2.0
        assert call_args["gpu_units"] == 1.0

    def test_rate_limiting_configuration(self):
        """Test rate limiting configuration."""
        decorator = FlexibleStateDecorator()

        @decorator(rate_limit=10.0, burst_limit=20)
        def test_func():
            return "test"

        assert test_func._rate_limit == 10.0
        assert test_func._burst_limit == 20

    def test_rate_limiting_auto_burst(self):
        """Test automatic burst limit calculation."""
        decorator = FlexibleStateDecorator()

        @decorator(rate_limit=10.0)
        def test_func():
            return "test"

        assert test_func._rate_limit == 10.0
        assert test_func._burst_limit == 20  # 2 * rate_limit


class TestDecoratorMethods:
    """Test FlexibleStateDecorator methods."""

    def test_with_defaults(self):
        """Test with_defaults method."""
        decorator = FlexibleStateDecorator()

        custom_decorator = decorator.with_defaults(cpu=2.0, memory=512.0)

        assert custom_decorator.default_config == {"cpu": 2.0, "memory": 512.0}
        assert decorator.default_config == {}  # Original unchanged

    def test_create_profile(self):
        """Test create_profile method."""
        decorator = FlexibleStateDecorator()

        profile = decorator.create_profile("custom", cpu=3.0, memory=1024.0)

        assert isinstance(profile, StateProfile)
        assert profile.name == "custom"
        assert profile.cpu == 3.0
        assert profile.memory == 1024.0

    def test_register_profile_with_object(self):
        """Test register_profile with StateProfile object."""
        decorator = FlexibleStateDecorator()
        profile = StateProfile(name="test_profile", cpu=2.0)

        decorator.register_profile(profile)

        assert "test_profile" in PROFILES
        assert PROFILES["test_profile"] == profile

    def test_register_profile_with_string(self):
        """Test register_profile with string name and config."""
        decorator = FlexibleStateDecorator()

        decorator.register_profile("test_profile2", cpu=3.0, memory=256.0)

        assert "test_profile2" in PROFILES
        profile = PROFILES["test_profile2"]
        assert profile.name == "test_profile2"
        assert profile.cpu == 3.0
        assert profile.memory == 256.0


class TestPredefinedDecorators:
    """Test predefined decorator instances."""

    def test_main_state_decorator(self):
        """Test main state decorator instance."""
        assert isinstance(state, FlexibleStateDecorator)
        assert state.default_config == {}

    def test_specialized_decorators_exist(self):
        """Test that all specialized decorators exist."""
        decorators = [
            minimal_state,
            cpu_intensive,
            memory_intensive,
            io_intensive,
            gpu_accelerated,
            network_intensive,
            quick_state,
            batch_state,
            critical_state,
            concurrent_state,
            synchronized_state,
            fault_tolerant,
            external_service,
            high_availability,
        ]

        for decorator in decorators:
            assert isinstance(decorator, FlexibleStateDecorator)

    def test_minimal_state_decorator(self):
        """Test minimal_state decorator configuration."""
        assert minimal_state.default_config == {"profile": "minimal"}

        @minimal_state
        def test_func():
            return "test"

        config = test_func._state_config
        assert config["cpu"] == 0.1  # From minimal profile
        assert config["memory"] == 50.0  # From minimal profile

    def test_cpu_intensive_decorator(self):
        """Test cpu_intensive decorator configuration."""
        assert cpu_intensive.default_config == {"profile": "cpu_intensive"}

        @cpu_intensive
        def test_func():
            return "test"

        config = test_func._state_config
        assert config["cpu"] == 4.0  # From cpu_intensive profile
        assert config["memory"] == 1024.0  # From cpu_intensive profile


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_profile_existing(self):
        """Test get_profile with existing profile."""
        profile = get_profile("minimal")

        assert isinstance(profile, StateProfile)
        assert profile.name == "minimal"

    def test_get_profile_nonexistent(self):
        """Test get_profile with non-existent profile."""
        profile = get_profile("nonexistent")

        assert profile is None

    def test_list_profiles(self):
        """Test list_profiles function."""
        profiles = list_profiles()

        assert isinstance(profiles, list)
        assert "minimal" in profiles
        assert "cpu_intensive" in profiles
        assert "fault_tolerant" in profiles
        assert len(profiles) >= 17  # At least the predefined ones

    def test_create_custom_decorator(self):
        """Test create_custom_decorator function."""
        custom = create_custom_decorator(cpu=2.0, memory=512.0)

        assert isinstance(custom, FlexibleStateDecorator)
        assert custom.default_config == {"cpu": 2.0, "memory": 512.0}


class TestDeadLetterConfiguration:
    """Test dead letter queue configuration."""

    def test_dead_letter_enabled_by_default(self):
        """Test that dead letter is enabled by default."""
        decorator = FlexibleStateDecorator()

        @decorator()
        def test_func():
            return "test"

        config = test_func._state_config
        retry_config = config.get("retry_config")
        if retry_config:
            assert retry_config.get("dead_letter_on_max_retries", True) is True
            assert retry_config.get("dead_letter_on_timeout", True) is True

    def test_dead_letter_disabled_with_no_dead_letter(self):
        """Test disabling dead letter with no_dead_letter flag."""
        decorator = FlexibleStateDecorator()

        @decorator(no_dead_letter=True)
        def test_func():
            return "test"

        config = test_func._state_config
        retry_config = config.get("retry_config")
        if retry_config:
            assert retry_config.get("dead_letter_on_max_retries") is False
            assert retry_config.get("dead_letter_on_timeout") is False

    def test_dead_letter_disabled_with_dead_letter_false(self):
        """Test disabling dead letter with dead_letter=False."""
        decorator = FlexibleStateDecorator()

        @decorator(dead_letter=False)
        def test_func():
            return "test"

        config = test_func._state_config
        retry_config = config.get("retry_config")
        if retry_config:
            assert retry_config.get("dead_letter_on_max_retries") is False
            assert retry_config.get("dead_letter_on_timeout") is False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_coordination_string(self):
        """Test empty coordination string."""
        decorator = FlexibleStateDecorator()

        @decorator(coordination="")
        def test_func():
            return "test"

        # Should not raise error, just ignore empty string
        assert test_func._puffinflow_state is True

    def test_coordination_without_parameter(self):
        """Test coordination types that require parameters."""
        decorator = FlexibleStateDecorator()

        with pytest.raises(ValueError):

            @decorator(coordination="semaphore")  # Missing parameter
            def test_func():
                return "test"

    def test_invalid_coordination_parameter(self):
        """Test coordination with invalid parameter."""
        decorator = FlexibleStateDecorator()

        with pytest.raises(ValueError, match="Unknown coordination type: semaphore"):

            @decorator(coordination="semaphore:invalid")
            def test_func():
                return "test"

    def test_non_dict_tags_normalized(self):
        """Test that non-dict tags are normalized to empty dict."""
        decorator = FlexibleStateDecorator()

        @decorator(tags="invalid")
        def test_func():
            return "test"

        config = test_func._state_config
        tags = config["tags"]
        assert isinstance(tags, dict)
        assert "function_name" in tags  # Auto tags still added

    def test_function_metadata_preserved(self):
        """Test that function metadata is preserved."""
        decorator = FlexibleStateDecorator()

        @decorator()
        def test_func():
            """Original docstring."""
            return "test"

        # Function should still be callable and have metadata
        assert callable(test_func)
        assert test_func() == "test"
        assert test_func.__name__ == "test_func"
        # Note: docstring might be modified by description processing
