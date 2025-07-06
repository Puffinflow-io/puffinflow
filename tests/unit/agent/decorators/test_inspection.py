"""Tests for state inspection utilities."""

from unittest.mock import Mock

# Import the modules we're testing
from puffinflow.core.agent.decorators.inspection import (
    compare_states,
    get_state_config,
    get_state_coordination,
    get_state_rate_limit,
    get_state_requirements,
    get_state_summary,
    is_puffinflow_state,
    list_state_metadata,
)
from puffinflow.core.agent.state import Priority
from puffinflow.core.coordination.rate_limiter import RateLimitStrategy


class TestIsPuffinflowState:
    """Test is_puffinflow_state function."""

    def test_is_puffinflow_state_true(self):
        """Test function marked as PuffinFlow state."""

        def test_func():
            return "test"

        # Mark as PuffinFlow state
        test_func._puffinflow_state = True

        assert is_puffinflow_state(test_func) is True

    def test_is_puffinflow_state_false(self):
        """Test function not marked as PuffinFlow state."""

        def test_func():
            return "test"

        # Explicitly set to False
        test_func._puffinflow_state = False

        assert is_puffinflow_state(test_func) is False

    def test_is_puffinflow_state_missing_attribute(self):
        """Test function without _puffinflow_state attribute."""

        def test_func():
            return "test"

        # No _puffinflow_state attribute
        assert is_puffinflow_state(test_func) is False

    def test_is_puffinflow_state_with_callable_object(self):
        """Test with callable object instead of function."""

        class CallableClass:
            def __call__(self):
                return "test"

        obj = CallableClass()
        obj._puffinflow_state = True

        assert is_puffinflow_state(obj) is True


class TestGetStateConfig:
    """Test get_state_config function."""

    def test_get_state_config_exists(self):
        """Test getting state config when it exists."""

        def test_func():
            return "test"

        config = {"cpu": 2.0, "memory": 512.0}
        test_func._state_config = config

        result = get_state_config(test_func)
        assert result == config
        assert result is config  # Should return the same object

    def test_get_state_config_missing(self):
        """Test getting state config when it doesn't exist."""

        def test_func():
            return "test"

        result = get_state_config(test_func)
        assert result is None

    def test_get_state_config_empty_dict(self):
        """Test getting empty state config."""

        def test_func():
            return "test"

        test_func._state_config = {}

        result = get_state_config(test_func)
        assert result == {}


class TestGetStateRequirements:
    """Test get_state_requirements function."""

    def test_get_state_requirements_exists(self):
        """Test getting state requirements when they exist."""

        def test_func():
            return "test"

        # Mock ResourceRequirements object
        mock_requirements = Mock()
        test_func._resource_requirements = mock_requirements

        result = get_state_requirements(test_func)
        assert result is mock_requirements

    def test_get_state_requirements_missing(self):
        """Test getting state requirements when they don't exist."""

        def test_func():
            return "test"

        result = get_state_requirements(test_func)
        assert result is None


class TestGetStateRateLimit:
    """Test get_state_rate_limit function."""

    def test_get_state_rate_limit_exists(self):
        """Test getting rate limit when it exists."""

        def test_func():
            return "test"

        test_func._rate_limit = 10.0
        test_func._burst_limit = 20

        result = get_state_rate_limit(test_func)

        expected = {
            "rate": 10.0,
            "burst": 20,
            "strategy": RateLimitStrategy.TOKEN_BUCKET,  # Default strategy
        }
        assert result["rate"] == expected["rate"]
        assert result["burst"] == expected["burst"]
        assert result["strategy"] == expected["strategy"]

    def test_get_state_rate_limit_with_strategy(self):
        """Test getting rate limit with custom strategy."""

        def test_func():
            return "test"

        test_func._rate_limit = 5.0
        test_func._rate_strategy = RateLimitStrategy.SLIDING_WINDOW

        result = get_state_rate_limit(test_func)

        assert result["rate"] == 5.0
        assert result["burst"] is None  # No burst limit set
        assert result["strategy"] == RateLimitStrategy.SLIDING_WINDOW

    def test_get_state_rate_limit_missing(self):
        """Test getting rate limit when it doesn't exist."""

        def test_func():
            return "test"

        result = get_state_rate_limit(test_func)
        assert result is None

    def test_get_state_rate_limit_only_rate(self):
        """Test getting rate limit with only rate set."""

        def test_func():
            return "test"

        test_func._rate_limit = 15.0
        # No burst limit or strategy

        result = get_state_rate_limit(test_func)

        assert result["rate"] == 15.0
        assert result["burst"] is None
        assert result["strategy"] is RateLimitStrategy.TOKEN_BUCKET  # Default


class TestGetStateCoordination:
    """Test get_state_coordination function."""

    def test_get_state_coordination_exists(self):
        """Test getting coordination when it exists."""

        def test_func():
            return "test"

        mock_primitive = Mock()
        coordination_config = {"ttl": 30.0, "max_count": 5}

        test_func._coordination_primitive = mock_primitive
        test_func._coordination_config = coordination_config

        result = get_state_coordination(test_func)

        expected = {"type": mock_primitive, "config": coordination_config}
        assert result == expected

    def test_get_state_coordination_missing_primitive(self):
        """Test getting coordination when primitive is missing."""

        def test_func():
            return "test"

        # No coordination primitive
        result = get_state_coordination(test_func)
        assert result is None

    def test_get_state_coordination_primitive_none(self):
        """Test getting coordination when primitive is None."""

        def test_func():
            return "test"

        test_func._coordination_primitive = None

        result = get_state_coordination(test_func)
        assert result is None

    def test_get_state_coordination_missing_config(self):
        """Test getting coordination when config is missing."""

        def test_func():
            return "test"

        mock_primitive = Mock()
        test_func._coordination_primitive = mock_primitive
        # No coordination config

        result = get_state_coordination(test_func)

        expected = {
            "type": mock_primitive,
            "config": {},  # Default empty dict
        }
        assert result == expected


class TestListStateMetadata:
    """Test list_state_metadata function."""

    def test_list_state_metadata_not_puffinflow_state(self):
        """Test listing metadata for non-PuffinFlow state."""

        def test_func():
            return "test"

        # Not marked as PuffinFlow state
        result = list_state_metadata(test_func)
        assert result == {}

    def test_list_state_metadata_minimal(self):
        """Test listing metadata with minimal attributes."""

        def test_func():
            """Test function docstring."""
            return "test"

        test_func._puffinflow_state = True

        result = list_state_metadata(test_func)

        # Should have default values
        assert result["name"] == "test_func"
        assert result["description"] == "Test function docstring."
        assert result["tags"] == {}
        assert result["priority"] == Priority.NORMAL
        assert result["requirements"] is None
        assert result["rate_limit"] is None
        assert result["coordination"] is None
        assert result["dependencies"] == {}
        assert result["preemptible"] is False
        assert result["checkpoint_interval"] is None
        assert result["cleanup_on_failure"] is True

    def test_list_state_metadata_full(self):
        """Test listing metadata with all attributes set."""

        def test_func():
            return "test"

        # Set up all attributes
        test_func._puffinflow_state = True
        test_func._state_name = "custom_name"
        test_func._state_description = "Custom description"
        test_func._state_tags = {"env": "test", "version": "1.0"}
        test_func._priority = Priority.HIGH

        mock_requirements = Mock()
        test_func._resource_requirements = mock_requirements

        test_func._rate_limit = 10.0
        test_func._burst_limit = 20

        mock_primitive = Mock()
        test_func._coordination_primitive = mock_primitive
        test_func._coordination_config = {"ttl": 30.0}

        test_func._dependency_configs = {"dep1": Mock(), "dep2": Mock()}
        test_func._preemptible = True
        test_func._checkpoint_interval = 60.0
        test_func._cleanup_on_failure = False

        result = list_state_metadata(test_func)

        assert result["name"] == "custom_name"
        assert result["description"] == "Custom description"
        assert result["tags"] == {"env": "test", "version": "1.0"}
        assert result["priority"] == Priority.HIGH
        assert result["requirements"] is mock_requirements

        rate_limit = result["rate_limit"]
        assert rate_limit["rate"] == 10.0
        assert rate_limit["burst"] == 20

        coordination = result["coordination"]
        assert coordination["type"] is mock_primitive
        assert coordination["config"] == {"ttl": 30.0}

        assert len(result["dependencies"]) == 2
        assert result["preemptible"] is True
        assert result["checkpoint_interval"] == 60.0
        assert result["cleanup_on_failure"] is False

    def test_list_state_metadata_description_fallback(self):
        """Test description fallback logic."""

        def test_func():
            """Function docstring."""
            return "test"

        test_func._puffinflow_state = True
        # No _state_description set

        result = list_state_metadata(test_func)
        assert result["description"] == "Function docstring."

    def test_list_state_metadata_description_fallback_no_docstring(self):
        """Test description fallback when no docstring."""

        def test_func():
            return "test"

        test_func._puffinflow_state = True
        # No docstring, no _state_description

        result = list_state_metadata(test_func)
        assert result["description"] == "State: test_func"

    def test_list_state_metadata_empty_description_fallback(self):
        """Test description fallback when description is empty."""

        def test_func():
            """ """  # Whitespace only docstring
            return "test"

        test_func._puffinflow_state = True
        test_func._state_description = ""  # Empty description

        result = list_state_metadata(test_func)
        assert result["description"] == "State: test_func"


class TestCompareStates:
    """Test compare_states function."""

    def test_compare_states_identical(self):
        """Test comparing identical states."""

        def func1():
            return "test1"

        def func2():
            return "test2"

        config = {"cpu": 2.0, "memory": 512.0, "priority": Priority.HIGH}
        func1._state_config = config.copy()
        func2._state_config = config.copy()

        result = compare_states(func1, func2)
        assert result == {}

    def test_compare_states_different(self):
        """Test comparing different states."""

        def func1():
            return "test1"

        def func2():
            return "test2"

        func1._state_config = {"cpu": 2.0, "memory": 512.0, "priority": Priority.HIGH}
        func2._state_config = {"cpu": 4.0, "memory": 512.0, "priority": Priority.LOW}

        result = compare_states(func1, func2)

        expected = {
            "cpu": {"func1": 2.0, "func2": 4.0},
            "priority": {"func1": Priority.HIGH, "func2": Priority.LOW},
        }
        assert result == expected

    def test_compare_states_missing_keys(self):
        """Test comparing states with missing keys."""

        def func1():
            return "test1"

        def func2():
            return "test2"

        func1._state_config = {"cpu": 2.0, "memory": 512.0}
        func2._state_config = {"cpu": 2.0, "timeout": 60.0}

        result = compare_states(func1, func2)

        expected = {
            "memory": {"func1": 512.0, "func2": None},
            "timeout": {"func1": None, "func2": 60.0},
        }
        assert result == expected

    def test_compare_states_no_config(self):
        """Test comparing states with no config."""

        def func1():
            return "test1"

        def func2():
            return "test2"

        # No _state_config attributes
        result = compare_states(func1, func2)
        assert result == {}

    def test_compare_states_one_missing_config(self):
        """Test comparing states where one has no config."""

        def func1():
            return "test1"

        def func2():
            return "test2"

        func1._state_config = {"cpu": 2.0, "memory": 512.0}
        # func2 has no config

        result = compare_states(func1, func2)

        expected = {
            "cpu": {"func1": 2.0, "func2": None},
            "memory": {"func1": 512.0, "func2": None},
        }
        assert result == expected


class TestGetStateSummary:
    """Test get_state_summary function."""

    def test_get_state_summary_not_puffinflow_state(self):
        """Test summary for non-PuffinFlow state."""

        def test_func():
            return "test"

        result = get_state_summary(test_func)
        assert result == "test_func: Not a PuffinFlow state"

    def test_get_state_summary_no_config(self):
        """Test summary for PuffinFlow state with no config."""

        def test_func():
            return "test"

        test_func._puffinflow_state = True
        # No _state_config

        result = get_state_summary(test_func)
        assert result == "test_func: No configuration found"

    def test_get_state_summary_minimal(self):
        """Test summary with minimal configuration."""

        def test_func():
            return "test"

        test_func._puffinflow_state = True
        test_func._state_config = {}

        result = get_state_summary(test_func)
        assert result == "test_func:"

    def test_get_state_summary_resources_only(self):
        """Test summary with only resource configuration."""

        def test_func():
            return "test"

        test_func._puffinflow_state = True
        test_func._state_config = {"cpu": 2.0, "memory": 512.0, "gpu": 1.0}

        result = get_state_summary(test_func)

        expected = "test_func:\n  Resources: CPU=2.0, Memory=512.0MB, GPU=1.0"
        assert result == expected

    def test_get_state_summary_priority(self):
        """Test summary with priority configuration."""

        def test_func():
            return "test"

        test_func._puffinflow_state = True
        test_func._state_config = {"cpu": 1.0, "priority": Priority.HIGH}

        result = get_state_summary(test_func)

        expected = "test_func:\n  Resources: CPU=1.0\n  Priority: HIGH"
        assert result == expected

    def test_get_state_summary_normal_priority_skipped(self):
        """Test that normal priority is not shown in summary."""

        def test_func():
            return "test"

        test_func._puffinflow_state = True
        test_func._state_config = {"cpu": 1.0, "priority": Priority.NORMAL}

        result = get_state_summary(test_func)

        expected = "test_func:\n  Resources: CPU=1.0"
        assert result == expected

    def test_get_state_summary_coordination(self):
        """Test summary with coordination configuration."""

        def test_func():
            return "test"

        test_func._puffinflow_state = True
        test_func._state_config = {
            "mutex": True,
            "semaphore": 5,
            "barrier": 3,
            "rate_limit": 10.0,
        }

        result = get_state_summary(test_func)

        expected = "test_func:\n  Coordination: Mutex, Semaphore(5), Barrier(3), RateLimit(10.0/s)"
        assert result == expected

    def test_get_state_summary_dependencies(self):
        """Test summary with dependencies."""

        def test_func():
            return "test"

        test_func._puffinflow_state = True
        test_func._state_config = {"depends_on": ["state1", "state2", "state3"]}

        result = get_state_summary(test_func)

        expected = "test_func:\n  Dependencies: state1, state2, state3"
        assert result == expected

    def test_get_state_summary_full(self):
        """Test summary with full configuration."""

        def test_func():
            return "test"

        test_func._puffinflow_state = True
        test_func._state_config = {
            "cpu": 4.0,
            "memory": 1024.0,
            "gpu": 2.0,
            "priority": Priority.CRITICAL,
            "mutex": True,
            "rate_limit": 5.0,
            "depends_on": ["state1", "state2"],
        }

        result = get_state_summary(test_func)

        expected_lines = [
            "test_func:",
            "  Resources: CPU=4.0, Memory=1024.0MB, GPU=2.0",
            "  Priority: CRITICAL",
            "  Coordination: Mutex, RateLimit(5.0/s)",
            "  Dependencies: state1, state2",
        ]
        expected = "\n".join(expected_lines)
        assert result == expected

    def test_get_state_summary_zero_resources_filtered(self):
        """Test that zero resources are filtered out."""

        def test_func():
            return "test"

        test_func._puffinflow_state = True
        test_func._state_config = {
            "cpu": 2.0,
            "memory": 0,  # Should be filtered
            "gpu": 0.0,  # Should be filtered
            "io": 1.5,
            "network": 0,  # Should be filtered
        }

        result = get_state_summary(test_func)

        expected = "test_func:\n  Resources: CPU=2.0"
        assert result == expected


class TestInspectionIntegration:
    """Test inspection utilities working together."""

    def test_full_inspection_workflow(self):
        """Test complete inspection workflow."""

        def test_func():
            """Test function for inspection."""
            return "test"

        # Set up complete state configuration
        test_func._puffinflow_state = True
        test_func._state_name = "test_state"
        test_func._state_config = {
            "cpu": 2.0,
            "memory": 512.0,
            "priority": Priority.HIGH,
            "mutex": True,
            "rate_limit": 10.0,
            "depends_on": ["dep1", "dep2"],
        }
        test_func._state_description = "Test state description"
        test_func._state_tags = {"env": "test"}
        test_func._priority = Priority.HIGH

        mock_requirements = Mock()
        test_func._resource_requirements = mock_requirements
        test_func._rate_limit = 10.0
        test_func._burst_limit = 20

        mock_primitive = Mock()
        test_func._coordination_primitive = mock_primitive
        test_func._coordination_config = {"ttl": 30.0}

        # Test all inspection functions
        assert is_puffinflow_state(test_func) is True

        config = get_state_config(test_func)
        assert config["cpu"] == 2.0
        assert config["priority"] == Priority.HIGH

        requirements = get_state_requirements(test_func)
        assert requirements is mock_requirements

        rate_limit = get_state_rate_limit(test_func)
        assert rate_limit["rate"] == 10.0
        assert rate_limit["burst"] == 20

        coordination = get_state_coordination(test_func)
        assert coordination["type"] is mock_primitive
        assert coordination["config"] == {"ttl": 30.0}

        metadata = list_state_metadata(test_func)
        assert metadata["name"] == "test_state"
        assert metadata["description"] == "Test state description"
        assert metadata["tags"] == {"env": "test"}
        assert metadata["priority"] == Priority.HIGH

        summary = get_state_summary(test_func)
        assert "test_func:" in summary
        assert "CPU=2.0" in summary
        assert "Memory=512.0MB" in summary
        assert "Priority: HIGH" in summary
        assert "Mutex" in summary
        assert "RateLimit(10.0/s)" in summary
        assert "Dependencies: dep1, dep2" in summary

    def test_compare_complex_states(self):
        """Test comparing complex states."""

        def func1():
            return "test1"

        def func2():
            return "test2"

        # Set up similar but different configurations
        func1._state_config = {
            "cpu": 2.0,
            "memory": 512.0,
            "priority": Priority.HIGH,
            "mutex": True,
            "rate_limit": 10.0,
        }

        func2._state_config = {
            "cpu": 4.0,  # Different
            "memory": 512.0,  # Same
            "priority": Priority.HIGH,  # Same
            "semaphore": 5,  # Different coordination
            "rate_limit": 15.0,  # Different rate
        }

        differences = compare_states(func1, func2)

        expected_differences = {
            "cpu": {"func1": 2.0, "func2": 4.0},
            "mutex": {"func1": True, "func2": None},
            "semaphore": {"func1": None, "func2": 5},
            "rate_limit": {"func1": 10.0, "func2": 15.0},
        }

        assert differences == expected_differences


class TestInspectionEdgeCases:
    """Test edge cases in inspection utilities."""

    def test_inspection_with_none_values(self):
        """Test inspection functions with None values in config."""

        def test_func():
            return "test"

        test_func._puffinflow_state = True
        test_func._state_config = {
            "cpu": None,
            "memory": 512.0,
            "priority": None,
            "timeout": None,
        }

        config = get_state_config(test_func)
        assert config["cpu"] is None
        assert config["memory"] == 512.0
        assert config["priority"] is None

        summary = get_state_summary(test_func)
        # None/zero values should be filtered out
        assert "CPU=" not in summary
        assert "Memory=512.0MB" in summary

    def test_inspection_with_empty_collections(self):
        """Test inspection with empty collections."""

        def test_func():
            return "test"

        test_func._puffinflow_state = True
        test_func._state_config = {
            "depends_on": [],  # Empty list
            "tags": {},  # Empty dict
        }
        test_func._state_tags = {}
        test_func._dependency_configs = {}

        metadata = list_state_metadata(test_func)
        assert metadata["dependencies"] == {}
        assert metadata["tags"] == {}

        summary = get_state_summary(test_func)
        # Empty dependencies should not appear in summary
        assert "Dependencies:" not in summary

    def test_inspection_with_malformed_attributes(self):
        """Test inspection with malformed attributes."""

        def test_func():
            return "test"

        test_func._puffinflow_state = True
        test_func._state_config = "not a dict"  # Malformed

        # Should handle gracefully
        config = get_state_config(test_func)
        assert config == "not a dict"

        # Summary should handle this case
        summary = get_state_summary(test_func)
        assert "test_func:" in summary
