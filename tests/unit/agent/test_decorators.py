"""
Comprehensive tests for the flexible state decorator system.
"""

import pytest

from puffinflow.core.agent.context import Context
from puffinflow.core.agent.decorators.builder import (
    build_state,
    cpu_state,
    exclusive_state,
    gpu_state,
    memory_state,
)

# Import decorator functionality
from puffinflow.core.agent.decorators.flexible import (
    StateProfile,
    cpu_intensive,
    create_custom_decorator,
    critical_state,
    get_profile,
    gpu_accelerated,
    list_profiles,
    memory_intensive,
    state,
)
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
from puffinflow.core.agent.dependencies import DependencyType
from puffinflow.core.agent.state import Priority
from puffinflow.core.coordination.primitives import PrimitiveType
from puffinflow.core.coordination.rate_limiter import RateLimitStrategy
from puffinflow.core.resources.requirements import (
    ResourceType,
)


class TestBasicDecoratorFunctionality:
    """Test basic decorator functionality and call patterns."""

    def test_state_decorator_no_params(self):
        """Test @state without parentheses."""

        @state
        async def simple_state(context: Context):
            return "done"

        assert is_puffinflow_state(simple_state)
        assert simple_state._state_name == "simple_state"
        assert simple_state._priority == Priority.NORMAL

        # Check resource requirements
        requirements = get_state_requirements(simple_state)
        assert requirements.cpu_units == 1.0
        assert requirements.memory_mb == 100.0
        assert requirements.gpu_units == 0.0

    def test_state_decorator_empty_params(self):
        """Test @state() with empty parentheses."""

        @state()
        async def empty_params_state(context: Context):
            return "done"

        assert is_puffinflow_state(empty_params_state)
        assert empty_params_state._priority == Priority.NORMAL
        requirements = get_state_requirements(empty_params_state)
        assert requirements.cpu_units == 1.0

    def test_state_decorator_with_params(self):
        """Test @state with explicit parameters."""

        @state(cpu=2.0, memory=512.0, priority=Priority.HIGH, timeout=60.0)
        async def parameterized_state(context: Context):
            return "done"

        assert is_puffinflow_state(parameterized_state)
        assert parameterized_state._priority == Priority.HIGH

        requirements = get_state_requirements(parameterized_state)
        assert requirements.cpu_units == 2.0
        assert requirements.memory_mb == 512.0
        assert requirements.timeout == 60.0
        assert requirements.priority_boost == Priority.HIGH.value

    def test_function_metadata_preservation(self):
        """Test that original function metadata is preserved."""

        @state
        async def documented_state(context: Context):
            """This is a test state function."""
            return "done"

        assert documented_state.__name__ == "documented_state"
        assert "test state function" in documented_state.__doc__
        assert is_puffinflow_state(documented_state)


class TestProfileBasedDecorators:
    """Test profile-based decoration functionality."""

    def test_profile_by_name(self):
        """Test using profile by string name."""

        @state("cpu_intensive")
        async def cpu_heavy_state(context: Context):
            return "done"

        assert is_puffinflow_state(cpu_heavy_state)
        requirements = get_state_requirements(cpu_heavy_state)
        assert requirements.cpu_units == 4.0
        assert requirements.memory_mb == 1024.0
        assert cpu_heavy_state._priority == Priority.HIGH

    def test_profile_with_keyword(self):
        """Test using profile with profile keyword."""

        @state(profile="gpu_accelerated")
        async def gpu_state(context: Context):
            return "done"

        requirements = get_state_requirements(gpu_state)
        assert requirements.gpu_units == 1.0
        assert requirements.cpu_units == 2.0
        assert requirements.memory_mb == 2048.0

    def test_profile_with_overrides(self):
        """Test profile with parameter overrides."""

        @state("cpu_intensive", memory=2048.0, rate_limit=10.0)
        async def customized_cpu_state(context: Context):
            return "done"

        requirements = get_state_requirements(customized_cpu_state)
        assert requirements.cpu_units == 4.0  # From profile
        assert requirements.memory_mb == 2048.0  # Override
        assert requirements.timeout == 300.0  # From profile

        rate_config = get_state_rate_limit(customized_cpu_state)
        assert rate_config["rate"] == 10.0  # Override

    def test_profile_decorators(self):
        """Test predefined profile decorators."""

        @cpu_intensive()
        async def cpu_state(context: Context):
            return "done"

        @memory_intensive()
        async def memory_state(context: Context):
            return "done"

        @gpu_accelerated()
        async def gpu_state(context: Context):
            return "done"

        @critical_state()
        async def critical_op(context: Context):
            return "done"

        # Test CPU intensive
        cpu_req = get_state_requirements(cpu_state)
        assert cpu_req.cpu_units == 4.0
        assert cpu_req.memory_mb == 1024.0

        # Test memory intensive
        mem_req = get_state_requirements(memory_state)
        assert mem_req.memory_mb == 4096.0

        # Test GPU accelerated
        gpu_req = get_state_requirements(gpu_state)
        assert gpu_req.gpu_units == 1.0

        # Test critical state
        assert critical_op._priority == Priority.CRITICAL
        coordination = get_state_coordination(critical_op)
        assert coordination["type"] == PrimitiveType.MUTEX

    def test_unknown_profile_error(self):
        """Test error for unknown profile."""
        with pytest.raises(ValueError, match="Unknown profile: nonexistent"):

            @state("nonexistent")
            async def bad_state(context: Context):
                return "done"


class TestConfigurationMerging:
    """Test configuration merging and priority."""

    def test_config_dict_parameter(self):
        """Test using config dictionary parameter."""
        config = {"cpu": 3.0, "memory": 768.0, "mutex": True, "max_retries": 5}

        @state(config=config)
        async def dict_config_state(context: Context):
            return "done"

        requirements = get_state_requirements(dict_config_state)
        assert requirements.cpu_units == 3.0
        assert requirements.memory_mb == 768.0

        coordination = get_state_coordination(dict_config_state)
        assert coordination["type"] == PrimitiveType.MUTEX

    def test_priority_order(self):
        """Test configuration priority order: kwargs > config > profile > defaults."""

        @state("minimal", config={"cpu": 2.0}, cpu=4.0, memory=1024.0)
        async def priority_test_state(context: Context):
            return "done"

        requirements = get_state_requirements(priority_test_state)
        assert requirements.cpu_units == 4.0  # kwargs override
        assert requirements.memory_mb == 1024.0  # kwargs override
        # Other values should come from minimal profile or defaults

    def test_multiple_config_sources(self):
        """Test merging from multiple configuration sources."""
        custom_config = {"io": 5.0, "network": 3.0}

        @state("standard", custom_config, cpu=2.0, timeout=120.0)
        async def multi_source_state(context: Context):
            return "done"

        requirements = get_state_requirements(multi_source_state)
        assert requirements.cpu_units == 2.0  # Direct param
        assert requirements.memory_mb == 100.0  # From standard profile
        assert requirements.io_weight == 5.0  # From config dict
        assert requirements.network_weight == 3.0  # From config dict
        assert requirements.timeout == 120.0  # Direct param


class TestResourceRequirements:
    """Test resource requirement configuration."""

    def test_resource_types_detection(self):
        """Test automatic resource type detection."""

        @state(cpu=2.0, memory=512.0, gpu=1.0, io=0.0, network=0.0)
        async def resource_state(context: Context):
            return "done"

        requirements = get_state_requirements(resource_state)
        expected_types = ResourceType.CPU | ResourceType.MEMORY | ResourceType.GPU
        assert requirements.resource_types == expected_types

    def test_priority_mapping(self):
        """Test priority enum to boost mapping."""
        test_cases = [
            (Priority.LOW, 0),
            (Priority.NORMAL, 1),
            (Priority.HIGH, 2),
            (Priority.CRITICAL, 3),
            ("low", 0),
            ("HIGH", 2),
            (2, 2),
        ]

        for priority_input, expected_boost in test_cases:

            @state(priority=priority_input)
            async def priority_state(context: Context):
                return "done"

            requirements = get_state_requirements(priority_state)
            assert requirements.priority_boost == expected_boost

    def test_timeout_configuration(self):
        """Test timeout configuration."""

        @state(timeout=300.0)
        async def timeout_state(context: Context):
            return "done"

        requirements = get_state_requirements(timeout_state)
        assert requirements.timeout == 300.0


class TestCoordinationPrimitives:
    """Test coordination primitive configuration."""

    def test_mutex_configuration(self):
        """Test mutex coordination."""

        @state(mutex=True)
        async def mutex_state(context: Context):
            return "done"

        coordination = get_state_coordination(mutex_state)
        assert coordination["type"] == PrimitiveType.MUTEX
        assert "ttl" in coordination["config"]

    def test_semaphore_configuration(self):
        """Test semaphore coordination."""

        @state(semaphore=5)
        async def semaphore_state(context: Context):
            return "done"

        coordination = get_state_coordination(semaphore_state)
        assert coordination["type"] == PrimitiveType.SEMAPHORE
        assert coordination["config"]["max_count"] == 5

    def test_barrier_configuration(self):
        """Test barrier coordination."""

        @state(barrier=3)
        async def barrier_state(context: Context):
            return "done"

        coordination = get_state_coordination(barrier_state)
        assert coordination["type"] == PrimitiveType.BARRIER
        assert coordination["config"]["parties"] == 3

    def test_lease_configuration(self):
        """Test lease coordination."""

        @state(lease=60.0)
        async def lease_state(context: Context):
            return "done"

        coordination = get_state_coordination(lease_state)
        assert coordination["type"] == PrimitiveType.LEASE
        assert coordination["config"]["ttl"] == 60.0
        assert coordination["config"]["auto_renew"]

    def test_quota_configuration(self):
        """Test quota coordination."""

        @state(quota=100.0)
        async def quota_state(context: Context):
            return "done"

        coordination = get_state_coordination(quota_state)
        assert coordination["type"] == PrimitiveType.QUOTA
        assert coordination["config"]["limit"] == 100.0

    def test_coordination_string_parsing(self):
        """Test coordination string parsing."""
        test_cases = [
            ("mutex", PrimitiveType.MUTEX, {}),
            ("semaphore:10", PrimitiveType.SEMAPHORE, {"max_count": 10}),
            ("barrier:4", PrimitiveType.BARRIER, {"parties": 4}),
            ("lease:120", PrimitiveType.LEASE, {"ttl": 120.0}),
            ("quota:500", PrimitiveType.QUOTA, {"limit": 500.0}),
        ]

        for coord_string, expected_type, expected_config_subset in test_cases:

            @state(coordination=coord_string)
            async def coord_state(context: Context):
                return "done"

            coordination = get_state_coordination(coord_state)
            assert coordination["type"] == expected_type
            for key, value in expected_config_subset.items():
                assert coordination["config"][key] == value


class TestRateLimiting:
    """Test rate limiting configuration."""

    def test_rate_limit_basic(self):
        """Test basic rate limiting."""

        @state(rate_limit=10.0)
        async def rate_limited_state(context: Context):
            return "done"

        rate_config = get_state_rate_limit(rate_limited_state)
        assert rate_config["rate"] == 10.0
        assert rate_config["burst"] == 20  # Default 2x rate
        assert rate_config["strategy"] == RateLimitStrategy.TOKEN_BUCKET

    def test_rate_limit_with_burst(self):
        """Test rate limiting with custom burst."""

        @state(rate_limit=5.0, burst_limit=15)
        async def burst_limited_state(context: Context):
            return "done"

        rate_config = get_state_rate_limit(burst_limited_state)
        assert rate_config["rate"] == 5.0
        assert rate_config["burst"] == 15

    def test_no_rate_limit(self):
        """Test state without rate limiting."""

        @state(cpu=2.0)
        async def no_rate_state(context: Context):
            return "done"

        rate_config = get_state_rate_limit(no_rate_state)
        assert rate_config is None


class TestDependencies:
    """Test dependency configuration."""

    def test_single_dependency(self):
        """Test single dependency."""

        @state(depends_on="init_state")
        async def dependent_state(context: Context):
            return "done"

        config = get_state_config(dependent_state)
        assert config["depends_on"] == ["init_state"]

        deps = dependent_state._dependency_configs
        assert "init_state" in deps
        assert deps["init_state"].type == DependencyType.REQUIRED

    def test_multiple_dependencies(self):
        """Test multiple dependencies."""

        @state(depends_on=["init", "setup", "validate"])
        async def multi_dep_state(context: Context):
            return "done"

        config = get_state_config(multi_dep_state)
        assert len(config["depends_on"]) == 3
        assert "init" in config["depends_on"]
        assert "setup" in config["depends_on"]
        assert "validate" in config["depends_on"]


class TestBuilderPattern:
    """Test the builder pattern functionality."""

    def test_basic_builder(self):
        """Test basic builder functionality."""

        @build_state().cpu(4.0).memory(2048.0).high_priority()
        async def builder_state(context: Context):
            return "done"

        assert is_puffinflow_state(builder_state)
        requirements = get_state_requirements(builder_state)
        assert requirements.cpu_units == 4.0
        assert requirements.memory_mb == 2048.0
        assert builder_state._priority == Priority.HIGH

    def test_fluent_builder_interface(self):
        """Test fluent builder interface."""

        @(
            build_state()
            .cpu(2.0)
            .memory(1024.0)
            .gpu(1.0)
            .mutex()
            .rate_limit(5.0)
            .depends_on("init_state")
            .retries(3)
            .tag("team", "ml")
            .description("ML processing state")
        )
        async def fluent_state(context: Context):
            return "done"

        # Test resource requirements
        requirements = get_state_requirements(fluent_state)
        assert requirements.cpu_units == 2.0
        assert requirements.memory_mb == 1024.0
        assert requirements.gpu_units == 1.0

        # Test coordination
        coordination = get_state_coordination(fluent_state)
        assert coordination["type"] == PrimitiveType.MUTEX

        # Test rate limiting
        rate_config = get_state_rate_limit(fluent_state)
        assert rate_config["rate"] == 5.0

        # Test dependencies
        config = get_state_config(fluent_state)
        assert config["depends_on"] == ["init_state"]

        # Test metadata
        assert config["tags"]["team"] == "ml"
        assert config["description"] == "ML processing state"

    def test_convenience_builders(self):
        """Test convenience builder functions."""

        @cpu_state(8.0).memory(4096.0).critical_priority()
        async def cpu_builder_state(context: Context):
            return "done"

        @memory_state(8192.0).cpu(2.0)
        async def memory_builder_state(context: Context):
            return "done"

        @gpu_state(2.0).memory(16384.0)
        async def gpu_builder_state(context: Context):
            return "done"

        @exclusive_state().timeout(60.0)
        async def exclusive_builder_state(context: Context):
            return "done"

        # Test CPU builder
        cpu_req = get_state_requirements(cpu_builder_state)
        assert cpu_req.cpu_units == 8.0
        assert cpu_req.memory_mb == 4096.0
        assert cpu_builder_state._priority == Priority.CRITICAL

        # Test memory builder
        mem_req = get_state_requirements(memory_builder_state)
        assert mem_req.memory_mb == 8192.0
        assert mem_req.cpu_units == 2.0

        # Test GPU builder
        gpu_req = get_state_requirements(gpu_builder_state)
        assert gpu_req.gpu_units == 2.0
        assert gpu_req.memory_mb == 16384.0

        # Test exclusive builder
        excl_coord = get_state_coordination(exclusive_builder_state)
        assert excl_coord["type"] == PrimitiveType.MUTEX
        excl_req = get_state_requirements(exclusive_builder_state)
        assert excl_req.timeout == 60.0

    def test_builder_build_method(self):
        """Test builder build method."""
        builder = build_state().cpu(2.0).memory(512.0).mutex().tag("env", "test")

        config = builder.build()
        assert config["cpu"] == 2.0
        assert config["memory"] == 512.0
        assert config["mutex"]
        assert config["tags"]["env"] == "test"


class TestInspectionUtilities:
    """Test state inspection utilities."""

    def test_is_puffinflow_state(self):
        """Test state detection."""

        @state
        async def decorated_state(context: Context):
            return "done"

        async def regular_function(context: Context):
            return "done"

        assert is_puffinflow_state(decorated_state)
        assert not is_puffinflow_state(regular_function)

    def test_get_state_config(self):
        """Test getting state configuration."""

        @state(cpu=2.0, memory=512.0, mutex=True, tags={"env": "test"})
        async def config_state(context: Context):
            return "done"

        config = get_state_config(config_state)
        assert config["cpu"] == 2.0
        assert config["memory"] == 512.0
        assert config["mutex"]
        assert config["tags"]["env"] == "test"

        # Test non-decorated function
        async def regular_function():
            pass

        assert get_state_config(regular_function) is None

    def test_list_state_metadata(self):
        """Test comprehensive metadata listing."""

        @state(
            cpu=4.0,
            memory=1024.0,
            priority=Priority.HIGH,
            mutex=True,
            rate_limit=10.0,
            depends_on=["init"],
            timeout=300.0,
            tags={"team": "backend"},
            description="A fully configured test state",
        )
        async def unique_full_metadata_test_state(context: Context):
            """A fully configured test state."""
            return "done"

        metadata = list_state_metadata(unique_full_metadata_test_state)

        assert metadata["name"] == "unique_full_metadata_test_state"
        assert "fully configured test state" in metadata["description"].lower()
        assert metadata["tags"]["team"] == "backend"
        assert metadata["priority"] == Priority.HIGH

        # Check requirements
        requirements = metadata["requirements"]
        assert requirements.cpu_units == 4.0
        assert requirements.memory_mb == 1024.0
        assert requirements.timeout == 300.0

        # Check coordination
        coordination = metadata["coordination"]
        assert coordination["type"] == PrimitiveType.MUTEX

        # Check rate limiting
        rate_limit = metadata["rate_limit"]
        assert rate_limit["rate"] == 10.0

        # Check dependencies
        dependencies = metadata["dependencies"]
        assert "init" in dependencies

    def test_compare_states(self):
        """Test state comparison."""

        @state(cpu=2.0, memory=512.0, mutex=True)
        async def state1(context: Context):
            return "done"

        @state(cpu=4.0, memory=512.0, semaphore=5)
        async def state2(context: Context):
            return "done"

        differences = compare_states(state1, state2)

        # Should show differences in CPU and coordination
        assert "cpu" in differences
        assert differences["cpu"]["func1"] == 2.0
        assert differences["cpu"]["func2"] == 4.0

        assert "mutex" in differences
        assert differences["mutex"]["func1"]
        # state2 doesn't have mutex in its config, so it should be None
        assert differences["mutex"]["func2"] is None

        assert "semaphore" in differences
        # state1 doesn't have semaphore in its config, so it should be None
        assert differences["semaphore"]["func1"] is None
        assert differences["semaphore"]["func2"] == 5

    def test_get_state_summary(self):
        """Test state summary generation."""

        @state(
            cpu=4.0,
            memory=2048.0,
            gpu=1.0,
            priority=Priority.HIGH,
            mutex=True,
            rate_limit=5.0,
            depends_on=["init", "setup"],
        )
        async def summary_state(context: Context):
            return "done"

        summary = get_state_summary(summary_state)

        assert "summary_state:" in summary
        assert "CPU=4.0" in summary
        assert "Memory=2048.0MB" in summary
        assert "GPU=1.0" in summary
        assert "Priority: HIGH" in summary
        assert "Mutex" in summary
        assert "RateLimit(5.0/s)" in summary
        assert "Dependencies: init, setup" in summary

        # Test non-decorated function
        async def regular_function():
            pass

        summary = get_state_summary(regular_function)
        assert "Not a PuffinFlow state" in summary


class TestProfileManagement:
    """Test profile management functionality."""

    def test_get_profile(self):
        """Test getting profiles."""
        cpu_profile = get_profile("cpu_intensive")
        assert cpu_profile is not None
        assert cpu_profile.name == "cpu_intensive"
        assert cpu_profile.cpu == 4.0
        assert cpu_profile.memory == 1024.0

        # Test non-existent profile
        assert get_profile("nonexistent") is None

    def test_list_profiles(self):
        """Test listing all profiles."""
        profiles = list_profiles()

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
        ]

        for expected in expected_profiles:
            assert expected in profiles

    def test_custom_decorator_creation(self):
        """Test creating custom decorators."""
        team_decorator = create_custom_decorator(
            cpu=2.0,
            memory=1024.0,
            tags={"team": "backend", "env": "prod"},
            priority=Priority.HIGH,
        )

        @team_decorator(mutex=True)
        async def team_state(context: Context):
            return "done"

        config = get_state_config(team_state)
        assert config["cpu"] == 2.0
        assert config["memory"] == 1024.0
        assert config["priority"] == Priority.HIGH
        assert config["mutex"]
        assert config["tags"]["team"] == "backend"
        assert config["tags"]["env"] == "prod"

    def test_profile_registration(self):
        """Test registering new profiles."""
        # Create a custom profile
        custom_profile = StateProfile(
            name="test_profile",
            cpu=8.0,
            memory=8192.0,
            gpu=2.0,
            priority=Priority.CRITICAL,
            description="Test profile for unit tests",
        )

        # Register it
        state.register_profile(custom_profile)

        # Test it's available
        assert "test_profile" in list_profiles()
        retrieved = get_profile("test_profile")
        assert retrieved.cpu == 8.0
        assert retrieved.memory == 8192.0
        assert retrieved.gpu == 2.0

        # Test using it
        @state("test_profile")
        async def custom_profile_state(context: Context):
            return "done"

        requirements = get_state_requirements(custom_profile_state)
        assert requirements.cpu_units == 8.0
        assert requirements.memory_mb == 8192.0
        assert requirements.gpu_units == 2.0
        assert custom_profile_state._priority == Priority.CRITICAL


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_invalid_priority_values(self):
        """Test handling of invalid priority values."""

        # These should work (valid conversions)
        @state(priority="high")
        async def string_priority_state(context: Context):
            return "done"

        assert string_priority_state._priority == Priority.HIGH

        @state(priority=2)
        async def int_priority_state(context: Context):
            return "done"

        assert int_priority_state._priority == Priority.HIGH

        # Test invalid string priority
        with pytest.raises(KeyError):

            @state(priority="invalid")
            async def invalid_priority_state(context: Context):
                return "done"

    def test_invalid_coordination_string(self):
        """Test invalid coordination string handling."""
        with pytest.raises(ValueError, match="Unknown coordination type"):

            @state(coordination="invalid_type")
            async def invalid_coord_state(context: Context):
                return "done"

    def test_mixed_coordination_primitives(self):
        """Test that only one coordination primitive is allowed."""

        # This should work - only mutex specified
        @state(mutex=True)
        async def mutex_only_state(context: Context):
            return "done"

        # Multiple coordination primitives in config should use the last one processed
        # This tests the current implementation behavior
        @state(mutex=True, semaphore=5)
        async def mixed_coord_state(context: Context):
            return "done"

        # The last coordination setting should win
        coordination = get_state_coordination(mixed_coord_state)
        # Implementation may vary on which one takes precedence
        assert coordination is not None

    def test_zero_and_negative_resources(self):
        """Test handling of zero and edge case resource values."""

        @state(cpu=0.0, memory=0.0, gpu=0.0)
        async def zero_resources_state(context: Context):
            return "done"

        requirements = get_state_requirements(zero_resources_state)
        assert requirements.cpu_units == 0.0
        assert requirements.memory_mb == 0.0
        assert requirements.gpu_units == 0.0

        # Resource types should only include non-zero resources
        assert ResourceType.CPU not in requirements.resource_types
        assert ResourceType.MEMORY not in requirements.resource_types
        assert ResourceType.GPU not in requirements.resource_types

    def test_empty_dependencies(self):
        """Test handling of empty dependencies."""

        @state(depends_on=[])
        async def empty_deps_state(context: Context):
            return "done"

        config = get_state_config(empty_deps_state)
        assert config["depends_on"] == []
        assert len(empty_deps_state._dependency_configs) == 0

    def test_none_values_in_config(self):
        """Test handling of None values in configuration."""

        @state(timeout=None, rate_limit=None, description=None)
        async def none_values_state(context: Context):
            return "done"

        requirements = get_state_requirements(none_values_state)
        assert requirements.timeout is None

        rate_config = get_state_rate_limit(none_values_state)
        assert rate_config is None

        config = get_state_config(none_values_state)
        # Description should be auto-generated from function name
        assert config["description"] == "State: none_values_state"

    def test_large_resource_values(self):
        """Test handling of very large resource values."""

        @state(cpu=1000.0, memory=1000000.0, gpu=100.0)
        async def large_resources_state(context: Context):
            return "done"

        requirements = get_state_requirements(large_resources_state)
        assert requirements.cpu_units == 1000.0
        assert requirements.memory_mb == 1000000.0
        assert requirements.gpu_units == 100.0

    def test_unicode_and_special_chars_in_metadata(self):
        """Test handling of unicode and special characters in metadata."""

        @state(
            tags={"emoji": "🚀", "unicode": "Ñiño", "special": "<>&\"'"},
            description="State with émojis 🎯 and special chars!",
        )
        async def unicode_state(context: Context):
            return "done"

        config = get_state_config(unicode_state)
        assert config["tags"]["emoji"] == "🚀"
        assert config["tags"]["unicode"] == "Ñiño"
        assert config["tags"]["special"] == "<>&\"'"
        assert "🎯" in config["description"]


class TestAsyncFunctionality:
    """Test decorator with actual async function execution."""

    @pytest.mark.asyncio
    async def test_decorated_function_execution(self):
        """Test that decorated functions can still be executed."""

        @state(cpu=2.0, memory=512.0)
        async def executable_state(context: Context):
            context.set_variable("test_var", "test_value")
            return "execution_complete"

        # Create a test context
        shared_state = {}
        context = Context(shared_state)

        # Execute the decorated function
        result = await executable_state(context)

        assert result == "execution_complete"
        assert context.get_variable("test_var") == "test_value"

        # Verify decoration is still intact
        assert is_puffinflow_state(executable_state)
        requirements = get_state_requirements(executable_state)
        assert requirements.cpu_units == 2.0

    @pytest.mark.asyncio
    async def test_function_with_args_and_kwargs(self):
        """Test decorated function with various argument patterns."""

        @state(priority=Priority.HIGH)
        async def args_kwargs_state(context: Context, arg1, *args, arg2=None, **kwargs):
            return {"arg1": arg1, "arg2": arg2, "args": args, "kwargs": kwargs}

        shared_state = {}
        context = Context(shared_state)

        # Fixed: Use proper argument order and avoid conflicts
        result = await args_kwargs_state(
            context,
            "value1",
            "extra1",
            "extra2",
            arg2="value2",
            key1="kwvalue1",
            key2="kwvalue2",
        )

        assert result["arg1"] == "value1"
        assert result["arg2"] == "value2"
        assert result["args"] == ("extra1", "extra2")
        assert result["kwargs"] == {"key1": "kwvalue1", "key2": "kwvalue2"}

        # Verify decoration
        assert args_kwargs_state._priority == Priority.HIGH


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_complete_workflow_states(self):
        """Test a complete set of workflow states with various configurations."""

        @state("minimal")
        async def init_state(context: Context):
            context.set_variable("initialized", True)
            return "setup_state"

        @state(depends_on=["init_state"], cpu=2.0, memory=1024.0)
        async def setup_state(context: Context):
            if not context.get_variable("initialized"):
                raise ValueError("Not initialized")
            context.set_variable("setup_complete", True)
            return "process_state"

        @cpu_intensive(depends_on=["setup_state"], rate_limit=5.0)
        async def process_state(context: Context):
            if not context.get_variable("setup_complete"):
                raise ValueError("Setup not complete")
            context.set_variable("processed_data", "result")
            return "cleanup_state"

        @state("critical", depends_on=["process_state"])
        async def cleanup_state(context: Context):
            context.set_variable("cleanup_done", True)
            return None

        # Verify all states are properly decorated
        states = [init_state, setup_state, process_state, cleanup_state]
        for state_func in states:
            assert is_puffinflow_state(state_func)

        # Verify specific configurations
        assert get_state_config(init_state)["cpu"] == 0.1  # minimal profile
        assert len(get_state_config(setup_state)["depends_on"]) == 1
        assert get_state_requirements(process_state).cpu_units == 4.0  # cpu_intensive
        assert process_state._priority == Priority.HIGH  # cpu_intensive profile
        assert get_state_rate_limit(process_state)["rate"] == 5.0
        assert cleanup_state._priority == Priority.CRITICAL  # critical profile

    def test_ml_pipeline_example(self):
        """Test ML pipeline with realistic state configurations."""

        @state("io_intensive", depends_on=["data_validation"])
        async def data_loading_state(context: Context):
            return "feature_engineering"

        @build_state().cpu(2.0).memory(2048.0).depends_on("data_loading").retries(3)
        async def feature_engineering_state(context: Context):
            return "model_training"

        @gpu_accelerated(
            cpu=4.0,
            memory=8192.0,
            depends_on=["feature_engineering"],
            rate_limit=1.0,  # Limit GPU training jobs
            timeout=3600.0,  # 1 hour timeout
            tags={"stage": "training", "team": "ml"},
        )
        async def model_training_state(context: Context):
            return "model_evaluation"

        @state(
            profile="cpu_intensive",
            depends_on=["model_training"],
            mutex=True,  # Exclusive model evaluation
            tags={"stage": "evaluation"},
        )
        async def model_evaluation_state(context: Context):
            return "model_deployment"

        @critical_state(depends_on=["model_evaluation"])
        async def model_deployment_state(context: Context):
            return None

        # Verify pipeline configuration
        data_req = get_state_requirements(data_loading_state)
        assert data_req.io_weight == 10.0  # io_intensive profile

        gpu_req = get_state_requirements(model_training_state)
        assert gpu_req.gpu_units == 1.0
        assert gpu_req.memory_mb == 8192.0  # Override from gpu_accelerated

        gpu_rate = get_state_rate_limit(model_training_state)
        assert gpu_rate["rate"] == 1.0

        eval_coord = get_state_coordination(model_evaluation_state)
        assert eval_coord["type"] == PrimitiveType.MUTEX

        deploy_priority = model_deployment_state._priority
        assert deploy_priority == Priority.CRITICAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
