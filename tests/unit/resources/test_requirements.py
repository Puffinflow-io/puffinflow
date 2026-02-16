"""
Comprehensive unit tests for ResourceRequirements and ResourceType.

Tests cover:
- ResourceType enum functionality and flag operations
- ResourceRequirements initialization and defaults
- Priority property getter/setter logic
- Resource type combinations and validations
- Edge cases and attribute modifications
- Integration with Priority enum
- Resource amount calculations and validations
- Serialization/representation methods
- Complex scenario testing
- Performance and stress testing
"""

import sys
import threading
import time
from dataclasses import asdict, fields

import pytest

from puffinflow.core.agent.state import Priority

# Import the classes under test
from puffinflow.core.resources.requirements import (
    ResourceRequirements,
    ResourceType,
    get_resource_amount,
)


class TestResourceType:
    """Test ResourceType enum functionality."""

    def test_resource_type_values(self):
        """Test ResourceType enum values are powers of 2."""
        assert ResourceType.NONE.value == 0
        assert ResourceType.CPU.value == 1
        assert ResourceType.MEMORY.value == 2
        assert ResourceType.IO.value == 4
        assert ResourceType.NETWORK.value == 8
        assert ResourceType.GPU.value == 16

        # Verify they are proper flags (powers of 2)
        for rt in ResourceType:
            if rt != ResourceType.NONE and rt != ResourceType.ALL:
                assert (
                    rt.value & (rt.value - 1)
                ) == 0, f"{rt.name} is not a power of 2"

    def test_resource_type_all_combination(self):
        """Test ResourceType.ALL includes all resource types."""
        expected = (
            ResourceType.CPU
            | ResourceType.MEMORY
            | ResourceType.IO
            | ResourceType.NETWORK
            | ResourceType.GPU
        )
        assert expected == ResourceType.ALL
        assert ResourceType.ALL.value == 31  # 1 + 2 + 4 + 8 + 16

    def test_resource_type_flag_operations(self):
        """Test flag operations on ResourceType."""
        # Single type
        cpu_only = ResourceType.CPU
        assert ResourceType.CPU in cpu_only
        assert ResourceType.MEMORY not in cpu_only

        # Combined types
        cpu_and_memory = ResourceType.CPU | ResourceType.MEMORY
        assert ResourceType.CPU in cpu_and_memory
        assert ResourceType.MEMORY in cpu_and_memory
        assert ResourceType.IO not in cpu_and_memory

        # All types
        for rt in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            assert rt in ResourceType.ALL

    def test_resource_type_operations_comprehensive(self):
        """Test comprehensive flag operations."""
        # Test XOR
        cpu_xor_memory = ResourceType.CPU ^ ResourceType.MEMORY
        assert ResourceType.CPU in cpu_xor_memory
        assert ResourceType.MEMORY in cpu_xor_memory
        assert cpu_xor_memory == (ResourceType.CPU | ResourceType.MEMORY)

        # Test intersection
        all_vs_cpu_mem = ResourceType.ALL & (ResourceType.CPU | ResourceType.MEMORY)
        assert all_vs_cpu_mem == (ResourceType.CPU | ResourceType.MEMORY)

        # Test negation
        not_cpu = ResourceType.ALL & ~ResourceType.CPU
        assert ResourceType.CPU not in not_cpu
        for rt in [
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            assert rt in not_cpu

    def test_resource_type_combinations(self):
        """Test various ResourceType combinations."""
        # CPU + Memory
        combo1 = ResourceType.CPU | ResourceType.MEMORY
        assert combo1.value == 3  # 1 + 2

        # IO + Network
        combo2 = ResourceType.IO | ResourceType.NETWORK
        assert combo2.value == 12  # 4 + 8

        # CPU + GPU
        combo3 = ResourceType.CPU | ResourceType.GPU
        assert combo3.value == 17  # 1 + 16

        # All except GPU
        combo4 = ResourceType.ALL & ~ResourceType.GPU
        assert ResourceType.CPU in combo4
        assert ResourceType.MEMORY in combo4
        assert ResourceType.IO in combo4
        assert ResourceType.NETWORK in combo4
        assert ResourceType.GPU not in combo4

    def test_resource_type_none(self):
        """Test ResourceType.NONE behavior."""
        assert ResourceType.NONE.value == 0
        for rt in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            assert rt not in ResourceType.NONE

        # NONE with other types
        cpu_with_none = ResourceType.CPU | ResourceType.NONE
        assert cpu_with_none == ResourceType.CPU

    def test_resource_type_string_representation(self):
        """Test string representation of ResourceType."""
        assert str(ResourceType.CPU) == "ResourceType.CPU"
        assert repr(ResourceType.CPU) == "<ResourceType.CPU: 1>"

        # Test combined types
        combined = ResourceType.CPU | ResourceType.MEMORY
        # Note: The exact string representation may vary by Python version
        assert "CPU" in str(combined) or "3" in str(combined)

    def test_resource_type_iteration(self):
        """Test iteration over ResourceType enum."""
        all_types = list(ResourceType)

        # Define the expected basic resource types (power-of-2 values)
        expected_basic_types = [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]

        # Test that all expected basic types are present in iteration
        for rt in expected_basic_types:
            assert rt in all_types, f"{rt} not found in ResourceType iteration"

        # Test that all iterated types are either basic types or known composite types
        # (Handle different Python versions that may include NONE/ALL in iteration)
        for rt in all_types:
            assert rt in expected_basic_types or rt in [
                ResourceType.NONE,
                ResourceType.ALL,
            ], f"Unexpected ResourceType in iteration: {rt}"

        # Verify that we have at least the expected basic types
        # (Don't do strict length check since some platforms may include NONE/ALL)
        basic_types_found = [rt for rt in all_types if rt in expected_basic_types]
        assert (
            len(basic_types_found) == len(expected_basic_types)
        ), f"Missing basic types. Expected: {expected_basic_types}, Found: {basic_types_found}"

        # Verify NONE and ALL are always accessible regardless of iteration behavior
        assert ResourceType.NONE.value == 0
        assert ResourceType.ALL.value == 31

        # Test that ALL is the combination of all basic types
        assert (
            ResourceType.CPU
            | ResourceType.MEMORY
            | ResourceType.IO
            | ResourceType.NETWORK
            | ResourceType.GPU
        ) == ResourceType.ALL

        # Test that each basic type is a power of 2 (valid flag)
        for rt in expected_basic_types:
            assert (
                rt.value > 0 and (rt.value & (rt.value - 1)) == 0
            ), f"{rt.name} value {rt.value} is not a power of 2"


class TestGetResourceAmount:
    """Test get_resource_amount function."""

    def test_get_resource_amount_basic(self):
        """Test basic get_resource_amount functionality."""
        req = ResourceRequirements(
            cpu_units=2.0,
            memory_mb=512.0,
            io_weight=3.0,
            network_weight=2.5,
            gpu_units=1.0,
        )

        assert get_resource_amount(req, ResourceType.CPU) == 2.0
        assert get_resource_amount(req, ResourceType.MEMORY) == 512.0
        assert get_resource_amount(req, ResourceType.IO) == 3.0
        assert get_resource_amount(req, ResourceType.NETWORK) == 2.5
        assert get_resource_amount(req, ResourceType.GPU) == 1.0

    def test_get_resource_amount_none_and_all(self):
        """Test get_resource_amount with NONE and ALL."""
        req = ResourceRequirements(cpu_units=1.0, memory_mb=100.0)

        # NONE should return 0
        assert get_resource_amount(req, ResourceType.NONE) == 0.0

        # ALL should return sum of all resources
        expected_total = 1.0 + 100.0 + 1.0 + 1.0 + 0.0  # cpu + mem + io + net + gpu
        assert get_resource_amount(req, ResourceType.ALL) == expected_total

    def test_get_resource_amount_zero_values(self):
        """Test get_resource_amount with zero values."""
        req = ResourceRequirements(
            cpu_units=0.0,
            memory_mb=0.0,
            io_weight=0.0,
            network_weight=0.0,
            gpu_units=0.0,
        )

        for rt in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            assert get_resource_amount(req, rt) == 0.0

    def test_get_resource_amount_invalid_type(self):
        """Test get_resource_amount with invalid resource type."""
        req = ResourceRequirements()

        # Test with a combined type (should raise error or return sum)
        combined = ResourceType.CPU | ResourceType.MEMORY
        # This behavior depends on implementation - adjust based on actual function
        try:
            result = get_resource_amount(req, combined)
            # If it doesn't raise an error, verify it makes sense
            assert isinstance(result, (int, float))
        except (ValueError, KeyError):
            # Expected if function doesn't handle combined types
            pass


class TestResourceRequirementsInitialization:
    """Test ResourceRequirements initialization."""

    def test_default_initialization(self):
        """Test ResourceRequirements with default values."""
        req = ResourceRequirements()

        assert req.cpu_units == 1.0
        assert req.memory_mb == 100.0
        assert req.io_weight == 1.0
        assert req.network_weight == 1.0
        assert req.gpu_units == 0.0
        assert req.priority_boost == 0
        assert req.timeout is None
        assert req.resource_types == ResourceType.ALL

    def test_custom_initialization(self):
        """Test ResourceRequirements with custom values."""
        req = ResourceRequirements(
            cpu_units=2.5,
            memory_mb=512.0,
            io_weight=3.0,
            network_weight=2.5,
            gpu_units=1.0,
            priority_boost=2,
            timeout=30.0,
            resource_types=ResourceType.CPU | ResourceType.MEMORY,
        )

        assert req.cpu_units == 2.5
        assert req.memory_mb == 512.0
        assert req.io_weight == 3.0
        assert req.network_weight == 2.5
        assert req.gpu_units == 1.0
        assert req.priority_boost == 2
        assert req.timeout == 30.0
        assert req.resource_types == ResourceType.CPU | ResourceType.MEMORY

    def test_keyword_only_initialization(self):
        """Test that initialization works with keyword arguments only."""
        # This tests the dataclass behavior
        req = ResourceRequirements(cpu_units=4.0, memory_mb=1024.0, gpu_units=2.0)

        assert req.cpu_units == 4.0
        assert req.memory_mb == 1024.0
        assert req.gpu_units == 2.0
        # Defaults for others
        assert req.io_weight == 1.0
        assert req.network_weight == 1.0

    def test_partial_initialization(self):
        """Test ResourceRequirements with partial values."""
        req = ResourceRequirements(cpu_units=4.0, memory_mb=1024.0)

        assert req.cpu_units == 4.0
        assert req.memory_mb == 1024.0
        # Other values should be defaults
        assert req.io_weight == 1.0
        assert req.network_weight == 1.0
        assert req.gpu_units == 0.0
        assert req.priority_boost == 0
        assert req.timeout is None
        assert req.resource_types == ResourceType.ALL

    def test_zero_values(self):
        """Test ResourceRequirements with zero values."""
        req = ResourceRequirements(
            cpu_units=0.0,
            memory_mb=0.0,
            io_weight=0.0,
            network_weight=0.0,
            gpu_units=0.0,
            priority_boost=0,
        )

        assert req.cpu_units == 0.0
        assert req.memory_mb == 0.0
        assert req.io_weight == 0.0
        assert req.network_weight == 0.0
        assert req.gpu_units == 0.0
        assert req.priority_boost == 0

    def test_dataclass_fields(self):
        """Test that ResourceRequirements has expected dataclass fields."""
        req_fields = {field.name for field in fields(ResourceRequirements)}
        expected_fields = {
            "cpu_units",
            "memory_mb",
            "io_weight",
            "network_weight",
            "gpu_units",
            "priority_boost",
            "timeout",
            "resource_types",
        }
        assert req_fields == expected_fields


class TestPriorityProperty:
    """Test priority property getter/setter logic."""

    def test_priority_getter_default(self):
        """Test priority property getter with default priority_boost."""
        req = ResourceRequirements()  # priority_boost=0 by default
        assert req.priority == Priority.LOW

    def test_priority_getter_mapping(self):
        """Test priority property getter with different priority_boost values."""
        test_cases = [
            (0, Priority.LOW),
            (1, Priority.NORMAL),
            (2, Priority.HIGH),
            (3, Priority.CRITICAL),
            (5, Priority.CRITICAL),  # Any value >= 3 should be CRITICAL
            (100, Priority.CRITICAL),
        ]

        for boost_value, expected_priority in test_cases:
            req = ResourceRequirements(priority_boost=boost_value)
            assert (
                req.priority == expected_priority
            ), f"Failed for boost_value={boost_value}"

    def test_priority_getter_negative_values(self):
        """Test priority property getter with negative priority_boost values."""
        # Negative values should still map to LOW
        for negative_value in [-1, -10, -100]:
            req = ResourceRequirements(priority_boost=negative_value)
            assert req.priority == Priority.LOW

    def test_priority_setter_with_enum(self):
        """Test priority property setter with Priority enum values."""
        req = ResourceRequirements()

        for priority in Priority:
            req.priority = priority
            assert req.priority_boost == priority.value
            assert req.priority == priority

    def test_priority_setter_with_int(self):
        """Test priority property setter with integer values."""
        req = ResourceRequirements()

        test_cases = [
            (0, Priority.LOW),
            (1, Priority.NORMAL),
            (2, Priority.HIGH),
            (3, Priority.CRITICAL),
            (10, Priority.CRITICAL),  # High values map to CRITICAL
        ]

        for int_value, expected_priority in test_cases:
            req.priority = int_value
            assert req.priority_boost == int_value
            assert req.priority == expected_priority

    def test_priority_setter_with_invalid_type(self):
        """Test priority property setter with invalid types."""
        req = ResourceRequirements()

        # Test with string
        with pytest.raises((TypeError, ValueError)):
            req.priority = "high"

        # Test with float
        with pytest.raises((TypeError, ValueError)):
            req.priority = 2.5

    def test_priority_roundtrip(self):
        """Test priority getter/setter roundtrip."""
        req = ResourceRequirements()

        for priority in Priority:
            req.priority = priority
            assert req.priority == priority
            assert req.priority_boost == priority.value

    def test_priority_consistency(self):
        """Test priority consistency across multiple operations."""
        req = ResourceRequirements()

        # Test multiple modifications
        req.priority = Priority.HIGH
        req.priority_boost += 1  # Should now be CRITICAL
        assert req.priority == Priority.CRITICAL

        req.priority = Priority.LOW
        assert req.priority_boost == Priority.LOW.value


class TestResourceTypesHandling:
    """Test resource_types field functionality."""

    def test_default_resource_types(self):
        """Test default resource_types value."""
        req = ResourceRequirements()
        assert req.resource_types == ResourceType.ALL

        # All types should be included
        for rt in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            assert rt in req.resource_types

    def test_custom_resource_types(self):
        """Test custom resource_types combinations."""
        test_cases = [
            (
                ResourceType.CPU,
                [ResourceType.CPU],
                [ResourceType.MEMORY, ResourceType.IO],
            ),
            (
                ResourceType.CPU | ResourceType.MEMORY,
                [ResourceType.CPU, ResourceType.MEMORY],
                [ResourceType.IO, ResourceType.NETWORK, ResourceType.GPU],
            ),
            (
                ResourceType.ALL & ~ResourceType.GPU,
                [
                    ResourceType.CPU,
                    ResourceType.MEMORY,
                    ResourceType.IO,
                    ResourceType.NETWORK,
                ],
                [ResourceType.GPU],
            ),
        ]

        for resource_types, should_include, should_exclude in test_cases:
            req = ResourceRequirements(resource_types=resource_types)
            assert req.resource_types == resource_types

            for rt in should_include:
                assert rt in req.resource_types
            for rt in should_exclude:
                assert rt not in req.resource_types

    def test_none_resource_types(self):
        """Test ResourceType.NONE behavior."""
        req = ResourceRequirements(resource_types=ResourceType.NONE)
        assert req.resource_types == ResourceType.NONE

        # No types should be included
        for rt in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            assert rt not in req.resource_types

    def test_resource_types_modification(self):
        """Test modifying resource_types after initialization."""
        req = ResourceRequirements()

        # Start with ALL
        assert req.resource_types == ResourceType.ALL

        # Change to CPU only
        req.resource_types = ResourceType.CPU
        assert req.resource_types == ResourceType.CPU
        assert ResourceType.CPU in req.resource_types
        assert ResourceType.MEMORY not in req.resource_types

        # Change to combined
        req.resource_types = ResourceType.CPU | ResourceType.GPU
        assert ResourceType.CPU in req.resource_types
        assert ResourceType.GPU in req.resource_types
        assert ResourceType.MEMORY not in req.resource_types


class TestAttributeModification:
    """Test modification of ResourceRequirements attributes."""

    def test_modify_resource_amounts(self):
        """Test modifying resource amount attributes."""
        req = ResourceRequirements()

        # Test each resource type
        modifications = [
            ("cpu_units", 4.0),
            ("memory_mb", 2048.0),
            ("io_weight", 5.0),
            ("network_weight", 3.5),
            ("gpu_units", 2.0),
        ]

        for attr, value in modifications:
            setattr(req, attr, value)
            assert getattr(req, attr) == value

    def test_modify_priority_boost(self):
        """Test modifying priority_boost attribute."""
        req = ResourceRequirements()

        test_values = [0, 1, 2, 3, 10, -5]
        for value in test_values:
            req.priority_boost = value
            assert req.priority_boost == value

    def test_modify_timeout(self):
        """Test modifying timeout attribute."""
        req = ResourceRequirements()

        assert req.timeout is None

        # Set various timeout values
        timeout_values = [60.0, 0.0, 3600.0, 0.1]
        for timeout in timeout_values:
            req.timeout = timeout
            assert req.timeout == timeout

        # Reset to None
        req.timeout = None
        assert req.timeout is None

    def test_chained_modifications(self):
        """Test chaining multiple attribute modifications."""
        req = ResourceRequirements()

        # Chain modifications
        req.cpu_units = 2.0
        req.memory_mb = 512.0
        req.priority = Priority.HIGH
        req.timeout = 30.0

        # Verify all changes
        assert req.cpu_units == 2.0
        assert req.memory_mb == 512.0
        assert req.priority == Priority.HIGH
        assert req.timeout == 30.0


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_negative_values(self):
        """Test ResourceRequirements with negative values."""
        req = ResourceRequirements(
            cpu_units=-1.0,
            memory_mb=-100.0,
            io_weight=-2.0,
            network_weight=-1.5,
            gpu_units=-0.5,
            priority_boost=-5,
        )

        assert req.cpu_units == -1.0
        assert req.memory_mb == -100.0
        assert req.io_weight == -2.0
        assert req.network_weight == -1.5
        assert req.gpu_units == -0.5
        assert req.priority_boost == -5

    def test_very_large_values(self):
        """Test ResourceRequirements with very large values."""
        large_value = sys.float_info.max / 10  # Avoid overflow

        req = ResourceRequirements(
            cpu_units=large_value,
            memory_mb=1e9,
            io_weight=999999.9,
            network_weight=1e6,
            gpu_units=100.0,
            priority_boost=1000,
        )

        assert req.cpu_units == large_value
        assert req.memory_mb == 1e9
        assert req.io_weight == 999999.9
        assert req.network_weight == 1e6
        assert req.gpu_units == 100.0
        assert req.priority_boost == 1000
        assert req.priority == Priority.CRITICAL

    def test_very_small_values(self):
        """Test ResourceRequirements with very small positive values."""
        small_value = sys.float_info.min * 10

        req = ResourceRequirements(
            cpu_units=small_value,
            memory_mb=1e-6,
            io_weight=0.001,
            network_weight=1e-3,
            gpu_units=0.0001,
        )

        assert req.cpu_units == small_value
        assert req.memory_mb == 1e-6
        assert req.io_weight == 0.001
        assert req.network_weight == 1e-3
        assert req.gpu_units == 0.0001

    def test_fractional_values(self):
        """Test ResourceRequirements with fractional values."""
        req = ResourceRequirements(
            cpu_units=0.5,
            memory_mb=128.5,
            io_weight=1.25,
            network_weight=0.75,
            gpu_units=0.25,
            timeout=30.5,
        )

        assert req.cpu_units == 0.5
        assert req.memory_mb == 128.5
        assert req.io_weight == 1.25
        assert req.network_weight == 0.75
        assert req.gpu_units == 0.25
        assert req.timeout == 30.5

    def test_special_float_values(self):
        """Test ResourceRequirements with special float values."""
        import math

        # Test with infinity (should be allowed if no validation)
        try:
            req = ResourceRequirements(cpu_units=float("inf"))
            assert math.isinf(req.cpu_units)
        except (ValueError, OverflowError):
            # Expected if validation rejects infinite values
            pass

        # Test with NaN (should be rejected or handled)
        try:
            req = ResourceRequirements(cpu_units=float("nan"))
            assert math.isnan(req.cpu_units)
        except (ValueError, TypeError):
            # Expected if validation rejects NaN values
            pass

    def test_equality(self):
        """Test equality comparison of ResourceRequirements."""
        req1 = ResourceRequirements(cpu_units=2.0, memory_mb=512.0)
        req2 = ResourceRequirements(cpu_units=2.0, memory_mb=512.0)
        req3 = ResourceRequirements(cpu_units=3.0, memory_mb=512.0)
        req4 = ResourceRequirements(cpu_units=2.0, memory_mb=512.0, priority_boost=1)

        assert req1 == req2
        assert req1 != req3
        assert req1 != req4
        assert req2 != req3

    def test_inequality_operators(self):
        """Test inequality operators if implemented."""
        req1 = ResourceRequirements(cpu_units=1.0)
        req2 = ResourceRequirements(cpu_units=2.0)

        # These may not be implemented, so test conditionally
        try:
            assert req1 != req2
            assert req1 != req2
        except TypeError:
            # Expected if comparison operators aren't implemented
            pass

    def test_hash_consistency(self):
        """Test that ResourceRequirements handles hashing appropriately."""
        req1 = ResourceRequirements(cpu_units=2.0, memory_mb=512.0)

        # ResourceRequirements should not be hashable (it's a mutable dataclass)
        with pytest.raises(TypeError, match="unhashable type"):
            hash(req1)

    def test_copy_behavior(self):
        """Test copying behavior of ResourceRequirements."""
        import copy

        req1 = ResourceRequirements(
            cpu_units=2.0,
            memory_mb=512.0,
            resource_types=ResourceType.CPU | ResourceType.MEMORY,
        )

        # Shallow copy
        req2 = copy.copy(req1)
        assert req1 == req2
        assert req1 is not req2

        # Deep copy
        req3 = copy.deepcopy(req1)
        assert req1 == req3
        assert req1 is not req3

        # Modify original and verify copies are independent
        req1.cpu_units = 4.0
        assert (
            req2.cpu_units == 2.0
        )  # Shallow copy should be independent for primitive values
        assert req3.cpu_units == 2.0  # Deep copy should be independent


class TestIntegrationWithPriority:
    """Test integration with Priority enum."""

    def test_all_priority_levels(self):
        """Test all Priority enum levels."""
        for priority in Priority:
            req = ResourceRequirements()
            req.priority = priority
            assert req.priority == priority
            assert req.priority_boost == priority.value

    def test_priority_enum_values(self):
        """Test Priority enum values are as expected."""
        expected_values = {
            Priority.LOW: 0,
            Priority.NORMAL: 1,
            Priority.HIGH: 2,
            Priority.CRITICAL: 3,
        }

        for priority, expected_value in expected_values.items():
            assert priority.value == expected_value

    def test_priority_inheritance_patterns(self):
        """Test priority behavior with inheritance patterns."""
        base_req = ResourceRequirements(priority_boost=1)
        assert base_req.priority == Priority.NORMAL

        # Create new requirement based on base
        derived_req = ResourceRequirements(
            cpu_units=base_req.cpu_units,
            memory_mb=base_req.memory_mb,
            priority_boost=base_req.priority_boost + 1,
        )
        assert derived_req.priority == Priority.HIGH

    def test_priority_edge_cases(self):
        """Test priority edge cases."""
        req = ResourceRequirements()

        # Test boundary values
        boundary_tests = [
            (-1, Priority.LOW),  # Below minimum
            (0, Priority.LOW),  # Minimum
            (3, Priority.CRITICAL),  # Maximum defined
            (4, Priority.CRITICAL),  # Above maximum
            (1000, Priority.CRITICAL),  # Very high value
        ]

        for boost_value, expected_priority in boundary_tests:
            req.priority_boost = boost_value
            assert req.priority == expected_priority


class TestResourceTypeCombinations:
    """Test complex ResourceType combinations."""

    def test_compute_resources_only(self):
        """Test compute-only resource types."""
        compute_types = ResourceType.CPU | ResourceType.MEMORY | ResourceType.GPU
        req = ResourceRequirements(resource_types=compute_types)

        # Should include
        for rt in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU]:
            assert rt in req.resource_types

        # Should exclude
        for rt in [ResourceType.IO, ResourceType.NETWORK]:
            assert rt not in req.resource_types

    def test_io_resources_only(self):
        """Test IO-only resource types."""
        io_types = ResourceType.IO | ResourceType.NETWORK
        req = ResourceRequirements(resource_types=io_types)

        # Should include
        for rt in [ResourceType.IO, ResourceType.NETWORK]:
            assert rt in req.resource_types

        # Should exclude
        for rt in [ResourceType.CPU, ResourceType.MEMORY, ResourceType.GPU]:
            assert rt not in req.resource_types

    def test_single_resource_types(self):
        """Test each resource type individually."""
        all_types = [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]

        for target_type in all_types:
            req = ResourceRequirements(resource_types=target_type)
            assert target_type in req.resource_types

            # Verify no other types are included
            for other_type in all_types:
                if other_type != target_type:
                    assert other_type not in req.resource_types

    def test_resource_type_arithmetic(self):
        """Test arithmetic operations on resource types."""
        # Union
        combined = ResourceType.CPU | ResourceType.MEMORY
        req = ResourceRequirements(resource_types=combined)
        assert ResourceType.CPU in req.resource_types
        assert ResourceType.MEMORY in req.resource_types

        # Intersection
        all_types = ResourceType.ALL
        cpu_mem = ResourceType.CPU | ResourceType.MEMORY
        intersection = all_types & cpu_mem
        req2 = ResourceRequirements(resource_types=intersection)
        assert req2.resource_types == cpu_mem

        # Difference
        no_gpu = ResourceType.ALL & ~ResourceType.GPU
        req3 = ResourceRequirements(resource_types=no_gpu)
        assert ResourceType.GPU not in req3.resource_types
        for rt in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
        ]:
            assert rt in req3.resource_types

    def test_resource_type_complex_combinations(self):
        """Test complex resource type combinations."""
        # All except memory and GPU
        complex_combo = ResourceType.ALL & ~(ResourceType.MEMORY | ResourceType.GPU)
        req = ResourceRequirements(resource_types=complex_combo)

        # Should include
        for rt in [ResourceType.CPU, ResourceType.IO, ResourceType.NETWORK]:
            assert rt in req.resource_types

        # Should exclude
        for rt in [ResourceType.MEMORY, ResourceType.GPU]:
            assert rt not in req.resource_types


class TestSerialization:
    """Test serialization and representation methods."""

    def test_string_representation(self):
        """Test string representation of ResourceRequirements."""
        req = ResourceRequirements(cpu_units=2.0, memory_mb=512.0, priority_boost=1)

        str_repr = str(req)
        assert "ResourceRequirements" in str_repr
        assert "cpu_units" in str_repr or "2.0" in str_repr

        repr_str = repr(req)
        assert "ResourceRequirements" in repr_str

    def test_dataclass_asdict(self):
        """Test converting ResourceRequirements to dictionary."""
        req = ResourceRequirements(
            cpu_units=2.0,
            memory_mb=512.0,
            io_weight=3.0,
            priority_boost=1,
            timeout=30.0,
            resource_types=ResourceType.CPU | ResourceType.MEMORY,
        )

        req_dict = asdict(req)

        assert req_dict["cpu_units"] == 2.0
        assert req_dict["memory_mb"] == 512.0
        assert req_dict["io_weight"] == 3.0
        assert req_dict["priority_boost"] == 1
        assert req_dict["timeout"] == 30.0
        assert req_dict["resource_types"] == ResourceType.CPU | ResourceType.MEMORY

    def test_recreation_from_dict(self):
        """Test recreating ResourceRequirements from dictionary."""
        original = ResourceRequirements(
            cpu_units=2.0,
            memory_mb=512.0,
            priority_boost=2,
            resource_types=ResourceType.CPU | ResourceType.MEMORY,
        )

        # Convert to dict and back
        req_dict = asdict(original)
        recreated = ResourceRequirements(**req_dict)

        assert recreated == original


class TestValidation:
    """Test validation scenarios."""

    def test_resource_requirements_with_get_resource_amount(self):
        """Test ResourceRequirements integration with get_resource_amount."""
        req = ResourceRequirements(
            cpu_units=2.0,
            memory_mb=512.0,
            io_weight=3.0,
            network_weight=2.5,
            gpu_units=1.0,
        )

        # Test that get_resource_amount works correctly
        assert get_resource_amount(req, ResourceType.CPU) == req.cpu_units
        assert get_resource_amount(req, ResourceType.MEMORY) == req.memory_mb
        assert get_resource_amount(req, ResourceType.IO) == req.io_weight
        assert get_resource_amount(req, ResourceType.NETWORK) == req.network_weight
        assert get_resource_amount(req, ResourceType.GPU) == req.gpu_units

    def test_validation_with_resource_pool_patterns(self):
        """Test patterns that would be used by ResourcePool."""
        req = ResourceRequirements(
            cpu_units=2.0, memory_mb=1024.0, gpu_units=0.5, priority_boost=2
        )

        # Test patterns that ResourcePool might use
        total_resources = 0.0
        for rt in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            amount = get_resource_amount(req, rt)
            assert amount >= 0.0  # Should be non-negative for typical use
            total_resources += amount

        assert total_resources > 0.0  # Should have some resource requirements

    def test_resource_type_consistency(self):
        """Test consistency between resource_types and actual resource amounts."""
        req = ResourceRequirements(
            cpu_units=2.0,
            memory_mb=0.0,  # No memory requested
            io_weight=0.0,  # No IO requested
            network_weight=0.0,  # No network requested
            gpu_units=1.0,
            resource_types=ResourceType.CPU | ResourceType.GPU,  # Only CPU and GPU
        )

        # Verify that resource_types matches what's actually requested
        assert ResourceType.CPU in req.resource_types
        assert ResourceType.GPU in req.resource_types
        assert get_resource_amount(req, ResourceType.CPU) > 0
        assert get_resource_amount(req, ResourceType.GPU) > 0

        # These should be zero
        assert get_resource_amount(req, ResourceType.MEMORY) == 0
        assert get_resource_amount(req, ResourceType.IO) == 0
        assert get_resource_amount(req, ResourceType.NETWORK) == 0


class TestConcurrency:
    """Test thread safety and concurrent access patterns."""

    def test_concurrent_priority_modifications(self):
        """Test concurrent modifications to priority don't cause issues."""
        req = ResourceRequirements()
        results = []
        errors = []

        def modify_priority(priority_val):
            try:
                req.priority_boost = priority_val
                results.append(req.priority)
            except Exception as e:
                errors.append(e)

        # Create multiple threads modifying priority
        threads = []
        for i in range(10):
            thread = threading.Thread(target=modify_priority, args=(i % 4,))
            threads.append(thread)

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors occurred
        assert len(errors) == 0
        assert len(results) == 10

    def test_concurrent_resource_modifications(self):
        """Test concurrent modifications to resource amounts."""
        req = ResourceRequirements()
        modifications = []

        def modify_resources(multiplier):
            try:
                req.cpu_units = 1.0 * multiplier
                req.memory_mb = 100.0 * multiplier
                modifications.append(multiplier)
            except Exception as e:
                modifications.append(f"Error: {e}")

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=modify_resources, args=(i + 1,))
            threads.append(thread)

        # Execute concurrently
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all modifications completed
        assert len(modifications) == 5
        assert all(isinstance(m, (int, float)) for m in modifications)


class TestPerformance:
    """Test performance characteristics."""

    def test_creation_performance(self):
        """Test ResourceRequirements creation performance."""
        import time

        start_time = time.time()

        # Create many instances
        requirements = []
        for i in range(1000):
            req = ResourceRequirements(
                cpu_units=i * 0.1, memory_mb=i * 10, priority_boost=i % 4
            )
            requirements.append(req)

        end_time = time.time()
        creation_time = end_time - start_time

        # Should be fast (less than 1 second for 1000 creations)
        assert creation_time < 1.0
        assert len(requirements) == 1000

    def test_property_access_performance(self):
        """Test property access performance."""
        req = ResourceRequirements(cpu_units=2.0, priority_boost=2)

        start_time = time.time()

        # Access properties many times
        for _ in range(10000):
            _ = req.priority
            _ = req.cpu_units
            _ = req.resource_types

        end_time = time.time()
        access_time = end_time - start_time

        # Property access should be very fast
        assert access_time < 0.1

    def test_get_resource_amount_performance(self):
        """Test get_resource_amount function performance."""
        req = ResourceRequirements(
            cpu_units=2.0,
            memory_mb=512.0,
            io_weight=3.0,
            network_weight=2.5,
            gpu_units=1.0,
        )

        start_time = time.time()

        # Call get_resource_amount many times
        for _ in range(10000):
            for rt in [
                ResourceType.CPU,
                ResourceType.MEMORY,
                ResourceType.IO,
                ResourceType.NETWORK,
                ResourceType.GPU,
            ]:
                _ = get_resource_amount(req, rt)

        end_time = time.time()
        function_time = end_time - start_time

        # Function calls should be reasonably fast
        assert function_time < 1.0


class TestPostInitBehavior:
    """Test __post_init__ method edge cases and error handling."""

    def test_invalid_resource_types_fallback(self):
        """Test fallback behavior when resource_types is invalid."""
        import logging
        from unittest.mock import patch

        # Test with non-ResourceType value
        with patch.object(
            logging.getLogger("puffinflow.core.resources.requirements"), "warning"
        ) as mock_warn:
            # Create instance with invalid resource_types via manual setting
            req = ResourceRequirements()
            # Force invalid resource_types to trigger auto-determination
            object.__setattr__(req, "resource_types", "invalid_string")
            req.__post_init__()
            mock_warn.assert_called()

    def test_invalid_resource_types_triggers_auto_determination(self):
        """Test that invalid resource_types triggers auto-determination."""
        req = ResourceRequirements(cpu_units=2.0, memory_mb=0.0)
        # Force invalid resource_types to trigger auto-determination
        object.__setattr__(req, "resource_types", "invalid_string")
        req.__post_init__()

        # Should have been auto-determined based on non-zero values
        assert ResourceType.CPU in req.resource_types

    def test_auto_determine_resource_types_all_zero(self):
        """Test auto-determination when all resource amounts are zero."""
        req = ResourceRequirements(
            cpu_units=0.0,
            memory_mb=0.0,
            io_weight=0.0,
            network_weight=0.0,
            gpu_units=0.0,
        )
        # Force auto-determination
        req._auto_determine_resource_types()

        # Should default to ALL when no resources are requested
        assert req.resource_types == ResourceType.ALL

    def test_auto_determine_resource_types_partial(self):
        """Test auto-determination with partial resource amounts."""
        req = ResourceRequirements(
            cpu_units=2.0,
            memory_mb=0.0,
            io_weight=0.0,
            network_weight=1.0,
            gpu_units=0.0,
        )
        req._auto_determine_resource_types()

        # Should include only CPU and NETWORK
        assert ResourceType.CPU in req.resource_types
        assert ResourceType.NETWORK in req.resource_types
        assert ResourceType.MEMORY not in req.resource_types
        assert ResourceType.IO not in req.resource_types
        assert ResourceType.GPU not in req.resource_types

    def test_safe_check_resource_type_fallback(self):
        """Test safe_check_resource_type fallback mechanisms."""
        import logging
        from unittest.mock import Mock, patch

        from puffinflow.core.resources.requirements import safe_check_resource_type

        req = ResourceRequirements(cpu_units=2.0)

        # Test TypeError fallback
        with patch.object(
            logging.getLogger("puffinflow.core.resources.requirements"), "warning"
        ) as mock_warn:
            mock_resource_types = Mock()
            mock_resource_types.__and__ = Mock(side_effect=TypeError("Bitwise failed"))
            mock_resource_types.value = 1
            req.resource_types = mock_resource_types

            result = safe_check_resource_type(req, ResourceType.CPU)
            mock_warn.assert_called()
            assert isinstance(result, bool)

    def test_safe_check_resource_type_double_fallback(self):
        """Test safe_check_resource_type double fallback to attribute check."""
        import logging
        from unittest.mock import Mock, patch

        from puffinflow.core.resources.requirements import safe_check_resource_type

        req = ResourceRequirements(cpu_units=2.0)

        # Test double fallback when both bitwise operations fail
        with patch.object(
            logging.getLogger("puffinflow.core.resources.requirements"), "error"
        ) as mock_error:
            mock_resource_types = Mock()
            mock_resource_types.__and__ = Mock(side_effect=TypeError("Bitwise failed"))
            mock_resource_types.value = Mock()
            mock_resource_types.value.__and__ = Mock(
                side_effect=Exception("Value fallback failed")
            )
            req.resource_types = mock_resource_types

            result = safe_check_resource_type(req, ResourceType.CPU)
            mock_error.assert_called()
            assert result is True  # Should fallback to checking cpu_units > 0

    def test_get_resource_amount_disabled_resource(self):
        """Test get_resource_amount when resource type is disabled."""
        from puffinflow.core.resources.requirements import get_resource_amount

        req = ResourceRequirements(
            cpu_units=2.0,
            memory_mb=512.0,
            resource_types=ResourceType.CPU,  # Only CPU enabled
        )

        # Memory should return 0 even though memory_mb is set
        result = get_resource_amount(req, ResourceType.MEMORY)
        assert result == 0.0

    def test_get_resource_amount_combined_type_fallback(self):
        """Test get_resource_amount with combined types using fallback logic."""
        from unittest.mock import patch

        from puffinflow.core.resources.requirements import get_resource_amount

        req = ResourceRequirements(cpu_units=2.0, memory_mb=512.0)
        combined_type = ResourceType.CPU | ResourceType.MEMORY

        # Mock the 'in' operator to fail, forcing value-based fallback
        with patch.object(
            ResourceType, "__contains__", side_effect=TypeError("Contains failed")
        ):
            result = get_resource_amount(req, combined_type)
            # Should still work using value-based comparison
            assert result > 0


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_machine_learning_workload_scenario(self):
        """Test scenario resembling ML workload requirements."""
        ml_req = ResourceRequirements(
            cpu_units=4.0,  # Multi-core processing
            memory_mb=8192.0,  # Large memory for datasets
            gpu_units=2.0,  # GPU acceleration
            io_weight=5.0,  # Heavy disk I/O for data loading
            network_weight=3.0,  # Network for distributed training
            priority_boost=3,  # Critical priority
            timeout=7200.0,  # 2 hour timeout
            resource_types=ResourceType.ALL,  # Needs all resource types
        )

        # Verify requirements make sense
        assert ml_req.cpu_units >= 1.0
        assert ml_req.memory_mb >= 1000.0
        assert ml_req.gpu_units > 0.0
        assert ml_req.priority == Priority.CRITICAL
        assert ml_req.timeout == 7200.0

        # Verify all resource types are requested
        for rt in [
            ResourceType.CPU,
            ResourceType.MEMORY,
            ResourceType.IO,
            ResourceType.NETWORK,
            ResourceType.GPU,
        ]:
            assert rt in ml_req.resource_types
            assert get_resource_amount(ml_req, rt) > 0.0

    def test_web_service_scenario(self):
        """Test scenario resembling web service requirements."""
        web_req = ResourceRequirements(
            cpu_units=1.0,  # Moderate CPU
            memory_mb=512.0,  # Moderate memory
            gpu_units=0.0,  # No GPU needed
            io_weight=2.0,  # Some disk I/O
            network_weight=5.0,  # Heavy network usage
            priority_boost=1,  # Normal priority
            timeout=30.0,  # Quick timeout
            resource_types=ResourceType.ALL & ~ResourceType.GPU,  # All except GPU
        )

        assert web_req.priority == Priority.NORMAL
        assert web_req.gpu_units == 0.0
        assert ResourceType.GPU not in web_req.resource_types
        assert ResourceType.NETWORK in web_req.resource_types
        assert get_resource_amount(web_req, ResourceType.NETWORK) > 0.0

    def test_batch_processing_scenario(self):
        """Test scenario resembling batch processing requirements."""
        batch_req = ResourceRequirements(
            cpu_units=8.0,  # High CPU for parallel processing
            memory_mb=4096.0,  # Large memory for batch data
            gpu_units=0.0,  # CPU-only processing
            io_weight=10.0,  # Very heavy I/O
            network_weight=1.0,  # Minimal network
            priority_boost=0,  # Low priority (can wait)
            timeout=None,  # No timeout (can run indefinitely)
            resource_types=ResourceType.CPU | ResourceType.MEMORY | ResourceType.IO,
        )

        assert batch_req.priority == Priority.LOW
        assert batch_req.timeout is None
        assert ResourceType.GPU not in batch_req.resource_types
        assert ResourceType.NETWORK not in batch_req.resource_types
        assert get_resource_amount(batch_req, ResourceType.IO) == 10.0

    def test_requirements_scaling(self):
        """Test scaling requirements up and down."""
        base_req = ResourceRequirements(
            cpu_units=1.0,
            memory_mb=100.0,
            io_weight=1.0,
            network_weight=1.0,
            gpu_units=0.0,
        )

        # Scale up by factor of 4
        scale_factor = 4.0
        scaled_req = ResourceRequirements(
            cpu_units=base_req.cpu_units * scale_factor,
            memory_mb=base_req.memory_mb * scale_factor,
            io_weight=base_req.io_weight * scale_factor,
            network_weight=base_req.network_weight * scale_factor,
            gpu_units=base_req.gpu_units * scale_factor,
            priority_boost=base_req.priority_boost,
            timeout=base_req.timeout,
            resource_types=base_req.resource_types,
        )

        assert scaled_req.cpu_units == 4.0
        assert scaled_req.memory_mb == 400.0
        assert scaled_req.io_weight == 4.0
        assert scaled_req.network_weight == 4.0
        assert scaled_req.gpu_units == 0.0  # 0 * 4 = 0

    def test_requirements_merging(self):
        """Test merging multiple requirements."""
        req1 = ResourceRequirements(cpu_units=2.0, memory_mb=512.0, priority_boost=1)

        req2 = ResourceRequirements(
            cpu_units=1.0, memory_mb=256.0, gpu_units=1.0, priority_boost=2
        )

        # Merge by taking maximum of each resource
        merged_req = ResourceRequirements(
            cpu_units=max(req1.cpu_units, req2.cpu_units),
            memory_mb=max(req1.memory_mb, req2.memory_mb),
            io_weight=max(req1.io_weight, req2.io_weight),
            network_weight=max(req1.network_weight, req2.network_weight),
            gpu_units=max(req1.gpu_units, req2.gpu_units),
            priority_boost=max(req1.priority_boost, req2.priority_boost),
            # Combine resource types
            resource_types=req1.resource_types | req2.resource_types,
        )

        assert merged_req.cpu_units == 2.0  # max(2.0, 1.0)
        assert merged_req.memory_mb == 512.0  # max(512.0, 256.0)
        assert merged_req.gpu_units == 1.0  # max(0.0, 1.0)
        assert merged_req.priority == Priority.HIGH  # priority_boost=2


if __name__ == "__main__":
    # Run tests with pytest - install pytest-cov for coverage if needed
    # pip install pytest pytest-cov
    pytest.main([__file__, "-v", "--tb=short"])
