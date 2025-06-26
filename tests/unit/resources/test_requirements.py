"""
Comprehensive unit tests for ResourceRequirements and ResourceType.

Tests cover:
- ResourceType enum functionality and flag operations
- ResourceRequirements initialization and defaults
- Priority property getter/setter logic
- Resource type combinations and validations
- Edge cases and attribute modifications
- Integration with Priority enum
"""

import pytest
from unittest.mock import patch

# Import the classes under test
from src.puffinflow.core.resources.requirements import ResourceRequirements, ResourceType
from src.puffinflow.core.agent.state import Priority


class TestResourceType:
    """Test ResourceType enum functionality."""

    def test_resource_type_values(self):
        """Test ResourceType enum values."""
        assert ResourceType.NONE.value == 0
        assert ResourceType.CPU.value == 1
        assert ResourceType.MEMORY.value == 2
        assert ResourceType.IO.value == 4
        assert ResourceType.NETWORK.value == 8
        assert ResourceType.GPU.value == 16

    def test_resource_type_all_combination(self):
        """Test ResourceType.ALL includes all resource types."""
        expected = ResourceType.CPU | ResourceType.MEMORY | ResourceType.IO | ResourceType.NETWORK | ResourceType.GPU
        assert ResourceType.ALL == expected
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
        assert ResourceType.CPU in ResourceType.ALL
        assert ResourceType.MEMORY in ResourceType.ALL
        assert ResourceType.IO in ResourceType.ALL
        assert ResourceType.NETWORK in ResourceType.ALL
        assert ResourceType.GPU in ResourceType.ALL

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
        assert ResourceType.CPU not in ResourceType.NONE
        assert ResourceType.MEMORY not in ResourceType.NONE

        # NONE with other types
        cpu_with_none = ResourceType.CPU | ResourceType.NONE
        assert cpu_with_none == ResourceType.CPU


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
            resource_types=ResourceType.CPU | ResourceType.MEMORY
        )

        assert req.cpu_units == 2.5
        assert req.memory_mb == 512.0
        assert req.io_weight == 3.0
        assert req.network_weight == 2.5
        assert req.gpu_units == 1.0
        assert req.priority_boost == 2
        assert req.timeout == 30.0
        assert req.resource_types == ResourceType.CPU | ResourceType.MEMORY

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
            priority_boost=0
        )

        assert req.cpu_units == 0.0
        assert req.memory_mb == 0.0
        assert req.io_weight == 0.0
        assert req.network_weight == 0.0
        assert req.gpu_units == 0.0
        assert req.priority_boost == 0


class TestPriorityProperty:
    """Test priority property getter/setter logic."""

    def test_priority_getter_default(self):
        """Test priority property getter with default priority_boost."""
        req = ResourceRequirements()  # priority_boost=0 by default
        assert req.priority == Priority.LOW

    def test_priority_getter_mapping(self):
        """Test priority property getter with different priority_boost values."""
        # LOW (0)
        req_low = ResourceRequirements(priority_boost=0)
        assert req_low.priority == Priority.LOW

        # NORMAL (1)
        req_normal = ResourceRequirements(priority_boost=1)
        assert req_normal.priority == Priority.NORMAL

        # HIGH (2)
        req_high = ResourceRequirements(priority_boost=2)
        assert req_high.priority == Priority.HIGH

        # CRITICAL (3+)
        req_critical = ResourceRequirements(priority_boost=3)
        assert req_critical.priority == Priority.CRITICAL

        req_critical_high = ResourceRequirements(priority_boost=5)
        assert req_critical_high.priority == Priority.CRITICAL

    def test_priority_setter_with_enum(self):
        """Test priority property setter with Priority enum values."""
        req = ResourceRequirements()

        # Set to LOW
        req.priority = Priority.LOW
        assert req.priority_boost == Priority.LOW.value
        assert req.priority == Priority.LOW

        # Set to NORMAL
        req.priority = Priority.NORMAL
        assert req.priority_boost == Priority.NORMAL.value
        assert req.priority == Priority.NORMAL

        # Set to HIGH
        req.priority = Priority.HIGH
        assert req.priority_boost == Priority.HIGH.value
        assert req.priority == Priority.HIGH

        # Set to CRITICAL
        req.priority = Priority.CRITICAL
        assert req.priority_boost == Priority.CRITICAL.value
        assert req.priority == Priority.CRITICAL

    def test_priority_setter_with_int(self):
        """Test priority property setter with integer values."""
        req = ResourceRequirements()

        # Set with integer
        req.priority = 2
        assert req.priority_boost == 2
        assert req.priority == Priority.HIGH

        req.priority = 0
        assert req.priority_boost == 0
        assert req.priority == Priority.LOW

    def test_priority_roundtrip(self):
        """Test priority getter/setter roundtrip."""
        req = ResourceRequirements()

        for priority in [Priority.LOW, Priority.NORMAL, Priority.HIGH, Priority.CRITICAL]:
            req.priority = priority
            assert req.priority == priority
            assert req.priority_boost == priority.value


class TestResourceTypesHandling:
    """Test resource_types field functionality."""

    def test_default_resource_types(self):
        """Test default resource_types value."""
        req = ResourceRequirements()
        assert req.resource_types == ResourceType.ALL

        # All types should be included
        assert ResourceType.CPU in req.resource_types
        assert ResourceType.MEMORY in req.resource_types
        assert ResourceType.IO in req.resource_types
        assert ResourceType.NETWORK in req.resource_types
        assert ResourceType.GPU in req.resource_types

    def test_custom_resource_types(self):
        """Test custom resource_types combinations."""
        # CPU only
        req_cpu = ResourceRequirements(resource_types=ResourceType.CPU)
        assert ResourceType.CPU in req_cpu.resource_types
        assert ResourceType.MEMORY not in req_cpu.resource_types

        # CPU + Memory
        req_cpu_mem = ResourceRequirements(
            resource_types=ResourceType.CPU | ResourceType.MEMORY
        )
        assert ResourceType.CPU in req_cpu_mem.resource_types
        assert ResourceType.MEMORY in req_cpu_mem.resource_types
        assert ResourceType.IO not in req_cpu_mem.resource_types

        # All except GPU
        req_no_gpu = ResourceRequirements(
            resource_types=ResourceType.ALL & ~ResourceType.GPU
        )
        assert ResourceType.CPU in req_no_gpu.resource_types
        assert ResourceType.MEMORY in req_no_gpu.resource_types
        assert ResourceType.IO in req_no_gpu.resource_types
        assert ResourceType.NETWORK in req_no_gpu.resource_types
        assert ResourceType.GPU not in req_no_gpu.resource_types

    def test_none_resource_types(self):
        """Test ResourceType.NONE behavior."""
        req = ResourceRequirements(resource_types=ResourceType.NONE)
        assert req.resource_types == ResourceType.NONE

        # No types should be included
        assert ResourceType.CPU not in req.resource_types
        assert ResourceType.MEMORY not in req.resource_types
        assert ResourceType.IO not in req.resource_types
        assert ResourceType.NETWORK not in req.resource_types
        assert ResourceType.GPU not in req.resource_types


class TestAttributeModification:
    """Test modification of ResourceRequirements attributes."""

    def test_modify_resource_amounts(self):
        """Test modifying resource amount attributes."""
        req = ResourceRequirements()

        # Modify CPU
        req.cpu_units = 4.0
        assert req.cpu_units == 4.0

        # Modify memory
        req.memory_mb = 2048.0
        assert req.memory_mb == 2048.0

        # Modify IO weight
        req.io_weight = 5.0
        assert req.io_weight == 5.0

        # Modify network weight
        req.network_weight = 3.5
        assert req.network_weight == 3.5

        # Modify GPU
        req.gpu_units = 2.0
        assert req.gpu_units == 2.0

    def test_modify_priority_boost(self):
        """Test modifying priority_boost attribute."""
        req = ResourceRequirements()

        req.priority_boost = 3
        assert req.priority_boost == 3
        assert req.priority == Priority.CRITICAL

        req.priority_boost = 1
        assert req.priority_boost == 1
        assert req.priority == Priority.NORMAL

    def test_modify_timeout(self):
        """Test modifying timeout attribute."""
        req = ResourceRequirements()

        assert req.timeout is None

        req.timeout = 60.0
        assert req.timeout == 60.0

        req.timeout = None
        assert req.timeout is None

    def test_modify_resource_types(self):
        """Test modifying resource_types attribute."""
        req = ResourceRequirements()

        # Change to CPU only
        req.resource_types = ResourceType.CPU
        assert req.resource_types == ResourceType.CPU

        # Change to CPU + Memory
        req.resource_types = ResourceType.CPU | ResourceType.MEMORY
        assert ResourceType.CPU in req.resource_types
        assert ResourceType.MEMORY in req.resource_types
        assert ResourceType.IO not in req.resource_types


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_negative_values(self):
        """Test ResourceRequirements with negative values."""
        # Should be allowed (validation happens elsewhere)
        req = ResourceRequirements(
            cpu_units=-1.0,
            memory_mb=-100.0,
            io_weight=-2.0,
            network_weight=-1.5,
            gpu_units=-0.5
        )

        assert req.cpu_units == -1.0
        assert req.memory_mb == -100.0
        assert req.io_weight == -2.0
        assert req.network_weight == -1.5
        assert req.gpu_units == -0.5

    def test_very_large_values(self):
        """Test ResourceRequirements with very large values."""
        req = ResourceRequirements(
            cpu_units=1000000.0,
            memory_mb=1e9,
            io_weight=999999.9,
            network_weight=1e6,
            gpu_units=100.0,
            priority_boost=1000
        )

        assert req.cpu_units == 1000000.0
        assert req.memory_mb == 1e9
        assert req.io_weight == 999999.9
        assert req.network_weight == 1e6
        assert req.gpu_units == 100.0
        assert req.priority_boost == 1000
        assert req.priority == Priority.CRITICAL  # Should still map to CRITICAL

    def test_fractional_values(self):
        """Test ResourceRequirements with fractional values."""
        req = ResourceRequirements(
            cpu_units=0.5,
            memory_mb=128.5,
            io_weight=1.25,
            network_weight=0.75,
            gpu_units=0.25
        )

        assert req.cpu_units == 0.5
        assert req.memory_mb == 128.5
        assert req.io_weight == 1.25
        assert req.network_weight == 0.75
        assert req.gpu_units == 0.25

    def test_extreme_priority_boost(self):
        """Test priority property with extreme priority_boost values."""
        # Very negative
        req_neg = ResourceRequirements(priority_boost=-100)
        assert req_neg.priority == Priority.LOW

        # Very positive
        req_pos = ResourceRequirements(priority_boost=100)
        assert req_pos.priority == Priority.CRITICAL

    def test_equality(self):
        """Test equality comparison of ResourceRequirements."""
        req1 = ResourceRequirements(cpu_units=2.0, memory_mb=512.0)
        req2 = ResourceRequirements(cpu_units=2.0, memory_mb=512.0)
        req3 = ResourceRequirements(cpu_units=3.0, memory_mb=512.0)

        assert req1 == req2
        assert req1 != req3

    def test_hash_consistency(self):
        """Test that ResourceRequirements is not hashable by default."""
        req1 = ResourceRequirements(cpu_units=2.0, memory_mb=512.0)

        # ResourceRequirements should not be hashable (it's a mutable dataclass)
        with pytest.raises(TypeError, match="unhashable type"):
            hash(req1)


class TestIntegrationWithPriority:
    """Test integration with Priority enum."""

    def test_all_priority_levels(self):
        """Test all Priority enum levels."""
        # Test each priority level
        for priority in Priority:
            req = ResourceRequirements()
            req.priority = priority
            assert req.priority == priority
            assert req.priority_boost == priority.value

    def test_priority_enum_values(self):
        """Test Priority enum values are as expected."""
        # This ensures our mapping logic is correct
        assert Priority.LOW.value == 0
        assert Priority.NORMAL.value == 1
        assert Priority.HIGH.value == 2
        assert Priority.CRITICAL.value == 3

    def test_priority_inheritance(self):
        """Test priority behavior with inheritance patterns."""
        base_req = ResourceRequirements(priority_boost=1)
        assert base_req.priority == Priority.NORMAL

        # Create new requirement based on base
        derived_req = ResourceRequirements(
            cpu_units=base_req.cpu_units,
            memory_mb=base_req.memory_mb,
            priority_boost=base_req.priority_boost + 1
        )
        assert derived_req.priority == Priority.HIGH


class TestResourceTypeCombinations:
    """Test complex ResourceType combinations."""

    def test_compute_resources_only(self):
        """Test compute-only resource types."""
        compute_types = ResourceType.CPU | ResourceType.MEMORY | ResourceType.GPU
        req = ResourceRequirements(resource_types=compute_types)

        assert ResourceType.CPU in req.resource_types
        assert ResourceType.MEMORY in req.resource_types
        assert ResourceType.GPU in req.resource_types
        assert ResourceType.IO not in req.resource_types
        assert ResourceType.NETWORK not in req.resource_types

    def test_io_resources_only(self):
        """Test IO-only resource types."""
        io_types = ResourceType.IO | ResourceType.NETWORK
        req = ResourceRequirements(resource_types=io_types)

        assert ResourceType.IO in req.resource_types
        assert ResourceType.NETWORK in req.resource_types
        assert ResourceType.CPU not in req.resource_types
        assert ResourceType.MEMORY not in req.resource_types
        assert ResourceType.GPU not in req.resource_types

    def test_single_resource_type(self):
        """Test each resource type individually."""
        for resource_type in [ResourceType.CPU, ResourceType.MEMORY,
                             ResourceType.IO, ResourceType.NETWORK, ResourceType.GPU]:
            req = ResourceRequirements(resource_types=resource_type)
            assert resource_type in req.resource_types

            # Verify no other types are included
            for other_type in [ResourceType.CPU, ResourceType.MEMORY,
                              ResourceType.IO, ResourceType.NETWORK, ResourceType.GPU]:
                if other_type != resource_type:
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
        assert ResourceType.CPU in req3.resource_types


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])