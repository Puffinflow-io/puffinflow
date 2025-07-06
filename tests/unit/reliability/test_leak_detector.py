"""
Comprehensive unit tests for leak_detector.py module.

Tests cover:
- Basic allocation/release tracking
- Leak detection with timing
- Metrics and reporting
- Edge cases and error conditions
- Data integrity and isolation
- Global instance functionality
"""

from dataclasses import asdict
from unittest.mock import patch

import pytest

# Import the module under test
from puffinflow.core.reliability.leak_detector import (
    ResourceAllocation,
    ResourceLeak,
    ResourceLeakDetector,
    leak_detector,
)


class TestResourceLeak:
    """Test ResourceLeak dataclass."""

    def test_resource_leak_creation(self):
        """Test ResourceLeak can be created with all required fields."""
        resources = {"cpu": 2.0, "memory": 1024}
        leak = ResourceLeak(
            state_name="test_state",
            agent_name="test_agent",
            resources=resources,
            allocated_at=1000.0,
            held_for_seconds=350.0,
            leak_threshold_seconds=300.0,
        )

        assert leak.state_name == "test_state"
        assert leak.agent_name == "test_agent"
        assert leak.resources == resources
        assert leak.allocated_at == 1000.0
        assert leak.held_for_seconds == 350.0
        assert leak.leak_threshold_seconds == 300.0

    def test_resource_leak_is_dataclass(self):
        """Test ResourceLeak is properly configured as dataclass."""
        resources = {"cpu": 1.0}
        leak = ResourceLeak(
            state_name="state1",
            agent_name="agent1",
            resources=resources,
            allocated_at=100.0,
            held_for_seconds=400.0,
            leak_threshold_seconds=300.0,
        )

        # Should be able to convert to dict
        leak_dict = asdict(leak)
        assert "state_name" in leak_dict
        assert "resources" in leak_dict


class TestResourceAllocation:
    """Test ResourceAllocation dataclass."""

    def test_resource_allocation_creation(self):
        """Test ResourceAllocation can be created with required fields."""
        resources = {"gpu": 1, "memory": 512}
        allocation = ResourceAllocation(
            state_name="compute_state",
            agent_name="worker_agent",
            resources=resources,
            allocated_at=2000.0,
        )

        assert allocation.state_name == "compute_state"
        assert allocation.agent_name == "worker_agent"
        assert allocation.resources == resources
        assert allocation.allocated_at == 2000.0


class TestResourceLeakDetector:
    """Test ResourceLeakDetector class functionality."""

    @pytest.fixture
    def detector(self):
        """Create a fresh ResourceLeakDetector for each test."""
        return ResourceLeakDetector(leak_threshold_seconds=300.0)

    @pytest.fixture
    def sample_resources(self):
        """Sample resource dictionary for testing."""
        return {"cpu": 2.0, "memory": 1024, "gpu": 1}

    def test_init_default_threshold(self):
        """Test detector initialization with default threshold."""
        detector = ResourceLeakDetector()
        assert detector.leak_threshold == 300.0
        assert detector.allocations == {}
        assert detector.detected_leaks == []
        assert detector._max_leaks_history == 100

    def test_init_custom_threshold(self):
        """Test detector initialization with custom threshold."""
        detector = ResourceLeakDetector(leak_threshold_seconds=600.0)
        assert detector.leak_threshold == 600.0
        assert detector.allocations == {}
        assert detector.detected_leaks == []

    def test_track_allocation_basic(self, detector, sample_resources):
        """Test basic resource allocation tracking."""
        with patch("time.time", return_value=1000.0):
            detector.track_allocation("state1", "agent1", sample_resources)

        key = "agent1:state1"
        assert key in detector.allocations

        allocation = detector.allocations[key]
        assert allocation.state_name == "state1"
        assert allocation.agent_name == "agent1"
        assert allocation.resources == sample_resources
        assert allocation.allocated_at == 1000.0

    def test_track_allocation_copies_resources(self, detector):
        """Test that resource dictionaries are copied, not referenced."""
        original_resources = {"cpu": 1.0}

        with patch("time.time", return_value=1000.0):
            detector.track_allocation("state1", "agent1", original_resources)

        # Modify original resources
        original_resources["cpu"] = 5.0

        # Stored resources should remain unchanged
        allocation = detector.allocations["agent1:state1"]
        assert allocation.resources["cpu"] == 1.0

    def test_track_allocation_multiple_states(self, detector, sample_resources):
        """Test tracking multiple allocations."""
        with patch("time.time", return_value=1000.0):
            detector.track_allocation("state1", "agent1", sample_resources)
            detector.track_allocation("state2", "agent1", {"memory": 512})
            detector.track_allocation("state1", "agent2", {"cpu": 1.0})

        assert len(detector.allocations) == 3
        assert "agent1:state1" in detector.allocations
        assert "agent1:state2" in detector.allocations
        assert "agent2:state1" in detector.allocations

    def test_track_allocation_overwrites_existing(self, detector, sample_resources):
        """Test that tracking same state/agent overwrites existing allocation."""
        # First allocation
        with patch("time.time", return_value=1000.0):
            detector.track_allocation("state1", "agent1", {"cpu": 1.0})

        # Second allocation for same state/agent
        with patch("time.time", return_value=2000.0):
            detector.track_allocation("state1", "agent1", sample_resources)

        assert len(detector.allocations) == 1
        allocation = detector.allocations["agent1:state1"]
        assert allocation.allocated_at == 2000.0
        assert allocation.resources == sample_resources

    def test_track_release_removes_allocation(self, detector, sample_resources):
        """Test that track_release removes the allocation."""
        with patch("time.time", return_value=1000.0):
            detector.track_allocation("state1", "agent1", sample_resources)

        assert "agent1:state1" in detector.allocations

        detector.track_release("state1", "agent1")
        assert "agent1:state1" not in detector.allocations

    def test_track_release_nonexistent_allocation(self, detector):
        """Test track_release with non-existent allocation does nothing."""
        # Should not raise an exception
        detector.track_release("nonexistent", "agent1")
        assert len(detector.allocations) == 0

    def test_track_release_partial(self, detector, sample_resources):
        """Test releasing one allocation while others remain."""
        with patch("time.time", return_value=1000.0):
            detector.track_allocation("state1", "agent1", sample_resources)
            detector.track_allocation("state2", "agent1", {"memory": 256})

        detector.track_release("state1", "agent1")

        assert "agent1:state1" not in detector.allocations
        assert "agent1:state2" in detector.allocations

    def test_detect_leaks_no_allocations(self, detector):
        """Test leak detection with no allocations."""
        leaks = detector.detect_leaks()
        assert leaks == []

    def test_detect_leaks_no_leaks(self, detector, sample_resources):
        """Test leak detection when allocations are within threshold."""
        current_time = 1000.0
        with patch("time.time", return_value=current_time):
            detector.track_allocation("state1", "agent1", sample_resources)

        # Check immediately (0 seconds held)
        with patch("time.time", return_value=current_time):
            leaks = detector.detect_leaks()
            assert leaks == []

        # Check just under threshold (299 seconds held)
        with patch("time.time", return_value=current_time + 299):
            leaks = detector.detect_leaks()
            assert leaks == []

    def test_detect_leaks_single_leak(self, detector, sample_resources):
        """Test leak detection with single leaked resource."""
        allocated_time = 1000.0
        current_time = allocated_time + 350.0  # 50 seconds over threshold

        with patch("time.time", return_value=allocated_time):
            detector.track_allocation("state1", "agent1", sample_resources)

        with patch("time.time", return_value=current_time):
            leaks = detector.detect_leaks()

        assert len(leaks) == 1
        leak = leaks[0]
        assert leak.state_name == "state1"
        assert leak.agent_name == "agent1"
        assert leak.resources == sample_resources
        assert leak.allocated_at == allocated_time
        assert leak.held_for_seconds == 350.0
        assert leak.leak_threshold_seconds == 300.0

    def test_detect_leaks_multiple_leaks(self, detector):
        """Test leak detection with multiple leaked resources."""
        allocated_time = 1000.0
        current_time = allocated_time + 400.0

        with patch("time.time", return_value=allocated_time):
            detector.track_allocation("state1", "agent1", {"cpu": 1.0})
            detector.track_allocation("state2", "agent1", {"memory": 512})
            detector.track_allocation("state3", "agent2", {"gpu": 1})

        with patch("time.time", return_value=current_time):
            leaks = detector.detect_leaks()

        assert len(leaks) == 3

        # Check all leaks have correct timing
        for leak in leaks:
            assert leak.held_for_seconds == 400.0
            assert leak.leak_threshold_seconds == 300.0

    def test_detect_leaks_mixed_leak_status(self, detector):
        """Test leak detection with mix of leaked and non-leaked resources."""
        allocated_time = 1000.0
        current_time = allocated_time + 350.0

        with patch("time.time", return_value=allocated_time):
            detector.track_allocation("leaked", "agent1", {"cpu": 1.0})

        # Add a more recent allocation that hasn't leaked
        with patch("time.time", return_value=allocated_time + 200):
            detector.track_allocation("recent", "agent1", {"memory": 256})

        with patch("time.time", return_value=current_time):
            leaks = detector.detect_leaks()

        assert len(leaks) == 1
        assert leaks[0].state_name == "leaked"

    def test_detect_leaks_builds_history(self, detector, sample_resources):
        """Test that detect_leaks builds leak history."""
        allocated_time = 1000.0

        with patch("time.time", return_value=allocated_time):
            detector.track_allocation("state1", "agent1", sample_resources)

        # First detection - adds to history
        with patch("time.time", return_value=allocated_time + 350):
            detector.detect_leaks()
            assert len(detector.detected_leaks) == 1

        # Second detection - should not duplicate
        with patch("time.time", return_value=allocated_time + 400):
            detector.detect_leaks()
            assert len(detector.detected_leaks) == 1  # Still only 1

    def test_detect_leaks_history_deduplication(self, detector):
        """Test that duplicate leaks are not added to history."""
        allocated_time = 1000.0

        with patch("time.time", return_value=allocated_time):
            detector.track_allocation("state1", "agent1", {"cpu": 1.0})

        # Multiple detections of same leak
        with patch("time.time", return_value=allocated_time + 350):
            detector.detect_leaks()
        with patch("time.time", return_value=allocated_time + 400):
            detector.detect_leaks()
        with patch("time.time", return_value=allocated_time + 450):
            detector.detect_leaks()

        # Should only have one entry in history
        assert len(detector.detected_leaks) == 1

    def test_detect_leaks_history_trimming(self, detector):
        """Test that leak history is trimmed to max size."""
        # Override max history for testing
        detector._max_leaks_history = 2

        allocated_time = 1000.0

        # Create 3 different leaks
        with patch("time.time", return_value=allocated_time):
            detector.track_allocation("state1", "agent1", {"cpu": 1.0})
            detector.track_allocation("state2", "agent2", {"memory": 256})
            detector.track_allocation("state3", "agent3", {"gpu": 1})

        with patch("time.time", return_value=allocated_time + 350):
            detector.detect_leaks()

        # Should only keep the last 2 leaks
        assert len(detector.detected_leaks) == 2

    def test_detect_leaks_copies_resources(self, detector):
        """Test that detected leaks contain copies of resource data."""
        resources = {"cpu": 1.0}
        allocated_time = 1000.0

        with patch("time.time", return_value=allocated_time):
            detector.track_allocation("state1", "agent1", resources)

        with patch("time.time", return_value=allocated_time + 350):
            leaks = detector.detect_leaks()

        # Modify original resources
        resources["cpu"] = 5.0

        # Leak should have original values
        assert leaks[0].resources["cpu"] == 1.0

    @pytest.mark.parametrize(
        "threshold,hold_time,should_leak",
        [
            (300, 299, False),  # Just under threshold
            (300, 300, False),  # Exactly at threshold
            (300, 301, True),  # Just over threshold
            (60, 61, True),  # Short threshold
            (3600, 3601, True),  # Long threshold
            (0, 1, True),  # Zero threshold
        ],
    )
    def test_detect_leaks_threshold_boundaries(self, threshold, hold_time, should_leak):
        """Test leak detection at various threshold boundaries."""
        detector = ResourceLeakDetector(leak_threshold_seconds=threshold)
        allocated_time = 1000.0

        with patch("time.time", return_value=allocated_time):
            detector.track_allocation("state1", "agent1", {"cpu": 1.0})

        with patch("time.time", return_value=allocated_time + hold_time):
            leaks = detector.detect_leaks()

        if should_leak:
            assert len(leaks) == 1
            assert leaks[0].held_for_seconds == hold_time
        else:
            assert len(leaks) == 0

    def test_get_metrics_empty(self, detector):
        """Test get_metrics with no allocations or leaks."""
        metrics = detector.get_metrics()

        assert metrics["total_allocations"] == 0
        assert metrics["current_leaks"] == 0
        assert metrics["total_detected_leaks"] == 0
        assert metrics["leak_threshold_seconds"] == 300.0
        assert metrics["oldest_allocation_age"] is None
        assert metrics["leaks_by_agent"] == {}

    def test_get_metrics_with_allocations(self, detector, sample_resources):
        """Test get_metrics with active allocations."""
        allocated_time = 1000.0

        with patch("time.time", return_value=allocated_time):
            detector.track_allocation("state1", "agent1", sample_resources)
            detector.track_allocation("state2", "agent2", {"memory": 256})

        current_time = allocated_time + 100
        with patch("time.time", return_value=current_time):
            metrics = detector.get_metrics()

        assert metrics["total_allocations"] == 2
        assert metrics["current_leaks"] == 0  # Not leaked yet
        assert metrics["oldest_allocation_age"] == 100.0

    def test_get_metrics_with_leaks(self, detector):
        """Test get_metrics with leaked resources."""
        allocated_time = 1000.0

        with patch("time.time", return_value=allocated_time):
            detector.track_allocation("state1", "agent1", {"cpu": 1.0})
            detector.track_allocation("state2", "agent2", {"memory": 256})

        current_time = allocated_time + 350
        with patch("time.time", return_value=current_time):
            metrics = detector.get_metrics()

        assert metrics["total_allocations"] == 2
        assert metrics["current_leaks"] == 2
        assert metrics["total_detected_leaks"] == 2

    def test_get_metrics_leaks_by_agent(self, detector):
        """Test get_metrics groups leaks by agent correctly."""
        allocated_time = 1000.0

        with patch("time.time", return_value=allocated_time):
            # agent1 has 2 leaks
            detector.track_allocation("state1", "agent1", {"cpu": 1.0})
            detector.track_allocation("state2", "agent1", {"memory": 256})
            # agent2 has 1 leak
            detector.track_allocation("state3", "agent2", {"gpu": 1})
            # agent3 has no leaks (recent allocation)

        with patch("time.time", return_value=allocated_time + 200):
            detector.track_allocation("state4", "agent3", {"disk": 100})

        current_time = allocated_time + 350
        with patch("time.time", return_value=current_time):
            metrics = detector.get_metrics()

        expected_leaks_by_agent = {"agent1": 2, "agent2": 1}
        assert metrics["leaks_by_agent"] == expected_leaks_by_agent

    def test_get_oldest_allocation_age_empty(self, detector):
        """Test _get_oldest_allocation_age with no allocations."""
        age = detector._get_oldest_allocation_age()
        assert age is None

    def test_get_oldest_allocation_age_single(self, detector, sample_resources):
        """Test _get_oldest_allocation_age with single allocation."""
        allocated_time = 1000.0
        current_time = allocated_time + 150

        with patch("time.time", return_value=allocated_time):
            detector.track_allocation("state1", "agent1", sample_resources)

        with patch("time.time", return_value=current_time):
            age = detector._get_oldest_allocation_age()

        assert age == 150.0

    def test_get_oldest_allocation_age_multiple(self, detector):
        """Test _get_oldest_allocation_age with multiple allocations."""
        base_time = 1000.0
        current_time = base_time + 300

        # Different allocation times
        with patch("time.time", return_value=base_time):
            detector.track_allocation("oldest", "agent1", {"cpu": 1.0})

        with patch("time.time", return_value=base_time + 100):
            detector.track_allocation("middle", "agent1", {"memory": 256})

        with patch("time.time", return_value=base_time + 200):
            detector.track_allocation("newest", "agent1", {"gpu": 1})

        with patch("time.time", return_value=current_time):
            age = detector._get_oldest_allocation_age()

        # Should return age of oldest allocation
        assert age == 300.0

    def test_group_leaks_by_agent_empty(self, detector):
        """Test _group_leaks_by_agent with empty list."""
        groups = detector._group_leaks_by_agent([])
        assert groups == {}

    def test_group_leaks_by_agent_single_agent(self, detector):
        """Test _group_leaks_by_agent with single agent."""
        leaks = [
            ResourceLeak("state1", "agent1", {}, 1000, 350, 300),
            ResourceLeak("state2", "agent1", {}, 1000, 400, 300),
        ]

        groups = detector._group_leaks_by_agent(leaks)
        assert groups == {"agent1": 2}

    def test_group_leaks_by_agent_multiple_agents(self, detector):
        """Test _group_leaks_by_agent with multiple agents."""
        leaks = [
            ResourceLeak("state1", "agent1", {}, 1000, 350, 300),
            ResourceLeak("state2", "agent1", {}, 1000, 400, 300),
            ResourceLeak("state3", "agent2", {}, 1000, 500, 300),
            ResourceLeak("state4", "agent3", {}, 1000, 600, 300),
            ResourceLeak("state5", "agent2", {}, 1000, 700, 300),
        ]

        groups = detector._group_leaks_by_agent(leaks)
        expected = {"agent1": 2, "agent2": 2, "agent3": 1}
        assert groups == expected

    def test_clear_leak_history(self, detector, sample_resources):
        """Test clear_leak_history removes all detected leaks."""
        allocated_time = 1000.0

        with patch("time.time", return_value=allocated_time):
            detector.track_allocation("state1", "agent1", sample_resources)

        # Generate some leak history
        with patch("time.time", return_value=allocated_time + 350):
            detector.detect_leaks()

        assert len(detector.detected_leaks) > 0

        detector.clear_leak_history()
        assert len(detector.detected_leaks) == 0

    def test_clear_leak_history_preserves_allocations(self, detector, sample_resources):
        """Test clear_leak_history doesn't affect current allocations."""
        allocated_time = 1000.0

        with patch("time.time", return_value=allocated_time):
            detector.track_allocation("state1", "agent1", sample_resources)

        # Generate leak history
        with patch("time.time", return_value=allocated_time + 350):
            detector.detect_leaks()

        detector.clear_leak_history()

        # Allocations should still exist
        assert len(detector.allocations) == 1
        assert "agent1:state1" in detector.allocations


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_resource_dict(self):
        """Test tracking allocation with empty resource dictionary."""
        detector = ResourceLeakDetector()
        empty_resources = {}

        with patch("time.time", return_value=1000.0):
            detector.track_allocation("state1", "agent1", empty_resources)

        allocation = detector.allocations["agent1:state1"]
        assert allocation.resources == {}

    def test_none_values_in_resources(self):
        """Test tracking allocation with None values in resources."""
        detector = ResourceLeakDetector()
        resources_with_none = {"cpu": None, "memory": 1024}

        with patch("time.time", return_value=1000.0):
            detector.track_allocation("state1", "agent1", resources_with_none)

        allocation = detector.allocations["agent1:state1"]
        assert allocation.resources["cpu"] is None
        assert allocation.resources["memory"] == 1024

    def test_zero_threshold(self):
        """Test detector with zero threshold - everything should leak immediately."""
        detector = ResourceLeakDetector(leak_threshold_seconds=0.0)

        with patch("time.time", return_value=1000.0):
            detector.track_allocation("state1", "agent1", {"cpu": 1.0})

        # Even immediately, should be a leak with zero threshold
        with patch("time.time", return_value=1000.001):  # 1ms later
            leaks = detector.detect_leaks()
            assert len(leaks) == 1

    def test_negative_threshold(self):
        """Test detector with negative threshold."""
        detector = ResourceLeakDetector(leak_threshold_seconds=-100.0)

        with patch("time.time", return_value=1000.0):
            detector.track_allocation("state1", "agent1", {"cpu": 1.0})

        # Should immediately be a leak
        with patch("time.time", return_value=1000.0):
            leaks = detector.detect_leaks()
            assert len(leaks) == 1

    def test_very_large_threshold(self):
        """Test detector with very large threshold."""
        detector = ResourceLeakDetector(leak_threshold_seconds=1e10)  # ~316 years

        with patch("time.time", return_value=1000.0):
            detector.track_allocation("state1", "agent1", {"cpu": 1.0})

        # Even after a long time, shouldn't leak
        with patch("time.time", return_value=1000.0 + 86400 * 365):  # 1 year later
            leaks = detector.detect_leaks()
            assert len(leaks) == 0

    def test_special_characters_in_names(self):
        """Test state and agent names with special characters."""
        detector = ResourceLeakDetector()

        special_state = "state-with_special.chars:123"
        special_agent = "agent@domain.com/path"

        with patch("time.time", return_value=1000.0):
            detector.track_allocation(special_state, special_agent, {"cpu": 1.0})

        key = f"{special_agent}:{special_state}"
        assert key in detector.allocations

        allocation = detector.allocations[key]
        assert allocation.state_name == special_state
        assert allocation.agent_name == special_agent

    def test_unicode_names(self):
        """Test state and agent names with unicode characters."""
        detector = ResourceLeakDetector()

        unicode_state = "状態テスト"
        unicode_agent = "агент_тест"

        with patch("time.time", return_value=1000.0):
            detector.track_allocation(unicode_state, unicode_agent, {"cpu": 1.0})

        key = f"{unicode_agent}:{unicode_state}"
        assert key in detector.allocations

    def test_very_long_names(self):
        """Test very long state and agent names."""
        detector = ResourceLeakDetector()

        long_state = "a" * 1000
        long_agent = "b" * 1000

        with patch("time.time", return_value=1000.0):
            detector.track_allocation(long_state, long_agent, {"cpu": 1.0})

        key = f"{long_agent}:{long_state}"
        assert key in detector.allocations

    def test_massive_resource_dict(self):
        """Test tracking allocation with very large resource dictionary."""
        detector = ResourceLeakDetector()

        # Create large resource dict
        massive_resources = {f"resource_{i}": float(i) for i in range(1000)}

        with patch("time.time", return_value=1000.0):
            detector.track_allocation("state1", "agent1", massive_resources)

        allocation = detector.allocations["agent1:state1"]
        assert len(allocation.resources) == 1000
        assert allocation.resources["resource_0"] == 0.0
        assert allocation.resources["resource_999"] == 999.0


class TestGlobalInstance:
    """Test the global leak_detector instance."""

    def test_global_instance_exists(self):
        """Test that global leak_detector instance exists and is correct type."""
        assert leak_detector is not None
        assert isinstance(leak_detector, ResourceLeakDetector)

    def test_global_instance_default_config(self):
        """Test that global instance has default configuration."""
        assert leak_detector.leak_threshold == 300.0
        assert leak_detector._max_leaks_history == 100

    def test_global_instance_functionality(self):
        """Test that global instance functions correctly."""
        # Clear any existing state
        leak_detector.allocations.clear()
        leak_detector.clear_leak_history()

        with patch("time.time", return_value=1000.0):
            leak_detector.track_allocation("test_state", "test_agent", {"cpu": 1.0})

        assert len(leak_detector.allocations) == 1

        # Clean up
        leak_detector.track_release("test_state", "test_agent")
        leak_detector.clear_leak_history()


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_workflow_lifecycle(self):
        """Test complete workflow: allocate -> detect -> release -> clean."""
        detector = ResourceLeakDetector(leak_threshold_seconds=100.0)
        resources = {"cpu": 2.0, "memory": 1024}

        # 1. Allocate resources
        with patch("time.time", return_value=1000.0):
            detector.track_allocation("workflow_state", "worker_agent", resources)

        # 2. Check no leaks initially
        with patch("time.time", return_value=1050.0):  # 50s later
            leaks = detector.detect_leaks()
            assert len(leaks) == 0

        # 3. Check leak detected after threshold
        with patch("time.time", return_value=1150.0):  # 150s later
            leaks = detector.detect_leaks()
            assert len(leaks) == 1

            leak = leaks[0]
            assert leak.state_name == "workflow_state"
            assert leak.held_for_seconds == 150.0

        # 4. Release resources
        detector.track_release("workflow_state", "worker_agent")

        # 5. Verify no current leaks but history preserved
        with patch("time.time", return_value=1200.0):
            leaks = detector.detect_leaks()
            assert len(leaks) == 0
            assert len(detector.detected_leaks) == 1  # History preserved

    def test_multiple_agents_workflow(self):
        """Test scenario with multiple agents and mixed leak patterns."""
        detector = ResourceLeakDetector(leak_threshold_seconds=200.0)

        base_time = 1000.0

        # Agent1: Allocates early, will leak
        with patch("time.time", return_value=base_time):
            detector.track_allocation("long_task", "agent1", {"cpu": 4.0})

        # Agent2: Allocates later, won't leak yet
        with patch("time.time", return_value=base_time + 100):
            detector.track_allocation("medium_task", "agent2", {"memory": 512})

        # Agent3: Allocates much later, won't leak
        with patch("time.time", return_value=base_time + 180):
            detector.track_allocation("quick_task", "agent3", {"gpu": 1})

        # Check leaks at base_time + 250
        with patch("time.time", return_value=base_time + 250):
            leaks = detector.detect_leaks()
            metrics = detector.get_metrics()

        # Only agent1 should have leaked
        assert len(leaks) == 1
        assert leaks[0].agent_name == "agent1"
        assert leaks[0].held_for_seconds == 250.0

        # Metrics should reflect the state
        assert metrics["total_allocations"] == 3
        assert metrics["current_leaks"] == 1
        assert metrics["leaks_by_agent"] == {"agent1": 1}
        assert metrics["oldest_allocation_age"] == 250.0

    def test_release_and_reallocate_cycle(self):
        """Test release and reallocation of same state/agent."""
        detector = ResourceLeakDetector(leak_threshold_seconds=100.0)

        # First allocation cycle
        with patch("time.time", return_value=1000.0):
            detector.track_allocation("cycling_state", "agent1", {"cpu": 1.0})

        # Let it leak
        with patch("time.time", return_value=1150.0):
            leaks = detector.detect_leaks()
            assert len(leaks) == 1

        # Release
        detector.track_release("cycling_state", "agent1")

        # Reallocate same state/agent
        with patch("time.time", return_value=1200.0):
            detector.track_allocation("cycling_state", "agent1", {"cpu": 2.0})

        # Should not leak immediately
        with patch("time.time", return_value=1250.0):  # 50s later
            leaks = detector.detect_leaks()
            assert len(leaks) == 0

        # But should leak after threshold
        with patch("time.time", return_value=1350.0):  # 150s later
            leaks = detector.detect_leaks()
            assert len(leaks) == 1
            assert leaks[0].held_for_seconds == 150.0

    def test_history_management_under_load(self):
        """Test leak history management with many leaks."""
        detector = ResourceLeakDetector(leak_threshold_seconds=50.0)
        detector._max_leaks_history = 5  # Small history for testing

        base_time = 1000.0

        # Create more leaks than history can hold
        for i in range(10):
            with patch("time.time", return_value=base_time + i):
                detector.track_allocation(f"state_{i}", f"agent_{i}", {"cpu": 1.0})

        # Trigger leak detection
        with patch("time.time", return_value=base_time + 100):
            leaks = detector.detect_leaks()

        # Should detect all 10 current leaks
        assert len(leaks) == 10

        # But history should be trimmed to max size
        assert len(detector.detected_leaks) == 5

    @pytest.mark.parametrize("num_allocations", [1, 10, 100, 1000])
    def test_performance_with_many_allocations(self, num_allocations):
        """Test performance doesn't degrade significantly with many allocations."""
        detector = ResourceLeakDetector()

        # Allocate many resources
        with patch("time.time", return_value=1000.0):
            for i in range(num_allocations):
                detector.track_allocation(f"state_{i}", f"agent_{i % 10}", {"cpu": 1.0})

        # Measure time for leak detection
        import time as time_module

        with patch("time.time", return_value=1400.0):  # All should leak
            start = time_module.perf_counter()
            leaks = detector.detect_leaks()
            metrics = detector.get_metrics()
            end = time_module.perf_counter()

        # Verify correctness
        assert len(leaks) == num_allocations
        assert metrics["total_allocations"] == num_allocations
        assert metrics["current_leaks"] == num_allocations

        # Performance should be reasonable (adjust threshold as needed)
        execution_time = end - start
        assert execution_time < 2.0  # Should complete within 2 seconds (relaxed for CI)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
