"""Tests for StateMachineCore (runs against whichever backend is available)."""

import pytest

from puffinflow.core.agent._core import StateMachineCore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _simple_configs():
    """3 linear states: a -> b -> c"""
    return [
        ("a", 1, 3, []),
        ("b", 1, 3, ["a"]),
        ("c", 1, 3, ["b"]),
    ]


def _parallel_configs():
    """3 independent states with different priorities."""
    return [
        ("low", 0, 3, []),
        ("normal", 1, 3, []),
        ("high", 2, 3, []),
    ]


def _diamond_configs():
    """Diamond dependency: a -> (b, c) -> d"""
    return [
        ("a", 1, 3, []),
        ("b", 1, 3, ["a"]),
        ("c", 1, 3, ["a"]),
        ("d", 1, 3, ["b", "c"]),
    ]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic_creation(self):
        core = StateMachineCore(_simple_configs())
        assert core.num_states() == 3

    def test_empty_creation(self):
        core = StateMachineCore([])
        assert core.num_states() == 0
        assert core.is_done()

    def test_single_state(self):
        core = StateMachineCore([("only", 1, 3, [])])
        assert core.num_states() == 1


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------

class TestQueue:
    def test_add_to_queue(self):
        core = StateMachineCore(_simple_configs())
        core.add_to_queue("a", 0)
        assert core.queue_len() == 1
        assert core.has_queued()
        assert not core.is_done()

    def test_duplicate_add_is_noop(self):
        core = StateMachineCore(_simple_configs())
        core.add_to_queue("a", 0)
        core.add_to_queue("a", 0)
        assert core.queue_len() == 1

    def test_add_unknown_state(self):
        core = StateMachineCore(_simple_configs())
        core.add_to_queue("nonexistent", 0)
        assert core.queue_len() == 0

    def test_get_ready_states(self):
        core = StateMachineCore(_simple_configs())
        core.add_to_queue("a", 0)
        ready = core.get_ready_states()
        assert ready == ["a"]
        assert core.queue_len() == 0  # popped

    def test_blocked_state_stays_queued(self):
        core = StateMachineCore(_simple_configs())
        core.add_to_queue("b", 0)  # depends on a
        ready = core.get_ready_states()
        assert ready == []
        assert core.queue_len() == 1  # b is still queued


# ---------------------------------------------------------------------------
# Linear workflow (a -> b -> c)
# ---------------------------------------------------------------------------

class TestLinearWorkflow:
    def test_full_linear_execution(self):
        core = StateMachineCore(_simple_configs())
        core.add_to_queue("a", 0)

        # Step 1: a
        ready = core.get_ready_states()
        assert ready == ["a"]
        core.mark_running("a")
        assert "a" in core.get_running_states()

        newly_queued = core.mark_completed("a")
        assert "b" in newly_queued
        assert "a" in core.get_completed_states()
        assert "a" in core.get_completed_once()

        # Step 2: b
        ready = core.get_ready_states()
        assert ready == ["b"]
        core.mark_running("b")
        newly_queued = core.mark_completed("b")
        assert "c" in newly_queued

        # Step 3: c
        ready = core.get_ready_states()
        assert ready == ["c"]
        core.mark_running("c")
        core.mark_completed("c")

        assert core.is_done()
        assert len(core.get_completed_states()) == 3


# ---------------------------------------------------------------------------
# Parallel states
# ---------------------------------------------------------------------------

class TestParallelStates:
    def test_all_independent_ready(self):
        core = StateMachineCore(_parallel_configs())
        core.add_to_queue("low", 0)
        core.add_to_queue("normal", 0)
        core.add_to_queue("high", 0)

        ready = core.get_ready_states()
        assert len(ready) == 3
        # Highest priority should come first
        assert ready[0] == "high"


# ---------------------------------------------------------------------------
# Diamond dependency (a -> b,c -> d)
# ---------------------------------------------------------------------------

class TestDiamondWorkflow:
    def test_diamond_execution(self):
        core = StateMachineCore(_diamond_configs())
        core.add_to_queue("a", 0)

        # a runs
        ready = core.get_ready_states()
        assert ready == ["a"]
        core.mark_running("a")
        newly = core.mark_completed("a")
        assert set(newly) == {"b", "c"}

        # b and c run in parallel
        ready = core.get_ready_states()
        assert set(ready) == {"b", "c"}

        core.mark_running("b")
        core.mark_running("c")

        # Complete b first — d should NOT be queued yet
        newly = core.mark_completed("b")
        assert "d" not in newly

        # Complete c — now d should be queued
        newly = core.mark_completed("c")
        assert "d" in newly

        # d runs
        ready = core.get_ready_states()
        assert ready == ["d"]
        core.mark_running("d")
        core.mark_completed("d")

        assert core.is_done()
        assert len(core.get_completed_states()) == 4


# ---------------------------------------------------------------------------
# handle_result
# ---------------------------------------------------------------------------

class TestHandleResult:
    def test_none_result(self):
        core = StateMachineCore(_simple_configs())
        core.handle_result("a", None)
        assert core.queue_len() == 0

    def test_string_result(self):
        core = StateMachineCore(_simple_configs())
        core.add_to_queue("a", 0)
        core.get_ready_states()
        core.mark_running("a")
        core.mark_completed("a")
        core.handle_result("a", "b")
        # b may already be queued from mark_completed dep check,
        # but duplicate add should be a noop
        assert core.has_queued()

    def test_list_result(self):
        configs = [
            ("start", 1, 3, []),
            ("opt_a", 1, 3, []),
            ("opt_b", 1, 3, []),
        ]
        core = StateMachineCore(configs)
        core.handle_result("start", ["opt_a", "opt_b"])
        assert core.queue_len() == 2

    def test_unknown_state_in_result(self):
        core = StateMachineCore(_simple_configs())
        core.handle_result("a", "nonexistent")
        assert core.queue_len() == 0

    def test_completed_state_not_requeued(self):
        core = StateMachineCore(_simple_configs())
        core.add_to_queue("a", 0)
        core.get_ready_states()
        core.mark_running("a")
        core.mark_completed("a")
        initial_queue = core.queue_len()
        core.handle_result("x", "a")  # a already completed
        assert core.queue_len() == initial_queue


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

class TestFailure:
    def test_mark_failed(self):
        core = StateMachineCore(_simple_configs())
        core.add_to_queue("a", 0)
        core.get_ready_states()
        core.mark_running("a")
        core.mark_failed("a")

        assert core.get_state_status("a") == "failed"
        assert "a" not in core.get_running_states()

    def test_failed_state_not_in_running(self):
        core = StateMachineCore(_simple_configs())
        core.add_to_queue("a", 0)
        core.get_ready_states()
        core.mark_running("a")
        core.mark_failed("a")
        assert core.get_running_states() == []


# ---------------------------------------------------------------------------
# Status queries
# ---------------------------------------------------------------------------

class TestStatus:
    def test_initial_status(self):
        core = StateMachineCore(_simple_configs())
        assert core.get_state_status("a") == "pending"
        assert core.get_state_status("b") == "pending"

    def test_unknown_state_status(self):
        core = StateMachineCore(_simple_configs())
        assert core.get_state_status("nonexistent") == "pending"

    def test_status_transitions(self):
        core = StateMachineCore(_simple_configs())
        core.add_to_queue("a", 0)
        core.get_ready_states()

        core.mark_running("a")
        assert core.get_state_status("a") == "running"

        core.mark_completed("a")
        assert core.get_state_status("a") == "completed"


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_all(self):
        core = StateMachineCore(_simple_configs())
        core.add_to_queue("a", 0)
        core.get_ready_states()
        core.mark_running("a")
        core.mark_completed("a")

        core.reset()

        assert core.is_done()
        assert core.get_completed_states() == []
        assert core.get_completed_once() == []
        assert core.get_running_states() == []
        assert core.get_state_status("a") == "pending"
        assert core.queue_len() == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_priority_boost(self):
        configs = [
            ("a", 0, 3, []),
            ("b", 0, 3, []),
        ]
        core = StateMachineCore(configs)
        core.add_to_queue("a", 0)
        core.add_to_queue("b", 10)  # boosted

        ready = core.get_ready_states()
        assert ready[0] == "b"  # boosted should come first

    def test_is_done_with_running(self):
        core = StateMachineCore(_simple_configs())
        core.add_to_queue("a", 0)
        core.get_ready_states()
        core.mark_running("a")
        assert not core.is_done()  # still running

    def test_has_queued_false_when_empty(self):
        core = StateMachineCore(_simple_configs())
        assert not core.has_queued()

    def test_many_states(self):
        """Test with more than 64 states (multi-word bitset)."""
        n = 100
        configs = [(f"s{i}", 1, 3, []) for i in range(n)]
        core = StateMachineCore(configs)
        assert core.num_states() == n

        for i in range(n):
            core.add_to_queue(f"s{i}", 0)
        assert core.queue_len() == n

        ready = core.get_ready_states()
        assert len(ready) == n

        for name in ready:
            core.mark_running(name)
            core.mark_completed(name)

        assert core.is_done()
        assert len(core.get_completed_states()) == n
