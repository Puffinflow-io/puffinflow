"""Pure Python fallback for StateMachineCore.

Provides the same API as the Rust extension but implemented in pure Python.
Key optimizations over the previous Agent internals:
- O(1) queue membership via _in_queue set (replaces O(N) linear scan)
- Monotonic counter instead of time.time() for queue ordering
- Integer-indexed arrays instead of string dict lookups in hot path
"""

from __future__ import annotations

import heapq

# Status constants matching Rust side
_PENDING = 0
_RUNNING = 1
_COMPLETED = 2
_FAILED = 3

_STATUS_NAMES = ("pending", "running", "completed", "failed")


class StateMachineCore:
    """Pure Python state machine core for workflow execution.

    Manages queue, dependency tracking, and state metadata using
    integer-indexed arrays for performance.

    Args:
        state_configs: List of (name, priority, max_retries, dep_names) tuples.
    """

    __slots__ = (
        "_all_state_names_set",
        "_attempts",
        "_completed",
        "_completed_once",
        "_dependents",
        "_deps",
        "_heap",
        "_in_queue",
        "_max_retries",
        "_n",
        "_name_to_idx",
        "_names",
        "_priority",
        "_running",
        "_seq",
        "_status",
    )

    def __init__(self, state_configs: list) -> None:
        n = len(state_configs)
        self._n = n
        self._names: list[str] = []
        self._name_to_idx: dict[str, int] = {}

        self._priority: list[int] = [0] * n
        self._max_retries: list[int] = [3] * n
        self._attempts: list[int] = [0] * n
        self._status: list[int] = [_PENDING] * n

        # Dependency storage: list of dep indices per state
        self._deps: list[list[int]] = [[] for _ in range(n)]
        # Reverse dependency index: state_idx -> list of dependent state indices
        self._dependents: list[list[int]] = [[] for _ in range(n)]

        # Priority queue: (neg_priority, sequence, state_index)
        self._heap: list[tuple[int, int, int]] = []
        self._in_queue: set[int] = set()
        self._running: set[int] = set()
        self._completed: set[int] = set()
        self._completed_once: set[int] = set()
        self._seq: int = 0

        # First pass: register names
        for i, cfg in enumerate(state_configs):
            name = cfg[0]
            self._names.append(name)
            self._name_to_idx[name] = i
            self._priority[i] = cfg[1]
            self._max_retries[i] = cfg[2]

        self._all_state_names_set = set(self._name_to_idx.keys())

        # Second pass: resolve dependencies
        for i, cfg in enumerate(state_configs):
            dep_names = cfg[3]
            for dn in dep_names:
                dep_idx = self._name_to_idx.get(dn)
                if dep_idx is not None:
                    self._deps[i].append(dep_idx)
                    self._dependents[dep_idx].append(i)

    def add_to_queue(self, state_name: str, priority_boost: int = 0) -> None:
        """Add state to priority queue with O(1) membership guard."""
        idx = self._name_to_idx.get(state_name)
        if idx is None:
            return
        if idx in self._in_queue:
            return
        self._in_queue.add(idx)
        neg_pri = -(self._priority[idx] + priority_boost)
        self._seq += 1
        heapq.heappush(self._heap, (neg_pri, self._seq, idx))

    def get_ready_states(self) -> list[str]:
        """Pop ready states from heap. Returns list of state names."""
        ready: list[str] = []
        reinsert: list[tuple[int, int, int]] = []

        while self._heap:
            entry = heapq.heappop(self._heap)
            idx = entry[2]

            # Lazy deletion: skip if no longer in queue
            if idx not in self._in_queue:
                continue

            if self._can_run(idx):
                self._in_queue.discard(idx)
                ready.append(self._names[idx])
            elif idx not in self._completed_once:
                reinsert.append(entry)
            else:
                # Already completed, remove from queue
                self._in_queue.discard(idx)

        for item in reinsert:
            heapq.heappush(self._heap, item)

        return ready

    def mark_running(self, state_name: str) -> None:
        """Mark state as running."""
        idx = self._name_to_idx.get(state_name)
        if idx is None:
            return
        self._running.add(idx)
        self._status[idx] = _RUNNING

    def mark_completed(self, state_name: str) -> list[str]:
        """Mark state completed, check dependents, queue newly-ready states.

        Returns list of newly queued state names.
        """
        idx = self._name_to_idx.get(state_name)
        if idx is None:
            return []

        self._running.discard(idx)
        self._completed.add(idx)
        self._completed_once.add(idx)
        self._status[idx] = _COMPLETED
        self._attempts[idx] += 1

        # Check dependents
        newly_queued: list[str] = []
        for dep_idx in self._dependents[idx]:
            if dep_idx in self._completed_once:
                continue
            if dep_idx in self._running:
                continue
            if dep_idx in self._in_queue:
                continue
            if self._can_run(dep_idx):
                self._in_queue.add(dep_idx)
                neg_pri = -self._priority[dep_idx]
                self._seq += 1
                heapq.heappush(self._heap, (neg_pri, self._seq, dep_idx))
                newly_queued.append(self._names[dep_idx])

        return newly_queued

    def handle_result(self, state_name: str, result: object) -> None:
        """Parse None/str/list[str] result and queue next states."""
        if result is None:
            return
        if isinstance(result, str):
            if result in self._all_state_names_set:
                r_idx = self._name_to_idx[result]
                if r_idx not in self._completed_once:
                    self.add_to_queue(result)
        elif isinstance(result, list):
            for ns in result:
                if isinstance(ns, str) and ns in self._all_state_names_set:
                    ns_idx = self._name_to_idx[ns]
                    if ns_idx not in self._completed_once:
                        self.add_to_queue(ns)

    def mark_failed(self, state_name: str) -> None:
        """Mark state as failed."""
        idx = self._name_to_idx.get(state_name)
        if idx is None:
            return
        self._running.discard(idx)
        self._in_queue.discard(idx)
        self._status[idx] = _FAILED

    def is_done(self) -> bool:
        """Check if all work is complete."""
        return not self._in_queue and not self._running

    def has_queued(self) -> bool:
        """Check if there are states in the queue."""
        return bool(self._in_queue)

    def get_completed_states(self) -> list[str]:
        """Get list of currently completed state names."""
        return [self._names[i] for i in self._completed]

    def get_completed_once(self) -> list[str]:
        """Get list of state names that have completed at least once."""
        return [self._names[i] for i in self._completed_once]

    def get_running_states(self) -> list[str]:
        """Get list of currently running state names."""
        return [self._names[i] for i in self._running]

    def get_state_status(self, state_name: str) -> str:
        """Get status string for a state."""
        idx = self._name_to_idx.get(state_name)
        if idx is None:
            return "pending"
        return _STATUS_NAMES[self._status[idx]]

    def queue_len(self) -> int:
        """Get number of states in queue."""
        return len(self._in_queue)

    def num_states(self) -> int:
        """Get total number of states."""
        return self._n

    def reset(self) -> None:
        """Clear all tracking state."""
        for i in range(self._n):
            self._status[i] = _PENDING
            self._attempts[i] = 0
        self._heap.clear()
        self._in_queue.clear()
        self._running.clear()
        self._completed.clear()
        self._completed_once.clear()
        self._seq = 0

    def _can_run(self, idx: int) -> bool:
        """Check if state at index can run."""
        if idx in self._running:
            return False
        if idx in self._completed_once:
            return False
        deps = self._deps[idx]
        if not deps:
            return True
        completed = self._completed
        return all(d in completed for d in deps)
