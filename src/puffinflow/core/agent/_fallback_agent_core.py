"""Pure Python fallback for AgentCore.

Provides the same API as the Rust AgentCore but implemented in pure Python.
Uses __slots__ and parallel arrays for memory efficiency.
States are added incrementally via add_state().
"""

from __future__ import annotations

import heapq
from typing import List, Optional


# Status constants matching Rust side
_PENDING = 0
_RUNNING = 1
_COMPLETED = 2
_FAILED = 3

_STATUS_NAMES = ("pending", "running", "completed", "failed")


class FallbackAgentCore:
    """Pure Python AgentCore matching the Rust AgentCore API.

    States are added incrementally via add_state() rather than batch
    construction. Supports validation caching and retry config.
    """

    __slots__ = (
        "agent_name",
        "_names",
        "_name_to_idx",
        "_priority",
        "_max_retries",
        "_attempts",
        "_status",
        "_last_execution",
        "_last_success",
        "_deps",
        "_dependents",
        "_heap",
        "_in_queue",
        "_running",
        "_completed",
        "_completed_once",
        "_seq",
        "_validated",
        "_cached_entry_seq",
        "_cached_entry_par",
        "_retry_max",
        "_retry_initial_delay",
        "_retry_exp_base",
        "_retry_jitter",
    )

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self._names: list[str] = []
        self._name_to_idx: dict[str, int] = {}

        self._priority: list[int] = []
        self._max_retries: list[int] = []
        self._attempts: list[int] = []
        self._status: list[int] = []
        self._last_execution: list[float] = []
        self._last_success: list[float] = []

        self._deps: list[list[int]] = []
        self._dependents: list[list[int]] = []

        self._heap: list[tuple[int, int, int]] = []
        self._in_queue: set[int] = set()
        self._running: set[int] = set()
        self._completed: set[int] = set()
        self._completed_once: set[int] = set()
        self._seq: int = 0

        self._validated: bool = False
        self._cached_entry_seq: list[int] = []
        self._cached_entry_par: list[int] = []

        self._retry_max: list[int] = []
        self._retry_initial_delay: list[float] = []
        self._retry_exp_base: list[float] = []
        self._retry_jitter: list[bool] = []

    def add_state(
        self,
        name: str,
        priority: int,
        max_retries: int,
        dep_names: list[str],
        retry_delay: float = 1.0,
        retry_base: float = 2.0,
        retry_jitter: bool = True,
    ) -> None:
        """Register a state, build dep graph, grow parallel arrays."""
        if name in self._name_to_idx:
            raise ValueError(f"State '{name}' already exists")

        idx = len(self._names)
        self._names.append(name)
        self._name_to_idx[name] = idx
        self._priority.append(priority)
        self._max_retries.append(max_retries)
        self._attempts.append(0)
        self._status.append(_PENDING)
        self._last_execution.append(0.0)
        self._last_success.append(0.0)

        # Resolve dependencies
        deps: list[int] = []
        for dn in dep_names:
            dep_idx = self._name_to_idx.get(dn)
            if dep_idx is not None:
                deps.append(dep_idx)
                # Grow dependents if needed
                while len(self._dependents) <= dep_idx:
                    self._dependents.append([])
                self._dependents[dep_idx].append(idx)
        self._deps.append(deps)
        # Ensure dependents covers this index
        while len(self._dependents) <= idx:
            self._dependents.append([])

        # Retry config
        self._retry_max.append(max_retries)
        self._retry_initial_delay.append(retry_delay)
        self._retry_exp_base.append(retry_base)
        self._retry_jitter.append(retry_jitter)

        self._validated = False

    def validate(self, mode: str) -> None:
        """Iterative cycle detection (Kahn's algorithm) + cache entry states."""
        n = len(self._names)
        if n == 0:
            raise ValueError(
                "No states defined. Agent must have at least one state to run."
            )

        # Kahn's algorithm for cycle detection
        in_degree = [len(self._deps[i]) for i in range(n)]
        queue: list[int] = [i for i in range(n) if in_degree[i] == 0]
        visited = 0
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            visited += 1
            if node < len(self._dependents):
                for dep_idx in self._dependents[node]:
                    in_degree[dep_idx] -= 1
                    if in_degree[dep_idx] == 0:
                        queue.append(dep_idx)
        if visited != n:
            raise ValueError(
                "Circular dependency detected in workflow. "
                "State dependencies form a cycle, which would prevent execution."
            )

        # Cache entry states
        entry_all = [i for i in range(n) if not self._deps[i]]
        if not entry_all:
            raise ValueError(
                "Sequential execution mode requires at least one state without "
                "dependencies to serve as an entry point. All states have "
                "dependencies, creating a deadlock."
            )

        self._cached_entry_seq = [entry_all[0]] if entry_all else []
        self._cached_entry_par = entry_all
        self._validated = True

    def prepare_run(self, mode: str) -> List[str]:
        """Validate if needed, reset tracking, return entry state names."""
        if not self._validated:
            self.validate(mode)

        n = len(self._names)

        # Reset metadata
        for i in range(n):
            self._status[i] = _PENDING
            self._attempts[i] = 0
            self._last_execution[i] = 0.0
            self._last_success[i] = 0.0

        # Reset queue state
        self._heap.clear()
        self._in_queue.clear()
        self._running.clear()
        self._completed.clear()
        self._completed_once.clear()
        self._seq = 0

        entries = (
            self._cached_entry_par if mode == "parallel" else self._cached_entry_seq
        )
        return [self._names[i] for i in entries]

    # ---- StateMachineCore-compatible methods ----

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

    def get_ready_states(self) -> List[str]:
        """Pop ready states from heap."""
        ready: list[str] = []
        reinsert: list[tuple[int, int, int]] = []

        while self._heap:
            entry = heapq.heappop(self._heap)
            idx = entry[2]
            if idx not in self._in_queue:
                continue
            if self._can_run(idx):
                self._in_queue.discard(idx)
                ready.append(self._names[idx])
            elif idx not in self._completed_once:
                reinsert.append(entry)
            else:
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

    def mark_completed(self, state_name: str) -> List[str]:
        """Mark state completed, check dependents, queue newly-ready states."""
        idx = self._name_to_idx.get(state_name)
        if idx is None:
            return []

        self._running.discard(idx)
        self._completed.add(idx)
        self._completed_once.add(idx)
        self._status[idx] = _COMPLETED
        self._attempts[idx] += 1

        import time

        now = time.time()
        self._last_execution[idx] = now
        self._last_success[idx] = now

        newly_queued: list[str] = []
        if idx < len(self._dependents):
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
            idx = self._name_to_idx.get(result)
            if idx is not None and idx not in self._completed_once:
                self.add_to_queue(result)
        elif isinstance(result, list):
            for ns in result:
                if isinstance(ns, str):
                    ns_idx = self._name_to_idx.get(ns)
                    if ns_idx is not None and ns_idx not in self._completed_once:
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
        return not self._in_queue and not self._running

    def has_queued(self) -> bool:
        return bool(self._in_queue)

    def get_completed_states(self) -> List[str]:
        return [self._names[i] for i in self._completed]

    def get_completed_once(self) -> List[str]:
        return [self._names[i] for i in self._completed_once]

    def get_running_states(self) -> List[str]:
        return [self._names[i] for i in self._running]

    def get_state_status(self, state_name: str) -> str:
        idx = self._name_to_idx.get(state_name)
        if idx is None:
            return "pending"
        return _STATUS_NAMES[self._status[idx]]

    def queue_len(self) -> int:
        return len(self._in_queue)

    def num_states(self) -> int:
        return len(self._names)

    def reset(self) -> None:
        n = len(self._names)
        for i in range(n):
            self._status[i] = _PENDING
            self._attempts[i] = 0
            self._last_execution[i] = 0.0
            self._last_success[i] = 0.0
        self._heap.clear()
        self._in_queue.clear()
        self._running.clear()
        self._completed.clear()
        self._completed_once.clear()
        self._seq = 0

    # ---- Metadata accessors ----

    def get_state_attempts(self, state_name: str) -> int:
        idx = self._name_to_idx.get(state_name)
        return self._attempts[idx] if idx is not None else 0

    def get_state_max_retries(self, state_name: str) -> int:
        idx = self._name_to_idx.get(state_name)
        return self._max_retries[idx] if idx is not None else 0

    def get_state_priority(self, state_name: str) -> int:
        idx = self._name_to_idx.get(state_name)
        return self._priority[idx] if idx is not None else 0

    def get_state_last_execution(self, state_name: str) -> Optional[float]:
        idx = self._name_to_idx.get(state_name)
        if idx is None:
            return None
        v = self._last_execution[idx]
        return None if v == 0.0 else v

    def get_state_last_success(self, state_name: str) -> Optional[float]:
        idx = self._name_to_idx.get(state_name)
        if idx is None:
            return None
        v = self._last_success[idx]
        return None if v == 0.0 else v

    def get_all_state_names(self) -> List[str]:
        return list(self._names)

    def get_dependencies(self, state_name: str) -> List[str]:
        idx = self._name_to_idx.get(state_name)
        if idx is None:
            return []
        return [self._names[d] for d in self._deps[idx]]

    def get_dependents(self, state_name: str) -> List[str]:
        idx = self._name_to_idx.get(state_name)
        if idx is None:
            return []
        if idx < len(self._dependents):
            return [self._names[d] for d in self._dependents[idx]]
        return []

    def should_retry(self, state_name: str) -> bool:
        idx = self._name_to_idx.get(state_name)
        if idx is None:
            return False
        return self._attempts[idx] < self._retry_max[idx]

    def get_retry_delay(self, state_name: str, attempt: int) -> float:
        idx = self._name_to_idx.get(state_name)
        if idx is None:
            return 0.0
        delay = self._retry_initial_delay[idx] * (
            self._retry_exp_base[idx] ** attempt
        )
        delay = min(delay, 60.0)
        if self._retry_jitter[idx]:
            delay *= 0.75
        return delay

    def _can_run(self, idx: int) -> bool:
        if idx in self._running:
            return False
        if idx in self._completed_once:
            return False
        deps = self._deps[idx]
        if not deps:
            return True
        completed = self._completed
        return all(d in completed for d in deps)
