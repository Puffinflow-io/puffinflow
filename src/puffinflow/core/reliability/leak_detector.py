"""Resource leak detection."""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta


@dataclass
class ResourceLeak:
    state_name: str
    agent_name: str
    resources: Dict[str, float]
    allocated_at: float
    held_for_seconds: float
    leak_threshold_seconds: float


@dataclass
class ResourceAllocation:
    state_name: str
    agent_name: str
    resources: Dict[str, float]
    allocated_at: float


class ResourceLeakDetector:
    """Detect and handle resource leaks"""

    def __init__(self, leak_threshold_seconds: float = 300.0):  # 5 minutes default
        self.leak_threshold = leak_threshold_seconds
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.detected_leaks: List[ResourceLeak] = []
        self._max_leaks_history = 100

    def track_allocation(self, state_name: str, agent_name: str, resources: Dict[str, float]):
        """Track resource allocation"""
        key = f"{agent_name}:{state_name}"
        self.allocations[key] = ResourceAllocation(
            state_name=state_name,
            agent_name=agent_name,
            resources=resources.copy(),
            allocated_at=time.time()
        )

    def track_release(self, state_name: str, agent_name: str):
        """Track resource release"""
        key = f"{agent_name}:{state_name}"
        if key in self.allocations:
            del self.allocations[key]

    def detect_leaks(self) -> List[ResourceLeak]:
        """Find resources held too long"""
        current_leaks = []
        now = time.time()

        for key, allocation in self.allocations.items():
            held_for = now - allocation.allocated_at

            if held_for > self.leak_threshold:
                leak = ResourceLeak(
                    state_name=allocation.state_name,
                    agent_name=allocation.agent_name,
                    resources=allocation.resources.copy(),
                    allocated_at=allocation.allocated_at,
                    held_for_seconds=held_for,
                    leak_threshold_seconds=self.leak_threshold
                )
                current_leaks.append(leak)

                # Add to history if not already there
                if not any(l.state_name == leak.state_name and
                           l.agent_name == leak.agent_name and
                           abs(l.allocated_at - leak.allocated_at) < 1.0
                           for l in self.detected_leaks):
                    self.detected_leaks.append(leak)

        # Trim history
        if len(self.detected_leaks) > self._max_leaks_history:
            self.detected_leaks = self.detected_leaks[-self._max_leaks_history:]

        return current_leaks

    def get_metrics(self) -> Dict[str, any]:
        """Get leak detection metrics"""
        current_leaks = self.detect_leaks()

        return {
            "total_allocations": len(self.allocations),
            "current_leaks": len(current_leaks),
            "total_detected_leaks": len(self.detected_leaks),
            "leak_threshold_seconds": self.leak_threshold,
            "oldest_allocation_age": self._get_oldest_allocation_age(),
            "leaks_by_agent": self._group_leaks_by_agent(current_leaks)
        }

    def _get_oldest_allocation_age(self) -> Optional[float]:
        """Get age of oldest allocation"""
        if not self.allocations:
            return None

        now = time.time()
        oldest = min(alloc.allocated_at for alloc in self.allocations.values())
        return now - oldest

    def _group_leaks_by_agent(self, leaks: List[ResourceLeak]) -> Dict[str, int]:
        """Group leaks by agent"""
        groups = {}
        for leak in leaks:
            groups[leak.agent_name] = groups.get(leak.agent_name, 0) + 1
        return groups

    def clear_leak_history(self):
        """Clear leak detection history"""
        self.detected_leaks.clear()


# Global leak detector instance
leak_detector = ResourceLeakDetector()