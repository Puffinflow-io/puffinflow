"""Bulkhead pattern for resource isolation."""
import asyncio
from dataclasses import dataclass
from typing import Dict, Set, Optional, Any
from contextlib import asynccontextmanager


@dataclass
class BulkheadConfig:
    name: str
    max_concurrent: int
    max_queue_size: int = 100
    timeout: float = 30.0


class BulkheadFullError(Exception):
    """Raised when bulkhead is at capacity"""
    pass


class Bulkhead:
    """Isolate resources to prevent cascading failures"""

    def __init__(self, config: BulkheadConfig):
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent)
        self._queue_size = 0
        self._active_tasks: Set[asyncio.Task] = set()

    @asynccontextmanager
    async def isolate(self):
        """Execute function within bulkhead constraints"""
        # Check queue capacity
        if self._queue_size >= self.config.max_queue_size:
            raise BulkheadFullError(f"Bulkhead {self.config.name} queue full")

        self._queue_size += 1
        try:
            # Wait for semaphore with timeout
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                raise BulkheadFullError(f"Bulkhead {self.config.name} timeout waiting for slot")

            try:
                yield
            finally:
                self._semaphore.release()
        finally:
            self._queue_size -= 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get bulkhead metrics"""
        return {
            "name": self.config.name,
            "max_concurrent": self.config.max_concurrent,
            "available_slots": self._semaphore._value,
            "queue_size": self._queue_size,
            "max_queue_size": self.config.max_queue_size,
            "active_tasks": len(self._active_tasks)
        }


# Global bulkhead registry
class BulkheadRegistry:
    """Simple registry for bulkheads"""

    def __init__(self):
        self._bulkheads: Dict[str, Bulkhead] = {}

    def get_or_create(self, name: str, config: Optional[BulkheadConfig] = None) -> Bulkhead:
        """Get existing or create new bulkhead"""
        if name not in self._bulkheads:
            if config is None:
                config = BulkheadConfig(name=name, max_concurrent=5)
            self._bulkheads[name] = Bulkhead(config)
        return self._bulkheads[name]

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all bulkheads"""
        return {name: bulkhead.get_metrics() for name, bulkhead in self._bulkheads.items()}


# Global registry instance
bulkhead_registry = BulkheadRegistry()