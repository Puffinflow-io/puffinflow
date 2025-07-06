"""Lightweight circuit breaker implementation."""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class CircuitState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # For half-open state
    timeout: float = 30.0
    name: str = "default"


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open"""

    pass


class CircuitBreaker:
    """Lightweight circuit breaker for state execution"""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def protect(self) -> AsyncGenerator[None, None]:
        """Context manager for protecting code blocks"""
        async with self._lock:
            await self._check_state()

            if self.state == CircuitState.OPEN:
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.config.name}' is OPEN"
                )

        time.time()
        try:
            yield
            await self._record_success()
        except Exception:
            await self._record_failure()
            raise

    async def _check_state(self) -> None:
        """Check and update circuit breaker state"""
        now = time.time()

        if (
            self.state == CircuitState.OPEN
            and now - self._last_failure_time >= self.config.recovery_timeout
        ):
            self.state = CircuitState.HALF_OPEN
            self._success_count = 0

        elif self.state == CircuitState.HALF_OPEN:
            if self._success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self._failure_count = 0

    async def _record_success(self) -> None:
        """Record successful execution"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self._success_count += 1
            elif self.state == CircuitState.CLOSED:
                self._failure_count = max(
                    0, self._failure_count - 1
                )  # Gradually recover

    async def _record_failure(self) -> None:
        """Record failed execution"""
        async with self._lock:
            now = time.time()
            self._failure_count += 1
            self._last_failure_time = int(now)
            self._success_count = 0

            if (
                self.state == CircuitState.CLOSED
                and self._failure_count >= self.config.failure_threshold
            ) or self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics"""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_failure_time": self._last_failure_time,
        }

    async def force_open(self) -> None:
        """Manually open the circuit"""
        async with self._lock:
            self.state = CircuitState.OPEN
            self._last_failure_time = int(time.time())

    async def force_close(self) -> None:
        """Manually close the circuit"""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self._failure_count = 0


# Global circuit breaker registry
class CircuitBreakerRegistry:
    """Simple registry for circuit breakers"""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}

    def get_or_create(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing or create new circuit breaker"""
        if name not in self._breakers:
            if config is None:
                config = CircuitBreakerConfig(name=name)
            self._breakers[name] = CircuitBreaker(config)
        return self._breakers[name]

    def get_all_metrics(self) -> dict[str, dict[str, Any]]:
        """Get metrics for all circuit breakers"""
        return {name: breaker.get_metrics() for name, breaker in self._breakers.items()}


# Global registry instance
circuit_registry = CircuitBreakerRegistry()
