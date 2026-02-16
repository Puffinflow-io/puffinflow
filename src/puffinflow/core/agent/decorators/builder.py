"""
Builder pattern for constructing state configurations.
"""

from typing import Any, Callable, Optional, Union

from ..state import Priority


class StateBuilder:
    """Builder pattern for constructing state configurations."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}

    # Resource methods
    def cpu(self, units: float) -> "StateBuilder":
        """Set CPU units."""
        self._config["cpu"] = units
        return self

    def memory(self, mb: float) -> "StateBuilder":
        """Set memory in MB."""
        self._config["memory"] = mb
        return self

    def gpu(self, units: float) -> "StateBuilder":
        """Set GPU units."""
        self._config["gpu"] = units
        return self

    def io(self, weight: float) -> "StateBuilder":
        """Set I/O weight."""
        self._config["io"] = weight
        return self

    def network(self, weight: float) -> "StateBuilder":
        """Set network weight."""
        self._config["network"] = weight
        return self

    def resources(
        self,
        cpu: Optional[float] = None,
        memory: Optional[float] = None,
        gpu: Optional[float] = None,
        io: Optional[float] = None,
        network: Optional[float] = None,
    ) -> "StateBuilder":
        """Set multiple resources at once."""
        if cpu is not None:
            self._config["cpu"] = cpu
        if memory is not None:
            self._config["memory"] = memory
        if gpu is not None:
            self._config["gpu"] = gpu
        if io is not None:
            self._config["io"] = io
        if network is not None:
            self._config["network"] = network
        return self

    # Priority and timing
    def priority(self, level: Union[Priority, int, str]) -> "StateBuilder":
        """Set priority level."""
        self._config["priority"] = level
        return self

    def high_priority(self) -> "StateBuilder":
        """Set high priority."""
        self._config["priority"] = Priority.HIGH
        return self

    def critical_priority(self) -> "StateBuilder":
        """Set critical priority."""
        self._config["priority"] = Priority.CRITICAL
        return self

    def low_priority(self) -> "StateBuilder":
        """Set low priority."""
        self._config["priority"] = Priority.LOW
        return self

    def timeout(self, seconds: float) -> "StateBuilder":
        """Set execution timeout."""
        self._config["timeout"] = seconds
        return self

    # Rate limiting
    def rate_limit(self, rate: float, burst: Optional[int] = None) -> "StateBuilder":
        """Set rate limiting."""
        self._config["rate_limit"] = rate
        if burst:
            self._config["burst_limit"] = burst
        return self

    def throttle(self, rate: float) -> "StateBuilder":
        """Alias for rate_limit."""
        return self.rate_limit(rate)

    # Coordination
    def mutex(self) -> "StateBuilder":
        """Enable mutual exclusion."""
        self._config["mutex"] = True
        return self

    def exclusive(self) -> "StateBuilder":
        """Alias for mutex."""
        return self.mutex()

    def semaphore(self, count: int) -> "StateBuilder":
        """Enable semaphore with count."""
        self._config["semaphore"] = count
        return self

    def concurrent(self, max_concurrent: int) -> "StateBuilder":
        """Alias for semaphore."""
        return self.semaphore(max_concurrent)

    def barrier(self, parties: int) -> "StateBuilder":
        """Enable barrier synchronization."""
        self._config["barrier"] = parties
        return self

    def synchronized(self, parties: int) -> "StateBuilder":
        """Alias for barrier."""
        return self.barrier(parties)

    def lease(self, duration: float) -> "StateBuilder":
        """Enable time-based lease."""
        self._config["lease"] = duration
        return self

    def quota(self, limit: float) -> "StateBuilder":
        """Enable quota management."""
        self._config["quota"] = limit
        return self

    # Dependencies
    def depends_on(self, *states: str) -> "StateBuilder":
        """Set state dependencies."""
        self._config["depends_on"] = list(states)
        return self

    def after(self, *states: str) -> "StateBuilder":
        """Alias for depends_on."""
        return self.depends_on(*states)

    # Retry configuration
    def retry(
        self,
        max_retries: int,
        initial_delay: float = 1.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        dead_letter: bool = True,
        circuit_breaker: bool = False,
    ) -> "StateBuilder":
        """Set retry configuration with dead letter options and circuit breaker integration"""
        self._config["retry_config"] = {
            "max_retries": max_retries,
            "initial_delay": initial_delay,
            "exponential_base": exponential_base,
            "jitter": jitter,
            "dead_letter_on_max_retries": dead_letter,
            "dead_letter_on_timeout": dead_letter,
        }

        # Enable circuit breaker if requested for retry scenarios
        if circuit_breaker:
            self._config["circuit_breaker"] = True

        return self

    def retries(self, count: int) -> "StateBuilder":
        """Set simple retry count."""
        self._config["max_retries"] = count
        return self

    def no_retry(self) -> "StateBuilder":
        """Disable retries."""
        self._config["max_retries"] = 0
        return self

    # Dead letter configuration
    def enable_dead_letter(self) -> "StateBuilder":
        """Enable dead letter queue on failure"""
        self._config["dead_letter"] = True
        return self

    def disable_dead_letter(self) -> "StateBuilder":
        """Disable dead letter queue - fail permanently instead"""
        self._config["no_dead_letter"] = True
        return self

    def no_dlq(self) -> "StateBuilder":
        """Alias for disable_dead_letter"""
        return self.disable_dead_letter()

    def with_dead_letter(self, enabled: bool = True) -> "StateBuilder":
        """Configure dead letter behavior"""
        self._config["dead_letter"] = enabled
        return self

    # NEW: Circuit Breaker methods
    def circuit_breaker(
        self,
        enabled: bool = True,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
    ) -> "StateBuilder":
        """Enable circuit breaker with configuration"""
        self._config["circuit_breaker"] = enabled
        if enabled:
            self._config["circuit_breaker_config"] = {
                "failure_threshold": failure_threshold,
                "recovery_timeout": recovery_timeout,
                "success_threshold": success_threshold,
            }
        return self

    def with_circuit_breaker(self, **config: Any) -> "StateBuilder":
        """Enable circuit breaker with custom configuration"""
        self._config["circuit_breaker"] = True
        self._config["circuit_breaker_config"] = config
        return self

    def protected(
        self, failure_threshold: int = 3, recovery_timeout: float = 30.0
    ) -> "StateBuilder":
        """Enable circuit breaker with sensible defaults for protection"""
        return self.circuit_breaker(True, failure_threshold, recovery_timeout)

    def fragile(
        self, failure_threshold: int = 2, recovery_timeout: float = 120.0
    ) -> "StateBuilder":
        """Enable circuit breaker for fragile operations (low threshold, long recovery)"""
        return self.circuit_breaker(True, failure_threshold, recovery_timeout)

    # NEW: Bulkhead methods
    def bulkhead(
        self,
        enabled: bool = True,
        max_concurrent: int = 5,
        max_queue_size: int = 100,
        timeout: float = 30.0,
    ) -> "StateBuilder":
        """Enable bulkhead with configuration"""
        self._config["bulkhead"] = enabled
        if enabled:
            self._config["bulkhead_config"] = {
                "max_concurrent": max_concurrent,
                "max_queue_size": max_queue_size,
                "timeout": timeout,
            }
        return self

    def with_bulkhead(self, **config: Any) -> "StateBuilder":
        """Enable bulkhead with custom configuration"""
        self._config["bulkhead"] = True
        self._config["bulkhead_config"] = config
        return self

    def isolated(self, max_concurrent: int = 3) -> "StateBuilder":
        """Enable bulkhead with isolation (limited concurrency)"""
        return self.bulkhead(True, max_concurrent)

    def single_threaded(self) -> "StateBuilder":
        """Enable bulkhead with single thread execution"""
        return self.bulkhead(True, max_concurrent=1)

    def highly_concurrent(self, max_concurrent: int = 20) -> "StateBuilder":
        """Enable bulkhead allowing high concurrency"""
        return self.bulkhead(True, max_concurrent)

    # NEW: Leak Detection methods
    def leak_detection(self, enabled: bool = True) -> "StateBuilder":
        """Configure resource leak detection"""
        self._config["leak_detection"] = enabled
        return self

    def no_leak_detection(self) -> "StateBuilder":
        """Disable resource leak detection"""
        self._config["leak_detection"] = False
        return self

    # NEW: Combined reliability methods
    def fault_tolerant(
        self,
        circuit_breaker: bool = True,
        bulkhead: bool = True,
        max_concurrent: int = 3,
        failure_threshold: int = 3,
    ) -> "StateBuilder":
        """Enable comprehensive fault tolerance"""
        if circuit_breaker:
            self.circuit_breaker(True, failure_threshold)
        if bulkhead:
            self.bulkhead(True, max_concurrent)
        return self.retry(5, dead_letter=True, circuit_breaker=circuit_breaker)

    def production_ready(self) -> "StateBuilder":
        """Apply production-ready reliability patterns"""
        return (
            self.circuit_breaker(True, failure_threshold=5, recovery_timeout=60.0)
            .bulkhead(True, max_concurrent=5)
            .retry(3, dead_letter=True)
            .leak_detection(True)
        )

    def external_call(self, timeout: float = 30.0) -> "StateBuilder":
        """Configure for external service calls"""
        return (
            self.circuit_breaker(True, failure_threshold=2, recovery_timeout=30.0)
            .bulkhead(True, max_concurrent=10)
            .timeout(timeout)
            .retry(3, dead_letter=True)
        )

    # Metadata
    def tag(self, key: str, value: Any) -> "StateBuilder":
        """Add a tag."""
        if "tags" not in self._config:
            self._config["tags"] = {}
        self._config["tags"][key] = value
        return self

    def tags(self, **tags: Any) -> "StateBuilder":
        """Add multiple tags."""
        if "tags" not in self._config:
            self._config["tags"] = {}
        self._config["tags"].update(tags)
        return self

    def description(self, desc: str) -> "StateBuilder":
        """Set description."""
        self._config["description"] = desc
        return self

    def describe(self, desc: str) -> "StateBuilder":
        """Alias for description."""
        return self.description(desc)

    # Profile application
    def profile(self, name: str) -> "StateBuilder":
        """Apply a profile."""
        self._config["profile"] = name
        return self

    def like(self, name: str) -> "StateBuilder":
        """Alias for profile."""
        return self.profile(name)

    # Advanced options
    def preemptible(self, value: bool = True) -> "StateBuilder":
        """Set preemptible flag."""
        self._config["preemptible"] = value
        return self

    def checkpoint_every(self, seconds: float) -> "StateBuilder":
        """Set checkpoint interval."""
        self._config["checkpoint_interval"] = seconds
        return self

    def cleanup_on_failure(self, value: bool = True) -> "StateBuilder":
        """Set cleanup on failure flag."""
        self._config["cleanup_on_failure"] = value
        return self

    # Build methods
    def build(self) -> dict[str, Any]:
        """Build and return the configuration dictionary."""
        return self._config.copy()

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Use as decorator."""
        from .flexible import state

        decorator = state(config=self._config)
        return decorator(func)  # type: ignore[no-any-return]

    def decorator(self) -> Callable[..., Any]:
        """Get decorator function."""
        from .flexible import state

        return state(config=self._config)


def build_state() -> StateBuilder:
    """Create a new state builder."""
    return StateBuilder()


# Convenience builder functions
def cpu_state(units: float) -> StateBuilder:
    """Start building a CPU-intensive state."""
    return StateBuilder().cpu(units)


def memory_state(mb: float) -> StateBuilder:
    """Start building a memory-intensive state."""
    return StateBuilder().memory(mb)


def gpu_state(units: float) -> StateBuilder:
    """Start building a GPU state."""
    return StateBuilder().gpu(units)


def exclusive_state() -> StateBuilder:
    """Start building an exclusive state."""
    return StateBuilder().mutex()


def concurrent_state(max_concurrent: int) -> StateBuilder:
    """Start building a concurrent state."""
    return StateBuilder().semaphore(max_concurrent)


def high_priority_state() -> StateBuilder:
    """Start building a high priority state."""
    return StateBuilder().high_priority()


def critical_state() -> StateBuilder:
    """Start building a critical state."""
    return StateBuilder().critical_priority()


# NEW: Reliability-focused builders
def fault_tolerant_state() -> StateBuilder:
    """Start building a fault-tolerant state."""
    return StateBuilder().fault_tolerant()


def external_service_state(timeout: float = 30.0) -> StateBuilder:
    """Start building a state for external service calls."""
    return StateBuilder().external_call(timeout)


def production_state() -> StateBuilder:
    """Start building a production-ready state."""
    return StateBuilder().production_ready()


def protected_state(failure_threshold: int = 3) -> StateBuilder:
    """Start building a circuit breaker protected state."""
    return StateBuilder().protected(failure_threshold)


def isolated_state(max_concurrent: int = 3) -> StateBuilder:
    """Start building an isolated (bulkhead) state."""
    return StateBuilder().isolated(max_concurrent)
