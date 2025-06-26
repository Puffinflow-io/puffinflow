"""
Builder pattern for constructing state configurations.
"""

from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass
from src.puffinflow.core.agent.state import Priority
from src.puffinflow.core.agent.decorators.flexible import state, StateProfile


class StateBuilder:
    """Builder pattern for constructing state configurations."""
    
    def __init__(self):
        self._config = {}
    
    # Resource methods
    def cpu(self, units: float) -> 'StateBuilder':
        """Set CPU units."""
        self._config['cpu'] = units
        return self
    
    def memory(self, mb: float) -> 'StateBuilder':
        """Set memory in MB."""
        self._config['memory'] = mb
        return self
    
    def gpu(self, units: float) -> 'StateBuilder':
        """Set GPU units."""
        self._config['gpu'] = units
        return self
    
    def io(self, weight: float) -> 'StateBuilder':
        """Set I/O weight."""
        self._config['io'] = weight
        return self
    
    def network(self, weight: float) -> 'StateBuilder':
        """Set network weight."""
        self._config['network'] = weight
        return self
    
    def resources(self, cpu: float = None, memory: float = None, 
                 gpu: float = None, io: float = None, network: float = None) -> 'StateBuilder':
        """Set multiple resources at once."""
        if cpu is not None:
            self._config['cpu'] = cpu
        if memory is not None:
            self._config['memory'] = memory
        if gpu is not None:
            self._config['gpu'] = gpu
        if io is not None:
            self._config['io'] = io
        if network is not None:
            self._config['network'] = network
        return self
    
    # Priority and timing
    def priority(self, level: Union[Priority, int, str]) -> 'StateBuilder':
        """Set priority level."""
        self._config['priority'] = level
        return self
    
    def high_priority(self) -> 'StateBuilder':
        """Set high priority."""
        return self.priority(Priority.HIGH)
    
    def critical_priority(self) -> 'StateBuilder':
        """Set critical priority."""
        return self.priority(Priority.CRITICAL)
    
    def low_priority(self) -> 'StateBuilder':
        """Set low priority."""
        return self.priority(Priority.LOW)
    
    def timeout(self, seconds: float) -> 'StateBuilder':
        """Set execution timeout."""
        self._config['timeout'] = seconds
        return self
    
    # Rate limiting
    def rate_limit(self, rate: float, burst: int = None) -> 'StateBuilder':
        """Set rate limiting."""
        self._config['rate_limit'] = rate
        if burst is not None:
            self._config['burst_limit'] = burst
        return self
    
    def throttle(self, rate: float) -> 'StateBuilder':
        """Alias for rate_limit."""
        return self.rate_limit(rate)
    
    # Coordination
    def mutex(self) -> 'StateBuilder':
        """Enable mutual exclusion."""
        self._config['mutex'] = True
        return self
    
    def exclusive(self) -> 'StateBuilder':
        """Alias for mutex."""
        return self.mutex()
    
    def semaphore(self, count: int) -> 'StateBuilder':
        """Enable semaphore with count."""
        self._config['semaphore'] = count
        return self
    
    def concurrent(self, max_concurrent: int) -> 'StateBuilder':
        """Alias for semaphore."""
        return self.semaphore(max_concurrent)
    
    def barrier(self, parties: int) -> 'StateBuilder':
        """Enable barrier synchronization."""
        self._config['barrier'] = parties
        return self
    
    def synchronized(self, parties: int) -> 'StateBuilder':
        """Alias for barrier."""
        return self.barrier(parties)
    
    def lease(self, duration: float) -> 'StateBuilder':
        """Enable time-based lease."""
        self._config['lease'] = duration
        return self
    
    def quota(self, limit: float) -> 'StateBuilder':
        """Enable quota management."""
        self._config['quota'] = limit
        return self
    
    # Dependencies
    def depends_on(self, *states: str) -> 'StateBuilder':
        """Set state dependencies."""
        if len(states) == 1:
            self._config['depends_on'] = [states[0]]
        else:
            self._config['depends_on'] = list(states)
        return self
    
    def after(self, *states: str) -> 'StateBuilder':
        """Alias for depends_on."""
        return self.depends_on(*states)
    
    # Retry configuration
    def retries(self, max_retries: int, delay: float = 1.0, 
               exponential: bool = True, jitter: bool = True) -> 'StateBuilder':
        """Set retry configuration."""
        self._config['max_retries'] = max_retries
        self._config['retry_delay'] = delay
        self._config['retry_exponential'] = exponential
        self._config['retry_jitter'] = jitter
        return self
    
    def no_retry(self) -> 'StateBuilder':
        """Disable retries."""
        return self.retries(0)
    
    # Metadata
    def tag(self, key: str, value: Any) -> 'StateBuilder':
        """Add a tag."""
        if 'tags' not in self._config:
            self._config['tags'] = {}
        self._config['tags'][key] = value
        return self
    
    def tags(self, **tags) -> 'StateBuilder':
        """Add multiple tags."""
        if 'tags' not in self._config:
            self._config['tags'] = {}
        self._config['tags'].update(tags)
        return self
    
    def description(self, desc: str) -> 'StateBuilder':
        """Set description."""
        self._config['description'] = desc
        return self
    
    def describe(self, desc: str) -> 'StateBuilder':
        """Alias for description."""
        return self.description(desc)
    
    # Profile application
    def profile(self, name: str) -> 'StateBuilder':
        """Apply a profile."""
        self._config['profile'] = name
        return self
    
    def like(self, name: str) -> 'StateBuilder':
        """Alias for profile."""
        return self.profile(name)
    
    # Advanced options
    def preemptible(self, value: bool = True) -> 'StateBuilder':
        """Set preemptible flag."""
        self._config['preemptible'] = value
        return self
    
    def checkpoint_every(self, seconds: float) -> 'StateBuilder':
        """Set checkpoint interval."""
        self._config['checkpoint_interval'] = seconds
        return self
    
    def cleanup_on_failure(self, value: bool = True) -> 'StateBuilder':
        """Set cleanup on failure flag."""
        self._config['cleanup_on_failure'] = value
        return self
    
    # Build methods
    def build(self) -> Dict[str, Any]:
        """Build and return the configuration dictionary."""
        return self._config.copy()
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        return state(**self._config)(func)
    
    def decorator(self) -> Callable:
        """Get decorator function."""
        return state(**self._config)


def build_state() -> StateBuilder:
    """Create a new state builder."""
    return StateBuilder()


# Convenience builder functions
def cpu_state(units: float) -> StateBuilder:
    """Start building a CPU-intensive state."""
    return build_state().cpu(units)


def memory_state(mb: float) -> StateBuilder:
    """Start building a memory-intensive state."""
    return build_state().memory(mb)


def gpu_state(units: float) -> StateBuilder:
    """Start building a GPU state."""
    return build_state().gpu(units)


def exclusive_state() -> StateBuilder:
    """Start building an exclusive state."""
    return build_state().mutex()


def concurrent_state(max_concurrent: int) -> StateBuilder:
    """Start building a concurrent state."""
    return build_state().semaphore(max_concurrent)


def high_priority_state() -> StateBuilder:
    """Start building a high priority state."""
    return build_state().high_priority()


def critical_state() -> StateBuilder:
    """Start building a critical state."""
    return build_state().critical_priority()