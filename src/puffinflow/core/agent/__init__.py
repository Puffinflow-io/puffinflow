"""Agent module for workflow orchestrator with flexible decorators and reliability patterns."""

# Core agent functionality
from .base import Agent
from .context import Context
from .state import (
    Priority, AgentStatus, StateStatus, StateMetadata, PrioritizedState,
    RetryPolicy, StateResult, DeadLetter
)
from .checkpoint import AgentCheckpoint
from .dependencies import DependencyType, DependencyLifecycle, DependencyConfig

# Import decorator functionality
from .decorators.flexible import (
    state, StateProfile,
    minimal_state, cpu_intensive, memory_intensive, io_intensive,
    gpu_accelerated, network_intensive, quick_state, batch_state,
    critical_state, concurrent_state, synchronized_state,
    fault_tolerant, external_service, high_availability,  # NEW
    get_profile, list_profiles, create_custom_decorator
)
from .decorators.builder import (
    StateBuilder, build_state, cpu_state, memory_state, gpu_state,
    exclusive_state, concurrent_state, high_priority_state, critical_state,
    fault_tolerant_state, external_service_state, production_state,  # NEW
    protected_state, isolated_state  # NEW
)
from .decorators.inspection import (
    is_puffinflow_state, get_state_config, get_state_requirements,
    get_state_rate_limit, get_state_coordination, list_state_metadata,
    compare_states, get_state_summary
)

# NEW: Import reliability patterns
from ..reliability import (
    CircuitBreaker, CircuitState, CircuitBreakerError,
    Bulkhead, BulkheadFullError,
    ResourceLeakDetector, ResourceLeak
)

__all__ = [
    # Core classes
    "Agent",
    "Context",
    "AgentCheckpoint",

    # State types
    "Priority",
    "AgentStatus",
    "StateStatus",
    "StateMetadata",
    "PrioritizedState",
    "StateResult",
    "RetryPolicy",
    "DeadLetter",

    # Context types
    "Context",

    # Dependencies
    "DependencyType",
    "DependencyLifecycle",
    "DependencyConfig",

    # Checkpoint
    "AgentCheckpoint",

    # Main Flexible Decorator
    "state",
    "StateProfile",

    # Profile-based Decorators
    "minimal_state",
    "cpu_intensive",
    "memory_intensive",
    "io_intensive",
    "gpu_accelerated",
    "network_intensive",
    "quick_state",
    "batch_state",
    "critical_state",
    "concurrent_state",
    "synchronized_state",
    "fault_tolerant",  # NEW
    "external_service",  # NEW
    "high_availability",  # NEW

    # Profile Management
    "get_profile",
    "list_profiles",
    "create_custom_decorator",

    # Builder Pattern
    "StateBuilder",
    "build_state",
    "cpu_state",
    "memory_state",
    "gpu_state",
    "exclusive_state",
    "high_priority_state",
    "fault_tolerant_state",  # NEW
    "external_service_state",  # NEW
    "production_state",  # NEW
    "protected_state",  # NEW
    "isolated_state",  # NEW

    # Inspection and Utilities
    "is_puffinflow_state",
    "get_state_config",
    "get_state_requirements",
    "get_state_rate_limit",
    "get_state_coordination",
    "list_state_metadata",
    "compare_states",
    "get_state_summary",

    # NEW: Reliability Patterns
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerError",
    "Bulkhead",
    "BulkheadFullError",
    "ResourceLeakDetector",
    "ResourceLeak",

    # Helper functions
    "create_team_decorator",
    "create_environment_decorator",
    "create_service_decorator"
]


# Helper functions for common use cases
def create_team_decorator(team_name: str, **defaults):
    """Create a team-specific decorator with common defaults."""
    team_defaults = {
        'tags': {'team': team_name},
        **defaults
    }
    return create_custom_decorator(**team_defaults)


def create_environment_decorator(env: str, **defaults):
    """Create an environment-specific decorator."""
    env_defaults = {
        'tags': {'environment': env},
        **defaults
    }
    return create_custom_decorator(**env_defaults)


def create_service_decorator(service_name: str, **defaults):
    """Create a service-specific decorator."""
    service_defaults = {
        'tags': {'service': service_name},
        **defaults
    }
    return create_custom_decorator(**service_defaults)


# NEW: Reliability-focused team decorators
def create_reliable_team_decorator(team_name: str, **defaults):
    """Create a team decorator with reliability patterns enabled."""
    reliable_defaults = {
        'tags': {'team': team_name, 'reliability': 'enabled'},
        'circuit_breaker': True,
        'bulkhead': True,
        'leak_detection': True,
        **defaults
    }
    return create_custom_decorator(**reliable_defaults)


def create_external_team_decorator(team_name: str, **defaults):
    """Create a team decorator optimized for external service calls."""
    external_defaults = {
        'tags': {'team': team_name, 'type': 'external'},
        'circuit_breaker': True,
        'circuit_breaker_config': {'failure_threshold': 2, 'recovery_timeout': 30.0},
        'bulkhead': True,
        'bulkhead_config': {'max_concurrent': 10},
        'timeout': 30.0,
        'max_retries': 3,
        **defaults
    }
    return create_custom_decorator(**external_defaults)


# Add new functions to __all__
__all__.extend([
    "create_reliable_team_decorator",
    "create_external_team_decorator"
])

# Enhanced example usage documentation
__doc__ = """
Usage Examples with Reliability Patterns:

1. Basic Circuit Breaker:
   @state(circuit_breaker=True)
   async def fragile_operation(context):
       pass

2. Bulkhead Isolation:
   @state(bulkhead=True, bulkhead_config={'max_concurrent': 3})
   async def resource_intensive(context):
       pass

3. Full Reliability Stack:
   @fault_tolerant
   async def production_critical(context):
       pass

4. Builder Pattern with Reliability:
   @build_state().protected().isolated(2).retry(5)
   async def resilient_state(context):
       pass

5. External Service Call:
   @external_service
   async def api_call(context):
       pass
   
   # Or with builder:
   @external_service_state(timeout=10.0)
   async def quick_api_call(context):
       pass

6. Custom Reliability Configuration:
   @state(
       circuit_breaker=True,
       circuit_breaker_config={
           'failure_threshold': 2,
           'recovery_timeout': 60.0
       },
       bulkhead=True,
       bulkhead_config={
           'max_concurrent': 5,
           'max_queue_size': 20
       }
   )
   async def custom_reliable_state(context):
       pass

7. Builder with Custom Reliability:
   @(build_state()
     .circuit_breaker(failure_threshold=3, recovery_timeout=45.0)
     .bulkhead(max_concurrent=4, timeout=20.0)
     .leak_detection(True)
     .production_ready())
   async def enterprise_state(context):
       pass

8. Team-specific Reliability:
   api_team = create_external_team_decorator('api-team')
   
   @api_team(timeout=15.0)
   async def team_api_call(context):
       pass

9. Reliability Profiles:
   @state('fault_tolerant')  # Pre-configured profile
   async def auto_reliable(context):
       pass
   
   @state('external_service')  # Optimized for external calls
   async def service_integration(context):
       pass

10. Production Monitoring:
    agent = Agent("prod", enable_circuit_breaker=True, enable_bulkhead=True)
    
    # Check reliability metrics
    status = agent.get_resource_status()
    print("Circuit Breaker:", status.get("circuit_breaker"))
    print("Bulkhead:", status.get("bulkhead"))
    print("Resource Leaks:", status.get("resource_leaks"))

Available Reliability Profiles:
- fault_tolerant: Circuit breaker + bulkhead + enhanced retries
- external_service: Optimized for external API calls with timeouts
- high_availability: Maximum reliability for critical operations

Builder Convenience Methods:
- .protected() - Enable circuit breaker with sensible defaults
- .isolated() - Enable bulkhead with limited concurrency
- .fault_tolerant() - Enable comprehensive fault tolerance
- .production_ready() - Apply all production reliability patterns
- .external_call() - Configure for external service integration

For detailed documentation, see the reliability module.
"""