"""
Flexible state decorator with optional parameters and multiple configuration methods.
"""

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional, Union

from ...coordination.primitives import PrimitiveType
from ...resources.requirements import ResourceRequirements, ResourceType
from ..state import Priority


@dataclass
class StateProfile:
    """Predefined state configuration profiles."""

    name: str
    cpu: float = 1.0
    memory: float = 100.0
    io: float = 1.0
    network: float = 1.0
    gpu: float = 0.0
    priority: Priority = Priority.NORMAL
    timeout: Optional[float] = None
    rate_limit: Optional[float] = None
    burst_limit: Optional[int] = None
    coordination: Optional[str] = None  # 'mutex', 'semaphore:5', 'barrier:3', etc.
    max_retries: int = 3
    tags: dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    # NEW: Reliability patterns
    circuit_breaker: bool = False
    circuit_breaker_config: Optional[dict[str, Any]] = None
    bulkhead: bool = False
    bulkhead_config: Optional[dict[str, Any]] = None
    leak_detection: bool = True  # Default enabled

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        import copy

        result = {}
        for key, value in self.__dict__.items():
            if value is not None and key != "name":
                # Deep copy to avoid modifying original profile configurations
                if isinstance(value, dict):
                    result[key] = copy.deepcopy(value)
                else:
                    result[key] = value
        return result


# Predefined profiles
PROFILES = {
    "minimal": StateProfile(
        name="minimal",
        cpu=0.1,
        memory=50.0,
        priority=Priority.NORMAL,
        max_retries=1,
        circuit_breaker=False,  # Keep minimal lightweight
        bulkhead=False,
        leak_detection=False,
        tags={"profile": "minimal"},
    ),
    "standard": StateProfile(
        name="standard",
        cpu=1.0,
        memory=100.0,
        priority=Priority.NORMAL,
        max_retries=3,
        circuit_breaker=False,  # Standard doesn't need extra protection
        bulkhead=False,
        leak_detection=True,  # But enable leak detection
        tags={"profile": "standard"},
    ),
    "cpu_intensive": StateProfile(
        name="cpu_intensive",
        cpu=4.0,
        memory=1024.0,
        priority=Priority.HIGH,
        timeout=300.0,
        max_retries=3,
        circuit_breaker=True,  # CPU intensive operations can fail
        bulkhead=True,  # Isolate CPU intensive work
        bulkhead_config={"max_concurrent": 2},  # Limit concurrent CPU work
        leak_detection=True,
        tags={"profile": "cpu_intensive", "workload": "compute"},
    ),
    "memory_intensive": StateProfile(
        name="memory_intensive",
        cpu=2.0,
        memory=4096.0,
        priority=Priority.HIGH,
        timeout=600.0,
        max_retries=3,
        circuit_breaker=True,
        bulkhead=True,
        bulkhead_config={"max_concurrent": 3},
        leak_detection=True,
        tags={"profile": "memory_intensive", "workload": "memory"},
    ),
    "io_intensive": StateProfile(
        name="io_intensive",
        cpu=1.0,
        memory=256.0,
        io=10.0,
        priority=Priority.NORMAL,
        timeout=120.0,
        max_retries=5,
        circuit_breaker=True,  # IO operations can fail often
        circuit_breaker_config={"failure_threshold": 3, "recovery_timeout": 30.0},
        bulkhead=True,
        bulkhead_config={"max_concurrent": 5},
        leak_detection=True,
        tags={"profile": "io_intensive", "workload": "io"},
    ),
    "gpu_accelerated": StateProfile(
        name="gpu_accelerated",
        cpu=2.0,
        memory=2048.0,
        gpu=1.0,
        priority=Priority.HIGH,
        timeout=900.0,
        max_retries=2,
        circuit_breaker=True,
        bulkhead=True,
        bulkhead_config={"max_concurrent": 1},  # Only one GPU operation at a time
        leak_detection=True,
        tags={"profile": "gpu_accelerated", "workload": "gpu"},
    ),
    "network_intensive": StateProfile(
        name="network_intensive",
        cpu=1.0,
        memory=512.0,
        network=10.0,
        priority=Priority.NORMAL,
        timeout=60.0,
        max_retries=5,
        circuit_breaker=True,  # Network operations are unreliable
        circuit_breaker_config={"failure_threshold": 2, "recovery_timeout": 20.0},
        bulkhead=True,
        bulkhead_config={"max_concurrent": 10},
        leak_detection=True,
        tags={"profile": "network_intensive", "workload": "network"},
    ),
    "quick": StateProfile(
        name="quick",
        cpu=0.5,
        memory=50.0,
        priority=Priority.NORMAL,
        timeout=30.0,
        rate_limit=100.0,
        max_retries=2,
        circuit_breaker=False,  # Quick operations shouldn't need circuit breaker
        bulkhead=True,
        bulkhead_config={"max_concurrent": 20},  # Allow many quick operations
        leak_detection=False,  # Quick operations shouldn't leak
        tags={"profile": "quick", "speed": "fast"},
    ),
    "batch": StateProfile(
        name="batch",
        cpu=2.0,
        memory=1024.0,
        priority=Priority.LOW,
        timeout=1800.0,
        max_retries=3,
        circuit_breaker=True,
        bulkhead=True,
        bulkhead_config={"max_concurrent": 3},
        leak_detection=True,
        tags={"profile": "batch", "workload": "batch"},
    ),
    "critical": StateProfile(
        name="critical",
        cpu=2.0,
        memory=512.0,
        priority=Priority.CRITICAL,
        coordination="mutex",
        max_retries=3,
        circuit_breaker=False,  # Critical operations should not be circuit broken
        bulkhead=True,
        bulkhead_config={"max_concurrent": 1},  # Exclusive execution
        leak_detection=True,
        tags={"profile": "critical", "importance": "high"},
    ),
    "concurrent": StateProfile(
        name="concurrent",
        cpu=1.0,
        memory=256.0,
        priority=Priority.NORMAL,
        coordination="semaphore:5",
        max_retries=3,
        circuit_breaker=True,
        bulkhead=True,
        bulkhead_config={"max_concurrent": 5},  # Match semaphore limit
        leak_detection=True,
        tags={"profile": "concurrent", "concurrency": "limited"},
    ),
    "synchronized": StateProfile(
        name="synchronized",
        cpu=1.0,
        memory=200.0,
        priority=Priority.NORMAL,
        coordination="barrier:3",
        max_retries=3,
        circuit_breaker=False,  # Barrier synchronization shouldn't be circuit broken
        bulkhead=False,  # Barriers need to coordinate
        leak_detection=True,
        tags={"profile": "synchronized", "sync": "barrier"},
    ),
    # Dead letter specific profiles
    "resilient": StateProfile(
        name="resilient",
        cpu=1.0,
        memory=200.0,
        priority=Priority.NORMAL,
        max_retries=5,
        circuit_breaker=True,  # Resilient means protected
        circuit_breaker_config={"failure_threshold": 5, "recovery_timeout": 60.0},
        bulkhead=True,
        bulkhead_config={"max_concurrent": 3},
        leak_detection=True,
        tags={"profile": "resilient", "dead_letter": "enabled"},
    ),
    "critical_no_dlq": StateProfile(
        name="critical_no_dlq",
        cpu=2.0,
        memory=512.0,
        priority=Priority.CRITICAL,
        max_retries=3,
        circuit_breaker=False,  # Critical should fail fast, not be circuit broken
        bulkhead=True,
        bulkhead_config={"max_concurrent": 1},
        leak_detection=True,
        tags={"profile": "critical", "dead_letter": "disabled"},
    ),
    # NEW: Reliability-focused profiles
    "fault_tolerant": StateProfile(
        name="fault_tolerant",
        cpu=1.0,
        memory=256.0,
        priority=Priority.NORMAL,
        max_retries=5,
        circuit_breaker=True,
        circuit_breaker_config={"failure_threshold": 3, "recovery_timeout": 45.0},
        bulkhead=True,
        bulkhead_config={"max_concurrent": 4, "max_queue_size": 20},
        leak_detection=True,
        tags={"profile": "fault_tolerant", "reliability": "high"},
    ),
    "external_service": StateProfile(
        name="external_service",
        cpu=0.5,
        memory=128.0,
        priority=Priority.NORMAL,
        timeout=30.0,
        max_retries=3,
        circuit_breaker=True,
        circuit_breaker_config={"failure_threshold": 2, "recovery_timeout": 30.0},
        bulkhead=True,
        bulkhead_config={"max_concurrent": 8, "timeout": 10.0},
        leak_detection=True,
        tags={"profile": "external_service", "type": "integration"},
    ),
    "high_availability": StateProfile(
        name="high_availability",
        cpu=1.5,
        memory=512.0,
        priority=Priority.HIGH,
        max_retries=10,
        circuit_breaker=True,
        circuit_breaker_config={"failure_threshold": 5, "recovery_timeout": 120.0},
        bulkhead=True,
        bulkhead_config={"max_concurrent": 2, "max_queue_size": 50},
        leak_detection=True,
        tags={"profile": "high_availability", "sla": "99.9%"},
    ),
}


class FlexibleStateDecorator:
    """
    Flexible state decorator that supports multiple configuration methods.
    """

    def __init__(self) -> None:
        self.default_config: dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        """
        Handle multiple call patterns:
        - @state
        - @state()
        - @state(profile='cpu_intensive')
        - @state(cpu=2.0, memory=512.0)
        - @state(config={'cpu': 2.0})
        """
        # Case 1: @state (direct decoration without parentheses)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            func = args[0]
            # Still need to merge configurations to resolve profiles
            final_config = self._merge_configurations()
            return self._decorate_function(func, final_config)

        # Case 2: @state() or @state(params...)
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            # Merge all configuration sources
            final_config = self._merge_configurations(*args, **kwargs)
            return self._decorate_function(func, final_config)

        return decorator

    def _merge_configurations(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Merge configuration from multiple sources in priority order."""
        final_config = {}

        # Apply default profile first if present in default_config
        default_profile = self.default_config.get("profile")
        if default_profile and default_profile in PROFILES:
            profile_config = PROFILES[default_profile].to_dict()
            final_config.update(profile_config)

        # Then apply other default config (excluding profile key)
        for key, value in self.default_config.items():
            if key != "profile":
                final_config[key] = value

        # Process positional arguments
        for arg in args:
            if isinstance(arg, dict):
                # Direct config dictionary
                final_config.update(arg)
            elif isinstance(arg, str):
                # Profile name - validate it exists
                if arg not in PROFILES:
                    raise ValueError(f"Unknown profile: {arg}")
                profile_config = PROFILES[arg].to_dict()
                final_config.update(profile_config)
            elif isinstance(arg, StateProfile):
                # Profile object
                profile_config = arg.to_dict()
                final_config.update(profile_config)

        # Process keyword arguments (highest priority)
        # Handle special cases
        config_dict = kwargs.pop("config", {})
        profile_name = kwargs.pop("profile", None)

        # Apply profile first
        if profile_name:
            if profile_name not in PROFILES:
                raise ValueError(f"Unknown profile: {profile_name}")
            profile_config = PROFILES[profile_name].to_dict()
            final_config.update(profile_config)

        # Apply config dict
        if config_dict:
            final_config.update(config_dict)

        # Apply direct keyword arguments (highest priority)
        final_config.update(kwargs)

        return final_config

    def _decorate_function(self, func: Callable, config: dict[str, Any]) -> Callable:
        """Apply decoration with merged configuration."""
        # Set default values for any missing configuration
        defaults: dict[str, Any] = {
            "cpu": 1.0,
            "memory": 100.0,
            "io": 1.0,
            "network": 1.0,
            "gpu": 0.0,
            "priority": Priority.NORMAL,
            "timeout": None,
            "rate_limit": None,
            "burst_limit": None,
            "coordination": None,
            "depends_on": [],
            "max_retries": 3,
            "tags": {},
            "description": None,
            # NEW: Reliability defaults
            "circuit_breaker": False,
            "circuit_breaker_config": None,
            "bulkhead": False,
            "bulkhead_config": None,
            "leak_detection": True,
        }

        # Merge defaults with provided config (only for missing keys)
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value

        # Process and validate configuration
        config = self._process_configuration(config, func)

        # Apply configuration to function
        return self._apply_configuration(func, config)

    def _process_configuration(
        self, config: dict[str, Any], func: Callable
    ) -> dict[str, Any]:
        """Process and validate configuration."""
        # Normalize priority
        priority = config["priority"]
        if isinstance(priority, str):
            try:
                priority = getattr(Priority, priority.upper())
            except AttributeError as e:
                raise KeyError(f"Invalid priority: {priority}") from e
        elif isinstance(priority, int):
            priority = Priority(priority)
        config["priority"] = priority

        # Process coordination string
        coordination = config.get("coordination")
        if coordination and isinstance(coordination, str):
            coord_config = self._parse_coordination_string(coordination)
            config.update(coord_config)

        # Normalize dependencies
        depends_on = config.get("depends_on")
        if isinstance(depends_on, str):
            config["depends_on"] = [depends_on]
        elif depends_on is None:
            config["depends_on"] = []

        # Auto-generate description if not provided
        if not config.get("description"):
            # Use function docstring if available, otherwise generate from name
            if func.__doc__ and func.__doc__.strip():
                config["description"] = func.__doc__.strip()
            else:
                config["description"] = f"State: {func.__name__}"

        # Process tags
        tags = config.get("tags", {})
        if not isinstance(tags, dict):
            tags = {}

        # Add automatic tags
        auto_tags = {"function_name": func.__name__, "decorated_at": "runtime"}
        auto_tags.update(tags)
        config["tags"] = auto_tags

        # Process dead letter configuration
        dead_letter_enabled = config.get("dead_letter", True)
        no_dead_letter = config.get("no_dead_letter", False)

        if no_dead_letter:
            dead_letter_enabled = False

        # Process retry configuration with dead letter
        retry_config = config.get("retry_config")
        max_retries = config.get("max_retries", config.get("retries", 3))

        if retry_config or max_retries != 3 or not dead_letter_enabled:
            if isinstance(retry_config, dict):
                retry_config["dead_letter_on_max_retries"] = dead_letter_enabled
                retry_config["dead_letter_on_timeout"] = dead_letter_enabled
            else:
                retry_config = {
                    "max_retries": max_retries,
                    "dead_letter_on_max_retries": dead_letter_enabled,
                    "dead_letter_on_timeout": dead_letter_enabled,
                }
            config["retry_config"] = retry_config

        # NEW: Process reliability patterns
        self._process_reliability_config(config, func)

        return config

    def _process_reliability_config(
        self, config: dict[str, Any], func: Callable
    ) -> None:
        """Process reliability pattern configuration."""
        # Circuit breaker configuration
        if config.get("circuit_breaker"):
            cb_config = config.get("circuit_breaker_config", {})
            if not isinstance(cb_config, dict):
                cb_config = {}

            # Set defaults
            cb_defaults = {
                "name": f"{func.__name__}_circuit_breaker",
                "failure_threshold": 5,
                "recovery_timeout": 60.0,
                "success_threshold": 3,
                "timeout": 30.0,
            }

            for key, default in cb_defaults.items():
                if key not in cb_config:
                    cb_config[key] = default

            config["circuit_breaker_config"] = cb_config

        # Bulkhead configuration
        if config.get("bulkhead"):
            bh_config = config.get("bulkhead_config", {})
            if not isinstance(bh_config, dict):
                bh_config = {}

            # Set defaults
            bh_defaults = {
                "name": f"{func.__name__}_bulkhead",
                "max_concurrent": 5,
                "max_queue_size": 100,
                "timeout": 30.0,
            }

            for key, default in bh_defaults.items():
                if key not in bh_config:
                    bh_config[key] = default

            config["bulkhead_config"] = bh_config

    def _parse_coordination_string(self, coordination: str) -> dict[str, Any]:
        """Parse coordination string like 'mutex', 'semaphore:5', 'barrier:3'."""
        if ":" in coordination:
            coord_type, param_str = coordination.split(":", 1)
            try:
                param: Optional[int] = int(param_str)
            except ValueError:
                param = None
        else:
            coord_type = coordination
            param = None

        coord_type = coord_type.lower()

        if coord_type == "mutex":
            return {"mutex": True}
        elif coord_type == "semaphore":
            if param is None:
                raise ValueError(f"Unknown coordination type: {coord_type}")
            return {"semaphore": param}
        elif coord_type == "barrier":
            if param is None:
                raise ValueError(f"Unknown coordination type: {coord_type}")
            return {"barrier": param}
        elif coord_type == "lease":
            if param is None:
                raise ValueError(f"Unknown coordination type: {coord_type}")
            return {"lease": param}
        elif coord_type == "quota":
            if param is None:
                raise ValueError(f"Unknown coordination type: {coord_type}")
            return {"quota": param}
        else:
            raise ValueError(f"Unknown coordination type: {coord_type}")

    def _apply_configuration(self, func: Callable, config: dict[str, Any]) -> Callable:
        """Apply the final configuration to the function."""
        # Create resource requirements
        resource_types = ResourceType.NONE

        if config["cpu"] > 0:
            resource_types |= ResourceType.CPU
        if config["memory"] > 0:
            resource_types |= ResourceType.MEMORY
        if config["io"] > 0:
            resource_types |= ResourceType.IO
        if config["network"] > 0:
            resource_types |= ResourceType.NETWORK
        if config["gpu"] > 0:
            resource_types |= ResourceType.GPU

        requirements = ResourceRequirements(
            cpu_units=config["cpu"],
            memory_mb=config["memory"],
            io_weight=config["io"],
            network_weight=config["network"],
            gpu_units=config["gpu"],
            priority_boost=config["priority"].value,
            timeout=config["timeout"],
            resource_types=resource_types,
        )

        # Create dependency configurations
        from ..dependencies import DependencyType

        class DependencyConfig:
            def __init__(self, dep_type: Any) -> None:
                self.type = dep_type

        dependency_configs = {}
        deps = config.get("depends_on", [])
        for dep in deps:
            dependency_configs[dep] = DependencyConfig(DependencyType.REQUIRED)

        # Determine coordination primitive
        coordination_primitive = None
        coordination_config = {}

        if config.get("mutex"):
            coordination_primitive = PrimitiveType.MUTEX
            coordination_config = {"ttl": 30.0}
        elif config.get("semaphore"):
            coordination_primitive = PrimitiveType.SEMAPHORE
            coordination_config = {"max_count": config["semaphore"], "ttl": 30.0}
        elif config.get("barrier"):
            coordination_primitive = PrimitiveType.BARRIER
            coordination_config = {"parties": config["barrier"]}
        elif config.get("lease"):
            coordination_primitive = PrimitiveType.LEASE
            coordination_config = {"ttl": config["lease"], "auto_renew": True}
        elif config.get("quota"):
            coordination_primitive = PrimitiveType.QUOTA
            coordination_config = {"limit": config["quota"]}

        # CRITICAL: Mark as PuffinFlow state
        func._puffinflow_state = True  # type: ignore
        func._state_name = func.__name__  # type: ignore
        func._state_config = config  # type: ignore

        # Store all configuration as function attributes
        func._resource_requirements = requirements  # type: ignore
        func._priority = config["priority"]  # type: ignore
        func._dependency_configs = dependency_configs  # type: ignore
        func._coordination_primitive = coordination_primitive  # type: ignore
        func._coordination_config = coordination_config  # type: ignore

        # Store rate limiting
        if config["rate_limit"]:
            func._rate_limit = config["rate_limit"]  # type: ignore
            func._burst_limit = config["burst_limit"] or int(config["rate_limit"] * 2)  # type: ignore

        # NEW: Store reliability configurations
        if config.get("circuit_breaker"):
            func._circuit_breaker_enabled = True  # type: ignore
            func._circuit_breaker_config = config.get("circuit_breaker_config")  # type: ignore
        else:
            func._circuit_breaker_enabled = False  # type: ignore

        if config.get("bulkhead"):
            func._bulkhead_enabled = True  # type: ignore
            func._bulkhead_config = config.get("bulkhead_config")  # type: ignore
        else:
            func._bulkhead_enabled = False  # type: ignore

        func._leak_detection_enabled = config.get("leak_detection", True)  # type: ignore

        # Store metadata
        func._state_config = config  # type: ignore
        func._state_tags = config["tags"]  # type: ignore
        func._state_description = config["description"]  # type: ignore

        # Preserve function metadata
        func = wraps(func)(func)

        return func

    def with_defaults(self, **defaults: Any) -> "FlexibleStateDecorator":
        """Create a new decorator with different default values."""
        new_decorator = FlexibleStateDecorator()
        new_decorator.default_config = {**self.default_config, **defaults}
        return new_decorator

    def create_profile(self, name: str, **config: Any) -> StateProfile:
        """Create a new profile."""
        return StateProfile(name=name, **config)

    def register_profile(
        self, profile: Union[StateProfile, str], **config: Any
    ) -> None:
        """Register a new profile globally."""
        if isinstance(profile, str):
            profile = StateProfile(name=profile, **config)

        PROFILES[profile.name] = profile


# Create the main decorator instance
state = FlexibleStateDecorator()

# Create specialized decorators with defaults
minimal_state = state.with_defaults(profile="minimal")
cpu_intensive = state.with_defaults(profile="cpu_intensive")
memory_intensive = state.with_defaults(profile="memory_intensive")
io_intensive = state.with_defaults(profile="io_intensive")
gpu_accelerated = state.with_defaults(profile="gpu_accelerated")
network_intensive = state.with_defaults(profile="network_intensive")
quick_state = state.with_defaults(profile="quick")
batch_state = state.with_defaults(profile="batch")
critical_state = state.with_defaults(profile="critical")
concurrent_state = state.with_defaults(profile="concurrent")
synchronized_state = state.with_defaults(profile="synchronized")

# NEW: Reliability-focused decorators
fault_tolerant = state.with_defaults(profile="fault_tolerant")
external_service = state.with_defaults(profile="external_service")
high_availability = state.with_defaults(profile="high_availability")


def get_profile(name: str) -> Optional[StateProfile]:
    """Get a profile by name."""
    return PROFILES.get(name)


def list_profiles() -> list[str]:
    """List all available profile names."""
    return list(PROFILES.keys())


def create_custom_decorator(**defaults: Any) -> FlexibleStateDecorator:
    """Create a custom decorator with specific defaults."""
    return state.with_defaults(**defaults)
