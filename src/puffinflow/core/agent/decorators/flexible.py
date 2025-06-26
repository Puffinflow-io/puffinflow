"""
Flexible state decorator with optional parameters and multiple configuration methods.
"""

import functools
import inspect
from typing import Optional, Dict, Any, List, Union, Callable, Type
from dataclasses import dataclass, field, asdict
from enum import Enum

from src.puffinflow.core.agent.state import Priority
from src.puffinflow.core.resources.requirements import ResourceRequirements, ResourceType
from src.puffinflow.core.coordination.primitives import PrimitiveType
from src.puffinflow.core.coordination.rate_limiter import RateLimitStrategy
from src.puffinflow.core.agent.dependencies import DependencyConfig, DependencyType, DependencyLifecycle


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
    tags: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None and key != 'name':
                result[key] = value
        return result


# Predefined profiles
PROFILES = {
    'minimal': StateProfile(
        name='minimal',
        cpu=0.1,
        memory=50.0,
        description="Minimal resource state for lightweight operations"
    ),
    
    'standard': StateProfile(
        name='standard',
        cpu=1.0,
        memory=100.0,
        description="Standard state configuration"
    ),
    
    'cpu_intensive': StateProfile(
        name='cpu_intensive',
        cpu=4.0,
        memory=1024.0,
        priority=Priority.HIGH,
        timeout=300.0,
        description="CPU-intensive operations"
    ),
    
    'memory_intensive': StateProfile(
        name='memory_intensive',
        cpu=2.0,
        memory=4096.0,
        priority=Priority.HIGH,
        description="Memory-intensive operations"
    ),
    
    'io_intensive': StateProfile(
        name='io_intensive',
        cpu=1.0,
        memory=256.0,
        io=10.0,
        timeout=180.0,
        description="I/O intensive operations"
    ),
    
    'gpu_accelerated': StateProfile(
        name='gpu_accelerated',
        cpu=2.0,
        memory=2048.0,
        gpu=1.0,
        priority=Priority.HIGH,
        timeout=600.0,
        description="GPU-accelerated operations"
    ),
    
    'network_intensive': StateProfile(
        name='network_intensive',
        cpu=1.0,
        memory=512.0,
        network=10.0,
        timeout=120.0,
        description="Network-intensive operations"
    ),
    
    'quick': StateProfile(
        name='quick',
        cpu=0.5,
        memory=50.0,
        timeout=30.0,
        rate_limit=100.0,
        description="Quick, lightweight operations"
    ),
    
    'batch': StateProfile(
        name='batch',
        cpu=2.0,
        memory=1024.0,
        priority=Priority.LOW,
        timeout=1800.0,
        description="Batch processing operations"
    ),
    
    'critical': StateProfile(
        name='critical',
        cpu=2.0,
        memory=512.0,
        priority=Priority.CRITICAL,
        coordination='mutex',
        max_retries=5,
        timeout=60.0,
        description="Critical system operations"
    ),
    
    'concurrent': StateProfile(
        name='concurrent',
        cpu=1.0,
        memory=256.0,
        coordination='semaphore:5',
        description="Concurrent operations with limited parallelism"
    ),
    
    'synchronized': StateProfile(
        name='synchronized',
        cpu=1.0,
        memory=200.0,
        coordination='barrier:3',
        description="Synchronized operations waiting for multiple parties"
    )
}


class FlexibleStateDecorator:
    """
    Flexible state decorator that supports multiple configuration methods.
    """
    
    def __init__(self):
        self.default_config = {}
    
    def __call__(self, *args, **kwargs):
        """
        Handle multiple call patterns:
        - @state
        - @state()
        - @state(profile='cpu_intensive')
        - @state(cpu=2.0, memory=512.0)
        - @state(config={'cpu': 2.0})
        """
        
        # Case 1: @state (direct decoration without parentheses)
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return self._decorate_function(args[0], {})
        
        # Case 2: @state() or @state(params...)
        def decorator(func):
            # Merge all configuration sources
            final_config = self._merge_configurations(*args, **kwargs)
            return self._decorate_function(func, final_config)
        
        return decorator
    
    def _merge_configurations(self, *args, **kwargs) -> Dict[str, Any]:
        """Merge configuration from multiple sources in priority order."""
        final_config = self.default_config.copy()
        
        # FIXED: Apply default profile first if present
        default_profile = self.default_config.get('profile')
        if default_profile and default_profile in PROFILES:
            profile_config = PROFILES[default_profile].to_dict()
            final_config.update(profile_config)

        # Process positional arguments
        for arg in args:
            if isinstance(arg, dict):
                # Direct config dictionary
                final_config.update(arg)
            elif isinstance(arg, str):
                # Profile name
                if arg in PROFILES:
                    profile_config = PROFILES[arg].to_dict()
                    final_config.update(profile_config)
                else:
                    raise ValueError(f"Unknown profile: {arg}")
            elif isinstance(arg, StateProfile):
                # Profile object
                profile_config = arg.to_dict()
                final_config.update(profile_config)
        
        # Process keyword arguments (highest priority)
        # Handle special cases
        config_dict = kwargs.pop('config', {})
        profile_name = kwargs.pop('profile', None)
        
        # Apply profile first
        if profile_name:
            if profile_name in PROFILES:
                profile_config = PROFILES[profile_name].to_dict()
                final_config.update(profile_config)
            else:
                raise ValueError(f"Unknown profile: {profile_name}")
        
        # Apply config dict
        if config_dict:
            final_config.update(config_dict)
        
        # Apply direct keyword arguments (highest priority)
        final_config.update(kwargs)
        
        return final_config
    
    def _decorate_function(self, func: Callable, config: Dict[str, Any]) -> Callable:
        """Apply decoration with merged configuration."""
        
        # Set default values for any missing configuration
        defaults = {
            'cpu': 1.0,
            'memory': 100.0,
            'io': 1.0,
            'network': 1.0,
            'gpu': 0.0,
            'priority': Priority.NORMAL,
            'timeout': None,
            'rate_limit': None,
            'burst_limit': None,
            'coordination': None,
            'depends_on': None,
            'dependency_type': DependencyType.REQUIRED,
            'dependency_lifecycle': DependencyLifecycle.ALWAYS,
            'max_retries': 3,
            'retry_delay': 1.0,
            'retry_exponential': True,
            'retry_jitter': True,
            'tags': {},
            'description': None,
            'preemptible': False,
            'checkpoint_interval': None,
            'health_check': None,
            'cleanup_on_failure': True,
        }
        
        # Merge defaults with provided config
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value
        
        # Process and validate configuration
        config = self._process_configuration(config, func)
        
        # Apply configuration to function
        return self._apply_configuration(func, config)
    
    def _process_configuration(self, config: Dict[str, Any], func: Callable) -> Dict[str, Any]:
        """Process and validate configuration."""
        
        # Normalize priority
        priority = config['priority']
        if isinstance(priority, str):
            config['priority'] = Priority[priority.upper()]
        elif isinstance(priority, int):
            config['priority'] = Priority(priority)
        
        # Process coordination string
        coordination = config.get('coordination')
        if isinstance(coordination, str):
            config.update(self._parse_coordination_string(coordination))
        
        # Normalize dependencies
        depends_on = config.get('depends_on')
        if depends_on:
            if isinstance(depends_on, str):
                config['depends_on'] = [depends_on]
            elif not isinstance(depends_on, list):
                config['depends_on'] = list(depends_on)
        
        # Auto-generate description if not provided
        if not config.get('description'):
            config['description'] = func.__doc__ or f"State: {func.__name__}"
        
        # Process tags
        tags = config.get('tags', {})
        if not isinstance(tags, dict):
            config['tags'] = {}
        else:
            # Add automatic tags
            auto_tags = {
                'function_name': func.__name__,
                'module': func.__module__,
                'decorated_at': 'runtime'  # Could be timestamp
            }
            config['tags'] = {**auto_tags, **tags}
        
        return config
    
    def _parse_coordination_string(self, coordination: str) -> Dict[str, Any]:
        """Parse coordination string like 'mutex', 'semaphore:5', 'barrier:3'."""
        result = {}
        
        if ':' in coordination:
            coord_type, param = coordination.split(':', 1)
            param = int(param)
        else:
            coord_type = coordination
            param = None
        
        coord_type = coord_type.lower()
        
        if coord_type == 'mutex':
            result['mutex'] = True
        elif coord_type == 'semaphore':
            result['semaphore'] = param or 1
        elif coord_type == 'barrier':
            result['barrier'] = param or 2
        elif coord_type == 'lease':
            result['lease'] = param or 30.0
        elif coord_type == 'quota':
            result['quota'] = param or 100.0
        else:
            raise ValueError(f"Unknown coordination type: {coord_type}")
        
        return result
    
    def _apply_configuration(self, func: Callable, config: Dict[str, Any]) -> Callable:
        """Apply the final configuration to the function."""
        
        # Create resource requirements
        resource_types = ResourceType.NONE
        if config['cpu'] > 0:
            resource_types |= ResourceType.CPU
        if config['memory'] > 0:
            resource_types |= ResourceType.MEMORY
        if config['io'] > 0:
            resource_types |= ResourceType.IO
        if config['network'] > 0:
            resource_types |= ResourceType.NETWORK
        if config['gpu'] > 0:
            resource_types |= ResourceType.GPU
        
        requirements = ResourceRequirements(
            cpu_units=config['cpu'],
            memory_mb=config['memory'],
            io_weight=config['io'],
            network_weight=config['network'],
            gpu_units=config['gpu'],
            timeout=config['timeout'],
            resource_types=resource_types,
            priority_boost=config['priority'].value
        )
        
        # Create dependency configurations
        dependency_configs = {}
        deps = config.get('depends_on', [])
        if deps:
            for dep in deps:
                dependency_configs[dep] = DependencyConfig(
                    type=config['dependency_type'],
                    lifecycle=config['dependency_lifecycle'],
                    timeout=config.get('dependency_timeout')
                )
        
        # Determine coordination primitive
        coordination_primitive = None
        coordination_config = {}
        
        if config.get('mutex'):
            coordination_primitive = PrimitiveType.MUTEX
            coordination_config = {'ttl': 30.0}
        elif config.get('semaphore') is not None:
            coordination_primitive = PrimitiveType.SEMAPHORE
            coordination_config = {'max_count': config['semaphore'], 'ttl': 30.0}
        elif config.get('barrier') is not None:
            coordination_primitive = PrimitiveType.BARRIER
            coordination_config = {'parties': config['barrier']}
        elif config.get('lease') is not None:
            coordination_primitive = PrimitiveType.LEASE
            coordination_config = {'ttl': config['lease'], 'auto_renew': True}
        elif config.get('quota') is not None:
            coordination_primitive = PrimitiveType.QUOTA
            coordination_config = {'limit': config['quota']}
        
        # Store all configuration as function attributes
        func._puffinflow_state = True
        func._state_config = config
        func._resource_requirements = requirements
        func._dependency_configs = dependency_configs
        func._coordination_primitive = coordination_primitive
        func._coordination_config = coordination_config
        func._priority = config['priority']
        
        # Store rate limiting
        if config.get('rate_limit') is not None:
            func._rate_limit = config['rate_limit']
            func._burst_limit = config.get('burst_limit') or int(config['rate_limit'] * 2)
            func._rate_strategy = config.get('rate_strategy', RateLimitStrategy.TOKEN_BUCKET)
        
        # Store metadata
        func._state_name = func.__name__
        func._state_description = config['description']
        func._state_tags = config['tags']
        func._preemptible = config['preemptible']
        func._checkpoint_interval = config.get('checkpoint_interval')
        func._health_check = config.get('health_check')
        func._cleanup_on_failure = config['cleanup_on_failure']
        
        # Preserve function metadata
        functools.update_wrapper(func, func)
        
        return func
    
    def with_defaults(self, **defaults) -> 'FlexibleStateDecorator':
        """Create a new decorator with different default values."""
        new_decorator = FlexibleStateDecorator()
        new_decorator.default_config = {**self.default_config, **defaults}
        return new_decorator
    
    def create_profile(self, name: str, **config) -> StateProfile:
        """Create a new profile."""
        return StateProfile(name=name, **config)
    
    def register_profile(self, profile: Union[StateProfile, str], **config):
        """Register a new profile globally."""
        if isinstance(profile, str):
            profile = StateProfile(name=profile, **config)
        PROFILES[profile.name] = profile


# Create the main decorator instance
state = FlexibleStateDecorator()

# Create specialized decorators with defaults
minimal_state = state.with_defaults(profile='minimal')
cpu_intensive = state.with_defaults(profile='cpu_intensive')
memory_intensive = state.with_defaults(profile='memory_intensive')
io_intensive = state.with_defaults(profile='io_intensive')
gpu_accelerated = state.with_defaults(profile='gpu_accelerated')
network_intensive = state.with_defaults(profile='network_intensive')
quick_state = state.with_defaults(profile='quick')
batch_state = state.with_defaults(profile='batch')
critical_state = state.with_defaults(profile='critical')
concurrent_state = state.with_defaults(profile='concurrent')
synchronized_state = state.with_defaults(profile='synchronized')


def get_profile(name: str) -> Optional[StateProfile]:
    """Get a profile by name."""
    return PROFILES.get(name)


def list_profiles() -> List[str]:
    """List all available profile names."""
    return list(PROFILES.keys())


def create_custom_decorator(**defaults) -> FlexibleStateDecorator:
    """Create a custom decorator with specific defaults."""
    return state.with_defaults(**defaults)