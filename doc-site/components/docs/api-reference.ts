export const apiReferenceMarkdown = `# API Reference

Complete reference for all Puffinflow classes, methods, and configuration options. Based on the actual framework implementation with practical examples.

---

## Core Classes

### Agent

The main orchestration class for creating and managing workflows with enterprise-grade features.

**File: agent_example.py**

\`\`\`python
     1→from puffinflow import Agent, ExecutionMode, Priority
     2→from puffinflow.core.patterns import RetryPolicy, CircuitBreakerConfig
     3→from puffinflow.core.storage import FileCheckpointStorage
     4→
     5→# Basic agent creation
     6→agent = Agent("my-workflow")
     7→
     8→# Advanced agent with full configuration
     9→advanced_agent = Agent(
    10→    name="production-agent",
    11→    max_concurrent=10,
    12→    enable_dead_letter=True,
    13→    state_timeout=300.0,
    14→    retry_policy=RetryPolicy(
    15→        max_retries=3,
    16→        backoff_strategy="exponential"
    17→    ),
    18→    circuit_breaker_config=CircuitBreakerConfig(
    19→        failure_threshold=5,
    20→        recovery_timeout=60.0
    21→    ),
    22→    checkpoint_storage=FileCheckpointStorage("./checkpoints")
    23→)
\`\`\`

#### Constructor Parameters

\`\`\`python
Agent(
    name: str,
    resource_pool: Optional[ResourcePool] = None,
    retry_policy: Optional[RetryPolicy] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    bulkhead_config: Optional[BulkheadConfig] = None,
    max_concurrent: int = 5,
    enable_dead_letter: bool = True,
    state_timeout: Optional[float] = None,
    checkpoint_storage: Optional[CheckpointStorage] = None
)
\`\`\`

**Parameters:**
- **name** (str): Unique identifier for the agent
- **resource_pool** (ResourcePool): Custom resource allocation pool
- **retry_policy** (RetryPolicy): Default retry behavior for failed states
- **circuit_breaker_config** (CircuitBreakerConfig): Circuit breaker settings
- **bulkhead_config** (BulkheadConfig): Isolation and concurrency limits  
- **max_concurrent** (int): Maximum concurrent state executions
- **enable_dead_letter** (bool): Enable dead letter queue for failed states
- **state_timeout** (float): Global timeout for all states
- **checkpoint_storage** (CheckpointStorage): Persistence backend for checkpoints

#### Key Methods

##### add_state()

Register a state function with dependencies and resource requirements:

**File: add_state_example.py**

\`\`\`python
     1→from puffinflow import Agent, state, Priority
     2→
     3→agent = Agent("workflow-demo")
     4→
     5→# Method 1: Using add_state with parameters
     6→async def fetch_data(context):
     7→    # State implementation
     8→    pass
     9→
    10→agent.add_state(
    11→    name="fetch_data",
    12→    func=fetch_data,
    13→    dependencies=["initialize"],
    14→    cpu=2.0,
    15→    memory=1024,
    16→    timeout=60.0,
    17→    max_retries=3,
    18→    priority=Priority.HIGH
    19→)
    20→
    21→# Method 2: Using @state decorator (recommended)
    22→@state(
    23→    cpu=2.0,
    24→    memory=1024, 
    25→    timeout=60.0,
    26→    priority=Priority.HIGH,
    27→    max_retries=3
    28→)
    29→async def process_data(context):
    30→    # State implementation with decorator
    31→    pass
    32→
    33→agent.add_state("process_data", process_data, dependencies=["fetch_data"])
\`\`\`

##### run()

Execute the agent workflow:

\`\`\`python
# Basic execution
result = await agent.run()

# With execution mode
result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)

# With timeout and starting state
result = await agent.run(
    timeout=300.0,
    start_state="custom_entry_point",
    initial_context={"batch_size": 1000}
)
\`\`\`

**Parameters:**
- **execution_mode** (ExecutionMode): PARALLEL or SEQUENTIAL
- **timeout** (float): Maximum execution time for entire workflow
- **start_state** (str): Custom entry point state
- **initial_context** (dict): Initial context variables

##### Variable Management

Direct agent variable access:

\`\`\`python
# Set variables directly on agent
agent.set_variable("config", {"api_key": "secret123"})
agent.set_shared_variable("global_counter", 0)

# Get variables
config = agent.get_variable("config")
counter = agent.get_shared_variable("global_counter")

# Property-based access
agent.variables["user_id"] = "user123"
user_id = agent.variables["user_id"]
\`\`\`

##### Workflow Control

\`\`\`python
# Pause/resume execution
await agent.pause()
await agent.resume()

# Cancel states
agent.cancel_state("slow_operation")
agent.cancel_all()

# Checkpoint management
checkpoint_id = await agent.save_checkpoint()
success = await agent.load_checkpoint(checkpoint_id)
\`\`\`

---

## State Decorators and Profiles

### @state Decorator

The primary decorator for defining workflow states with resource management:

**File: state_decorators.py**

\`\`\`python
     1→from puffinflow import state, Priority, cpu_intensive, memory_intensive
     2→
     3→# Basic state
     4→@state
     5→async def simple_task(context):
     6→    pass
     7→
     8→# State with resource specification
     9→@state(cpu=4.0, memory=2048, timeout=120.0, priority=Priority.HIGH)
    10→async def intensive_task(context):
    11→    pass
    12→
    13→# Using predefined profiles
    14→@state(profile='cpu_intensive')
    15→async def cpu_heavy_task(context):
    16→    pass
    17→
    18→# Specialized decorators (shorthand)
    19→@cpu_intensive
    20→async def another_cpu_task(context):
    21→    pass
    22→
    23→@memory_intensive
    24→async def memory_heavy_task(context):
    25→    pass
\`\`\`

### Predefined Profiles

Ready-to-use resource profiles for common scenarios:

| Profile | CPU | Memory | Timeout | Use Case |
|---------|-----|--------|---------|----------|
| **minimal** | 0.1 | 50MB | 30s | Lightweight operations |
| **standard** | 1.0 | 100MB | 60s | Default balanced profile |
| **cpu_intensive** | 4.0 | 1024MB | 300s | Computation-heavy tasks |
| **memory_intensive** | 2.0 | 4096MB | 600s | Large data processing |
| **io_intensive** | 1.0 | 256MB | 120s | File/network operations |
| **gpu_accelerated** | 2.0 | 2048MB | 900s | GPU computation |
| **network_intensive** | 1.0 | 512MB | 180s | API calls and networking |
| **quick** | 0.5 | 64MB | 30s | Fast operations |
| **batch** | 2.0 | 1024MB | 3600s | Long-running batch jobs |
| **critical** | 4.0 | 2048MB | 60s | High-priority exclusive tasks |
| **fault_tolerant** | 1.0 | 256MB | 60s | High reliability with retries |

### Specialized Decorators

**File: specialized_decorators.py**

\`\`\`python
     1→from puffinflow import (
     2→    cpu_intensive, memory_intensive, io_intensive,
     3→    gpu_accelerated, network_intensive, critical_state,
     4→    fault_tolerant, external_service
     5→)
     6→
     7→@cpu_intensive  # 4.0 CPU, 1024MB memory
     8→async def machine_learning_training(context):
     9→    pass
    10→
    11→@memory_intensive  # 2.0 CPU, 4096MB memory  
    12→async def large_dataset_processing(context):
    13→    pass
    14→
    15→@gpu_accelerated  # 2.0 CPU, 2048MB memory, 1.0 GPU
    16→async def neural_network_inference(context):
    17→    pass
    18→
    19→@network_intensive  # With circuit breaker
    20→async def api_integration(context):
    21→    pass
    22→
    23→@critical_state  # High priority, exclusive execution
    24→async def emergency_response(context):
    25→    pass
    26→
    27→@fault_tolerant  # Max retries, dead letter queue
    28→async def unreliable_operation(context):
    29→    pass
    30→
    31→@external_service  # API-focused with timeout/retries
    32→async def third_party_integration(context):
    33→    pass
\`\`\`

---

## Context and Data Management

### Context Class

Rich context management with multiple data types and validation:

**File: context_usage.py**

\`\`\`python
     1→from pydantic import BaseModel
     2→from typing import Dict, Any
     3→
     4→class UserProfile(BaseModel):
     5→    user_id: str
     6→    email: str
     7→    preferences: Dict[str, Any]
     8→
     9→@state
    10→async def context_demo(context):
    11→    # Regular variables
    12→    context.set_variable("user_data", {"id": "123", "name": "John"})
    13→    user_data = context.get_variable("user_data")
    14→    
    15→    # Type-checked variables with Pydantic
    16→    profile = UserProfile(user_id="123", email="john@example.com", preferences={})
    17→    context.set_validated_data("profile", profile)
    18→    validated_profile = context.get_validated_data("profile", UserProfile)
    19→    
    20→    # Immutable constants
    21→    context.set_constant("API_VERSION", "v2.1")
    22→    api_version = context.get_constant("API_VERSION")
    23→    
    24→    # Secure secrets (encrypted storage)
    25→    context.set_secret("api_key", "sk-abc123def456")
    26→    api_key = context.get_secret("api_key")
    27→    
    28→    # Cached data with TTL
    29→    context.set_cached("user_permissions", ["read", "write"], ttl=300)  # 5 minutes
    30→    permissions = context.get_cached("user_permissions")
    31→    
    32→    # State outputs (results for other states)
    33→    context.set_output("processed_count", 1500)
    34→    context.set_output("success_rate", 98.5)
    35→    
    36→    # Metadata (non-functional data)
    37→    context.set_metadata("processing_timestamp", "2024-01-15T10:30:00Z")
    38→    context.set_metadata("worker_id", "worker-001")
\`\`\`

### Context Methods Reference

#### Variable Storage Methods

\`\`\`python
# Basic variable operations
context.set_variable(key: str, value: Any)
context.get_variable(key: str, default: Any = None) -> Any
context.has_variable(key: str) -> bool
context.delete_variable(key: str)

# Type-checked variables (with Pydantic models)
context.set_typed_variable(key: str, value: Any)
context.set_validated_data(key: str, model: BaseModel)
context.get_validated_data(key: str, model_class: Type[BaseModel]) -> BaseModel

# Immutable constants
context.set_constant(key: str, value: Any)
context.get_constant(key: str) -> Any

# Secure secrets (encrypted)
context.set_secret(key: str, value: str)
context.get_secret(key: str) -> str

# Cached data with TTL
context.set_cached(key: str, value: Any, ttl: int = 300)
context.get_cached(key: str) -> Any

# State outputs
context.set_output(key: str, value: Any)
context.get_output(key: str) -> Any

# Metadata storage
context.set_metadata(key: str, value: Any)
context.get_metadata(key: str) -> Any
\`\`\`

#### Advanced Context Features

\`\`\`python
# Variable watching (react to changes)
def on_user_change(old_value, new_value):
    print(f"User changed from {old_value} to {new_value}")

context.watch_variable("current_user", on_user_change)

# Bulk operations
context.set_variables({"key1": "value1", "key2": "value2"})
all_vars = context.get_all_variables()

# Context snapshots
snapshot = context.create_snapshot()
context.restore_snapshot(snapshot)
\`\`\`

---

## Multi-Agent Coordination

### Agent Teams

Coordinate multiple agents working together:

**File: team_coordination.py**

\`\`\`python
     1→from puffinflow import Agent, AgentTeam, create_team
     2→
     3→# Create individual agents
     4→data_processor = Agent("data-processor")
     5→validator = Agent("validator") 
     6→reporter = Agent("reporter")
     7→
     8→# Create team
     9→team = create_team("data-pipeline-team")
    10→team.add_agent("processor", data_processor)
    11→team.add_agent("validator", validator)
    12→team.add_agent("reporter", reporter)
    13→
    14→# Define team workflow
    15→team.set_workflow([
    16→    ("processor", "fetch_and_process"),
    17→    ("validator", "validate_results"),
    18→    ("reporter", "generate_report")
    19→])
    20→
    21→# Execute team workflow
    22→team_result = await team.run()
    23→
    24→# Access individual agent results
    25→processor_result = team_result.get_agent_result("processor")
    26→validation_result = team_result.get_agent_result("validator")
\`\`\`

### Agent Groups and Pools

**File: agent_groups.py**

\`\`\`python
     1→from puffinflow import AgentGroup, AgentPool, WorkQueue
     2→
     3→# Parallel execution group
     4→agents = [Agent(f"worker-{i}") for i in range(3)]
     5→group = AgentGroup(agents)
     6→
     7→# Run all agents in parallel
     8→group_result = await group.run_parallel()
     9→
    10→# Process results from all agents
    11→for agent_name, result in group_result.agent_results.items():
    12→    print(f"{agent_name}: {result.status}")
    13→
    14→# Agent pool for work processing
    15→class WorkerAgent(Agent):
    16→    def __init__(self, name):
    17→        super().__init__(name)
    18→        # Add worker-specific states
    19→        pass
    20→
    21→# Create pool of workers
    22→pool = AgentPool(WorkerAgent, pool_size=5)
    23→work_queue = WorkQueue()
    24→
    25→# Add work items to queue
    26→for i in range(100):
    27→    work_queue.add_work({"task_id": i, "data": f"item_{i}"})
    28→
    29→# Process work with pool
    30→pool_results = await pool.process_queue(work_queue)
\`\`\`

### Fluent API

Chain operations with a fluent interface:

\`\`\`python
from puffinflow import Agents

# Fluent agent coordination
result = await (
    Agents([agent1, agent2, agent3])
    .run_parallel()
    .timeout(300.0)
    .with_shared_context({"batch_size": 1000})
    .then(lambda results: combine_results(results))
    .catch(lambda error: handle_error(error))
)

# Sequential chaining
pipeline_result = await (
    Agents([extractor, transformer, loader])
    .run_sequential()
    .with_retry_policy(RetryPolicy(max_retries=3))
    .execute()
)
\`\`\`

---

## Resource Management

### ResourceRequirements

Define precise resource needs for states:

\`\`\`python
from puffinflow.core.resources import ResourceRequirements, ResourceType

requirements = ResourceRequirements(
    cpu_units=4.0,              # 4 CPU cores
    memory_mb=2048.0,           # 2GB memory
    io_weight=2.0,              # I/O priority weight
    network_weight=1.5,         # Network priority weight
    gpu_units=1.0,              # 1 GPU unit
    priority_boost=10,          # Priority adjustment
    timeout=300.0,              # 5 minute timeout
    resource_types=ResourceType.CPU | ResourceType.MEMORY | ResourceType.GPU
)
\`\`\`

### Resource Pools and Allocation

**File: resource_management.py**

\`\`\`python
     1→from puffinflow.core.resources import (
     2→    ResourcePool, FirstFitAllocator, PriorityAllocator
     3→)
     4→
     5→# Create resource pool with allocation strategy
     6→pool = ResourcePool(
     7→    total_cpu=16.0,
     8→    total_memory=32768.0,  # 32GB
     9→    total_gpu=4.0,
    10→    allocator=PriorityAllocator()
    11→)
    12→
    13→# Create agent with custom resource pool
    14→agent = Agent("resource-managed-agent", resource_pool=pool)
    15→
    16→# Resource allocation strategies available:
    17→# - FirstFitAllocator: First available slot
    18→# - BestFitAllocator: Most efficient fit
    19→# - WorstFitAllocator: Largest available slot
    20→# - PriorityAllocator: Priority-based allocation
    21→# - FairShareAllocator: Equal distribution
\`\`\`

---

## Reliability Patterns

### Circuit Breaker

Prevent cascading failures in external service calls:

**File: circuit_breaker.py**

\`\`\`python
     1→from puffinflow.core.patterns import CircuitBreakerConfig
     2→
     3→# Circuit breaker configuration
     4→cb_config = CircuitBreakerConfig(
     5→    failure_threshold=5,        # Open after 5 failures
     6→    recovery_timeout=60.0,      # Wait 60s before retry
     7→    success_threshold=3,        # Close after 3 successes
     8→    timeout=30.0               # Request timeout
     9→)
    10→
    11→# Agent with circuit breaker
    12→agent = Agent("api-client", circuit_breaker_config=cb_config)
    13→
    14→# State-level circuit breaker
    15→@state(
    16→    circuit_breaker=True,
    17→    circuit_breaker_config={
    18→        "failure_threshold": 3,
    19→        "recovery_timeout": 30.0
    20→    }
    21→)
    22→async def external_api_call(context):
    23→    # This state is protected by circuit breaker
    24→    response = await make_api_request()
    25→    return response
\`\`\`

### Bulkhead Pattern

Isolate resources to prevent resource exhaustion:

\`\`\`python
from puffinflow.core.patterns import BulkheadConfig

# Bulkhead configuration
bulkhead_config = BulkheadConfig(
    max_concurrent=3,           # Max 3 concurrent executions
    max_queue_size=10,          # Queue up to 10 waiting requests
    timeout=30.0,               # Queue timeout
    rejection_policy="drop"     # Drop policy when full
)

# State with bulkhead
@state(
    bulkhead=True,
    bulkhead_config={
        "max_concurrent": 2,
        "max_queue_size": 5
    }
)
async def database_operation(context):
    # This state is protected by bulkhead
    pass
\`\`\`

### Retry Policies

Configure intelligent retry behavior:

\`\`\`python
from puffinflow.core.patterns import RetryPolicy

# Exponential backoff retry policy
retry_policy = RetryPolicy(
    max_retries=5,
    backoff_strategy="exponential",  # exponential, linear, fixed
    initial_delay=1.0,
    max_delay=60.0,
    backoff_multiplier=2.0,
    jitter=True,
    dead_letter_on_max_retries=True
)

# Agent with retry policy
agent = Agent("resilient-agent", retry_policy=retry_policy)
\`\`\`

---

## Scheduling System

### Agent Scheduling

Schedule agent execution with flexible timing:

**File: scheduling.py**

\`\`\`python
     1→from puffinflow import Agent
     2→
     3→agent = Agent("scheduled-workflow")
     4→
     5→# Schedule with cron-like syntax
     6→scheduled_execution = agent.schedule(
     7→    "daily at 09:00",
     8→    source="database",
     9→    inputs={"batch_size": 1000}
    10→)
    11→
    12→# Fluent scheduling API
    13→agent.every("5 minutes").with_inputs(
    14→    batch_size=100,
    15→    source="api"
    16→).run()
    17→
    18→# Daily scheduling with secrets
    19→agent.daily("09:00").with_secrets(
    20→    api_key="secret123"
    21→).with_constants(
    22→    timeout=300
    23→).run()
    24→
    25→# Hourly scheduling (30 minutes past hour)
    26→agent.hourly(30).with_inputs(
    27→    check_interval=3600
    28→).run()
\`\`\`

### Magic Prefix Inputs

Use special prefixes to control how input values are stored:

\`\`\`python
# Magic prefix examples
inputs = {
    "secret:api_key": "sk-abc123",           # Stored as secret
    "const:version": "v1.0.0",               # Stored as constant
    "cache:300:user_data": user_profile,     # Cached for 5 minutes
    "typed:config": configuration_object     # Type-checked storage
}

agent.schedule("daily at 06:00", inputs=inputs)
\`\`\`

---

## Observability System

### Observability Decorators

Automatic metrics and tracing for states:

**File: observability.py**

\`\`\`python
     1→from puffinflow.core.observability import observe, trace_state
     2→from puffinflow import state
     3→
     4→# Automatic observability
     5→@observe  # Adds metrics, tracing, and monitoring
     6→@state(cpu=2.0, memory=1024)
     7→async def monitored_task(context):
     8→    # Automatically tracked:
     9→    # - Execution time
    10→    # - Resource usage
    11→    # - Success/failure rates
    12→    # - Custom metrics
    13→    pass
    14→
    15→# Detailed execution tracing
    16→@trace_state  # Distributed tracing
    17→@state
    18→async def traced_operation(context):
    19→    # Creates trace spans for:
    20→    # - State execution
    21→    # - Context operations  
    22→    # - External calls
    23→    pass
    24→
    25→# Combined observability
    26→@observe
    27→@trace_state
    28→@cpu_intensive
    29→async def fully_monitored_task(context):
    30→    # Maximum observability
    31→    pass
\`\`\`

### Custom Metrics

Collect custom business and performance metrics:

\`\`\`python
from puffinflow.core.observability import get_metrics_collector

@state
async def business_metrics_task(context):
    metrics = get_metrics_collector()
    
    # Counter metrics
    metrics.increment_counter("orders_processed", 1, tags={"region": "us-east"})
    
    # Histogram metrics (for distributions)
    metrics.record_histogram("processing_time", 2.5, tags={"operation": "validation"})
    
    # Gauge metrics (for current values)
    metrics.set_gauge("active_connections", 45, tags={"service": "database"})
    
    # Custom business metrics
    metrics.record_business_metric("revenue_generated", 1250.75, currency="USD")
\`\`\`

---

## Persistence and Checkpointing

### Checkpoint Storage

Save and restore workflow state for fault tolerance:

**File: checkpointing.py**

\`\`\`python
     1→from puffinflow.core.storage import FileCheckpointStorage, DatabaseCheckpointStorage
     2→
     3→# File-based checkpoint storage
     4→file_storage = FileCheckpointStorage(
     5→    checkpoint_dir="./checkpoints",
     6→    format="json",  # json, pickle, msgpack
     7→    compression=True
     8→)
     9→
    10→# Database checkpoint storage
    11→db_storage = DatabaseCheckpointStorage(
    12→    connection_string="postgresql://localhost/checkpoints",
    13→    table_name="agent_checkpoints"
    14→)
    15→
    16→# Agent with checkpoint storage
    17→agent = Agent("fault-tolerant-agent", checkpoint_storage=file_storage)
    18→
    19→# Manual checkpoint operations
    20→@state
    21→async def checkpoint_example(context):
    22→    # Save checkpoint before risky operation
    23→    checkpoint_id = await context.agent.save_checkpoint()
    24→    
    25→    try:
    26→        # Risky operation that might fail
    27→        result = await risky_operation()
    28→        return result
    29→    except Exception as e:
    30→        # Restore from checkpoint on failure
    31→        await context.agent.load_checkpoint(checkpoint_id)
    32→        raise e
\`\`\`

### Dead Letter Queue

Handle failed states with dead letter queue:

\`\`\`python
# Agent with dead letter queue enabled
agent = Agent("resilient-agent", enable_dead_letter=True)

# Access dead letter queue
dead_letters = agent.get_dead_letter_queue()

# Process failed states
for failed_state in dead_letters:
    print(f"Failed state: {failed_state.state_name}")
    print(f"Error: {failed_state.error}")
    print(f"Attempts: {failed_state.retry_count}")
    
    # Retry failed state
    if failed_state.retry_count < 5:
        await agent.retry_dead_letter(failed_state.id)
\`\`\`

---

## Execution Modes and Control

### Execution Modes

Control how states are executed within the workflow:

\`\`\`python
from puffinflow import ExecutionMode

# Parallel execution (default)
# All states without dependencies run simultaneously
result = await agent.run(execution_mode=ExecutionMode.PARALLEL)

# Sequential execution  
# States run one after another based on return values
result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)
\`\`\`

### State Flow Control

Control workflow progression through return values:

\`\`\`python
@state
async def flow_control_example(context):
    user_type = context.get_variable("user_type")
    
    if user_type == "premium":
        return "premium_workflow"        # Single next state
    elif user_type == "standard":
        return ["process_basic", "send_email"]  # Multiple parallel states
    else:
        return None                      # End workflow
\`\`\`

### Cross-Agent Transitions

Transition between states across different agents:

\`\`\`python
@state
async def cross_agent_transition(context):
    # Process data locally
    processed_data = await process_locally()
    context.set_variable("processed_data", processed_data)
    
    # Transition to different agent
    return (other_agent, "validate_data")
\`\`\`

---

## Result Handling

### AgentResult

Comprehensive result container with execution details:

\`\`\`python
# Execute agent and get result
result = await agent.run()

# Result properties
print(f"Status: {result.status}")                    # SUCCESS, FAILED, TIMEOUT
print(f"Execution time: {result.execution_time}")    # Total time in seconds
print(f"States executed: {len(result.executed_states)}")

# Access variables from result
final_data = result.get_variable("processed_data")
output_metrics = result.get_output("performance_metrics")

# Execution statistics
stats = result.get_execution_stats()
print(f"Total states: {stats.total_states}")
print(f"Successful: {stats.successful_states}")
print(f"Failed: {stats.failed_states}")
print(f"Average execution time: {stats.avg_execution_time}")

# Error handling
if result.status == "FAILED":
    print(f"Failure reason: {result.error}")
    print(f"Failed state: {result.failed_state}")
    
    # Access failure context
    failure_context = result.get_failure_context()
    print(f"State variables at failure: {failure_context.variables}")
\`\`\`

---

## Configuration Classes

### Priority Levels

Define execution priority for states:

\`\`\`python
from puffinflow import Priority

@state(priority=Priority.CRITICAL)    # Highest priority, preempts others
async def emergency_task(context): pass

@state(priority=Priority.HIGH)        # High priority
async def important_task(context): pass

@state(priority=Priority.NORMAL)      # Default priority
async def regular_task(context): pass

@state(priority=Priority.LOW)         # Low priority, runs when resources available
async def background_task(context): pass
\`\`\`

### Resource Types

Specify which resource types a state requires:

\`\`\`python
from puffinflow.core.resources import ResourceType

@state(
    cpu=2.0,
    memory=1024,
    resource_types=ResourceType.CPU | ResourceType.MEMORY | ResourceType.IO
)
async def multi_resource_task(context):
    # Uses CPU, memory, and I/O resources
    pass
\`\`\`

---

This API reference covers the complete Puffinflow framework with accurate class definitions, method signatures, and practical examples. All code examples are based on the actual framework implementation and demonstrate real-world usage patterns.
`.trim();