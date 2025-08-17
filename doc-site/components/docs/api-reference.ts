export const apiReferenceMarkdown = `# API Reference

Complete reference for core PuffinFlow classes, decorators, and coordination utilities.

## Core Classes

### Agent

The main class for creating and executing workflows (state machines).

\`\`\`python
from puffinflow import Agent

class Agent:
    def __init__(
        self,
        name: str,
        resource_pool: Optional[ResourcePool] = None,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        bulkhead_config: Optional[BulkheadConfig] = None,
        max_concurrent: int = 5,
        enable_dead_letter: bool = True,
        state_timeout: Optional[float] = None,
        checkpoint_storage: Optional[CheckpointStorage] = None,
        **kwargs: Any,
    )
\`\`\`

**Key Parameters:**
- \`name\`: Agent identifier.
- \`max_concurrent\`: Max states to execute concurrently.
- \`retry_policy\`: Default \`RetryPolicy\` for states.
- \`enable_dead_letter\`: Send exhausted failures to DLQ.
- \`state_timeout\`: Default per-state timeout.
- \`checkpoint_storage\`: \`FileCheckpointStorage\` or \`MemoryCheckpointStorage\`.

**Primary Methods:**

#### \`add_state(name, func, *, dependencies=None, resources=None, priority=None, retry_policy=None, coordination_primitives=None, max_retries=None) -> None\`
Register an async state function with optional dependencies and execution config.

#### \`run(timeout=None, initial_context=None, execution_mode=ExecutionMode.SEQUENTIAL) -> AgentResult\`
Execute the workflow. Returns \`AgentResult\` with variables, outputs, metadata, and metrics.

#### Common helpers
- \`set_variable/get_variable\`: Manage shared variables.
- \`set_output/get_output\`: Produce and read final outputs.
- \`set_cached/get_cached\`: Cache values with TTL.
- \`save_checkpoint/load_checkpoint\`: Persist/restore execution.

---

### Context

Data sharing and state management within and across states. Exposed to every state as the \`context\` parameter.

#### Variable Management

#### \`set_variable(key: str, value: Any) -> None\`
Stores a variable in the context.

**Parameters:**
- \`key\` (str): Variable name
- \`value\` (Any): Variable value

#### \`get_variable(key: str, default: Any = None) -> Any\`
Retrieves a variable from the context.

**Parameters:**
- \`key\` (str): Variable name
- \`default\` (Any): Default value if key doesn't exist

**Returns:**
- \`Any\`: Variable value or default

#### Other
- \`get_variable_keys() -> set[str]\`: Enumerate variable names.

#### Type-Safe Variables

#### \`set_typed_variable(key: str, value: T) -> None\`
Stores a type-locked variable.

**Parameters:**
- \`key\` (str): Variable name
- \`value\` (T): Variable value (type is locked)

#### \`get_typed_variable(key: str, expected: Optional[type] = None) -> Any\`
Retrieves a type-locked variable.

**Parameters:**
- \`key\` (str): Variable name
- \`type_hint\` (Type[T], optional): Type hint for IDE support

**Returns:**
- \`T\`: Variable value with type guarantee

#### Validated Data

#### \`set_validated_data(key: str, value: BaseModel) -> None\`
Stores Pydantic model data with validation.

**Parameters:**
- \`key\` (str): Variable name
- \`value\` (BaseModel): Pydantic model instance

#### \`get_validated_data(key: str, model_class: Type[BaseModel]) -> Optional[BaseModel]\`
Retrieves and validates Pydantic model data.

**Parameters:**
- \`key\` (str): Variable name
- \`model_class\` (Type[BaseModel]): Pydantic model class

**Returns:**
- \`BaseModel\`: Validated model instance

#### Constants

#### \`set_constant(key: str, value: Any) -> None\`
Stores an immutable constant.

**Parameters:**
- \`key\` (str): Constant name
- \`value\` (Any): Constant value

#### \`get_constant(key: str) -> Any\`
Retrieves a constant value.

**Parameters:**
- \`key\` (str): Constant name

**Returns:**
- \`Any\`: Constant value

#### Secrets Management

#### \`set_secret(key: str, value: str) -> None\`
Stores sensitive data securely.

**Parameters:**
- \`key\` (str): Secret name
- \`value\` (str): Secret value

#### \`get_secret(key: str) -> str\`
Retrieves a secret value.

**Parameters:**
- \`key\` (str): Secret name

**Returns:**
- \`str\`: Secret value

#### Cached Data

#### \`set_cached(key: str, value: Any, ttl: Optional[int] = None) -> None\`
Stores data with time-to-live expiration.

**Parameters:**
- \`key\` (str): Cache key
- \`value\` (Any): Cached value
- \`ttl\` (float): Time-to-live in seconds

#### \`get_cached(key: str, default: Any = None) -> Any\`
Retrieves cached data if not expired.

**Parameters:**
- \`key\` (str): Cache key
- \`default\` (Any): Default value if expired/missing

**Returns:**
- \`Any\`: Cached value or default

#### State-Local Data

#### \`set_state(key: str, value: Any) -> None\`
Stores data local to the current state.

**Parameters:**
- \`key\` (str): State variable name
- \`value\` (Any): State variable value

#### \`get_state(key: str, default: Any = None) -> Any\`
Retrieves state-local data.

**Parameters:**
- \`key\` (str): State variable name
- \`default\` (Any): Default value if not found

**Returns:**
- \`Any\`: State variable value or default

---

## Decorators

### @state

Production-grade decorator for configuring states with resources, reliability, and coordination.

\`\`\`python
from puffinflow import state, Priority

@state(
    cpu=1.0,
    memory=512,
    gpu=0.0,
    io=1.0,
    priority=Priority.NORMAL,
    timeout=None,
    max_retries=3,
    rate_limit=None,
    burst_limit=None,
    # Coordination (any of): mutex=True | semaphore=5 | barrier=3 | lease=30 | quota=100
    mutex=False,
    semaphore=None,
    barrier=None,
    lease=None,
    quota=None,
    # Metadata
    depends_on=[],
    tags={}
)
async def my_state(context):
    ...
\`\`\`

Profiles like \`cpu_intensive\`, \`gpu_accelerated\`, \`io_intensive\` are available via imports from \`puffinflow\`.

---

## Enums and Constants

### Priority

State scheduling priority.

\`\`\`python
from puffinflow import Priority  # IntEnum

# Values (higher is more urgent):
# LOW=0, NORMAL=1, HIGH=2, CRITICAL=3
\`\`\`

**Usage:**
\`\`\`python
@state(priority=Priority.HIGH)
async def high_priority_state(context):
    pass
\`\`\`

---

## Coordination

### AgentTeam

Manages coordinated execution of multiple agents.

\`\`\`python
from puffinflow import AgentTeam

class AgentTeam:
    def __init__(self, agents: List[Agent], name: str = "team")
\`\`\`

**Parameters:**
- \`agents\` (List[Agent]): List of agents to coordinate
- \`name\` (str): Team identifier

**Methods:**

#### \`execute_parallel() -> Dict[str, Context]\`
Executes all agents in parallel.

**Returns:**
- \`Dict[str, Context]\`: Results from each agent

#### \`execute_sequential() -> List[Context]\`
Executes agents one after another.

**Returns:**
- \`List[Context]\`: Ordered results from each agent

**Example:**
\`\`\`python
from puffinflow import Agent, AgentTeam

agent1 = Agent("worker1")
agent2 = Agent("worker2")

team = AgentTeam([agent1, agent2], name="processing_team")
results = await team.execute_parallel()
\`\`\`

### AgentPool

Manages a pool of identical agents for load balancing.

\`\`\`python
from puffinflow import AgentPool

class AgentPool:
    def __init__(self, agent_factory: Callable[[], Agent], size: int = 5)
\`\`\`

**Parameters:**
- \`agent_factory\` (Callable): Function that creates agent instances
- \`size\` (int): Number of agents in the pool

**Methods:**

#### \`submit_task(initial_context: Dict) -> Awaitable[Context]\`
Submits a task to the next available agent.

**Parameters:**
- \`initial_context\` (Dict): Initial context for the task

**Returns:**
- \`Awaitable[Context]\`: Task result

**Example:**
\`\`\`python
def create_worker():
    agent = Agent("worker")

    @agent.state
    async def process_task(context):
        data = context.get_variable("task_data")
        result = await process_data(data)
        context.set_variable("result", result)
        return None

    return agent

pool = AgentPool(create_worker, size=10)
result = await pool.submit_task({"task_data": "work_item"})
\`\`\`

---

## Observability

### Metrics

Prometheus-compatible metrics provider.

\`\`\`python
from puffinflow.core.observability.metrics import PrometheusMetricsProvider
from puffinflow.core.observability.config import MetricsConfig

provider = PrometheusMetricsProvider(MetricsConfig(namespace="puffinflow"))
reqs = provider.counter("requests_total", "Total requests", labels=["route"])
latency = provider.histogram("request_duration_seconds", labels=["route"])
inflight = provider.gauge("inflight_requests")

reqs.record(1, route="/predict")
inflight.record(5)
latency.record(0.234, route="/predict")

# Export Prometheus text format if needed
text = provider.export_metrics()
\`\`\`

### Tracing

OpenTelemetry-compatible tracing provider.

\`\`\`python
from puffinflow.core.observability.tracing import OpenTelemetryTracingProvider
from puffinflow.core.observability.config import TracingConfig

tracer = OpenTelemetryTracingProvider(TracingConfig(service_name="puffinflow-app"))
with tracer.span("state_execute", span_type=SpanType.STATE) as span:
    span.set_attribute("state", "generate_report")
    # ... do work ...
\`\`\`

---

## Configuration

Key configuration lives in constructors and decorators (no global AgentConfig class).
- \`max_concurrent_states\` (int): Maximum states running concurrently
- \`default_timeout\` (float): Default timeout for states
- \`enable_checkpointing\` (bool): Enable automatic checkpointing
- \`checkpoint_interval\` (float): Checkpoint frequency in seconds
- \`enable_metrics\` (bool): Enable metrics collection
- \`enable_tracing\` (bool): Enable distributed tracing
- \`log_level\` (str): Logging level

**Example:**
\`\`\`python
config = AgentConfig(
    max_concurrent_states=20,
    default_timeout=600.0,
    enable_checkpointing=True,
    enable_metrics=True
)

agent = Agent("configured_agent", config=config)
\`\`\`

---

## Error Handling

### Common Exceptions

#### \`StateExecutionError\`
Raised when state execution fails.

\`\`\`python
from puffinflow.exceptions import StateExecutionError

try:
    await agent.run()
except StateExecutionError as e:
    print(f"State '{e.state_name}' failed: {e.message}")
\`\`\`

#### \`ResourceAllocationError\`
Raised when resource allocation fails.

\`\`\`python
from puffinflow.exceptions import ResourceAllocationError

try:
    await agent.run()
except ResourceAllocationError as e:
    print(f"Resource allocation failed: {e.message}")
\`\`\`

#### \`ContextVariableError\`
Raised when context variable operations fail.

\`\`\`python
from puffinflow.exceptions import ContextVariableError

try:
    value = context.get_variable("nonexistent_key")
except ContextVariableError as e:
    print(f"Context error: {e.message}")
\`\`\`

---

## Utilities

### Checkpoint Management

#### \`save_checkpoint(context: Context, filepath: str) -> None\`
Saves workflow state to file.

**Parameters:**
- \`context\` (Context): Context to save
- \`filepath\` (str): Path to save checkpoint

#### \`load_checkpoint(filepath: str) -> Context\`
Loads workflow state from file.

**Parameters:**
- \`filepath\` (str): Path to checkpoint file

**Returns:**
- \`Context\`: Restored context

**Example:**
\`\`\`python
from puffinflow.utils import save_checkpoint, load_checkpoint

# Save checkpoint
save_checkpoint(context, "workflow_checkpoint.json")

# Load checkpoint
restored_context = load_checkpoint("workflow_checkpoint.json")
\`\`\`

---

## Type Hints

Complete type definitions for better IDE support:

\`\`\`python
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from puffinflow import Context, Agent, Priority

# State function signature
StateFunction = Callable[[Context], Awaitable[Optional[Union[str, List[str]]]]]

# Agent factory signature
AgentFactory = Callable[[], Agent]

# Context data types
ContextData = Dict[str, Any]
StateResult = Optional[Union[str, List[str]]]
\`\`\`

This reference covers all major PuffinFlow APIs. For complete implementation details, see the source code and additional documentation sections.
`.trim();
