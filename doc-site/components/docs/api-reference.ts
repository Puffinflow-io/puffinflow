export const apiReferenceMarkdown = `# API Reference

Complete reference for all PuffinFlow classes, methods, and functions.

## Core Classes

### Agent

The main class for creating and managing workflow agents.

\`\`\`python
from puffinflow import Agent

class Agent:
    def __init__(self, name: str, config: Optional[AgentConfig] = None)
\`\`\`

**Parameters:**
- \`name\` (str): Unique identifier for the agent
- \`config\` (AgentConfig, optional): Configuration settings

**Methods:**

#### \`add_state(name: str, func: Callable, dependencies: Optional[List[str]] = None) -> None\`
Registers a state function with the agent.

**Parameters:**
- \`name\` (str): Unique state identifier
- \`func\` (Callable): Async function to execute
- \`dependencies\` (List[str], optional): List of state names that must complete first

**Example:**
\`\`\`python
async def my_state(context):
    return "next_state"

agent.add_state("my_state", my_state)
agent.add_state("dependent_state", other_func, dependencies=["my_state"])
\`\`\`

#### \`run(initial_context: Optional[Dict] = None) -> Context\`
Executes the agent workflow.

**Parameters:**
- \`initial_context\` (Dict, optional): Initial context variables

**Returns:**
- \`Context\`: Final context containing all workflow results

**Example:**
\`\`\`python
result = await agent.run(initial_context={"input": "data"})
output = result.get_variable("output")
\`\`\`

#### \`state(func: Optional[Callable] = None, **kwargs) -> Callable\`
Decorator to register state functions directly.

**Parameters:**
- \`func\` (Callable, optional): Function to decorate
- \`**kwargs\`: Resource and configuration options

**Example:**
\`\`\`python
@agent.state(cpu=2.0, memory=1024)
async def my_state(context):
    return "next_state"
\`\`\`

---

### Context

Provides data sharing and state management across workflow states.

\`\`\`python
class Context:
    def __init__(self, workflow_id: str, initial_data: Optional[Dict] = None)
\`\`\`

**Properties:**
- \`workflow_id\` (str): Unique workflow identifier
- \`execution_id\` (str): Unique execution identifier

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

#### \`has_variable(key: str) -> bool\`
Checks if a variable exists in the context.

**Parameters:**
- \`key\` (str): Variable name

**Returns:**
- \`bool\`: True if variable exists

#### \`clear_variable(key: str) -> None\`
Removes a variable from the context.

**Parameters:**
- \`key\` (str): Variable name to remove

#### Type-Safe Variables

#### \`set_typed_variable(key: str, value: T) -> None\`
Stores a type-locked variable.

**Parameters:**
- \`key\` (str): Variable name
- \`value\` (T): Variable value (type is locked)

#### \`get_typed_variable(key: str, type_hint: Optional[Type[T]] = None) -> T\`
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

#### \`get_validated_data(key: str, model_class: Type[BaseModel]) -> BaseModel\`
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

#### \`set_cached(key: str, value: Any, ttl: float) -> None\`
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

Decorator for configuring state functions with resource management and behavior options.

\`\`\`python
from puffinflow import state

@state(
    cpu: float = 1.0,
    memory: int = 512,
    gpu: float = 0.0,
    io: float = 1.0,
    priority: Priority = Priority.NORMAL,
    timeout: float = 300.0,
    max_retries: int = 0,
    retry_delay: float = 1.0,
    rate_limit: float = 0.0,
    burst_limit: int = 0,
    preemptible: bool = False
)
async def my_state(context: Context) -> Optional[Union[str, List[str]]]
\`\`\`

**Parameters:**

#### Resource Allocation
- \`cpu\` (float): CPU units to allocate (default: 1.0)
- \`memory\` (int): Memory in MB to allocate (default: 512)
- \`gpu\` (float): GPU units to allocate (default: 0.0)
- \`io\` (float): I/O bandwidth units (default: 1.0)

#### Execution Control
- \`priority\` (Priority): Execution priority (default: Priority.NORMAL)
- \`timeout\` (float): Maximum execution time in seconds (default: 300.0)
- \`preemptible\` (bool): Allow preemption for higher priority tasks (default: False)

#### Retry Configuration
- \`max_retries\` (int): Maximum retry attempts (default: 0)
- \`retry_delay\` (float): Delay between retries in seconds (default: 1.0)

#### Rate Limiting
- \`rate_limit\` (float): Operations per second limit (default: 0.0 = no limit)
- \`burst_limit\` (int): Burst capacity above rate limit (default: 0)

**Example:**
\`\`\`python
@state(
    cpu=2.0,
    memory=1024,
    priority=Priority.HIGH,
    max_retries=3,
    timeout=60.0
)
async def important_task(context):
    # High-priority task with retries
    result = await critical_operation()
    context.set_variable("result", result)
    return "next_state"
\`\`\`

---

## Enums and Constants

### Priority

Defines execution priority levels for states.

\`\`\`python
from puffinflow import Priority

class Priority(Enum):
    CRITICAL = 5
    HIGH = 4
    NORMAL = 3
    LOW = 2
    BACKGROUND = 1
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

### MetricsCollector

Collects and tracks performance metrics.

\`\`\`python
from puffinflow.observability import MetricsCollector

class MetricsCollector:
    def __init__(self, namespace: str = "puffinflow")
\`\`\`

**Methods:**

#### \`increment(metric_name: str, value: float = 1.0, tags: Optional[Dict] = None) -> None\`
Increments a counter metric.

#### \`gauge(metric_name: str, value: float, tags: Optional[Dict] = None) -> None\`
Sets a gauge metric value.

#### \`timer(metric_name: str, tags: Optional[Dict] = None) -> ContextManager\`
Context manager for timing operations.

**Example:**
\`\`\`python
metrics = MetricsCollector()

@state
async def monitored_state(context):
    metrics.increment("state_executions")
    
    with metrics.timer("processing_time"):
        result = await process_data()
    
    metrics.gauge("result_size", len(result))
    return "next_state"
\`\`\`

---

## Configuration

### AgentConfig

Configuration settings for agent behavior.

\`\`\`python
from puffinflow import AgentConfig

class AgentConfig:
    def __init__(
        self,
        max_concurrent_states: int = 10,
        default_timeout: float = 300.0,
        enable_checkpointing: bool = True,
        checkpoint_interval: float = 30.0,
        enable_metrics: bool = True,
        enable_tracing: bool = False,
        log_level: str = "INFO"
    )
\`\`\`

**Parameters:**
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