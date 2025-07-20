export const apiReferenceMarkdown = `# API Reference

Comprehensive reference for all PuffinFlow classes, methods, and functions. This reference covers the complete API surface area including core agent functionality, coordination primitives, observability features, resource management, and reliability patterns.

## Core Classes

### Agent

The main orchestration class for creating and managing workflow agents. Agents are the primary building blocks for workflow orchestration in PuffinFlow.

\`\`\`python
from puffinflow import Agent

class Agent:
    def __init__(self, name: str, config: Optional[AgentConfig] = None)
\`\`\`

**Parameters:**
- \`name\` (str): Unique identifier for the agent. Used for logging, metrics, and coordination.
- \`config\` (AgentConfig, optional): Configuration settings for resource limits, execution mode, and observability.

**Properties:**
- \`name\` (str): Agent's unique identifier
- \`status\` (AgentStatus): Current execution status (PENDING, RUNNING, COMPLETED, FAILED)
- \`states\` (Dict[str, StateMetadata]): Registered state functions and their metadata
- \`context\` (Context): Shared context for data passing between states
- \`execution_mode\` (ExecutionMode): Execution strategy (PARALLEL or SEQUENTIAL)

**Methods:**

#### \`add_state(name: str, func: Callable, dependencies: Optional[List[str]] = None, **kwargs) -> None\`
Registers a state function with the agent, including optional dependencies and resource requirements.

**Parameters:**
- \`name\` (str): Unique state identifier within the agent
- \`func\` (Callable): Async function to execute for this state
- \`dependencies\` (List[str], optional): List of state names that must complete before this state can execute
- \`**kwargs\`: Additional state configuration (cpu, memory, timeout, max_retries, priority)

**Example:**
\`\`\`python
# Basic state registration
async def data_processing(context):
    data = context.get_variable("input_data")
    processed = await process_data(data)
    context.set_variable("processed_data", processed)
    return "analysis_state"

agent.add_state("data_processing", data_processing)

# State with dependencies and resources
agent.add_state(
    "analysis_state", 
    analysis_function,
    dependencies=["data_processing"],
    cpu=2.0,
    memory=1024,
    timeout=60.0,
    max_retries=3
)
\`\`\`

#### \`run(initial_context: Optional[Dict] = None, execution_mode: Optional[ExecutionMode] = None) -> Context\`
Executes the agent workflow with optional initial context and execution mode override.

**Parameters:**
- \`initial_context\` (Dict, optional): Initial context variables to set before execution
- \`execution_mode\` (ExecutionMode, optional): Override default execution mode for this run

**Returns:**
- \`Context\`: Final context containing all workflow results and state data

**Example:**
\`\`\`python
# Run with initial data
result = await agent.run(initial_context={
    "input_data": [1, 2, 3, 4, 5],
    "config": {"batch_size": 100}
})
output = result.get_variable("final_result")

# Run in sequential mode
result = await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)

# Run without initial data
result = await agent.run()
\`\`\`

### Context

Provides type-safe data sharing and state management across workflow states.

\`\`\`python
class Context:
    def __init__(self, workflow_id: str, initial_data: Optional[Dict] = None)
\`\`\`

**Properties:**
- \`workflow_id\` (str): Unique workflow identifier
- \`execution_id\` (str): Unique execution identifier for this run
- \`agent_name\` (str): Name of the agent executing this context
- \`start_time\` (float): Workflow start timestamp

**Methods:**

#### \`set_variable(key: str, value: Any) -> None\`
Stores a variable in the context.

**Parameters:**
- \`key\` (str): Variable name
- \`value\` (Any): Variable value

#### \`get_variable(key: str, default: Any = None) -> Any\`
Retrieves a variable from the context.

**Parameters:**
- \`key\` (str): Variable name to retrieve
- \`default\` (Any): Default value if variable doesn't exist

**Returns:**
- \`Any\`: Variable value or default

#### \`has_variable(key: str) -> bool\`
Checks if a variable exists in the context.

**Parameters:**
- \`key\` (str): Variable name to check

**Returns:**
- \`bool\`: True if variable exists, False otherwise

## Decorators

### @state

Primary decorator for defining workflow states with resource requirements.

\`\`\`python
from puffinflow import state, Priority

@state(
    cpu: float = 1.0,
    memory: int = 512,
    timeout: Optional[float] = None,
    max_retries: int = 0,
    priority: Priority = Priority.NORMAL
)
async def state_function(context: Context) -> Optional[Union[str, List[str]]]:
    pass
\`\`\`

### Specialized Decorators

- \`@cpu_intensive\`: Pre-configured for CPU-heavy tasks
- \`@memory_intensive\`: Pre-configured for memory-heavy tasks  
- \`@gpu_accelerated\`: Pre-configured for GPU-accelerated tasks
- \`@io_intensive\`: Pre-configured for I/O-heavy tasks
- \`@network_intensive\`: Pre-configured for network-heavy tasks
- \`@critical_state\`: Pre-configured for critical operations

## Coordination

### AgentTeam
Manages multiple agents working together.

### AgentPool  
Manages a pool of identical agents for load distribution.

### Coordination Primitives
- **Semaphore**: Controls concurrent access to limited resources
- **Mutex**: Provides exclusive access to shared resources
- **Barrier**: Synchronizes multiple agents at specific points
- **Event**: Provides asynchronous signaling between agents

## Observability

### MetricsCollector
Collects and manages workflow metrics.

### Tracing
Distributed tracing for workflow execution.

## Resource Management

### ResourceRequirements
Defines resource requirements for states and agents.

### ResourcePool
Manages shared resource pools across workflows.

### QuotaManager
Manages API quotas and rate limits.

## Reliability Patterns

### CircuitBreaker
Prevents cascade failures by temporarily blocking failing operations.

### Bulkhead
Isolates resources to prevent one failing component from affecting others.

### ResourceLeakDetector
Monitors and detects resource leaks in workflows.

## Execution Modes

- \`ExecutionMode.PARALLEL\`: States without dependencies run concurrently
- \`ExecutionMode.SEQUENTIAL\`: First state runs, flow controlled by return values

## Priority Levels

- \`Priority.LOW\`: Background operations
- \`Priority.NORMAL\`: Standard operations  
- \`Priority.HIGH\`: Important operations
- \`Priority.CRITICAL\`: Critical operations

## Status Types

### AgentStatus
- \`PENDING\`: Agent created but not started
- \`RUNNING\`: Agent currently executing
- \`COMPLETED\`: Agent finished successfully
- \`FAILED\`: Agent execution failed
- \`CANCELLED\`: Agent execution was cancelled

### StateStatus  
- \`PENDING\`: State waiting to execute
- \`RUNNING\`: State currently executing
- \`COMPLETED\`: State finished successfully
- \`FAILED\`: State execution failed
- \`SKIPPED\`: State was skipped
- \`RETRYING\`: State is being retried

## Utility Functions

- \`create_pipeline(*agents)\`: Creates sequential pipeline
- \`create_team(*agents)\`: Creates parallel team
- \`run_agents_parallel(*agents)\`: Runs agents in parallel
- \`run_agents_sequential(*agents)\`: Runs agents sequentially
- \`get_settings()\`: Returns current settings
- \`get_features()\`: Returns enabled features
- \`get_version()\`: Returns version string
- \`get_info()\`: Returns package information

This API reference provides comprehensive coverage of PuffinFlow's functionality. For practical examples and patterns, see the [Getting Started Guide](#docs/getting-started) and [Best Practices](#docs/best-practices).`;