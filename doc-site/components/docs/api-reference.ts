export const apiReferenceMarkdown = `# API Reference

Comprehensive reference for all PuffinFlow classes, methods, and functions. This reference covers the complete API surface area including core agent functionality, coordination primitives, observability features, resource management, and reliability patterns, with detailed examples and usage patterns for production workloads.

## Core Classes

### Agent

The main orchestration class for creating and managing workflow agents. Agents are the primary building blocks for workflow orchestration in PuffinFlow, providing sophisticated state management, resource allocation, and execution control.

\`\`\`python
from puffinflow import Agent, AgentConfig, ExecutionMode, Priority

class Agent:
    def __init__(self, name: str, config: Optional[AgentConfig] = None)
\`\`\`

**Parameters:**
- \`name\` (str): Unique identifier for the agent within the application. Used for logging, metrics collection, coordination with other agents, and debugging. Should be descriptive and follow naming conventions (e.g., "data-processor", "ml-inference-agent").
- \`config\` (AgentConfig, optional): Configuration object containing resource limits, execution mode preferences, observability settings, retry policies, and timeout configurations. If not provided, uses system defaults.

**Properties:**
- \`name\` (str): Agent's unique identifier, immutable after creation
- \`status\` (AgentStatus): Current execution status: PENDING (created but not started), RUNNING (actively executing), COMPLETED (finished successfully), FAILED (encountered unrecoverable error), CANCELLED (execution was terminated)
- \`states\` (Dict[str, StateMetadata]): Registry of all registered state functions with their metadata, including dependencies, resource requirements, and execution history
- \`context\` (Context): Shared context instance for data passing between states, thread-safe and type-aware
- \`execution_mode\` (ExecutionMode): Current execution strategy - PARALLEL (states without dependencies run concurrently) or SEQUENTIAL (controlled flow based on return values)
- \`resource_allocation\` (ResourceAllocation): Current resource allocation status including CPU, memory, GPU, and custom resource usage
- \`metrics\` (AgentMetrics): Performance metrics including execution time, state transition counts, error rates, and resource utilization
- \`dependencies\` (Dict[str, List[str]]): State dependency graph showing execution order constraints

**Methods:**

#### \`add_state(name: str, func: Callable, dependencies: Optional[List[str]] = None, **kwargs) -> None\`
Registers a state function with the agent, including optional dependencies and resource requirements. States are the fundamental execution units in PuffinFlow workflows.

**Parameters:**
- \`name\` (str): Unique state identifier within the agent. Must be valid Python identifier and descriptive of the state's purpose
- \`func\` (Callable): Async function to execute for this state. Must accept a Context parameter and optionally return next state name(s)
- \`dependencies\` (List[str], optional): List of state names that must complete successfully before this state can execute. Creates execution order constraints
- \`**kwargs\`: Additional state configuration options:
  - \`cpu\` (float): CPU cores required (0.1-16.0, default: 1.0)
  - \`memory\` (int): Memory in MB (64-32768, default: 512)
  - \`gpu\` (float): GPU units required (0.0-8.0, default: 0.0)
  - \`timeout\` (float): Maximum execution time in seconds (default: None)
  - \`max_retries\` (int): Maximum retry attempts (0-10, default: 0)
  - \`priority\` (Priority): Execution priority (LOW, NORMAL, HIGH, CRITICAL, default: NORMAL)
  - \`rate_limit\` (float): Maximum calls per second (default: None)
  - \`isolation_level\` (IsolationLevel): State isolation level for debugging and testing

**Raises:**
- \`ValueError\`: If state name is invalid, duplicate, or dependencies create cycles
- \`ResourceError\`: If resource requirements exceed system capabilities
- \`ConfigurationError\`: If state configuration is invalid

**Examples:**
\`\`\`python
# Basic state registration with minimal configuration
async def data_validation(context):
    \"\"\"Validate input data format and constraints.\"\"\"
    data = context.get_variable("input_data")
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Input data must be a non-empty list")
    
    validated_count = 0
    for item in data:
        if validate_item(item):
            validated_count += 1
    
    context.set_variable("validated_data", data)
    context.set_metric("validation_success_rate", validated_count / len(data))
    return "data_processing"  # Next state to execute

agent.add_state("data_validation", data_validation)

# Advanced state with full configuration
async def ml_inference(context):
    \"\"\"Perform ML model inference with GPU acceleration.\"\"\"
    import torch
    
    model = context.get_variable("trained_model")
    input_data = context.get_variable("processed_data")
    
    # GPU inference with batch processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    results = []
    batch_size = context.get_variable("batch_size", 32)
    
    with torch.no_grad():
        for i in range(0, len(input_data), batch_size):
            batch = input_data[i:i+batch_size]
            batch_tensor = torch.tensor(batch, device=device)
            
            predictions = model(batch_tensor)
            results.extend(predictions.cpu().numpy().tolist())
            
            # Update progress metrics
            progress = (i + len(batch)) / len(input_data)
            context.set_metric("inference_progress", progress)
    
    context.set_variable("inference_results", results)
    context.set_metric("inference_accuracy", calculate_accuracy(results))
    return "results_postprocessing"

agent.add_state(
    name="ml_inference",
    func=ml_inference,
    dependencies=["data_processing", "model_loading"],  # Must wait for these states
    cpu=2.0,              # Requires 2 CPU cores
    memory=4096,          # Requires 4GB RAM
    gpu=1.0,              # Requires 1 GPU unit
    timeout=300.0,        # 5 minute timeout
    max_retries=2,        # Retry up to 2 times on failure
    priority=Priority.HIGH,  # High priority execution
    rate_limit=10.0,      # Max 10 inferences per second
    isolation_level=IsolationLevel.READ_COMMITTED
)

# State with complex dependencies and error handling
async def data_aggregation(context):
    \"\"\"Aggregate results from multiple parallel processing streams.\"\"\"
    try:
        # Wait for all parallel processing to complete
        stream_a_results = context.get_variable("stream_a_results")
        stream_b_results = context.get_variable("stream_b_results") 
        stream_c_results = context.get_variable("stream_c_results")
        
        # Validate all streams completed successfully
        if not all([stream_a_results, stream_b_results, stream_c_results]):
            raise RuntimeError("Not all processing streams completed successfully")
        
        # Aggregate results with error handling
        aggregated = {
            "total_records": len(stream_a_results) + len(stream_b_results) + len(stream_c_results),
            "stream_a_summary": summarize_results(stream_a_results),
            "stream_b_summary": summarize_results(stream_b_results),
            "stream_c_summary": summarize_results(stream_c_results),
            "aggregation_timestamp": datetime.utcnow().isoformat()
        }
        
        context.set_variable("aggregated_results", aggregated)
        context.set_metric("aggregation_success", True)
        
        return "final_validation"
        
    except Exception as e:
        context.set_variable("aggregation_error", str(e))
        context.set_metric("aggregation_success", False)
        context.add_alert("error", f"Aggregation failed: {str(e)}")
        return "error_handling"  # Route to error handling state

agent.add_state(
    name="data_aggregation",
    func=data_aggregation,
    dependencies=["parallel_stream_a", "parallel_stream_b", "parallel_stream_c"],
    cpu=1.0,
    memory=2048,
    timeout=120.0,
    max_retries=1,
    priority=Priority.NORMAL
)
\`\`\`

#### \`run(initial_context: Optional[Dict] = None, execution_mode: Optional[ExecutionMode] = None, checkpoint_path: Optional[str] = None) -> Context\`
Executes the agent workflow with comprehensive configuration options, checkpointing support, and detailed execution tracking.

**Parameters:**
- \`initial_context\` (Dict, optional): Initial context variables to set before execution. Can include input data, configuration parameters, and external resource handles
- \`execution_mode\` (ExecutionMode, optional): Override default execution mode for this specific run:
  - \`ExecutionMode.PARALLEL\`: States without dependencies execute concurrently, maximizing throughput
  - \`ExecutionMode.SEQUENTIAL\`: States execute in sequence based on return values, providing deterministic flow control
- \`checkpoint_path\` (str, optional): Path for saving execution checkpoints. Enables resumption after interruption
- \`dry_run\` (bool, optional): If True, validates workflow without executing states (default: False)
- \`debug_mode\` (bool, optional): Enable detailed debug logging and state inspection (default: False)
- \`timeout\` (float, optional): Overall workflow timeout in seconds (default: None)
- \`resource_limits\` (ResourceLimits, optional): Override default resource limits for this execution
- \`metrics_enabled\` (bool, optional): Enable detailed metrics collection (default: True)

**Returns:**
- \`Context\`: Final context containing all workflow results, state execution history, performance metrics, and error information

**Raises:**
- \`ExecutionError\`: If workflow execution fails due to state errors, resource exhaustion, or timeout
- \`ValidationError\`: If workflow validation fails in dry run mode
- \`CheckpointError\`: If checkpoint loading or saving fails
- \`ResourceError\`: If required resources are unavailable

**Examples:**
\`\`\`python
# Basic workflow execution
result = await agent.run(initial_context={
    "input_data": [1, 2, 3, 4, 5],
    "batch_size": 100,
    "model_path": "/models/trained_model.pth"
})

# Extract results and metrics
processed_data = result.get_variable("final_results")
execution_time = result.get_metric("total_execution_time")
success_rate = result.get_metric("overall_success_rate")

print(f"Processed {len(processed_data)} items in {execution_time:.2f}s")
print(f"Success rate: {success_rate:.2%}")

# Advanced execution with full configuration
result = await agent.run(
    initial_context={
        "data_source": "postgresql://localhost:5432/production",
        "output_bucket": "s3://my-bucket/results/",
        "processing_config": {
            "batch_size": 1000,
            "parallel_workers": 8,
            "quality_threshold": 0.95
        }
    },
    execution_mode=ExecutionMode.PARALLEL,
    checkpoint_path="/tmp/workflow_checkpoints/",
    dry_run=False,
    debug_mode=True,
    timeout=1800.0,  # 30 minute timeout
    resource_limits=ResourceLimits(
        max_cpu=8.0,
        max_memory=16384,
        max_gpu=2.0
    ),
    metrics_enabled=True
)

# Comprehensive result analysis
print("=== Execution Summary ===")
print(f"Status: {result.get_execution_status()}")
print(f"Duration: {result.get_execution_duration():.2f}s")
print(f"States executed: {len(result.get_executed_states())}")
print(f"Peak memory usage: {result.get_peak_memory_usage():.1f}MB")
print(f"Average CPU usage: {result.get_average_cpu_usage():.1f}%")

# Error handling and debugging
if result.has_errors():
    print("\\n=== Errors ===")
    for error in result.get_errors():
        print(f"State: {error.state_name}")
        print(f"Error: {error.message}")
        print(f"Timestamp: {error.timestamp}")
        print(f"Stack trace: {error.stack_trace}")

# Performance analysis
print("\\n=== Performance Metrics ===")
state_metrics = result.get_state_metrics()
for state_name, metrics in state_metrics.items():
    print(f"{state_name}:")
    print(f"  Execution time: {metrics['execution_time']:.2f}s")
    print(f"  Memory peak: {metrics['peak_memory']:.1f}MB")
    print(f"  CPU usage: {metrics['avg_cpu']:.1f}%")
    print(f"  Retry count: {metrics['retry_count']}")

# Checkpoint and resumption example
checkpoint_path = "/tmp/my_workflow_checkpoint"
try:
    result = await agent.run(
        initial_context={"large_dataset": load_large_dataset()},
        checkpoint_path=checkpoint_path
    )
except ExecutionInterruption as e:
    print(f"Execution interrupted at state: {e.current_state}")
    print("Resuming from checkpoint...")
    
    # Resume from checkpoint
    result = await agent.resume_from_checkpoint(checkpoint_path)
    print(f"Resumed and completed successfully")

# Parallel execution comparison
print("\\n=== Execution Mode Comparison ===")

# Sequential execution
start_time = time.time()
sequential_result = await agent.run(
    initial_context=test_data,
    execution_mode=ExecutionMode.SEQUENTIAL
)
sequential_time = time.time() - start_time

# Parallel execution
start_time = time.time()
parallel_result = await agent.run(
    initial_context=test_data,
    execution_mode=ExecutionMode.PARALLEL
)
parallel_time = time.time() - start_time

print(f"Sequential execution: {sequential_time:.2f}s")
print(f"Parallel execution: {parallel_time:.2f}s")
print(f"Speedup: {sequential_time / parallel_time:.2f}x")
\`\`\`

#### \`add_checkpoint_handler(handler: CheckpointHandler) -> None\`
Registers a custom checkpoint handler for workflow state persistence and recovery.

**Parameters:**
- \`handler\` (CheckpointHandler): Custom checkpoint handler implementing save/load/validate operations

**Example:**
\`\`\`python
class S3CheckpointHandler(CheckpointHandler):
    def __init__(self, bucket_name: str, prefix: str):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client = boto3.client('s3')
    
    async def save_checkpoint(self, checkpoint_data: Dict, checkpoint_id: str) -> str:
        key = f"{self.prefix}/{checkpoint_id}.json"
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=json.dumps(checkpoint_data, default=str)
        )
        return f"s3://{self.bucket_name}/{key}"
    
    async def load_checkpoint(self, checkpoint_path: str) -> Dict:
        # Parse S3 path and load checkpoint data
        # Implementation details...
        pass
    
    async def validate_checkpoint(self, checkpoint_path: str) -> bool:
        # Validate checkpoint integrity
        # Implementation details...
        pass

# Register the custom checkpoint handler
s3_handler = S3CheckpointHandler("my-checkpoints-bucket", "workflow-checkpoints")
agent.add_checkpoint_handler(s3_handler)
\`\`\`

#### \`get_state_graph() -> StateGraph\`
Returns the complete state dependency graph for visualization and analysis.

**Returns:**
- \`StateGraph\`: Graph object containing nodes (states) and edges (dependencies) with execution metadata

**Example:**
\`\`\`python
# Get and analyze state graph
graph = agent.get_state_graph()

print(f"Total states: {len(graph.nodes)}")
print(f"Total dependencies: {len(graph.edges)}")
print(f"Execution paths: {len(graph.get_execution_paths())}")

# Identify critical path
critical_path = graph.get_critical_path()
print(f"Critical path: {' -> '.join(critical_path)}")

# Check for cycles
if graph.has_cycles():
    cycles = graph.detect_cycles()
    print(f"Warning: Dependency cycles detected: {cycles}")

# Export for visualization
graph.export_graphviz("/tmp/workflow.dot")
graph.export_mermaid("/tmp/workflow.mmd")
\`\`\`

### Context

Provides thread-safe, type-aware data sharing and state management across workflow states. The Context class is the backbone of data flow in PuffinFlow, offering multiple storage mechanisms optimized for different use cases.

\`\`\`python
from puffinflow import Context, ValidationModel
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

class Context:
    def __init__(self, 
                 workflow_id: str, 
                 initial_data: Optional[Dict] = None,
                 validation_enabled: bool = True,
                 metrics_enabled: bool = True,
                 isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED)
\`\`\`

**Parameters:**
- \`workflow_id\` (str): Unique workflow identifier for this execution instance
- \`initial_data\` (Dict, optional): Initial context variables to populate
- \`validation_enabled\` (bool): Enable type validation and schema checking (default: True)
- \`metrics_enabled\` (bool): Enable automatic metrics collection (default: True) 
- \`isolation_level\` (IsolationLevel): Transaction isolation level for concurrent access

**Properties:**
- \`workflow_id\` (str): Unique workflow identifier, immutable after creation
- \`execution_id\` (str): Unique execution identifier for this specific run
- \`agent_name\` (str): Name of the agent executing this context
- \`start_time\` (float): Workflow start timestamp (Unix timestamp)
- \`current_state\` (str): Currently executing state name
- \`execution_history\` (List[StateExecution]): Complete history of state executions
- \`resource_usage\` (ResourceUsage): Current and peak resource usage statistics
- \`error_count\` (int): Total number of errors encountered
- \`retry_count\` (int): Total number of retries performed

**Core Variable Management Methods:**

#### \`set_variable(key: str, value: Any, ttl: Optional[float] = None, persistent: bool = False) -> None\`
Stores a variable in the context with optional expiration and persistence.

**Parameters:**
- \`key\` (str): Variable name, must be valid identifier
- \`value\` (Any): Variable value, will be deep-copied for thread safety
- \`ttl\` (float, optional): Time-to-live in seconds, auto-expires after this duration
- \`persistent\` (bool): Whether to persist across checkpoint saves/loads (default: False)

**Raises:**
- \`ValueError\`: If key is invalid or conflicts with reserved names
- \`TypeError\`: If value type is not serializable and persistent=True

**Examples:**
\`\`\`python
# Basic variable storage
context.set_variable("user_id", 12345)
context.set_variable("processing_config", {"batch_size": 100, "timeout": 30})

# Temporary variables with TTL (auto-expire after 1 hour)
context.set_variable("cache_data", expensive_computation_result(), ttl=3600)

# Persistent variables (survive checkpoints and agent restarts)
context.set_variable("workflow_state", {"stage": "processing", "progress": 0.5}, persistent=True)

# Large data structures
large_dataset = load_massive_dataset()
context.set_variable("dataset", large_dataset)  # Automatically deep-copied
\`\`\`

#### \`get_variable(key: str, default: Any = None) -> Any\`
Retrieves a variable from the context with comprehensive error handling.

**Parameters:**
- \`key\` (str): Variable name to retrieve
- \`default\` (Any): Default value if variable doesn't exist or has expired

**Returns:**
- \`Any\`: Variable value, default value, or None

**Raises:**
- \`KeyError\`: If key doesn't exist and no default provided (when strict mode enabled)

**Examples:**
\`\`\`python
# Basic retrieval
user_id = context.get_variable("user_id")
config = context.get_variable("config", {"default": "values"})

# Safe retrieval with type hints
dataset: List[Dict] = context.get_variable("processed_dataset", [])
batch_size: int = context.get_variable("batch_size", 32)

# Complex data structure retrieval
processing_state = context.get_variable("processing_state", {})
current_stage = processing_state.get("stage", "initial")
progress = processing_state.get("progress", 0.0)
\`\`\`

#### \`has_variable(key: str) -> bool\`
Checks if a variable exists and hasn't expired.

**Parameters:**
- \`key\` (str): Variable name to check

**Returns:**
- \`bool\`: True if variable exists and is valid, False otherwise

**Example:**
\`\`\`python
# Conditional processing based on variable existence
if context.has_variable("cached_results"):
    results = context.get_variable("cached_results")
    print("Using cached results")
else:
    results = perform_expensive_computation()
    context.set_variable("cached_results", results, ttl=1800)  # Cache for 30 minutes
\`\`\`

**Advanced Data Management Methods:**

#### \`set_typed_variable(key: str, value: T, type_hint: Type[T]) -> None\`
Stores a variable with strict type enforcement and validation.

**Parameters:**
- \`key\` (str): Variable name
- \`value\` (T): Variable value  
- \`type_hint\` (Type[T]): Expected type for runtime validation

**Raises:**
- \`TypeError\`: If value doesn't match type hint

**Example:**
\`\`\`python
from typing import List, Dict

# Type-safe variable storage
context.set_typed_variable("user_ids", [1, 2, 3, 4], List[int])
context.set_typed_variable("config", {"timeout": 30}, Dict[str, int])

# This will raise TypeError
try:
    context.set_typed_variable("user_ids", "invalid", List[int])  # TypeError!
except TypeError as e:
    print(f"Type validation failed: {e}")
\`\`\`

#### \`set_validated_data(key: str, data: Dict, model: Type[BaseModel]) -> None\`
Stores data with Pydantic model validation for complex structured data.

**Parameters:**
- \`key\` (str): Variable name
- \`data\` (Dict): Data to validate and store
- \`model\` (Type[BaseModel]): Pydantic model for validation

**Example:**
\`\`\`python
from pydantic import BaseModel, validator
from typing import Optional

class UserProfile(BaseModel):
    user_id: int
    email: str
    age: Optional[int] = None
    preferences: Dict[str, Any] = {}
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v
    
    @validator('age')
    def validate_age(cls, v):
        if v is not None and (v < 0 or v > 150):
            raise ValueError('Age must be between 0 and 150')
        return v

# Store validated data
user_data = {
    "user_id": 12345,
    "email": "user@example.com", 
    "age": 25,
    "preferences": {"theme": "dark", "language": "en"}
}

context.set_validated_data("user_profile", user_data, UserProfile)

# Retrieve validated data (automatically typed)
profile: UserProfile = context.get_variable("user_profile")
print(f"User {profile.user_id} prefers {profile.preferences['theme']} theme")
\`\`\`

#### \`set_secret(key: str, value: str, encryption_key: Optional[str] = None) -> None\`
Stores sensitive information with encryption and secure handling.

**Parameters:**
- \`key\` (str): Secret identifier
- \`value\` (str): Secret value to encrypt and store
- \`encryption_key\` (str, optional): Custom encryption key, uses default if not provided

**Security Features:**
- Values are encrypted at rest
- Memory is cleared after use
- Never logged or included in checkpoints
- Automatic key rotation support

**Example:**
\`\`\`python
# Store API keys and credentials securely
context.set_secret("openai_api_key", "sk-1234567890abcdef")
context.set_secret("database_password", "super_secret_password")
context.set_secret("jwt_token", user_jwt_token, encryption_key=custom_key)

# Retrieve secrets (automatically decrypted)
api_key = context.get_secret("openai_api_key")
db_password = context.get_secret("database_password")

# Use secrets safely
async with aiohttp.ClientSession() as session:
    headers = {"Authorization": f"Bearer {context.get_secret('jwt_token')}"}
    async with session.get("https://api.example.com/data", headers=headers) as response:
        data = await response.json()
\`\`\`

#### \`set_cached(key: str, value: Any, ttl: float, cache_strategy: CacheStrategy = CacheStrategy.LRU) -> None\`
Stores data with intelligent caching and automatic cleanup.

**Parameters:**
- \`key\` (str): Cache key
- \`value\` (Any): Value to cache
- \`ttl\` (float): Time-to-live in seconds
- \`cache_strategy\` (CacheStrategy): Eviction strategy (LRU, LFU, FIFO)

**Example:**
\`\`\`python
# Cache expensive computation results
expensive_result = await perform_ml_inference(large_dataset)
context.set_cached("ml_results", expensive_result, ttl=1800)  # 30 minutes

# Cache with different strategies
context.set_cached("frequent_data", data, ttl=3600, cache_strategy=CacheStrategy.LFU)
context.set_cached("temp_data", temp_result, ttl=60, cache_strategy=CacheStrategy.FIFO)

# Automatic cache management
cache_stats = context.get_cache_statistics()
print(f"Cache hit rate: {cache_stats.hit_rate:.2%}")
print(f"Memory usage: {cache_stats.memory_usage_mb:.1f}MB")
\`\`\`

**State Management Methods:**

#### \`set_state(state_name: str, key: str, value: Any) -> None\`
Stores state-specific data for debugging and state isolation.

**Parameters:**
- \`state_name\` (str): Name of the state this data belongs to
- \`key\` (str): State-specific variable name  
- \`value\` (Any): Value to store

**Example:**
\`\`\`python
# Store state-specific debug information
context.set_state("data_processing", "batch_count", 150)
context.set_state("data_processing", "processing_time", 45.2)
context.set_state("data_processing", "memory_peak", 2048)

# Store intermediate results for debugging
context.set_state("ml_inference", "model_confidence", 0.94)
context.set_state("ml_inference", "prediction_distribution", [0.1, 0.8, 0.1])
\`\`\`

#### \`get_state(state_name: str, key: str, default: Any = None) -> Any\`
Retrieves state-specific data.

**Example:**
\`\`\`python
# Retrieve state-specific information
batch_count = context.get_state("data_processing", "batch_count", 0)
confidence = context.get_state("ml_inference", "model_confidence", 0.0)

# Debug state transitions
previous_state_time = context.get_state("previous_state", "execution_time", 0)
current_state_start = time.time()
\`\`\`

**Output Management Methods:**

#### \`set_output(key: str, value: Any, metadata: Optional[Dict] = None) -> None\`
Marks variables as final workflow outputs with optional metadata.

**Parameters:**
- \`key\` (str): Output identifier
- \`value\` (Any): Output value
- \`metadata\` (Dict, optional): Additional metadata about the output

**Example:**
\`\`\`python
# Mark final results as outputs
final_predictions = run_final_inference(processed_data)
context.set_output("predictions", final_predictions, {
    "model_version": "v2.1.0",
    "confidence_threshold": 0.8,
    "prediction_count": len(final_predictions),
    "processing_time": processing_duration
})

# Multiple outputs with different metadata
context.set_output("summary_report", generate_report(), {"format": "json"})
context.set_output("visualization", create_plots(), {"format": "png", "dpi": 300})
context.set_output("raw_metrics", collect_metrics(), {"units": "various"})
\`\`\`

#### \`get_output(key: str) -> Any\`
Retrieves marked workflow outputs.

**Example:**
\`\`\`python
# Retrieve outputs at workflow completion
predictions = context.get_output("predictions")  
report = context.get_output("summary_report")

# Get all outputs
all_outputs = context.get_all_outputs()
for output_name, output_data in all_outputs.items():
    print(f"Output '{output_name}': {type(output_data)} with {len(str(output_data))} chars")
\`\`\`

**Metrics and Monitoring Methods:**

#### \`set_metric(key: str, value: Union[int, float], timestamp: Optional[float] = None) -> None\`
Records workflow metrics for monitoring and analysis.

**Parameters:**
- \`key\` (str): Metric name
- \`value\` (Union[int, float]): Metric value
- \`timestamp\` (float, optional): Metric timestamp, uses current time if not provided

**Example:**
\`\`\`python
# Performance metrics
context.set_metric("processing_speed", 1250.5)  # items per second
context.set_metric("memory_usage_mb", 2048)
context.set_metric("api_response_time", 0.15)   # seconds
context.set_metric("error_rate", 0.02)          # percentage

# Business metrics  
context.set_metric("revenue_generated", 15750.00)
context.set_metric("customer_satisfaction", 4.2)  # out of 5
context.set_metric("conversion_rate", 0.08)        # 8%

# Custom timestamps for historical data
historical_timestamp = datetime(2024, 1, 15).timestamp()
context.set_metric("historical_baseline", 100.0, historical_timestamp)
\`\`\`

#### \`increment_metric(key: str, value: Union[int, float] = 1) -> None\`
Increments a counter metric.

**Example:**
\`\`\`python
# Count events and operations
context.increment_metric("api_calls_made")           # Increment by 1
context.increment_metric("bytes_processed", 1024)   # Increment by specific amount
context.increment_metric("errors_encountered")
context.increment_metric("successful_predictions", batch_size)

# Track cumulative values
for item in batch:
    if process_item(item):
        context.increment_metric("successful_items")
    else:
        context.increment_metric("failed_items")
\`\`\`

#### \`get_metric(key: str, default: Union[int, float] = 0) -> Union[int, float]\`
Retrieves metric values.

**Example:**
\`\`\`python
# Get current metric values
current_speed = context.get_metric("processing_speed", 0.0)
total_errors = context.get_metric("errors_encountered", 0)
memory_usage = context.get_metric("memory_usage_mb", 0)

# Calculate derived metrics
success_rate = context.get_metric("successful_items", 0) / max(1, context.get_metric("total_items", 1))
throughput = context.get_metric("items_processed", 0) / context.get_execution_duration()

print(f"Success rate: {success_rate:.2%}")
print(f"Throughput: {throughput:.1f} items/second")
\`\`\`

**Alert and Logging Methods:**

#### \`add_alert(level: str, message: str, metadata: Optional[Dict] = None) -> None\`
Records alerts and notifications for monitoring systems.

**Parameters:**
- \`level\` (str): Alert level - "info", "warning", "error", "critical"
- \`message\` (str): Alert message
- \`metadata\` (Dict, optional): Additional alert context

**Example:**
\`\`\`python
# Different alert levels
context.add_alert("info", "Processing started", {"batch_size": 1000})
context.add_alert("warning", "High memory usage detected", {
    "current_usage": 15360,
    "threshold": 16384,
    "recommendation": "Consider reducing batch size"
})

context.add_alert("error", "API rate limit exceeded", {
    "api_endpoint": "https://api.example.com/v1/data", 
    "retry_after": 60,
    "current_rate": 1000
})

context.add_alert("critical", "Database connection lost", {
    "connection_pool": "primary",
    "retry_attempts": 3,
    "last_error": "Connection timeout after 30s"
})
\`\`\`

#### \`get_alerts(level: Optional[str] = None) -> List[Alert]\`
Retrieves recorded alerts, optionally filtered by level.

**Example:**
\`\`\`python
# Get all alerts
all_alerts = context.get_alerts()
print(f"Total alerts: {len(all_alerts)}")

# Get alerts by level
critical_alerts = context.get_alerts("critical")
error_alerts = context.get_alerts("error")

# Process alerts for monitoring
for alert in context.get_alerts(["warning", "error", "critical"]):
    send_to_monitoring_system({
        "timestamp": alert.timestamp,
        "level": alert.level,
        "message": alert.message,
        "workflow_id": context.workflow_id,
        "state": context.current_state,
        "metadata": alert.metadata
    })
\`\`\`

**Advanced Context Methods:**

#### \`create_checkpoint() -> str\`
Creates a checkpoint of current context state for recovery.

**Returns:**
- \`str\`: Checkpoint identifier for later restoration

**Example:**
\`\`\`python
# Create checkpoint before risky operation
checkpoint_id = context.create_checkpoint()

try:
    risky_operation()
    expensive_computation()
    context.set_variable("results", computation_results)
except Exception as e:
    # Restore from checkpoint on failure
    context.restore_checkpoint(checkpoint_id)
    context.add_alert("error", f"Operation failed, restored from checkpoint: {e}")
    return "error_handling_state"
\`\`\`

#### \`get_execution_summary() -> ExecutionSummary\`
Returns comprehensive execution summary with metrics and timing information.

**Returns:**
- \`ExecutionSummary\`: Object containing execution statistics, resource usage, and performance metrics

**Example:**
\`\`\`python
# Get detailed execution summary
summary = context.get_execution_summary()

print(f"Workflow: {summary.workflow_id}")
print(f"Duration: {summary.total_duration:.2f}s")
print(f"States executed: {len(summary.executed_states)}")
print(f"Peak memory: {summary.peak_memory_mb:.1f}MB")
print(f"Total CPU time: {summary.total_cpu_time:.2f}s")
print(f"Success rate: {summary.success_rate:.2%}")
print(f"Error count: {summary.error_count}")

# Resource utilization analysis
print("\\nResource Utilization:")
for resource, usage in summary.resource_usage.items():
    print(f"  {resource}: {usage.average:.1f} (peak: {usage.peak:.1f})")

# State execution breakdown
print("\\nState Performance:")
for state_name, metrics in summary.state_metrics.items():
    print(f"  {state_name}: {metrics.duration:.2f}s "
          f"(CPU: {metrics.cpu_usage:.1f}%, Memory: {metrics.memory_usage:.1f}MB)")
\`\`\`

## Decorators and State Functions

### @state Decorator

The primary decorator for defining workflow states with comprehensive resource management and execution control.

\`\`\`python
from puffinflow import state, Priority, ExecutionMode
from typing import Optional, Union, List

@state(
    # Resource requirements
    cpu: float = 1.0,                    # CPU cores (0.1-16.0)
    memory: int = 512,                   # Memory in MB (64-32768)  
    gpu: float = 0.0,                    # GPU units (0.0-8.0)
    disk: int = 0,                       # Temporary disk in MB
    
    # Execution constraints
    timeout: Optional[float] = None,      # Max execution time in seconds
    max_retries: int = 0,                # Retry attempts (0-10)
    priority: Priority = Priority.NORMAL, # Execution priority
    
    # Rate limiting
    rate_limit: Optional[float] = None,   # Max calls per second
    burst_limit: Optional[int] = None,    # Burst capacity
    
    # Advanced options
    isolation_level: IsolationLevel = IsolationLevel.READ_COMMITTED,
    cache_results: bool = False,          # Cache state results
    metrics_enabled: bool = True,         # Enable metrics collection
    checkpoint_enabled: bool = True,      # Enable checkpointing
    
    # Dependencies and coordination
    depends_on: Optional[List[str]] = None,  # State dependencies
    mutex: Optional[str] = None,             # Mutex resource name
    semaphore: Optional[str] = None,         # Semaphore resource name
    
    # Error handling
    on_failure: Optional[str] = None,        # Failure handler state
    on_timeout: Optional[str] = None,        # Timeout handler state
    on_retry: Optional[str] = None,          # Retry handler state
)
async def state_function(context: Context) -> Optional[Union[str, List[str]]]:
    \"\"\"
    State function that processes data and returns next state(s).
    
    Args:
        context: Context object for data sharing and metrics
        
    Returns:
        Optional[Union[str, List[str]]]: Next state name(s) to execute, or None to end
        
    Raises:
        Any exceptions will trigger retry logic and error handling
    \"\"\"
    pass
\`\`\`

**Advanced State Function Examples:**

\`\`\`python
# High-performance data processing state
@state(
    cpu=4.0,                    # Use 4 CPU cores
    memory=8192,               # 8GB memory allocation
    timeout=300.0,             # 5 minute timeout
    max_retries=2,             # Retry up to 2 times
    priority=Priority.HIGH,    # High priority scheduling
    rate_limit=100.0,          # Process max 100 items/second
    cache_results=True,        # Cache results for reuse
    checkpoint_enabled=True    # Enable state checkpointing
)
async def parallel_data_processing(context: Context) -> str:
    \"\"\"Process large dataset with parallel processing and monitoring.\"\"\"
    import asyncio
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    
    # Get input data and configuration
    dataset = context.get_variable("input_dataset")
    batch_size = context.get_variable("batch_size", 1000)
    num_workers = min(mp.cpu_count(), len(dataset) // batch_size)
    
    context.set_metric("dataset_size", len(dataset))
    context.set_metric("batch_size", batch_size)
    context.set_metric("worker_count", num_workers)
    
    # Create data chunks for parallel processing
    chunks = [dataset[i:i+batch_size] for i in range(0, len(dataset), batch_size)]
    
    start_time = time.time()
    processed_results = []
    
    # Process chunks in parallel with progress tracking
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_data_chunk, chunk): i 
            for i, chunk in enumerate(chunks)
        }
        
        completed = 0
        for future in asyncio.as_completed(futures):
            try:
                chunk_result = await future
                processed_results.append(chunk_result)
                completed += 1
                
                # Update progress metrics
                progress = completed / len(chunks)
                context.set_metric("processing_progress", progress)
                context.increment_metric("chunks_completed")
                
                if completed % 10 == 0:  # Log every 10 chunks
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(chunks) - completed) / rate if rate > 0 else 0
                    
                    context.add_alert("info", f"Processing progress: {progress:.1%}", {
                        "completed_chunks": completed,
                        "total_chunks": len(chunks),
                        "processing_rate": f"{rate:.1f} chunks/sec",
                        "eta_seconds": eta
                    })
                
            except Exception as e:
                context.add_alert("error", f"Chunk processing failed: {str(e)}")
                context.increment_metric("failed_chunks")
    
    # Aggregate results and calculate final metrics
    total_items = sum(len(result) for result in processed_results)
    processing_time = time.time() - start_time
    throughput = total_items / processing_time
    
    context.set_variable("processed_data", processed_results)
    context.set_metric("total_items_processed", total_items)
    context.set_metric("processing_time_seconds", processing_time)
    context.set_metric("throughput_items_per_second", throughput)
    context.set_metric("processing_efficiency", throughput / num_workers)
    
    context.add_alert("info", f"Processing completed successfully", {
        "total_items": total_items,
        "processing_time": f"{processing_time:.2f}s",
        "throughput": f"{throughput:.1f} items/sec",
        "efficiency": f"{throughput/num_workers:.1f} items/worker/sec"
    })
    
    return "results_validation"

# ML inference state with GPU acceleration and fallback
@state(
    cpu=2.0,
    memory=4096,
    gpu=1.0,                   # Prefer GPU acceleration
    timeout=120.0,
    max_retries=1,
    priority=Priority.HIGH,
    rate_limit=50.0,           # Limit inference rate
    on_failure="cpu_fallback_inference",  # Fallback to CPU on GPU failure
    cache_results=True,
    metrics_enabled=True
)
async def gpu_ml_inference(context: Context) -> str:
    \"\"\"Perform ML inference with GPU acceleration and automatic fallback.\"\"\"
    import torch
    import torch.nn.functional as F
    
    try:
        # Check GPU availability
        if not torch.cuda.is_available():
            context.add_alert("warning", "GPU not available, falling back to CPU")
            return "cpu_fallback_inference"
        
        device = torch.device("cuda")
        context.set_metric("device_type", "gpu")
        
        # Load model and data
        model = context.get_variable("trained_model").to(device)
        input_data = context.get_variable("inference_data")
        batch_size = context.get_variable("inference_batch_size", 32)
        
        model.eval()
        results = []
        
        start_time = time.time()
        total_batches = len(input_data) // batch_size + (1 if len(input_data) % batch_size else 0)
        
        with torch.no_grad():
            for i, batch_start in enumerate(range(0, len(input_data), batch_size)):
                batch_end = min(batch_start + batch_size, len(input_data))
                batch_data = input_data[batch_start:batch_end]
                
                # Convert to tensor and move to GPU
                batch_tensor = torch.tensor(batch_data, device=device, dtype=torch.float32)
                
                # Perform inference
                batch_start_time = time.time()
                outputs = model(batch_tensor)
                predictions = F.softmax(outputs, dim=1)
                
                # Move results back to CPU and store
                batch_results = predictions.cpu().numpy().tolist()
                results.extend(batch_results)
                
                # Track batch metrics
                batch_time = time.time() - batch_start_time
                context.set_metric(f"batch_{i}_time", batch_time)
                context.set_metric(f"batch_{i}_size", len(batch_data))
                context.increment_metric("batches_processed")
                
                # GPU memory monitoring
                if i % 10 == 0:  # Check every 10 batches
                    gpu_memory_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    gpu_memory_cached = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                    context.set_metric("gpu_memory_used_mb", gpu_memory_used)
                    context.set_metric("gpu_memory_cached_mb", gpu_memory_cached)
                    
                    # Clean up GPU memory if needed
                    if gpu_memory_used > 3072:  # More than 3GB used
                        torch.cuda.empty_cache()
                        context.increment_metric("gpu_cache_clears")
                
                # Progress tracking
                progress = (i + 1) / total_batches
                context.set_metric("inference_progress", progress)
        
        # Clean up GPU resources
        del model, batch_tensor
        torch.cuda.empty_cache()
        
        # Calculate final metrics
        total_time = time.time() - start_time
        inference_rate = len(results) / total_time
        avg_batch_time = total_time / total_batches
        
        context.set_variable("inference_results", results)
        context.set_metric("total_inference_time", total_time)
        context.set_metric("inference_rate", inference_rate)
        context.set_metric("average_batch_time", avg_batch_time)
        context.set_metric("total_predictions", len(results))
        
        context.add_alert("info", f"GPU inference completed", {
            "total_predictions": len(results),
            "inference_time": f"{total_time:.2f}s",
            "inference_rate": f"{inference_rate:.1f} predictions/sec",
            "device": "GPU"
        })
        
        return "results_postprocessing"
        
    except Exception as e:
        context.add_alert("error", f"GPU inference failed: {str(e)}")
        context.set_metric("gpu_inference_failed", True)
        return "cpu_fallback_inference"  # Automatic fallback

# Database operation state with connection management
@state(
    cpu=1.0,
    memory=1024,
    timeout=60.0,
    max_retries=3,
    priority=Priority.NORMAL,
    semaphore="database_connections",  # Limit concurrent DB connections
    on_failure="database_error_handler",
    metrics_enabled=True
)
async def database_operations(context: Context) -> str:
    \"\"\"Perform database operations with connection pooling and error handling.\"\"\"
    import asyncpg
    import asyncio
    from contextlib import asynccontextmanager
    
    @asynccontextmanager
    async def get_db_connection():
        conn = None
        try:
            # Get database connection with timeout
            conn = await asyncpg.connect(
                context.get_secret("database_url"),
                timeout=10.0
            )
            context.increment_metric("db_connections_created")
            yield conn
        except Exception as e:
            context.add_alert("error", f"Database connection failed: {str(e)}")
            context.increment_metric("db_connection_failures")
            raise
        finally:
            if conn:
                await conn.close()
                context.increment_metric("db_connections_closed")
    
    try:
        # Get queries to execute
        queries = context.get_variable("database_queries", [])
        results = []
        
        async with get_db_connection() as conn:
            # Begin transaction
            async with conn.transaction():
                for i, query_info in enumerate(queries):
                    query = query_info["query"]
                    params = query_info.get("params", [])
                    query_type = query_info.get("type", "select")
                    
                    query_start = time.time()
                    
                    try:
                        if query_type == "select":
                            rows = await conn.fetch(query, *params)
                            result = [dict(row) for row in rows]
                        elif query_type in ["insert", "update", "delete"]:
                            result = await conn.execute(query, *params)
                        else:
                            result = await conn.fetchval(query, *params)
                        
                        results.append({
                            "query_id": i,
                            "result": result,
                            "row_count": len(result) if isinstance(result, list) else 1
                        })
                        
                        # Track query metrics
                        query_time = time.time() - query_start
                        context.set_metric(f"query_{i}_time", query_time)
                        context.increment_metric("successful_queries")
                        
                    except Exception as e:
                        context.add_alert("error", f"Query {i} failed: {str(e)}")
                        context.increment_metric("failed_queries")
                        results.append({
                            "query_id": i,
                            "error": str(e),
                            "result": None
                        })
        
        # Store results and final metrics
        total_queries = len(queries)
        successful_queries = context.get_metric("successful_queries", 0)
        success_rate = successful_queries / total_queries if total_queries > 0 else 0
        
        context.set_variable("database_results", results)
        context.set_metric("total_queries_executed", total_queries)
        context.set_metric("query_success_rate", success_rate)
        
        if success_rate >= 0.9:  # 90% success threshold
            context.add_alert("info", f"Database operations completed successfully", {
                "total_queries": total_queries,
                "success_rate": f"{success_rate:.1%}"
            })
            return "data_aggregation"
        else:
            context.add_alert("warning", f"Database operations partially failed", {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "success_rate": f"{success_rate:.1%}"
            })
            return "database_error_handler"
            
    except Exception as e:
        context.add_alert("error", f"Database transaction failed: {str(e)}")
        return "database_error_handler"
\`\`\`

### Specialized State Decorators

PuffinFlow provides pre-configured decorators for common workload patterns:

#### \`@cpu_intensive\`
Pre-configured for CPU-heavy computational tasks.

\`\`\`python
from puffinflow import cpu_intensive

@cpu_intensive(
    cores=4.0,                 # Use 4 CPU cores
    timeout=600.0,             # 10 minute timeout
    priority=Priority.HIGH     # High priority scheduling
)
async def matrix_multiplication(context: Context) -> str:
    \"\"\"Perform large-scale matrix operations.\"\"\"
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    
    matrices = context.get_variable("input_matrices")
    chunk_size = len(matrices) // mp.cpu_count()
    
    def multiply_chunk(matrix_chunk):
        results = []
        for matrix in matrix_chunk:
            result = np.dot(matrix, matrix.T)
            eigenvals = np.linalg.eigvals(result)
            results.append(eigenvals)
        return results
    
    # Parallel processing across multiple CPU cores
    with ProcessPoolExecutor(max_workers=4) as executor:
        chunks = [matrices[i:i+chunk_size] for i in range(0, len(matrices), chunk_size)]
        futures = [executor.submit(multiply_chunk, chunk) for chunk in chunks]
        
        all_results = []
        for future in futures:
            chunk_results = future.result()
            all_results.extend(chunk_results)
    
    context.set_variable("computation_results", all_results)
    context.set_metric("matrices_processed", len(matrices))
    
    return "results_analysis"
\`\`\`

#### \`@gpu_accelerated\`
Pre-configured for GPU-accelerated workloads.

\`\`\`python
from puffinflow import gpu_accelerated

@gpu_accelerated(
    gpu_units=2.0,             # Use 2 GPU units
    gpu_memory_gb=8.0,         # Reserve 8GB GPU memory
    fallback_to_cpu=True,      # Automatic CPU fallback
    timeout=300.0
)
async def deep_learning_training(context: Context) -> str:
    \"\"\"Train deep learning model with GPU acceleration.\"\"\"
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context.set_metric("device_used", str(device))
    
    # Load model and data
    model = context.get_variable("model").to(device)
    train_loader = context.get_variable("train_dataloader")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    epochs = context.get_variable("training_epochs", 10)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 100 == 0:
                context.set_metric(f"epoch_{epoch}_batch_{batch_idx}_loss", loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        context.set_metric(f"epoch_{epoch}_avg_loss", avg_loss)
        
        # GPU memory monitoring
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            context.set_metric(f"epoch_{epoch}_gpu_memory_gb", gpu_memory)
    
    context.set_variable("trained_model", model.cpu())  # Move back to CPU for storage
    return "model_evaluation"
\`\`\`

#### \`@io_intensive\`
Pre-configured for I/O-heavy operations.

\`\`\`python
from puffinflow import io_intensive

@io_intensive(
    concurrent_limit=50,       # Max 50 concurrent I/O operations
    timeout=300.0,             # 5 minute timeout
    rate_limit=100.0,          # Max 100 operations/second
    retry_policy="exponential_backoff"
)
async def bulk_file_processing(context: Context) -> str:
    \"\"\"Process multiple files concurrently with I/O optimization.\"\"\"
    import aiofiles
    import aiohttp
    import asyncio
    from asyncio import Semaphore
    
    file_urls = context.get_variable("file_urls")
    semaphore = Semaphore(50)  # Limit concurrent operations
    
    async def process_single_file(session, url):
        async with semaphore:
            try:
                # Download file
                async with session.get(url) as response:
                    content = await response.read()
                
                # Process file content
                processed_content = await process_file_content(content)
                
                # Save processed file
                output_path = generate_output_path(url)
                async with aiofiles.open(output_path, 'wb') as f:
                    await f.write(processed_content)
                
                context.increment_metric("files_processed_successfully")
                return {"url": url, "status": "success", "output_path": output_path}
                
            except Exception as e:
                context.increment_metric("files_failed")
                context.add_alert("warning", f"File processing failed: {url} - {str(e)}")
                return {"url": url, "status": "failed", "error": str(e)}
    
    # Process all files concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [process_single_file(session, url) for url in file_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Aggregate results
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - successful
    
    context.set_variable("file_processing_results", results)
    context.set_metric("total_files", len(file_urls))
    context.set_metric("successful_files", successful)
    context.set_metric("failed_files", failed)
    context.set_metric("success_rate", successful / len(file_urls))
    
    return "results_summary"
\`\`\`

## Coordination and Multi-Agent Systems

### AgentTeam

Manages multiple agents working together on a shared workflow with coordination and synchronization.

\`\`\`python
from puffinflow import AgentTeam, TeamConfig, CoordinationStrategy

class AgentTeam:
    def __init__(self, 
                 name: str,
                 agents: List[Agent], 
                 config: Optional[TeamConfig] = None,
                 coordination_strategy: CoordinationStrategy = CoordinationStrategy.PARALLEL)
\`\`\`

**Parameters:**
- \`name\` (str): Unique team identifier
- \`agents\` (List[Agent]): List of agents to coordinate
- \`config\` (TeamConfig, optional): Team-level configuration including resource limits and coordination settings
- \`coordination_strategy\` (CoordinationStrategy): How agents coordinate - PARALLEL, SEQUENTIAL, PIPELINE, or CUSTOM

**Key Methods:**

#### \`execute(shared_context: Context, coordination_mode: CoordinationMode = CoordinationMode.COLLABORATIVE) -> TeamResult\`
Executes all team agents with shared context and coordination.

**Example:**
\`\`\`python
# Create specialized agents for different tasks
data_agent = Agent("data_processor")
data_agent.add_state("load_data", load_data_function)
data_agent.add_state("clean_data", clean_data_function)

analysis_agent = Agent("data_analyzer") 
analysis_agent.add_state("analyze_data", analyze_data_function)
analysis_agent.add_state("generate_insights", generate_insights_function)

ml_agent = Agent("ml_processor")
ml_agent.add_state("train_model", train_model_function)
ml_agent.add_state("evaluate_model", evaluate_model_function)

# Create coordinated team
team = AgentTeam(
    name="data_analysis_team",
    agents=[data_agent, analysis_agent, ml_agent],
    coordination_strategy=CoordinationStrategy.PIPELINE
)

# Execute with shared context
shared_context = Context("team_workflow", {
    "dataset_path": "/data/large_dataset.csv",
    "model_config": {"algorithm": "random_forest", "n_estimators": 100}
})

team_result = await team.execute(shared_context, CoordinationMode.COLLABORATIVE)

# Access results from all agents
for agent_name, agent_result in team_result.agent_results.items():
    print(f"Agent {agent_name}: {agent_result.status}")
    print(f"  Duration: {agent_result.execution_time:.2f}s")
    print(f"  States executed: {len(agent_result.executed_states)}")
\`\`\`

### AgentPool

Manages a pool of identical agents for load distribution and high-throughput processing.

\`\`\`python
from puffinflow import AgentPool, LoadBalancingStrategy

class AgentPool:
    def __init__(self, 
                 name: str,
                 agent_template: Agent,
                 pool_size: int,
                 load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
                 auto_scaling: bool = False)
\`\`\`

**Example:**
\`\`\`python
# Create agent template
template_agent = Agent("processing_agent")
template_agent.add_state("process_batch", process_batch_function)
template_agent.add_state("validate_results", validate_results_function)

# Create agent pool for high throughput
agent_pool = AgentPool(
    name="batch_processing_pool",
    agent_template=template_agent,
    pool_size=10,
    load_balancing=LoadBalancingStrategy.LEAST_LOADED,
    auto_scaling=True
)

# Process multiple batches in parallel
batch_tasks = [
    {"batch_id": i, "data": generate_batch_data(i)} 
    for i in range(100)
]

# Distribute tasks across agent pool
pool_results = []
for task in batch_tasks:
    context = Context(f"batch_{task['batch_id']}", task)
    future = agent_pool.submit(context)
    pool_results.append(future)

# Collect results
completed_results = await asyncio.gather(*pool_results)

# Analyze pool performance
pool_stats = agent_pool.get_performance_statistics()
print(f"Total tasks processed: {pool_stats.total_tasks}")
print(f"Average processing time: {pool_stats.avg_processing_time:.2f}s")
print(f"Throughput: {pool_stats.throughput:.1f} tasks/second")
print(f"Pool utilization: {pool_stats.utilization:.1%}")
\`\`\`

### Coordination Primitives

#### Semaphore
Controls concurrent access to limited resources.

\`\`\`python
from puffinflow import Semaphore

# Create semaphore for database connections
db_semaphore = Semaphore("database_connections", max_permits=5)

@state(semaphore="database_connections")
async def database_query(context: Context) -> str:
    # Automatically acquires semaphore permit
    # Max 5 concurrent database operations
    async with get_database_connection() as conn:
        result = await conn.fetch("SELECT * FROM large_table")
        context.set_variable("query_result", result)
    # Permit automatically released
    return "process_results"
\`\`\`

#### Mutex
Provides exclusive access to shared resources.

\`\`\`python
from puffinflow import Mutex

# Create mutex for exclusive file access
file_mutex = Mutex("shared_file_access")

@state(mutex="shared_file_access")
async def update_shared_file(context: Context) -> str:
    # Exclusive access to shared file
    async with aiofiles.open("shared_state.json", "r+") as f:
        current_state = json.loads(await f.read())
        current_state["last_updated"] = datetime.utcnow().isoformat()
        current_state["update_count"] += 1
        
        await f.seek(0)
        await f.write(json.dumps(current_state, indent=2))
        await f.truncate()
    
    return "next_state"
\`\`\`

#### Barrier
Synchronizes multiple agents at specific coordination points.

\`\`\`python
from puffinflow import Barrier

# Create barrier for multi-agent synchronization
sync_barrier = Barrier("processing_checkpoint", required_agents=3)

@state(barrier="processing_checkpoint")
async def synchronized_processing(context: Context) -> str:
    # Complete individual processing first
    individual_result = await process_individual_data(context)
    context.set_variable("individual_result", individual_result)
    
    # Wait for all agents to reach this point
    await sync_barrier.wait()
    
    # Now proceed with coordinated processing
    coordinated_result = await process_coordinated_data(context)
    context.set_variable("coordinated_result", coordinated_result)
    
    return "final_aggregation"
\`\`\`

#### Event
Provides asynchronous signaling between agents.

\`\`\`python
from puffinflow import Event

# Create events for agent coordination
data_ready_event = Event("data_preparation_complete")
model_ready_event = Event("model_training_complete")

# Agent 1: Data preparation
@state()
async def prepare_data(context: Context) -> str:
    dataset = await load_and_preprocess_data()
    context.set_variable("prepared_dataset", dataset)
    
    # Signal that data is ready
    await data_ready_event.set()
    return "wait_for_model"

# Agent 2: Model training (waits for data)
@state()
async def train_model(context: Context) -> str:
    # Wait for data preparation to complete
    await data_ready_event.wait()
    
    dataset = context.get_variable("prepared_dataset")
    trained_model = await train_ml_model(dataset)
    context.set_variable("trained_model", trained_model)
    
    # Signal that model is ready
    await model_ready_event.set()
    return "evaluate_model"

# Agent 3: Inference (waits for model)
@state()
async def run_inference(context: Context) -> str:
    # Wait for model training to complete
    await model_ready_event.wait()
    
    model = context.get_variable("trained_model")
    test_data = context.get_variable("test_dataset")
    
    predictions = await run_model_inference(model, test_data)
    context.set_variable("predictions", predictions)
    
    return "generate_report"
\`\`\`

## Observability and Monitoring

### MetricsCollector

Comprehensive metrics collection and aggregation for workflow monitoring and performance analysis.

\`\`\`python
from puffinflow import MetricsCollector, MetricType, AggregationStrategy

class MetricsCollector:
    def __init__(self, 
                 namespace: str,
                 export_interval: float = 60.0,
                 retention_period: float = 86400.0,
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.TIME_WINDOW)
\`\`\`

**Key Methods:**

#### \`counter(name: str, value: Union[int, float] = 1, tags: Optional[Dict[str, str]] = None) -> None\`
Records counter metrics for tracking events and operations.

#### \`gauge(name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None\`
Records gauge metrics for current system state values.

#### \`histogram(name: str, value: float, buckets: Optional[List[float]] = None, tags: Optional[Dict[str, str]] = None) -> None\`
Records histogram metrics for distribution analysis.

#### \`timer(name: str) -> ContextManager\`
Context manager for timing operations.

**Example:**
\`\`\`python
# Initialize metrics collector
metrics = MetricsCollector(
    namespace="puffinflow_workflow",
    export_interval=30.0,  # Export every 30 seconds
    retention_period=7200.0  # Keep metrics for 2 hours
)

@state(metrics_enabled=True)
async def monitored_processing(context: Context) -> str:
    \"\"\"State with comprehensive metrics collection.\"\"\"
    
    # Counter metrics
    metrics.counter("processing_started", tags={"workflow": "data_pipeline"})
    
    # Gauge metrics
    input_size = len(context.get_variable("input_data"))
    metrics.gauge("input_data_size", input_size, tags={"data_type": "batch"})
    
    # Timer metrics with context manager
    with metrics.timer("data_processing_duration"):
        processed_data = await expensive_data_processing()
    
    # Histogram metrics for distribution analysis
    for item in processed_data:
        processing_time = item.get("processing_time", 0)
        metrics.histogram(
            "item_processing_time", 
            processing_time,
            buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            tags={"item_type": item.get("type", "unknown")}
        )
    
    # Business metrics
    success_rate = calculate_success_rate(processed_data)
    metrics.gauge("processing_success_rate", success_rate, tags={"pipeline": "main"})
    
    # Custom metrics with multiple dimensions
    for category in ["A", "B", "C"]:
        category_items = [item for item in processed_data if item.get("category") == category]
        metrics.gauge(
            "category_item_count", 
            len(category_items),
            tags={"category": category, "workflow": "data_pipeline"}
        )
    
    context.set_variable("processed_data", processed_data)
    return "quality_validation"

# Export metrics to monitoring systems
await metrics.export_to_prometheus("http://prometheus:9090/api/v1/write")
await metrics.export_to_datadog(api_key="your_datadog_api_key")
await metrics.export_to_cloudwatch(region="us-west-2")
\`\`\`

### Distributed Tracing

Provides distributed tracing capabilities for complex workflow debugging and performance analysis.

\`\`\`python
from puffinflow import Tracer, Span, SpanKind

# Initialize distributed tracer
tracer = Tracer(
    service_name="puffinflow_workflow",
    jaeger_endpoint="http://jaeger:14268/api/traces",
    sampling_rate=1.0  # Trace 100% of requests for development
)

@state(tracing_enabled=True)
async def traced_operation(context: Context) -> str:
    \"\"\"State with distributed tracing.\"\"\"
    
    # Start root span
    with tracer.start_span("data_processing", kind=SpanKind.INTERNAL) as span:
        span.set_attribute("workflow.id", context.workflow_id)
        span.set_attribute("agent.name", context.agent_name)
        
        # Add child spans for sub-operations
        with tracer.start_span("load_data", parent=span) as load_span:
            data = await load_data_from_source()
            load_span.set_attribute("data.size", len(data))
            load_span.set_attribute("data.source", "database")
        
        with tracer.start_span("transform_data", parent=span) as transform_span:
            transformed = await transform_data(data)
            transform_span.set_attribute("transformation.type", "normalize")
            transform_span.set_attribute("output.size", len(transformed))
        
        with tracer.start_span("validate_data", parent=span) as validate_span:
            validation_result = await validate_data(transformed)
            validate_span.set_attribute("validation.passed", validation_result.is_valid)
            validate_span.set_attribute("validation.errors", len(validation_result.errors))
            
            if not validation_result.is_valid:
                validate_span.set_status("ERROR", "Data validation failed")
                span.set_status("ERROR", "Processing failed due to validation errors")
                
                # Record error details
                for error in validation_result.errors:
                    span.record_exception(error)
                
                return "error_handling"
        
        # Record success
        span.set_attribute("processing.status", "success")
        span.set_attribute("processing.duration", time.time() - span.start_time)
        
        context.set_variable("processed_data", transformed)
        return "next_processing_stage"
\`\`\`

### Alerting and Monitoring

\`\`\`python
from puffinflow import AlertManager, AlertRule, AlertSeverity, AlertChannel

# Configure alert manager
alert_manager = AlertManager([
    AlertChannel.SLACK(webhook_url="https://hooks.slack.com/services/..."),
    AlertChannel.EMAIL(smtp_config={...}),
    AlertChannel.PAGERDUTY(integration_key="...")
])

# Define alert rules
alert_rules = [
    AlertRule(
        name="high_error_rate",
        condition="error_rate > 0.05",  # More than 5% error rate
        severity=AlertSeverity.CRITICAL,
        channels=[AlertChannel.SLACK, AlertChannel.PAGERDUTY]
    ),
    AlertRule(
        name="slow_processing",
        condition="avg_processing_time > 300",  # More than 5 minutes
        severity=AlertSeverity.WARNING,
        channels=[AlertChannel.SLACK]
    ),
    AlertRule(
        name="resource_exhaustion",
        condition="memory_usage > 0.9",  # More than 90% memory usage
        severity=AlertSeverity.CRITICAL,
        channels=[AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.PAGERDUTY]
    )
]

alert_manager.register_rules(alert_rules)

@state(alerting_enabled=True)
async def monitored_state_with_alerts(context: Context) -> str:
    \"\"\"State with comprehensive alerting.\"\"\"
    
    start_time = time.time()
    errors = 0
    total_operations = 0
    
    try:
        operations = context.get_variable("operations_to_process")
        
        for operation in operations:
            total_operations += 1
            
            try:
                await process_operation(operation)
            except Exception as e:
                errors += 1
                context.add_alert(
                    "error", 
                    f"Operation failed: {str(e)[:200]}", 
                    {"operation_id": operation.get("id"), "error_type": type(e).__name__}
                )
        
        # Calculate metrics for alerting
        processing_time = time.time() - start_time
        error_rate = errors / total_operations if total_operations > 0 else 0
        avg_processing_time = processing_time / total_operations if total_operations > 0 else 0
        
        # Set metrics that alert rules will evaluate
        context.set_metric("error_rate", error_rate)
        context.set_metric("avg_processing_time", avg_processing_time)
        context.set_metric("memory_usage", get_memory_usage_percentage())
        
        # Trigger alerts based on conditions
        if error_rate > 0.1:  # More than 10% errors
            await alert_manager.trigger_alert(
                "critical_error_rate_exceeded",
                f"Error rate {error_rate:.1%} exceeds threshold",
                {
                    "error_rate": error_rate,
                    "total_operations": total_operations,
                    "failed_operations": errors,
                    "workflow_id": context.workflow_id
                }
            )
        
        return "results_analysis"
        
    except Exception as e:
        # Critical system error
        await alert_manager.trigger_alert(
            "system_failure",
            f"Critical system failure: {str(e)}",
            {
                "error_message": str(e),
                "stack_trace": traceback.format_exc(),
                "workflow_id": context.workflow_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            severity=AlertSeverity.CRITICAL
        )
        raise
\`\`\`

Resource management in PuffinFlow enables you to build robust, scalable workflows that efficiently utilize system resources while maintaining predictable performance under varying load conditions. The comprehensive API provides fine-grained control over every aspect of workflow execution, from basic state management to advanced distributed coordination and monitoring.`;