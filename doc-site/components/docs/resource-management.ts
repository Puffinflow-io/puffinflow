export const resourceManagementMarkdown = `# Resource Management

Puffinflow provides sophisticated resource management to ensure optimal system utilization, prevent resource exhaustion, and maintain fair allocation across workflows. This comprehensive guide covers resource allocation strategies, quota management, priority systems, coordination mechanisms, and advanced optimization techniques essential for building production-ready AI workflows that scale efficiently and cost-effectively.

## Resource Management Philosophy

Effective resource management in Puffinflow is built on these core principles:

- **Predictable Performance**: Resource constraints ensure consistent execution times under varying load, enabling reliable SLA commitments
- **Fair Allocation**: No single workflow can monopolize system resources, preventing resource starvation and ensuring equitable access
- **Graceful Degradation**: System maintains responsiveness even under high resource pressure through intelligent prioritization and queuing
- **Cost Optimization**: Intelligent resource allocation minimizes waste and cloud costs through efficient utilization and auto-scaling
- **Observability**: Comprehensive metrics for resource usage monitoring, profiling, and optimization with real-time dashboards
- **Elasticity**: Dynamic resource scaling based on workload patterns and demand forecasting
- **Fault Tolerance**: Resource management continues to function correctly even when individual components fail

## Understanding Resource Types

Puffinflow manages several types of resources, each with different characteristics, constraints, and optimization strategies:

### Core System Resources

| Resource Type | Description | Unit | Typical Constraints | Allocation Strategy | Monitoring |
|---------------|-------------|------|-------------------|-------------------|------------|
| **CPU** | Processing power | Cores (float) | 0.1 - 16.0 cores per state | Share-based allocation | CPU utilization, load average |
| **Memory** | RAM allocation | MB (int) | 64 - 32768 MB per state | Reserved allocation | Memory usage, garbage collection |
| **GPU** | Graphics processing | Units (float) | 0.0 - 8.0 GPU units | Exclusive or shared | GPU utilization, memory usage |
| **Disk** | Storage space | MB (int) | 0 - 10240 MB temporary storage | Copy-on-write, cleanup | I/O operations, space usage |
| **Network** | Bandwidth allocation | Mbps (float) | 1.0 - 1000.0 Mbps | QoS-based limiting | Throughput, latency metrics |

### External Resource Quotas

| Resource Type | Description | Constraints | Use Cases | Rate Limiting | Cost Tracking |
|---------------|-------------|-------------|-----------|---------------|---------------|
| **API Quotas** | External service limits | Requests/second, daily limits | OpenAI, AWS, GCP APIs | Token bucket, sliding window | Per-API cost tracking |
| **Database Connections** | Connection pool limits | Concurrent connections | PostgreSQL, MongoDB | Connection pooling | Connection duration metrics |
| **File Handles** | Open file descriptors | File count, size limits | Large file processing | File handle pooling | File operation metrics |
| **Custom Resources** | Domain-specific limits | User-defined constraints | License seats, custom quotas | Custom rate limiting | Usage-based billing |

## Detailed Resource Configuration

### Core Resource Allocation

#### CPU Resource Management

CPU resources in Puffinflow are managed through a sophisticated allocation system that balances performance with system stability:

\`\`\`python
from puffinflow import state, CPUProfile, Priority

# Different CPU allocation patterns
@state(cpu=0.25, cpu_profile=CPUProfile.BURST)  # Short bursts of activity
async def quick_validation(context):
    # Lightweight validation tasks
    data = context.get_variable("input_data")
    if not data or len(data) == 0:
        raise ValueError("Input data is empty")
    context.set_variable("validated", True)

@state(cpu=2.0, cpu_profile=CPUProfile.SUSTAINED)  # Sustained CPU usage
async def data_processing(context):
    # CPU-intensive data processing
    import pandas as pd
    data = context.get_variable("raw_data")
    
    # Process large dataset with multiple CPU cores
    processed = data.groupby("category").agg({
        "value": ["mean", "std", "count"],
        "timestamp": ["min", "max"]
    }).reset_index()
    
    context.set_variable("processed_data", processed)

@state(cpu=8.0, cpu_profile=CPUProfile.PARALLEL, cpu_affinity="numa-aware")
async def parallel_computation(context):
    # High-performance parallel computation with NUMA awareness
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    
    data_chunks = context.get_variable("data_chunks")
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = await asyncio.gather(*[
            executor.submit(process_chunk, chunk) 
            for chunk in data_chunks
        ])
    
    context.set_variable("computation_results", results)
\`\`\`

#### Memory Resource Management

Memory allocation in Puffinflow includes both reservation and monitoring capabilities:

\`\`\`python
from puffinflow import state, MemoryProfile

@state(
    memory=512,                           # Reserve 512MB
    memory_profile=MemoryProfile.STABLE,  # Steady memory usage
    memory_limit=1024,                    # Hard limit at 1GB
    memory_swap=False                     # Disable swap usage
)
async def data_analysis(context):
    """Memory-managed data analysis with monitoring."""
    import psutil
    import gc
    
    # Get initial memory baseline
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    context.set_metric("memory_baseline_mb", initial_memory)
    
    # Load and process data within memory constraints
    large_dataset = context.get_variable("large_dataset")
    
    # Process data in chunks to stay within memory limits
    chunk_size = 10000
    results = []
    
    for i in range(0, len(large_dataset), chunk_size):
        chunk = large_dataset[i:i+chunk_size]
        processed_chunk = analyze_chunk(chunk)
        results.append(processed_chunk)
        
        # Monitor memory usage and trigger garbage collection if needed
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        if current_memory > 800:  # Approaching limit
            gc.collect()
            context.increment_metric("gc_triggered_count")
    
    context.set_variable("analysis_results", results)
    context.set_metric("peak_memory_mb", 
                      max(context.get_metric("peak_memory_mb", 0), current_memory))

@state(
    memory=4096,                          # Large memory allocation
    memory_profile=MemoryProfile.GROWING, # Memory usage grows over time
    memory_monitoring=True,               # Enable detailed monitoring
    oom_handler="graceful_failure"        # Handle out-of-memory gracefully
)
async def ml_training(context):
    """ML model training with sophisticated memory management."""
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    
    # Configure PyTorch memory management
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    model = context.get_variable("model")
    training_data = context.get_variable("training_data")
    
    # Use memory-efficient data loading
    dataloader = DataLoader(
        training_data, 
        batch_size=32,
        pin_memory=True,
        num_workers=2
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        epoch_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Memory monitoring and cleanup
            if batch_idx % 100 == 0:
                if torch.cuda.is_available():
                    context.set_metric(f"gpu_memory_allocated_mb", 
                                     torch.cuda.memory_allocated() / 1024 / 1024)
                    torch.cuda.empty_cache()
        
        context.set_metric(f"epoch_{epoch}_loss", epoch_loss / len(dataloader))
    
    context.set_variable("trained_model", model)
\`\`\`

#### GPU Resource Management

GPU resources require special handling due to their specialized nature and high cost:

\`\`\`python
from puffinflow import state, GPUProfile, GPUMemoryStrategy

@state(
    gpu=1.0,                                    # Full GPU allocation
    gpu_profile=GPUProfile.INFERENCE,           # Optimized for inference
    gpu_memory_strategy=GPUMemoryStrategy.LAZY, # Lazy memory allocation
    gpu_memory_fraction=0.8,                    # Use 80% of GPU memory
    gpu_device_id=0                             # Specific GPU device
)
async def gpu_inference(context):
    """GPU-accelerated model inference with memory optimization."""
    import torch
    import torch.nn.functional as F
    
    # Ensure we're using the correct GPU
    device = torch.device(f"cuda:{context.get_resource_allocation().gpu_device_id}")
    
    model = context.get_variable("model").to(device)
    input_batch = context.get_variable("input_batch")
    
    # Configure memory optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    results = []
    batch_size = 32
    
    with torch.inference_mode():  # More efficient than no_grad for inference
        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i:i+batch_size]
            batch_tensor = torch.tensor(batch, device=device)
            
            # Run inference
            outputs = model(batch_tensor)
            predictions = F.softmax(outputs, dim=1)
            
            # Move back to CPU and append results
            results.extend(predictions.cpu().numpy().tolist())
            
            # Monitor GPU memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(device) / 1024 / 1024
                context.set_metric("gpu_memory_used_mb", memory_used)
    
    context.set_variable("inference_results", results)

@state(
    gpu=0.5,                                    # Shared GPU allocation
    gpu_profile=GPUProfile.TRAINING,            # Training workload
    gpu_memory_strategy=GPUMemoryStrategy.PREALLOC, # Pre-allocate memory
    gpu_compute_capability="7.5+"               # Require modern GPU
)
async def distributed_training(context):
    """Distributed GPU training with resource sharing."""
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    # Initialize distributed training
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    
    model = context.get_variable("model").to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # Training loop with distributed coordination
    train_data = context.get_variable("train_data")
    sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    dataloader = torch.utils.data.DataLoader(
        train_data, 
        sampler=sampler,
        batch_size=32,
        pin_memory=True
    )
    
    optimizer = torch.optim.AdamW(model.parameters())
    
    for epoch in range(5):
        sampler.set_epoch(epoch)
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            
            # Gradient synchronization happens automatically with DDP
            optimizer.step()
            
            # Log metrics from rank 0 only
            if dist.get_rank() == 0:
                context.set_metric(f"training_loss", loss.item())
    
    # Cleanup distributed training
    if dist.is_initialized():
        dist.destroy_process_group()
    
    context.set_variable("trained_model", model.module)
\`\`\`

### Advanced Resource Patterns

#### Priority-Based Resource Allocation

\`\`\`python
from puffinflow import state, Priority, ResourcePool

# Create a resource pool for managing shared resources
ml_resource_pool = ResourcePool(
    name="ml_inference_pool",
    cpu_quota=16.0,
    memory_quota=32768,
    gpu_quota=4.0
)

@state(
    priority=Priority.CRITICAL,
    resource_pool=ml_resource_pool,
    cpu=4.0,
    memory=8192,
    gpu=2.0,
    preemption_policy="allow"  # Can preempt lower priority tasks
)
async def critical_ml_inference(context):
    """Critical ML inference with highest priority resource access."""
    # This state can preempt lower priority states if resources are needed
    model = context.get_variable("critical_model")
    urgent_data = context.get_variable("urgent_inference_data")
    
    # Process with guaranteed resources
    results = await model.predict(urgent_data)
    context.set_variable("critical_results", results)

@state(
    priority=Priority.HIGH,
    resource_pool=ml_resource_pool,
    cpu=2.0,
    memory=4096,
    gpu=1.0,
    preemption_policy="graceful"  # Handle preemption gracefully
)
async def high_priority_inference(context):
    """High priority inference that can be preempted by critical tasks."""
    try:
        model = context.get_variable("model")
        data = context.get_variable("inference_data")
        
        # Check if we're being preempted
        if context.is_preemption_requested():
            # Save state for later resumption
            context.set_checkpoint("partial_data", data)
            raise PreemptionRequested("Yielding to critical task")
        
        results = await model.predict(data)
        context.set_variable("inference_results", results)
        
    except PreemptionRequested:
        context.set_variable("preemption_handled", True)
        raise  # Re-raise to trigger proper preemption handling

@state(
    priority=Priority.LOW,
    resource_pool=ml_resource_pool,
    cpu=1.0,
    memory=2048,
    gpu=0.5,
    preemption_policy="checkpoint"  # Checkpoint before preemption
)
async def background_processing(context):
    """Background processing that checkpoints on preemption."""
    data_chunks = context.get_variable("background_data")
    processed_chunks = context.get_variable("processed_chunks", [])
    
    for i, chunk in enumerate(data_chunks):
        # Check for preemption before processing each chunk
        if context.is_preemption_requested():
            context.set_checkpoint("processed_chunks", processed_chunks)
            context.set_checkpoint("next_chunk_index", i)
            raise PreemptionRequested(f"Checkpointed at chunk {i}")
        
        processed_chunk = process_data_chunk(chunk)
        processed_chunks.append(processed_chunk)
        
        # Update progress
        context.set_metric("chunks_processed", len(processed_chunks))
    
    context.set_variable("final_results", processed_chunks)
\`\`\`

#### Dynamic Resource Scaling

\`\`\`python
from puffinflow import state, DynamicResourceManager, ScalingPolicy

# Configure dynamic resource scaling
dynamic_manager = DynamicResourceManager(
    scaling_policy=ScalingPolicy.PREDICTIVE,
    scale_up_threshold=0.8,      # Scale up when 80% utilized
    scale_down_threshold=0.3,    # Scale down when 30% utilized
    min_resources={"cpu": 1.0, "memory": 512},
    max_resources={"cpu": 16.0, "memory": 32768}
)

@state(
    resource_manager=dynamic_manager,
    cpu="auto",                  # Dynamic CPU allocation
    memory="auto",              # Dynamic memory allocation
    scaling_triggers=["queue_depth", "response_time", "error_rate"]
)
async def auto_scaling_processor(context):
    """Processor that automatically scales resources based on workload."""
    import time
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Get current workload metrics
    queue_depth = context.get_metric("queue_depth", 0)
    avg_response_time = context.get_metric("avg_response_time", 0)
    
    # Determine optimal resource allocation
    if queue_depth > 100 and avg_response_time > 5.0:
        # High load - request more resources
        await context.request_resource_scaling(
            cpu_multiplier=2.0,
            memory_multiplier=1.5,
            reason="High queue depth and response time"
        )
    elif queue_depth < 10 and avg_response_time < 1.0:
        # Low load - release excess resources
        await context.request_resource_scaling(
            cpu_multiplier=0.5,
            memory_multiplier=0.7,
            reason="Low utilization - cost optimization"
        )
    
    # Process workload with dynamically allocated resources
    work_items = context.get_variable("work_queue")
    current_resources = context.get_current_resource_allocation()
    
    # Adjust parallelism based on available resources
    max_workers = min(len(work_items), int(current_resources.cpu * 2))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        start_time = time.time()
        
        results = await asyncio.gather(*[
            executor.submit(process_work_item, item)
            for item in work_items[:max_workers]
        ])
        
        processing_time = time.time() - start_time
        
        # Update metrics for future scaling decisions
        context.set_metric("processing_time", processing_time)
        context.set_metric("throughput", len(results) / processing_time)
        context.set_metric("resource_efficiency", 
                          len(results) / (current_resources.cpu * processing_time))
    
    context.set_variable("processed_results", results)
\`\`\`

## Rate Limiting and Quota Management

### API Rate Limiting

\`\`\`python
from puffinflow import state, RateLimiter, QuotaManager, APIQuota

# Configure API quotas
openai_quota = APIQuota(
    name="openai",
    requests_per_minute=3000,
    requests_per_day=200000,
    tokens_per_minute=90000,
    cost_per_1k_tokens=0.002
)

anthropic_quota = APIQuota(
    name="anthropic",
    requests_per_minute=1000,
    requests_per_day=100000,
    tokens_per_minute=100000,
    cost_per_1k_tokens=0.008
)

# Global quota manager
quota_manager = QuotaManager([openai_quota, anthropic_quota])

@state(
    rate_limit=50.0,                    # 50 requests per second max
    burst_limit=100,                   # Allow bursts up to 100 requests
    quota_manager=quota_manager,
    api_provider="openai",
    cost_tracking=True
)
async def llm_processing(context):
    """LLM processing with comprehensive rate limiting and cost tracking."""
    import openai
    import time
    from tenacity import retry, stop_after_attempt, wait_exponential
    
    # Get rate limiter for this state
    rate_limiter = context.get_rate_limiter()
    
    prompts = context.get_variable("prompts")
    results = []
    total_tokens = 0
    total_cost = 0.0
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def make_api_call(prompt):
        # Wait for rate limit clearance
        await rate_limiter.acquire()
        
        # Check quota availability
        if not quota_manager.check_availability("openai", tokens=1000):
            raise QuotaExceededException("OpenAI quota exceeded")
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            
            # Track usage
            tokens_used = response["usage"]["total_tokens"]
            cost = quota_manager.calculate_cost("openai", tokens_used)
            
            # Update quotas
            quota_manager.consume_quota("openai", 
                                      requests=1, 
                                      tokens=tokens_used)
            
            return {
                "result": response["choices"][0]["message"]["content"],
                "tokens": tokens_used,
                "cost": cost
            }
            
        except openai.RateLimitError as e:
            # Dynamic backoff for rate limits
            backoff_time = rate_limiter.get_backoff_time()
            await asyncio.sleep(backoff_time)
            raise
    
    # Process prompts with rate limiting
    for prompt in prompts:
        try:
            result = await make_api_call(prompt)
            results.append(result["result"])
            total_tokens += result["tokens"]
            total_cost += result["cost"]
            
            # Update real-time metrics
            context.set_metric("tokens_consumed", total_tokens)
            context.set_metric("api_cost", total_cost)
            context.increment_metric("successful_api_calls")
            
        except Exception as e:
            context.increment_metric("failed_api_calls")
            context.set_variable("error", str(e))
    
    # Store results and metrics
    context.set_variable("llm_results", results)
    context.set_variable("total_api_cost", total_cost)
    context.set_variable("total_tokens_used", total_tokens)
    
    # Log quota status
    quota_status = quota_manager.get_quota_status("openai")
    context.set_metric("quota_remaining_percent", 
                      quota_status["requests_remaining"] / quota_status["requests_limit"] * 100)

@state(
    rate_limit=10.0,                   # Conservative rate limit
    quota_manager=quota_manager,
    api_provider="anthropic",
    fallback_provider="openai",        # Fallback if quota exceeded
    cost_optimization=True
)
async def multi_provider_llm(context):
    """Multi-provider LLM calls with automatic fallback and cost optimization."""
    prompt = context.get_variable("prompt")
    
    # Try primary provider first
    primary_available = quota_manager.check_availability("anthropic", tokens=2000)
    fallback_available = quota_manager.check_availability("openai", tokens=2000)
    
    if primary_available:
        provider = "anthropic"
        model = "claude-3-sonnet"
    elif fallback_available:
        provider = "openai" 
        model = "gpt-4"
        context.increment_metric("fallback_activations")
    else:
        raise QuotaExceededException("All provider quotas exceeded")
    
    # Make API call with selected provider
    if provider == "anthropic":
        # Anthropic API call implementation
        response = await call_anthropic_api(prompt, model)
    else:
        # OpenAI API call implementation
        response = await call_openai_api(prompt, model)
    
    # Track cross-provider costs and usage
    cost = quota_manager.calculate_cost(provider, response["tokens"])
    quota_manager.consume_quota(provider, 
                               requests=1, 
                               tokens=response["tokens"])
    
    context.set_variable("llm_response", response["content"])
    context.set_variable("provider_used", provider)
    context.set_variable("api_cost", cost)
    
    # Cost optimization metrics
    context.set_metric(f"{provider}_cost", cost)
    context.set_metric("cross_provider_cost_efficiency", 
                      response["quality_score"] / cost if cost > 0 else 0)
\`\`\`

### Database Connection Management

\`\`\`python
from puffinflow import state, ConnectionPool, DatabaseQuota

# Configure database connection pools
postgres_pool = ConnectionPool(
    name="postgres_main",
    max_connections=50,
    min_connections=5,
    connection_timeout=30.0,
    idle_timeout=300.0,
    max_lifetime=3600.0
)

redis_pool = ConnectionPool(
    name="redis_cache",
    max_connections=100,
    min_connections=10,
    connection_timeout=5.0,
    idle_timeout=60.0
)

@state(
    connection_pool=postgres_pool,
    max_connections=5,              # Max concurrent connections for this state
    transaction_timeout=60.0,       # Transaction timeout
    isolation_level="READ_COMMITTED"
)
async def database_operations(context):
    """Database operations with connection pooling and transaction management."""
    import asyncpg
    import asyncio
    
    queries = context.get_variable("batch_queries")
    results = []
    
    # Get connection from pool
    async with postgres_pool.acquire() as conn:
        # Begin transaction
        async with conn.transaction(isolation="read_committed"):
            try:
                for query_data in queries:
                    query = query_data["query"]
                    params = query_data.get("params", [])
                    
                    if query_data["type"] == "select":
                        result = await conn.fetch(query, *params)
                        results.append([dict(row) for row in result])
                    elif query_data["type"] == "insert":
                        result = await conn.execute(query, *params)
                        results.append({"affected_rows": int(result.split()[-1])})
                    elif query_data["type"] == "update":
                        result = await conn.execute(query, *params)
                        results.append({"affected_rows": int(result.split()[-1])})
                    
                    # Track query performance
                    context.increment_metric("database_queries_executed")
                
                # Commit transaction (automatic with context manager)
                context.increment_metric("database_transactions_committed")
                
            except Exception as e:
                # Transaction will rollback automatically
                context.increment_metric("database_transactions_failed")
                context.set_variable("database_error", str(e))
                raise
    
    context.set_variable("query_results", results)
    context.set_metric("database_connection_pool_size", postgres_pool.size)
    context.set_metric("database_active_connections", postgres_pool.active_connections)

@state(
    connection_pool=redis_pool,
    max_connections=10,
    operation_timeout=5.0
)
async def cache_operations(context):
    """Redis cache operations with connection pooling."""
    import aioredis
    
    cache_operations = context.get_variable("cache_operations")
    results = {}
    
    async with redis_pool.acquire() as redis_conn:
        pipe = redis_conn.pipeline()
        
        # Batch operations for efficiency
        for operation in cache_operations:
            op_type = operation["type"]
            key = operation["key"]
            
            if op_type == "get":
                pipe.get(key)
            elif op_type == "set":
                value = operation["value"]
                ttl = operation.get("ttl", 3600)
                pipe.setex(key, ttl, value)
            elif op_type == "delete":
                pipe.delete(key)
        
        # Execute all operations
        pipe_results = await pipe.execute()
        
        # Process results
        for i, operation in enumerate(cache_operations):
            operation_id = operation.get("id", f"op_{i}")
            results[operation_id] = pipe_results[i]
        
        context.increment_metric("cache_operations_executed", len(cache_operations))
    
    context.set_variable("cache_results", results)
    context.set_metric("redis_connection_pool_size", redis_pool.size)
\`\`\`

## Resource Monitoring and Optimization

### Real-time Resource Monitoring

\`\`\`python
from puffinflow import state, ResourceMonitor, AlertingRule

# Configure resource monitoring
resource_monitor = ResourceMonitor(
    sampling_interval=1.0,          # Sample every second
    history_retention=3600,         # Keep 1 hour of history
    alert_thresholds={
        "cpu_usage": 85.0,          # Alert if CPU > 85%
        "memory_usage": 90.0,       # Alert if memory > 90%
        "gpu_usage": 95.0,          # Alert if GPU > 95%
        "disk_usage": 80.0          # Alert if disk > 80%
    }
)

@state(
    resource_monitor=resource_monitor,
    cpu=4.0,
    memory=8192,
    monitoring_level="detailed",     # Enable detailed monitoring
    alert_on_anomalies=True
)
async def monitored_computation(context):
    """Computation with comprehensive resource monitoring and alerting."""
    import psutil
    import GPUtil
    import time
    import numpy as np
    
    # Start monitoring
    monitor = context.get_resource_monitor()
    monitor.start_monitoring()
    
    try:
        # Get computational workload
        computation_tasks = context.get_variable("computation_tasks")
        results = []
        
        for i, task in enumerate(computation_tasks):
            task_start = time.time()
            
            # Monitor resources before task execution
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            
            if GPUtil.getGPUs():
                gpu = GPUtil.getGPUs()[0]
                gpu_usage = gpu.load * 100
                gpu_memory_usage = gpu.memoryUtil * 100
            else:
                gpu_usage = gpu_memory_usage = 0
            
            # Record pre-task metrics
            context.set_metric(f"task_{i}_cpu_before", cpu_percent)
            context.set_metric(f"task_{i}_memory_before", memory_info.percent)
            context.set_metric(f"task_{i}_gpu_before", gpu_usage)
            
            # Execute task
            if task["type"] == "cpu_intensive":
                result = await cpu_intensive_task(task["data"])
            elif task["type"] == "memory_intensive":
                result = await memory_intensive_task(task["data"])
            elif task["type"] == "gpu_intensive":
                result = await gpu_intensive_task(task["data"])
            else:
                result = await generic_task(task["data"])
            
            task_duration = time.time() - task_start
            results.append(result)
            
            # Monitor resources after task execution
            cpu_percent_after = psutil.cpu_percent(interval=0.1)
            memory_info_after = psutil.virtual_memory()
            
            # Record post-task metrics
            context.set_metric(f"task_{i}_cpu_after", cpu_percent_after)
            context.set_metric(f"task_{i}_memory_after", memory_info_after.percent)
            context.set_metric(f"task_{i}_duration", task_duration)
            context.set_metric(f"task_{i}_efficiency", 
                             result.get("operations_completed", 0) / task_duration)
            
            # Check for resource anomalies
            cpu_spike = cpu_percent_after - cpu_percent
            memory_spike = memory_info_after.percent - memory_info.percent
            
            if cpu_spike > 50 or memory_spike > 30:
                context.add_alert(
                    level="warning",
                    message=f"Resource spike detected in task {i}: "
                           f"CPU +{cpu_spike:.1f}%, Memory +{memory_spike:.1f}%"
                )
            
            # Adaptive delays based on resource usage
            if cpu_percent_after > 90 or memory_info_after.percent > 95:
                adaptive_delay = min(5.0, cpu_percent_after / 20)
                await asyncio.sleep(adaptive_delay)
                context.increment_metric("adaptive_delays_triggered")
        
        # Final resource summary
        final_stats = monitor.get_current_stats()
        context.set_metric("peak_cpu_usage", final_stats["peak_cpu"])
        context.set_metric("peak_memory_usage", final_stats["peak_memory"])
        context.set_metric("average_gpu_usage", final_stats["avg_gpu"])
        context.set_metric("resource_efficiency_score", 
                          calculate_efficiency_score(final_stats))
        
        context.set_variable("computation_results", results)
        
    finally:
        # Stop monitoring and generate report
        monitor.stop_monitoring()
        monitoring_report = monitor.generate_report()
        context.set_variable("resource_monitoring_report", monitoring_report)

async def cpu_intensive_task(data):
    """Simulate CPU-intensive computation."""
    import numpy as np
    
    # Matrix multiplication to simulate CPU load
    matrices = [np.random.rand(500, 500) for _ in range(10)]
    results = []
    
    for matrix in matrices:
        # Perform expensive computation
        eigenvalues = np.linalg.eigvals(matrix @ matrix.T)
        results.append(eigenvalues.mean())
    
    return {"operations_completed": len(matrices), "results": results}

async def memory_intensive_task(data):
    """Simulate memory-intensive computation."""
    import numpy as np
    
    # Create large arrays to simulate memory usage
    large_arrays = []
    
    for i in range(5):
        # Allocate 100MB arrays
        array = np.random.rand(1000, 1000, 10)  # ~80MB
        large_arrays.append(array)
        
        # Process array
        processed = np.fft.fft2(array[:, :, 0])
        summary = np.abs(processed).mean()
    
    return {"operations_completed": len(large_arrays), "memory_peak_mb": 400}

async def gpu_intensive_task(data):
    """Simulate GPU-intensive computation."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {"error": "GPU not available", "operations_completed": 0}
        
        device = torch.device("cuda")
        
        # Create large tensors on GPU
        tensors = []
        for i in range(3):
            tensor = torch.randn(2000, 2000, device=device)
            tensors.append(tensor)
        
        # Perform GPU computations
        results = []
        for tensor in tensors:
            # Matrix operations on GPU
            result = torch.matmul(tensor, tensor.T)
            eigenvals = torch.linalg.eigvals(result)
            results.append(eigenvals.mean().item())
        
        # Clean up GPU memory
        for tensor in tensors:
            del tensor
        torch.cuda.empty_cache()
        
        return {"operations_completed": len(tensors), "results": results}
        
    except ImportError:
        return {"error": "PyTorch not available", "operations_completed": 0}
\`\`\`

### Performance Optimization and Tuning

\`\`\`python
from puffinflow import state, PerformanceProfiler, OptimizationHints

@state(
    cpu=2.0,
    memory=4096,
    profiler=PerformanceProfiler(
        profile_type="comprehensive",
        include_memory_profiling=True,
        include_cpu_profiling=True,
        sampling_rate=1000  # 1000 samples per second
    ),
    optimization_hints=OptimizationHints(
        enable_auto_tuning=True,
        cache_intermediate_results=True,
        use_multiprocessing=True
    )
)
async def performance_optimized_processing(context):
    """Processing with comprehensive performance profiling and optimization."""
    import cProfile
    import pstats
    import io
    import tracemalloc
    import time
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    import multiprocessing as mp
    
    # Start performance profiling
    profiler = cProfile.Profile()
    tracemalloc.start()
    profiler.enable()
    
    # Get profiler from context
    perf_profiler = context.get_performance_profiler()
    perf_profiler.start_profiling()
    
    try:
        # Load and analyze workload
        workload = context.get_variable("processing_workload")
        workload_size = len(workload)
        
        # Adaptive strategy based on workload characteristics
        if workload_size < 100:
            # Small workload - use threading
            executor_type = "thread"
            max_workers = min(4, workload_size)
        elif workload_size < 1000:
            # Medium workload - use process pool
            executor_type = "process"
            max_workers = min(mp.cpu_count(), workload_size // 10)
        else:
            # Large workload - use hybrid approach
            executor_type = "hybrid"
            max_workers = mp.cpu_count()
        
        context.set_metric("executor_type", executor_type)
        context.set_metric("max_workers", max_workers)
        
        # Cache intermediate results if beneficial
        cache = {}
        cache_hits = 0
        
        results = []
        
        if executor_type == "thread":
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for item in workload:
                    # Check cache first
                    cache_key = generate_cache_key(item)
                    if cache_key in cache:
                        results.append(cache[cache_key])
                        cache_hits += 1
                        continue
                    
                    future = executor.submit(process_item_optimized, item)
                    futures.append((future, cache_key))
                
                # Collect results and update cache
                for future, cache_key in futures:
                    result = future.result()
                    results.append(result)
                    cache[cache_key] = result
        
        elif executor_type == "process":
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Use chunking for better process utilization
                chunk_size = max(1, workload_size // (max_workers * 4))
                chunks = [workload[i:i+chunk_size] 
                         for i in range(0, workload_size, chunk_size)]
                
                futures = [
                    executor.submit(process_chunk_optimized, chunk)
                    for chunk in chunks
                ]
                
                # Collect and flatten results
                for future in futures:
                    chunk_results = future.result()
                    results.extend(chunk_results)
        
        elif executor_type == "hybrid":
            # Process large chunks in separate processes
            # Use threading for I/O within each process
            chunk_size = max(1, workload_size // max_workers)
            chunks = [workload[i:i+chunk_size] 
                     for i in range(0, workload_size, chunk_size)]
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_chunk_with_threading, chunk)
                    for chunk in chunks
                ]
                
                for future in futures:
                    chunk_results = future.result()
                    results.extend(chunk_results)
        
        # Performance analysis
        processing_time = time.time()
        
        # Memory profiling results
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # CPU profiling results
        profiler.disable()
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        # Store profiling data
        context.set_variable("cpu_profile", s.getvalue())
        context.set_metric("peak_memory_mb", peak / 1024 / 1024)
        context.set_metric("current_memory_mb", current / 1024 / 1024)
        context.set_metric("cache_hit_rate", cache_hits / workload_size if workload_size > 0 else 0)
        
        # Calculate performance metrics
        throughput = workload_size / processing_time
        efficiency = throughput / max_workers  # Items per worker per second
        
        context.set_metric("throughput_items_per_sec", throughput)
        context.set_metric("worker_efficiency", efficiency)
        context.set_metric("total_processing_time", processing_time)
        
        # Performance optimization recommendations
        optimization_report = generate_optimization_report(
            executor_type=executor_type,
            throughput=throughput,
            memory_usage=peak,
            cache_hit_rate=cache_hits / workload_size,
            worker_count=max_workers
        )
        
        context.set_variable("optimization_report", optimization_report)
        context.set_variable("processing_results", results)
        
    finally:
        # Stop profiling
        perf_profiler.stop_profiling()
        profiling_report = perf_profiler.generate_report()
        context.set_variable("performance_profile", profiling_report)

def process_item_optimized(item):
    """Optimized item processing function."""
    # Simulate processing with different computational patterns
    if item.get("type") == "cpu_intensive":
        # CPU-bound operation
        import math
        result = sum(math.sqrt(i) for i in range(1000))
        return {"processed": True, "value": result, "type": "cpu"}
    
    elif item.get("type") == "memory_intensive":
        # Memory-bound operation
        data = [i**2 for i in range(10000)]
        return {"processed": True, "value": sum(data), "type": "memory"}
    
    else:
        # Generic processing
        import time
        time.sleep(0.001)  # Simulate I/O
        return {"processed": True, "value": item.get("value", 0) * 2, "type": "generic"}

def process_chunk_optimized(chunk):
    """Optimized chunk processing for process pools."""
    return [process_item_optimized(item) for item in chunk]

def process_chunk_with_threading(chunk):
    """Process chunk using threading for I/O operations."""
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_item_optimized, item) for item in chunk]
        return [future.result() for future in futures]

def generate_cache_key(item):
    """Generate cache key for item."""
    import hashlib
    import json
    
    item_str = json.dumps(item, sort_keys=True)
    return hashlib.md5(item_str.encode()).hexdigest()

def generate_optimization_report(executor_type, throughput, memory_usage, cache_hit_rate, worker_count):
    """Generate performance optimization recommendations."""
    recommendations = []
    
    # Throughput analysis
    if throughput < 10:
        recommendations.append("Low throughput detected - consider increasing worker count or using process pools")
    elif throughput > 1000:
        recommendations.append("High throughput achieved - current configuration is well-optimized")
    
    # Memory analysis
    if memory_usage > 1024 * 1024 * 1024:  # 1GB
        recommendations.append("High memory usage - consider processing in smaller chunks or using streaming")
    
    # Cache analysis
    if cache_hit_rate < 0.1:
        recommendations.append("Low cache hit rate - caching may not be beneficial for this workload")
    elif cache_hit_rate > 0.8:
        recommendations.append("High cache hit rate - caching is very effective")
    
    # Concurrency analysis
    if executor_type == "thread" and worker_count > 8:
        recommendations.append("High thread count - consider switching to process pool for CPU-bound work")
    elif executor_type == "process" and worker_count < 2:
        recommendations.append("Low process count - may not be utilizing available CPU cores")
    
    return {
        "executor_type": executor_type,
        "throughput": throughput,
        "memory_usage_mb": memory_usage / 1024 / 1024,
        "cache_hit_rate": cache_hit_rate,
        "worker_count": worker_count,
        "recommendations": recommendations,
        "optimization_score": calculate_optimization_score(throughput, memory_usage, cache_hit_rate)
    }

def calculate_optimization_score(throughput, memory_usage, cache_hit_rate):
    """Calculate overall optimization score."""
    # Normalize metrics to 0-100 scale
    throughput_score = min(100, throughput / 10)  # 10 items/sec = 100 points
    memory_score = max(0, 100 - (memory_usage / (1024 * 1024 * 10)))  # 10MB = 100 points
    cache_score = cache_hit_rate * 100
    
    # Weighted average
    return (throughput_score * 0.5 + memory_score * 0.3 + cache_score * 0.2)
\`\`\`

## Cost Optimization Strategies

### Cloud Cost Management

\`\`\`python
from puffinflow import state, CostOptimizer, CloudProvider

# Configure cost optimization
cost_optimizer = CostOptimizer(
    cloud_provider=CloudProvider.AWS,
    cost_tracking_enabled=True,
    budget_alerts_enabled=True,
    auto_scaling_enabled=True,
    spot_instance_usage=True
)

@state(
    cost_optimizer=cost_optimizer,
    instance_type="spot",           # Use spot instances for cost savings
    max_cost_per_hour=5.0,         # Maximum acceptable cost
    cost_monitoring=True,
    auto_shutdown_on_budget=True
)
async def cost_optimized_processing(context):
    """Processing with comprehensive cost optimization and monitoring."""
    import boto3
    import time
    from datetime import datetime, timedelta
    
    # Initialize cost tracking
    cost_tracker = context.get_cost_optimizer()
    session_start = datetime.now()
    
    try:
        # Check current budget status
        current_budget = cost_tracker.get_current_budget_usage()
        if current_budget["usage_percentage"] > 90:
            context.add_alert(
                level="warning", 
                message=f"Budget usage at {current_budget['usage_percentage']}%"
            )
        
        # Get processing workload
        workload = context.get_variable("workload")
        estimated_cost = cost_tracker.estimate_processing_cost(workload)
        
        if estimated_cost > context.get_config("max_cost_per_hour", 5.0):
            # Optimize workload for cost
            workload = await optimize_workload_for_cost(workload, estimated_cost)
            context.set_variable("cost_optimized_workload", True)
        
        # Choose optimal instance configuration
        optimal_config = cost_tracker.get_optimal_instance_config(
            cpu_requirements=workload["cpu_needed"],
            memory_requirements=workload["memory_needed"],
            duration_estimate=workload["estimated_duration"]
        )
        
        context.set_metric("instance_type", optimal_config["instance_type"])
        context.set_metric("estimated_hourly_cost", optimal_config["hourly_cost"])
        
        # Process with cost monitoring
        results = []
        total_cost = 0.0
        
        for i, task in enumerate(workload["tasks"]):
            task_start = time.time()
            
            # Monitor spot instance status
            if optimal_config["instance_type"] == "spot":
                spot_status = cost_tracker.check_spot_instance_status()
                if spot_status["interruption_warning"]:
                    # Save state and request on-demand instance
                    context.set_checkpoint(f"task_{i}_state", task)
                    await cost_tracker.request_on_demand_fallback()
                    context.increment_metric("spot_interruption_handled")
            
            # Execute task
            task_result = await process_task_with_cost_tracking(task, cost_tracker)
            results.append(task_result["result"])
            
            # Track costs
            task_cost = task_result["cost"]
            total_cost += task_cost
            
            context.set_metric(f"task_{i}_cost", task_cost)
            context.set_metric("cumulative_cost", total_cost)
            
            # Cost-based early termination
            if total_cost > context.get_config("max_total_cost", 50.0):
                context.add_alert(
                    level="error",
                    message=f"Cost limit reached: \${total_cost:.2f}"
                )
                break
            
            # Dynamic cost optimization
            if i % 10 == 0:  # Every 10 tasks
                current_efficiency = len(results) / total_cost if total_cost > 0 else 0
                if current_efficiency < 1.0:  # Less than 1 task per dollar
                    # Reduce resource allocation to optimize cost
                    await cost_tracker.optimize_resource_allocation(
                        target_efficiency=2.0
                    )
        
        # Final cost analysis
        session_duration = (datetime.now() - session_start).total_seconds() / 3600
        cost_per_hour = total_cost / session_duration if session_duration > 0 else 0
        
        context.set_metric("session_duration_hours", session_duration)
        context.set_metric("cost_per_hour", cost_per_hour)
        context.set_metric("total_session_cost", total_cost)
        context.set_metric("cost_efficiency", len(results) / total_cost if total_cost > 0 else 0)
        
        # Generate cost optimization report
        cost_report = cost_tracker.generate_cost_report()
        savings_report = cost_tracker.calculate_savings_vs_on_demand()
        
        context.set_variable("processing_results", results)
        context.set_variable("cost_report", cost_report)
        context.set_variable("savings_report", savings_report)
        
    except Exception as e:
        # Ensure cleanup to avoid unnecessary costs
        await cost_tracker.emergency_shutdown()
        raise

async def optimize_workload_for_cost(workload, estimated_cost):
    """Optimize workload configuration for cost efficiency."""
    # Reduce precision for non-critical tasks
    for task in workload["tasks"]:
        if task.get("priority", "medium") != "high":
            task["precision"] = "medium"  # Reduce from "high" to "medium"
    
    # Use smaller batch sizes to reduce memory requirements
    original_batch_size = workload.get("batch_size", 100)
    workload["batch_size"] = max(10, original_batch_size // 2)
    
    # Enable aggressive caching
    workload["enable_caching"] = True
    workload["cache_ttl"] = 3600  # Cache for 1 hour
    
    return workload

async def process_task_with_cost_tracking(task, cost_tracker):
    """Process individual task with detailed cost tracking."""
    import time
    
    task_start = time.time()
    resource_usage_start = cost_tracker.get_current_resource_usage()
    
    # Simulate task processing
    if task["type"] == "compute":
        # CPU-intensive task
        result = await compute_task(task["data"])
        cost_model = "cpu_intensive"
    elif task["type"] == "memory":
        # Memory-intensive task  
        result = await memory_task(task["data"])
        cost_model = "memory_intensive"
    elif task["type"] == "api":
        # API-based task
        result = await api_task(task["data"])
        cost_model = "api_based"
    else:
        result = await generic_task(task["data"])
        cost_model = "generic"
    
    task_duration = time.time() - task_start
    resource_usage_end = cost_tracker.get_current_resource_usage()
    
    # Calculate task-specific costs
    cost_breakdown = cost_tracker.calculate_task_cost(
        duration=task_duration,
        resource_usage_start=resource_usage_start,
        resource_usage_end=resource_usage_end,
        cost_model=cost_model
    )
    
    return {
        "result": result,
        "cost": cost_breakdown["total_cost"],
        "cost_breakdown": cost_breakdown,
        "duration": task_duration
    }

# Example task processing functions
async def compute_task(data):
    """CPU-intensive computation task."""
    import numpy as np
    
    # Simulate compute work
    matrices = [np.random.rand(100, 100) for _ in range(10)]
    result = np.sum([np.trace(m @ m.T) for m in matrices])
    return {"computed_value": result, "matrices_processed": len(matrices)}

async def memory_task(data):
    """Memory-intensive processing task."""
    import numpy as np
    
    # Create large data structures
    large_array = np.random.rand(1000, 1000)  # ~8MB
    processed = np.fft.fft2(large_array)
    result = np.abs(processed).sum()
    
    return {"processed_sum": result, "array_size": large_array.shape}

async def api_task(data):
    """API-based processing task."""
    import aiohttp
    import asyncio
    
    # Simulate API calls
    async with aiohttp.ClientSession() as session:
        # Mock API endpoint
        url = "https://api.example.com/process"
        payload = {"data": data}
        
        try:
            async with session.post(url, json=payload, timeout=10) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"api_result": result, "status": "success"}
                else:
                    return {"error": f"API error: {response.status}", "status": "error"}
        except asyncio.TimeoutError:
            return {"error": "API timeout", "status": "timeout"}

async def generic_task(data):
    """Generic task processing."""
    import time
    
    # Simulate processing
    await asyncio.sleep(0.1)
    return {"processed": True, "input_size": len(str(data))}
\`\`\`

## Troubleshooting Resource Issues

### Common Resource Problems and Solutions

#### Problem: Resource Exhaustion
\`\`\`python
from puffinflow import state, ResourceExhaustionHandler

@state(
    cpu=2.0,
    memory=4096,
    resource_exhaustion_handler=ResourceExhaustionHandler(
        strategy="graceful_degradation",
        fallback_resources={"cpu": 1.0, "memory": 2048},
        emergency_shutdown_threshold=0.95
    )
)
async def resource_exhaustion_resilient(context):
    """State that handles resource exhaustion gracefully."""
    import psutil
    import gc
    
    try:
        workload = context.get_variable("workload")
        results = []
        
        for item in workload:
            # Check resource availability before processing
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            if memory_percent > 90:
                # Memory exhaustion detected
                context.add_alert(
                    level="warning",
                    message=f"Memory usage at {memory_percent}%"
                )
                
                # Trigger garbage collection
                gc.collect()
                
                # Switch to memory-efficient processing
                result = await memory_efficient_processing(item)
                context.increment_metric("memory_efficient_fallbacks")
                
            elif cpu_percent > 95:
                # CPU exhaustion detected
                context.add_alert(
                    level="warning", 
                    message=f"CPU usage at {cpu_percent}%"
                )
                
                # Add processing delay to reduce CPU pressure
                await asyncio.sleep(0.1)
                result = await process_with_backoff(item)
                context.increment_metric("cpu_backoff_activations")
                
            else:
                # Normal processing
                result = await normal_processing(item)
            
            results.append(result)
        
        context.set_variable("processed_results", results)
        
    except ResourceExhaustionException as e:
        # Handle critical resource exhaustion
        context.set_variable("resource_exhaustion_error", str(e))
        
        # Attempt graceful degradation
        partial_results = await emergency_processing_mode(workload[:10])
        context.set_variable("partial_results", partial_results)
        context.increment_metric("emergency_mode_activations")

async def memory_efficient_processing(item):
    """Process item with minimal memory usage."""
    # Process in smaller chunks
    data_chunks = chunk_data(item["data"], chunk_size=100)
    results = []
    
    for chunk in data_chunks:
        chunk_result = process_data_chunk(chunk)
        results.append(chunk_result)
        # Clear intermediate variables
        del chunk_result
    
    return {"results": results, "mode": "memory_efficient"}

async def process_with_backoff(item):
    """Process with CPU backoff to reduce load."""
    import time
    
    # Add delays between operations
    data = item["data"]
    result = []
    
    for i, element in enumerate(data):
        processed_element = process_element(element)
        result.append(processed_element)
        
        # Add backoff delay every 10 elements
        if i % 10 == 0:
            await asyncio.sleep(0.01)
    
    return {"results": result, "mode": "cpu_backoff"}

async def emergency_processing_mode(limited_workload):
    """Emergency processing with minimal resources."""
    # Process only critical items
    critical_items = [item for item in limited_workload if item.get("priority") == "critical"]
    
    results = []
    for item in critical_items:
        # Minimal processing
        simple_result = {"id": item.get("id"), "status": "emergency_processed"}
        results.append(simple_result)
    
    return results
\`\`\`

### Resource Leak Detection

\`\`\`python
from puffinflow import state, ResourceLeakDetector

@state(
    resource_leak_detector=ResourceLeakDetector(
        check_interval=30.0,          # Check every 30 seconds
        memory_growth_threshold=50,    # Alert if memory grows by 50MB
        file_handle_threshold=100,     # Alert if file handles > 100
        connection_threshold=50        # Alert if connections > 50
    )
)
async def leak_monitored_processing(context):
    """Processing with resource leak detection and prevention."""
    import psutil
    import gc
    import weakref
    import threading
    
    # Start leak detection
    leak_detector = context.get_resource_leak_detector()
    leak_detector.start_monitoring()
    
    # Track resource baselines
    initial_memory = psutil.Process().memory_info().rss
    initial_threads = threading.active_count()
    initial_file_handles = len(psutil.Process().open_files())
    
    context.set_metric("initial_memory_mb", initial_memory / 1024 / 1024)
    context.set_metric("initial_threads", initial_threads)
    context.set_metric("initial_file_handles", initial_file_handles)
    
    # Create weak references to track object lifecycle
    created_objects = []
    
    try:
        workload = context.get_variable("workload")
        results = []
        
        for i, item in enumerate(workload):
            # Process item with resource tracking
            result = await process_with_resource_tracking(
                item, created_objects, leak_detector
            )
            results.append(result)
            
            # Periodic resource checks
            if i % 100 == 0:
                current_memory = psutil.Process().memory_info().rss
                memory_growth = (current_memory - initial_memory) / 1024 / 1024
                
                if memory_growth > 100:  # More than 100MB growth
                    # Potential memory leak detected
                    leak_detector.trigger_leak_analysis()
                    
                    # Force garbage collection
                    collected = gc.collect()
                    context.increment_metric("forced_gc_runs")
                    context.set_metric("objects_collected", collected)
                    
                    # Check if memory was reclaimed
                    after_gc_memory = psutil.Process().memory_info().rss
                    memory_reclaimed = (current_memory - after_gc_memory) / 1024 / 1024
                    
                    if memory_reclaimed < 10:  # Less than 10MB reclaimed
                        context.add_alert(
                            level="error",
                            message=f"Potential memory leak: {memory_growth:.1f}MB growth, "
                                   f"only {memory_reclaimed:.1f}MB reclaimed"
                        )
                
                # Check thread count
                current_threads = threading.active_count()
                if current_threads > initial_threads + 10:
                    context.add_alert(
                        level="warning",
                        message=f"Thread count increased from {initial_threads} to {current_threads}"
                    )
                
                # Check file handles
                current_file_handles = len(psutil.Process().open_files())
                if current_file_handles > initial_file_handles + 50:
                    context.add_alert(
                        level="warning",
                        message=f"File handle count increased from {initial_file_handles} "
                               f"to {current_file_handles}"
                    )
        
        # Final resource audit
        final_memory = psutil.Process().memory_info().rss
        final_threads = threading.active_count()
        final_file_handles = len(psutil.Process().open_files())
        
        # Check for object leaks using weak references
        leaked_objects = [ref for ref in created_objects if ref() is not None]
        
        context.set_metric("final_memory_mb", final_memory / 1024 / 1024)
        context.set_metric("memory_growth_mb", (final_memory - initial_memory) / 1024 / 1024)
        context.set_metric("thread_growth", final_threads - initial_threads)
        context.set_metric("file_handle_growth", final_file_handles - initial_file_handles)
        context.set_metric("potential_leaked_objects", len(leaked_objects))
        
        context.set_variable("processing_results", results)
        
        # Generate leak detection report
        leak_report = leak_detector.generate_report()
        context.set_variable("resource_leak_report", leak_report)
        
    finally:
        leak_detector.stop_monitoring()
        
        # Cleanup any remaining resources
        await cleanup_resources(created_objects, context)

async def process_with_resource_tracking(item, created_objects, leak_detector):
    """Process item while tracking resource creation."""
    import tempfile
    import sqlite3
    import weakref
    
    # Simulate resource-intensive processing
    temp_files = []
    db_connections = []
    large_objects = []
    
    try:
        # Create temporary resources
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_files.append(temp_file)
        created_objects.append(weakref.ref(temp_file))
        
        # Database connection
        db_conn = sqlite3.connect(":memory:")
        db_connections.append(db_conn)
        created_objects.append(weakref.ref(db_conn))
        
        # Large data structure
        large_data = [i * 2 for i in range(10000)]
        large_objects.append(large_data)
        created_objects.append(weakref.ref(large_data))
        
        # Process the item
        result = {"processed": True, "item_id": item.get("id")}
        
        # Write temporary data
        temp_file.write(str(result).encode())
        temp_file.flush()
        
        # Database operations
        cursor = db_conn.cursor()
        cursor.execute("CREATE TABLE temp (id INTEGER, value TEXT)")
        cursor.execute("INSERT INTO temp VALUES (?, ?)", 
                      (item.get("id"), str(result)))
        db_conn.commit()
        
        return result
        
    except Exception as e:
        leak_detector.record_error_with_resources(
            error=str(e),
            temp_files=len(temp_files),
            db_connections=len(db_connections),
            large_objects=len(large_objects)
        )
        raise
    finally:
        # Proper resource cleanup
        for temp_file in temp_files:
            try:
                temp_file.close()
                os.unlink(temp_file.name)
            except:
                pass
        
        for db_conn in db_connections:
            try:
                db_conn.close()
            except:
                pass
        
        # Clear references to large objects
        large_objects.clear()

async def cleanup_resources(created_objects, context):
    """Cleanup any leaked resources."""
    cleaned_count = 0
    
    for weak_ref in created_objects:
        obj = weak_ref()
        if obj is not None:
            try:
                # Attempt to cleanup based on object type
                if hasattr(obj, 'close'):
                    obj.close()
                elif hasattr(obj, '__del__'):
                    del obj
                cleaned_count += 1
            except:
                pass
    
    context.set_metric("resources_cleaned", cleaned_count)
    
    # Force garbage collection
    import gc
    collected = gc.collect()
    context.set_metric("final_gc_collected", collected)
\`\`\`

## Quick Reference

### Resource Configuration Syntax

\`\`\`python
from puffinflow import state, Priority

@state(
    # Core resources
    cpu=2.0,                    # CPU cores required
    memory=1024,                # Memory in MB
    gpu=1.0,                    # GPU units required
    disk=500,                   # Temporary disk space in MB
    
    # Execution constraints
    timeout=60.0,               # Maximum execution time
    max_retries=3,              # Retry attempts on failure
    priority=Priority.HIGH,     # Execution priority
    
    # Rate limiting
    rate_limit=10.0,            # Max calls per second
    burst_limit=20,             # Burst capacity
    
    # Coordination
    semaphore_size=5,           # Concurrent access limit
    mutex=True,                 # Exclusive access required
    
    # Monitoring
    monitor_resources=True,     # Enable resource monitoring
    metric_tags=["service:ml"]  # Custom metric tags
)
async def example_state(context):
    pass
\`\`\`

### Common Resource Patterns

\`\`\`python
# Quick validation (micro workload)
@state(cpu=0.25, memory=128, timeout=5.0)
async def quick_validation(context): pass

# API integration (I/O bound)
@state(cpu=0.5, memory=256, rate_limit=10.0, timeout=30.0)
async def api_integration(context): pass

# Data processing (CPU bound)
@state(cpu=4.0, memory=2048, timeout=300.0)
async def data_processing(context): pass

# ML inference (GPU accelerated)
@state(cpu=2.0, memory=4096, gpu=1.0, timeout=120.0)
async def ml_inference(context): pass

# Critical operation (high priority)
@state(cpu=2.0, memory=1024, priority=Priority.CRITICAL, max_retries=5)
async def critical_operation(context): pass
\`\`\`

Resource management in PuffinFlow enables you to build robust, scalable workflows that efficiently utilize system resources while maintaining predictable performance under varying load conditions. The comprehensive monitoring, optimization, and troubleshooting capabilities ensure your workflows can handle production workloads cost-effectively and reliably.`;