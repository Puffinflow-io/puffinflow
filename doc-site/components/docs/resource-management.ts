export const resourceManagementMarkdown = `# Resource Management

Control CPU, memory, GPU, and custom resources to build scalable, production-ready workflows. Puffinflow provides intelligent resource allocation, quota management, and optimization patterns to prevent resource exhaustion and ensure fair allocation.

---

## Quick Start: Resource-Controlled Workflow

**See resource management in action with a complete example:**

**File: resource_demo.py**

\`\`\`python
     1→import asyncio
     2→from puffinflow import Agent, state, Priority, cpu_intensive, memory_intensive
     3→from puffinflow.core.resources import ResourcePool, PriorityAllocator
     4→
     5→# Create resource pool with limits
     6→resource_pool = ResourcePool(
     7→    total_cpu=8.0,      # 8 CPU cores available
     8→    total_memory=16384,  # 16GB memory available
     9→    total_gpu=2.0,      # 2 GPU units available
    10→    allocator=PriorityAllocator()
    11→)
    12→
    13→# Create agent with resource management
    14→agent = Agent("resource-demo", resource_pool=resource_pool)
    15→
    16→@state(cpu=1.0, memory=256, timeout=30.0, priority=Priority.NORMAL)
    17→async def fetch_data(context):
    18→    print("📊 Fetching data (light resource usage)...")
    19→    await asyncio.sleep(1)
    20→    context.set_variable("raw_data", list(range(10000)))
    21→    print("✅ Data fetched")
    22→    return "process_data"
    23→
    24→@cpu_intensive  # Uses 4.0 CPU, 1024MB memory profile
    25→async def process_data(context):
    26→    print("🔥 CPU-intensive processing...")
    27→    data = context.get_variable("raw_data")
    28→    
    29→    # Simulate CPU-heavy work
    30→    result = sum(x * x for x in data)
    31→    await asyncio.sleep(2)
    32→    
    33→    context.set_variable("processed_result", result)
    34→    print(f"✅ Processing complete: {result}")
    35→    return "analyze_memory"
    36→
    37→@memory_intensive  # Uses 2.0 CPU, 4096MB memory profile
    38→async def analyze_memory(context):
    39→    print("💾 Memory-intensive analysis...")
    40→    
    41→    # Simulate memory-heavy work
    42→    large_data = [[i] * 1000 for i in range(1000)]
    43→    analysis = {"memory_usage": "high", "data_size": len(large_data)}
    44→    
    45→    context.set_variable("analysis", analysis)
    46→    print("✅ Memory analysis complete")
    47→    return "gpu_task"
    48→
    49→@state(cpu=2.0, memory=2048, gpu=1.0, timeout=60.0, priority=Priority.HIGH)
    50→async def gpu_task(context):
    51→    print("🎮 GPU-accelerated computation...")
    52→    await asyncio.sleep(1.5)
    53→    
    54→    gpu_result = {"model_inference": "complete", "accuracy": 0.95}
    55→    context.set_variable("gpu_result", gpu_result)
    56→    print("✅ GPU task complete")
    57→    return None
    58→
    59→# Register states
    60→agent.add_state("fetch_data", fetch_data)
    61→agent.add_state("process_data", process_data, dependencies=["fetch_data"])
    62→agent.add_state("analyze_memory", analyze_memory, dependencies=["process_data"])
    63→agent.add_state("gpu_task", gpu_task, dependencies=["analyze_memory"])
    64→
    65→async def main():
    66→    print("🚀 Starting resource-managed workflow...")
    67→    print(f"Available resources: {resource_pool.total_cpu} CPU, {resource_pool.total_memory}MB RAM, {resource_pool.total_gpu} GPU")
    68→    
    69→    result = await agent.run()
    70→    
    71→    print("🎯 Workflow complete!")
    72→    print(f"Final results: {result.get_variable('gpu_result')}")
    73→
    74→if __name__ == "__main__":
    75→    asyncio.run(main())
\`\`\`

**Run the example:**
\`\`\`bash
python resource_demo.py
\`\`\`

**Output:**
\`\`\`
🚀 Starting resource-managed workflow...
Available resources: 8.0 CPU, 16384MB RAM, 2.0 GPU
📊 Fetching data (light resource usage)...
✅ Data fetched
🔥 CPU-intensive processing...
✅ Processing complete: 333283335000
💾 Memory-intensive analysis...
✅ Memory analysis complete
🎮 GPU-accelerated computation...
✅ GPU task complete  
🎯 Workflow complete!
Final results: {'model_inference': 'complete', 'accuracy': 0.95}
\`\`\`

---

## Resource Types and Allocation

### System Resources

Puffinflow manages these core system resources:

| Resource | Unit | Range | Purpose | Example Usage |
|----------|------|-------|---------|---------------|
| **CPU** | Cores | 0.1 - 16.0 | Processing power | Machine learning, data processing |
| **Memory** | MB | 64 - 32768 | RAM allocation | Large datasets, caching |
| **GPU** | Units | 0.0 - 8.0 | Graphics processing | Neural networks, image processing |
| **I/O** | Weight | 0.1 - 10.0 | Disk/network priority | File operations, database queries |
| **Network** | Weight | 0.1 - 10.0 | Network bandwidth | API calls, data transfer |

### Resource Profiles

**Ready-to-use profiles for common scenarios:**

**File: profiles_demo.py**

\`\`\`python
     1→from puffinflow import state, cpu_intensive, memory_intensive, gpu_accelerated
     2→from puffinflow import io_intensive, network_intensive, critical_state
     3→
     4→# Predefined profiles with automatic resource allocation
     5→
     6→@state(profile='minimal')      # 0.1 CPU, 50MB memory, 30s timeout
     7→async def lightweight_task(context):
     8→    print("⚡ Minimal resource task")
     9→
    10→@state(profile='standard')     # 1.0 CPU, 100MB memory, 60s timeout
    11→async def balanced_task(context):
    12→    print("⚖️ Standard balanced task")
    13→
    14→@cpu_intensive                 # 4.0 CPU, 1024MB memory, 300s timeout
    15→async def machine_learning_training(context):
    16→    print("🧠 Training ML model...")
    17→
    18→@memory_intensive              # 2.0 CPU, 4096MB memory, 600s timeout
    19→async def big_data_processing(context):
    20→    print("📊 Processing large dataset...")
    21→
    22→@gpu_accelerated               # 2.0 CPU, 2048MB memory, 1.0 GPU, 900s timeout
    23→async def neural_network_inference(context):
    24→    print("🎯 Running neural network inference...")
    25→
    26→@io_intensive                  # 1.0 CPU, 256MB memory, circuit breaker enabled
    27→async def file_processing(context):
    28→    print("📁 Processing files...")
    29→
    30→@network_intensive             # 1.0 CPU, 512MB memory, retry policies
    31→async def api_integration(context):
    32→    print("🌐 Calling external APIs...")
    33→
    34→@critical_state                # 4.0 CPU, 2048MB memory, highest priority
    35→async def emergency_response(context):
    36→    print("🚨 Critical operation...")
\`\`\`

---

## Advanced Resource Control

### Custom Resource Requirements

Define precise resource needs with ResourceRequirements:

**File: custom_resources.py**

\`\`\`python
     1→from puffinflow.core.resources import ResourceRequirements, ResourceType
     2→from puffinflow import Agent, state
     3→
     4→# Define custom resource requirements
     5→requirements = ResourceRequirements(
     6→    cpu_units=3.5,              # 3.5 CPU cores
     7→    memory_mb=2048.0,           # 2GB memory
     8→    io_weight=2.0,              # High I/O priority
     9→    network_weight=1.5,         # Medium network priority
    10→    gpu_units=0.5,              # Half GPU unit
    11→    priority_boost=10,          # Priority adjustment
    12→    timeout=180.0,              # 3 minute timeout
    13→    resource_types=ResourceType.CPU | ResourceType.MEMORY | ResourceType.GPU
    14→)
    15→
    16→@state(
    17→    cpu=requirements.cpu_units,
    18→    memory=int(requirements.memory_mb),
    19→    gpu=requirements.gpu_units,
    20→    timeout=requirements.timeout,
    21→    io_weight=requirements.io_weight
    22→)
    23→async def custom_resource_task(context):
    24→    print(f"🔧 Running with custom resources:")
    25→    print(f"   CPU: {requirements.cpu_units} cores")
    26→    print(f"   Memory: {requirements.memory_mb}MB")
    27→    print(f"   GPU: {requirements.gpu_units} units")
    28→    print(f"   I/O Weight: {requirements.io_weight}")
    29→    
    30→    # Simulate resource-intensive work
    31→    await asyncio.sleep(2)
    32→    print("✅ Custom resource task complete")
\`\`\`

### Resource Pools and Allocation Strategies

**Control how resources are allocated across states:**

**File: resource_pools.py**

\`\`\`python
     1→from puffinflow.core.resources import (
     2→    ResourcePool, FirstFitAllocator, BestFitAllocator,
     3→    PriorityAllocator, FairShareAllocator
     4→)
     5→from puffinflow import Agent, state, Priority
     6→
     7→# Strategy 1: First Fit (fastest allocation)
     8→first_fit_pool = ResourcePool(
     9→    total_cpu=16.0,
    10→    total_memory=32768,
    11→    allocator=FirstFitAllocator()  # First available slot
    12→)
    13→
    14→# Strategy 2: Best Fit (most efficient)
    15→best_fit_pool = ResourcePool(
    16→    total_cpu=16.0,
    17→    total_memory=32768,
    18→    allocator=BestFitAllocator()  # Most efficient fit
    19→)
    20→
    21→# Strategy 3: Priority-based (high priority first)
    22→priority_pool = ResourcePool(
    23→    total_cpu=16.0,
    24→    total_memory=32768,
    25→    allocator=PriorityAllocator()  # Priority-based allocation
    26→)
    27→
    28→# Strategy 4: Fair Share (equal distribution)
    29→fair_share_pool = ResourcePool(
    30→    total_cpu=16.0,
    31→    total_memory=32768,
    32→    allocator=FairShareAllocator()  # Equal distribution
    33→)
    34→
    35→# Create agents with different allocation strategies
    36→fast_agent = Agent("fast-allocator", resource_pool=first_fit_pool)
    37→efficient_agent = Agent("efficient-allocator", resource_pool=best_fit_pool)
    38→priority_agent = Agent("priority-allocator", resource_pool=priority_pool)
    39→fair_agent = Agent("fair-allocator", resource_pool=fair_share_pool)
    40→
    41→@state(cpu=2.0, memory=1024, priority=Priority.HIGH)
    42→async def high_priority_task(context):
    43→    print("🔥 High priority task executing...")
    44→    await asyncio.sleep(1)
    45→    print("✅ High priority task complete")
    46→
    47→@state(cpu=1.0, memory=512, priority=Priority.LOW)
    48→async def low_priority_task(context):
    49→    print("📋 Low priority task executing...")
    50→    await asyncio.sleep(2)
    51→    print("✅ Low priority task complete")
\`\`\`

---

## Resource Monitoring and Optimization

### Real-time Resource Tracking

**Monitor resource usage during workflow execution:**

**File: resource_monitoring.py**

\`\`\`python
     1→import asyncio
     2→import time
     3→from puffinflow import Agent, state
     4→from puffinflow.core.resources import ResourcePool, ResourceMonitor
     5→
     6→# Create monitored resource pool
     7→pool = ResourcePool(total_cpu=8.0, total_memory=8192)
     8→monitor = ResourceMonitor(pool)
     9→agent = Agent("monitored-agent", resource_pool=pool)
    10→
    11→@state(cpu=2.0, memory=1024, timeout=60.0)
    12→async def resource_tracked_task(context):
    13→    print("📊 Starting resource-tracked task...")
    14→    
    15→    # Get current resource usage
    16→    usage_start = monitor.get_current_usage()
    17→    print(f"🔋 Resources at start: {usage_start.cpu_used}/{usage_start.cpu_total} CPU, {usage_start.memory_used}/{usage_start.memory_total}MB memory")
    18→    
    19→    # Simulate work with periodic monitoring
    20→    for i in range(5):
    21→        await asyncio.sleep(0.5)
    22→        current_usage = monitor.get_current_usage()
    23→        print(f"   Step {i+1}: CPU {current_usage.cpu_utilization:.1f}%, Memory {current_usage.memory_utilization:.1f}%")
    24→    
    25→    # Check for resource warnings
    26→    if monitor.is_resource_constrained():
    27→        print("⚠️ System under resource pressure")
    28→        
    29→        # Get resource recommendations
    30→        recommendations = monitor.get_optimization_recommendations()
    31→        for rec in recommendations:
    32→            print(f"💡 Recommendation: {rec}")
    33→    
    34→    usage_end = monitor.get_current_usage()
    35→    print(f"🏁 Resources at end: {usage_end.cpu_used}/{usage_end.cpu_total} CPU, {usage_end.memory_used}/{usage_end.memory_total}MB memory")
    36→    
    37→    # Store resource metrics
    38→    context.set_output("resource_metrics", {
    39→        "start_cpu_utilization": usage_start.cpu_utilization,
    40→        "end_cpu_utilization": usage_end.cpu_utilization,
    41→        "start_memory_utilization": usage_start.memory_utilization,
    42→        "end_memory_utilization": usage_end.memory_utilization,
    43→        "peak_cpu": monitor.get_peak_cpu_usage(),
    44→        "peak_memory": monitor.get_peak_memory_usage()
    45→    })
    46→    
    47→    print("✅ Resource tracking complete")
    48→
    49→# Add resource monitoring callbacks
    50→@monitor.on_resource_threshold(cpu_threshold=0.8, memory_threshold=0.9)
    51→async def resource_warning_callback(usage_info):
    52→    print(f"🚨 Resource Warning: CPU {usage_info.cpu_utilization:.1f}%, Memory {usage_info.memory_utilization:.1f}%")
    53→
    54→@monitor.on_resource_exhausted()
    55→async def resource_exhausted_callback(resource_type):
    56→    print(f"💥 Resource Exhausted: {resource_type}")
    57→    # Could implement emergency response here
\`\`\`

### Resource Leak Detection

**Automatically detect and handle resource leaks:**

**File: leak_detection.py**

\`\`\`python
     1→from puffinflow import Agent, state
     2→from puffinflow.core.resources import ResourceLeakDetector, LeakDetectionConfig
     3→
     4→# Configure leak detection
     5→leak_config = LeakDetectionConfig(
     6→    memory_threshold_mb=1000,     # Alert if memory usage exceeds 1GB
     7→    cpu_threshold_percent=80,     # Alert if CPU usage exceeds 80%
     8→    monitoring_interval=5.0,      # Check every 5 seconds
     9→    leak_tolerance_duration=30.0, # Allow high usage for 30s before alerting
    10→    auto_cleanup=True             # Automatically clean up detected leaks
    11→)
    12→
    13→leak_detector = ResourceLeakDetector(leak_config)
    14→agent = Agent("leak-monitored-agent")
    15→
    16→@state(cpu=1.0, memory=512, timeout=120.0)
    17→async def potential_leak_task(context):
    18→    print("🔍 Task with potential resource leak...")
    19→    
    20→    # Start leak detection for this task
    21→    with leak_detector.monitor_task("potential_leak_task"):
    22→        # Simulate gradual memory leak
    23→        data_accumulator = []
    24→        
    25→        for i in range(100):
    26→            # Each iteration "leaks" more memory
    27→            large_data = [j for j in range(i * 1000)]
    28→            data_accumulator.append(large_data)
    29→            
    30→            await asyncio.sleep(0.1)
    31→            
    32→            # Check if leak detector found issues
    33→            if leak_detector.has_detected_leaks():
    34→                print("🚨 Memory leak detected!")
    35→                leaks = leak_detector.get_detected_leaks()
    36→                
    37→                for leak in leaks:
    38→                    print(f"   Leak type: {leak.resource_type}")
    39→                    print(f"   Current usage: {leak.current_usage}")
    40→                    print(f"   Threshold: {leak.threshold}")
    41→                    print(f"   Duration: {leak.duration_seconds}s")
    42→                
    43→                # Manual cleanup if auto_cleanup is disabled
    44→                if not leak_config.auto_cleanup:
    45→                    print("🧹 Manual cleanup triggered")
    46→                    data_accumulator.clear()  # Clean up the leak
    47→                    break
    48→    
    49→    print("✅ Task complete (leak detection finished)")
    50→
    51→# Leak detection callbacks
    52→@leak_detector.on_leak_detected
    53→async def handle_memory_leak(leak_info):
    54→    print(f"💧 Leak detected in {leak_info.task_name}: {leak_info.resource_type}")
    55→    
    56→    # Could implement custom cleanup logic here
    57→    if leak_info.severity == "critical":
    58→        print("🚨 Critical leak - implementing emergency measures")
    59→        # Emergency response (restart task, scale resources, etc.)
    60→
    61→@leak_detector.on_leak_resolved
    62→async def handle_leak_resolved(leak_info):
    63→    print(f"✅ Leak resolved in {leak_info.task_name}")
\`\`\`

---

## Resource Quotas and Limits

### Setting Resource Quotas

**Enforce resource limits and quotas across workflows:**

**File: resource_quotas.py**

\`\`\`python
     1→from puffinflow import Agent, state, Priority
     2→from puffinflow.core.resources import ResourceQuota, QuotaManager, QuotaPolicy
     3→
     4→# Define resource quotas for different user types
     5→basic_quota = ResourceQuota(
     6→    max_cpu=2.0,           # 2 CPU cores max
     7→    max_memory=1024,       # 1GB memory max
     8→    max_gpu=0.0,           # No GPU access
     9→    max_concurrent_states=3,  # 3 states max at once
    10→    daily_cpu_hours=10.0,  # 10 CPU hours per day
    11→    daily_memory_gb_hours=5.0  # 5GB-hours per day
    12→)
    13→
    14→premium_quota = ResourceQuota(
    15→    max_cpu=8.0,           # 8 CPU cores max
    16→    max_memory=8192,       # 8GB memory max
    17→    max_gpu=2.0,           # 2 GPU units max
    18→    max_concurrent_states=10,  # 10 states max at once
    19→    daily_cpu_hours=100.0, # 100 CPU hours per day
    20→    daily_memory_gb_hours=50.0  # 50GB-hours per day
    21→)
    22→
    23→# Create quota manager
    24→quota_manager = QuotaManager()
    25→quota_manager.set_user_quota("basic_user", basic_quota)
    26→quota_manager.set_user_quota("premium_user", premium_quota)
    27→
    28→# Quota policy configuration
    29→quota_policy = QuotaPolicy(
    30→    enforcement_mode="strict",     # strict, warning, or disabled
    31→    grace_period_seconds=60.0,     # Allow 60s over quota
    32→    quota_reset_schedule="daily",  # daily, weekly, monthly
    33→    overage_penalty_factor=2.0     # 2x resource cost for overages
    34→)
    35→
    36→quota_manager.set_policy(quota_policy)
    37→
    38→# Create agent with quota enforcement
    39→agent = Agent("quota-managed-agent")
    40→agent.set_quota_manager(quota_manager)
    41→
    42→@state(cpu=1.0, memory=512, timeout=60.0)
    43→async def quota_checked_task(context):
    44→    print("🎫 Starting quota-checked task...")
    45→    
    46→    # Get current quota usage
    47→    user_id = context.get_variable("user_id", "basic_user")
    48→    quota_usage = quota_manager.get_user_usage(user_id)
    49→    
    50→    print(f"📊 Current quota usage for {user_id}:")
    51→    print(f"   CPU: {quota_usage.cpu_used:.1f}/{quota_usage.cpu_limit:.1f} cores")
    52→    print(f"   Memory: {quota_usage.memory_used}/{quota_usage.memory_limit}MB")
    53→    print(f"   GPU: {quota_usage.gpu_used:.1f}/{quota_usage.gpu_limit:.1f} units")
    54→    print(f"   Daily CPU hours: {quota_usage.daily_cpu_hours_used:.1f}/{quota_usage.daily_cpu_hours_limit:.1f}")
    55→    
    56→    # Check if we're approaching limits
    57→    if quota_usage.is_approaching_limit(threshold=0.8):
    58→        print("⚠️ Approaching quota limits")
    59→        approaching_limits = quota_usage.get_approaching_limits()
    60→        for limit_type in approaching_limits:
    61→            print(f"   {limit_type} usage is at {quota_usage.get_utilization(limit_type):.1f}%")
    62→    
    63→    # Simulate work
    64→    await asyncio.sleep(2)
    65→    
    66→    print("✅ Quota-checked task complete")
    67→
    68→# Quota violation handlers
    69→@quota_manager.on_quota_exceeded
    70→async def handle_quota_exceeded(user_id, resource_type, current_usage, limit):
    71→    print(f"🚫 Quota exceeded for {user_id}: {resource_type} usage {current_usage} > limit {limit}")
    72→    
    73→    # Could implement quota upgrade prompts, throttling, etc.
    74→    if resource_type == "cpu":
    75→        print("💡 Consider upgrading to premium for higher CPU limits")
    76→
    77→@quota_manager.on_quota_warning
    78→async def handle_quota_warning(user_id, resource_type, utilization_percent):
    79→    print(f"⚠️ Quota warning for {user_id}: {resource_type} at {utilization_percent:.1f}% capacity")
\`\`\`

---

## Performance Optimization Patterns

### Resource-Aware Task Scheduling

**Optimize task execution based on available resources:**

**File: resource_scheduling.py**

\`\`\`python
     1→from puffinflow import Agent, state, Priority
     2→from puffinflow.core.resources import ResourceAwareScheduler, SchedulingStrategy
     3→
     4→# Create resource-aware scheduler
     5→scheduler = ResourceAwareScheduler(
     6→    strategy=SchedulingStrategy.RESOURCE_OPTIMAL,  # Balance resource utilization
     7→    preemption_enabled=True,                       # Allow high priority preemption
     8→    load_balancing_enabled=True,                   # Distribute load evenly
     9→    resource_prediction_enabled=True              # Predict resource needs
    10→)
    11→
    12→agent = Agent("resource-scheduled-agent")
    13→agent.set_scheduler(scheduler)
    14→
    15→# States with different resource profiles
    16→@state(cpu=0.5, memory=128, priority=Priority.LOW)
    17→async def background_cleanup(context):
    18→    print("🧹 Background cleanup (low priority, light resources)...")
    19→    await asyncio.sleep(5)  # Long-running but low priority
    20→    print("✅ Cleanup complete")
    21→
    22→@state(cpu=2.0, memory=1024, priority=Priority.NORMAL)
    23→async def data_processing(context):
    24→    print("⚙️ Data processing (normal priority, moderate resources)...")
    25→    await asyncio.sleep(3)
    26→    context.set_variable("processed_data", {"status": "complete"})
    27→    print("✅ Data processing complete")
    28→    return "generate_report"
    29→
    30→@state(cpu=1.0, memory=512, priority=Priority.HIGH)
    31→async def generate_report(context):
    32→    print("📊 Report generation (high priority)...")
    33→    processed_data = context.get_variable("processed_data")
    34→    await asyncio.sleep(1)
    35→    print("✅ Report generated")
    36→
    37→@state(cpu=4.0, memory=2048, priority=Priority.CRITICAL)
    38→async def emergency_analysis(context):
    39→    print("🚨 Emergency analysis (critical priority, high resources)...")
    40→    await asyncio.sleep(2)
    41→    print("✅ Emergency analysis complete")
    42→
    43→# Register states
    44→agent.add_state("cleanup", background_cleanup)
    45→agent.add_state("process", data_processing)
    46→agent.add_state("report", generate_report)
    47→agent.add_state("emergency", emergency_analysis)
    48→
    49→async def demonstrate_scheduling():
    50→    print("🎯 Demonstrating resource-aware scheduling...")
    51→    
    52→    # Start multiple workflows with different priorities
    53→    tasks = []
    54→    
    55→    # Start background cleanup (low priority)
    56→    cleanup_agent = Agent("cleanup-agent")
    57→    cleanup_agent.add_state("cleanup", background_cleanup)
    58→    tasks.append(cleanup_agent.run())
    59→    
    60→    # Start normal data processing
    61→    process_agent = Agent("process-agent")
    62→    process_agent.add_state("process", data_processing)
    63→    process_agent.add_state("report", generate_report)
    64→    tasks.append(process_agent.run())
    65→    
    66→    # Emergency task (should preempt others)
    67→    emergency_agent = Agent("emergency-agent")
    68→    emergency_agent.add_state("emergency", emergency_analysis)
    69→    tasks.append(emergency_agent.run())
    70→    
    71→    # Wait for all tasks
    72→    results = await asyncio.gather(*tasks)
    73→    print("🏁 All scheduled tasks complete")
    74→    
    75→    return results
\`\`\`

### Resource Pooling Strategies

**Implement different pooling strategies for optimal resource usage:**

**File: pooling_strategies.py**

\`\`\`python
     1→from puffinflow import Agent, state
     2→from puffinflow.core.resources import (
     3→    DynamicResourcePool, StaticResourcePool,
     4→    ElasticResourcePool, HierarchicalResourcePool
     5→)
     6→
     7→# Strategy 1: Static Pool (fixed resources)
     8→static_pool = StaticResourcePool(
     9→    total_cpu=8.0,
    10→    total_memory=16384,
    11→    total_gpu=2.0
    12→)
    13→
    14→# Strategy 2: Dynamic Pool (adjusts based on demand)
    15→dynamic_pool = DynamicResourcePool(
    16→    min_cpu=2.0,           # Minimum guaranteed resources
    17→    max_cpu=16.0,          # Maximum available resources
    18→    min_memory=1024,
    19→    max_memory=32768,
    20→    scaling_factor=1.5,    # Scale by 1.5x when needed
    21→    scale_up_threshold=0.8, # Scale up at 80% utilization
    22→    scale_down_threshold=0.3 # Scale down at 30% utilization
    23→)
    24→
    25→# Strategy 3: Elastic Pool (cloud-like scaling)
    26→elastic_pool = ElasticResourcePool(
    27→    base_cpu=4.0,          # Always-available base resources
    28→    base_memory=4096,
    29→    burst_cpu=12.0,        # Additional burst capacity
    30→    burst_memory=12288,
    31→    burst_cost_multiplier=2.0,  # Burst resources cost 2x
    32→    burst_timeout=300.0    # Burst capacity timeout
    33→)
    34→
    35→# Strategy 4: Hierarchical Pool (tiered resource allocation)
    36→hierarchical_pool = HierarchicalResourcePool([
    37→    {"name": "critical", "cpu": 4.0, "memory": 8192, "priority": "CRITICAL"},
    38→    {"name": "high", "cpu": 2.0, "memory": 4096, "priority": "HIGH"},
    39→    {"name": "normal", "cpu": 1.0, "memory": 2048, "priority": "NORMAL"},
    40→    {"name": "low", "cpu": 0.5, "memory": 1024, "priority": "LOW"}
    41→])
    42→
    43→# Demonstrate different pooling strategies
    44→agents = {
    45→    "static": Agent("static-agent", resource_pool=static_pool),
    46→    "dynamic": Agent("dynamic-agent", resource_pool=dynamic_pool),
    47→    "elastic": Agent("elastic-agent", resource_pool=elastic_pool),
    48→    "hierarchical": Agent("hierarchical-agent", resource_pool=hierarchical_pool)
    49→}
    50→
    51→@state(cpu=2.0, memory=1024, timeout=30.0)
    52→async def test_pooling_strategy(context):
    53→    strategy_name = context.get_variable("strategy_name")
    54→    print(f"🎯 Testing {strategy_name} pooling strategy...")
    55→    
    56→    # Simulate resource-intensive work
    57→    await asyncio.sleep(2)
    58→    
    59→    print(f"✅ {strategy_name} strategy test complete")
    60→    return {"strategy": strategy_name, "status": "success"}
    61→
    62→async def compare_pooling_strategies():
    63→    print("🔬 Comparing resource pooling strategies...")
    64→    
    65→    results = {}
    66→    
    67→    for strategy_name, agent in agents.items():
    68→        agent.add_state("test", test_pooling_strategy)
    69→        
    70→        start_time = time.time()
    71→        result = await agent.run(initial_context={"strategy_name": strategy_name})
    72→        end_time = time.time()
    73→        
    74→        results[strategy_name] = {
    75→            "execution_time": end_time - start_time,
    76→            "result": result.get_variable("strategy_name")
    77→        }
    78→    
    79→    print("📊 Pooling Strategy Comparison Results:")
    80→    for strategy, data in results.items():
    81→        print(f"   {strategy}: {data['execution_time']:.2f}s")
    82→    
    83→    return results
\`\`\`

---

## Production Best Practices

### Resource Management in Production

**Essential patterns for production deployments:**

**File: production_patterns.py**

\`\`\`python
     1→import logging
     2→from puffinflow import Agent, state, Priority
     3→from puffinflow.core.resources import (
     4→    ProductionResourceManager, ResourceHealthChecker,
     5→    ResourceAlertManager, ResourceOptimizer
     6→)
     7→
     8→# Configure production resource management
     9→resource_manager = ProductionResourceManager(
    10→    cpu_overcommit_ratio=1.2,      # 20% CPU overcommit
    11→    memory_overcommit_ratio=1.0,   # No memory overcommit
    12→    resource_reservation_buffer=0.1, # 10% buffer for emergencies
    13→    enable_resource_prediction=True,  # Predict future needs
    14→    enable_auto_scaling=True,        # Auto-scale resources
    15→    enable_cost_optimization=True    # Optimize for cost
    16→)
    17→
    18→# Health checking for resource systems
    19→health_checker = ResourceHealthChecker(
    20→    check_interval_seconds=30.0,     # Check every 30 seconds
    21→    cpu_health_threshold=0.9,        # Alert if CPU > 90%
    22→    memory_health_threshold=0.85,    # Alert if memory > 85%
    23→    disk_health_threshold=0.8,       # Alert if disk > 80%
    24→    network_health_threshold=0.7     # Alert if network > 70%
    25→)
    26→
    27→# Alert management for resource issues
    28→alert_manager = ResourceAlertManager(
    29→    email_alerts=True,
    30→    slack_webhook_url="https://hooks.slack.com/services/...",
    31→    pagerduty_integration_key="your_pagerduty_key",
    32→    alert_escalation_timeout=300.0   # Escalate after 5 minutes
    33→)
    34→
    35→# Resource optimizer for cost and performance
    36→optimizer = ResourceOptimizer(
    37→    optimization_goal="balanced",    # balanced, performance, cost
    38→    optimization_interval=3600.0,    # Optimize every hour
    39→    min_optimization_improvement=0.05 # Must improve by 5%
    40→)
    41→
    42→# Production agent with full resource management
    43→production_agent = Agent(
    44→    "production-workflow",
    45→    resource_manager=resource_manager
    46→)
    47→
    48→@state(cpu=2.0, memory=1024, timeout=300.0, priority=Priority.HIGH)
    49→async def production_critical_task(context):
    50→    print("🏭 Production critical task starting...")
    51→    
    52→    try:
    53→        # Get resource health status
    54→        health_status = health_checker.get_current_health()
    55→        
    56→        if not health_status.is_healthy:
    57→            print("⚠️ System health issues detected:")
    58→            for issue in health_status.issues:
    59→                print(f"   {issue.severity}: {issue.message}")
    60→                
    61→                # Send alert for critical issues
    62→                if issue.severity == "critical":
    63→                    await alert_manager.send_alert(
    64→                        title="Critical Resource Issue",
    65→                        message=issue.message,
    66→                        severity="critical"
    67→                    )
    68→        
    69→        # Get resource optimization recommendations
    70→        optimization_recs = optimizer.get_recommendations()
    71→        if optimization_recs:
    72→            print("💡 Resource optimization recommendations:")
    73→            for rec in optimization_recs:
    74→                print(f"   {rec.type}: {rec.description} (savings: {rec.estimated_savings})")
    75→        
    76→        # Simulate critical production work
    77→        await asyncio.sleep(10)
    78→        
    79→        # Log resource usage for analysis
    80→        resource_usage = resource_manager.get_current_usage()
    81→        logging.info(f"Production task resource usage: CPU {resource_usage.cpu_utilization:.1f}%, Memory {resource_usage.memory_utilization:.1f}%")
    82→        
    83→        print("✅ Production critical task complete")
    84→        
    85→    except Exception as e:
    86→        # Handle production errors with resource context
    87→        resource_context = resource_manager.get_resource_context()
    88→        error_message = f"Production task failed: {str(e)}"
    89→        error_details = f"Resource context: {resource_context}"
    90→        
    91→        logging.error(f"{error_message}\\n{error_details}")
    92→        
    93→        # Send critical alert
    94→        await alert_manager.send_alert(
    95→            title="Production Task Failure",
    96→            message=f"{error_message}\\n{error_details}",
    97→            severity="critical"
    98→        )
    99→        
   100→        raise
   101→
   102→# Resource monitoring callbacks for production
   103→@resource_manager.on_resource_pressure
   104→async def handle_resource_pressure(pressure_info):
   105→    print(f"🔥 Resource pressure detected: {pressure_info.resource_type} at {pressure_info.utilization:.1f}%")
   106→    
   107→    # Implement load shedding or scaling
   108→    if pressure_info.utilization > 0.95:
   109→        print("🚨 Implementing emergency load shedding")
   110→        # Could pause non-critical tasks, scale up resources, etc.
   111→
   112→@optimizer.on_optimization_completed
   113→async def handle_optimization_complete(optimization_result):
   114→    print(f"⚡ Resource optimization complete:")
   115→    print(f"   Performance improvement: {optimization_result.performance_improvement:.1f}%")
   116→    print(f"   Cost reduction: $\{optimization_result.cost_savings:.2f}/hour")
   117→    print(f"   Resource efficiency: {optimization_result.efficiency_improvement:.1f}%")
\`\`\`

---

## Quick Reference

### Resource Allocation Syntax

\`\`\`python
# Basic resource allocation
@state(cpu=2.0, memory=1024, timeout=60.0)
async def my_task(context): pass

# Using predefined profiles
@cpu_intensive        # 4.0 CPU, 1024MB memory
@memory_intensive      # 2.0 CPU, 4096MB memory
@gpu_accelerated       # 2.0 CPU, 2048MB memory, 1.0 GPU

# Priority-based allocation
@state(cpu=1.0, memory=512, priority=Priority.HIGH)
async def important_task(context): pass

# Custom resource requirements
@state(cpu=3.5, memory=2048, gpu=0.5, io_weight=2.0, network_weight=1.5)
async def complex_task(context): pass
\`\`\`

### Resource Pool Creation

\`\`\`python
from puffinflow.core.resources import ResourcePool, PriorityAllocator

# Create resource pool
pool = ResourcePool(
    total_cpu=16.0,
    total_memory=32768,
    total_gpu=4.0,
    allocator=PriorityAllocator()
)

# Agent with resource pool
agent = Agent("my-agent", resource_pool=pool)
\`\`\`

### Resource Monitoring

\`\`\`python
from puffinflow.core.resources import ResourceMonitor

monitor = ResourceMonitor(resource_pool)

# Get current usage
usage = monitor.get_current_usage()
print(f"CPU: {usage.cpu_utilization:.1f}%")
print(f"Memory: {usage.memory_utilization:.1f}%")

# Check for resource constraints
if monitor.is_resource_constrained():
    recommendations = monitor.get_optimization_recommendations()
\`\`\`

Resource management in Puffinflow ensures your workflows run efficiently, scale properly, and maintain high performance under varying load conditions. Start with basic resource allocation and gradually adopt advanced patterns as your needs grow.
`.trim();