export const resourceManagementMarkdown = `# Resource Management

Puffinflow provides sophisticated resource management to ensure optimal system utilization, prevent resource exhaustion, and maintain fair allocation across workflows. Understanding how to configure resource constraints is crucial for building scalable, reliable workflows that perform well under load.

## Understanding Resource Management

Resource management in Puffinflow involves several key concepts:

- **Resource Allocation**: Reserving CPU, memory, GPU, and I/O capacity for operations
- **Rate Limiting**: Controlling the frequency of operations to respect API quotas and service limits
- **Coordination**: Managing access to shared resources using synchronization primitives
- **Quotas**: Enforcing per-user, per-tenant, or per-application resource limits
- **Priority Management**: Ensuring critical operations get precedence during resource contention

## How to Think About Your Resource Needs

Before configuring resource constraints, ask yourself these questions:

### 1. **What Type of Work Are You Doing?**

\`\`\`python
# CPU-intensive work (data processing, calculations)
@state(cpu=4.0, memory=1024)
async def heavy_computation(context):
    # Mathematical modeling, data analysis, cryptography
    pass

# Memory-intensive work (large datasets, caching)
@state(cpu=2.0, memory=8192)
async def large_dataset_processing(context):
    # Data science, machine learning training, large file processing
    pass

# I/O-intensive work (file operations, database queries)
@state(cpu=1.0, memory=512, io=10.0)
async def file_processing(context):
    # File uploads, database operations, log processing
    pass

# GPU-accelerated work (machine learning, graphics)
@state(cpu=2.0, memory=4096, gpu=1.0)
async def ml_inference(context):
    # Deep learning, computer vision, scientific computing
    pass
\`\`\`

### 2. **What Are Your Performance Requirements?**

\`\`\`python
# Real-time operations (low latency required)
@state(cpu=2.0, memory=1024, priority=Priority.HIGH, timeout=5.0)
async def real_time_api(context):
    # User-facing APIs, real-time analytics
    pass

# Batch operations (high throughput, can be slower)
@state(cpu=1.0, memory=512, priority=Priority.NORMAL, timeout=300.0)
async def batch_processing(context):
    # Data pipelines, report generation, cleanup tasks
    pass

# Background operations (can be interrupted)
@state(cpu=0.5, memory=256, priority=Priority.LOW, preemptible=True)
async def background_maintenance(context):
    # Garbage collection, statistics gathering, archiving
    pass
\`\`\`

### 3. **What External Dependencies Do You Have?**

\`\`\`python
# External API calls (need rate limiting)
@state(rate_limit=10.0, burst_limit=20, timeout=15.0)
async def external_api_call(context):
    # Third-party APIs, web services, external databases
    pass

# Shared resources (need coordination)
@state(semaphore=3, timeout=30.0)
async def shared_database_access(context):
    # Connection pools, shared files, exclusive resources
    pass

# Critical system operations (need exclusive access)
@state(mutex=True, priority=Priority.CRITICAL, timeout=60.0)
async def system_maintenance(context):
    # Schema migrations, system updates, configuration changes
    pass
\`\`\`

---

## Step-by-Step Resource Configuration Guide

### Step 1: Analyze Your Workload Characteristics

Start by understanding what your operation actually does:

\`\`\`python
import asyncio
import time
from puffinflow import Agent
from puffinflow.decorators import state
from puffinflow.core.agent.state import Priority

agent = Agent("resource-analysis-agent")

# Profile your operations first
@state  # Start with no constraints to understand baseline behavior
async def analyze_workload(context):
    """
    Profile this operation to understand its resource characteristics:
    1. How much CPU does it use?
    2. How much memory does it need?
    3. Does it do I/O operations?
    4. How long does it typically take?
    5. Does it call external services?
    """
    start_time = time.time()

    # Your actual workload here
    # Example: data processing
    data_size = 1000000
    processed_items = []

    for i in range(data_size):
        # CPU-intensive calculation
        result = i ** 2 + i ** 0.5
        processed_items.append(result)

        # Simulate periodic I/O
        if i % 100000 == 0:
            await asyncio.sleep(0.01)  # I/O operation

    execution_time = time.time() - start_time
    memory_estimate = len(processed_items) * 8  # Rough memory usage

    print(f"📊 Workload Analysis:")
    print(f"   ⏱️ Execution time: {execution_time:.2f} seconds")
    print(f"   💾 Memory usage: ~{memory_estimate / 1024 / 1024:.1f} MB")
    print(f"   🔢 Items processed: {data_size:,}")
    print(f"   ⚡ Processing rate: {data_size / execution_time:.0f} items/sec")

    context.set_variable("workload_profile", {
        "execution_time": execution_time,
        "memory_mb": memory_estimate / 1024 / 1024,
        "processing_rate": data_size / execution_time,
        "io_operations": data_size // 100000
    })
\`\`\`

### Step 2: Set Appropriate Resource Limits

Based on your analysis, configure resource constraints:

\`\`\`python
# Based on profiling, this operation needs:
# - High CPU (mathematical calculations)
# - Moderate memory (storing results)
# - Some I/O (periodic writes)
@state(
    cpu=3.0,        # 3 CPU cores (observed high CPU usage)
    memory=2048,    # 2GB memory (observed ~500MB + safety margin)
    io=5.0,         # Medium I/O priority (periodic I/O operations)
    timeout=120.0   # 2-minute timeout (observed 60s + safety margin)
)
async def optimized_data_processing(context):
    """Now properly configured based on profiling"""
    start_time = time.time()

    # Same workload as before, but now with proper resource allocation
    data_size = 1000000
    processed_items = []

    for i in range(data_size):
        result = i ** 2 + i ** 0.5
        processed_items.append(result)

        if i % 100000 == 0:
            await asyncio.sleep(0.01)

    execution_time = time.time() - start_time
    print(f"✅ Optimized processing completed in {execution_time:.2f}s")

    context.set_variable("optimized_result", {
        "execution_time": execution_time,
        "items_processed": data_size
    })
\`\`\`

### Step 3: Add Coordination for Shared Resources

If your operation accesses shared resources, add coordination:

\`\`\`python
# Shared database connection pool (max 5 connections)
@state(
    cpu=1.0,
    memory=512,
    semaphore=5,    # Max 5 concurrent database operations
    timeout=30.0
)
async def database_operation(context):
    """Multiple instances can run, but limited by connection pool"""
    print("🗄️ Accessing shared database...")

    # Simulate database work
    await asyncio.sleep(2.0)

    print("✅ Database operation completed")
    context.set_variable("db_result", "success")

# Exclusive system resource (only one at a time)
@state(
    cpu=2.0,
    memory=1024,
    mutex=True,     # Exclusive access required
    timeout=60.0,
    priority=Priority.HIGH
)
async def exclusive_system_operation(context):
    """Only one instance can run at a time"""
    print("🔒 Performing exclusive system operation...")

    # Critical system work that requires exclusive access
    await asyncio.sleep(5.0)

    print("✅ Exclusive operation completed")
    context.set_variable("exclusive_result", "success")

# Synchronized batch processing (wait for all participants)
@state(
    cpu=1.0,
    memory=512,
    barrier=3,      # Wait for 3 instances to start together
    timeout=45.0
)
async def synchronized_batch_job(context):
    """Starts only when 3 instances are ready"""
    job_id = context.get_state("job_id", "unknown")

    print(f"🚀 Starting synchronized job {job_id}...")

    # Coordinated work
    await asyncio.sleep(3.0)

    print(f"✅ Synchronized job {job_id} completed")
    context.set_variable(f"sync_result_{job_id}", "success")
\`\`\`

### Step 4: Configure Rate Limiting for External Services

For operations that call external APIs or services:

\`\`\`python
# High-volume API with burst capacity
@state(
    cpu=0.5,
    memory=256,
    rate_limit=10.0,    # 10 requests per second
    burst_limit=25,     # Can burst up to 25 requests
    timeout=15.0
)
async def high_volume_api_call(context):
    """API that can handle high volume with bursts"""
    call_id = context.get_state("api_calls", 0) + 1
    context.set_state("api_calls", call_id)

    print(f"🌐 High-volume API call #{call_id}")

    # Simulate API call
    await asyncio.sleep(0.1)

    context.set_variable(f"api_result_{call_id}", "success")

# Rate-limited premium service
@state(
    cpu=0.5,
    memory=256,
    rate_limit=2.0,     # 2 requests per second (expensive service)
    timeout=30.0
)
async def premium_service_call(context):
    """Expensive external service with strict rate limits"""
    call_id = context.get_state("premium_calls", 0) + 1
    context.set_state("premium_calls", call_id)

    print(f"💰 Premium service call #{call_id}")

    # Simulate slower, expensive service
    await asyncio.sleep(0.5)

    context.set_variable(f"premium_result_{call_id}", "success")

# Critical service with very limited capacity
@state(
    cpu=1.0,
    memory=512,
    rate_limit=0.5,     # 1 request every 2 seconds
    timeout=60.0,
    priority=Priority.HIGH
)
async def critical_service_call(context):
    """Critical service that must be used sparingly"""
    call_id = context.get_state("critical_calls", 0) + 1
    context.set_state("critical_calls", call_id)

    print(f"🔥 Critical service call #{call_id}")

    # Simulate critical service interaction
    await asyncio.sleep(2.0)

    context.set_variable(f"critical_result_{call_id}", "success")
\`\`\`

---

## Common Resource Management Patterns

### Pattern 1: CPU-Intensive Scientific Computing

\`\`\`python
@state(
    cpu=8.0,            # Use 8 CPU cores for parallel computation
    memory=4096,        # 4GB for intermediate results
    timeout=1800.0,     # 30-minute timeout for long computations
    priority=Priority.NORMAL
)
async def scientific_simulation(context):
    """
    Use this pattern for:
    - Mathematical modeling
    - Scientific simulations
    - Data analysis
    - Cryptographic operations
    """
    simulation_id = context.get_state("simulation_id", 1)

    print(f"🧮 Running scientific simulation #{simulation_id}")
    print(f"   📊 Allocated: 8 CPU cores, 4GB memory")

    # Simulate CPU-intensive scientific computation
    iterations = 1000000
    for i in range(iterations):
        # Complex mathematical operations
        result = (i ** 2.5) * 3.14159 + (i ** 0.5)

        if i % 100000 == 0:
            progress = (i / iterations) * 100
            print(f"   🔄 Progress: {progress:.1f}%")
            await asyncio.sleep(0)  # Yield control briefly

    print(f"✅ Scientific simulation #{simulation_id} completed")
    context.set_variable(f"simulation_result_{simulation_id}", {
        "iterations": iterations,
        "status": "completed"
    })
\`\`\`

### Pattern 2: Memory-Intensive Data Processing

\`\`\`python
@state(
    cpu=2.0,            # Moderate CPU for data manipulation
    memory=16384,       # 16GB for large datasets
    io=8.0,             # High I/O for reading/writing data
    timeout=3600.0,     # 1-hour timeout for large datasets
    priority=Priority.NORMAL
)
async def large_dataset_analysis(context):
    """
    Use this pattern for:
    - Big data processing
    - Machine learning training data preparation
    - Large file analysis
    - In-memory databases
    """
    dataset_id = context.get_state("dataset_id", "dataset_001")

    print(f"📊 Analyzing large dataset: {dataset_id}")
    print(f"   📊 Allocated: 2 CPU cores, 16GB memory, high I/O")

    # Simulate loading large dataset into memory
    print("   📥 Loading dataset into memory...")
    dataset_size = 10000000  # 10M records
    dataset = []

    for i in range(dataset_size):
        # Simulate complex data structure
        record = {
            "id": i,
            "value": i * 2.5,
            "category": f"cat_{i % 100}",
            "metadata": {"processed": False, "score": i * 0.001}
        }
        dataset.append(record)

        if i % 1000000 == 0:
            print(f"   📈 Loaded {i:,} records...")
            await asyncio.sleep(0)

    print("   🔄 Processing dataset...")
    processed_count = 0
    for record in dataset:
        # Data processing logic
        record["metadata"]["processed"] = True
        record["metadata"]["score"] *= 1.1
        processed_count += 1

        if processed_count % 1000000 == 0:
            print(f"   ⚙️ Processed {processed_count:,} records...")
            await asyncio.sleep(0)

    print(f"✅ Dataset analysis completed: {len(dataset):,} records")
    context.set_variable(f"analysis_result_{dataset_id}", {
        "records_processed": len(dataset),
        "status": "completed"
    })
\`\`\`

### Pattern 3: GPU-Accelerated Machine Learning

\`\`\`python
@state(
    cpu=4.0,            # 4 CPU cores for data preparation
    memory=8192,        # 8GB for model and data
    gpu=2.0,            # 2 GPU units for training/inference
    timeout=7200.0,     # 2-hour timeout for training
    priority=Priority.HIGH,
    rate_limit=1.0      # Limit GPU usage to prevent overload
)
async def ml_model_training(context):
    """
    Use this pattern for:
    - Deep learning model training
    - GPU-accelerated inference
    - Computer vision processing
    - Scientific computing with CUDA
    """
    model_id = context.get_state("model_id", "model_v1")

    print(f"🤖 Training ML model: {model_id}")
    print(f"   📊 Allocated: 4 CPU cores, 8GB memory, 2 GPUs")

    # Simulate ML training pipeline
    phases = [
        ("Data Loading", 30),
        ("Data Preprocessing", 45),
        ("Model Initialization", 15),
        ("Training Loop", 300),
        ("Model Validation", 60),
        ("Model Saving", 20)
    ]

    total_time = sum(duration for _, duration in phases)
    elapsed = 0

    for phase_name, duration in phases:
        print(f"   🧠 {phase_name}...")

        # Simulate GPU-intensive work
        await asyncio.sleep(duration / 100)  # Compressed time for demo
        elapsed += duration

        progress = (elapsed / total_time) * 100
        print(f"   📈 Overall progress: {progress:.1f}%")

    print(f"✅ ML model training completed: {model_id}")
    context.set_variable(f"model_result_{model_id}", {
        "model_id": model_id,
        "training_time": total_time,
        "status": "trained"
    })
\`\`\`

### Pattern 4: I/O-Intensive File Processing

\`\`\`python
@state(
    cpu=1.0,            # Low CPU for I/O coordination
    memory=1024,        # Moderate memory for buffers
    io=15.0,            # Very high I/O priority
    network=10.0,       # High network for remote files
    timeout=1800.0,     # 30-minute timeout for large files
    priority=Priority.NORMAL
)
async def bulk_file_processing(context):
    """
    Use this pattern for:
    - File uploads/downloads
    - Log processing
    - Backup operations
    - Data import/export
    """
    batch_id = context.get_state("batch_id", "batch_001")

    print(f"📁 Processing file batch: {batch_id}")
    print(f"   📊 Allocated: 1 CPU core, 1GB memory, high I/O priority")

    # Simulate bulk file operations
    file_count = 1000
    processed_files = 0
    total_size_mb = 0

    for file_num in range(file_count):
        file_size_mb = (file_num % 50) + 1  # 1-50MB files

        # Simulate file I/O operations
        print(f"   📄 Processing file {file_num + 1}/{file_count} ({file_size_mb}MB)")

        # Read operation
        await asyncio.sleep(file_size_mb * 0.001)  # Simulate read time

        # Process operation (minimal CPU)
        await asyncio.sleep(0.001)

        # Write operation
        await asyncio.sleep(file_size_mb * 0.001)  # Simulate write time

        processed_files += 1
        total_size_mb += file_size_mb

        if file_num % 100 == 0:
            print(f"   📈 Progress: {processed_files}/{file_count} files, {total_size_mb}MB")

    print(f"✅ Bulk file processing completed: {processed_files} files, {total_size_mb}MB")
    context.set_variable(f"file_result_{batch_id}", {
        "files_processed": processed_files,
        "total_size_mb": total_size_mb,
        "status": "completed"
    })
\`\`\`

---

## Advanced Coordination Patterns

### Pattern 1: Producer-Consumer with Semaphore

\`\`\`python
# Producer: Creates work items (limited by rate)
@state(
    cpu=1.0,
    memory=512,
    rate_limit=5.0,     # Produce 5 items per second
    timeout=60.0
)
async def work_producer(context):
    """Produces work items at a controlled rate"""
    item_id = context.get_state("produced_items", 0) + 1
    context.set_state("produced_items", item_id)

    work_item = {
        "id": item_id,
        "data": f"work_data_{item_id}",
        "created_at": time.time()
    }

    print(f"🏭 Produced work item #{item_id}")
    context.set_variable(f"work_item_{item_id}", work_item)

# Consumer: Processes work items (limited by resource pool)
@state(
    cpu=2.0,
    memory=1024,
    semaphore=3,        # Max 3 concurrent consumers
    timeout=30.0
)
async def work_consumer(context):
    """Consumes and processes work items"""
    consumer_id = context.get_state("consumer_id", 1)

    # Find available work item
    for i in range(1, 100):  # Look for work items
        work_item = context.get_variable(f"work_item_{i}")
        if work_item and not context.get_variable(f"processed_{i}"):
            print(f"🔧 Consumer {consumer_id} processing item #{i}")

            # Simulate processing work
            await asyncio.sleep(2.0)

            context.set_variable(f"processed_{i}", True)
            print(f"✅ Consumer {consumer_id} completed item #{i}")
            break
\`\`\`

### Pattern 2: Barrier Synchronization for Batch Jobs

\`\`\`python
# Phase 1: Data collection (all must complete before phase 2)
@state(
    cpu=1.0,
    memory=512,
    barrier=3,          # Wait for 3 data collectors
    timeout=120.0
)
async def data_collection_phase(context):
    """Collect data from different sources"""
    collector_id = context.get_state("collector_id", 1)

    print(f"📥 Data collector {collector_id} starting...")

    # Simulate data collection from different sources
    collection_time = 2.0 + (collector_id * 0.5)  # Different collection times
    await asyncio.sleep(collection_time)

    data = {
        "collector_id": collector_id,
        "data_size": collector_id * 1000,
        "collection_time": collection_time
    }

    context.set_variable(f"collected_data_{collector_id}", data)
    print(f"✅ Data collector {collector_id} completed")

# Phase 2: Data processing (starts only after all collection is done)
@state(
    cpu=4.0,
    memory=2048,
    timeout=180.0,
    priority=Priority.HIGH
)
async def data_processing_phase(context):
    """Process all collected data together"""
    print("🔄 Starting data processing phase...")

    # Gather all collected data
    all_data = []
    total_size = 0

    for i in range(1, 4):  # 3 collectors
        data = context.get_variable(f"collected_data_{i}")
        if data:
            all_data.append(data)
            total_size += data["data_size"]

    print(f"   📊 Processing {len(all_data)} datasets, total size: {total_size}")

    # Simulate intensive data processing
    await asyncio.sleep(5.0)

    result = {
        "datasets_processed": len(all_data),
        "total_size": total_size,
        "processing_time": 5.0
    }

    context.set_variable("processing_result", result)
    print("✅ Data processing phase completed")
\`\`\`

### Pattern 3: Lease-Based Resource Management

\`\`\`python
# Short-term exclusive access to shared resource
@state(
    cpu=1.0,
    memory=512,
    lease=10.0,         # 10-second lease on shared resource
    timeout=15.0
)
async def short_term_exclusive_access(context):
    """Brief exclusive access to shared resource"""
    task_id = context.get_state("task_id", 1)

    print(f"🎫 Task {task_id} acquired 10-second lease")

    # Quick operations on shared resource
    operations = ["read_config", "update_status", "write_log"]

    for operation in operations:
        print(f"   ⚙️ Performing {operation}...")
        await asyncio.sleep(1.0)

    print(f"✅ Task {task_id} completed within lease period")
    context.set_variable(f"lease_result_{task_id}", "completed")

# Long-term resource reservation
@state(
    cpu=2.0,
    memory=1024,
    lease=300.0,        # 5-minute lease for extended work
    timeout=350.0
)
async def long_term_resource_reservation(context):
    """Extended exclusive access for complex operations"""
    reservation_id = context.get_state("reservation_id", 1)

    print(f"🏗️ Reservation {reservation_id} acquired 5-minute lease")

    # Complex operations requiring extended exclusive access
    phases = [
        ("initialization", 30),
        ("data_migration", 120),
        ("validation", 60),
        ("cleanup", 30)
    ]

    for phase_name, duration in phases:
        print(f"   🔄 Phase: {phase_name} ({duration}s)")
        await asyncio.sleep(duration / 10)  # Compressed time

    print(f"✅ Reservation {reservation_id} completed all phases")
    context.set_variable(f"reservation_result_{reservation_id}", "completed")
\`\`\`

---

## Quota Management Strategies

### Multi-Tenant Resource Quotas

\`\`\`python
import asyncio
from puffinflow import Agent
from puffinflow.decorators import state

# Sophisticated quota management system
class AdvancedQuotaManager:
    def __init__(self):
        self.quotas = {
            "tenant_startup": {
                "cpu_hours": 10.0,
                "memory_gb_hours": 20.0,
                "gpu_hours": 0.0,
                "api_calls": 1000,
                "storage_gb": 5.0
            },
            "tenant_growth": {
                "cpu_hours": 100.0,
                "memory_gb_hours": 200.0,
                "gpu_hours": 5.0,
                "api_calls": 10000,
                "storage_gb": 50.0
            },
            "tenant_enterprise": {
                "cpu_hours": 1000.0,
                "memory_gb_hours": 2000.0,
                "gpu_hours": 100.0,
                "api_calls": 100000,
                "storage_gb": 500.0
            }
        }

        self.usage = {
            "tenant_startup": {
                "cpu_hours": 7.5,
                "memory_gb_hours": 15.2,
                "gpu_hours": 0.0,
                "api_calls": 750,
                "storage_gb": 3.2
            },
            "tenant_growth": {
                "cpu_hours": 45.8,
                "memory_gb_hours": 89.5,
                "gpu_hours": 2.1,
                "api_calls": 6750,
                "storage_gb": 28.9
            },
            "tenant_enterprise": {
                "cpu_hours": 234.7,
                "memory_gb_hours": 456.3,
                "gpu_hours": 23.8,
                "api_calls": 45600,
                "storage_gb": 189.4
            }
        }

    def check_quota(self, tenant, resource, amount):
        """Check if tenant has quota available for resource"""
        quota = self.quotas[tenant][resource]
        current = self.usage[tenant][resource]
        return current + amount <= quota

    def consume_quota(self, tenant, resource, amount):
        """Consume quota if available"""
        if self.check_quota(tenant, resource, amount):
            self.usage[tenant][resource] += amount
            return True
        return False

    def get_quota_status(self, tenant):
        """Get detailed quota status for tenant"""
        quota = self.quotas[tenant]
        usage = self.usage[tenant]

        status = {}
        for resource in quota:
            used = usage[resource]
            limit = quota[resource]
            percentage = (used / limit * 100) if limit > 0 else 0

            status[resource] = {
                "used": used,
                "limit": limit,
                "percentage": percentage,
                "available": limit - used,
                "status": self._get_quota_health(percentage)
            }

        return status

    def _get_quota_health(self, percentage):
        """Determine quota health status"""
        if percentage >= 95:
            return "critical"
        elif percentage >= 80:
            return "warning"
        elif percentage >= 60:
            return "moderate"
        else:
            return "healthy"

quota_manager = AdvancedQuotaManager()
agent = Agent("quota-management-agent")

@state(
    cpu=2.0,
    memory=1024,
    rate_limit=5.0,
    timeout=30.0
)
async def api_operation_with_quotas(context):
    """API operation with comprehensive quota checking"""
    tenant_id = context.get_variable("tenant_id")
    operation_id = context.get_state("api_operations", 0) + 1
    context.set_state("api_operations", operation_id)

    print(f"🌐 API operation #{operation_id} for {tenant_id}")

    # Check multiple quota types
    quotas_needed = {
        "api_calls": 1,
        "cpu_hours": 0.01,  # Rough estimate: 2 CPU * 30s max / 3600s
        "memory_gb_hours": 0.008  # 1GB * 30s max / 3600s
    }

    # Pre-flight quota check
    quota_available = True
    for resource, amount in quotas_needed.items():
        if not quota_manager.check_quota(tenant_id, resource, amount):
            print(f"❌ API operation #{operation_id} rejected: {resource} quota exceeded")
            context.set_variable(f"api_quota_exceeded_{operation_id}", resource)
            quota_available = False
            break

    if quota_available:
        # Consume quotas
        for resource, amount in quotas_needed.items():
            quota_manager.consume_quota(tenant_id, resource, amount)

        # Perform the operation
        await asyncio.sleep(0.5)  # Simulate API work

        print(f"✅ API operation #{operation_id} completed")
        context.set_variable(f"api_success_{operation_id}", True)

@state(
    cpu=4.0,
    memory=4096,
    gpu=1.0,
    timeout=600.0
)
async def compute_job_with_quotas(context):
    """Compute-intensive job with quota tracking"""
    tenant_id = context.get_variable("tenant_id")
    job_id = context.get_state("compute_jobs", 0) + 1
    context.set_state("compute_jobs", job_id)

    print(f"💻 Compute job #{job_id} for {tenant_id}")

    # Estimate resource consumption
    estimated_duration_hours = 0.25  # 15 minutes
    quotas_needed = {
        "cpu_hours": 4.0 * estimated_duration_hours,
        "memory_gb_hours": 4.0 * estimated_duration_hours,
        "gpu_hours": 1.0 * estimated_duration_hours
    }

    # Check all quotas before starting
    quota_available = True
    for resource, amount in quotas_needed.items():
        if not quota_manager.check_quota(tenant_id, resource, amount):
            print(f"❌ Compute job #{job_id} rejected: {resource} quota exceeded")
            context.set_variable(f"compute_quota_exceeded_{job_id}", resource)
            quota_available = False
            break

    if quota_available:
        # Reserve quotas
        for resource, amount in quotas_needed.items():
            quota_manager.consume_quota(tenant_id, resource, amount)

        print(f"   📊 Reserved: {quotas_needed['cpu_hours']:.2f} CPU hours, "
              f"{quotas_needed['gpu_hours']:.2f} GPU hours")

        # Perform compute job
        await asyncio.sleep(2.0)  # Simulate compute work

        print(f"✅ Compute job #{job_id} completed")
        context.set_variable(f"compute_success_{job_id}", True)

@state
async def quota_monitoring_report(context):
    """Generate comprehensive quota monitoring report"""
    tenant_id = context.get_variable("tenant_id")

    print(f"📊 Quota Monitoring Report for {tenant_id}")
    print("=" * 60)

    status = quota_manager.get_quota_status(tenant_id)

    for resource, info in status.items():
        health_emoji = {
            "healthy": "🟢",
            "moderate": "🟡",
            "warning": "🟠",
            "critical": "🔴"
        }.get(info["status"], "⚪")

        print(f"{health_emoji} {resource.upper()}:")
        print(f"   Used: {info['used']:.2f} / {info['limit']:.2f} ({info['percentage']:.1f}%)")
        print(f"   Available: {info['available']:.2f}")
        print(f"   Status: {info['status']}")
        print()

    # Generate recommendations
    recommendations = []
    for resource, info in status.items():
        if info["status"] == "critical":
            recommendations.append(f"🚨 URGENT: Upgrade {resource} quota immediately")
        elif info["status"] == "warning":
            recommendations.append(f"⚠️ Consider upgrading {resource} quota soon")
        elif info["status"] == "moderate":
            recommendations.append(f"💡 Monitor {resource} usage closely")

    if recommendations:
        print("📋 Recommendations:")
        for rec in recommendations:
            print(f"   {rec}")
    else:
        print("✅ All quotas are healthy")

    context.set_variable("quota_report", status)

# Test different tenant types
tenant_types = ["tenant_startup", "tenant_growth", "tenant_enterprise"]

for tenant_type in tenant_types:
    # Setup tenant context
    setup_task = f"setup_{tenant_type}"

    @state
    async def setup_tenant_context(context, tenant=tenant_type):
        context.set_variable("tenant_id", tenant)
        print(f"🏢 Setting up context for {tenant}")

    agent.add_state(setup_task, setup_tenant_context)

    # Add operations for each tenant
    api_tasks = []
    for i in range(2):
        task_name = f"api_operation_{tenant_type}_{i}"
        agent.add_state(task_name, api_operation_with_quotas, dependencies=[setup_task])
        api_tasks.append(task_name)

    compute_tasks = []
    for i in range(1):
        task_name = f"compute_job_{tenant_type}_{i}"
        agent.add_state(task_name, compute_job_with_quotas, dependencies=[setup_task])
        compute_tasks.append(task_name)

    # Add monitoring report
    report_task = f"quota_report_{tenant_type}"
    agent.add_state(report_task, quota_monitoring_report,
                    dependencies=api_tasks + compute_tasks)
\`\`\`

---

## Best Practices for Resource Configuration

### 1. **Start Conservative, Then Optimize**

\`\`\`python
# ✅ Good approach - Start with conservative estimates
@state(cpu=1.0, memory=512, timeout=30.0)  # Conservative baseline
async def new_operation(context):
    # Monitor and profile this operation
    pass

# Then optimize based on actual behavior
@state(cpu=2.0, memory=1024, timeout=60.0)  # Optimized based on profiling
async def optimized_operation(context):
    # Same operation with better resource allocation
    pass

# ❌ Avoid - Over-allocating without data
@state(cpu=16.0, memory=32768, timeout=3600.0)  # Probably wasteful
async def over_allocated_operation(context):
    pass
\`\`\`

### 2. **Match Resources to Workload Characteristics**

\`\`\`python
# ✅ Good - Different patterns for different workloads

# CPU-bound: High CPU, moderate memory
@state(cpu=6.0, memory=2048)
async def mathematical_modeling(context): pass

# Memory-bound: Moderate CPU, high memory
@state(cpu=2.0, memory=16384)
async def large_dataset_processing(context): pass

# I/O-bound: Low CPU, moderate memory, high I/O
@state(cpu=1.0, memory=1024, io=15.0)
async def file_processing(context): pass

# Network-bound: Low CPU, low memory, high network, rate limited
@state(cpu=0.5, memory=256, network=10.0, rate_limit=20.0)
async def api_aggregation(context): pass

# ❌ Avoid - Same configuration for different workloads
@state(cpu=4.0, memory=4096)  # One size fits all approach
\`\`\`

### 3. **Use Coordination Appropriately**

\`\`\`python
# ✅ Good - Choose the right coordination primitive

# Exclusive access for critical updates
@state(mutex=True, priority=Priority.CRITICAL)
async def database_schema_update(context): pass

# Limited concurrency for resource pools
@state(semaphore=5)  # Max 5 database connections
async def database_query(context): pass

# Synchronized start for batch jobs
@state(barrier=3)  # Wait for 3 workers
async def distributed_batch_job(context): pass

# Temporary exclusive access
@state(lease=30.0)  # 30-second exclusive lease
async def configuration_update(context): pass

# ❌ Avoid - Wrong coordination for the use case
@state(mutex=True)  # Unnecessarily exclusive for read operations
async def read_only_query(context): pass
\`\`\`

### 4. **Implement Sensible Rate Limiting**

\`\`\`python
# ✅ Good - Rate limits based on service characteristics

# High-throughput service with burst capacity
@state(rate_limit=50.0, burst_limit=100)
async def internal_microservice_call(context): pass

# Third-party API with strict limits
@state(rate_limit=10.0, burst_limit=15)
async def external_api_call(context): pass

# Expensive ML service
@state(rate_limit=1.0)  # 1 request per second
async def ml_inference_service(context): pass

# Critical system service
@state(rate_limit=0.1)  # 1 request per 10 seconds
async def system_admin_operation(context): pass

# ❌ Avoid - Rate limits that don't match reality
@state(rate_limit=1000.0)  # Unrealistically high
async def realistic_api_call(context): pass

@state(rate_limit=0.01)  # Unrealistically low for normal operations
async def normal_operation(context): pass
\`\`\`

### 5. **Monitor and Adjust Based on Data**

\`\`\`python
@state
async def resource_monitoring_and_optimization(context):
    """
    Implement monitoring to continuously optimize resource allocation
    """
    print("📊 Resource utilization analysis...")

    # Collect performance metrics
    operations_data = {
        "data_processing": {
            "avg_cpu_usage": 85,  # 85% of allocated CPU
            "avg_memory_usage": 60,  # 60% of allocated memory
            "avg_duration": 45,  # 45 seconds average
            "success_rate": 98  # 98% success rate
        },
        "api_calls": {
            "avg_cpu_usage": 15,  # Only 15% CPU usage
            "avg_memory_usage": 25,  # 25% memory usage
            "avg_duration": 2,  # 2 seconds average
            "success_rate": 99.5  # 99.5% success rate
        }
    }

    recommendations = []

    for operation, metrics in operations_data.items():
        print(f"\n🔍 Analyzing {operation}:")
        print(f"   CPU: {metrics['avg_cpu_usage']}% utilization")
        print(f"   Memory: {metrics['avg_memory_usage']}% utilization")
        print(f"   Duration: {metrics['avg_duration']}s average")
        print(f"   Success: {metrics['success_rate']}%")

        # Generate optimization recommendations
        if metrics['avg_cpu_usage'] > 90:
            recommendations.append(f"🔴 {operation}: Increase CPU allocation")
        elif metrics['avg_cpu_usage'] < 30:
            recommendations.append(f"🟡 {operation}: Consider reducing CPU allocation")

        if metrics['avg_memory_usage'] > 90:
            recommendations.append(f"🔴 {operation}: Increase memory allocation")
        elif metrics['avg_memory_usage'] < 30:
            recommendations.append(f"🟡 {operation}: Consider reducing memory allocation")

        if metrics['success_rate'] < 95:
            recommendations.append(f"🔴 {operation}: Investigate failure causes")

    print(f"\n📋 Optimization Recommendations:")
    for rec in recommendations:
        print(f"   {rec}")

    if not recommendations:
        print("   ✅ All operations are well-optimized")

    context.set_variable("optimization_recommendations", recommendations)
\`\`\`

---

## When to Use Different Features

### Resource Allocation Decision Tree

\`\`\`python
"""
Resource Allocation Decision Framework:

1. CPU Allocation:
   - cpu=0.5-1.0: Light coordination, simple logic
   - cpu=2.0-4.0: Data processing, API orchestration
   - cpu=4.0-8.0: Scientific computing, complex analysis
   - cpu=8.0+: Parallel processing, mathematical modeling

2. Memory Allocation:
   - memory=256-512: Simple operations, coordination tasks
   - memory=1024-4096: Data processing, caching
   - memory=4096-16384: Large datasets, ML training
   - memory=16384+: Big data, in-memory databases

3. GPU Allocation:
   - gpu=0.0: No GPU needed
   - gpu=0.5-1.0: Light ML inference, image processing
   - gpu=1.0-4.0: ML training, scientific computing
   - gpu=4.0+: Distributed ML, high-performance computing

4. I/O Priority:
   - io=1.0: Minimal file operations
   - io=5.0-10.0: Regular file processing
   - io=10.0+: Bulk file operations, data pipelines

5. Network Priority:
   - network=1.0: Occasional API calls
   - network=5.0-10.0: Regular external service calls
   - network=10.0+: High-volume data transfer
"""

# Example decision process
def recommend_resources(operation_type, data_size, external_calls, duration):
    """Recommend resource configuration based on operation characteristics"""

    recommendations = {
        "cpu": 1.0,
        "memory": 512,
        "gpu": 0.0,
        "io": 1.0,
        "network": 1.0,
        "timeout": 30.0,
        "rate_limit": None,
        "coordination": None
    }

    # Adjust based on operation type
    if operation_type == "data_processing":
        recommendations.update({
            "cpu": 4.0,
            "memory": 2048,
            "timeout": 300.0
        })
    elif operation_type == "ml_training":
        recommendations.update({
            "cpu": 4.0,
            "memory": 8192,
            "gpu": 2.0,
            "timeout": 3600.0
        })
    elif operation_type == "file_processing":
        recommendations.update({
            "cpu": 1.0,
            "memory": 1024,
            "io": 10.0,
            "timeout": 600.0
        })
    elif operation_type == "api_orchestration":
        recommendations.update({
            "cpu": 1.0,
            "memory": 512,
            "network": 8.0,
            "timeout": 60.0,
            "rate_limit": 10.0
        })

    # Adjust based on data size
    if data_size > 10000000:  # 10M+ records
        recommendations["memory"] *= 4
        recommendations["timeout"] *= 2
    elif data_size > 1000000:  # 1M+ records
        recommendations["memory"] *= 2

    # Adjust based on external calls
    if external_calls > 100:
        recommendations["rate_limit"] = 20.0
        recommendations["network"] = 10.0
    elif external_calls > 10:
        recommendations["rate_limit"] = 5.0
        recommendations["network"] = 5.0

    # Adjust based on duration
    if duration > 3600:  # > 1 hour
        recommendations["timeout"] = duration * 1.5
        recommendations["coordination"] = "lease"
    elif duration > 300:  # > 5 minutes
        recommendations["timeout"] = duration * 1.2

    return recommendations

# Usage example
def generate_decorator_config(operation_type, data_size=0, external_calls=0, duration=30):
    """Generate @state decorator configuration"""
    config = recommend_resources(operation_type, data_size, external_calls, duration)

    decorator_parts = []

    # Required resources
    if config["cpu"] != 1.0:
        decorator_parts.append(f"cpu={config['cpu']}")
    if config["memory"] != 512:
        decorator_parts.append(f"memory={config['memory']}")
    if config["gpu"] > 0:
        decorator_parts.append(f"gpu={config['gpu']}")
    if config["io"] != 1.0:
        decorator_parts.append(f"io={config['io']}")
    if config["network"] != 1.0:
        decorator_parts.append(f"network={config['network']}")

    # Performance settings
    if config["timeout"] != 30.0:
        decorator_parts.append(f"timeout={config['timeout']}")
    if config["rate_limit"]:
        decorator_parts.append(f"rate_limit={config['rate_limit']}")

    # Coordination
    if config["coordination"] == "lease":
        decorator_parts.append("lease=300.0")

    decorator_config = ", ".join(decorator_parts)

    print(f"Recommended configuration for {operation_type}:")
    print(f"@state({decorator_config})")

    return config

# Examples
print("🔍 Resource Configuration Recommendations:\n")

generate_decorator_config("data_processing", data_size=5000000, duration=180)
print()

generate_decorator_config("ml_training", data_size=1000000, duration=1800)
print()

generate_decorator_config("file_processing", data_size=100000, duration=300)
print()

generate_decorator_config("api_orchestration", external_calls=50, duration=45)
\`\`\`

---

## Quick Reference

### **Basic Resource Allocation**
\`\`\`python
@state(cpu=4.0, memory=2048)            # 4 cores, 2GB RAM
@state(gpu=1.0, memory=8192)            # 1 GPU, 8GB RAM
@state(io=10.0, network=5.0)            # High I/O, moderate network
\`\`\`

### **Rate Limiting**
\`\`\`python
@state(rate_limit=10.0)                 # 10 requests per second
@state(rate_limit=5.0, burst_limit=15)  # 5/sec with burst to 15
\`\`\`

### **Coordination Primitives**
\`\`\`python
@state(mutex=True)                      # Exclusive access
@state(semaphore=5)                     # Max 5 concurrent
@state(barrier=3)                       # Wait for 3 participants
@state(lease=30.0)                      # 30-second lease
\`\`\`

### **Combined Configuration**
\`\`\`python
@state(
    cpu=2.0,                            # Resource requirements
    memory=1024,
    rate_limit=2.0,                     # Rate limiting
    semaphore=3,                        # Coordination
    timeout=30.0,                       # Error handling
    priority=Priority.HIGH              # Execution priority
)
async def comprehensive_operation(context):
    pass
\`\`\`

### **Predefined Resource Profiles**

Puffinflow provides some predefined decorators for common patterns, but you should always validate these match your actual requirements:

\`\`\`python
# These are convenience decorators with predefined resource allocations:
# @cpu_intensive     # Roughly equivalent to @state(cpu=4.0, memory=1024)
# @memory_intensive  # Roughly equivalent to @state(cpu=2.0, memory=4096)
# @gpu_accelerated   # Roughly equivalent to @state(cpu=2.0, memory=2048, gpu=1.0)

# ⚠️ Important: Always profile your workloads and adjust these as needed!
# The predefined decorators are starting points, not optimized configurations.

# ✅ Better approach - Start with manual configuration based on your analysis:
@state(cpu=3.0, memory=1536, timeout=180.0)  # Tuned for your specific workload
async def your_cpu_intensive_task(context):
    pass
\`\`\`

Resource management is about understanding your workload characteristics and configuring appropriate constraints to ensure optimal performance, reliability, and fair resource sharing across your workflows!
`.trim();
