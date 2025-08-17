export const coordinationMarkdown = `# Coordination

Coordination is like conducting an orchestra - you need to make sure all the different parts work together at the right time. Puffinflow comes with powerful built-in coordination primitives that make it easy to synchronize multiple workflows, share resources safely, and orchestrate complex operations.

## Why Coordination Matters

**Without coordination:**
- Multiple workflows compete for the same resources and crash
- Operations run in the wrong order and produce incorrect results
- No way to synchronize parallel operations
- Resource exhaustion from too many concurrent operations
- Race conditions and data corruption

**With Puffinflow's built-in coordination:**
- Resources are shared safely between workflows
- Operations happen in the correct order automatically
- Parallel tasks can synchronize at specific points
- Resource usage is controlled and optimized
- Race conditions are eliminated

## Part 1: Built-in Semaphores (Controlling Access)

Semaphores are like tickets for a concert - only a limited number of people can get in at once. Puffinflow's built-in semaphores make resource limiting automatic and safe:

\`\`\`python
import asyncio
import time
import random
from puffinflow import Agent, state
from puffinflow.core.coordination.primitives import Semaphore

agent = Agent("semaphore-demo")

# Create built-in semaphores for different resources
database_pool = Semaphore("database_connections", max_count=3)
api_rate_limiter = Semaphore("openai_api_calls", max_count=5)
gpu_resources = Semaphore("gpu_compute", max_count=2)

@state(timeout=30.0)
async def database_operation(context):
    """Database operation using built-in semaphore"""

    task_id = context.get_variable("task_id", "task_001")

    print(f"🗄️ Task {task_id}: Requesting database connection...")

    # Use built-in semaphore to limit database connections
    async with database_pool:
        print(f"✅ Task {task_id}: Got database connection!")
        print(f"   Available connections: {database_pool.available_permits}")

        # Simulate database work
        await asyncio.sleep(2.0)

        # Simulate database query
        user_data = {
            "user_id": task_id,
            "name": f"User {task_id}",
            "created_at": time.time()
        }

        context.set_variable("user_data", user_data)
        print(f"   Task {task_id}: Database operation completed")

    # Semaphore automatically released here
    print(f"🔓 Task {task_id}: Released database connection")
    return "process_user_data"

@state(timeout=15.0)
async def process_user_data(context):
    """Process user data with API rate limiting"""

    task_id = context.get_variable("task_id")
    user_data = context.get_variable("user_data")

    print(f"🤖 Task {task_id}: Making AI API call...")

    # Use built-in semaphore for API rate limiting
    async with api_rate_limiter:
        print(f"✅ Task {task_id}: Got API rate limit slot!")
        print(f"   Available API slots: {api_rate_limiter.available_permits}")

        # Simulate AI API call
        await asyncio.sleep(1.5)

        # Process the user data
        processed_data = {
            "original": user_data,
            "ai_analysis": f"Analysis for {user_data['name']}",
            "confidence": random.uniform(0.8, 0.95),
            "processed_at": time.time()
        }

        context.set_variable("processed_data", processed_data)
        print(f"   Task {task_id}: AI processing completed")

    print(f"🔓 Task {task_id}: Released API rate limit slot")
    context.set_output("result", processed_data)
    return None

# Add states to agent
agent.add_state("database_operation", database_operation)
agent.add_state("process_user_data", process_user_data)

# Demo built-in semaphores with multiple concurrent tasks
async def demo_builtin_semaphores():
    print("🎫 Built-in Semaphores Demo\\n")

    # Create many tasks to show semaphore limiting
    tasks = []
    for i in range(8):  # 8 tasks, but only 3 can use DB and 5 can use API
        task = asyncio.create_task(
            agent.run(
                initial_state="database_operation",
                initial_context={"task_id": f"task_{i+1:02d}"}
            )
        )
        tasks.append(task)

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print(f"\\n📊 Results: {len([r for r in results if not isinstance(r, Exception)])} successful")
    print("Notice how tasks waited for semaphore permits!")

if __name__ == "__main__":
    asyncio.run(demo_builtin_semaphores())

# Add API rate limited task state
agent.add_state("api_rate_limited_task", api_rate_limited_task)

@state(timeout=25.0, max_retries=3)
async def api_rate_limited_task(context):
    """Task with API rate limiting using semaphore"""
    task_id = context.get_variable("task_id", "api_task_001")

    print(f"🌐 Task {task_id}: Requesting API quota...")

    # Multiple API calls with rate limiting
    api_calls = ["chat_completion", "embedding", "moderation"]
    results = []

    for call_type in api_calls:
        try:
            # Acquire API quota
            quota_acquired = await api_rate_limiter.acquire(
                requester_id=f"{task_id}_{call_type}",
                timeout=15.0
            )

            if not quota_acquired:
                print(f"⚠️ Task {task_id}: API quota timeout for {call_type}")
                continue

            print(f"✅ Task {task_id}: API quota acquired for {call_type}")

            # Simulate API call
            call_start = time.time()
            await asyncio.sleep(1.0)  # Simulate API latency
            call_duration = time.time() - call_start

            results.append({
                "call_type": call_type,
                "duration": call_duration,
                "status": "success"
            })

            print(f"📞 Task {task_id}: {call_type} API call completed ({call_duration:.2f}s)")

        finally:
            # Release API quota
            await api_rate_limiter.release(f"{task_id}_{call_type}")

    context.set_variable(f"api_results_{task_id}", {
        "calls_completed": len(results),
        "total_calls_attempted": len(api_calls),
        "results": results
    })

@state(timeout=60.0)
async def gpu_computation_task(context):
    """Task requiring exclusive GPU resources"""
    task_id = context.get_variable("task_id", "gpu_task_001")
    model_type = context.get_variable("model_type", "transformer")

    print(f"🔥 Task {task_id}: Requesting GPU for {model_type} processing...")

    try:
        # Acquire GPU resource
        gpu_acquired = await gpu_resources.acquire(
            requester_id=task_id,
            timeout=30.0
        )

        if not gpu_acquired:
            print(f"❌ Task {task_id}: GPU resource timeout")
            return

        print(f"✅ Task {task_id}: GPU resource acquired")

        # Simulate GPU-intensive computation
        computation_phases = [
            {"phase": "model_loading", "duration": 3.0},
            {"phase": "data_preprocessing", "duration": 1.5},
            {"phase": "inference", "duration": 8.0},
            {"phase": "postprocessing", "duration": 1.0}
        ]

        total_compute_time = 0
        for phase in computation_phases:
            print(f"   🧠 Task {task_id}: {phase['phase']}...")
            await asyncio.sleep(phase['duration'])
            total_compute_time += phase['duration']

        context.set_variable(f"gpu_result_{task_id}", {
            "model_type": model_type,
            "computation_time": total_compute_time,
            "phases_completed": len(computation_phases),
            "status": "success"
        })

        print(f"✅ Task {task_id}: GPU computation completed ({total_compute_time:.1f}s)")

    finally:
        # Always release GPU resource
        await gpu_resources.release(task_id)
        print(f"🔓 Task {task_id}: GPU resource released")

# Create multiple tasks to demonstrate semaphore coordination
for i in range(8):  # More tasks than semaphore allows
    coordination_agent.add_state(f"db_task_{i}",
        lambda ctx, task_num=i: database_intensive_task({**ctx.shared_state, "task_id": f"db_{task_num}"}))

    coordination_agent.add_state(f"api_task_{i}",
        lambda ctx, task_num=i: api_rate_limited_task({**ctx.shared_state, "task_id": f"api_{task_num}"}))

# Add GPU tasks (more than available GPUs)
for i in range(4):
    coordination_agent.add_state(f"gpu_task_{i}",
        lambda ctx, task_num=i: gpu_computation_task({
            **ctx.shared_state,
            "task_id": f"gpu_{task_num}",
            "model_type": ["transformer", "cnn", "rnn", "bert"][task_num]
        }))
\`\`\`

---

## Mutexes: Exclusive Access Protection

### Critical Section Management

Use mutexes to protect shared resources that require exclusive access:

\`\`\`python
from puffinflow.core.coordination.primitives import Mutex

# Shared resource protection
shared_data_mutex = Mutex("shared_data_access")
config_update_mutex = Mutex("configuration_updates")
audit_log_mutex = Mutex("audit_logging")

mutex_agent = Agent("mutex-coordination")

class SharedDataStore:
    def __init__(self):
        self.data = {}
        self.version = 0
        self.last_updated = time.time()

    async def update_data(self, key: str, value: Any, updater_id: str):
        """Thread-safe data update"""
        async with shared_data_mutex.acquire_context(updater_id):
            print(f"   🔒 {updater_id}: Acquired exclusive access to shared data")

            old_value = self.data.get(key)
            self.data[key] = value
            self.version += 1
            self.last_updated = time.time()

            # Simulate processing time
            await asyncio.sleep(0.5)

            print(f"   📝 {updater_id}: Updated {key}: {old_value} -> {value} (v{self.version})")

            return self.version

    async def read_data(self, reader_id: str) -> dict:
        """Thread-safe data reading"""
        async with shared_data_mutex.acquire_context(reader_id):
            print(f"   👀 {reader_id}: Reading shared data (v{self.version})")

            # Simulate read processing
            await asyncio.sleep(0.1)

            return {
                "data": self.data.copy(),
                "version": self.version,
                "last_updated": self.last_updated
            }

# Global shared store
shared_store = SharedDataStore()

@state(timeout=20.0)
async def data_writer_task(context):
    """Task that writes to shared data store"""
    writer_id = context.get_variable("writer_id", "writer_001")

    print(f"✍️ Writer {writer_id}: Starting data updates...")

    # Multiple updates to shared store
    updates = [
        {"key": f"metric_{writer_id}", "value": 42},
        {"key": f"status_{writer_id}", "value": "active"},
        {"key": f"timestamp_{writer_id}", "value": time.time()}
    ]

    completed_updates = []

    for update in updates:
        try:
            print(f"📝 Writer {writer_id}: Updating {update['key']}...")

            version = await shared_store.update_data(
                update['key'],
                update['value'],
                writer_id
            )

            completed_updates.append({
                **update,
                "version": version,
                "updated_at": time.time()
            })

            # Wait between updates
            await asyncio.sleep(0.2)

        except Exception as e:
            print(f"❌ Writer {writer_id}: Update failed for {update['key']}: {e}")

    context.set_variable(f"writer_results_{writer_id}", {
        "updates_completed": len(completed_updates),
        "total_updates_attempted": len(updates),
        "updates": completed_updates
    })

    print(f"✅ Writer {writer_id}: Completed {len(completed_updates)} updates")

@state(timeout=15.0)
async def data_reader_task(context):
    """Task that reads from shared data store"""
    reader_id = context.get_variable("reader_id", "reader_001")

    print(f"👀 Reader {reader_id}: Starting data reads...")

    # Multiple reads with analysis
    read_results = []

    for i in range(3):
        try:
            print(f"📖 Reader {reader_id}: Reading data (iteration {i+1})...")

            data_snapshot = await shared_store.read_data(reader_id)

            # Analyze data
            analysis = {
                "read_iteration": i + 1,
                "data_version": data_snapshot["version"],
                "keys_count": len(data_snapshot["data"]),
                "data_age_seconds": time.time() - data_snapshot["last_updated"],
                "timestamp": time.time()
            }

            read_results.append(analysis)

            print(f"📊 Reader {reader_id}: Data v{analysis['data_version']}, {analysis['keys_count']} keys")

            # Wait between reads
            await asyncio.sleep(1.0)

        except Exception as e:
            print(f"❌ Reader {reader_id}: Read failed: {e}")

    context.set_variable(f"reader_results_{reader_id}", {
        "reads_completed": len(read_results),
        "read_results": read_results
    })

    print(f"✅ Reader {reader_id}: Completed {len(read_results)} reads")

@state(timeout=25.0)
async def configuration_manager_task(context):
    """Task that manages system configuration with exclusive access"""
    manager_id = context.get_variable("manager_id", "config_mgr_001")

    print(f"⚙️ Config Manager {manager_id}: Managing system configuration...")

    try:
        # Acquire exclusive configuration access
        async with config_update_mutex.acquire_context(manager_id):
            print(f"🔒 Config Manager {manager_id}: Acquired exclusive configuration access")

            # Simulate complex configuration operations
            config_operations = [
                {"operation": "backup_current_config", "duration": 1.0},
                {"operation": "validate_new_config", "duration": 2.0},
                {"operation": "apply_configuration", "duration": 1.5},
                {"operation": "restart_services", "duration": 3.0},
                {"operation": "verify_deployment", "duration": 1.0}
            ]

            operation_results = []

            for operation in config_operations:
                op_start = time.time()

                print(f"   ⚙️ Config Manager {manager_id}: {operation['operation']}...")
                await asyncio.sleep(operation['duration'])

                op_duration = time.time() - op_start
                operation_results.append({
                    "operation": operation['operation'],
                    "duration": op_duration,
                    "status": "completed"
                })

            context.set_variable(f"config_results_{manager_id}", {
                "operations_completed": len(operation_results),
                "total_duration": sum(op['duration'] for op in operation_results),
                "operations": operation_results
            })

            print(f"✅ Config Manager {manager_id}: Configuration update completed")

    except Exception as e:
        print(f"❌ Config Manager {manager_id}: Configuration update failed: {e}")

# Create multiple concurrent tasks to demonstrate mutex coordination
for i in range(3):
    mutex_agent.add_state(f"writer_{i}",
        lambda ctx, writer_num=i: data_writer_task({**ctx.shared_state, "writer_id": f"writer_{writer_num}"}))

    mutex_agent.add_state(f"reader_{i}",
        lambda ctx, reader_num=i: data_reader_task({**ctx.shared_state, "reader_id": f"reader_{reader_num}"}))

# Add configuration manager
mutex_agent.add_state("config_manager",
    lambda ctx: configuration_manager_task({**ctx.shared_state, "manager_id": "config_mgr_001"}))
\`\`\`

---

## Barriers: Synchronized Parallel Execution

### Coordinated Batch Processing

Use barriers to synchronize parallel tasks at specific points:

\`\`\`python
from puffinflow.core.coordination.primitives import Barrier

# Barrier for synchronized starts and checkpoints
batch_start_barrier = Barrier("batch_processing_start", parties=4)
checkpoint_barrier = Barrier("processing_checkpoint", parties=4)
completion_barrier = Barrier("batch_completion", parties=4)

barrier_agent = Agent("barrier-coordination")

@state(timeout=45.0)
async def batch_worker_task(context):
    """Worker task that participates in barrier synchronization"""
    worker_id = context.get_variable("worker_id", "worker_001")
    batch_id = context.get_variable("batch_id", "batch_001")

    print(f"👷 Worker {worker_id}: Initializing for batch {batch_id}...")

    # Initialization phase
    init_time = time.time()
    await asyncio.sleep(1.0 + (hash(worker_id) % 10) * 0.1)  # Variable init time
    init_duration = time.time() - init_time

    print(f"🔄 Worker {worker_id}: Waiting for all workers to initialize...")

    # Wait for all workers to complete initialization
    await batch_start_barrier.wait(worker_id, timeout=30.0)

    print(f"🚀 Worker {worker_id}: All workers initialized, starting batch processing...")

    # Phase 1: Data processing
    phase1_start = time.time()
    phase1_data = []

    for i in range(100):
        # Simulate data processing
        processed_item = f"{worker_id}_item_{i}"
        phase1_data.append(processed_item)

        if i % 25 == 0:
            print(f"   📊 Worker {worker_id}: Processed {i+1}/100 items")

        await asyncio.sleep(0.01)  # Simulate processing time

    phase1_duration = time.time() - phase1_start

    print(f"✅ Worker {worker_id}: Phase 1 complete, waiting at checkpoint...")

    # Checkpoint: Wait for all workers to complete phase 1
    await checkpoint_barrier.wait(worker_id, timeout=60.0)

    print(f"🔄 Worker {worker_id}: All workers reached checkpoint, starting phase 2...")

    # Phase 2: Aggregation and validation
    phase2_start = time.time()

    # Simulate cross-worker data validation
    validation_results = []
    for i in range(10):
        # Simulate validation operations
        validation_result = {
            "check_id": f"{worker_id}_validation_{i}",
            "status": "passed" if i % 7 != 0 else "failed",
            "timestamp": time.time()
        }
        validation_results.append(validation_result)
        await asyncio.sleep(0.2)

    phase2_duration = time.time() - phase2_start

    # Store worker results
    worker_results = {
        "worker_id": worker_id,
        "batch_id": batch_id,
        "initialization_duration": init_duration,
        "phase1_duration": phase1_duration,
        "phase1_items_processed": len(phase1_data),
        "phase2_duration": phase2_duration,
        "validation_results": validation_results,
        "total_duration": init_duration + phase1_duration + phase2_duration
    }

    context.set_variable(f"worker_results_{worker_id}", worker_results)

    print(f"✅ Worker {worker_id}: Phase 2 complete, waiting for batch completion...")

    # Final barrier: Wait for all workers to complete all phases
    await completion_barrier.wait(worker_id, timeout=90.0)

    print(f"🎉 Worker {worker_id}: Batch {batch_id} completed successfully!")

@state(timeout=60.0)
async def batch_coordinator_task(context):
    """Coordinator that orchestrates the barrier-synchronized batch"""
    coordinator_id = "batch_coordinator"
    batch_id = context.get_variable("batch_id", "batch_001")

    print(f"🎭 Coordinator: Managing batch {batch_id}...")

    # Monitor barrier states
    batch_start_time = time.time()

    # Wait for all phases to complete by monitoring the completion barrier
    try:
        print("⏳ Coordinator: Waiting for batch completion...")

        # In a real implementation, this would monitor the barrier state
        # Here we simulate waiting for the completion barrier
        await asyncio.sleep(35.0)  # Estimated batch completion time

        batch_total_time = time.time() - batch_start_time

        # Collect results from all workers
        worker_results = []
        for i in range(4):
            worker_result = context.get_variable(f"worker_results_worker_{i}")
            if worker_result:
                worker_results.append(worker_result)

        # Generate batch summary
        batch_summary = {
            "batch_id": batch_id,
            "coordinator_id": coordinator_id,
            "total_workers": len(worker_results),
            "batch_duration": batch_total_time,
            "avg_worker_duration": sum(w["total_duration"] for w in worker_results) / len(worker_results) if worker_results else 0,
            "total_items_processed": sum(w["phase1_items_processed"] for w in worker_results),
            "successful_workers": len([w for w in worker_results if w["total_duration"] > 0])
        }

        context.set_output(f"batch_summary_{batch_id}", batch_summary)

        print(f"📊 Coordinator: Batch {batch_id} Summary:")
        print(f"   Workers: {batch_summary['total_workers']}")
        print(f"   Duration: {batch_summary['batch_duration']:.2f}s")
        print(f"   Items processed: {batch_summary['total_items_processed']}")

    except Exception as e:
        print(f"❌ Coordinator: Batch coordination failed: {e}")

# Create worker tasks that will synchronize via barriers
for i in range(4):
    barrier_agent.add_state(f"worker_{i}",
        lambda ctx, worker_num=i: batch_worker_task({
            **ctx.shared_state,
            "worker_id": f"worker_{worker_num}",
            "batch_id": "batch_001"
        }))

# Add coordinator
barrier_agent.add_state("batch_coordinator",
    lambda ctx: batch_coordinator_task({**ctx.shared_state, "batch_id": "batch_001"}))
\`\`\`

---

## Event-Driven Coordination

### Asynchronous Notification System

Use events for loose coupling and asynchronous coordination:

\`\`\`python
from puffinflow.core.coordination.primitives import Event
from typing import Callable, Dict
import asyncio

class EventBus:
    def __init__(self):
        self.events = {}
        self.subscribers = {}

    def create_event(self, event_name: str) -> Event:
        """Create a new event"""
        if event_name not in self.events:
            self.events[event_name] = Event(event_name)
            self.subscribers[event_name] = []
        return self.events[event_name]

    def subscribe(self, event_name: str, handler: Callable, subscriber_id: str):
        """Subscribe to an event"""
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []

        self.subscribers[event_name].append({
            "handler": handler,
            "subscriber_id": subscriber_id
        })

    async def publish(self, event_name: str, data: Dict = None):
        """Publish an event to all subscribers"""
        if event_name in self.events:
            event = self.events[event_name]
            await event.set(data or {})

            # Notify all subscribers
            for subscriber in self.subscribers.get(event_name, []):
                try:
                    await subscriber["handler"](data or {})
                except Exception as e:
                    print(f"❌ Event handler error for {subscriber['subscriber_id']}: {e}")

# Global event bus
event_bus = EventBus()

# Create events
data_ready_event = event_bus.create_event("data_ready")
processing_complete_event = event_bus.create_event("processing_complete")
error_occurred_event = event_bus.create_event("error_occurred")
system_shutdown_event = event_bus.create_event("system_shutdown")

event_agent = Agent("event-coordination")

@state(timeout=30.0)
async def data_producer_task(context):
    """Producer that generates data and signals when ready"""
    producer_id = context.get_variable("producer_id", "producer_001")

    print(f"🏭 Producer {producer_id}: Starting data production...")

    try:
        # Simulate data generation
        datasets = []
        for i in range(5):
            print(f"   📊 Producer {producer_id}: Generating dataset {i+1}/5...")

            # Simulate data creation time
            await asyncio.sleep(2.0)

            dataset = {
                "dataset_id": f"{producer_id}_dataset_{i}",
                "size_mb": 100 + (i * 50),
                "format": "parquet",
                "created_at": time.time(),
                "producer_id": producer_id
            }

            datasets.append(dataset)

            # Publish data ready event for each dataset
            await event_bus.publish("data_ready", {
                "dataset": dataset,
                "producer_id": producer_id,
                "total_datasets": 5,
                "completed_datasets": i + 1
            })

            print(f"📤 Producer {producer_id}: Published data_ready event for dataset {i+1}")

        # Store results
        context.set_variable(f"producer_results_{producer_id}", {
            "datasets_created": len(datasets),
            "total_size_mb": sum(d["size_mb"] for d in datasets),
            "datasets": datasets
        })

        print(f"✅ Producer {producer_id}: All data production complete")

    except Exception as e:
        # Publish error event
        await event_bus.publish("error_occurred", {
            "error_type": "data_production_error",
            "producer_id": producer_id,
            "error_message": str(e)
        })
        raise

@state(timeout=60.0)
async def data_consumer_task(context):
    """Consumer that waits for data ready events"""
    consumer_id = context.get_variable("consumer_id", "consumer_001")

    print(f"🍽️ Consumer {consumer_id}: Waiting for data events...")

    processed_datasets = []

    # Subscribe to data ready events
    async def handle_data_ready(event_data):
        dataset = event_data.get("dataset", {})
        producer_id = event_data.get("producer_id", "unknown")

        print(f"📥 Consumer {consumer_id}: Received data from {producer_id}")

        # Process the dataset
        processing_start = time.time()

        # Simulate data processing
        await asyncio.sleep(1.5)

        processing_duration = time.time() - processing_start

        processed_dataset = {
            **dataset,
            "processed_by": consumer_id,
            "processing_duration": processing_duration,
            "processed_at": time.time()
        }

        processed_datasets.append(processed_dataset)

        print(f"✅ Consumer {consumer_id}: Processed dataset {dataset['dataset_id']} ({processing_duration:.2f}s)")

    # Subscribe to events
    event_bus.subscribe("data_ready", handle_data_ready, consumer_id)

    # Wait for data events
    datasets_to_process = 5  # Expecting 5 datasets
    timeout_start = time.time()

    while len(processed_datasets) < datasets_to_process:
        if time.time() - timeout_start > 50.0:  # 50 second timeout
            print(f"⏰ Consumer {consumer_id}: Timeout waiting for data")
            break

        await asyncio.sleep(0.5)  # Check periodically

    # Store results
    context.set_variable(f"consumer_results_{consumer_id}", {
        "datasets_processed": len(processed_datasets),
        "total_processing_time": sum(d["processing_duration"] for d in processed_datasets),
        "processed_datasets": processed_datasets
    })

    # Publish processing complete event
    await event_bus.publish("processing_complete", {
        "consumer_id": consumer_id,
        "datasets_processed": len(processed_datasets),
        "completion_time": time.time()
    })

    print(f"🎉 Consumer {consumer_id}: Processing complete, published completion event")

@state(timeout=25.0)
async def monitoring_task(context):
    """Monitor system events and generate reports"""
    monitor_id = "system_monitor"

    print(f"👁️ Monitor: Starting event monitoring...")

    event_log = []

    # Subscribe to all events
    async def log_data_ready(event_data):
        event_log.append({
            "event_type": "data_ready",
            "timestamp": time.time(),
            "producer_id": event_data.get("producer_id"),
            "dataset_id": event_data.get("dataset", {}).get("dataset_id")
        })
        print(f"📝 Monitor: Logged data_ready event")

    async def log_processing_complete(event_data):
        event_log.append({
            "event_type": "processing_complete",
            "timestamp": time.time(),
            "consumer_id": event_data.get("consumer_id"),
            "datasets_processed": event_data.get("datasets_processed")
        })
        print(f"📝 Monitor: Logged processing_complete event")

    async def log_error(event_data):
        event_log.append({
            "event_type": "error_occurred",
            "timestamp": time.time(),
            "error_type": event_data.get("error_type"),
            "component": event_data.get("producer_id") or event_data.get("consumer_id")
        })
        print(f"🚨 Monitor: Logged error event")

    # Subscribe to events
    event_bus.subscribe("data_ready", log_data_ready, monitor_id)
    event_bus.subscribe("processing_complete", log_processing_complete, monitor_id)
    event_bus.subscribe("error_occurred", log_error, monitor_id)

    # Monitor for events
    monitoring_start = time.time()

    while time.time() - monitoring_start < 20.0:  # Monitor for 20 seconds
        await asyncio.sleep(1.0)

    # Generate monitoring report
    report = {
        "monitor_id": monitor_id,
        "monitoring_duration": time.time() - monitoring_start,
        "events_logged": len(event_log),
        "event_types": {},
        "event_timeline": event_log
    }

    # Count events by type
    for event in event_log:
        event_type = event["event_type"]
        report["event_types"][event_type] = report["event_types"].get(event_type, 0) + 1

    context.set_output("monitoring_report", report)

    print(f"📊 Monitor: Monitoring complete")
    print(f"   Events logged: {len(event_log)}")
    print(f"   Event types: {report['event_types']}")

# Create producer and consumer tasks
event_agent.add_state("data_producer",
    lambda ctx: data_producer_task({**ctx.shared_state, "producer_id": "producer_001"}))

for i in range(2):  # 2 consumers
    event_agent.add_state(f"data_consumer_{i}",
        lambda ctx, consumer_num=i: data_consumer_task({
            **ctx.shared_state,
            "consumer_id": f"consumer_{consumer_num}"
        }))

event_agent.add_state("system_monitor", monitoring_task)
\`\`\`

---

## Distributed Workflow Orchestration

### Multi-Agent Coordination Patterns

\`\`\`python
from typing import List, Dict, Any
from enum import Enum

class WorkflowState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class DistributedWorkflowOrchestrator:
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.agents = {}
        self.workflow_state = WorkflowState.PENDING
        self.coordination_events = {}
        self.results = {}

    def register_agent(self, agent_id: str, agent_config: Dict):
        """Register an agent in the distributed workflow"""
        self.agents[agent_id] = {
            "config": agent_config,
            "state": WorkflowState.PENDING,
            "results": None,
            "dependencies": agent_config.get("dependencies", [])
        }

    async def execute_workflow(self, context):
        """Execute the distributed workflow"""
        print(f"🎭 Orchestrator: Starting distributed workflow {self.workflow_id}")

        self.workflow_state = WorkflowState.RUNNING

        try:
            # Execute agents in dependency order
            execution_order = self._calculate_execution_order()

            for agent_id in execution_order:
                await self._execute_agent(agent_id, context)

            self.workflow_state = WorkflowState.COMPLETED
            print(f"✅ Orchestrator: Workflow {self.workflow_id} completed successfully")

        except Exception as e:
            self.workflow_state = WorkflowState.FAILED
            print(f"❌ Orchestrator: Workflow {self.workflow_id} failed: {e}")
            raise

    def _calculate_execution_order(self) -> List[str]:
        """Calculate agent execution order based on dependencies"""
        # Topological sort implementation
        visited = set()
        order = []

        def visit(agent_id):
            if agent_id in visited:
                return
            visited.add(agent_id)

            dependencies = self.agents[agent_id]["dependencies"]
            for dep in dependencies:
                if dep in self.agents:
                    visit(dep)

            order.append(agent_id)

        for agent_id in self.agents:
            visit(agent_id)

        return order

    async def _execute_agent(self, agent_id: str, context):
        """Execute a specific agent"""
        agent_config = self.agents[agent_id]["config"]

        print(f"🤖 Orchestrator: Executing agent {agent_id}")

        self.agents[agent_id]["state"] = WorkflowState.RUNNING

        try:
            # Check dependencies
            for dep_id in agent_config["dependencies"]:
                if self.agents[dep_id]["state"] != WorkflowState.COMPLETED:
                    raise Exception(f"Dependency {dep_id} not completed")

            # Execute agent function
            agent_func = agent_config["function"]
            result = await agent_func(context, agent_id)

            # Store results
            self.agents[agent_id]["results"] = result
            self.agents[agent_id]["state"] = WorkflowState.COMPLETED
            self.results[agent_id] = result

            print(f"✅ Orchestrator: Agent {agent_id} completed")

        except Exception as e:
            self.agents[agent_id]["state"] = WorkflowState.FAILED
            print(f"❌ Orchestrator: Agent {agent_id} failed: {e}")
            raise

# Create distributed workflow orchestrator
orchestrator = DistributedWorkflowOrchestrator("distributed_ai_pipeline")

# Define agent functions
async def data_ingestion_agent(context, agent_id):
    """Agent responsible for data ingestion"""
    print(f"   📥 {agent_id}: Starting data ingestion...")

    # Simulate data ingestion
    await asyncio.sleep(3.0)

    ingested_data = {
        "records_ingested": 10000,
        "data_sources": ["database", "api", "files"],
        "ingestion_time": 3.0,
        "data_quality_score": 0.95
    }

    print(f"   ✅ {agent_id}: Ingested {ingested_data['records_ingested']} records")
    return ingested_data

async def data_validation_agent(context, agent_id):
    """Agent responsible for data validation"""
    print(f"   🔍 {agent_id}: Starting data validation...")

    # Get ingestion results
    ingestion_results = orchestrator.results.get("data_ingestion")
    if not ingestion_results:
        raise Exception("No ingestion data available")

    # Simulate validation
    await asyncio.sleep(2.0)

    validation_results = {
        "records_validated": ingestion_results["records_ingested"],
        "validation_errors": 50,
        "error_rate": 0.005,
        "validation_time": 2.0,
        "passed_quality_checks": True
    }

    print(f"   ✅ {agent_id}: Validated {validation_results['records_validated']} records")
    return validation_results

async def ai_processing_agent(context, agent_id):
    """Agent responsible for AI processing"""
    print(f"   🧠 {agent_id}: Starting AI processing...")

    # Get validation results
    validation_results = orchestrator.results.get("data_validation")
    if not validation_results:
        raise Exception("No validation data available")

    # Simulate AI processing
    await asyncio.sleep(5.0)

    ai_results = {
        "records_processed": validation_results["records_validated"] - validation_results["validation_errors"],
        "model_predictions": 9950,
        "confidence_score": 0.87,
        "processing_time": 5.0,
        "model_version": "v2.1.0"
    }

    print(f"   ✅ {agent_id}: Processed {ai_results['records_processed']} records with AI")
    return ai_results

async def results_aggregation_agent(context, agent_id):
    """Agent responsible for aggregating results"""
    print(f"   📊 {agent_id}: Starting results aggregation...")

    # Get all previous results
    ingestion_results = orchestrator.results.get("data_ingestion", {})
    validation_results = orchestrator.results.get("data_validation", {})
    ai_results = orchestrator.results.get("ai_processing", {})

    # Simulate aggregation
    await asyncio.sleep(1.0)

    aggregated_results = {
        "pipeline_summary": {
            "total_records": ingestion_results.get("records_ingested", 0),
            "validation_errors": validation_results.get("validation_errors", 0),
            "successful_predictions": ai_results.get("model_predictions", 0),
            "overall_success_rate": ai_results.get("model_predictions", 0) / max(ingestion_results.get("records_ingested", 1), 1),
            "total_pipeline_time": 11.0  # Sum of all processing times
        },
        "quality_metrics": {
            "data_quality_score": ingestion_results.get("data_quality_score", 0),
            "validation_pass_rate": 1 - validation_results.get("error_rate", 0),
            "ai_confidence_score": ai_results.get("confidence_score", 0)
        },
        "aggregation_time": 1.0
    }

    print(f"   ✅ {agent_id}: Aggregation complete")
    return aggregated_results

# Register agents with dependencies
orchestrator.register_agent("data_ingestion", {
    "function": data_ingestion_agent,
    "dependencies": []
})

orchestrator.register_agent("data_validation", {
    "function": data_validation_agent,
    "dependencies": ["data_ingestion"]
})

orchestrator.register_agent("ai_processing", {
    "function": ai_processing_agent,
    "dependencies": ["data_validation"]
})

orchestrator.register_agent("results_aggregation", {
    "function": results_aggregation_agent,
    "dependencies": ["data_ingestion", "data_validation", "ai_processing"]
})

# Orchestration agent
orchestration_agent = Agent("distributed-orchestration")

@state(timeout=120.0)
async def execute_distributed_workflow(context):
    """Execute the complete distributed workflow"""
    print("🎭 Starting distributed workflow orchestration...")

    workflow_start = time.time()

    try:
        await orchestrator.execute_workflow(context)

        workflow_duration = time.time() - workflow_start

        # Store final results
        final_results = {
            "workflow_id": orchestrator.workflow_id,
            "workflow_state": orchestrator.workflow_state.value,
            "total_duration": workflow_duration,
            "agent_results": orchestrator.results,
            "execution_summary": {
                "agents_executed": len(orchestrator.agents),
                "successful_agents": len([a for a in orchestrator.agents.values() if a["state"] == WorkflowState.COMPLETED]),
                "failed_agents": len([a for a in orchestrator.agents.values() if a["state"] == WorkflowState.FAILED])
            }
        }

        context.set_output("distributed_workflow_results", final_results)

        print(f"🎉 Distributed workflow completed in {workflow_duration:.2f}s")

    except Exception as e:
        print(f"❌ Distributed workflow failed: {e}")
        raise

orchestration_agent.add_state("execute_workflow", execute_distributed_workflow)
\`\`\`

---

## Best Practices Summary

### Coordination Design Principles

1. **Choose the Right Primitive**
   - **Semaphore**: Limited resource pools
   - **Mutex**: Exclusive access protection
   - **Barrier**: Synchronized parallel execution
   - **Event**: Asynchronous notifications

2. **Design for Deadlock Prevention**
   - Always acquire resources in consistent order
   - Use timeouts on all coordination operations
   - Implement proper cleanup in finally blocks

3. **Monitor Coordination Health**
   - Track resource utilization
   - Monitor barrier wait times
   - Log coordination events

4. **Handle Failures Gracefully**
   - Release resources on exceptions
   - Provide fallback coordination mechanisms
   - Implement coordination timeouts

### Quick Reference

\`\`\`python
# Semaphore: Control concurrent access
semaphore = Semaphore("resource_pool", max_count=5)
async with semaphore.acquire_context(requester_id):
    # Protected code here
    pass

# Mutex: Exclusive access
mutex = Mutex("shared_resource")
async with mutex.acquire_context(requester_id):
    # Critical section here
    pass

# Barrier: Synchronized execution
barrier = Barrier("sync_point", parties=4)
await barrier.wait(participant_id, timeout=30.0)

# Event: Asynchronous signaling
event = Event("data_ready")
await event.wait(timeout=60.0)
await event.set({"data": "payload"})
\`\`\`

Coordination in Puffinflow enables you to build sophisticated, scalable workflows that can manage complex interactions between distributed components while maintaining consistency and reliability.
`.trim();
