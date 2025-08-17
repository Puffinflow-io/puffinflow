export const resourceManagementMarkdown = `# Resource Management

Resource management in Puffinflow is like organizing a busy workspace. You tell each worker how much equipment they need, when they can use shared tools, and how to avoid conflicts. This keeps everything running smoothly without anyone getting overwhelmed.

## Why Resource Management Matters

**Without resource management:**
- Your computer might run out of memory and crash
- Heavy tasks might slow down everything else
- API calls might hit rate limits and fail
- Multiple workflows might fight for the same resources

**With Puffinflow's resource management:**
- Each state gets exactly what it needs
- Your system stays stable and responsive
- API limits are respected automatically
- Workflows share resources fairly

## Part 1: The Essentials (Start Here)

### Basic CPU and Memory

The two most important resources are CPU (processing power) and Memory (storage space):

\`\`\`python
from puffinflow import Agent, state

agent = Agent("resource-demo")

# Light task - like reading email
@state(cpu=0.5, memory=256)
async def light_task(context):
    print("✅ Quick check completed")
    return "medium_task"

# Medium task - like processing a document
@state(cpu=2.0, memory=1024)
async def medium_task(context):
    # Simulate some data processing
    data = list(range(10000))
    result = [x * 2 for x in data]
    context.set_variable("processed_data", result)
    return "heavy_task"

# Heavy task - like machine learning
@state(cpu=4.0, memory=4096)
async def heavy_task(context):
    import time
    time.sleep(2)  # Simulate heavy work
    print("💪 Heavy computation finished")
    return None

# Add states to agent
agent.add_state("light_task", light_task)
agent.add_state("medium_task", medium_task)
agent.add_state("heavy_task", heavy_task)
\`\`\`

### Understanding the Numbers

**CPU (Processing Power):**
- \`cpu=0.5\` - Half a processor core (light work)
- \`cpu=1.0\` - One full processor core (normal work)
- \`cpu=2.0\` - Two processor cores (heavy work)
- \`cpu=4.0+\` - Multiple cores (very heavy work)

**Memory (Storage Space):**
- \`memory=256\` - 256 MB (basic tasks)
- \`memory=512\` - 512 MB (default, most tasks)
- \`memory=1024\` - 1 GB (data processing)
- \`memory=4096+\` - 4+ GB (large datasets)

### Timeouts: Preventing Tasks from Getting Stuck

Sometimes tasks might hang forever. Use timeouts to prevent this:

\`\`\`python
@state(cpu=1.0, memory=512, timeout=30.0)
async def might_get_stuck(context):
    """
    This task has a 30-second timeout
    If it takes longer, Puffinflow will stop it and move on
    """
    import asyncio
    print("⏱️ Starting task that might take a while...")
    await asyncio.sleep(5)  # This is fine (under 30 seconds)
    print("✅ Task completed in time!")
    return None

# Add state to agent
agent.add_state("might_get_stuck", might_get_stuck)
\`\`\`

### Retries: Making Tasks More Reliable

If a task fails, you can make it try again automatically:

\`\`\`python
@state(cpu=1.0, memory=512, max_retries=3, timeout=10.0)
async def might_fail(context):
    """
    This task will retry up to 3 times if it fails
    Perfect for network calls or external services
    """
    import random

    print("🎲 Attempting task that might fail...")

    # Simulate a task that fails sometimes
    if random.random() < 0.7:  # 70% chance of failure
        print("❌ Task failed, will retry...")
        raise Exception("Random failure for demo")

    print("✅ Task succeeded!")
    context.set_variable("success", True)
    return None

# Add state to agent
agent.add_state("might_fail", might_fail)
\`\`\`

### Rate Limiting: Being Nice to External Services

When calling APIs or external services, don't overwhelm them:

\`\`\`python
@state(cpu=0.5, memory=256, rate_limit=2.0)
async def call_api(context):
    """
    rate_limit=2.0 means max 2 calls per second
    This prevents you from hitting API rate limits
    """
    import asyncio
    print("🌐 Calling external API...")
    await asyncio.sleep(0.1)  # Simulate network delay

    result = {"data": "API response", "status": "success"}
    context.set_variable("api_result", result)
    return "process_response"

# Add state to agent
agent.add_state("call_api", call_api)
\`\`\`

## Part 2: Specialized Resources

### GPU for Machine Learning

If you're doing machine learning or graphics work:

\`\`\`python
@agent.state(cpu=2.0, memory=4096, gpu=1.0)
async def ml_inference(context):
    """
    gpu=1.0 requests one GPU unit
    Perfect for machine learning models
    """
    model_input = context.get_variable("input_data")
    print("🤖 Running AI model on GPU...")

    # Simulate GPU computation
    await asyncio.sleep(1)
    result = f"AI processed: {model_input}"

    context.set_variable("ai_result", result)
    return None
\`\`\`

### I/O for File Operations

For tasks that read/write lots of files:

\`\`\`python
@agent.state(cpu=1.0, memory=1024, io=10.0)
async def process_large_file(context):
    """
    io=10.0 requests high I/O bandwidth
    Good for file processing, database operations
    """
    print("📁 Processing large file...")

    # Simulate heavy file I/O
    await asyncio.sleep(2)

    context.set_variable("file_processed", True)
    return None
\`\`\`

### Network for Data Transfer

For tasks that transfer lots of data:

\`\`\`python
@agent.state(cpu=0.5, memory=512, network=8.0, rate_limit=5.0)
async def download_data(context):
    """
    network=8.0 requests high network bandwidth
    Combined with rate_limit for responsible usage
    """
    print("📥 Downloading large dataset...")

    # Simulate network download
    await asyncio.sleep(3)

    context.set_variable("data_downloaded", True)
    return None
\`\`\`

## Part 3: Coordination (Sharing Resources Safely)

### Mutex: One at a Time

Sometimes only one task should access a resource at a time:

\`\`\`python
@agent.state(cpu=1.0, memory=512, mutex=True)
async def update_shared_file(context):
    """
    mutex=True means only one instance of this state can run at a time
    Perfect for updating files or databases that don't support concurrent writes
    """
    print("📝 Updating shared configuration file...")

    # Only one task can do this at a time
    current_config = {"version": 1, "updated": True}
    context.set_variable("config", current_config)

    await asyncio.sleep(1)  # Simulate file update
    print("✅ Configuration updated safely")
    return None
\`\`\`

### Semaphore: Limited Concurrent Access

Control how many tasks can run simultaneously:

\`\`\`python
@agent.state(cpu=1.0, memory=512, semaphore=3)
async def database_query(context):
    """
    semaphore=3 means max 3 instances can run at the same time
    Prevents overwhelming your database with too many connections
    """
    query_id = context.get_variable("query_id", "unknown")
    print(f"🗄️ Running database query {query_id}")

    # Simulate database work
    await asyncio.sleep(2)

    context.set_variable("query_result", f"Data for query {query_id}")
    print(f"✅ Query {query_id} completed")
    return None
\`\`\`

### Barrier: Wait for Everyone

Make multiple tasks wait for each other:

\`\`\`python
@agent.state(cpu=1.0, memory=512, barrier=3)
async def parallel_data_prep(context):
    """
    barrier=3 means wait until 3 tasks reach this point, then all continue together
    Perfect for synchronizing parallel work
    """
    task_id = context.get_variable("task_id", "unknown")
    print(f"📊 Task {task_id} preparing data...")

    # Do individual work first
    await asyncio.sleep(1)
    print(f"Task {task_id} ready, waiting for others...")

    # This point waits for all 3 tasks
    print(f"🚀 All tasks ready! Task {task_id} proceeding...")
    return None
\`\`\`

### Lease: Time-Limited Access

Reserve a resource for a specific amount of time:

\`\`\`python
@agent.state(cpu=2.0, memory=1024, lease=60.0)
async def batch_processing(context):
    """
    lease=60.0 reserves resources for 60 seconds
    Good for batch jobs that need guaranteed time
    """
    print("⏰ Starting batch job with 60-second lease...")

    # You have guaranteed access to resources for 60 seconds
    for i in range(10):
        print(f"Processing batch {i+1}/10")
        await asyncio.sleep(5)  # 50 seconds total

    print("✅ Batch processing completed within lease time")
    return None
\`\`\`

## Part 4: Priority and Performance

### Task Priority

Some tasks are more important than others:

\`\`\`python
from puffinflow import Priority

@agent.state(cpu=2.0, memory=1024, priority=Priority.HIGH)
async def urgent_task(context):
    """
    priority=Priority.HIGH makes this task run before normal tasks
    Available priorities: CRITICAL, HIGH, NORMAL, LOW, BACKGROUND
    """
    print("🚨 Running high-priority urgent task")
    await asyncio.sleep(1)
    return None

@agent.state(cpu=1.0, memory=512, priority=Priority.LOW)
async def background_cleanup(context):
    """
    priority=Priority.LOW makes this run when system isn't busy
    Perfect for maintenance tasks
    """
    print("🧹 Running background cleanup")
    await asyncio.sleep(2)
    return None
\`\`\`

### Burst Limits for Rate Limiting

Allow temporary spikes in activity:

\`\`\`python
@agent.state(cpu=0.5, memory=256, rate_limit=2.0, burst_limit=10)
async def api_with_bursts(context):
    """
    rate_limit=2.0: normally 2 calls per second
    burst_limit=10: but allow up to 10 calls in a burst
    """
    print("🌐 Making API call (with burst capability)")
    # Can handle sudden spikes in activity
    return None
\`\`\`

## Complete Real-World Example: Image Processing Service

\`\`\`python
import asyncio
from puffinflow import Agent, Priority

agent = Agent("image-service")

@agent.state(cpu=0.5, memory=256, rate_limit=10.0)
async def receive_upload(context):
    """Handle image upload - light but rate-limited"""
    print("📤 Image uploaded")
    context.set_variable("image_path", "/tmp/photo.jpg")
    context.set_variable("image_size", "2MB")
    return "validate_image"

@agent.state(cpu=1.0, memory=512, timeout=30.0, max_retries=2)
async def validate_image(context):
    """Validate image format - might fail, so retry"""
    image_path = context.get_variable("image_path")
    print(f"🔍 Validating: {image_path}")

    # Simulate validation that might fail
    await asyncio.sleep(1)

    context.set_variable("valid", True)
    return "resize_image"

@agent.state(cpu=2.0, memory=1024, timeout=60.0, io=5.0)
async def resize_image(context):
    """Resize image - CPU and I/O intensive"""
    image_path = context.get_variable("image_path")
    print(f"🖼️ Resizing: {image_path}")

    # Simulate image processing
    await asyncio.sleep(2)

    context.set_variable("resized_path", "/tmp/photo_resized.jpg")
    return "apply_ai_filters"

@agent.state(cpu=3.0, memory=2048, gpu=1.0, timeout=120.0,
             priority=Priority.HIGH, semaphore=2)
async def apply_ai_filters(context):
    """AI processing - high priority, GPU, limited concurrency"""
    resized_path = context.get_variable("resized_path")
    print(f"🤖 Applying AI filters: {resized_path}")

    # Simulate AI processing on GPU
    await asyncio.sleep(3)

    context.set_variable("filtered_path", "/tmp/photo_ai.jpg")
    return "save_to_gallery"

@agent.state(cpu=0.5, memory=256, rate_limit=5.0, mutex=True)
async def save_to_gallery(context):
    """Save final image - rate-limited, exclusive access to gallery"""
    filtered_path = context.get_variable("filtered_path")
    print(f"💾 Saving to gallery: {filtered_path}")

    # Update gallery database (needs exclusive access)
    await asyncio.sleep(1)

    context.set_output("final_image", filtered_path)
    context.set_output("processing_complete", True)
    return None

# Run the service
async def main():
    await agent.run(initial_state="receive_upload")

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

## Quick Reference Guide

### Resource Allocation Cheat Sheet

**For CPU:**
- **Light tasks** (API calls, simple logic): \`cpu=0.5-1.0\`
- **Normal tasks** (data processing): \`cpu=1.0-2.0\`
- **Heavy tasks** (ML, complex math): \`cpu=3.0-8.0\`

**For Memory:**
- **Small data** (basic variables): \`memory=256-512\`
- **Medium data** (processing files): \`memory=1024-2048\`
- **Large data** (big datasets): \`memory=4096+\`

**For Timeouts:**
- **Quick tasks**: \`timeout=10-30\` seconds
- **Normal tasks**: \`timeout=60-300\` seconds
- **Long tasks**: \`timeout=600+\` seconds

**For Rate Limits:**
- **External APIs**: \`rate_limit=1.0-10.0\` (check API docs)
- **Database calls**: \`rate_limit=5.0-20.0\`
- **File operations**: \`rate_limit=2.0-10.0\`

### Common Patterns

\`\`\`python
# Pattern 1: Simple API call
@agent.state(cpu=0.5, memory=256, rate_limit=5.0, timeout=10.0, max_retries=3)
async def api_call(context):
    pass

# Pattern 2: Data processing
@agent.state(cpu=2.0, memory=1024, timeout=300.0)
async def process_data(context):
    pass

# Pattern 3: Machine learning
@agent.state(cpu=3.0, memory=4096, gpu=1.0, timeout=600.0)
async def ml_training(context):
    pass

# Pattern 4: File processing
@agent.state(cpu=1.0, memory=1024, io=10.0, timeout=300.0)
async def process_files(context):
    pass

# Pattern 5: Database operation
@agent.state(cpu=1.0, memory=512, semaphore=5, timeout=60.0)
async def database_query(context):
    pass

# Pattern 6: Background task
@agent.state(cpu=0.5, memory=256, priority=Priority.LOW, timeout=3600.0)
async def background_cleanup(context):
    pass
\`\`\`

## Advanced Features (When You're Ready)

### Resource Quotas

Set limits per user or tenant:

\`\`\`python
# This is configured at the agent/system level
# Each user gets max 4 CPU cores and 8GB memory total
agent.set_quota("user_123", cpu=4.0, memory=8192)
\`\`\`

### Preemptible Tasks

Allow high-priority tasks to interrupt low-priority ones:

\`\`\`python
@agent.state(cpu=2.0, memory=1024, priority=Priority.LOW, preemptible=True)
async def can_be_interrupted(context):
    """This task can be paused if something urgent comes up"""
    pass
\`\`\`

### Resource Monitoring

Track how much resources you're actually using:

\`\`\`python
@agent.state(cpu=2.0, memory=1024, enable_monitoring=True)
async def monitored_task(context):
    """Resource usage will be tracked and reported"""
    pass
\`\`\`

## Tips for Beginners

1. **Start small** - Begin with basic \`cpu\`, \`memory\`, and \`timeout\` settings
2. **Monitor real usage** - Check how much your tasks actually use
3. **Add retries for network calls** - External services can be unreliable
4. **Use rate limits with APIs** - Respect external service limits
5. **Test with real data** - Resource needs change with data size
6. **Don't over-optimize early** - Start simple, optimize when needed

## What Each Feature Is Good For

- **CPU/Memory/GPU**: Basic resource allocation (use always)
- **Timeout**: Preventing stuck tasks (use always)
- **Max retries**: Handling failures (use for network/external calls)
- **Rate limiting**: Respecting API limits (use for external services)
- **Mutex**: Exclusive access (use for file/database writes)
- **Semaphore**: Limited concurrency (use for connection pools)
- **Barrier**: Synchronization (use for parallel coordination)
- **Lease**: Guaranteed time (use for batch jobs)
- **Priority**: Importance ordering (use sparingly)
- **I/O/Network**: Specialized bandwidth (use for heavy file/network work)

Start with the essentials and add advanced features as your needs grow. Resource management helps your workflows run smoothly and reliably!
`.trim();
