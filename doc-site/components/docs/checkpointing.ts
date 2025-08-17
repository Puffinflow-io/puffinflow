
export const checkpointingMarkdown = `# Checkpointing

Checkpointing is like creating save points in a video game - you can save your progress and continue from exactly where you left off if something goes wrong. This is incredibly useful for long-running workflows, handling system crashes, and working with cloud services that might restart your machine.

## Why Use Checkpoints?

**Without checkpointing:**
- If your workflow crashes after 2 hours, you start over from the beginning
- Cloud interruptions (like spot instances shutting down) waste all your work
- Long-running tasks are fragile and unreliable
- You can't afford to use cheaper, interruptible cloud resources

**With Puffinflow's checkpointing:**
- Resume from exactly where you left off after any interruption
- Use cheaper cloud resources safely (spot instances, preemptible VMs)
- Long-running workflows become robust and reliable
- Save money by recovering gracefully from interruptions

## Part 1: The Basics (Start Here)

### What Gets Saved in a Checkpoint?

A checkpoint saves everything about your workflow's current state:
- Which states have completed
- All data stored in context variables
- Current progress and position in the workflow
- Timestamp of when the checkpoint was created

### Your First Checkpoint

\`\`\`python
import asyncio
from puffinflow import Agent, state

agent = Agent("my-first-checkpoint")

@state
async def step_one(context):
    print("🔄 Step 1: Processing data...")
    # Do some work
    context.set_variable("step1_result", "completed")
    return "step_two"

@state
async def step_two(context):
    print("🔄 Step 2: Analyzing results...")
    # Do more work
    result = context.get_variable("step1_result")
    context.set_variable("step2_result", f"analyzed {result}")
    return "step_three"

@state
async def step_three(context):
    print("🔄 Step 3: Generating report...")
    # Final work
    context.set_variable("final_report", "Report completed!")
    return None

async def main():
    # Run the workflow
    await agent.run(initial_state="step_one")

    # Create a checkpoint (save point)
    checkpoint = agent.create_checkpoint()
    print("✅ Checkpoint created!")
    print(f"Saved {len(checkpoint.completed_states)} completed states")

    # Later, if something goes wrong, you can restore:
    # await agent.restore_from_checkpoint(checkpoint)

# Add states to agent
agent.add_state("step_one", step_one)
agent.add_state("step_two", step_two)
agent.add_state("step_three", step_three)

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

### Saving and Loading Progress

\`\`\`python
import asyncio
from puffinflow import Agent, state

agent = Agent("save-load-demo")

@state
async def long_process(context):
    """Simulate a long process that we want to save progress for"""

    # Check if we're resuming from a saved state
    progress = context.get_variable("progress", {"completed": 0, "total": 10})

    print(f"Starting from item {progress['completed']}")

    for i in range(progress["completed"], progress["total"]):
        print(f"📊 Processing item {i + 1}/{progress['total']}")

        # Simulate work
        await asyncio.sleep(1)

        # Update progress
        progress["completed"] = i + 1
        context.set_variable("progress", progress)

        # Save checkpoint every 3 items
        if (i + 1) % 3 == 0:
            checkpoint = agent.create_checkpoint()
            print(f"💾 Checkpoint saved at item {i + 1}")

    print("✅ All items processed!")
    return None

# Add state to agent
agent.add_state("long_process", long_process)

async def main():
    # Run workflow
    await agent.run(initial_state="long_process")

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

## Part 2: Automatic Checkpointing

Instead of manually saving checkpoints, you can have Puffinflow automatically save progress at regular intervals:

\`\`\`python
# Automatically save every 30 seconds
@state(checkpoint_interval=30.0)
async def auto_save_task(context):
    """
    This state will automatically create a checkpoint every 30 seconds
    You don't have to do anything - Puffinflow handles it!
    """
    print("⏰ Starting task with automatic checkpointing...")

    # Simulate a long-running task
    for minute in range(5):  # 5 minutes of work
        print(f"🔄 Working... minute {minute + 1}/5")

        # Track our progress
        progress = {
            "current_minute": minute + 1,
            "total_minutes": 5,
            "completion": ((minute + 1) / 5) * 100
        }
        context.set_variable("work_progress", progress)

        # Do 60 seconds of work (simulated)
        for second in range(60):
            await asyncio.sleep(1)  # 1 second of "work"

            # Every 30 seconds, Puffinflow automatically saves a checkpoint!
            if second % 30 == 0 and second > 0:
                print("💾 Auto-checkpoint saved!")

    print("✅ Task completed!")
    return None

# Add state to agent
agent.add_state("auto_save_task", auto_save_task)
\`\`\`

### Different Checkpoint Frequencies

\`\`\`python
# For quick tasks: checkpoint frequently
@state(checkpoint_interval=10.0)  # Every 10 seconds
async def frequent_saves(context):
    """For tasks where you don't want to lose much progress"""
    pass

# For expensive tasks: checkpoint less often
@state(checkpoint_interval=300.0)  # Every 5 minutes
async def expensive_saves(context):
    """For tasks where checkpointing itself is expensive"""
    pass

# For critical tasks: checkpoint very frequently
@state(checkpoint_interval=5.0)   # Every 5 seconds
async def critical_saves(context):
    """For absolutely critical tasks"""
    pass

# Add states to agent
agent.add_state("frequent_saves", frequent_saves)
agent.add_state("expensive_saves", expensive_saves)
agent.add_state("critical_saves", critical_saves)
\`\`\`

## Part 3: Smart Recovery

### Resuming Exactly Where You Left Off

\`\`\`python
import asyncio
from puffinflow import Agent, state

agent = Agent("smart-recovery")

@state
async def data_processor(context):
    """A processor that knows how to resume intelligently"""

    # Check if we're resuming from a checkpoint
    batch_info = context.get_variable("batch_processing", {
        "current_batch": 0,
        "total_batches": 20,
        "items_per_batch": 50,
        "total_processed": 0
    })

    if batch_info["current_batch"] > 0:
        print(f"🔄 Resuming from batch {batch_info['current_batch']}")
        print(f"📊 Already processed {batch_info['total_processed']} items")
    else:
        print("🚀 Starting fresh data processing")

    # Continue from where we left off
    for batch_num in range(batch_info["current_batch"], batch_info["total_batches"]):
        print(f"📦 Processing batch {batch_num + 1}/{batch_info['total_batches']}")

        # Process items in this batch
        for item_num in range(batch_info["items_per_batch"]):
            # Simulate processing an item
            await asyncio.sleep(0.1)
            batch_info["total_processed"] += 1

        # Update progress
        batch_info["current_batch"] = batch_num + 1
        context.set_variable("batch_processing", batch_info)

        # Save checkpoint after each batch
        checkpoint = agent.create_checkpoint()
        print(f"💾 Batch {batch_num + 1} complete, checkpoint saved")

        # Show progress
        completion_percent = (batch_info["current_batch"] / batch_info["total_batches"]) * 100
        print(f"📈 Overall progress: {completion_percent:.1f}%")

    print("🎉 All batches processed successfully!")
    context.set_output("total_items_processed", batch_info["total_processed"])
    return None

# Add state to agent
agent.add_state("data_processor", data_processor)

async def main():
    await agent.run(initial_state="data_processor")

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

### Handling Different Types of Interruptions

\`\`\`python
import asyncio
import signal
from puffinflow import Agent, state

agent = Agent("interruption-handler")

class GracefulCheckpointer:
    def __init__(self, agent):
        self.agent = agent
        self.checkpoint_file = "emergency_checkpoint.json"

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.handle_shutdown)  # Cloud shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)   # Ctrl+C

    def handle_shutdown(self, signum, frame):
        """Save checkpoint when system is shutting down"""
        print(f"🛑 Shutdown signal received: {signum}")
        print("💾 Creating emergency checkpoint...")

        try:
            checkpoint = self.agent.create_checkpoint()
            # In real usage, save to persistent storage (S3, database, etc.)
            print("✅ Emergency checkpoint saved!")
            print("🔄 Workflow can be resumed later")
        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")

        exit(0)

@state
async def resilient_task(context):
    """A task that handles interruptions gracefully"""

    # Set up graceful checkpointing
    checkpointer = GracefulCheckpointer(agent)

    work_state = context.get_variable("work_state", {
        "phase": "initialization",
        "completed_work_units": 0,
        "total_work_units": 1000
    })

    print(f"🔄 Resuming {work_state['phase']} phase")
    print(f"📊 {work_state['completed_work_units']}/{work_state['total_work_units']} work units done")

    # Do the work with regular checkpoints
    for work_unit in range(work_state["completed_work_units"], work_state["total_work_units"]):
        # Simulate work
        await asyncio.sleep(0.01)

        # Update state
        work_state["completed_work_units"] = work_unit + 1

        # Update phase based on progress
        progress = work_unit / work_state["total_work_units"]
        if progress < 0.3:
            work_state["phase"] = "initialization"
        elif progress < 0.7:
            work_state["phase"] = "processing"
        else:
            work_state["phase"] = "finalization"

        context.set_variable("work_state", work_state)

        # Checkpoint every 50 work units
        if (work_unit + 1) % 50 == 0:
            checkpoint = agent.create_checkpoint()
            print(f"💾 Checkpoint: {work_unit + 1}/1000 units ({progress*100:.1f}%)")

    print("🎉 All work completed!")
    return None

# Add state to agent
agent.add_state("resilient_task", resilient_task)
\`\`\`

## Part 4: File Storage and Persistence

### Saving Checkpoints to Files

\`\`\`python
import json
import asyncio
from pathlib import Path
from puffinflow import Agent, state

agent = Agent("file-checkpoints")

@state
async def file_processor(context):
    """Process files and save checkpoints to disk"""

    file_state = context.get_variable("file_state", {
        "processed_files": [],
        "current_index": 0,
        "total_files": 100
    })

    print(f"📁 Processing files {file_state['current_index']}/{file_state['total_files']}")

    # Process remaining files
    for i in range(file_state["current_index"], file_state["total_files"]):
        filename = f"file_{i+1:03d}.txt"
        print(f"🔄 Processing {filename}")

        # Simulate file processing
        await asyncio.sleep(0.5)

        # Update state
        file_state["processed_files"].append(filename)
        file_state["current_index"] = i + 1
        context.set_variable("file_state", file_state)

        # Save to file every 10 files
        if (i + 1) % 10 == 0:
            checkpoint = agent.create_checkpoint()
            save_checkpoint_to_file(checkpoint, f"checkpoint_{i+1:03d}.json")
            print(f"💾 Checkpoint saved: {i+1} files processed")

    print("✅ All files processed!")
    return None

# Add state to agent
agent.add_state("file_processor", file_processor)

def save_checkpoint_to_file(checkpoint, filename):
    """Save checkpoint data to a JSON file"""
    checkpoint_data = {
        "timestamp": checkpoint.timestamp,
        "agent_name": checkpoint.agent_name,
        "completed_states": list(checkpoint.completed_states),
        "shared_state": checkpoint.shared_state
    }

    # Create checkpoints directory
    Path("checkpoints").mkdir(exist_ok=True)

    # Save to file
    with open(f"checkpoints/{filename}", 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

    print(f"📄 Checkpoint saved to checkpoints/{filename}")

def load_checkpoint_from_file(filename):
    """Load checkpoint data from a JSON file"""
    try:
        with open(f"checkpoints/{filename}", 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Checkpoint file not found: {filename}")
        return None

async def main():
    # Check for existing checkpoint
    checkpoint_data = load_checkpoint_from_file("latest_checkpoint.json")

    if checkpoint_data:
        print("🔄 Found existing checkpoint, resuming...")
        # In real usage, you'd recreate the checkpoint object and restore
        # await agent.restore_from_checkpoint(checkpoint)
    else:
        print("🚀 No checkpoint found, starting fresh")

    await agent.run(initial_state="file_processor")

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

## Part 5: Cloud Resilience

### Handling Cloud Interruptions

\`\`\`python
import asyncio
import boto3
from puffinflow import Agent, state

agent = Agent("cloud-resilient")

class CloudCheckpointManager:
    def __init__(self, agent, bucket_name):
        self.agent = agent
        self.bucket_name = bucket_name
        self.s3 = boto3.client('s3')

    def save_to_cloud(self, checkpoint):
        """Save checkpoint to cloud storage (S3)"""
        checkpoint_key = f"checkpoints/{checkpoint.agent_name}_{int(checkpoint.timestamp)}.json"

        checkpoint_data = {
            "timestamp": checkpoint.timestamp,
            "agent_name": checkpoint.agent_name,
            "completed_states": list(checkpoint.completed_states),
            "shared_state": checkpoint.shared_state
        }

        try:
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=checkpoint_key,
                Body=json.dumps(checkpoint_data),
                ContentType='application/json'
            )
            print(f"☁️ Checkpoint saved to S3: {checkpoint_key}")
            return checkpoint_key
        except Exception as e:
            print(f"❌ Failed to save to S3: {e}")
            return None

    def load_from_cloud(self, checkpoint_key):
        """Load checkpoint from cloud storage"""
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=checkpoint_key)
            checkpoint_data = json.loads(response['Body'].read())
            print(f"☁️ Checkpoint loaded from S3: {checkpoint_key}")
            return checkpoint_data
        except Exception as e:
            print(f"❌ Failed to load from S3: {e}")
            return None

@state
async def cloud_safe_processing(context):
    """Processing that survives cloud interruptions"""

    # Set up cloud checkpoint manager
    cloud_manager = CloudCheckpointManager(agent, "my-workflow-checkpoints")

    work_progress = context.get_variable("cloud_work", {
        "stage": "data_collection",
        "items_processed": 0,
        "total_items": 500,
        "start_time": time.time()
    })

    print(f"☁️ Cloud processing stage: {work_progress['stage']}")
    print(f"📊 Progress: {work_progress['items_processed']}/{work_progress['total_items']}")

    # Process items with cloud-safe checkpointing
    for item_id in range(work_progress["items_processed"], work_progress["total_items"]):
        # Simulate processing
        await asyncio.sleep(0.1)

        # Update progress
        work_progress["items_processed"] = item_id + 1

        # Update stage based on progress
        progress_percent = (item_id + 1) / work_progress["total_items"]
        if progress_percent < 0.33:
            work_progress["stage"] = "data_collection"
        elif progress_percent < 0.66:
            work_progress["stage"] = "data_processing"
        else:
            work_progress["stage"] = "data_output"

        context.set_variable("cloud_work", work_progress)

        # Save to cloud every 25 items (checkpoint frequently for spot instances)
        if (item_id + 1) % 25 == 0:
            checkpoint = agent.create_checkpoint()
            cloud_key = cloud_manager.save_to_cloud(checkpoint)

            if cloud_key:
                # Store the cloud key so we can resume later
                context.set_variable("latest_cloud_checkpoint", cloud_key)
                print(f"☁️ Cloud checkpoint: {progress_percent*100:.1f}% complete")

    print("🎉 Cloud processing completed successfully!")
    return None

# Add state to agent
agent.add_state("cloud_safe_processing", cloud_safe_processing)
\`\`\`

## Part 6: Progress Tracking and Monitoring

### Advanced Progress Tracking

\`\`\`python
import time
import asyncio
from puffinflow import Agent, state

agent = Agent("progress-tracker")

@state
async def tracked_workflow(context):
    """Workflow with comprehensive progress tracking"""

    # Initialize or resume progress tracking
    progress = context.get_variable("detailed_progress", {
        "workflow_id": f"wf_{int(time.time())}",
        "start_time": time.time(),
        "phases": {
            "initialization": {"status": "pending", "duration": 0},
            "data_loading": {"status": "pending", "duration": 0},
            "processing": {"status": "pending", "duration": 0},
            "validation": {"status": "pending", "duration": 0},
            "output": {"status": "pending", "duration": 0}
        },
        "current_phase": "initialization",
        "overall_progress": 0,
        "estimated_completion": None
    })

    phases = ["initialization", "data_loading", "processing", "validation", "output"]
    phase_work = [10, 25, 100, 15, 20]  # Work units per phase

    for phase_idx, phase_name in enumerate(phases):
        phase_start = time.time()
        progress["current_phase"] = phase_name
        progress["phases"][phase_name]["status"] = "running"

        print(f"🔄 Phase: {phase_name}")

        # Simulate phase work
        work_units = phase_work[phase_idx]
        for work_unit in range(work_units):
            await asyncio.sleep(0.1)

            # Calculate progress
            phase_progress = (work_unit + 1) / work_units
            overall_progress = (phase_idx + phase_progress) / len(phases)
            progress["overall_progress"] = overall_progress

            # Estimate completion time
            elapsed = time.time() - progress["start_time"]
            if overall_progress > 0:
                estimated_total = elapsed / overall_progress
                estimated_remaining = estimated_total - elapsed
                progress["estimated_completion"] = time.time() + estimated_remaining

            # Update context
            context.set_variable("detailed_progress", progress)

            # Print progress every 10 work units
            if (work_unit + 1) % 10 == 0:
                print(f"   📊 {phase_name}: {phase_progress*100:.1f}% | Overall: {overall_progress*100:.1f}%")
                if progress["estimated_completion"]:
                    remaining_mins = (progress["estimated_completion"] - time.time()) / 60
                    print(f"   ⏱️ Estimated {remaining_mins:.1f} minutes remaining")

        # Complete phase
        phase_duration = time.time() - phase_start
        progress["phases"][phase_name]["status"] = "completed"
        progress["phases"][phase_name]["duration"] = phase_duration
        print(f"✅ {phase_name} completed in {phase_duration:.1f}s")

        # Checkpoint after each phase
        checkpoint = agent.create_checkpoint()
        print(f"💾 Checkpoint saved after {phase_name}")

    # Final summary
    total_duration = time.time() - progress["start_time"]
    print(f"\\n🎉 Workflow completed in {total_duration:.1f} seconds!")
    print("📊 Phase Summary:")
    for phase, info in progress["phases"].items():
        print(f"   {phase}: {info['duration']:.1f}s")

    context.set_output("workflow_summary", {
        "total_duration": total_duration,
        "phases_completed": len(phases),
        "final_status": "success"
    })

    return None

# Add state to agent
agent.add_state("tracked_workflow", tracked_workflow)

async def main():
    await agent.run(initial_state="tracked_workflow")

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

## Quick Decision Guide

### When to Use Checkpoints?

- **Long-running workflows** (> 10 minutes): Always use checkpoints
- **Cloud workflows**: Essential for spot instances and preemptible VMs
- **Expensive computations**: Protect against losing costly work
- **Multi-step pipelines**: Save progress between major steps
- **Batch processing**: Checkpoint after each batch
- **Critical workflows**: When you can't afford to restart

### Checkpoint Frequency Guidelines

| Workflow Duration | Checkpoint Interval | Reasoning |
|-------------------|-------------------|-----------|
| **< 5 minutes** | Manual only | Short enough to restart |
| **5-30 minutes** | Every 2-5 minutes | Balance overhead vs. lost work |
| **30-120 minutes** | Every 5-10 minutes | Frequent enough to minimize loss |
| **> 2 hours** | Every 10-30 minutes | Essential for recovery |

### Storage Recommendations

| Environment | Storage Method | Best For |
|-------------|---------------|----------|
| **Local Development** | Memory checkpoints | Testing and debugging |
| **Single Machine** | File checkpoints | Local production |
| **Cloud (Single Instance)** | File + periodic S3 | Basic cloud resilience |
| **Cloud (Distributed)** | S3/Database | Full cloud resilience |

## Common Patterns

### Pattern 1: Simple Progress Saving
\`\`\`python
@state
async def simple_task(context):
    progress = context.get_variable("progress", 0)

    for i in range(progress, 100):
        # Do work
        context.set_variable("progress", i + 1)

        if (i + 1) % 10 == 0:
            agent.create_checkpoint()

# Add state to agent
agent.add_state("simple_task", simple_task)
\`\`\`

### Pattern 2: Batch Processing
\`\`\`python
@state
async def batch_processor(context):
    batch_state = context.get_variable("batch_state", {"current": 0, "total": 50})

    for batch in range(batch_state["current"], batch_state["total"]):
        process_batch(batch)
        batch_state["current"] = batch + 1
        context.set_variable("batch_state", batch_state)
        agent.create_checkpoint()  # Checkpoint after each batch

# Add state to agent
agent.add_state("batch_processor", batch_processor)
\`\`\`

### Pattern 3: Automatic Checkpointing
\`\`\`python
@state(checkpoint_interval=60.0)  # Every minute
async def auto_checkpoint_task(context):
    # Long-running work with automatic checkpoints
    for i in range(1000):
        do_work()
        await asyncio.sleep(1)  # Checkpoints happen automatically

# Add state to agent
agent.add_state("auto_checkpoint_task", auto_checkpoint_task)
\`\`\`

### Pattern 4: Phase-Based Checkpointing
\`\`\`python
@state
async def phase_processor(context):
    phases = ["load", "process", "validate", "save"]
    current_phase = context.get_variable("current_phase", 0)

    for phase_idx in range(current_phase, len(phases)):
        execute_phase(phases[phase_idx])
        context.set_variable("current_phase", phase_idx + 1)
        agent.create_checkpoint()  # Checkpoint after each phase

# Add state to agent
agent.add_state("phase_processor", phase_processor)
\`\`\`

## Tips for Beginners

1. **Start with manual checkpoints** - Create them at logical points in your workflow
2. **Use automatic checkpoints for long tasks** - Set checkpoint_interval for tasks > 30 minutes
3. **Test your recovery** - Practice restoring from checkpoints during development
4. **Track meaningful progress** - Store enough information to resume intelligently
5. **Don't over-checkpoint** - Too frequent checkpointing can slow down your workflow
6. **Use cloud storage for production** - Local files don't survive instance restarts

## What Checkpointing Protects Against

- **System crashes**: Hardware failures, out-of-memory errors
- **Cloud interruptions**: Spot instance termination, planned maintenance
- **Network failures**: Lost connections, temporary outages
- **Power failures**: Data center issues, local power problems
- **Human errors**: Accidental termination, deployment issues
- **Resource exhaustion**: Running out of disk space, memory leaks

Checkpointing transforms fragile workflows into robust, production-ready systems that can handle any interruption gracefully!
`.trim();
