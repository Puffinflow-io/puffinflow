
export const checkpointingMarkdown = `# Checkpoints & State Persistence

One of Puffinflow's most powerful features is the ability to save workflow progress and resume execution exactly where you left off. This is essential for long-running workflows, handling system failures, and managing workflows in cloud environments.

## Why Use Checkpoints?

Checkpoints solve critical problems in workflow orchestration:

- Failure Recovery: Resume workflows after crashes or interruptions
- Cloud Resilience: Handle preemptible instances and spot interruptions
- Long-Running Tasks: Save progress in workflows that take hours or days
- Cost Optimization: Use cheaper, interruptible cloud resources safely
- Development: Test and debug workflows without losing progress

## Overview

| Feature | Use Case | Benefit |
|---------|----------|---------|
| **Automatic Checkpoints** | System crashes, interruptions | Zero-loss recovery |
| **Manual Checkpoints** | Strategic save points | Controlled persistence |
| **State Restoration** | Resume after failure | Continue from exact point |
| **Progress Tracking** | Long-running workflows | Monitor completion |
| **Cloud Resilience** | Preemptible instances | Cost-effective execution |

---

## Quick Start

### Creating Your First Checkpoint

\`\`\`python
import asyncio
from puffinflow import Agent
from puffinflow.decorators import state

agent = Agent("my-workflow")

@state
async def process_data(context):
    # Do some work
    context.set_variable("processed_items", 100)
    print("Data processing complete")

# Add state to agent
agent.add_state("process_data", process_data)

async def main():
    # Run workflow
    await agent.run()

    # Create checkpoint
    checkpoint = agent.create_checkpoint()
    print(f"Checkpoint created with {len(checkpoint.completed_states)} completed states")

    # Later: restore from checkpoint
    agent.reset()  # Reset agent state
    await agent.restore_from_checkpoint(checkpoint)
    print("Workflow restored from checkpoint!")

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

---

## Basic Checkpoint Usage

### Manual Checkpointing

\`\`\`python
import asyncio
from puffinflow import Agent
from puffinflow.decorators import state

agent = Agent("data-pipeline")

@state
async def extract_data(context):
    print("Extracting data...")
    # Simulate data extraction
    data = {"records": 1000, "source": "database"}
    context.set_variable("extracted_data", data)
    return "transform_data"

@state
async def transform_data(context):
    print("Transforming data...")
    data = context.get_variable("extracted_data")
    # Transform the data
    transformed = {"processed_records": data["records"], "status": "cleaned"}
    context.set_variable("transformed_data", transformed)
    return "load_data"

@state
async def load_data(context):
    print("Loading data...")
    data = context.get_variable("transformed_data")
    # Load to destination
    context.set_variable("load_complete", True)
    print(f"Loaded {data['processed_records']} records")

# Build workflow
agent.add_state("extract_data", extract_data)
agent.add_state("transform_data", transform_data, dependencies=["extract_data"])
agent.add_state("load_data", load_data, dependencies=["transform_data"])

async def main():
    # Run with checkpointing
    await agent.run()

    # Save progress
    checkpoint = agent.create_checkpoint()

    # Simulate failure and recovery
    new_agent = Agent("data-pipeline")
    new_agent.add_state("extract_data", extract_data)
    new_agent.add_state("transform_data", transform_data, dependencies=["extract_data"])
    new_agent.add_state("load_data", load_data, dependencies=["transform_data"])

    # Restore and continue
    await new_agent.restore_from_checkpoint(checkpoint)
    print("Pipeline restored successfully!")

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

---

## Automatic Checkpointing

For long-running operations, use automatic checkpointing to save progress at regular intervals:

\`\`\`python
@state(checkpoint_interval=30.0)  # Checkpoint every 30 seconds
async def long_running_task(context):
    print("Starting long-running analysis...")

    total_steps = 10
    for step in range(total_steps):
        print(f"   Processing step {step + 1}/{total_steps}")

        # Simulate work
        await asyncio.sleep(5)

        # Track progress
        progress = {
            "current_step": step + 1,
            "total_steps": total_steps,
            "completion_percentage": ((step + 1) / total_steps) * 100
        }
        context.set_variable("analysis_progress", progress)

        # Automatic checkpoint happens every 30 seconds
        print(f"   Step {step + 1} complete ({progress['completion_percentage']:.1f}%)")

    context.set_variable("analysis_complete", True)
    print("Analysis complete!")
\`\`\`

---

## Smart Resumption

Design workflows that intelligently resume from where they left off:

\`\`\`python
@state
async def smart_processor(context):
    """Processor that knows how to resume from any point"""

    # Check if we're resuming
    progress = context.get_variable("processing_progress")

    if progress:
        print(f"Resuming from item {progress['last_processed']}")
        start_from = progress["last_processed"] + 1
    else:
        print("Starting fresh processing")
        start_from = 0
        progress = {"last_processed": -1, "total_items": 100}

    # Process items
    for i in range(start_from, progress["total_items"]):
        print(f"   Processing item {i + 1}")

        # Simulate work
        await asyncio.sleep(0.1)

        # Update progress
        progress["last_processed"] = i
        context.set_variable("processing_progress", progress)

        # Manual checkpoint every 10 items
        if (i + 1) % 10 == 0:
            checkpoint = agent.create_checkpoint()
            print(f"   Checkpoint saved at item {i + 1}")

    print("Processing complete!")
\`\`\`

---

## Cloud-Resilient Workflows

Handle cloud interruptions gracefully with persistent checkpointing:

\`\`\`python
import signal
import json
from pathlib import Path

class CloudResilienceManager:
    def __init__(self, agent, checkpoint_file="workflow.checkpoint"):
        self.agent = agent
        self.checkpoint_file = Path(checkpoint_file)

        # Handle cloud interruption signals
        signal.signal(signal.SIGTERM, self.handle_interruption)

    def handle_interruption(self, signum, frame):
        """Save checkpoint before cloud instance terminates"""
        print(f"Cloud interruption detected (signal {signum})")
        print("Saving checkpoint...")

        checkpoint = self.agent.create_checkpoint()
        self.save_to_file(checkpoint)
        print("Checkpoint saved, ready for restart")

    def save_to_file(self, checkpoint):
        """Save checkpoint to persistent storage"""
        data = {
            "timestamp": checkpoint.timestamp,
            "agent_name": checkpoint.agent_name,
            "completed_states": list(checkpoint.completed_states),
            "shared_state": checkpoint.shared_state
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self):
        """Load checkpoint from persistent storage"""
        if not self.checkpoint_file.exists():
            return None

        with open(self.checkpoint_file, 'r') as f:
            return json.load(f)

# Usage
async def main():
    agent = Agent("cloud-workflow")
    manager = CloudResilienceManager(agent)

    # Try to resume from previous checkpoint
    saved_state = manager.load_from_file()
    if saved_state:
        print("Resuming from previous run...")
        # Restore logic here
    else:
        print("Starting new workflow...")

    # Add your workflow states
    # ...

    try:
        await agent.run()
        # Clean up checkpoint on success
        if manager.checkpoint_file.exists():
            manager.checkpoint_file.unlink()
    except KeyboardInterrupt:
        print("Saving checkpoint before exit...")
        checkpoint = agent.create_checkpoint()
        manager.save_to_file(checkpoint)
\`\`\`

---

## Best Practices

### DO

\`\`\`python
# Store incremental progress
@state
async def good_processor(context):
    progress = context.get_variable("progress", {"completed": 0, "total": 1000})

    for i in range(progress["completed"], progress["total"]):
        # Do work
        await process_item(i)

        # Update progress frequently
        progress["completed"] = i + 1
        context.set_variable("progress", progress)

        # Checkpoint at logical intervals
        if i % 100 == 0:
            checkpoint = agent.create_checkpoint()

# Use descriptive checkpoint state
@state
async def descriptive_state(context):
    state_info = {
        "phase": "data_processing",
        "batch_number": 5,
        "processed_records": 1500,
        "next_action": "validate_results",
        "estimated_completion": "2024-01-15T10:30:00Z"
    }
    context.set_variable("processing_state", state_info)
\`\`\`

### AVOID

\`\`\`python
# All-or-nothing processing
@state
async def bad_processor(context):
    results = []
    for i in range(1000):  # No intermediate checkpoints
        results.append(await process_item(i))
    context.set_variable("results", results)  # Only saves at the end

# Ambiguous state
@state
async def unclear_state(context):
    context.set_variable("status", "running")  # Not helpful for resumption
    context.set_variable("count", 42)         # What does this count?
\`\`\`

---

## Configuration Options

### Checkpoint Intervals

\`\`\`python
# Different checkpoint strategies
@state(checkpoint_interval=10.0)   # Every 10 seconds
async def frequent_checkpoints(context): pass

@state(checkpoint_interval=300.0)  # Every 5 minutes
async def moderate_checkpoints(context): pass

# Manual checkpointing at logical points
@state
async def manual_checkpoints(context):
    for batch in get_batches():
        process_batch(batch)
        if batch.is_milestone():
            checkpoint = agent.create_checkpoint()
\`\`\`

### Checkpoint Conditions

\`\`\`python
@state
async def conditional_checkpoints(context):
    items_processed = 0

    for item in get_items():
        process_item(item)
        items_processed += 1

        # Checkpoint based on conditions
        if items_processed % 100 == 0:  # Every 100 items
            checkpoint = agent.create_checkpoint()

        if time.time() % 300 == 0:  # Every 5 minutes
            checkpoint = agent.create_checkpoint()
\`\`\`

---

## Quick Reference

### Core Methods

\`\`\`python
# Create checkpoint
checkpoint = agent.create_checkpoint()

# Restore from checkpoint
await agent.restore_from_checkpoint(checkpoint)

# Automatic checkpointing
@state(checkpoint_interval=30.0)
async def auto_checkpoint_state(context): pass
\`\`\`

### Progress Tracking

\`\`\`python
# Store progress
context.set_variable("progress", {
    "phase": "processing",
    "completed": 150,
    "total": 1000,
    "start_time": time.time()
})

# Resume from progress
progress = context.get_variable("progress")
if progress:
    start_from = progress["completed"]
\`\`\`

### File Persistence

\`\`\`python
# Save to file
import json
checkpoint_data = {
    "timestamp": checkpoint.timestamp,
    "completed_states": list(checkpoint.completed_states),
    "shared_state": checkpoint.shared_state
}
with open("checkpoint.json", "w") as f:
    json.dump(checkpoint_data, f)

# Load from file
with open("checkpoint.json", "r") as f:
    data = json.load(f)
\`\`\`

---

## Common Patterns

### Batch Processing

\`\`\`python
@state
async def batch_processor(context):
    batch_state = context.get_variable("batch_state", {
        "current_batch": 0,
        "total_batches": 10,
        "processed_items": 0
    })

    for batch_id in range(batch_state["current_batch"], batch_state["total_batches"]):
        items = get_batch(batch_id)
        for item in items:
            process_item(item)
            batch_state["processed_items"] += 1

        batch_state["current_batch"] = batch_id + 1
        context.set_variable("batch_state", batch_state)

        # Checkpoint after each batch
        checkpoint = agent.create_checkpoint()
        print(f"Batch {batch_id + 1} complete, checkpoint saved")
\`\`\`

### Time-Based Processing

\`\`\`python
@state
async def time_based_processor(context):
    start_time = context.get_variable("start_time", time.time())
    duration = 3600  # 1 hour

    while time.time() - start_time < duration:
        # Do work
        await process_chunk()

        # Update progress
        elapsed = time.time() - start_time
        progress = (elapsed / duration) * 100
        context.set_variable("time_progress", {
            "elapsed": elapsed,
            "progress_percent": progress,
            "estimated_remaining": duration - elapsed
        })

        await asyncio.sleep(10)  # Process every 10 seconds
\`\`\`

---

With Puffinflow's checkpoint system, your workflows become resilient, cost-effective, and production-ready!
`.trim();
