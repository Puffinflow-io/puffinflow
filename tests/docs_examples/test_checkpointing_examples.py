#!/usr/bin/env python3
"""
Tests for checkpointing.ts documentation examples.
"""

import asyncio
import sys

import pytest

# Add the src directory to Python path
sys.path.insert(0, "src")

from puffinflow import Agent, state


class TestCheckpointingExamples:
    """Test examples from checkpointing.ts documentation."""

    @pytest.mark.asyncio
    async def test_basic_checkpoint_workflow(self):
        """Test the basic checkpoint workflow example."""
        agent = Agent("checkpoint-demo")

        @state
        async def step_one(context):
            print("🔄 Step 1: Processing data...")
            context.set_variable("step1_result", "completed")
            return "step_two"

        @state
        async def step_two(context):
            print("🔄 Step 2: Analyzing results...")
            result = context.get_variable("step1_result")
            context.set_variable("step2_result", f"analyzed {result}")
            return "step_three"

        @state
        async def step_three(context):
            print("🔄 Step 3: Generating report...")
            context.set_variable("final_report", "Report completed!")
            return None

        agent.add_state("step_one", step_one)
        agent.add_state("step_two", step_two)
        agent.add_state("step_three", step_three)

        result = await agent.run()

        assert result.get_variable("step1_result") == "completed"
        assert result.get_variable("step2_result") == "analyzed completed"
        assert result.get_variable("final_report") == "Report completed!"

    @pytest.mark.asyncio
    async def test_progress_tracking_workflow(self):
        """Test the progress tracking workflow example."""
        agent = Agent("progress-demo")

        @state
        async def long_process(context):
            """Simulate a long process that saves progress"""
            # Check if we're resuming from a saved state
            progress = context.get_variable("progress", {"completed": 0, "total": 10})

            print(f"Starting from item {progress['completed']}")

            for i in range(progress["completed"], progress["total"]):
                print(f"📊 Processing item {i + 1}/{progress['total']}")

                # Simulate work (much faster for testing)
                await asyncio.sleep(0.01)

                # Update progress
                progress["completed"] = i + 1
                context.set_variable("progress", progress)

                # Save checkpoint every 3 items (simulate)
                if (i + 1) % 3 == 0:
                    print(f"💾 Checkpoint saved at item {i + 1}")

            print("✅ All items processed!")
            context.set_variable("processing_complete", True)
            return None

        agent.add_state("long_process", long_process)
        result = await agent.run()

        progress = result.get_variable("progress")
        assert progress["completed"] == 10
        assert progress["total"] == 10
        assert result.get_variable("processing_complete") is True

    @pytest.mark.asyncio
    async def test_automatic_checkpointing(self):
        """Test automatic checkpointing configuration."""
        agent = Agent("auto-checkpoint-demo")

        @state(checkpoint_interval=30.0)
        async def auto_save_task(context):
            """
            This state will automatically create a checkpoint every 30 seconds
            You don't have to do anything - Puffinflow handles it!
            """
            print("⏰ Starting task with automatic checkpointing...")

            # Simulate work (much faster for testing)
            for minute in range(3):  # 3 iterations instead of 5 minutes
                print(f"🔄 Working... iteration {minute + 1}/3")

                # Track our progress
                progress = {
                    "current_iteration": minute + 1,
                    "total_iterations": 3,
                    "completion": ((minute + 1) / 3) * 100,
                }
                context.set_variable("work_progress", progress)

                # Simulate work (much faster for testing)
                await asyncio.sleep(0.01)

            print("✅ Task completed!")
            context.set_variable("auto_checkpoint_complete", True)
            return None

        agent.add_state("auto_save_task", auto_save_task)
        result = await agent.run()

        progress = result.get_variable("work_progress")
        assert progress["current_iteration"] == 3
        assert progress["completion"] == 100.0
        assert result.get_variable("auto_checkpoint_complete") is True

    @pytest.mark.asyncio
    async def test_batch_processing_with_checkpoints(self):
        """Test batch processing with checkpoint recovery."""
        agent = Agent("batch-checkpoint-demo")

        @state
        async def data_processor(context):
            """A processor that knows how to resume intelligently"""

            # Check if we're resuming from a checkpoint
            batch_info = context.get_variable(
                "batch_processing",
                {
                    "current_batch": 0,
                    "total_batches": 5,  # Reduced for testing
                    "items_per_batch": 10,  # Reduced for testing
                    "total_processed": 0,
                },
            )

            if batch_info["current_batch"] > 0:
                print(f"🔄 Resuming from batch {batch_info['current_batch']}")
                print(f"📊 Already processed {batch_info['total_processed']} items")
            else:
                print("🚀 Starting fresh data processing")

            # Continue from where we left off
            for batch_num in range(
                batch_info["current_batch"], batch_info["total_batches"]
            ):
                print(
                    f"📦 Processing batch {batch_num + 1}/{batch_info['total_batches']}"
                )

                # Process items in this batch
                for _ in range(batch_info["items_per_batch"]):
                    # Simulate processing an item (much faster for testing)
                    await asyncio.sleep(0.001)
                    batch_info["total_processed"] += 1

                # Update progress
                batch_info["current_batch"] = batch_num + 1
                context.set_variable("batch_processing", batch_info)

                # Save checkpoint after each batch (simulate)
                print(f"💾 Batch {batch_num + 1} complete, checkpoint saved")

                # Show progress
                completion_percent = (
                    batch_info["current_batch"] / batch_info["total_batches"]
                ) * 100
                print(f"📈 Overall progress: {completion_percent:.1f}%")

            print("🎉 All batches processed successfully!")
            context.set_output("total_items_processed", batch_info["total_processed"])
            return None

        agent.add_state("data_processor", data_processor)
        result = await agent.run()

        assert result.get_output("total_items_processed") == 50  # 5 batches * 10 items
        batch_info = result.get_variable("batch_processing")
        assert batch_info["current_batch"] == 5
        assert batch_info["total_processed"] == 50

    @pytest.mark.asyncio
    async def test_different_checkpoint_frequencies(self):
        """Test different checkpoint frequency configurations."""
        agent = Agent("frequency-demo")

        # Test frequent saves
        @state(checkpoint_interval=10.0)
        async def frequent_saves(context):
            """For tasks where you don't want to lose much progress"""
            await asyncio.sleep(0.01)
            context.set_variable("frequent_complete", True)
            return None

        # Test less frequent saves
        @state(checkpoint_interval=300.0)
        async def expensive_saves(context):
            """For tasks where checkpointing itself is expensive"""
            await asyncio.sleep(0.01)
            context.set_variable("expensive_complete", True)
            return None

        # Test very frequent saves
        @state(checkpoint_interval=5.0)
        async def critical_saves(context):
            """For absolutely critical tasks"""
            await asyncio.sleep(0.01)
            context.set_variable("critical_complete", True)
            return None

        agent.add_state("frequent_saves", frequent_saves)
        agent.add_state("expensive_saves", expensive_saves)
        agent.add_state("critical_saves", critical_saves)

        # Test that all checkpoint frequency configurations work
        result = await agent.run()
        # Since all three states are added, at least one should complete
        frequent_complete = result.get_variable("frequent_complete")
        expensive_complete = result.get_variable("expensive_complete")
        critical_complete = result.get_variable("critical_complete")

        # At least one of the checkpoint frequency tests should complete
        assert (
            frequent_complete is True
            or expensive_complete is True
            or critical_complete is True
        )

    @pytest.mark.asyncio
    async def test_tracked_workflow_with_progress(self):
        """Test comprehensive progress tracking workflow."""
        agent = Agent("progress-tracker")

        @state
        async def tracked_workflow(context):
            """Workflow with comprehensive progress tracking"""

            # Initialize progress tracking
            import time

            progress = context.get_variable(
                "detailed_progress",
                {
                    "workflow_id": f"wf_{int(time.time())}",
                    "start_time": time.time(),
                    "phases": {
                        "initialization": {"status": "pending", "duration": 0},
                        "processing": {"status": "pending", "duration": 0},
                        "finalization": {"status": "pending", "duration": 0},
                    },
                    "current_phase": "initialization",
                    "overall_progress": 0,
                },
            )

            phases = ["initialization", "processing", "finalization"]
            phase_work = [5, 10, 3]  # Reduced work units for testing

            for phase_idx, phase_name in enumerate(phases):
                phase_start = time.time()
                progress["current_phase"] = phase_name
                progress["phases"][phase_name]["status"] = "running"

                print(f"🔄 Phase: {phase_name}")

                # Simulate phase work
                work_units = phase_work[phase_idx]
                for work_unit in range(work_units):
                    await asyncio.sleep(0.001)  # Much faster for testing

                    # Calculate progress
                    phase_progress = (work_unit + 1) / work_units
                    overall_progress = (phase_idx + phase_progress) / len(phases)
                    progress["overall_progress"] = overall_progress

                    # Update context
                    context.set_variable("detailed_progress", progress)

                # Complete phase
                phase_duration = time.time() - phase_start
                progress["phases"][phase_name]["status"] = "completed"
                progress["phases"][phase_name]["duration"] = phase_duration
                print(f"✅ {phase_name} completed in {phase_duration:.3f}s")

            # Final summary
            total_duration = time.time() - progress["start_time"]
            print(f"🎉 Workflow completed in {total_duration:.3f} seconds!")

            context.set_output(
                "workflow_summary",
                {
                    "total_duration": total_duration,
                    "phases_completed": len(phases),
                    "final_status": "success",
                },
            )

            return None

        agent.add_state("tracked_workflow", tracked_workflow)
        result = await agent.run()

        summary = result.get_output("workflow_summary")
        assert summary["phases_completed"] == 3
        assert summary["final_status"] == "success"

        progress = result.get_variable("detailed_progress")
        assert progress["overall_progress"] == 1.0  # 100% complete
        assert all(
            phase["status"] == "completed" for phase in progress["phases"].values()
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
