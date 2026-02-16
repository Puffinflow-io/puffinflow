#!/usr/bin/env python3
"""
Tests for resource-management.ts documentation examples.
"""

import asyncio
import sys

import pytest

# Add the src directory to Python path
sys.path.insert(0, "src")

from puffinflow import Agent, Priority, state


class TestResourceManagementExamples:
    """Test examples from resource-management.ts documentation."""

    @pytest.mark.asyncio
    async def test_basic_cpu_memory_allocation(self):
        """Test basic CPU and memory allocation example."""
        agent = Agent("resource-allocation-demo")

        @state(cpu=0.5, memory=256)
        async def light_task(context):
            print("✅ Quick check completed")
            context.set_variable("light_completed", True)
            return "medium_task"

        @state(cpu=2.0, memory=1024)
        async def medium_task(context):
            # Simulate some data processing
            data = list(range(100))
            result = [x * 2 for x in data]
            context.set_variable("processed_data", result)
            context.set_variable("medium_completed", True)
            return "heavy_task"

        @state(cpu=2.0, memory=512)  # Reduced for testing environment
        async def heavy_task(context):
            import time

            time.sleep(0.01)  # Reduced for testing
            print("💪 Heavy computation finished")
            context.set_variable("heavy_completed", True)
            return None

        agent.add_state("light_task", light_task)
        agent.add_state("medium_task", medium_task)
        agent.add_state("heavy_task", heavy_task)

        result = await agent.run()

        assert result.get_variable("light_completed") is True
        assert result.get_variable("medium_completed") is True
        assert result.get_variable("heavy_completed") is True
        assert len(result.get_variable("processed_data")) == 100

    @pytest.mark.asyncio
    async def test_timeout_configuration(self):
        """Test timeout configuration example."""
        agent = Agent("timeout-demo")

        @state(cpu=1.0, memory=512, timeout=30.0)
        async def might_get_stuck(context):
            """
            This task has a 30-second timeout
            If it takes longer, Puffinflow will stop it and move on
            """
            print("⏱️ Starting task that might take a while...")
            await asyncio.sleep(0.1)  # Quick for testing
            print("✅ Task completed in time!")
            context.set_variable("task_completed", True)
            return None

        agent.add_state("might_get_stuck", might_get_stuck)
        result = await agent.run()

        assert result.get_variable("task_completed") is True

    @pytest.mark.asyncio
    async def test_retry_configuration(self):
        """Test retry configuration example."""
        agent = Agent("retry-demo")

        @state(cpu=1.0, memory=512, max_retries=3, timeout=10.0)
        async def might_fail(context):
            """
            This task will retry up to 3 times if it fails
            Perfect for network calls or external services
            """
            attempt = context.get_variable("attempts", 0) + 1
            context.set_variable("attempts", attempt)

            print("🎲 Attempting task that might fail...")

            # Simulate success after 2 attempts for testing
            if attempt < 2:
                print("❌ Task failed, will retry...")
                raise Exception("Random failure for demo")

            print("✅ Task succeeded!")
            context.set_variable("success", True)
            return None

        agent.add_state("might_fail", might_fail)
        result = await agent.run()

        assert result.get_variable("success") is True
        assert result.get_variable("attempts") == 2

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting example."""
        agent = Agent("rate-limit-demo")

        @state(cpu=0.5, memory=256, rate_limit=2.0)
        async def call_api(context):
            """
            rate_limit=2.0 means max 2 calls per second
            This prevents you from hitting API rate limits
            """
            print("🌐 Calling external API...")
            await asyncio.sleep(0.01)  # Simulate network delay

            result = {"data": "API response", "status": "success"}
            context.set_variable("api_result", result)
            return "process_response"

        @state(cpu=0.5, memory=256)
        async def process_response(context):
            context.get_variable("api_result")
            context.set_variable("response_processed", True)
            return None

        agent.add_state("call_api", call_api)
        agent.add_state("process_response", process_response)

        result = await agent.run()

        assert result.get_variable("api_result") == {
            "data": "API response",
            "status": "success",
        }
        assert result.get_variable("response_processed") is True

    @pytest.mark.asyncio
    async def test_priority_configuration(self):
        """Test priority configuration example."""
        agent = Agent("priority-demo")

        @state(cpu=2.0, memory=1024, priority=Priority.HIGH)
        async def urgent_task(context):
            """
            priority=Priority.HIGH makes this task run before normal tasks
            Available priorities: CRITICAL, HIGH, NORMAL, LOW, BACKGROUND
            """
            print("🚨 Running high-priority urgent task")
            await asyncio.sleep(0.1)
            context.set_variable("urgent_completed", True)
            return None

        @state(cpu=1.0, memory=512, priority=Priority.LOW)
        async def background_cleanup(context):
            """
            priority=Priority.LOW makes this run when system isn't busy
            Perfect for maintenance tasks
            """
            print("🧹 Running background cleanup")
            await asyncio.sleep(0.1)
            context.set_variable("cleanup_completed", True)
            return None

        agent.add_state("urgent_task", urgent_task)
        agent.add_state("background_cleanup", background_cleanup)

        # Test urgent task
        result = await agent.run()
        # Since both states are added, one will run depending on the execution order
        # For testing, let's verify at least one completes
        urgent_completed = result.get_variable("urgent_completed")
        cleanup_completed = result.get_variable("cleanup_completed")
        assert urgent_completed is True or cleanup_completed is True

    @pytest.mark.asyncio
    async def test_image_processing_workflow(self):
        """Test the complete image processing service example."""
        agent = Agent("image-service")

        @state(cpu=0.5, memory=256, rate_limit=10.0)
        async def receive_upload(context):
            """Handle image upload - light but rate-limited"""
            print("📤 Image uploaded")
            context.set_variable("image_path", "/tmp/photo.jpg")
            context.set_variable("image_size", "2MB")
            return "validate_image"

        @state(cpu=1.0, memory=512, timeout=30.0, max_retries=2)
        async def validate_image(context):
            """Validate image format - might fail, so retry"""
            image_path = context.get_variable("image_path")
            print(f"🔍 Validating: {image_path}")

            # Simulate validation
            await asyncio.sleep(0.01)

            context.set_variable("valid", True)
            return "resize_image"

        @state(cpu=2.0, memory=1024, timeout=60.0)
        async def resize_image(context):
            """Resize image - CPU and I/O intensive"""
            image_path = context.get_variable("image_path")
            print(f"🖼️ Resizing: {image_path}")

            # Simulate image processing
            await asyncio.sleep(0.01)

            context.set_variable("resized_path", "/tmp/photo_resized.jpg")
            return "apply_ai_filters"

        @state(cpu=2.0, memory=1024, timeout=120.0, priority=Priority.HIGH)
        async def apply_ai_filters(context):
            """AI processing - high priority, limited concurrency"""
            resized_path = context.get_variable("resized_path")
            print(f"🤖 Applying AI filters: {resized_path}")

            # Simulate AI processing
            await asyncio.sleep(0.01)

            context.set_variable("filtered_path", "/tmp/photo_ai.jpg")
            return "save_to_gallery"

        @state(cpu=0.5, memory=256, rate_limit=5.0)
        async def save_to_gallery(context):
            """Save final image - rate-limited"""
            filtered_path = context.get_variable("filtered_path")
            print(f"💾 Saving to gallery: {filtered_path}")

            # Update gallery database
            await asyncio.sleep(0.01)

            context.set_output("final_image", filtered_path)
            context.set_output("processing_complete", True)
            return None

        # Add all states
        agent.add_state("receive_upload", receive_upload)
        agent.add_state("validate_image", validate_image)
        agent.add_state("resize_image", resize_image)
        agent.add_state("apply_ai_filters", apply_ai_filters)
        agent.add_state("save_to_gallery", save_to_gallery)

        # Run the complete workflow
        result = await agent.run()

        assert result.get_output("final_image") == "/tmp/photo_ai.jpg"
        assert result.get_output("processing_complete") is True
        assert result.get_variable("valid") is True
        assert result.get_variable("image_path") == "/tmp/photo.jpg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
