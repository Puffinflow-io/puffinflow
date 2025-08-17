#!/usr/bin/env python3
"""
Comprehensive tests for all documentation examples to ensure they work exactly as shown.
"""

import asyncio
import sys

import pytest
from pydantic import BaseModel

# Add the src directory to Python path
sys.path.insert(0, "src")

from puffinflow import Agent, state
from puffinflow.core.agent.state import RetryPolicy


class TestErrorHandlingExamples:
    """Test examples from error-handling.ts documentation."""

    @pytest.mark.asyncio
    async def test_simple_retries_example(self):
        """Test the simple retries example from docs."""
        agent = Agent("retry-demo")

        @state(max_retries=3)
        async def call_flaky_api(context):
            """This API fails sometimes, but usually works if you try again"""
            attempt = context.get_variable("attempts", 0) + 1
            context.set_variable("attempts", attempt)

            print(f"🌐 Calling API (attempt {attempt})...")

            # Simulate success after a few attempts for testing
            if attempt < 3:
                print(f"❌ API failed on attempt {attempt}")
                raise Exception("API temporarily unavailable")

            print(f"✅ API succeeded on attempt {attempt}!")
            context.set_variable("api_data", {"result": "success"})
            return None

        agent.add_state("call_flaky_api", call_flaky_api)

        result = await agent.run()

        assert result.get_variable("api_data") == {"result": "success"}
        assert result.get_variable("attempts") >= 2  # Should succeed after retries

    @pytest.mark.asyncio
    async def test_timeout_example(self):
        """Test the timeout example from docs."""
        agent = Agent("timeout-demo")

        @state(timeout=10.0, max_retries=2)
        async def might_hang(context):
            """
            This will timeout after 10 seconds if it gets stuck
            If it times out, it will retry up to 2 more times
            """
            print("⏱️ Starting operation that might hang...")

            # Simulate work that completes quickly for test
            await asyncio.sleep(0.1)

            print("✅ Operation completed!")
            context.set_variable("completed", True)
            return None

        agent.add_state("might_hang", might_hang)
        result = await agent.run()

        assert result.get_variable("completed") is True

    @pytest.mark.asyncio
    async def test_exponential_backoff_example(self):
        """Test the exponential backoff example from docs."""
        agent = Agent("backoff-demo")

        # Smart retry policy that waits longer each time
        smart_retry = RetryPolicy(
            max_retries=4,
            initial_delay=0.01,  # Reduced for testing
            exponential_base=2.0,  # Double each time
            jitter=True,  # Add randomness to prevent thundering herd
        )

        @state
        async def overloaded_service(context):
            """
            This service gets overwhelmed easily, so we:
            - Give it more time to recover between retries
            - Add randomness so multiple clients don't retry at once
            """
            attempt = context.get_variable("service_attempts", 0) + 1
            context.set_variable("service_attempts", attempt)

            print(f"🔄 Calling overloaded service (attempt {attempt})...")

            # Simulate a service that's more likely to work with fewer concurrent calls
            if attempt < 3:  # Fail first 2 attempts for testing
                print(f"❌ Service overloaded (attempt {attempt})")
                raise Exception("Service temporarily overloaded")

            print("✅ Service call succeeded!")
            context.set_variable("service_result", {"status": "completed"})
            return None

        agent.add_state(
            "overloaded_service", overloaded_service, retry_policy=smart_retry
        )
        result = await agent.run()

        assert result.get_variable("service_result") == {"status": "completed"}


class TestResourceManagementExamples:
    """Test examples from resource-management.ts documentation."""

    @pytest.mark.asyncio
    async def test_basic_cpu_memory_example(self):
        """Test the basic CPU and memory example from docs."""
        agent = Agent("resource-demo")

        @state(cpu=0.5, memory=256)
        async def light_task(context):
            print("✅ Quick check completed")
            context.set_variable("light_done", True)
            return "medium_task"

        @state(cpu=2.0, memory=1024)
        async def medium_task(context):
            # Simulate some data processing
            data = list(range(100))  # Smaller for testing
            result = [x * 2 for x in data]
            context.set_variable("processed_data", result)
            context.set_variable("medium_done", True)
            return "heavy_task"

        @state(cpu=2.0, memory=512)  # Reduced for testing environment
        async def heavy_task(context):
            import time

            time.sleep(0.01)  # Reduced for testing
            print("💪 Heavy computation finished")
            context.set_variable("heavy_done", True)
            return None

        agent.add_state("light_task", light_task)
        agent.add_state("medium_task", medium_task)
        agent.add_state("heavy_task", heavy_task)

        result = await agent.run()

        assert result.get_variable("light_done") is True
        assert result.get_variable("medium_done") is True
        assert result.get_variable("heavy_done") is True
        assert len(result.get_variable("processed_data")) == 100

    @pytest.mark.asyncio
    async def test_timeout_resource_example(self):
        """Test the timeout resource example from docs."""
        agent = Agent("timeout-resource-demo")

        @state(cpu=1.0, memory=512, timeout=30.0)
        async def might_get_stuck(context):
            """
            This task has a 30-second timeout
            If it takes longer, Puffinflow will stop it and move on
            """
            print("⏱️ Starting task that might take a while...")
            await asyncio.sleep(0.1)  # Reduced for testing
            print("✅ Task completed in time!")
            context.set_variable("completed", True)
            return None

        agent.add_state("might_get_stuck", might_get_stuck)
        result = await agent.run()

        assert result.get_variable("completed") is True

    @pytest.mark.asyncio
    async def test_retry_resource_example(self):
        """Test the retry resource example from docs."""
        agent = Agent("retry-resource-demo")

        @state(cpu=1.0, memory=512, max_retries=3, timeout=10.0)
        async def might_fail(context):
            """
            This task will retry up to 3 times if it fails
            Perfect for network calls or external services
            """
            attempt = context.get_variable("attempts", 0) + 1
            context.set_variable("attempts", attempt)

            print("🎲 Attempting task that might fail...")

            # Simulate a task that fails sometimes - succeed on 2nd attempt
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
    async def test_rate_limiting_example(self):
        """Test the rate limiting example from docs."""
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
            context.set_variable("processed", True)
            return None

        agent.add_state("call_api", call_api)
        agent.add_state("process_response", process_response)

        result = await agent.run()

        assert result.get_variable("api_result") == {
            "data": "API response",
            "status": "success",
        }
        assert result.get_variable("processed") is True


class TestContextAndDataExamples:
    """Test examples from context-and-data.ts documentation."""

    @pytest.mark.asyncio
    async def test_basic_data_sharing_example(self):
        """Test the basic data sharing example from docs."""
        agent = Agent("context-demo")

        @state
        async def fetch_user(context):
            user_data = {"id": 123, "name": "Alice", "email": "alice@example.com"}
            context.set_variable("user", user_data)
            context.set_variable("timestamp", "2025-01-15T10:30:00Z")
            return "process_user"

        @state
        async def process_user(context):
            user = context.get_variable("user")
            timestamp = context.get_variable("timestamp")

            # Use default values for optional data
            context.get_variable("settings", {"theme": "default"})

            print(f"Processing {user['name']} at {timestamp}")
            context.set_variable("processed", True)
            return "send_welcome"

        @state
        async def send_welcome(context):
            context.get_variable("user")
            context.set_variable("welcome_sent", True)
            return None

        agent.add_state("fetch_user", fetch_user)
        agent.add_state("process_user", process_user)
        agent.add_state("send_welcome", send_welcome)

        result = await agent.run()

        assert result.get_variable("user") == {
            "id": 123,
            "name": "Alice",
            "email": "alice@example.com",
        }
        assert result.get_variable("timestamp") == "2025-01-15T10:30:00Z"
        assert result.get_variable("processed") is True
        assert result.get_variable("welcome_sent") is True

    @pytest.mark.asyncio
    async def test_type_safe_variables_example(self):
        """Test the type-safe variables example from docs."""
        agent = Agent("type-safe-demo")

        @state
        async def initialize(context):
            context.set_typed_variable("user_count", 100)  # Locked to int
            context.set_typed_variable("avg_score", 85.5)  # Locked to float
            return "process"

        @state
        async def process(context):
            context.set_typed_variable("user_count", 150)  # Works

            count = context.get_typed_variable("user_count")
            score = context.get_typed_variable("avg_score")

            print(f"Processing {count} users with avg score {score}")
            context.set_variable("processing_done", True)
            return None

        agent.add_state("initialize", initialize)
        agent.add_state("process", process)

        result = await agent.run()

        assert result.get_typed_variable("user_count") == 150
        assert result.get_typed_variable("avg_score") == 85.5
        assert result.get_variable("processing_done") is True

    @pytest.mark.asyncio
    async def test_validated_data_example(self):
        """Test the validated data example from docs."""

        class User(BaseModel):
            id: int
            name: str
            email: str  # Using str instead of EmailStr for testing
            age: int

        agent = Agent("validation-demo")

        @state
        async def create_user(context):
            user = User(id=123, name="Alice", email="alice@example.com", age=28)
            context.set_validated_data("user", user)
            return "update_user"

        @state
        async def update_user(context):
            user = context.get_validated_data("user", User)
            user.age = 29
            context.set_validated_data("user", user)  # Re-validates automatically
            context.set_variable("updated", True)
            return None

        agent.add_state("create_user", create_user)
        agent.add_state("update_user", update_user)

        result = await agent.run()

        user = result.get_validated_data("user", User)
        assert user.id == 123
        assert user.name == "Alice"
        assert user.age == 29
        assert result.get_variable("updated") is True

    @pytest.mark.asyncio
    async def test_constants_and_secrets_example(self):
        """Test the constants and secrets example from docs."""
        agent = Agent("config-demo")

        @state
        async def setup(context):
            # Configuration that won't change
            context.set_constant("api_url", "https://api.example.com")
            context.set_constant("max_retries", 3)

            # Sensitive data stored securely
            context.set_secret("api_key", "sk-1234567890abcdef")
            context.set_secret("db_password", "super_secure_password")
            return "make_request"

        @state
        async def make_request(context):
            url = context.get_constant("api_url")
            api_key = context.get_secret("api_key")

            # Don't log real secrets!
            print(f"Making request to {url} with key {api_key[:8]}...")

            context.set_variable("request_made", True)
            return None

        agent.add_state("setup", setup)
        agent.add_state("make_request", make_request)

        result = await agent.run()

        assert result.get_constant("api_url") == "https://api.example.com"
        assert result.get_constant("max_retries") == 3
        assert result.get_secret("api_key") == "sk-1234567890abcdef"
        assert result.get_variable("request_made") is True

    @pytest.mark.asyncio
    async def test_cached_data_example(self):
        """Test the cached data example from docs."""
        agent = Agent("cache-demo")

        @state
        async def cache_session(context):
            context.set_cached("user_session", {"user_id": 123}, ttl=300)  # 5 minutes
            context.set_cached("temp_token", "abc123", ttl=60)  # 1 minute
            return "use_cache"

        @state
        async def use_cache(context):
            session = context.get_cached("user_session", default="EXPIRED")
            token = context.get_cached("temp_token", default="EXPIRED")

            context.set_variable("session_valid", session != "EXPIRED")
            context.set_variable("token_valid", token != "EXPIRED")

            if session != "EXPIRED":
                print(f"Active session: {session}")
            else:
                print("Session expired, need to re-authenticate")
            return None

        agent.add_state("cache_session", cache_session)
        agent.add_state("use_cache", use_cache)

        result = await agent.run()

        # Should be valid immediately after caching
        assert result.get_variable("session_valid") is True
        assert result.get_variable("token_valid") is True

    @pytest.mark.asyncio
    async def test_workflow_outputs_example(self):
        """Test the workflow outputs example from docs."""
        agent = Agent("output-demo")

        @state
        async def calculate_metrics(context):
            orders = [{"amount": 100}, {"amount": 200}, {"amount": 150}]
            total = sum(order["amount"] for order in orders)

            # Mark as final outputs
            context.set_output("total_revenue", total)
            context.set_output("order_count", len(orders))
            context.set_output("avg_order_value", total / len(orders))
            return "send_report"

        @state
        async def send_report(context):
            revenue = context.get_output("total_revenue")
            count = context.get_output("order_count")
            avg = context.get_output("avg_order_value")

            print(f"Report: ${revenue} revenue from {count} orders (avg: ${avg:.2f})")
            context.set_variable("report_sent", True)
            return None

        agent.add_state("calculate_metrics", calculate_metrics)
        agent.add_state("send_report", send_report)

        result = await agent.run()

        assert result.get_output("total_revenue") == 450
        assert result.get_output("order_count") == 3
        assert result.get_output("avg_order_value") == 150.0
        assert result.get_variable("report_sent") is True

    @pytest.mark.asyncio
    async def test_complete_context_example(self):
        """Test the complete example from context-and-data docs."""

        class Order(BaseModel):
            id: int
            total: float
            customer_email: str

        agent = Agent("order-processor")

        @state
        async def setup(context):
            context.set_constant("tax_rate", 0.08)
            context.set_secret("payment_key", "pk_123456")
            return "process_order"

        @state
        async def process_order(context):
            # Validated order data
            order = Order(id=123, total=99.99, customer_email="user@example.com")
            context.set_validated_data("order", order)

            # Cache session temporarily
            context.set_cached("session", {"order_id": order.id}, ttl=3600)

            # Type-safe tracking
            context.set_typed_variable("amount_charged", order.total)
            return "finalize"

        @state
        async def finalize(context):
            order = context.get_validated_data("order", Order)
            amount = context.get_typed_variable("amount_charged")

            # Final outputs
            context.set_output("order_id", order.id)
            context.set_output("amount_processed", amount)

            print(f"✅ Order {order.id} completed: ${amount}")
            return None

        agent.add_state("setup", setup)
        agent.add_state("process_order", process_order)
        agent.add_state("finalize", finalize)

        result = await agent.run()

        assert result.get_output("order_id") == 123
        assert result.get_output("amount_processed") == 99.99
        assert result.get_constant("tax_rate") == 0.08


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
