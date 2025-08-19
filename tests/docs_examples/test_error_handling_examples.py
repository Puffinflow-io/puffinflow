"""Test all examples from the error handling documentation."""

import asyncio

import pytest

from puffinflow import Agent, state
from puffinflow.core.agent.state import Priority


@pytest.mark.asyncio
class TestErrorHandlingExamples:
    """Test examples from error-handling.ts documentation."""

    async def test_basic_retry_configuration(self):
        """Test basic retry configuration examples."""
        agent = Agent("error-handling-agent")

        @state(max_retries=3)
        async def stable_api_call(context):
            attempts = context.get_state("attempts", 0) + 1
            context.set_state("attempts", attempts)

            # First attempt fails, second succeeds
            if attempts == 1:
                raise Exception(f"Temporary API error (attempt {attempts})")

            context.set_variable("api_result", "success")

        @state(max_retries=1)
        async def expensive_operation(context):
            attempts = context.get_state("expensive_attempts", 0) + 1
            context.set_state("expensive_attempts", attempts)

            # Only retry once due to cost
            if attempts == 1:
                raise Exception("Expensive operation failed on first try")

            context.set_variable("expensive_result", "success")

        agent.add_state("stable_api_call", stable_api_call)
        agent.add_state(
            "expensive_operation", expensive_operation, dependencies=["stable_api_call"]
        )

        result = await agent.run()

        # Verify retry mechanisms worked
        assert result.get_variable("api_result") == "success"
        assert result.get_variable("expensive_result") == "success"

    async def test_timeout_configuration(self):
        """Test timeout configuration examples."""
        agent = Agent("timeout-agent")

        @state(timeout=5.0)
        async def quick_health_check(context):
            try:
                # This should complete within timeout
                await asyncio.sleep(0.1)
                context.set_variable("health_status", "healthy")
            except asyncio.TimeoutError:
                context.set_variable("health_status", "timeout")

        @state(timeout=2.0, max_retries=3)
        async def real_time_api_call(context):
            attempt = context.get_state("rt_attempts", 0) + 1
            context.set_state("rt_attempts", attempt)

            try:
                # First two attempts timeout, third succeeds
                if attempt <= 2:
                    await asyncio.sleep(3.0)  # This will timeout
                else:
                    await asyncio.sleep(0.1)  # This will succeed

                context.set_variable("rt_result", "success")
            except asyncio.TimeoutError:
                if attempt >= 4:  # Final attempt
                    context.set_variable("rt_result", "timeout")

        agent.add_state("quick_health_check", quick_health_check)
        agent.add_state(
            "real_time_api_call",
            real_time_api_call,
            dependencies=["quick_health_check"],
        )

        result = await agent.run()

        # Verify timeout handling
        assert result.get_variable("health_status") == "healthy"
        assert result.get_variable("rt_result") == "success"

    async def test_priority_based_error_handling(self):
        """Test priority-based error handling examples."""
        agent = Agent("priority-agent", max_concurrent=10)

        @state(priority=Priority.CRITICAL, max_retries=5, timeout=60.0)
        async def critical_system_operation(context):
            attempt = context.get_state("critical_attempts", 0) + 1
            context.set_state("critical_attempts", attempt)

            # Critical operations get aggressive retry
            if attempt <= 2:
                raise Exception(f"Critical system not ready (attempt {attempt})")

            context.set_variable("critical_status", "operational")

        @state(priority=Priority.HIGH, max_retries=3, timeout=30.0)
        async def user_facing_operation(context):
            attempt = context.get_state("user_attempts", 0) + 1
            context.set_state("user_attempts", attempt)

            # Succeeds on second attempt
            if attempt == 1:
                raise Exception(f"User operation failed (attempt {attempt})")

            context.set_variable("user_status", "completed")

        @state(priority=Priority.NORMAL, max_retries=2, timeout=15.0)
        async def business_logic_operation(context):
            attempt = context.get_state("business_attempts", 0) + 1
            context.set_state("business_attempts", attempt)

            # Succeeds on second attempt
            if attempt < 2:
                raise Exception(f"Business logic issue (attempt {attempt})")

            context.set_variable("business_status", "completed")

        @state(priority=Priority.LOW, max_retries=1, timeout=10.0)
        async def background_operation(context):
            attempt = context.get_state("background_attempts", 0) + 1
            context.set_state("background_attempts", attempt)

            # Succeeds on first attempt for test
            context.set_variable("background_status", "completed")

        agent.add_state("critical_system_operation", critical_system_operation)
        agent.add_state(
            "user_facing_operation",
            user_facing_operation,
            dependencies=["critical_system_operation"],
        )
        agent.add_state(
            "business_logic_operation",
            business_logic_operation,
            dependencies=["user_facing_operation"],
        )
        agent.add_state(
            "background_operation",
            background_operation,
            dependencies=["business_logic_operation"],
        )

        result = await agent.run()

        # Verify priority-based execution
        assert result.get_variable("critical_status") == "operational"
        assert result.get_variable("user_status") == "completed"
        assert result.get_variable("business_status") == "completed"
        assert result.get_variable("background_status") == "completed"

    async def test_graceful_degradation_pattern(self):
        """Test graceful degradation pattern."""
        agent = Agent("graceful-degradation-agent")

        @state(max_retries=2, timeout=10.0)
        async def primary_service_call(context):
            # Always fails for this test
            raise Exception("Primary service unavailable")

        @state(max_retries=1, timeout=5.0)
        async def fallback_service_call(context):
            # Always fails for this test
            raise Exception("Fallback service unavailable")

        @state(max_retries=0)
        async def degraded_mode_operation(context):
            # Always succeeds
            await asyncio.sleep(0.01)
            context.set_variable("service_result", "degraded_mode")

        @state
        async def orchestrate_with_fallbacks(context):
            try:
                await primary_service_call(context)
            except Exception:
                try:
                    await fallback_service_call(context)
                except Exception:
                    await degraded_mode_operation(context)

            result = context.get_variable("service_result")

            # Set appropriate status based on which service worked
            if result == "primary_success":
                context.set_variable("system_status", "fully_operational")
            elif result == "fallback_success":
                context.set_variable("system_status", "reduced_functionality")
            else:
                context.set_variable("system_status", "degraded_mode")

        agent.add_state("orchestrate_with_fallbacks", orchestrate_with_fallbacks)

        result = await agent.run()

        # Verify graceful degradation
        assert result.get_variable("service_result") == "degraded_mode"
        assert result.get_variable("system_status") == "degraded_mode"

    async def test_error_handling_with_state_isolation(self):
        """Test that error handling works correctly with state isolation."""
        agent = Agent("error-isolation-agent")

        @state(max_retries=2)
        async def failing_state(context):
            attempt = context.get_state("fail_attempts", 0) + 1
            context.set_state("fail_attempts", attempt)

            if attempt <= 1:
                raise Exception(f"Transient failure (attempt {attempt})")

            context.set_variable("fail_result", "eventually_succeeded")

        @state(max_retries=0)
        async def success_state(context):
            context.set_variable("success_result", "always_works")

        agent.add_state("failing_state", failing_state)
        agent.add_state("success_state", success_state, dependencies=["failing_state"])

        result = await agent.run()

        # Verify both states completed correctly
        assert result.get_variable("fail_result") == "eventually_succeeded"
        assert result.get_variable("success_result") == "always_works"
