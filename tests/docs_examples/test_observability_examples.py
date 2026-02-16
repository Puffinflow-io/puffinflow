#!/usr/bin/env python3
"""
Tests for observability.ts documentation examples.
"""

import asyncio
import sys
import time

import pytest

# Add the src directory to Python path
sys.path.insert(0, "src")

from puffinflow import Agent, Priority, state


class TestObservabilityExamples:
    """Test examples from observability.ts documentation."""

    @pytest.mark.asyncio
    async def test_basic_metrics_collection(self):
        """Test basic metrics collection example."""
        agent = Agent("metrics-demo")

        @state(timeout=30.0)
        async def process_user_request(context):
            """Process request with metrics tracking"""

            user_id = context.get_variable("user_id", "user_123")
            request_type = context.get_variable("request_type", "search")

            print(f"🔄 Processing {request_type} request from {user_id}")

            # Simulate processing work
            start_time = time.time()
            await asyncio.sleep(0.01)  # Reduced for testing
            processing_time = time.time() - start_time

            # Track metrics
            context.set_variable("processing_time", processing_time)
            context.set_variable("request_processed", True)
            context.set_variable("result", "success")

            print(f"✅ Request completed successfully in {processing_time:.3f}s")
            return "generate_metrics_summary"

        @state
        async def generate_metrics_summary(context):
            """Generate metrics summary"""
            print("📊 Metrics collection completed!")
            print("   ✅ State execution times tracked")
            print("   ✅ Processing time recorded")
            print("   ✅ Success status tracked")

            processing_time = context.get_variable("processing_time", 0)
            context.set_output("metrics_collected", True)
            context.set_output("processing_duration", processing_time)
            return None

        agent.add_state("process_user_request", process_user_request)
        agent.add_state("generate_metrics_summary", generate_metrics_summary)

        # Run with sample context
        result = await agent.run(
            initial_context={"user_id": "user_001", "request_type": "search"}
        )

        assert result.get_variable("request_processed") is True
        assert result.get_variable("result") == "success"
        assert result.get_output("metrics_collected") is True
        assert result.get_output("processing_duration") > 0

    @pytest.mark.asyncio
    async def test_distributed_tracing_workflow(self):
        """Test distributed tracing workflow example."""
        agent = Agent("tracing-demo")

        @state(timeout=30.0)
        async def start_traced_workflow(context):
            """Start workflow with tracing"""

            user_id = context.get_variable("user_id", "user_123")
            print(f"🚀 Starting traced workflow for {user_id}")

            # Simulate authentication
            await asyncio.sleep(0.01)

            if user_id.startswith("user_"):
                context.set_variable("authenticated", True)
                print("✅ Authentication successful")
            else:
                context.set_variable("authenticated", False)
                print("❌ Authentication failed")

            return "load_user_data"

        @state(timeout=30.0)
        async def load_user_data(context):
            """Load user data with tracing"""

            if not context.get_variable("authenticated", False):
                return "handle_auth_failure"

            user_id = context.get_variable("user_id")
            print(f"🗃️ Loading data for {user_id}")

            # Simulate database query
            await asyncio.sleep(0.01)

            user_data = {
                "user_id": user_id,
                "name": f"User {user_id.split('_')[1]}",
                "subscription": "premium",
            }

            context.set_variable("user_data", user_data)
            return "personalize_experience"

        @state(timeout=30.0)
        async def personalize_experience(context):
            """Personalize with tracing"""

            user_data = context.get_variable("user_data")
            print(f"🎨 Personalizing for {user_data['name']}")

            # Simulate AI processing
            await asyncio.sleep(0.01)

            recommendations = ["item_1", "item_2", "item_3"]
            context.set_variable("recommendations", recommendations)
            context.set_output("personalization_complete", True)

            print(f"✅ Generated {len(recommendations)} recommendations")
            return None

        @state(timeout=10.0)
        async def handle_auth_failure(context):
            """Handle authentication failure"""

            print("❌ Authentication failed")
            context.set_output("auth_status", "failed")
            return None

        # Add states to agent
        agent.add_state("start_traced_workflow", start_traced_workflow)
        agent.add_state("load_user_data", load_user_data)
        agent.add_state("personalize_experience", personalize_experience)
        agent.add_state("handle_auth_failure", handle_auth_failure)

        # Test successful workflow
        result = await agent.run(initial_context={"user_id": "user_001"})

        assert result.get_variable("authenticated") is True
        assert result.get_variable("user_data")["user_id"] == "user_001"
        assert len(result.get_variable("recommendations")) == 3
        assert result.get_output("personalization_complete") is True

        # Test failed authentication with a new agent instance
        failed_agent = Agent("tracing-demo-failed")

        # Add states to the failed agent
        failed_agent.add_state("start_traced_workflow", start_traced_workflow)
        failed_agent.add_state("load_user_data", load_user_data)
        failed_agent.add_state("personalize_experience", personalize_experience)
        failed_agent.add_state("handle_auth_failure", handle_auth_failure)

        result = await failed_agent.run(initial_context={"user_id": "invalid_user"})

        assert result.get_variable("authenticated") is False
        assert result.get_output("auth_status") == "failed"

    @pytest.mark.asyncio
    async def test_structured_logging_example(self):
        """Test structured logging example."""
        agent = Agent("logging-demo")

        @state(timeout=30.0)
        async def api_request_with_logging(context):
            """API request with structured logging"""

            request_id = context.get_variable("request_id", "req_001")
            user_id = context.get_variable("user_id", "user_123")
            endpoint = context.get_variable("endpoint", "/api/profile")

            print(f"🌐 Starting API request {request_id}")
            print(f"   User: {user_id}")
            print(f"   Endpoint: {endpoint}")

            start_time = time.time()

            # Simulate API call
            await asyncio.sleep(0.01)

            duration_ms = int((time.time() - start_time) * 1000)

            # Log successful API request
            print(f"✅ API request completed in {duration_ms}ms")

            context.set_variable("api_status", "success")
            context.set_variable("api_duration", duration_ms)
            return "database_operation"

        @state(timeout=30.0)
        async def database_operation(context):
            """Database operation with logging"""

            request_id = context.get_variable("request_id")
            print(f"🗄️ Database operation for request {request_id}")

            start_time = time.time()

            # Simulate database query
            await asyncio.sleep(0.01)

            duration_ms = int((time.time() - start_time) * 1000)

            print(f"✅ Database query completed in {duration_ms}ms")

            context.set_variable("db_status", "success")
            context.set_variable("db_duration", duration_ms)
            return "complete_request"

        @state(timeout=30.0)
        async def complete_request(context):
            """Complete request with logging"""

            request_id = context.get_variable("request_id")

            print(f"📝 Completing request {request_id}")
            print(f"   API Status: {context.get_variable('api_status')}")
            print(f"   DB Status: {context.get_variable('db_status')}")

            context.set_output("status", "completed")
            context.set_output("request_id", request_id)
            return None

        # Add states to agent
        agent.add_state("api_request_with_logging", api_request_with_logging)
        agent.add_state("database_operation", database_operation)
        agent.add_state("complete_request", complete_request)

        # Run the logging workflow
        result = await agent.run(
            initial_context={
                "request_id": "req_001",
                "user_id": "user_123",
                "endpoint": "/api/profile",
            }
        )

        assert result.get_variable("api_status") == "success"
        assert result.get_variable("db_status") == "success"
        assert result.get_output("status") == "completed"
        assert result.get_output("request_id") == "req_001"
        assert result.get_variable("api_duration") > 0
        assert result.get_variable("db_duration") > 0

    @pytest.mark.asyncio
    async def test_health_monitoring_example(self):
        """Test health monitoring example."""
        agent = Agent("health-monitor")

        @state(timeout=30.0)
        async def monitored_operation(context):
            """Operation with health monitoring"""

            print("🔍 Running monitored operation...")

            # Simulate health monitoring
            health_metrics = {
                "cpu_usage": 45.0,
                "memory_usage": 60.0,
                "response_time": 0.150,
                "error_rate": 0.01,
            }

            # Simulate work
            await asyncio.sleep(0.01)
            context.set_variable("business_metric", 42)

            # Calculate health status
            health_status = "healthy"
            if health_metrics["cpu_usage"] > 80:
                health_status = "degraded"
            if health_metrics["error_rate"] > 0.05:
                health_status = "unhealthy"

            health_report = {
                "overall_status": health_status,
                "metrics": health_metrics,
                "timestamp": time.time(),
            }

            print(f"📊 Health Status: {health_status}")

            context.set_variable("health_report", health_report)
            context.set_output("health_status", health_report)
            return None

        agent.add_state("monitored_operation", monitored_operation)

        # Run the health monitoring workflow
        result = await agent.run()

        health_report = result.get_output("health_status")
        assert health_report["overall_status"] == "healthy"
        assert health_report["metrics"]["cpu_usage"] == 45.0
        assert result.get_variable("business_metric") == 42

    @pytest.mark.asyncio
    async def test_production_observability_workflow(self):
        """Test production observability workflow."""
        agent = Agent("production-workflow")

        @state(timeout=30.0, priority=Priority.HIGH)
        async def production_ready_state(context):
            """Production state with full observability"""

            print("🚀 Running production state with full observability")

            # Simulate production work
            start_time = time.time()
            await asyncio.sleep(0.01)
            execution_time = time.time() - start_time

            # Business metrics
            business_kpi = 95.5
            user_satisfaction = 4.8

            context.set_variable("business_kpi", business_kpi)
            context.set_variable("user_satisfaction", user_satisfaction)
            context.set_variable("execution_time", execution_time)

            # Set outputs for monitoring
            context.set_output("production_complete", True)
            context.set_output(
                "performance_metrics",
                {
                    "execution_time": execution_time,
                    "business_kpi": business_kpi,
                    "user_satisfaction": user_satisfaction,
                },
            )

            print(f"✅ Production operation completed in {execution_time:.3f}s")
            print(f"   Business KPI: {business_kpi}")
            print(f"   User Satisfaction: {user_satisfaction}")

            return None

        agent.add_state("production_ready_state", production_ready_state)

        # Run the production workflow
        result = await agent.run()

        assert result.get_output("production_complete") is True

        metrics = result.get_output("performance_metrics")
        assert metrics["business_kpi"] == 95.5
        assert metrics["user_satisfaction"] == 4.8
        assert metrics["execution_time"] > 0

        assert result.get_variable("business_kpi") == 95.5
        assert result.get_variable("user_satisfaction") == 4.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
