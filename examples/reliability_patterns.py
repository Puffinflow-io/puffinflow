"""
Reliability Patterns Examples

This example demonstrates reliability patterns and fault tolerance:
- Circuit breakers for external service calls
- Bulkhead isolation patterns
- Resource leak detection
- Retry mechanisms and error handling
"""

import asyncio
import random
import time

from puffinflow import (
    Agent,
    Bulkhead,
    BulkheadConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    Context,
    ResourceLeakDetector,
    external_service,
    fault_tolerant,
    state,
)


class ExternalServiceAgent(Agent):
    """Agent that simulates calls to external services with reliability patterns."""

    def __init__(self, name: str, service_reliability: float = 0.8):
        super().__init__(name)
        self.set_variable("service_reliability", service_reliability)

        # Configure circuit breaker
        cb_config = CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout=5.0, expected_exception=Exception
        )
        self.circuit_breaker = CircuitBreaker(cb_config)

        # Register all decorated states
        self.add_state("call_external_api", self.call_external_api)
        self.add_state("process_response", self.process_response)
        self.add_state("handle_failure", self.handle_failure)

    @external_service(timeout=5.0, retries=3)
    async def call_external_api(self, context: Context):
        """Make a call to an external API with circuit breaker protection."""
        reliability = self.get_variable("service_reliability", 0.8)

        async def api_call():
            # Simulate external API call
            await asyncio.sleep(0.1)

            # Simulate service failures based on reliability
            if random.random() > reliability:
                raise Exception(
                    f"External service failure (reliability: {reliability})"
                )

            return {
                "status": "success",
                "data": {"result": "api_response", "timestamp": time.time()},
                "response_time": 0.1,
            }

        try:
            # Use circuit breaker to protect the call
            result = await self.circuit_breaker.call(api_call)

            context.set_output("api_response", result)
            context.set_metric("api_call_success", 1)

            print(f"{self.name} successfully called external API")
            return "process_response"

        except Exception as e:
            context.set_output("api_error", str(e))
            context.set_metric("api_call_success", 0)

            print(f"{self.name} external API call failed: {e}")
            return "handle_failure"

    @state(cpu=0.5, memory=128.0)
    async def process_response(self, context: Context):
        """Process successful API response."""
        response = context.get_output("api_response", {})

        processed_data = {
            "original_data": response.get("data", {}),
            "processed_at": time.time(),
            "processing_status": "completed",
        }

        context.set_output("processed_data", processed_data)
        print(f"{self.name} processed API response successfully")
        return None

    @fault_tolerant(retries=2)
    async def handle_failure(self, context: Context):
        """Handle API call failures with fallback logic."""
        error = context.get_output("api_error", "Unknown error")

        # Implement fallback logic
        fallback_data = {
            "status": "fallback",
            "data": {"result": "cached_response", "timestamp": time.time()},
            "error": error,
            "fallback_reason": "external_service_unavailable",
        }

        context.set_output("fallback_data", fallback_data)
        context.set_metric("fallback_used", 1)

        print(f"{self.name} used fallback due to: {error}")
        return None


class DatabaseAgent(Agent):
    """Agent that demonstrates bulkhead isolation for database operations."""

    def __init__(self, name: str):
        super().__init__(name)

        # Configure bulkhead for database operations
        bulkhead_config = BulkheadConfig(max_concurrent_calls=5, max_wait_duration=10.0)
        self.db_bulkhead = Bulkhead(bulkhead_config)

        # Register all decorated states
        self.add_state("execute_query", self.execute_query)
        self.add_state("log_operation", self.log_operation)
        self.add_state("handle_db_error", self.handle_db_error)

    @state(cpu=2.0, memory=512.0)
    async def execute_query(self, context: Context):
        """Execute database query with bulkhead isolation."""
        query_type = context.get_input("query_type", "SELECT")

        async def db_operation():
            # Simulate database operation
            operation_time = random.uniform(0.1, 0.5)
            await asyncio.sleep(operation_time)

            # Simulate occasional database errors
            if random.random() < 0.1:  # 10% failure rate
                raise Exception("Database connection timeout")

            return {
                "query_type": query_type,
                "rows_affected": random.randint(1, 1000),
                "execution_time": operation_time,
                "status": "completed",
            }

        try:
            # Use bulkhead to limit concurrent database operations
            result = await self.db_bulkhead.call(db_operation)

            context.set_output("query_result", result)
            context.set_metric("db_operation_success", 1)

            print(f"{self.name} executed {query_type} query successfully")
            return "log_operation"

        except Exception as e:
            context.set_output("db_error", str(e))
            context.set_metric("db_operation_success", 0)

            print(f"{self.name} database operation failed: {e}")
            return "handle_db_error"

    @state(cpu=0.2, memory=64.0)
    async def log_operation(self, context: Context):
        """Log successful database operation."""
        result = context.get_output("query_result", {})

        log_entry = {
            "timestamp": time.time(),
            "operation": "database_query",
            "status": "success",
            "details": result,
        }

        context.set_output("operation_log", log_entry)
        print(f"{self.name} logged successful operation")
        return None

    @state(cpu=0.1, memory=32.0)
    async def handle_db_error(self, context: Context):
        """Handle database errors."""
        error = context.get_output("db_error", "Unknown database error")

        error_log = {
            "timestamp": time.time(),
            "operation": "database_query",
            "status": "error",
            "error_message": error,
            "recovery_action": "retry_with_backoff",
        }

        context.set_output("error_log", error_log)
        print(f"{self.name} logged database error: {error}")
        return None


class ResourceIntensiveAgent(Agent):
    """Agent that demonstrates resource leak detection."""

    def __init__(self, name: str):
        super().__init__(name)
        self.leak_detector = ResourceLeakDetector()
        self.resources = []

        # Register all decorated states
        self.add_state("allocate_resources", self.allocate_resources)
        self.add_state("process_with_resources", self.process_with_resources)
        self.add_state("cleanup_resources", self.cleanup_resources)

    @state(cpu=4.0, memory=2048.0)
    async def allocate_resources(self, context: Context):
        """Allocate resources with leak detection."""
        resource_count = context.get_input("resource_count", 10)

        # Start monitoring for resource leaks
        self.leak_detector.start_monitoring()

        try:
            # Simulate resource allocation
            for i in range(resource_count):
                resource = {
                    "id": f"resource_{i}",
                    "type": "memory_buffer",
                    "size": 1024 * 1024,  # 1MB
                    "allocated_at": time.time(),
                }
                self.resources.append(resource)
                await asyncio.sleep(0.01)  # Simulate allocation time

            context.set_output("allocated_resources", len(self.resources))
            context.set_metric("resource_allocation_rate", len(self.resources) / 0.1)

            print(f"{self.name} allocated {len(self.resources)} resources")
            return "process_with_resources"

        except Exception as e:
            context.set_output("allocation_error", str(e))
            return "cleanup_resources"

    @state(cpu=2.0, memory=1024.0)
    async def process_with_resources(self, context: Context):
        """Process data using allocated resources."""
        # Simulate processing with allocated resources
        processing_time = len(self.resources) * 0.01
        await asyncio.sleep(processing_time)

        # Simulate occasional processing errors that might cause leaks
        if random.random() < 0.2:  # 20% chance of error
            print(f"{self.name} processing error occurred - potential resource leak!")
            return "cleanup_resources"

        result = {
            "processed_items": len(self.resources) * 10,
            "processing_time": processing_time,
            "resources_used": len(self.resources),
        }

        context.set_output("processing_result", result)
        print(f"{self.name} processed data successfully")
        return "cleanup_resources"

    @state(cpu=0.5, memory=128.0)
    async def cleanup_resources(self, context: Context):
        """Clean up allocated resources."""
        # Check for resource leaks before cleanup
        leak_report = self.leak_detector.check_leaks()

        if leak_report.get("potential_leaks", 0) > 0:
            print(
                f"{self.name} detected {leak_report['potential_leaks']} potential resource leaks"
            )
            context.set_metric(
                "resource_leaks_detected", leak_report["potential_leaks"]
            )

        # Clean up resources
        cleaned_count = len(self.resources)
        self.resources.clear()

        # Stop monitoring
        self.leak_detector.stop_monitoring()

        context.set_output("cleaned_resources", cleaned_count)
        context.set_metric("cleanup_efficiency", 1.0)

        print(f"{self.name} cleaned up {cleaned_count} resources")
        return None


async def demonstrate_circuit_breaker():
    """Demonstrate circuit breaker pattern."""
    print("=== Circuit Breaker Pattern ===")

    # Create agents with different service reliabilities
    reliable_agent = ExternalServiceAgent("reliable-service", 0.9)
    unreliable_agent = ExternalServiceAgent("unreliable-service", 0.3)

    agents = [reliable_agent, unreliable_agent]

    # Run multiple calls to trigger circuit breaker
    for i in range(5):
        print(f"\nRound {i + 1}:")

        for agent in agents:
            try:
                result = await agent.run()
                status = "SUCCESS" if result.get_output("api_response") else "FALLBACK"
                print(f"  {agent.name}: {status}")
            except Exception as e:
                print(f"  {agent.name}: ERROR - {e}")

    print()


async def demonstrate_bulkhead_isolation():
    """Demonstrate bulkhead isolation pattern."""
    print("=== Bulkhead Isolation Pattern ===")

    # Create multiple database agents
    db_agents = [DatabaseAgent(f"db-agent-{i}") for i in range(3)]

    # Simulate concurrent database operations
    tasks = []
    for agent in db_agents:
        # Set different query types
        agent.set_variable("query_type", random.choice(["SELECT", "INSERT", "UPDATE"]))
        tasks.append(agent.run())

    # Run all agents concurrently
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    execution_time = time.time() - start_time

    print(f"Concurrent execution completed in {execution_time:.2f} seconds")

    # Analyze results
    successful = 0
    failed = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  db-agent-{i}: EXCEPTION - {result}")
            failed += 1
        else:
            query_result = result.get_output("query_result")
            if query_result:
                print(f"  db-agent-{i}: SUCCESS - {query_result['rows_affected']} rows")
                successful += 1
            else:
                print(f"  db-agent-{i}: FAILED")
                failed += 1

    print(f"Results: {successful} successful, {failed} failed")
    print()


async def demonstrate_resource_leak_detection():
    """Demonstrate resource leak detection."""
    print("=== Resource Leak Detection ===")

    # Create resource-intensive agents
    agents = [
        ResourceIntensiveAgent("resource-agent-1"),
        ResourceIntensiveAgent("resource-agent-2"),
    ]

    # Set different resource allocation patterns
    agents[0].set_variable("resource_count", 5)  # Normal allocation
    agents[1].set_variable("resource_count", 15)  # Heavy allocation

    # Run agents and monitor for leaks
    for agent in agents:
        print(f"Running {agent.name}...")
        result = await agent.run()

        allocated = result.get_output("allocated_resources", 0)
        cleaned = result.get_output("cleaned_resources", 0)
        leaks = result.get_metric("resource_leaks_detected", 0)

        print(f"  Allocated: {allocated}, Cleaned: {cleaned}, Leaks detected: {leaks}")

    print()


async def demonstrate_fault_tolerance():
    """Demonstrate comprehensive fault tolerance."""
    print("=== Comprehensive Fault Tolerance ===")

    # Create a mixed workload with different reliability patterns
    agents = [
        ExternalServiceAgent("api-service", 0.7),
        DatabaseAgent("primary-db"),
        ResourceIntensiveAgent("batch-processor"),
    ]

    # Set up different scenarios
    agents[2].set_variable("resource_count", 8)

    # Run all agents and collect reliability metrics
    start_time = time.time()
    results = []

    for agent in agents:
        try:
            result = await agent.run()
            results.append((agent.name, result, None))
        except Exception as e:
            results.append((agent.name, None, e))

    total_time = time.time() - start_time

    print(f"Fault tolerance test completed in {total_time:.2f} seconds")
    print("\nReliability Report:")

    for agent_name, result, error in results:
        if error:
            print(f"  {agent_name}: CRITICAL FAILURE - {error}")
        elif result:
            # Analyze success metrics
            success_metrics = [
                result.get_metric("api_call_success", None),
                result.get_metric("db_operation_success", None),
                result.get_metric("cleanup_efficiency", None),
            ]

            success_rate = sum(m for m in success_metrics if m is not None)
            fallback_used = result.get_metric("fallback_used", 0)

            status = "HEALTHY" if success_rate > 0 else "DEGRADED"
            if fallback_used > 0:
                status += " (FALLBACK)"

            print(f"  {agent_name}: {status}")
        else:
            print(f"  {agent_name}: UNKNOWN STATUS")

    print()


async def run_stress_test():
    """Run a stress test to validate reliability patterns."""
    print("=== Reliability Stress Test ===")

    # Create a large number of agents with varying reliability
    stress_agents = []

    for i in range(10):
        reliability = random.uniform(0.5, 0.95)
        agent = ExternalServiceAgent(f"stress-agent-{i}", reliability)
        stress_agents.append(agent)

    # Run stress test
    print(f"Running stress test with {len(stress_agents)} agents...")

    start_time = time.time()
    tasks = [agent.run() for agent in stress_agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    stress_time = time.time() - start_time

    # Analyze stress test results
    successful = 0
    failed = 0
    fallbacks = 0

    for result in results:
        if isinstance(result, Exception):
            failed += 1
        else:
            if result.get_output("api_response"):
                successful += 1
            elif result.get_output("fallback_data"):
                fallbacks += 1
            else:
                failed += 1

    print(f"Stress test completed in {stress_time:.2f} seconds")
    print(f"Results: {successful} successful, {fallbacks} fallbacks, {failed} failed")
    print(f"Success rate: {(successful + fallbacks) / len(stress_agents) * 100:.1f}%")
    print()


async def main():
    """Run all reliability pattern examples."""
    print("PuffinFlow Reliability Patterns Examples")
    print("=" * 50)

    await demonstrate_circuit_breaker()
    await demonstrate_bulkhead_isolation()
    await demonstrate_resource_leak_detection()
    await demonstrate_fault_tolerance()
    await run_stress_test()

    print("All reliability pattern examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
