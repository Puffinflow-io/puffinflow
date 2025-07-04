"""End-to-end tests for microservices and event-driven scenarios.

Additional E2E tests that complement the main workflow tests.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, Mock

from puffinflow import (
    Agent, Context, state, run_agents_parallel, run_agents_sequential
)


class MicroserviceAgent(Agent):
    """Agent that simulates a microservice."""
    
    def __init__(self, name: str, service_config: dict):
        super().__init__(name)
        self.set_variable("service_config", service_config)
        self.add_state('health_check', self.health_check)
        self.add_state('process_request', self.process_request)
        self.add_state('respond', self.respond)
    
    @state(cpu=0.3, memory=128.0)
    async def health_check(self, context: Context):
        """Check service health."""
        config = self.get_variable("service_config", {})
        service_type = config.get("type", "unknown")
        
        await asyncio.sleep(0.05)
        
        # Simulate health check
        is_healthy = True  # Assume healthy for test
        
        context.set_output("health_status", "healthy" if is_healthy else "unhealthy")
        context.set_output("service_type", service_type)
        
        if not is_healthy:
            return None  # Stop if unhealthy
        
        return "process_request"
    
    @state(cpu=1.0, memory=256.0)
    async def process_request(self, context: Context):
        """Process service request."""
        config = self.get_variable("service_config", {})
        processing_time = config.get("processing_time", 0.2)
        
        await asyncio.sleep(processing_time)
        
        # Simulate request processing
        result = {
            "service": self.name,
            "processed_at": time.time(),
            "processing_time": processing_time,
            "status": "success"
        }
        
        context.set_output("processing_result", result)
        context.set_metric("service_latency", processing_time)
        
        return "respond"
    
    @state(cpu=0.2, memory=64.0)
    async def respond(self, context: Context):
        """Send response."""
        processing_result = context.get_output("processing_result", {})
        
        await asyncio.sleep(0.05)
        
        response = {
            "response_id": f"resp_{int(time.time())}",
            "service_result": processing_result,
            "response_time": 0.05
        }
        
        context.set_output("service_response", response)
        context.set_output("completed", True)
        
        return None


class EventProducerAgent(Agent):
    """Agent that produces events."""
    
    def __init__(self, name: str, event_config: dict):
        super().__init__(name)
        self.set_variable("event_config", event_config)
        self.add_state('generate_event', self.generate_event)
    
    @state(cpu=0.5, memory=128.0)
    async def generate_event(self, context: Context):
        config = self.get_variable("event_config", {})
        event_type = config.get("type", "generic")
        
        await asyncio.sleep(0.1)
        
        event = {
            "event_id": f"evt_{int(time.time())}_{self.name}",
            "type": event_type,
            "producer": self.name,
            "timestamp": time.time(),
            "data": config.get("data", {})
        }
        
        context.set_output("event", event)
        context.set_output("event_produced", True)
        context.set_metric("event_generation_time", 0.1)
        
        return None


class EventConsumerAgent(Agent):
    """Agent that consumes and processes events."""
    
    def __init__(self, name: str, consumer_config: dict):
        super().__init__(name)
        self.set_variable("consumer_config", consumer_config)
        self.add_state('consume_event', self.consume_event)
        self.add_state('process_event', self.process_event)
    
    @state(cpu=0.3, memory=128.0)
    async def consume_event(self, context: Context):
        config = self.get_variable("consumer_config", {})
        
        await asyncio.sleep(0.05)
        
        # Simulate event consumption
        context.set_output("event_consumed", True)
        context.set_output("consumer_type", config.get("type", "generic"))
        
        return "process_event"
    
    @state(cpu=1.0, memory=256.0)
    async def process_event(self, context: Context):
        config = self.get_variable("consumer_config", {})
        processing_time = config.get("processing_time", 0.2)
        
        await asyncio.sleep(processing_time)
        
        processing_result = {
            "consumer": self.name,
            "processed_at": time.time(),
            "processing_time": processing_time,
            "result": "processed_successfully"
        }
        
        context.set_output("processing_result", processing_result)
        context.set_output("event_processed", True)
        context.set_metric("event_processing_time", processing_time)
        
        return None


@pytest.mark.e2e
@pytest.mark.asyncio
class TestMicroservicesOrchestration:
    """Test microservices orchestration scenarios."""
    
    async def test_microservices_orchestration(self):
        """Test orchestrating multiple microservices."""
        
        # Create microservices
        service_configs = [
            {"type": "auth", "processing_time": 0.1},
            {"type": "data", "processing_time": 0.3},
            {"type": "notification", "processing_time": 0.15},
            {"type": "logging", "processing_time": 0.05}
        ]
        
        microservices = [
            MicroserviceAgent(f"{config['type']}-service", config)
            for config in service_configs
        ]
        
        # Execute microservices orchestration
        start_time = time.time()
        service_results = await run_agents_parallel(microservices)
        total_time = time.time() - start_time
        
        # Verify microservices orchestration
        assert len(service_results) == 4
        
        # All services should complete successfully
        for service_name, result in service_results.items():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            assert result.get_output("health_status") == "healthy"
            assert result.get_output("completed") is True
            
            # Verify service-specific behavior
            service_response = result.get_output("service_response", {})
            assert "response_id" in service_response
            assert "service_result" in service_response
        
        # Verify orchestration timing
        # Parallel execution should be faster than sequential
        max_individual_time = max(config["processing_time"] for config in service_configs)
        assert total_time <= max_individual_time + 0.5  # Allow overhead
        
        # Verify service types
        service_types = [
            result.get_output("service_type")
            for result in service_results.values()
        ]
        expected_types = ["auth", "data", "notification", "logging"]
        assert set(service_types) == set(expected_types)
    
    async def test_service_dependency_chain(self):
        """Test services with dependencies executing in sequence."""
        
        # Create services with dependencies (auth -> data -> notification -> logging)
        dependent_services = [
            MicroserviceAgent("auth-service", {"type": "auth", "processing_time": 0.1}),
            MicroserviceAgent("data-service", {"type": "data", "processing_time": 0.2}),
            MicroserviceAgent("notification-service", {"type": "notification", "processing_time": 0.15}),
            MicroserviceAgent("logging-service", {"type": "logging", "processing_time": 0.05})
        ]
        
        # Execute services sequentially to simulate dependencies
        start_time = time.time()
        sequential_results = await run_agents_sequential(dependent_services)
        total_time = time.time() - start_time
        
        # Verify dependency chain execution
        assert len(sequential_results) == 4
        
        # Verify execution order
        service_names = list(sequential_results.keys())
        expected_order = ["auth-service", "data-service", "notification-service", "logging-service"]
        assert service_names == expected_order
        
        # All services should complete successfully
        for result in sequential_results.values():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            assert result.get_output("completed") is True
        
        # Sequential execution should take longer than parallel
        expected_min_time = sum(config["processing_time"] for config in [
            {"processing_time": 0.1}, {"processing_time": 0.2}, 
            {"processing_time": 0.15}, {"processing_time": 0.05}
        ])
        assert total_time >= expected_min_time * 0.8  # Allow some variance
    
    async def test_service_failure_recovery(self):
        """Test microservices behavior when some services fail."""
        
        class FailingMicroservice(MicroserviceAgent):
            def __init__(self, name: str, service_config: dict, should_fail: bool = False):
                super().__init__(name, service_config)
                self.set_variable("should_fail", should_fail)
            
            @state(cpu=1.0, memory=256.0)
            async def process_request(self, context: Context):
                if self.get_variable("should_fail", False):
                    context.set_output("error", "Service intentionally failed")
                    raise RuntimeError("Simulated service failure")
                
                return await super().process_request(context)
        
        # Create mix of working and failing services
        mixed_services = [
            FailingMicroservice("working-service-1", {"type": "auth", "processing_time": 0.1}, should_fail=False),
            FailingMicroservice("failing-service", {"type": "data", "processing_time": 0.2}, should_fail=True),
            FailingMicroservice("working-service-2", {"type": "notification", "processing_time": 0.15}, should_fail=False)
        ]
        
        # Execute services and handle failures
        results = []
        for service in mixed_services:
            try:
                result = await service.run()
                results.append(("success", result))
            except Exception as e:
                results.append(("failure", str(e)))
        
        # Verify mixed results
        assert len(results) == 3
        
        success_results = [r for r in results if r[0] == "success"]
        failure_results = [r for r in results if r[0] == "failure"]
        
        assert len(success_results) == 2  # Two working services
        assert len(failure_results) == 1   # One failing service
        
        # Verify successful services completed properly
        for _, result in success_results:
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            assert result.get_output("completed") is True


@pytest.mark.e2e
@pytest.mark.asyncio
class TestEventDrivenWorkflows:
    """Test event-driven workflow scenarios."""
    
    async def test_event_driven_workflow(self):
        """Test event-driven workflow orchestration."""
        
        # Create event-driven workflow
        event_configs = [
            {"type": "user_action", "data": {"user_id": 123, "action": "login"}},
            {"type": "system_alert", "data": {"severity": "high", "component": "database"}},
            {"type": "data_update", "data": {"table": "users", "records": 50}}
        ]
        
        consumer_configs = [
            {"type": "analytics", "processing_time": 0.15},
            {"type": "alerting", "processing_time": 0.1},
            {"type": "audit", "processing_time": 0.25}
        ]
        
        # Create producers and consumers
        producers = [
            EventProducerAgent(f"producer-{i}", config)
            for i, config in enumerate(event_configs)
        ]
        
        consumers = [
            EventConsumerAgent(f"consumer-{i}", config)
            for i, config in enumerate(consumer_configs)
        ]
        
        # Execute event-driven workflow
        start_time = time.time()
        
        # Phase 1: Generate events
        producer_results = await run_agents_parallel(producers)
        
        # Phase 2: Process events
        consumer_results = await run_agents_parallel(consumers)
        
        total_time = time.time() - start_time
        
        # Verify event-driven workflow
        assert len(producer_results) == 3
        assert len(consumer_results) == 3
        
        # Verify event production
        produced_events = []
        for result in producer_results.values():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            assert result.get_output("event_produced") is True
            
            event = result.get_output("event", {})
            assert "event_id" in event
            assert "type" in event
            assert "timestamp" in event
            produced_events.append(event)
        
        # Verify event consumption
        for result in consumer_results.values():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            assert result.get_output("event_consumed") is True
            assert result.get_output("event_processed") is True
            
            processing_result = result.get_output("processing_result", {})
            assert processing_result["result"] == "processed_successfully"
        
        # Verify event types
        event_types = [event["type"] for event in produced_events]
        expected_event_types = ["user_action", "system_alert", "data_update"]
        assert set(event_types) == set(expected_event_types)
        
        # Verify consumer types
        consumer_types = [
            result.get_output("consumer_type")
            for result in consumer_results.values()
        ]
        expected_consumer_types = ["analytics", "alerting", "audit"]
        assert set(consumer_types) == set(expected_consumer_types)
        
        # Event-driven workflow should be efficient
        assert total_time <= 1.0  # Should complete within 1 second
    
    async def test_event_filtering_and_routing(self):
        """Test event filtering and routing to appropriate consumers."""
        
        class FilteringEventConsumer(EventConsumerAgent):
            def __init__(self, name: str, consumer_config: dict, event_filter: list):
                super().__init__(name, consumer_config)
                self.set_variable("event_filter", event_filter)
            
            @state(cpu=0.3, memory=128.0)
            async def consume_event(self, context: Context):
                config = self.get_variable("consumer_config", {})
                event_filter = self.get_variable("event_filter", [])
                
                await asyncio.sleep(0.05)
                
                # Simulate event filtering
                # In real scenario, this would check actual event content
                simulated_event_type = config.get("simulated_event_type", "generic")
                
                if simulated_event_type in event_filter:
                    context.set_output("event_consumed", True)
                    context.set_output("consumer_type", config.get("type", "generic"))
                    return "process_event"
                else:
                    context.set_output("event_filtered_out", True)
                    context.set_output("consumer_type", config.get("type", "generic"))
                    return None  # Skip processing
        
        # Create specialized consumers with filters
        specialized_consumers = [
            FilteringEventConsumer(
                "security-consumer", 
                {"type": "security", "processing_time": 0.1, "simulated_event_type": "security_alert"}, 
                ["security_alert", "auth_failure"]
            ),
            FilteringEventConsumer(
                "analytics-consumer", 
                {"type": "analytics", "processing_time": 0.2, "simulated_event_type": "user_action"}, 
                ["user_action", "page_view"]
            ),
            FilteringEventConsumer(
                "monitoring-consumer", 
                {"type": "monitoring", "processing_time": 0.15, "simulated_event_type": "system_metric"}, 
                ["system_metric", "performance_data"]
            )
        ]
        
        # Execute filtering consumers
        consumer_results = await run_agents_parallel(specialized_consumers)
        
        # Verify filtering behavior
        assert len(consumer_results) == 3
        
        # Check which consumers processed events vs filtered them out
        processed_consumers = []
        filtered_consumers = []
        
        for name, result in consumer_results.items():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            
            if result.get_output("event_consumed", False):
                processed_consumers.append(name)
                assert result.get_output("event_processed", False) is True
            elif result.get_output("event_filtered_out", False):
                filtered_consumers.append(name)
        
        # Verify that filtering worked (some consumers processed, others filtered)
        assert len(processed_consumers) >= 1  # At least one should process
        assert len(filtered_consumers) >= 1   # At least one should filter
        
        # Verify consumer types
        consumer_types = [
            result.get_output("consumer_type")
            for result in consumer_results.values()
        ]
        expected_types = ["security", "analytics", "monitoring"]
        assert set(consumer_types) == set(expected_types)
    
    async def test_event_stream_processing(self):
        """Test processing a stream of events over time."""
        
        class StreamProcessorAgent(Agent):
            def __init__(self, name: str, batch_size: int = 3):
                super().__init__(name)
                self.set_variable("batch_size", batch_size)
                self.set_variable("processed_events", [])
                self.add_state('collect_events', self.collect_events)
                self.add_state('process_batch', self.process_batch)
            
            @state(cpu=0.5, memory=256.0)
            async def collect_events(self, context: Context):
                batch_size = self.get_variable("batch_size", 3)
                
                # Simulate collecting events over time
                collected_events = []
                for i in range(batch_size):
                    await asyncio.sleep(0.05)  # Simulate event arrival
                    event = {
                        "event_id": f"stream_event_{i}",
                        "timestamp": time.time(),
                        "data": f"event_data_{i}"
                    }
                    collected_events.append(event)
                
                context.set_output("collected_events", collected_events)
                context.set_output("batch_size", len(collected_events))
                
                return "process_batch"
            
            @state(cpu=2.0, memory=512.0)
            async def process_batch(self, context: Context):
                collected_events = context.get_output("collected_events", [])
                
                # Simulate batch processing
                await asyncio.sleep(0.2)
                
                processed_events = []
                for event in collected_events:
                    processed_event = {
                        "original_id": event["event_id"],
                        "processed_at": time.time(),
                        "processed_data": f"processed_{event['data']}"
                    }
                    processed_events.append(processed_event)
                
                context.set_output("processed_events", processed_events)
                context.set_output("processing_rate", len(processed_events) / 0.2)
                context.set_metric("batch_processing_time", 0.2)
                
                return None
        
        # Create stream processors with different batch sizes
        stream_processors = [
            StreamProcessorAgent("stream-processor-small", batch_size=2),
            StreamProcessorAgent("stream-processor-medium", batch_size=4),
            StreamProcessorAgent("stream-processor-large", batch_size=6)
        ]
        
        # Execute stream processing
        start_time = time.time()
        stream_results = await run_agents_parallel(stream_processors)
        total_time = time.time() - start_time
        
        # Verify stream processing
        assert len(stream_results) == 3
        
        total_events_processed = 0
        processing_rates = []
        
        for result in stream_results.values():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            
            processed_events = result.get_output("processed_events", [])
            batch_size = result.get_output("batch_size", 0)
            processing_rate = result.get_output("processing_rate", 0)
            
            assert len(processed_events) == batch_size
            total_events_processed += len(processed_events)
            processing_rates.append(processing_rate)
            
            # Verify event processing
            for processed_event in processed_events:
                assert "original_id" in processed_event
                assert "processed_at" in processed_event
                assert "processed_data" in processed_event
        
        # Verify total processing
        assert total_events_processed == 12  # 2 + 4 + 6
        
        # Verify processing rates
        assert all(rate > 0 for rate in processing_rates)
        
        # Stream processing should be efficient
        assert total_time <= 2.0  # Should complete within 2 seconds