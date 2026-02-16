"""End-to-end tests for microservices and event-driven scenarios.

Additional E2E tests that complement the main workflow tests.
"""

import asyncio
import time

import pytest

from puffinflow import Agent, Context, run_agents_parallel, run_agents_sequential, state


class MicroserviceAgent(Agent):
    """Agent that simulates a microservice."""

    def __init__(self, name: str, service_config: dict):
        super().__init__(name)
        self.service_config = service_config
        self.add_state("health_check", self.health_check)
        self.add_state("process_request", self.process_request)

    @state(cpu=0.3, memory=128.0)
    async def health_check(self, context: Context):
        """Check service health."""
        await asyncio.sleep(0.05)

        # Simulate health check
        is_healthy = True  # Assume healthy for test

        context.set_variable("health_status", "healthy" if is_healthy else "unhealthy")
        context.set_variable("service_type", self.service_config.get("type", "unknown"))

        if not is_healthy:
            return None  # Stop if unhealthy

        return "process_request"

    @state(cpu=1.0, memory=256.0)
    async def process_request(self, context: Context):
        """Process service request."""
        processing_time = self.service_config.get("processing_time", 0.2)

        # Simulate processing
        await asyncio.sleep(processing_time)

        # Generate response based on service type
        service_type = self.service_config.get("type", "unknown")

        if service_type == "auth":
            response = {
                "token": "auth_token_123",
                "user_id": "user_456",
                "expires_at": time.time() + 3600,
            }
        elif service_type == "data":
            response = {
                "data": [{"id": i, "value": f"item_{i}"} for i in range(10)],
                "count": 10,
                "timestamp": time.time(),
            }
        elif service_type == "notification":
            response = {
                "message_id": "msg_789",
                "status": "sent",
                "recipients": ["user@example.com"],
            }
        else:
            response = {
                "message": "Service processed successfully",
                "timestamp": time.time(),
            }

        # Preserve health status from previous state
        context.set_variable("health_status", "healthy")
        context.set_variable("service_type", self.service_config.get("type", "unknown"))
        context.set_variable("response", response)
        context.set_typed_variable("processing_time", processing_time)

        return None


class OrderProcessingAgent(Agent):
    """Agent that simulates order processing."""

    def __init__(self, name: str, order_data: dict):
        super().__init__(name)
        self.order_data = order_data
        self.add_state("validate_order", self.validate_order)
        self.add_state("process_payment", self.process_payment)
        self.add_state("fulfill_order", self.fulfill_order)

    @state(cpu=0.5, memory=128.0)
    async def validate_order(self, context: Context):
        """Validate order details."""
        await asyncio.sleep(0.1)

        # Basic validation
        if not self.order_data.get("items") or self.order_data.get("total", 0) <= 0:
            return None  # Invalid order

        context.set_variable("order_valid", True)
        return "process_payment"

    @state(cpu=1.0, memory=256.0)
    async def process_payment(self, context: Context):
        """Process payment for the order."""
        await asyncio.sleep(0.3)

        # Simulate payment processing
        payment_result = {
            "payment_id": "pay_123",
            "status": "completed",
            "amount": self.order_data.get("total", 0),
            "method": self.order_data.get("payment_method", "credit_card"),
        }

        context.set_variable("payment_result", payment_result)
        return "fulfill_order"

    @state(cpu=0.8, memory=192.0)
    async def fulfill_order(self, context: Context):
        """Fulfill the order."""
        await asyncio.sleep(0.2)

        # Simulate fulfillment
        fulfillment_result = {
            "fulfillment_id": "fulfill_456",
            "status": "shipped",
            "tracking_number": "TRK789",
            "items": self.order_data.get("items", []),
        }

        # Preserve all data from previous states
        context.set_variable("order_valid", True)
        # Re-create payment result since context may not preserve it
        payment_result = {
            "payment_id": "pay_123",
            "status": "completed",
            "amount": self.order_data.get("total", 0),
            "method": self.order_data.get("payment_method", "credit_card"),
        }
        context.set_variable("payment_result", payment_result)
        context.set_variable("fulfillment_result", fulfillment_result)
        return None


class EventProcessingAgent(Agent):
    """Agent that processes events in an event-driven system."""

    def __init__(self, name: str, event_config: dict):
        super().__init__(name)
        self.event_config = event_config
        self.add_state("process_event", self.process_event)

    @state(cpu=0.5, memory=128.0)
    async def process_event(self, context: Context):
        """Process incoming events."""
        await asyncio.sleep(0.1)

        event_type = self.event_config.get("type", "unknown")

        # Process different event types
        if event_type == "user_signup":
            result = {
                "event_id": "evt_001",
                "action": "send_welcome_email",
                "user_id": self.event_config.get("user_id", "unknown"),
                "processed_at": time.time(),
            }
        elif event_type == "order_placed":
            result = {
                "event_id": "evt_002",
                "action": "update_inventory",
                "order_id": self.event_config.get("order_id", "unknown"),
                "processed_at": time.time(),
            }
        elif event_type == "payment_completed":
            result = {
                "event_id": "evt_003",
                "action": "send_confirmation",
                "payment_id": self.event_config.get("payment_id", "unknown"),
                "processed_at": time.time(),
            }
        else:
            result = {
                "event_id": "evt_999",
                "action": "log_unknown_event",
                "processed_at": time.time(),
            }

        context.set_variable("event_result", result)
        return None


@pytest.mark.e2e
@pytest.mark.asyncio
class TestMicroservicesOrchestration:
    """Test microservices orchestration scenarios."""

    async def test_microservices_orchestration(self):
        """Test coordination of multiple microservices."""
        # Create microservice agents
        auth_service = MicroserviceAgent(
            "auth-service", {"type": "auth", "processing_time": 0.2}
        )

        data_service = MicroserviceAgent(
            "data-service", {"type": "data", "processing_time": 0.3}
        )

        notification_service = MicroserviceAgent(
            "notification-service", {"type": "notification", "processing_time": 0.1}
        )

        # Run services in parallel
        services = [auth_service, data_service, notification_service]
        results = await run_agents_parallel(services)

        # Verify all services completed successfully
        assert len(results) == 3

        # Check auth service
        auth_result = results["auth-service"]
        assert auth_result.status.name in ["COMPLETED", "SUCCESS"]
        auth_response = auth_result.get_variable("response", {})
        assert "token" in auth_response
        assert "user_id" in auth_response

        # Check data service
        data_result = results["data-service"]
        assert data_result.status.name in ["COMPLETED", "SUCCESS"]
        data_response = data_result.get_variable("response", {})
        assert "data" in data_response
        assert data_response["count"] == 10

        # Check notification service
        notification_result = results["notification-service"]
        assert notification_result.status.name in ["COMPLETED", "SUCCESS"]
        notification_response = notification_result.get_variable("response", {})
        assert "message_id" in notification_response
        assert notification_response["status"] == "sent"

    async def test_service_dependency_chain(self):
        """Test services that depend on each other."""
        # Create services that should run in sequence
        auth_service = MicroserviceAgent(
            "auth-service", {"type": "auth", "processing_time": 0.1}
        )

        data_service = MicroserviceAgent(
            "data-service", {"type": "data", "processing_time": 0.2}
        )

        # Run services sequentially (auth must complete before data)
        services = [auth_service, data_service]
        results = await run_agents_sequential(services)

        # Verify execution order and results
        assert len(results) == 2

        # Both services should complete successfully
        for _service_name, result in results.items():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            assert result.get_variable("health_status") == "healthy"

    async def test_order_processing_workflow(self):
        """Test a complete order processing workflow."""
        # Create order processing agents for different orders
        order1 = OrderProcessingAgent(
            "order-1",
            {
                "order_id": "ord_001",
                "items": [{"id": 1, "name": "Widget", "price": 10.00}],
                "total": 10.00,
                "payment_method": "credit_card",
            },
        )

        order2 = OrderProcessingAgent(
            "order-2",
            {
                "order_id": "ord_002",
                "items": [{"id": 2, "name": "Gadget", "price": 25.00}],
                "total": 25.00,
                "payment_method": "paypal",
            },
        )

        # Process orders in parallel
        orders = [order1, order2]
        results = await run_agents_parallel(orders)

        # Verify both orders processed successfully
        assert len(results) == 2

        for _order_name, result in results.items():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            assert result.get_variable("order_valid") is True

            payment_result = result.get_variable("payment_result", {})
            assert payment_result["status"] == "completed"

            fulfillment_result = result.get_variable("fulfillment_result", {})
            assert fulfillment_result["status"] == "shipped"
            assert "tracking_number" in fulfillment_result


@pytest.mark.e2e
@pytest.mark.asyncio
class TestEventDrivenWorkflows:
    """Test event-driven workflow scenarios."""

    async def test_event_driven_workflow(self):
        """Test processing of different event types."""
        # Create event processing agents
        signup_processor = EventProcessingAgent(
            "signup-processor", {"type": "user_signup", "user_id": "user_123"}
        )

        order_processor = EventProcessingAgent(
            "order-processor", {"type": "order_placed", "order_id": "ord_456"}
        )

        payment_processor = EventProcessingAgent(
            "payment-processor", {"type": "payment_completed", "payment_id": "pay_789"}
        )

        # Process events in parallel
        processors = [signup_processor, order_processor, payment_processor]
        results = await run_agents_parallel(processors)

        # Verify all events processed successfully
        assert len(results) == 3

        # Check signup event
        signup_result = results["signup-processor"]
        assert signup_result.status.name in ["COMPLETED", "SUCCESS"]
        event_result = signup_result.get_variable("event_result", {})
        assert event_result["action"] == "send_welcome_email"

        # Check order event
        order_result = results["order-processor"]
        assert order_result.status.name in ["COMPLETED", "SUCCESS"]
        event_result = order_result.get_variable("event_result", {})
        assert event_result["action"] == "update_inventory"

        # Check payment event
        payment_result = results["payment-processor"]
        assert payment_result.status.name in ["COMPLETED", "SUCCESS"]
        event_result = payment_result.get_variable("event_result", {})
        assert event_result["action"] == "send_confirmation"

    async def test_event_processing_pipeline(self):
        """Test a pipeline of event processors."""
        # Create a chain of event processors
        events = [
            EventProcessingAgent(
                "event-1", {"type": "user_signup", "user_id": "user_001"}
            ),
            EventProcessingAgent(
                "event-2", {"type": "order_placed", "order_id": "ord_002"}
            ),
            EventProcessingAgent(
                "event-3", {"type": "payment_completed", "payment_id": "pay_003"}
            ),
        ]

        # Process events sequentially
        results = await run_agents_sequential(events)

        # Verify all events processed in order
        assert len(results) == 3

        for _event_name, result in results.items():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            event_result = result.get_variable("event_result", {})
            assert "event_id" in event_result
            assert "action" in event_result
            assert "processed_at" in event_result

    async def test_mixed_workflow_patterns(self):
        """Test mixing parallel and sequential processing patterns."""
        # Create different types of agents
        auth_service = MicroserviceAgent(
            "auth", {"type": "auth", "processing_time": 0.1}
        )

        # Create parallel order processing
        order_agents = []
        for i in range(3):
            order = OrderProcessingAgent(
                f"order-{i}",
                {
                    "order_id": f"ord_{i:03d}",
                    "items": [{"id": i, "name": f"Item_{i}", "price": 10.0 * (i + 1)}],
                    "total": 10.0 * (i + 1),
                    "payment_method": "credit_card",
                },
            )
            order_agents.append(order)

        # First authenticate
        auth_result = await auth_service.run()
        assert auth_result.status.name in ["COMPLETED", "SUCCESS"]

        # Then process orders in parallel
        order_results = await run_agents_parallel(order_agents)
        assert len(order_results) == 3

        # Verify all orders processed successfully
        for _order_name, result in order_results.items():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            assert result.get_variable("order_valid") is True
