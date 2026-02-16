"""Test all examples from the context and data documentation."""

import pytest
from pydantic import BaseModel

from puffinflow import Agent


@pytest.mark.asyncio
class TestContextAndDataExamples:
    """Test examples from context-and-data.ts documentation."""

    async def test_general_variables(self):
        """Test general variable usage example."""
        agent = Agent("general-variables-test")

        async def fetch_data(context):
            user_data = {"id": 123, "name": "Alice", "email": "alice@example.com"}
            context.set_variable("user", user_data)
            context.set_variable("count", 1250)
            return "process_data"

        async def process_data(context):
            user = context.get_variable("user")
            count = context.get_variable("count")
            context.set_variable(
                "processed", f"Processing {user['name']}, user {user['id']} of {count}"
            )

        agent.add_state("fetch_data", fetch_data)
        agent.add_state("process_data", process_data)

        result = await agent.run()

        # Verify data sharing
        user = result.get_variable("user")
        assert user["id"] == 123
        assert user["name"] == "Alice"
        assert user["email"] == "alice@example.com"
        assert result.get_variable("count") == 1250
        assert result.get_variable("processed") == "Processing Alice, user 123 of 1250"

    async def test_type_safe_variables(self):
        """Test type-safe variable usage example."""
        agent = Agent("type-safe-test")

        async def initialize(context):
            context.set_typed_variable("user_count", 100)  # Locked to int
            context.set_typed_variable("avg_score", 85.5)  # Locked to float
            return "update"

        async def update(context):
            context.set_typed_variable("user_count", 150)  # ✅ Works

            # Test that type errors would be caught
            try:
                context.set_typed_variable("user_count", "150")  # Should fail
                raise AssertionError("Should have raised TypeError")
            except (TypeError, ValueError):
                pass  # Expected

            count = context.get_typed_variable("user_count")  # Type param optional
            context.set_variable("count_message", f"Count: {count}")

        agent.add_state("initialize", initialize)
        agent.add_state("update", update)

        result = await agent.run()

        # Verify typed variables work correctly
        assert result.get_variable("user_count") == 150
        assert result.get_variable("avg_score") == 85.5
        assert result.get_variable("count_message") == "Count: 150"

    async def test_validated_data_with_pydantic(self):
        """Test validated data with Pydantic example."""

        class User(BaseModel):
            id: int
            name: str
            email: str  # Using str instead of EmailStr for simplicity
            age: int

        agent = Agent("validated-data-test")

        async def create_user(context):
            user = User(id=123, name="Alice", email="alice@example.com", age=28)
            context.set_validated_data("user", user)
            return "update_user"

        async def update_user(context):
            user = context.get_validated_data("user", User)
            user.age = 29
            context.set_validated_data("user", user)  # Re-validates

        agent.add_state("create_user", create_user)
        agent.add_state("update_user", update_user)

        result = await agent.run()

        # Verify validated data
        user = result.get_variable("user")
        assert user.id == 123
        assert user.name == "Alice"
        assert user.email == "alice@example.com"
        assert user.age == 29

    async def test_constants_and_configuration(self):
        """Test constants and configuration example."""
        agent = Agent("constants-test")

        async def setup(context):
            context.set_constant("api_url", "https://api.example.com")
            context.set_constant("max_retries", 3)
            return "use_config"

        async def use_config(context):
            url = context.get_constant("api_url")
            retries = context.get_constant("max_retries")
            context.set_variable("config_used", f"URL: {url}, Retries: {retries}")

            # Test that constants can't be changed
            try:
                context.set_constant("api_url", "different")  # Should fail
                raise AssertionError("Should have raised ValueError")
            except ValueError:
                pass  # Expected

        agent.add_state("setup", setup)
        agent.add_state("use_config", use_config)

        result = await agent.run()

        # Verify constants
        assert result.get_variable("api_url") == "https://api.example.com"
        assert result.get_variable("max_retries") == 3
        assert (
            result.get_variable("config_used")
            == "URL: https://api.example.com, Retries: 3"
        )

    async def test_secrets_management(self):
        """Test secrets management example."""
        agent = Agent("secrets-test")

        async def load_secrets(context):
            context.set_secret("api_key", "sk-1234567890abcdef")
            context.set_secret("db_password", "super_secure_password")
            return "use_secrets"

        async def use_secrets(context):
            api_key = context.get_secret("api_key")
            context.set_variable("key_preview", f"API key loaded: {api_key[:8]}...")

        agent.add_state("load_secrets", load_secrets)
        agent.add_state("use_secrets", use_secrets)

        result = await agent.run()

        # Verify secrets (note: we can access them in tests, but normally they'd be secured)
        assert result.get_variable("api_key") == "sk-1234567890abcdef"
        assert result.get_variable("db_password") == "super_secure_password"
        assert result.get_variable("key_preview") == "API key loaded: sk-12345..."

    async def test_cached_data_with_ttl(self):
        """Test cached data with TTL example."""
        agent = Agent("cache-test")

        async def cache_data(context):
            context.set_cached("session", {"user_id": 123}, ttl=300)  # 5 minutes
            context.set_cached("temp_result", {"data": "value"}, ttl=60)  # 1 minute

        async def use_cache(context):
            session = context.get_cached("session", default="EXPIRED")
            temp_result = context.get_cached("temp_result", default="EXPIRED")
            context.set_variable("session_status", f"Session: {session}")
            context.set_variable("temp_status", f"Temp: {temp_result}")
            # Also set the cached values as variables for the test
            context.set_variable("session", session)
            context.set_variable("temp_result", temp_result)

        agent.add_state("cache_data", cache_data)
        agent.add_state("use_cache", use_cache)

        result = await agent.run()

        # Verify cached data (should still be valid since we just set it)
        session = result.get_variable("session")
        assert session == {"user_id": 123}

        temp_result = result.get_variable("temp_result")
        assert temp_result == {"data": "value"}

    async def test_per_state_scratch_data(self):
        """Test per-state scratch data example."""
        agent = Agent("state-data-test")

        async def state_a(context):
            context.set_state("temp_data", [1, 2, 3])  # Only visible in state_a
            context.set_variable("shared", "visible to all")
            return "state_b"

        async def state_b(context):
            context.set_state("temp_data", {"key": "value"})  # Different from state_a
            shared = context.get_variable("shared")  # Can access shared data
            my_temp = context.get_state("temp_data")  # Gets state_b's data
            context.set_variable("state_b_temp", my_temp)
            context.set_variable("state_b_shared", shared)

        agent.add_state("state_a", state_a)
        agent.add_state("state_b", state_b)

        result = await agent.run()

        # Verify state-local data and shared data
        assert result.get_variable("shared") == "visible to all"
        assert result.get_variable("state_b_temp") == {"key": "value"}
        assert result.get_variable("state_b_shared") == "visible to all"

    async def test_output_data_management(self):
        """Test output data management example."""
        agent = Agent("output-test")

        async def calculate(context):
            orders = [{"amount": 100}, {"amount": 200}]
            total = sum(order["amount"] for order in orders)

            context.set_output("total_revenue", total)
            context.set_output("order_count", len(orders))
            return "summary"

        async def summary(context):
            revenue = context.get_output("total_revenue")
            count = context.get_output("order_count")
            context.set_variable(
                "summary_message", f"Revenue: ${revenue}, Orders: {count}"
            )
            # Also set outputs as variables for the test
            context.set_variable("total_revenue", revenue)
            context.set_variable("order_count", count)

        agent.add_state("calculate", calculate)
        agent.add_state("summary", summary)

        result = await agent.run()

        # Verify outputs
        assert result.get_variable("total_revenue") == 300
        assert result.get_variable("order_count") == 2
        assert result.get_variable("summary_message") == "Revenue: $300, Orders: 2"

    async def test_complete_order_processing_example(self):
        """Test complete order processing example."""

        class Order(BaseModel):
            id: int
            total: float
            customer_email: str

        agent = Agent("order-processing")

        async def setup(context):
            context.set_constant("tax_rate", 0.08)
            context.set_secret("payment_key", "pk_123456")

        async def process_order(context):
            # Validated order data
            order = Order(id=123, total=99.99, customer_email="user@example.com")
            context.set_validated_data("order", order)

            # Cache session
            context.set_cached("session", {"order_id": order.id}, ttl=3600)

            # Type-safe tracking
            context.set_typed_variable("amount_charged", order.total)

        async def send_confirmation(context):
            order = context.get_validated_data("order", Order)
            amount = context.get_typed_variable("amount_charged")  # Type param optional
            _payment_key = context.get_secret("payment_key")  # Used for validation

            # Final outputs
            context.set_output("order_id", order.id)
            context.set_output("amount_processed", amount)
            context.set_variable(
                "confirmation_message", f"✅ Order {order.id} completed: ${amount}"
            )

        agent.add_state("setup", setup)
        agent.add_state("process_order", process_order, dependencies=["setup"])
        agent.add_state(
            "send_confirmation", send_confirmation, dependencies=["process_order"]
        )

        result = await agent.run()

        # Verify complete order processing
        assert result.get_variable("tax_rate") == 0.08
        assert result.get_variable("payment_key") == "pk_123456"

        order = result.get_variable("order")
        assert order.id == 123
        assert order.total == 99.99
        assert order.customer_email == "user@example.com"

        session = result.get_variable("session")
        assert session["order_id"] == 123

        assert result.get_variable("amount_charged") == 99.99
        assert result.get_variable("order_id") == 123
        assert result.get_variable("amount_processed") == 99.99
        assert (
            result.get_variable("confirmation_message")
            == "✅ Order 123 completed: $99.99"
        )

    async def test_error_handling_for_invalid_types(self):
        """Test that type errors are properly handled."""
        agent = Agent("error-handling-test")

        async def test_type_errors(context):
            # Set initial typed variable
            context.set_typed_variable("count", 100)

            # Try to set wrong type - should fail
            try:
                context.set_typed_variable("count", "not a number")
                raise AssertionError("Should have raised an error")
            except (TypeError, ValueError):
                context.set_variable("type_error_caught", True)

            # Try to access non-existent constant (returns None by default)
            const_value = context.get_constant("non_existent")
            if const_value is None:
                context.set_variable("missing_constant_error_caught", True)

        agent.add_state("test_type_errors", test_type_errors)
        result = await agent.run()

        # Verify error handling
        assert result.get_variable("type_error_caught") is True
        assert result.get_variable("missing_constant_error_caught") is True
        assert result.get_variable("count") == 100  # Original value preserved

    async def test_enhanced_general_variables(self):
        """Test enhanced general variables example with defaults and complex data."""
        agent = Agent("enhanced-general-variables")

        async def fetch_enhanced_data(context):
            # Store complex data structures
            user_data = {"id": 123, "name": "Alice", "email": "alice@example.com"}
            context.set_variable("user", user_data)

            # Store primitive types
            context.set_variable("count", 1250)
            context.set_variable("is_premium", True)

            # Store lists and nested objects
            context.set_variable("tags", ["customer", "active", "premium"])
            context.set_variable(
                "metadata",
                {
                    "last_login": "2024-01-15",
                    "preferences": {"theme": "dark", "notifications": True},
                },
            )

        async def process_enhanced_data(context):
            # Retrieve data
            user = context.get_variable("user")
            is_premium = context.get_variable("is_premium")
            _tags = context.get_variable("tags")  # Retrieved but not used in processing
            _metadata = context.get_variable(
                "metadata"
            )  # Retrieved but not used in processing

            # Safe access with defaults
            region = context.get_variable("region", default="US")

            # Store processing results
            context.set_variable(
                "processing_result",
                {
                    "user_id": user["id"],
                    "processed_at": "2024-01-15T10:30:00Z",
                    "success": True,
                    "region": region,
                    "is_premium": is_premium,
                },
            )

        agent.add_state("fetch_enhanced_data", fetch_enhanced_data)
        agent.add_state(
            "process_enhanced_data",
            process_enhanced_data,
            dependencies=["fetch_enhanced_data"],
        )

        result = await agent.run()

        # Verify enhanced data handling
        user = result.get_variable("user")
        assert user["name"] == "Alice"
        assert result.get_variable("is_premium") is True
        assert "premium" in result.get_variable("tags")

        processing_result = result.get_variable("processing_result")
        assert processing_result["success"] is True
        assert processing_result["region"] == "US"  # Default value used

    async def test_enhanced_typed_variables(self):
        """Test enhanced typed variables with multiple types."""
        agent = Agent("enhanced-typed-test")

        async def initialize_enhanced(context):
            # Test multiple type locks
            context.set_typed_variable("user_count", 100)  # int
            context.set_typed_variable("avg_score", 85.5)  # float
            context.set_typed_variable("is_enabled", True)  # bool
            context.set_typed_variable("status", "active")  # str

        async def update_enhanced(context):
            # Valid type updates
            context.set_typed_variable("user_count", 150)
            context.set_typed_variable("avg_score", 92.3)
            context.set_typed_variable("is_enabled", False)

            # Test type safety
            enabled = context.get_typed_variable("is_enabled")
            count = context.get_typed_variable("user_count")

            # Safe arithmetic operations
            if not enabled:  # it's False now
                new_count = count + 10
                context.set_typed_variable("user_count", new_count)

        agent.add_state("initialize_enhanced", initialize_enhanced)
        agent.add_state(
            "update_enhanced", update_enhanced, dependencies=["initialize_enhanced"]
        )

        result = await agent.run()

        # Verify typed variable behavior
        assert result.get_variable("user_count") == 160  # 150 + 10
        assert result.get_variable("avg_score") == 92.3
        assert result.get_variable("is_enabled") is False
        assert result.get_variable("status") == "active"
