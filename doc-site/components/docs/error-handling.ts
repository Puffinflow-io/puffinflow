export const errorHandlingMarkdown = `# Error Handling & Resilience

When building workflows, things will go wrong. APIs will be down, networks will hiccup, and databases will be busy. Puffinflow helps you handle these problems gracefully so your workflows stay reliable and don't crash when something unexpected happens.

## Why Error Handling Matters

**Without error handling:**
- One failed API call crashes your entire workflow
- Temporary network issues cause permanent failures
- Your system becomes unreliable and frustrating to use
- You lose data and waste computing resources

**With Puffinflow's error handling:**
- Failed operations automatically retry and often succeed
- Temporary problems are handled without human intervention
- Your system stays stable even when dependencies fail
- Critical failures are captured for later investigation

## Part 1: The Basics (Start Here)

### Simple Retries

The most common error handling is just trying again when something fails:

\`\`\`python
import asyncio
import random
from puffinflow import Agent, state

agent = Agent("retry-demo")

# Try up to 3 times if this fails
@state(max_retries=3)
async def call_flaky_api(context):
    """This API fails sometimes, but usually works if you try again"""

    attempt = context.get_variable("attempts", 0) + 1
    context.set_variable("attempts", attempt)

    print(f"🌐 Calling API (attempt {attempt})...")

    # Simulate an API that fails 60% of the time
    if random.random() < 0.6:
        print(f"❌ API failed on attempt {attempt}")
        raise Exception("API temporarily unavailable")

    print(f"✅ API succeeded on attempt {attempt}!")
    context.set_variable("api_data", {"result": "success"})
    return None

# Add state to agent
agent.add_state("call_flaky_api", call_flaky_api)

# Run it and see the retries in action
async def main():
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

### Timeouts: Don't Wait Forever

Sometimes operations get stuck. Use timeouts to prevent this:

\`\`\`python
@state(timeout=10.0, max_retries=2)
async def might_hang(context):
    """
    This will timeout after 10 seconds if it gets stuck
    If it times out, it will retry up to 2 more times
    """
    print("⏱️ Starting operation that might hang...")

    # Simulate work that might take too long
    delay = random.uniform(5, 15)  # Sometimes over 10 seconds
    await asyncio.sleep(delay)

    print("✅ Operation completed!")
    return None

# Add state to agent
agent.add_state("might_hang", might_hang)
\`\`\`

### When to Use Basic Retries

- **API calls** that sometimes fail but usually work
- **File operations** that might hit temporary locks
- **Database queries** that occasionally timeout
- **Network operations** with intermittent connectivity

## Part 2: Smarter Retries

### Exponential Backoff: Give Services Time to Recover

Instead of retrying immediately, wait longer between each attempt:

\`\`\`python
from puffinflow.core.agent.state import RetryPolicy

# Smart retry policy that waits longer each time
smart_retry = RetryPolicy(
    max_retries=4,
    initial_delay=1.0,      # Wait 1 second first time
    exponential_base=2.0,   # Double each time: 1s, 2s, 4s, 8s
    jitter=True            # Add randomness to prevent thundering herd
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
    if random.random() < 0.7:
        print(f"❌ Service overloaded (attempt {attempt})")
        raise Exception("Service temporarily overloaded")

    print(f"✅ Service call succeeded!")
    context.set_variable("service_result", {"status": "completed"})
    return None

# Add state with smart retry policy
agent.add_state("overloaded_service", overloaded_service, retry_policy=smart_retry)
\`\`\`

### Different Retry Strategies for Different Problems

\`\`\`python
# For rate-limited APIs: wait consistently
rate_limit_retry = RetryPolicy(
    max_retries=3,
    initial_delay=2.0,
    exponential_base=1.0,  # No exponential growth: 2s, 2s, 2s
    jitter=False          # Consistent timing for rate limits
)

# For unreliable networks: aggressive retry
network_retry = RetryPolicy(
    max_retries=5,
    initial_delay=0.5,
    exponential_base=2.0,  # Quick escalation: 0.5s, 1s, 2s, 4s, 8s
    jitter=True           # Randomize to avoid conflicts
)

# For expensive operations: conservative retry
expensive_retry = RetryPolicy(
    max_retries=2,
    initial_delay=5.0,
    exponential_base=2.0,  # Slow and careful: 5s, 10s
    jitter=False          # Predictable for cost control
)

@state
async def rate_limited_api(context):
    """For APIs with strict rate limits"""
    print("🚦 Calling rate-limited API...")
    # Implementation here
    pass

@state
async def network_operation(context):
    """For flaky network connections"""
    print("🌐 Network operation...")
    # Implementation here
    pass

@state
async def expensive_computation(context):
    """For operations that cost money or resources"""
    print("💰 Expensive operation...")
    # Implementation here
    pass

# Add states with different retry policies
agent.add_state("rate_limited_api", rate_limited_api, retry_policy=rate_limit_retry)
agent.add_state("network_operation", network_operation, retry_policy=network_retry)
agent.add_state("expensive_computation", expensive_computation, retry_policy=expensive_retry)
\`\`\`

## Part 3: Advanced Protection

### Circuit Breaker: Stop Trying When Service is Down

When a service is completely down, stop trying for a while to let it recover:

\`\`\`python
@state
async def external_service_call(context):
    """
    Circuit breaker will:
    1. Try normally at first
    2. If it fails too many times, stop trying for a while
    3. After some time, try once to see if it's back
    4. If successful, resume normal operation
    """
    print("🔌 Calling external service...")

    # Simulate a service that might be completely down
    if random.random() < 0.8:  # High failure rate
        print("❌ Service is down")
        raise Exception("External service unavailable")

    print("✅ Service call succeeded!")
    context.set_variable("external_data", {"response": "success"})
    return None

# Add state with circuit breaker configuration
from puffinflow.core.reliability import CircuitBreakerConfig
circuit_config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)
retry_policy = RetryPolicy(max_retries=3)
agent.add_state("external_service_call", external_service_call,
                retry_policy=retry_policy, circuit_breaker_config=circuit_config)
\`\`\`

### Bulkhead: Don't Let One Problem Affect Everything

Isolate different types of operations so problems in one area don't spread:

\`\`\`python
@state
async def database_query(context):
    """
    Bulkhead ensures that:
    - Database problems don't affect API calls
    - Limited number of concurrent database connections
    - Other operations can continue even if database is slow
    """
    print("🗄️ Running database query...")

    # Simulate database that might be slow or unavailable
    if random.random() < 0.4:
        print("❌ Database connection failed")
        raise Exception("Database timeout")

    print("✅ Database query completed!")
    context.set_variable("query_result", {"rows": 42})
    return None

@state
async def file_processing(context):
    """
    This runs in a separate bulkhead from database operations
    So file problems won't affect database operations and vice versa
    """
    print("📁 Processing file...")

    # Simulate file operation
    if random.random() < 0.3:
        print("❌ File processing failed")
        raise Exception("File access denied")

    print("✅ File processed!")
    context.set_variable("file_result", {"processed": True})
    return None

# Add states with bulkhead configuration
from puffinflow.core.reliability import BulkheadConfig
db_bulkhead = BulkheadConfig(name="database", max_concurrent=2)
file_bulkhead = BulkheadConfig(name="file_ops", max_concurrent=3)

agent.add_state("database_query", database_query,
                retry_policy=RetryPolicy(max_retries=2), bulkhead_config=db_bulkhead)
agent.add_state("file_processing", file_processing,
                retry_policy=RetryPolicy(max_retries=3), bulkhead_config=file_bulkhead)
\`\`\`

### Dead Letter Queue: Save Failed Operations for Later

When all else fails, save the operation for manual investigation:

\`\`\`python
@agent.state(max_retries=3, dead_letter=True)
async def critical_payment_processing(context):
    """
    If this fails after all retries:
    - The failure is saved in a "dead letter queue"
    - You can investigate and fix the problem later
    - No payment data is lost
    """
    payment_id = context.get_variable("payment_id")
    attempt = context.get_variable("payment_attempts", 0) + 1
    context.set_variable("payment_attempts", attempt)

    print(f"💳 Processing payment {payment_id} (attempt {attempt})...")

    # Simulate payment processing that might fail
    if random.random() < 0.9:  # Very high failure rate for demo
        print(f"❌ Payment processing failed")
        raise Exception("Payment gateway error")

    print(f"✅ Payment {payment_id} processed successfully!")
    context.set_variable("payment_result", {"status": "completed"})
    return None
\`\`\`

## Part 4: Priority and Coordination

### Priority: Important Things First

Make sure critical operations get resources even when the system is busy:

\`\`\`python
from puffinflow import Priority

@agent.state(priority=Priority.CRITICAL, max_retries=5)
async def emergency_alert(context):
    """
    Critical priority means:
    - This runs before normal priority tasks
    - Gets resources even when system is busy
    - More aggressive retry policy
    """
    print("🚨 Sending emergency alert...")

    if random.random() < 0.3:
        print("❌ Alert delivery failed")
        raise Exception("Alert system unavailable")

    print("✅ Emergency alert sent!")
    return None

@agent.state(priority=Priority.LOW, max_retries=1)
async def background_cleanup(context):
    """
    Low priority means:
    - Runs when system isn't busy
    - Fewer retries (not critical)
    - Won't interfere with important work
    """
    print("🧹 Running background cleanup...")

    if random.random() < 0.5:
        print("❌ Cleanup failed (not critical)")
        raise Exception("Cleanup interrupted")

    print("✅ Cleanup completed!")
    return None
\`\`\`

## Part 5: Complete Real-World Example

Here's a comprehensive example showing multiple error handling techniques:

\`\`\`python
import asyncio
import random
from puffinflow import Agent, Priority
from puffinflow.core.agent.base import RetryPolicy

agent = Agent("ecommerce-order")

# Smart retry for payment processing
payment_retry = RetryPolicy(
    max_retries=3,
    initial_delay=2.0,
    exponential_base=1.5,
    jitter=True,
    dead_letter_on_max_retries=True  # Save failed payments
)

@agent.state(timeout=5.0, max_retries=2)
async def validate_order(context):
    """Quick validation with short timeout"""
    order_id = context.get_variable("order_id", "ORD-12345")
    print(f"✅ Validating order {order_id}...")

    # Fast operation, rarely fails
    if random.random() < 0.1:
        raise Exception("Invalid order data")

    context.set_variable("validation_status", "valid")
    return "check_inventory"

@agent.state(timeout=15.0, max_retries=3, bulkhead=True)
async def check_inventory(context):
    """Check inventory with bulkhead isolation"""
    order_id = context.get_variable("order_id", "ORD-12345")
    print(f"📦 Checking inventory for {order_id}...")

    # Inventory service can be slow
    if random.random() < 0.3:
        raise Exception("Inventory service timeout")

    context.set_variable("inventory_status", "available")
    return "process_payment"

@agent.state(
    retry_policy=payment_retry,
    priority=Priority.HIGH,
    circuit_breaker=True,
    dead_letter=True
)
async def process_payment(context):
    """Critical payment processing with all protections"""
    order_id = context.get_variable("order_id", "ORD-12345")
    amount = context.get_variable("amount", 99.99)

    attempt = context.get_variable("payment_attempts", 0) + 1
    context.set_variable("payment_attempts", attempt)

    print(f"💳 Processing payment for {order_id}: \${amount} (attempt {attempt})...")

    # Payment processing with various failure modes
    if random.random() < 0.6:
        error_types = [
            "Card declined",
            "Payment gateway timeout",
            "Insufficient funds",
            "Bank communication error"
        ]
        error = random.choice(error_types)
        print(f"❌ Payment failed: {error}")
        raise Exception(f"Payment error: {error}")

    print(f"✅ Payment successful for {order_id}!")
    context.set_variable("payment_status", "completed")
    return "send_confirmation"

@agent.state(timeout=10.0, max_retries=2, priority=Priority.NORMAL)
async def send_confirmation(context):
    """Send confirmation with moderate retry"""
    order_id = context.get_variable("order_id", "ORD-12345")
    print(f"📧 Sending confirmation for {order_id}...")

    # Email service occasionally fails
    if random.random() < 0.2:
        raise Exception("Email service unavailable")

    print(f"✅ Confirmation sent for {order_id}!")
    context.set_output("order_completed", True)
    context.set_output("order_id", order_id)
    return None

# Run the complete order process
async def main():
    initial_data = {
        "order_id": "ORD-67890",
        "amount": 149.99
    }

    result = await agent.run(
        initial_state="validate_order",
        initial_context=initial_data
    )

    if result.get_output("order_completed"):
        print(f"\\n🎉 Order {result.get_output('order_id')} completed successfully!")
    else:
        print("\\n❌ Order processing failed")

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

## Quick Decision Guide

### How Many Retries Should I Use?

- **Quick operations** (< 1 second): 2-3 retries
- **Normal operations** (1-10 seconds): 3-5 retries
- **Slow operations** (10+ seconds): 1-3 retries
- **Expensive operations** (costs money/resources): 1-2 retries

### When to Use Advanced Features?

**Circuit Breaker**: When calling external services that might go down completely

**Bulkhead**: When you have different types of operations (database, files, APIs) that shouldn't affect each other

**Dead Letter Queue**: For critical operations where you can't afford to lose data

**Priority**: When some operations are much more important than others

### Common Patterns

\`\`\`python
# Pattern 1: Simple API call
@agent.state(timeout=10.0, max_retries=3)
async def basic_api_call(context):
    pass

# Pattern 2: Critical operation
@agent.state(
    priority=Priority.HIGH,
    max_retries=5,
    dead_letter=True
)
async def critical_operation(context):
    pass

# Pattern 3: External service
@agent.state(
    timeout=15.0,
    max_retries=3,
    circuit_breaker=True,
    bulkhead=True
)
async def external_service_call(context):
    pass

# Pattern 4: Database operation
@agent.state(
    timeout=30.0,
    max_retries=2,
    bulkhead=True
)
async def database_operation(context):
    pass

# Pattern 5: Background task
@agent.state(
    priority=Priority.LOW,
    max_retries=1,
    timeout=300.0
)
async def background_task(context):
    pass
\`\`\`

## Tips for Beginners

1. **Start simple** - Begin with just \`max_retries\` and \`timeout\`
2. **Add timeouts always** - Never let operations hang forever
3. **Use retries for network calls** - APIs and external services fail often
4. **Monitor your dead letter queues** - Check what's failing and why
5. **Test with failures** - Simulate errors to see how your error handling works
6. **Don't over-retry expensive operations** - Some failures should fail fast

## What Each Feature Protects Against

- **max_retries**: Temporary failures, network hiccups
- **timeout**: Hung operations, slow responses
- **circuit_breaker**: Cascade failures, completely down services
- **bulkhead**: Resource exhaustion, one failure affecting everything
- **dead_letter**: Data loss, operations that need manual intervention
- **priority**: Important operations being delayed by less important ones

Error handling makes your workflows robust and reliable. Start with the basics and add advanced features as your needs grow!
`.trim();
