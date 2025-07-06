export const errorHandlingMarkdown = `# Error Handling & Resilience

Building robust workflows means expecting things to go wrong and handling failures gracefully. Puffinflow provides comprehensive error handling, retry mechanisms, circuit breakers, and recovery patterns that can be configured directly in state decorators to create resilient, production-ready workflows.

## Understanding Error Handling Needs

Before diving into implementation, it's crucial to understand what types of failures your workflow might encounter and choose the appropriate resilience patterns.

### Types of Failures

| Failure Type | Characteristics | Recommended Approach |
|---|---|---|
| **Transient** | Temporary network issues, momentary service unavailability | Retry with backoff |
| **Timeout** | Operations taking too long | Timeout + retry |
| **Resource Exhaustion** | CPU, memory, or connection limits reached | Rate limiting + bulkheads |
| **Cascade Failures** | One service failure affecting others | Circuit breakers |
| **Persistent** | Configuration errors, invalid data | Dead letter queues |

### Decision Framework

Ask yourself these questions to determine your error handling strategy:

1. **How critical is this operation?** → Priority level
2. **How likely is it to fail transiently?** → Retry configuration
3. **How expensive is it to retry?** → Backoff strategy
4. **Could it cause cascade failures?** → Circuit breaker
5. **Does it compete for limited resources?** → Bulkhead isolation
6. **What happens if it ultimately fails?** → Dead letter queue

---

## Basic Retry Configuration

Start with simple retry mechanisms for operations that might fail transiently.

### When to Use Retries

- **Network operations** that might experience temporary connectivity issues
- **External API calls** that could hit rate limits or temporary outages
- **Database operations** that might encounter lock timeouts
- **File I/O operations** that could face temporary permission issues

### Simple Retry Setup

\`\`\`python
import asyncio
import random
from puffinflow import Agent
from puffinflow.decorators import state

agent = Agent("error-handling-agent")

# For operations with low failure rates
@state(max_retries=3)
async def stable_api_call(context):
    print("🌐 Calling stable API...")

    attempts = context.get_state("attempts", 0) + 1
    context.set_state("attempts", attempts)

    # 20% failure rate - mostly reliable
    if random.random() < 0.2:
        raise Exception(f"Temporary API error (attempt {attempts})")

    print("✅ API call succeeded")
    context.set_variable("api_result", "success")

# For more unreliable operations
@state(max_retries=5)
async def flaky_service_call(context):
    print("🎲 Calling flaky service...")

    attempts = context.get_state("flaky_attempts", 0) + 1
    context.set_state("flaky_attempts", attempts)

    # 60% failure rate - needs more retries
    if random.random() < 0.6:
        raise Exception(f"Service unavailable (attempt {attempts})")

    print("✅ Flaky service succeeded")
    context.set_variable("flaky_result", "success")

# For critical operations that should rarely retry
@state(max_retries=1)
async def expensive_operation(context):
    print("💰 Running expensive operation...")

    attempts = context.get_state("expensive_attempts", 0) + 1
    context.set_state("expensive_attempts", attempts)

    # Only retry once due to cost
    if attempts == 1 and random.random() < 0.3:
        raise Exception("Expensive operation failed on first try")

    print("✅ Expensive operation completed")
    context.set_variable("expensive_result", "success")
\`\`\`

### Choosing Retry Counts

| Operation Type | Suggested Retries | Reasoning |
|---|---|---|
| **Stable APIs** | 2-3 | Low failure rate, quick recovery |
| **Flaky Services** | 3-5 | Higher failure rate, may need multiple attempts |
| **Expensive Operations** | 1-2 | High cost, limited retry budget |
| **Critical Systems** | 5-10 | Must succeed, willing to wait |

---

## Custom Retry Policies

When basic retry counts aren't enough, create custom retry policies with sophisticated backoff strategies.

### Understanding Backoff Strategies

\`\`\`python
import asyncio
import time
from puffinflow import Agent
from puffinflow.decorators import state
from puffinflow.core.agent.base import RetryPolicy

agent = Agent("retry-policy-agent")

# Aggressive retry for network operations
network_retry = RetryPolicy(
    max_retries=5,
    initial_delay=0.5,       # Start with 500ms
    exponential_base=2.0,    # Double each time: 0.5s, 1s, 2s, 4s, 8s
    jitter=True             # Add randomization to prevent thundering herd
)

# Conservative retry for database operations
database_retry = RetryPolicy(
    max_retries=3,
    initial_delay=1.0,       # Start with 1 second
    exponential_base=1.5,    # Slower growth: 1s, 1.5s, 2.25s
    jitter=False            # Predictable timing for database connections
)

# Linear retry for API rate limits
rate_limit_retry = RetryPolicy(
    max_retries=4,
    initial_delay=2.0,       # Start with 2 seconds
    exponential_base=1.0,    # No exponential growth: 2s, 2s, 2s, 2s
    jitter=False            # Consistent for rate limit windows
)

@state(retry_policy=network_retry)
async def network_operation(context):
    print("🌐 Network operation with aggressive retry...")

    start_time = context.get_state("start_time") or time.time()
    context.set_state("start_time", start_time)
    attempt = context.get_state("attempts", 0) + 1
    context.set_state("attempts", attempt)

    print(f"   Attempt #{attempt} (elapsed: {time.time() - start_time:.1f}s)")

    # Simulate network issues that resolve over time
    if attempt < 3:
        raise Exception(f"Network timeout (attempt {attempt})")

    print("✅ Network operation succeeded!")
    context.set_variable("network_result", "success")

@state(retry_policy=database_retry)
async def database_operation(context):
    print("🗄️ Database operation with conservative retry...")

    start_time = context.get_state("db_start") or time.time()
    context.set_state("db_start", start_time)
    attempt = context.get_state("db_attempts", 0) + 1
    context.set_state("db_attempts", attempt)

    print(f"   DB attempt #{attempt} (elapsed: {time.time() - start_time:.1f}s)")

    # Database issues that need time to resolve
    if attempt < 2:
        raise Exception(f"Database lock timeout (attempt {attempt})")

    print("✅ Database operation succeeded!")
    context.set_variable("database_result", "success")

@state(retry_policy=rate_limit_retry)
async def rate_limited_api(context):
    print("🚦 Rate-limited API with linear retry...")

    start_time = context.get_state("api_start") or time.time()
    context.set_state("api_start", start_time)
    attempt = context.get_state("api_attempts", 0) + 1
    context.set_state("api_attempts", attempt)

    print(f"   API attempt #{attempt} (elapsed: {time.time() - start_time:.1f}s)")

    # Rate limit that needs consistent spacing
    if attempt < 3:
        raise Exception(f"Rate limit exceeded (attempt {attempt})")

    print("✅ Rate-limited API succeeded!")
    context.set_variable("api_result", "success")
\`\`\`

### When to Use Each Backoff Strategy

| Strategy | When to Use | Example Use Cases |
|---|---|---|
| **Exponential** | Transient failures, network issues | External APIs, microservices |
| **Linear** | Rate limiting, resource contention | API quotas, database connections |
| **Fixed** | Predictable recovery times | Scheduled maintenance windows |

### Jitter: Preventing Thundering Herd

\`\`\`python
# With jitter - recommended for most cases
jitter_retry = RetryPolicy(
    max_retries=3,
    initial_delay=1.0,
    exponential_base=2.0,
    jitter=True  # Adds ±25% randomization
)

# Without jitter - use when timing is critical
precise_retry = RetryPolicy(
    max_retries=3,
    initial_delay=1.0,
    exponential_base=2.0,
    jitter=False  # Exact timing: 1s, 2s, 4s
)

@state(retry_policy=jitter_retry)
async def distributed_operation(context):
    """Multiple instances won't retry at exactly the same time"""
    print("🌍 Distributed operation with jitter...")
    # Implementation here

@state(retry_policy=precise_retry)
async def synchronized_operation(context):
    """Predictable retry timing for coordination"""
    print("⏰ Synchronized operation with precise timing...")
    # Implementation here
\`\`\`

---

## Timeout Configuration

Timeouts prevent operations from hanging indefinitely and ensure system responsiveness.

### When to Use Timeouts

- **External API calls** that might hang
- **Database queries** that could deadlock
- **File operations** on slow storage
- **Any operation** with unpredictable duration

### Timeout Strategies

\`\`\`python
import asyncio
import random
from puffinflow import Agent
from puffinflow.decorators import state

agent = Agent("timeout-agent")

@state(timeout=5.0)  # Short timeout for quick operations
async def quick_health_check(context):
    print("⚡ Quick health check (5s limit)...")

    try:
        # Simulate health check
        await asyncio.sleep(random.uniform(1.0, 8.0))
        print("✅ Health check passed")
        context.set_variable("health_status", "healthy")
    except asyncio.TimeoutError:
        print("❌ Health check timed out")
        context.set_variable("health_status", "timeout")

@state(timeout=30.0, max_retries=2)  # Medium timeout with retries
async def data_processing_task(context):
    print("📊 Data processing (30s limit)...")

    attempt = context.get_state("processing_attempts", 0) + 1
    context.set_state("processing_attempts", attempt)

    try:
        # Simulate data processing that might get stuck
        processing_time = random.uniform(10.0, 45.0)
        await asyncio.sleep(processing_time)
        print("✅ Data processing completed")
        context.set_variable("processing_result", "completed")
    except asyncio.TimeoutError:
        print(f"❌ Data processing timed out (attempt {attempt})")
        if attempt >= 3:  # Final attempt
            context.set_variable("processing_result", "failed")

@state(timeout=120.0)  # Long timeout for expensive operations
async def ml_model_training(context):
    print("🤖 ML model training (120s limit)...")

    try:
        # Simulate model training
        training_time = random.uniform(60.0, 180.0)
        await asyncio.sleep(training_time)
        print("✅ Model training completed")
        context.set_variable("model_status", "trained")
    except asyncio.TimeoutError:
        print("❌ Model training timed out")
        context.set_variable("model_status", "timeout")

@state(timeout=2.0, max_retries=3)  # Very short timeout for real-time operations
async def real_time_api_call(context):
    print("⚡ Real-time API call (2s limit)...")

    attempt = context.get_state("rt_attempts", 0) + 1
    context.set_state("rt_attempts", attempt)

    try:
        # Real-time operations must be fast
        response_time = random.uniform(0.5, 4.0)
        await asyncio.sleep(response_time)
        print("✅ Real-time API responded")
        context.set_variable("rt_result", "success")
    except asyncio.TimeoutError:
        print(f"❌ Real-time API timed out (attempt {attempt})")
        if attempt >= 4:  # Final attempt
            context.set_variable("rt_result", "timeout")
\`\`\`

### Choosing Timeout Values

| Operation Type | Suggested Timeout | Considerations |
|---|---|---|
| **Health Checks** | 2-5 seconds | Must be fast for monitoring |
| **API Calls** | 10-30 seconds | Balance responsiveness vs success |
| **Database Queries** | 5-15 seconds | Prevent deadlock accumulation |
| **File Operations** | 30-60 seconds | Account for disk/network speed |
| **ML/AI Operations** | 60-300 seconds | Complex operations need time |

---

## Priority-Based Error Handling

Use priorities to ensure critical operations get resources and attention during system stress.

### Understanding Priority Levels

\`\`\`python
import asyncio
import random
from puffinflow import Agent
from puffinflow.decorators import state, critical_state
from puffinflow.core.agent.state import Priority

agent = Agent("priority-agent", max_concurrent=2)

@state(priority=Priority.CRITICAL, max_retries=5, timeout=60.0)
async def critical_system_operation(context):
    """
    Critical operations that must succeed:
    - System health checks
    - Security validations
    - Data consistency checks
    """
    print("🔥 Critical system operation...")

    attempt = context.get_state("critical_attempts", 0) + 1
    context.set_state("critical_attempts", attempt)
    print(f"   Critical attempt #{attempt} (CRITICAL PRIORITY)")

    # Critical operations get aggressive retry
    if attempt <= 3:
        raise Exception(f"Critical system not ready (attempt {attempt})")

    print("✅ Critical system operational!")
    context.set_variable("critical_status", "operational")

@state(priority=Priority.HIGH, max_retries=3, timeout=30.0)
async def user_facing_operation(context):
    """
    High priority operations affecting users:
    - User authentication
    - Payment processing
    - Core business logic
    """
    print("👤 User-facing operation...")

    attempt = context.get_state("user_attempts", 0) + 1
    context.set_state("user_attempts", attempt)
    print(f"   User operation attempt #{attempt} (HIGH PRIORITY)")

    # High priority gets reasonable retry
    if random.random() < 0.4:
        raise Exception(f"User operation failed (attempt {attempt})")

    print("✅ User operation completed!")
    context.set_variable("user_status", "completed")

@state(priority=Priority.NORMAL, max_retries=2, timeout=15.0)
async def business_logic_operation(context):
    """
    Normal priority operations:
    - Data processing
    - Report generation
    - Routine maintenance
    """
    print("📊 Business logic operation...")

    attempt = context.get_state("business_attempts", 0) + 1
    context.set_state("business_attempts", attempt)
    print(f"   Business attempt #{attempt} (NORMAL PRIORITY)")

    # Normal operations get standard retry
    if attempt < 2:
        raise Exception(f"Business logic issue (attempt {attempt})")

    print("✅ Business logic completed!")
    context.set_variable("business_status", "completed")

@state(priority=Priority.LOW, max_retries=1, timeout=10.0)
async def background_operation(context):
    """
    Low priority operations that can fail:
    - Analytics collection
    - Cache warming
    - Optional optimizations
    """
    print("🧹 Background operation...")

    attempt = context.get_state("background_attempts", 0) + 1
    context.set_state("background_attempts", attempt)
    print(f"   Background attempt #{attempt} (LOW PRIORITY)")

    # Background operations are expendable
    if random.random() < 0.7:  # High failure rate is acceptable
        raise Exception(f"Background operation failed (attempt {attempt})")

    print("✅ Background operation completed!")
    context.set_variable("background_status", "completed")

@state
async def priority_summary(context):
    print("📋 Priority-based execution summary...")

    critical = context.get_variable("critical_status")
    user = context.get_variable("user_status")
    business = context.get_variable("business_status")
    background = context.get_variable("background_status")

    print(f"🔥 Critical: {'✅ ' + critical if critical else '❌ Failed'}")
    print(f"👤 User-facing: {'✅ ' + user if user else '❌ Failed'}")
    print(f"📊 Business logic: {'✅ ' + business if business else '❌ Failed'}")
    print(f"🧹 Background: {'✅ ' + background if background else '❌ Failed (OK)'}")

    # Determine system health based on priority
    essential_services = [critical, user, business]
    healthy_services = sum(1 for service in essential_services if service)

    if healthy_services == 3:
        print("🎉 System fully operational!")
    elif healthy_services >= 2:
        print("⚠️ System operational with some issues")
    else:
        print("🚨 System experiencing critical issues")
\`\`\`

### Priority Decision Matrix

| Operation Type | Priority Level | Retry Strategy | Timeout |
|---|---|---|---|
| **System Health** | CRITICAL | Aggressive (5+ retries) | Long (60s+) |
| **User Requests** | HIGH | Moderate (3-4 retries) | Medium (30s) |
| **Business Logic** | NORMAL | Standard (2-3 retries) | Short (15s) |
| **Analytics** | LOW | Minimal (1 retry) | Very Short (5s) |

---

## Circuit Breaker Pattern

Circuit breakers prevent cascade failures by stopping calls to failing services and allowing them time to recover.

### When to Use Circuit Breakers

- **External service dependencies** that might become overwhelmed
- **Database connections** that could be exhausted
- **Any operation** that could cause cascade failures
- **Expensive operations** where failure is costly

### Circuit Breaker States

1. **CLOSED** - Normal operation, calls pass through
2. **OPEN** - Service is failing, calls are rejected immediately
3. **HALF_OPEN** - Testing if service has recovered

\`\`\`python
from puffinflow.decorators import state
from puffinflow.core.reliability.circuit_breaker import CircuitBreakerConfig

# Circuit breaker for external APIs
api_circuit_config = CircuitBreakerConfig(
    failure_threshold=3,        # Open after 3 consecutive failures
    recovery_timeout=30.0,      # Wait 30 seconds before testing recovery
    success_threshold=2,        # Close after 2 consecutive successes
    timeout=10.0               # Individual operation timeout
)

# Circuit breaker for database operations
db_circuit_config = CircuitBreakerConfig(
    failure_threshold=5,        # More tolerant of database issues
    recovery_timeout=60.0,      # Longer recovery time
    success_threshold=3,        # Need more successes to trust again
    timeout=15.0
)

@state(
    circuit_breaker=True,
    circuit_breaker_config=api_circuit_config,
    max_retries=2
)
async def external_payment_api(context):
    """
    Payment API protected by circuit breaker:
    - Prevents overwhelming failed payment service
    - Fails fast when service is down
    - Automatically tests recovery
    """
    print("💳 External payment API call...")

    attempt = context.get_state("payment_attempts", 0) + 1
    context.set_state("payment_attempts", attempt)

    # Simulate external API that might be down
    if random.random() < 0.7:  # 70% failure rate when service is struggling
        raise Exception(f"Payment service unavailable (attempt {attempt})")

    print("✅ Payment processed successfully!")
    context.set_variable("payment_result", "success")

@state(
    circuit_breaker=True,
    circuit_breaker_config=db_circuit_config,
    max_retries=1
)
async def user_database_query(context):
    """
    Database query protected by circuit breaker:
    - Prevents connection pool exhaustion
    - Protects against cascading database failures
    """
    print("🗄️ User database query...")

    attempt = context.get_state("db_query_attempts", 0) + 1
    context.set_state("db_query_attempts", attempt)

    # Simulate database that might be overloaded
    if random.random() < 0.6:  # 60% failure rate during overload
        raise Exception(f"Database connection timeout (attempt {attempt})")

    print("✅ User data retrieved!")
    context.set_variable("user_data", {"id": 12345, "name": "John Doe"})

@state(
    circuit_breaker=True,
    circuit_breaker_config=CircuitBreakerConfig(
        failure_threshold=2,    # Very sensitive
        recovery_timeout=120.0, # Long recovery period
        success_threshold=5     # Need many successes
    )
)
async def fragile_ml_service(context):
    """
    ML service that's known to be fragile:
    - Opens circuit quickly to prevent resource waste
    - Takes time to recover
    - Requires multiple successes to trust again
    """
    print("🤖 Fragile ML service call...")

    attempt = context.get_state("ml_attempts", 0) + 1
    context.set_state("ml_attempts", attempt)

    # Very unreliable service
    if random.random() < 0.8:  # 80% failure rate
        raise Exception(f"ML service error (attempt {attempt})")

    print("✅ ML inference completed!")
    context.set_variable("ml_result", "inference_complete")
\`\`\`

### Circuit Breaker Configuration Guide

| Service Type | Failure Threshold | Recovery Timeout | Success Threshold |
|---|---|---|---|
| **Stable APIs** | 5-10 failures | 30-60 seconds | 2-3 successes |
| **Unreliable Services** | 2-3 failures | 60-120 seconds | 3-5 successes |
| **Critical Systems** | 3-5 failures | 30-45 seconds | 2-3 successes |
| **Expensive Operations** | 2-3 failures | 120+ seconds | 5+ successes |

---

## Bulkhead Isolation Pattern

Bulkheads isolate different types of operations to prevent failures in one area from affecting others.

### When to Use Bulkheads

- **Different types of operations** that compete for resources
- **External service calls** that might overwhelm connection pools
- **CPU/memory intensive tasks** that could starve other operations
- **Any operation** where you want to limit concurrent execution

\`\`\`python
from puffinflow.decorators import state
from puffinflow.core.reliability.bulkhead import BulkheadConfig

# Bulkhead for database operations
database_bulkhead = BulkheadConfig(
    name="database_operations",
    max_concurrent=3,           # Only 3 concurrent DB operations
    max_queue_size=10,         # Queue up to 10 waiting operations
    timeout=30.0               # Max wait time in queue
)

# Bulkhead for external API calls
external_api_bulkhead = BulkheadConfig(
    name="external_apis",
    max_concurrent=5,           # 5 concurrent API calls
    max_queue_size=20,         # Larger queue for API calls
    timeout=15.0               # Shorter wait time
)

# Bulkhead for heavy computation
compute_bulkhead = BulkheadConfig(
    name="heavy_compute",
    max_concurrent=2,           # Limited CPU-intensive operations
    max_queue_size=5,          # Small queue
    timeout=60.0               # Longer wait for expensive operations
)

@state(
    bulkhead=True,
    bulkhead_config=database_bulkhead,
    max_retries=2
)
async def user_profile_query(context):
    """
    Database operation isolated in its own bulkhead:
    - Won't overwhelm database connection pool
    - Database issues won't affect API calls
    - Predictable resource usage
    """
    print("👤 User profile query (database bulkhead)...")

    user_id = random.randint(1000, 9999)

    # Simulate database query
    await asyncio.sleep(random.uniform(1.0, 3.0))

    # 20% failure rate for database timeouts
    if random.random() < 0.2:
        raise Exception(f"Database timeout for user {user_id}")

    print(f"✅ User {user_id} profile retrieved!")
    context.set_variable(f"user_profile_{user_id}", {"status": "loaded"})

@state(
    bulkhead=True,
    bulkhead_config=external_api_bulkhead,
    max_retries=3
)
async def external_service_call(context):
    """
    External API call isolated in its own bulkhead:
    - Won't overwhelm external service with too many connections
    - External service issues won't affect database operations
    - Can handle more concurrent calls than database
    """
    print("🌍 External service call (API bulkhead)...")

    service_id = random.randint(1000, 9999)

    # Simulate external API call
    await asyncio.sleep(random.uniform(0.5, 2.0))

    # 30% failure rate for external service issues
    if random.random() < 0.3:
        raise Exception(f"External service error {service_id}")

    print(f"✅ External service {service_id} responded!")
    context.set_variable(f"external_result_{service_id}", {"status": "success"})

@state(
    bulkhead=True,
    bulkhead_config=compute_bulkhead,
    max_retries=1
)
async def heavy_computation(context):
    """
    CPU-intensive operation isolated in its own bulkhead:
    - Won't starve other operations of CPU
    - Limited concurrent execution prevents system overload
    - Expensive operations get dedicated resources
    """
    print("🔥 Heavy computation (compute bulkhead)...")

    task_id = random.randint(1000, 9999)

    # Simulate heavy computation
    await asyncio.sleep(random.uniform(3.0, 8.0))

    # 10% failure rate for computation errors
    if random.random() < 0.1:
        raise Exception(f"Computation failed for task {task_id}")

    print(f"✅ Heavy computation {task_id} completed!")
    context.set_variable(f"compute_result_{task_id}", {"status": "completed"})

# Create multiple instances to test bulkhead limits
for i in range(6):  # More than max_concurrent to test queuing
    agent.add_state(f"user_profile_query_{i}", user_profile_query)
    agent.add_state(f"external_service_call_{i}", external_service_call)

for i in range(4):  # More than compute bulkhead limit
    agent.add_state(f"heavy_computation_{i}", heavy_computation)
\`\`\`

### Bulkhead Sizing Guide

| Operation Type | Max Concurrent | Queue Size | Reasoning |
|---|---|---|---|
| **Database** | 3-5 | 10-15 | Connection pool limits |
| **External APIs** | 5-10 | 20-50 | Rate limiting considerations |
| **CPU Intensive** | 1-2 | 3-5 | Prevent CPU starvation |
| **Memory Intensive** | 2-4 | 5-10 | Memory availability |

---

## Dead Letter Queue Handling

Dead letter queues capture operations that have exhausted all retry attempts, allowing for manual intervention or alternative processing.

### When to Use Dead Letter Queues

- **Critical operations** that must not be lost even if they fail
- **Operations with side effects** that shouldn't be retried indefinitely
- **Complex workflows** where manual intervention might be needed
- **Audit trails** where you need to track all failures

\`\`\`python
from puffinflow.decorators import state
from puffinflow.core.agent.base import RetryPolicy

# Retry policy that sends failures to dead letter queue
dlq_retry_policy = RetryPolicy(
    max_retries=3,
    initial_delay=1.0,
    exponential_base=2.0,
    jitter=True,
    dead_letter_on_max_retries=True,    # Send to DLQ after exhausting retries
    dead_letter_on_timeout=True         # Send to DLQ on timeout as well
)

@state(
    retry_policy=dlq_retry_policy,
    timeout=10.0,
    dead_letter=True  # Enable dead letter queue for this state
)
async def payment_processing(context):
    """
    Payment processing with DLQ:
    - Critical operation that must not be lost
    - Failed payments need manual review
    - Audit trail required for compliance
    """
    print("💳 Processing payment...")

    payment_id = random.randint(10000, 99999)
    context.set_state("payment_id", payment_id)

    attempt = context.get_state("payment_attempts", 0) + 1
    context.set_state("payment_attempts", attempt)

    # Simulate payment processing that might fail
    if random.random() < 0.8:  # High failure rate for demonstration
        error_types = [
            "Insufficient funds",
            "Card expired",
            "Bank communication error",
            "Fraud detection triggered"
        ]
        error = random.choice(error_types)
        raise Exception(f"Payment failed: {error} (payment {payment_id})")

    print(f"✅ Payment {payment_id} processed successfully!")
    context.set_variable("payment_result", {"payment_id": payment_id, "status": "success"})

@state(
    max_retries=2,
    timeout=15.0,
    dead_letter=True
)
async def user_notification(context):
    """
    User notification with DLQ:
    - Important user communications
    - Failed notifications need follow-up
    - Track delivery failures for user experience
    """
    print("📧 Sending user notification...")

    notification_id = random.randint(10000, 99999)
    context.set_state("notification_id", notification_id)

    attempt = context.get_state("notification_attempts", 0) + 1
    context.set_state("notification_attempts", attempt)

    # Simulate notification that might fail due to various issues
    if random.random() < 0.6:  # 60% failure rate
        error_types = [
            "Email service unavailable",
            "Invalid email address",
            "SMS gateway timeout",
            "Push notification service error"
        ]
        error = random.choice(error_types)
        raise Exception(f"Notification failed: {error} (notification {notification_id})")

    print(f"✅ Notification {notification_id} sent successfully!")
    context.set_variable("notification_result", {"notification_id": notification_id, "status": "sent"})

@state(
    retry_policy=RetryPolicy(
        max_retries=5,
        initial_delay=2.0,
        exponential_base=1.5,
        dead_letter_on_max_retries=True
    ),
    timeout=30.0,
    dead_letter=True
)
async def data_sync_operation(context):
    """
    Data synchronization with DLQ:
    - Critical data consistency operation
    - Failed syncs need investigation
    - Data integrity must be maintained
    """
    print("🔄 Data synchronization...")

    sync_id = random.randint(10000, 99999)
    context.set_state("sync_id", sync_id)

    attempt = context.get_state("sync_attempts", 0) + 1
    context.set_state("sync_attempts", attempt)

    # Simulate data sync that fails due to data issues
    if random.random() < 0.9:  # Very high failure rate to demonstrate DLQ
        error_types = [
            "Data validation failed",
            "Source system unavailable",
            "Destination system readonly",
            "Data conflict detected"
        ]
        error = random.choice(error_types)
        raise Exception(f"Data sync failed: {error} (sync {sync_id})")

    print(f"✅ Data sync {sync_id} completed successfully!")
    context.set_variable("sync_result", {"sync_id": sync_id, "status": "synced"})

@state
async def check_dead_letter_queue(context):
    """
    Monitor dead letter queue for failed operations
    """
    print("📥 Checking dead letter queue...")

    # Get dead letters from agent
    dead_letters = agent.get_dead_letters()

    print(f"📊 Dead Letter Queue Status:")
    print(f"   Total dead letters: {len(dead_letters)}")

    if dead_letters:
        print("   Failed operations requiring attention:")

        for dl in dead_letters:
            print(f"   💀 {dl.state_name}:")
            print(f"      Agent: {dl.agent_name}")
            print(f"      Error: {dl.error_type}")
            print(f"      Message: {dl.error_message}")
            print(f"      Attempts: {dl.attempts}")
            print(f"      Timeout: {dl.timeout_occurred}")
            print(f"      Failed at: {dl.failed_at}")

            # Show context snapshot for debugging
            if dl.context_snapshot:
                print(f"      Context: {dl.context_snapshot}")
    else:
        print("   ✅ No items in dead letter queue")

    context.set_variable("dlq_status", {
        "count": len(dead_letters),
        "items": [dl.state_name for dl in dead_letters]
    })

# Add operations that will likely end up in DLQ
agent.add_state("payment_processing", payment_processing)
agent.add_state("user_notification", user_notification)
agent.add_state("data_sync_operation", data_sync_operation)
agent.add_state("check_dead_letter_queue", check_dead_letter_queue,
                dependencies=["payment_processing", "user_notification", "data_sync_operation"])
\`\`\`

### Dead Letter Queue Best Practices

1. **Monitor DLQ regularly** - Set up alerts for items in the queue
2. **Include context** - Capture enough information for debugging
3. **Plan remediation** - Have processes for handling dead letters
4. **Set retention** - Don't let dead letters accumulate indefinitely

---

## Comprehensive Error Handling Patterns

For production systems, combine multiple resilience patterns for robust operation.

### The Resilient Service Pattern

\`\`\`python
from puffinflow.decorators import state
from puffinflow.core.agent.state import Priority
from puffinflow.core.agent.base import RetryPolicy
from puffinflow.core.reliability.circuit_breaker import CircuitBreakerConfig
from puffinflow.core.reliability.bulkhead import BulkheadConfig

# Comprehensive configuration for a critical service
critical_service_retry = RetryPolicy(
    max_retries=5,
    initial_delay=1.0,
    exponential_base=2.0,
    jitter=True,
    dead_letter_on_max_retries=True
)

critical_service_circuit = CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=60.0,
    success_threshold=2,
    timeout=30.0
)

critical_service_bulkhead = BulkheadConfig(
    name="critical_service_operations",
    max_concurrent=2,
    max_queue_size=5,
    timeout=45.0
)

@state(
    # Priority and basic error handling
    priority=Priority.CRITICAL,
    timeout=30.0,

    # Retry configuration
    retry_policy=critical_service_retry,

    # Resilience patterns
    circuit_breaker=True,
    circuit_breaker_config=critical_service_circuit,
    bulkhead=True,
    bulkhead_config=critical_service_bulkhead,

    # Dead letter queue
    dead_letter=True,

    # Resource allocation
    cpu=2.0,
    memory=1024,

    # Rate limiting
    rate_limit=5.0
)
async def critical_business_operation(context):
    """
    A mission-critical operation with comprehensive error handling:

    ✅ Critical priority - gets resources first
    ✅ 5 retry attempts with exponential backoff + jitter
    ✅ Circuit breaker - protects against cascade failures
    ✅ Bulkhead isolation - won't overwhelm system resources
    ✅ 30-second timeout - prevents hanging
    ✅ Dead letter queue - captures ultimate failures
    ✅ Resource allocation - guaranteed CPU and memory
    ✅ Rate limiting - protects downstream services
    """
    print("🎯 Critical business operation starting...")

    operation_id = random.randint(100000, 999999)
    context.set_state("operation_id", operation_id)

    attempt = context.get_state("attempts", 0) + 1
    context.set_state("attempts", attempt)

    print(f"   Operation {operation_id}, attempt #{attempt}")
    print(f"   🛡️ Protected by: retry, circuit breaker, bulkhead, DLQ")

    # Simulate complex business operation
    await asyncio.sleep(random.uniform(2.0, 5.0))

    # Progressive success rate - gets better with retries
    success_rate = min(0.95, 0.3 + (attempt * 0.15))

    if random.random() > success_rate:
        error_types = [
            "Database transaction conflict",
            "External service dependency failure",
            "Business rule validation error",
            "Resource allocation timeout",
            "Network partition detected"
        ]
        error = random.choice(error_types)
        print(f"   ❌ {error}")
        raise Exception(f"{error} (operation {operation_id}, attempt {attempt})")

    print(f"   ✅ Operation {operation_id} completed successfully!")

    context.set_variable("critical_result", {
        "operation_id": operation_id,
        "attempts": attempt,
        "status": "success",
        "resilience_features": [
            "priority_scheduling",
            "retry_with_backoff",
            "circuit_breaker_protection",
            "bulkhead_isolation",
            "timeout_protection",
            "dead_letter_queue",
            "resource_allocation",
            "rate_limiting"
        ]
    })
\`\`\`

### The Graceful Degradation Pattern

\`\`\`python
@state(max_retries=2, timeout=10.0)
async def primary_service_call(context):
    """Try primary service first"""
    print("🎯 Attempting primary service...")

    # Simulate primary service that might be down
    if random.random() < 0.7:  # 70% failure rate
        raise Exception("Primary service unavailable")

    print("✅ Primary service succeeded")
    context.set_variable("service_result", "primary_success")

@state(max_retries=1, timeout=5.0)
async def fallback_service_call(context):
    """Fallback to secondary service"""
    print("🔄 Attempting fallback service...")

    # Simulate more reliable fallback service
    if random.random() < 0.2:  # 20% failure rate
        raise Exception("Fallback service unavailable")

    print("✅ Fallback service succeeded")
    context.set_variable("service_result", "fallback_success")

@state(max_retries=0)
async def degraded_mode_operation(context):
    """Operate with limited functionality"""
    print("⚠️ Operating in degraded mode...")

    # Always succeeds but with limited functionality
    await asyncio.sleep(0.5)

    print("✅ Degraded mode operation completed")
    context.set_variable("service_result", "degraded_mode")

@state
async def orchestrate_with_fallbacks(context):
    """Orchestrate service calls with graceful degradation"""
    print("🎭 Orchestrating service calls with fallbacks...")

    try:
        # Try primary service
        await agent.run_state("primary_service_call")

    except Exception as e:
        print(f"Primary service failed: {e}")

        try:
            # Try fallback service
            await agent.run_state("fallback_service_call")

        except Exception as e:
            print(f"Fallback service failed: {e}")

            # Use degraded mode
            await agent.run_state("degraded_mode_operation")

    result = context.get_variable("service_result")
    print(f"Final result: {result}")

    # Set appropriate status based on which service worked
    if result == "primary_success":
        context.set_variable("system_status", "fully_operational")
    elif result == "fallback_success":
        context.set_variable("system_status", "reduced_functionality")
    else:
        context.set_variable("system_status", "degraded_mode")
\`\`\`

---

## Decision Framework: Choosing Error Handling Strategies

Use this step-by-step framework to determine the right error handling approach for your operations.

### Step 1: Assess Operation Characteristics

\`\`\`python
# Ask yourself these questions:

# 1. How critical is this operation?
@state(priority=Priority.CRITICAL)  # Must succeed
@state(priority=Priority.HIGH)      # Important for users
@state(priority=Priority.NORMAL)    # Standard business logic
@state(priority=Priority.LOW)       # Nice to have

# 2. How likely is it to fail transiently?
@state(max_retries=5)  # High transient failure rate
@state(max_retries=2)  # Low transient failure rate
@state(max_retries=0)  # Failures are likely permanent

# 3. How expensive is each attempt?
@state(max_retries=1)   # Very expensive (ML training, large data processing)
@state(max_retries=3)   # Moderately expensive (database writes, API calls)
@state(max_retries=5)   # Inexpensive (health checks, simple queries)

# 4. How long should you wait?
@state(timeout=2.0)    # Real-time operations
@state(timeout=30.0)   # Standard operations
@state(timeout=300.0)  # Long-running operations
\`\`\`

### Step 2: Consider Failure Impact

\`\`\`python
# 5. Could failure cause cascade effects?
@state(circuit_breaker=True)  # Yes - protect downstream services

# 6. Does it compete for limited resources?
@state(bulkhead=True)  # Yes - isolate resource usage

# 7. Should failures be preserved for analysis?
@state(dead_letter=True)  # Yes - capture for manual review

# 8. Are there alternative approaches?
# Implement fallback pattern (see graceful degradation example)
\`\`\`

### Step 3: Implementation Template

\`\`\`python
# Template for comprehensive error handling
@state(
    # Basic characteristics
    priority=Priority.HIGH,         # Choose based on criticality
    timeout=30.0,                  # Choose based on operation duration

    # Retry strategy
    max_retries=3,                 # Choose based on failure likelihood
    retry_policy=RetryPolicy(      # Choose based on failure pattern
        initial_delay=1.0,
        exponential_base=2.0,
        jitter=True
    ),

    # Resilience patterns (add as needed)
    circuit_breaker=True,          # For external dependencies
    bulkhead=True,                 # For resource isolation
    dead_letter=True,              # For critical operations

    # Resource allocation (if needed)
    cpu=2.0,
    memory=1024,

    # Rate limiting (if needed)
    rate_limit=10.0
)
async def your_operation(context):
    # Your implementation here
    pass
\`\`\`

---

## Monitoring and Observability

Track error handling effectiveness with built-in monitoring.

\`\`\`python
@state
async def error_handling_health_check(context):
    """Monitor the health of error handling mechanisms"""
    print("🔍 Error handling health check...")

    # Check dead letter queue
    dead_letters = agent.get_dead_letters()
    dlq_count = len(dead_letters)

    # Check circuit breaker states (if available)
    # This would depend on your circuit breaker implementation

    # Analyze recent failures
    recent_failures = []
    for dl in dead_letters[-10:]:  # Last 10 failures
        recent_failures.append({
            "state": dl.state_name,
            "error": dl.error_type,
            "attempts": dl.attempts,
            "timeout": dl.timeout_occurred,
            "context_snapshot": dl.context_snapshot
        })

    context.set_variable("error_health", {
        "dlq_count": dlq_count,
        "recent_failures": recent_failures
    })

@state
async def generate_error_report(context):
    """Generate a summary report of system resilience"""
    print("📄 Generating error handling report...")

    error_health = context.get_variable("error_health")

    if not error_health:
        print("   No error health data available.")
        return

    dlq_count = error_health.get("dlq_count", 0)
    recent_failures = error_health.get("recent_failures", [])

    print(f"   Dead Letter Queue Size: {dlq_count}")

    if recent_failures:
        print("   Recent Failures:")

        failure_counts = {}
        for failure in recent_failures:
            state_name = failure['state']
            failure_counts[state_name] = failure_counts.get(state_name, 0) + 1

        for state, count in failure_counts.items():
            print(f"   - {state}: {count} recent failure(s)")

    # Add more complex analysis here, e.g., failure rates, etc.

    if dlq_count > 5:
        print("   🔴 WARNING: Dead letter queue is growing. Needs investigation.")
    elif dlq_count > 0:
        print("   🟡 INFO: Items found in dead letter queue.")
    else:
        print("   ✅ INFO: Dead letter queue is empty.")
\`\`\`

---

## Best Practices Summary

1. **Start Simple, Add Complexity as Needed**
   - Begin with basic retries and timeouts, then add more sophisticated patterns like circuit breakers and bulkheads as you identify specific needs.

2. **Configure in Decorators for Clarity**
   - Keep resilience logic declarative and close to your state definitions. This makes workflows easier to read and maintain.

3. **Match Patterns to Failure Types**
   - Use the right tool for the job. Don't use a circuit breaker for data validation errors; use a DLQ.

4. **Monitor and Adjust**
   - Error handling is not "set it and forget it." Monitor your DLQs, circuit breaker states, and failure rates to fine-tune your configurations.

5. **Plan for Ultimate Failures**
   - Always have a plan for what happens when an operation exhausts all resilience mechanisms. Dead letter queues are your last line of defense.

---

## Quick Reference

### Basic Error Handling
\`\`\`python
@state(max_retries=3)
@state(timeout=30.0)
@state(priority=Priority.HIGH)
\`\`\`

### Advanced Retry Policies
\`\`\`python
@state(retry_policy=RetryPolicy(
    max_retries=5,
    initial_delay=1.0,
    exponential_base=2.0,
    jitter=True
))
\`\`\`

### Resilience Patterns
\`\`\`python
@state(circuit_breaker=True)
@state(bulkhead=True)
@state(dead_letter=True)
\`\`\`

### Comprehensive Protection
\`\`\`python
@state(
    priority=Priority.CRITICAL,
    max_retries=5,
    timeout=30.0,
    circuit_breaker=True,
    bulkhead=True,
    dead_letter=True,
    rate_limit=10.0
)
async def production_ready_operation(context):
    pass
\`\`\`

Error handling and resilience patterns ensure your workflows gracefully handle failures, maintain system stability, and provide the reliability needed for production systems!
`.trim();
