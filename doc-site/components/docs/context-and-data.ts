export const contextAndDataMarkdown = `# Context and Data

The Context system is how states share data in Puffinflow. It's a secure, typed data store that every state can read from and write to, making your workflows robust and maintainable.

## Why Context Matters

**The Problem:** In async workflows, sharing data between functions usually means:
- Global variables (dangerous with concurrency)
- Passing parameters everywhere (verbose and brittle)
- Manual serialization (error-prone)

**The Solution:** Puffinflow's Context acts as a secure, shared memory space that every state can safely access.

## Basic Data Sharing

Use \`set_variable()\` and \`get_variable()\` for most data sharing:

\`\`\`python
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
    settings = context.get_variable("settings", {"theme": "default"})

    print(f"Processing {user['name']} at {timestamp}")
    return "send_welcome"

# Add states to agent
agent.add_state("fetch_user", fetch_user)
agent.add_state("process_user", process_user)
\`\`\`

## Data Types Available

| Method | Use Case | Example |
|--------|----------|---------|
| \`set_variable()\` | General data sharing | User data, lists, dicts |
| \`set_typed_variable()\` | Type-safe data | Counts, scores (enforces type) |
| \`set_validated_data()\` | Structured data | Pydantic models |
| \`set_constant()\` | Configuration | API URLs, settings |
| \`set_secret()\` | Sensitive data | API keys, passwords |
| \`set_cached()\` | Temporary data | TTL expiration |
| \`set_output()\` | Final results | Workflow outputs |

## Type-Safe Variables

Use \`set_typed_variable()\` to enforce consistent data types:

\`\`\`python
@state
async def initialize(context):
    context.set_typed_variable("user_count", 100)      # Locked to int
    context.set_typed_variable("avg_score", 85.5)      # Locked to float
    return "process"

@state
async def process(context):
    context.set_typed_variable("user_count", 150)      # ✅ Works
    # context.set_typed_variable("user_count", "150")  # ❌ TypeError

    count = context.get_typed_variable("user_count")
    print(f"Processing {count} users")

# Add states to agent
agent.add_state("initialize", initialize)
agent.add_state("process", process)
\`\`\`

## Validated Data with Pydantic

Use \`set_validated_data()\` for structured data with automatic validation:

\`\`\`python
from pydantic import BaseModel, EmailStr

class User(BaseModel):
    id: int
    name: str
    email: EmailStr
    age: int

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

# Add states to agent
agent.add_state("create_user", create_user)
agent.add_state("update_user", update_user)
\`\`\`

## Configuration and Secrets

Use \`set_constant()\` for immutable configuration and \`set_secret()\` for sensitive data:

\`\`\`python
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

    # context.set_constant("api_url", "different")  # ❌ ValueError: Constants are immutable

# Add states to agent
agent.add_state("setup", setup)
agent.add_state("make_request", make_request)
\`\`\`

## Cached Data with TTL

Use \`set_cached()\` for temporary data that expires:

\`\`\`python
@state
async def cache_session(context):
    context.set_cached("user_session", {"user_id": 123}, ttl=300)  # 5 minutes
    context.set_cached("temp_token", "abc123", ttl=60)            # 1 minute
    return "use_cache"

@state
async def use_cache(context):
    session = context.get_cached("user_session", default="EXPIRED")
    if session != "EXPIRED":
        print(f"Active session: {session}")
    else:
        print("Session expired, need to re-authenticate")

# Add states to agent
agent.add_state("cache_session", cache_session)
agent.add_state("use_cache", use_cache)
\`\`\`

## Workflow Outputs

Use \`set_output()\` to mark final workflow results:

\`\`\`python
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

    print(f"Report: \${revenue} revenue from {count} orders (avg: \${avg:.2f})")

# Add states to agent
agent.add_state("calculate_metrics", calculate_metrics)
agent.add_state("send_report", send_report)
\`\`\`

## Complete Example

\`\`\`python
import asyncio
from pydantic import BaseModel
from puffinflow import Agent, state

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

    print(f"✅ Order {order.id} completed: \${amount}")

# Add states to agent
agent.add_state("setup", setup)
agent.add_state("process_order", process_order)
agent.add_state("finalize", finalize)

# Run the workflow
async def main():
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

## Best Practices

**Choose the right method:**
- **\`set_variable()\`** - Default choice for most data (90% of use cases)
- **\`set_constant()\`** - Configuration that never changes
- **\`set_secret()\`** - API keys and sensitive data only
- **\`set_output()\`** - Final workflow results
- **\`set_typed_variable()\`** - When you need strict type consistency
- **\`set_validated_data()\`** - Complex structured data from external sources
- **\`set_cached()\`** - Data that expires (don't overuse)

**Quick tips:**
1. **Start simple** - Use \`set_variable()\` for most data sharing
2. **Validate external data** - Use Pydantic models for data from APIs
3. **Never log secrets** - Only retrieve when absolutely needed
4. **Use appropriate TTL** - Don't cache sensitive data too long
5. **Prefer local variables** - For temporary data within a single state

The Context system gives you the flexibility to handle any data scenario while maintaining type safety and security.
`.trim();
