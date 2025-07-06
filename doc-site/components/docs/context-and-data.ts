export const contextAndDataMarkdown = `# Context and Data

The **Context system** is Puffinflow's powerful data sharing mechanism that goes far beyond simple variables. It provides type safety, validation, caching, secrets management, and more - all designed to make your workflows robust and maintainable.

## Why Context Matters

**The Problem:** In async workflows, sharing data between functions usually means:
- Global variables (dangerous with concurrency)
- Passing parameters everywhere (verbose and brittle)
- Manual serialization (error-prone)

**The Solution:** Puffinflow's Context acts as a secure, typed, shared memory space that every state can safely read from and write to.

## Quick Overview

The Context object provides several data storage mechanisms:

| Method | Use Case | Features |
|--------|----------|----------|
| \`set_variable()\` | General data sharing | Simple, flexible |
| \`set_typed_variable()\` | Type-safe data | Locks Python types |
| \`set_validated_data()\` | Structured data | Pydantic validation |
| \`set_constant()\` | Configuration | Immutable values |
| \`set_secret()\` | Sensitive data | Secure storage |
| \`set_cached()\` | Temporary data | TTL expiration |
| \`set_state()\` | Per-state scratch | State-local data |

## General Variables (Most Common)

Use \`set_variable()\` and \`get_variable()\` for most data sharing:

\`\`\`python
async def fetch_data(context):
    user_data = {"id": 123, "name": "Alice", "email": "alice@example.com"}
    context.set_variable("user", user_data)
    context.set_variable("count", 1250)

async def process_data(context):
    user = context.get_variable("user")
    count = context.get_variable("count")
    print(f"Processing {user['name']}, user {user['id']} of {count}")
\`\`\`

## Type-Safe Variables

Use \`set_typed_variable()\` to enforce consistent data types:

\`\`\`python
async def initialize(context):
    context.set_typed_variable("user_count", 100)      # Locked to int
    context.set_typed_variable("avg_score", 85.5)      # Locked to float

async def update(context):
    context.set_typed_variable("user_count", 150)      # ✅ Works
    # context.set_typed_variable("user_count", "150")  # ❌ TypeError

    count = context.get_typed_variable("user_count")   # Type param optional
    print(f"Count: {count}")
\`\`\`

> **Note:** The type parameter in \`get_typed_variable("key", int)\` is optional. You can just use \`get_typed_variable("key")\` for cleaner code. The type parameter is mainly for static type checkers.

## Validated Data with Pydantic

Use \`set_validated_data()\` for structured data with automatic validation:

\`\`\`python
from pydantic import BaseModel, EmailStr

class User(BaseModel):
    id: int
    name: str
    email: EmailStr
    age: int

async def create_user(context):
    user = User(id=123, name="Alice", email="alice@example.com", age=28)
    context.set_validated_data("user", user)

async def update_user(context):
    user = context.get_validated_data("user", User)
    user.age = 29
    context.set_validated_data("user", user)  # Re-validates
\`\`\`

## Constants and Configuration

Use \`set_constant()\` for immutable configuration:

\`\`\`python
async def setup(context):
    context.set_constant("api_url", "https://api.example.com")
    context.set_constant("max_retries", 3)

async def use_config(context):
    url = context.get_constant("api_url")
    retries = context.get_constant("max_retries")
    # context.set_constant("api_url", "different")  # ❌ ValueError
\`\`\`

## Secrets Management

Use \`set_secret()\` for sensitive data:

\`\`\`python
async def load_secrets(context):
    context.set_secret("api_key", "sk-1234567890abcdef")
    context.set_secret("db_password", "super_secure_password")

async def use_secrets(context):
    api_key = context.get_secret("api_key")
    # Use for API calls (don't print real secrets!)
    print(f"API key loaded: {api_key[:8]}...")
\`\`\`

## Cached Data with TTL

Use \`set_cached()\` for temporary data that expires:

\`\`\`python
async def cache_data(context):
    context.set_cached("session", {"user_id": 123}, ttl=300)  # 5 minutes
    context.set_cached("temp_result", {"data": "value"}, ttl=60)   # 1 minute

async def use_cache(context):
    session = context.get_cached("session", default="EXPIRED")
    print(f"Session: {session}")
\`\`\`

## Per-State Scratch Data

Use \`set_state()\` for data local to individual states:

\`\`\`python
async def state_a(context):
    context.set_state("temp_data", [1, 2, 3])  # Only visible in state_a
    context.set_variable("shared", "visible to all")

async def state_b(context):
    context.set_state("temp_data", {"key": "value"})  # Different from state_a
    shared = context.get_variable("shared")  # Can access shared data
    my_temp = context.get_state("temp_data")  # Gets state_b's data
\`\`\`

> **Note:** For most use cases, regular local variables are simpler and better than \`set_state()\`:
> \`\`\`python
> # Instead of context.set_state("temp", data)
> # Just use: temp_data = [1, 2, 3]
> \`\`\`
> Only use \`set_state()\` if you need to inspect a state's internal data from outside for debugging/monitoring purposes.

## Output Data Management

Use \`set_output()\` for final workflow results:

\`\`\`python
async def calculate(context):
    orders = [{"amount": 100}, {"amount": 200}]
    total = sum(order["amount"] for order in orders)

    context.set_output("total_revenue", total)
    context.set_output("order_count", len(orders))

async def summary(context):
    revenue = context.get_output("total_revenue")
    count = context.get_output("order_count")
    print(f"Revenue: \${revenue}, Orders: {count}")
\`\`\`

## Complete Example: Order Processing

\`\`\`python
import asyncio
from pydantic import BaseModel
from puffinflow import Agent

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
    payment_key = context.get_secret("payment_key")

    # Final outputs
    context.set_output("order_id", order.id)
    context.set_output("amount_processed", amount)
    print(f"✅ Order {order.id} completed: \${amount}")

agent.add_state("setup", setup)
agent.add_state("process_order", process_order, dependencies=["setup"])
agent.add_state("send_confirmation", send_confirmation, dependencies=["process_order"])

if __name__ == "__main__":
    asyncio.run(agent.run())
\`\`\`

## Best Practices

### Choose the Right Method

- **\`set_variable()\`** - Default choice for most data (use 90% of the time)
- **\`set_constant()\`** - For configuration that shouldn't change
- **\`set_secret()\`** - For API keys and passwords
- **\`set_output()\`** - For final workflow results
- **\`set_typed_variable()\`** - Only when you need strict type consistency
- **\`set_validated_data()\`** - Only for complex structured data
- **\`set_cached()\`** - Only when you need TTL expiration
- **\`set_state()\`** - Almost never (use local variables instead)

### Quick Tips

1. **Start simple** - Use \`set_variable()\` for most data sharing
2. **Validate early** - Use Pydantic models for external data
3. **Never log secrets** - Only retrieve when needed
4. **Set appropriate TTL** - Don't cache sensitive data too long
5. **Use local variables** - Instead of \`set_state()\` for temporary data

The Context system gives you flexibility to handle any data scenario while maintaining type safety and security.
`.trim();
