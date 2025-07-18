export const contextAndDataMarkdown = `# Context and Data

The **Context system** is Puffinflow's most powerful feature for data management and sharing across workflow states. It goes far beyond simple variables, providing a comprehensive solution for type safety, validation, caching, secrets management, and state isolation - all designed to make your workflows robust, maintainable, and production-ready.

## Why Context Matters

**The Problem:** In traditional async workflows, sharing data between functions creates several challenges:
- **Global variables** are dangerous with concurrency and can cause race conditions
- **Passing parameters everywhere** becomes verbose, brittle, and hard to maintain
- **Manual serialization** is error-prone and doesn't preserve types
- **No validation** means bad data can corrupt your entire workflow
- **Security issues** when sensitive data is mixed with regular variables
- **Memory leaks** when temporary data isn't cleaned up properly

**The Solution:** Puffinflow's Context acts as a secure, typed, shared memory space that every state can safely read from and write to. It provides:

- **Thread-safe** data sharing across concurrent states
- **Type enforcement** to prevent runtime errors
- **Automatic validation** using Pydantic models
- **Secure storage** for sensitive information like API keys
- **TTL-based caching** for temporary data that expires automatically
- **State isolation** for debugging and monitoring individual states
- **Immutable constants** for configuration that shouldn't change during execution

## Quick Overview

The Context object provides several specialized data storage mechanisms, each optimized for different use cases:

| Method | Use Case | Features | Thread-Safe | Validation | TTL Support |
|--------|----------|----------|-------------|------------|-------------|
| \`set_variable()\` | General data sharing | Simple, flexible, any Python object | ✅ | ❌ | ❌ |
| \`set_typed_variable()\` | Type-safe data | Enforces Python type consistency | ✅ | Type checking | ❌ |
| \`set_validated_data()\` | Structured data | Pydantic model validation | ✅ | Full validation | ❌ |
| \`set_constant()\` | Configuration | Immutable once set | ✅ | ❌ | ❌ |
| \`set_secret()\` | Sensitive data | Secure storage, no logging | ✅ | ❌ | ❌ |
| \`set_cached()\` | Temporary data | Automatic TTL expiration | ✅ | ❌ | ✅ |
| \`set_state()\` | Per-state scratch | State-local debugging data | ✅ | ❌ | ❌ |
| \`set_output()\` | Final results | Workflow outputs/results | ✅ | ❌ | ❌ |

**🎯 Pro Tip:** Start with \`set_variable()\` for most use cases, then upgrade to specialized methods when you need their specific features.

## General Variables (Most Common)

Use \`set_variable()\` and \`get_variable()\` for most data sharing. This is your go-to method for 90% of use cases - it's simple, flexible, and handles any Python object.

**Key Features:**
- **Thread-safe**: Multiple states can read/write safely
- **No type restrictions**: Store dicts, lists, objects, primitives
- **Persistent**: Data survives across all workflow states
- **Simple API**: Just set and get with string keys

\`\`\`python
async def fetch_data(context):
    # Store complex data structures
    user_data = {"id": 123, "name": "Alice", "email": "alice@example.com"}
    context.set_variable("user", user_data)
    
    # Store primitive types
    context.set_variable("count", 1250)
    context.set_variable("is_premium", True)
    
    # Store lists and nested objects
    context.set_variable("tags", ["customer", "active", "premium"])
    context.set_variable("metadata", {
        "last_login": "2024-01-15",
        "preferences": {"theme": "dark", "notifications": True}
    })

async def process_data(context):
    # Retrieve and use the data
    user = context.get_variable("user")
    count = context.get_variable("count")
    is_premium = context.get_variable("is_premium")
    tags = context.get_variable("tags")
    
    # Safe access with defaults
    region = context.get_variable("region", default="US")
    
    print(f"Processing {user['name']}, user {user['id']} of {count}")
    print(f"Premium: {is_premium}, Tags: {', '.join(tags)}")
    print(f"Region: {region}")
    
    # Update existing variables
    context.set_variable("status", "processed")
    
    # Store processing results
    context.set_variable("processing_result", {
        "user_id": user["id"],
        "processed_at": "2024-01-15T10:30:00Z",
        "success": True
    })
\`\`\`

**💡 Best Practices:**
- Use descriptive key names like \`"user_data"\` instead of \`"data"\`
- Provide defaults with \`get_variable("key", default="fallback")\` for optional data
- Group related data in dictionaries rather than many separate variables
- Consider upgrading to typed or validated methods for critical data

## Type-Safe Variables

Use \`set_typed_variable()\` when you need strict type consistency across your workflow. The first call locks the type, and all subsequent calls must use the same Python type.

**When to Use:**
- **Counter variables** that should always be integers
- **Score/percentage values** that should always be floats
- **Feature flags** that should always be booleans
- **Critical numeric data** where type confusion could cause bugs

**Key Features:**
- **Type locking**: First call determines the allowed type forever
- **Runtime validation**: Automatic type checking on every set
- **Type preservation**: Values maintain their exact Python type
- **Error prevention**: Prevents accidental string-to-number bugs

\`\`\`python
async def initialize(context):
    # First call locks the type
    context.set_typed_variable("user_count", 100)      # Locked to int
    context.set_typed_variable("avg_score", 85.5)      # Locked to float
    context.set_typed_variable("is_enabled", True)     # Locked to bool
    context.set_typed_variable("status", "active")     # Locked to str

async def update_counters(context):
    # These work - same types as initial values
    context.set_typed_variable("user_count", 150)      # ✅ int → int
    context.set_typed_variable("avg_score", 92.3)      # ✅ float → float
    context.set_typed_variable("is_enabled", False)    # ✅ bool → bool
    
    # These would fail - type mismatches
    # context.set_typed_variable("user_count", "150")    # ❌ str → int (TypeError)
    # context.set_typed_variable("avg_score", 92)        # ❌ int → float (TypeError)
    # context.set_typed_variable("is_enabled", "false")  # ❌ str → bool (TypeError)

async def use_typed_data(context):
    # Retrieve with automatic type safety
    count = context.get_typed_variable("user_count")     # Always int
    score = context.get_typed_variable("avg_score")      # Always float
    enabled = context.get_typed_variable("is_enabled")   # Always bool
    
    # Type parameter is optional but helps IDEs
    count_typed = context.get_typed_variable("user_count", int)
    
    # Safe arithmetic operations
    if enabled:
        new_count = count + 10  # Always safe - count is guaranteed int
        new_avg = score * 1.1   # Always safe - score is guaranteed float
        context.set_typed_variable("user_count", new_count)
        context.set_typed_variable("avg_score", new_avg)

async def demonstrate_type_errors(context):
    # This will raise TypeError - demonstrating type safety
    try:
        context.set_typed_variable("user_count", 100)  # First call - locks to int
        context.set_typed_variable("user_count", "200")  # This fails!
    except TypeError as e:
        print(f"Type error caught: {e}")
        # Workflow continues safely
\`\`\`

**⚡ Performance Note:** Type checking happens on every \`set_typed_variable()\` call, so there's a small overhead. Use regular \`set_variable()\` for high-frequency updates where types are already guaranteed.

**🔧 IDE Integration:** The optional type parameter in \`get_typed_variable("key", int)\` provides better autocomplete and type hints in IDEs like PyCharm and VS Code.

## Validated Data with Pydantic

Use \`set_validated_data()\` when working with complex structured data that needs validation. This method leverages Pydantic models to ensure data integrity and catch errors early.

**When to Use:**
- **External API responses** that need validation
- **User input data** that could be malformed
- **Complex data structures** with multiple fields
- **Data that changes** and needs re-validation
- **Inter-service communication** data

**Key Features:**
- **Automatic validation**: Pydantic validates on every set operation
- **Type conversion**: Converts compatible types automatically
- **Field validation**: Custom validators and constraints
- **Nested models**: Support for complex nested structures
- **Error reporting**: Detailed validation error messages

\`\`\`python
from pydantic import BaseModel, EmailStr, validator, Field
from typing import List, Optional
from datetime import datetime

class Address(BaseModel):
    street: str
    city: str
    zip_code: str = Field(..., min_length=5, max_length=10)
    country: str = "US"

class User(BaseModel):
    id: int = Field(..., gt=0)  # Must be positive
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr  # Automatic email validation
    age: int = Field(..., ge=13, le=120)  # Age constraints
    addresses: List[Address] = []
    last_login: Optional[datetime] = None
    tags: List[str] = []
    
    @validator('name')
    def name_must_be_capitalized(cls, v):
        return v.title()
    
    @validator('tags')
    def validate_tags(cls, v):
        # Custom business logic validation
        allowed_tags = ['premium', 'customer', 'active', 'trial']
        invalid_tags = [tag for tag in v if tag not in allowed_tags]
        if invalid_tags:
            raise ValueError(f'Invalid tags: {invalid_tags}')
        return v

async def create_user(context):
    # Pydantic validates during model creation
    user = User(
        id=123, 
        name="alice smith",  # Will be capitalized to "Alice Smith"
        email="alice@example.com", 
        age=28,
        addresses=[
            Address(
                street="123 Main St",
                city="San Francisco",
                zip_code="94105"
            )
        ],
        tags=["premium", "customer"]
    )
    
    # Store with validation
    context.set_validated_data("user", user)
    
    # Store other validated models
    company_data = {
        "name": "Acme Corp",
        "employees": 500,
        "founded": 1995
    }
    # Convert dict to model and validate
    class Company(BaseModel):
        name: str
        employees: int = Field(..., gt=0)
        founded: int = Field(..., ge=1800, le=2024)
    
    company = Company(**company_data)
    context.set_validated_data("company", company)

async def update_user(context):
    # Retrieve validated data
    user = context.get_validated_data("user", User)
    company = context.get_validated_data("company", Company)
    
    # Modify data
    user.age = 29
    user.last_login = datetime.now()
    user.tags.append("active")
    
    # Re-validation happens automatically
    context.set_validated_data("user", user)  # Validates again
    
    # Demonstrate validation errors
    try:
        user.age = 150  # Invalid age
        context.set_validated_data("user", user)
    except ValueError as e:
        print(f"Validation error: {e}")
        # Reset to valid age
        user.age = 29
        context.set_validated_data("user", user)

async def process_external_data(context):
    # Handle potentially bad external data
    external_user_data = {
        "id": "123",  # String, but Pydantic will convert to int
        "name": "bob jones",  # Will be capitalized
        "email": "bob@example.com",
        "age": "30",  # String, but Pydantic will convert to int
        "tags": ["customer", "active"]
    }
    
    try:
        user = User(**external_user_data)
        context.set_validated_data("external_user", user)
        print(f"✅ Valid user: {user.name} (age {user.age})")
    except ValueError as validation_error:
        # Handle validation failures gracefully
        print(f"❌ Invalid user data: {validation_error}")
        context.set_variable("validation_error", str(validation_error))

async def demonstrate_nested_validation(context):
    # Complex nested structure validation
    user_with_address = User(
        id=456,
        name="charlie brown",
        email="charlie@example.com",
        age=35,
        addresses=[
            Address(street="456 Oak Ave", city="Portland", zip_code="97201"),
            Address(street="789 Pine St", city="Seattle", zip_code="98101")
        ],
        tags=["premium"]
    )
    
    context.set_validated_data("complex_user", user_with_address)
    
    # Retrieve and modify nested data
    user = context.get_validated_data("complex_user", User)
    
    # Add a new address
    new_address = Address(
        street="321 Elm St",
        city="Vancouver",
        zip_code="V6B1A1"  # Canadian postal code format
    )
    user.addresses.append(new_address)
    
    # Re-validate the entire structure
    context.set_validated_data("complex_user", user)
\`\`\`

**🔍 Validation Benefits:**
- **Catch errors early**: Invalid data fails fast, not in production
- **Type safety**: Guaranteed field types after validation
- **Data normalization**: Consistent format (e.g., capitalized names)
- **Documentation**: Pydantic models serve as data contracts
- **IDE support**: Better autocomplete and type hints

**⚠️ Important Notes:**
- Validation happens on every \`set_validated_data()\` call
- Failed validation raises \`ValueError\` with detailed error messages
- Use try/except blocks when processing external data
- Consider performance impact for high-frequency updates

## Constants and Configuration

Use \`set_constant()\` for configuration values that should never change during workflow execution. Once set, constants are immutable and provide a safe way to share configuration across your entire workflow.

**When to Use:**
- **API endpoints** and service URLs
- **Retry counts** and timeout values
- **Feature flags** that shouldn't change mid-execution
- **Environment-specific settings** (dev/staging/prod)
- **Business rules** and limits that are fixed for a run

**Key Features:**
- **Immutability**: Cannot be changed once set (raises ValueError)
- **Thread-safe**: Safe for concurrent access across states
- **Global scope**: Available to all states in the workflow
- **Error prevention**: Prevents accidental configuration changes
- **Clear intent**: Explicitly marks values as unchangeable

\`\`\`python
async def setup_configuration(context):
    # API and service configuration
    context.set_constant("api_base_url", "https://api.example.com")
    context.set_constant("auth_endpoint", "/oauth/token")
    context.set_constant("data_endpoint", "/api/v1/data")
    
    # Retry and timeout settings
    context.set_constant("max_retries", 3)
    context.set_constant("request_timeout", 30)
    context.set_constant("backoff_factor", 2.0)
    
    # Business rules and limits
    context.set_constant("max_items_per_batch", 100)
    context.set_constant("rate_limit_per_second", 10)
    context.set_constant("max_file_size_mb", 50)
    
    # Feature flags for this workflow run
    context.set_constant("enable_caching", True)
    context.set_constant("enable_metrics", True)
    context.set_constant("debug_mode", False)
    
    # Environment-specific settings
    import os
    environment = os.getenv("ENVIRONMENT", "development")
    context.set_constant("environment", environment)
    
    if environment == "production":
        context.set_constant("log_level", "INFO")
        context.set_constant("enable_detailed_logging", False)
    else:
        context.set_constant("log_level", "DEBUG")
        context.set_constant("enable_detailed_logging", True)

async def make_api_request(context):
    # Use configuration constants
    base_url = context.get_constant("api_base_url")
    auth_endpoint = context.get_constant("auth_endpoint")
    max_retries = context.get_constant("max_retries")
    timeout = context.get_constant("request_timeout")
    
    # Build request URL
    full_auth_url = f"{base_url}{auth_endpoint}"
    
    # Use retry configuration
    for attempt in range(max_retries):
        try:
            # Simulate API request with configured timeout
            print(f"Attempting request to {full_auth_url} (timeout: {timeout}s)")
            # ... actual API request logic ...
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Retry {attempt + 1}/{max_retries} failed: {e}")

async def process_data_batch(context):
    # Use business rule constants
    max_batch_size = context.get_constant("max_items_per_batch")
    rate_limit = context.get_constant("rate_limit_per_second")
    enable_caching = context.get_constant("enable_caching")
    
    # Process data according to configuration
    data_items = range(250)  # Example: 250 items to process
    
    # Split into batches based on configuration
    for i in range(0, len(data_items), max_batch_size):
        batch = data_items[i:i + max_batch_size]
        
        print(f"Processing batch of {len(batch)} items (limit: {max_batch_size})")
        
        # Use caching if enabled
        if enable_caching:
            # Check cache first
            cache_key = f"batch_{i // max_batch_size}"
            cached_result = context.get_cached(cache_key)
            if cached_result:
                print(f"Using cached result for {cache_key}")
                continue
        
        # Rate limiting based on configuration
        import time
        time.sleep(1 / rate_limit)  # Respect rate limit

async def demonstrate_immutability(context):
    # Constants cannot be changed once set
    try:
        # This will work - first time setting
        context.set_constant("new_config", "initial_value")
        print("✅ New constant set successfully")
        
        # This will fail - trying to change existing constant
        context.set_constant("new_config", "different_value")
        print("❌ This line should never execute")
        
    except ValueError as e:
        print(f"✅ Immutability enforced: {e}")
        
    # Reading constants always works
    value = context.get_constant("new_config")
    print(f"Current value: {value}")
    
    # Non-existent constants return None
    missing = context.get_constant("does_not_exist")
    print(f"Missing constant: {missing}")
    
    # Provide defaults for optional constants
    optional_feature = context.get_constant("optional_feature", default=False)
    print(f"Optional feature enabled: {optional_feature}")

async def use_environment_config(context):
    # Use environment-specific configuration
    environment = context.get_constant("environment")
    log_level = context.get_constant("log_level")
    detailed_logging = context.get_constant("enable_detailed_logging")
    
    print(f"Running in {environment} environment")
    print(f"Log level: {log_level}")
    
    if detailed_logging:
        print("🔍 Detailed logging enabled - full debug info available")
        # Log detailed information
    else:
        print("📝 Standard logging - production mode")
        # Log only essential information

# Example of setting up different configurations for different environments
async def setup_environment_specific_config(context):
    import os
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        context.set_constant("api_base_url", "https://api.production.com")
        context.set_constant("max_retries", 5)
        context.set_constant("enable_metrics", True)
        context.set_constant("cache_ttl", 3600)  # 1 hour
    elif env == "staging":
        context.set_constant("api_base_url", "https://api.staging.com")
        context.set_constant("max_retries", 3)
        context.set_constant("enable_metrics", True)
        context.set_constant("cache_ttl", 1800)  # 30 minutes
    else:  # development
        context.set_constant("api_base_url", "https://api.dev.com")
        context.set_constant("max_retries", 1)
        context.set_constant("enable_metrics", False)
        context.set_constant("cache_ttl", 300)   # 5 minutes
    
    context.set_constant("environment", env)
\`\`\`

**🔒 Immutability Benefits:**
- **Prevents bugs**: Configuration can't accidentally change during execution
- **Thread safety**: Multiple states can safely read constants simultaneously
- **Clear contracts**: Other developers know these values won't change
- **Debugging**: Eliminates configuration drift as a source of issues

**💡 Best Practices:**
- Set all constants early in your workflow (usually in a setup state)
- Use descriptive names: \`"max_retry_count"\` not \`"retries"\`
- Group related constants with prefixes: \`"api_base_url"\`, \`"api_timeout"\`
- Provide sensible defaults with \`get_constant("key", default=value)\`
- Use environment variables to configure constants for different environments

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
