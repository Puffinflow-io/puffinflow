export const observabilityMarkdown = `# Observability

Observability is like having X-ray vision for your workflows - you can see what's happening inside them, understand why they're slow or failing, and know exactly where problems occur. Puffinflow comes with powerful built-in observability features that work out of the box.

## Why Observability Matters

**Without observability:**
- You only know something is broken when users complain
- Debugging feels like searching for a needle in a haystack
- Performance problems are mysterious and hard to fix
- You can't tell if changes make things better or worse
- Scaling decisions are based on guesswork

**With Puffinflow's built-in observability:**
- You spot problems before users notice them
- You can trace any request from start to finish
- Performance bottlenecks are clearly visible
- You have data to make informed decisions
- Debugging becomes systematic and efficient

## Part 1: Built-in Metrics Collection

Puffinflow automatically collects metrics for all your workflows - just enable it:

\`\`\`python
import asyncio
import time
from puffinflow import Agent, state
from puffinflow.core.observability.metrics import PrometheusMetricsProvider
from puffinflow.core.observability.config import MetricsConfig

# Set up Puffinflow's built-in metrics
metrics_config = MetricsConfig(
    namespace="my_app",
    enable_detailed_metrics=True,
    cardinality_limit=10000
)
metrics_provider = PrometheusMetricsProvider(metrics_config)

agent = Agent("metrics-demo")

# Enable automatic metrics collection
@state(
    metrics_enabled=True,  # This enables Puffinflow's built-in metrics
    custom_metrics=["execution_time", "memory_usage", "api_calls"]
)
async def process_user_request(context):
    """Process request with automatic metrics collection"""

    user_id = context.get_variable("user_id", "user_123")
    request_type = context.get_variable("request_type", "search")

    print(f"🔄 Processing {request_type} request from {user_id}")

    # Puffinflow automatically tracks:
    # - State execution time
    # - Memory usage during execution
    # - Success/failure rates
    # - Resource utilization

    # Simulate processing work
    await asyncio.sleep(1.2)

    # Add custom business metrics using built-in provider
    request_counter = metrics_provider.counter(
        "user_requests_total",
        "Total user requests",
        ["request_type", "status"]
    )

    response_time = metrics_provider.histogram(
        "request_duration_seconds",
        "Request processing time",
        ["request_type"]
    )

    # Record custom metrics
    request_counter.record(1, request_type=request_type, status="success")
    response_time.record(1.2, request_type=request_type)

    print(f"✅ Request completed successfully")
    context.set_variable("result", "success")

    return "generate_metrics_summary"

@state(metrics_enabled=True)
async def generate_metrics_summary(context):
    """Generate metrics summary using built-in capabilities"""
    print("📊 Built-in metrics are automatically collected!")
    print("   ✅ State execution times")
    print("   ✅ Memory usage patterns")
    print("   ✅ Success/failure rates")
    print("   ✅ Resource utilization")
    print("   ✅ Custom business metrics")

    # Export metrics in Prometheus format
    metrics_text = metrics_provider.export_metrics()
    print(f"\\n📈 Metrics exported ({len(metrics_text)} characters)")

    context.set_output("metrics_collected", True)
    return None

# Add states to agent
agent.add_state("process_user_request", process_user_request)
agent.add_state("generate_metrics_summary", generate_metrics_summary)

# Run with built-in metrics
async def demo_builtin_metrics():
    contexts = [
        {"user_id": "user_001", "request_type": "search"},
        {"user_id": "user_002", "request_type": "upload"},
        {"user_id": "user_003", "request_type": "analysis"},
    ]

    for ctx in contexts:
        await agent.run(
            initial_state="process_user_request",
            initial_context=ctx
        )
        agent.reset()

if __name__ == "__main__":
    asyncio.run(demo_builtin_metrics())
\`\`\`

## Advanced Metrics Configuration

\`\`\`python
from puffinflow import Agent, state, Priority

# Configure agent with comprehensive metrics
agent = Agent(
    "advanced-metrics",
    # Built-in metrics configuration
    enable_metrics=True,
    metrics_config=MetricsConfig(
        namespace="production_app",
        enable_detailed_metrics=True,
        collect_system_metrics=True,  # CPU, memory, disk I/O
        collect_gc_metrics=True,      # Garbage collection stats
        cardinality_limit=50000       # Prevent metric explosion
    )
)

@state(
    metrics_enabled=True,
    custom_metrics=[
        "execution_time",      # Built-in: time spent in this state
        "memory_usage",        # Built-in: memory used during execution
        "api_calls",          # Custom: count of API calls made
        "processed_items",    # Custom: business metric
        "error_count"         # Custom: error tracking
    ],
    priority=Priority.HIGH
)
async def comprehensive_processing(context):
    """State with comprehensive built-in metrics"""

    print("🔍 Processing with comprehensive metrics...")

    # Puffinflow automatically tracks all enabled metrics
    # Your business logic here
    items = context.get_variable("items", list(range(100)))

    processed = 0
    api_calls = 0
    errors = 0

    for item in items:
        try:
            # Simulate processing
            await asyncio.sleep(0.01)
            processed += 1

            # Simulate API call every 10 items
            if item % 10 == 0:
                api_calls += 1

        except Exception:
            errors += 1

    # Update custom metrics (Puffinflow tracks these automatically)
    context.set_variable("processed_items_count", processed)
    context.set_variable("api_calls_made", api_calls)
    context.set_variable("errors_encountered", errors)

    print(f"✅ Processed {processed} items, made {api_calls} API calls")
    return None

# Add state to agent
agent.add_state("comprehensive_processing", comprehensive_processing)

# The metrics are automatically exported to Prometheus format
# Access via agent.get_metrics() or metrics_provider.export_metrics()
\`\`\`

## Part 2: Built-in Distributed Tracing

Puffinflow includes OpenTelemetry-compatible distributed tracing:

\`\`\`python
from puffinflow.core.observability.tracing import OpenTelemetryTracingProvider
from puffinflow.core.observability.config import TracingConfig
from puffinflow.core.observability.interfaces import SpanType

# Configure built-in tracing
tracing_config = TracingConfig(
    service_name="my-workflow-service",
    sampling_rate=1.0,  # Trace 100% for demo
    enable_console_export=True,
    enable_jaeger_export=False  # Set to True for Jaeger
)

tracer = OpenTelemetryTracingProvider(tracing_config)
agent = Agent("tracing-demo")

@state(
    # Enable automatic tracing for this state
    enable_tracing=True,
    trace_sampling_rate=1.0
)
async def start_traced_workflow(context):
    """Start workflow with built-in tracing"""

    user_id = context.get_variable("user_id", "user_123")

    print(f"🚀 Starting traced workflow for {user_id}")

    # Puffinflow automatically creates spans for:
    # - State execution
    # - Context operations
    # - Inter-state transitions
    # - Error conditions

    # Create custom span using built-in tracer
    with tracer.span("user_authentication", span_type=SpanType.STATE) as span:
        span.set_attribute("user_id", user_id)
        span.set_attribute("auth_method", "oauth")

        # Simulate authentication
        await asyncio.sleep(0.3)

        if user_id.startswith("user_"):
            span.set_attribute("auth_result", "success")
            context.set_variable("authenticated", True)
        else:
            span.set_attribute("auth_result", "failure")
            span.set_status("error", "Invalid user ID format")
            context.set_variable("authenticated", False)

    return "load_user_data"

@state(enable_tracing=True)
async def load_user_data(context):
    """Load user data with automatic tracing"""

    if not context.get_variable("authenticated", False):
        return "handle_auth_failure"

    user_id = context.get_variable("user_id")

    print(f"🗃️ Loading data for {user_id}")

    # Built-in tracing automatically captures:
    # - Function entry/exit
    # - Exception handling
    # - Context variable access
    # - State transitions

    # Add custom span for database operation
    with tracer.span("database_query", span_type=SpanType.DATABASE) as span:
        span.set_attribute("query_type", "user_lookup")
        span.set_attribute("user_id", user_id)

        # Simulate database query
        await asyncio.sleep(0.8)

        user_data = {
            "user_id": user_id,
            "name": f"User {user_id.split('_')[1]}",
            "subscription": "premium"
        }

        span.set_attribute("records_returned", 1)
        context.set_variable("user_data", user_data)

    return "personalize_experience"

@state(enable_tracing=True)
async def personalize_experience(context):
    """Personalize with built-in tracing"""

    user_data = context.get_variable("user_data")

    print(f"🎨 Personalizing for {user_data['name']}")

    # Custom span for AI/ML operation
    with tracer.span("ai_personalization", span_type=SpanType.AI_MODEL) as span:
        span.set_attribute("model_type", "recommendation_engine")
        span.set_attribute("user_tier", user_data["subscription"])

        # Simulate AI processing
        await asyncio.sleep(0.5)

        recommendations = ["item_1", "item_2", "item_3"]
        span.set_attribute("recommendations_count", len(recommendations))

        context.set_variable("recommendations", recommendations)

    print(f"✅ Generated {len(recommendations)} recommendations")
    return None

@state(enable_tracing=True)
async def handle_auth_failure(context):
    """Handle authentication failure with tracing"""

    print("❌ Authentication failed")

    # Puffinflow automatically traces error conditions
    with tracer.span("auth_failure_handler") as span:
        span.set_status("error", "User authentication failed")
        span.set_attribute("error_type", "authentication_error")

        context.set_output("auth_status", "failed")

    return None

# Add states to agent
agent.add_state("start_traced_workflow", start_traced_workflow)
agent.add_state("load_user_data", load_user_data)
agent.add_state("personalize_experience", personalize_experience)
agent.add_state("handle_auth_failure", handle_auth_failure)

# Demo built-in tracing
async def demo_builtin_tracing():
    print("🎬 Built-in Distributed Tracing Demo\\n")

    users = ["user_001", "user_002", "invalid_user"]

    for user_id in users:
        print(f"--- Tracing workflow for {user_id} ---")

        # Puffinflow automatically creates trace context
        await agent.run(
            initial_state="start_traced_workflow",
            initial_context={"user_id": user_id}
        )

        # Access trace information
        trace_summary = agent.get_trace_summary()
        if trace_summary:
            print(f"📊 Trace Summary:")
            print(f"   Total spans: {trace_summary.get('span_count', 0)}")
            print(f"   Duration: {trace_summary.get('total_duration', 0):.3f}s")
            print(f"   Errors: {trace_summary.get('error_count', 0)}")

        agent.reset()
        print()

if __name__ == "__main__":
    asyncio.run(demo_builtin_tracing())
\`\`\`

## Part 3: Built-in Structured Logging

Puffinflow provides structured logging that integrates with your workflow context:

\`\`\`python
import logging
from puffinflow.core.observability.logging import StructuredLogger
from puffinflow.core.observability.config import LoggingConfig

# Configure built-in structured logging
logging_config = LoggingConfig(
    level="INFO",
    format="json",  # or "text"
    include_context=True,
    include_trace_info=True
)

# Get Puffinflow's structured logger
logger = StructuredLogger("workflow", config=logging_config)

agent = Agent("logging-demo")

@state(
    # Enable automatic logging for this state
    enable_logging=True,
    log_level="INFO"
)
async def api_request_with_logging(context):
    """Handle request with built-in structured logging"""

    request_id = context.get_variable("request_id", f"req_{int(time.time())}")
    user_id = context.get_variable("user_id", "anonymous")
    endpoint = context.get_variable("endpoint", "/api/data")

    # Puffinflow automatically logs:
    # - State entry/exit
    # - Execution duration
    # - Context variables accessed
    # - Errors and exceptions
    # - Trace correlation IDs

    # Use built-in structured logger for custom events
    logger.info(
        "Processing API request",
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "endpoint": endpoint,
            "method": "GET"
        }
    )

    try:
        # Simulate API processing
        await asyncio.sleep(0.5)

        logger.info(
            "API request successful",
            extra={
                "request_id": request_id,
                "response_code": 200,
                "duration_ms": 500
            }
        )

        context.set_variable("api_status", "success")
        return "database_operation"

    except Exception as e:
        # Puffinflow automatically logs exceptions with full context
        logger.error(
            "API request failed",
            extra={
                "request_id": request_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "response_code": 500
            }
        )
        raise

@state(enable_logging=True)
async def database_operation(context):
    """Database operation with built-in logging"""

    request_id = context.get_variable("request_id")

    # Built-in logger automatically includes:
    # - Agent name and state name
    # - Execution context
    # - Trace/span IDs for correlation
    # - Timestamp and log level

    logger.info(
        "Starting database query",
        extra={
            "request_id": request_id,
            "query_type": "SELECT",
            "table": "users"
        }
    )

    start_time = time.time()

    try:
        # Simulate database work
        await asyncio.sleep(0.8)

        duration_ms = int((time.time() - start_time) * 1000)

        logger.info(
            "Database query completed",
            extra={
                "request_id": request_id,
                "duration_ms": duration_ms,
                "rows_returned": 1
            }
        )

        return "complete_request"

    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)

        logger.error(
            "Database query failed",
            extra={
                "request_id": request_id,
                "duration_ms": duration_ms,
                "error_type": type(e).__name__
            }
        )
        raise

@state(enable_logging=True, log_level="DEBUG")
async def complete_request(context):
    """Complete request with debug logging"""

    request_id = context.get_variable("request_id")

    # Enable debug logging for detailed information
    logger.debug(
        "Completing request processing",
        extra={
            "request_id": request_id,
            "api_status": context.get_variable("api_status"),
            "total_states": 3
        }
    )

    logger.info(
        "Request processing completed successfully",
        extra={"request_id": request_id}
    )

    context.set_output("status", "completed")
    return None

# Add states to agent
agent.add_state("api_request_with_logging", api_request_with_logging)
agent.add_state("database_operation", database_operation)
agent.add_state("complete_request", complete_request)

# Demo built-in structured logging
async def demo_builtin_logging():
    print("📝 Built-in Structured Logging Demo\\n")

    requests = [
        {"request_id": "req_001", "user_id": "user_123", "endpoint": "/api/profile"},
        {"request_id": "req_002", "user_id": "user_456", "endpoint": "/api/settings"},
    ]

    for req in requests:
        print(f"Processing request: {req['request_id']}")
        await agent.run(
            initial_state="api_request_with_logging",
            initial_context=req
        )
        agent.reset()

if __name__ == "__main__":
    asyncio.run(demo_builtin_logging())
\`\`\`

## Part 4: Built-in Health Monitoring

Puffinflow includes health monitoring capabilities:

\`\`\`python
from puffinflow.core.observability.health import HealthMonitor
from puffinflow.core.observability.config import HealthConfig

# Configure built-in health monitoring
health_config = HealthConfig(
    check_interval=30.0,  # Run health checks every 30 seconds
    enable_system_checks=True,
    enable_dependency_checks=True
)

health_monitor = HealthMonitor(health_config)
agent = Agent("health-demo")

@state(
    # Enable automatic health reporting
    health_check=True,
    health_check_interval=60.0
)
async def monitored_operation(context):
    """Operation with built-in health monitoring"""

    print("🔍 Running monitored operation...")

    # Puffinflow automatically monitors:
    # - State execution health
    # - Resource usage
    # - Error rates
    # - Performance metrics

    # Register custom health check
    health_monitor.register_check(
        "custom_business_check",
        lambda: context.get_variable("business_metric", 0) > 0,
        severity="warning"
    )

    # Simulate work
    await asyncio.sleep(1.0)
    context.set_variable("business_metric", 42)

    # Get current health status
    health_status = health_monitor.get_health_status()
    print(f"📊 Health Status: {health_status['overall_status']}")

    context.set_output("health_status", health_status)
    return None

# Add state to agent
agent.add_state("monitored_operation", monitored_operation)

# Built-in health monitoring runs automatically
# Access via agent.get_health_status() or health_monitor.get_status()
\`\`\`

## Part 5: Integration with External Systems

### Prometheus Integration

\`\`\`python
# Export to Prometheus (built-in)
from puffinflow.integrations.prometheus import PrometheusExporter

# Configure Prometheus export
prometheus_exporter = PrometheusExporter(
    port=8000,
    path="/metrics",
    registry=metrics_provider.registry
)

# Start Prometheus endpoint
prometheus_exporter.start()

# Metrics are automatically exported at http://localhost:8000/metrics
\`\`\`

### Jaeger Tracing Integration

\`\`\`python
# Configure Jaeger export (built-in)
tracing_config = TracingConfig(
    service_name="my-service",
    enable_jaeger_export=True,
    jaeger_endpoint="http://jaeger:14268/api/traces"
)

tracer = OpenTelemetryTracingProvider(tracing_config)
# Traces automatically exported to Jaeger
\`\`\`

### Grafana Dashboard Integration

\`\`\`python
# Use built-in Grafana dashboard templates
from puffinflow.integrations.grafana import GrafanaDashboard

dashboard = GrafanaDashboard.from_template("puffinflow_standard")
dashboard.export("my-workflow-dashboard.json")
\`\`\`

## Configuration Examples

### Production Observability Setup

\`\`\`python
from puffinflow import Agent, state
from puffinflow.core.observability import ObservabilityConfig

# Complete production observability configuration
observability_config = ObservabilityConfig(
    # Metrics
    metrics=MetricsConfig(
        namespace="production_app",
        enable_detailed_metrics=True,
        collect_system_metrics=True,
        cardinality_limit=100000
    ),

    # Tracing
    tracing=TracingConfig(
        service_name="puffinflow-production",
        sampling_rate=0.1,  # 10% sampling for production
        enable_jaeger_export=True,
        jaeger_endpoint="http://jaeger:14268/api/traces"
    ),

    # Logging
    logging=LoggingConfig(
        level="INFO",
        format="json",
        include_context=True,
        include_trace_info=True
    ),

    # Health monitoring
    health=HealthConfig(
        check_interval=60.0,
        enable_system_checks=True,
        enable_dependency_checks=True
    )
)

# Create agent with full observability
agent = Agent(
    "production-workflow",
    observability_config=observability_config
)

@state(
    metrics_enabled=True,
    enable_tracing=True,
    enable_logging=True,
    health_check=True,
    custom_metrics=["business_kpi", "user_satisfaction"]
)
async def production_ready_state(context):
    """Production state with full observability"""

    # All observability features work automatically!
    # - Metrics collected
    # - Traces created
    # - Logs structured
    # - Health monitored

    # Your business logic here
    await asyncio.sleep(1.0)

    context.set_variable("business_kpi", 95.5)
    context.set_variable("user_satisfaction", 4.8)

    return None

# Add state to agent
agent.add_state("production_ready_state", production_ready_state)
\`\`\`

## Quick Reference

### Enable Built-in Observability

\`\`\`python
# Metrics
@state(metrics_enabled=True, custom_metrics=["execution_time"])

# Tracing
@state(enable_tracing=True, trace_sampling_rate=1.0)

# Logging
@state(enable_logging=True, log_level="INFO")

# Health monitoring
@state(health_check=True, health_check_interval=60.0)

# All features together
@state(
    metrics_enabled=True,
    enable_tracing=True,
    enable_logging=True,
    health_check=True
)
\`\`\`

### Access Observability Data

\`\`\`python
# Get metrics
metrics = agent.get_metrics()
prometheus_text = metrics_provider.export_metrics()

# Get trace information
trace_summary = agent.get_trace_summary()
current_spans = tracer.get_active_spans()

# Get health status
health_status = agent.get_health_status()

# Get logs (configured via logging framework)
logs = logger.get_recent_logs()
\`\`\`

## Tips for Beginners

1. **Start with built-in features** - Enable \`metrics_enabled=True\` on your states
2. **Use the configuration objects** - MetricsConfig, TracingConfig, etc. for setup
3. **Enable gradually** - Start with metrics, then add tracing and logging
4. **Use the integrations** - Built-in Prometheus, Jaeger, and Grafana support
5. **Monitor the collectors** - Built-in observability has minimal overhead
6. **Check the exports** - Use \`export_metrics()\` and similar methods to see data

## What's Built Into Puffinflow

✅ **Automatic Metrics Collection**
- State execution times and success rates
- Memory usage and resource consumption
- Custom business metrics
- Prometheus-compatible export

✅ **Distributed Tracing**
- OpenTelemetry-compatible spans
- Automatic context propagation
- Jaeger and console export
- Performance analysis

✅ **Structured Logging**
- JSON-formatted logs with context
- Automatic trace correlation
- Configurable log levels
- Integration with standard logging

✅ **Health Monitoring**
- Built-in system health checks
- Custom health check registration
- Dependency monitoring
- Status aggregation

✅ **External Integrations**
- Prometheus metrics export
- Jaeger tracing export
- Grafana dashboard templates
- Standard observability stack support

Puffinflow's built-in observability gives you production-ready monitoring without any custom implementation required!`.trim();
