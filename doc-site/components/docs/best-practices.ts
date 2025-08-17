export const bestPracticesMarkdown = `# Best Practices Guide

This guide provides recommended patterns, conventions, and practices for building robust, maintainable, and scalable Puffinflow workflows.

## Code Organization

### Project Structure

Organize your Puffinflow projects with a clear, consistent structure:

\`\`\`
my_project/
├── agents/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── ml_trainer.py
│   └── notification_service.py
├── workflows/
│   ├── __init__.py
│   ├── etl_pipeline.py
│   └── ml_pipeline.py
├── models/
│   ├── __init__.py
│   └── data_models.py
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── resources.py
├── tests/
│   ├── test_agents.py
│   └── test_workflows.py
├── requirements.txt
└── main.py
\`\`\`

### Agent Class Organization

Structure agent classes with clear separation of concerns:

\`\`\`python
from puffinflow import Agent, state, Priority
from puffinflow.observability import MetricsCollector
import logging

logger = logging.getLogger(__name__)
metrics = MetricsCollector(namespace="data_processor")

class DataProcessorAgent(Agent):
    """Handles data processing workflows with validation and monitoring."""

    def __init__(self, name: str = "data-processor"):
        super().__init__(name)
        self._setup_states()

    def _setup_states(self):
        """Register all states for this agent."""
        self.add_state("validate_input", self.validate_input)
        self.add_state("process_data", self.process_data, dependencies=["validate_input"])
        self.add_state("save_results", self.save_results, dependencies=["process_data"])
        self.add_state("cleanup", self.cleanup, dependencies=["save_results"])

    @state(
        cpu=1.0,
        memory=512,
        priority=Priority.HIGH,
        max_retries=3,
        timeout=30.0
    )
    async def validate_input(self, context):
        """Validate input data with comprehensive error handling."""
        logger.info("Starting input validation")
        metrics.increment("validation_started")

        try:
            raw_data = context.get_variable("input_data")

            # Validation logic
            if not raw_data:
                raise ValueError("Input data is empty")

            validated_data = self._validate_data_schema(raw_data)
            context.set_variable("validated_data", validated_data)

            logger.info("Input validation successful")
            metrics.increment("validation_success")
            return "process_data"

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            metrics.increment("validation_failed")
            context.set_variable("error", str(e))
            return "cleanup"

    @state(
        cpu=4.0,
        memory=2048,
        priority=Priority.NORMAL,
        max_retries=2,
        timeout=300.0
    )
    async def process_data(self, context):
        """Process validated data with progress tracking."""
        logger.info("Starting data processing")
        metrics.increment("processing_started")

        with metrics.timer("processing_duration"):
            try:
                validated_data = context.get_variable("validated_data")

                # Processing logic with progress tracking
                total_items = len(validated_data)
                processed_items = []

                for i, item in enumerate(validated_data):
                    processed_item = await self._process_item(item)
                    processed_items.append(processed_item)

                    # Progress tracking
                    progress = (i + 1) / total_items
                    context.set_state("progress", progress)

                    if i % 100 == 0:  # Log every 100 items
                        logger.info(f"Processed {i + 1}/{total_items} items")

                context.set_variable("processed_data", processed_items)
                metrics.gauge("processed_items_count", len(processed_items))

                logger.info(f"Processing completed: {len(processed_items)} items")
                metrics.increment("processing_success")
                return "save_results"

            except Exception as e:
                logger.error(f"Processing failed: {e}")
                metrics.increment("processing_failed")
                context.set_variable("error", str(e))
                return "cleanup"

    @state(
        cpu=1.0,
        memory=512,
        priority=Priority.NORMAL,
        max_retries=3,
        timeout=60.0
    )
    async def save_results(self, context):
        """Save processing results with atomic operations."""
        logger.info("Starting result save")
        metrics.increment("save_started")

        try:
            processed_data = context.get_variable("processed_data")

            # Atomic save operation
            result_id = await self._save_to_database(processed_data)

            context.set_variable("result_id", result_id)
            logger.info(f"Results saved with ID: {result_id}")
            metrics.increment("save_success")
            return "cleanup"

        except Exception as e:
            logger.error(f"Save failed: {e}")
            metrics.increment("save_failed")
            context.set_variable("error", str(e))
            return "cleanup"

    @state(
        cpu=0.5,
        memory=256,
        priority=Priority.LOW,
        max_retries=1,
        timeout=30.0
    )
    async def cleanup(self, context):
        """Clean up resources and handle final state."""
        logger.info("Starting cleanup")

        try:
            # Always perform cleanup
            await self._cleanup_temp_files()

            # Check if workflow was successful
            if context.has_variable("result_id"):
                result_id = context.get_variable("result_id")
                logger.info(f"Workflow completed successfully: {result_id}")
                metrics.increment("workflow_success")
            else:
                error = context.get_variable("error", "Unknown error")
                logger.error(f"Workflow failed: {error}")
                metrics.increment("workflow_failed")

            return None  # End workflow

        except Exception as e:
            logger.critical(f"Cleanup failed: {e}")
            metrics.increment("cleanup_failed")
            return None

    # Helper methods
    def _validate_data_schema(self, data):
        """Validate data against expected schema."""
        # Implementation details
        return data

    async def _process_item(self, item):
        """Process a single data item."""
        # Implementation details
        return item

    async def _save_to_database(self, data):
        """Save data to database atomically."""
        # Implementation details
        return "result_123"

    async def _cleanup_temp_files(self):
        """Clean up temporary files and resources."""
        # Implementation details
        pass
\`\`\`

## Error Handling Patterns

### Hierarchical Error Handling

Structure error handling with multiple levels of recovery:

\`\`\`python
@state(max_retries=3, timeout=60.0)
async def robust_operation(context):
    """Robust operation with multiple error recovery levels."""
    try:
        # Primary operation
        result = await primary_operation()
        context.set_variable("result", result)
        return "success_state"

    except TransientError as e:
        # Level 1: Transient errors - retry with exponential backoff
        retry_count = context.get_state("retry_count", 0)
        if retry_count < 3:
            logger.warning(f"Transient error (retry {retry_count + 1}): {e}")
            context.set_state("retry_count", retry_count + 1)
            await asyncio.sleep(2 ** retry_count)  # Exponential backoff
            return "robust_operation"  # Retry same state
        else:
            logger.error("Max retries exceeded for transient error")
            return "fallback_operation"

    except RecoverableError as e:
        # Level 2: Recoverable errors - try alternative approach
        logger.warning(f"Recoverable error, trying fallback: {e}")
        context.set_variable("fallback_reason", str(e))
        return "fallback_operation"

    except CriticalError as e:
        # Level 3: Critical errors - immediate failure
        logger.error(f"Critical error: {e}")
        context.set_variable("critical_error", str(e))
        return "error_handler"

    except Exception as e:
        # Level 4: Unknown errors - log and fail safely
        logger.critical(f"Unknown error: {e}")
        context.set_variable("unknown_error", str(e))
        return "error_handler"

@state(max_retries=1, timeout=30.0)
async def fallback_operation(context):
    """Alternative operation when primary fails."""
    try:
        # Alternative approach
        result = await fallback_approach()
        context.set_variable("result", result)
        context.set_variable("used_fallback", True)
        return "success_state"

    except Exception as e:
        logger.error(f"Fallback also failed: {e}")
        context.set_variable("fallback_error", str(e))
        return "error_handler"
\`\`\`

### Resource-Aware Error Handling

Handle errors differently based on resource constraints:

\`\`\`python
@state(cpu=4.0, memory=2048, max_retries=2)
async def resource_intensive_operation(context):
    """Handle errors based on resource availability."""
    try:
        result = await expensive_operation()
        return "success_state"

    except MemoryError as e:
        # Reduce memory requirements and retry
        logger.warning("Memory error, reducing batch size")
        current_batch = context.get_state("batch_size", 1000)
        new_batch = max(100, current_batch // 2)
        context.set_state("batch_size", new_batch)
        return "reduced_batch_operation"

    except TimeoutError as e:
        # Increase timeout and resources for retry
        logger.warning("Timeout error, increasing resources")
        return "high_resource_operation"
\`\`\`

## Data Management Best Practices

### Type-Safe Data Patterns

Use Pydantic models for complex data structures:

\`\`\`python
from pydantic import BaseModel, validator
from typing import List, Optional
from datetime import datetime

class ProcessingConfig(BaseModel):
    batch_size: int = 1000
    max_retries: int = 3
    timeout_seconds: float = 300.0
    enable_monitoring: bool = True

    @validator('batch_size')
    def batch_size_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('batch_size must be positive')
        return v

class DataItem(BaseModel):
    id: str
    content: str
    metadata: dict
    created_at: datetime
    processed_at: Optional[datetime] = None

class ProcessingResult(BaseModel):
    items_processed: int
    success_count: int
    error_count: int
    processing_time: float
    errors: List[str] = []

@state
async def typed_processing(context):
    """Process data with full type safety."""
    # Get typed configuration
    config = context.get_validated_data("config", ProcessingConfig)

    # Get and validate input data
    raw_items = context.get_variable("raw_items")
    items = [DataItem(**item) for item in raw_items]

    # Process with type safety
    result = ProcessingResult(
        items_processed=len(items),
        success_count=0,
        error_count=0,
        processing_time=0.0
    )

    start_time = time.time()

    for item in items:
        try:
            # Process item
            processed_item = await process_item(item)
            result.success_count += 1
        except Exception as e:
            result.error_count += 1
            result.errors.append(f"Item {item.id}: {str(e)}")

    result.processing_time = time.time() - start_time

    # Store typed result
    context.set_validated_data("result", result)
    return "save_results"
\`\`\`

### Context Data Lifecycle

Manage data lifecycle effectively:

\`\`\`python
@state
async def data_lifecycle_example(context):
    """Example of proper data lifecycle management."""

    # 1. Constants - Set once, never change
    context.set_constant("api_version", "v1.2.0")
    context.set_constant("max_file_size", 10 * 1024 * 1024)  # 10MB

    # 2. Configuration - Validated data
    config = ProcessingConfig(batch_size=500, max_retries=2)
    context.set_validated_data("config", config)

    # 3. Secrets - Secure storage
    context.set_secret("api_key", os.getenv("API_KEY"))

    # 4. Working data - Regular variables
    context.set_variable("batch_id", str(uuid.uuid4()))

    # 5. Temporary data - Auto-expiring cache
    context.set_cached("temp_results", [], ttl=300)  # 5 minutes

    # 6. State-local data - Scoped to current state
    context.set_state("processing_start", time.time())

    return "next_state"

@state
async def cleanup_data(context):
    """Clean up data at appropriate times."""

    # Clear large temporary data
    context.clear_variable("large_dataset")

    # Keep essential results
    essential_data = {
        "result_id": context.get_variable("result_id"),
        "status": "completed",
        "summary": context.get_variable("summary")
    }

    context.set_variable("final_result", essential_data)
    return None
\`\`\`

## Performance Optimization

### Resource Allocation Strategies

Optimize resource allocation based on workload characteristics:

\`\`\`python
# CPU-intensive tasks
@state(cpu=4.0, memory=1024, priority=Priority.NORMAL)
async def cpu_intensive_task(context):
    """Allocate CPU resources for computation-heavy work."""
    # Mathematical modeling, data analysis
    pass

# Memory-intensive tasks
@state(cpu=2.0, memory=8192, priority=Priority.NORMAL)
async def memory_intensive_task(context):
    """Allocate memory for large data processing."""
    # Large dataset processing, caching
    pass

# I/O-intensive tasks
@state(cpu=1.0, memory=512, io=5.0, priority=Priority.LOW)
async def io_intensive_task(context):
    """Optimize for I/O operations."""
    # File processing, network operations
    pass

# Time-critical tasks
@state(cpu=2.0, memory=1024, priority=Priority.HIGH, timeout=10.0)
async def time_critical_task(context):
    """High priority with tight deadline."""
    # Real-time processing, user-facing operations
    pass
\`\`\`

### Batch Processing Patterns

Handle large datasets efficiently:

\`\`\`python
@state(cpu=2.0, memory=2048)
async def batch_processor(context):
    """Process large datasets in optimized batches."""

    dataset = context.get_variable("large_dataset")
    config = context.get_validated_data("config", ProcessingConfig)

    # Determine optimal batch size based on available memory
    available_memory = 2048  # MB from state configuration
    item_size_mb = estimate_item_memory_usage(dataset[0]) if dataset else 1
    optimal_batch_size = min(
        config.batch_size,
        max(1, int(available_memory * 0.8 / item_size_mb))
    )

    logger.info(f"Processing {len(dataset)} items in batches of {optimal_batch_size}")

    results = []
    for i in range(0, len(dataset), optimal_batch_size):
        batch = dataset[i:i + optimal_batch_size]

        # Process batch with progress tracking
        batch_results = await process_batch(batch)
        results.extend(batch_results)

        # Update progress
        progress = min(1.0, (i + optimal_batch_size) / len(dataset))
        context.set_state("progress", progress)

        # Allow other tasks to run
        await asyncio.sleep(0.1)

    context.set_variable("processed_results", results)
    return "save_results"
\`\`\`

## Testing Strategies

### Unit Testing Agent States

Create comprehensive tests for individual states:

\`\`\`python
import pytest
from unittest.mock import AsyncMock, patch
from puffinflow import Agent, Context

@pytest.mark.asyncio
async def test_validate_input_success():
    """Test successful input validation."""
    agent = DataProcessorAgent()
    context = Context("test-workflow")

    # Setup test data
    test_data = {"key": "value", "items": [1, 2, 3]}
    context.set_variable("input_data", test_data)

    # Execute state
    result = await agent.validate_input(context)

    # Assertions
    assert result == "process_data"
    assert context.has_variable("validated_data")
    assert context.get_variable("validated_data") == test_data

@pytest.mark.asyncio
async def test_validate_input_failure():
    """Test input validation with invalid data."""
    agent = DataProcessorAgent()
    context = Context("test-workflow")

    # Setup invalid test data
    context.set_variable("input_data", None)

    # Execute state
    result = await agent.validate_input(context)

    # Assertions
    assert result == "cleanup"
    assert context.has_variable("error")
    assert "empty" in context.get_variable("error").lower()

@pytest.mark.asyncio
async def test_process_data_with_mocks():
    """Test data processing with mocked dependencies."""
    agent = DataProcessorAgent()
    context = Context("test-workflow")

    # Setup test data
    test_data = [{"id": 1}, {"id": 2}]
    context.set_variable("validated_data", test_data)

    # Mock the processing function
    with patch.object(agent, '_process_item', new_callable=AsyncMock) as mock_process:
        mock_process.return_value = {"id": 1, "processed": True}

        # Execute state
        result = await agent.process_data(context)

        # Assertions
        assert result == "save_results"
        assert context.has_variable("processed_data")
        assert len(context.get_variable("processed_data")) == 2
        assert mock_process.call_count == 2
\`\`\`

### Integration Testing

Test complete workflows end-to-end:

\`\`\`python
@pytest.mark.asyncio
async def test_complete_workflow():
    """Test complete data processing workflow."""
    agent = DataProcessorAgent()

    # Setup initial context
    initial_data = {
        "input_data": [
            {"id": 1, "value": "test1"},
            {"id": 2, "value": "test2"}
        ]
    }

    # Mock external dependencies
    with patch.object(agent, '_save_to_database', new_callable=AsyncMock) as mock_save:
        mock_save.return_value = "result_123"

        # Run complete workflow
        final_context = await agent.run(initial_context=initial_data)

        # Assertions
        assert final_context.has_variable("result_id")
        assert final_context.get_variable("result_id") == "result_123"
        assert not final_context.has_variable("error")

@pytest.mark.asyncio
async def test_workflow_error_handling():
    """Test workflow behavior under error conditions."""
    agent = DataProcessorAgent()

    # Setup initial context with invalid data
    initial_data = {"input_data": None}

    # Run workflow
    final_context = await agent.run(initial_context=initial_data)

    # Assertions
    assert final_context.has_variable("error")
    assert not final_context.has_variable("result_id")
\`\`\`

## Monitoring and Observability

### Comprehensive Metrics Collection

Implement detailed monitoring:

\`\`\`python
from puffinflow.observability import MetricsCollector, AlertManager

class MonitoredAgent(Agent):
    """Agent with comprehensive monitoring."""

    def __init__(self, name: str):
        super().__init__(name)
        self.metrics = MetricsCollector(namespace=f"agent_{name}")
        self.alerts = AlertManager()

    @state(cpu=2.0, memory=1024)
    async def monitored_operation(self, context):
        """Operation with comprehensive monitoring."""

        # Start timing
        operation_timer = self.metrics.start_timer("operation_duration")
        self.metrics.increment("operations_started")

        try:
            # Tag metrics with context
            tags = {
                "workflow_id": context.workflow_id,
                "state": "monitored_operation"
            }

            # Business logic with monitoring
            with self.metrics.timer("processing_time", tags=tags):
                result = await self._do_processing()

            # Success metrics
            self.metrics.increment("operations_successful", tags=tags)
            self.metrics.gauge("result_size", len(result), tags=tags)

            # Performance metrics
            memory_usage = self._get_memory_usage()
            self.metrics.gauge("memory_usage_mb", memory_usage, tags=tags)

            context.set_variable("result", result)
            return "next_state"

        except Exception as e:
            # Error metrics
            error_tags = {**tags, "error_type": type(e).__name__}
            self.metrics.increment("operations_failed", tags=error_tags)

            # Alert on critical errors
            if isinstance(e, CriticalError):
                self.alerts.send_alert(
                    level="critical",
                    message=f"Critical error in {context.workflow_id}: {e}",
                    tags=error_tags
                )

            raise

        finally:
            # Always record duration
            operation_timer.stop()

    async def _do_processing(self):
        """Mock processing function."""
        await asyncio.sleep(0.1)
        return ["item1", "item2", "item3"]

    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
\`\`\`

## Deployment Best Practices

### Configuration Management

Use environment-based configuration:

\`\`\`python
# config/settings.py
import os
from pydantic import BaseModel

class AppConfig(BaseModel):
    """Application configuration with environment variable support."""

    # Agent configuration
    max_concurrent_agents: int = int(os.getenv("MAX_CONCURRENT_AGENTS", "10"))
    default_timeout: float = float(os.getenv("DEFAULT_TIMEOUT", "300.0"))

    # Resource configuration
    default_cpu_limit: float = float(os.getenv("DEFAULT_CPU_LIMIT", "2.0"))
    default_memory_limit: int = int(os.getenv("DEFAULT_MEMORY_LIMIT", "1024"))

    # Monitoring configuration
    enable_metrics: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    metrics_port: int = int(os.getenv("METRICS_PORT", "8080"))

    # Database configuration
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///puffinflow.db")

    # External service configuration
    api_base_url: str = os.getenv("API_BASE_URL", "https://api.example.com")
    api_timeout: float = float(os.getenv("API_TIMEOUT", "30.0"))

# Usage in agents
config = AppConfig()

@state(
    cpu=config.default_cpu_limit,
    memory=config.default_memory_limit,
    timeout=config.default_timeout
)
async def configured_state(context):
    """State using environment-based configuration."""
    # Use configuration
    api_url = config.api_base_url
    timeout = config.api_timeout

    # Implementation
    pass
\`\`\`

### Health Checks and Readiness

Implement health monitoring:

\`\`\`python
class HealthCheckAgent(Agent):
    """Agent for health checking and readiness probes."""

    def __init__(self):
        super().__init__("health-check")
        self.setup_health_states()

    def setup_health_states(self):
        """Setup health check states."""
        self.add_state("check_dependencies", self.check_dependencies)
        self.add_state("check_resources", self.check_resources)
        self.add_state("report_health", self.report_health)

    @state(cpu=0.1, memory=128, timeout=10.0)
    async def check_dependencies(self, context):
        """Check external dependencies."""
        health_status = {"dependencies": {}}

        # Check database
        try:
            await self._check_database()
            health_status["dependencies"]["database"] = "healthy"
        except Exception as e:
            health_status["dependencies"]["database"] = f"unhealthy: {e}"

        # Check external APIs
        try:
            await self._check_external_api()
            health_status["dependencies"]["external_api"] = "healthy"
        except Exception as e:
            health_status["dependencies"]["external_api"] = f"unhealthy: {e}"

        context.set_variable("dependency_status", health_status)
        return "check_resources"

    @state(cpu=0.1, memory=128, timeout=5.0)
    async def check_resources(self, context):
        """Check system resources."""
        import psutil

        resource_status = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }

        context.set_variable("resource_status", resource_status)
        return "report_health"

    @state(cpu=0.1, memory=128, timeout=5.0)
    async def report_health(self, context):
        """Generate final health report."""
        dependency_status = context.get_variable("dependency_status")
        resource_status = context.get_variable("resource_status")

        # Determine overall health
        is_healthy = True
        issues = []

        # Check dependencies
        for dep, status in dependency_status["dependencies"].items():
            if not status == "healthy":
                is_healthy = False
                issues.append(f"Dependency {dep}: {status}")

        # Check resources
        if resource_status["cpu_percent"] > 90:
            is_healthy = False
            issues.append(f"High CPU usage: {resource_status['cpu_percent']}%")

        if resource_status["memory_percent"] > 90:
            is_healthy = False
            issues.append(f"High memory usage: {resource_status['memory_percent']}%")

        health_report = {
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "dependencies": dependency_status["dependencies"],
            "resources": resource_status,
            "issues": issues
        }

        context.set_variable("health_report", health_report)

        # Log health status
        if is_healthy:
            logger.info("Health check passed")
        else:
            logger.warning(f"Health check failed: {issues}")

        return None

    async def _check_database(self):
        """Check database connectivity."""
        # Implementation specific to your database
        pass

    async def _check_external_api(self):
        """Check external API connectivity."""
        # Implementation specific to your external dependencies
        pass

# Usage for health endpoints
async def health_check_endpoint():
    """Health check endpoint for load balancers."""
    health_agent = HealthCheckAgent()
    try:
        context = await asyncio.wait_for(
            health_agent.run(),
            timeout=15.0
        )
        report = context.get_variable("health_report")
        return report
    except asyncio.TimeoutError:
        return {
            "status": "unhealthy",
            "error": "Health check timeout",
            "timestamp": datetime.utcnow().isoformat()
        }
\`\`\`

This comprehensive best practices guide provides a foundation for building robust, maintainable, and scalable Puffinflow applications.
`.trim();
