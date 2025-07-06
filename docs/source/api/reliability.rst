Reliability API
===============

The reliability module provides fault tolerance patterns including circuit breakers, bulkheads, and resource leak detection to ensure robust workflow execution.

Fault Tolerance Patterns
-------------------------

Circuit Breaker
~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.reliability.circuit_breaker
   :members:
   :undoc-members:
   :show-inheritance:

Bulkhead Pattern
~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.reliability.bulkhead
   :members:
   :undoc-members:
   :show-inheritance:

Resource Management
-------------------

Resource Leak Detection
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.reliability.leak_detector
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Circuit Breaker Pattern
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import Agent, Context, state
   from puffinflow.core.reliability import CircuitBreaker, CircuitBreakerConfig

   class ResilientAgent(Agent):
       def __init__(self):
           super().__init__()

           # Configure circuit breaker for external API calls
           self.api_circuit_breaker = CircuitBreaker(
               CircuitBreakerConfig(
                   failure_threshold=5,
                   recovery_timeout=30,
                   expected_exception=ConnectionError
               )
           )

       @state
       async def call_external_api(self, ctx: Context) -> None:
           """Call external API with circuit breaker protection."""
           try:
               async with self.api_circuit_breaker:
                   response = await external_api_call()
                   ctx.api_response = response
           except CircuitBreakerOpenError:
               # Circuit breaker is open, use fallback
               ctx.api_response = await get_cached_response()

Advanced Circuit Breaker Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow.core.reliability import CircuitBreaker, CircuitBreakerConfig

   # Custom circuit breaker with advanced configuration
   config = CircuitBreakerConfig(
       failure_threshold=10,           # Open after 10 failures
       recovery_timeout=60,            # Try to recover after 60 seconds
       expected_exception=(            # Exceptions that count as failures
           ConnectionError,
           TimeoutError,
           HTTPError
       ),
       success_threshold=3,            # Need 3 successes to close
       timeout=30,                     # Individual call timeout
       fallback_function=get_fallback_data
   )

   circuit_breaker = CircuitBreaker(config)

   class AdvancedResilientAgent(Agent):
       @state
       async def protected_operation(self, ctx: Context) -> None:
           async with circuit_breaker:
               ctx.result = await risky_operation()

Bulkhead Pattern
~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow.core.reliability import Bulkhead, BulkheadConfig

   class IsolatedAgent(Agent):
       def __init__(self):
           super().__init__()

           # Create separate bulkheads for different operations
           self.cpu_bulkhead = Bulkhead(
               BulkheadConfig(
                   max_concurrent_calls=4,
                   max_wait_duration=10,
                   name="cpu_intensive_operations"
               )
           )

           self.io_bulkhead = Bulkhead(
               BulkheadConfig(
                   max_concurrent_calls=20,
                   max_wait_duration=5,
                   name="io_operations"
               )
           )

       @state
       async def cpu_intensive_task(self, ctx: Context) -> None:
           """CPU intensive task isolated in its own bulkhead."""
           async with self.cpu_bulkhead:
               ctx.cpu_result = await heavy_computation()

       @state
       async def io_intensive_task(self, ctx: Context) -> None:
           """I/O intensive task isolated in its own bulkhead."""
           async with self.io_bulkhead:
               ctx.io_result = await database_operation()

Resource Leak Detection
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow.core.reliability import ResourceLeakDetector

   class LeakAwareAgent(Agent):
       def __init__(self):
           super().__init__()
           self.leak_detector = ResourceLeakDetector(
               check_interval=60,          # Check every 60 seconds
               memory_threshold=0.8,       # Alert if memory usage > 80%
               file_handle_threshold=1000, # Alert if file handles > 1000
               connection_threshold=100    # Alert if connections > 100
           )

           # Start monitoring
           self.leak_detector.start_monitoring()

       @state
       async def resource_using_task(self, ctx: Context) -> None:
           """Task that uses resources with leak detection."""
           # Open file with proper cleanup
           async with aiofiles.open('data.txt', 'r') as file:
               data = await file.read()

           # Database connection with proper cleanup
           async with database.connection() as conn:
               result = await conn.execute(query)

           ctx.result = result

       async def cleanup(self):
           """Cleanup method called on agent shutdown."""
           await self.leak_detector.stop_monitoring()
           await super().cleanup()

Combined Reliability Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow.core.reliability import (
       CircuitBreaker, CircuitBreakerConfig,
       Bulkhead, BulkheadConfig,
       ResourceLeakDetector
   )

   class HighlyResilientAgent(Agent):
       def __init__(self):
           super().__init__()

           # Circuit breaker for external dependencies
           self.external_service_cb = CircuitBreaker(
               CircuitBreakerConfig(
                   failure_threshold=5,
                   recovery_timeout=30,
                   expected_exception=ServiceUnavailableError
               )
           )

           # Bulkhead for resource isolation
           self.processing_bulkhead = Bulkhead(
               BulkheadConfig(
                   max_concurrent_calls=10,
                   max_wait_duration=15
               )
           )

           # Resource leak detection
           self.leak_detector = ResourceLeakDetector()
           self.leak_detector.start_monitoring()

       @state
       async def resilient_processing(self, ctx: Context) -> None:
           """Highly resilient processing with multiple patterns."""
           try:
               # Use bulkhead for resource isolation
               async with self.processing_bulkhead:
                   # Use circuit breaker for external calls
                   async with self.external_service_cb:
                       external_data = await fetch_external_data()

                   # Process data locally
                   processed_data = await process_data(external_data)
                   ctx.result = processed_data

           except BulkheadFullError:
               # Bulkhead is full, queue for later processing
               await self.queue_for_retry(ctx)

           except CircuitBreakerOpenError:
               # Circuit breaker is open, use cached data
               ctx.result = await get_cached_data()

Retry Mechanisms with Reliability Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from puffinflow.core.reliability import CircuitBreaker, exponential_backoff

   class RetryableAgent(Agent):
       def __init__(self):
           super().__init__()
           self.circuit_breaker = CircuitBreaker(
               CircuitBreakerConfig(failure_threshold=3)
           )

       @state
       async def resilient_operation_with_retry(self, ctx: Context) -> None:
           """Operation with retry logic and circuit breaker."""
           max_retries = 3
           base_delay = 1

           for attempt in range(max_retries + 1):
               try:
                   async with self.circuit_breaker:
                       ctx.result = await unreliable_operation()
                       return  # Success, exit retry loop

               except CircuitBreakerOpenError:
                   # Circuit breaker is open, don't retry
                   ctx.result = await get_fallback_result()
                   return

               except TransientError as e:
                   if attempt == max_retries:
                       # Last attempt failed, re-raise
                       raise

                   # Calculate delay with exponential backoff
                   delay = base_delay * (2 ** attempt)
                   await asyncio.sleep(delay)

Health Checks and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow.core.reliability import HealthChecker, HealthCheck

   class MonitoredAgent(Agent):
       def __init__(self):
           super().__init__()
           self.health_checker = HealthChecker()

           # Register health checks
           self.health_checker.add_check("database", self.check_database)
           self.health_checker.add_check("external_api", self.check_external_api)
           self.health_checker.add_check("memory_usage", self.check_memory)

       async def check_database(self) -> HealthCheck:
           """Check database connectivity."""
           try:
               await database.ping()
               return HealthCheck(
                   name="database",
                   status="healthy",
                   message="Database connection is active"
               )
           except Exception as e:
               return HealthCheck(
                   name="database",
                   status="unhealthy",
                   message=f"Database check failed: {e}"
               )

       async def check_external_api(self) -> HealthCheck:
           """Check external API availability."""
           try:
               response = await external_api.health()
               return HealthCheck(
                   name="external_api",
                   status="healthy" if response.status == 200 else "degraded",
                   message=f"API status: {response.status}"
               )
           except Exception as e:
               return HealthCheck(
                   name="external_api",
                   status="unhealthy",
                   message=f"API check failed: {e}"
               )

       async def check_memory(self) -> HealthCheck:
           """Check memory usage."""
           import psutil
           memory_percent = psutil.virtual_memory().percent

           if memory_percent > 90:
               status = "unhealthy"
           elif memory_percent > 80:
               status = "degraded"
           else:
               status = "healthy"

           return HealthCheck(
               name="memory_usage",
               status=status,
               message=f"Memory usage: {memory_percent}%"
           )

       @state
       async def monitored_processing(self, ctx: Context) -> None:
           """Processing with health monitoring."""
           # Check health before processing
           health_status = await self.health_checker.check_all()

           if not health_status.is_healthy():
               raise HealthCheckFailedError(
                   f"Health checks failed: {health_status.failed_checks}"
               )

           # Proceed with processing
           ctx.result = await process_data()

Graceful Degradation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow.core.reliability import GracefulDegradation

   class DegradableAgent(Agent):
       def __init__(self):
           super().__init__()
           self.degradation = GracefulDegradation()

       @state
       async def adaptive_processing(self, ctx: Context) -> None:
           """Processing that adapts based on system health."""
           system_health = await self.check_system_health()

           if system_health.cpu_usage > 90:
               # High CPU usage - use simplified processing
               ctx.result = await self.simple_processing(ctx.input_data)

           elif system_health.memory_usage > 85:
               # High memory usage - process in smaller batches
               ctx.result = await self.batch_processing(ctx.input_data)

           elif not system_health.external_services_available:
               # External services down - use cached data
               ctx.result = await self.offline_processing(ctx.input_data)

           else:
               # Normal processing
               ctx.result = await self.full_processing(ctx.input_data)

Timeout and Deadline Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from puffinflow.core.reliability import TimeoutManager

   class TimeoutAwareAgent(Agent):
       def __init__(self):
           super().__init__()
           self.timeout_manager = TimeoutManager()

       @state
       async def time_bounded_operation(self, ctx: Context) -> None:
           """Operation with strict time bounds."""
           try:
               # Set operation timeout
               async with self.timeout_manager.timeout(30):  # 30 seconds max
                   ctx.result = await long_running_operation()

           except asyncio.TimeoutError:
               # Handle timeout gracefully
               ctx.result = await get_partial_result()
               ctx.timeout_occurred = True

       @state
       async def deadline_aware_processing(self, ctx: Context) -> None:
           """Processing that respects deadlines."""
           deadline = ctx.get('deadline')
           if deadline:
               remaining_time = (deadline - datetime.utcnow()).total_seconds()

               if remaining_time <= 0:
                   raise DeadlineExceededError("Processing deadline exceeded")

               # Adjust processing based on remaining time
               if remaining_time < 10:
                   ctx.result = await quick_processing(ctx.input_data)
               else:
                   ctx.result = await full_processing(ctx.input_data)
