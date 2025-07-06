Observability API
=================

The observability module provides comprehensive monitoring, tracing, metrics collection, and alerting capabilities for PuffinFlow workflows.

Core Observability
------------------

Core Components
~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.observability.core
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
~~~~~~~~~~~~~

.. automodule:: puffinflow.core.observability.config
   :members:
   :undoc-members:
   :show-inheritance:

Interfaces
~~~~~~~~~~

.. automodule:: puffinflow.core.observability.interfaces
   :members:
   :undoc-members:
   :show-inheritance:

Agent Observability
-------------------

Agent Monitoring
~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.observability.agent
   :members:
   :undoc-members:
   :show-inheritance:

Context Tracking
~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.observability.context
   :members:
   :undoc-members:
   :show-inheritance:

Metrics and Monitoring
----------------------

Metrics Collection
~~~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.observability.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Distributed Tracing
~~~~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.observability.tracing
   :members:
   :undoc-members:
   :show-inheritance:

Event Management
~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.observability.events
   :members:
   :undoc-members:
   :show-inheritance:

Alerting and Notifications
--------------------------

Alerting System
~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.observability.alerting
   :members:
   :undoc-members:
   :show-inheritance:

Decorators
----------

Observability Decorators
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.observability.decorators
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Observability Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import Agent, Context, state
   from puffinflow.core.observability import ObservabilityConfig, setup_observability

   # Configure observability
   config = ObservabilityConfig(
       enable_metrics=True,
       enable_tracing=True,
       enable_logging=True,
       metrics_port=8080,
       jaeger_endpoint="http://localhost:14268/api/traces"
   )

   # Setup observability
   setup_observability(config)

   class MonitoredAgent(Agent):
       @state
       async def process_data(self, ctx: Context) -> None:
           """This method will be automatically traced and monitored."""
           ctx.result = await process_data()

Metrics Collection
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow.core.observability import MetricsCollector, Counter, Histogram, Gauge

   class MetricsAwareAgent(Agent):
       def __init__(self):
           super().__init__()
           self.metrics = MetricsCollector()

           # Define custom metrics
           self.request_counter = Counter(
               "agent_requests_total",
               "Total number of agent requests"
           )
           self.processing_time = Histogram(
               "agent_processing_seconds",
               "Time spent processing requests"
           )
           self.active_tasks = Gauge(
               "agent_active_tasks",
               "Number of currently active tasks"
           )

       @state
       async def process_request(self, ctx: Context) -> None:
           self.request_counter.inc()
           self.active_tasks.inc()

           with self.processing_time.time():
               ctx.result = await process_request(ctx.request)

           self.active_tasks.dec()

Distributed Tracing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow.core.observability import trace, get_tracer
   from opentelemetry import trace as otel_trace

   class TracedAgent(Agent):
       def __init__(self):
           super().__init__()
           self.tracer = get_tracer(__name__)

       @state
       @trace("data_processing")
       async def process_data(self, ctx: Context) -> None:
           """Automatically traced method."""
           with self.tracer.start_as_current_span("fetch_data") as span:
               data = await fetch_data()
               span.set_attribute("data_size", len(data))

           with self.tracer.start_as_current_span("transform_data") as span:
               result = await transform_data(data)
               span.set_attribute("result_size", len(result))
               ctx.result = result

Custom Event Tracking
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow.core.observability import EventTracker, Event

   class EventAwareAgent(Agent):
       def __init__(self):
           super().__init__()
           self.event_tracker = EventTracker()

       @state
       async def process_with_events(self, ctx: Context) -> None:
           # Track custom events
           await self.event_tracker.track(Event(
               name="processing_started",
               data={"input_size": len(ctx.input_data)},
               timestamp=datetime.utcnow()
           ))

           try:
               result = await process_data(ctx.input_data)
               ctx.result = result

               await self.event_tracker.track(Event(
                   name="processing_completed",
                   data={"output_size": len(result)},
                   timestamp=datetime.utcnow()
               ))
           except Exception as e:
               await self.event_tracker.track(Event(
                   name="processing_failed",
                   data={"error": str(e)},
                   timestamp=datetime.utcnow()
               ))
               raise

Alerting Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow.core.observability import AlertManager, Alert, AlertRule

   # Configure alerting
   alert_manager = AlertManager()

   # Define alert rules
   high_error_rate = AlertRule(
       name="high_error_rate",
       condition="error_rate > 0.05",
       severity="critical",
       description="Error rate is above 5%"
   )

   long_processing_time = AlertRule(
       name="long_processing_time",
       condition="avg_processing_time > 30",
       severity="warning",
       description="Average processing time is above 30 seconds"
   )

   # Register alert rules
   await alert_manager.add_rule(high_error_rate)
   await alert_manager.add_rule(long_processing_time)

   # Configure notification channels
   await alert_manager.add_webhook_channel(
       "slack",
       "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
   )

   await alert_manager.add_email_channel(
       "ops_team",
       ["ops@company.com", "alerts@company.com"]
   )

Health Checks and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow.core.observability import HealthChecker, HealthCheck

   class HealthAwareAgent(Agent):
       def __init__(self):
           super().__init__()
           self.health_checker = HealthChecker()

           # Register health checks
           self.health_checker.add_check(
               "database_connection",
               self.check_database_health
           )
           self.health_checker.add_check(
               "external_api",
               self.check_api_health
           )

       async def check_database_health(self) -> HealthCheck:
           try:
               await database.ping()
               return HealthCheck(
                   name="database_connection",
                   status="healthy",
                   message="Database connection is active"
               )
           except Exception as e:
               return HealthCheck(
                   name="database_connection",
                   status="unhealthy",
                   message=f"Database connection failed: {e}"
               )

       async def check_api_health(self) -> HealthCheck:
           try:
               response = await external_api.health_check()
               return HealthCheck(
                   name="external_api",
                   status="healthy" if response.status == 200 else "degraded",
                   message=f"API responded with status {response.status}"
               )
           except Exception as e:
               return HealthCheck(
                   name="external_api",
                   status="unhealthy",
                   message=f"API health check failed: {e}"
               )

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow.core.observability import PerformanceMonitor, monitor_performance

   class PerformanceAwareAgent(Agent):
       def __init__(self):
           super().__init__()
           self.perf_monitor = PerformanceMonitor()

       @state
       @monitor_performance(
           track_memory=True,
           track_cpu=True,
           track_io=True
       )
       async def resource_intensive_task(self, ctx: Context) -> None:
           """This method will be monitored for resource usage."""
           # Memory usage will be tracked
           large_data = await load_large_dataset()

           # CPU usage will be tracked
           processed_data = await cpu_intensive_processing(large_data)

           # I/O operations will be tracked
           await save_to_storage(processed_data)

           ctx.result = processed_data

       async def get_performance_stats(self):
           """Get performance statistics."""
           return await self.perf_monitor.get_stats()

Log Correlation and Structured Logging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import structlog
   from puffinflow.core.observability import setup_structured_logging

   # Setup structured logging with correlation
   setup_structured_logging(
       level="INFO",
       format="json",
       include_trace_id=True
   )

   class LoggingAgent(Agent):
       def __init__(self):
           super().__init__()
           self.logger = structlog.get_logger(__name__)

       @state
       async def process_with_logging(self, ctx: Context) -> None:
           # Structured logging with automatic trace correlation
           self.logger.info(
               "Processing started",
               input_size=len(ctx.input_data),
               agent_id=self.id
           )

           try:
               result = await process_data(ctx.input_data)
               ctx.result = result

               self.logger.info(
                   "Processing completed successfully",
                   output_size=len(result),
                   processing_time=ctx.elapsed_time
               )
           except Exception as e:
               self.logger.error(
                   "Processing failed",
                   error=str(e),
                   error_type=type(e).__name__,
                   input_size=len(ctx.input_data)
               )
               raise
