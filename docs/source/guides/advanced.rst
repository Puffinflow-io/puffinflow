Advanced Guide
==============

This guide covers advanced PuffinFlow features and patterns for building sophisticated workflow orchestration systems.

Advanced Agent Patterns
------------------------

Custom State Builders
~~~~~~~~~~~~~~~~~~~~~

Create custom state builders for complex state management:

.. code-block:: python

   from puffinflow import Agent, Context, StateBuilder, state

   class CustomStateBuilder(StateBuilder):
       def __init__(self, validation_rules=None):
           super().__init__()
           self.validation_rules = validation_rules or []

       def with_validation(self, rule):
           """Add validation rule to the state."""
           self.validation_rules.append(rule)
           return self

       def with_timeout(self, seconds):
           """Add timeout to the state."""
           self.timeout = seconds
           return self

       def build(self, func):
           """Build the state with custom logic."""
           async def wrapper(agent_self, ctx: Context):
               # Pre-execution validation
               for rule in self.validation_rules:
                   if not rule(ctx):
                       raise ValidationError(f"Validation failed: {rule.__name__}")

               # Execute with timeout if specified
               if hasattr(self, 'timeout'):
                   return await asyncio.wait_for(func(agent_self, ctx), self.timeout)
               else:
                   return await func(agent_self, ctx)

           return state(func=wrapper, **self.kwargs)

   class AdvancedAgent(Agent):
       @CustomStateBuilder().with_validation(lambda ctx: hasattr(ctx, 'input_data')).with_timeout(30).build
       async def validated_processing(self, ctx: Context) -> None:
           """Processing with validation and timeout."""
           ctx.result = await complex_processing(ctx.input_data)

Dynamic State Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Register states dynamically at runtime:

.. code-block:: python

   from puffinflow import Agent, Context, state

   class DynamicAgent(Agent):
       def __init__(self):
           super().__init__()
           self.dynamic_states = {}

       def register_dynamic_state(self, name: str, func, dependencies=None):
           """Register a state dynamically."""
           @state(depends_on=dependencies or [])
           async def dynamic_state(self, ctx: Context):
               return await func(self, ctx)

           # Add to agent's state registry
           setattr(self, name, dynamic_state.__get__(self, type(self)))
           self.dynamic_states[name] = dynamic_state

       async def execute_dynamic_workflow(self, workflow_definition):
           """Execute a workflow defined at runtime."""
           for step in workflow_definition:
               if step['name'] not in self.dynamic_states:
                   self.register_dynamic_state(
                       step['name'],
                       step['function'],
                       step.get('dependencies')
                   )

           # Execute the dynamically created workflow
           return await self.run()

   # Usage example
   async def custom_processing(agent, ctx):
       ctx.custom_result = "Dynamic processing complete"

   agent = DynamicAgent()
   workflow = [
       {
           'name': 'dynamic_step',
           'function': custom_processing,
           'dependencies': []
       }
   ]

   result = await agent.execute_dynamic_workflow(workflow)

Advanced Resource Management
----------------------------

Custom Allocation Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement custom resource allocation strategies:

.. code-block:: python

   from puffinflow.core.resources import AllocationStrategy, ResourceRequirements

   class PriorityBasedAllocation(AllocationStrategy):
       """Allocate resources based on agent priority."""

       def __init__(self, priority_weights=None):
           self.priority_weights = priority_weights or {
               'critical': 1.0,
               'high': 0.8,
               'medium': 0.6,
               'low': 0.4
           }

       async def allocate(self, requirements: ResourceRequirements, available_resources: dict):
           """Allocate resources based on priority."""
           priority = requirements.metadata.get('priority', 'medium')
           weight = self.priority_weights.get(priority, 0.5)

           # Calculate allocation based on priority weight
           allocated = {}
           for resource_type, requested in requirements.resources.items():
               available = available_resources.get(resource_type, 0)
               allocated[resource_type] = min(requested * weight, available)

           return allocated

   class SmartResourcePool(ResourcePool):
       """Resource pool with intelligent allocation."""

       def __init__(self, **resources):
           super().__init__(**resources)
           self.allocation_history = []
           self.performance_metrics = {}

       async def allocate_with_learning(self, agent_id: str, requirements: ResourceRequirements):
           """Allocate resources and learn from performance."""
           # Get historical performance for this agent
           agent_history = [h for h in self.allocation_history if h['agent_id'] == agent_id]

           if agent_history:
               # Adjust allocation based on historical performance
               avg_efficiency = sum(h['efficiency'] for h in agent_history) / len(agent_history)
               if avg_efficiency > 0.8:
                   # High efficiency agent gets priority
                   requirements.metadata['priority'] = 'high'

           allocation = await self.allocate(requirements)

           # Track allocation for learning
           self.allocation_history.append({
               'agent_id': agent_id,
               'allocation': allocation,
               'timestamp': datetime.utcnow()
           })

           return allocation

Resource Monitoring and Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement advanced resource monitoring:

.. code-block:: python

   from puffinflow.core.resources import ResourceMonitor
   import psutil
   import asyncio

   class AdvancedResourceMonitor(ResourceMonitor):
       """Advanced resource monitoring with predictive capabilities."""

       def __init__(self):
           super().__init__()
           self.metrics_history = []
           self.prediction_model = None

       async def collect_system_metrics(self):
           """Collect comprehensive system metrics."""
           cpu_percent = psutil.cpu_percent(interval=1)
           memory = psutil.virtual_memory()
           disk = psutil.disk_usage('/')
           network = psutil.net_io_counters()

           metrics = {
               'timestamp': datetime.utcnow(),
               'cpu_percent': cpu_percent,
               'memory_percent': memory.percent,
               'memory_available': memory.available,
               'disk_percent': (disk.used / disk.total) * 100,
               'network_bytes_sent': network.bytes_sent,
               'network_bytes_recv': network.bytes_recv
           }

           self.metrics_history.append(metrics)

           # Keep only last 1000 metrics
           if len(self.metrics_history) > 1000:
               self.metrics_history = self.metrics_history[-1000:]

           return metrics

       async def predict_resource_needs(self, time_horizon_minutes=30):
           """Predict future resource needs."""
           if len(self.metrics_history) < 10:
               return None

           # Simple trend analysis (in production, use proper ML models)
           recent_metrics = self.metrics_history[-10:]
           cpu_trend = (recent_metrics[-1]['cpu_percent'] - recent_metrics[0]['cpu_percent']) / 10
           memory_trend = (recent_metrics[-1]['memory_percent'] - recent_metrics[0]['memory_percent']) / 10

           predicted_cpu = recent_metrics[-1]['cpu_percent'] + (cpu_trend * time_horizon_minutes)
           predicted_memory = recent_metrics[-1]['memory_percent'] + (memory_trend * time_horizon_minutes)

           return {
               'predicted_cpu_percent': max(0, min(100, predicted_cpu)),
               'predicted_memory_percent': max(0, min(100, predicted_memory)),
               'confidence': 0.7  # Simple confidence score
           }

       async def optimize_allocation(self, pending_agents):
           """Optimize resource allocation based on predictions."""
           prediction = await self.predict_resource_needs()

           if prediction and prediction['predicted_cpu_percent'] > 90:
               # High CPU predicted, prioritize CPU-light tasks
               return sorted(pending_agents, key=lambda a: a.cpu_requirements)
           elif prediction and prediction['predicted_memory_percent'] > 90:
               # High memory predicted, prioritize memory-light tasks
               return sorted(pending_agents, key=lambda a: a.memory_requirements)

           return pending_agents

Advanced Coordination Patterns
------------------------------

Event-Driven Coordination
~~~~~~~~~~~~~~~~~~~~~~~~~

Implement sophisticated event-driven coordination:

.. code-block:: python

   from puffinflow.core.coordination import EventBus
   import asyncio
   from typing import Dict, List, Callable

   class AdvancedEventBus(EventBus):
       """Event bus with advanced features."""

       def __init__(self):
           super().__init__()
           self.event_filters = {}
           self.event_transformers = {}
           self.event_history = []

       def add_filter(self, event_type: str, filter_func: Callable):
           """Add filter for specific event type."""
           if event_type not in self.event_filters:
               self.event_filters[event_type] = []
           self.event_filters[event_type].append(filter_func)

       def add_transformer(self, event_type: str, transformer_func: Callable):
           """Add transformer for specific event type."""
           self.event_transformers[event_type] = transformer_func

       async def publish_with_processing(self, event_type: str, data: dict):
           """Publish event with filtering and transformation."""
           # Apply filters
           if event_type in self.event_filters:
               for filter_func in self.event_filters[event_type]:
                   if not filter_func(data):
                       return  # Event filtered out

           # Apply transformation
           if event_type in self.event_transformers:
               data = self.event_transformers[event_type](data)

           # Store in history
           self.event_history.append({
               'type': event_type,
               'data': data,
               'timestamp': datetime.utcnow()
           })

           # Publish event
           await self.publish(event_type, data)

   class EventDrivenOrchestrator:
       """Orchestrator using advanced event-driven patterns."""

       def __init__(self):
           self.event_bus = AdvancedEventBus()
           self.agents = {}
           self.workflows = {}

       def register_workflow(self, name: str, workflow_definition: dict):
           """Register a workflow definition."""
           self.workflows[name] = workflow_definition

       async def execute_event_driven_workflow(self, workflow_name: str):
           """Execute workflow based on events."""
           workflow = self.workflows[workflow_name]

           for step in workflow['steps']:
               # Wait for trigger event
               if 'trigger_event' in step:
                   await self.event_bus.wait_for(step['trigger_event'])

               # Execute step
               agent = self.agents[step['agent']]
               result = await agent.run()

               # Publish completion event
               await self.event_bus.publish_with_processing(
                   f"{step['name']}_completed",
                   {'result': result, 'step': step['name']}
               )

Saga Pattern Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement the Saga pattern for distributed transactions:

.. code-block:: python

   from puffinflow import Agent, Context, state
   from typing import List, Dict, Any
   import asyncio

   class SagaStep:
       """Represents a step in a saga."""

       def __init__(self, name: str, action: Callable, compensation: Callable):
           self.name = name
           self.action = action
           self.compensation = compensation
           self.executed = False
           self.compensated = False

   class SagaOrchestrator(Agent):
       """Orchestrates saga execution with compensation."""

       def __init__(self, saga_steps: List[SagaStep]):
           super().__init__()
           self.saga_steps = saga_steps
           self.executed_steps = []

       @state
       async def execute_saga(self, ctx: Context) -> None:
           """Execute saga with automatic compensation on failure."""
           try:
               for step in self.saga_steps:
                   await step.action(ctx)
                   step.executed = True
                   self.executed_steps.append(step)

               ctx.saga_completed = True

           except Exception as e:
               # Saga failed, execute compensations in reverse order
               await self.compensate_saga(ctx, e)
               raise

       async def compensate_saga(self, ctx: Context, original_error: Exception):
           """Execute compensation actions for completed steps."""
           ctx.saga_failed = True
           ctx.original_error = str(original_error)

           # Execute compensations in reverse order
           for step in reversed(self.executed_steps):
               if step.executed and not step.compensated:
                   try:
                       await step.compensation(ctx)
                       step.compensated = True
                   except Exception as comp_error:
                       # Log compensation failure but continue
                       print(f"Compensation failed for {step.name}: {comp_error}")

   # Usage example
   async def book_flight(ctx: Context):
       ctx.flight_booking = await flight_service.book(ctx.flight_details)

   async def cancel_flight(ctx: Context):
       await flight_service.cancel(ctx.flight_booking['id'])

   async def book_hotel(ctx: Context):
       ctx.hotel_booking = await hotel_service.book(ctx.hotel_details)

   async def cancel_hotel(ctx: Context):
       await hotel_service.cancel(ctx.hotel_booking['id'])

   # Create saga
   travel_saga = SagaOrchestrator([
       SagaStep("book_flight", book_flight, cancel_flight),
       SagaStep("book_hotel", book_hotel, cancel_hotel)
   ])

Advanced Observability
----------------------

Custom Metrics and Tracing
~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement custom observability solutions:

.. code-block:: python

   from puffinflow.core.observability import MetricsCollector
   from opentelemetry import trace
   from prometheus_client import Counter, Histogram, Gauge
   import time

   class AdvancedMetricsCollector(MetricsCollector):
       """Advanced metrics collection with custom metrics."""

       def __init__(self):
           super().__init__()

           # Business metrics
           self.business_transactions = Counter(
               'business_transactions_total',
               'Total business transactions',
               ['transaction_type', 'status']
           )

           self.processing_latency = Histogram(
               'processing_latency_seconds',
               'Processing latency distribution',
               ['operation_type'],
               buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
           )

           self.active_workflows = Gauge(
               'active_workflows',
               'Number of active workflows',
               ['workflow_type']
           )

       def record_business_transaction(self, transaction_type: str, status: str):
           """Record business transaction."""
           self.business_transactions.labels(
               transaction_type=transaction_type,
               status=status
           ).inc()

       def record_processing_latency(self, operation_type: str, duration: float):
           """Record processing latency."""
           self.processing_latency.labels(
               operation_type=operation_type
           ).observe(duration)

   class TracedAgent(Agent):
       """Agent with comprehensive tracing."""

       def __init__(self):
           super().__init__()
           self.tracer = trace.get_tracer(__name__)
           self.metrics = AdvancedMetricsCollector()

       @state
       async def traced_processing(self, ctx: Context) -> None:
           """Processing with comprehensive tracing."""
           with self.tracer.start_as_current_span("traced_processing") as span:
               start_time = time.time()

               try:
                   # Add span attributes
                   span.set_attribute("input_size", len(str(ctx.input_data)))
                   span.set_attribute("agent_id", self.id)

                   # Nested span for sub-operation
                   with self.tracer.start_as_current_span("data_validation") as validation_span:
                       await self.validate_input(ctx)
                       validation_span.set_attribute("validation_passed", True)

                   # Main processing
                   with self.tracer.start_as_current_span("core_processing") as processing_span:
                       result = await self.process_data(ctx.input_data)
                       processing_span.set_attribute("output_size", len(str(result)))
                       ctx.result = result

                   # Record success metrics
                   duration = time.time() - start_time
                   self.metrics.record_processing_latency("traced_processing", duration)
                   self.metrics.record_business_transaction("data_processing", "success")

                   span.set_attribute("processing_duration", duration)
                   span.set_status(trace.Status(trace.StatusCode.OK))

               except Exception as e:
                   # Record error metrics
                   self.metrics.record_business_transaction("data_processing", "error")

                   span.set_attribute("error", str(e))
                   span.set_status(trace.Status(
                       trace.StatusCode.ERROR,
                       description=str(e)
                   ))
                   raise

Distributed Tracing Correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement distributed tracing across multiple services:

.. code-block:: python

   from opentelemetry import trace, baggage
   from opentelemetry.propagate import inject, extract
   import aiohttp

   class DistributedAgent(Agent):
       """Agent with distributed tracing support."""

       def __init__(self):
           super().__init__()
           self.tracer = trace.get_tracer(__name__)

       @state
       async def call_external_service(self, ctx: Context) -> None:
           """Call external service with trace propagation."""
           with self.tracer.start_as_current_span("external_service_call") as span:
               # Prepare headers for trace propagation
               headers = {}
               inject(headers)

               # Add baggage for cross-service context
               baggage.set_baggage("user_id", ctx.get("user_id"))
               baggage.set_baggage("request_id", ctx.get("request_id"))

               async with aiohttp.ClientSession() as session:
                   async with session.post(
                       "http://external-service/api/process",
                       json=ctx.input_data,
                       headers=headers
                   ) as response:
                       result = await response.json()

                       span.set_attribute("http.status_code", response.status)
                       span.set_attribute("http.url", str(response.url))

                       ctx.external_result = result

       @state
       async def process_with_context(self, ctx: Context) -> None:
           """Process data with distributed context."""
           # Extract context from incoming request
           if hasattr(ctx, 'headers'):
               parent_context = extract(ctx.headers)
               with trace.use_span(trace.get_current_span(parent_context)):
                   await self.internal_processing(ctx)
           else:
               await self.internal_processing(ctx)

Advanced Error Handling and Recovery
------------------------------------

Circuit Breaker with Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement circuit breaker with detailed metrics:

.. code-block:: python

   from puffinflow.core.reliability import CircuitBreaker, CircuitBreakerConfig
   from prometheus_client import Counter, Histogram, Gauge
   import time

   class MetricsCircuitBreaker(CircuitBreaker):
       """Circuit breaker with comprehensive metrics."""

       def __init__(self, config: CircuitBreakerConfig, name: str):
           super().__init__(config)
           self.name = name

           # Metrics
           self.calls_total = Counter(
               f'circuit_breaker_calls_total',
               'Total circuit breaker calls',
               ['name', 'state', 'result']
           )

           self.state_transitions = Counter(
               f'circuit_breaker_state_transitions_total',
               'Circuit breaker state transitions',
               ['name', 'from_state', 'to_state']
           )

           self.failure_rate = Gauge(
               f'circuit_breaker_failure_rate',
               'Current failure rate',
               ['name']
           )

       async def __aenter__(self):
           start_time = time.time()
           current_state = self.state

           try:
               result = await super().__aenter__()

               # Record successful call
               self.calls_total.labels(
                   name=self.name,
                   state=current_state,
                   result='success'
               ).inc()

               return result

           except Exception as e:
               # Record failed call
               self.calls_total.labels(
                   name=self.name,
                   state=current_state,
                   result='failure'
               ).inc()

               # Update failure rate
               self.failure_rate.labels(name=self.name).set(self.get_failure_rate())

               raise

Adaptive Retry Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

Implement adaptive retry mechanisms:

.. code-block:: python

   import asyncio
   import random
   from typing import List, Type
   import time

   class AdaptiveRetryStrategy:
       """Adaptive retry strategy that learns from failures."""

       def __init__(self):
           self.failure_history = []
           self.success_history = []

       async def execute_with_retry(self, func, *args, **kwargs):
           """Execute function with adaptive retry."""
           max_retries = self.calculate_max_retries()
           base_delay = self.calculate_base_delay()

           for attempt in range(max_retries + 1):
               try:
                   start_time = time.time()
                   result = await func(*args, **kwargs)

                   # Record success
                   duration = time.time() - start_time
                   self.success_history.append({
                       'timestamp': time.time(),
                       'duration': duration,
                       'attempt': attempt
                   })

                   return result

               except Exception as e:
                   # Record failure
                   self.failure_history.append({
                       'timestamp': time.time(),
                       'error_type': type(e).__name__,
                       'attempt': attempt
                   })

                   if attempt == max_retries:
                       raise

                   # Calculate adaptive delay
                   delay = self.calculate_adaptive_delay(attempt, base_delay)
                   await asyncio.sleep(delay)

       def calculate_max_retries(self) -> int:
           """Calculate max retries based on recent success rate."""
           if not self.failure_history and not self.success_history:
               return 3  # Default

           recent_failures = len([f for f in self.failure_history
                                if time.time() - f['timestamp'] < 300])  # Last 5 minutes
           recent_successes = len([s for s in self.success_history
                                 if time.time() - s['timestamp'] < 300])

           if recent_failures > recent_successes * 2:
               return 5  # High failure rate, more retries
           else:
               return 3  # Normal retry count

       def calculate_adaptive_delay(self, attempt: int, base_delay: float) -> float:
           """Calculate adaptive delay based on recent patterns."""
           # Exponential backoff with jitter
           delay = base_delay * (2 ** attempt)

           # Add jitter to prevent thundering herd
           jitter = random.uniform(0.1, 0.3) * delay

           # Adjust based on recent failure patterns
           recent_failures = [f for f in self.failure_history
                            if time.time() - f['timestamp'] < 60]  # Last minute

           if len(recent_failures) > 5:
               delay *= 1.5  # Increase delay if many recent failures

           return delay + jitter

This completes the advanced guide, covering sophisticated patterns and techniques for building robust, scalable workflow orchestration systems with PuffinFlow.
