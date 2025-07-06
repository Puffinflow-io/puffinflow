Coordination API
================

The coordination module provides advanced multi-agent orchestration, team management, and parallel execution capabilities.

Agent Teams and Groups
-----------------------

Agent Team
~~~~~~~~~~

.. automodule:: puffinflow.core.coordination.agent_team
   :members:
   :undoc-members:
   :show-inheritance:

Agent Group
~~~~~~~~~~~

.. automodule:: puffinflow.core.coordination.agent_group
   :members:
   :undoc-members:
   :show-inheritance:

Agent Pool
~~~~~~~~~~

.. automodule:: puffinflow.core.coordination.agent_pool
   :members:
   :undoc-members:
   :show-inheritance:

Orchestration
-------------

Coordinator
~~~~~~~~~~~

.. automodule:: puffinflow.core.coordination.coordinator
   :members:
   :undoc-members:
   :show-inheritance:

Fluent API
~~~~~~~~~~

.. automodule:: puffinflow.core.coordination.fluent_api
   :members:
   :undoc-members:
   :show-inheritance:

Coordination Primitives
-----------------------

Core Primitives
~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.coordination.primitives
   :members:
   :undoc-members:
   :show-inheritance:

Rate Limiting
~~~~~~~~~~~~~

.. automodule:: puffinflow.core.coordination.rate_limiter
   :members:
   :undoc-members:
   :show-inheritance:

Deadlock Detection
~~~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.coordination.deadlock
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Team Coordination
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import AgentTeam, Agent, Context, state

   class DataFetcher(Agent):
       @state
       async def fetch(self, ctx: Context) -> None:
           ctx.data = await fetch_data()

   class DataProcessor(Agent):
       @state
       async def process(self, ctx: Context) -> None:
           ctx.processed = process_data(ctx.data)

   # Create and run team
   team = AgentTeam([DataFetcher(), DataProcessor()])
   result = await team.run()

Parallel Agent Execution
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import run_agents_parallel, Agent

   class WorkerAgent(Agent):
       def __init__(self, worker_id: int):
           super().__init__()
           self.worker_id = worker_id

       @state
       async def work(self, ctx: Context) -> None:
           ctx.result = await do_work(self.worker_id)

   # Run multiple agents in parallel
   agents = [WorkerAgent(i) for i in range(5)]
   results = await run_agents_parallel(agents)

Agent Pool with Work Queue
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import AgentPool, WorkQueue, WorkItem

   class TaskProcessor(Agent):
       @state
       async def process_task(self, ctx: Context) -> None:
           task = ctx.work_item.data
           ctx.result = await process_task(task)

   # Create work queue and agent pool
   queue = WorkQueue()
   pool = AgentPool(TaskProcessor, pool_size=10)

   # Add work items
   for task in tasks:
       await queue.put(WorkItem(data=task))

   # Process all items
   results = await pool.process_queue(queue)

Fluent API Usage
~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import create_team, create_pipeline

   # Create a processing pipeline
   pipeline = (create_pipeline()
               .add_stage(DataFetcher())
               .add_stage(DataValidator())
               .add_stage(DataProcessor())
               .add_stage(DataSaver())
               .build())

   result = await pipeline.run()

   # Create a parallel team
   team = (create_team()
           .add_parallel_group([
               WorkerAgent(1),
               WorkerAgent(2),
               WorkerAgent(3)
           ])
           .add_sequential_stage(AggregatorAgent())
           .build())

   result = await team.run()

Rate Limited Execution
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import AgentPool, RateLimiter

   # Create rate-limited agent pool
   rate_limiter = RateLimiter(max_calls=10, time_window=60)  # 10 calls per minute
   pool = AgentPool(
       APICallAgent,
       pool_size=5,
       rate_limiter=rate_limiter
   )

   results = await pool.process_queue(work_queue)

Event-Driven Coordination
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import EventBus, Agent

   class ProducerAgent(Agent):
       def __init__(self, event_bus: EventBus):
           super().__init__()
           self.event_bus = event_bus

       @state
       async def produce(self, ctx: Context) -> None:
           data = await generate_data()
           await self.event_bus.publish("data_ready", data)

   class ConsumerAgent(Agent):
       def __init__(self, event_bus: EventBus):
           super().__init__()
           self.event_bus = event_bus

       @state
       async def consume(self, ctx: Context) -> None:
           data = await self.event_bus.wait_for("data_ready")
           ctx.result = await process_data(data)

   # Set up event-driven coordination
   event_bus = EventBus()
   producer = ProducerAgent(event_bus)
   consumer = ConsumerAgent(event_bus)

   # Run both agents
   results = await run_agents_parallel([producer, consumer])
