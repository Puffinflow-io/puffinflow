Agent API
=========

The agent module provides the core workflow orchestration functionality with state-based execution, dependency management, and checkpointing capabilities.

Core Agent Classes
------------------

.. automodule:: puffinflow.core.agent.base
   :members:
   :undoc-members:
   :show-inheritance:

Agent State Management
----------------------

.. automodule:: puffinflow.core.agent.state
   :members:
   :undoc-members:
   :show-inheritance:

Context and Data Management
---------------------------

.. automodule:: puffinflow.core.agent.context
   :members:
   :undoc-members:
   :show-inheritance:

Dependency Resolution
---------------------

.. automodule:: puffinflow.core.agent.dependencies
   :members:
   :undoc-members:
   :show-inheritance:

Checkpointing and Persistence
-----------------------------

.. automodule:: puffinflow.core.agent.checkpoint
   :members:
   :undoc-members:
   :show-inheritance:

Decorators
----------

State Decorators
~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.agent.decorators.builder
   :members:
   :undoc-members:
   :show-inheritance:

Flexible Decorators
~~~~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.agent.decorators.flexible
   :members:
   :undoc-members:
   :show-inheritance:

Inspection Utilities
~~~~~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.agent.decorators.inspection
   :members:
   :undoc-members:
   :show-inheritance:

Scheduling
----------

Scheduler Core
~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.agent.scheduling.scheduler
   :members:
   :undoc-members:
   :show-inheritance:

Input Processing
~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.agent.scheduling.inputs
   :members:
   :undoc-members:
   :show-inheritance:

Parser Utilities
~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.agent.scheduling.parser
   :members:
   :undoc-members:
   :show-inheritance:

Builder Patterns
~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.agent.scheduling.builder
   :members:
   :undoc-members:
   :show-inheritance:

Scheduling Exceptions
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.agent.scheduling.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Agent
~~~~~~~~~~~

.. code-block:: python

   from puffinflow import Agent, Context, state

   class SimpleAgent(Agent):
       @state
       async def process_data(self, ctx: Context) -> None:
           """Process some data."""
           ctx.result = "processed"

   # Run the agent
   agent = SimpleAgent()
   result = await agent.run()

Agent with Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import Agent, Context, state

   class DataPipeline(Agent):
       @state
       async def fetch_data(self, ctx: Context) -> None:
           """Fetch data from source."""
           ctx.raw_data = await fetch_from_api()

       @state(depends_on=["fetch_data"])
       async def process_data(self, ctx: Context) -> None:
           """Process the fetched data."""
           ctx.processed_data = process(ctx.raw_data)

       @state(depends_on=["process_data"])
       async def save_data(self, ctx: Context) -> None:
           """Save processed data."""
           await save_to_db(ctx.processed_data)

Agent with Resource Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import Agent, Context, state, cpu_intensive, memory_intensive

   class ResourceIntensiveAgent(Agent):
       @state
       @cpu_intensive(cores=4)
       async def cpu_heavy_task(self, ctx: Context) -> None:
           """CPU intensive computation."""
           ctx.result = await heavy_computation()

       @state
       @memory_intensive(memory_mb=2048)
       async def memory_heavy_task(self, ctx: Context) -> None:
           """Memory intensive processing."""
           ctx.large_data = await load_large_dataset()

Agent with Checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import Agent, Context, state

   class CheckpointedAgent(Agent):
       def __init__(self):
           super().__init__(enable_checkpointing=True)

       @state
       async def long_running_task(self, ctx: Context) -> None:
           """Long running task with checkpointing."""
           for i in range(1000):
               await process_item(i)
               if i % 100 == 0:
                   await self.save_checkpoint(ctx)
