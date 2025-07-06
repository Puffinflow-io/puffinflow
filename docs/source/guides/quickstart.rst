Quick Start Guide
=================

This guide will help you get started with PuffinFlow quickly and efficiently.

Installation
------------

Basic Installation
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install puffinflow

With Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # With observability features
   pip install puffinflow[observability]

   # With all optional dependencies
   pip install puffinflow[all]

   # For development
   pip install puffinflow[dev]

Your First Agent
----------------

Let's create a simple agent that processes data:

.. code-block:: python

   import asyncio
   from puffinflow import Agent, Context, state

   class HelloWorldAgent(Agent):
       @state(profile="quick")
       async def greet(self, ctx: Context) -> None:
           """A simple greeting state."""
           name = ctx.get('name', 'World')
           ctx.greeting = f"Hello, {name}!"
           print(ctx.greeting)

   async def main():
       # Create and run the agent
       agent = HelloWorldAgent()

       # Run with custom input
       context = Context({'name': 'PuffinFlow'})
       result = await agent.run(context)
       print(f"Status: {result.status}")
       print(f"Greeting: {result.context.greeting}")

   if __name__ == "__main__":
       asyncio.run(main())

Multi-State Workflow
--------------------

Now let's create a more complex workflow with multiple states and dependencies:

.. code-block:: python

   import asyncio
   from puffinflow import Agent, Context, state

   class DataPipeline(Agent):
       """A simple data processing pipeline."""

       @state(profile="io_intensive")
       async def load_data(self, ctx: Context) -> None:
           """Load data from source."""
           # Simulate loading data
           await asyncio.sleep(0.1)
           ctx.raw_data = [1, 2, 3, 4, 5]
           print(f"Loaded {len(ctx.raw_data)} records")

       @state(depends_on=["load_data"], profile="cpu_intensive")
       async def transform_data(self, ctx: Context) -> None:
           """Transform the loaded data."""
           # Transform data (multiply by 2)
           ctx.processed_data = [x * 2 for x in ctx.raw_data]
           print(f"Transformed data: {ctx.processed_data}")

       @state(depends_on=["transform_data"], profile="io_intensive")
       async def save_data(self, ctx: Context) -> None:
           """Save processed data."""
           # Simulate saving
           await asyncio.sleep(0.1)
           ctx.saved_location = "/tmp/processed_data.json"
           ctx.records_saved = len(ctx.processed_data)
           print(f"Saved {ctx.records_saved} records to {ctx.saved_location}")

   async def main():
       pipeline = DataPipeline()
       result = await pipeline.run()

       print(f"\nPipeline Status: {result.status}")
       print(f"Records processed: {result.context.records_saved}")
       print(f"Saved to: {result.context.saved_location}")

   asyncio.run(main())

AI/ML Quick Examples
--------------------

Here are some quick examples for AI/ML workflows:

Simple RAG Agent
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import Agent, Context, state
   from puffinflow.core.coordination import RateLimiter

   class SimpleRAG(Agent):
       """A basic RAG implementation."""

       def __init__(self):
           super().__init__()
           self.rate_limiter = RateLimiter(max_calls=10, time_window=60)

       @state(profile="cpu_intensive")
       async def embed_query(self, ctx: Context) -> None:
           """Generate embedding for the query."""
           # Simulate embedding generation
           import numpy as np
           ctx.query_embedding = np.random.randn(384).tolist()

       @state(depends_on=["embed_query"], profile="memory_intensive")
       async def retrieve_documents(self, ctx: Context) -> None:
           """Retrieve relevant documents."""
           # Simulate document retrieval
           ctx.retrieved_docs = [
               {"text": "Sample document 1", "score": 0.9},
               {"text": "Sample document 2", "score": 0.8}
           ]

       @state(depends_on=["retrieve_documents"], profile="external_service")
       async def generate_response(self, ctx: Context) -> None:
           """Generate response using LLM."""
           async with self.rate_limiter:
               # Simulate LLM call
               await asyncio.sleep(1.0)
               ctx.response = f"Answer to '{ctx.query}' based on retrieved documents"

   # Usage
   async def main():
       rag = SimpleRAG()
       context = Context({'query': 'What is machine learning?'})
       result = await rag.run(context)
       print(f"Response: {result.context.response}")

   asyncio.run(main())

Model Training Pipeline
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import Agent, Context, state, AgentTeam

   class DataLoader(Agent):
       @state(profile="io_intensive")
       async def load_training_data(self, ctx: Context) -> None:
           """Load training data."""
           # Simulate data loading
           await asyncio.sleep(0.5)
           ctx.train_data = list(range(1000))
           ctx.val_data = list(range(100, 200))

   class ModelTrainer(Agent):
       @state(depends_on=["load_training_data"], profile="gpu_accelerated")
       async def train_model(self, ctx: Context) -> None:
           """Train the model."""
           # Simulate training
           await asyncio.sleep(2.0)
           ctx.model_accuracy = 0.95
           ctx.model_path = "/models/trained_model.pt"

   class ModelEvaluator(Agent):
       @state(depends_on=["train_model"], profile="cpu_intensive")
       async def evaluate_model(self, ctx: Context) -> None:
           """Evaluate model performance."""
           ctx.test_accuracy = ctx.model_accuracy - 0.02  # Simulate test performance
           ctx.evaluation_complete = True

   async def main():
       # Create training pipeline
       training_team = AgentTeam([
           DataLoader(),
           ModelTrainer(),
           ModelEvaluator()
       ])

       result = await training_team.run()
       print(f"Training complete! Test accuracy: {result.context.test_accuracy:.2f}")

   asyncio.run(main())

       # Create context with input data
       context = Context({'name': 'PuffinFlow'})

       # Run the agent
       result = await agent.run(context)

       print(f"Agent completed with status: {result.status}")
       print(f"Final greeting: {result.context.greeting}")

   if __name__ == "__main__":
       asyncio.run(main())

Multi-State Workflow
--------------------

Create an agent with multiple states and dependencies:

.. code-block:: python

   from puffinflow import Agent, Context, state

   class DataProcessingAgent(Agent):
       @state
       async def fetch_data(self, ctx: Context) -> None:
           """Fetch data from a source."""
           # Simulate data fetching
           await asyncio.sleep(1)
           ctx.raw_data = [1, 2, 3, 4, 5]
           print(f"Fetched data: {ctx.raw_data}")

       @state(depends_on=["fetch_data"])
       async def process_data(self, ctx: Context) -> None:
           """Process the fetched data."""
           # Process the data (multiply by 2)
           ctx.processed_data = [x * 2 for x in ctx.raw_data]
           print(f"Processed data: {ctx.processed_data}")

       @state(depends_on=["process_data"])
       async def save_results(self, ctx: Context) -> None:
           """Save the processed results."""
           # Simulate saving
           await asyncio.sleep(0.5)
           ctx.saved = True
           print("Results saved successfully!")

   async def main():
       agent = DataProcessingAgent()
       result = await agent.run()

       print(f"Workflow completed: {result.status}")
       print(f"Final data: {result.context.processed_data}")

   asyncio.run(main())

Resource Management
-------------------

Add resource requirements to your agents:

.. code-block:: python

   from puffinflow import Agent, Context, state, cpu_intensive, memory_intensive

   class ResourceAwareAgent(Agent):
       @state
       @cpu_intensive(cores=2)
       async def cpu_heavy_task(self, ctx: Context) -> None:
           """CPU intensive computation."""
           # Simulate CPU intensive work
           result = sum(i * i for i in range(100000))
           ctx.cpu_result = result
           print(f"CPU task result: {result}")

       @state
       @memory_intensive(memory_mb=1024)
       async def memory_heavy_task(self, ctx: Context) -> None:
           """Memory intensive processing."""
           # Simulate memory intensive work
           large_list = list(range(100000))
           ctx.memory_result = len(large_list)
           print(f"Memory task processed {len(large_list)} items")

   async def main():
       agent = ResourceAwareAgent()
       result = await agent.run()
       print(f"Resource-aware workflow completed: {result.status}")

   asyncio.run(main())

Error Handling and Retry
-------------------------

Add error handling and retry logic:

.. code-block:: python

   import random
   from puffinflow import Agent, Context, state

   class ResilientAgent(Agent):
       @state(retry_count=3, retry_delay=1.0)
       async def unreliable_task(self, ctx: Context) -> None:
           """A task that might fail randomly."""
           if random.random() < 0.7:  # 70% chance of failure
               raise Exception("Random failure occurred!")

           ctx.success = True
           print("Task completed successfully!")

       @state(depends_on=["unreliable_task"])
       async def cleanup_task(self, ctx: Context) -> None:
           """Cleanup task that runs after the main task."""
           print("Performing cleanup...")
           ctx.cleaned_up = True

   async def main():
       agent = ResilientAgent()
       try:
           result = await agent.run()
           print(f"Agent completed: {result.status}")
       except Exception as e:
           print(f"Agent failed: {e}")

   asyncio.run(main())

Agent Coordination
------------------

Coordinate multiple agents working together:

.. code-block:: python

   from puffinflow import Agent, Context, state, AgentTeam

   class ProducerAgent(Agent):
       @state
       async def produce_data(self, ctx: Context) -> None:
           """Produce data for processing."""
           ctx.data = list(range(10))
           print(f"Produced data: {ctx.data}")

   class ProcessorAgent(Agent):
       @state
       async def process_data(self, ctx: Context) -> None:
           """Process data from producer."""
           if hasattr(ctx, 'data'):
               ctx.processed = [x ** 2 for x in ctx.data]
               print(f"Processed data: {ctx.processed}")
           else:
               print("No data to process")

   class ConsumerAgent(Agent):
       @state
       async def consume_data(self, ctx: Context) -> None:
           """Consume processed data."""
           if hasattr(ctx, 'processed'):
               ctx.sum = sum(ctx.processed)
               print(f"Sum of processed data: {ctx.sum}")

   async def main():
       # Create a team of agents
       team = AgentTeam([
           ProducerAgent(),
           ProcessorAgent(),
           ConsumerAgent()
       ])

       result = await team.run()
       print(f"Team completed: {result.status}")

   asyncio.run(main())

Parallel Execution
------------------

Run multiple agents in parallel:

.. code-block:: python

   from puffinflow import Agent, Context, state, run_agents_parallel

   class WorkerAgent(Agent):
       def __init__(self, worker_id: int):
           super().__init__()
           self.worker_id = worker_id

       @state
       async def do_work(self, ctx: Context) -> None:
           """Simulate work being done."""
           await asyncio.sleep(random.uniform(1, 3))
           ctx.result = f"Worker {self.worker_id} completed"
           print(ctx.result)

   async def main():
       # Create multiple worker agents
       workers = [WorkerAgent(i) for i in range(5)]

       # Run all workers in parallel
       results = await run_agents_parallel(workers)

       print("All workers completed:")
       for i, result in enumerate(results):
           print(f"  Worker {i}: {result.status}")

   asyncio.run(main())

Checkpointing and Recovery
--------------------------

Enable checkpointing for long-running workflows:

.. code-block:: python

   from puffinflow import Agent, Context, state

   class CheckpointedAgent(Agent):
       def __init__(self):
           super().__init__(enable_checkpointing=True)

       @state
       async def long_running_task(self, ctx: Context) -> None:
           """A long-running task with checkpointing."""
           for i in range(10):
               # Simulate work
               await asyncio.sleep(0.5)
               ctx.progress = i + 1

               # Save checkpoint every 3 iterations
               if (i + 1) % 3 == 0:
                   await self.save_checkpoint(ctx)
                   print(f"Checkpoint saved at progress: {ctx.progress}")

           ctx.completed = True
           print("Long-running task completed!")

   async def main():
       agent = CheckpointedAgent()

       # Try to restore from checkpoint first
       restored = await agent.restore_from_checkpoint()
       if restored:
           print("Restored from checkpoint")

       result = await agent.run()
       print(f"Agent completed: {result.status}")

   asyncio.run(main())

Configuration and Settings
--------------------------

Configure PuffinFlow for your environment:

.. code-block:: python

   from puffinflow import get_settings, Agent, Context, state

   # Get current settings
   settings = get_settings()
   print(f"Default timeout: {settings.default_timeout}")
   print(f"Max retries: {settings.max_retries}")

   # You can also override settings
   class ConfiguredAgent(Agent):
       def __init__(self):
           super().__init__(
               timeout=30,  # 30 second timeout
               max_retries=5,  # Maximum 5 retries
               enable_checkpointing=True
           )

       @state
       async def configured_task(self, ctx: Context) -> None:
           """Task with custom configuration."""
           ctx.message = "Task with custom settings"
           print(ctx.message)

   async def main():
       agent = ConfiguredAgent()
       result = await agent.run()
       print(f"Configured agent completed: {result.status}")

   asyncio.run(main())

Next Steps
----------

Now that you've learned the basics, you can:

1. **Explore Advanced Features**: Check out the :doc:`advanced` guide for more sophisticated patterns
2. **See More Examples**: Browse the :doc:`examples` section for real-world use cases
3. **Read the API Documentation**: Dive deep into the :doc:`../api/index` for complete reference
4. **Add Observability**: Learn about monitoring and tracing in the observability documentation
5. **Implement Reliability Patterns**: Explore circuit breakers and bulkheads for fault tolerance

Common Patterns
---------------

Here are some common patterns you'll use frequently:

Sequential Processing
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class SequentialAgent(Agent):
       @state
       async def step_1(self, ctx: Context) -> None:
           ctx.step1_result = "Step 1 complete"

       @state(depends_on=["step_1"])
       async def step_2(self, ctx: Context) -> None:
           ctx.step2_result = "Step 2 complete"

       @state(depends_on=["step_2"])
       async def step_3(self, ctx: Context) -> None:
           ctx.final_result = "All steps complete"

Conditional Execution
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ConditionalAgent(Agent):
       @state
       async def check_condition(self, ctx: Context) -> None:
           ctx.should_process = ctx.get('input_value', 0) > 10

       @state(depends_on=["check_condition"])
       async def conditional_processing(self, ctx: Context) -> None:
           if ctx.should_process:
               ctx.result = "Processing performed"
           else:
               ctx.result = "Processing skipped"

Data Pipeline
~~~~~~~~~~~~~

.. code-block:: python

   class DataPipelineAgent(Agent):
       @state
       async def extract(self, ctx: Context) -> None:
           """Extract data from source."""
           ctx.raw_data = await extract_from_source()

       @state(depends_on=["extract"])
       async def transform(self, ctx: Context) -> None:
           """Transform the extracted data."""
           ctx.transformed_data = await transform_data(ctx.raw_data)

       @state(depends_on=["transform"])
       async def load(self, ctx: Context) -> None:
           """Load data to destination."""
           await load_to_destination(ctx.transformed_data)
           ctx.pipeline_complete = True

This completes the quick start guide. You now have the foundation to build powerful workflows with PuffinFlow!
