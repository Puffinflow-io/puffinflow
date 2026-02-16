Migration Guide
===============

This guide helps you migrate to PuffinFlow from other workflow orchestration frameworks or upgrade between PuffinFlow versions.

Migrating from Other Frameworks
--------------------------------

From Airflow
~~~~~~~~~~~~

**Key Differences:**

- **Agent-based**: PuffinFlow uses agents instead of DAGs
- **Async-first**: Built for async/await patterns
- **Type-safe**: Strong typing throughout
- **Resource-aware**: Built-in resource management

**Migration Steps:**

1. **Convert DAGs to Agent Teams**

.. code-block:: python

   # Airflow DAG
   from airflow import DAG
   from airflow.operators.python import PythonOperator

   dag = DAG('data_pipeline', schedule_interval='@daily')

   def extract_data():
       # Extract logic
       pass

   def transform_data():
       # Transform logic
       pass

   extract_task = PythonOperator(
       task_id='extract',
       python_callable=extract_data,
       dag=dag
   )

   transform_task = PythonOperator(
       task_id='transform',
       python_callable=transform_data,
       dag=dag
   )

   extract_task >> transform_task

.. code-block:: python

   # PuffinFlow equivalent
   from puffinflow import Agent, Context
   from puffinflow.core.coordination import AgentTeam

   class ExtractAgent(Agent):
       async def run(self, ctx: Context) -> None:
           # Extract logic (now async)
           data = await self.extract_data()
           ctx.extracted_data = data

   class TransformAgent(Agent):
       async def run(self, ctx: Context) -> None:
           # Transform logic
           data = ctx.extracted_data
           transformed = await self.transform_data(data)
           ctx.transformed_data = transformed

   # Create team with dependencies
   team = AgentTeam([
       ExtractAgent(),
       TransformAgent()
   ])

   # Run the pipeline
   result = await team.run()

2. **Handle Scheduling**

.. code-block:: python

   # PuffinFlow with external scheduler (e.g., APScheduler)
   from apscheduler.schedulers.asyncio import AsyncIOScheduler

   scheduler = AsyncIOScheduler()

   async def run_pipeline():
       team = AgentTeam([ExtractAgent(), TransformAgent()])
       await team.run()

   scheduler.add_job(
       run_pipeline,
       'cron',
       hour=0,  # Daily at midnight
       id='data_pipeline'
   )

   scheduler.start()

From Prefect
~~~~~~~~~~~~

**Key Differences:**

- **Agent abstraction**: Higher-level than Prefect tasks
- **Built-in coordination**: Native multi-agent patterns
- **Resource management**: Integrated resource allocation
- **Observability**: Built-in monitoring and tracing

**Migration Steps:**

1. **Convert Flows to Agent Teams**

.. code-block:: python

   # Prefect flow
   from prefect import flow, task

   @task
   def extract_data():
       return "extracted_data"

   @task
   def transform_data(data):
       return f"transformed_{data}"

   @flow
   def data_pipeline():
       data = extract_data()
       result = transform_data(data)
       return result

.. code-block:: python

   # PuffinFlow equivalent
   from puffinflow import Agent, Context
   from puffinflow.core.coordination import AgentTeam

   class DataPipelineTeam(AgentTeam):
       def __init__(self):
           super().__init__([
               ExtractAgent(),
               TransformAgent()
           ])

   # Usage
   pipeline = DataPipelineTeam()
   result = await pipeline.run()

From Celery
~~~~~~~~~~~

**Key Differences:**

- **Structured workflows**: Beyond simple task queues
- **Context sharing**: Built-in state management
- **Coordination patterns**: Advanced agent coordination
- **Type safety**: Strong typing and validation

**Migration Steps:**

1. **Convert Tasks to Agents**

.. code-block:: python

   # Celery tasks
   from celery import Celery

   app = Celery('tasks')

   @app.task
   def process_item(item_id):
       # Process item
       return f"processed_{item_id}"

   @app.task
   def aggregate_results(results):
       return sum(results)

.. code-block:: python

   # PuffinFlow equivalent
   from puffinflow import Agent, Context
   from puffinflow.core.coordination import AgentPool

   class ProcessItemAgent(Agent):
       async def run(self, ctx: Context) -> None:
           item_id = ctx.item_id
           result = await self.process_item(item_id)
           ctx.result = result

   class AggregateAgent(Agent):
       async def run(self, ctx: Context) -> None:
           results = ctx.all_results
           ctx.final_result = sum(results)

   # Process multiple items
   pool = AgentPool(ProcessItemAgent, pool_size=10)
   results = await pool.process_batch(item_contexts)

Version Migration
-----------------

From 0.x to 1.0
~~~~~~~~~~~~~~~

**Breaking Changes:**

1. **Agent Interface Changes**

.. code-block:: python

   # v0.x
   class MyAgent(Agent):
       def execute(self, context):
           return "result"

.. code-block:: python

   # v1.0
   class MyAgent(Agent):
       async def run(self, ctx: Context) -> None:
           ctx.result = "result"

2. **Context API Changes**

.. code-block:: python

   # v0.x
   context.set_data("key", value)
   value = context.get_data("key")

.. code-block:: python

   # v1.0
   ctx.key = value
   value = ctx.key

3. **Coordination API Changes**

.. code-block:: python

   # v0.x
   from puffinflow.coordination import Coordinator

   coordinator = Coordinator()
   coordinator.add_agent(agent1)
   coordinator.add_agent(agent2)
   result = coordinator.run()

.. code-block:: python

   # v1.0
   from puffinflow.core.coordination import AgentTeam

   team = AgentTeam([agent1, agent2])
   result = await team.run()

**Migration Script:**

.. code-block:: python

   # migration_script.py
   import ast
   import re
   from pathlib import Path

   def migrate_agent_class(content):
       """Migrate agent class definitions."""
       # Replace execute method with run method
       content = re.sub(
           r'def execute\(self, context\):',
           r'async def run(self, ctx: Context) -> None:',
           content
       )

       # Replace context usage
       content = re.sub(r'context\.get_data\("([^"]+)"\)', r'ctx.\1', content)
       content = re.sub(r'context\.set_data\("([^"]+)", ([^)]+)\)', r'ctx.\1 = \2', content)

       return content

   def migrate_file(file_path):
       """Migrate a single Python file."""
       with open(file_path, 'r') as f:
           content = f.read()

       # Apply migrations
       content = migrate_agent_class(content)

       # Add necessary imports
       if 'from puffinflow' in content:
           content = 'from puffinflow import Agent, Context\n' + content

       with open(file_path, 'w') as f:
           f.write(content)

   # Run migration on all Python files
   for py_file in Path('.').rglob('*.py'):
       if 'puffinflow' in py_file.read_text():
           migrate_file(py_file)
           print(f"Migrated: {py_file}")

Common Migration Patterns
-------------------------

State Management
~~~~~~~~~~~~~~~~

**Before (various frameworks):**

.. code-block:: python

   # Global state or external storage
   import redis

   redis_client = redis.Redis()

   def task1():
       result = process_data()
       redis_client.set('task1_result', result)

   def task2():
       data = redis_client.get('task1_result')
       return transform(data)

**After (PuffinFlow):**

.. code-block:: python

   # Built-in context sharing
   class Task1Agent(Agent):
       async def run(self, ctx: Context) -> None:
           result = await self.process_data()
           ctx.task1_result = result

   class Task2Agent(Agent):
       async def run(self, ctx: Context) -> None:
           data = ctx.task1_result
           ctx.final_result = await self.transform(data)

Error Handling
~~~~~~~~~~~~~~

**Before:**

.. code-block:: python

   # Manual retry logic
   import time

   def unreliable_task():
       max_retries = 3
       for attempt in range(max_retries):
           try:
               return risky_operation()
           except Exception as e:
               if attempt == max_retries - 1:
                   raise
               time.sleep(2 ** attempt)

**After (PuffinFlow):**

.. code-block:: python

   # Built-in reliability patterns
   from puffinflow.core.reliability import CircuitBreaker

   class ReliableAgent(Agent):
       def __init__(self):
           super().__init__()
           self.circuit_breaker = CircuitBreaker(
               failure_threshold=3,
               recovery_timeout=30
           )

       async def run(self, ctx: Context) -> None:
           async with self.circuit_breaker:
               ctx.result = await self.risky_operation()

Resource Management
~~~~~~~~~~~~~~~~~~~

**Before:**

.. code-block:: python

   # Manual resource management
   import asyncio

   semaphore = asyncio.Semaphore(5)  # Limit concurrent operations

   async def limited_task():
       async with semaphore:
           return await expensive_operation()

**After (PuffinFlow):**

.. code-block:: python

   # Built-in resource management
   from puffinflow.core.resources import ResourcePool

   class ResourceManagedAgent(Agent):
       def __init__(self):
           super().__init__()
           self.resource_pool = ResourcePool(
               max_concurrent=5,
               max_memory_mb=1024
           )

       async def run(self, ctx: Context) -> None:
           async with self.resource_pool.acquire():
               ctx.result = await self.expensive_operation()

Testing Migration
-----------------

**Before:**

.. code-block:: python

   # Framework-specific testing
   def test_airflow_dag():
       from airflow.models import DagBag

       dagbag = DagBag()
       dag = dagbag.get_dag('my_dag')
       assert dag is not None

**After (PuffinFlow):**

.. code-block:: python

   # Standard async testing
   import pytest
   from puffinflow.testing import create_test_context

   @pytest.mark.asyncio
   async def test_agent():
       agent = MyAgent()
       ctx = create_test_context({'input': 'test_data'})

       result = await agent.run(ctx)

       assert ctx.output == 'expected_result'

Migration Checklist
--------------------

Pre-Migration
~~~~~~~~~~~~~

- [ ] **Audit current workflows** - Document existing processes
- [ ] **Identify dependencies** - Map data flow and dependencies
- [ ] **Plan testing strategy** - How to validate migrated workflows
- [ ] **Set up development environment** - Install PuffinFlow and tools
- [ ] **Create backup** - Backup existing code and configurations

During Migration
~~~~~~~~~~~~~~~~

- [ ] **Start with simple workflows** - Migrate least complex first
- [ ] **Convert incrementally** - One workflow at a time
- [ ] **Test thoroughly** - Validate each migrated component
- [ ] **Update documentation** - Keep docs current with changes
- [ ] **Monitor performance** - Compare before/after metrics

Post-Migration
~~~~~~~~~~~~~~

- [ ] **Performance validation** - Ensure acceptable performance
- [ ] **Integration testing** - Test with external systems
- [ ] **Team training** - Train team on new patterns
- [ ] **Documentation update** - Complete documentation overhaul
- [ ] **Monitoring setup** - Configure observability
- [ ] **Cleanup** - Remove old framework dependencies

Best Practices
--------------

Gradual Migration
~~~~~~~~~~~~~~~~~

1. **Start with new workflows** in PuffinFlow
2. **Migrate simple workflows** first
3. **Keep both systems running** during transition
4. **Migrate complex workflows** last
5. **Decommission old system** after validation

Testing Strategy
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create comprehensive test suite
   import pytest
   from puffinflow.testing import MockAgent, create_test_context

   class TestMigratedWorkflow:
       @pytest.mark.asyncio
       async def test_data_flow(self):
           """Test that data flows correctly between agents."""
           team = DataProcessingTeam()
           ctx = create_test_context({'input_data': sample_data})

           result = await team.run(ctx)

           assert ctx.final_result == expected_result

       @pytest.mark.asyncio
       async def test_error_handling(self):
           """Test error handling in migrated workflow."""
           agent = ProcessingAgent()
           ctx = create_test_context({'invalid_data': None})

           with pytest.raises(ValidationError):
               await agent.run(ctx)

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Monitor migration performance
   from puffinflow.core.observability import MetricsCollector

   metrics = MetricsCollector()

   class MonitoredAgent(Agent):
       async def run(self, ctx: Context) -> None:
           with metrics.timer('agent_execution_time'):
               # Agent logic here
               pass

           metrics.increment('agent_executions')

Getting Help
------------

**Resources:**

- **Documentation**: https://puffinflow.readthedocs.io/
- **GitHub Issues**: https://github.com/yourusername/puffinflow/issues
- **Community Discord**: https://discord.gg/puffinflow
- **Migration Support**: migration@puffinflow.dev

**Common Issues:**

1. **Import errors** - Check Python path and dependencies
2. **Async/await confusion** - Review async programming patterns
3. **Context sharing** - Understand PuffinFlow context model
4. **Performance differences** - Profile and optimize as needed

**Professional Services:**

For complex migrations, consider professional migration services:
- **Assessment and planning**
- **Custom migration tools**
- **Training and support**
- **Performance optimization**

Contact: services@puffinflow.dev
