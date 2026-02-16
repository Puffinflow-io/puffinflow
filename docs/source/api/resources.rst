Resources API
=============

The resources module provides advanced resource management capabilities including allocation strategies, quotas, and resource pools.

Resource Management
-------------------

Resource Requirements
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.resources.requirements
   :members:
   :undoc-members:
   :show-inheritance:

Resource Pool
~~~~~~~~~~~~~

.. automodule:: puffinflow.core.resources.pool
   :members:
   :undoc-members:
   :show-inheritance:

Resource Allocation
~~~~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.resources.allocation
   :members:
   :undoc-members:
   :show-inheritance:

Quota Management
~~~~~~~~~~~~~~~~

.. automodule:: puffinflow.core.resources.quotas
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Resource Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import Agent, Context, state, ResourceRequirements, ResourceType

   class ResourceAwareAgent(Agent):
       @state
       async def cpu_intensive_task(self, ctx: Context) -> None:
           """Task requiring specific CPU resources."""
           # Resource requirements are automatically handled
           result = await heavy_computation()
           ctx.result = result

       def get_resource_requirements(self) -> ResourceRequirements:
           return ResourceRequirements(
               cpu_cores=4,
               memory_mb=2048,
               gpu_memory_mb=1024,
               network_bandwidth_mbps=100
           )

Resource Pool Management
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import ResourcePool, ResourceType, AllocationStrategy

   # Create a resource pool
   pool = ResourcePool(
       cpu_cores=16,
       memory_mb=32768,
       gpu_memory_mb=8192,
       allocation_strategy=AllocationStrategy.FAIR_SHARE
   )

   # Register agents with the pool
   await pool.register_agent(agent1)
   await pool.register_agent(agent2)

   # Run agents with resource management
   results = await pool.run_agents([agent1, agent2])

Custom Allocation Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import AllocationStrategy, ResourcePool

   class PriorityAllocationStrategy(AllocationStrategy):
       """Custom allocation strategy based on agent priority."""

       async def allocate(self, requirements, available_resources):
           # Custom allocation logic
           if requirements.priority == "high":
               return self.allocate_premium(requirements, available_resources)
           else:
               return self.allocate_standard(requirements, available_resources)

   # Use custom strategy
   pool = ResourcePool(
       cpu_cores=16,
       memory_mb=32768,
       allocation_strategy=PriorityAllocationStrategy()
   )

Quota Management
~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import QuotaManager, ResourceType

   # Set up quota management
   quota_manager = QuotaManager()

   # Define quotas for different users/teams
   await quota_manager.set_quota("team_a", ResourceType.CPU_CORES, 8)
   await quota_manager.set_quota("team_a", ResourceType.MEMORY_MB, 16384)
   await quota_manager.set_quota("team_b", ResourceType.CPU_CORES, 4)
   await quota_manager.set_quota("team_b", ResourceType.MEMORY_MB, 8192)

   # Check quota before allocation
   can_allocate = await quota_manager.check_quota(
       "team_a",
       ResourceRequirements(cpu_cores=6, memory_mb=12288)
   )

   if can_allocate:
       await quota_manager.allocate_resources("team_a", requirements)

Resource Decorators
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import Agent, Context, state
   from puffinflow import cpu_intensive, memory_intensive, gpu_accelerated

   class OptimizedAgent(Agent):
       @state
       @cpu_intensive(cores=4, priority="high")
       async def cpu_task(self, ctx: Context) -> None:
           """CPU intensive task with specific requirements."""
           ctx.cpu_result = await cpu_heavy_computation()

       @state
       @memory_intensive(memory_mb=4096, swap_allowed=False)
       async def memory_task(self, ctx: Context) -> None:
           """Memory intensive task."""
           ctx.large_data = await load_large_dataset()

       @state
       @gpu_accelerated(gpu_memory_mb=2048, cuda_cores=1024)
       async def gpu_task(self, ctx: Context) -> None:
           """GPU accelerated computation."""
           ctx.gpu_result = await gpu_computation()

Dynamic Resource Scaling
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import ResourcePool, Agent

   class AdaptiveAgent(Agent):
       async def adapt_resources(self, ctx: Context) -> ResourceRequirements:
           """Dynamically adjust resource requirements based on workload."""
           workload_size = len(ctx.input_data)

           if workload_size > 10000:
               return ResourceRequirements(
                   cpu_cores=8,
                   memory_mb=8192,
                   priority="high"
               )
           elif workload_size > 1000:
               return ResourceRequirements(
                   cpu_cores=4,
                   memory_mb=4096,
                   priority="medium"
               )
           else:
               return ResourceRequirements(
                   cpu_cores=2,
                   memory_mb=2048,
                   priority="low"
               )

   # Use with adaptive resource pool
   pool = ResourcePool(
       cpu_cores=32,
       memory_mb=65536,
       enable_dynamic_scaling=True
   )

Resource Monitoring
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import ResourcePool, ResourceMonitor

   # Set up resource monitoring
   monitor = ResourceMonitor()
   pool = ResourcePool(
       cpu_cores=16,
       memory_mb=32768,
       monitor=monitor
   )

   # Run agents with monitoring
   results = await pool.run_agents(agents)

   # Get resource usage statistics
   stats = await monitor.get_usage_stats()
   print(f"Peak CPU usage: {stats.peak_cpu_usage}%")
   print(f"Peak memory usage: {stats.peak_memory_usage}MB")
   print(f"Average allocation time: {stats.avg_allocation_time}ms")

Resource Constraints and Limits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import ResourcePool, ResourceConstraints

   # Define resource constraints
   constraints = ResourceConstraints(
       max_cpu_per_agent=8,
       max_memory_per_agent=16384,
       max_concurrent_agents=10,
       timeout_seconds=300
   )

   pool = ResourcePool(
       cpu_cores=64,
       memory_mb=131072,
       constraints=constraints
   )

   # Agents will be automatically constrained by these limits
   results = await pool.run_agents(agents)
