"""
Resource Management Examples

This example demonstrates resource management capabilities:
- Resource pools and allocation
- Resource requirements and constraints
- Quota management
- Resource-aware agent execution
"""

import asyncio
import time

from puffinflow import (
    Agent,
    AllocationStrategy,
    Context,
    QuotaManager,
    ResourcePool,
    ResourceRequirements,
    ResourceType,
    cpu_intensive,
    memory_intensive,
    state,
)


class ResourceAwareAgent(Agent):
    """Agent that demonstrates resource-aware execution."""

    def __init__(self, name: str, workload_size: str = "medium"):
        super().__init__(name)
        self.set_variable("workload_size", workload_size)
        # Register all decorated states
        self.add_state("analyze_requirements", self.analyze_requirements)
        self.add_state("execute_workload", self.execute_workload)

    @state(cpu=1.0, memory=256.0)
    async def analyze_requirements(self, context: Context):
        """Analyze resource requirements based on workload."""
        workload = self.get_variable("workload_size", "medium")

        # Define resource requirements based on workload
        requirements = {
            "small": {"cpu": 1.0, "memory": 256.0, "duration": 0.1},
            "medium": {"cpu": 2.0, "memory": 512.0, "duration": 0.2},
            "large": {"cpu": 4.0, "memory": 1024.0, "duration": 0.4},
            "xlarge": {"cpu": 8.0, "memory": 2048.0, "duration": 0.8},
        }

        req = requirements.get(workload, requirements["medium"])

        context.set_output("resource_requirements", req)
        context.set_metric("estimated_duration", req["duration"])

        print(f"{self.name} analyzed {workload} workload: {req}")
        return "execute_workload"

    @state(cpu=2.0, memory=512.0)  # Default resources, will be adjusted
    async def execute_workload(self, context: Context):
        """Execute the workload with appropriate resources."""
        requirements = context.get_output("resource_requirements", {})
        duration = requirements.get("duration", 0.2)

        # Simulate resource-intensive work
        start_time = time.time()
        await asyncio.sleep(duration)
        actual_duration = time.time() - start_time

        context.set_output(
            "execution_result",
            {
                "workload": self.get_variable("workload_size"),
                "planned_duration": duration,
                "actual_duration": actual_duration,
                "efficiency": duration / actual_duration
                if actual_duration > 0
                else 1.0,
            },
        )

        print(f"{self.name} completed workload in {actual_duration:.2f}s")
        return None


class DatabaseAgent(Agent):
    """Agent that simulates database operations with resource constraints."""

    def __init__(self, name: str):
        super().__init__(name)
        # Register all decorated states
        self.add_state("query_database", self.query_database)
        self.add_state("optimize_query", self.optimize_query)

    @cpu_intensive(cpu=2.0, memory=1024.0)
    async def query_database(self, context: Context):
        """Execute database queries."""
        # Simulate database query execution
        await asyncio.sleep(0.3)

        result = {
            "query_type": "SELECT",
            "rows_processed": 10000,
            "execution_time": 0.3,
            "memory_used": 1024.0,
            "cpu_utilization": 2.0,
        }

        context.set_output("query_result", result)
        context.set_metric(
            "rows_per_second", result["rows_processed"] / result["execution_time"]
        )

        print(f"{self.name} processed {result['rows_processed']} rows")
        return "optimize_query"

    @state(cpu=1.0, memory=512.0)
    async def optimize_query(self, context: Context):
        """Optimize query performance."""
        query_result = context.get_output("query_result", {})

        # Simulate query optimization
        await asyncio.sleep(0.1)

        optimization = {
            "original_time": query_result.get("execution_time", 0),
            "optimized_time": query_result.get("execution_time", 0) * 0.8,
            "improvement": 20.0,
            "recommendations": [
                "Add index on frequently queried columns",
                "Use query result caching",
                "Optimize JOIN operations",
            ],
        }

        context.set_output("optimization_result", optimization)
        print(f"{self.name} optimized query by {optimization['improvement']}%")
        return None


class MLTrainingAgent(Agent):
    """Agent that simulates ML training with high resource requirements."""

    def __init__(self, name: str):
        super().__init__(name)
        # Register all decorated states
        self.add_state("train_model", self.train_model)

    @memory_intensive(memory=4096.0, cpu=8.0)
    async def train_model(self, context: Context):
        """Train a machine learning model."""
        # Simulate model training
        epochs = 5
        for epoch in range(epochs):
            await asyncio.sleep(0.1)  # Simulate training time per epoch
            accuracy = 0.6 + (epoch * 0.08)  # Improving accuracy

            context.set_metric(f"epoch_{epoch}_accuracy", accuracy)
            print(f"{self.name} epoch {epoch + 1}/{epochs}, accuracy: {accuracy:.2f}")

        final_result = {
            "model_type": "neural_network",
            "epochs": epochs,
            "final_accuracy": accuracy,
            "training_time": epochs * 0.1,
            "memory_peak": 4096.0,
            "cpu_hours": 8.0 * (epochs * 0.1) / 3600,
        }

        context.set_output("training_result", final_result)
        print(f"{self.name} completed training with {accuracy:.2f} accuracy")
        return None


async def demonstrate_resource_pool():
    """Demonstrate resource pool management."""
    print("=== Resource Pool Management ===")

    # Create a resource pool
    pool = ResourcePool(cpu_cores=16.0, memory_gb=32.0, gpu_count=2, storage_gb=1000.0)

    print(f"Created resource pool: {pool.cpu_cores} CPU, {pool.memory_gb}GB RAM")

    # Create resource requirements
    small_req = ResourceRequirements(cpu=2.0, memory=1.0, priority="normal")

    large_req = ResourceRequirements(cpu=8.0, memory=16.0, priority="high")

    print(f"Small workload requirements: {small_req.cpu} CPU, {small_req.memory}GB RAM")
    print(f"Large workload requirements: {large_req.cpu} CPU, {large_req.memory}GB RAM")

    # Check resource availability
    can_allocate_small = pool.can_allocate(small_req)
    can_allocate_large = pool.can_allocate(large_req)

    print(f"Can allocate small workload: {can_allocate_small}")
    print(f"Can allocate large workload: {can_allocate_large}")

    print()


async def demonstrate_quota_management():
    """Demonstrate quota management."""
    print("=== Quota Management ===")

    # Create quota manager
    quota_manager = QuotaManager()

    # Set quotas for different users/teams
    quota_manager.set_quota("team_a", ResourceType.CPU, 10.0)
    quota_manager.set_quota("team_a", ResourceType.MEMORY, 20.0)
    quota_manager.set_quota("team_b", ResourceType.CPU, 6.0)
    quota_manager.set_quota("team_b", ResourceType.MEMORY, 12.0)

    print("Set quotas:")
    print(
        f"  Team A: {quota_manager.get_quota('team_a', ResourceType.CPU)} CPU, "
        f"{quota_manager.get_quota('team_a', ResourceType.MEMORY)} GB RAM"
    )
    print(
        f"  Team B: {quota_manager.get_quota('team_b', ResourceType.CPU)} CPU, "
        f"{quota_manager.get_quota('team_b', ResourceType.MEMORY)} GB RAM"
    )

    # Check quota usage
    team_a_cpu_usage = quota_manager.get_usage("team_a", ResourceType.CPU)
    team_a_memory_usage = quota_manager.get_usage("team_a", ResourceType.MEMORY)

    print(f"Team A current usage: {team_a_cpu_usage} CPU, {team_a_memory_usage} GB RAM")

    # Try to allocate resources
    can_allocate = quota_manager.can_allocate("team_a", ResourceType.CPU, 5.0)
    print(f"Team A can allocate 5.0 CPU: {can_allocate}")

    print()


async def run_resource_aware_agents():
    """Run agents with different resource requirements."""
    print("=== Resource-Aware Agent Execution ===")

    # Create agents with different workload sizes
    agents = [
        ResourceAwareAgent("small-workload", "small"),
        ResourceAwareAgent("medium-workload", "medium"),
        ResourceAwareAgent("large-workload", "large"),
    ]

    # Run agents and measure resource usage
    start_time = time.time()

    results = {}
    for agent in agents:
        result = await agent.run()
        results[agent.name] = result

        execution_result = result.get_output("execution_result", {})
        print(f"  {agent.name}: {execution_result.get('efficiency', 0):.2f} efficiency")

    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    print()


async def run_specialized_agents():
    """Run specialized agents with specific resource patterns."""
    print("=== Specialized Agent Execution ===")

    # Create specialized agents
    db_agent = DatabaseAgent("database-processor")
    ml_agent = MLTrainingAgent("ml-trainer")

    # Run database agent
    print("Running database agent...")
    db_result = await db_agent.run()
    query_result = db_result.get_output("query_result", {})
    optimization = db_result.get_output("optimization_result", {})

    print(f"  Database: {query_result.get('rows_processed', 0)} rows processed")
    print(f"  Optimization: {optimization.get('improvement', 0)}% improvement")

    # Run ML training agent
    print("Running ML training agent...")
    ml_result = await ml_agent.run()
    training_result = ml_result.get_output("training_result", {})

    print(f"  ML Training: {training_result.get('final_accuracy', 0):.2f} accuracy")
    print(f"  Training time: {training_result.get('training_time', 0):.2f} seconds")

    print()


async def demonstrate_allocation_strategies():
    """Demonstrate different allocation strategies."""
    print("=== Allocation Strategies ===")

    # Create resource pool
    pool = ResourcePool(cpu_cores=8.0, memory_gb=16.0)

    # Create multiple resource requests
    requests = [
        ResourceRequirements(cpu=2.0, memory=4.0, priority="high"),
        ResourceRequirements(cpu=3.0, memory=6.0, priority="normal"),
        ResourceRequirements(cpu=4.0, memory=8.0, priority="low"),
        ResourceRequirements(cpu=1.0, memory=2.0, priority="high"),
    ]

    print("Resource requests:")
    for i, req in enumerate(requests):
        print(
            f"  Request {i+1}: {req.cpu} CPU, {req.memory}GB RAM, priority: {req.priority}"
        )

    # Simulate different allocation strategies
    strategies = [
        AllocationStrategy.FIRST_FIT,
        AllocationStrategy.BEST_FIT,
        AllocationStrategy.PRIORITY_BASED,
    ]

    for strategy in strategies:
        print(f"\nUsing {strategy.value} strategy:")
        allocated = []
        remaining_cpu = pool.cpu_cores
        remaining_memory = pool.memory_gb

        for i, req in enumerate(requests):
            if remaining_cpu >= req.cpu and remaining_memory >= req.memory:
                allocated.append(i + 1)
                remaining_cpu -= req.cpu
                remaining_memory -= req.memory

        print(f"  Allocated requests: {allocated}")
        print(f"  Remaining resources: {remaining_cpu} CPU, {remaining_memory}GB RAM")

    print()


async def run_concurrent_resource_usage():
    """Demonstrate concurrent resource usage patterns."""
    print("=== Concurrent Resource Usage ===")

    # Create multiple agents that will run concurrently
    concurrent_agents = [
        ResourceAwareAgent(f"concurrent-{i}", "medium") for i in range(3)
    ]

    # Run agents concurrently and measure resource contention
    start_time = time.time()

    # Use asyncio.gather to run agents concurrently
    tasks = [agent.run() for agent in concurrent_agents]
    results = await asyncio.gather(*tasks)

    concurrent_time = time.time() - start_time

    print(f"Concurrent execution completed in {concurrent_time:.2f} seconds")

    # Analyze results
    total_efficiency = 0
    for i, result in enumerate(results):
        execution_result = result.get_output("execution_result", {})
        efficiency = execution_result.get("efficiency", 0)
        total_efficiency += efficiency
        print(f"  Agent {i+1}: {efficiency:.2f} efficiency")

    average_efficiency = total_efficiency / len(results)
    print(f"Average efficiency: {average_efficiency:.2f}")
    print()


async def main():
    """Run all resource management examples."""
    print("PuffinFlow Resource Management Examples")
    print("=" * 50)

    await demonstrate_resource_pool()
    await demonstrate_quota_management()
    await run_resource_aware_agents()
    await run_specialized_agents()
    await demonstrate_allocation_strategies()
    await run_concurrent_resource_usage()

    print("All resource management examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
