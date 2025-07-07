#!/usr/bin/env python3
"""
Benchmark suite for core agent execution performance.
"""

import asyncio
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from puffinflow.core.agent.base import Agent
from puffinflow.core.coordination.coordinator import AgentCoordinator
from puffinflow.core.coordination.primitives import CoordinationPrimitive
from puffinflow.core.observability.metrics import PrometheusMetricsProvider
from puffinflow.core.resources.pool import ResourcePool
from puffinflow.core.resources.requirements import ResourceRequirements

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class BenchmarkResult:
    """Benchmark result container."""

    name: str
    duration_ms: float
    memory_mb: float
    cpu_percent: float
    iterations: int
    min_time: float
    max_time: float
    median_time: float
    std_dev: float


class BenchmarkRunner:
    """Benchmark runner with resource monitoring."""

    def __init__(self):
        self.process = psutil.Process()
        self.results: list[BenchmarkResult] = []

    def run_benchmark(
        self,
        name: str,
        func,
        iterations: int = 100,
        warmup_iterations: int = 10,
        *args,
        **kwargs,
    ) -> BenchmarkResult:
        """Run a benchmark with performance monitoring."""
        print(f"Running benchmark: {name}")

        # Warmup
        for _ in range(warmup_iterations):
            if asyncio.iscoroutinefunction(func):
                asyncio.run(func(*args, **kwargs))
            else:
                func(*args, **kwargs)

        # Benchmark
        times = []
        memory_before = self.process.memory_info().rss / 1024 / 1024
        cpu_before = self.process.cpu_percent()

        for _ in range(iterations):
            start_time = time.perf_counter()

            if asyncio.iscoroutinefunction(func):
                asyncio.run(func(*args, **kwargs))
            else:
                func(*args, **kwargs)

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        memory_after = self.process.memory_info().rss / 1024 / 1024
        cpu_after = self.process.cpu_percent()

        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0

        result = BenchmarkResult(
            name=name,
            duration_ms=avg_time,
            memory_mb=memory_after - memory_before,
            cpu_percent=cpu_after - cpu_before,
            iterations=iterations,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            std_dev=std_dev,
        )

        self.results.append(result)
        print(f"  Average: {avg_time:.2f}ms, Memory: {result.memory_mb:.2f}MB")
        return result

    def print_results(self):
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 120)
        print("BENCHMARK RESULTS")
        print("=" * 120)
        print(
            f"{'Benchmark':<40} {'Avg (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Median (ms)':<12} {'StdDev':<10} {'Memory (MB)':<12} {'CPU %':<8}"
        )
        print("-" * 120)

        for result in self.results:
            print(
                f"{result.name:<40} {result.duration_ms:<10.2f} {result.min_time:<10.2f} {result.max_time:<10.2f} {result.median_time:<12.2f} {result.std_dev:<10.2f} {result.memory_mb:<12.2f} {result.cpu_percent:<8.2f}"
            )

        print("=" * 120)


class AgentBenchmarks:
    """Agent execution benchmarks."""

    def __init__(self):
        self.resource_pool = ResourcePool()
        self.coordinator = AgentCoordinator()
        self.metrics_provider = PrometheusMetricsProvider()

    def create_simple_agent(self) -> Agent:
        """Create a simple agent for benchmarking."""
        agent = Agent(name="benchmark_agent")

        @agent.state()
        def simple_computation(ctx):
            # Simple computation to benchmark
            result = sum(range(1000))
            ctx.set_variable("result", result)
            return result

        return agent

    def create_complex_agent(self) -> Agent:
        """Create a complex agent with dependencies and resource requirements."""
        agent = Agent(name="complex_agent")

        @agent.state()
        def init_state(ctx):
            ctx.set_variable("data", list(range(100)))
            return "initialized"

        @agent.state(depends_on=["init_state"])
        def process_data(ctx):
            data = ctx.get_variable("data")
            processed = [x * 2 for x in data]
            ctx.set_variable("processed", processed)
            return processed

        @agent.state(depends_on=["process_data"])
        def aggregate_data(ctx):
            processed = ctx.get_variable("processed")
            result = sum(processed)
            ctx.set_variable("final_result", result)
            return result

        return agent

    def create_resource_heavy_agent(self) -> Agent:
        """Create an agent with resource requirements."""
        agent = Agent(name="resource_agent")

        @agent.state(
            resource_requirements=ResourceRequirements(
                cpu_cores=1, memory_mb=100, custom_resources={"special_resource": 1}
            )
        )
        def resource_intensive_task(ctx):
            # Simulate resource-intensive computation
            result = sum(x**2 for x in range(10000))
            ctx.set_variable("compute_result", result)
            return result

        return agent

    async def benchmark_simple_agent_execution(self):
        """Benchmark simple agent execution."""
        agent = self.create_simple_agent()
        await agent.run()

    async def benchmark_complex_agent_execution(self):
        """Benchmark complex agent with dependencies."""
        agent = self.create_complex_agent()
        await agent.run()

    async def benchmark_resource_heavy_agent(self):
        """Benchmark agent with resource requirements."""
        agent = self.create_resource_heavy_agent()
        await agent.run()

    async def benchmark_concurrent_agents(self, num_agents: int = 10):
        """Benchmark concurrent agent execution."""
        agents = [self.create_simple_agent() for _ in range(num_agents)]

        # Run all agents concurrently
        await asyncio.gather(*[agent.run() for agent in agents])

    async def benchmark_state_dependency_resolution(self):
        """Benchmark state dependency resolution."""
        agent = self.create_complex_agent()
        # Just test the dependency resolution, not full execution
        ready_states = agent._get_ready_states()
        return len(ready_states)

    def benchmark_resource_acquisition(self):
        """Benchmark resource pool acquisition."""
        requirements = ResourceRequirements(cpu_cores=1, memory_mb=50)

        # Acquire and release resources
        resource_id = self.resource_pool.acquire(requirements, timeout=1.0)
        if resource_id:
            self.resource_pool.release(resource_id)

    def benchmark_coordination_primitive(self):
        """Benchmark coordination primitive operations."""
        primitive = CoordinationPrimitive("test_primitive", "lock")

        # Acquire and release
        primitive.acquire(timeout=1.0)
        primitive.release()

    def benchmark_metrics_recording(self):
        """Benchmark metrics recording."""
        metric = self.metrics_provider.get_counter("test_counter")
        metric.record(1.0)


def main():
    """Main benchmark runner."""
    runner = BenchmarkRunner()
    benchmarks = AgentBenchmarks()

    print("Starting PuffinFlow Core Agent Benchmarks")
    print("=" * 50)

    # Agent execution benchmarks
    runner.run_benchmark(
        "Simple Agent Execution",
        benchmarks.benchmark_simple_agent_execution,
        iterations=50,
    )

    runner.run_benchmark(
        "Complex Agent Execution",
        benchmarks.benchmark_complex_agent_execution,
        iterations=30,
    )

    runner.run_benchmark(
        "Resource Heavy Agent", benchmarks.benchmark_resource_heavy_agent, iterations=20
    )

    runner.run_benchmark(
        "Concurrent Agents (10)",
        benchmarks.benchmark_concurrent_agents,
        iterations=10,
        num_agents=10,
    )

    runner.run_benchmark(
        "State Dependency Resolution",
        benchmarks.benchmark_state_dependency_resolution,
        iterations=1000,
    )

    # Resource management benchmarks
    runner.run_benchmark(
        "Resource Acquisition",
        benchmarks.benchmark_resource_acquisition,
        iterations=1000,
    )

    runner.run_benchmark(
        "Coordination Primitive",
        benchmarks.benchmark_coordination_primitive,
        iterations=1000,
    )

    runner.run_benchmark(
        "Metrics Recording", benchmarks.benchmark_metrics_recording, iterations=10000
    )

    # Print final results
    runner.print_results()

    return runner.results


if __name__ == "__main__":
    results = main()
