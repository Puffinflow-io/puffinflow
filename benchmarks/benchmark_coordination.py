#!/usr/bin/env python3
"""
Benchmark suite for coordination and synchronization performance.
"""

import asyncio
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import psutil

from puffinflow.core.agent.base import Agent
from puffinflow.core.coordination.agent_pool import AgentPool, WorkProcessor
from puffinflow.core.coordination.coordinator import AgentCoordinator
from puffinflow.core.coordination.primitives import Barrier, CoordinationPrimitive
from puffinflow.core.coordination.rate_limiter import RateLimiter

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
    throughput_ops_per_sec: float


class CoordinationBenchmarkRunner:
    """Specialized benchmark runner for coordination operations."""

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

        start_benchmark = time.perf_counter()

        for _ in range(iterations):
            start_time = time.perf_counter()

            if asyncio.iscoroutinefunction(func):
                asyncio.run(func(*args, **kwargs))
            else:
                func(*args, **kwargs)

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        end_benchmark = time.perf_counter()
        total_time = end_benchmark - start_benchmark

        memory_after = self.process.memory_info().rss / 1024 / 1024
        cpu_after = self.process.cpu_percent()

        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        throughput = iterations / total_time

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
            throughput_ops_per_sec=throughput,
        )

        self.results.append(result)
        print(f"  Average: {avg_time:.2f}ms, Throughput: {throughput:.2f} ops/sec")
        return result

    def print_results(self):
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 140)
        print("COORDINATION AND SYNCHRONIZATION BENCHMARK RESULTS")
        print("=" * 140)
        print(
            f"{'Benchmark':<40} {'Avg (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Median (ms)':<12} {'StdDev':<10} {'Throughput (ops/s)':<18} {'Memory (MB)':<12} {'CPU %':<8}"
        )
        print("-" * 140)

        for result in self.results:
            print(
                f"{result.name:<40} {result.duration_ms:<10.2f} {result.min_time:<10.2f} {result.max_time:<10.2f} {result.median_time:<12.2f} {result.std_dev:<10.2f} {result.throughput_ops_per_sec:<18.2f} {result.memory_mb:<12.2f} {result.cpu_percent:<8.2f}"
            )

        print("=" * 140)


class CoordinationBenchmarks:
    """Coordination and synchronization benchmarks."""

    def __init__(self):
        self.coordinator = AgentCoordinator()
        self.primitives: dict[str, CoordinationPrimitive] = {}
        self.barriers: dict[str, Barrier] = {}
        self.agent_pools: dict[str, AgentPool] = {}
        self.rate_limiters: dict[str, RateLimiter] = {}

    def create_simple_agent(self, name: str = "benchmark_agent") -> Agent:
        """Create a simple agent for benchmarking."""
        agent = Agent(name=name)

        @agent.state()
        def simple_task(ctx):
            result = sum(range(100))
            ctx.set_variable("result", result)
            return result

        return agent

    def benchmark_coordination_primitive_lock(self):
        """Benchmark coordination primitive lock operations."""
        primitive = CoordinationPrimitive("test_lock", "lock")

        # Acquire and release
        success = primitive.acquire(timeout=1.0)
        if success:
            primitive.release()

        return success

    def benchmark_coordination_primitive_semaphore(self):
        """Benchmark coordination primitive semaphore operations."""
        primitive = CoordinationPrimitive("test_semaphore", "semaphore", max_count=5)

        # Acquire and release
        success = primitive.acquire(timeout=1.0)
        if success:
            primitive.release()

        return success

    def benchmark_coordination_primitive_barrier(self):
        """Benchmark coordination primitive barrier operations."""
        barrier = Barrier("test_barrier", party_count=1)

        # Wait on barrier (should complete immediately with party_count=1)
        result = barrier.wait(timeout=1.0)
        return result

    def benchmark_concurrent_lock_contention(self, num_threads: int = 10):
        """Benchmark lock contention with multiple threads."""
        primitive = CoordinationPrimitive("contention_lock", "lock")
        results = []

        def acquire_release():
            success = primitive.acquire(timeout=1.0)
            if success:
                time.sleep(0.01)  # Simulate some work
                primitive.release()
            return success

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(acquire_release) for _ in range(num_threads)]
            results = [f.result() for f in futures]

        return sum(results)

    def benchmark_semaphore_contention(self, num_threads: int = 10):
        """Benchmark semaphore contention with multiple threads."""
        primitive = CoordinationPrimitive(
            "contention_semaphore", "semaphore", max_count=3
        )
        results = []

        def acquire_release():
            success = primitive.acquire(timeout=1.0)
            if success:
                time.sleep(0.01)  # Simulate some work
                primitive.release()
            return success

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(acquire_release) for _ in range(num_threads)]
            results = [f.result() for f in futures]

        return sum(results)

    def benchmark_barrier_synchronization(self, num_threads: int = 5):
        """Benchmark barrier synchronization with multiple threads."""
        barrier = Barrier("sync_barrier", party_count=num_threads)
        results = []

        def wait_on_barrier():
            return barrier.wait(timeout=5.0)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(wait_on_barrier) for _ in range(num_threads)]
            results = [f.result() for f in futures]

        return sum(results)

    def benchmark_rate_limiter(self):
        """Benchmark rate limiter operations."""
        rate_limiter = RateLimiter(max_calls=100, time_window=1.0)

        # Test rate limiting
        return rate_limiter.can_proceed()

    def benchmark_rate_limiter_contention(self, num_threads: int = 20):
        """Benchmark rate limiter under contention."""
        rate_limiter = RateLimiter(max_calls=10, time_window=1.0)
        results = []

        def check_rate_limit():
            return rate_limiter.can_proceed()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(check_rate_limit) for _ in range(num_threads)]
            results = [f.result() for f in futures]

        return sum(results)

    async def benchmark_agent_coordinator_state_execution(self):
        """Benchmark agent coordinator state execution."""
        agent = self.create_simple_agent()

        # Coordinate state execution
        await self.coordinator.coordinate_state_execution(agent, "simple_task", {})

        return True

    def benchmark_agent_pool_creation(self):
        """Benchmark agent pool creation."""
        agent_pool = AgentPool(
            name="benchmark_pool",
            agent_factory=lambda: self.create_simple_agent(),
            min_size=1,
            max_size=10,
        )

        return agent_pool.name == "benchmark_pool"

    async def benchmark_agent_pool_work_processing(self):
        """Benchmark agent pool work processing."""
        agent_pool = AgentPool(
            name="work_pool",
            agent_factory=lambda: self.create_simple_agent(),
            min_size=1,
            max_size=5,
        )

        # Start the pool
        await agent_pool.start()

        # Submit work
        work_item = {
            "type": "agent_execution",
            "agent_name": "benchmark_agent",
            "inputs": {},
        }

        try:
            await agent_pool.submit_work(work_item)
            return True
        finally:
            await agent_pool.stop()

    def benchmark_work_processor_creation(self):
        """Benchmark work processor creation."""
        processor = WorkProcessor(
            name="benchmark_processor",
            agent_factory=lambda: self.create_simple_agent(),
            max_workers=5,
        )

        return processor.name == "benchmark_processor"

    async def benchmark_coordinator_enhance_agent(self):
        """Benchmark coordinator agent enhancement."""
        agent = self.create_simple_agent()

        # Enhance agent with coordination
        enhanced_agent = self.coordinator.enhance_agent(agent)

        # Run the enhanced agent
        await enhanced_agent.run()

        return True

    def benchmark_primitive_state_management(self):
        """Benchmark primitive state management operations."""
        primitive = CoordinationPrimitive("state_test", "lock")

        # Check initial state
        is_available = primitive.is_available()

        # Acquire
        success = primitive.acquire(timeout=1.0)

        # Check state after acquire
        is_available_after = primitive.is_available()

        # Release
        if success:
            primitive.release()

        # Check state after release
        is_available_final = primitive.is_available()

        return (
            is_available and success and not is_available_after and is_available_final
        )

    def benchmark_quota_management(self):
        """Benchmark quota management in primitives."""
        primitive = CoordinationPrimitive("quota_test", "semaphore", max_count=5)

        # Acquire multiple times
        acquisitions = []
        for _ in range(5):
            success = primitive.acquire(timeout=1.0)
            acquisitions.append(success)

        # Try to acquire when quota is full
        over_quota = primitive.acquire(timeout=0.1)

        # Release all
        for success in acquisitions:
            if success:
                primitive.release()

        return sum(acquisitions) == 5 and not over_quota


def main():
    """Main benchmark runner."""
    runner = CoordinationBenchmarkRunner()
    benchmarks = CoordinationBenchmarks()

    print("Starting PuffinFlow Coordination and Synchronization Benchmarks")
    print("=" * 70)

    # Basic coordination primitive benchmarks
    runner.run_benchmark(
        "Coordination Primitive Lock",
        benchmarks.benchmark_coordination_primitive_lock,
        iterations=10000,
    )

    runner.run_benchmark(
        "Coordination Primitive Semaphore",
        benchmarks.benchmark_coordination_primitive_semaphore,
        iterations=10000,
    )

    runner.run_benchmark(
        "Coordination Primitive Barrier",
        benchmarks.benchmark_coordination_primitive_barrier,
        iterations=1000,
    )

    # Contention benchmarks
    runner.run_benchmark(
        "Concurrent Lock Contention (10 threads)",
        benchmarks.benchmark_concurrent_lock_contention,
        iterations=100,
        num_threads=10,
    )

    runner.run_benchmark(
        "Semaphore Contention (10 threads)",
        benchmarks.benchmark_semaphore_contention,
        iterations=100,
        num_threads=10,
    )

    runner.run_benchmark(
        "Barrier Synchronization (5 threads)",
        benchmarks.benchmark_barrier_synchronization,
        iterations=50,
        num_threads=5,
    )

    # Rate limiting benchmarks
    runner.run_benchmark(
        "Rate Limiter", benchmarks.benchmark_rate_limiter, iterations=10000
    )

    runner.run_benchmark(
        "Rate Limiter Contention (20 threads)",
        benchmarks.benchmark_rate_limiter_contention,
        iterations=100,
        num_threads=20,
    )

    # Agent coordination benchmarks
    runner.run_benchmark(
        "Agent Coordinator State Execution",
        benchmarks.benchmark_agent_coordinator_state_execution,
        iterations=100,
    )

    runner.run_benchmark(
        "Agent Pool Creation", benchmarks.benchmark_agent_pool_creation, iterations=1000
    )

    runner.run_benchmark(
        "Agent Pool Work Processing",
        benchmarks.benchmark_agent_pool_work_processing,
        iterations=20,
    )

    runner.run_benchmark(
        "Work Processor Creation",
        benchmarks.benchmark_work_processor_creation,
        iterations=1000,
    )

    runner.run_benchmark(
        "Coordinator Agent Enhancement",
        benchmarks.benchmark_coordinator_enhance_agent,
        iterations=50,
    )

    # State management benchmarks
    runner.run_benchmark(
        "Primitive State Management",
        benchmarks.benchmark_primitive_state_management,
        iterations=5000,
    )

    runner.run_benchmark(
        "Quota Management", benchmarks.benchmark_quota_management, iterations=1000
    )

    # Print final results
    runner.print_results()

    return runner.results


if __name__ == "__main__":
    results = main()
