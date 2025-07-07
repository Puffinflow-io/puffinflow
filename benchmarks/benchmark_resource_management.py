#!/usr/bin/env python3
"""
Benchmark suite for resource management performance.
"""

import asyncio
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import psutil

from puffinflow.core.resources.allocation import (
    BestFitAllocator,
    FirstFitAllocator,
    PriorityAllocator,
)
from puffinflow.core.resources.pool import ResourcePool
from puffinflow.core.resources.quotas import ResourceQuotas
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
    throughput_ops_per_sec: float


class ResourceBenchmarkRunner:
    """Specialized benchmark runner for resource management."""

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
        print("RESOURCE MANAGEMENT BENCHMARK RESULTS")
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


class ResourceManagementBenchmarks:
    """Resource management benchmarks."""

    def __init__(self):
        self.pool = ResourcePool()
        self.active_allocations: list[str] = []
        self.lock = threading.Lock()

    def setup_pool_with_resources(self, cpu_cores: int = 16, memory_mb: int = 8192):
        """Setup resource pool with specific resources."""
        self.pool = ResourcePool()
        # Add some custom resources for testing
        self.pool._available_resources = {
            "cpu_cores": cpu_cores,
            "memory_mb": memory_mb,
            "custom_resource_1": 10,
            "custom_resource_2": 5,
            "gpu_memory": 2048,
        }

    def benchmark_single_resource_acquisition(self):
        """Benchmark single resource acquisition and release."""
        requirements = ResourceRequirements(cpu_cores=1, memory_mb=512)

        resource_id = self.pool.acquire(requirements, timeout=1.0)
        if resource_id:
            self.pool.release(resource_id)

        return resource_id is not None

    def benchmark_complex_resource_acquisition(self):
        """Benchmark complex resource acquisition with multiple resource types."""
        requirements = ResourceRequirements(
            cpu_cores=2,
            memory_mb=1024,
            custom_resources={"custom_resource_1": 2, "gpu_memory": 256},
        )

        resource_id = self.pool.acquire(requirements, timeout=1.0)
        if resource_id:
            self.pool.release(resource_id)

        return resource_id is not None

    def benchmark_resource_contention(self):
        """Benchmark resource acquisition under contention."""
        requirements = ResourceRequirements(cpu_cores=4, memory_mb=2048)

        # Try to acquire large resources that may cause contention
        resource_id = self.pool.acquire(requirements, timeout=0.1)
        if resource_id:
            # Hold for a very short time to simulate contention
            time.sleep(0.001)
            self.pool.release(resource_id)

        return resource_id is not None

    def benchmark_concurrent_acquisitions(self, num_threads: int = 10):
        """Benchmark concurrent resource acquisitions."""

        def acquire_and_release():
            requirements = ResourceRequirements(cpu_cores=1, memory_mb=256)
            resource_id = self.pool.acquire(requirements, timeout=1.0)
            if resource_id:
                time.sleep(0.01)  # Simulate some work
                self.pool.release(resource_id)
            return resource_id is not None

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(acquire_and_release) for _ in range(num_threads)]
            results = [f.result() for f in futures]

        return sum(results)

    def benchmark_quota_checking(self):
        """Benchmark quota checking performance."""
        quotas = ResourceQuotas()
        quotas.set_quota("cpu_cores", 8)
        quotas.set_quota("memory_mb", 4096)

        # Check if allocation is within quota
        allocation = {"cpu_cores": 2, "memory_mb": 1024}
        return quotas.check_quota(allocation)

    def benchmark_allocator_first_fit(self):
        """Benchmark FirstFit allocation strategy."""
        allocator = FirstFitAllocator()

        # Setup available resources
        available = {"cpu_cores": 16, "memory_mb": 8192, "custom_resource_1": 10}

        requirements = ResourceRequirements(
            cpu_cores=2, memory_mb=1024, custom_resources={"custom_resource_1": 2}
        )

        result = allocator.allocate(available, requirements)
        return result is not None

    def benchmark_allocator_best_fit(self):
        """Benchmark BestFit allocation strategy."""
        allocator = BestFitAllocator()

        # Setup available resources
        available = {"cpu_cores": 16, "memory_mb": 8192, "custom_resource_1": 10}

        requirements = ResourceRequirements(
            cpu_cores=2, memory_mb=1024, custom_resources={"custom_resource_1": 2}
        )

        result = allocator.allocate(available, requirements)
        return result is not None

    def benchmark_allocator_priority(self):
        """Benchmark Priority allocation strategy."""
        allocator = PriorityAllocator()

        # Setup available resources
        available = {"cpu_cores": 16, "memory_mb": 8192, "custom_resource_1": 10}

        requirements = ResourceRequirements(
            cpu_cores=2, memory_mb=1024, custom_resources={"custom_resource_1": 2}
        )

        result = allocator.allocate(available, requirements, priority=5)
        return result is not None

    def benchmark_resource_pool_can_allocate(self):
        """Benchmark resource pool allocation checking."""
        requirements = ResourceRequirements(cpu_cores=1, memory_mb=512)
        return self.pool._can_allocate(requirements)

    def benchmark_resource_pool_preemption(self):
        """Benchmark resource pool preemption logic."""
        # First, fill up the pool
        requirements = ResourceRequirements(cpu_cores=8, memory_mb=4096)
        resource_id = self.pool.acquire(requirements, timeout=1.0)

        if resource_id:
            # Now try to trigger preemption
            high_priority_requirements = ResourceRequirements(
                cpu_cores=2, memory_mb=1024
            )

            # This should trigger preemption logic
            preempted = self.pool._try_preemption(
                high_priority_requirements, priority=10
            )

            # Clean up
            self.pool.release(resource_id)

            return preempted

        return False

    def benchmark_resource_leak_detection(self):
        """Benchmark resource leak detection."""
        # Simulate a resource leak scenario
        requirements = ResourceRequirements(cpu_cores=1, memory_mb=256)

        # Acquire but don't release immediately
        resource_id = self.pool.acquire(requirements, timeout=1.0)

        if resource_id:
            # Check if leak detection works
            leak_detected = len(self.pool._allocated_resources) > 0

            # Clean up
            self.pool.release(resource_id)

            return leak_detected

        return False


def main():
    """Main benchmark runner."""
    runner = ResourceBenchmarkRunner()
    benchmarks = ResourceManagementBenchmarks()

    print("Starting PuffinFlow Resource Management Benchmarks")
    print("=" * 60)

    # Setup resource pool
    benchmarks.setup_pool_with_resources()

    # Basic resource acquisition benchmarks
    runner.run_benchmark(
        "Single Resource Acquisition",
        benchmarks.benchmark_single_resource_acquisition,
        iterations=1000,
    )

    runner.run_benchmark(
        "Complex Resource Acquisition",
        benchmarks.benchmark_complex_resource_acquisition,
        iterations=500,
    )

    runner.run_benchmark(
        "Resource Contention", benchmarks.benchmark_resource_contention, iterations=200
    )

    runner.run_benchmark(
        "Concurrent Acquisitions (10 threads)",
        benchmarks.benchmark_concurrent_acquisitions,
        iterations=50,
        num_threads=10,
    )

    # Quota and allocation strategy benchmarks
    runner.run_benchmark(
        "Quota Checking", benchmarks.benchmark_quota_checking, iterations=10000
    )

    runner.run_benchmark(
        "FirstFit Allocator", benchmarks.benchmark_allocator_first_fit, iterations=5000
    )

    runner.run_benchmark(
        "BestFit Allocator", benchmarks.benchmark_allocator_best_fit, iterations=5000
    )

    runner.run_benchmark(
        "Priority Allocator", benchmarks.benchmark_allocator_priority, iterations=5000
    )

    # Advanced resource pool operations
    runner.run_benchmark(
        "Resource Pool Can Allocate",
        benchmarks.benchmark_resource_pool_can_allocate,
        iterations=10000,
    )

    runner.run_benchmark(
        "Resource Pool Preemption",
        benchmarks.benchmark_resource_pool_preemption,
        iterations=100,
    )

    runner.run_benchmark(
        "Resource Leak Detection",
        benchmarks.benchmark_resource_leak_detection,
        iterations=1000,
    )

    # Print final results
    runner.print_results()

    return runner.results


if __name__ == "__main__":
    results = main()
