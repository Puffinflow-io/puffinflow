#!/usr/bin/env python3
"""
Simple benchmark for PuffinFlow performance metrics.
"""

import asyncio
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    HAS_PUFFINFLOW = True
except ImportError:
    HAS_PUFFINFLOW = False


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
    throughput_ops_per_sec: float = 0


class SimpleBenchmarkRunner:
    """Simple benchmark runner."""

    def __init__(self):
        self.process = psutil.Process()
        self.results: list[BenchmarkResult] = []

    def run_benchmark(self, name: str, func, iterations: int = 1000, warmup: int = 100):
        """Run a benchmark function and collect metrics."""
        print(f"Running benchmark: {name}")

        # Warmup
        for _ in range(warmup):
            func()

        # Monitor initial state
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Run benchmark
        times = []
        cpu_times = []

        for _i in range(iterations):
            cpu_before = self.process.cpu_percent()
            start_time = time.time()

            func()

            end_time = time.time()
            cpu_after = self.process.cpu_percent()

            duration = (end_time - start_time) * 1000  # Convert to ms
            times.append(duration)
            cpu_times.append(cpu_after - cpu_before)

        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0

        # Monitor final state
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory

        avg_cpu = statistics.mean(cpu_times) if cpu_times else 0
        throughput = 1000 / avg_time if avg_time > 0 else 0  # ops/second

        result = BenchmarkResult(
            name=name,
            duration_ms=avg_time,
            memory_mb=memory_usage,
            cpu_percent=avg_cpu,
            iterations=iterations,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            std_dev=std_dev,
            throughput_ops_per_sec=throughput,
        )

        self.results.append(result)
        print(f"  Completed: {avg_time:.2f}ms avg, {throughput:.2f} ops/s")

        return result


def benchmark_basic_operations():
    """Benchmark basic Python operations."""
    runner = SimpleBenchmarkRunner()

    # Basic list operations
    def list_creation():
        return list(range(100))

    def list_processing():
        data = list(range(1000))
        return [x * 2 for x in data if x % 2 == 0]

    def dict_operations():
        d = {}
        for i in range(100):
            d[str(i)] = i * 2
        return d

    # Run benchmarks
    runner.run_benchmark("List Creation (100 items)", list_creation, iterations=10000)
    runner.run_benchmark(
        "List Processing (1000 items)", list_processing, iterations=5000
    )
    runner.run_benchmark(
        "Dict Operations (100 items)", dict_operations, iterations=5000
    )

    return runner.results


def benchmark_async_operations():
    """Benchmark async operations."""
    runner = SimpleBenchmarkRunner()

    async def simple_async_task():
        await asyncio.sleep(0.001)  # 1ms sleep
        return "completed"

    async def concurrent_async_tasks():
        tasks = [simple_async_task() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        return results

    def run_async_benchmark():
        return asyncio.run(simple_async_task())

    def run_concurrent_benchmark():
        return asyncio.run(concurrent_async_tasks())

    runner.run_benchmark("Simple Async Task", run_async_benchmark, iterations=1000)
    runner.run_benchmark(
        "Concurrent Async Tasks (10)", run_concurrent_benchmark, iterations=500
    )

    return runner.results


def benchmark_resource_intensive():
    """Benchmark resource-intensive operations."""
    runner = SimpleBenchmarkRunner()

    def cpu_intensive():
        # CPU-bound calculation
        total = 0
        for i in range(10000):
            total += i * i
        return total

    def memory_intensive():
        # Memory allocation
        data = []
        for _i in range(1000):
            data.append(list(range(100)))
        return len(data)

    runner.run_benchmark(
        "CPU Intensive (10k operations)", cpu_intensive, iterations=1000
    )
    runner.run_benchmark(
        "Memory Intensive (100k items)", memory_intensive, iterations=500
    )

    return runner.results


def main():
    """Main benchmark function."""
    print("🚀 Starting Simple PuffinFlow Benchmarks")
    print("=" * 50)

    all_results = []

    # Run different benchmark categories
    print("\n📊 Basic Operations Benchmarks")
    basic_results = benchmark_basic_operations()
    all_results.extend(basic_results)

    print("\n⚡ Async Operations Benchmarks")
    async_results = benchmark_async_operations()
    all_results.extend(async_results)

    print("\n🔥 Resource Intensive Benchmarks")
    resource_results = benchmark_resource_intensive()
    all_results.extend(resource_results)

    # Print summary
    print("\n" + "=" * 80)
    print("🎯 BENCHMARK SUMMARY")
    print("=" * 80)

    for result in all_results:
        print(
            f"{result.name:<40} | {result.duration_ms:>8.2f}ms | {result.throughput_ops_per_sec:>10.2f} ops/s"
        )

    print("=" * 80)

    # Find best performers
    fastest = min(all_results, key=lambda x: x.duration_ms)
    highest_throughput = max(all_results, key=lambda x: x.throughput_ops_per_sec)

    print(f"🏆 Fastest Operation: {fastest.name} ({fastest.duration_ms:.2f}ms)")
    print(
        f"⚡ Highest Throughput: {highest_throughput.name} ({highest_throughput.throughput_ops_per_sec:.2f} ops/s)"
    )

    return all_results


if __name__ == "__main__":
    main()
