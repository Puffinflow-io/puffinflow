#!/usr/bin/env python3
"""
Benchmark suite for observability performance.
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
from puffinflow.core.observability.alerting import AlertManager
from puffinflow.core.observability.core import ObservabilityCore
from puffinflow.core.observability.events import EventManager
from puffinflow.core.observability.metrics import PrometheusMetricsProvider
from puffinflow.core.observability.tracing import OpenTelemetryTracingProvider

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


class ObservabilityBenchmarkRunner:
    """Specialized benchmark runner for observability operations."""

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
        print("OBSERVABILITY BENCHMARK RESULTS")
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


class ObservabilityBenchmarks:
    """Observability benchmarks."""

    def __init__(self):
        self.metrics_provider = PrometheusMetricsProvider()
        self.tracing_provider = OpenTelemetryTracingProvider()
        self.event_manager = EventManager()
        self.alert_manager = AlertManager()
        self.observability_core = ObservabilityCore()
        self.counters = {}
        self.histograms = {}
        self.gauges = {}

    def create_simple_agent(self) -> Agent:
        """Create a simple agent for benchmarking."""
        agent = Agent(name="observability_agent")

        @agent.state()
        def monitored_task(ctx):
            result = sum(range(1000))
            ctx.set_variable("result", result)
            return result

        return agent

    def benchmark_counter_recording(self):
        """Benchmark counter metric recording."""
        if "test_counter" not in self.counters:
            self.counters["test_counter"] = self.metrics_provider.get_counter(
                "test_counter"
            )

        counter = self.counters["test_counter"]
        counter.record(1.0)
        return True

    def benchmark_histogram_recording(self):
        """Benchmark histogram metric recording."""
        if "test_histogram" not in self.histograms:
            self.histograms["test_histogram"] = self.metrics_provider.get_histogram(
                "test_histogram"
            )

        histogram = self.histograms["test_histogram"]
        histogram.record(42.5)
        return True

    def benchmark_gauge_recording(self):
        """Benchmark gauge metric recording."""
        if "test_gauge" not in self.gauges:
            self.gauges["test_gauge"] = self.metrics_provider.get_gauge("test_gauge")

        gauge = self.gauges["test_gauge"]
        gauge.record(78.9)
        return True

    def benchmark_metric_with_labels(self):
        """Benchmark metric recording with labels."""
        counter = self.metrics_provider.get_counter("labeled_counter")
        counter.record(1.0, labels={"service": "test", "method": "benchmark"})
        return True

    def benchmark_concurrent_metric_recording(self, num_threads: int = 10):
        """Benchmark concurrent metric recording."""
        counter = self.metrics_provider.get_counter("concurrent_counter")
        results = []

        def record_metric():
            counter.record(1.0)
            return True

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(record_metric) for _ in range(num_threads)]
            results = [f.result() for f in futures]

        return sum(results)

    def benchmark_metric_cardinality_protection(self):
        """Benchmark metric cardinality protection."""
        counter = self.metrics_provider.get_counter("cardinality_test")

        # Try to create high cardinality
        for i in range(10):
            counter.record(1.0, labels={"dynamic_label": f"value_{i}"})

        return True

    def benchmark_tracing_span_creation(self):
        """Benchmark tracing span creation."""
        span = self.tracing_provider.start_span("test_span")
        if span:
            span.end()
        return span is not None

    def benchmark_tracing_span_with_attributes(self):
        """Benchmark tracing span with attributes."""
        span = self.tracing_provider.start_span(
            "attributed_span",
            attributes={
                "service.name": "benchmark",
                "operation.type": "test",
                "user.id": "123",
            },
        )
        if span:
            span.end()
        return span is not None

    def benchmark_nested_span_creation(self):
        """Benchmark nested span creation."""
        parent_span = self.tracing_provider.start_span("parent_span")

        if parent_span:
            child_span = self.tracing_provider.start_span("child_span")
            if child_span:
                child_span.end()
            parent_span.end()

        return parent_span is not None

    def benchmark_span_event_recording(self):
        """Benchmark span event recording."""
        span = self.tracing_provider.start_span("event_span")

        if span:
            span.add_event("test_event", {"key": "value"})
            span.end()

        return span is not None

    def benchmark_event_manager_emit(self):
        """Benchmark event manager event emission."""
        event_data = {
            "type": "test_event",
            "timestamp": time.time(),
            "data": {"key": "value"},
        }

        self.event_manager.emit("test_event", event_data)
        return True

    def benchmark_event_manager_with_handlers(self):
        """Benchmark event manager with handlers."""

        # Register a handler
        def test_handler(event_data):
            pass

        self.event_manager.register_handler("benchmark_event", test_handler)

        # Emit event
        event_data = {
            "type": "benchmark_event",
            "timestamp": time.time(),
            "data": {"processed": True},
        }

        self.event_manager.emit("benchmark_event", event_data)
        return True

    def benchmark_alert_manager_check(self):
        """Benchmark alert manager condition checking."""

        # Define a simple alert condition
        def cpu_high_condition():
            return psutil.cpu_percent(interval=0.1) > 90

        # Check condition
        self.alert_manager.check_condition("cpu_high", cpu_high_condition)
        return True

    def benchmark_alert_manager_trigger(self):
        """Benchmark alert manager trigger."""

        # Define a simple alert condition that always triggers
        def always_true_condition():
            return True

        # Check condition (should trigger)
        self.alert_manager.check_condition("always_true", always_true_condition)
        return True

    def benchmark_observability_core_integration(self):
        """Benchmark observability core integration."""
        # Use observability core to record metrics and traces
        with self.observability_core.trace_operation("benchmark_operation"):
            self.observability_core.record_metric("operation_count", 1.0)
            time.sleep(0.001)  # Simulate work

        return True

    async def benchmark_agent_observability_integration(self):
        """Benchmark agent with observability integration."""
        agent = self.create_simple_agent()

        # Enable observability for the agent
        agent.enable_observability(
            metrics_provider=self.metrics_provider,
            tracing_provider=self.tracing_provider,
        )

        # Run the agent
        await agent.run()

        return True

    def benchmark_metric_collection(self):
        """Benchmark metric collection/export."""
        # Record some metrics
        counter = self.metrics_provider.get_counter("collection_test")
        counter.record(10.0)

        histogram = self.metrics_provider.get_histogram("collection_histogram")
        histogram.record(25.5)

        gauge = self.metrics_provider.get_gauge("collection_gauge")
        gauge.record(50.0)

        # Simulate collection
        metrics = self.metrics_provider.collect_metrics()
        return len(metrics) > 0

    def benchmark_high_frequency_metrics(self):
        """Benchmark high-frequency metric recording."""
        counter = self.metrics_provider.get_counter("high_frequency_counter")

        # Record multiple times quickly
        for _ in range(100):
            counter.record(1.0)

        return True

    def benchmark_metric_memory_usage(self):
        """Benchmark metric memory usage."""
        # Create many metrics to test memory usage
        counters = []
        for i in range(50):
            counter = self.metrics_provider.get_counter(f"memory_test_counter_{i}")
            counters.append(counter)

        # Record on all counters
        for counter in counters:
            counter.record(1.0)

        return len(counters) == 50

    def benchmark_trace_context_propagation(self):
        """Benchmark trace context propagation."""
        # Start a parent span
        parent_span = self.tracing_provider.start_span("parent_context")

        if parent_span:
            # Create child operations
            for i in range(5):
                child_span = self.tracing_provider.start_span(f"child_operation_{i}")
                if child_span:
                    child_span.end()

            parent_span.end()

        return parent_span is not None


def main():
    """Main benchmark runner."""
    runner = ObservabilityBenchmarkRunner()
    benchmarks = ObservabilityBenchmarks()

    print("Starting PuffinFlow Observability Benchmarks")
    print("=" * 50)

    # Metrics benchmarks
    runner.run_benchmark(
        "Counter Recording", benchmarks.benchmark_counter_recording, iterations=50000
    )

    runner.run_benchmark(
        "Histogram Recording",
        benchmarks.benchmark_histogram_recording,
        iterations=50000,
    )

    runner.run_benchmark(
        "Gauge Recording", benchmarks.benchmark_gauge_recording, iterations=50000
    )

    runner.run_benchmark(
        "Metric with Labels", benchmarks.benchmark_metric_with_labels, iterations=10000
    )

    runner.run_benchmark(
        "Concurrent Metric Recording (10 threads)",
        benchmarks.benchmark_concurrent_metric_recording,
        iterations=1000,
        num_threads=10,
    )

    runner.run_benchmark(
        "Metric Cardinality Protection",
        benchmarks.benchmark_metric_cardinality_protection,
        iterations=1000,
    )

    # Tracing benchmarks
    runner.run_benchmark(
        "Tracing Span Creation",
        benchmarks.benchmark_tracing_span_creation,
        iterations=10000,
    )

    runner.run_benchmark(
        "Tracing Span with Attributes",
        benchmarks.benchmark_tracing_span_with_attributes,
        iterations=10000,
    )

    runner.run_benchmark(
        "Nested Span Creation",
        benchmarks.benchmark_nested_span_creation,
        iterations=5000,
    )

    runner.run_benchmark(
        "Span Event Recording",
        benchmarks.benchmark_span_event_recording,
        iterations=10000,
    )

    # Event management benchmarks
    runner.run_benchmark(
        "Event Manager Emit", benchmarks.benchmark_event_manager_emit, iterations=10000
    )

    runner.run_benchmark(
        "Event Manager with Handlers",
        benchmarks.benchmark_event_manager_with_handlers,
        iterations=5000,
    )

    # Alert management benchmarks
    runner.run_benchmark(
        "Alert Manager Check", benchmarks.benchmark_alert_manager_check, iterations=1000
    )

    runner.run_benchmark(
        "Alert Manager Trigger",
        benchmarks.benchmark_alert_manager_trigger,
        iterations=1000,
    )

    # Integration benchmarks
    runner.run_benchmark(
        "Observability Core Integration",
        benchmarks.benchmark_observability_core_integration,
        iterations=1000,
    )

    runner.run_benchmark(
        "Agent Observability Integration",
        benchmarks.benchmark_agent_observability_integration,
        iterations=100,
    )

    runner.run_benchmark(
        "Metric Collection", benchmarks.benchmark_metric_collection, iterations=1000
    )

    runner.run_benchmark(
        "High Frequency Metrics",
        benchmarks.benchmark_high_frequency_metrics,
        iterations=1000,
    )

    runner.run_benchmark(
        "Metric Memory Usage", benchmarks.benchmark_metric_memory_usage, iterations=100
    )

    runner.run_benchmark(
        "Trace Context Propagation",
        benchmarks.benchmark_trace_context_propagation,
        iterations=1000,
    )

    # Print final results
    runner.print_results()

    return runner.results


if __name__ == "__main__":
    results = main()
