#!/usr/bin/env python3
"""
Orchestration-focused benchmarks that measure metrics that actually matter
for workflow orchestration frameworks, not just basic operations.
"""

import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import psutil

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from puffinflow.core.agent.base import Agent  # noqa: E402


@dataclass
class OrchestrationMetric:
    """Container for orchestration-specific metrics."""

    name: str
    value: float
    unit: str
    description: str
    is_better_lower: bool = True  # True if lower values are better


@dataclass
class OrchestrationBenchmarkResult:
    """Results for orchestration-focused benchmarks."""

    name: str
    framework: str
    metrics: list[OrchestrationMetric]
    iterations: int
    success_rate: float
    timestamp: str


class OrchestrationBenchmarkRunner:
    """Runner for orchestration-specific benchmarks."""

    def __init__(self):
        self.process = psutil.Process()
        self.results: list[OrchestrationBenchmarkResult] = []

    def run_orchestration_benchmark(
        self,
        name: str,
        framework: str,
        benchmark_func: Callable,
        iterations: int = 50,
        warmup_iterations: int = 5,
    ) -> OrchestrationBenchmarkResult:
        """Run an orchestration-focused benchmark."""

        print(f"Running {name} for {framework}...")

        # Warmup
        import contextlib

        for _ in range(warmup_iterations):
            with contextlib.suppress(Exception):
                benchmark_func()

        # Collect metrics over multiple iterations
        all_metrics = []
        successful_runs = 0

        for _ in range(iterations):
            try:
                metrics = benchmark_func()
                if metrics:
                    all_metrics.append(metrics)
                    successful_runs += 1
            except Exception as e:
                print(f"  Benchmark iteration failed: {e}")

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        success_rate = successful_runs / iterations if iterations > 0 else 0.0

        result = OrchestrationBenchmarkResult(
            name=name,
            framework=framework,
            metrics=aggregated_metrics,
            iterations=successful_runs,
            success_rate=success_rate,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        self.results.append(result)
        return result

    def _aggregate_metrics(
        self, all_metrics: list[list[OrchestrationMetric]]
    ) -> list[OrchestrationMetric]:
        """Aggregate metrics across iterations."""
        if not all_metrics:
            return []

        metric_groups = {}
        for metrics in all_metrics:
            for metric in metrics:
                if metric.name not in metric_groups:
                    metric_groups[metric.name] = []
                metric_groups[metric.name].append(metric.value)

        aggregated = []
        for name, values in metric_groups.items():
            # Find the metric template
            template_metric = None
            for metrics in all_metrics:
                for metric in metrics:
                    if metric.name == name:
                        template_metric = metric
                        break
                if template_metric:
                    break

            if template_metric and values:
                avg_value = statistics.mean(values)
                aggregated.append(
                    OrchestrationMetric(
                        name=name,
                        value=avg_value,
                        unit=template_metric.unit,
                        description=template_metric.description,
                        is_better_lower=template_metric.is_better_lower,
                    )
                )

        return aggregated


class WorkflowComplexityBenchmarks:
    """Benchmarks for workflow complexity handling."""

    def deep_dependency_chain(self) -> list[OrchestrationMetric]:
        """Test deep dependency chains (common in data pipelines)."""
        chain_depth = 20

        class ChainAgent(Agent):
            def __init__(
                self, agent_id: int, depends_on: Optional["ChainAgent"] = None
            ):
                super().__init__(name=f"chain_agent_{agent_id}")
                self.agent_id = agent_id
                self.depends_on = depends_on
                self.execution_time = 0

            async def run(self):
                start_time = time.perf_counter()

                # Wait for dependency if exists
                if self.depends_on and hasattr(self.depends_on, "result"):
                    dependency_start = time.perf_counter()
                    while not hasattr(self.depends_on, "result"):
                        await asyncio.sleep(0.001)
                    dependency_wait = time.perf_counter() - dependency_start
                else:
                    dependency_wait = 0

                # Simulate work
                await asyncio.sleep(0.01)

                self.execution_time = time.perf_counter() - start_time
                self.result = {
                    "agent_id": self.agent_id,
                    "dependency_wait": dependency_wait,
                }
                return self.result

        # Create chain
        agents = []
        for i in range(chain_depth):
            depends_on = agents[-1] if agents else None
            agent = ChainAgent(i, depends_on)
            agents.append(agent)

        # Measure execution
        start_time = time.perf_counter()

        async def run_chain():
            tasks = [agent.run() for agent in agents]
            await asyncio.gather(*tasks)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_chain())
        finally:
            loop.close()

        total_time = time.perf_counter() - start_time

        # Calculate metrics
        total_work_time = sum(agent.execution_time for agent in agents)
        parallelization_efficiency = (
            (total_work_time / total_time) if total_time > 0 else 0
        )
        avg_dependency_wait = statistics.mean(
            [
                agent.result.get("dependency_wait", 0)
                for agent in agents
                if hasattr(agent, "result") and agent.result
            ]
        )

        return [
            OrchestrationMetric(
                "dependency_chain_latency",
                total_time * 1000,
                "ms",
                "Total time to execute deep dependency chain",
                True,
            ),
            OrchestrationMetric(
                "parallelization_efficiency",
                parallelization_efficiency,
                "ratio",
                "How well the framework parallelizes independent work",
                False,
            ),
            OrchestrationMetric(
                "dependency_wait_time",
                avg_dependency_wait * 1000,
                "ms",
                "Average time agents wait for dependencies",
                True,
            ),
        ]

    def wide_fanout_pattern(self) -> list[OrchestrationMetric]:
        """Test wide fanout patterns (one producer, many consumers)."""
        fanout_width = 50

        class ProducerAgent(Agent):
            def __init__(self):
                super().__init__(name="producer")
                self.start_time = None

            async def run(self):
                self.start_time = time.perf_counter()
                await asyncio.sleep(0.05)  # Simulate work
                self.result = {"data": list(range(1000)), "timestamp": time.time()}
                return self.result

        class ConsumerAgent(Agent):
            def __init__(self, consumer_id: int, producer: ProducerAgent):
                super().__init__(name=f"consumer_{consumer_id}")
                self.consumer_id = consumer_id
                self.producer = producer
                self.wait_time = 0

            async def run(self):
                wait_start = time.perf_counter()

                # Wait for producer
                while not hasattr(self.producer, "result"):
                    await asyncio.sleep(0.001)

                self.wait_time = time.perf_counter() - wait_start

                # Process data
                data = self.producer.result["data"]
                processed = [x * self.consumer_id for x in data[:10]]  # Process subset

                return {"consumer_id": self.consumer_id, "processed": processed}

        # Create producer and consumers
        producer = ProducerAgent()
        consumers = [ConsumerAgent(i, producer) for i in range(fanout_width)]

        # Measure execution
        start_time = time.perf_counter()

        async def run_fanout():
            # Start producer first
            producer_task = asyncio.create_task(producer.run())

            # Start consumers
            consumer_tasks = [
                asyncio.create_task(consumer.run()) for consumer in consumers
            ]

            # Wait for all
            await producer_task
            await asyncio.gather(*consumer_tasks)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_fanout())
        finally:
            loop.close()

        total_time = time.perf_counter() - start_time

        # Calculate metrics
        avg_consumer_wait = statistics.mean([c.wait_time for c in consumers])
        max_consumer_wait = max([c.wait_time for c in consumers])
        consumer_start_spread = max_consumer_wait - min(
            [c.wait_time for c in consumers]
        )

        return [
            OrchestrationMetric(
                "fanout_total_time",
                total_time * 1000,
                "ms",
                "Total time for fanout pattern execution",
                True,
            ),
            OrchestrationMetric(
                "consumer_avg_wait",
                avg_consumer_wait * 1000,
                "ms",
                "Average wait time for consumers",
                True,
            ),
            OrchestrationMetric(
                "consumer_start_spread",
                consumer_start_spread * 1000,
                "ms",
                "Spread in consumer start times (lower = better coordination)",
                True,
            ),
        ]

    def diamond_dependency_pattern(self) -> list[OrchestrationMetric]:
        """Test diamond dependency patterns (complex DAG resolution)."""

        class DiamondAgent(Agent):
            def __init__(
                self, name: str, dependencies: Optional[list["DiamondAgent"]] = None
            ):
                super().__init__(name=name)
                self.dependencies = dependencies or []
                self.execution_start = None
                self.dependency_wait_time = 0

            async def run(self):
                self.execution_start = time.perf_counter()

                # Wait for all dependencies
                wait_start = time.perf_counter()
                for dep in self.dependencies:
                    while not hasattr(dep, "result"):
                        await asyncio.sleep(0.001)
                self.dependency_wait_time = time.perf_counter() - wait_start

                # Simulate work
                await asyncio.sleep(0.02)

                # Combine dependency results
                dep_results = [dep.result for dep in self.dependencies]
                self.result = {
                    "name": self.name,
                    "dependencies": [dep.name for dep in self.dependencies],
                    "dep_results": dep_results,
                }
                return self.result

        # Create diamond pattern: A -> [B, C] -> D
        agent_a = DiamondAgent("A")
        agent_b = DiamondAgent("B", [agent_a])
        agent_c = DiamondAgent("C", [agent_a])
        agent_d = DiamondAgent("D", [agent_b, agent_c])

        agents = [agent_a, agent_b, agent_c, agent_d]

        # Measure execution
        start_time = time.perf_counter()

        async def run_diamond():
            tasks = [agent.run() for agent in agents]
            await asyncio.gather(*tasks)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_diamond())
        finally:
            loop.close()

        total_time = time.perf_counter() - start_time

        # Calculate metrics
        dag_efficiency = (
            (0.06 / total_time) if total_time > 0 else 0
        )  # 3 steps * 0.02s each
        max_dependency_wait = max([a.dependency_wait_time for a in agents])

        return [
            OrchestrationMetric(
                "diamond_dag_time",
                total_time * 1000,
                "ms",
                "Time to resolve and execute diamond DAG",
                True,
            ),
            OrchestrationMetric(
                "dag_execution_efficiency",
                dag_efficiency,
                "ratio",
                "How efficiently the DAG is executed vs theoretical minimum",
                False,
            ),
            OrchestrationMetric(
                "max_dependency_resolution_time",
                max_dependency_wait * 1000,
                "ms",
                "Maximum time any agent waits for dependencies",
                True,
            ),
        ]


class ErrorHandlingBenchmarks:
    """Benchmarks for error handling and resilience."""

    def cascading_failure_resilience(self) -> list[OrchestrationMetric]:
        """Test resilience to cascading failures."""

        class FailureProneAgent(Agent):
            def __init__(self, name: str, failure_rate: float = 0.3):
                super().__init__(name=name)
                self.failure_rate = failure_rate
                self.attempts = 0
                self.success = False

            async def run(self):
                self.attempts += 1

                # Simulate random failures
                import random

                if random.random() < self.failure_rate:
                    raise RuntimeError(
                        f"Agent {self.name} failed on attempt {self.attempts}"
                    )

                await asyncio.sleep(0.01)
                self.success = True
                self.result = {"name": self.name, "attempts": self.attempts}
                return self.result

        # Create agents with varying failure rates
        agents = [
            FailureProneAgent("critical_1", 0.1),
            FailureProneAgent("critical_2", 0.1),
            FailureProneAgent("normal_1", 0.3),
            FailureProneAgent("normal_2", 0.3),
            FailureProneAgent("flaky_1", 0.5),
            FailureProneAgent("flaky_2", 0.5),
        ]

        start_time = time.perf_counter()
        successful_agents = 0
        total_attempts = 0

        # Run with retries
        async def run_with_retries():
            nonlocal successful_agents, total_attempts

            for agent in agents:
                max_retries = 3
                for attempt in range(max_retries + 1):
                    try:
                        await agent.run()
                        successful_agents += 1
                        total_attempts += agent.attempts
                        break
                    except RuntimeError:
                        if attempt == max_retries:
                            total_attempts += agent.attempts
                        continue

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_with_retries())
        finally:
            loop.close()

        total_time = time.perf_counter() - start_time
        success_rate = successful_agents / len(agents)
        avg_attempts = total_attempts / len(agents)

        return [
            OrchestrationMetric(
                "failure_handling_time",
                total_time * 1000,
                "ms",
                "Time to handle failures and retries",
                True,
            ),
            OrchestrationMetric(
                "failure_recovery_rate",
                success_rate,
                "ratio",
                "Proportion of agents that eventually succeeded",
                False,
            ),
            OrchestrationMetric(
                "retry_efficiency",
                avg_attempts,
                "attempts",
                "Average attempts needed per agent",
                True,
            ),
        ]

    def partial_failure_isolation(self) -> list[OrchestrationMetric]:
        """Test isolation of partial failures."""

        class IsolatedAgent(Agent):
            def __init__(self, name: str, should_fail: bool = False):
                super().__init__(name=name)
                self.should_fail = should_fail
                self.isolation_tested = False

            async def run(self):
                if self.should_fail:
                    await asyncio.sleep(0.01)
                    raise RuntimeError(f"Agent {self.name} intentionally failed")

                # Test isolation - should complete even if others fail
                await asyncio.sleep(0.02)
                self.isolation_tested = True
                self.result = {"name": self.name, "isolated": True}
                return self.result

        # Create mix of failing and succeeding agents
        agents = [
            IsolatedAgent("good_1", False),
            IsolatedAgent("good_2", False),
            IsolatedAgent("good_3", False),
            IsolatedAgent("bad_1", True),
            IsolatedAgent("bad_2", True),
        ]

        start_time = time.perf_counter()

        async def run_isolated():
            tasks = []
            for agent in agents:
                task = asyncio.create_task(agent.run())
                tasks.append((agent, task))

            # Gather with return_exceptions to prevent one failure from stopping others
            results = await asyncio.gather(
                *[task for _, task in tasks], return_exceptions=True
            )
            return results

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(run_isolated())
        finally:
            loop.close()

        total_time = time.perf_counter() - start_time

        # Check isolation effectiveness
        good_agents = [a for a in agents if not a.should_fail]
        isolated_successfully = sum(1 for a in good_agents if a.isolation_tested)
        isolation_rate = isolated_successfully / len(good_agents) if good_agents else 0

        # Count successful vs failed results
        successful_results = sum(1 for r in results if not isinstance(r, Exception))
        failure_isolation_efficiency = successful_results / len(agents)

        return [
            OrchestrationMetric(
                "isolation_execution_time",
                total_time * 1000,
                "ms",
                "Time to execute with isolated failures",
                True,
            ),
            OrchestrationMetric(
                "failure_isolation_rate",
                isolation_rate,
                "ratio",
                "Rate of successful isolation from failures",
                False,
            ),
            OrchestrationMetric(
                "partial_success_rate",
                failure_isolation_efficiency,
                "ratio",
                "Proportion of agents that completed despite other failures",
                False,
            ),
        ]


class ResourceContentionBenchmarks:
    """Benchmarks for resource contention and throttling."""

    def concurrent_resource_pressure(self) -> list[OrchestrationMetric]:
        """Test performance under resource pressure."""

        class ResourceIntensiveAgent(Agent):
            def __init__(self, name: str, cpu_intensity: float = 1.0):
                super().__init__(name=name)
                self.cpu_intensity = cpu_intensity
                self.actual_work_time = 0
                self.wait_time = 0

            async def run(self):
                start_time = time.perf_counter()

                # CPU-intensive work (scaled by intensity)
                work_start = time.perf_counter()
                iterations = int(10000 * self.cpu_intensity)
                result = sum(i * i for i in range(iterations))
                self.actual_work_time = time.perf_counter() - work_start

                # I/O simulation
                await asyncio.sleep(0.01)

                self.wait_time = (
                    time.perf_counter() - start_time - self.actual_work_time
                )
                self.result = {"name": self.name, "computation": result}
                return self.result

        # Create agents with different resource requirements
        agents = [ResourceIntensiveAgent(f"light_{i}", 0.5) for i in range(10)] + [
            ResourceIntensiveAgent(f"heavy_{i}", 2.0) for i in range(5)
        ]

        start_time = time.perf_counter()

        async def run_under_pressure():
            # Use a semaphore to limit concurrency (simulate resource constraints)
            semaphore = asyncio.Semaphore(8)  # Limit concurrent execution

            async def run_with_limit(agent):
                async with semaphore:
                    return await agent.run()

            tasks = [run_with_limit(agent) for agent in agents]
            await asyncio.gather(*tasks)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_under_pressure())
        finally:
            loop.close()

        total_time = time.perf_counter() - start_time

        # Calculate contention metrics
        total_work_time = sum(a.actual_work_time for a in agents)
        total_wait_time = sum(a.wait_time for a in agents)
        contention_overhead = (
            total_wait_time / total_work_time if total_work_time > 0 else 0
        )
        resource_utilization = total_work_time / total_time if total_time > 0 else 0

        return [
            OrchestrationMetric(
                "resource_contention_time",
                total_time * 1000,
                "ms",
                "Total time under resource pressure",
                True,
            ),
            OrchestrationMetric(
                "contention_overhead_ratio",
                contention_overhead,
                "ratio",
                "Ratio of wait time to work time",
                True,
            ),
            OrchestrationMetric(
                "resource_utilization_efficiency",
                resource_utilization,
                "ratio",
                "How efficiently resources are utilized",
                False,
            ),
        ]

    def memory_pressure_handling(self) -> list[OrchestrationMetric]:
        """Test handling of memory pressure."""

        class MemoryIntensiveAgent(Agent):
            def __init__(self, name: str, memory_mb: int = 10):
                super().__init__(name=name)
                self.memory_mb = memory_mb
                self.peak_memory = 0

            async def run(self):
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024

                # Allocate memory
                data = []
                for _ in range(self.memory_mb):
                    # Allocate roughly 1MB
                    chunk = [0] * (256 * 1024)  # 256K integers ≈ 1MB
                    data.append(chunk)
                    await asyncio.sleep(0.001)  # Yield control

                current_memory = process.memory_info().rss / 1024 / 1024
                self.peak_memory = current_memory - initial_memory

                # Simulate work with the data
                len(data)

                # Cleanup
                del data

                self.result = {"name": self.name, "memory_used": self.peak_memory}
                return self.result

        # Create agents with different memory requirements
        agents = [MemoryIntensiveAgent(f"small_{i}", 5) for i in range(8)] + [
            MemoryIntensiveAgent(f"large_{i}", 20) for i in range(3)
        ]

        process = psutil.Process()
        initial_system_memory = process.memory_info().rss / 1024 / 1024

        start_time = time.perf_counter()

        async def run_memory_test():
            tasks = [agent.run() for agent in agents]
            await asyncio.gather(*tasks)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_memory_test())
        finally:
            loop.close()

        total_time = time.perf_counter() - start_time
        final_system_memory = process.memory_info().rss / 1024 / 1024

        total_allocated = sum(a.peak_memory for a in agents)
        memory_efficiency = (
            total_allocated / (final_system_memory - initial_system_memory)
            if (final_system_memory - initial_system_memory) > 0
            else 0
        )

        return [
            OrchestrationMetric(
                "memory_pressure_time",
                total_time * 1000,
                "ms",
                "Time to execute under memory pressure",
                True,
            ),
            OrchestrationMetric(
                "memory_allocation_efficiency",
                memory_efficiency,
                "ratio",
                "Efficiency of memory allocation and cleanup",
                False,
            ),
            OrchestrationMetric(
                "peak_memory_usage",
                final_system_memory - initial_system_memory,
                "MB",
                "Peak additional memory used during execution",
                True,
            ),
        ]


class ScalabilityBenchmarks:
    """Benchmarks for scalability characteristics."""

    def horizontal_scaling_efficiency(self) -> list[OrchestrationMetric]:
        """Test how efficiently the framework scales with agent count."""

        scaling_results = []

        for agent_count in [10, 25, 50, 100]:

            class ScalingAgent(Agent):
                def __init__(self, agent_id: int):
                    super().__init__(name=f"scaling_agent_{agent_id}")
                    self.agent_id = agent_id

                async def run(self):
                    # Consistent work regardless of scale
                    await asyncio.sleep(0.01)
                    result = sum(range(100))
                    self.result = {"agent_id": self.agent_id, "result": result}
                    return self.result

            agents = [ScalingAgent(i) for i in range(agent_count)]

            start_time = time.perf_counter()

            async def run_scaling_test(agents_list):
                tasks = [agent.run() for agent in agents_list]
                await asyncio.gather(*tasks)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_scaling_test(agents))
            finally:
                loop.close()

            execution_time = time.perf_counter() - start_time
            throughput = agent_count / execution_time

            scaling_results.append(
                {
                    "agent_count": agent_count,
                    "time": execution_time,
                    "throughput": throughput,
                }
            )

        # Calculate scaling efficiency
        baseline_throughput = scaling_results[0]["throughput"]
        final_throughput = scaling_results[-1]["throughput"]
        scaling_efficiency = (
            final_throughput / baseline_throughput if baseline_throughput > 0 else 0
        )

        # Calculate throughput consistency (lower std dev is better)
        throughputs = [r["throughput"] for r in scaling_results]
        throughput_variance = (
            statistics.stdev(throughputs) / statistics.mean(throughputs)
            if len(throughputs) > 1
            else 0
        )

        return [
            OrchestrationMetric(
                "scaling_efficiency",
                scaling_efficiency,
                "ratio",
                "How well throughput scales with agent count",
                False,
            ),
            OrchestrationMetric(
                "throughput_consistency",
                throughput_variance,
                "cv",
                "Coefficient of variation in throughput (lower is better)",
                True,
            ),
            OrchestrationMetric(
                "max_throughput",
                max(throughputs),
                "agents/sec",
                "Maximum achieved throughput",
                False,
            ),
        ]

    def coordination_overhead_scaling(self) -> list[OrchestrationMetric]:
        """Test coordination overhead as system scales."""

        coordination_results = []

        for coord_complexity in [5, 15, 30, 50]:  # Number of coordination points

            class CoordinationAgent(Agent):
                def __init__(
                    self, agent_id: int, coordination_points: list[asyncio.Event]
                ):
                    super().__init__(name=f"coord_agent_{agent_id}")
                    self.agent_id = agent_id
                    self.coordination_points = coordination_points
                    self.coordination_time = 0

                async def run(self):
                    coord_start = time.perf_counter()

                    # Wait for coordination points
                    for i, event in enumerate(self.coordination_points):
                        if i % 3 == self.agent_id % 3:  # Only wait for relevant events
                            await event.wait()

                    self.coordination_time = time.perf_counter() - coord_start

                    # Do actual work
                    await asyncio.sleep(0.005)

                    self.result = {
                        "agent_id": self.agent_id,
                        "coordination_time": self.coordination_time,
                    }
                    return self.result

            # Create coordination events
            events = [asyncio.Event() for _ in range(coord_complexity)]
            agents = [
                CoordinationAgent(i, events) for i in range(20)
            ]  # Fixed agent count

            start_time = time.perf_counter()

            async def run_coordination_test(agents_list, events_list):
                # Start agents
                tasks = [agent.run() for agent in agents_list]

                # Trigger coordination events gradually
                async def trigger_events(event_list):
                    for event in event_list:
                        await asyncio.sleep(0.002)
                        event.set()

                trigger_task = asyncio.create_task(trigger_events(events_list))

                # Wait for all
                await asyncio.gather(*tasks, trigger_task)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_coordination_test(agents, events))
            finally:
                loop.close()

            total_time = time.perf_counter() - start_time
            avg_coordination_time = statistics.mean(
                [a.coordination_time for a in agents]
            )
            coordination_overhead = (
                avg_coordination_time / total_time if total_time > 0 else 0
            )

            coordination_results.append(
                {
                    "complexity": coord_complexity,
                    "overhead": coordination_overhead,
                    "avg_coord_time": avg_coordination_time,
                }
            )

        # Calculate coordination scaling metrics
        overhead_growth = (
            coordination_results[-1]["overhead"] / coordination_results[0]["overhead"]
            if coordination_results[0]["overhead"] > 0
            else 0
        )
        max_coordination_time = max(r["avg_coord_time"] for r in coordination_results)

        return [
            OrchestrationMetric(
                "coordination_overhead_growth",
                overhead_growth,
                "ratio",
                "How coordination overhead grows with complexity",
                True,
            ),
            OrchestrationMetric(
                "max_coordination_latency",
                max_coordination_time * 1000,
                "ms",
                "Maximum coordination latency observed",
                True,
            ),
            OrchestrationMetric(
                "coordination_scalability",
                1.0 / overhead_growth if overhead_growth > 0 else 1.0,
                "ratio",
                "Inverse of overhead growth (higher is better)",
                False,
            ),
        ]


def run_orchestration_benchmarks():
    """Run all orchestration-focused benchmarks."""

    runner = OrchestrationBenchmarkRunner()

    # Initialize benchmark suites
    complexity_bench = WorkflowComplexityBenchmarks()
    error_bench = ErrorHandlingBenchmarks()
    resource_bench = ResourceContentionBenchmarks()
    scalability_bench = ScalabilityBenchmarks()

    benchmarks = [
        ("Deep Dependency Chain", complexity_bench.deep_dependency_chain),
        ("Wide Fanout Pattern", complexity_bench.wide_fanout_pattern),
        ("Diamond DAG Pattern", complexity_bench.diamond_dependency_pattern),
        ("Cascading Failure Resilience", error_bench.cascading_failure_resilience),
        ("Partial Failure Isolation", error_bench.partial_failure_isolation),
        ("Resource Contention", resource_bench.concurrent_resource_pressure),
        ("Memory Pressure", resource_bench.memory_pressure_handling),
        ("Horizontal Scaling", scalability_bench.horizontal_scaling_efficiency),
        ("Coordination Overhead", scalability_bench.coordination_overhead_scaling),
    ]

    print("🎯 Orchestration-Focused Benchmarks")
    print("=" * 80)
    print("Measuring metrics that actually matter for workflow orchestration:")
    print("• Workflow complexity handling")
    print("• Error resilience and isolation")
    print("• Resource contention management")
    print("• Scalability characteristics")
    print("=" * 80)

    for name, benchmark_func in benchmarks:
        try:
            runner.run_orchestration_benchmark(
                name=name,
                framework="PuffinFlow",
                benchmark_func=benchmark_func,
                iterations=10,  # Fewer iterations for complex benchmarks
            )
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    return runner.results


def print_orchestration_results(results: list[OrchestrationBenchmarkResult]):
    """Print orchestration benchmark results."""

    print("\n" + "=" * 100)
    print("📊 ORCHESTRATION BENCHMARK RESULTS")
    print("=" * 100)

    for result in results:
        print(f"\n🎯 {result.name}")
        print(
            f"   Framework: {result.framework} | Success Rate: {result.success_rate:.1%} | Iterations: {result.iterations}"
        )
        print("-" * 80)

        for metric in result.metrics:
            direction = "↓" if metric.is_better_lower else "↑"
            print(f"   {direction} {metric.name}: {metric.value:.3f} {metric.unit}")
            print(f"     {metric.description}")

        print()

    # Generate insights
    print("\n🔍 KEY INSIGHTS")
    print("-" * 80)

    # Analyze patterns across metrics
    all_metrics = {}
    for result in results:
        for metric in result.metrics:
            if metric.name not in all_metrics:
                all_metrics[metric.name] = []
            all_metrics[metric.name].append(metric.value)

    # Identify potential bottlenecks
    print("\n🚨 Potential Bottlenecks:")
    for metric_name, values in all_metrics.items():
        if "time" in metric_name.lower() or "latency" in metric_name.lower():
            avg_val = statistics.mean(values)
            if avg_val > 100:  # If average time > 100ms
                print(f"   • High {metric_name}: {avg_val:.1f}ms average")

    # Identify strengths
    print("\n✅ Framework Strengths:")
    for metric_name, values in all_metrics.items():
        if "efficiency" in metric_name.lower() or "rate" in metric_name.lower():
            avg_val = statistics.mean(values)
            if avg_val > 0.8:  # If efficiency > 80%
                print(f"   • Good {metric_name}: {avg_val:.1%} average")


def save_orchestration_results(
    results: list[OrchestrationBenchmarkResult], filename: str
):
    """Save results to JSON file."""

    serializable_results = []
    for result in results:
        serializable_metrics = []
        for metric in result.metrics:
            serializable_metrics.append(
                {
                    "name": metric.name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "description": metric.description,
                    "is_better_lower": metric.is_better_lower,
                }
            )

        serializable_results.append(
            {
                "name": result.name,
                "framework": result.framework,
                "metrics": serializable_metrics,
                "iterations": result.iterations,
                "success_rate": result.success_rate,
                "timestamp": result.timestamp,
            }
        )

    with Path(filename).open("w") as f:
        json.dump(
            {
                "orchestration_benchmarks": serializable_results,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "description": "Orchestration-focused benchmarks measuring workflow complexity, error handling, resource management, and scalability",
            },
            f,
            indent=2,
        )


def main():
    """Main function to run orchestration benchmarks."""

    print("🚀 Starting Orchestration-Focused Benchmarks")
    print(
        "These benchmarks measure what actually matters for orchestration frameworks:"
    )
    print("• How well does it handle complex workflow patterns?")
    print("• How resilient is it to failures?")
    print("• How efficiently does it manage resources under pressure?")
    print("• How well does it scale with complexity?")
    print()

    try:
        # Run benchmarks
        results = run_orchestration_benchmarks()

        # Print results
        print_orchestration_results(results)

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"orchestration_benchmark_results_{timestamp}.json"
        save_orchestration_results(results, filename)

        print("\n✅ Orchestration benchmarks completed successfully!")
        print(f"📊 Results saved to: {filename}")
        print(f"📈 Total benchmarks run: {len(results)}")

        return results

    except Exception as e:
        print(f"\n❌ Orchestration benchmarks failed: {e}")
        import traceback

        traceback.print_exc()
        return []


if __name__ == "__main__":
    main()
