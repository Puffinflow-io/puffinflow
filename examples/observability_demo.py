"""
Observability Demo

This example demonstrates observability and monitoring capabilities:
- Agent monitoring and metrics collection
- Tracing and logging
- Performance monitoring
- Custom observability configurations
"""

import asyncio
import random
import time

from puffinflow import Agent, Context, cpu_intensive, state
from puffinflow.core.observability import (
    observe,
    trace_state,
)


class MonitoredAgent(Agent):
    """Agent with comprehensive monitoring and observability."""

    def __init__(self, name: str):
        super().__init__(name)
        self.metrics = {}
        self.traces = []

        # Register all decorated states
        self.add_state("initialize_monitoring", self.initialize_monitoring)
        self.add_state("collect_baseline_metrics", self.collect_baseline_metrics)
        self.add_state("start_monitoring", self.start_monitoring)
        self.add_state("analyze_performance", self.analyze_performance)

    @observe(metrics=["duration", "calls", "success_rate"])
    @state(cpu=1.0, memory=256.0)
    async def initialize_monitoring(self, context: Context):
        """Initialize monitoring and observability systems."""
        start_time = time.time()

        # Set up monitoring configuration
        monitoring_config = {
            "agent_id": self.name,
            "monitoring_enabled": True,
            "metrics_collection": True,
            "trace_collection": True,
            "log_level": "INFO",
        }

        self.set_variable("monitoring_config", monitoring_config)

        # Record initialization metrics
        init_duration = time.time() - start_time
        self.record_metric("initialization_time", init_duration)

        context.set_output("monitoring_status", "initialized")
        context.set_metric("init_duration", init_duration)

        print(f"{self.name} monitoring initialized in {init_duration:.3f}s")
        return "collect_baseline_metrics"

    @trace_state(span_name="baseline_collection")
    @state(cpu=0.5, memory=128.0)
    async def collect_baseline_metrics(self, context: Context):
        """Collect baseline performance metrics."""
        baseline_metrics = {
            "cpu_baseline": 0.1,
            "memory_baseline": 128.0,
            "response_time_baseline": 0.05,
            "throughput_baseline": 100.0,
            "error_rate_baseline": 0.01,
        }

        # Simulate baseline collection
        await asyncio.sleep(0.1)

        for metric_name, value in baseline_metrics.items():
            self.record_metric(metric_name, value)

        context.set_output("baseline_metrics", baseline_metrics)
        print(f"{self.name} collected baseline metrics")
        return "start_monitoring"

    @observe(metrics=["performance", "resource_usage"])
    @cpu_intensive(cpu=2.0, memory=512.0)
    async def start_monitoring(self, context: Context):
        """Start active monitoring of agent performance."""
        monitoring_duration = 0.5  # Monitor for 500ms
        sample_interval = 0.1  # Sample every 100ms

        performance_samples = []

        start_time = time.time()
        while time.time() - start_time < monitoring_duration:
            # Simulate performance monitoring
            sample = {
                "timestamp": time.time(),
                "cpu_usage": random.uniform(1.5, 2.5),
                "memory_usage": random.uniform(400, 600),
                "response_time": random.uniform(0.05, 0.15),
                "active_connections": random.randint(5, 15),
            }

            performance_samples.append(sample)

            # Record real-time metrics
            self.record_metric("cpu_usage", sample["cpu_usage"])
            self.record_metric("memory_usage", sample["memory_usage"])

            await asyncio.sleep(sample_interval)

        # Calculate aggregated metrics
        avg_cpu = sum(s["cpu_usage"] for s in performance_samples) / len(
            performance_samples
        )
        avg_memory = sum(s["memory_usage"] for s in performance_samples) / len(
            performance_samples
        )
        max_response_time = max(s["response_time"] for s in performance_samples)

        monitoring_result = {
            "samples_collected": len(performance_samples),
            "average_cpu": avg_cpu,
            "average_memory": avg_memory,
            "max_response_time": max_response_time,
            "monitoring_duration": monitoring_duration,
        }

        context.set_output("monitoring_result", monitoring_result)
        context.set_metric("avg_cpu_usage", avg_cpu)
        context.set_metric("avg_memory_usage", avg_memory)

        print(f"{self.name} monitoring completed: {len(performance_samples)} samples")
        return "analyze_performance"

    @state(cpu=1.0, memory=256.0)
    async def analyze_performance(self, context: Context):
        """Analyze collected performance data."""
        monitoring_result = context.get_output("monitoring_result", {})
        baseline_metrics = context.get_output("baseline_metrics", {})

        # Compare current performance with baseline
        analysis = {
            "cpu_variance": monitoring_result.get("average_cpu", 0)
            - baseline_metrics.get("cpu_baseline", 0),
            "memory_variance": monitoring_result.get("average_memory", 0)
            - baseline_metrics.get("memory_baseline", 0),
            "performance_score": 0.85 + random.uniform(-0.1, 0.1),
            "recommendations": [],
        }

        # Generate recommendations based on analysis
        if analysis["cpu_variance"] > 1.0:
            analysis["recommendations"].append("Consider CPU optimization")

        if analysis["memory_variance"] > 200:
            analysis["recommendations"].append("Monitor memory usage patterns")

        if analysis["performance_score"] < 0.8:
            analysis["recommendations"].append("Performance tuning required")
        else:
            analysis["recommendations"].append("Performance within acceptable range")

        context.set_output("performance_analysis", analysis)
        context.set_metric("performance_score", analysis["performance_score"])

        print(
            f"{self.name} performance analysis: {analysis['performance_score']:.2f} score"
        )
        return None

    def record_metric(self, name: str, value: float):
        """Record a custom metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({"value": value, "timestamp": time.time()})

    def get_metric_summary(self, name: str) -> dict:
        """Get summary statistics for a metric."""
        if name not in self.metrics:
            return {}

        values = [m["value"] for m in self.metrics[name]]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else None,
        }


class TracingAgent(Agent):
    """Agent that demonstrates distributed tracing capabilities."""

    def __init__(self, name: str):
        super().__init__(name)
        # Register all decorated states
        self.add_state("process_data_pipeline", self.process_data_pipeline)

    @trace_state(span_name="data_processing_pipeline")
    @state(cpu=2.0, memory=512.0)
    async def process_data_pipeline(self, context: Context):
        """Process data through a traced pipeline."""
        pipeline_id = f"pipeline_{int(time.time())}"

        # Start main pipeline trace
        with self.create_trace_span("pipeline_execution") as pipeline_span:
            pipeline_span.set_attribute("pipeline_id", pipeline_id)
            pipeline_span.set_attribute("data_size", 1000)

            # Stage 1: Data ingestion
            with self.create_trace_span("data_ingestion") as ingestion_span:
                await asyncio.sleep(0.1)
                ingestion_span.set_attribute("records_ingested", 1000)
                ingestion_span.set_attribute("ingestion_rate", 10000)

            # Stage 2: Data validation
            with self.create_trace_span("data_validation") as validation_span:
                await asyncio.sleep(0.05)
                validation_errors = random.randint(0, 5)
                validation_span.set_attribute("validation_errors", validation_errors)
                validation_span.set_attribute(
                    "validation_success", validation_errors == 0
                )

            # Stage 3: Data transformation
            with self.create_trace_span("data_transformation") as transform_span:
                await asyncio.sleep(0.15)
                transform_span.set_attribute(
                    "transformation_type", "normalize_and_enrich"
                )
                transform_span.set_attribute("output_records", 950)

            # Stage 4: Data output
            with self.create_trace_span("data_output") as output_span:
                await asyncio.sleep(0.08)
                output_span.set_attribute("output_format", "json")
                output_span.set_attribute("compression_ratio", 0.7)

        pipeline_result = {
            "pipeline_id": pipeline_id,
            "total_duration": 0.38,
            "records_processed": 950,
            "stages_completed": 4,
            "success": True,
        }

        context.set_output("pipeline_result", pipeline_result)
        print(f"{self.name} completed traced pipeline: {pipeline_id}")
        return None

    def create_trace_span(self, span_name: str):
        """Create a trace span (simplified implementation)."""
        return TraceSpan(span_name, self.name)


class TraceSpan:
    """Simplified trace span implementation for demonstration."""

    def __init__(self, name: str, agent_name: str):
        self.name = name
        self.agent_name = agent_name
        self.attributes = {}
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"  TRACE START: {self.agent_name}.{self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"  TRACE END: {self.agent_name}.{self.name} ({duration:.3f}s)")
        if self.attributes:
            print(f"    Attributes: {self.attributes}")

    def set_attribute(self, key: str, value):
        """Set a trace attribute."""
        self.attributes[key] = value


class AlertingAgent(Agent):
    """Agent that demonstrates alerting and anomaly detection."""

    def __init__(self, name: str):
        super().__init__(name)
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 90.0,
            "error_rate": 5.0,
            "response_time": 1.0,
        }
        self.alerts = []

        # Register all decorated states
        self.add_state("monitor_system_health", self.monitor_system_health)
        self.add_state("generate_health_report", self.generate_health_report)

    @state(cpu=1.0, memory=256.0)
    async def monitor_system_health(self, context: Context):
        """Monitor system health and generate alerts."""
        # Simulate system metrics
        current_metrics = {
            "cpu_usage": random.uniform(60, 95),
            "memory_usage": random.uniform(70, 95),
            "error_rate": random.uniform(0, 10),
            "response_time": random.uniform(0.1, 2.0),
            "disk_usage": random.uniform(50, 90),
            "network_latency": random.uniform(10, 100),
        }

        # Check for threshold violations
        alerts_triggered = []

        for metric, value in current_metrics.items():
            if metric in self.alert_thresholds:
                threshold = self.alert_thresholds[metric]
                if value > threshold:
                    alert = {
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                        "severity": self.calculate_severity(value, threshold),
                        "timestamp": time.time(),
                        "agent": self.name,
                    }
                    alerts_triggered.append(alert)
                    self.alerts.append(alert)

        context.set_output("current_metrics", current_metrics)
        context.set_output("alerts_triggered", alerts_triggered)
        context.set_metric("alerts_count", len(alerts_triggered))

        if alerts_triggered:
            print(f"{self.name} triggered {len(alerts_triggered)} alerts")
            for alert in alerts_triggered:
                print(
                    f"  ALERT: {alert['metric']} = {alert['value']:.1f} "
                    f"(threshold: {alert['threshold']}) - {alert['severity']}"
                )
        else:
            print(f"{self.name} system health normal")

        return "generate_health_report"

    @state(cpu=0.5, memory=128.0)
    async def generate_health_report(self, context: Context):
        """Generate a comprehensive health report."""
        current_metrics = context.get_output("current_metrics", {})
        alerts_triggered = context.get_output("alerts_triggered", [])

        # Calculate overall health score
        health_score = 100.0
        for alert in alerts_triggered:
            severity_penalty = {"LOW": 5, "MEDIUM": 15, "HIGH": 30, "CRITICAL": 50}
            health_score -= severity_penalty.get(alert["severity"], 10)

        health_score = max(0, health_score)

        health_report = {
            "overall_health_score": health_score,
            "health_status": self.get_health_status(health_score),
            "metrics_summary": current_metrics,
            "active_alerts": len(alerts_triggered),
            "total_alerts": len(self.alerts),
            "recommendations": self.generate_recommendations(alerts_triggered),
            "report_timestamp": time.time(),
        }

        context.set_output("health_report", health_report)
        context.set_metric("health_score", health_score)

        print(
            f"{self.name} health report: {health_score:.1f}/100 ({health_report['health_status']})"
        )
        return None

    def calculate_severity(self, value: float, threshold: float) -> str:
        """Calculate alert severity based on threshold violation."""
        ratio = value / threshold
        if ratio >= 2.0:
            return "CRITICAL"
        elif ratio >= 1.5:
            return "HIGH"
        elif ratio >= 1.2:
            return "MEDIUM"
        else:
            return "LOW"

    def get_health_status(self, score: float) -> str:
        """Get health status based on score."""
        if score >= 90:
            return "EXCELLENT"
        elif score >= 75:
            return "GOOD"
        elif score >= 50:
            return "FAIR"
        elif score >= 25:
            return "POOR"
        else:
            return "CRITICAL"

    def generate_recommendations(self, alerts: list) -> list:
        """Generate recommendations based on alerts."""
        recommendations = []

        for alert in alerts:
            metric = alert["metric"]
            if metric == "cpu_usage":
                recommendations.append(
                    "Consider scaling CPU resources or optimizing workloads"
                )
            elif metric == "memory_usage":
                recommendations.append(
                    "Monitor memory leaks and consider increasing memory allocation"
                )
            elif metric == "error_rate":
                recommendations.append(
                    "Investigate error patterns and implement error handling"
                )
            elif metric == "response_time":
                recommendations.append(
                    "Optimize performance bottlenecks and consider caching"
                )

        if not recommendations:
            recommendations.append("System operating within normal parameters")

        return list(set(recommendations))  # Remove duplicates


async def demonstrate_basic_observability():
    """Demonstrate basic observability features."""
    print("=== Basic Observability ===")

    # Create and run monitored agent
    monitored_agent = MonitoredAgent("monitored-agent")
    result = await monitored_agent.run()

    # Display monitoring results
    monitoring_result = result.get_output("monitoring_result", {})
    performance_analysis = result.get_output("performance_analysis", {})

    print(f"Monitoring samples: {monitoring_result.get('samples_collected', 0)}")
    print(f"Performance score: {performance_analysis.get('performance_score', 0):.2f}")
    print(f"Recommendations: {performance_analysis.get('recommendations', [])}")

    # Display custom metrics
    print("\nCustom Metrics Summary:")
    for metric_name in ["cpu_usage", "memory_usage", "initialization_time"]:
        summary = monitored_agent.get_metric_summary(metric_name)
        if summary:
            print(
                f"  {metric_name}: avg={summary['avg']:.2f}, "
                f"min={summary['min']:.2f}, max={summary['max']:.2f}"
            )

    print()


async def demonstrate_distributed_tracing():
    """Demonstrate distributed tracing capabilities."""
    print("=== Distributed Tracing ===")

    # Create and run tracing agent
    tracing_agent = TracingAgent("tracing-agent")
    result = await tracing_agent.run()

    pipeline_result = result.get_output("pipeline_result", {})
    print(f"Pipeline completed: {pipeline_result.get('pipeline_id', 'unknown')}")
    print(f"Records processed: {pipeline_result.get('records_processed', 0)}")
    print(f"Total duration: {pipeline_result.get('total_duration', 0):.3f}s")
    print()


async def demonstrate_alerting_system():
    """Demonstrate alerting and health monitoring."""
    print("=== Alerting System ===")

    # Create and run alerting agent
    alerting_agent = AlertingAgent("alerting-agent")
    result = await alerting_agent.run()

    health_report = result.get_output("health_report", {})
    alerts_triggered = result.get_output("alerts_triggered", [])

    print(f"Health Score: {health_report.get('overall_health_score', 0):.1f}/100")
    print(f"Health Status: {health_report.get('health_status', 'UNKNOWN')}")
    print(f"Active Alerts: {len(alerts_triggered)}")

    if health_report.get("recommendations"):
        print("Recommendations:")
        for rec in health_report["recommendations"]:
            print(f"  - {rec}")

    print()


async def run_observability_comparison():
    """Compare performance with and without observability."""
    print("=== Observability Performance Impact ===")

    # Run without observability
    start_time = time.time()
    simple_agent = Agent("simple-agent")
    simple_agent.add_state("dummy", lambda ctx: None)
    await simple_agent.run()
    simple_time = time.time() - start_time

    # Run with observability
    start_time = time.time()
    monitored_agent = MonitoredAgent("comparison-monitored")
    await monitored_agent.run()
    monitored_time = time.time() - start_time

    overhead = (
        ((monitored_time - simple_time) / simple_time) * 100 if simple_time > 0 else 0
    )

    print(f"Simple agent execution: {simple_time:.3f}s")
    print(f"Monitored agent execution: {monitored_time:.3f}s")
    print(f"Observability overhead: {overhead:.1f}%")
    print()


async def main():
    """Run all observability examples."""
    print("PuffinFlow Observability Examples")
    print("=" * 50)

    await demonstrate_basic_observability()
    await demonstrate_distributed_tracing()
    await demonstrate_alerting_system()
    await run_observability_comparison()

    print("All observability examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
