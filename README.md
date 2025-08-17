# 🐧 PuffinFlow

[![PyPI version](https://badge.fury.io/py/puffinflow.svg)](https://badge.fury.io/py/puffinflow)
[![Python versions](https://img.shields.io/pypi/pyversions/puffinflow.svg)](https://pypi.org/project/puffinflow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PuffinFlow is a powerful Python framework for developers who need to rapidly prototype LLM workflows and seamlessly transition them to production-ready systems.**

Perfect for AI engineers, data scientists, and backend developers who want to focus on workflow logic rather than infrastructure complexity.

## Get started

Install PuffinFlow:

```bash
pip install puffinflow
```

Then, create an agent using the state decorator:

```python
from puffinflow import Agent, state

class DataProcessor(Agent):
    @state(cpu=2.0, memory=1024.0)
    async def fetch_data(self, context):
        """Fetch data from external source."""
        data = await get_external_data()
        context.set_variable("raw_data", data)
        return "validate_data" if data else "error"

    @state(cpu=1.0, memory=512.0)
    async def validate_data(self, context):
        """Validate the fetched data."""
        data = context.get_variable("raw_data")
        if self.is_valid(data):
            return "process_data"
        return "error"

    @state(cpu=4.0, memory=2048.0)
    async def process_data(self, context):
        """Process the validated data."""
        data = context.get_variable("raw_data")
        result = await self.transform_data(data)
        context.set_output("processed_data", result)
        return "complete"

# Run the agent
agent = DataProcessor("data-processor")
result = await agent.run()
```

For more information, see the [Documentation](https://puffinflow.readthedocs.io/) and [Examples](./examples/). To learn how to build complex multi-agent workflows with coordination and observability, see the [Advanced Examples](./examples/).

## Core benefits

PuffinFlow bridges the gap between quick prototyping and production deployment. Start building your LLM workflow in minutes, then scale to production without rewriting code:

**Prototype to Production**: Begin with simple agents and seamlessly add resource management, observability, and coordination as your needs grow.

**Intelligent resource management**: Automatically allocate and manage CPU, memory, and other resources based on state requirements with built-in quotas and limits.

**Zero-config observability**: Comprehensive monitoring with OpenTelemetry integration, custom metrics, distributed tracing, and real-time alerting that works out of the box.

**Built-in reliability**: Circuit breakers, bulkheads, and leak detection ensure robust operation under various failure conditions without additional configuration.

**Agent coordination**: Scale from single agents to complex multi-agent workflows with teams, pools, and orchestrators using the same simple API.

**Production performance**: Achieve high throughput with sub-millisecond latency for basic operations, designed for real-world production workloads.

## PuffinFlow's ecosystem

While PuffinFlow can be used standalone, it integrates with popular Python frameworks and tools:

**FastAPI & Django** — Seamlessly integrate PuffinFlow agents into web applications with built-in async support and resource management.

**Celery & Redis** — Enhance existing task queues with stateful workflows, advanced coordination, and comprehensive monitoring.

**OpenTelemetry** — Full observability stack with distributed tracing, metrics collection, and integration with monitoring platforms like Prometheus and Jaeger.

**Kubernetes** — Production-ready deployment with container orchestration, automatic scaling, and cloud-native observability.

## Additional resources

- **[Documentation](https://puffinflow.readthedocs.io/)**: Complete guides and API reference
- **[Examples](./examples/)**: Ready-to-run code examples for common patterns
- **[Advanced Guides](./docs/source/guides/)**: Deep dives into resource management, coordination, and observability
- **[Benchmarks](./benchmarks/)**: Performance metrics and optimization guides

---

## Real-World Examples

### 🔥 Image Processing Pipeline
```python
class ImageProcessor(Agent):
    @state(cpu=2.0, memory=1024.0)
    async def resize_image(self, context):
        image_url = context.get_variable("image_url")
        resized = await resize_image(image_url, size=(800, 600))
        context.set_variable("resized_image", resized)
        return "add_watermark"

    @state(cpu=1.0, memory=512.0)
    async def add_watermark(self, context):
        image = context.get_variable("resized_image")
        watermarked = await add_watermark(image)
        context.set_variable("final_image", watermarked)
        return "upload_to_storage"

    @state(cpu=1.0, memory=256.0)
    async def upload_to_storage(self, context):
        image = context.get_variable("final_image")
        url = await upload_to_s3(image)
        context.set_output("result_url", url)
        return "complete"
```

### 🤖 ML Model Training
```python
class MLTrainer(Agent):
    @state(cpu=8.0, memory=4096.0)
    async def train_model(self, context):
        dataset = context.get_variable("dataset")
        model = await train_neural_network(dataset)
        context.set_variable("model", model)
        context.set_output("accuracy", model.accuracy)

        if model.accuracy > 0.9:
            return "deploy_model"
        return "retrain_with_more_data"

    @state(cpu=2.0, memory=1024.0)
    async def deploy_model(self, context):
        model = context.get_variable("model")
        await deploy_to_production(model)
        context.set_output("deployment_status", "success")
        return "complete"
```

### 🔄 Multi-Agent Coordination
```python
from puffinflow import create_team, AgentTeam

# Coordinate multiple agents
email_team = create_team([
    EmailValidator("validator"),
    EmailProcessor("processor"),
    EmailTracker("tracker")
])

# Execute with built-in coordination
result = await email_team.execute_parallel()
```

---

## 🎯 Use Cases

**📊 Data Pipelines** — Build resilient ETL workflows with automatic retries, resource management, and comprehensive monitoring.

**🤖 ML Workflows** — Orchestrate training pipelines, model deployment, and inference workflows with checkpointing and observability.

**🌐 Microservices** — Coordinate distributed services with circuit breakers, bulkheads, and intelligent load balancing.

**⚡ Event Processing** — Handle high-throughput event streams with backpressure control and automatic scaling.

## 📊 Performance

PuffinFlow is built for production workloads with excellent performance characteristics:

### Latest Benchmark Results

PuffinFlow delivers excellent performance across different operation types. Here are the latest benchmark results:

#### Core Performance Metrics
| Operation Type | Avg Latency | Throughput | Use Case |
|---|---|---|---|
| Simple Task Execution | 0.39ms | 2,574 ops/s | Basic agent operations |
| Multi-Task Workflows | 0.44ms | 2,276 ops/s | Complex workflow coordination |
| Coordination Primitives | 1.19ms | 842 ops/s | Agent synchronization |
| Framework Comparison¹ | 0.50ms | 1,983 ops/s | vs. other frameworks |

#### Performance Distribution
- **Min Latency**: 0.06ms (fastest operations)
- **Max Latency**: 17.1ms (complex workflows under load)
- **Median Latency**: 0.06-1.15ms (typical operations)
- **Standard Deviation**: 0.04-2.4ms (consistency measure)

*¹ Benchmarked against Dagster, Prefect, and LangGraph*

**System Specifications**: Linux WSL2, 16 cores, 3.68GB RAM, Python 3.12

#### Detailed Benchmark Categories
- **Simple Task Execution**: Basic agent lifecycle and state execution
- **Multi-Task Workflows**: Complex multi-step agent coordination
- **Coordination Primitives**: Synchronization and coordination overhead
- **Resource Management**: CPU and memory allocation performance
- **Framework Comparison**: Performance vs. popular orchestration frameworks

[View detailed benchmarks →](./benchmarks/)

## 🤝 Community & Support

- **[🐛 Issues](https://github.com/m-ahmed-elbeskeri/puffinflow-main/issues)** — Bug reports and feature requests
- **[💬 Discussions](https://github.com/m-ahmed-elbeskeri/puffinflow-main/discussions)** — Community Q&A
- **[📧 Email](mailto:mohamed.ahmed.4894@gmail.com)** — Direct contact for support

## Acknowledgements

PuffinFlow is inspired by workflow orchestration principles and builds upon the Python async ecosystem. The framework emphasizes practical workflow management with production-ready features. PuffinFlow is built by Mohamed Ahmed, designed for developers who need reliable, observable, and scalable workflow orchestration.

## 📜 License

PuffinFlow is released under the [MIT License](LICENSE). Free for commercial and personal use.

---

<div align="center">

**Ready to build production-ready workflows?**

[Get Started →](https://puffinflow.readthedocs.io/) | [View Examples →](./examples/) | [Join Community →](https://github.com/m-ahmed-elbeskeri/puffinflow-main/discussions)

</div>
