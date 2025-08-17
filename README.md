# PuffinFlow

[![PyPI version](https://badge.fury.io/py/puffinflow.svg)](https://badge.fury.io/py/puffinflow)
[![Python versions](https://img.shields.io/pypi/pyversions/puffinflow.svg)](https://pypi.org/project/puffinflow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PuffinFlow is a high-performance Python framework for building production-ready LLM workflows and multi-agent systems.**

Perfect for AI engineers, data scientists, and backend developers who need to build reliable, scalable, and observable workflow orchestration systems.

## Quick Start

Install PuffinFlow:

```bash
pip install puffinflow
```

Create your first agent with state management:

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

## Core Features

**Production-Ready Performance**: Sub-millisecond latency for basic operations with throughput exceeding 12,000 ops/s.

**Intelligent Resource Management**: Automatic allocation and management of CPU, memory, and other resources with built-in quotas and limits.

**Zero-Configuration Observability**: Comprehensive monitoring with OpenTelemetry integration, custom metrics, distributed tracing, and real-time alerting.

**Built-in Reliability**: Circuit breakers, bulkheads, timeout handling, and leak detection ensure robust operation under failure conditions.

**Multi-Agent Coordination**: Scale from single agents to complex multi-agent workflows with teams, pools, and orchestrators.

**Seamless Development Experience**: Prototype quickly and transition to production without code rewrites.

## Performance Benchmarks

PuffinFlow delivers exceptional performance in production workloads. Our comprehensive benchmark suite compares PuffinFlow against leading orchestration frameworks.

### Framework Comparison Results

| Framework | Simple Tasks | Multi-Task Workflows | Coordination | Overall Score |
|-----------|-------------|-------------------|-------------|---------------|
| **PuffinFlow** | **0.08ms (12,519 ops/s)** | **0.09ms (11,559 ops/s)** | 1.12ms (890 ops/s) | **1st Place** |
| LangGraph | 0.44ms (2,294 ops/s) | 0.63ms (1,586 ops/s) | 1.53ms (655 ops/s) | 2nd Place |
| Dagster | 21.18ms (47 ops/s) | 36.23ms (28 ops/s) | **1.06ms (946 ops/s)** | 3rd Place |
| Prefect | 228.21ms (4 ops/s) | 580.76ms (2 ops/s) | 51.53ms (19 ops/s) | 4th Place |

### Key Performance Metrics

**Simple Task Execution**
- PuffinFlow: **265x faster** than Dagster, **2,844x faster** than Prefect
- Consistent sub-millisecond performance with minimal variance

**Multi-Task Workflows** 
- PuffinFlow: **402x faster** than Dagster, **6,453x faster** than Prefect
- Excellent scaling characteristics for complex workflows

**Coordination Primitives**
- Competitive performance with specialized coordination frameworks
- Balanced performance across all operation types

### System Specifications
- **Platform**: Linux WSL2
- **CPU**: 16 cores @ 2.3GHz
- **Memory**: 3.68GB RAM
- **Python**: 3.12.3

*Benchmarks conducted using identical test scenarios across all frameworks. Results represent average performance over 50 iterations per test.*

## Real-World Examples

### Image Processing Pipeline
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

### ML Model Training Workflow
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

### Multi-Agent Coordination
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

## Use Cases

**Data Pipelines**: Build resilient ETL workflows with automatic retries, resource management, and comprehensive monitoring.

**ML Workflows**: Orchestrate training pipelines, model deployment, and inference workflows with checkpointing and observability.

**Microservices**: Coordinate distributed services with circuit breakers, bulkheads, and intelligent load balancing.

**Event Processing**: Handle high-throughput event streams with backpressure control and automatic scaling.

**API Orchestration**: Coordinate complex API interactions with built-in retry policies and error handling.

## Ecosystem Integration

PuffinFlow integrates seamlessly with popular Python frameworks:

**FastAPI & Django**: Native async support for web application integration with automatic resource management.

**Celery & Redis**: Enhance existing task queues with stateful workflows, advanced coordination, and monitoring.

**OpenTelemetry**: Complete observability stack with distributed tracing, metrics, and monitoring platform integration.

**Kubernetes**: Production-ready deployment with container orchestration and cloud-native observability.

## Architecture

PuffinFlow is built on a robust, production-tested architecture:

- **Agent-Based Design**: Modular, stateful agents with lifecycle management
- **Resource Pooling**: Intelligent allocation and management of compute resources
- **Coordination Layer**: Built-in primitives for multi-agent synchronization
- **Observability Core**: Comprehensive monitoring and telemetry collection
- **Reliability Systems**: Circuit breakers, bulkheads, and failure detection

## Documentation & Resources

- **[Documentation](https://puffinflow.readthedocs.io/)**: Complete guides and API reference
- **[Examples](./examples/)**: Ready-to-run code examples for common patterns
- **[Advanced Guides](./docs/source/guides/)**: Deep dives into resource management, coordination, and observability
- **[Benchmarks](./benchmarks/)**: Performance metrics and comparison studies

## Community & Support

- **[Issues](https://github.com/m-ahmed-elbeskeri/puffinflow-main/issues)**: Bug reports and feature requests
- **[Discussions](https://github.com/m-ahmed-elbeskeri/puffinflow-main/discussions)**: Community Q&A and discussions
- **[Email](mailto:mohamed.ahmed.4894@gmail.com)**: Direct contact for support and partnerships

## Contributing

We welcome contributions from the community. Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## License

PuffinFlow is released under the [MIT License](LICENSE). Free for commercial and personal use.

---

<div align="center">

**Ready to build production-ready workflows?**

[Get Started](https://puffinflow.readthedocs.io/) | [View Examples](./examples/) | [Join Community](https://github.com/m-ahmed-elbeskeri/puffinflow-main/discussions)

</div>