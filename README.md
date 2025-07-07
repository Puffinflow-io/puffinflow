# 🐧 PuffinFlow

[![PyPI version](https://badge.fury.io/py/puffinflow.svg)](https://badge.fury.io/py/puffinflow)
[![Python versions](https://img.shields.io/pypi/pyversions/puffinflow.svg)](https://pypi.org/project/puffinflow/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Stop writing complex workflow orchestration code. Just define agents and let PuffinFlow handle the rest.**

Turn this messy workflow code:
```python
# 😵 Before: Complex orchestration nightmare
def process_data():
    try:
        raw_data = fetch_data()
        if validate_data(raw_data):
            processed = transform_data(raw_data)
            if processed:
                result = analyze_data(processed)
                if result.confidence > 0.8:
                    store_results(result)
                    notify_completion()
                else:
                    retry_analysis()
    except Exception as e:
        handle_error(e)
        maybe_retry()
```

Into this simple, reliable agent:
```python
# 🚀 After: Clean, self-managing workflow
class DataPipeline(Agent):
    @state(cpu=2.0, memory=1024.0)
    async def fetch_data(self, context):
        data = await get_external_data()
        return "validate" if data else "error"

    @state(cpu=1.0, memory=512.0)
    async def validate(self, context):
        return "transform" if context.data.is_valid else "error"

    @state(cpu=4.0, memory=2048.0)
    async def transform(self, context):
        context.result = await process_data(context.data)
        return "complete"
```

**567,000+ operations/second** • **Sub-millisecond latency** • **Production-ready**

---

## Why Developers Love PuffinFlow

**🔥 No More Workflow Hell** - Stop writing nested try/catch blocks and complex state management
**⚡ Insanely Fast** - 567K+ ops/sec performance that beats traditional orchestrators
**🧠 Smart Resource Management** - Automatically handles CPU/memory allocation and scaling
**🛠️ Works With Your Stack** - Drop into FastAPI, Django, or any Python app

## Get Started in 30 Seconds

```bash
pip install puffinflow
```

```python
from puffinflow import Agent, state

class EmailProcessor(Agent):
    @state
    async def validate_email(self, context):
        if "@" in context.email:
            return "send_email"
        return "invalid"

    @state
    async def send_email(self, context):
        await send_email(context.email, context.message)
        return "complete"

# Run it
agent = EmailProcessor()
result = await agent.run({"email": "user@example.com", "message": "Hello!"})
```

**That's it.** PuffinFlow handles retries, resource management, error handling, and scaling automatically.

---

## Real-World Examples

### 🔥 Image Processing Pipeline
```python
class ImageProcessor(Agent):
    @state(cpu=2.0)
    async def resize_image(self, context):
        image = await resize(context.image_url, size=(800, 600))
        context.processed_image = image
        return "add_watermark"

    @state(cpu=1.0)
    async def add_watermark(self, context):
        watermarked = await add_watermark(context.processed_image)
        context.final_image = watermarked
        return "upload_to_s3"

    @state(cpu=1.0)
    async def upload_to_s3(self, context):
        url = await upload_to_s3(context.final_image)
        context.result_url = url
        return "complete"
```

### 🤖 ML Model Training
```python
class MLTrainer(Agent):
    @state(cpu=8.0, memory=4096.0)
    async def train_model(self, context):
        model = await train_neural_network(context.dataset)
        if model.accuracy > 0.9:
            return "deploy_model"
        return "retrain_with_more_data"

    @state(cpu=2.0)
    async def deploy_model(self, context):
        await deploy_to_production(context.model)
        return "complete"
```

### 🧠 LLM Agent Pipeline
```python
class LLMAgent(Agent):
    @state(cpu=2.0, memory=2048.0)
    async def generate_response(self, context):
        response = await llm_call(context.prompt)
        context.raw_response = response
        return "validate_output"

    @state
    async def validate_output(self, context):
        if is_safe_content(context.raw_response):
            return "format_response"
        return "regenerate_with_filter"

    @state
    async def format_response(self, context):
        context.final_response = format_markdown(context.raw_response)
        return "complete"
```

### 🔄 Start Simple, Scale to Production
```python
# Development: Simple single-agent workflow
dev_agent = EmailProcessor()
await dev_agent.run({"email": "test@example.com"})

# Production: Multi-agent coordination with monitoring
production_team = create_team([
    EmailValidator("validator"),
    EmailProcessor("processor"),
    EmailTracker("tracker")
])
await production_team.execute_with_monitoring()
```

---

## 🎯 Use Cases

### 📊 Data Pipelines
Build resilient ETL workflows with automatic retries, resource management, and monitoring.

### 🤖 ML Workflows
Orchestrate training pipelines, model deployment, and inference workflows with checkpointing and rollback.

### 🌐 Microservices
Coordinate distributed services with circuit breakers, bulkheads, and intelligent load balancing.

### ⚡ Event Processing
Handle high-throughput event streams with backpressure control and automatic scaling.

---

## 📚 Learn More

- **[📖 Documentation](https://puffinflow.readthedocs.io/)** - Complete guides and API reference
- **[🚀 Examples](./examples/)** - Ready-to-run code examples
- **[🎯 Tutorials](./docs/source/guides/)** - Step-by-step learning path
- **[🔧 Advanced Patterns](./docs/source/guides/advanced.rst)** - Production deployment strategies

---

## 🤝 Community & Support

- **[🐛 Issues](https://github.com/m-ahmed-elbeskeri/puffinflow/issues)** - Bug reports and feature requests
- **[💬 Discussions](https://github.com/m-ahmed-elbeskeri/puffinflow/discussions)** - Community Q&A
- **[📧 Email](mailto:mohamed.ahmed.4894@gmail.com)** - Direct contact for support

---

## 🚀 Production Ready

**Deploy with confidence** - PuffinFlow handles the hard stuff:
- **Automatic retries** with exponential backoff
- **Resource management** and CPU/memory limits
- **Built-in monitoring** and observability
- **Kubernetes integration** for container deployments
- **Type safety** throughout the entire framework

---

## 📊 Performance

PuffinFlow is built for production workloads with excellent performance characteristics:

### Core Performance Metrics
- **567,000+ operations/second** for basic agent operations
- **27,000+ operations/second** for complex data processing
- **1,100+ operations/second** for CPU-intensive tasks
- **Sub-millisecond** state transition latency (0.00-1.97ms range)

### Benchmark Results (Latest)
| Operation Type | Avg Latency | Throughput | Use Case |
|---|---|---|---|
| Agent State Transitions | 0.00ms | 567,526 ops/s | Basic workflow steps |
| Data Processing | 0.04ms | 27,974 ops/s | ETL operations |
| Resource Management | 0.01ms | 104,719 ops/s | Memory/CPU allocation |
| Async Coordination | 1.23ms | 811 ops/s | Multi-agent workflows |
| CPU-Intensive Tasks | 0.91ms | 1,100 ops/s | ML training steps |

*Benchmarks run on: Linux WSL2, 16 cores, 3.68GB RAM, Python 3.12*

[View detailed benchmarks →](./benchmarks/)

---

## 📜 License

PuffinFlow is released under the [MIT License](LICENSE). Free for commercial and personal use.

---

<div align="center">

**Ready to build something amazing?**

[Get Started →](https://puffinflow.readthedocs.io/en/latest/guides/quickstart.html) | [View Examples →](./examples/) | [Join Community →](https://github.com/m-ahmed-elbeskeri/puffinflow/discussions)

</div>
