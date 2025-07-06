# 🐧 PuffinFlow

[![PyPI version](https://badge.fury.io/py/puffinflow.svg)](https://badge.fury.io/py/puffinflow)
[![Python versions](https://img.shields.io/pypi/pyversions/puffinflow.svg)](https://pypi.org/project/puffinflow/)
[![CI](https://github.com/yourusername/puffinflow/workflows/CI/badge.svg)](https://github.com/yourusername/puffinflow/actions)
[![Coverage](https://codecov.io/gh/yourusername/puffinflow/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/puffinflow)
[![Documentation](https://readthedocs.org/projects/puffinflow/badge/?version=latest)](https://puffinflow.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Build resilient, scalable workflows with confidence.**

PuffinFlow is a modern Python framework that transforms complex workflows into simple, manageable agent-based systems. Whether you're building data pipelines, ML workflows, or distributed microservices, PuffinFlow provides the tools to create reliable, observable, and resource-efficient applications.

---

## Why PuffinFlow?

🎯 **Start Simple, Scale Smart** - Begin with basic agents and seamlessly grow to complex multi-agent orchestrations  
🔒 **Built for Production** - Enterprise-grade reliability patterns, observability, and resource management  
⚡ **Performance First** - Async-native design with intelligent resource allocation and optimization  
🧩 **Framework Agnostic** - Integrates beautifully with FastAPI, Celery, Kubernetes, and your existing stack

## ✨ What Makes PuffinFlow Special

### 🎯 Agent-Based Architecture
Transform complex logic into simple, reusable agents with state-based workflows and automatic dependency resolution.

### 🚀 Production-Ready Performance  
- **Async-first design** with full asyncio support
- **Intelligent resource management** with quotas and allocation strategies
- **Built-in checkpointing** for workflow persistence and recovery
- **Automatic retry mechanisms** with exponential backoff and circuit breakers

### 🔧 Developer Experience
- **Type-safe decorators** for defining agent states and resource requirements
- **Flexible context system** for seamless data flow between states
- **Comprehensive observability** with metrics, tracing, and alerting
- **Zero-config testing** with built-in fixtures and mocking support

### 🌐 Enterprise Integration
- **Framework agnostic** - works with FastAPI, Celery, Django, and more
- **Kubernetes native** with built-in deployment patterns
- **Security first** with automated secret scanning and secure defaults
- **Monitoring ready** with OpenTelemetry and Prometheus integration

---

## 🚀 Quick Start

### Installation

```bash
pip install puffinflow
```

### Your First Agent

Create a simple data processing agent in under 10 lines:

```python
import asyncio
from puffinflow import Agent, state, Context

class DataProcessor(Agent):
    @state(cpu=1.0, memory=512.0)
    async def process_data(self, context: Context):
        # Your business logic here
        data = context.get_input("raw_data", [])
        processed = [x * 2 for x in data]
        context.set_output("processed_data", processed)
        return None  # Workflow complete

# Run the agent
async def main():
    agent = DataProcessor("my-processor")
    result = await agent.run(inputs={"raw_data": [1, 2, 3, 4, 5]})
    print(f"Result: {result.get_output('processed_data')}")

asyncio.run(main())
```

### Multi-State Workflows

Build complex workflows with automatic state transitions:

```python
class MLPipeline(Agent):
    @state(priority="high", cpu=2.0, memory=1024.0)
    async def load_data(self, context: Context):
        # Load and validate data
        context.set_output("dataset_size", 10000)
        return "preprocess"
    
    @state(cpu=4.0, memory=2048.0)
    async def preprocess(self, context: Context):
        # Feature engineering and preprocessing
        await asyncio.sleep(2)  # Simulate processing
        context.set_output("features_ready", True)
        return "train_model"
    
    @state(cpu=8.0, memory=4096.0)
    async def train_model(self, context: Context):
        # Model training
        context.set_output("model_accuracy", 0.94)
        return "evaluate"
    
    @state(cpu=1.0, memory=512.0)
    async def evaluate(self, context: Context):
        accuracy = context.get_output("model_accuracy")
        if accuracy > 0.9:
            context.set_output("status", "model_ready")
        else:
            context.set_output("status", "retrain_needed")
        return None
```

### Team Coordination

Orchestrate multiple agents working together:

```python
from puffinflow import create_team, run_agents_parallel

# Create specialized agents
data_collector = DataCollector("collector")
data_processor = DataProcessor("processor") 
model_trainer = MLPipeline("trainer")

# Run them as a coordinated team
team = create_team([data_collector, data_processor, model_trainer])
results = await team.execute()

# Or run agents in parallel for independent tasks
results = await run_agents_parallel([
    (agent1, {"input": "data1"}),
    (agent2, {"input": "data2"}),
    (agent3, {"input": "data3"})
])
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

## 🔒 Production Ready

### 🛡️ Security First
- **Automated secret scanning** with TruffleHog integration
- **Dependency vulnerability checks** with safety and bandit
- **Secure defaults** for all configuration options
- **Type safety** with comprehensive mypy coverage

### 🚀 Quality Assurance  
- **95%+ test coverage** across unit, integration, and end-to-end tests
- **Multi-version testing** on Python 3.9-3.12
- **Automated linting** with ruff, black, and isort
- **Performance benchmarks** to prevent regressions

### 📦 Release Pipeline
- **Semantic versioning** with automated changelog generation
- **PyPI publishing** with signed releases
- **Documentation deployment** with version management
- **Docker images** for containerized deployments

---

## 📊 Performance

PuffinFlow is built for production workloads:

- **10,000+ agents/second** throughput on standard hardware
- **< 50MB memory** overhead for typical workloads  
- **Sub-millisecond** state transition latency
- **Linear scaling** with coordinated agent teams

[View detailed benchmarks →](./benchmarks/)

---

## 📜 License

PuffinFlow is released under the [MIT License](LICENSE). Free for commercial and personal use.

---

<div align="center">

**Ready to build something amazing?**

[Get Started →](https://puffinflow.readthedocs.io/en/latest/guides/quickstart.html) | [View Examples →](./examples/) | [Join Community →](https://github.com/yourusername/puffinflow/discussions)

</div>
