🐧 PuffinFlow Documentation
===========================

.. image:: https://badge.fury.io/py/puffinflow.svg
   :target: https://badge.fury.io/py/puffinflow
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/puffinflow.svg
   :target: https://pypi.org/project/puffinflow/
   :alt: Python versions

.. image:: https://github.com/yourusername/puffinflow/workflows/CI/badge.svg
   :target: https://github.com/yourusername/puffinflow/actions
   :alt: CI

.. image:: https://codecov.io/gh/yourusername/puffinflow/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/yourusername/puffinflow
   :alt: Coverage

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

A powerful Python workflow orchestration framework with advanced resource management, state persistence, and async execution.

✨ Features
-----------

**Core Framework:**
- 🚀 **Async-first design** with full asyncio support
- 🎯 **State-based workflow management** with dependency resolution and automatic transitions
- 💾 **Built-in checkpointing** for workflow persistence and recovery
- 🔧 **Advanced resource management** with quotas, allocation strategies, and GPU scheduling
- 🔄 **Automatic retry mechanisms** with exponential backoff and circuit breakers
- 📊 **Priority-based execution** with configurable scheduling and natural language syntax

**AI/ML Capabilities:**
- 🤖 **RAG System Support** - Complete pipelines for document ingestion, embedding generation, and retrieval
- 🧠 **LLM Integration** - Prompt routing, model selection, and fine-tuning workflows
- 📈 **Self-RAG** - Self-improving systems with reflection and iterative enhancement
- 🕸️ **Graph RAG** - Knowledge graph construction and graph-enhanced retrieval
- 🔀 **Prompt Routing** - Intelligent model selection based on query analysis
- 🎯 **Model Fine-tuning** - End-to-end pipelines for training and deployment

**Enterprise Features:**
- 🎛️ **Flexible context system** for state data management with TTL caching
- 🔌 **Easy integration** with FastAPI, Celery, and Kubernetes
- 📈 **Built-in monitoring** and observability with OpenTelemetry support
- 🛡️ **Reliability patterns** - Circuit breakers, bulkheads, and leak detection
- 🤝 **Multi-agent coordination** - Teams, pools, groups with message passing
- 🧪 **Comprehensive testing** with 95%+ code coverage
- 🔒 **Security scanning** with TruffleHog secret detection

🚀 Quick Start
---------------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install puffinflow

   # With observability features
   pip install puffinflow[observability]

   # With all optional dependencies
   pip install puffinflow[all]

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import asyncio
   from puffinflow import Agent, Context, state

   class DataProcessor(Agent):
       @state(profile="io_intensive")
       async def fetch_data(self, ctx: Context) -> None:
           """Fetch data from external source."""
           data = await fetch_external_data()
           ctx.data = data

       @state(depends_on=["fetch_data"], profile="cpu_intensive")
       async def process_data(self, ctx: Context) -> None:
           """Process the fetched data."""
           processed = await process(ctx.data)
           ctx.processed_data = processed

       @state(depends_on=["process_data"], profile="io_intensive")
       async def save_results(self, ctx: Context) -> None:
           """Save processed results."""
           await save_to_database(ctx.processed_data)

   async def main():
       agent = DataProcessor()
       result = await agent.run()
       print(f"Workflow completed: {result.status}")

   if __name__ == "__main__":
       asyncio.run(main())

AI/ML Quick Start
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from puffinflow import Agent, Context, state, AgentTeam
   from puffinflow.core.coordination import RateLimiter

   class RAGAgent(Agent):
       """Simple RAG implementation."""

       def __init__(self):
           super().__init__()
           self.rate_limiter = RateLimiter(max_calls=10, time_window=60)

       @state(profile="gpu_accelerated")
       async def generate_embeddings(self, ctx: Context) -> None:
           """Generate embeddings for documents."""
           # Your embedding logic here
           ctx.embeddings = await generate_embeddings(ctx.documents)

       @state(depends_on=["generate_embeddings"], profile="external_service")
       async def query_llm(self, ctx: Context) -> None:
           """Query LLM with retrieved context."""
           async with self.rate_limiter:
               response = await llm_api_call(ctx.query, ctx.retrieved_docs)
               ctx.response = response

   # Usage
   async def main():
       rag = RAGAgent()
       context = Context({'documents': docs, 'query': 'What is machine learning?'})
       result = await rag.run(context)
       print(result.context.response)

📚 Documentation
----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guides/quickstart
   guides/advanced
   guides/examples
   guides/migration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/agent
   api/coordination
   api/resources
   api/observability
   api/reliability

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
   contributing
   security

🔗 Links
---------

* **Source Code**: https://github.com/yourusername/puffinflow
* **Issue Tracker**: https://github.com/yourusername/puffinflow/issues
* **PyPI Package**: https://pypi.org/project/puffinflow/
* **Documentation**: https://puffinflow.readthedocs.io

📄 License
-----------

This project is licensed under the MIT License - see the `LICENSE <https://github.com/yourusername/puffinflow/blob/main/LICENSE>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
