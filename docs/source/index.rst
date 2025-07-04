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

- 🚀 **Async-first design** with full asyncio support
- 🎯 **State-based workflow management** with dependency resolution
- 💾 **Built-in checkpointing** for workflow persistence and recovery
- 🔧 **Advanced resource management** with quotas and allocation strategies
- 🔄 **Automatic retry mechanisms** with exponential backoff
- 📊 **Priority-based execution** with configurable scheduling
- 🎛️ **Flexible context system** for state data management
- 🔌 **Easy integration** with FastAPI, Celery, and Kubernetes
- 📈 **Built-in monitoring** and observability features
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
       @state
       async def fetch_data(self, ctx: Context) -> None:
           """Fetch data from external source."""
           data = await fetch_external_data()
           ctx.data = data

       @state
       async def process_data(self, ctx: Context) -> None:
           """Process the fetched data."""
           processed = await process(ctx.data)
           ctx.processed_data = processed

       @state
       async def save_results(self, ctx: Context) -> None:
           """Save processed results."""
           await save_to_database(ctx.processed_data)

   async def main():
       agent = DataProcessor()
       result = await agent.run()
       print(f"Workflow completed: {result.status}")

   if __name__ == "__main__":
       asyncio.run(main())

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