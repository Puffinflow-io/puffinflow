# 🐧 Puffinflow

[![PyPI version](https://badge.fury.io/py/puffinflow.svg)](https://badge.fury.io/py/puffinflow)
[![Python versions](https://img.shields.io/pypi/pyversions/puffinflow.svg)](https://pypi.org/project/puffinflow/)
[![CI](https://github.com/yourusername/puffinflow/workflows/CI/badge.svg)](https://github.com/yourusername/puffinflow/actions)
[![Coverage](https://codecov.io/gh/yourusername/puffinflow/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/puffinflow)
[![Documentation](https://readthedocs.org/projects/puffinflow/badge/?version=latest)](https://puffinflow.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![image](https://github.com/user-attachments/assets/eb167fd1-3cb1-43c3-8a48-f2a6813e7629)

A powerful Python workflow orchestration framework with advanced resource management, state persistence, and async execution.

## ✨ Features

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

## 🚀 Quick Start

### Installation

```bash
pip install puffinflow


puffinflow/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── release.yml
│   │   ├── docs.yml
│   │   └── security.yml
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   └── question.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── dependabot.yml
├── docs/
│   ├── source/
│   │   ├── _static/
│   │   ├── _templates/
│   │   ├── api/
│   │   │   ├── agent.rst
│   │   │   ├── resources.rst
│   │   │   └── index.rst
│   │   ├── guides/
│   │   │   ├── quickstart.rst
│   │   │   ├── advanced.rst
│   │   │   ├── examples.rst
│   │   │   └── migration.rst
│   │   ├── conf.py
│   │   ├── index.rst
│   │   └── changelog.rst
│   ├── Makefile
│   └── requirements.txt
├── examples/
│   ├── basic/
│   │   ├── simple_workflow.py
│   │   ├── sequential_states.py
│   │   └── parallel_execution.py
│   ├── advanced/
│   │   ├── resource_management.py
│   │   ├── checkpointing.py
│   │   ├── error_handling.py
│   │   └── custom_allocators.py
│   ├── integrations/
│   │   ├── fastapi_integration.py
│   │   ├── celery_integration.py
│   │   └── kubernetes_deployment.py
│   └── README.md
├── src/
│   └── puffinflow/
│       ├── __init__.py
│       ├── version.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── agent/
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   ├── context.py
│       │   │   ├── state.py
│       │   │   ├── dependencies.py
│       │   │   └── checkpoint.py
│       │   ├── resources/
│       │   │   ├── __init__.py
│       │   │   ├── pool.py
│       │   │   ├── requirements.py
│       │   │   ├── quotas.py
│       │   │   └── allocation.py
│       │   ├── exceptions.py
│       │   └── utils.py
│       ├── integrations/
│       │   ├── __init__.py
│       │   ├── fastapi.py
│       │   ├── celery.py
│       │   ├── kubernetes.py
│       │   └── monitoring.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py
│       │   ├── commands/
│       │   │   ├── __init__.py
│       │   │   ├── run.py
│       │   │   ├── validate.py
│       │   │   └── monitor.py
│       │   └── utils.py
│       └── py.typed
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── agent/
│   │   │   ├── __init__.py
│   │   │   ├── test_base.py
│   │   │   ├── test_context.py
│   │   │   ├── test_state.py
│   │   │   ├── test_dependencies.py
│   │   │   └── test_checkpoint.py
│   │   ├── resources/
│   │   │   ├── __init__.py
│   │   │   ├── test_pool.py
│   │   │   ├── test_requirements.py
│   │   │   ├── test_quotas.py
│   │   │   └── test_allocation.py
│   │   └── test_utils.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_workflows.py
│   │   ├── test_resource_management.py
│   │   ├── test_checkpointing.py
│   │   └── test_error_scenarios.py
│   ├── performance/
│   │   ├── __init__.py
│   │   ├── test_benchmarks.py
│   │   └── test_memory_usage.py
│   └── fixtures/
│       ├── __init__.py
│       ├── sample_workflows.py
│       └── test_data.py
├── benchmarks/
│   ├── __init__.py
│   ├── agent_performance.py
│   ├── resource_allocation.py
│   └── memory_usage.py
├── scripts/
│   ├── build.sh
│   ├── test.sh
│   ├── docs.sh
│   ├── lint.sh
│   └── release.sh
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── MANIFEST.in
├── README.md
├── CHANGELOG.md
├── SECURITY.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── tox.ini
└── Makefile
