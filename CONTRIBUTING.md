# Contributing to PuffinFlow

Thank you for your interest in contributing to PuffinFlow! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Security](#security)
- [Release Process](#release-process)

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of workflow orchestration concepts

### Ways to Contribute

- 🐛 **Bug Reports**: Help us identify and fix issues
- ✨ **Feature Requests**: Suggest new functionality
- 💻 **Code Contributions**: Implement bug fixes or new features
- 📚 **Documentation**: Improve guides, examples, and API docs
- 🧪 **Testing**: Add test coverage or improve existing tests
- 🎨 **Examples**: Create tutorials and usage examples

## 🛠 Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/puffinflow.git
cd puffinflow

# Add upstream remote
git remote add upstream https://github.com/puffinflow/puffinflow.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Installation

```bash
# Run tests to ensure everything works
pytest tests/

# Run linting
pre-commit run --all-files

# Check type hints
mypy src/
```

## 🔄 Contributing Process

### 1. Create an Issue

Before starting work, create an issue to discuss:
- Bug reports: Use the bug report template
- Feature requests: Use the feature request template
- Security issues: Use private reporting for sensitive issues

### 2. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git rebase upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Make Changes

- Follow our [code standards](#code-standards)
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

### 4. Test Your Changes

```bash
# Run full test suite
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v

# Check coverage
pytest tests/ --cov=src/puffinflow --cov-report=html

# Run linting and formatting
pre-commit run --all-files

# Type checking
mypy src/
```

### 5. Submit Pull Request

- Use our PR template
- Link to related issues
- Provide clear description of changes
- Include test results and coverage info

### 6. Code Review Process

- Maintainers will review your PR
- Address feedback promptly
- Keep PR updated with main branch
- Be patient - reviews may take time

## 📏 Code Standards

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Automated checks

### Code Guidelines

#### Python Style

```python
# Good: Clear, descriptive names
def calculate_resource_allocation(agent_name: str, requirements: ResourceRequirements) -> Allocation:
    """Calculate resource allocation for an agent."""
    pass

# Good: Type hints
def process_state(state: AgentState) -> AgentResult:
    """Process agent state and return result."""
    pass

# Good: Docstrings
class Agent:
    """A workflow agent that can execute states.

    Args:
        name: Unique identifier for the agent
        context: Optional execution context

    Example:
        >>> agent = Agent("data-processor")
        >>> result = await agent.run()
    """
```

#### Error Handling

```python
# Good: Specific exception handling
try:
    result = resource_pool.acquire(agent_name, requirements)
except ResourceExhaustionError as e:
    logger.error(f"Failed to acquire resources for {agent_name}: {e}")
    raise AgentExecutionError(f"Resource allocation failed: {e}") from e

# Good: Context managers for cleanup
async with resource_pool.acquire(agent_name, requirements) as allocation:
    result = await agent.execute(state)
    return result
```

#### Async/Await

```python
# Good: Proper async/await usage
async def execute_workflow(agents: List[Agent]) -> WorkflowResult:
    """Execute agents concurrently."""
    tasks = [agent.run() for agent in agents]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return WorkflowResult(results)
```

### Directory Structure

```
src/puffinflow/
├── core/                   # Core functionality
│   ├── agent/             # Agent implementation
│   ├── coordination/      # Multi-agent coordination
│   ├── resources/         # Resource management
│   ├── observability/     # Monitoring and metrics
│   └── reliability/       # Reliability patterns
├── cli/                   # Command-line interface
├── examples/              # Usage examples
└── utils/                 # Utility functions

tests/
├── unit/                  # Unit tests
├── integration/           # Integration tests
├── e2e/                   # End-to-end tests
└── performance/           # Performance tests
```

## 🧪 Testing

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Test scalability and performance

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch
from puffinflow.core.agent import Agent

class TestAgent:
    """Test cases for Agent class."""

    def test_agent_creation(self):
        """Test basic agent creation."""
        agent = Agent("test-agent")
        assert agent.name == "test-agent"
        assert agent.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_agent_execution(self):
        """Test agent state execution."""
        agent = Agent("test-agent")

        @agent.state
        async def process_data(context):
            return {"result": "processed"}

        result = await agent.run_state("process_data")
        assert result["result"] == "processed"

    def test_resource_requirements(self):
        """Test agent resource requirements."""
        agent = Agent("test-agent")
        requirements = agent.get_resource_requirements()
        assert isinstance(requirements, ResourceRequirements)
```

### Test Fixtures

```python
# conftest.py
@pytest.fixture
def mock_resource_pool():
    """Mock resource pool for testing."""
    pool = Mock(spec=ResourcePool)
    pool.acquire.return_value = MockAllocation()
    return pool

@pytest.fixture
async def sample_agent():
    """Create a sample agent for testing."""
    agent = Agent("test-agent")

    @agent.state
    async def sample_state(context):
        return {"status": "success"}

    return agent
```

### Performance Testing

```python
import time
import pytest
from memory_profiler import profile

def test_agent_creation_performance():
    """Test agent creation performance."""
    start_time = time.time()

    agents = [Agent(f"agent-{i}") for i in range(1000)]

    end_time = time.time()
    assert end_time - start_time < 1.0  # Should create 1000 agents in <1s

@profile
def test_memory_usage():
    """Test memory usage patterns."""
    agents = []
    for i in range(100):
        agent = Agent(f"agent-{i}")
        agents.append(agent)

    # Memory usage should be reasonable
    assert len(agents) == 100
```

## 📚 Documentation

### Docstring Standards

We follow Google-style docstrings:

```python
def calculate_allocation(requirements: ResourceRequirements,
                        available: ResourceCapacity) -> Optional[Allocation]:
    """Calculate resource allocation based on requirements and availability.

    This function implements the core allocation algorithm, considering
    resource types, priorities, and availability constraints.

    Args:
        requirements: Resource requirements specification
        available: Currently available resource capacity

    Returns:
        Resource allocation if possible, None if insufficient resources

    Raises:
        ValueError: If requirements are invalid
        ResourceError: If allocation algorithm fails

    Example:
        >>> requirements = ResourceRequirements(cpu=2.0, memory=1024)
        >>> capacity = ResourceCapacity(cpu=8.0, memory=4096)
        >>> allocation = calculate_allocation(requirements, capacity)
        >>> print(allocation.cpu)  # 2.0
    """
```

### README Updates

When adding features, update relevant README sections:
- Installation instructions
- Quick start guide
- API reference
- Examples

### Examples

Create comprehensive examples for new features:

```python
# examples/new_feature_example.py
"""
Example demonstrating new feature usage.

This example shows how to use the new coordination feature
to manage multiple agents in a complex workflow.
"""

import asyncio
from puffinflow import Agent, Coordinator

async def main():
    # Create coordinator
    coordinator = Coordinator()

    # Create agents
    agent1 = Agent("data-processor")
    agent2 = Agent("data-analyzer")

    # Add agents to coordinator
    coordinator.add_agent(agent1)
    coordinator.add_agent(agent2)

    # Execute coordinated workflow
    result = await coordinator.execute_workflow()
    print(f"Workflow result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔒 Security

### Security Guidelines

1. **Never commit secrets**: Use environment variables or secure secret management
2. **Validate inputs**: Always validate and sanitize user inputs
3. **Follow least privilege**: Grant minimal necessary permissions
4. **Keep dependencies updated**: Regularly update to patched versions

### Reporting Security Issues

- **Public issues**: Use GitHub issues for general security improvements
- **Sensitive vulnerabilities**: Use GitHub's private vulnerability reporting
- **Critical issues**: Contact maintainers directly

### Security Testing

```python
def test_input_validation():
    """Test that inputs are properly validated."""
    agent = Agent("test-agent")

    # Test injection attempts
    with pytest.raises(ValueError):
        agent.set_name("'; DROP TABLE agents; --")

    # Test oversized inputs
    with pytest.raises(ValueError):
        huge_name = "a" * 10000
        agent.set_name(huge_name)
```

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features (backward compatible)
- Patch: Bug fixes (backward compatible)

### Release Checklist

1. **Update CHANGELOG.md**
2. **Update version numbers**
3. **Run full test suite**
4. **Update documentation**
5. **Create release PR**
6. **Tag release**
7. **Publish to PyPI**

### Changelog Format

```markdown
## [1.2.0] - 2024-01-15

### Added
- New coordination features for multi-agent workflows
- Performance monitoring dashboard
- Resource leak detection

### Changed
- Improved error handling in agent execution
- Updated resource allocation algorithm

### Fixed
- Fixed memory leak in resource pool
- Resolved race condition in coordinator

### Security
- Updated dependencies to patch vulnerabilities
- Improved input validation
```

## 🎉 Recognition

Contributors are recognized in:
- CHANGELOG.md for their contributions
- GitHub contributors page
- Release notes for significant contributions

### Hall of Fame

Special recognition for:
- 🐛 **Bug Hunters**: Find and report critical issues
- ✨ **Feature Champions**: Implement major new functionality
- 📚 **Documentation Heroes**: Significantly improve documentation
- 🛡️ **Security Guardians**: Identify and fix security issues

## ❓ Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community discussions
- **Documentation**: Comprehensive guides and API reference

### Asking Questions

When asking for help:
1. Check existing issues and documentation first
2. Provide minimal reproducible example
3. Include environment details (OS, Python version, etc.)
4. Be specific about the problem and expected behavior

### Mentorship

New contributors can request mentorship:
- Comment on "good first issue" tickets
- Ask questions in discussions
- Request code review guidance

---

**Thank you for contributing to PuffinFlow!**

Your contributions help make workflow orchestration more accessible and powerful for everyone. We appreciate your time, effort, and expertise in making PuffinFlow better.

For questions about contributing, please open a GitHub issue or start a discussion. We're here to help! 🚀
