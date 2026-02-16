Contributing to PuffinFlow
==========================

We welcome contributions to PuffinFlow! This guide will help you get started with contributing to the project.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. **Fork and Clone the Repository**

.. code-block:: bash

   git clone https://github.com/yourusername/puffinflow.git
   cd puffinflow

2. **Set Up Development Environment**

.. code-block:: bash

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install

3. **Verify Installation**

.. code-block:: bash

   # Run tests
   pytest

   # Run linting
   ruff check src/ tests/
   black --check src/ tests/

   # Run type checking
   mypy src/

Development Workflow
~~~~~~~~~~~~~~~~~~~~

1. **Create a Feature Branch**

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. **Make Your Changes**

   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Tests and Checks**

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=src/puffinflow --cov-report=html

   # Run linting and formatting
   ruff check src/ tests/
   black src/ tests/

   # Run type checking
   mypy src/

4. **Commit Your Changes**

.. code-block:: bash

   git add .
   git commit -m "feat: add new feature description"

5. **Push and Create Pull Request**

.. code-block:: bash

   git push origin feature/your-feature-name

Types of Contributions
----------------------

Bug Fixes
~~~~~~~~~~

**Finding Bugs:**
- Check existing issues on GitHub
- Test edge cases and error conditions
- Review code for potential issues

**Reporting Bugs:**
- Use the bug report template
- Include minimal reproduction example
- Provide system information and versions
- Include relevant logs and error messages

**Fixing Bugs:**
- Reference the issue number in your PR
- Add regression tests
- Update documentation if needed

New Features
~~~~~~~~~~~~

**Feature Requests:**
- Open an issue to discuss the feature first
- Explain the use case and benefits
- Consider backward compatibility
- Propose API design

**Implementing Features:**
- Follow existing code patterns
- Add comprehensive tests
- Update documentation
- Consider performance implications

Documentation
~~~~~~~~~~~~~

**Types of Documentation:**
- API documentation (docstrings)
- User guides and tutorials
- Examples and recipes
- Architecture documentation

**Documentation Standards:**
- Use clear, concise language
- Include code examples
- Test all code examples
- Follow reStructuredText format

Testing
~~~~~~~

**Test Types:**
- Unit tests for individual components
- Integration tests for workflows
- Performance tests for benchmarks
- End-to-end tests for complete scenarios

**Test Guidelines:**
- Aim for high test coverage (>90%)
- Test both success and failure cases
- Use descriptive test names
- Mock external dependencies

Code Style Guidelines
---------------------

Python Style
~~~~~~~~~~~~

We follow PEP 8 with some modifications:

.. code-block:: python

   # Good: Clear, descriptive names
   class DataProcessingAgent(Agent):
       async def process_user_data(self, ctx: Context) -> None:
           user_records = await self.fetch_user_records(ctx.user_ids)
           processed_data = self.transform_records(user_records)
           ctx.processed_data = processed_data

   # Bad: Unclear names
   class DPA(Agent):
       async def proc(self, ctx):
           data = await self.fetch(ctx.ids)
           result = self.transform(data)
           ctx.result = result

**Formatting:**
- Line length: 88 characters (Black default)
- Use double quotes for strings
- Use trailing commas in multi-line structures
- Sort imports with isort

**Type Hints:**
- Use type hints for all public APIs
- Use `typing` module for complex types
- Document return types for async functions

.. code-block:: python

   from typing import List, Dict, Optional, Union
   from puffinflow import Agent, Context

   class TypedAgent(Agent):
       async def process_items(
           self,
           ctx: Context,
           items: List[Dict[str, Union[str, int]]]
       ) -> Optional[List[str]]:
           """Process items and return results."""
           # Implementation here
           pass

Documentation Style
~~~~~~~~~~~~~~~~~~~

**Docstring Format:**
Use Google-style docstrings:

.. code-block:: python

   async def process_data(self, ctx: Context, batch_size: int = 100) -> None:
       """Process data in batches.

       Args:
           ctx: The execution context containing input data.
           batch_size: Number of items to process in each batch.

       Raises:
           ValueError: If batch_size is less than 1.
           ProcessingError: If data processing fails.

       Example:
           >>> agent = DataProcessor()
           >>> ctx = Context({'data': [1, 2, 3, 4, 5]})
           >>> await agent.process_data(ctx, batch_size=2)
       """

**Code Comments:**
- Explain why, not what
- Use comments for complex algorithms
- Document assumptions and constraints

.. code-block:: python

   # Use exponential backoff to avoid overwhelming the API
   # during temporary failures
   delay = base_delay * (2 ** attempt)
   await asyncio.sleep(delay)

Testing Guidelines
------------------

Test Structure
~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from puffinflow import Agent, Context
   from puffinflow.testing import MockAgent, create_test_context

   class TestDataProcessor:
       """Test suite for DataProcessor agent."""

       @pytest.fixture
       def agent(self):
           """Create agent instance for testing."""
           return DataProcessor()

       @pytest.fixture
       def sample_context(self):
           """Create sample context for testing."""
           return create_test_context({
               'input_data': [1, 2, 3, 4, 5],
               'batch_size': 2
           })

       @pytest.mark.asyncio
       async def test_successful_processing(self, agent, sample_context):
           """Test successful data processing."""
           result = await agent.run(sample_context)

           assert result.status == 'completed'
           assert len(result.context.processed_data) == 5
           assert all(x > 0 for x in result.context.processed_data)

       @pytest.mark.asyncio
       async def test_empty_input_handling(self, agent):
           """Test handling of empty input data."""
           ctx = create_test_context({'input_data': []})
           result = await agent.run(ctx)

           assert result.status == 'completed'
           assert result.context.processed_data == []

       @pytest.mark.asyncio
       async def test_invalid_batch_size(self, agent, sample_context):
           """Test error handling for invalid batch size."""
           sample_context.batch_size = 0

           with pytest.raises(ValueError, match="batch_size must be greater than 0"):
               await agent.run(sample_context)

Test Utilities
~~~~~~~~~~~~~~

Use provided test utilities:

.. code-block:: python

   from puffinflow.testing import (
       MockAgent,
       create_test_context,
       assert_agent_completed,
       assert_context_contains
   )

   @pytest.mark.asyncio
   async def test_agent_coordination():
       """Test agent coordination with mocks."""
       # Create mock agents
       mock_fetcher = MockAgent(return_data={'data': [1, 2, 3]})
       mock_processor = MockAgent(return_data={'processed': [2, 4, 6]})

       # Test coordination
       team = AgentTeam([mock_fetcher, mock_processor])
       result = await team.run()

       # Assertions
       assert_agent_completed(result)
       assert_context_contains(result.context, 'processed')

Performance Testing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pytest
   from puffinflow.testing import benchmark_agent

   @pytest.mark.benchmark
   def test_agent_performance(benchmark):
       """Benchmark agent performance."""
       agent = DataProcessor()
       context = create_test_context({'input_data': list(range(1000))})

       result = benchmark(agent.run, context)

       # Performance assertions
       assert result.execution_time < 1.0  # Should complete in under 1 second
       assert result.memory_usage < 100 * 1024 * 1024  # Under 100MB

Pull Request Guidelines
-----------------------

PR Checklist
~~~~~~~~~~~~

Before submitting a pull request:

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Changelog updated (for significant changes)
- [ ] Type hints added
- [ ] Performance impact considered

PR Description Template
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

   ## Description
   Brief description of changes and motivation.

   ## Type of Change
   - [ ] Bug fix (non-breaking change that fixes an issue)
   - [ ] New feature (non-breaking change that adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update

   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests added/updated
   - [ ] Manual testing performed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Tests pass locally

Review Process
~~~~~~~~~~~~~~

1. **Automated Checks**: CI/CD pipeline runs tests and checks
2. **Code Review**: Maintainers review code for quality and design
3. **Testing**: Reviewers may test functionality manually
4. **Approval**: At least one maintainer approval required
5. **Merge**: Maintainer merges after all checks pass

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~~

We are committed to providing a welcoming and inclusive environment. Please read our `Code of Conduct <https://github.com/yourusername/puffinflow/blob/main/CODE_OF_CONDUCT.md>`_.

Communication
~~~~~~~~~~~~~

**Channels:**
- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: General questions and ideas
- Discord: Real-time chat and community support
- Email: security@puffinflow.dev for security issues

**Guidelines:**
- Be respectful and constructive
- Search existing issues before creating new ones
- Provide clear, detailed information
- Follow up on your contributions

Recognition
~~~~~~~~~~~

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Documentation acknowledgments
- Community highlights

Getting Help
------------

**For Contributors:**
- Read existing code and tests for examples
- Ask questions in GitHub Discussions
- Join our Discord community
- Attend community office hours

**For Maintainers:**
- Review contribution guidelines regularly
- Provide constructive feedback
- Help onboard new contributors
- Maintain project standards

Release Process
---------------

For Maintainers
~~~~~~~~~~~~~~~

1. **Version Planning**: Plan features for next release
2. **Feature Freeze**: Stop accepting new features
3. **Testing**: Comprehensive testing of release candidate
4. **Documentation**: Update docs and changelog
5. **Release**: Tag and publish release
6. **Announcement**: Announce release to community

Versioning
~~~~~~~~~~

We follow Semantic Versioning:
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Thank You
---------

Thank you for contributing to PuffinFlow! Your contributions help make workflow orchestration better for everyone.

**Special Thanks:**
- All contributors who have submitted code, documentation, and bug reports
- Community members who provide support and feedback
- Organizations that sponsor development time

For questions about contributing, please reach out to the maintainers or join our community discussions.
