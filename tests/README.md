# PuffinFlow Test Suite

This directory contains comprehensive tests for the PuffinFlow workflow orchestration framework, organized into three main categories: unit tests, integration tests, and end-to-end (E2E) tests.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
├── integration/             # Integration tests for component interactions
├── e2e/                     # End-to-end tests for complete workflows
└── README.md               # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)
- Test individual components in isolation
- Mock external dependencies
- Fast execution (< 1 second per test)
- High code coverage focus
- Run with: `python run_tests.py unit`

### Integration Tests (`tests/integration/`)
- Test interactions between multiple components
- Validate coordination patterns and resource management
- Medium execution time (1-10 seconds per test)
- Focus on component integration correctness
- Run with: `python run_tests.py integration`

Key integration test areas:
- **Agent Coordination** (`test_agent_coordination.py`):
  - Parallel and sequential agent execution
  - Team-based coordination patterns
  - Context sharing and message passing
  - Dynamic agent creation and conditional coordination
  - Failure handling in coordinated workflows

- **Resource and Reliability** (`test_resource_reliability.py`):
  - Resource pool allocation and contention
  - Circuit breaker integration with agents
  - Bulkhead isolation patterns
  - Combined reliability patterns
  - Observability integration with failures

### End-to-End Tests (`tests/e2e/`)
- Test complete workflows from start to finish
- Simulate real-world usage scenarios
- Longer execution time (10+ seconds per test)
- Focus on user experience and system behavior
- Run with: `python run_tests.py e2e`

Key E2E test areas:
- **Complete Workflows** (`test_complete_workflows.py`):
  - Data processing pipelines (ingestion → transformation → storage → monitoring)
  - Workflow failure recovery mechanisms
  - High-throughput scenarios with dynamic scaling
  - Batch processing workflows

- **Microservices Scenarios** (`test_microservices_scenarios.py`):
  - Microservices orchestration patterns
  - Service dependency chains
  - Service failure recovery
  - Event-driven workflows
  - Event filtering and routing
  - Stream processing scenarios

## Running Tests

### Using the Test Runner Script

The project includes a convenient test runner script (`run_tests.py`) that provides easy access to different test suites:

```bash
# Run unit tests only
python run_tests.py unit

# Run integration tests only
python run_tests.py integration

# Run end-to-end tests only
python run_tests.py e2e

# Run all tests
python run_tests.py all

# Run tests with coverage report
python run_tests.py coverage

# Additional options
python run_tests.py unit --verbose --fail-fast
python run_tests.py integration --parallel 4
```

### Using pytest Directly

You can also run tests directly with pytest:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/ -m unit
pytest tests/integration/ -m integration
pytest tests/e2e/ -m e2e

# Run with coverage
pytest tests/ --cov=src/puffinflow --cov-report=html

# Run specific test files
pytest tests/integration/test_agent_coordination.py
pytest tests/e2e/test_complete_workflows.py

# Run with verbose output
pytest -v tests/integration/

# Stop on first failure
pytest -x tests/unit/
```

## Test Markers

Tests are organized using pytest markers defined in `pytest.ini`:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.asyncio` - Async tests (automatically handled)

## Test Scenarios

### Integration Test Scenarios

#### Agent Coordination Patterns
- **Parallel Execution**: Multiple agents running simultaneously
- **Sequential Execution**: Agents running in dependency order
- **Team Coordination**: Agents working together as teams
- **Context Sharing**: Agents sharing data through context
- **Dynamic Creation**: Creating agents based on runtime conditions
- **Failure Handling**: Graceful handling of agent failures

#### Resource and Reliability Patterns
- **Resource Contention**: Multiple agents competing for limited resources
- **Circuit Breaker Integration**: Preventing cascade failures
- **Bulkhead Isolation**: Isolating failures to prevent system-wide impact
- **Combined Patterns**: Multiple reliability patterns working together
- **Observability Integration**: Monitoring and metrics during failures

### End-to-End Test Scenarios

#### Complete Workflow Patterns
- **Data Processing Pipeline**:
  - Data ingestion from external sources
  - Data transformation and validation
  - Data storage and indexing
  - Monitoring and alerting
- **Failure Recovery**: Automatic recovery from various failure modes
- **Dynamic Scaling**: Scaling agents based on load conditions
- **Batch Processing**: Processing large datasets in batches

#### Microservices and Event-Driven Patterns
- **Microservices Orchestration**: Coordinating multiple services
- **Service Dependencies**: Managing service dependency chains
- **Service Failure Recovery**: Handling individual service failures
- **Event-Driven Workflows**: Processing events through multiple consumers
- **Event Filtering**: Routing events to appropriate consumers
- **Stream Processing**: Processing continuous event streams

## Test Data and Fixtures

### Common Test Patterns

Most tests follow these patterns:

1. **Setup**: Create test agents with specific configurations
2. **Execution**: Run agents individually or in coordination
3. **Verification**: Assert expected outcomes and behaviors
4. **Cleanup**: Automatic cleanup through pytest fixtures

### Test Agent Examples

Tests use realistic agent implementations that simulate real-world scenarios:

```python
class DataIngestionAgent(Agent):
    """Simulates data ingestion from external sources."""

    @state(cpu=1.0, memory=512.0)
    async def ingest_data(self, context: Context):
        # Simulate data ingestion
        await asyncio.sleep(0.2)
        context.set_output("records_ingested", 1000)
        return "validate_data"

class DataTransformationAgent(Agent):
    """Simulates data transformation operations."""

    @state(cpu=2.0, memory=1024.0)
    async def transform_data(self, context: Context):
        # Simulate data transformation
        await asyncio.sleep(0.3)
        context.set_output("records_transformed", 800)
        return None
```

## Performance Expectations

### Test Execution Times

- **Unit Tests**: < 1 second per test, < 30 seconds total
- **Integration Tests**: 1-10 seconds per test, < 2 minutes total
- **E2E Tests**: 10-60 seconds per test, < 10 minutes total

### Resource Usage

Tests are designed to be resource-efficient:
- Memory usage: < 100MB during test execution
- CPU usage: Moderate, with controlled async operations
- No external dependencies required

## Continuous Integration

The test suite is designed for CI/CD environments:

```yaml
# Example GitHub Actions workflow
- name: Run Unit Tests
  run: python run_tests.py unit

- name: Run Integration Tests
  run: python run_tests.py integration

- name: Run E2E Tests
  run: python run_tests.py e2e

- name: Generate Coverage Report
  run: python run_tests.py coverage
```

## Extending the Test Suite

### Adding New Integration Tests

1. Create test file in `tests/integration/`
2. Use `@pytest.mark.integration` marker
3. Focus on component interactions
4. Follow existing patterns for agent creation and coordination

### Adding New E2E Tests

1. Create test file in `tests/e2e/`
2. Use `@pytest.mark.e2e` marker
3. Create complete workflow scenarios
4. Simulate real-world usage patterns

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*` (e.g., `TestAgentCoordination`)
- Test methods: `test_*` (e.g., `test_parallel_agent_execution`)
- Use descriptive names that explain the scenario being tested

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **Async Test Issues**: Use `@pytest.mark.asyncio` for async tests
3. **Resource Conflicts**: Tests are designed to avoid conflicts
4. **Timing Issues**: Tests include appropriate delays and timeouts

### Debug Mode

Run tests with additional debugging:

```bash
# Verbose output with full tracebacks
pytest -v --tb=long tests/integration/

# Stop on first failure with debugging
pytest -x --pdb tests/e2e/

# Run specific test with output
pytest -s tests/integration/test_agent_coordination.py::TestAgentCoordination::test_parallel_execution
```

## Contributing

When contributing new tests:

1. Follow the existing test structure and patterns
2. Include appropriate markers (`@pytest.mark.integration`, `@pytest.mark.e2e`)
3. Write descriptive test names and docstrings
4. Ensure tests are deterministic and don't rely on external services
5. Add documentation for new test scenarios

## Test Coverage Goals

- **Unit Tests**: > 90% line coverage
- **Integration Tests**: > 80% interaction coverage
- **E2E Tests**: > 70% workflow coverage
- **Overall**: > 85% combined coverage

Run coverage reports with:
```bash
python run_tests.py coverage
```

Coverage reports are generated in `htmlcov/` directory.
