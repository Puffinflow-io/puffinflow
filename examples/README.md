# PuffinFlow Examples

This directory contains comprehensive examples demonstrating the capabilities of the PuffinFlow workflow orchestration framework.

## Examples Overview

### 1. Basic Agent (`basic_agent.py`)
Demonstrates fundamental PuffinFlow concepts:
- Creating simple agents with state decorators
- Working with context and agent variables
- Basic agent execution patterns
- Error handling and state transitions

**Key Features:**
- `SimpleAgent`: Basic workflow with initialization, setup, processing, and finalization
- `DataProcessor`: Data validation and processing with error handling
- Resource-aware state decorators (`@cpu_intensive`, `@memory_intensive`)

### 2. Coordination Examples (`coordination_examples.py`)
Shows multi-agent coordination patterns:
- Parallel and sequential agent execution
- Agent teams and collaboration
- Message passing between agents
- Event-driven coordination

**Key Features:**
- `DataCollector`, `DataProcessor`, `DataAggregator`: Pipeline processing
- Parallel vs sequential execution comparisons
- Team-based coordination with `AgentTeam`
- Mixed coordination patterns

### 3. Resource Management (`resource_management.py`)
Demonstrates resource allocation and management:
- Resource pools and allocation strategies
- Resource requirements and constraints
- Quota management
- Resource-aware agent execution

**Key Features:**
- `ResourceAwareAgent`: Adapts behavior based on workload size
- `DatabaseAgent`, `MLTrainingAgent`: Specialized resource patterns
- Resource pool management and allocation strategies
- Concurrent resource usage patterns

### 4. Reliability Patterns (`reliability_patterns.py`)
Shows fault tolerance and reliability patterns:
- Circuit breakers for external service calls
- Bulkhead isolation patterns
- Resource leak detection
- Retry mechanisms and error handling

**Key Features:**
- `ExternalServiceAgent`: Circuit breaker protection for API calls
- `DatabaseAgent`: Bulkhead isolation for database operations
- `ResourceIntensiveAgent`: Resource leak detection and cleanup
- Comprehensive fault tolerance testing

### 5. Observability Demo (`observability_demo.py`)
Demonstrates monitoring and observability:
- Agent monitoring and metrics collection
- Distributed tracing
- Performance monitoring
- Custom observability configurations

**Key Features:**
- `MonitoredAgent`: Comprehensive performance monitoring
- `TracingAgent`: Distributed tracing with span management
- `AlertingAgent`: Health monitoring and alerting
- Performance impact analysis

### 6. Advanced Workflows (`advanced_workflows.py`)
Shows complex workflow patterns:
- Multi-stage workflow orchestration
- Conditional execution and branching
- Dynamic workflow generation
- Error recovery and compensation

**Key Features:**
- `WorkflowOrchestrator`: Complex multi-stage workflow management
- `ConditionalWorkflowAgent`: Dynamic execution path selection
- `DynamicWorkflowGenerator`: Runtime workflow generation
- Comprehensive workflow testing

## Running the Examples

### Prerequisites
Ensure you have PuffinFlow installed and properly configured:

```bash
# Install dependencies
pip install -e .

# Verify installation
python -c "import puffinflow; print(f'PuffinFlow {puffinflow.__version__} ready')"
```

### Running Individual Examples

Each example can be run independently:

```bash
# Basic agent examples
python examples/basic_agent.py

# Coordination examples
python examples/coordination_examples.py

# Resource management examples
python examples/resource_management.py

# Reliability patterns
python examples/reliability_patterns.py

# Observability demo
python examples/observability_demo.py

# Advanced workflows
python examples/advanced_workflows.py
```

### Running All Examples

Use the test runner to execute all examples:

```bash
python examples/run_all_examples.py
```

## Example Output

Each example provides detailed console output showing:
- Agent execution progress
- Performance metrics
- Resource usage
- Error handling
- Results and recommendations

### Sample Output Format
```
PuffinFlow Basic Agent Examples
==================================================
=== Running Simple Agent Example ===
Agent simple-agent initialized
Agent simple-agent setup complete
Agent simple-agent processed data: 333283335000
Agent simple-agent finalized
Agent completed with status: completed
Final state: None
Outputs: {'status': 'initialized', 'setup_complete': True, ...}
```

## Key Concepts Demonstrated

### 1. Agent Lifecycle
- State-based execution model
- Context management
- Variable persistence
- Error handling

### 2. Resource Management
- CPU, memory, and I/O resource allocation
- Resource pools and quotas
- Dynamic resource scaling
- Resource leak detection

### 3. Coordination Patterns
- Parallel execution
- Sequential pipelines
- Team-based coordination
- Message passing

### 4. Reliability Patterns
- Circuit breakers
- Bulkhead isolation
- Retry mechanisms
- Fallback strategies

### 5. Observability
- Performance monitoring
- Distributed tracing
- Metrics collection
- Health monitoring

### 6. Advanced Features
- Dynamic workflow generation
- Conditional execution
- Multi-stage orchestration
- Error recovery

## Customization

Each example can be customized by modifying:
- Agent configurations
- Resource requirements
- Workflow parameters
- Monitoring settings

### Example Customization
```python
# Customize resource requirements
@state(cpu=4.0, memory=2048.0, priority="high")
async def custom_processing_state(self, context):
    # Your custom logic here
    pass

# Customize workflow behavior
agent = CustomAgent("my-agent")
agent.set_variable("batch_size", 5000)
agent.set_variable("quality_threshold", 0.95)
```

## Best Practices Demonstrated

1. **State Design**: Clear, focused states with single responsibilities
2. **Resource Awareness**: Appropriate resource allocation for different workloads
3. **Error Handling**: Comprehensive error handling and recovery
4. **Monitoring**: Built-in observability and performance tracking
5. **Scalability**: Patterns that scale from single agents to complex workflows
6. **Maintainability**: Clean, documented code with clear separation of concerns

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PuffinFlow is properly installed
2. **Resource Constraints**: Adjust resource requirements if running on limited hardware
3. **Timeout Issues**: Increase timeout values for slower systems
4. **Memory Issues**: Reduce batch sizes or concurrent agents

### Debug Mode

Enable debug output by setting environment variables:
```bash
export PUFFINFLOW_LOG_LEVEL=DEBUG
export PUFFINFLOW_TRACE_ENABLED=true
python examples/basic_agent.py
```

## Contributing

To add new examples:
1. Follow the existing pattern and structure
2. Include comprehensive documentation
3. Add error handling and logging
4. Update this README with the new example
5. Test thoroughly before submitting

## Support

For questions or issues with the examples:
- Check the main PuffinFlow documentation
- Review the source code comments
- Run examples with debug logging enabled
- Create an issue with detailed error information
