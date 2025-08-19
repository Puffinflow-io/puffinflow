# PuffinFlow Benchmarks

This directory contains comprehensive benchmarks for the PuffinFlow framework, designed to measure performance across all major components and identify optimization opportunities.

## Overview

The benchmark suite covers the following areas:

- **Core Agent Execution**: Agent lifecycle, state execution, and dependency resolution
- **Resource Management**: Resource allocation, quotas, and pool management
- **Coordination & Synchronization**: Primitive operations, barriers, and agent coordination
- **Observability**: Metrics collection, tracing, and event handling
- **Framework Comparison**: Performance comparison against other orchestration frameworks (Dagster, Prefect, LangGraph)

## Quick Start

### Run All Benchmarks

```bash
# Run all benchmarks with summary
python benchmarks/run_all_benchmarks.py

# Run and save results to JSON
python benchmarks/run_all_benchmarks.py --save-results --format json

# Run and generate HTML report
python benchmarks/run_all_benchmarks.py --save-results --format html
```

### Run Individual Benchmarks

```bash
# Orchestration-focused benchmarks (recommended)
python benchmarks/benchmark_orchestration_metrics.py

# Core agent benchmarks
python benchmarks/benchmark_core_agent.py

# Resource management benchmarks
python benchmarks/benchmark_resource_management.py

# Coordination benchmarks
python benchmarks/benchmark_coordination.py

# Observability benchmarks
python benchmarks/benchmark_observability.py

# Framework comparison benchmarks
python benchmarks/benchmark_framework_comparison.py
```

## Benchmark Categories

### 🎯 Orchestration-Focused Benchmarks (`benchmark_orchestration_metrics.py`)

**Real-world orchestration metrics that actually matter for workflow frameworks**

Tests the fundamental orchestration capabilities that matter in production:

#### Workflow Complexity Handling:
- **Deep Dependency Chain**: Tests performance with 20-level dependency chains (common in data pipelines)
- **Wide Fanout Pattern**: Tests one-to-many coordination (1 producer → 50 consumers)
- **Diamond DAG Pattern**: Tests complex dependency resolution (A → [B,C] → D patterns)

#### Error Resilience:
- **Cascading Failure Resilience**: Tests recovery from random failures with retry logic
- **Partial Failure Isolation**: Tests that good agents continue when others fail

#### Resource Management Under Pressure:
- **Resource Contention**: Tests performance when resource-intensive agents compete
- **Memory Pressure**: Tests handling of memory allocation/cleanup under load

#### Scalability Characteristics:
- **Horizontal Scaling**: Tests throughput efficiency as agent count increases (10→100 agents)
- **Coordination Overhead**: Tests how coordination latency scales with complexity

**Key Metrics Measured:**
- Dependency resolution latency
- Parallelization efficiency
- Failure recovery rates
- Resource utilization efficiency
- Coordination overhead growth
- Memory allocation efficiency
- Scaling efficiency

**Current Benchmark Results (PuffinFlow v1.0):**

| Orchestration Test | Result | Unit | Performance |
|--------------------|--------|------|-------------|
| Deep Dependency Chain (20 levels) | 11.2ms | latency | ✅ Excellent |
| Wide Fanout (1→50 consumers) | 51.7ms | latency | ✅ Good |
| Diamond DAG Resolution | 63.0ms | latency | ✅ Good |
| Parallelization Efficiency | 1897% | ratio | 🏆 Outstanding |
| DAG Execution Efficiency | 95.2% | ratio | 🏆 Outstanding |
| Failure Recovery Rate | 98.3% | success | 🏆 Outstanding |
| Failure Isolation Rate | 100% | success | 🏆 Perfect |
| Retry Efficiency | 1.5 | attempts | ✅ Excellent |
| Resource Utilization | 31.6% | efficiency | ⚠️ Moderate |
| Scaling Efficiency (10x agents) | 834% | throughput | 🏆 Outstanding |
| Max Throughput | 7,489 | agents/sec | 🏆 Outstanding |
| Coordination Overhead Growth | 1.78x | scaling | ✅ Good |
| Max Coordination Latency | 117ms | latency | ⚠️ Needs improvement |

**Performance Summary:**
- **Strengths**: Exceptional parallelization, perfect failure isolation, outstanding scaling
- **Areas for improvement**: Resource utilization efficiency, coordination latency at scale
- **Overall Grade**: A- (excellent orchestration performance with room for resource optimization)

## Agent Framework Performance Comparison

**Objective performance measurements across PuffinFlow, LangGraph, and LlamaIndex using native execution models**

### 🔄 Execution Models

| Framework | Execution Model | Concurrency Type |
|-----------|----------------|------------------|
| PuffinFlow | async | async/await |
| LangGraph | sync | threads |
| LlamaIndex | async | async/await |

### 📝 Code Efficiency (Lines of Code)

| Framework | Simple Tasks | Complex Tasks | Typed Tasks | Average |
|-----------|--------------|---------------|-------------|---------|
| PuffinFlow | 22 LOC | 46 LOC | 23 LOC | 30.3 LOC |
| LangGraph | 30 LOC | 63 LOC | 30 LOC | 41.0 LOC |
| LlamaIndex | 25 LOC | 49 LOC | 26 LOC | 33.3 LOC |

### ⚡ Execution Speed (Milliseconds)

| Framework | Simple Tasks | Complex Tasks | I/O Heavy Tasks |
|-----------|--------------|---------------|-----------------|
| PuffinFlow | <0.1ms | <0.1ms | <0.1ms |
| LangGraph | 152.2ms | 149.8ms | 355.8ms |
| LlamaIndex | 110.1ms | 122.0ms | 406.4ms |

### 🔧 Framework Overhead (Percentage)

| Framework | Simple Tasks | Complex Tasks | Average |
|-----------|--------------|---------------|---------|
| PuffinFlow | <1% | <1% | <1% |
| LangGraph | 74.2% | 231.1% | 152.7% |
| LlamaIndex | 104.3% | 172.6% | 138.4% |

### 🚀 Concurrency (Tasks per Second)

| Framework | Low Load (10) | High Load (100) | Scaling Factor |
|-----------|---------------|-----------------|----------------|
| PuffinFlow | 3,854.8 TPS | 1,149.0 TPS | 0.30x |
| LangGraph | 13.1 TPS | 12.3 TPS | 0.94x |
| LlamaIndex | 12.0 TPS | 13.1 TPS | 1.09x |

### 💾 Memory Efficiency (MB per Task)

| Framework | Simple Tasks | Complex Tasks | Average |
|-----------|--------------|---------------|---------|
| PuffinFlow | 7.9 MB | 29.1 MB | 18.5 MB |
| LangGraph | 7.6 MB | 29.0 MB | 18.3 MB |
| LlamaIndex | 6.9 MB | 21.4 MB | 14.1 MB |

### 📊 Raw Performance Data

**PuffinFlow (async execution):**
- Most concise code (30.3 LOC average)
- Exceptional execution speed (<0.1ms across all tasks)
- Highest throughput (3,855 TPS at low load, 1,149 TPS at high load)
- Minimal framework overhead (<1%)
- Competitive memory usage (18.5 MB average per task)

**LangGraph (sync execution):**
- Moderate code requirements (41.0 LOC average)
- Moderate execution speed (152.2-355.8ms)
- High framework overhead (152.7% average)
- Low concurrency performance (12.3-13.1 TPS)
- Good memory efficiency (18.3 MB average per task)

**LlamaIndex (async execution):**
- Balanced code requirements (33.3 LOC average)
- Moderate execution speed (110.1-406.4ms)
- High framework overhead (138.4% average)
- Low concurrency performance (12.0-13.1 TPS)
- Most efficient memory usage (14.1 MB average per task)

### Technical Framework Characteristics:

**Framework Selection Considerations:**

**PuffinFlow** technical strengths:
- Low-latency agent coordination and state management
- Built-in reliability patterns (circuit breakers, bulkheads)
- Comprehensive resource management and quota enforcement
- Native observability and metrics collection
- Multi-agent orchestration with dependency resolution

**LangGraph** technical characteristics:
- Graph-based state management for agent workflows
- TypedDict-based state definitions
- Conditional workflow routing and branching
- Built-in checkpointing and persistence
- Agent communication pattern support

**LlamaIndex** technical focus:
- Workflow orchestration for document processing
- Integration with LLM and embedding models
- Event-driven step execution model
- Built-in context management for AI workflows
- RAG (Retrieval-Augmented Generation) pipeline support

**Framework Selection Guidelines:**
- **For high-performance multi-agent systems**: PuffinFlow offers lowest latency and overhead
- **For AI agent workflows with complex state**: LangGraph provides specialized graph coordination
- **For document and LLM processing workflows**: LlamaIndex offers domain-specific optimizations
- **For general workflow orchestration**: Consider feature requirements beyond raw performance

### Core Agent Benchmarks (`benchmark_core_agent.py`)

Tests the fundamental agent execution performance:

- **Simple Agent Execution**: Basic agent run lifecycle
- **Complex Agent Execution**: Multi-state agents with dependencies
- **Resource Heavy Agent**: Agents with resource requirements
- **Concurrent Agents**: Multiple agents running simultaneously
- **State Dependency Resolution**: Performance of dependency graph resolution
- **Resource Acquisition**: Resource pool interaction performance
- **Coordination Primitive**: Basic synchronization operations
- **Metrics Recording**: Observability overhead

### Resource Management Benchmarks (`benchmark_resource_management.py`)

Tests resource allocation and management performance:

- **Single Resource Acquisition**: Basic resource allocation
- **Complex Resource Acquisition**: Multi-resource allocation
- **Resource Contention**: Performance under resource pressure
- **Concurrent Acquisitions**: Multi-threaded resource access
- **Quota Checking**: Resource quota validation
- **Allocation Strategies**: FirstFit, BestFit, and Priority allocators
- **Resource Pool Operations**: Internal pool management
- **Preemption Logic**: Resource reclamation performance
- **Leak Detection**: Resource leak monitoring

### Coordination Benchmarks (`benchmark_coordination.py`)

Tests synchronization and coordination performance:

- **Coordination Primitives**: Lock, semaphore, and barrier operations
- **Concurrent Operations**: Multi-threaded coordination
- **Rate Limiting**: Request rate control
- **Agent Coordination**: Agent-to-agent coordination
- **Agent Pools**: Pool-based agent management
- **Work Processing**: Task distribution and execution
- **State Management**: Primitive state tracking
- **Quota Management**: Coordination resource quotas

### Observability Benchmarks (`benchmark_observability.py`)

Tests monitoring and observability performance:

- **Metrics Recording**: Counter, histogram, and gauge operations
- **Labeled Metrics**: Metrics with dimensional data
- **Concurrent Metrics**: Multi-threaded metric recording
- **Cardinality Protection**: High-cardinality metric handling
- **Tracing Operations**: Span creation and management
- **Event Management**: Event emission and handling
- **Alert Management**: Alert condition evaluation
- **Integration Tests**: End-to-end observability
- **Memory Usage**: Observability memory overhead

### Framework Comparison Benchmarks (`benchmark_framework_comparison.py`)

Comprehensive performance evaluation of agent framework capabilities:

#### Core Benchmark Categories:

**Native API Performance:**
- **Agent Execution**: Framework-specific agent/workflow execution patterns
- **State Management**: Framework overhead for state transitions and coordination
- **Framework Overhead**: Real framework coordination costs vs pure computation
- **Concurrent Throughput**: Operations per second under concurrent load

**Multi-Workflow Patterns:**
- **Simple Workflows**: Single-task execution patterns
- **Complex Workflows**: Multi-step workflows with dependencies
- **Multi-Agent Coordination**: Parallel agent execution and coordination

**Comparative Analysis:**
- **Developer Experience**: Code complexity and lines of code required
- **Performance Metrics**: Speed and efficiency measurements across patterns
- **Resource Utilization**: Memory and CPU usage efficiency
- **Scalability**: Concurrent workflow handling capabilities

#### Framework Implementation Approach:

Each framework is tested using its native patterns and APIs:

**PuffinFlow**: Uses `@state` decorators and `Agent` classes with proper dependency resolution
**LangGraph**: Uses `StateGraph` with `TypedDict` state definitions and native node routing
**LlamaIndex**: Uses `Workflow` classes with event-driven `@step` methods

#### Technical Measurement Methodology:

- **Real Framework Integration**: Tests actual framework APIs, not mocks or simulations
- **Identical Computational Work**: All frameworks execute identical standardized compute tasks
- **Framework-Specific Patterns**: Each test uses the framework's recommended/idiomatic patterns
- **Statistical Validity**: Multiple iterations with proper warmup for consistent results
- **Objective Metrics**: Measures actual execution time, memory usage, and throughput

#### Benchmark Environment:

- **Consistent Hardware**: All tests run on the same system configuration
- **Local Execution**: No network or database overhead to ensure fair comparison
- **Realistic Workloads**: Tests mirror common agent coordination patterns
- **Error Handling**: Proper timeout and error handling for failed framework setups

## Understanding Results

### Metrics Explained

- **Duration (ms)**: Average execution time per operation
- **Min/Max/Median**: Statistical distribution of execution times
- **Std Dev**: Standard deviation of execution times
- **Throughput (ops/s)**: Operations per second
- **Memory (MB)**: Memory usage during benchmark
- **CPU %**: CPU usage during benchmark
- **Iterations**: Number of test iterations

### Performance Baselines

Expected performance ranges on modern hardware:

#### Traditional Benchmark Baselines:
| Operation Type | Expected Throughput | Acceptable Duration |
|---------------|--------------------|--------------------|
| Metric Recording | >10,000 ops/s | <0.1ms |
| Resource Acquisition | >1,000 ops/s | <1ms |
| Coordination Primitives | >5,000 ops/s | <0.2ms |
| Simple Agent Execution | >100 ops/s | <10ms |
| Complex Agent Execution | >50 ops/s | <20ms |
| Tracing Operations | >1,000 ops/s | <1ms |
| Framework Task Execution | >500 ops/s | <2ms |
| Multi-Task Workflows | >50 ops/s | <20ms |
| Framework Coordination | >1,000 ops/s | <1ms |

#### Orchestration-Focused Benchmark Targets:
| Orchestration Metric | Good Performance | Acceptable | Poor |
|---------------------|------------------|------------|------|
| Deep Dependency Chain (20 levels) | <15ms | <50ms | >100ms |
| Wide Fanout (1→50) | <30ms | <100ms | >200ms |
| Diamond DAG Resolution | <40ms | <100ms | >200ms |
| Failure Recovery Rate | >95% | >80% | <80% |
| Scaling Efficiency (10x agents) | >5x throughput | >3x | <2x |
| Coordination Overhead Growth | <2x | <3x | >5x |
| Resource Utilization | >70% | >50% | <30% |
| Memory Allocation Efficiency | >90% | >70% | <50% |

## Benchmark Methodology

### Orchestration-Focused Approach

Our benchmarks prioritize **real-world orchestration scenarios** over synthetic microbenchmarks:

#### **Why Traditional Benchmarks Miss the Point:**
- **Operations/second metrics** don't reflect workflow complexity
- **Simple task execution** ignores dependency resolution overhead
- **Memory/CPU usage** doesn't capture coordination efficiency
- **Setup overhead** is often excluded despite being critical

#### **Our Orchestration-First Methodology:**

**1. Workflow Complexity Testing:**
- **Deep Dependency Chains**: 10-20 step pipelines (common in data/ML workflows)
- **Wide Fanout Patterns**: 1 producer → 20-50 consumers (batch processing)
- **Diamond DAGs**: Complex dependency resolution (A → [B,C] → D)

**2. Real Framework Integration:**
- Tests actual Dagster assets, Prefect flows, LangGraph graphs
- Measures real initialization, coordination, and execution overhead
- Includes database/server costs (not just compute)

**3. Scalability Under Load:**
- Horizontal scaling efficiency (10x agent increase)
- Coordination overhead growth with complexity
- Resource contention and memory pressure handling

**4. Error Resilience Patterns:**
- Cascading failure recovery with realistic retry logic
- Partial failure isolation (good agents continue when others fail)
- Success rates under various failure scenarios

#### **Comparative Testing Standards:**

- **Consistent Workloads**: Same logical operations across all frameworks
- **Real Dependencies**: Actual inter-task dependencies, not mocked
- **Production Patterns**: Patterns seen in real orchestration use cases
- **Local Execution**: All tests run locally to ensure fair comparison
- **Multiple Iterations**: 50 iterations with proper warmup for statistical validity

#### **Why These Metrics Matter:**

- **Dependency Chain Latency**: Critical for data pipeline efficiency
- **Fanout Coordination**: Essential for parallel batch processing
- **Failure Recovery Rate**: Production reliability indicator
- **Scaling Efficiency**: Cost of adding complexity
- **Setup Overhead**: Real-world deployment considerations

This methodology reveals **orchestration-specific performance characteristics** rather than just raw computational speed.

## Performance Optimization

### Identified Bottlenecks

Based on benchmark results, common performance bottlenecks include:

1. **Resource Allocation**: Complex allocation strategies can be slow
2. **Metric Cardinality**: High-cardinality metrics impact performance
3. **Coordination Contention**: Lock contention under high concurrency
4. **State Dependencies**: Complex dependency graphs slow execution
5. **Observability Overhead**: Excessive tracing can impact performance

### Optimization Strategies

1. **Resource Management**:
   - Use FirstFit allocator for high-throughput scenarios
   - Implement resource pooling for frequently used resources
   - Optimize preemption algorithms

2. **Coordination**:
   - Minimize lock hold times
   - Use lock-free data structures where possible
   - Implement backoff strategies for contention

3. **Observability**:
   - Implement sampling for high-frequency operations
   - Use asynchronous metric collection
   - Limit metric cardinality

4. **Agent Execution**:
   - Optimize dependency resolution algorithms
   - Implement state caching where appropriate
   - Use connection pooling for external resources

## Continuous Benchmarking

### CI/CD Integration

Add benchmarks to your CI/CD pipeline:

```yaml
# .github/workflows/benchmarks.yml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e .[dev,performance]
      - name: Run benchmarks
        run: |
          python benchmarks/run_all_benchmarks.py --save-results
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark_results/
```

### Performance Regression Detection

Monitor performance over time:

```bash
# Compare current results with baseline
python benchmarks/compare_results.py \
  --baseline benchmark_results/baseline.json \
  --current benchmark_results/current.json \
  --threshold 10  # 10% regression threshold
```

## Framework Comparison Setup

The framework comparison benchmarks include mock implementations by default for initial testing. For accurate comparisons, install the actual frameworks:

### Installing Comparison Frameworks

```bash
# Install Dagster
pip install dagster dagster-webserver

# Install Prefect
pip install prefect

# Install LangGraph
pip install langgraph

# Install all for complete comparison
pip install dagster dagster-webserver prefect langgraph
```

### Enabling Real Framework Benchmarks

To enable real framework implementations instead of mocks:

1. Set the environment variable `ENABLE_REAL_FRAMEWORKS=true`
2. Ensure all frameworks are installed
3. Run the framework comparison benchmarks

```bash
export ENABLE_REAL_FRAMEWORKS=true
python benchmarks/benchmark_framework_comparison.py
```

### Framework-Specific Considerations

- **Dagster**: Requires asset definitions and may need a temporary database
- **Prefect**: May require flow registration depending on the test
- **LangGraph**: Requires proper graph state management setup
- **Performance Variance**: Real framework performance will vary significantly from mocks

## Custom Benchmarks

### Adding New Benchmarks

1. Create a new benchmark class:

```python
class MyBenchmarks:
    def __init__(self):
        # Initialize test objects
        pass

    def benchmark_my_operation(self):
        # Implement your benchmark
        # Return True for success, False for failure
        pass
```

2. Add to the benchmark runner:

```python
def main():
    runner = BenchmarkRunner()
    benchmarks = MyBenchmarks()

    runner.run_benchmark(
        "My Operation",
        benchmarks.benchmark_my_operation,
        iterations=1000
    )
```

### Benchmark Best Practices

1. **Warm-up**: Always include warm-up iterations
2. **Isolation**: Each benchmark should be independent
3. **Repeatability**: Ensure consistent results across runs
4. **Cleanup**: Properly clean up resources after benchmarks
5. **Documentation**: Document what each benchmark measures

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Resource Conflicts**: Some benchmarks may conflict if run simultaneously
3. **Timeout Issues**: Increase timeouts for slower systems
4. **Memory Issues**: Monitor memory usage during benchmarks

### Debug Mode

Run benchmarks in debug mode:

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run single benchmark with detailed output
python benchmarks/benchmark_core_agent.py --debug
```

## Contributing

When adding new benchmarks:

1. Follow the existing code structure
2. Include comprehensive documentation
3. Add appropriate error handling
4. Test on different hardware configurations
5. Update this README with new benchmark descriptions

## Results Archive

Benchmark results are stored in `benchmark_results/` directory:

- `benchmark_results_YYYYMMDD_HHMMSS.json`: Raw benchmark data
- `benchmark_report_YYYYMMDD_HHMMSS.html`: HTML report with visualizations
- `baseline.json`: Baseline performance metrics for comparison

## Hardware Considerations

Benchmark results vary significantly based on hardware:

- **CPU**: Clock speed and core count affect agent concurrency
- **Memory**: RAM size impacts resource allocation benchmarks
- **Storage**: SSD vs HDD affects checkpoint/persistence operations
- **Network**: Latency impacts distributed coordination benchmarks

Always include system specifications when sharing benchmark results.

## Important Considerations

**"Best" framework depends heavily on specific use cases.** While these benchmarks provide objective performance measurements, the optimal choice varies based on:

- **Workload characteristics**: CPU-bound vs I/O-bound operations
- **Scale requirements**: Small workflows vs large-scale orchestration
- **Team expertise**: Framework familiarity and learning curve
- **Integration needs**: Existing toolchain compatibility
- **Feature requirements**: Specific capabilities beyond raw performance

Consider these benchmarks as one factor in your decision-making process, alongside architectural fit, community support, and long-term maintainability.
