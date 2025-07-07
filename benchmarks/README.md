# PuffinFlow Benchmarks

This directory contains comprehensive benchmarks for the PuffinFlow framework, designed to measure performance across all major components and identify optimization opportunities.

## Overview

The benchmark suite covers the following areas:

- **Core Agent Execution**: Agent lifecycle, state execution, and dependency resolution
- **Resource Management**: Resource allocation, quotas, and pool management
- **Coordination & Synchronization**: Primitive operations, barriers, and agent coordination
- **Observability**: Metrics collection, tracing, and event handling

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
# Core agent benchmarks
python benchmarks/benchmark_core_agent.py

# Resource management benchmarks
python benchmarks/benchmark_resource_management.py

# Coordination benchmarks
python benchmarks/benchmark_coordination.py

# Observability benchmarks
python benchmarks/benchmark_observability.py
```

## Benchmark Categories

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

| Operation Type | Expected Throughput | Acceptable Duration |
|---------------|--------------------|--------------------|
| Metric Recording | >10,000 ops/s | <0.1ms |
| Resource Acquisition | >1,000 ops/s | <1ms |
| Coordination Primitives | >5,000 ops/s | <0.2ms |
| Simple Agent Execution | >100 ops/s | <10ms |
| Complex Agent Execution | >50 ops/s | <20ms |
| Tracing Operations | >1,000 ops/s | <1ms |

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
