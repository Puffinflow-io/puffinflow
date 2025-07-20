export const resourceManagementMarkdown = `# Resource Management

Puffinflow provides sophisticated resource management to ensure optimal system utilization, prevent resource exhaustion, and maintain fair allocation across workflows. This comprehensive guide covers resource allocation strategies, quota management, priority systems, and coordination mechanisms essential for building production-ready AI workflows.

## Resource Management Philosophy

Effective resource management in Puffinflow is built on these core principles:

- **Predictable Performance**: Resource constraints ensure consistent execution times under varying load
- **Fair Allocation**: No single workflow can monopolize system resources
- **Graceful Degradation**: System maintains responsiveness even under high resource pressure
- **Cost Optimization**: Intelligent resource allocation minimizes waste and cloud costs
- **Observability**: Comprehensive metrics for resource usage monitoring and optimization

## Understanding Resource Types

Puffinflow manages several types of resources, each with different characteristics and constraints:

### Core System Resources

| Resource Type | Description | Unit | Typical Constraints |
|---------------|-------------|------|-------------------|
| **CPU** | Processing power | Cores (float) | 0.1 - 16.0 cores per state |
| **Memory** | RAM allocation | MB (int) | 64 - 32768 MB per state |
| **GPU** | Graphics processing | Units (float) | 0.0 - 8.0 GPU units |
| **Disk** | Storage space | MB (int) | 0 - 10240 MB temporary storage |
| **Network** | Bandwidth allocation | Mbps (float) | 1.0 - 1000.0 Mbps |

### External Resource Quotas

| Resource Type | Description | Constraints | Use Cases |
|---------------|-------------|-------------|-----------|
| **API Quotas** | External service limits | Requests/second | OpenAI, AWS, GCP APIs |
| **Database Connections** | Connection pool limits | Concurrent connections | PostgreSQL, MongoDB |
| **File Handles** | Open file descriptors | File count | Large file processing |
| **Custom Resources** | Domain-specific limits | User-defined | License seats, custom quotas |

## Quick Reference

### Resource Configuration Syntax

\`\`\`python
from puffinflow import state, Priority

@state(
    # Core resources
    cpu=2.0,                    # CPU cores required
    memory=1024,                # Memory in MB
    gpu=1.0,                    # GPU units required
    disk=500,                   # Temporary disk space in MB
    
    # Execution constraints
    timeout=60.0,               # Maximum execution time
    max_retries=3,              # Retry attempts on failure
    priority=Priority.HIGH,     # Execution priority
    
    # Rate limiting
    rate_limit=10.0,            # Max calls per second
    burst_limit=20,             # Burst capacity
    
    # Coordination
    semaphore_size=5,           # Concurrent access limit
    mutex=True,                 # Exclusive access required
    
    # Monitoring
    monitor_resources=True,     # Enable resource monitoring
    metric_tags=["service:ml"]  # Custom metric tags
)
async def example_state(context):
    pass
\`\`\`

### Common Resource Patterns

\`\`\`python
# Quick validation (micro workload)
@state(cpu=0.25, memory=128, timeout=5.0)
async def quick_validation(context): pass

# API integration (I/O bound)
@state(cpu=0.5, memory=256, rate_limit=10.0, timeout=30.0)
async def api_integration(context): pass

# Data processing (CPU bound)
@state(cpu=4.0, memory=2048, timeout=300.0)
async def data_processing(context): pass

# ML inference (GPU accelerated)
@state(cpu=2.0, memory=4096, gpu=1.0, timeout=120.0)
async def ml_inference(context): pass

# Critical operation (high priority)
@state(cpu=2.0, memory=1024, priority=Priority.CRITICAL, max_retries=5)
async def critical_operation(context): pass
\`\`\`

Resource management in PuffinFlow enables you to build robust, scalable workflows that efficiently utilize system resources while maintaining predictable performance under varying load conditions.`;