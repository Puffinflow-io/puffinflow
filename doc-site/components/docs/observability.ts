export const observabilityMarkdown = `# Observability & Monitoring

Puffinflow provides comprehensive observability capabilities to monitor, trace, and debug your AI workflows in production. This guide covers metrics collection, distributed tracing, logging strategies, alerting, and performance monitoring to give you complete visibility into your system's behavior.

## Observability Philosophy

**Observability is more than monitoring** - it's about understanding:
- **What** is happening in your workflows
- **Why** performance degrades or failures occur  
- **Where** bottlenecks and issues originate
- **When** problems start and how they evolve
- **How** to quickly diagnose and resolve issues

## Core Observability Features

| Feature | Purpose | Use Cases |
|---------|---------|-----------|
| **Metrics** | Quantitative measurements | Performance tracking, SLA monitoring |
| **Tracing** | Request flow tracking | Debugging, performance analysis |
| **Logging** | Event and error recording | Troubleshooting, audit trails |
| **Alerting** | Proactive notifications | Incident response, SLA violations |
| **Dashboards** | Visual data representation | Operations monitoring, reporting |

---

## Metrics Collection & Performance Monitoring

### Built-in Metrics Framework

Puffinflow automatically collects comprehensive metrics for all workflow operations:

\`\`\`python
import asyncio
import time
from typing import Dict, List, Any
from puffinflow import Agent
from puffinflow import state
from puffinflow.core.agent.state import Priority
from puffinflow.core.observability.metrics import MetricsCollector, Counter, Histogram, Gauge

# Initialize metrics collector
metrics = MetricsCollector(namespace="puffinflow_app")

# Create custom metrics
request_counter = Counter("workflow_requests_total", "Total workflow requests", ["workflow_type", "status"])
response_time = Histogram("workflow_duration_seconds", "Workflow execution time", ["workflow_name"])
active_workflows = Gauge("active_workflows_current", "Currently active workflows")
queue_depth = Gauge("workflow_queue_depth", "Workflow queue depth", ["priority"])

observability_agent = Agent("observability-demo")

@state(
    priority=Priority.NORMAL,
    timeout=30.0,
    # Enable automatic metrics collection
    metrics_enabled=True,
    custom_metrics=["execution_time", "memory_usage", "api_calls"]
)
async def ai_processing_task(context):
    """AI processing task with comprehensive metrics"""
    print("🤖 AI processing with metrics...")
    
    # Start timing
    start_time = time.time()
    
    # Custom business metrics
    task_id = context.get_variable("task_id", "task_001")
    model_type = context.get_variable("model_type", "gpt-4")
    
    try:
        # Increment request counter
        request_counter.inc({"workflow_type": "ai_processing", "status": "started"})
        active_workflows.inc()
        
        # Simulate AI processing work
        processing_steps = [
            {"name": "input_validation", "duration": 0.1},
            {"name": "model_inference", "duration": 2.5},
            {"name": "output_processing", "duration": 0.3},
            {"name": "result_validation", "duration": 0.1}
        ]
        
        step_metrics = []
        for step in processing_steps:
            step_start = time.time()
            
            print(f"   🔄 {step['name']}...")
            await asyncio.sleep(step['duration'])
            
            step_duration = time.time() - step_start
            step_metrics.append({
                "step": step['name'],
                "duration": step_duration,
                "timestamp": step_start
            })
            
            # Record step-level metrics
            metrics.record_histogram(
                f"ai_step_duration_seconds",
                step_duration,
                {"step_name": step['name'], "model_type": model_type}
            )
        
        # Calculate total processing time
        total_duration = time.time() - start_time
        response_time.observe(total_duration, {"workflow_name": "ai_processing"})
        
        # Record business metrics
        context.set_variable("processing_metrics", {
            "task_id": task_id,
            "model_type": model_type,
            "total_duration": total_duration,
            "steps": step_metrics,
            "tokens_processed": 1500,  # Simulated
            "api_calls_made": 3,
            "cache_hits": 2,
            "cache_misses": 1
        })
        
        # Success metrics
        request_counter.inc({"workflow_type": "ai_processing", "status": "completed"})
        
        print(f"✅ AI processing completed in {total_duration:.2f}s")
        
    except Exception as e:
        # Error metrics
        request_counter.inc({"workflow_type": "ai_processing", "status": "failed"})
        print(f"❌ AI processing failed: {e}")
        raise
    
    finally:
        active_workflows.dec()

@state(
    timeout=15.0,
    metrics_enabled=True
)
async def data_pipeline_task(context):
    """Data pipeline with custom metrics"""
    print("📊 Data pipeline with metrics...")
    
    start_time = time.time()
    
    try:
        # Simulate data pipeline metrics
        pipeline_metrics = {
            "records_processed": 0,
            "records_failed": 0,
            "bytes_processed": 0,
            "api_calls": 0,
            "cache_operations": 0
        }
        
        # Simulate processing batches
        batch_sizes = [100, 150, 120, 80, 200]
        
        for i, batch_size in enumerate(batch_sizes):
            batch_start = time.time()
            
            print(f"   📦 Processing batch {i+1}: {batch_size} records")
            
            # Simulate processing time based on batch size
            processing_time = batch_size * 0.001
            await asyncio.sleep(processing_time)
            
            # Update metrics
            pipeline_metrics["records_processed"] += batch_size
            pipeline_metrics["bytes_processed"] += batch_size * 1024  # 1KB per record
            pipeline_metrics["api_calls"] += 1
            
            # Record batch metrics
            batch_duration = time.time() - batch_start
            metrics.record_histogram(
                "batch_processing_duration_seconds",
                batch_duration,
                {"batch_size_range": get_size_range(batch_size)}
            )
            
            # Record throughput
            throughput = batch_size / batch_duration
            metrics.record_gauge(
                "batch_throughput_records_per_second",
                throughput,
                {"pipeline_stage": "processing"}
            )
        
        total_duration = time.time() - start_time
        
        # Record pipeline completion metrics
        metrics.record_histogram(
            "pipeline_total_duration_seconds",
            total_duration
        )
        
        metrics.record_counter(
            "pipeline_records_total",
            pipeline_metrics["records_processed"],
            {"status": "processed"}
        )
        
        context.set_variable("pipeline_metrics", pipeline_metrics)
        
        print(f"✅ Pipeline completed: {pipeline_metrics['records_processed']} records in {total_duration:.2f}s")
        
    except Exception as e:
        metrics.record_counter("pipeline_errors_total", 1, {"error_type": type(e).__name__})
        raise

def get_size_range(size: int) -> str:
    """Categorize batch sizes for metrics"""
    if size < 50:
        return "small"
    elif size < 150:
        return "medium"
    else:
        return "large"

@state(rate_limit=0.5, timeout=10.0)  # Run every 2 seconds
async def collect_system_metrics(context):
    """Collect system-wide metrics"""
    print("📈 Collecting system metrics...")
    
    # Simulate system metrics collection
    system_metrics = {
        "cpu_usage_percent": 45.2,
        "memory_usage_percent": 67.8,
        "disk_usage_percent": 34.1,
        "network_throughput_mbps": 125.5,
        "active_connections": 42,
        "queue_sizes": {
            "high_priority": 5,
            "normal_priority": 23,
            "low_priority": 8
        }
    }
    
    # Record system metrics
    for metric_name, value in system_metrics.items():
        if metric_name != "queue_sizes":
            metrics.record_gauge(f"system_{metric_name}", value)
    
    # Record queue metrics
    for priority, depth in system_metrics["queue_sizes"].items():
        queue_depth.set(depth, {"priority": priority})
    
    # Calculate and record derived metrics
    total_queue_depth = sum(system_metrics["queue_sizes"].values())
    metrics.record_gauge("total_queue_depth", total_queue_depth)
    
    # Memory utilization efficiency
    memory_efficiency = (system_metrics["memory_usage_percent"] / 100) * (system_metrics["cpu_usage_percent"] / 100)
    metrics.record_gauge("resource_efficiency_ratio", memory_efficiency)
    
    context.set_variable("system_metrics", system_metrics)
    
    print(f"📊 System metrics: CPU {system_metrics['cpu_usage_percent']}%, "
          f"Memory {system_metrics['memory_usage_percent']}%, "
          f"Queue depth {total_queue_depth}")

@state
async def generate_metrics_report(context):
    """Generate comprehensive metrics report"""
    print("📋 Generating metrics report...")
    
    # Gather metrics from all tasks
    processing_metrics = context.get_variable("processing_metrics", {})
    pipeline_metrics = context.get_variable("pipeline_metrics", {})
    system_metrics = context.get_variable("system_metrics", {})
    
    # Generate aggregated report
    metrics_report = {
        "collection_timestamp": time.time(),
        "workflow_performance": {
            "ai_processing": {
                "total_duration": processing_metrics.get("total_duration", 0),
                "tokens_processed": processing_metrics.get("tokens_processed", 0),
                "api_calls": processing_metrics.get("api_calls_made", 0),
                "cache_hit_ratio": calculate_cache_hit_ratio(processing_metrics)
            },
            "data_pipeline": {
                "records_processed": pipeline_metrics.get("records_processed", 0),
                "throughput_records_per_second": calculate_throughput(pipeline_metrics),
                "data_volume_mb": pipeline_metrics.get("bytes_processed", 0) / (1024 * 1024)
            }
        },
        "system_health": {
            "resource_utilization": {
                "cpu": system_metrics.get("cpu_usage_percent", 0),
                "memory": system_metrics.get("memory_usage_percent", 0),
                "disk": system_metrics.get("disk_usage_percent", 0)
            },
            "capacity_metrics": {
                "active_connections": system_metrics.get("active_connections", 0),
                "total_queue_depth": sum(system_metrics.get("queue_sizes", {}).values())
            }
        }
    }
    
    context.set_output("metrics_report", metrics_report)
    
    print("📊 Metrics Report Summary:")
    print(f"   🤖 AI Processing: {processing_metrics.get('total_duration', 0):.2f}s, {processing_metrics.get('tokens_processed', 0)} tokens")
    print(f"   📊 Data Pipeline: {pipeline_metrics.get('records_processed', 0)} records processed")
    print(f"   💻 System: CPU {system_metrics.get('cpu_usage_percent', 0)}%, Memory {system_metrics.get('memory_usage_percent', 0)}%")

def calculate_cache_hit_ratio(metrics: dict) -> float:
    """Calculate cache hit ratio from metrics"""
    hits = metrics.get("cache_hits", 0)
    misses = metrics.get("cache_misses", 0)
    total = hits + misses
    return (hits / total * 100) if total > 0 else 0

def calculate_throughput(metrics: dict) -> float:
    """Calculate processing throughput"""
    records = metrics.get("records_processed", 0)
    # This would come from timing data in real implementation
    estimated_duration = 5.0  # Simulated
    return records / estimated_duration if estimated_duration > 0 else 0
\`\`\`

---

## Distributed Tracing

### Request Flow Tracking

Track requests across multiple agents and services for complete visibility:

\`\`\`python
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class TraceSpan:
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "active"
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict] = field(default_factory=list)

class DistributedTracer:
    def __init__(self):
        self.active_spans = {}
        self.completed_spans = []
    
    def start_span(self, operation_name: str, parent_span_id: Optional[str] = None, trace_id: Optional[str] = None) -> TraceSpan:
        """Start a new tracing span"""
        span_id = str(uuid.uuid4())[:8]
        if not trace_id:
            trace_id = str(uuid.uuid4())[:8]
        
        span = TraceSpan(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time()
        )
        
        self.active_spans[span_id] = span
        return span
    
    def finish_span(self, span_id: str, status: str = "success"):
        """Finish a tracing span"""
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.end_time = time.time()
            span.status = status
            
            self.completed_spans.append(span)
            del self.active_spans[span_id]
    
    def add_span_tag(self, span_id: str, key: str, value: str):
        """Add tag to span"""
        if span_id in self.active_spans:
            self.active_spans[span_id].tags[key] = value
    
    def add_span_log(self, span_id: str, message: str, level: str = "info"):
        """Add log entry to span"""
        if span_id in self.active_spans:
            self.active_spans[span_id].logs.append({
                "timestamp": time.time(),
                "message": message,
                "level": level
            })
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace"""
        spans = []
        
        # Get completed spans
        spans.extend([span for span in self.completed_spans if span.trace_id == trace_id])
        
        # Get active spans
        spans.extend([span for span in self.active_spans.values() if span.trace_id == trace_id])
        
        return sorted(spans, key=lambda s: s.start_time)

# Global tracer instance
tracer = DistributedTracer()

# Distributed tracing agent
tracing_agent = Agent("distributed-tracing-demo")

@state(timeout=20.0)
async def orchestrator_task(context):
    """Main orchestrator with distributed tracing"""
    print("🎭 Orchestrator starting with tracing...")
    
    # Start root span
    trace_id = str(uuid.uuid4())[:8]
    root_span = tracer.start_span("workflow_orchestration", trace_id=trace_id)
    
    try:
        tracer.add_span_tag(root_span.span_id, "workflow_type", "ai_pipeline")
        tracer.add_span_tag(root_span.span_id, "user_id", "user_123")
        tracer.add_span_log(root_span.span_id, "Orchestration started")
        
        # Store trace context
        context.set_variable("trace_id", trace_id)
        context.set_variable("parent_span_id", root_span.span_id)
        
        # Simulate orchestrator work
        await asyncio.sleep(0.5)
        
        # Call dependent services
        await ai_service_call(context)
        await data_service_call(context)
        await notification_service_call(context)
        
        tracer.add_span_log(root_span.span_id, "All services completed successfully")
        tracer.finish_span(root_span.span_id, "success")
        
        print(f"✅ Orchestration completed (trace: {trace_id})")
        
    except Exception as e:
        tracer.add_span_log(root_span.span_id, f"Orchestration failed: {str(e)}", "error")
        tracer.finish_span(root_span.span_id, "error")
        raise

async def ai_service_call(context):
    """AI service with child span"""
    trace_id = context.get_variable("trace_id")
    parent_span_id = context.get_variable("parent_span_id")
    
    # Start child span
    ai_span = tracer.start_span("ai_inference", parent_span_id=parent_span_id, trace_id=trace_id)
    
    try:
        tracer.add_span_tag(ai_span.span_id, "service_name", "ai_service")
        tracer.add_span_tag(ai_span.span_id, "model_type", "gpt-4")
        tracer.add_span_log(ai_span.span_id, "Starting AI inference")
        
        print("   🤖 AI service processing...")
        
        # Simulate AI processing steps
        steps = ["input_validation", "model_loading", "inference", "output_formatting"]
        
        for step in steps:
            step_span = tracer.start_span(f"ai_{step}", parent_span_id=ai_span.span_id, trace_id=trace_id)
            
            tracer.add_span_log(step_span.span_id, f"Executing {step}")
            await asyncio.sleep(0.3)  # Simulate step work
            
            tracer.finish_span(step_span.span_id, "success")
        
        tracer.add_span_log(ai_span.span_id, "AI inference completed")
        tracer.finish_span(ai_span.span_id, "success")
        
        print("   ✅ AI service completed")
        
    except Exception as e:
        tracer.add_span_log(ai_span.span_id, f"AI service failed: {str(e)}", "error")
        tracer.finish_span(ai_span.span_id, "error")
        raise

async def data_service_call(context):
    """Data service with parallel operations"""
    trace_id = context.get_variable("trace_id")
    parent_span_id = context.get_variable("parent_span_id")
    
    # Start data service span
    data_span = tracer.start_span("data_processing", parent_span_id=parent_span_id, trace_id=trace_id)
    
    try:
        tracer.add_span_tag(data_span.span_id, "service_name", "data_service")
        tracer.add_span_log(data_span.span_id, "Starting data processing")
        
        print("   📊 Data service processing...")
        
        # Parallel data operations
        operations = ["fetch_data", "validate_data", "transform_data", "store_data"]
        
        # Simulate parallel execution with tracing
        tasks = []
        for operation in operations:
            task = asyncio.create_task(
                execute_data_operation(operation, data_span.span_id, trace_id)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        tracer.add_span_log(data_span.span_id, "All data operations completed")
        tracer.finish_span(data_span.span_id, "success")
        
        print("   ✅ Data service completed")
        
    except Exception as e:
        tracer.add_span_log(data_span.span_id, f"Data service failed: {str(e)}", "error")
        tracer.finish_span(data_span.span_id, "error")
        raise

async def execute_data_operation(operation: str, parent_span_id: str, trace_id: str):
    """Execute individual data operation with tracing"""
    op_span = tracer.start_span(operation, parent_span_id=parent_span_id, trace_id=trace_id)
    
    try:
        tracer.add_span_tag(op_span.span_id, "operation_type", operation)
        tracer.add_span_log(op_span.span_id, f"Executing {operation}")
        
        # Simulate operation-specific work
        if operation == "fetch_data":
            await asyncio.sleep(0.8)  # Slower operation
            tracer.add_span_tag(op_span.span_id, "records_fetched", "1500")
        elif operation == "validate_data":
            await asyncio.sleep(0.3)
            tracer.add_span_tag(op_span.span_id, "validation_errors", "0")
        elif operation == "transform_data":
            await asyncio.sleep(0.5)
            tracer.add_span_tag(op_span.span_id, "transformations_applied", "5")
        elif operation == "store_data":
            await asyncio.sleep(0.4)
            tracer.add_span_tag(op_span.span_id, "records_stored", "1500")
        
        tracer.finish_span(op_span.span_id, "success")
        
    except Exception as e:
        tracer.add_span_log(op_span.span_id, f"Operation failed: {str(e)}", "error")
        tracer.finish_span(op_span.span_id, "error")
        raise

async def notification_service_call(context):
    """Notification service with external dependencies"""
    trace_id = context.get_variable("trace_id")
    parent_span_id = context.get_variable("parent_span_id")
    
    notif_span = tracer.start_span("notification_service", parent_span_id=parent_span_id, trace_id=trace_id)
    
    try:
        tracer.add_span_tag(notif_span.span_id, "service_name", "notification_service")
        tracer.add_span_log(notif_span.span_id, "Starting notification delivery")
        
        print("   📧 Notification service processing...")
        
        # Simulate external service calls with tracing
        external_calls = [
            {"service": "email_gateway", "duration": 0.6},
            {"service": "sms_gateway", "duration": 0.4},
            {"service": "push_notification", "duration": 0.2}
        ]
        
        for call in external_calls:
            call_span = tracer.start_span(
                f"external_call_{call['service']}", 
                parent_span_id=notif_span.span_id, 
                trace_id=trace_id
            )
            
            tracer.add_span_tag(call_span.span_id, "external_service", call['service'])
            tracer.add_span_tag(call_span.span_id, "call_type", "http_request")
            
            await asyncio.sleep(call['duration'])
            
            tracer.add_span_tag(call_span.span_id, "response_status", "200")
            tracer.finish_span(call_span.span_id, "success")
        
        tracer.add_span_log(notif_span.span_id, "All notifications sent successfully")
        tracer.finish_span(notif_span.span_id, "success")
        
        print("   ✅ Notification service completed")
        
    except Exception as e:
        tracer.add_span_log(notif_span.span_id, f"Notification service failed: {str(e)}", "error")
        tracer.finish_span(notif_span.span_id, "error")
        raise

@state
async def analyze_trace_data(context):
    """Analyze collected trace data"""
    print("🔍 Analyzing trace data...")
    
    trace_id = context.get_variable("trace_id")
    trace_spans = tracer.get_trace(trace_id)
    
    if not trace_spans:
        print("⚠️ No trace data found")
        return
    
    # Analyze trace performance
    trace_analysis = {
        "trace_id": trace_id,
        "total_spans": len(trace_spans),
        "total_duration": 0,
        "service_breakdown": {},
        "critical_path": [],
        "error_count": 0
    }
    
    # Find root span
    root_spans = [span for span in trace_spans if span.parent_span_id is None]
    if root_spans:
        root_span = root_spans[0]
        if root_span.end_time:
            trace_analysis["total_duration"] = root_span.end_time - root_span.start_time
    
    # Analyze by service
    for span in trace_spans:
        service_name = span.tags.get("service_name", "unknown")
        if service_name not in trace_analysis["service_breakdown"]:
            trace_analysis["service_breakdown"][service_name] = {
                "span_count": 0,
                "total_duration": 0,
                "error_count": 0
            }
        
        trace_analysis["service_breakdown"][service_name]["span_count"] += 1
        
        if span.end_time:
            duration = span.end_time - span.start_time
            trace_analysis["service_breakdown"][service_name]["total_duration"] += duration
        
        if span.status == "error":
            trace_analysis["error_count"] += 1
            trace_analysis["service_breakdown"][service_name]["error_count"] += 1
    
    # Find critical path (longest duration spans)
    completed_spans = [span for span in trace_spans if span.end_time]
    if completed_spans:
        completed_spans.sort(key=lambda s: (s.end_time - s.start_time), reverse=True)
        trace_analysis["critical_path"] = [
            {
                "operation": span.operation_name,
                "duration": span.end_time - span.start_time,
                "service": span.tags.get("service_name", "unknown")
            }
            for span in completed_spans[:5]  # Top 5 slowest operations
        ]
    
    context.set_output("trace_analysis", trace_analysis)
    
    print(f"📊 Trace Analysis (ID: {trace_id}):")
    print(f"   Total spans: {trace_analysis['total_spans']}")
    print(f"   Total duration: {trace_analysis['total_duration']:.3f}s")
    print(f"   Errors: {trace_analysis['error_count']}")
    
    if trace_analysis['critical_path']:
        print("   🐌 Slowest operations:")
        for op in trace_analysis['critical_path'][:3]:
            print(f"      {op['operation']}: {op['duration']:.3f}s ({op['service']})")
\`\`\`

---

## Structured Logging & Event Tracking

### Comprehensive Logging Framework

\`\`\`python
import logging
import json
from enum import Enum
from typing import Any, Dict, Optional

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class StructuredLogger:
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = logging.getLogger(component_name)
        
        # Configure structured logging format
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log(self, level: LogLevel, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log structured message with context"""
        log_entry = {
            "component": self.component_name,
            "message": message,
            "context": context or {},
            **kwargs
        }
        
        log_message = json.dumps(log_entry, default=str)
        
        if level == LogLevel.DEBUG:
            self.logger.debug(log_message)
        elif level == LogLevel.INFO:
            self.logger.info(log_message)
        elif level == LogLevel.WARNING:
            self.logger.warning(log_message)
        elif level == LogLevel.ERROR:
            self.logger.error(log_message)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(log_message)
    
    def debug(self, message: str, **kwargs):
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.log(LogLevel.CRITICAL, message, **kwargs)

# Logging demonstration agent
logging_agent = Agent("structured-logging-demo")

# Create component loggers
workflow_logger = StructuredLogger("workflow_engine")
ai_logger = StructuredLogger("ai_service")
data_logger = StructuredLogger("data_pipeline")

@state(timeout=25.0)
async def workflow_with_logging(context):
    """Workflow with comprehensive structured logging"""
    print("📝 Workflow with structured logging...")
    
    # Start workflow logging
    workflow_logger.info(
        "Workflow execution started",
        workflow_id=context.workflow_id,
        user_id="user_123",
        workflow_type="ai_data_pipeline"
    )
    
    try:
        # Log workflow parameters
        workflow_params = {
            "input_data_size": 1500,
            "processing_mode": "batch",
            "model_type": "gpt-4",
            "priority": "normal"
        }
        
        workflow_logger.debug(
            "Workflow parameters configured",
            parameters=workflow_params
        )
        
        # Execute workflow steps with logging
        await ai_processing_with_logging(context)
        await data_processing_with_logging(context)
        
        # Log successful completion
        workflow_logger.info(
            "Workflow execution completed successfully",
            workflow_id=context.workflow_id,
            duration_seconds=3.5,  # Would be calculated
            records_processed=1500,
            api_calls_made=5
        )
        
    except Exception as e:
        # Log error with full context
        workflow_logger.error(
            "Workflow execution failed",
            workflow_id=context.workflow_id,
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=str(e.__traceback__) if hasattr(e, '__traceback__') else None
        )
        raise

async def ai_processing_with_logging(context):
    """AI processing with detailed logging"""
    ai_logger.info("AI processing started")
    
    try:
        # Log model initialization
        ai_logger.debug(
            "Initializing AI model",
            model_type="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )
        
        # Simulate processing with progress logging
        total_items = 100
        for i in range(0, total_items, 20):
            await asyncio.sleep(0.3)
            
            # Log progress
            ai_logger.debug(
                "Processing progress",
                items_processed=i + 20,
                total_items=total_items,
                progress_percentage=((i + 20) / total_items) * 100
            )
        
        # Log performance metrics
        ai_logger.info(
            "AI processing completed",
            total_items=total_items,
            tokens_used=850,
            api_calls=3,
            cache_hits=2,
            processing_duration_seconds=1.5
        )
        
    except Exception as e:
        ai_logger.error(
            "AI processing failed",
            error_type=type(e).__name__,
            error_message=str(e),
            items_processed_before_failure=60
        )
        raise

async def data_processing_with_logging(context):
    """Data processing with structured event logging"""
    data_logger.info("Data processing pipeline started")
    
    try:
        # Log data source information
        data_logger.info(
            "Connecting to data source",
            source_type="database",
            connection_string="postgresql://localhost:5432/prod",
            timeout_seconds=30
        )
        
        # Simulate data processing stages
        stages = [
            {"name": "extract", "duration": 0.8, "records": 1500},
            {"name": "validate", "duration": 0.3, "errors": 5},
            {"name": "transform", "duration": 0.6, "transformations": 8},
            {"name": "load", "duration": 0.4, "loaded": 1495}
        ]
        
        for stage in stages:
            stage_start = time.time()
            
            data_logger.info(
                f"Starting {stage['name']} stage",
                stage=stage['name'],
                expected_duration=stage['duration']
            )
            
            await asyncio.sleep(stage['duration'])
            
            stage_actual_duration = time.time() - stage_start
            
            # Log stage completion with metrics
            stage_metrics = {k: v for k, v in stage.items() if k != "duration"}
            
            data_logger.info(
                f"Completed {stage['name']} stage",
                stage=stage['name'],
                duration_seconds=stage_actual_duration,
                metrics=stage_metrics
            )
            
            # Log performance warnings
            if stage_actual_duration > stage['duration'] * 1.2:
                data_logger.warning(
                    f"Stage {stage['name']} exceeded expected duration",
                    stage=stage['name'],
                    expected_duration=stage['duration'],
                    actual_duration=stage_actual_duration,
                    performance_impact="moderate"
                )
        
        # Log final pipeline metrics
        data_logger.info(
            "Data processing pipeline completed",
            total_records_processed=1495,
            total_errors=5,
            error_rate_percentage=0.33,
            pipeline_duration_seconds=2.1
        )
        
    except Exception as e:
        data_logger.error(
            "Data processing pipeline failed",
            error_type=type(e).__name__,
            error_message=str(e),
            pipeline_stage="unknown"
        )
        raise

@state
async def log_analysis_demo(context):
    """Demonstrate log analysis capabilities"""
    print("🔍 Log analysis demonstration...")
    
    # In a real implementation, this would parse actual log files
    # Here we simulate analyzing collected logs
    
    simulated_logs = [
        {"level": "info", "component": "workflow_engine", "message": "Workflow execution started"},
        {"level": "debug", "component": "ai_service", "message": "Processing progress", "progress_percentage": 40},
        {"level": "warning", "component": "data_pipeline", "message": "Stage extract exceeded expected duration"},
        {"level": "info", "component": "ai_service", "message": "AI processing completed"},
        {"level": "error", "component": "external_api", "message": "API rate limit exceeded"},
        {"level": "info", "component": "workflow_engine", "message": "Workflow execution completed"}
    ]
    
    # Analyze log patterns
    log_analysis = {
        "total_logs": len(simulated_logs),
        "by_level": {},
        "by_component": {},
        "warnings_and_errors": [],
        "performance_issues": []
    }
    
    for log_entry in simulated_logs:
        level = log_entry["level"]
        component = log_entry["component"]
        
        # Count by level
        log_analysis["by_level"][level] = log_analysis["by_level"].get(level, 0) + 1
        
        # Count by component
        log_analysis["by_component"][component] = log_analysis["by_component"].get(component, 0) + 1
        
        # Collect warnings and errors
        if level in ["warning", "error", "critical"]:
            log_analysis["warnings_and_errors"].append({
                "level": level,
                "component": component,
                "message": log_entry["message"]
            })
        
        # Identify performance issues
        if "exceeded" in log_entry["message"] or "slow" in log_entry["message"]:
            log_analysis["performance_issues"].append({
                "component": component,
                "issue": log_entry["message"]
            })
    
    context.set_output("log_analysis", log_analysis)
    
    print("📊 Log Analysis Results:")
    print(f"   Total logs: {log_analysis['total_logs']}")
    print(f"   By level: {log_analysis['by_level']}")
    print(f"   Warnings/Errors: {len(log_analysis['warnings_and_errors'])}")
    print(f"   Performance issues: {len(log_analysis['performance_issues'])}")
\`\`\`

---

## Alerting & Notification Systems

### Intelligent Alerting Framework

\`\`\`python
from dataclasses import dataclass
from typing import List, Callable, Dict
from enum import Enum

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"

@dataclass
class Alert:
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    component: str
    timestamp: float
    tags: Dict[str, str]
    metrics: Dict[str, Any]

class AlertManager:
    def __init__(self):
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_history = []
        self.notification_channels = {}
    
    def add_alert_rule(self, name: str, condition: Callable, severity: AlertSeverity, channels: List[AlertChannel]):
        """Add alert rule"""
        self.alert_rules.append({
            "name": name,
            "condition": condition,
            "severity": severity,
            "channels": channels
        })
    
    def evaluate_alerts(self, metrics: Dict[str, Any], context: Dict[str, Any] = None):
        """Evaluate all alert rules against current metrics"""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            try:
                if rule["condition"](metrics, context or {}):
                    alert = Alert(
                        alert_id=f"alert_{int(time.time())}_{rule['name']}",
                        title=f"{rule['name']} Alert",
                        description=f"Alert condition triggered for {rule['name']}",
                        severity=rule["severity"],
                        component=context.get("component", "unknown") if context else "unknown",
                        timestamp=time.time(),
                        tags=context.get("tags", {}) if context else {},
                        metrics=metrics
                    )
                    
                    triggered_alerts.append(alert)
                    self.active_alerts[alert.alert_id] = alert
                    
            except Exception as e:
                print(f"Error evaluating alert rule {rule['name']}: {e}")
        
        return triggered_alerts
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = time.time()
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]

# Global alert manager
alert_manager = AlertManager()

# Define alert conditions
def high_error_rate_condition(metrics: dict, context: dict) -> bool:
    """Alert on high error rate"""
    error_rate = metrics.get("error_rate_percentage", 0)
    return error_rate > 5.0

def slow_response_time_condition(metrics: dict, context: dict) -> bool:
    """Alert on slow response times"""
    avg_response_time = metrics.get("avg_response_time_seconds", 0)
    return avg_response_time > 2.0

def resource_exhaustion_condition(metrics: dict, context: dict) -> bool:
    """Alert on resource exhaustion"""
    cpu_usage = metrics.get("cpu_usage_percent", 0)
    memory_usage = metrics.get("memory_usage_percent", 0)
    return cpu_usage > 90 or memory_usage > 95

def queue_backup_condition(metrics: dict, context: dict) -> bool:
    """Alert on queue backup"""
    total_queue_depth = metrics.get("total_queue_depth", 0)
    return total_queue_depth > 100

# Register alert rules
alert_manager.add_alert_rule(
    "High Error Rate",
    high_error_rate_condition,
    AlertSeverity.HIGH,
    [AlertChannel.EMAIL, AlertChannel.SLACK]
)

alert_manager.add_alert_rule(
    "Slow Response Time",
    slow_response_time_condition,
    AlertSeverity.MEDIUM,
    [AlertChannel.SLACK]
)

alert_manager.add_alert_rule(
    "Resource Exhaustion",
    resource_exhaustion_condition,
    AlertSeverity.CRITICAL,
    [AlertChannel.EMAIL, AlertChannel.SMS, AlertChannel.SLACK]
)

alert_manager.add_alert_rule(
    "Queue Backup",
    queue_backup_condition,
    AlertSeverity.MEDIUM,
    [AlertChannel.SLACK]
)

# Alerting demonstration agent
alerting_agent = Agent("alerting-demo")

@state(rate_limit=0.5, timeout=15.0)  # Check every 2 seconds
async def monitor_and_alert(context):
    """Monitor system metrics and trigger alerts"""
    print("🚨 Monitoring system for alert conditions...")
    
    # Simulate collecting current metrics
    current_metrics = {
        "error_rate_percentage": 7.5,  # High error rate
        "avg_response_time_seconds": 1.2,  # Normal response time
        "cpu_usage_percent": 85,  # High but acceptable CPU
        "memory_usage_percent": 92,  # High memory usage
        "total_queue_depth": 25,  # Normal queue depth
        "active_connections": 150,
        "api_calls_per_minute": 1200
    }
    
    # Alert context
    alert_context = {
        "component": "workflow_engine",
        "environment": "production",
        "tags": {
            "service": "ai_pipeline",
            "version": "1.2.3",
            "datacenter": "us-east-1"
        }
    }
    
    # Evaluate alert conditions
    triggered_alerts = alert_manager.evaluate_alerts(current_metrics, alert_context)
    
    # Process triggered alerts
    if triggered_alerts:
        print(f"🔔 {len(triggered_alerts)} alert(s) triggered:")
        
        for alert in triggered_alerts:
            print(f"   {alert.severity.value.upper()}: {alert.title}")
            print(f"      Component: {alert.component}")
            print(f"      Metrics: {alert.metrics}")
            
            # Simulate sending notifications
            await send_alert_notifications(alert)
    else:
        print("✅ No alert conditions triggered")
    
    # Store monitoring results
    context.set_variable("current_metrics", current_metrics)
    context.set_variable("triggered_alerts", [alert.__dict__ for alert in triggered_alerts])
    context.set_output("active_alert_count", len(alert_manager.active_alerts))

async def send_alert_notifications(alert: Alert):
    """Send alert notifications through configured channels"""
    print(f"📤 Sending {alert.severity.value} alert notifications...")
    
    # Simulate sending to different channels
    channels = ["slack", "email"]  # Would be determined by alert rules
    
    for channel in channels:
        try:
            await asyncio.sleep(0.1)  # Simulate network call
            print(f"   ✅ Sent to {channel}")
            
        except Exception as e:
            print(f"   ❌ Failed to send to {channel}: {e}")

@state
async def alert_dashboard(context):
    """Generate alert dashboard and summary"""
    print("📊 Generating alert dashboard...")
    
    active_alerts = list(alert_manager.active_alerts.values())
    alert_history = alert_manager.alert_history[-10:]  # Last 10 resolved alerts
    
    # Analyze alert patterns
    alert_summary = {
        "active_alerts": len(active_alerts),
        "alerts_by_severity": {},
        "alerts_by_component": {},
        "recent_resolved": len(alert_history),
        "alert_frequency": {},
        "top_alert_sources": {}
    }
    
    # Analyze active alerts
    for alert in active_alerts:
        severity = alert.severity.value
        component = alert.component
        
        alert_summary["alerts_by_severity"][severity] = alert_summary["alerts_by_severity"].get(severity, 0) + 1
        alert_summary["alerts_by_component"][component] = alert_summary["alerts_by_component"].get(component, 0) + 1
    
    # Analyze historical patterns
    all_alerts = active_alerts + alert_history
    for alert in all_alerts:
        # Count alert frequency by title
        title = alert.title
        alert_summary["alert_frequency"][title] = alert_summary["alert_frequency"].get(title, 0) + 1
    
    # Identify top alert sources
    alert_summary["top_alert_sources"] = dict(
        sorted(alert_summary["alert_frequency"].items(), key=lambda x: x[1], reverse=True)[:5]
    )
    
    context.set_output("alert_summary", alert_summary)
    
    print("🎯 Alert Dashboard Summary:")
    print(f"   Active alerts: {alert_summary['active_alerts']}")
    print(f"   By severity: {alert_summary['alerts_by_severity']}")
    print(f"   Recent resolved: {alert_summary['recent_resolved']}")
    
    if alert_summary['top_alert_sources']:
        print("   Top alert sources:")
        for source, count in list(alert_summary['top_alert_sources'].items())[:3]:
            print(f"      {source}: {count} occurrences")
\`\`\`

---

## Performance Profiling & Optimization

### Advanced Performance Analysis

\`\`\`python
import cProfile
import pstats
from typing import Dict, List
import sys
from io import StringIO

class PerformanceProfiler:
    def __init__(self):
        self.profiles = {}
        self.benchmarks = {}
    
    def profile_function(self, func_name: str, func, *args, **kwargs):
        """Profile a function execution"""
        profiler = cProfile.Profile()
        
        result = None
        try:
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            
            # Capture profile stats
            stats_stream = StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            self.profiles[func_name] = {
                "stats_output": stats_stream.getvalue(),
                "total_calls": stats.total_calls,
                "total_time": stats.total_tt,
                "timestamp": time.time()
            }
            
        except Exception as e:
            print(f"Profiling error for {func_name}: {e}")
        
        return result
    
    def get_profile_summary(self, func_name: str) -> Dict:
        """Get profile summary for a function"""
        return self.profiles.get(func_name, {})

# Global profiler
profiler = PerformanceProfiler()

# Performance monitoring agent
performance_agent = Agent("performance-monitoring")

@state(timeout=30.0)
async def cpu_intensive_task_with_profiling(context):
    """CPU-intensive task with performance profiling"""
    print("🔥 CPU-intensive task with profiling...")
    
    def cpu_heavy_computation():
        """Simulate CPU-heavy computation"""
        result = 0
        for i in range(1000000):
            result += i * i
            if i % 100000 == 0:
                # Simulate some string operations
                temp = f"processing_{i}"
                temp = temp.upper().lower()
        return result
    
    # Profile the computation
    start_time = time.time()
    result = profiler.profile_function("cpu_heavy_computation", cpu_heavy_computation)
    total_time = time.time() - start_time
    
    # Get profiling results
    profile_data = profiler.get_profile_summary("cpu_heavy_computation")
    
    performance_metrics = {
        "function_name": "cpu_heavy_computation",
        "result": result,
        "total_execution_time": total_time,
        "profiled_time": profile_data.get("total_time", 0),
        "total_function_calls": profile_data.get("total_calls", 0),
        "performance_overhead": total_time - profile_data.get("total_time", 0)
    }
    
    context.set_variable("cpu_performance", performance_metrics)
    
    print(f"✅ CPU task completed: {result}")
    print(f"   Execution time: {total_time:.3f}s")
    print(f"   Function calls: {profile_data.get('total_calls', 0)}")

@state(timeout=20.0)
async def memory_intensive_task_with_monitoring(context):
    """Memory-intensive task with memory monitoring"""
    print("💾 Memory-intensive task with monitoring...")
    
    memory_usage_snapshots = []
    
    def get_memory_usage():
        """Simulate memory usage measurement"""
        # In real implementation, would use psutil or similar
        import sys
        return sys.getsizeof(locals()) + sys.getsizeof(globals())
    
    # Initial memory snapshot
    initial_memory = get_memory_usage()
    memory_usage_snapshots.append({"stage": "start", "memory_bytes": initial_memory, "timestamp": time.time()})
    
    # Simulate memory-intensive operations
    data_structures = {}
    
    # Stage 1: Create large data structures
    print("   📈 Stage 1: Creating large data structures...")
    large_list = list(range(100000))
    large_dict = {i: f"value_{i}" for i in range(50000)}
    data_structures["large_list"] = large_list
    data_structures["large_dict"] = large_dict
    
    memory_after_stage1 = get_memory_usage()
    memory_usage_snapshots.append({"stage": "after_creation", "memory_bytes": memory_after_stage1, "timestamp": time.time()})
    
    await asyncio.sleep(0.5)
    
    # Stage 2: Process data
    print("   ⚙️ Stage 2: Processing data...")
    processed_data = []
    for i in range(0, len(large_list), 1000):
        chunk = large_list[i:i+1000]
        processed_chunk = [x * 2 for x in chunk]
        processed_data.extend(processed_chunk)
    
    memory_after_stage2 = get_memory_usage()
    memory_usage_snapshots.append({"stage": "after_processing", "memory_bytes": memory_after_stage2, "timestamp": time.time()})
    
    await asyncio.sleep(0.3)
    
    # Stage 3: Cleanup
    print("   🧹 Stage 3: Cleanup...")
    del large_list, large_dict, processed_data
    data_structures.clear()
    
    memory_after_cleanup = get_memory_usage()
    memory_usage_snapshots.append({"stage": "after_cleanup", "memory_bytes": memory_after_cleanup, "timestamp": time.time()})
    
    # Analyze memory usage pattern
    max_memory = max(snapshot["memory_bytes"] for snapshot in memory_usage_snapshots)
    memory_growth = max_memory - initial_memory
    memory_efficiency = (memory_after_cleanup - initial_memory) / max(memory_growth, 1)
    
    memory_analysis = {
        "initial_memory_bytes": initial_memory,
        "peak_memory_bytes": max_memory,
        "final_memory_bytes": memory_after_cleanup,
        "memory_growth_bytes": memory_growth,
        "memory_efficiency_ratio": memory_efficiency,
        "usage_snapshots": memory_usage_snapshots
    }
    
    context.set_variable("memory_analysis", memory_analysis)
    
    print(f"✅ Memory task completed")
    print(f"   Peak memory: {max_memory:,} bytes")
    print(f"   Memory growth: {memory_growth:,} bytes")
    print(f"   Efficiency ratio: {memory_efficiency:.3f}")

@state(timeout=25.0)
async def io_performance_benchmark(context):
    """I/O performance benchmarking"""
    print("📀 I/O performance benchmark...")
    
    io_benchmarks = {}
    
    # File I/O benchmark
    print("   📁 Testing file I/O performance...")
    file_io_start = time.time()
    
    # Simulate file operations
    test_data = "Test data " * 1000  # 9KB of test data
    write_times = []
    read_times = []
    
    for i in range(10):
        # Write test
        write_start = time.time()
        # In real implementation: write to actual file
        await asyncio.sleep(0.01)  # Simulate file write
        write_time = time.time() - write_start
        write_times.append(write_time)
        
        # Read test
        read_start = time.time()
        # In real implementation: read from actual file
        await asyncio.sleep(0.005)  # Simulate file read
        read_time = time.time() - read_start
        read_times.append(read_time)
    
    file_io_total = time.time() - file_io_start
    
    io_benchmarks["file_io"] = {
        "total_time": file_io_total,
        "avg_write_time": sum(write_times) / len(write_times),
        "avg_read_time": sum(read_times) / len(read_times),
        "write_throughput_ops_per_sec": len(write_times) / sum(write_times),
        "read_throughput_ops_per_sec": len(read_times) / sum(read_times)
    }
    
    # Network I/O benchmark
    print("   🌐 Testing network I/O performance...")
    network_io_start = time.time()
    
    response_times = []
    for i in range(5):
        request_start = time.time()
        # Simulate HTTP request
        await asyncio.sleep(0.1 + (i * 0.02))  # Variable latency
        response_time = time.time() - request_start
        response_times.append(response_time)
    
    network_io_total = time.time() - network_io_start
    
    io_benchmarks["network_io"] = {
        "total_time": network_io_total,
        "avg_response_time": sum(response_times) / len(response_times),
        "min_response_time": min(response_times),
        "max_response_time": max(response_times),
        "requests_per_second": len(response_times) / sum(response_times)
    }
    
    context.set_variable("io_benchmarks", io_benchmarks)
    
    print(f"✅ I/O benchmark completed")
    print(f"   File I/O: {io_benchmarks['file_io']['write_throughput_ops_per_sec']:.1f} writes/sec")
    print(f"   Network I/O: {io_benchmarks['network_io']['avg_response_time']*1000:.1f}ms avg response")

@state
async def generate_performance_report(context):
    """Generate comprehensive performance analysis report"""
    print("📊 Generating performance report...")
    
    # Gather performance data
    cpu_performance = context.get_variable("cpu_performance", {})
    memory_analysis = context.get_variable("memory_analysis", {})
    io_benchmarks = context.get_variable("io_benchmarks", {})
    
    # Create comprehensive report
    performance_report = {
        "report_timestamp": time.time(),
        "cpu_analysis": {
            "execution_time_seconds": cpu_performance.get("total_execution_time", 0),
            "function_calls": cpu_performance.get("total_function_calls", 0),
            "profiling_overhead_seconds": cpu_performance.get("performance_overhead", 0),
            "calls_per_second": cpu_performance.get("total_function_calls", 0) / max(cpu_performance.get("total_execution_time", 1), 0.001)
        },
        "memory_analysis": {
            "peak_memory_mb": memory_analysis.get("peak_memory_bytes", 0) / (1024 * 1024),
            "memory_growth_mb": memory_analysis.get("memory_growth_bytes", 0) / (1024 * 1024),
            "efficiency_ratio": memory_analysis.get("memory_efficiency_ratio", 0),
            "memory_leak_indicator": memory_analysis.get("final_memory_bytes", 0) > memory_analysis.get("initial_memory_bytes", 0) * 1.1
        },
        "io_performance": {
            "file_write_ops_per_sec": io_benchmarks.get("file_io", {}).get("write_throughput_ops_per_sec", 0),
            "file_read_ops_per_sec": io_benchmarks.get("file_io", {}).get("read_throughput_ops_per_sec", 0),
            "network_avg_latency_ms": io_benchmarks.get("network_io", {}).get("avg_response_time", 0) * 1000,
            "network_requests_per_sec": io_benchmarks.get("network_io", {}).get("requests_per_second", 0)
        }
    }
    
    # Performance scoring
    scores = calculate_performance_scores(performance_report)
    performance_report["performance_scores"] = scores
    performance_report["overall_score"] = sum(scores.values()) / len(scores)
    
    context.set_output("performance_report", performance_report)
    
    print("🎯 Performance Report Summary:")
    print(f"   Overall Score: {performance_report['overall_score']:.1f}/100")
    print(f"   CPU Score: {scores['cpu_score']:.1f}/100")
    print(f"   Memory Score: {scores['memory_score']:.1f}/100")
    print(f"   I/O Score: {scores['io_score']:.1f}/100")
    
    # Performance recommendations
    recommendations = generate_performance_recommendations(performance_report)
    if recommendations:
        print("💡 Performance Recommendations:")
        for rec in recommendations:
            print(f"   - {rec}")

def calculate_performance_scores(report: dict) -> dict:
    """Calculate performance scores from 0-100"""
    scores = {}
    
    # CPU Score (based on efficiency)
    cpu_calls_per_sec = report["cpu_analysis"]["calls_per_second"]
    cpu_score = min(100, (cpu_calls_per_sec / 100000) * 100)  # Normalize to 100k calls/sec
    scores["cpu_score"] = cpu_score
    
    # Memory Score (based on efficiency and leak detection)
    memory_efficiency = report["memory_analysis"]["efficiency_ratio"]
    memory_leak = report["memory_analysis"]["memory_leak_indicator"]
    memory_score = (1 - abs(memory_efficiency)) * 100
    if memory_leak:
        memory_score *= 0.7  # Penalize for potential memory leaks
    scores["memory_score"] = max(0, memory_score)
    
    # I/O Score (based on throughput and latency)
    file_write_score = min(100, (report["io_performance"]["file_write_ops_per_sec"] / 1000) * 100)
    network_latency_score = max(0, 100 - (report["io_performance"]["network_avg_latency_ms"] / 10))
    io_score = (file_write_score + network_latency_score) / 2
    scores["io_score"] = io_score
    
    return scores

def generate_performance_recommendations(report: dict) -> List[str]:
    """Generate performance optimization recommendations"""
    recommendations = []
    
    scores = report["performance_scores"]
    
    if scores["cpu_score"] < 70:
        recommendations.append("Consider optimizing CPU-intensive operations or algorithm complexity")
    
    if scores["memory_score"] < 70:
        recommendations.append("Review memory usage patterns and implement garbage collection optimization")
    
    if report["memory_analysis"]["memory_leak_indicator"]:
        recommendations.append("Investigate potential memory leaks in data structure cleanup")
    
    if scores["io_score"] < 70:
        recommendations.append("Optimize I/O operations with caching, batching, or async patterns")
    
    if report["io_performance"]["network_avg_latency_ms"] > 200:
        recommendations.append("Consider connection pooling or CDN usage for network operations")
    
    return recommendations
\`\`\`

---

## Best Practices Summary

### Observability Implementation Strategy

1. **Start with Metrics**
   - Begin with built-in metrics collection
   - Add custom business metrics gradually
   - Focus on key performance indicators

2. **Implement Structured Logging**
   - Use consistent log format across components
   - Include relevant context in every log entry
   - Separate operational logs from application logs

3. **Add Distributed Tracing**
   - Implement for multi-service workflows
   - Track request flows across agent boundaries
   - Monitor critical path performance

4. **Configure Smart Alerting**
   - Define clear severity levels
   - Avoid alert fatigue with appropriate thresholds
   - Include actionable information in alerts

5. **Monitor Continuously**
   - Regular performance profiling
   - Resource utilization tracking
   - User experience metrics

### Quick Reference

\`\`\`python
# Enable built-in metrics
@state(metrics_enabled=True, custom_metrics=["execution_time", "memory_usage"])
async def monitored_task(context): pass

# Structured logging
logger = StructuredLogger("component_name")
logger.info("Operation completed", user_id="123", duration=1.5)

# Distributed tracing
span = tracer.start_span("operation_name")
tracer.add_span_tag(span.span_id, "key", "value")
tracer.finish_span(span.span_id, "success")

# Custom metrics
metrics.record_counter("requests_total", 1, {"status": "success"})
metrics.record_histogram("duration_seconds", 1.5)

# Alerting
alert_manager.evaluate_alerts(current_metrics, context)
\`\`\`

Observability transforms your Puffinflow workflows from black boxes into transparent, debuggable, and optimizable systems. Start with basic metrics and logging, then gradually add more sophisticated monitoring as your system grows in complexity.
`.trim();