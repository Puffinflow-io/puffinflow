export const gettingStartedMarkdown = `# Getting Started with Puffinflow

**Build your first observable data processing workflow in 5 minutes.** This complete example shows how data flows through a Puffinflow workflow with full observability, monitoring, and enterprise-grade features.

---

## The Observable Data Flow Example

We'll build a **customer analytics workflow** with **full observability** that:
1. **Fetches** customer data from an API with distributed tracing
2. **Processes** the data to calculate metrics with performance monitoring
3. **Generates** a business report with structured logging
4. **Sends** notifications to stakeholders with error tracking

Each step includes metrics collection, performance monitoring, and comprehensive logging to demonstrate production-ready workflows.

---

## Step 1: Install Puffinflow

\`\`\`bash
pip install puffinflow
\`\`\`

---

## Step 2: Setup Observability Configuration

**File: observability_config.py**

\`\`\`python
     1→from puffinflow.observability import ObservabilityConfig, MetricsCollector
     2→from puffinflow.observability import TracingProvider, LoggingConfig
     3→from puffinflow.observability import AlertManager, HealthChecker
     4→import logging
     5→
     6→# Configure comprehensive observability
     7→observability_config = ObservabilityConfig(
     8→    # Metrics collection every 5 seconds
     9→    metrics_interval=5.0,
    10→    
    11→    # Distributed tracing with 100% sampling
    12→    tracing_enabled=True,
    13→    sampling_rate=1.0,
    14→    
    15→    # Structured logging configuration
    16→    logging_config=LoggingConfig(
    17→        level=logging.INFO,
    18→        format="json",
    19→        include_context=True,
    20→        include_performance=True
    21→    ),
    22→    
    23→    # Health monitoring
    24→    health_check_interval=10.0,
    25→    
    26→    # Alert thresholds
    27→    alert_config={
    28→        "execution_time_threshold": 60.0,
    29→        "memory_usage_threshold": 0.8,
    30→        "error_rate_threshold": 0.1
    31→    }
    32→)
    33→
    34→# Initialize metrics collector
    35→metrics = MetricsCollector(
    36→    namespace="customer_analytics",
    37→    tags={"environment": "production", "service": "analytics"}
    38→)
    39→
    40→# Setup distributed tracing
    41→tracer = TracingProvider.get_tracer("customer-analytics-workflow")
    42→\`\`\`

---

## Step 3: Create the Observable Workflow

**File: customer_analytics.py**

\`\`\`python
     1→import asyncio
     2→import time
     3→from typing import Dict, Any
     4→from puffinflow import Agent, state, Priority
     5→from puffinflow.observability import observe, metric, trace, log_structured
     6→from observability_config import observability_config, metrics, tracer
     7→
     8→# Create the workflow agent with observability
     9→analytics_agent = Agent(
    10→    name="customer-analytics",
    11→    observability_config=observability_config
    12→)
    13→
    14→@state(
    15→    cpu=1.0,
    16→    memory=512,
    17→    timeout=30.0,
    18→    priority=Priority.NORMAL
    19→)
    20→@observe(metrics=metrics, tracer=tracer)
    21→async def fetch_customer_data(context):
    22→    """Step 1: Fetch customer data from API with observability"""
    23→    
    24→    # Start distributed trace
    25→    with tracer.start_span("fetch_customer_data") as span:
    26→        span.set_attribute("step", "data_fetching")
    27→        span.set_attribute("source", "customer_api")
    28→        
    29→        # Structured logging
    30→        log_structured("info", "Starting customer data fetch", {
    31→            "step": "fetch_customer_data",
    32→            "timestamp": time.time(),
    33→            "trace_id": span.get_span_context().trace_id
    34→        })
    35→        
    36→        print("📊 Fetching customer data from API...")
    37→        
    38→        # Start timing metric
    39→        fetch_timer = metrics.start_timer("api_fetch_duration")
    40→        
    41→        try:
    42→            # Simulate API call delay
    43→            await asyncio.sleep(1.5)
    44→            
    45→            # Mock customer data from API
    46→            customer_data = {
    47→                "total_customers": 2500,
    48→                "active_customers": 1875,
    49→                "new_signups_today": 45,
    50→                "churned_customers": 23,
    51→                "revenue_data": {
    52→                    "total_revenue": 125000,
    53→                    "monthly_recurring": 89000,
    54→                    "one_time_purchases": 36000
    55→                },
    56→                "engagement_metrics": {
    57→                    "daily_active_users": 892,
    58→                    "weekly_active_users": 1456,
    59→                    "average_session_time": 24.5
    60→                }
    61→            }
    62→            
    63→            # Record metrics
    64→            metrics.counter("customers_fetched").increment(customer_data["total_customers"])
    65→            metrics.gauge("active_customers").set(customer_data["active_customers"])
    66→            metrics.histogram("api_response_size").record(len(str(customer_data)))
    67→            
    68→            # Stop timing metric
    69→            fetch_timer.stop()
    70→            
    71→            # Store data in context for next step
    72→            context.set_variable("raw_customer_data", customer_data)
    73→            context.set_variable("fetch_timestamp", time.time())
    74→            
    75→            # Add trace attributes
    76→            span.set_attribute("customers_count", customer_data["total_customers"])
    77→            span.set_attribute("status", "success")
    78→            
    79→            # Success logging
    80→            log_structured("info", "Customer data fetch completed", {
    81→                "step": "fetch_customer_data",
    82→                "customers_count": customer_data["total_customers"],
    83→                "execution_time": fetch_timer.elapsed(),
    84→                "status": "success"
    85→            })
    86→            
    87→            print(f"✅ Fetched data for {customer_data['total_customers']} customers")
    88→            
    89→            # Tell Puffinflow which step to run next
    90→            return "process_metrics"
    91→            
    92→        except Exception as e:
    93→            # Error metrics and logging
    94→            metrics.counter("fetch_errors").increment()
    95→            span.set_attribute("error", True)
    96→            span.set_attribute("error_message", str(e))
    97→            
    98→            log_structured("error", "Customer data fetch failed", {
    99→                "step": "fetch_customer_data",
   100→                "error": str(e),
   101→                "trace_id": span.get_span_context().trace_id
   102→            })
   103→            
   104→            raise
    48→
   105→@state(
   106→    cpu=2.0,
   107→    memory=1024,
   108→    timeout=45.0,
   109→    priority=Priority.HIGH
   110→)
   111→@observe(metrics=metrics, tracer=tracer)
   112→async def process_metrics(context):
   113→    """Step 2: Process data and calculate business metrics with monitoring"""
   114→    
   115→    # Start distributed trace
   116→    with tracer.start_span("process_metrics") as span:
   117→        span.set_attribute("step", "data_processing")
   118→        
   119→        # Structured logging
   120→        log_structured("info", "Starting metrics processing", {
   121→            "step": "process_metrics",
   122→            "timestamp": time.time(),
   123→            "trace_id": span.get_span_context().trace_id
   124→        })
   125→        
   126→        print("🧮 Processing customer metrics...")
   127→        
   128→        # Start performance monitoring
   129→        processing_timer = metrics.start_timer("metrics_processing_duration")
   130→        memory_monitor = metrics.gauge("memory_usage_mb")
   131→        
   132→        try:
   133→            # Get data from previous step
   134→            raw_data = context.get_variable("raw_customer_data")
   135→            
   136→            # Monitor memory usage
   137→            import psutil
   138→            process = psutil.Process()
   139→            memory_monitor.set(process.memory_info().rss / 1024 / 1024)
   140→            
   141→            # Simulate processing time
   142→            await asyncio.sleep(2.0)
   143→            
   144→            # Calculate key business metrics
   145→            processed_metrics = {
   146→                "customer_health": {
   147→                    "total_customers": raw_data["total_customers"],
   148→                    "active_rate": round((raw_data["active_customers"] / raw_data["total_customers"]) * 100, 2),
   149→                    "churn_rate": round((raw_data["churned_customers"] / raw_data["total_customers"]) * 100, 2),
   150→                    "growth_rate": round((raw_data["new_signups_today"] / raw_data["total_customers"]) * 100, 2)
   151→                },
   152→                "revenue_analysis": {
   153→                    "total_revenue": raw_data["revenue_data"]["total_revenue"],
   154→                    "revenue_per_customer": round(raw_data["revenue_data"]["total_revenue"] / raw_data["total_customers"], 2),
   155→                    "mrr_percentage": round((raw_data["revenue_data"]["monthly_recurring"] / raw_data["revenue_data"]["total_revenue"]) * 100, 2),
   156→                    "avg_customer_value": round(raw_data["revenue_data"]["total_revenue"] / raw_data["active_customers"], 2)
   157→                },
   158→                "engagement_insights": {
   159→                    "dau_to_total_ratio": round((raw_data["engagement_metrics"]["daily_active_users"] / raw_data["total_customers"]) * 100, 2),
   160→                    "wau_to_total_ratio": round((raw_data["engagement_metrics"]["weekly_active_users"] / raw_data["total_customers"]) * 100, 2),
   161→                    "avg_session_minutes": raw_data["engagement_metrics"]["average_session_time"]
   162→                }
   163→            }
   164→            
   165→            # Record processing metrics
   166→            metrics.counter("metrics_calculated").increment(len(processed_metrics))
   167→            metrics.histogram("revenue_per_customer").record(processed_metrics["revenue_analysis"]["revenue_per_customer"])
   168→            metrics.gauge("active_customer_rate").set(processed_metrics["customer_health"]["active_rate"])
   169→            
   170→            # Store processed metrics for next step
   171→            context.set_variable("business_metrics", processed_metrics)
   172→            context.set_variable("processing_timestamp", time.time())
   173→            
   174→            # Stop performance monitoring
   175→            processing_timer.stop()
   176→            
   177→            # Add trace attributes
   178→            span.set_attribute("metrics_calculated", len(processed_metrics))
   179→            span.set_attribute("revenue_per_customer", processed_metrics["revenue_analysis"]["revenue_per_customer"])
   180→            span.set_attribute("status", "success")
   181→            
   182→            # Success logging
   183→            log_structured("info", "Metrics processing completed", {
   184→                "step": "process_metrics",
   185→                "metrics_count": len(processed_metrics),
   186→                "revenue_per_customer": processed_metrics["revenue_analysis"]["revenue_per_customer"],
   187→                "execution_time": processing_timer.elapsed(),
   188→                "status": "success"
   189→            })
   190→            
   191→            print(f"💰 Revenue per customer: $\{processed_metrics['revenue_analysis']['revenue_per_customer']}")
   192→            print(f"📈 Active customer rate: {processed_metrics['customer_health']['active_rate']}%")
   193→            
   194→            # Continue to report generation
   195→            return "generate_report"
   196→            
   197→        except Exception as e:
   198→            # Error metrics and logging
   199→            metrics.counter("processing_errors").increment()
   200→            span.set_attribute("error", True)
   201→            span.set_attribute("error_message", str(e))
   202→            
   203→            log_structured("error", "Metrics processing failed", {
   204→                "step": "process_metrics",
   205→                "error": str(e),
   206→                "trace_id": span.get_span_context().trace_id
   207→            })
   208→            
   209→            raise
    95→
    96→@state(
    97→    cpu=1.0,
    98→    memory=512,
    99→    timeout=20.0,
   100→    priority=Priority.NORMAL
   101→)
   102→async def generate_report(context):
   103→    """Step 3: Generate business report from processed metrics"""
   104→    print("📋 Generating business report...")
   105→    
   106→    # Get processed metrics from previous step
   107→    metrics = context.get_variable("business_metrics")
   108→    fetch_time = context.get_variable("fetch_timestamp")
   109→    process_time = context.get_variable("processing_timestamp")
   110→    
   111→    # Simulate report generation
   112→    await asyncio.sleep(1.0)
   113→    
   114→    # Create comprehensive report
   115→    report = {
   116→        "report_id": f"analytics_report_{int(time.time())}",
   117→        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
   118→        "summary": {
   119→            "total_customers": metrics["customer_health"]["total_customers"],
   120→            "active_rate": f"{metrics['customer_health']['active_rate']}%",
   121→            "revenue_per_customer": f"$\{metrics['revenue_analysis']['revenue_per_customer']}",
   122→            "churn_rate": f"{metrics['customer_health']['churn_rate']}%"
   123→        },
   124→        "detailed_metrics": metrics,
   125→        "performance": {
   126→            "data_fetch_duration": round(process_time - fetch_time, 2),
   127→            "processing_duration": round(time.time() - process_time, 2),
   128→            "total_workflow_duration": round(time.time() - fetch_time, 2)
   129→        },
   130→        "recommendations": generate_recommendations(metrics)
   131→    }
   132→    
   133→    # Store final report for next step
   134→    context.set_variable("final_report", report)
   135→    
   136→    print(f"📊 Report generated: {report['report_id']}")
   137→    print(f"⏱️  Total workflow time: {report['performance']['total_workflow_duration']}s")
   138→    
   139→    # Continue to notification step
   140→    return "send_notifications"
   141→
   142→@state(
   143→    cpu=0.5,
   144→    memory=256,
   145→    timeout=15.0,
   146→    max_retries=3
   147→)
   148→async def send_notifications(context):
   149→    """Step 4: Send report notifications to stakeholders"""
   150→    print("📧 Sending notifications to stakeholders...")
   151→    
   152→    # Get final report from previous step
   153→    report = context.get_variable("final_report")
   154→    
   155→    # Simulate sending notifications to different channels
   156→    notification_channels = [
   157→        {"type": "email", "recipients": ["ceo@company.com", "analytics@company.com"]},
   158→        {"type": "slack", "channel": "#analytics-alerts"},
   159→        {"type": "dashboard", "url": "/dashboard/customer-analytics"}
   160→    ]
   161→    
   162→    successful_notifications = []
   163→    
   164→    for channel in notification_channels:
   165→        try:
   166→            # Simulate notification sending
   167→            await asyncio.sleep(0.5)
   168→            
   169→            notification_result = {
   170→                "channel": channel["type"],
   171→                "status": "sent",
   172→                "timestamp": time.time(),
   173→                "summary": {
   174→                    "customers": report["summary"]["total_customers"], 
   175→                    "revenue_per_customer": report["summary"]["revenue_per_customer"]
   176→                }
   177→            }
   178→            
   179→            successful_notifications.append(notification_result)
   180→            print(f"✅ Sent to {channel['type']}")
   181→            
   182→        except Exception as e:
   183→            print(f"⚠️ Failed to send to {channel['type']}: {e}")
   184→    
   185→    # Store notification results
   186→    context.set_variable("notification_results", successful_notifications)
   187→    
   188→    print(f"📤 Notifications sent to {len(successful_notifications)} channels")
   189→    
   190→    # End workflow (return None)
   191→    return None
   192→
   193→def generate_recommendations(metrics: Dict[str, Any]) -> list:
   194→    """Generate business recommendations based on metrics"""
   195→    recommendations = []
   196→    
   197→    customer_health = metrics["customer_health"]
   198→    revenue_analysis = metrics["revenue_analysis"]
   199→    engagement = metrics["engagement_insights"]
   200→    
   201→    if customer_health["churn_rate"] > 5.0:
   202→        recommendations.append("High churn rate detected - implement retention campaign")
   203→    
   204→    if customer_health["active_rate"] < 70.0:
   205→        recommendations.append("Low customer activation - review onboarding process")
   206→    
   207→    if engagement["dau_to_total_ratio"] < 30.0:
   208→        recommendations.append("Low daily engagement - consider feature improvements")
   209→    
   210→    if revenue_analysis["revenue_per_customer"] < 40.0:
   211→        recommendations.append("Low revenue per customer - explore upselling opportunities")
   212→    
   213→    return recommendations
   214→
   215→# Register all workflow steps
   216→analytics_agent.add_state("fetch_data", fetch_customer_data)
   217→analytics_agent.add_state("process_metrics", process_metrics)
   218→analytics_agent.add_state("generate_report", generate_report)
   219→analytics_agent.add_state("send_notifications", send_notifications)
   220→
   221→async def run_customer_analytics():
   222→    """Run the complete customer analytics workflow"""
   223→    print("🚀 Starting Customer Analytics Workflow")
   224→    print("=" * 50)
   225→    
   226→    # Run the workflow starting from fetch_data
   227→    result = await analytics_agent.run(
   228→        start_state="fetch_data",
   229→        execution_mode="SEQUENTIAL"
   230→    )
   231→    
   232→    print("=" * 50)
   233→    print("✨ Workflow Complete!")
   234→    
   235→    # Access final results
   236→    final_report = result.get_variable("final_report")
   237→    notifications = result.get_variable("notification_results")
   238→    
   239→    print(f"📊 Generated report: {final_report['report_id']}")
   240→    print(f"📧 Sent {len(notifications)} notifications")
   241→    print(f"💡 Recommendations: {len(final_report['recommendations'])}")
   242→    
   243→    return result
   244→
   245→if __name__ == "__main__":
   246→    # Run the workflow
   247→    asyncio.run(run_customer_analytics())
   248→\`\`\`

---

## Step 4: Run the Observable Workflow

Save both files and run:

\`\`\`bash
python customer_analytics.py
\`\`\`

**Observable Output with Metrics:**
\`\`\`
🚀 Starting Customer Analytics Workflow
==================================================
[INFO] Starting customer data fetch | trace_id=abc123 step=fetch_customer_data
📊 Fetching customer data from API...
[METRIC] api_fetch_duration=1.52s customers_fetched=2500 active_customers=1875
✅ Fetched data for 2500 customers
[INFO] Customer data fetch completed | execution_time=1.52s status=success

[INFO] Starting metrics processing | trace_id=abc123 step=process_metrics  
🧮 Processing customer metrics...
[METRIC] memory_usage_mb=45.2 metrics_calculated=3 revenue_per_customer=50.0
💰 Revenue per customer: $50.0
📈 Active customer rate: 75.0%
[INFO] Metrics processing completed | metrics_count=3 execution_time=2.01s

[INFO] Starting report generation | trace_id=abc123 step=generate_report
📋 Generating business report...
[METRIC] report_generation_duration=1.03s recommendations_generated=2
📊 Report generated: analytics_report_1674645234
⏱️  Total workflow time: 4.5s

[INFO] Starting notifications | trace_id=abc123 step=send_notifications
📧 Sending notifications to stakeholders...
[METRIC] notification_success_rate=100% channels_notified=3
✅ Sent to email ✅ Sent to slack ✅ Sent to dashboard
📤 Notifications sent to 3 channels
==================================================
✨ Workflow Complete!
📊 Generated report: analytics_report_1674645234
📧 Sent 3 notifications | 💡 Recommendations: 2

[OBSERVABILITY SUMMARY]
• Total Execution Time: 4.56s
• Memory Peak Usage: 67.8MB  
• Traces Generated: 4 spans
• Metrics Collected: 12 data points
• Error Rate: 0% (0/4 steps failed)
\`\`\`

---

## Understanding the Observable Data Flow

### 🔄 How Data Flows with Full Observability

**1. Fetch Step (Lines 21-104)** - **Distributed Tracing + Metrics**
\`\`\`python
# Observability Setup
with tracer.start_span("fetch_customer_data") as span:
    fetch_timer = metrics.start_timer("api_fetch_duration")
    
    # Data Flow
    # Input: None (starting step)
    # Process: API call with performance monitoring
    customer_data = {...}  # Mock API response
    
    # Metrics Collection
    metrics.counter("customers_fetched").increment(2500)
    metrics.gauge("active_customers").set(1875)
    
    # Output: raw_customer_data → stored in context
    context.set_variable("raw_customer_data", customer_data)
    return "process_metrics"  # Next step with trace context
\`\`\`

**2. Process Step (Lines 112-209)** - **Performance Monitoring + Memory Tracking**
\`\`\`python
# Observability Setup
with tracer.start_span("process_metrics") as span:
    processing_timer = metrics.start_timer("metrics_processing_duration")
    memory_monitor = metrics.gauge("memory_usage_mb")
    
    # Data Flow
    # Input: raw_customer_data ← retrieved from context
    raw_data = context.get_variable("raw_customer_data")
    # Process: Calculate business metrics with monitoring
    processed_metrics = {...}  # Business calculations
    
    # Performance Metrics
    metrics.histogram("revenue_per_customer").record(50.0)
    metrics.gauge("active_customer_rate").set(75.0)
    
    # Output: business_metrics → stored in context
    context.set_variable("business_metrics", processed_metrics)
    return "generate_report"  # Next step
\`\`\`

### 🎯 Key Observability Concepts

**Distributed Tracing = Request Journey**
\`\`\`python
with tracer.start_span("step_name") as span:
    span.set_attribute("key", "value")  # Add context
    span.set_attribute("status", "success")  # Track outcomes
    # span automatically links to parent traces
\`\`\`

**Metrics Collection = Performance Data**
\`\`\`python
# Different metric types for different use cases
metrics.counter("events_count").increment()      # Counts
metrics.gauge("current_value").set(42)           # Current state  
metrics.histogram("duration").record(1.5)       # Distributions
timer = metrics.start_timer("operation_time")   # Time tracking
\`\`\`

**Structured Logging = Searchable Context**
\`\`\`python
log_structured("info", "Operation completed", {
    "step": "process_metrics",
    "execution_time": 2.01,
    "trace_id": span.get_span_context().trace_id,
    "status": "success"
})
\`\`\`

**Observability Decorators = Automatic Monitoring**
\`\`\`python
@observe(metrics=metrics, tracer=tracer)  # Auto-instrumentation
@state(cpu=2.0, memory=1024)             # Resource monitoring
async def process_step(context):
    # Automatic trace creation, timing, and error tracking
    pass
\`\`\`

---

## What You've Built

✅ **Complete observable data processing pipeline**  
✅ **Distributed tracing** across all workflow steps  
✅ **Performance metrics** (timing, memory, throughput)  
✅ **Structured logging** with trace correlation  
✅ **Error tracking** and alerting capabilities  
✅ **Resource monitoring** (CPU, memory, timeouts)  
✅ **Production-ready observability** patterns  

---

## Next Steps

**Dive deeper into production-ready features:**

- **[Observability →](/docs/observability)** - Advanced monitoring, alerting, and dashboards  
- **[Resource Management →](/docs/resource-management)** - Scale with CPU/memory controls and resource pools
- **[Error Handling →](/docs/error-handling)** - Build fault-tolerant workflows with circuit breakers  
- **[API Reference →](/docs/api-reference)** - Complete framework documentation

**Enhance your observable workflow:**
- Add custom metrics for business KPIs
- Implement distributed tracing across microservices
- Set up alerting rules for performance thresholds
- Create observability dashboards and reports
- Add parallel processing with trace correlation
- Implement error recovery with metric tracking

You now understand how to build **production-ready, observable workflows** with Puffinflow!
`.trim();