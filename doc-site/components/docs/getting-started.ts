export const gettingStartedMarkdown = `# Getting Started with PuffinFlow

**PuffinFlow** is a Python workflow orchestration framework specifically designed for AI-first engineering teams. It bridges the gap between rapid prototyping and production-ready systems, providing Airflow-style orchestration for async functions with modern AI workload optimizations.

## What Makes PuffinFlow Different?

Unlike traditional workflow tools, PuffinFlow is built from the ground up for:

- **🤖 AI/ML Workflows**: Optimized for LLM chains, model training, and data processing pipelines
- **⚡ Async-First**: Native async/await support with intelligent concurrency management  
- **🎯 Resource-Aware**: Automatic CPU, memory, and GPU allocation with intelligent scheduling
- **🔄 Fault-Tolerant**: Built-in retry policies, circuit breakers, and checkpointing
- **📊 Observable**: OpenTelemetry integration, Prometheus metrics, and real-time monitoring
- **🚀 Production-Ready**: From prototype to production with zero config changes

## Prerequisites

- **Python 3.8+** (3.9+ recommended for optimal performance)
- Basic familiarity with \`async/await\` in Python
- Understanding of workflow concepts (states, transitions, context)
- 10 minutes to build your first production-ready AI workflow! ⏱️

## Installation

\`\`\`bash
pip install puffinflow
\`\`\`

For development with all optional dependencies:
\`\`\`bash
pip install "puffinflow[dev,observability,integrations]"
\`\`\`

## Your First Agent

PuffinFlow organizes workflows around **Agents** - intelligent orchestrators that manage state execution, resource allocation, and error handling. Let's create your first agent:

\`\`\`python
import asyncio
from puffinflow import Agent

# Create an agent - the core abstraction in PuffinFlow
agent = Agent("my-first-ai-agent")

# Define a state (an async function that does work)
async def welcome_state(context):
    """A simple state that introduces PuffinFlow concepts."""
    print("🐧 Welcome to PuffinFlow!")
    print(f"Agent ID: {context.agent_id}")
    print(f"Workflow ID: {context.workflow_id}")
    
    # Store data in context for other states to use
    context.set_variable("greeting_sent", True)
    context.set_variable("timestamp", "2024-01-15T10:30:00Z")
    
    print("✅ Welcome state completed")
    return None  # Workflow ends here

# Register the state with the agent
agent.add_state("welcome", welcome_state)

# Run the workflow
if __name__ == "__main__":
    result = asyncio.run(agent.run())
    # Agent.run() returns AgentResult. Access variables via the dict.
    print(f"Workflow completed with {len(result.variables)} variables stored")
\`\`\`

**Output:**
\`\`\`
🐧 Welcome to PuffinFlow!
Agent ID: my-first-ai-agent
Workflow ID: my-first-ai-agent-20240115-103000
✅ Welcome state completed
Workflow completed with 2 variables stored
\`\`\`

🎉 **Congratulations!** You just ran your first PuffinFlow agent.

## Understanding Agent States

In PuffinFlow, **states** are the building blocks of your workflow. Each state is an async function that:
- Receives a \`context\` object containing shared data
- Performs some work (API calls, data processing, etc.)
- Returns the name of the next state to execute (or \`None\` to end)

### 🎯 **Sequential Execution by Default**

**PuffinFlow uses sequential execution by default**, meaning states run one after another in the order they complete and return next states. This provides predictable, ordered workflow execution that's perfect for most use cases:

\`\`\`python
# Default behavior - states run sequentially
result = await agent.run()  # Sequential execution

# For parallel execution when needed
from puffinflow import ExecutionMode
result = await agent.run(execution_mode=ExecutionMode.PARALLEL)
\`\`\`

This sequential-first approach ensures your workflows are easy to debug, reason about, and maintain. You can always opt into parallel execution when you need maximum performance for independent tasks.

### Method 1: Simple Function States

For rapid prototyping and simple workflows:

\`\`\`python
async def process_user_data(context):
    """Simple state function - perfect for getting started."""
    user_id = context.get_variable("user_id")
    
    # Simulate API call
    user_data = await fetch_user_from_api(user_id)
    
    # Store result for next state
    context.set_variable("user_data", user_data)
    print(f"✅ Fetched data for user {user_id}")
    
    return "validate_data"  # Move to next state
\`\`\`

### Method 2: Decorator-Enhanced States (Recommended)

For production workflows with resource management and reliability:

\`\`\`python
from puffinflow import Agent, state

class UserProcessorAgent(Agent):
    @state(
        cpu=2.0,           # Request 2 CPU cores
        memory=1024,       # Request 1GB memory
        timeout=30.0,      # 30 second timeout
        max_retries=3      # Retry up to 3 times on failure
    )
    async def process_user_data(self, context):
        """Production-ready state with resource management."""
        user_id = context.get_variable("user_id")
        
        # Simulate resource-intensive processing
        user_data = await self.heavy_data_processing(user_id)
        
        context.set_variable("processed_data", user_data)
        return "send_notification"
    
    @state(cpu=0.5, memory=256, priority="low")
    async def send_notification(self, context):
        """Lightweight notification state."""
        data = context.get_variable("processed_data")
        await self.send_email_notification(data)
        return None  # End workflow

# Usage
agent = UserProcessorAgent("user-processor")
result = await agent.run(initial_context={"user_id": 12345})
\`\`\`

> **💡 Pro Tip**: Start with simple functions, then add the \`@state\` decorator when you need production features like resource allocation, retries, timeouts, and priority scheduling.

## Context: The Heart of Data Flow

The **Context** object is PuffinFlow's intelligent data management system. It provides type-safe data sharing between states with built-in caching, validation, and persistence. Think of it as a smart, shared memory that flows through your entire workflow.

### Basic Context Operations

\`\`\`python
import asyncio
from puffinflow import Agent

# Create an analytics pipeline agent
analytics_agent = Agent("business-analytics-pipeline")

async def fetch_business_data(context):
    """Fetch data from multiple sources concurrently."""
    print("📊 Fetching business data from APIs...")
    
    # Simulate concurrent API calls
    import aiohttp
    async with aiohttp.ClientSession() as session:
        # In real usage, these would be actual API calls  
        user_data = {"total_users": 1250, "active_users": 980}
        revenue_data = {"total_revenue": 45000, "monthly_recurring": 32000}
        analytics_data = {"page_views": 125000, "conversion_rate": 0.034}
    
    # Store all data in context with type hints
    context.set_variable("user_metrics", user_data)
    context.set_variable("revenue_metrics", revenue_data) 
    context.set_variable("analytics_metrics", analytics_data)
    context.set_variable("data_fetch_timestamp", "2024-01-15T10:30:00Z")
    
    print("✅ Business data fetched successfully")
    return "calculate_kpis"

async def calculate_kpis(context):
    """Calculate key performance indicators from raw data."""
    print("🧮 Calculating business KPIs...")
    
    # Retrieve data from context (with automatic type checking)
    user_metrics = context.get_variable("user_metrics")
    revenue_metrics = context.get_variable("revenue_metrics")
    analytics_metrics = context.get_variable("analytics_metrics")
    
    # Calculate KPIs
    kpis = {
        "revenue_per_user": revenue_metrics["total_revenue"] / user_metrics["total_users"],
        "user_engagement": user_metrics["active_users"] / user_metrics["total_users"],
        "conversion_value": analytics_metrics["page_views"] * analytics_metrics["conversion_rate"],
        "mrr_per_user": revenue_metrics["monthly_recurring"] / user_metrics["active_users"]
    }
    
    # Store calculated KPIs
    context.set_variable("business_kpis", kpis)
    context.set_variable("calculation_timestamp", "2024-01-15T10:31:00Z")
    
    print(f"💰 Revenue per user: \${kpis['revenue_per_user']:.2f}")
    print(f"👥 User engagement: {kpis['user_engagement']:.1%}")
    print("✅ KPI calculations completed")
    
    return "generate_insights"

async def generate_insights(context):
    """Generate business insights using calculated KPIs."""
    print("🧠 Generating business insights...")
    
    kpis = context.get_variable("business_kpis")
    
    # Generate insights based on KPI thresholds
    insights = []
    
    if kpis["revenue_per_user"] > 35:
        insights.append("💡 High RPU indicates strong monetization")
    
    if kpis["user_engagement"] > 0.75:
        insights.append("🎯 Excellent user engagement levels")
    
    if kpis["mrr_per_user"] > 30:
        insights.append("📈 Strong recurring revenue per user")
    
    # Store insights and create summary report
    report = {
        "kpis": kpis,
        "insights": insights,
        "generated_at": "2024-01-15T10:32:00Z",
        "report_id": f"RPT-{hash(str(kpis))}"
    }
    
    context.set_variable("business_report", report)
    
    print("📊 Business Insights Generated:")
    for insight in insights:
        print(f"   {insight}")
    
    return "send_report"

async def send_report(context):
    """Send the business report to stakeholders."""
    print("📧 Preparing business report...")
    
    report = context.get_variable("business_report")
    
    # In production, this would send to email/Slack/dashboard
    print(f"📈 Report {report['report_id']} ready for distribution")
    print(f"   KPIs calculated: {len(report['kpis'])}")
    print(f"   Insights generated: {len(report['insights'])}")
    
    context.set_variable("report_sent", True)
    print("✅ Business report sent to stakeholders!")

# Build the analytics pipeline
analytics_agent.add_state("fetch_data", fetch_business_data)
analytics_agent.add_state("calculate_kpis", calculate_kpis)
analytics_agent.add_state("generate_insights", generate_insights)
analytics_agent.add_state("send_report", send_report)

# Execute the complete pipeline
if __name__ == "__main__":
    result = asyncio.run(analytics_agent.run())
    
    # Access final results
    final_report = result.get_variable("business_report")
    print(f"\\n🎉 Pipeline completed! Report ID: {final_report['report_id']}")
\`\`\`

**Sample Output:**
\`\`\`
📊 Fetching business data from APIs...
✅ Business data fetched successfully
🧮 Calculating business KPIs...
💰 Revenue per user: $36.00
👥 User engagement: 78.4%
✅ KPI calculations completed
🧠 Generating business insights...
📊 Business Insights Generated:
   💡 High RPU indicates strong monetization
   🎯 Excellent user engagement levels
   📈 Strong recurring revenue per user
📧 Preparing business report...
📈 Report RPT-1234567890 ready for distribution
   KPIs calculated: 4
   Insights generated: 3
✅ Business report sent to stakeholders!

🎉 Pipeline completed! Report ID: RPT-1234567890
\`\`\`

### Context Features

The Context object provides powerful features beyond simple variable storage:

\`\`\`python
# Type-safe and validated storage
# - Use set_typed_variable for type-consistent primitives/objects
# - Use set_validated_data for Pydantic models
context.set_typed_variable("user_count", 42)
context.set_validated_data("user", user_model_instance)

# Built-in caching with TTL (seconds)
context.set_cached("api_response", data, ttl=300)  # Cache for 5 minutes

# Metadata tracking
context.set_metadata("processing_time", 1.2)
context.set_metadata("data_source", "production_api")

# Metrics collection
context.increment_metric("api_calls")
context.set_metric("response_size", len(response))
\`\`\`

## Workflow Control Patterns

PuffinFlow provides flexible workflow execution patterns that adapt to your specific use case. Understanding these patterns is crucial for building efficient, scalable agents.

### 1. Sequential Execution (Linear Pipeline)

Perfect for data pipelines where each step depends on the previous one:

\`\`\`python
from puffinflow import Agent

# Create a machine learning training pipeline
ml_pipeline = Agent("ml-training-pipeline")

async def load_dataset(context):
    """Load and validate training data."""
    print("📊 Loading training dataset...")
    
    # Simulate dataset loading
    dataset = {"features": 10000, "samples": 50000, "labels": ["positive", "negative"]}
    context.set_variable("dataset", dataset)
    context.set_variable("dataset_size", dataset["samples"])
    
    print(f"✅ Dataset loaded: {dataset['samples']} samples, {dataset['features']} features")
    return "preprocess_data"  # Next state

async def preprocess_data(context):
    """Clean and prepare data for training."""
    print("🧹 Preprocessing data...")
    
    dataset = context.get_variable("dataset")
    
    # Simulate preprocessing
    processed_data = {
        "normalized_features": dataset["features"],
        "train_samples": int(dataset["samples"] * 0.8),
        "val_samples": int(dataset["samples"] * 0.2),
        "preprocessing_method": "standard_scaler"
    }
    
    context.set_variable("processed_data", processed_data)
    print(f"✅ Data preprocessed: {processed_data['train_samples']} train, {processed_data['val_samples']} validation")
    
    return "train_model"

async def train_model(context):
    """Train the ML model."""
    print("🤖 Training machine learning model...")
    
    processed_data = context.get_variable("processed_data")
    
    # Simulate model training
    model_stats = {
        "accuracy": 0.94,
        "precision": 0.92,
        "recall": 0.89,
        "training_time": 45.2,
        "model_type": "random_forest"
    }
    
    context.set_variable("trained_model", model_stats)
    print(f"✅ Model trained: {model_stats['accuracy']:.1%} accuracy in {model_stats['training_time']}s")
    
    return "evaluate_model"

async def evaluate_model(context):
    """Evaluate model performance."""
    print("📈 Evaluating model performance...")
    
    model_stats = context.get_variable("trained_model")
    
    if model_stats["accuracy"] > 0.9:
        print("🎉 Model meets performance criteria!")
        context.set_variable("model_approved", True)
        return "deploy_model"
    else:
        print("❌ Model performance insufficient")
        context.set_variable("model_approved", False)
        return "retrain_model"  # Would loop back to training

async def deploy_model(context):
    """Deploy the approved model."""
    model_stats = context.get_variable("trained_model")
    
    print(f"🚀 Deploying model with {model_stats['accuracy']:.1%} accuracy")
    context.set_variable("deployment_status", "deployed")
    print("✅ Model successfully deployed to production!")

# Build sequential pipeline - each state runs after the previous completes
ml_pipeline.add_state("load_dataset", load_dataset)
ml_pipeline.add_state("preprocess_data", preprocess_data)
ml_pipeline.add_state("train_model", train_model)
ml_pipeline.add_state("evaluate_model", evaluate_model)
ml_pipeline.add_state("deploy_model", deploy_model)

# Execute the pipeline
if __name__ == "__main__":
    result = asyncio.run(ml_pipeline.run())
    deployment_status = result.get_variable("deployment_status")
    print(f"Pipeline result: {deployment_status}")
\`\`\`

### 2. Static Dependencies

Explicitly declare what must complete before each state runs:

\`\`\`python
async def fetch_user_data(context):
    print("👥 Fetching user data...")
    await asyncio.sleep(0.5)  # Simulate API call
    context.set_variable("user_count", 1250)

async def fetch_sales_data(context):
    print("💰 Fetching sales data...")
    await asyncio.sleep(0.3)  # Simulate API call
    context.set_variable("revenue", 45000)

async def generate_report(context):
    print("📊 Generating report...")
    users = context.get_variable("user_count")
    revenue = context.get_variable("revenue")
    print(f"Revenue per user: \${revenue/users:.2f}")

# fetch_user_data and fetch_sales_data run in parallel
# generate_report waits for BOTH to complete
agent.add_state("fetch_user_data", fetch_user_data)
agent.add_state("fetch_sales_data", fetch_sales_data)
agent.add_state("generate_report", generate_report,
                dependencies=["fetch_user_data", "fetch_sales_data"])
\`\`\`

### 3. Dynamic Flow Control

Return state names from functions to decide what runs next:

\`\`\`python
async def check_user_type(context):
    print("🔍 Checking user type...")
    user_type = "premium"  # Could come from database
    context.set_variable("user_type", user_type)

    # Dynamic routing based on data
    if user_type == "premium":
        return "premium_flow"
    else:
        return "basic_flow"

async def premium_flow(context):
    print("⭐ Premium user workflow")
    context.set_variable("features", ["advanced_analytics", "priority_support"])
    return "send_welcome"

async def basic_flow(context):
    print("👋 Basic user workflow")
    context.set_variable("features", ["basic_analytics"])
    return "send_welcome"

async def send_welcome(context):
    user_type = context.get_variable("user_type")
    features = context.get_variable("features")
    print(f"✉️ Welcome {user_type} user! Features: {', '.join(features)}")

# Add all states
agent.add_state("check_user_type", check_user_type)
agent.add_state("premium_flow", premium_flow)
agent.add_state("basic_flow", basic_flow)
agent.add_state("send_welcome", send_welcome)
\`\`\`

### Dynamic Parallel Execution

Even within sequential workflows, return a list of state names to run multiple states at once:

\`\`\`python
async def process_order(context):
    print("📦 Processing order...")
    context.set_variable("order_id", "ORD-123")

    # Run these three states in parallel
    return ["send_confirmation", "update_inventory", "charge_payment"]

async def send_confirmation(context):
    order_id = context.get_variable("order_id")
    print(f"📧 Confirmation sent for {order_id}")

async def update_inventory(context):
    print("📋 Inventory updated")

async def charge_payment(context):
    order_id = context.get_variable("order_id")
    print(f"💳 Payment processed for {order_id}")
\`\`\`

## Complete Example: Data Pipeline

\`\`\`python
import asyncio
from puffinflow import Agent

agent = Agent("data-pipeline")

async def extract(context):
    data = {"sales": [100, 200, 150], "customers": ["Alice", "Bob", "Charlie"]}
    context.set_variable("raw_data", data)
    print("✅ Data extracted")

async def transform(context):
    raw_data = context.get_variable("raw_data")
    total_sales = sum(raw_data["sales"])
    customer_count = len(raw_data["customers"])

    transformed = {
        "total_sales": total_sales,
        "customer_count": customer_count,
        "avg_sale": total_sales / customer_count
    }

    context.set_variable("processed_data", transformed)
    print("✅ Data transformed")

async def load(context):
    processed_data = context.get_variable("processed_data")
    print(f"✅ Saved: {processed_data}")

# Set up the pipeline - runs sequentially
agent.add_state("extract", extract)
agent.add_state("transform", transform, dependencies=["extract"])
agent.add_state("load", load, dependencies=["transform"])

if __name__ == "__main__":
    asyncio.run(agent.run())
\`\`\`

## When to Use the Decorator

Add the \`@state\` decorator when you need advanced features later:

\`\`\`python
from puffinflow import state

# Advanced features example (you don't need this initially)
@state(cpu=2.0, memory=1024, priority="high", timeout=60.0)
async def intensive_task(context):
    # This state gets 2 CPU units, 1GB memory, high priority, 60s timeout
    pass
\`\`\`

## Quick Reference

### Flow Control Methods
\`\`\`python
# Sequential (default)
agent.add_state("first", first_function)
agent.add_state("second", second_function)

# Dependencies
agent.add_state("dependent", function, dependencies=["first", "second"])

# Dynamic routing
async def router(context):
    return "next_state"           # Single state
    return ["state1", "state2"]   # Parallel states
\`\`\`

### Context Methods
- \`context.set_variable(key, value)\` - Store data
- \`context.get_variable(key)\` - Retrieve data

### State Return Values
- \`None\` - Continue normally
- \`"state_name"\` - Run specific state next
- \`["state1", "state2"]\` - Run multiple states in parallel

## 🤖 Complete AI Workflow Example

Here's a real-world example showing how to build an AI research assistant that:
1. Takes a query
2. Searches for information
3. Analyzes findings with an LLM
4. Generates a final report

\`\`\`python
import asyncio
import json
from puffinflow import Agent

# Simulate external APIs
async def search_web(query):
    """Simulate web search API"""
    await asyncio.sleep(0.2)
    return [
        {"title": f"Article about {query}", "content": f"Detailed info on {query}..."},
        {"title": f"{query} trends", "content": f"Latest trends in {query}..."}
    ]

async def call_llm(prompt):
    """Simulate LLM API call"""
    await asyncio.sleep(0.5)
    return f"AI Analysis: {prompt[:50]}..."

# Create the research agent
research_agent = Agent("ai-research-assistant")

async def validate_query(context):
    """Validate and prepare the search query"""
    query = context.get_variable("search_query", "")

    if not query or len(query) < 3:
        print("❌ Invalid query - too short")
        return None  # End workflow

    # Clean and prepare query
    clean_query = query.strip().lower()
    context.set_variable("clean_query", clean_query)

    print(f"✅ Query validated: '{clean_query}'")
    return "search_information"

async def search_information(context):
    """Search for information on the web"""
    query = context.get_variable("clean_query")

    print(f"🔍 Searching for: {query}")
    results = await search_web(query)

    context.set_variable("search_results", results)
    print(f"✅ Found {len(results)} results")

    return "analyze_results"

async def analyze_results(context):
    """Use LLM to analyze search results"""
    results = context.get_variable("search_results")
    query = context.get_variable("clean_query")

    print("🧠 Analyzing results with AI...")

    # Prepare prompt for LLM
    prompt = f"""
    Analyze these search results for query '{query}':
    {json.dumps(results, indent=2)}

    Provide key insights and trends.
    """

    analysis = await call_llm(prompt)
    context.set_variable("analysis", analysis)

    print("✅ Analysis complete")
    return "generate_report"

async def generate_report(context):
    """Generate final research report"""
    query = context.get_variable("search_query")
    analysis = context.get_variable("analysis")
    results = context.get_variable("search_results")

    print("📝 Generating final report...")

    # Create structured report
    report = {
        "query": query,
        "sources_found": len(results),
        "analysis": analysis,
        "generated_at": "2024-01-15 10:30:00",
        "confidence": "high"
    }

    context.set_variable("final_report", report)

    print("🎉 Research Report Generated!")
    print(f"Query: {report['query']}")
    print(f"Sources: {report['sources_found']}")
    print(f"Analysis: {report['analysis']}")

    return None  # End workflow

# Wire up the workflow
research_agent.add_state("validate_query", validate_query)
research_agent.add_state("search_information", search_information)
research_agent.add_state("analyze_results", analyze_results)
research_agent.add_state("generate_report", generate_report)

async def run_research(query):
    """Run a complete research workflow"""
    print(f"🚀 Starting research on: '{query}'")
    print("-" * 50)

    result = await research_agent.run(
        initial_context={"search_query": query}
    )

    print("-" * 50)
    print("✨ Research complete!")

    return result.get_variable("final_report")

# Example usage
if __name__ == "__main__":
    report = asyncio.run(run_research("machine learning trends 2024"))
    print(f"\\nFinal report available in context: {report is not None}")
\`\`\`

**Expected Output:**
\`\`\`
🚀 Starting research on: 'machine learning trends 2024'
--------------------------------------------------
✅ Query validated: 'machine learning trends 2024'
🔍 Searching for: machine learning trends 2024
✅ Found 2 results
🧠 Analyzing results with AI...
✅ Analysis complete
📝 Generating final report...
🎉 Research Report Generated!
Query: machine learning trends 2024
Sources: 2
Analysis: AI Analysis: Analyze these search results for query 'machine...
--------------------------------------------------
✨ Research complete!

Final report available in context: True
\`\`\`

## 🚀 Real-World Production Example

Here's a complete example of a production-ready document processing workflow:

\`\`\`python
import asyncio
import logging
from pathlib import Path
from puffinflow import Agent, state, Priority
from puffinflow.core.observability.metrics import PrometheusMetricsProvider
from puffinflow.core.observability.config import MetricsConfig
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Prometheus metrics provider
provider = PrometheusMetricsProvider(MetricsConfig(namespace="document_processor"))

# Create production agent
processor = Agent("document-processor")

@state(
    cpu=2.0,
    memory=1024,
    priority=Priority.HIGH,
    max_retries=3,
    timeout=120.0
)
async def validate_document(context):
    """Validate uploaded document format and size."""
    logger.info("Starting document validation")
    provider.counter("validation_total", labels=["status"]).record(1, status="started")
    
    try:
        file_path = context.get_variable("file_path")
        file_size = Path(file_path).stat().st_size
        
        # Validate file size (max 10MB)
        if file_size > 10 * 1024 * 1024:
            context.set_variable("error", "File too large")
            provider.counter("validation_total", labels=["status","reason"]).record(1, status="failed", reason="file_size")
            return "error_handler"
        
        # Validate file format
        if not file_path.lower().endswith(('.pdf', '.docx', '.txt')):
            context.set_variable("error", "Unsupported file format")
            provider.counter("validation_total", labels=["status","reason"]).record(1, status="failed", reason="format")
            return "error_handler"
        
        context.set_variable("file_size", file_size)
        logger.info(f"Document validated: {file_size} bytes")
        provider.counter("validation_total", labels=["status"]).record(1, status="succeeded")
        
        return "extract_content"
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        context.set_variable("error", str(e))
        provider.counter("validation_total", labels=["status","reason"]).record(1, status="failed", reason="exception")
        return "error_handler"

@state(
    cpu=4.0,
    memory=2048,
    priority=Priority.NORMAL,
    max_retries=2,
    timeout=300.0
)
async def extract_content(context):
    """Extract text content from document."""
    logger.info("Starting content extraction")
    provider.counter("extraction_total", labels=["status"]).record(1, status="started")
    
    start_time = time.time()
        try:
            file_path = context.get_variable("file_path")
            
            # Simulate content extraction
            await asyncio.sleep(2)  # Replace with actual extraction
            
            content = f"Extracted content from {file_path}"
            word_count = len(content.split())
            
            context.set_variable("content", content)
            context.set_variable("word_count", word_count)
            
            logger.info(f"Content extracted: {word_count} words")
            provider.gauge("content_word_count").record(word_count)
            provider.histogram("extraction_time_seconds").record(time.time() - start_time)
            provider.counter("extraction_total", labels=["status"]).record(1, status="succeeded")
            
            return "analyze_content"
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            context.set_variable("error", str(e))
            provider.counter("extraction_total", labels=["status"]).record(1, status="failed")
            return "error_handler"

@state(
    cpu=2.0,
    memory=1024,
    priority=Priority.NORMAL,
    max_retries=1,
    timeout=180.0
)
async def analyze_content(context):
    """Analyze content with AI/ML processing."""
    logger.info("Starting content analysis")
    provider.counter("analysis_total", labels=["status"]).record(1, status="started")
    
    start = time.time()
        try:
            content = context.get_variable("content")
            word_count = context.get_variable("word_count")
            
            # Simulate AI analysis
            await asyncio.sleep(1)  # Replace with actual AI processing
            
            analysis = {
                "sentiment": "positive",
                "topics": ["technology", "business"],
                "summary": f"Document contains {word_count} words about technology and business.",
                "confidence": 0.95
            }
            
            context.set_variable("analysis", analysis)
            logger.info(f"Analysis complete: {analysis['sentiment']} sentiment")
            provider.gauge("analysis_confidence").record(analysis["confidence"])
            provider.histogram("analysis_time_seconds").record(time.time() - start)
            provider.counter("analysis_total", labels=["status"]).record(1, status="succeeded")
            
            return "save_results"
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            context.set_variable("error", str(e))
            provider.counter("analysis_total", labels=["status"]).record(1, status="failed")
            return "error_handler"

@state(
    cpu=1.0,
    memory=512,
    priority=Priority.NORMAL,
    max_retries=2,
    timeout=60.0
)
async def save_results(context):
    """Save processing results to database."""
    logger.info("Saving results")
    provider.counter("save_total", labels=["status"]).record(1, status="started")
    
    try:
        analysis = context.get_variable("analysis")
        file_path = context.get_variable("file_path")
        
        # Simulate database save
        await asyncio.sleep(0.5)  # Replace with actual database operation
        
        result_id = f"doc_{hash(file_path)}"
        results = {
            "id": result_id,
            "file_path": file_path,
            "analysis": analysis,
            "processed_at": "2024-01-15T10:30:00Z"
        }
        
        context.set_variable("results", results)
        logger.info(f"Results saved with ID: {result_id}")
        provider.counter("save_total", labels=["status"]).record(1, status="succeeded")
        
        return "send_notification"
        
    except Exception as e:
        logger.error(f"Save error: {e}")
        context.set_variable("error", str(e))
        provider.counter("save_total", labels=["status"]).record(1, status="failed")
        return "error_handler"

@state(
    cpu=0.5,
    memory=256,
    priority=Priority.LOW,
    max_retries=3,
    timeout=30.0
)
async def send_notification(context):
    """Send completion notification."""
    logger.info("Sending notification")
    provider.counter("notification_total", labels=["status"]).record(1, status="started")
    
    try:
        results = context.get_variable("results")
        
        # Simulate notification
        await asyncio.sleep(0.2)  # Replace with actual notification service
        
        notification = {
            "type": "success",
            "message": f"Document {results['id']} processed successfully",
            "timestamp": "2024-01-15T10:35:00Z"
        }
        
        context.set_variable("notification", notification)
        logger.info("Notification sent successfully")
        provider.counter("notification_total", labels=["status"]).record(1, status="succeeded")
        
        return None  # End workflow
        
    except Exception as e:
        logger.error(f"Notification error: {e}")
        context.set_variable("error", str(e))
        provider.counter("notification_total", labels=["status"]).record(1, status="failed")
        return "error_handler"

@state(
    cpu=0.5,
    memory=256,
    priority=Priority.HIGH,
    max_retries=1,
    timeout=30.0
)
async def error_handler(context):
    """Handle errors and cleanup."""
    logger.error("Handling workflow error")
    provider.counter("errors_total", labels=["type"]).record(1, type="handled")
    
    try:
        error = context.get_variable("error")
        file_path = context.get_variable("file_path", "unknown")
        
        # Log error details
        logger.error(f"Workflow failed for {file_path}: {error}")
        
        # Cleanup resources
        await cleanup_resources(file_path)
        
        # Send error notification
        error_notification = {
            "type": "error",
            "message": f"Document processing failed: {error}",
            "file_path": file_path,
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        context.set_variable("error_notification", error_notification)
        logger.info("Error handling completed")
        
        return None  # End workflow
        
    except Exception as e:
        logger.critical(f"Error handler failed: {e}")
        provider.counter("errors_total", labels=["type"]).record(1, type="handler_failed")
        return None

async def cleanup_resources(file_path):
    """Cleanup any allocated resources."""
    logger.info(f"Cleaning up resources for {file_path}")
    # Add cleanup logic here
    pass

# Example usage
async def process_document(file_path: str):
    """Process a document through the complete workflow."""
    logger.info(f"Starting document processing: {file_path}")
    
    try:
        # Run workflow with error handling
        result = await processor.run(
            initial_context={"file_path": file_path}
        )
        
        # Optionally persist a checkpoint (configure FileCheckpointStorage for file persistence)
        await processor.save_checkpoint()
        
        # Check results
        results = result.get_variable("results")
        if results is not None:
            logger.info(f"Processing completed successfully: {results['id']}")
            return results
        else:
            error = result.get_variable("error", "Unknown error")
            logger.error(f"Processing failed: {error}")
            return None
            
    except Exception as e:
        logger.critical(f"Workflow execution failed: {e}")
        provider.counter("workflow_total", labels=["status"]).record(1, status="failed")
        return None

# Production usage
if __name__ == "__main__":
    # Process multiple documents
    documents = [
        "/path/to/document1.pdf",
        "/path/to/document2.docx",
        "/path/to/document3.txt"
    ]
    
    async def main():
        tasks = []
        for doc in documents:
            task = asyncio.create_task(process_document(doc))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Summary
        successful = sum(1 for r in results if r is not None and not isinstance(r, Exception))
        logger.info(f"Processed {successful}/{len(documents)} documents successfully")
    
    asyncio.run(main())
\`\`\`

This example demonstrates:
- **Production-ready error handling** with retry logic and cleanup
- **Comprehensive monitoring** with metrics and logging
- **Resource management** with appropriate CPU/memory allocation
- **Prioritization** of critical vs. background tasks
- **Fault tolerance** with checkpointing and recovery
- **Concurrent processing** of multiple documents

## 📚 Common Patterns and Best Practices

### 1. **Error Handling Pattern**

\`\`\`python
@state(max_retries=3, timeout=60.0)
async def robust_state(context):
    try:
        # Your business logic
        result = await risky_operation()
        context.set_variable("result", result)
        return "success_state"
    except SpecificError as e:
        logger.warning(f"Recoverable error: {e}")
        context.set_variable("retry_count", context.get_state("retry_count", 0) + 1)
        return "retry_state"
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        context.set_variable("error", str(e))
        return "error_handler"
\`\`\`

### 2. **Data Validation Pattern**

\`\`\`python
from pydantic import BaseModel, ValidationError

class InputData(BaseModel):
    user_id: int
    email: str
    preferences: dict

@state
async def validate_input(context):
    try:
        raw_data = context.get_variable("raw_input")
        validated_data = InputData(**raw_data)
        context.set_validated_data("input", validated_data)
        return "process_data"
    except ValidationError as e:
        context.set_variable("validation_errors", e.errors())
        return "validation_error"
\`\`\`

### 3. **Resource Optimization Pattern**

\`\`\`python
@state(cpu=0.5, memory=256, priority=Priority.LOW)
async def lightweight_task(context):
    # Light processing
    return "next_state"

@state(cpu=4.0, memory=2048, priority=Priority.HIGH)
async def heavy_task(context):
    # CPU/memory intensive processing
    return "next_state"
\`\`\`

### 4. **Monitoring Pattern**

\`\`\`python
from puffinflow.core.observability.metrics import PrometheusMetricsProvider
from puffinflow.core.observability.config import MetricsConfig
import time

provider = PrometheusMetricsProvider(MetricsConfig(namespace="getting_started"))

@state
async def monitored_state(context):
    provider.counter("state_executions_total").record(1)
    
    start_time = time.time()
    try:
        result = await business_operation()
        duration = time.time() - start_time
        
        provider.histogram("operation_duration_seconds").record(duration)
        provider.gauge("result_size").record(len(result))
        provider.counter("successful_operations_total").record(1)
        
        context.set_variable("result", result)
        return "next_state"
    except Exception:
        provider.counter("failed_operations_total").record(1)
        raise
\`\`\`

## 🎯 Next Steps

You now know the fundamentals! Here's what to explore next:

1. **[Context and Data →](#docs/context-and-data)** - Deep dive into data management and validation
2. **[Resource Management →](#docs/resource-management)** - Control CPU, memory, and rate limits
3. **[Error Handling →](#docs/error-handling)** - Build resilient workflows with retries and circuit breakers
4. **[Checkpointing →](#docs/checkpointing)** - Save and resume progress for long-running workflows
5. **[Observability →](#docs/observability)** - Monitor and debug your workflows in production
6. **[API Reference →](#docs/api-reference)** - Complete reference for all classes and methods
7. **[Troubleshooting →](#docs/troubleshooting)** - Solve common issues and debug problems

**Pro tip:** Start simple with basic workflows, then gradually add advanced features as your needs grow! 🌱
`.trim();
