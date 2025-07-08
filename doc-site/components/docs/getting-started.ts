export const gettingStartedMarkdown = `# Getting Started with Puffinflow

## Prerequisites

- **Python 3.8+**
- Basic familiarity with \`async/await\` in Python
- 5 minutes to get your first workflow running! ⏱️

## Installation

\`\`\`bash
pip install puffinflow
\`\`\`

## Your First Workflow

Create a complete workflow in just **3 simple steps**:

\`\`\`python
import asyncio
from puffinflow import Agent

# 1. Create an agent
agent = Agent("my-first-workflow")

# 2. Define a state (just a regular async function)
async def hello_world(context):
    print("Hello, Puffinflow! 🐧")
    print(f"Workflow ID: {context.workflow_id}")
    return None

# 3. Add state and run it
agent.add_state("hello_world", hello_world)

if __name__ == "__main__":
    asyncio.run(agent.run())
\`\`\`

**Output:**
\`\`\`
Hello, Puffinflow! 🐧
Workflow ID: my-first-workflow
\`\`\`

🎉 **Congratulations!** You just ran your first Puffinflow workflow.

## Two Ways to Define States

For simple workflows, both approaches work identically:

### Plain Functions (Simplest)
\`\`\`python
async def process_data(context):
    context.set_variable("result", "Hello!")
    return None
\`\`\`

### With Decorator (For Advanced Features Later)
\`\`\`python
from puffinflow import state

@state
async def process_data(context):
    context.set_variable("result", "Hello!")
    return None
\`\`\`

> **The difference?** None for basic workflows! The decorator becomes useful when you later want to add resource management, priorities, rate limiting, etc. Start simple, add the decorator when you need advanced features.

## Sharing Data Between States

The **context** is how states communicate with each other. Think of it as a type-safe shared memory:

\`\`\`python
import asyncio
from puffinflow import Agent

agent = Agent("data-pipeline")

async def fetch_data(context):
    # Simulate fetching data from an API
    print("📊 Fetching user data...")

    # Store data in context
    context.set_variable("user_count", 1250)
    context.set_variable("revenue", 45000)
    print("✅ Data fetched successfully")

async def calculate_metrics(context):
    # Get data from previous state
    users = context.get_variable("user_count")
    revenue = context.get_variable("revenue")

    # Calculate and store result
    revenue_per_user = revenue / users
    context.set_variable("revenue_per_user", revenue_per_user)

    print(f"💰 Revenue per user: \${revenue_per_user:.2f}")
    print("✅ Metrics calculated")

async def send_report(context):
    # Use the calculated metric
    rpu = context.get_variable("revenue_per_user")

    print(f"📧 Sending report: RPU is \${rpu:.2f}")
    print("✅ Report sent!")

# Add states to workflow
agent.add_state("fetch_data", fetch_data)
agent.add_state("calculate_metrics", calculate_metrics)
agent.add_state("send_report", send_report)

# Run the complete pipeline
asyncio.run(agent.run())
\`\`\`

**Output:**
\`\`\`
📊 Fetching user data...
✅ Data fetched successfully
💰 Revenue per user: $36.00
✅ Metrics calculated
📧 Sending report: RPU is $36.00
✅ Report sent!
\`\`\`

## Three Ways to Control Workflow Flow

Puffinflow gives you three powerful ways to control how your workflow executes:

### 1. Sequential Execution (Default)

States run in the order you add them:

\`\`\`python
agent = Agent("sequential-workflow")

async def step_one(context):
    print("Step 1: Preparing data")
    context.set_variable("step1_done", True)

async def step_two(context):
    print("Step 2: Processing data")
    context.set_variable("step2_done", True)

async def step_three(context):
    print("Step 3: Finalizing")
    print("All steps complete!")

# Runs in this exact order: step_one → step_two → step_three
agent.add_state("step_one", step_one)
agent.add_state("step_two", step_two)
agent.add_state("step_three", step_three)
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

### Parallel Execution

Return a list of state names to run multiple states at once:

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
from puffinflow.observability import MetricsCollector
from puffinflow.utils import save_checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize metrics
metrics = MetricsCollector(namespace="document_processor")

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
    metrics.increment("validation_started")
    
    try:
        file_path = context.get_variable("file_path")
        file_size = Path(file_path).stat().st_size
        
        # Validate file size (max 10MB)
        if file_size > 10 * 1024 * 1024:
            context.set_variable("error", "File too large")
            metrics.increment("validation_failed", tags={"reason": "file_size"})
            return "error_handler"
        
        # Validate file format
        if not file_path.lower().endswith(('.pdf', '.docx', '.txt')):
            context.set_variable("error", "Unsupported file format")
            metrics.increment("validation_failed", tags={"reason": "format"})
            return "error_handler"
        
        context.set_variable("file_size", file_size)
        logger.info(f"Document validated: {file_size} bytes")
        metrics.increment("validation_succeeded")
        
        return "extract_content"
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        context.set_variable("error", str(e))
        metrics.increment("validation_failed", tags={"reason": "exception"})
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
    metrics.increment("extraction_started")
    
    with metrics.timer("extraction_time"):
        try:
            file_path = context.get_variable("file_path")
            
            # Simulate content extraction
            await asyncio.sleep(2)  # Replace with actual extraction
            
            content = f"Extracted content from {file_path}"
            word_count = len(content.split())
            
            context.set_variable("content", content)
            context.set_variable("word_count", word_count)
            
            logger.info(f"Content extracted: {word_count} words")
            metrics.gauge("content_word_count", word_count)
            metrics.increment("extraction_succeeded")
            
            return "analyze_content"
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            context.set_variable("error", str(e))
            metrics.increment("extraction_failed")
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
    metrics.increment("analysis_started")
    
    with metrics.timer("analysis_time"):
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
            metrics.gauge("analysis_confidence", analysis["confidence"])
            metrics.increment("analysis_succeeded")
            
            return "save_results"
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            context.set_variable("error", str(e))
            metrics.increment("analysis_failed")
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
    metrics.increment("save_started")
    
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
        metrics.increment("save_succeeded")
        
        return "send_notification"
        
    except Exception as e:
        logger.error(f"Save error: {e}")
        context.set_variable("error", str(e))
        metrics.increment("save_failed")
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
    metrics.increment("notification_started")
    
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
        metrics.increment("notification_succeeded")
        
        return None  # End workflow
        
    except Exception as e:
        logger.error(f"Notification error: {e}")
        context.set_variable("error", str(e))
        metrics.increment("notification_failed")
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
    metrics.increment("error_handled")
    
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
        metrics.increment("error_handler_failed")
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
        context = await processor.run(
            initial_context={"file_path": file_path}
        )
        
        # Save checkpoint periodically
        save_checkpoint(context, f"checkpoint_{hash(file_path)}.json")
        
        # Check results
        if context.has_variable("results"):
            results = context.get_variable("results")
            logger.info(f"Processing completed successfully: {results['id']}")
            return results
        else:
            error = context.get_variable("error", "Unknown error")
            logger.error(f"Processing failed: {error}")
            return None
            
    except Exception as e:
        logger.critical(f"Workflow execution failed: {e}")
        metrics.increment("workflow_failed")
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
from puffinflow.observability import MetricsCollector

metrics = MetricsCollector()

@state
async def monitored_state(context):
    metrics.increment("state_executions")
    
    start_time = time.time()
    try:
        with metrics.timer("operation_duration"):
            result = await business_operation()
        
        metrics.gauge("result_size", len(result))
        metrics.increment("successful_operations")
        
        context.set_variable("result", result)
        return "next_state"
    except Exception as e:
        metrics.increment("failed_operations")
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
