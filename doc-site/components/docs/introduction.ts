export const introductionMarkdown = `
# Puffinflow Agent Framework

A lightweight Python framework for orchestrating AI agents and data workflows with deterministic, resource-aware execution built for today's AI-first engineering teams.

## What is Puffinflow?

Puffinflow is the modern workflow orchestration framework designed from the ground up for AI and LLM applications. Inspired by Airflow's DAG concept but reimagined for the async, resource-constrained world of AI workloads, Puffinflow gives you **Airflow-style wiring for async functions**—but trimmed down to what you actually need when building with LLMs, vector databases, web scraping, and other I/O-intensive operations.

### The Modern AI Development Problem

Building production AI systems involves stitching together multiple async operations that each have their own challenges:

**🔥 Common Pain Points:**
- **Async Complexity**: Managing \`asyncio\` tasks, preventing race conditions, handling concurrent API calls
- **Resource Management**: LLM APIs have rate limits, cost per token, and memory constraints
- **Data Flow**: Passing structured data between OpenAI calls, embeddings, and storage operations
- **Error Handling**: Retrying failed API calls, handling partial failures, maintaining consistency
- **Observability**: Monitoring token usage, API costs, performance bottlenecks
- **Scalability**: Running hundreds of autonomous agents without overwhelming external APIs

**💔 What We've All Tried:**
- **Pure asyncio**: Quickly becomes callback hell with complex error handling
- **Celery**: Overkill for AI workloads, poor async support, complex setup
- **Airflow**: Too heavy, not designed for high-frequency async tasks
- **Custom solutions**: Reinventing the wheel, missing edge cases, hard to maintain

### The Puffinflow Solution

Puffinflow solves these problems by providing a lightweight, AI-first framework that handles the orchestration complexity while letting you focus on your domain logic.

**🎯 Core Design Principles:**
- **AI-Native**: Built specifically for LLM, embedding, and vector database workflows
- **Async-First**: Every operation is async by default, optimized for I/O-heavy AI workloads
- **Resource-Aware**: Built-in rate limiting, token tracking, and memory management
- **Type-Safe**: Strong typing for data flow between AI operations
- **Fault-Tolerant**: Automatic retries, checkpointing, and graceful error handling

**Key Benefits:**
- **🚀 Simple**: Define states as async functions, wire them together with dependencies
- **🔒 Safe**: Built-in context management prevents race conditions and data corruption
- **⚡ Fast**: Optimized for high-concurrency AI workloads with parallel execution
- **🛡️ Reliable**: Automatic checkpointing, retry logic, and graceful failure recovery
- **📊 Observable**: Rich metrics for token usage, API costs, performance, and errors
- **🎛️ Resource-Aware**: Built-in rate limiting, memory management, and quota tracking
- **🔧 Production-Ready**: Comprehensive error handling, logging, and monitoring capabilities

## Why Another Workflow Tool?

The existing workflow orchestration tools weren't built for the modern AI stack. Here's how Puffinflow addresses the specific challenges AI engineers face daily:

| Your Headache | How Puffinflow Helps | Traditional Tools |
|--------------|---------------------|------------------|
| **Async spaghetti** – callback hell, tangled asyncio tasks | Register tiny, focused states; Puffinflow's scheduler runs them safely and in order | Celery: Poor async support<br/>Airflow: Sync-first design |
| **Global variables & race-conditions** | Built-in, type-locked Context lets every step pass data without foot-guns | Manual threading locks<br/>Redis/database as shared state |
| **"Rate limit exceeded" from day-one** | Opt-in rate-limit helpers keep you under OpenAI/vendor quotas without manual back-off logic | Custom retry logic<br/>Manual queue management |
| **Cloud pre-emptions wiping work** | One-liner checkpoints freeze progress so you can resume exactly where you left off | Losing hours of LLM work<br/>Manual state persistence |
| **LLM token costs spiraling** | Built-in token tracking and cost monitoring across all API calls | Manual logging<br/>Surprise bills |
| **Debugging complex AI chains** | Rich observability with state-by-state execution tracking | Black box execution<br/>Printf debugging |
| **Managing 100s of concurrent agents** | Resource-aware scheduling prevents overwhelming external APIs | Manual semaphores<br/>API timeouts |

### Real-World Examples

**Before Puffinflow:**
\`\`\`python
# 😵 The async spaghetti we've all written
async def process_documents(docs):
    results = []
    semaphore = asyncio.Semaphore(5)  # Manual rate limiting
    
    async def process_single(doc):
        async with semaphore:
            try:
                # Extract text
                text = await extract_text(doc)
                
                # Generate embeddings
                embeddings = await openai_embed(text)
                
                # Store in vector DB
                await vector_store.upsert(embeddings)
                
                # Generate summary with GPT
                summary = await openai_complete(f"Summarize: {text}")
                
                return {"doc": doc, "summary": summary}
            except Exception as e:
                # What if this fails halfway through?
                # We've lost the embeddings work!
                logger.error(f"Failed processing {doc}: {e}")
                return None
    
    tasks = [process_single(doc) for doc in docs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # Now what? How do we retry failed docs? Track costs?
    return [r for r in results if r is not None]
\`\`\`

**With Puffinflow:**
\`\`\`python
# 😌 Clean, maintainable, and resilient
@state(cpu=1.0, memory=512, max_retries=3)
async def extract_text(context):
    doc = context.get_variable("document")
    text = await extract_text_api(doc)
    context.set_variable("text", text)
    return "generate_embeddings"

@state(cpu=0.5, memory=256, rate_limit=10)  # 10 calls/sec to stay under quota
async def generate_embeddings(context):
    text = context.get_variable("text")
    embeddings = await openai_embed(text)
    context.set_variable("embeddings", embeddings)
    return "store_embeddings"

@state(cpu=0.5, memory=256, max_retries=2)
async def store_embeddings(context):
    embeddings = context.get_variable("embeddings")
    await vector_store.upsert(embeddings)
    return "generate_summary"

@state(cpu=1.0, memory=512, rate_limit=5, timeout=60.0)
async def generate_summary(context):
    text = context.get_variable("text")
    summary = await openai_complete(f"Summarize: {text}")
    context.set_variable("summary", summary)
    # Automatic checkpointing means partial work is never lost!

# Process 100s of documents safely
for doc in documents:
    agent = Agent(f"process-{doc.id}")
    agent.add_state("extract_text", extract_text)
    agent.add_state("generate_embeddings", generate_embeddings) 
    agent.add_state("store_embeddings", store_embeddings)
    agent.add_state("generate_summary", generate_summary)
    
    # Each agent runs independently with built-in fault tolerance
    asyncio.create_task(agent.run(initial_context={"document": doc}))
\`\`\`

## When to Choose Puffinflow

### ✅ Perfect for:

**AI & LLM Applications:**
- **Multi-step LLM chains** with tight token budgets and API quotas (GPT-4, Claude, etc.)
- **RAG pipelines** that embed, search, and generate responses
- **Document processing** with extraction, analysis, and summarization
- **Agent swarms** running hundreds of concurrent autonomous agents
- **Vector database workflows** with embeddings, search, and updates

**Resource-Constrained Environments:**
- **Exact resumption after interruption** (cloud pre-emptible nodes, CI jobs, lambda timeouts)
- **Cost-sensitive workloads** needing precise token and API call tracking
- **Rate-limited APIs** requiring intelligent back-off and queue management
- **Memory-constrained environments** needing fine-grained resource allocation

**Complex Orchestration Needs:**
- **Multi-agent coordination** where agents need to share state and synchronize
- **Typed shared memory** to avoid prompt-format drift between states
- **Fault-tolerant pipelines** that gracefully handle partial failures
- **Observable workflows** requiring detailed metrics and monitoring

### ✅ Great for:

**Team Productivity:**
- **Rapid prototyping** of AI workflows without infrastructure overhead
- **Production deployments** that need reliability without complexity
- **Teams transitioning** from custom asyncio solutions to structured orchestration
- **Projects requiring** deterministic, reproducible execution across environments

**Technical Scenarios:**
- **Async-heavy workloads** with lots of I/O (API calls, database operations)
- **Event-driven workflows** that react to external triggers
- **Batch processing** that needs parallelization with resource controls
- **Microservice orchestration** for coordinating multiple services

### ✅ Consider Puffinflow if:

- You're spending more time debugging async code than building features
- You need Airflow-like DAG capabilities but with better async support
- Your team wants to focus on AI logic, not infrastructure plumbing
- You're tired of reinventing workflow orchestration for every project
- You need built-in observability without external dependencies

### ❌ Not ideal for:

**Simple Use Cases:**
- **Single-function scripts** that don't need orchestration
- **Synchronous, sequential workloads** without async operations
- **Basic automation** that can be handled with cron jobs

**Different Problem Domains:**
- **Real-time streaming** applications (use Kafka Streams, Apache Flink)
- **Traditional ETL pipelines** with batch scheduling (use Airflow, dbt)
- **High-frequency trading** or ultra-low latency applications
- **Simple web applications** that don't need workflow orchestration

**Resource-Unlimited Environments:**
- **Workflows that don't care about** API costs, rate limits, or resource usage
- **Teams that prefer** building custom solutions from scratch
- **Legacy systems** that can't adopt async Python patterns

### 🤔 Still Not Sure?

**Try Puffinflow if you've ever said:**
- "This async code is getting out of hand"
- "We're getting rate limited by OpenAI again"
- "The cloud instance got preempted and we lost 3 hours of work"
- "I can't debug which part of the AI pipeline failed"
- "Our token costs are spiraling out of control"
- "We need better coordination between our agents"

**Stick with your current solution if:**
- Your current async orchestration works perfectly
- You don't need resource management or rate limiting
- Your workflows are simple and never fail
- You don't mind building observability from scratch

## Quick Example: AI Research Assistant

Here's a complete AI workflow that demonstrates Puffinflow's key features. This example shows how to build a research assistant that gathers information, analyzes it with an LLM, and generates a comprehensive report:

\`\`\`python
import asyncio
from puffinflow import Agent, state
from puffinflow.observability import MetricsCollector

# Initialize metrics collection
metrics = MetricsCollector(namespace="research_assistant")

# Create the research agent
agent = Agent("research-assistant")

@state(cpu=1.0, memory=512, timeout=30.0, max_retries=2)
async def gather_info(context):
    """Search for information on the web."""
    query = context.get_variable("search_query")
    
    # Track the operation
    metrics.increment("searches_started")
    
    with metrics.timer("search_duration"):
        # Simulate web search API call
        await asyncio.sleep(0.1)  # Simulate network delay
        results = [
            {"title": f"Article about {query}", "content": f"Detailed content about {query}..."},
            {"title": f"{query} - Latest Research", "content": f"Recent findings on {query}..."},
            {"title": f"Industry Analysis: {query}", "content": f"Market analysis of {query}..."}
        ]
    
    # Store results in context for next state
    context.set_variable("raw_results", results)
    context.set_variable("search_count", len(results))
    
    metrics.gauge("articles_found", len(results))
    metrics.increment("searches_completed")
    
    print(f"✅ Found {len(results)} articles about '{query}'")
    return "analyze_results"

@state(cpu=2.0, memory=1024, timeout=60.0, rate_limit=5)  # Rate limit LLM calls
async def analyze_results(context):
    """Use LLM to analyze the gathered information."""
    results = context.get_variable("raw_results")
    query = context.get_variable("search_query")
    
    metrics.increment("llm_calls_started")
    
    with metrics.timer("llm_analysis_duration"):
        # Simulate LLM API call (GPT-4, Claude, etc.)
        await asyncio.sleep(0.5)  # Simulate LLM processing time
        
        # Create comprehensive analysis
        topics = [f"topic_{i+1}" for i in range(len(results))]
        analysis = {
            "summary": f"Comprehensive analysis of {len(results)} articles about {query}",
            "key_topics": topics,
            "sentiment": "neutral",
            "confidence": 0.92,
            "word_count": sum(len(r["content"]) for r in results),
            "sources_analyzed": len(results)
        }
    
    # Store analysis results
    context.set_variable("analysis", analysis)
    
    # Track metrics
    metrics.gauge("analysis_confidence", analysis["confidence"])
    metrics.gauge("sources_analyzed", analysis["sources_analyzed"])
    metrics.increment("llm_calls_completed")
    
    print(f"🧠 Analysis complete: {analysis['confidence']:.0%} confidence")
    return "generate_report"

@state(cpu=1.0, memory=512, timeout=45.0, max_retries=1)
async def generate_report(context):
    """Generate the final research report."""
    query = context.get_variable("search_query")
    analysis = context.get_variable("analysis")
    raw_results = context.get_variable("raw_results")
    
    metrics.increment("reports_started")
    
    with metrics.timer("report_generation_duration"):
        # Create structured report
        report = {
            "title": f"Research Report: {query.title()}",
            "query": query,
            "executive_summary": analysis["summary"],
            "key_findings": analysis["key_topics"],
            "sentiment_analysis": analysis["sentiment"],
            "confidence_score": analysis["confidence"],
            "methodology": {
                "sources_searched": len(raw_results),
                "sources_analyzed": analysis["sources_analyzed"],
                "analysis_method": "LLM-powered content analysis"
            },
            "metadata": {
                "generated_at": "2024-01-15T10:30:00Z",
                "agent_id": agent.name,
                "word_count": analysis["word_count"]
            }
        }
    
    # Store final report
    context.set_variable("final_report", report)
    context.set_output("research_report", report)  # Mark as workflow output
    
    # Final metrics
    metrics.gauge("report_word_count", analysis["word_count"])
    metrics.increment("reports_completed")
    
    print(f"📊 Report generated: '{report['title']}'")
    print(f"   Sources: {report['methodology']['sources_analyzed']}")
    print(f"   Confidence: {report['confidence_score']:.0%}")
    
    return None  # End of workflow

# Wire up the workflow with dependencies
agent.add_state("gather_info", gather_info)
agent.add_state("analyze_results", analyze_results)
agent.add_state("generate_report", generate_report)

# Example usage
async def run_research(query: str):
    """Run a complete research workflow."""
    print(f"🚀 Starting research on: '{query}'")
    print("=" * 50)
    
    # Run the agent with initial context
    result = await agent.run(initial_context={"search_query": query})
    
    # Get the final report
    report = result.get_output("research_report")
    
    print("=" * 50)
    print("📋 Research Complete!")
    print(f"Title: {report['title']}")
    print(f"Confidence: {report['confidence_score']:.0%}")
    print(f"Sources: {report['methodology']['sources_analyzed']}")
    
    return report

# Run multiple research queries concurrently
async def main():
    queries = [
        "machine learning trends 2024",
        "sustainable energy solutions",
        "remote work productivity tools"
    ]
    
    # Process all queries in parallel
    tasks = [run_research(query) for query in queries]
    reports = await asyncio.gather(*tasks)
    
    print(f"\\n🎉 Completed {len(reports)} research reports!")
    
    # Print metrics summary
    print("\\n📊 Metrics Summary:")
    print(f"   Total searches: {metrics.get_counter('searches_completed')}")
    print(f"   Total LLM calls: {metrics.get_counter('llm_calls_completed')}")
    print(f"   Total reports: {metrics.get_counter('reports_completed')}")

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

**Expected Output:**
\`\`\`
🚀 Starting research on: 'machine learning trends 2024'
==================================================
✅ Found 3 articles about 'machine learning trends 2024'
🧠 Analysis complete: 92% confidence
📊 Report generated: 'Research Report: Machine Learning Trends 2024'
   Sources: 3
   Confidence: 92%
==================================================
📋 Research Complete!
Title: Research Report: Machine Learning Trends 2024
Confidence: 92%
Sources: 3

🎉 Completed 3 research reports!

📊 Metrics Summary:
   Total searches: 3
   Total LLM calls: 3
   Total reports: 3
\`\`\`

**🔥 What This Example Demonstrates:**

1. **Resource Management**: Each state specifies CPU, memory, and timeout requirements
2. **Rate Limiting**: LLM calls are rate-limited to respect API quotas
3. **Error Handling**: Automatic retries for transient failures
4. **Data Flow**: Clean data passing between states via context
5. **Observability**: Built-in metrics tracking for performance monitoring
6. **Concurrent Execution**: Multiple research workflows running in parallel
7. **Type Safety**: Structured data with clear contracts between states
8. **Production Ready**: Comprehensive error handling and resource management

**🚀 Ready to Build?**

This is just a taste of what Puffinflow can do. The framework handles all the complex orchestration, resource management, and observability while you focus on your AI logic.

**Next Steps:**
- **[🚀 Get Started →](#docs/getting-started)** - Build your first workflow in 5 minutes
- **[📚 Context & Data →](#docs/context-and-data)** - Master data flow between states  
- **[⚡ Resource Management →](#docs/resource-management)** - Control CPU, memory, and rate limits
- **[🔧 Error Handling →](#docs/error-handling)** - Build resilient workflows
- **[📊 Observability →](#docs/observability)** - Monitor and debug your workflows

Ready to tame your async chaos? Let's dive in! 🐧
`.trim();
