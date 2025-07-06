export const introductionMarkdown = `
# Puffinflow Agent Framework

A lightweight Python framework for orchestrating AI agents and data workflows with deterministic, resource-aware execution built for today's AI-first engineering teams.

## What is Puffinflow?

Puffinflow is inspired by Airflow-style DAGs but designed specifically for modern LLM stacks. Think of it as **Airflow-style wiring for async functions**—but trimmed down to what you actually need when you're juggling OpenAI calls, scraping, vector-store writes, or any other I/O-heavy jobs.

### The Problem
If you've ever tried to stitch together a handful of OpenAI calls, a scraping routine, a vector-store write, and a Slack notification—all while tip-toeing around \`async\` race conditions—you already know why Puffinflow exists.

### The Solution
Whether you're orchestrating OpenAI calls, vector-store pipelines, or long-running autonomous agents, Puffinflow provides the scaffolding so you can focus on domain logic instead of infrastructure concerns.

**Key Benefits:**
- **🚀 Simple**: Define states as async functions, wire them together
- **🔒 Safe**: Built-in context management prevents race conditions
- **⚡ Fast**: Optimized for high-concurrency AI workloads
- **🛡️ Reliable**: Automatic checkpointing and recovery
- **📊 Observable**: Rich metrics and monitoring out of the box

## Why Another Workflow Tool?
| Your Headache                                     | How Puffinflow Helps                                                                               |
|-----------------------------------------------------|----------------------------------------------------------------------------------------------------|
| Async spaghetti – callback hell, tangled asyncio tasks | Register tiny, focused states; Puffinflow's scheduler runs them safely and in order                |
| Global variables & race-conditions                  | A built-in, type-locked Context lets every step pass data without the foot-guns                      |
| "Rate limit exceeded" from day-one                  | Opt-in rate-limit helpers keep you under OpenAI or vendor quotas—without manual back-off logic         |
| Cloud pre-emptions wiping work                      | One-liner checkpoints freeze progress so you can resume exactly where you left off                   |

## When to Choose Puffinflow

### ✅ Perfect for:
- **Multi-step LLM chains** with tight token budgets and API quotas
- **Hundreds of concurrent autonomous agents** that coordinate through shared resources
- **Exact resumption after interruption** (cloud pre-emptible nodes, CI jobs)
- **Typed shared memory** to avoid prompt-format drift between states

### ✅ Great for:
- Complex agent workflows with dependencies and coordination
- Resource-constrained environments needing quota management
- Teams that want Airflow-like orchestration without the operational overhead
- Projects requiring deterministic, reproducible execution

### ❌ Not ideal for:
- Simple scripts that don't need orchestration
- Synchronous, non-concurrent workloads
- Traditional ETL pipelines (use Airflow instead)
- Real-time streaming applications

## Quick Example

Here's what a simple AI workflow looks like:

\`\`\`python
from puffinflow import Agent

agent = Agent("research-assistant")

@agent.state
async def gather_info(context):
    query = context.get_variable("search_query")
    results = await search_web(query)
    context.set_variable("raw_results", results)
    return "analyze_results"

@agent.state
async def analyze_results(context):
    results = context.get_variable("raw_results")
    analysis = await llm.analyze(results)
    context.set_variable("analysis", analysis)
    return "generate_report"

@agent.state
async def generate_report(context):
    analysis = context.get_variable("analysis")
    report = await llm.generate_report(analysis)
    context.set_variable("final_report", report)
    return None  # End of workflow

# Run it
await agent.run(initial_context={"search_query": "latest AI trends"})
\`\`\`

Ready to tame your async chaos? Let's dive in! 🐧
`.trim();
