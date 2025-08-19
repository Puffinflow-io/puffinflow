export const gettingStartedMarkdown = `# Getting Started with Puffinflow

Puffinflow turns your Python functions into robust, fault-tolerant workflows. Perfect for AI pipelines, data processing, and any multi-step async work that needs reliability.

## Installation

\`\`\`bash
pip install puffinflow
\`\`\`

## Core Concept

**Agent**: Your workflow orchestrator
**States**: Individual steps (just async Python functions)
**Context**: Shared data between states

## Your First Workflow

Create a simple 3-step data processing workflow:

\`\`\`python
import asyncio
from puffinflow import Agent, state

# Create an agent
agent = Agent("data-processor")

@state
async def fetch_data(context):
    """Step 1: Get some data"""
    data = {"users": ["Alice", "Bob", "Charlie"]}
    context.set_variable("raw_data", data)
    return "process_data"

@state
async def process_data(context):
    """Step 2: Transform the data"""
    raw_data = context.get_variable("raw_data")
    processed = [f"Hello, {user}!" for user in raw_data["users"]]
    context.set_variable("greetings", processed)
    return "save_results"

@state
async def save_results(context):
    """Step 3: Output results"""
    greetings = context.get_variable("greetings")
    print("Results:")
    for greeting in greetings:
        print(f"  {greeting}")
    # Return None to end the workflow
    return None

# Add states to agent
agent.add_state("fetch_data", fetch_data)
agent.add_state("process_data", process_data)
agent.add_state("save_results", save_results)

# Run it
async def main():
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

**Output:**
\`\`\`
Results:
  Hello, Alice!
  Hello, Bob!
  Hello, Charlie!
\`\`\`

## How It Works

1. **Define states** with \`@agent.state\` - each state does one thing
2. **Share data** using \`context.set_variable()\` and \`context.get_variable()\`
3. **Control flow** by returning the name of the next state (or \`None\` to end)
4. **Run the workflow** with \`agent.run(initial_state="start_state")\`

## Alternative: Without Decorators

If you prefer not using decorators:

\`\`\`python
async def my_function(context):
    print("Hello from Puffinflow!")
    return None

agent = Agent("simple-workflow")
agent.add_state("hello", my_function)

await agent.run()
\`\`\`

## Execution Modes

Puffinflow supports two execution modes to control how your workflow runs:

### Sequential Mode (Default)
States run one after another in a linear flow:

\`\`\`python
from puffinflow import Agent, ExecutionMode, state

agent = Agent("sequential-workflow")

@state
async def step_one(context):
    context.set_variable("step", 1)
    return "step_two"  # Explicitly control next step

@state
async def step_two(context):
    context.set_variable("step", 2)
    # End workflow

agent.add_state("step_one", step_one)
agent.add_state("step_two", step_two)

await agent.run(execution_mode=ExecutionMode.SEQUENTIAL)
\`\`\`

### Parallel Mode
All independent states run concurrently for maximum performance:

\`\`\`python
agent = Agent("parallel-workflow")

@state
async def fetch_users(context):
    # This runs in parallel with fetch_orders
    context.set_variable("users", ["Alice", "Bob"])

@state
async def fetch_orders(context):
    # This runs in parallel with fetch_users
    context.set_variable("orders", [{"id": 1}, {"id": 2}])

@state
async def generate_report(context):
    # Waits for both parallel states to complete
    users = context.get_variable("users")
    orders = context.get_variable("orders")
    context.set_variable("report", f"Report: {len(users)} users, {len(orders)} orders")

agent.add_state("fetch_users", fetch_users)
agent.add_state("fetch_orders", fetch_orders)
agent.add_state("generate_report", generate_report, dependencies=["fetch_users", "fetch_orders"])

await agent.run(execution_mode=ExecutionMode.PARALLEL)
\`\`\`

Use **sequential mode** for linear workflows and **parallel mode** when you have independent operations that can run concurrently.

## Next Steps

Now that you have a working workflow, explore:

- **[Error Handling](/docs/error-handling)** - Add retries and fault tolerance
- **[Context & Data](/docs/context-and-data)** - Advanced data sharing patterns
- **[Examples](https://github.com/puffinflow/examples)** - Real-world workflow examples

Ready to build something robust? 🐧
`.trim();
