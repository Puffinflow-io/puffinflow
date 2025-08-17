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
from puffinflow import Agent

# Create an agent
agent = Agent("data-processor")

@agent.state
async def fetch_data(context):
    """Step 1: Get some data"""
    data = {"users": ["Alice", "Bob", "Charlie"]}
    context.set_variable("raw_data", data)
    return "process_data"

@agent.state  
async def process_data(context):
    """Step 2: Transform the data"""
    raw_data = context.get_variable("raw_data")
    processed = [f"Hello, {user}!" for user in raw_data["users"]]
    context.set_variable("greetings", processed)
    return "save_results"

@agent.state
async def save_results(context):
    """Step 3: Output results"""
    greetings = context.get_variable("greetings")
    print("Results:")
    for greeting in greetings:
        print(f"  {greeting}")
    # Return None to end the workflow
    return None

# Run it
async def main():
    await agent.run(initial_state="fetch_data")

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

await agent.run(initial_state="hello")
\`\`\`

## Next Steps

Now that you have a working workflow, explore:

- **[Error Handling](/docs/error-handling)** - Add retries and fault tolerance
- **[Context & Data](/docs/context-and-data)** - Advanced data sharing patterns  
- **[Examples](https://github.com/puffinflow/examples)** - Real-world workflow examples

Ready to build something robust? 🐧
`.trim();
