"""Test to understand state execution order when no explicit transitions are defined."""
import asyncio

from puffinflow import Agent


async def main():
    agent = Agent("no-transitions-test")

    async def initialize(context):
        print("Initialize state running")
        context.set_variable("initialized", True)
        return None  # No explicit transitions

    async def update(context):
        print("Update state running")
        initialized = context.get_variable("initialized", False)
        print(f"Initialized: {initialized}")
        context.set_variable("updated", True)
        return None  # No explicit transitions

    # Add states without dependencies
    agent.add_state("initialize", initialize)
    agent.add_state("update", update)

    print("Starting agent run...")
    print(f"States: {list(agent.states.keys())}")
    print(f"Dependencies: {agent.dependencies}")

    result = await agent.run()

    print(f"Agent status: {result.status}")
    print(f"Initialized: {result.get_variable('initialized')}")
    print(f"Updated: {result.get_variable('updated')}")

    # Print state metadata
    for state_name in agent.states:
        metadata = agent.state_metadata.get(state_name)
        print(
            f"State {state_name}: status={metadata.status if metadata else 'unknown'}"
        )


if __name__ == "__main__":
    asyncio.run(main())
