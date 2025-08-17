"""Test to debug state execution order and potential race conditions."""
import asyncio

from puffinflow import Agent


async def main():
    agent = Agent("state-order-debug")

    execution_order = []

    async def initialize(context):
        execution_order.append("initialize_start")
        print("Initialize state running")
        await asyncio.sleep(0.1)  # Small delay to see ordering
        context.set_variable("initialized", True)
        execution_order.append("initialize_end")
        return None

    async def update(context):
        execution_order.append("update_start")
        print("Update state running")

        # This is where the issue might occur - accessing a variable
        # that might not be set yet if update runs before initialize
        initialized = context.get_variable("initialized", False)
        print(f"Update sees initialized: {initialized}")

        if not initialized:
            print("ERROR: Update state ran before initialize!")
            raise ValueError("Update state ran before initialize completed")

        context.set_variable("updated", True)
        execution_order.append("update_end")
        return None

    # Add states without dependencies - order matters here
    agent.add_state("initialize", initialize)
    agent.add_state("update", update)

    print("Starting agent run...")
    print(f"States in order: {list(agent.states.keys())}")

    try:
        result = await agent.run()
        print(f"Agent status: {result.status}")
        print(f"Execution order: {execution_order}")
        print(f"Initialized: {result.get_variable('initialized')}")
        print(f"Updated: {result.get_variable('updated')}")
    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"Execution order: {execution_order}")


if __name__ == "__main__":
    asyncio.run(main())
