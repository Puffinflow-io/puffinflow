"""Test simple dependency to understand how it works."""
import asyncio

from puffinflow import Agent


async def main():
    agent = Agent("dep-test")

    async def state1(context):
        print("State 1 running")
        context.set_variable("state1_done", True)
        return None

    async def state2(context):
        print("State 2 running")
        state1_done = context.get_variable("state1_done")
        print(f"State 1 done: {state1_done}")
        context.set_variable("state2_done", True)
        return None

    agent.add_state("state1", state1)
    agent.add_state("state2", state2, dependencies=["state1"])

    print("Starting agent run...")
    result = await agent.run()

    print(f"State 1 done: {result.get_variable('state1_done')}")
    print(f"State 2 done: {result.get_variable('state2_done')}")


if __name__ == "__main__":
    asyncio.run(main())
