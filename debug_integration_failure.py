import asyncio
import logging

import pytest

from puffinflow import Agent, Context, ResourcePool, ResourceRequirements, state

# Add some debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DebugAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)
        self.add_state("debug_state", self.debug_state)

    @state(cpu=1.0, memory=256.0)
    async def debug_state(self, context: Context):
        logger.info("Debug state started")
        await asyncio.sleep(0.01)
        return None


@pytest.mark.asyncio
async def test_debug_resource_types():
    """Debug test to understand ResourceType enum issues."""

    # Test 1: Check if ResourceType imports correctly
    try:
        from puffinflow import ResourceType

        print(f"ResourceType class: {ResourceType}")
        print(f"ResourceType.CPU: {ResourceType.CPU}")
        print(f"ResourceType.MEMORY: {ResourceType.MEMORY}")
        print(f"Type of ResourceType.CPU: {type(ResourceType.CPU)}")

        # Test bitwise operations directly
        result = ResourceType.CPU & ResourceType.MEMORY
        print(f"CPU & MEMORY = {result}")

        result2 = ResourceType.ALL & ResourceType.CPU
        print(f"ALL & CPU = {result2}")

    except Exception as e:
        print(f"Error importing or using ResourceType: {e}")
        print(f"Error type: {type(e)}")
        import traceback

        traceback.print_exc()

    # Test 2: Check ResourceRequirements creation
    try:
        req = ResourceRequirements(cpu_units=1.0, memory_mb=256.0)
        print(f"ResourceRequirements: {req}")
        print(f"resource_types: {req.resource_types}")
        print(f"Type of resource_types: {type(req.resource_types)}")

        # Test bitwise operation on requirements
        test_result = req.resource_types & ResourceType.CPU
        print(f"req.resource_types & CPU = {test_result}")

    except Exception as e:
        print(f"Error with ResourceRequirements: {e}")
        import traceback

        traceback.print_exc()

    # Test 3: Check state decorator
    try:
        agent = DebugAgent("debug-agent")

        # Check if the state has resource requirements
        state_func = agent.states["debug_state"]
        if hasattr(state_func, "_resource_requirements"):
            req = state_func._resource_requirements
            print(f"State requirements: {req}")
            print(f"State resource_types: {req.resource_types}")
            print(f"Type of state resource_types: {type(req.resource_types)}")
        else:
            print("State function has no _resource_requirements attribute")

    except Exception as e:
        print(f"Error with state decorator: {e}")
        import traceback

        traceback.print_exc()

    # Test 4: Try to run the agent
    try:
        agent = DebugAgent("debug-agent")
        resource_pool = ResourcePool(total_cpu=4.0, total_memory=1024.0)
        agent.resource_pool = resource_pool

        result = await agent.run()
        print(f"Agent run result: {result.status}")

        if result.error:
            print(f"Agent error: {result.error}")

        dead_letters = agent.get_dead_letters()
        if dead_letters:
            print(f"Dead letters: {dead_letters}")
            for dl in dead_letters:
                print(f"  Error: {dl.error_message}")
                print(f"  Type: {dl.error_type}")

    except Exception as e:
        print(f"Error running agent: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_debug_resource_types())
