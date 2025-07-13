"""Debug dynamic flow control."""
import asyncio

from puffinflow import Agent


async def main():
    agent = Agent("dynamic-flow-test")

    async def check_user_type(context):
        print("🔍 Checking user type...")
        user_type = "premium"
        context.set_variable("user_type", user_type)

        if user_type == "premium":
            print("  -> Returning premium_flow")
            return "premium_flow"
        else:
            print("  -> Returning basic_flow")
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

    print("Entry states:", agent._find_entry_states())
    print("Dependencies:", agent.dependencies)

    result = await agent.run()

    print(f"\nCompleted states: {result.metadata['states_completed']}")
    print(f"User type: {result.get_variable('user_type')}")
    print(f"Features: {result.get_variable('features')}")


if __name__ == "__main__":
    asyncio.run(main())
