#!/usr/bin/env python3
"""
Simple test script to verify PuffinFlow examples work correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Debug path information
print(f"Project root: {project_root}")
print(f"Src path: {src_path}")
print(f"Src path exists: {src_path.exists()}")


def test_basic_imports():
    """Test that basic PuffinFlow imports work."""
    print("Testing basic imports...")
    try:
        # Just test that imports work, don't need to store references
        from puffinflow import Agent, Context  # noqa: F401
        from puffinflow.core.agent import state  # noqa: F401

        print("✅ Basic imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_simple_agent():
    """Test creating and running a simple agent."""
    print("\nTesting simple agent creation...")
    try:
        import asyncio

        from puffinflow import Agent, Context
        from puffinflow.core.agent import state

        class TestAgent(Agent):
            def __init__(self, name):
                super().__init__(name)
                self.add_state("start", self.start)

            @state(cpu=1.0, memory=256.0)
            async def start(self, context: Context):
                context.set_output("message", "Hello from PuffinFlow!")
                return None

        async def run_test():
            agent = TestAgent("test-agent")
            result = await agent.run()
            return result.get_output("message") == "Hello from PuffinFlow!"

        success = asyncio.run(run_test())
        if success:
            print("✅ Simple agent test passed")
            return True
        else:
            print("❌ Simple agent test failed - wrong output")
            return False

    except Exception as e:
        print(f"❌ Simple agent test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("PuffinFlow Examples Test")
    print("=" * 40)

    tests = [
        test_basic_imports,
        test_simple_agent,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Examples are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed.")
        return 1


async def async_main():
    """Async wrapper for main."""
    return await main()


if __name__ == "__main__":
    exit_code = asyncio.run(async_main())
    sys.exit(exit_code)
