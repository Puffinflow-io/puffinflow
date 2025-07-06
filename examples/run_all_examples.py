"""
Example Test Runner

This script runs all PuffinFlow examples and verifies they work correctly.
It provides a comprehensive test of the framework's functionality.
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path

# Add the src directory to the path so we can import puffinflow
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import example modules
try:
    import advanced_workflows
    import basic_agent
    import coordination_examples
    import observability_demo
    import reliability_patterns
    import resource_management
except ImportError as e:
    print(f"Failed to import example modules: {e}")
    print("Make sure you're running from the examples directory")
    sys.exit(1)


class ExampleRunner:
    """Runner for executing and testing all examples."""

    def __init__(self):
        self.results = {}
        self.total_time = 0
        self.start_time = None

    async def run_example(self, name: str, example_main_func):
        """Run a single example and capture results."""
        print(f"\n{'='*60}")
        print(f"Running {name}")
        print(f"{'='*60}")

        start_time = time.time()
        success = False
        error_msg = None

        try:
            # Run the example's main function
            await example_main_func()
            success = True
            print(f"✅ {name} completed successfully")

        except Exception as e:
            error_msg = str(e)
            print(f"❌ {name} failed: {error_msg}")
            print(f"Traceback: {traceback.format_exc()}")

        execution_time = time.time() - start_time

        self.results[name] = {
            "success": success,
            "execution_time": execution_time,
            "error": error_msg,
        }

        print(f"Execution time: {execution_time:.2f} seconds")
        return success

    async def run_all_examples(self):
        """Run all examples in sequence."""
        self.start_time = time.time()

        print("PuffinFlow Examples Test Runner")
        print("=" * 60)
        print("Testing all examples to verify functionality...")

        # Define examples to run
        examples = [
            ("Basic Agent", basic_agent.main),
            ("Coordination Examples", coordination_examples.main),
            ("Resource Management", resource_management.main),
            ("Reliability Patterns", reliability_patterns.main),
            ("Observability Demo", observability_demo.main),
            ("Advanced Workflows", advanced_workflows.main),
        ]

        # Run each example
        successful_examples = 0
        for name, main_func in examples:
            success = await self.run_example(name, main_func)
            if success:
                successful_examples += 1

        self.total_time = time.time() - self.start_time

        # Generate summary report
        self.generate_summary_report(successful_examples, len(examples))

        return successful_examples == len(examples)

    def generate_summary_report(self, successful: int, total: int):
        """Generate a comprehensive summary report."""
        print(f"\n{'='*60}")
        print("EXAMPLE TEST SUMMARY")
        print(f"{'='*60}")

        print(f"Total Examples: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(f"Success Rate: {successful / total * 100:.1f}%")
        print(f"Total Execution Time: {self.total_time:.2f} seconds")

        print("\nDetailed Results:")
        print("-" * 60)

        for name, result in self.results.items():
            status = "✅ PASS" if result["success"] else "❌ FAIL"
            time_str = f"{result['execution_time']:.2f}s"
            print(f"{status:<8} {name:<25} {time_str:>8}")

            if not result["success"] and result["error"]:
                print(f"         Error: {result['error']}")

        print("-" * 60)

        # Performance analysis
        if self.results:
            avg_time = sum(r["execution_time"] for r in self.results.values()) / len(
                self.results
            )
            fastest = min(self.results.items(), key=lambda x: x[1]["execution_time"])
            slowest = max(self.results.items(), key=lambda x: x[1]["execution_time"])

            print("\nPerformance Analysis:")
            print(f"Average execution time: {avg_time:.2f}s")
            print(
                f"Fastest example: {fastest[0]} ({fastest[1]['execution_time']:.2f}s)"
            )
            print(
                f"Slowest example: {slowest[0]} ({slowest[1]['execution_time']:.2f}s)"
            )

        # Recommendations
        print("\nRecommendations:")
        failed_examples = [
            name for name, result in self.results.items() if not result["success"]
        ]

        if not failed_examples:
            print("🎉 All examples passed! PuffinFlow is working correctly.")
            print("   You can now use these examples as templates for your own agents.")
        else:
            print("⚠️  Some examples failed. Please check:")
            for name in failed_examples:
                print(f"   - {name}: {self.results[name]['error']}")
            print("   - Ensure all dependencies are installed")
            print("   - Check that PuffinFlow is properly configured")
            print("   - Review the error messages above for specific issues")

        print(f"\n{'='*60}")


async def run_quick_test():
    """Run a quick test to verify basic functionality."""
    print("Running quick functionality test...")

    try:
        # Test basic imports
        import puffinflow
        from puffinflow import Agent, Context, state

        print(f"✅ PuffinFlow {puffinflow.__version__} imported successfully")

        # Test basic agent creation
        class TestAgent(Agent):
            @state(cpu=1.0, memory=256.0)
            async def test_state(self, context: Context):
                context.set_output("test_result", "success")
                return None

        agent = TestAgent("test-agent")
        result = await agent.run()

        if result.get_output("test_result") == "success":
            print("✅ Basic agent functionality working")
            return True
        else:
            print("❌ Basic agent test failed")
            return False

    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


async def main():
    """Main function to run all tests."""
    print("PuffinFlow Examples Test Suite")
    print("=" * 60)

    # Run quick test first
    quick_test_passed = await run_quick_test()

    if not quick_test_passed:
        print("\n❌ Quick test failed. Please check your PuffinFlow installation.")
        print("   Try: pip install -e .")
        return False

    print("\n✅ Quick test passed. Running full example suite...\n")

    # Run all examples
    runner = ExampleRunner()
    all_passed = await runner.run_all_examples()

    # Exit with appropriate code
    if all_passed:
        print("\n🎉 All examples completed successfully!")
        return True
    else:
        print("\n⚠️  Some examples failed. Check the summary above.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test runner failed: {e}")
        traceback.print_exc()
        sys.exit(1)
