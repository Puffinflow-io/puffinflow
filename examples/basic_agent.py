"""
Basic Agent Examples

This module demonstrates the fundamental concepts of PuffinFlow agents:
- Creating simple agents with state decorators
- Basic state execution and context management
- Resource allocation and lifecycle management
"""

import asyncio
import time
from typing import Optional

from puffinflow import Agent, Context, cpu_intensive, memory_intensive, state


class SimpleAgent(Agent):
    """A simple agent that demonstrates basic functionality."""

    def __init__(self, name: str):
        super().__init__(name)
        # Register all decorated states
        self.add_state("initialize", self.initialize)
        self.add_state("setup", self.setup)
        self.add_state("process", self.process)
        self.add_state("finalize", self.finalize)

    @state(cpu=1.0, memory=256.0, priority="normal")
    async def initialize(self, context: Context):
        """Initialize the agent with some basic setup."""
        context.set_output("status", "initialized")
        context.set_metadata("init_time", time.time())
        self.set_variable("initialized", True)
        print(f"Agent {self.name} initialized")
        return "setup"

    @state(cpu=2.0, memory=512.0, priority="high")
    async def setup(self, context: Context):
        """Setup phase with higher resource requirements."""
        if not self.get_variable("initialized", False):
            raise ValueError("Agent not initialized")

        # Simulate some setup work
        await asyncio.sleep(0.1)

        context.set_output("setup_complete", True)
        context.set_metadata("setup_duration", 0.1)
        print(f"Agent {self.name} setup complete")
        return "process"

    @cpu_intensive(cpu=4.0, memory=1024.0)
    async def process(self, context: Context):
        """CPU-intensive processing state."""
        # Simulate CPU-intensive work
        start_time = time.time()

        # Some computational work
        result = sum(i * i for i in range(10000))

        duration = time.time() - start_time
        context.set_output("computation_result", result)
        context.set_metadata("processing_time", duration)

        print(f"Agent {self.name} processed data: {result}")
        return "finalize"

    @state(cpu=0.5, memory=128.0, priority="low")
    async def finalize(self, context: Context):
        """Finalization state with low resource requirements."""
        context.set_output("final_status", "completed")
        context.set_metadata("total_outputs", len(context.get_output_keys()))

        print(f"Agent {self.name} finalized")
        return None  # End of workflow


class DataProcessor(Agent):
    """An agent that processes data with different strategies."""

    def __init__(self, name: str, data: Optional[list] = None):
        super().__init__(name)
        # Register all decorated states
        self.add_state("validate_data", self.validate_data)
        self.add_state("process_data", self.process_data)
        self.add_state("generate_report", self.generate_report)
        self.add_state("error", self.error)

        self.set_variable("data", data or [])

    @state(cpu=1.0, memory=256.0)
    async def validate_data(self, context: Context):
        """Validate input data."""
        data = self.get_variable("data", [])

        if not data:
            context.set_output("error", "No data provided")
            return "error"

        if not all(isinstance(x, (int, float)) for x in data):
            context.set_output("error", "Invalid data types")
            return "error"

        context.set_output("data_size", len(data))
        context.set_metadata("validation_time", time.time())
        print(f"Data validation passed: {len(data)} items")
        return "process_data"

    @memory_intensive(memory=2048.0, cpu=2.0)
    async def process_data(self, context: Context):
        """Process the validated data."""
        data = self.get_variable("data", [])

        # Simulate memory-intensive processing
        processed_data = []
        for item in data:
            # Some complex processing
            processed_item = item**2 + item * 0.5
            processed_data.append(processed_item)
            await asyncio.sleep(0.001)  # Simulate processing time

        self.set_variable("processed_data", processed_data)
        context.set_output("processed_count", len(processed_data))
        context.set_output("average_value", sum(processed_data) / len(processed_data))

        print(f"Processed {len(processed_data)} items")
        return "generate_report"

    @state(cpu=0.5, memory=512.0)
    async def generate_report(self, context: Context):
        """Generate a processing report."""
        processed_data = self.get_variable("processed_data", [])
        original_data = self.get_variable("data", [])

        report = {
            "original_count": len(original_data),
            "processed_count": len(processed_data),
            "min_value": min(processed_data) if processed_data else 0,
            "max_value": max(processed_data) if processed_data else 0,
            "sum_value": sum(processed_data),
        }

        context.set_output("report", report)
        print(f"Report generated: {report}")
        return None

    @state(cpu=0.1, memory=64.0)
    async def error(self, context: Context):
        """Handle errors."""
        error_msg = context.get_output("error", "Unknown error")
        print(f"Error occurred: {error_msg}")
        context.set_output("status", "failed")
        return None


async def run_simple_agent():
    """Run the simple agent example."""
    print("=== Running Simple Agent Example ===")

    agent = SimpleAgent("simple-agent")
    result = await agent.run()

    print(f"Agent completed with status: {result.status}")
    print(f"Outputs: {result.outputs}")
    print()


async def run_data_processor():
    """Run the data processor example."""
    print("=== Running Data Processor Example ===")

    # Test with valid data
    data = [1, 2, 3, 4, 5, 10, 15, 20]
    processor = DataProcessor("data-processor", data)
    result = await processor.run()

    print(f"Processor completed with status: {result.status}")
    print(f"Report: {result.get_output('report')}")
    print()

    # Test with invalid data
    print("=== Testing Error Handling ===")
    invalid_processor = DataProcessor("invalid-processor", ["invalid", "data"])
    error_result = await invalid_processor.run()

    print(f"Error processor status: {error_result.status}")
    print(f"Error message: {error_result.get_output('error')}")
    print()


async def run_agent_with_variables():
    """Demonstrate agent variable management."""
    print("=== Agent Variables Example ===")

    agent = SimpleAgent("variable-agent")

    # Set some variables before running
    agent.set_variable("custom_setting", "production")
    agent.set_variable("batch_size", 1000)
    agent.set_variable("debug_mode", False)

    print(f"Variables before run: {agent.get_variable('custom_setting')}")

    await agent.run()

    print(f"Initialized flag: {agent.get_variable('initialized')}")
    print(f"Custom setting: {agent.get_variable('custom_setting')}")
    print()


async def main():
    """Run all basic agent examples."""
    await run_simple_agent()
    await run_data_processor()
    await run_agent_with_variables()


if __name__ == "__main__":
    asyncio.run(main())
