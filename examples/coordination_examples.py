"""
Coordination Examples

This example demonstrates multi-agent coordination patterns:
- Agent teams and collaboration
- Parallel and sequential execution
- Message passing between agents
- Event-driven coordination
"""

import asyncio
import time
from typing import Optional

from puffinflow import (
    Agent,
    Context,
    cpu_intensive,
    create_team,
    memory_intensive,
    run_agents_parallel,
    run_agents_sequential,
    state,
)


class DataCollector(Agent):
    """Agent that collects data from various sources."""

    def __init__(self, name: str):
        super().__init__(name)
        # Register all decorated states
        self.add_state("collect_data", self.collect_data)

    @state(cpu=1.0, memory=256.0)
    async def collect_data(self, context: Context):
        """Collect data from external sources."""
        # Simulate data collection
        await asyncio.sleep(0.2)

        data = {
            "source": self.name,
            "timestamp": time.time(),
            "records": list(range(100)),
            "metadata": {"quality": "high", "format": "json"},
        }

        context.set_output("collected_data", data)
        context.set_metric("collection_time", 0.2)

        print(f"{self.name} collected {len(data['records'])} records")
        return None


class DataProcessor(Agent):
    """Agent that processes collected data."""

    def __init__(self, name: str, processing_type: str = "standard"):
        super().__init__(name)
        self.set_variable("processing_type", processing_type)
        # Register all decorated states
        self.add_state("process_data", self.process_data)

    @cpu_intensive(cpu=2.0, memory=512.0)
    async def process_data(self, context: Context):
        """Process the input data."""
        processing_type = self.get_variable("processing_type", "standard")

        # Simulate different processing strategies
        if processing_type == "fast":
            await asyncio.sleep(0.1)
            multiplier = 1.0
        elif processing_type == "accurate":
            await asyncio.sleep(0.3)
            multiplier = 1.5
        else:  # standard
            await asyncio.sleep(0.2)
            multiplier = 1.2

        # Simulate processing
        processed_count = int(100 * multiplier)

        result = {
            "processor": self.name,
            "type": processing_type,
            "processed_count": processed_count,
            "quality_score": 0.95 if processing_type == "accurate" else 0.85,
            "timestamp": time.time(),
        }

        context.set_output("processed_data", result)
        context.set_metric("processing_efficiency", multiplier)

        print(f"{self.name} processed data using {processing_type} method")
        return None


class DataAggregator(Agent):
    """Agent that aggregates results from multiple processors."""

    def __init__(self, name: str):
        super().__init__(name)
        # Register all decorated states
        self.add_state("aggregate_results", self.aggregate_results)

    @memory_intensive(memory=1024.0, cpu=1.5)
    async def aggregate_results(self, context: Context):
        """Aggregate results from multiple sources."""
        # In a real scenario, this would receive data from other agents
        # For demo purposes, we'll simulate aggregation

        await asyncio.sleep(0.15)

        aggregated_result = {
            "total_records": 300,  # Sum from multiple processors
            "average_quality": 0.88,
            "processing_time": time.time(),
            "sources": ["collector-1", "collector-2", "collector-3"],
        }

        context.set_output("aggregated_data", aggregated_result)
        context.set_metric("aggregation_efficiency", 0.95)

        print(
            f"{self.name} aggregated data from {len(aggregated_result['sources'])} sources"
        )
        return None


class ReportGenerator(Agent):
    """Agent that generates final reports."""

    def __init__(self, name: str):
        super().__init__(name)
        # Register all decorated states
        self.add_state("generate_report", self.generate_report)

    @state(cpu=0.5, memory=256.0)
    async def generate_report(self, context: Context):
        """Generate a comprehensive report."""
        await asyncio.sleep(0.1)

        report = {
            "report_id": f"report_{int(time.time())}",
            "generated_by": self.name,
            "summary": {
                "total_processed": 300,
                "quality_score": 0.88,
                "processing_efficiency": 0.95,
            },
            "recommendations": [
                "Consider using accurate processing for critical data",
                "Monitor quality scores regularly",
                "Optimize aggregation pipeline",
            ],
            "generated_at": time.time(),
        }

        context.set_output("final_report", report)
        print(f"{self.name} generated report: {report['report_id']}")
        return None


async def run_parallel_coordination():
    """Demonstrate parallel agent coordination."""
    print("=== Parallel Coordination Example ===")

    # Create multiple data collectors
    collectors = [
        DataCollector("collector-1"),
        DataCollector("collector-2"),
        DataCollector("collector-3"),
    ]

    # Run collectors in parallel
    start_time = time.time()
    results = await run_agents_parallel(collectors)
    parallel_time = time.time() - start_time

    print(f"Parallel execution completed in {parallel_time:.2f} seconds")

    for agent_name, result in results.items():
        data = result.get_output("collected_data")
        if data:
            print(f"  {agent_name}: {len(data['records'])} records")

    print()


async def run_sequential_coordination():
    """Demonstrate sequential agent coordination."""
    print("=== Sequential Coordination Example ===")

    # Create processing pipeline
    agents = [
        DataCollector("sequential-collector"),
        DataProcessor("sequential-processor", "accurate"),
        DataAggregator("sequential-aggregator"),
        ReportGenerator("sequential-reporter"),
    ]

    # Run agents sequentially
    start_time = time.time()
    results = await run_agents_sequential(agents)
    sequential_time = time.time() - start_time

    print(f"Sequential execution completed in {sequential_time:.2f} seconds")

    # Show the pipeline results
    for agent_name, result in results.items():
        print(f"  {agent_name}: {result.status}")

    print()


async def run_team_coordination():
    """Demonstrate team-based coordination."""
    print("=== Team Coordination Example ===")

    # Create a processing team
    team = create_team("data-processing-team")

    # Add team members
    team.add_agent(DataCollector("team-collector"))
    team.add_agent(DataProcessor("team-processor-fast", "fast"))
    team.add_agent(DataProcessor("team-processor-accurate", "accurate"))
    team.add_agent(DataAggregator("team-aggregator"))
    team.add_agent(ReportGenerator("team-reporter"))

    # Run the team
    start_time = time.time()
    team_result = await team.run()
    team_time = time.time() - start_time

    print(f"Team execution completed in {team_time:.2f} seconds")
    print(f"Team status: {team_result.status}")
    print(f"Agents completed: {len(team_result.agent_results)}")

    # Show individual agent results
    for agent_name, result in team_result.agent_results.items():
        print(f"  {agent_name}: {result.status}")

    print()


async def run_mixed_coordination():
    """Demonstrate mixed coordination patterns."""
    print("=== Mixed Coordination Example ===")

    # Phase 1: Parallel data collection
    print("Phase 1: Parallel data collection")
    collectors = [DataCollector(f"mixed-collector-{i}") for i in range(3)]
    collection_results = await run_agents_parallel(collectors)

    # Phase 2: Parallel processing with different strategies
    print("Phase 2: Parallel processing")
    processors = [
        DataProcessor("mixed-processor-fast", "fast"),
        DataProcessor("mixed-processor-standard", "standard"),
        DataProcessor("mixed-processor-accurate", "accurate"),
    ]
    processing_results = await run_agents_parallel(processors)

    # Phase 3: Sequential aggregation and reporting
    print("Phase 3: Sequential aggregation and reporting")
    final_agents = [
        DataAggregator("mixed-aggregator"),
        ReportGenerator("mixed-reporter"),
    ]
    final_results = await run_agents_sequential(final_agents)

    print("Mixed coordination completed!")

    # Summary
    total_agents = (
        len(collection_results) + len(processing_results) + len(final_results)
    )
    print(f"Total agents executed: {total_agents}")
    print()


class MessagePassingAgent(Agent):
    """Agent that demonstrates message passing capabilities."""

    def __init__(self, name: str, message_target: Optional[str] = None):
        super().__init__(name)
        self.set_variable("message_target", message_target)
        self.set_variable("messages_received", [])
        # Register all decorated states
        self.add_state("send_message", self.send_message)

    @state(cpu=0.5, memory=128.0)
    async def send_message(self, context: Context):
        """Send a message to another agent."""
        target = self.get_variable("message_target")
        if target:
            message = {
                "from": self.name,
                "to": target,
                "content": f"Hello from {self.name}",
                "timestamp": time.time(),
            }

            # In a real implementation, this would use the team's message bus
            print(f"{self.name} sending message to {target}: {message['content']}")
            context.set_output("message_sent", message)

        return None

    def receive_message(self, message: dict):
        """Receive a message from another agent."""
        messages = self.get_variable("messages_received", [])
        messages.append(message)
        self.set_variable("messages_received", messages)
        print(
            f"{self.name} received message from {message['from']}: {message['content']}"
        )


async def run_message_passing():
    """Demonstrate message passing between agents."""
    print("=== Message Passing Example ===")

    # Create agents with message targets
    agent_a = MessagePassingAgent("agent-a", "agent-b")
    agent_b = MessagePassingAgent("agent-b", "agent-c")
    agent_c = MessagePassingAgent("agent-c", "agent-a")

    # Run agents and simulate message passing
    agents = [agent_a, agent_b, agent_c]
    results = await run_agents_parallel(agents)

    # Simulate message delivery (in real implementation, this would be automatic)
    for _agent_name, result in results.items():
        message = result.get_output("message_sent")
        if message:
            target_agent = next((a for a in agents if a.name == message["to"]), None)
            if target_agent:
                target_agent.receive_message(message)

    print("Message passing completed!")
    print()


async def main():
    """Run all coordination examples."""
    print("PuffinFlow Coordination Examples")
    print("=" * 50)

    await run_parallel_coordination()
    await run_sequential_coordination()
    await run_team_coordination()
    await run_mixed_coordination()
    await run_message_passing()

    print("All coordination examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
