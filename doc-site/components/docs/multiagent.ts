export const multiagentMarkdown = `# Multi-Agent Systems

Multi-agent systems are like having a team of specialists working together - each agent has their own expertise, but they collaborate to solve complex problems that would be difficult for any single agent. Puffinflow comes with powerful built-in multi-agent coordination that makes it easy to orchestrate teams of agents.

## Why Multi-Agent Systems Matter

**Without multi-agent coordination:**
- Single agents get overwhelmed by complex tasks
- No way to leverage specialized capabilities
- Poor scalability and resource utilization
- Bottlenecks when one agent handles everything
- No fault tolerance if the single agent fails

**With Puffinflow's built-in multi-agent systems:**
- Specialized agents handle what they do best
- Parallel processing dramatically improves performance
- Automatic load balancing and resource optimization
- Built-in fault tolerance and resilience
- Rich coordination patterns for complex workflows

## Part 1: Built-in Agent Teams

Agent teams are like project teams - each member has a role, and they work together toward a common goal:

\`\`\`python
import asyncio
import time
import random
from puffinflow import Agent, state, AgentTeam, Priority

# Create specialized agents for different tasks
data_agent = Agent("data-processor")
ai_agent = Agent("ai-analyzer")
results_agent = Agent("results-collector")

@state(priority=Priority.HIGH, timeout=30.0)
async def process_data(context):
    """Process raw data for analysis"""

    dataset_id = context.get_variable("dataset_id", "dataset_001")
    print(f"📊 Data Agent: Processing dataset {dataset_id}")

    # Simulate data processing
    await asyncio.sleep(2.0)

    processed_data = {
        "dataset_id": dataset_id,
        "records": [f"record_{i}" for i in range(1000)],
        "quality_score": 0.95,
        "processed_at": time.time()
    }

    context.set_variable("processed_data", processed_data)
    print(f"✅ Data Agent: Processed {len(processed_data['records'])} records")

    return None

@state(priority=Priority.NORMAL, timeout=45.0)
async def analyze_with_ai(context):
    """Perform AI analysis on processed data"""

    model_name = context.get_variable("model", "gpt-4")
    print(f"🤖 AI Agent: Running analysis with {model_name}")

    # Simulate AI analysis
    await asyncio.sleep(3.0)

    # Generate analysis results
    analysis_results = {
        "model_used": model_name,
        "predictions": [f"prediction_{i}" for i in range(1000)],
        "confidence_scores": [0.8 + random.random() * 0.2 for _ in range(1000)],
        "analysis_time": 3.0,
        "analyzed_at": time.time()
    }

    context.set_variable("analysis_results", analysis_results)
    print(f"✅ AI Agent: Generated {len(analysis_results['predictions'])} predictions")

    return None

@state(priority=Priority.LOW, timeout=20.0)
async def collect_results(context):
    """Collect and summarize results from other agents"""

    print("📈 Results Agent: Collecting team results...")

    # Access team-wide results using built-in team context
    team_results = context.get_team_results()

    summary = {
        "total_agents": len(team_results),
        "successful_agents": len([r for r in team_results.values() if r.is_success]),
        "total_processing_time": sum(r.execution_time for r in team_results.values()),
        "collected_at": time.time()
    }

    context.set_output("team_summary", summary)
    print(f"✅ Results Agent: Collected results from {summary['total_agents']} agents")

    return None

# Add states to each agent
data_agent.add_state("process_data", process_data)
ai_agent.add_state("analyze_with_ai", analyze_with_ai)
results_agent.add_state("collect_results", collect_results)

# Create built-in agent team
analysis_team = AgentTeam("analysis-team")
analysis_team.add_agents([data_agent, ai_agent, results_agent])

# Set shared context for the entire team
analysis_team.with_shared_context({
    "project_id": "ml_analysis_2024",
    "quality_threshold": 0.9,
    "max_retries": 3
})

# Demo built-in agent team
async def demo_builtin_agent_team():
    print("👥 Built-in Agent Team Demo\\n")

    # Run agents in parallel using built-in team coordination
    team_result = await analysis_team.run_parallel(
        timeout=60.0,
        initial_context={
            "dataset_id": "customer_data_2024",
            "model": "gpt-4-turbo"
        }
    )

    print(f"\\n📊 Team Results:")
    print(f"   Team name: {team_result.team_name}")
    print(f"   Execution time: {team_result.total_execution_time:.2f}s")
    print(f"   Successful agents: {len(team_result.successful_agents)}")
    print(f"   Failed agents: {len(team_result.failed_agents)}")

    # Access individual agent results
    for agent_name, result in team_result.agent_results.items():
        status_emoji = "✅" if result.is_success else "❌"
        print(f"   {status_emoji} {agent_name}: {result.execution_time:.2f}s")

    # Get team summary from results agent
    if "results-collector" in team_result.agent_results:
        summary = team_result.agent_results["results-collector"].outputs.get("team_summary")
        if summary:
            print(f"\\n📈 Team Summary: {summary}")

if __name__ == "__main__":
    asyncio.run(demo_builtin_agent_team())
\`\`\`

### Advanced Team Coordination

\`\`\`python
# Team with specialized roles and dependencies
from puffinflow import Agent, stateTeam, ExecutionStrategy

# Create team with execution strategy
advanced_team = AgentTeam("advanced-processing")

# Add agents with dependencies
advanced_team.add_agent(data_agent)
advanced_team.add_agent(ai_agent).depends_on("data-processor")
advanced_team.add_agent(results_agent).depends_on(["data-processor", "ai-analyzer"])

# Set team-wide variables
advanced_team.set_variable_for_all("environment", "production")
advanced_team.set_variable_for_all("debug_mode", False)

# Run with dependency-aware execution
result = await advanced_team.run_with_dependencies(timeout=120.0)
\`\`\`

## Part 2: Built-in Agent Pools

Agent pools are like having a group of identical workers ready to handle tasks:

\`\`\`python
from puffinflow import Agent, statePool, WorkItem

# Create a worker agent template
def create_worker_agent():
    """Factory function to create identical worker agents"""
    agent = Agent("worker")

    @agent.state(timeout=15.0, max_retries=2)
    async def process_work_item(context):
        """Process a work item"""

        worker_id = context.get_variable("worker_id", "unknown")
        item_data = context.get_variable("item_data", {})

        print(f"⚙️ Worker {worker_id}: Processing item {item_data.get('id', 'unknown')}")

        # Simulate variable processing time
        processing_time = 1.0 + random.random() * 3.0
        await asyncio.sleep(processing_time)

        # Simulate occasional failures
        if random.random() < 0.1:  # 10% failure rate
            print(f"❌ Worker {worker_id}: Processing failed")
            raise Exception("Processing failed due to data corruption")

        # Process the item
        result = {
            "item_id": item_data.get("id"),
            "worker_id": worker_id,
            "processing_time": processing_time,
            "result_data": f"processed_{item_data.get('value', 'unknown')}",
            "processed_at": time.time()
        }

        context.set_output("processing_result", result)
        print(f"✅ Worker {worker_id}: Completed in {processing_time:.2f}s")

        return None

    return agent

# Create built-in agent pool
worker_pool = AgentPool(
    agent_factory=create_worker_agent,
    size=5,  # 5 worker agents in the pool
    name="processing-pool"
)

# Demo built-in agent pool
async def demo_builtin_agent_pool():
    print("🏊 Built-in Agent Pool Demo\\n")

    # Create work items to process
    work_items = []
    for i in range(15):  # 15 items, 5 workers
        work_item = WorkItem(
            item_id=f"item_{i+1:03d}",
            data={
                "id": f"item_{i+1:03d}",
                "value": f"data_value_{i+1}",
                "priority": random.choice(["high", "normal", "low"])
            }
        )
        work_items.append(work_item)

    print(f"📦 Submitting {len(work_items)} work items to pool of {worker_pool.size} workers")

    # Submit all work items to the pool
    tasks = []
    for item in work_items:
        task = worker_pool.submit_task(item.data)
        tasks.append(task)

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Analyze results
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]

    print(f"\\n📊 Pool Processing Results:")
    print(f"   Total items: {len(work_items)}")
    print(f"   Successful: {len(successful)}")
    print(f"   Failed: {len(failed)}")

    # Show pool statistics
    pool_stats = worker_pool.get_pool_stats()
    print(f"   Pool utilization: {pool_stats['utilization']:.1%}")
    print(f"   Average task time: {pool_stats['average_task_time']:.2f}s")
    print(f"   Total tasks processed: {pool_stats['total_tasks']}")

if __name__ == "__main__":
    asyncio.run(demo_builtin_agent_pool())
\`\`\`

## Part 3: Built-in Agent Orchestrator

The orchestrator is like a conductor managing complex multi-stage workflows:

\`\`\`python
from puffinflow import Agent, stateOrchestrator, ExecutionStrategy, StageConfig

# Create specialized agents for different stages
ingestion_agent = Agent("data-ingestion")
validation_agent = Agent("data-validation")
analysis_agent = Agent("data-analysis")
reporting_agent = Agent("report-generation")

@state(timeout=20.0)
async def ingest_data(context):
    """Ingest data from various sources"""

    source = context.get_variable("data_source", "database")
    print(f"📥 Ingestion: Loading data from {source}")

    await asyncio.sleep(2.0)

    ingested_data = {
        "source": source,
        "record_count": 5000,
        "ingestion_time": 2.0,
        "data_quality": "good"
    }

    context.set_variable("ingested_data", ingested_data)
    print(f"✅ Ingestion: Loaded {ingested_data['record_count']} records")

    return None

@state(timeout=15.0)
async def validate_data(context):
    """Validate the ingested data"""

    ingested_data = context.get_variable("ingested_data", {})
    print(f"🔍 Validation: Checking {ingested_data.get('record_count', 0)} records")

    await asyncio.sleep(1.5)

    validation_results = {
        "records_validated": ingested_data.get("record_count", 0),
        "error_rate": 0.02,
        "validation_time": 1.5,
        "status": "passed"
    }

    context.set_variable("validation_results", validation_results)
    print(f"✅ Validation: {validation_results['status']} with {validation_results['error_rate']*100:.1f}% error rate")

    return None

@state(timeout=30.0)
async def analyze_data(context):
    """Analyze the validated data"""

    validation_results = context.get_variable("validation_results", {})
    print(f"🧠 Analysis: Processing {validation_results.get('records_validated', 0)} validated records")

    await asyncio.sleep(4.0)

    analysis_results = {
        "insights_generated": 25,
        "patterns_found": 8,
        "anomalies_detected": 3,
        "analysis_time": 4.0,
        "confidence_score": 0.92
    }

    context.set_variable("analysis_results", analysis_results)
    print(f"✅ Analysis: Found {analysis_results['patterns_found']} patterns, {analysis_results['anomalies_detected']} anomalies")

    return None

@state(timeout=10.0)
async def generate_report(context):
    """Generate final report"""

    ingested_data = context.get_variable("ingested_data", {})
    validation_results = context.get_variable("validation_results", {})
    analysis_results = context.get_variable("analysis_results", {})

    print("📊 Reporting: Generating comprehensive report")

    await asyncio.sleep(1.0)

    final_report = {
        "pipeline_summary": {
            "total_records": ingested_data.get("record_count", 0),
            "data_quality": validation_results.get("status", "unknown"),
            "insights_count": analysis_results.get("insights_generated", 0),
            "overall_confidence": analysis_results.get("confidence_score", 0)
        },
        "processing_times": {
            "ingestion": ingested_data.get("ingestion_time", 0),
            "validation": validation_results.get("validation_time", 0),
            "analysis": analysis_results.get("analysis_time", 0),
            "reporting": 1.0
        },
        "generated_at": time.time()
    }

    context.set_output("final_report", final_report)
    print(f"✅ Reporting: Report generated with {final_report['pipeline_summary']['insights_count']} insights")

    return None

# Create built-in agent orchestrator
data_pipeline = AgentOrchestrator("data-processing-pipeline")

# Add agents to orchestrator
data_pipeline.add_agents([ingestion_agent, validation_agent, analysis_agent, reporting_agent])

# Define execution stages with dependencies
data_pipeline.add_stage(
    name="data_ingestion",
    agents=[ingestion_agent],
    strategy=ExecutionStrategy.SEQUENTIAL
)

data_pipeline.add_stage(
    name="data_validation",
    agents=[validation_agent],
    strategy=ExecutionStrategy.SEQUENTIAL,
    depends_on=["data_ingestion"]
)

data_pipeline.add_stage(
    name="data_analysis",
    agents=[analysis_agent],
    strategy=ExecutionStrategy.SEQUENTIAL,
    depends_on=["data_validation"]
)

data_pipeline.add_stage(
    name="report_generation",
    agents=[reporting_agent],
    strategy=ExecutionStrategy.SEQUENTIAL,
    depends_on=["data_analysis"]
)

# Set global variables for the entire pipeline
data_pipeline.set_global_variable("environment", "production")
data_pipeline.set_global_variable("max_retries", 3)

# Demo built-in agent orchestrator
async def demo_builtin_orchestrator():
    print("🎭 Built-in Agent Orchestrator Demo\\n")

    # Run the complete orchestrated pipeline
    orchestration_result = await data_pipeline.run(
        initial_context={
            "data_source": "customer_database",
            "processing_date": "2024-03-15"
        }
    )

    print(f"\\n📊 Orchestration Results:")
    print(f"   Pipeline: {orchestration_result.orchestrator_name}")
    print(f"   Total execution time: {orchestration_result.total_execution_time:.2f}s")
    print(f"   Stages completed: {orchestration_result.stages_completed}")
    print(f"   Overall status: {orchestration_result.overall_status}")

    # Show stage-by-stage results
    print(f"\\n📋 Stage Results:")
    for stage_name, stage_result in orchestration_result.stage_results.items():
        status_emoji = "✅" if stage_result.is_success else "❌"
        print(f"   {status_emoji} {stage_name}: {stage_result.execution_time:.2f}s")

    # Get final report
    if orchestration_result.final_outputs.get("final_report"):
        report = orchestration_result.final_outputs["final_report"]
        print(f"\\n📈 Pipeline Summary:")
        summary = report["pipeline_summary"]
        print(f"   Records processed: {summary['total_records']}")
        print(f"   Data quality: {summary['data_quality']}")
        print(f"   Insights generated: {summary['insights_count']}")
        print(f"   Overall confidence: {summary['overall_confidence']:.1%}")

if __name__ == "__main__":
    asyncio.run(demo_builtin_orchestrator())
\`\`\`

## Part 4: Fluent Agent Coordination API

Puffinflow includes a fluent API for intuitive agent coordination:

\`\`\`python
from puffinflow import Agent, states, run_parallel_agents, run_sequential_agents

# Create agents for fluent coordination
data_loader = Agent("data-loader")
feature_extractor = Agent("feature-extractor")
model_trainer = Agent("model-trainer")
model_evaluator = Agent("model-evaluator")

@data_loader.state(timeout=10.0)
async def load_training_data(context):
    """Load training data"""
    dataset_name = context.get_variable("dataset", "training_data")
    print(f"📂 Data Loader: Loading {dataset_name}")

    await asyncio.sleep(1.0)

    context.set_variable("training_data", {
        "samples": 10000,
        "features": 50,
        "loaded_at": time.time()
    })

    return None

@feature_extractor.state(timeout=15.0)
async def extract_features(context):
    """Extract features from training data"""
    training_data = context.get_variable("training_data", {})
    print(f"🔧 Feature Extractor: Processing {training_data.get('samples', 0)} samples")

    await asyncio.sleep(2.0)

    context.set_variable("extracted_features", {
        "feature_count": 25,
        "extraction_time": 2.0,
        "quality_score": 0.88
    })

    return None

@model_trainer.state(timeout=30.0)
async def train_model(context):
    """Train ML model"""
    features = context.get_variable("extracted_features", {})
    print(f"🧠 Model Trainer: Training with {features.get('feature_count', 0)} features")

    await asyncio.sleep(5.0)

    context.set_variable("trained_model", {
        "model_type": "neural_network",
        "training_time": 5.0,
        "accuracy": 0.94,
        "trained_at": time.time()
    })

    return None

@model_evaluator.state(timeout=10.0)
async def evaluate_model(context):
    """Evaluate trained model"""
    model = context.get_variable("trained_model", {})
    print(f"📊 Model Evaluator: Evaluating {model.get('model_type', 'unknown')} model")

    await asyncio.sleep(1.5)

    evaluation_results = {
        "test_accuracy": 0.92,
        "precision": 0.91,
        "recall": 0.93,
        "f1_score": 0.92,
        "evaluation_time": 1.5
    }

    context.set_output("model_evaluation", evaluation_results)
    print(f"✅ Model Evaluator: F1 Score = {evaluation_results['f1_score']:.2f}")

    return None

# Demo fluent API coordination patterns
async def demo_fluent_coordination():
    print("🌊 Fluent Agent Coordination Demo\\n")

    # Pattern 1: Simple parallel execution
    print("=== Parallel Execution ===")
    parallel_result = await run_parallel_agents(
        data_loader, feature_extractor,
        timeout=20.0
    )

    print(f"Parallel execution: {len(parallel_result.successful_agents)} agents completed")

    # Pattern 2: Sequential pipeline
    print("\\n=== Sequential Pipeline ===")
    sequential_result = await run_sequential_agents(
        data_loader, feature_extractor, model_trainer, model_evaluator
    )

    print(f"Sequential pipeline: {sequential_result.total_execution_time:.2f}s total")

    # Pattern 3: Complex coordination with fluent API
    print("\\n=== Complex Coordination ===")
    complex_result = await (
        Agents()
        .add(data_loader)
        .then_add(feature_extractor)  # Sequential after data_loader
        .add_parallel([model_trainer, model_evaluator])  # These run in parallel
        .set_variable_for_all("experiment_id", "exp_001")
        .run_coordinated(timeout=60.0)
    )

    print(f"Complex coordination: {complex_result.overall_status}")

    # Pattern 4: Get best performing agent
    print("\\n=== Best Agent Selection ===")
    agents_to_compare = [model_trainer, model_evaluator]

    # Run agents and find best by execution time
    comparison_result = await (
        Agents()
        .add_many(agents_to_compare)
        .run_parallel()
    )

    fastest_agent = comparison_result.get_best_by("execution_time", maximize=False)
    print(f"Fastest agent: {fastest_agent.agent_name} ({fastest_agent.execution_time:.2f}s)")

if __name__ == "__main__":
    asyncio.run(demo_fluent_coordination())
\`\`\`

## Part 5: Built-in Inter-Agent Communication

Agents can communicate using Puffinflow's built-in message system:

\`\`\`python
from puffinflow import EventBus, Message, MessageType

# Create built-in event bus for agent communication
event_bus = EventBus("ml-pipeline-communication")

# Create communicating agents
coordinator_agent = Agent("coordinator")
worker_agents = [Agent(f"worker-{i}") for i in range(3)]

@state(timeout=30.0)
async def coordinate_workers(context):
    """Coordinate multiple worker agents"""

    task_count = context.get_variable("total_tasks", 10)
    print(f"📡 Coordinator: Distributing {task_count} tasks to workers")

    # Send tasks to workers using built-in messaging
    tasks_per_worker = task_count // len(worker_agents)

    for i, worker in enumerate(worker_agents):
        task_message = Message(
            sender_id="coordinator",
            recipient_id=worker.name,
            message_type=MessageType.COMMAND,
            content={
                "task_type": "data_processing",
                "task_count": tasks_per_worker,
                "worker_id": i,
                "deadline": time.time() + 25
            }
        )

        await event_bus.send_message(task_message)
        print(f"   📤 Sent {tasks_per_worker} tasks to {worker.name}")

    # Wait for completion messages from all workers
    completed_workers = 0
    total_processed = 0

    while completed_workers < len(worker_agents):
        completion_message = await event_bus.wait_for_message(
            recipient_id="coordinator",
            message_type=MessageType.RESPONSE,
            timeout=20.0
        )

        if completion_message:
            worker_result = completion_message.content
            total_processed += worker_result.get("tasks_completed", 0)
            completed_workers += 1

            print(f"   📥 {completion_message.sender_id}: {worker_result['tasks_completed']} tasks completed")

    context.set_output("coordination_result", {
        "total_tasks_assigned": task_count,
        "total_tasks_completed": total_processed,
        "workers_coordinated": completed_workers,
        "coordination_time": time.time() - context.get_variable("start_time", time.time())
    })

    print(f"✅ Coordinator: All workers completed! {total_processed} total tasks processed")

    return None

# Worker agent template
def create_worker_state(worker_name):
    async def process_assigned_tasks(context):
        """Process tasks assigned by coordinator"""

        print(f"👷 {worker_name}: Waiting for task assignment...")

        # Wait for task message from coordinator
        task_message = await event_bus.wait_for_message(
            recipient_id=worker_name,
            message_type=MessageType.COMMAND,
            timeout=15.0
        )

        if not task_message:
            print(f"❌ {worker_name}: No task received")
            return None

        task_info = task_message.content
        task_count = task_info.get("task_count", 0)

        print(f"🔄 {worker_name}: Processing {task_count} tasks...")

        # Simulate task processing
        processing_time = 1.0 + random.random() * 2.0
        await asyncio.sleep(processing_time)

        # Send completion message back to coordinator
        completion_message = Message(
            sender_id=worker_name,
            recipient_id="coordinator",
            message_type=MessageType.RESPONSE,
            content={
                "tasks_completed": task_count,
                "processing_time": processing_time,
                "worker_status": "completed"
            }
        )

        await event_bus.send_message(completion_message)
        print(f"✅ {worker_name}: Completed {task_count} tasks in {processing_time:.2f}s")

        return None

    return process_assigned_tasks

# Add worker states to agents
for i, worker in enumerate(worker_agents):
    worker.add_state("process_tasks", create_worker_state(worker.name))

# Demo inter-agent communication
async def demo_agent_communication():
    print("📡 Inter-Agent Communication Demo\\n")

    # Start coordinator and workers
    tasks = []

    # Start coordinator
    coordinator_task = asyncio.create_task(
        coordinator_agent.run(
            initial_state="coordinate_workers",
            initial_context={
                "total_tasks": 15,
                "start_time": time.time()
            }
        )
    )
    tasks.append(("coordinator", coordinator_task))

    # Start workers
    for worker in worker_agents:
        worker_task = asyncio.create_task(
            worker.run(initial_state="process_tasks")
        )
        tasks.append((worker.name, worker_task))

    # Wait for all to complete
    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

    print(f"\\n📊 Communication Results:")
    for (agent_name, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            print(f"   ❌ {agent_name}: Failed")
        else:
            print(f"   ✅ {agent_name}: Completed")

    # Show message statistics
    message_stats = event_bus.get_message_statistics()
    print(f"\\n📈 Message Statistics:")
    print(f"   Total messages sent: {message_stats['total_sent']}")
    print(f"   Total messages delivered: {message_stats['total_delivered']}")
    print(f"   Average delivery time: {message_stats['avg_delivery_time']:.3f}s")

if __name__ == "__main__":
    asyncio.run(demo_agent_communication())
\`\`\`

## Quick Reference

### Enable Built-in Multi-Agent Features

\`\`\`python
# Import built-in multi-agent classes
from puffinflow import (
    AgentTeam, AgentPool, AgentOrchestrator, Agents,
    EventBus, Message, MessageType, ExecutionStrategy
)

# Agent Team
team = AgentTeam("my-team")
team.add_agents([agent1, agent2, agent3])
result = await team.run_parallel(timeout=60.0)

# Agent Pool
pool = AgentPool(agent_factory=create_worker, size=5)
result = await pool.submit_task({"data": "work_item"})

# Agent Orchestrator
orchestrator = AgentOrchestrator("pipeline")
orchestrator.add_stage("stage1", [agent1], ExecutionStrategy.PARALLEL)
result = await orchestrator.run()

# Fluent API
result = await Agents().add_many([agent1, agent2]).run_parallel()

# Inter-agent communication
event_bus = EventBus("communication")
message = Message(sender_id="agent1", recipient_id="agent2",
                 message_type=MessageType.REQUEST, content={"task": "process"})
await event_bus.send_message(message)
\`\`\`

### Access Built-in Multi-Agent Results

\`\`\`python
# Team results
team_result.agent_results       # Individual agent results
team_result.successful_agents   # List of successful agents
team_result.failed_agents      # List of failed agents
team_result.total_execution_time # Total time

# Pool statistics
pool_stats = pool.get_pool_stats()
pool_stats['utilization']      # Pool utilization percentage
pool_stats['total_tasks']      # Total tasks processed

# Orchestration results
orch_result.stage_results      # Results by stage
orch_result.overall_status     # Overall pipeline status
orch_result.stages_completed   # Number of completed stages

# Fluent API results
fluent_result.get_best_by("execution_time", maximize=False)
fluent_result.filter_successful()
fluent_result.group_by_status()
\`\`\`

## Built-in Multi-Agent Features

### AgentTeam
\`\`\`python
from puffinflow import Agent, stateTeam

team = AgentTeam("processing-team")
team.add_agents([agent1, agent2, agent3])
team.with_shared_context({"project": "ml_pipeline"})

# Parallel execution
result = await team.run_parallel(timeout=60.0)

# Sequential execution
result = await team.run_sequential()

# Dependency-aware execution
result = await team.run_with_dependencies()
\`\`\`

### AgentPool
\`\`\`python
from puffinflow import Agent, statePool

def create_worker():
    return Agent("worker")

pool = AgentPool(
    agent_factory=create_worker,
    size=10,                    # Number of workers
    name="worker-pool"
)

# Submit tasks to pool
task_result = await pool.submit_task({"data": "work_item"})

# Get pool statistics
stats = pool.get_pool_stats()
\`\`\`

### AgentOrchestrator
\`\`\`python
from puffinflow import Agent, stateOrchestrator, ExecutionStrategy

orchestrator = AgentOrchestrator("data-pipeline")
orchestrator.add_agents([agent1, agent2, agent3])

# Add execution stages
orchestrator.add_stage(
    name="preprocessing",
    agents=[agent1],
    strategy=ExecutionStrategy.SEQUENTIAL
)

orchestrator.add_stage(
    name="processing",
    agents=[agent2, agent3],
    strategy=ExecutionStrategy.PARALLEL,
    depends_on=["preprocessing"]
)

# Run orchestrated pipeline
result = await orchestrator.run()
\`\`\`

### Fluent API
\`\`\`python
from puffinflow import Agent, states, run_parallel_agents

# Simple parallel execution
result = await run_parallel_agents(agent1, agent2, agent3, timeout=30.0)

# Complex coordination
result = await (
    Agents()
    .add(agent1)
    .then_add(agent2)           # Sequential after agent1
    .add_parallel([agent3, agent4])  # Parallel execution
    .set_variable_for_all("env", "prod")
    .run_coordinated()
)

# Get best performing agent
best = result.get_best_by("execution_time", maximize=False)
\`\`\`

### EventBus Communication
\`\`\`python
from puffinflow import EventBus, Message, MessageType

event_bus = EventBus("agent-communication")

# Send message
message = Message(
    sender_id="agent1",
    recipient_id="agent2",
    message_type=MessageType.REQUEST,
    content={"task": "process_data", "priority": "high"}
)
await event_bus.send_message(message)

# Wait for message
response = await event_bus.wait_for_message(
    recipient_id="agent1",
    message_type=MessageType.RESPONSE,
    timeout=30.0
)
\`\`\`

## Tips for Beginners

1. **Start with AgentTeam** - Use \`AgentTeam\` for simple parallel agent coordination
2. **Use AgentPool for identical workers** - Great for load balancing and horizontal scaling
3. **Try the fluent API** - Use \`Agents()\` for intuitive, readable coordination code
4. **Add dependencies gradually** - Start simple, then add stage dependencies as needed
5. **Enable communication when needed** - Use \`EventBus\` for agents that need to communicate
6. **Monitor team performance** - Check \`team_result.total_execution_time\` and success rates

## Common Multi-Agent Patterns

### Pattern 1: Parallel Processing Team
\`\`\`python
team = AgentTeam("parallel-processors")
team.add_agents([worker1, worker2, worker3])
result = await team.run_parallel(timeout=60.0)
\`\`\`

### Pattern 2: Worker Pool for Load Balancing
\`\`\`python
pool = AgentPool(create_worker, size=5)
tasks = [pool.submit_task(item) for item in work_items]
results = await asyncio.gather(*tasks)
\`\`\`

### Pattern 3: Sequential Pipeline
\`\`\`python
result = await run_sequential_agents(
    data_loader, processor, analyzer, reporter
)
\`\`\`

### Pattern 4: Complex Orchestration
\`\`\`python
orchestrator = AgentOrchestrator("complex-pipeline")
orchestrator.add_stage("ingestion", [ingest_agent])
orchestrator.add_stage("processing", [proc1, proc2], depends_on=["ingestion"])
result = await orchestrator.run()
\`\`\`

## What's Built Into Puffinflow

✅ **AgentTeam**: Team-based coordination with parallel/sequential execution
✅ **AgentPool**: Load balancing with identical worker agents
✅ **AgentOrchestrator**: Complex multi-stage pipeline orchestration
✅ **Fluent API (Agents)**: Intuitive, chainable coordination methods
✅ **EventBus**: Inter-agent messaging and communication
✅ **Message System**: Structured communication with types and priorities
✅ **Execution Strategies**: PARALLEL, SEQUENTIAL, PIPELINE, FAN_OUT, etc.
✅ **Dependency Management**: Stage dependencies and execution ordering
✅ **Result Aggregation**: Built-in result collection and analysis
✅ **Performance Monitoring**: Execution time tracking and statistics

Puffinflow's built-in multi-agent systems give you powerful coordination capabilities without any custom implementation required!
`.trim();
