export const multiagentMarkdown = `# Multi-Agent Systems & Collaboration

Puffinflow enables sophisticated multi-agent systems where multiple AI agents collaborate, coordinate, and communicate to solve complex problems. This guide covers agent communication patterns, team structures, hierarchical organizations, and collaborative workflows for building scalable AI systems.

## Multi-Agent Philosophy

**Multi-agent systems go beyond parallel processing** - they enable:
- **Specialization**: Agents focus on specific domains or tasks
- **Collaboration**: Agents share knowledge and coordinate actions
- **Scalability**: Distribute workload across multiple specialized components
- **Resilience**: System continues operating if individual agents fail
- **Emergence**: Complex behaviors arising from simple agent interactions

## Multi-Agent Architecture Patterns

| Pattern | Use Cases | Benefits | Complexity |
|---------|-----------|----------|------------|
| **Agent Pool** | Load balancing, horizontal scaling | High throughput, fault tolerance | Low |
| **Agent Pipeline** | Sequential processing, data transformation | Clear data flow, easy debugging | Medium |
| **Agent Hierarchy** | Command & control, delegation | Clear authority, structured coordination | High |
| **Agent Swarm** | Distributed problem solving, emergent behavior | Adaptive, self-organizing | Very High |
| **Agent Market** | Resource allocation, competitive optimization | Efficient resource use, dynamic adaptation | High |

---

## Agent Communication & Messaging

### Inter-Agent Communication Framework

Establish robust communication channels between agents:

\`\`\`python
import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from puffinflow import Agent
from puffinflow import state

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    COMMAND = "command"

class MessagePriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class Message:
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    expires_at: Optional[float] = None

class MessageBus:
    def __init__(self):
        self.message_queues = {}
        self.subscribers = {}
        self.message_handlers = {}
        self.delivered_messages = []
        self.failed_messages = []

    def register_agent(self, agent_id: str):
        """Register an agent with the message bus"""
        if agent_id not in self.message_queues:
            self.message_queues[agent_id] = asyncio.Queue()
            self.subscribers[agent_id] = set()
            self.message_handlers[agent_id] = {}

    async def send_message(self, message: Message) -> bool:
        """Send a message to a specific agent"""
        try:
            if message.recipient_id in self.message_queues:
                await self.message_queues[message.recipient_id].put(message)
                self.delivered_messages.append(message)

                print(f"📤 MessageBus: Delivered {message.message_type.value} from {message.sender_id} to {message.recipient_id}")
                return True
            else:
                print(f"❌ MessageBus: Recipient {message.recipient_id} not found")
                self.failed_messages.append(message)
                return False

        except Exception as e:
            print(f"❌ MessageBus: Failed to send message: {e}")
            self.failed_messages.append(message)
            return False

    async def broadcast_message(self, sender_id: str, content: Dict[str, Any], message_type: MessageType = MessageType.BROADCAST):
        """Broadcast a message to all registered agents"""
        broadcast_tasks = []

        for agent_id in self.message_queues.keys():
            if agent_id != sender_id:  # Don't send to sender
                message = Message(
                    message_id=f"broadcast_{int(time.time())}_{agent_id}",
                    sender_id=sender_id,
                    recipient_id=agent_id,
                    message_type=message_type,
                    priority=MessagePriority.NORMAL,
                    content=content
                )
                broadcast_tasks.append(self.send_message(message))

        results = await asyncio.gather(*broadcast_tasks, return_exceptions=True)
        successful_sends = sum(1 for result in results if result is True)

        print(f"📡 MessageBus: Broadcast from {sender_id} delivered to {successful_sends}/{len(broadcast_tasks)} agents")
        return successful_sends

    async def receive_message(self, agent_id: str, timeout: float = 5.0) -> Optional[Message]:
        """Receive a message for a specific agent"""
        try:
            if agent_id in self.message_queues:
                message = await asyncio.wait_for(
                    self.message_queues[agent_id].get(),
                    timeout=timeout
                )

                # Check if message has expired
                if message.expires_at and time.time() > message.expires_at:
                    print(f"⏰ MessageBus: Message {message.message_id} expired")
                    return None

                return message
            return None

        except asyncio.TimeoutError:
            return None

    def subscribe_to_topic(self, agent_id: str, topic: str, handler: Callable):
        """Subscribe an agent to a specific topic"""
        if agent_id not in self.message_handlers:
            self.message_handlers[agent_id] = {}

        self.message_handlers[agent_id][topic] = handler
        self.subscribers[agent_id].add(topic)

        print(f"📝 MessageBus: Agent {agent_id} subscribed to topic '{topic}'")

# Global message bus
message_bus = MessageBus()

# Multi-agent communication demonstration
communication_agent = Agent("agent-communication-demo")

@state(timeout=45.0)
async def coordinator_agent_task(context):
    """Coordinator agent that orchestrates multi-agent collaboration"""
    agent_id = "coordinator"

    print(f"🎭 {agent_id}: Starting coordination...")

    # Register with message bus
    message_bus.register_agent(agent_id)

    # Define work distribution
    work_assignments = [
        {
            "agent_id": "data_processor_1",
            "task_type": "data_ingestion",
            "parameters": {"source": "database", "batch_size": 1000}
        },
        {
            "agent_id": "data_processor_2",
            "task_type": "data_validation",
            "parameters": {"validation_rules": ["schema", "quality", "completeness"]}
        },
        {
            "agent_id": "ai_analyzer",
            "task_type": "ai_analysis",
            "parameters": {"model": "gpt-4", "analysis_type": "sentiment"}
        }
    ]

    # Send work assignments
    for assignment in work_assignments:
        message = Message(
            message_id=f"work_assignment_{assignment['agent_id']}",
            sender_id=agent_id,
            recipient_id=assignment["agent_id"],
            message_type=MessageType.COMMAND,
            priority=MessagePriority.HIGH,
            content={
                "command": "start_task",
                "assignment": assignment
            },
            expires_at=time.time() + 300  # 5 minute expiry
        )

        await message_bus.send_message(message)
        print(f"📋 {agent_id}: Assigned {assignment['task_type']} to {assignment['agent_id']}")

    # Monitor progress and collect results
    completed_tasks = []
    failed_tasks = []

    # Wait for responses from all agents
    for assignment in work_assignments:
        print(f"⏳ {agent_id}: Waiting for response from {assignment['agent_id']}...")

        response = await message_bus.receive_message(agent_id, timeout=30.0)

        if response and response.message_type == MessageType.RESPONSE:
            if response.content.get("status") == "completed":
                completed_tasks.append(response.content)
                print(f"✅ {agent_id}: Received completion from {response.sender_id}")
            else:
                failed_tasks.append(response.content)
                print(f"❌ {agent_id}: Received failure from {response.sender_id}")
        else:
            failed_tasks.append({"agent_id": assignment["agent_id"], "error": "timeout"})
            print(f"⏰ {agent_id}: Timeout waiting for {assignment['agent_id']}")

    # Generate coordination summary
    coordination_results = {
        "coordinator_id": agent_id,
        "total_assignments": len(work_assignments),
        "completed_tasks": len(completed_tasks),
        "failed_tasks": len(failed_tasks),
        "success_rate": len(completed_tasks) / len(work_assignments) if work_assignments else 0,
        "task_results": completed_tasks
    }

    context.set_variable("coordination_results", coordination_results)

    # Broadcast completion to all agents
    await message_bus.broadcast_message(agent_id, {
        "notification": "coordination_complete",
        "summary": coordination_results
    })

    print(f"🎉 {agent_id}: Coordination complete - {len(completed_tasks)}/{len(work_assignments)} tasks successful")

@state(timeout=30.0)
async def worker_agent_task(context):
    """Worker agent that receives and processes tasks"""
    agent_id = context.get_variable("agent_id", "worker_001")

    print(f"👷 {agent_id}: Starting as worker agent...")

    # Register with message bus
    message_bus.register_agent(agent_id)

    # Wait for work assignment
    print(f"📬 {agent_id}: Waiting for work assignment...")

    assignment_message = await message_bus.receive_message(agent_id, timeout=20.0)

    if not assignment_message:
        print(f"⏰ {agent_id}: No assignment received, timing out")
        return

    if assignment_message.message_type != MessageType.COMMAND:
        print(f"❌ {agent_id}: Unexpected message type: {assignment_message.message_type}")
        return

    # Process the assignment
    assignment = assignment_message.content.get("assignment", {})
    task_type = assignment.get("task_type", "unknown")
    parameters = assignment.get("parameters", {})

    print(f"📋 {agent_id}: Received assignment: {task_type}")

    try:
        # Simulate task processing based on type
        if task_type == "data_ingestion":
            await simulate_data_ingestion(agent_id, parameters)
        elif task_type == "data_validation":
            await simulate_data_validation(agent_id, parameters)
        elif task_type == "ai_analysis":
            await simulate_ai_analysis(agent_id, parameters)
        else:
            raise Exception(f"Unknown task type: {task_type}")

        # Send success response
        response = Message(
            message_id=f"response_{agent_id}_{int(time.time())}",
            sender_id=agent_id,
            recipient_id=assignment_message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=MessagePriority.NORMAL,
            content={
                "status": "completed",
                "task_type": task_type,
                "agent_id": agent_id,
                "completion_time": time.time(),
                "results": {
                    "processed_items": 1500,
                    "processing_time": 5.0,
                    "quality_score": 0.95
                }
            },
            correlation_id=assignment_message.message_id
        )

        await message_bus.send_message(response)
        print(f"✅ {agent_id}: Task {task_type} completed successfully")

    except Exception as e:
        # Send failure response
        error_response = Message(
            message_id=f"error_response_{agent_id}_{int(time.time())}",
            sender_id=agent_id,
            recipient_id=assignment_message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=MessagePriority.HIGH,
            content={
                "status": "failed",
                "task_type": task_type,
                "agent_id": agent_id,
                "error": str(e),
                "failure_time": time.time()
            },
            correlation_id=assignment_message.message_id
        )

        await message_bus.send_message(error_response)
        print(f"❌ {agent_id}: Task {task_type} failed: {e}")

# Simulation functions for different task types
async def simulate_data_ingestion(agent_id: str, parameters: Dict):
    """Simulate data ingestion task"""
    source = parameters.get("source", "unknown")
    batch_size = parameters.get("batch_size", 100)

    print(f"   📥 {agent_id}: Ingesting data from {source} (batch size: {batch_size})")
    await asyncio.sleep(3.0)  # Simulate processing time

    if source == "database":
        # Simulate potential database issues
        if time.time() % 10 < 1:  # 10% chance of failure
            raise Exception("Database connection timeout")

async def simulate_data_validation(agent_id: str, parameters: Dict):
    """Simulate data validation task"""
    rules = parameters.get("validation_rules", [])

    print(f"   🔍 {agent_id}: Validating data with rules: {', '.join(rules)}")
    await asyncio.sleep(2.0)  # Simulate processing time

    # Simulate validation failures
    if len(rules) > 2 and time.time() % 15 < 1:  # Rare failure
        raise Exception("Data quality check failed")

async def simulate_ai_analysis(agent_id: str, parameters: Dict):
    """Simulate AI analysis task"""
    model = parameters.get("model", "unknown")
    analysis_type = parameters.get("analysis_type", "unknown")

    print(f"   🧠 {agent_id}: Running {analysis_type} analysis with {model}")
    await asyncio.sleep(4.0)  # Simulate AI processing time

    # Simulate AI model issues
    if model == "gpt-4" and time.time() % 20 < 1:  # Rare API failure
        raise Exception("AI model API rate limit exceeded")

# Create the coordination workflow
communication_agent.add_state("coordinator", coordinator_agent_task)

# Create worker agents
worker_agents = ["data_processor_1", "data_processor_2", "ai_analyzer"]
for worker_id in worker_agents:
    communication_agent.add_state(f"worker_{worker_id}",
        lambda ctx, w_id=worker_id: worker_agent_task({**ctx.shared_state, "agent_id": w_id}))
\`\`\`

---

## Agent Teams & Hierarchical Organizations

### Team-Based Agent Structures

Organize agents into teams with specific roles and responsibilities:

\`\`\`python
from typing import Set, Dict, List
from enum import Enum

class AgentRole(Enum):
    LEADER = "leader"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    WORKER = "worker"
    OBSERVER = "observer"

class TeamStructure(Enum):
    FLAT = "flat"
    HIERARCHICAL = "hierarchical"
    MATRIX = "matrix"
    NETWORK = "network"

@dataclass
class AgentProfile:
    agent_id: str
    role: AgentRole
    specializations: List[str]
    capabilities: Dict[str, float]  # capability -> skill level (0-1)
    team_memberships: Set[str]
    reporting_to: Optional[str] = None
    direct_reports: Set[str] = field(default_factory=set)

class AgentTeam:
    def __init__(self, team_id: str, team_structure: TeamStructure):
        self.team_id = team_id
        self.structure = team_structure
        self.members: Dict[str, AgentProfile] = {}
        self.team_leader: Optional[str] = None
        self.communication_patterns = {}
        self.task_assignments = {}

    def add_member(self, agent_profile: AgentProfile):
        """Add an agent to the team"""
        self.members[agent_profile.agent_id] = agent_profile
        agent_profile.team_memberships.add(self.team_id)

        # Set team leader if this is the first leader role
        if agent_profile.role == AgentRole.LEADER and not self.team_leader:
            self.team_leader = agent_profile.agent_id

    def assign_task(self, task_id: str, task_requirements: Dict[str, float], deadline: float):
        """Assign a task to the best-suited team member"""
        best_agent = None
        best_score = 0.0

        for agent_id, profile in self.members.items():
            # Calculate suitability score
            score = 0.0
            for requirement, weight in task_requirements.items():
                if requirement in profile.capabilities:
                    score += profile.capabilities[requirement] * weight

            # Normalize by number of requirements
            score = score / len(task_requirements) if task_requirements else 0

            if score > best_score:
                best_score = score
                best_agent = agent_id

        if best_agent:
            self.task_assignments[task_id] = {
                "assigned_to": best_agent,
                "requirements": task_requirements,
                "deadline": deadline,
                "score": best_score,
                "assigned_at": time.time()
            }

            return best_agent

        return None

# Create specialized agent teams
data_team = AgentTeam("data_processing_team", TeamStructure.HIERARCHICAL)
ai_team = AgentTeam("ai_analysis_team", TeamStructure.FLAT)
coordination_team = AgentTeam("coordination_team", TeamStructure.NETWORK)

# Define agent profiles
agent_profiles = [
    # Data Processing Team
    AgentProfile(
        agent_id="data_team_leader",
        role=AgentRole.LEADER,
        specializations=["data_architecture", "team_management"],
        capabilities={"data_processing": 0.9, "leadership": 0.95, "coordination": 0.8},
        team_memberships=set()
    ),
    AgentProfile(
        agent_id="data_ingestion_specialist",
        role=AgentRole.SPECIALIST,
        specializations=["data_ingestion", "etl_pipelines"],
        capabilities={"data_processing": 0.95, "database_management": 0.9, "api_integration": 0.85},
        team_memberships=set(),
        reporting_to="data_team_leader"
    ),
    AgentProfile(
        agent_id="data_quality_specialist",
        role=AgentRole.SPECIALIST,
        specializations=["data_validation", "quality_assurance"],
        capabilities={"data_processing": 0.9, "quality_control": 0.95, "statistics": 0.8},
        team_memberships=set(),
        reporting_to="data_team_leader"
    ),

    # AI Analysis Team
    AgentProfile(
        agent_id="ml_specialist",
        role=AgentRole.SPECIALIST,
        specializations=["machine_learning", "model_training"],
        capabilities={"ai_analysis": 0.95, "model_optimization": 0.9, "data_science": 0.85},
        team_memberships=set()
    ),
    AgentProfile(
        agent_id="nlp_specialist",
        role=AgentRole.SPECIALIST,
        specializations=["natural_language_processing", "text_analysis"],
        capabilities={"ai_analysis": 0.9, "nlp": 0.95, "linguistics": 0.8},
        team_memberships=set()
    ),

    # Coordination Team
    AgentProfile(
        agent_id="master_coordinator",
        role=AgentRole.COORDINATOR,
        specializations=["workflow_orchestration", "resource_management"],
        capabilities={"coordination": 0.95, "resource_management": 0.9, "planning": 0.85},
        team_memberships=set()
    )
]

# Add agents to teams
for profile in agent_profiles:
    if "data" in profile.agent_id:
        data_team.add_member(profile)
    elif profile.specializations and any("ml" in spec or "nlp" in spec for spec in profile.specializations):
        ai_team.add_member(profile)
    elif profile.role == AgentRole.COORDINATOR:
        coordination_team.add_member(profile)

# Multi-agent team collaboration
team_agent = Agent("multi-agent-teams")

@state(timeout=60.0)
async def execute_team_based_workflow(context):
    """Execute a complex workflow using specialized agent teams"""
    print("🏢 Starting team-based multi-agent workflow...")

    # Define complex project requirements
    project_tasks = [
        {
            "task_id": "data_pipeline_setup",
            "requirements": {"data_processing": 0.8, "database_management": 0.7},
            "deadline": time.time() + 3600,
            "estimated_duration": 120
        },
        {
            "task_id": "data_quality_validation",
            "requirements": {"data_processing": 0.7, "quality_control": 0.9},
            "deadline": time.time() + 1800,
            "estimated_duration": 90
        },
        {
            "task_id": "machine_learning_analysis",
            "requirements": {"ai_analysis": 0.9, "model_optimization": 0.8},
            "deadline": time.time() + 2400,
            "estimated_duration": 180
        },
        {
            "task_id": "text_processing_analysis",
            "requirements": {"ai_analysis": 0.8, "nlp": 0.9},
            "deadline": time.time() + 2000,
            "estimated_duration": 150
        }
    ]

    # Assign tasks to appropriate teams and agents
    task_assignments = {}

    for task in project_tasks:
        task_id = task["task_id"]

        # Determine which team should handle the task
        if "data" in task_id:
            assigned_agent = data_team.assign_task(task_id, task["requirements"], task["deadline"])
            team_id = data_team.team_id
        elif "machine_learning" in task_id:
            assigned_agent = ai_team.assign_task(task_id, task["requirements"], task["deadline"])
            team_id = ai_team.team_id
        elif "text_processing" in task_id:
            assigned_agent = ai_team.assign_task(task_id, task["requirements"], task["deadline"])
            team_id = ai_team.team_id
        else:
            assigned_agent = coordination_team.assign_task(task_id, task["requirements"], task["deadline"])
            team_id = coordination_team.team_id

        if assigned_agent:
            task_assignments[task_id] = {
                "team_id": team_id,
                "agent_id": assigned_agent,
                "task": task
            }

            print(f"📋 Assigned {task_id} to {assigned_agent} in {team_id}")
        else:
            print(f"❌ Could not assign {task_id} - no suitable agent found")

    # Simulate task execution with team coordination
    completed_tasks = []

    for task_id, assignment in task_assignments.items():
        agent_id = assignment["agent_id"]
        team_id = assignment["team_id"]
        task = assignment["task"]

        print(f"🚀 Starting {task_id} with {agent_id}...")

        # Simulate task execution
        execution_start = time.time()

        # Simulate work (compressed time for demo)
        simulated_duration = task["estimated_duration"] / 30.0  # Compress 30x
        await asyncio.sleep(simulated_duration)

        execution_duration = time.time() - execution_start

        # Simulate task completion
        task_result = {
            "task_id": task_id,
            "agent_id": agent_id,
            "team_id": team_id,
            "status": "completed",
            "execution_duration": execution_duration,
            "quality_score": 0.85 + (hash(agent_id) % 100) / 1000,  # Simulated quality
            "completion_time": time.time()
        }

        completed_tasks.append(task_result)
        print(f"✅ Completed {task_id} in {execution_duration:.2f}s (quality: {task_result['quality_score']:.3f})")

    # Generate team performance analysis
    team_performance = analyze_team_performance(completed_tasks, task_assignments)

    context.set_variable("team_workflow_results", {
        "total_tasks": len(project_tasks),
        "completed_tasks": len(completed_tasks),
        "task_assignments": task_assignments,
        "completed_task_results": completed_tasks,
        "team_performance": team_performance
    })

    print(f"🎉 Team workflow completed: {len(completed_tasks)}/{len(project_tasks)} tasks finished")

def analyze_team_performance(completed_tasks: List[Dict], assignments: Dict) -> Dict:
    """Analyze performance of each team"""
    team_stats = {}

    for task in completed_tasks:
        team_id = task["team_id"]

        if team_id not in team_stats:
            team_stats[team_id] = {
                "tasks_completed": 0,
                "total_duration": 0,
                "avg_quality": 0,
                "agents_used": set()
            }

        team_stats[team_id]["tasks_completed"] += 1
        team_stats[team_id]["total_duration"] += task["execution_duration"]
        team_stats[team_id]["avg_quality"] += task["quality_score"]
        team_stats[team_id]["agents_used"].add(task["agent_id"])

    # Calculate averages
    for team_id, stats in team_stats.items():
        if stats["tasks_completed"] > 0:
            stats["avg_duration"] = stats["total_duration"] / stats["tasks_completed"]
            stats["avg_quality"] = stats["avg_quality"] / stats["tasks_completed"]
            stats["agents_used"] = list(stats["agents_used"])

    return team_stats

team_agent.add_state("execute_team_workflow", execute_team_based_workflow)
\`\`\`

---

## Agent Swarms & Emergent Behavior

### Self-Organizing Agent Collectives

Create agent swarms that exhibit emergent behavior and self-organization:

\`\`\`python
import random
import math
from typing import Tuple, List

@dataclass
class SwarmAgent:
    agent_id: str
    position: Tuple[float, float]  # x, y coordinates
    velocity: Tuple[float, float]  # vx, vy
    energy: float
    role: str
    local_memory: Dict[str, Any] = field(default_factory=dict)
    communication_range: float = 10.0

    def distance_to(self, other: 'SwarmAgent') -> float:
        """Calculate distance to another agent"""
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return math.sqrt(dx * dx + dy * dy)

    def can_communicate_with(self, other: 'SwarmAgent') -> bool:
        """Check if this agent can communicate with another"""
        return self.distance_to(other) <= self.communication_range

class SwarmIntelligence:
    def __init__(self, environment_size: Tuple[float, float] = (100.0, 100.0)):
        self.environment_size = environment_size
        self.agents: List[SwarmAgent] = []
        self.global_state = {}
        self.communication_network = {}
        self.collective_memory = {}

    def add_agent(self, agent: SwarmAgent):
        """Add an agent to the swarm"""
        self.agents.append(agent)
        self.communication_network[agent.agent_id] = set()

    def update_communication_network(self):
        """Update which agents can communicate with each other"""
        for agent in self.agents:
            self.communication_network[agent.agent_id] = set()

            for other_agent in self.agents:
                if agent.agent_id != other_agent.agent_id and agent.can_communicate_with(other_agent):
                    self.communication_network[agent.agent_id].add(other_agent.agent_id)

    def get_local_neighbors(self, agent_id: str) -> List[SwarmAgent]:
        """Get neighboring agents within communication range"""
        target_agent = next((a for a in self.agents if a.agent_id == agent_id), None)
        if not target_agent:
            return []

        neighbors = []
        for other_agent in self.agents:
            if other_agent.agent_id != agent_id and target_agent.can_communicate_with(other_agent):
                neighbors.append(other_agent)

        return neighbors

    async def execute_swarm_behavior(self, iterations: int = 10):
        """Execute swarm behavior for specified iterations"""
        for iteration in range(iterations):
            print(f"🐝 Swarm iteration {iteration + 1}/{iterations}")

            # Update communication network
            self.update_communication_network()

            # Execute behavior for each agent
            for agent in self.agents:
                await self.execute_agent_behavior(agent, iteration)

            # Update global swarm state
            self.update_global_state(iteration)

            await asyncio.sleep(0.5)  # Pause between iterations

    async def execute_agent_behavior(self, agent: SwarmAgent, iteration: int):
        """Execute behavior for a specific agent"""
        neighbors = self.get_local_neighbors(agent.agent_id)

        if agent.role == "explorer":
            await self.explorer_behavior(agent, neighbors, iteration)
        elif agent.role == "forager":
            await self.forager_behavior(agent, neighbors, iteration)
        elif agent.role == "coordinator":
            await self.coordinator_behavior(agent, neighbors, iteration)
        elif agent.role == "analyzer":
            await self.analyzer_behavior(agent, neighbors, iteration)

    async def explorer_behavior(self, agent: SwarmAgent, neighbors: List[SwarmAgent], iteration: int):
        """Explorer agents seek new areas and opportunities"""
        # Move towards unexplored areas
        target_x = random.uniform(0, self.environment_size[0])
        target_y = random.uniform(0, self.environment_size[1])

        # Adjust target based on neighbor positions (avoid crowding)
        if neighbors:
            neighbor_avg_x = sum(n.position[0] for n in neighbors) / len(neighbors)
            neighbor_avg_y = sum(n.position[1] for n in neighbors) / len(neighbors)

            # Move away from crowd
            repulsion_strength = 5.0
            target_x = agent.position[0] + (agent.position[0] - neighbor_avg_x) * repulsion_strength / 100
            target_y = agent.position[1] + (agent.position[1] - neighbor_avg_y) * repulsion_strength / 100

        # Update position
        dx = (target_x - agent.position[0]) * 0.1  # 10% movement per iteration
        dy = (target_y - agent.position[1]) * 0.1

        new_x = max(0, min(self.environment_size[0], agent.position[0] + dx))
        new_y = max(0, min(self.environment_size[1], agent.position[1] + dy))

        agent.position = (new_x, new_y)
        agent.velocity = (dx, dy)

        # Record exploration
        exploration_key = f"explored_{int(new_x//10)}_{int(new_y//10)}"
        agent.local_memory[exploration_key] = iteration

        # Share findings with neighbors
        for neighbor in neighbors:
            if neighbor.role in ["forager", "coordinator"]:
                # Simplified communication - share position info
                neighbor.local_memory[f"explorer_report_{agent.agent_id}"] = {
                    "position": agent.position,
                    "iteration": iteration,
                    "energy": agent.energy
                }

        print(f"   🔍 Explorer {agent.agent_id}: Moved to ({new_x:.1f}, {new_y:.1f}), {len(neighbors)} neighbors")

    async def forager_behavior(self, agent: SwarmAgent, neighbors: List[SwarmAgent], iteration: int):
        """Forager agents collect resources and optimize paths"""
        # Look for resource locations from explorer reports
        resource_locations = []

        for key, value in agent.local_memory.items():
            if key.startswith("explorer_report_") and isinstance(value, dict):
                resource_locations.append(value["position"])

        # Move towards promising locations
        if resource_locations:
            # Choose closest resource location
            current_pos = agent.position
            closest_resource = min(resource_locations,
                                 key=lambda pos: math.sqrt((pos[0] - current_pos[0])**2 + (pos[1] - current_pos[1])**2))

            # Move towards resource
            dx = (closest_resource[0] - current_pos[0]) * 0.2
            dy = (closest_resource[1] - current_pos[1]) * 0.2

            new_x = max(0, min(self.environment_size[0], current_pos[0] + dx))
            new_y = max(0, min(self.environment_size[1], current_pos[1] + dy))

            agent.position = (new_x, new_y)
            agent.velocity = (dx, dy)

            # Simulate resource collection
            if abs(new_x - closest_resource[0]) < 5 and abs(new_y - closest_resource[1]) < 5:
                agent.energy += 10  # Gained energy from resource
                agent.local_memory[f"resource_collected_{iteration}"] = {
                    "location": closest_resource,
                    "energy_gained": 10,
                    "iteration": iteration
                }

        # Share resource information with coordinators
        for neighbor in neighbors:
            if neighbor.role == "coordinator":
                neighbor.local_memory[f"forager_report_{agent.agent_id}"] = {
                    "position": agent.position,
                    "energy": agent.energy,
                    "resources_found": len([k for k in agent.local_memory.keys() if "resource_collected" in k]),
                    "iteration": iteration
                }

        print(f"   🍯 Forager {agent.agent_id}: Energy {agent.energy:.1f}, position ({agent.position[0]:.1f}, {agent.position[1]:.1f})")

    async def coordinator_behavior(self, agent: SwarmAgent, neighbors: List[SwarmAgent], iteration: int):
        """Coordinator agents optimize swarm organization"""
        # Collect information from all agent types
        explorer_reports = {}
        forager_reports = {}

        for key, value in agent.local_memory.items():
            if key.startswith("explorer_report_"):
                explorer_reports[key] = value
            elif key.startswith("forager_report_"):
                forager_reports[key] = value

        # Calculate swarm metrics
        swarm_metrics = {
            "active_explorers": len(explorer_reports),
            "active_foragers": len(forager_reports),
            "total_energy": sum(report.get("energy", 0) for report in forager_reports.values()),
            "exploration_coverage": len(set(
                f"{int(report['position'][0]//10)}_{int(report['position'][1]//10)}"
                for report in explorer_reports.values()
            )),
            "iteration": iteration
        }

        # Store in collective memory
        self.collective_memory[f"swarm_metrics_{iteration}"] = swarm_metrics

        # Coordinate swarm behavior adjustments
        if swarm_metrics["exploration_coverage"] < 5:  # Need more exploration
            agent.local_memory["swarm_directive"] = "increase_exploration"
        elif swarm_metrics["total_energy"] < 50:  # Need more foraging
            agent.local_memory["swarm_directive"] = "increase_foraging"
        else:
            agent.local_memory["swarm_directive"] = "maintain_balance"

        # Move to central location for better coordination
        center_x = self.environment_size[0] / 2
        center_y = self.environment_size[1] / 2

        dx = (center_x - agent.position[0]) * 0.05  # Slow movement to center
        dy = (center_y - agent.position[1]) * 0.05

        new_x = agent.position[0] + dx
        new_y = agent.position[1] + dy

        agent.position = (new_x, new_y)

        print(f"   🎯 Coordinator {agent.agent_id}: Directive '{swarm_metrics}', {len(neighbors)} neighbors")

    async def analyzer_behavior(self, agent: SwarmAgent, neighbors: List[SwarmAgent], iteration: int):
        """Analyzer agents process collective intelligence"""
        # Analyze patterns in collective memory
        recent_metrics = []
        for key, value in self.collective_memory.items():
            if key.startswith("swarm_metrics_") and value["iteration"] >= iteration - 3:
                recent_metrics.append(value)

        if recent_metrics:
            # Calculate trends
            avg_exploration = sum(m["exploration_coverage"] for m in recent_metrics) / len(recent_metrics)
            avg_energy = sum(m["total_energy"] for m in recent_metrics) / len(recent_metrics)

            # Generate insights
            insights = {
                "exploration_trend": "increasing" if len(recent_metrics) > 1 and recent_metrics[-1]["exploration_coverage"] > avg_exploration else "stable",
                "energy_trend": "increasing" if len(recent_metrics) > 1 and recent_metrics[-1]["total_energy"] > avg_energy else "stable",
                "swarm_efficiency": (avg_exploration * avg_energy) / 100,
                "iteration": iteration
            }

            agent.local_memory["swarm_analysis"] = insights

            # Share insights with coordinators
            for neighbor in neighbors:
                if neighbor.role == "coordinator":
                    neighbor.local_memory[f"analysis_report_{agent.agent_id}"] = insights

        print(f"   📊 Analyzer {agent.agent_id}: Analyzing swarm patterns, {len(recent_metrics)} data points")

    def update_global_state(self, iteration: int):
        """Update global swarm state"""
        total_energy = sum(agent.energy for agent in self.agents)
        avg_position_x = sum(agent.position[0] for agent in self.agents) / len(self.agents)
        avg_position_y = sum(agent.position[1] for agent in self.agents) / len(self.agents)

        self.global_state[f"iteration_{iteration}"] = {
            "total_agents": len(self.agents),
            "total_energy": total_energy,
            "center_of_mass": (avg_position_x, avg_position_y),
            "communication_connections": sum(len(connections) for connections in self.communication_network.values()),
            "timestamp": time.time()
        }

# Create swarm intelligence system
swarm_system = SwarmIntelligence(environment_size=(50.0, 50.0))

# Create diverse swarm agents
swarm_agents = [
    SwarmAgent("explorer_1", position=(10, 10), velocity=(0, 0), energy=100, role="explorer"),
    SwarmAgent("explorer_2", position=(40, 40), velocity=(0, 0), energy=100, role="explorer"),
    SwarmAgent("forager_1", position=(25, 25), velocity=(0, 0), energy=80, role="forager"),
    SwarmAgent("forager_2", position=(15, 35), velocity=(0, 0), energy=80, role="forager"),
    SwarmAgent("coordinator_1", position=(25, 25), velocity=(0, 0), energy=120, role="coordinator"),
    SwarmAgent("analyzer_1", position=(30, 20), velocity=(0, 0), energy=90, role="analyzer")
]

for agent in swarm_agents:
    swarm_system.add_agent(agent)

# Swarm behavior demonstration
swarm_agent = Agent("swarm-intelligence")

@state(timeout=120.0)
async def execute_swarm_intelligence(context):
    """Execute swarm intelligence demonstration"""
    print("🐝 Starting swarm intelligence simulation...")

    initial_state = {
        "agents": len(swarm_system.agents),
        "environment_size": swarm_system.environment_size,
        "agent_roles": {agent.role: len([a for a in swarm_system.agents if a.role == agent.role])
                       for agent in swarm_system.agents}
    }

    print(f"🎯 Initial swarm configuration: {initial_state}")

    # Execute swarm behavior
    await swarm_system.execute_swarm_behavior(iterations=8)

    # Analyze emergent behavior
    final_analysis = analyze_swarm_emergence(swarm_system)

    context.set_variable("swarm_results", {
        "initial_state": initial_state,
        "final_analysis": final_analysis,
        "global_state_history": swarm_system.global_state,
        "collective_memory": swarm_system.collective_memory
    })

    print("🎉 Swarm intelligence simulation completed")

def analyze_swarm_emergence(swarm: SwarmIntelligence) -> Dict:
    """Analyze emergent behaviors in the swarm"""
    analysis = {
        "energy_distribution": {agent.agent_id: agent.energy for agent in swarm.agents},
        "spatial_distribution": {agent.agent_id: agent.position for agent in swarm.agents},
        "communication_efficiency": len([conn for connections in swarm.communication_network.values()
                                       for conn in connections]) / (len(swarm.agents) * (len(swarm.agents) - 1)),
        "role_specialization": {},
        "collective_intelligence_indicators": {}
    }

    # Analyze role specialization
    for role in ["explorer", "forager", "coordinator", "analyzer"]:
        role_agents = [a for a in swarm.agents if a.role == role]
        if role_agents:
            analysis["role_specialization"][role] = {
                "count": len(role_agents),
                "avg_energy": sum(a.energy for a in role_agents) / len(role_agents),
                "memory_complexity": sum(len(a.local_memory) for a in role_agents) / len(role_agents)
            }

    # Collective intelligence indicators
    total_memory_entries = sum(len(agent.local_memory) for agent in swarm.agents)
    shared_knowledge = len(swarm.collective_memory)

    analysis["collective_intelligence_indicators"] = {
        "total_individual_memory": total_memory_entries,
        "shared_collective_memory": shared_knowledge,
        "knowledge_sharing_ratio": shared_knowledge / max(total_memory_entries, 1),
        "swarm_cohesion": calculate_swarm_cohesion(swarm.agents)
    }

    return analysis

def calculate_swarm_cohesion(agents: List[SwarmAgent]) -> float:
    """Calculate how cohesive the swarm is based on spatial clustering"""
    if len(agents) < 2:
        return 1.0

    # Calculate average distance between all agent pairs
    total_distance = 0
    pair_count = 0

    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents[i+1:], i+1):
            distance = agent1.distance_to(agent2)
            total_distance += distance
            pair_count += 1

    avg_distance = total_distance / pair_count if pair_count > 0 else 0

    # Normalize cohesion (lower distance = higher cohesion)
    max_possible_distance = math.sqrt(50**2 + 50**2)  # Diagonal of environment
    cohesion = 1.0 - (avg_distance / max_possible_distance)

    return max(0.0, min(1.0, cohesion))

swarm_agent.add_state("execute_swarm", execute_swarm_intelligence)
\`\`\`

---

## Best Practices Summary

### Multi-Agent System Design Principles

1. **Define Clear Agent Roles**
   - Specialized capabilities and responsibilities
   - Clear boundaries and interfaces
   - Complementary skill sets

2. **Establish Communication Protocols**
   - Standardized message formats
   - Reliable delivery mechanisms
   - Timeout and retry handling

3. **Design for Scalability**
   - Loose coupling between agents
   - Horizontal scaling capabilities
   - Load distribution strategies

4. **Implement Coordination Mechanisms**
   - Shared coordination primitives
   - Conflict resolution strategies
   - Synchronization points

5. **Monitor Emergent Behavior**
   - Track system-level metrics
   - Identify unexpected patterns
   - Adapt coordination strategies

### Multi-Agent Architecture Patterns

\`\`\`python
# Agent Pool Pattern
@state(max_concurrent=5)
async def pooled_worker(context): pass

# Agent Pipeline Pattern
@state(dependencies=["previous_stage"])
async def pipeline_stage(context): pass

# Agent Hierarchy Pattern
@state(priority=Priority.HIGH)  # Leader
async def coordinator(context): pass

@state(priority=Priority.NORMAL)  # Worker
async def subordinate(context): pass

# Agent Communication
message = Message(
    sender_id="agent_1",
    recipient_id="agent_2",
    message_type=MessageType.REQUEST,
    content={"task": "process_data"}
)
await message_bus.send_message(message)
\`\`\`

Multi-agent systems in Puffinflow enable you to build sophisticated AI applications that leverage the power of collaboration, specialization, and emergent intelligence to solve complex problems that would be difficult for single agents to handle.
`.trim();
