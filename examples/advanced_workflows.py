"""
Advanced Workflows Examples

This example demonstrates advanced workflow patterns and complex scenarios:
- Complex multi-stage workflows
- Conditional execution and branching
- Dynamic workflow generation
- Error recovery and compensation
- Workflow orchestration patterns
"""

import asyncio
import random
import time
from typing import Any, Optional

from puffinflow import (
    Agent,
    Context,
    cpu_intensive,
    create_team,
    memory_intensive,
    state,
)


class WorkflowOrchestrator(Agent):
    """Orchestrator that manages complex multi-stage workflows."""

    def __init__(self, name: str, workflow_config: Optional[dict[str, Any]] = None):
        super().__init__(name)
        self.workflow_config = workflow_config or {}
        self.workflow_state = {}
        self.execution_history = []

        # Register all decorated states
        self.add_state("initialize_workflow", self.initialize_workflow)
        self.add_state("plan_execution", self.plan_execution)
        self.add_state("execute_workflow", self.execute_workflow)
        self.add_state("finalize_workflow", self.finalize_workflow)

    @state(cpu=1.0, memory=256.0)
    async def initialize_workflow(self, context: Context):
        """Initialize the workflow with configuration and planning."""
        workflow_id = f"workflow_{int(time.time())}"

        # Set up workflow configuration
        default_config = {
            "max_parallel_stages": 3,
            "timeout_per_stage": 30.0,
            "retry_failed_stages": True,
            "enable_compensation": True,
            "workflow_type": "data_processing",
        }

        config = {**default_config, **self.workflow_config}

        self.workflow_state = {
            "workflow_id": workflow_id,
            "status": "initializing",
            "stages_completed": 0,
            "total_stages": 0,
            "start_time": time.time(),
            "config": config,
        }

        context.set_output("workflow_id", workflow_id)
        context.set_output("workflow_config", config)

        print(f"{self.name} initialized workflow: {workflow_id}")
        return "plan_execution"

    @state(cpu=2.0, memory=512.0)
    async def plan_execution(self, context: Context):
        """Plan the workflow execution strategy."""
        config = context.get_output("workflow_config", {})
        workflow_type = config.get("workflow_type", "data_processing")

        # Define execution plan based on workflow type
        execution_plans = {
            "data_processing": {
                "stages": [
                    {"name": "data_ingestion", "parallel": False, "critical": True},
                    {"name": "data_validation", "parallel": True, "critical": True},
                    {
                        "name": "data_transformation",
                        "parallel": True,
                        "critical": False,
                    },
                    {"name": "data_analysis", "parallel": False, "critical": False},
                    {"name": "report_generation", "parallel": False, "critical": True},
                ],
                "dependencies": {
                    "data_validation": ["data_ingestion"],
                    "data_transformation": ["data_validation"],
                    "data_analysis": ["data_transformation"],
                    "report_generation": ["data_analysis"],
                },
            },
            "ml_pipeline": {
                "stages": [
                    {"name": "data_preparation", "parallel": False, "critical": True},
                    {"name": "feature_engineering", "parallel": True, "critical": True},
                    {"name": "model_training", "parallel": True, "critical": True},
                    {"name": "model_validation", "parallel": False, "critical": True},
                    {"name": "model_deployment", "parallel": False, "critical": True},
                ],
                "dependencies": {
                    "feature_engineering": ["data_preparation"],
                    "model_training": ["feature_engineering"],
                    "model_validation": ["model_training"],
                    "model_deployment": ["model_validation"],
                },
            },
        }

        execution_plan = execution_plans.get(
            workflow_type, execution_plans["data_processing"]
        )

        self.workflow_state["total_stages"] = len(execution_plan["stages"])
        self.workflow_state["execution_plan"] = execution_plan

        context.set_output("execution_plan", execution_plan)
        context.set_metric("planned_stages", len(execution_plan["stages"]))

        print(f"{self.name} planned {len(execution_plan['stages'])} stages")
        return "execute_workflow"

    @cpu_intensive(cpu=4.0, memory=1024.0)
    async def execute_workflow(self, context: Context):
        """Execute the planned workflow."""
        execution_plan = context.get_output("execution_plan", {})
        stages = execution_plan.get("stages", [])
        dependencies = execution_plan.get("dependencies", {})

        completed_stages = set()
        failed_stages = set()
        stage_results = {}

        # Execute stages according to dependencies
        for stage in stages:
            stage_name = stage["name"]

            # Check if dependencies are satisfied
            stage_deps = dependencies.get(stage_name, [])
            if not all(dep in completed_stages for dep in stage_deps):
                print(f"  Skipping {stage_name} - dependencies not met")
                continue

            # Execute stage
            print(f"  Executing stage: {stage_name}")
            time.time()

            try:
                # Simulate stage execution
                execution_time = random.uniform(0.1, 0.3)
                await asyncio.sleep(execution_time)

                # Simulate occasional failures
                if random.random() < 0.1:  # 10% failure rate
                    raise Exception(f"Stage {stage_name} failed")

                stage_result = {
                    "stage": stage_name,
                    "status": "completed",
                    "execution_time": execution_time,
                    "output_size": random.randint(100, 1000),
                    "quality_score": random.uniform(0.8, 1.0),
                }

                stage_results[stage_name] = stage_result
                completed_stages.add(stage_name)

                self.workflow_state["stages_completed"] += 1

                print(f"    ✓ {stage_name} completed in {execution_time:.2f}s")

            except Exception as e:
                print(f"    ✗ {stage_name} failed: {e}")
                failed_stages.add(stage_name)

                # Handle critical stage failures
                if stage.get("critical", False):
                    print(
                        f"    Critical stage {stage_name} failed - initiating recovery"
                    )
                    recovery_result = await self.handle_stage_failure(
                        stage_name, str(e)
                    )
                    if recovery_result:
                        completed_stages.add(stage_name)
                        stage_results[stage_name] = recovery_result
                    else:
                        break  # Stop workflow on critical failure

        # Calculate workflow metrics
        total_time = time.time() - self.workflow_state["start_time"]
        success_rate = len(completed_stages) / len(stages) if stages else 0

        workflow_result = {
            "workflow_id": self.workflow_state["workflow_id"],
            "total_execution_time": total_time,
            "stages_completed": len(completed_stages),
            "stages_failed": len(failed_stages),
            "success_rate": success_rate,
            "stage_results": stage_results,
            "overall_status": "completed" if success_rate > 0.8 else "partial_failure",
        }

        context.set_output("workflow_result", workflow_result)
        context.set_metric("workflow_success_rate", success_rate)
        context.set_metric("total_execution_time", total_time)

        print(f"{self.name} workflow completed: {success_rate:.1%} success rate")
        return "finalize_workflow"

    async def handle_stage_failure(self, stage_name: str, error: str) -> dict[str, Any]:
        """Handle stage failure with recovery mechanisms."""
        print(f"    Attempting recovery for {stage_name}")

        # Simulate recovery attempt
        await asyncio.sleep(0.1)

        # 70% chance of successful recovery
        if random.random() < 0.7:
            return {
                "stage": stage_name,
                "status": "recovered",
                "execution_time": 0.1,
                "output_size": 50,  # Reduced output after recovery
                "quality_score": 0.6,  # Lower quality after recovery
                "recovery_method": "fallback_processing",
            }

        return None

    @state(cpu=0.5, memory=128.0)
    async def finalize_workflow(self, context: Context):
        """Finalize the workflow and generate summary."""
        workflow_result = context.get_output("workflow_result", {})

        # Generate workflow summary
        summary = {
            "workflow_id": workflow_result.get("workflow_id"),
            "execution_summary": {
                "total_time": workflow_result.get("total_execution_time", 0),
                "success_rate": workflow_result.get("success_rate", 0),
                "stages_completed": workflow_result.get("stages_completed", 0),
                "overall_status": workflow_result.get("overall_status", "unknown"),
            },
            "performance_metrics": {
                "avg_stage_time": self.calculate_avg_stage_time(workflow_result),
                "throughput": self.calculate_throughput(workflow_result),
                "efficiency_score": workflow_result.get("success_rate", 0) * 100,
            },
            "recommendations": self.generate_recommendations(workflow_result),
        }

        context.set_output("workflow_summary", summary)

        print(f"{self.name} workflow finalized")
        print(f"  Status: {summary['execution_summary']['overall_status']}")
        print(
            f"  Efficiency: {summary['performance_metrics']['efficiency_score']:.1f}%"
        )

        return None

    def calculate_avg_stage_time(self, workflow_result: dict[str, Any]) -> float:
        """Calculate average stage execution time."""
        stage_results = workflow_result.get("stage_results", {})
        if not stage_results:
            return 0.0

        total_time = sum(
            result.get("execution_time", 0) for result in stage_results.values()
        )
        return total_time / len(stage_results)

    def calculate_throughput(self, workflow_result: dict[str, Any]) -> float:
        """Calculate workflow throughput."""
        total_time = workflow_result.get("total_execution_time", 1)
        stages_completed = workflow_result.get("stages_completed", 0)
        return stages_completed / total_time

    def generate_recommendations(self, workflow_result: dict[str, Any]) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        success_rate = workflow_result.get("success_rate", 0)
        avg_stage_time = self.calculate_avg_stage_time(workflow_result)

        if success_rate < 0.9:
            recommendations.append("Improve error handling and recovery mechanisms")

        if avg_stage_time > 0.25:
            recommendations.append("Optimize stage execution performance")

        if workflow_result.get("stages_failed", 0) > 0:
            recommendations.append(
                "Review failed stages and implement preventive measures"
            )

        if not recommendations:
            recommendations.append("Workflow performing optimally")

        return recommendations


class ConditionalWorkflowAgent(Agent):
    """Agent that demonstrates conditional workflow execution."""

    def __init__(self, name: str):
        super().__init__(name)
        # Register all decorated states
        self.add_state("evaluate_conditions", self.evaluate_conditions)
        self.add_state("fast_track_processing", self.fast_track_processing)
        self.add_state("enhanced_validation", self.enhanced_validation)
        self.add_state("optimized_processing", self.optimized_processing)
        self.add_state("expedited_processing", self.expedited_processing)
        self.add_state("standard_processing", self.standard_processing)

    @state(cpu=1.0, memory=256.0)
    async def evaluate_conditions(self, context: Context):
        """Evaluate conditions to determine workflow path."""
        # Simulate different business conditions
        conditions = {
            "data_quality_score": random.uniform(0.6, 1.0),
            "system_load": random.uniform(0.3, 0.9),
            "business_priority": random.choice(["low", "medium", "high"]),
            "resource_availability": random.uniform(0.5, 1.0),
            "time_constraint": random.choice(["relaxed", "normal", "urgent"]),
        }

        # Determine execution path based on conditions
        execution_path = self.determine_execution_path(conditions)

        context.set_output("conditions", conditions)
        context.set_output("execution_path", execution_path)

        print(f"{self.name} evaluated conditions: {execution_path}")
        return execution_path

    def determine_execution_path(self, conditions: dict[str, Any]) -> str:
        """Determine which execution path to take based on conditions."""
        data_quality = conditions["data_quality_score"]
        system_load = conditions["system_load"]
        priority = conditions["business_priority"]
        time_constraint = conditions["time_constraint"]

        # High priority and good data quality
        if priority == "high" and data_quality > 0.8:
            return "fast_track_processing"

        # Low data quality requires validation
        elif data_quality < 0.7:
            return "enhanced_validation"

        # High system load requires optimization
        elif system_load > 0.8:
            return "optimized_processing"

        # Urgent time constraint
        elif time_constraint == "urgent":
            return "expedited_processing"

        # Default path
        else:
            return "standard_processing"

    @cpu_intensive(cpu=3.0, memory=768.0)
    async def fast_track_processing(self, context: Context):
        """Fast track processing for high priority items."""
        await asyncio.sleep(0.1)  # Fast processing

        result = {
            "processing_type": "fast_track",
            "execution_time": 0.1,
            "quality_level": "high",
            "throughput": 1000,
        }

        context.set_output("processing_result", result)
        print(f"{self.name} completed fast track processing")
        return None

    @state(cpu=2.0, memory=512.0)
    async def enhanced_validation(self, context: Context):
        """Enhanced validation for low quality data."""
        await asyncio.sleep(0.3)  # Thorough validation

        result = {
            "processing_type": "enhanced_validation",
            "execution_time": 0.3,
            "quality_level": "validated",
            "validation_score": 0.95,
            "corrections_made": random.randint(5, 20),
        }

        context.set_output("processing_result", result)
        print(f"{self.name} completed enhanced validation")
        return None

    @memory_intensive(memory=1024.0, cpu=1.5)
    async def optimized_processing(self, context: Context):
        """Optimized processing for high system load."""
        await asyncio.sleep(0.2)  # Optimized processing

        result = {
            "processing_type": "optimized",
            "execution_time": 0.2,
            "quality_level": "standard",
            "resource_efficiency": 0.9,
            "memory_optimized": True,
        }

        context.set_output("processing_result", result)
        print(f"{self.name} completed optimized processing")
        return None

    @state(cpu=4.0, memory=256.0, priority="high")
    async def expedited_processing(self, context: Context):
        """Expedited processing for urgent requests."""
        await asyncio.sleep(0.05)  # Very fast processing

        result = {
            "processing_type": "expedited",
            "execution_time": 0.05,
            "quality_level": "standard",
            "priority_boost": True,
            "sla_compliance": True,
        }

        context.set_output("processing_result", result)
        print(f"{self.name} completed expedited processing")
        return None

    @state(cpu=2.0, memory=512.0)
    async def standard_processing(self, context: Context):
        """Standard processing path."""
        await asyncio.sleep(0.2)  # Standard processing

        result = {
            "processing_type": "standard",
            "execution_time": 0.2,
            "quality_level": "standard",
            "cost_efficiency": 0.8,
        }

        context.set_output("processing_result", result)
        print(f"{self.name} completed standard processing")
        return None


class DynamicWorkflowGenerator(Agent):
    """Agent that generates workflows dynamically based on input requirements."""

    def __init__(self, name: str):
        super().__init__(name)
        # Register all decorated states
        self.add_state("analyze_requirements", self.analyze_requirements)
        self.add_state("validate_workflow", self.validate_workflow)
        self.add_state("optimize_workflow", self.optimize_workflow)
        self.add_state("execute_dynamic_workflow", self.execute_dynamic_workflow)

    @state(cpu=2.0, memory=512.0)
    async def analyze_requirements(self, context: Context):
        """Analyze input requirements to generate appropriate workflow."""
        # Simulate different types of requirements
        requirements = {
            "data_volume": random.choice(["small", "medium", "large", "xlarge"]),
            "processing_complexity": random.choice(["simple", "moderate", "complex"]),
            "accuracy_requirement": random.uniform(0.8, 0.99),
            "time_budget": random.uniform(1.0, 10.0),  # seconds
            "resource_constraints": {
                "max_cpu": random.uniform(2.0, 8.0),
                "max_memory": random.uniform(512, 2048),
            },
        }

        # Generate workflow specification
        workflow_spec = self.generate_workflow_spec(requirements)

        context.set_output("requirements", requirements)
        context.set_output("workflow_spec", workflow_spec)

        print(
            f"{self.name} generated workflow with {len(workflow_spec['stages'])} stages"
        )
        return "validate_workflow"

    def generate_workflow_spec(self, requirements: dict[str, Any]) -> dict[str, Any]:
        """Generate workflow specification based on requirements."""
        data_volume = requirements["data_volume"]
        complexity = requirements["processing_complexity"]
        accuracy = requirements["accuracy_requirement"]

        # Base workflow stages
        stages = ["input_validation"]

        # Add stages based on data volume
        if data_volume in ["large", "xlarge"]:
            stages.extend(["data_partitioning", "parallel_processing"])
        else:
            stages.append("sequential_processing")

        # Add stages based on complexity
        if complexity == "complex":
            stages.extend(["advanced_analysis", "optimization"])
        elif complexity == "moderate":
            stages.append("standard_analysis")

        # Add stages based on accuracy requirements
        if accuracy > 0.95:
            stages.extend(["quality_assurance", "validation"])

        # Always end with output generation
        stages.append("output_generation")

        return {
            "stages": stages,
            "estimated_duration": len(stages) * 0.2,
            "resource_requirements": self.calculate_resource_requirements(
                stages, requirements
            ),
            "parallelization_opportunities": self.identify_parallelization(stages),
        }

    def calculate_resource_requirements(
        self, stages: list[str], requirements: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate resource requirements for the workflow."""
        base_cpu = 1.0
        base_memory = 256.0

        # Adjust based on stages
        cpu_multiplier = 1.0 + (len(stages) * 0.2)
        memory_multiplier = 1.0 + (len(stages) * 0.3)

        # Adjust based on data volume
        volume_multipliers = {"small": 1.0, "medium": 1.5, "large": 2.5, "xlarge": 4.0}

        volume = requirements.get("data_volume", "medium")
        volume_mult = volume_multipliers.get(volume, 1.5)

        return {
            "cpu": min(
                base_cpu * cpu_multiplier * volume_mult,
                requirements["resource_constraints"]["max_cpu"],
            ),
            "memory": min(
                base_memory * memory_multiplier * volume_mult,
                requirements["resource_constraints"]["max_memory"],
            ),
        }

    def identify_parallelization(self, stages: list[str]) -> list[str]:
        """Identify stages that can be parallelized."""
        parallelizable = []

        parallel_candidates = [
            "parallel_processing",
            "advanced_analysis",
            "quality_assurance",
        ]

        for stage in stages:
            if stage in parallel_candidates:
                parallelizable.append(stage)

        return parallelizable

    @state(cpu=1.0, memory=256.0)
    async def validate_workflow(self, context: Context):
        """Validate the generated workflow specification."""
        workflow_spec = context.get_output("workflow_spec", {})
        requirements = context.get_output("requirements", {})

        # Perform validation checks
        validation_results = {
            "resource_feasibility": self.check_resource_feasibility(
                workflow_spec, requirements
            ),
            "time_feasibility": self.check_time_feasibility(
                workflow_spec, requirements
            ),
            "quality_feasibility": self.check_quality_feasibility(
                workflow_spec, requirements
            ),
            "overall_score": 0.0,
        }

        # Calculate overall validation score
        scores = [v for k, v in validation_results.items() if k != "overall_score"]
        validation_results["overall_score"] = sum(scores) / len(scores)

        context.set_output("validation_results", validation_results)
        context.set_metric("workflow_feasibility", validation_results["overall_score"])

        if validation_results["overall_score"] > 0.8:
            print(
                f"{self.name} workflow validation passed: {validation_results['overall_score']:.2f}"
            )
            return "execute_dynamic_workflow"
        else:
            print(
                f"{self.name} workflow validation failed: {validation_results['overall_score']:.2f}"
            )
            return "optimize_workflow"

    def check_resource_feasibility(
        self, workflow_spec: dict[str, Any], requirements: dict[str, Any]
    ) -> float:
        """Check if resource requirements are feasible."""
        required = workflow_spec.get("resource_requirements", {})
        constraints = requirements.get("resource_constraints", {})

        cpu_ratio = required.get("cpu", 0) / constraints.get("max_cpu", 1)
        memory_ratio = required.get("memory", 0) / constraints.get("max_memory", 1)

        return 1.0 - max(0, max(cpu_ratio, memory_ratio) - 1.0)

    def check_time_feasibility(
        self, workflow_spec: dict[str, Any], requirements: dict[str, Any]
    ) -> float:
        """Check if time requirements are feasible."""
        estimated_duration = workflow_spec.get("estimated_duration", 0)
        time_budget = requirements.get("time_budget", 1)

        time_ratio = estimated_duration / time_budget
        return 1.0 - max(0, time_ratio - 1.0)

    def check_quality_feasibility(
        self, workflow_spec: dict[str, Any], requirements: dict[str, Any]
    ) -> float:
        """Check if quality requirements can be met."""
        stages = workflow_spec.get("stages", [])
        accuracy_requirement = requirements.get("accuracy_requirement", 0.8)

        # Estimate achievable accuracy based on stages
        quality_stages = ["quality_assurance", "validation", "advanced_analysis"]
        quality_boost = sum(0.05 for stage in stages if stage in quality_stages)

        estimated_accuracy = 0.8 + quality_boost

        return min(1.0, estimated_accuracy / accuracy_requirement)

    @state(cpu=1.5, memory=384.0)
    async def optimize_workflow(self, context: Context):
        """Optimize workflow specification to meet requirements."""
        workflow_spec = context.get_output("workflow_spec", {})
        validation_results = context.get_output("validation_results", {})

        # Apply optimizations based on validation failures
        optimized_spec = workflow_spec.copy()

        if validation_results.get("resource_feasibility", 1.0) < 0.8:
            # Reduce resource requirements
            optimized_spec["resource_requirements"]["cpu"] *= 0.8
            optimized_spec["resource_requirements"]["memory"] *= 0.8
            print("  Applied resource optimization")

        if validation_results.get("time_feasibility", 1.0) < 0.8:
            # Add parallelization
            optimized_spec["parallelization_opportunities"].extend(
                ["input_validation", "output_generation"]
            )
            optimized_spec["estimated_duration"] *= 0.7
            print("  Applied time optimization")

        context.set_output("optimized_workflow_spec", optimized_spec)
        print(f"{self.name} optimized workflow specification")
        return "execute_dynamic_workflow"

    @cpu_intensive(cpu=3.0, memory=768.0)
    async def execute_dynamic_workflow(self, context: Context):
        """Execute the dynamically generated workflow."""
        # Use optimized spec if available, otherwise use original
        workflow_spec = context.get_output(
            "optimized_workflow_spec"
        ) or context.get_output("workflow_spec", {})

        stages = workflow_spec.get("stages", [])
        parallelizable = workflow_spec.get("parallelization_opportunities", [])

        execution_results = []
        total_start_time = time.time()

        # Execute stages
        for stage in stages:
            time.time()

            # Simulate stage execution
            if stage in parallelizable:
                # Simulate parallel execution (faster)
                await asyncio.sleep(0.05)
                execution_time = 0.05
            else:
                # Simulate sequential execution
                await asyncio.sleep(0.1)
                execution_time = 0.1

            stage_result = {
                "stage": stage,
                "execution_time": execution_time,
                "status": "completed",
                "parallel": stage in parallelizable,
            }

            execution_results.append(stage_result)
            print(
                f"  Executed {stage} ({'parallel' if stage in parallelizable else 'sequential'})"
            )

        total_execution_time = time.time() - total_start_time

        dynamic_result = {
            "total_stages": len(stages),
            "total_execution_time": total_execution_time,
            "parallel_stages": len([s for s in execution_results if s["parallel"]]),
            "sequential_stages": len(
                [s for s in execution_results if not s["parallel"]]
            ),
            "efficiency_score": len(stages) / total_execution_time,
            "stage_results": execution_results,
        }

        context.set_output("dynamic_execution_result", dynamic_result)
        context.set_metric(
            "dynamic_workflow_efficiency", dynamic_result["efficiency_score"]
        )

        print(f"{self.name} dynamic workflow completed in {total_execution_time:.2f}s")
        return None


async def demonstrate_complex_orchestration():
    """Demonstrate complex workflow orchestration."""
    print("=== Complex Workflow Orchestration ===")

    # Create orchestrator with different workflow types
    orchestrators = [
        WorkflowOrchestrator("data-orchestrator", {"workflow_type": "data_processing"}),
        WorkflowOrchestrator("ml-orchestrator", {"workflow_type": "ml_pipeline"}),
    ]

    # Run orchestrators
    for orchestrator in orchestrators:
        result = await orchestrator.run()

        workflow_summary = result.get_output("workflow_summary", {})
        execution_summary = workflow_summary.get("execution_summary", {})
        performance_metrics = workflow_summary.get("performance_metrics", {})

        print(f"  {orchestrator.name}:")
        print(f"    Status: {execution_summary.get('overall_status', 'unknown')}")
        print(f"    Success Rate: {execution_summary.get('success_rate', 0):.1%}")
        print(f"    Efficiency: {performance_metrics.get('efficiency_score', 0):.1f}%")

    print()


async def demonstrate_conditional_workflows():
    """Demonstrate conditional workflow execution."""
    print("=== Conditional Workflow Execution ===")

    # Create multiple conditional agents with different scenarios
    conditional_agents = [
        ConditionalWorkflowAgent(f"conditional-agent-{i}") for i in range(3)
    ]

    # Run agents and analyze execution paths
    execution_paths = {}

    for agent in conditional_agents:
        result = await agent.run()

        conditions = result.get_output("conditions", {})
        execution_path = result.get_output("execution_path", "unknown")
        processing_result = result.get_output("processing_result", {})

        if execution_path not in execution_paths:
            execution_paths[execution_path] = []

        execution_paths[execution_path].append(
            {"agent": agent.name, "conditions": conditions, "result": processing_result}
        )

        print(
            f"  {agent.name}: {execution_path} "
            f"({processing_result.get('execution_time', 0):.2f}s)"
        )

    # Summarize execution paths
    print("\nExecution Path Summary:")
    for path, executions in execution_paths.items():
        print(f"  {path}: {len(executions)} executions")

    print()


async def demonstrate_dynamic_workflows():
    """Demonstrate dynamic workflow generation."""
    print("=== Dynamic Workflow Generation ===")

    # Create dynamic workflow generators
    generators = [DynamicWorkflowGenerator(f"dynamic-generator-{i}") for i in range(2)]

    # Run generators and analyze results
    for generator in generators:
        result = await generator.run()

        requirements = result.get_output("requirements", {})
        workflow_spec = result.get_output("workflow_spec", {})
        dynamic_result = result.get_output("dynamic_execution_result", {})

        print(f"  {generator.name}:")
        print(f"    Data Volume: {requirements.get('data_volume', 'unknown')}")
        print(f"    Complexity: {requirements.get('processing_complexity', 'unknown')}")
        print(f"    Generated Stages: {len(workflow_spec.get('stages', []))}")
        print(
            f"    Execution Time: {dynamic_result.get('total_execution_time', 0):.2f}s"
        )
        print(
            f"    Efficiency: {dynamic_result.get('efficiency_score', 0):.1f} stages/sec"
        )

    print()


async def run_comprehensive_workflow_test():
    """Run a comprehensive test combining all workflow patterns."""
    print("=== Comprehensive Workflow Test ===")

    # Create a team with different workflow agents
    workflow_team = create_team("comprehensive-workflow-team")

    # Add different types of workflow agents
    workflow_team.add_agent(WorkflowOrchestrator("team-orchestrator"))
    workflow_team.add_agent(ConditionalWorkflowAgent("team-conditional"))
    workflow_team.add_agent(DynamicWorkflowGenerator("team-dynamic"))

    # Run the team
    start_time = time.time()
    team_result = await workflow_team.run()
    total_time = time.time() - start_time

    print(f"Comprehensive test completed in {total_time:.2f} seconds")
    print(f"Team status: {team_result.status}")
    print(f"Agents completed: {len(team_result.agent_results)}")

    # Analyze team results
    successful_agents = 0
    for agent_name, agent_result in team_result.agent_results.items():
        if agent_result.status == "completed":
            successful_agents += 1
        print(f"  {agent_name}: {agent_result.status}")

    print(
        f"Success rate: {successful_agents / len(team_result.agent_results) * 100:.1f}%"
    )
    print()


async def main():
    """Run all advanced workflow examples."""
    print("PuffinFlow Advanced Workflows Examples")
    print("=" * 50)

    await demonstrate_complex_orchestration()
    await demonstrate_conditional_workflows()
    await demonstrate_dynamic_workflows()
    await run_comprehensive_workflow_test()

    print("All advanced workflow examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
