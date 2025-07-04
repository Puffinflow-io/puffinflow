"""End-to-end tests for complete PuffinFlow workflows.

These tests simulate real-world scenarios and validate the entire system
from user perspective, including all components working together.
"""

import pytest
import asyncio
import time
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from puffinflow import (
    Agent, Context, AgentTeam, ResourcePool, CircuitBreaker, CircuitBreakerConfig,
    state, cpu_intensive, memory_intensive, io_intensive,
    run_agents_parallel, run_agents_sequential, create_team
)


class DataIngestionAgent(Agent):
    """Agent that simulates data ingestion from external sources."""
    
    def __init__(self, name: str, source_config: dict):
        super().__init__(name)
        self.set_variable("source_config", source_config)
        self.add_state('validate_source', self.validate_source)
        self.add_state('ingest_data', self.ingest_data)
        self.add_state('cleanup', self.cleanup)
    
    @state(cpu=0.5, memory=128.0)
    async def validate_source(self, context: Context):
        """Validate data source configuration."""
        config = self.get_variable("source_config", {})
        
        # Simulate validation
        await asyncio.sleep(0.1)
        
        if not config.get("url") or not config.get("format"):
            context.set_output("error", "Invalid source configuration")
            return "cleanup"
        
        context.set_output("validation_status", "passed")
        context.set_metadata("validated_at", time.time())
        return "ingest_data"
    
    @io_intensive(cpu=1.0, memory=256.0, io_weight=2.0)
    async def ingest_data(self, context: Context):
        """Ingest data from the source."""
        config = self.get_variable("source_config", {})
        
        # Simulate data ingestion
        await asyncio.sleep(0.3)
        
        # Generate mock data based on configuration
        data_size = config.get("expected_records", 1000)
        ingested_data = {
            "records": [{"id": i, "value": f"data_{i}"} for i in range(data_size)],
            "metadata": {
                "source": config.get("url", "unknown"),
                "format": config.get("format", "json"),
                "ingested_at": time.time(),
                "record_count": data_size
            }
        }
        
        context.set_output("ingested_data", ingested_data)
        context.set_output("record_count", data_size)
        context.set_metric("ingestion_rate", data_size / 0.3)
        
        return "cleanup"
    
    @state(cpu=0.2, memory=64.0)
    async def cleanup(self, context: Context):
        """Cleanup resources after ingestion."""
        await asyncio.sleep(0.05)
        
        context.set_output("cleanup_completed", True)
        context.set_metadata("cleanup_at", time.time())
        
        return None


class DataTransformationAgent(Agent):
    """Agent that transforms ingested data."""
    
    def __init__(self, name: str, transformation_rules: dict):
        super().__init__(name)
        self.set_variable("transformation_rules", transformation_rules)
        self.add_state('prepare_transformation', self.prepare_transformation)
        self.add_state('transform_data', self.transform_data)
        self.add_state('validate_output', self.validate_output)
    
    @state(cpu=0.5, memory=256.0)
    async def prepare_transformation(self, context: Context):
        """Prepare transformation pipeline."""
        rules = self.get_variable("transformation_rules", {})
        
        await asyncio.sleep(0.1)
        
        if not rules:
            context.set_output("error", "No transformation rules provided")
            return None
        
        context.set_output("pipeline_ready", True)
        context.set_metadata("pipeline_prepared_at", time.time())
        
        return "transform_data"
    
    @cpu_intensive(cpu=3.0, memory=512.0)
    async def transform_data(self, context: Context):
        """Transform the data according to rules."""
        rules = self.get_variable("transformation_rules", {})
        
        # Simulate CPU-intensive transformation
        await asyncio.sleep(0.4)
        
        # Mock transformation based on rules
        transform_type = rules.get("type", "filter")
        multiplier = rules.get("multiplier", 1.0)
        
        if transform_type == "filter":
            # Simulate filtering
            output_count = int(1000 * 0.8)  # 80% pass filter
        elif transform_type == "aggregate":
            # Simulate aggregation
            output_count = int(1000 * 0.1)  # Aggregate to 10%
        else:
            # Default transformation
            output_count = int(1000 * multiplier)
        
        transformed_data = {
            "transformed_records": output_count,
            "transformation_type": transform_type,
            "processing_time": 0.4,
            "efficiency": output_count / 1000
        }
        
        context.set_output("transformed_data", transformed_data)
        context.set_output("output_count", output_count)
        context.set_metric("transformation_efficiency", output_count / 1000)
        
        return "validate_output"
    
    @state(cpu=0.5, memory=128.0)
    async def validate_output(self, context: Context):
        """Validate transformation output."""
        transformed_data = context.get_output("transformed_data", {})
        
        await asyncio.sleep(0.1)
        
        output_count = transformed_data.get("transformed_records", 0)
        
        if output_count <= 0:
            context.set_output("validation_error", "No records after transformation")
            return None
        
        context.set_output("validation_passed", True)
        context.set_output("quality_score", min(1.0, output_count / 500))
        
        return None


class DataStorageAgent(Agent):
    """Agent that stores processed data."""
    
    def __init__(self, name: str, storage_config: dict):
        super().__init__(name)
        self.set_variable("storage_config", storage_config)
        self.add_state('prepare_storage', self.prepare_storage)
        self.add_state('store_data', self.store_data)
        self.add_state('verify_storage', self.verify_storage)
    
    @state(cpu=0.5, memory=128.0)
    async def prepare_storage(self, context: Context):
        """Prepare storage system."""
        config = self.get_variable("storage_config", {})
        
        await asyncio.sleep(0.1)
        
        storage_type = config.get("type", "file")
        if storage_type not in ["file", "database", "cloud"]:
            context.set_output("error", f"Unsupported storage type: {storage_type}")
            return None
        
        context.set_output("storage_prepared", True)
        context.set_metadata("storage_type", storage_type)
        
        return "store_data"
    
    @io_intensive(cpu=1.0, memory=256.0, io_weight=3.0)
    async def store_data(self, context: Context):
        """Store the processed data."""
        config = self.get_variable("storage_config", {})
        
        # Simulate storage operation
        await asyncio.sleep(0.2)
        
        storage_type = config.get("type", "file")
        batch_size = config.get("batch_size", 100)
        
        # Mock storage operation
        stored_records = 800  # Assume we're storing transformed data
        batches = (stored_records + batch_size - 1) // batch_size
        
        storage_result = {
            "stored_records": stored_records,
            "batches": batches,
            "storage_type": storage_type,
            "storage_time": 0.2,
            "throughput": stored_records / 0.2
        }
        
        context.set_output("storage_result", storage_result)
        context.set_output("stored_count", stored_records)
        context.set_metric("storage_throughput", stored_records / 0.2)
        
        return "verify_storage"
    
    @state(cpu=0.3, memory=64.0)
    async def verify_storage(self, context: Context):
        """Verify data was stored correctly."""
        storage_result = context.get_output("storage_result", {})
        
        await asyncio.sleep(0.05)
        
        stored_count = storage_result.get("stored_records", 0)
        
        # Simulate verification
        verification_passed = stored_count > 0
        
        context.set_output("verification_passed", verification_passed)
        context.set_output("final_count", stored_count)
        context.set_metric("storage_success_rate", 1.0 if verification_passed else 0.0)
        
        return None


class MonitoringAgent(Agent):
    """Agent that monitors the entire workflow."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.add_state('collect_metrics', self.collect_metrics)
        self.add_state('analyze_performance', self.analyze_performance)
        self.add_state('generate_report', self.generate_report)
    
    @state(cpu=0.5, memory=256.0)
    async def collect_metrics(self, context: Context):
        """Collect metrics from the workflow."""
        await asyncio.sleep(0.1)
        
        # Simulate collecting metrics from other agents
        metrics = {
            "workflow_start": time.time() - 2.0,  # Assume workflow started 2s ago
            "agents_monitored": 3,
            "total_records_processed": 800,
            "average_processing_time": 0.25,
            "error_count": 0
        }
        
        context.set_output("collected_metrics", metrics)
        context.set_metric("monitoring_efficiency", 1.0)
        
        return "analyze_performance"
    
    @cpu_intensive(cpu=2.0, memory=512.0)
    async def analyze_performance(self, context: Context):
        """Analyze workflow performance."""
        metrics = context.get_output("collected_metrics", {})
        
        await asyncio.sleep(0.3)
        
        # Simulate performance analysis
        total_records = metrics.get("total_records_processed", 0)
        processing_time = metrics.get("average_processing_time", 0)
        
        performance_score = min(1.0, total_records / (processing_time * 1000))
        
        analysis = {
            "performance_score": performance_score,
            "throughput": total_records / processing_time if processing_time > 0 else 0,
            "efficiency_rating": "high" if performance_score > 0.8 else "medium" if performance_score > 0.5 else "low",
            "bottlenecks": ["transformation"] if performance_score < 0.7 else [],
            "recommendations": ["optimize transformation rules"] if performance_score < 0.7 else ["maintain current configuration"]
        }
        
        context.set_output("performance_analysis", analysis)
        context.set_metric("analysis_accuracy", 0.95)
        
        return "generate_report"
    
    @state(cpu=0.5, memory=128.0)
    async def generate_report(self, context: Context):
        """Generate monitoring report."""
        analysis = context.get_output("performance_analysis", {})
        metrics = context.get_output("collected_metrics", {})
        
        await asyncio.sleep(0.1)
        
        report = {
            "report_id": f"workflow_report_{int(time.time())}",
            "generated_at": time.time(),
            "summary": {
                "total_records": metrics.get("total_records_processed", 0),
                "performance_score": analysis.get("performance_score", 0),
                "efficiency_rating": analysis.get("efficiency_rating", "unknown")
            },
            "details": {
                "metrics": metrics,
                "analysis": analysis
            },
            "recommendations": analysis.get("recommendations", [])
        }
        
        context.set_output("monitoring_report", report)
        context.set_output("report_generated", True)
        
        return None


@pytest.mark.e2e
@pytest.mark.asyncio
class TestCompleteDataProcessingWorkflow:
    """Test complete data processing workflow end-to-end."""
    
    async def test_successful_data_pipeline(self):
        """Test a complete successful data processing pipeline."""
        # Configure the workflow
        ingestion_config = {
            "url": "https://api.example.com/data",
            "format": "json",
            "expected_records": 1000
        }
        
        transformation_config = {
            "type": "filter",
            "multiplier": 0.8
        }
        
        storage_config = {
            "type": "database",
            "batch_size": 100
        }
        
        # Create agents
        ingestion_agent = DataIngestionAgent("data-ingester", ingestion_config)
        transformation_agent = DataTransformationAgent("data-transformer", transformation_config)
        storage_agent = DataStorageAgent("data-storage", storage_config)
        monitoring_agent = MonitoringAgent("workflow-monitor")
        
        # Execute the pipeline sequentially
        pipeline_agents = [ingestion_agent, transformation_agent, storage_agent]
        
        start_time = time.time()
        pipeline_results = await run_agents_sequential(pipeline_agents)
        pipeline_time = time.time() - start_time
        
        # Run monitoring in parallel
        monitoring_result = await monitoring_agent.run()
        
        total_time = time.time() - start_time
        
        # Verify pipeline execution
        assert len(pipeline_results) == 3
        
        # Verify ingestion
        ingestion_result = pipeline_results["data-ingester"]
        assert ingestion_result.status.name in ["COMPLETED", "SUCCESS"]
        assert ingestion_result.get_output("record_count") == 1000
        assert ingestion_result.get_output("cleanup_completed") is True
        
        # Verify transformation
        transformation_result = pipeline_results["data-transformer"]
        assert transformation_result.status.name in ["COMPLETED", "SUCCESS"]
        assert transformation_result.get_output("validation_passed") is True
        transformed_data = transformation_result.get_output("transformed_data", {})
        assert transformed_data["transformation_type"] == "filter"
        
        # Verify storage
        storage_result = pipeline_results["data-storage"]
        assert storage_result.status.name in ["COMPLETED", "SUCCESS"]
        assert storage_result.get_output("verification_passed") is True
        assert storage_result.get_output("stored_count") > 0
        
        # Verify monitoring
        assert monitoring_result.status.name in ["COMPLETED", "SUCCESS"]
        assert monitoring_result.get_output("report_generated") is True
        
        monitoring_report = monitoring_result.get_output("monitoring_report", {})
        assert "report_id" in monitoring_report
        assert "summary" in monitoring_report
        assert "recommendations" in monitoring_report
        
        # Verify end-to-end data flow
        ingested_count = ingestion_result.get_output("record_count", 0)
        transformed_count = transformation_result.get_output("output_count", 0)
        stored_count = storage_result.get_output("stored_count", 0)
        
        # Data should flow through the pipeline with expected transformations
        assert ingested_count == 1000
        assert transformed_count == 800  # 80% after filtering
        assert stored_count == 800  # All transformed data stored
        
        # Verify timing
        assert total_time >= 1.0  # Should take at least 1 second for all operations
        assert total_time <= 5.0  # Should complete within reasonable time
    
    async def test_workflow_with_failures_and_recovery(self):
        """Test workflow behavior with failures and recovery mechanisms."""
        
        class FailingTransformationAgent(DataTransformationAgent):
            def __init__(self, name: str, transformation_rules: dict, fail_on_attempt: int = 1):
                super().__init__(name, transformation_rules)
                self.set_variable("fail_on_attempt", fail_on_attempt)
                self.set_variable("current_attempt", 0)
            
            @cpu_intensive(cpu=3.0, memory=512.0)
            async def transform_data(self, context: Context):
                current_attempt = self.get_variable("current_attempt", 0) + 1
                self.set_variable("current_attempt", current_attempt)
                fail_on_attempt = self.get_variable("fail_on_attempt", 1)
                
                if current_attempt == fail_on_attempt:
                    context.set_output("error", f"Transformation failed on attempt {current_attempt}")
                    raise RuntimeError(f"Simulated failure on attempt {current_attempt}")
                
                # If we get here, the failure attempt has passed
                return await super().transform_data(context)
        
        # Configure workflow with failure-prone agent
        ingestion_config = {
            "url": "https://api.example.com/data",
            "format": "json",
            "expected_records": 500
        }
        
        transformation_config = {
            "type": "aggregate",
            "multiplier": 0.1
        }
        
        storage_config = {
            "type": "file",
            "batch_size": 50
        }
        
        # Create circuit breaker for reliability
        cb_config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.5,
            expected_exception=RuntimeError
        )
        circuit_breaker = CircuitBreaker(cb_config)
        
        # Create agents
        ingestion_agent = DataIngestionAgent("reliable-ingester", ingestion_config)
        failing_transformer = FailingTransformationAgent("failing-transformer", transformation_config, fail_on_attempt=1)
        storage_agent = DataStorageAgent("reliable-storage", storage_config)
        
        # Execute with circuit breaker protection
        async def protected_transformation():
            return await circuit_breaker.call(failing_transformer.run)
        
        # Run ingestion first
        ingestion_result = await ingestion_agent.run()
        assert ingestion_result.status.name in ["COMPLETED", "SUCCESS"]
        
        # Try transformation with circuit breaker (should fail first time)
        transformation_results = []
        transformation_exceptions = []
        
        for attempt in range(3):
            try:
                result = await protected_transformation()
                transformation_results.append(result)
                break  # Success, exit loop
            except Exception as e:
                transformation_exceptions.append(e)
                await asyncio.sleep(0.1)
        
        # Should have at least one failure
        assert len(transformation_exceptions) >= 1
        
        # If transformation eventually succeeded, continue with storage
        if transformation_results:
            storage_result = await storage_agent.run()
            assert storage_result.status.name in ["COMPLETED", "SUCCESS"]
        
        # Verify circuit breaker behavior
        assert circuit_breaker.failure_count >= 1
        assert circuit_breaker.state in ["CLOSED", "HALF_OPEN", "OPEN"]
    
    async def test_high_throughput_workflow(self):
        """Test workflow under high throughput conditions."""
        
        # Create resource pool for high throughput
        resource_pool = ResourcePool(
            total_cpu=8.0,
            total_memory=2048.0,
            total_io=200.0,
            total_network=200.0
        )
        
        # Configure high-throughput workflow
        high_throughput_configs = [
            {
                "url": f"https://api.example.com/data/batch_{i}",
                "format": "json",
                "expected_records": 2000
            }
            for i in range(5)  # 5 parallel ingestion sources
        ]
        
        transformation_config = {
            "type": "filter",
            "multiplier": 0.9
        }
        
        storage_config = {
            "type": "cloud",
            "batch_size": 200
        }
        
        # Create multiple ingestion agents for parallel processing
        ingestion_agents = [
            DataIngestionAgent(f"high-throughput-ingester-{i}", config)
            for i, config in enumerate(high_throughput_configs)
        ]
        
        # Set resource pool for all agents
        for agent in ingestion_agents:
            agent.resource_pool = resource_pool
        
        # Create processing agents
        transformation_agent = DataTransformationAgent("high-throughput-transformer", transformation_config)
        transformation_agent.resource_pool = resource_pool
        
        storage_agent = DataStorageAgent("high-throughput-storage", storage_config)
        storage_agent.resource_pool = resource_pool
        
        # Execute high-throughput workflow
        start_time = time.time()
        
        # Phase 1: Parallel ingestion
        ingestion_results = await run_agents_parallel(ingestion_agents)
        ingestion_time = time.time() - start_time
        
        # Phase 2: Transformation
        transformation_start = time.time()
        transformation_result = await transformation_agent.run()
        transformation_time = time.time() - transformation_start
        
        # Phase 3: Storage
        storage_start = time.time()
        storage_result = await storage_agent.run()
        storage_time = time.time() - storage_start
        
        total_time = time.time() - start_time
        
        # Verify high-throughput performance
        assert len(ingestion_results) == 5
        
        # All ingestion agents should complete successfully
        total_ingested = 0
        for result in ingestion_results.values():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            total_ingested += result.get_output("record_count", 0)
        
        assert total_ingested == 10000  # 5 * 2000 records
        
        # Verify transformation and storage
        assert transformation_result.status.name in ["COMPLETED", "SUCCESS"]
        assert storage_result.status.name in ["COMPLETED", "SUCCESS"]
        
        # Verify throughput metrics
        overall_throughput = total_ingested / total_time
        assert overall_throughput > 1000  # Should process > 1000 records/second
        
        # Parallel ingestion should be faster than sequential
        assert ingestion_time < 2.0  # Should complete in less than 2 seconds
        
        # Verify resource utilization was effective
        transformation_data = transformation_result.get_output("transformed_data", {})
        storage_data = storage_result.get_output("storage_result", {})
        
        assert transformation_data.get("efficiency", 0) > 0.8
        assert storage_data.get("throughput", 0) > 1000
    
    async def test_workflow_with_dynamic_scaling(self):
        """Test workflow that dynamically scales based on load."""
        
        class LoadBalancingAgent(Agent):
            def __init__(self, name: str, load_threshold: int = 1000):
                super().__init__(name)
                self.set_variable("load_threshold", load_threshold)
                self.add_state('assess_load', self.assess_load)
                self.add_state('scale_resources', self.scale_resources)
            
            @state(cpu=1.0, memory=256.0)
            async def assess_load(self, context: Context):
                threshold = self.get_variable("load_threshold", 1000)
                
                # Simulate load assessment
                await asyncio.sleep(0.1)
                
                current_load = 1500  # Simulate high load
                
                context.set_output("current_load", current_load)
                context.set_output("threshold", threshold)
                context.set_output("scaling_needed", current_load > threshold)
                
                return "scale_resources"
            
            @state(cpu=0.5, memory=128.0)
            async def scale_resources(self, context: Context):
                scaling_needed = context.get_output("scaling_needed", False)
                
                await asyncio.sleep(0.1)
                
                if scaling_needed:
                    # Simulate scaling up
                    context.set_output("scaling_action", "scale_up")
                    context.set_output("additional_instances", 2)
                else:
                    context.set_output("scaling_action", "maintain")
                    context.set_output("additional_instances", 0)
                
                context.set_output("scaling_completed", True)
                return None
        
        # Create dynamic scaling workflow
        load_balancer = LoadBalancingAgent("load-balancer", load_threshold=1000)
        
        # Base processing agents
        base_agents = [
            DataIngestionAgent("base-ingester", {
                "url": "https://api.example.com/data",
                "format": "json",
                "expected_records": 1500
            }),
            DataTransformationAgent("base-transformer", {
                "type": "filter",
                "multiplier": 0.8
            })
        ]
        
        # Execute load balancing
        load_balancer_result = await load_balancer.run()
        
        # Check if scaling is needed
        scaling_needed = load_balancer_result.get_output("scaling_needed", False)
        additional_instances = load_balancer_result.get_output("additional_instances", 0)
        
        assert scaling_needed is True
        assert additional_instances > 0
        
        # Create additional agents based on scaling decision
        scaled_agents = []
        if scaling_needed:
            for i in range(additional_instances):
                scaled_agents.append(
                    DataTransformationAgent(f"scaled-transformer-{i}", {
                        "type": "filter",
                        "multiplier": 0.8
                    })
                )
        
        # Execute scaled workflow
        all_agents = base_agents + scaled_agents
        
        start_time = time.time()
        results = await run_agents_parallel(all_agents)
        execution_time = time.time() - start_time
        
        # Verify scaling effectiveness
        assert len(results) == len(all_agents)
        assert len(results) > len(base_agents)  # Should have scaled up
        
        # All agents should complete successfully
        for result in results.values():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
        
        # Scaled execution should handle higher load
        total_processing_capacity = len(all_agents)
        assert total_processing_capacity >= 3  # Base 2 + at least 1 scaled
        
        # Verify load balancer decision was correct
        assert load_balancer_result.get_output("scaling_completed") is True
        assert load_balancer_result.get_output("scaling_action") == "scale_up"


@pytest.mark.e2e
@pytest.mark.asyncio
class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    async def test_batch_processing_scenario(self):
        """Test a typical batch processing scenario."""
        
        class BatchProcessorAgent(Agent):
            def __init__(self, name: str, batch_config: dict):
                super().__init__(name)
                self.set_variable("batch_config", batch_config)
                self.add_state('initialize_batch', self.initialize_batch)
                self.add_state('process_batch', self.process_batch)
                self.add_state('finalize_batch', self.finalize_batch)
            
            @state(cpu=0.5, memory=256.0)
            async def initialize_batch(self, context: Context):
                config = self.get_variable("batch_config", {})
                batch_size = config.get("batch_size", 100)
                
                await asyncio.sleep(0.1)
                
                context.set_output("batch_initialized", True)
                context.set_output("batch_size", batch_size)
                context.set_metadata("batch_start", time.time())
                
                return "process_batch"
            
            @cpu_intensive(cpu=4.0, memory=1024.0)
            async def process_batch(self, context: Context):
                batch_size = context.get_output("batch_size", 100)
                
                # Simulate intensive batch processing
                await asyncio.sleep(0.5)
                
                processed_items = batch_size
                success_rate = 0.95
                successful_items = int(processed_items * success_rate)
                
                context.set_output("processed_items", processed_items)
                context.set_output("successful_items", successful_items)
                context.set_output("success_rate", success_rate)
                context.set_metric("batch_throughput", processed_items / 0.5)
                
                return "finalize_batch"
            
            @state(cpu=0.3, memory=128.0)
            async def finalize_batch(self, context: Context):
                await asyncio.sleep(0.1)
                
                batch_start = context.get_metadata("batch_start", time.time())
                batch_duration = time.time() - batch_start
                
                context.set_output("batch_completed", True)
                context.set_output("batch_duration", batch_duration)
                context.set_metric("batch_efficiency", 0.95)
                
                return None
        
        # Create batch processing workflow
        batch_configs = [
            {"batch_size": 200, "priority": "high"},
            {"batch_size": 150, "priority": "medium"},
            {"batch_size": 100, "priority": "low"}
        ]
        
        batch_processors = [
            BatchProcessorAgent(f"batch-processor-{i}", config)
            for i, config in enumerate(batch_configs)
        ]
        
        # Execute batch processing
        start_time = time.time()
        batch_results = await run_agents_parallel(batch_processors)
        total_time = time.time() - start_time
        
        # Verify batch processing
        assert len(batch_results) == 3
        
        total_processed = 0
        total_successful = 0
        
        for result in batch_results.values():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            assert result.get_output("batch_completed") is True
            
            processed = result.get_output("processed_items", 0)
            successful = result.get_output("successful_items", 0)
            
            total_processed += processed
            total_successful += successful
        
        # Verify batch processing metrics
        assert total_processed == 450  # 200 + 150 + 100
        assert total_successful >= 400  # At least 90% success rate
        
        overall_success_rate = total_successful / total_processed
        assert overall_success_rate >= 0.9
        
        # Parallel processing should be efficient
        assert total_time <= 2.0  # Should complete within 2 seconds
    
    async def test_microservices_orchestration(self):
        """Test orchestrating multiple microservices."""
        
        class MicroserviceAgent(Agent):
            def __init__(self, name: str, service_config: dict):
                super().__init__(name)
                self.set_