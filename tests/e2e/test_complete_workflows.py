"""End-to-end tests for complete PuffinFlow workflows.

These tests simulate real-world scenarios and validate the entire system
from user perspective, including all components working together.
"""

import asyncio
import time

import pytest

from puffinflow import (
    Agent,
    Context,
    cpu_intensive,
    create_team,
    io_intensive,
    run_agents_parallel,
    run_agents_sequential,
    state,
)


class DataIngestionAgent(Agent):
    """Agent that simulates realistic data ingestion from external sources."""

    def __init__(self, name: str, source_config: dict):
        super().__init__(name)
        self.source_config = source_config
        self.add_state("validate_source", self.validate_source)
        self.add_state("ingest_data", self.ingest_data)

    @state(cpu=0.5, memory=128.0)
    async def validate_source(self, context: Context):
        """Validate data source configuration."""
        # Simulate validation time
        await asyncio.sleep(0.1)

        # Basic validation - check required fields
        if not self.source_config.get("url") or not self.source_config.get("format"):
            return None  # End execution on validation failure

        return "ingest_data"

    @io_intensive(cpu=1.0, memory=256.0, io_weight=2.0)
    async def ingest_data(self, context: Context):
        """Ingest data from the source."""
        # Simulate data ingestion time
        await asyncio.sleep(0.3)

        # Generate mock data based on configuration
        data_size = self.source_config.get("expected_records", 1000)
        ingested_data = {
            "records": [{"id": i, "value": f"data_{i}"} for i in range(data_size)],
            "metadata": {
                "source": self.source_config.get("url", "unknown"),
                "format": self.source_config.get("format", "json"),
                "ingested_at": time.time(),
                "record_count": data_size,
            },
        }

        context.set_variable("ingested_data", ingested_data)
        context.set_variable("record_count", data_size)

        return None


class DataTransformationAgent(Agent):
    """Agent that transforms ingested data."""

    def __init__(self, name: str, transformation_rules: dict):
        super().__init__(name)
        self.transformation_rules = transformation_rules
        self.add_state("transform_data", self.transform_data)

    @cpu_intensive(cpu=3.0, memory=512.0)
    async def transform_data(self, context: Context):
        """Transform the input data according to rules."""
        # Generate mock input data for transformation
        input_data = {
            "records": [{"id": i, "value": f"data_{i}"} for i in range(1000)],
            "metadata": {"source": "mock", "format": "json", "record_count": 1000},
        }

        # Simulate transformation work
        await asyncio.sleep(0.5)

        # Apply transformation rules
        transformed_records = []
        multiplier = self.transformation_rules.get("multiplier", 1.0)

        for record in input_data.get("records", []):
            if self.transformation_rules.get("type") == "filter":
                # Apply filtering logic
                if len(transformed_records) < len(input_data["records"]) * multiplier:
                    transformed_records.append(
                        {
                            "id": record["id"],
                            "value": record["value"],
                            "transformed": True,
                        }
                    )
            else:
                # Default transformation
                transformed_records.append(
                    {
                        "id": record["id"],
                        "value": f"transformed_{record['value']}",
                        "transformed": True,
                    }
                )

        transformed_data = {
            "records": transformed_records,
            "metadata": {
                "transformation_type": self.transformation_rules.get("type", "default"),
                "input_count": len(input_data.get("records", [])),
                "output_count": len(transformed_records),
                "transformed_at": time.time(),
            },
        }

        context.set_variable("transformed_data", transformed_data)
        context.set_variable("output_count", len(transformed_records))

        return None


class DataStorageAgent(Agent):
    """Agent that stores processed data."""

    def __init__(self, name: str, storage_config: dict):
        super().__init__(name)
        self.storage_config = storage_config
        self.add_state("store_data", self.store_data)

    @io_intensive(cpu=0.5, memory=256.0, io_weight=1.5)
    async def store_data(self, context: Context):
        """Store transformed data."""
        # Generate mock transformed data for storage
        transformed_data = {
            "records": [
                {"id": i, "value": f"data_{i}", "transformed": True} for i in range(800)
            ],
            "metadata": {
                "transformation_type": "filter",
                "input_count": 1000,
                "output_count": 800,
            },
        }

        # Simulate storage time
        await asyncio.sleep(0.2)

        # Store data (simulate batching)
        batch_size = self.storage_config.get("batch_size", 100)
        records = transformed_data.get("records", [])
        total_stored = 0

        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            # Simulate batch storage time
            await asyncio.sleep(0.01)
            total_stored += len(batch)

        context.set_variable("stored_count", total_stored)
        context.set_variable("storage_complete", True)

        return None


class MonitoringAgent(Agent):
    """Agent that monitors workflow execution."""

    def __init__(self, name: str):
        super().__init__(name)
        self.add_state("generate_report", self.generate_report)

    @state(cpu=0.3, memory=64.0)
    async def generate_report(self, context: Context):
        """Generate monitoring report."""
        # Simulate monitoring work
        await asyncio.sleep(0.1)

        # Generate a monitoring report
        report = {
            "report_id": f"monitor_{int(time.time())}",
            "timestamp": time.time(),
            "summary": {
                "workflow_status": "completed",
                "total_agents": 4,
                "execution_time": 1.2,
            },
            "recommendations": [
                "Consider increasing batch size for better performance",
                "Monitor memory usage during peak hours",
            ],
        }

        context.set_variable("monitoring_report", report)
        context.set_variable("report_generated", True)

        return None


class FileProcessingAgent(Agent):
    """Agent that processes files from a directory."""

    def __init__(self, name: str, directory_path: str):
        super().__init__(name)
        self.directory_path = directory_path
        self.add_state("process_files", self.process_files)

    @cpu_intensive(cpu=2.0, memory=256.0)
    async def process_files(self, context: Context):
        """Process files from directory."""
        # Simulate scanning and finding files
        await asyncio.sleep(0.1)
        files = [f"file_{i}.txt" for i in range(10)]

        # Simulate file processing
        processed_files = []
        for file_name in files:
            await asyncio.sleep(0.05)  # Simulate processing time
            processed_files.append(
                {
                    "name": file_name,
                    "size": 1024,  # Simulate file size
                    "processed_at": time.time(),
                }
            )

        # Set all variables
        context.set_variable("files_found", files)
        context.set_typed_variable("file_count", len(files))
        context.set_variable("processed_files", processed_files)
        context.set_typed_variable("processed_count", len(processed_files))

        return None


class APIServiceAgent(Agent):
    """Agent that simulates calling external APIs."""

    def __init__(self, name: str, api_config: dict):
        super().__init__(name)
        self.api_config = api_config
        self.add_state("call_api", self.call_api)

    @io_intensive(cpu=0.5, memory=128.0, io_weight=3.0)
    async def call_api(self, context: Context):
        """Call external API and process response."""
        # Simulate API call time
        await asyncio.sleep(0.4)

        # Generate mock API response
        response = {
            "status": "success",
            "data": {
                "items": [{"id": i, "name": f"item_{i}"} for i in range(50)],
                "total": 50,
                "timestamp": time.time(),
            },
            "metadata": {"api_version": "v1", "response_time": 0.4},
        }

        context.set_variable("api_response", response)
        context.set_typed_variable("items_retrieved", len(response["data"]["items"]))

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
            "expected_records": 1000,
        }

        transformation_config = {"type": "filter", "multiplier": 0.8}

        storage_config = {"type": "database", "batch_size": 100}

        # Create agents
        ingestion_agent = DataIngestionAgent("data-ingester", ingestion_config)
        transformation_agent = DataTransformationAgent(
            "data-transformer", transformation_config
        )
        storage_agent = DataStorageAgent("data-storage", storage_config)
        monitoring_agent = MonitoringAgent("workflow-monitor")

        # Execute the pipeline sequentially
        pipeline_agents = [ingestion_agent, transformation_agent, storage_agent]

        start_time = time.time()
        pipeline_results = await run_agents_sequential(pipeline_agents)
        time.time() - start_time

        # Run monitoring in parallel
        monitoring_result = await monitoring_agent.run()

        total_time = time.time() - start_time

        # Verify pipeline execution
        assert len(pipeline_results) == 3

        # Verify ingestion
        ingestion_result = pipeline_results["data-ingester"]
        assert ingestion_result.status.name in ["COMPLETED", "SUCCESS"]
        assert ingestion_result.get_variable("record_count") == 1000

        # Verify transformation
        transformation_result = pipeline_results["data-transformer"]
        assert transformation_result.status.name in ["COMPLETED", "SUCCESS"]
        assert (
            transformation_result.get_variable("output_count") == 800
        )  # 80% after filtering

        # Verify storage
        storage_result = pipeline_results["data-storage"]
        assert storage_result.status.name in ["COMPLETED", "SUCCESS"]
        assert (
            storage_result.get_variable("stored_count") == 800
        )  # All transformed data stored

        # Verify monitoring
        assert monitoring_result.status.name in ["COMPLETED", "SUCCESS"]
        assert monitoring_result.get_variable("report_generated") is True

        # Verify timing is reasonable
        assert total_time >= 0.9  # Should take at least 0.9 seconds for all operations
        assert total_time <= 5.0  # Should complete within reasonable time

    async def test_parallel_processing_workflow(self):
        """Test parallel processing of multiple data streams."""
        # Create multiple file processing agents
        agents = []
        for i in range(3):
            agent = FileProcessingAgent(f"file-processor-{i}", f"/tmp/data_{i}")
            agents.append(agent)

        # Run agents in parallel
        start_time = time.time()
        results = await run_agents_parallel(agents)
        execution_time = time.time() - start_time

        # Verify all agents completed successfully
        assert len(results) == 3

        total_processed = 0
        for _agent_name, result in results.items():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            assert result.get_variable("processed_count") == 10
            total_processed += result.get_variable("processed_count", 0)

        # Verify parallel processing was efficient
        assert total_processed == 30  # 3 agents x 10 files each
        assert execution_time < 2.0  # Should be faster than sequential

    async def test_team_coordination_workflow(self):
        """Test agent team coordination for complex workflows."""
        # Create API service agents
        api_agents = []
        for i in range(2):
            config = {"endpoint": f"https://api.service{i}.com", "timeout": 30}
            agent = APIServiceAgent(f"api-service-{i}", config)
            api_agents.append(agent)

        # Create a team
        team = create_team("api-team", api_agents)

        # Execute team workflow
        start_time = time.time()
        team_result = await team.run_parallel()
        execution_time = time.time() - start_time

        # Verify team coordination
        assert len(team_result.agent_results) == 2

        total_items = 0
        for _agent_name, result in team_result.agent_results.items():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            items = result.get_variable("items_retrieved", 0)
            total_items += items

        assert total_items == 100  # 2 agents x 50 items each
        assert execution_time < 3.0


@pytest.mark.e2e
@pytest.mark.asyncio
class TestRealWorldScenarios:
    """Test real-world scenarios that users might encounter."""

    async def test_batch_processing_scenario(self):
        """Test batch processing scenario similar to real ETL workflows."""
        # Create a realistic batch processing pipeline
        ingestion_agent = DataIngestionAgent(
            "batch-ingester",
            {
                "url": "https://data.warehouse.com/export",
                "format": "csv",
                "expected_records": 5000,
            },
        )

        transformation_agent = DataTransformationAgent(
            "batch-transformer",
            {
                "type": "filter",
                "multiplier": 0.9,  # 90% pass rate
            },
        )

        storage_agent = DataStorageAgent(
            "batch-storage", {"type": "warehouse", "batch_size": 500}
        )

        # Run the batch processing pipeline
        pipeline = [ingestion_agent, transformation_agent, storage_agent]
        results = await run_agents_sequential(pipeline)

        # Verify the batch process completed successfully
        assert len(results) == 3

        # Check that each agent completed successfully
        ingested = results["batch-ingester"].get_variable("record_count", 0)
        transformed = results["batch-transformer"].get_variable("output_count", 0)
        stored = results["batch-storage"].get_variable("stored_count", 0)

        # Each agent should have processed its own simulated data
        assert ingested == 5000  # Ingestion agent generated 5000 records
        assert (
            transformed == 900
        )  # Transformation agent filtered to 900 records (90% of 1000)
        assert stored == 800  # Storage agent stored its mock data (800 records)

    async def test_microservices_orchestration(self):
        """Test microservices orchestration scenario."""
        # Create microservice agents
        auth_service = APIServiceAgent(
            "auth-service",
            {"endpoint": "https://auth.service.com/validate", "timeout": 10},
        )

        data_service = APIServiceAgent(
            "data-service",
            {"endpoint": "https://data.service.com/fetch", "timeout": 30},
        )

        notification_service = APIServiceAgent(
            "notification-service",
            {"endpoint": "https://notify.service.com/send", "timeout": 15},
        )

        # Run services in a coordinated manner
        # First auth, then data and notification in parallel
        auth_result = await auth_service.run()
        assert auth_result.status.name in ["COMPLETED", "SUCCESS"]

        # Run data and notification services in parallel
        parallel_services = [data_service, notification_service]
        service_results = await run_agents_parallel(parallel_services)

        # Verify all services completed successfully
        assert len(service_results) == 2
        for _service_name, result in service_results.items():
            assert result.status.name in ["COMPLETED", "SUCCESS"]
            assert result.get_variable("items_retrieved", 0) > 0
