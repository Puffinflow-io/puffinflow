#!/usr/bin/env python3
"""
Tests for reliability.ts documentation examples.
"""

import asyncio
import sys
import time

import pytest

# Add the src directory to Python path
sys.path.insert(0, "src")

from puffinflow import Agent, ExecutionMode, Priority, state


class TestReliabilityExamples:
    """Test examples from reliability.ts documentation."""

    @pytest.mark.asyncio
    async def test_health_monitoring_workflow(self):
        """Test health monitoring workflow example."""
        health_monitor = Agent("health-monitor")

        @state(priority=Priority.HIGH, timeout=10.0, max_retries=1)
        async def database_health_check(context):
            """Monitor database connectivity and performance"""
            print("🔍 Database health check...")

            start_time = time.time()

            # Simulate database connection test
            await asyncio.sleep(0.01)  # Reduced for testing

            # Test basic query performance
            query_start = time.time()
            await asyncio.sleep(0.001)  # Reduced for testing
            query_time = time.time() - query_start

            total_time = time.time() - start_time

            health_status = {
                "status": "healthy",
                "connection_time": total_time,
                "query_time": query_time,
                "timestamp": time.time(),
            }

            # Set thresholds for performance alerts
            if query_time > 0.5:
                health_status["status"] = "degraded"
                health_status["warning"] = "Slow query performance"

            if total_time > 2.0:
                health_status["status"] = "unhealthy"
                health_status["error"] = "Connection timeout"

            context.set_variable("database_health", health_status)
            print(f"✅ Database: {health_status['status']} ({total_time:.3f}s)")
            return None

        @state(priority=Priority.HIGH, timeout=5.0, rate_limit=2.0)
        async def external_api_health_check(context):
            """Monitor external API dependencies"""
            print("🌐 External API health check...")

            apis_to_check = [
                {"name": "OpenAI API", "endpoint": "api.openai.com"},
                {"name": "Vector Store", "endpoint": "vectorstore.service"},
            ]

            api_statuses = {}

            for api in apis_to_check:
                start_time = time.time()

                # Simulate API health check
                await asyncio.sleep(0.001)  # Reduced for testing

                response_time = time.time() - start_time

                status = {
                    "status": "healthy",
                    "response_time": response_time,
                    "endpoint": api["endpoint"],
                    "timestamp": time.time(),
                }

                # Performance thresholds
                if response_time > 1.0:
                    status["status"] = "degraded"
                    status["warning"] = "High latency"

                api_statuses[api["name"]] = status
                print(f"✅ {api['name']}: {status['status']} ({response_time:.3f}s)")

            context.set_variable("api_health", api_statuses)
            return None

        @state(priority=Priority.NORMAL, timeout=15.0)
        async def system_resource_check(context):
            """Monitor system resource utilization"""
            print("📊 System resource check...")

            # Simulate resource monitoring
            resources = {
                "cpu_usage": 45.2,
                "memory_usage": 62.8,
                "disk_usage": 78.5,
                "network_latency": 12.3,
                "active_connections": 150,
            }

            resource_health = {}
            overall_status = "healthy"

            # Define thresholds
            thresholds = {
                "cpu_usage": {"warning": 70, "critical": 90},
                "memory_usage": {"warning": 80, "critical": 95},
                "disk_usage": {"warning": 85, "critical": 95},
                "network_latency": {"warning": 100, "critical": 500},
                "active_connections": {"warning": 1000, "critical": 1500},
            }

            for metric, value in resources.items():
                if metric in thresholds:
                    threshold = thresholds[metric]

                    if value >= threshold["critical"]:
                        status = "critical"
                        overall_status = "critical"
                    elif value >= threshold["warning"]:
                        status = "warning"
                        if overall_status == "healthy":
                            overall_status = "warning"
                    else:
                        status = "healthy"

                    resource_health[metric] = {
                        "value": value,
                        "status": status,
                        "threshold_warning": threshold["warning"],
                        "threshold_critical": threshold["critical"],
                    }

            resource_health["overall_status"] = overall_status
            resource_health["timestamp"] = time.time()

            context.set_variable("resource_health", resource_health)
            print(f"📈 Overall resource status: {overall_status}")
            return None

        @state
        async def aggregate_health_status(context):
            """Aggregate all health checks into overall system status"""
            print("🎯 Aggregating system health...")

            db_health = context.get_variable("database_health", {})
            api_health = context.get_variable("api_health", {})
            resource_health = context.get_variable("resource_health", {})

            # Calculate overall system health
            critical_issues = []
            warnings = []
            healthy_components = []

            # Check database
            if db_health.get("status") == "unhealthy":
                critical_issues.append("Database unavailable")
            elif db_health.get("status") == "degraded":
                warnings.append("Database performance degraded")
            else:
                healthy_components.append("Database")

            # Check APIs
            unhealthy_apis = [
                name
                for name, status in api_health.items()
                if status.get("status") == "unhealthy"
            ]
            degraded_apis = [
                name
                for name, status in api_health.items()
                if status.get("status") == "degraded"
            ]

            if unhealthy_apis:
                critical_issues.extend(
                    [f"{api} API unavailable" for api in unhealthy_apis]
                )
            if degraded_apis:
                warnings.extend([f"{api} API degraded" for api in degraded_apis])

            healthy_apis = [
                name
                for name, status in api_health.items()
                if status.get("status") == "healthy"
            ]
            healthy_components.extend([f"{api} API" for api in healthy_apis])

            # Check resources
            resource_status = resource_health.get("overall_status", "unknown")
            if resource_status == "critical":
                critical_issues.append("Resource exhaustion")
            elif resource_status == "warning":
                warnings.append("Resource utilization high")
            else:
                healthy_components.append("System resources")

            # Determine overall status
            if critical_issues:
                overall_status = "critical"
            elif warnings:
                overall_status = "degraded"
            else:
                overall_status = "healthy"

            health_summary = {
                "overall_status": overall_status,
                "critical_issues": critical_issues,
                "warnings": warnings,
                "healthy_components": healthy_components,
                "timestamp": time.time(),
            }

            context.set_output("system_health", health_summary)
            print(f"📊 Overall System Status: {overall_status.upper()}")
            return None

        # Add states to monitor
        health_monitor.add_state("database_health_check", database_health_check)
        health_monitor.add_state("external_api_health_check", external_api_health_check)
        health_monitor.add_state("system_resource_check", system_resource_check)
        health_monitor.add_state(
            "aggregate_health_status",
            aggregate_health_status,
            dependencies=[
                "database_health_check",
                "external_api_health_check",
                "system_resource_check",
            ],
        )

        # Run health monitoring workflow
        result = await health_monitor.run(execution_mode=ExecutionMode.PARALLEL)

        # Verify health checks completed
        assert result.get_variable("database_health")["status"] == "healthy"
        assert len(result.get_variable("api_health")) == 2
        assert result.get_variable("resource_health")["overall_status"] == "healthy"

        health_summary = result.get_output("system_health")
        assert health_summary["overall_status"] == "healthy"
        assert len(health_summary["healthy_components"]) >= 3

    @pytest.mark.asyncio
    async def test_service_degradation_levels(self):
        """Test service degradation level determination."""
        from enum import Enum

        class ServiceLevel(Enum):
            FULL = "full"
            REDUCED = "reduced"
            MINIMAL = "minimal"
            EMERGENCY = "emergency"

        degradation_agent = Agent("degradation-manager")

        @state(timeout=10.0, max_retries=2)
        async def determine_service_level(context):
            """Determine current service level based on system health"""
            print("🎚️ Determining service level...")

            system_health = context.get_variable("system_health", {})
            overall_status = system_health.get("overall_status", "healthy")

            critical_issues = system_health.get("critical_issues", [])
            warnings = system_health.get("warnings", [])

            # Service level decision logic
            if overall_status == "healthy":
                service_level = ServiceLevel.FULL
                print("✅ Service Level: FULL - All features available")

            elif overall_status == "degraded" and len(warnings) <= 2:
                service_level = ServiceLevel.REDUCED
                print("⚠️ Service Level: REDUCED - Some features may be slower")

            elif overall_status == "degraded" or (
                critical_issues and len(critical_issues) <= 1
            ):
                service_level = ServiceLevel.MINIMAL
                print("🟡 Service Level: MINIMAL - Core features only")

            else:
                service_level = ServiceLevel.EMERGENCY
                print("🔴 Service Level: EMERGENCY - Emergency operations only")

            # Store service level configuration
            service_config = {"level": service_level.value, "timestamp": time.time()}

            context.set_variable("service_level", service_config)
            context.set_output("current_service_level", service_level.value)
            return None

        degradation_agent.add_state("determine_service_level", determine_service_level)

        # Test with healthy system
        result = await degradation_agent.run(
            initial_context={
                "system_health": {
                    "overall_status": "healthy",
                    "critical_issues": [],
                    "warnings": [],
                }
            }
        )

        assert result.get_output("current_service_level") == "full"

        # Test with degraded system using a new agent instance
        degraded_agent = Agent("degradation-manager-degraded")
        degraded_agent.add_state("determine_service_level", determine_service_level)

        result = await degraded_agent.run(
            initial_context={
                "system_health": {
                    "overall_status": "degraded",
                    "critical_issues": [],
                    "warnings": ["High latency", "Memory usage high"],
                }
            }
        )

        assert result.get_output("current_service_level") == "reduced"

    @pytest.mark.asyncio
    async def test_data_consistency_validation(self):
        """Test data consistency and integrity validation."""
        consistency_agent = Agent("data-consistency-manager")

        @state(priority=Priority.HIGH, timeout=30.0)
        async def create_data_snapshot(context):
            """Create consistent snapshot of critical data"""
            print("📸 Creating data snapshot...")

            # Gather critical data
            critical_data = {
                "user_sessions": {
                    "session_1": {"user_id": "user_123", "created_at": time.time()}
                },
                "processing_queue": [
                    {"id": "item_1", "type": "process", "status": "pending"}
                ],
                "cache_state": {
                    "key_1": {"data": "value_1", "expires_at": time.time() + 3600}
                },
                "configuration": {
                    "service_level": "full",
                    "feature_flags": {"feature_a": True},
                },
            }

            # Create versioned snapshot
            current_version = context.get_variable("data_version", 0) + 1

            # Calculate checksum for integrity
            import json

            data_json = json.dumps(critical_data, sort_keys=True)
            checksum = str(hash(data_json))

            snapshot = {
                "version": current_version,
                "data": critical_data,
                "timestamp": time.time(),
                "checksum": checksum,
            }

            # Store snapshot
            context.set_variable("data_snapshot", snapshot)
            context.set_variable("data_version", current_version)

            print(f"✅ Snapshot v{current_version} created (checksum: {checksum[:8]})")
            return None

        @state(max_retries=3, timeout=15.0)
        async def validate_data_integrity(context):
            """Validate data integrity across system components"""
            print("🔍 Validating data integrity...")

            snapshot = context.get_variable("data_snapshot", {})
            if not snapshot:
                print("⚠️ No snapshot available for validation")
                return

            validation_results = {}

            # Validate each data component
            for component, data in snapshot["data"].items():
                try:
                    if component == "user_sessions":
                        # Validate session data structure
                        valid = isinstance(data, dict) and all(
                            isinstance(session, dict) and "user_id" in session
                            for session in data.values()
                        )
                        validation_results[component] = {
                            "valid": valid,
                            "count": len(data),
                        }

                    elif component == "processing_queue":
                        # Validate queue data
                        valid = isinstance(data, list) and all(
                            isinstance(item, dict) and "id" in item and "status" in item
                            for item in data
                        )
                        validation_results[component] = {
                            "valid": valid,
                            "count": len(data),
                        }

                    elif component == "cache_state":
                        # Validate cache consistency
                        valid = isinstance(data, dict) and all(
                            isinstance(value, dict)
                            and "data" in value
                            and "expires_at" in value
                            for value in data.values()
                        )
                        validation_results[component] = {
                            "valid": valid,
                            "count": len(data),
                        }

                    elif component == "configuration":
                        # Validate configuration data
                        valid = (
                            isinstance(data, dict)
                            and "service_level" in data
                            and "feature_flags" in data
                        )
                        validation_results[component] = {
                            "valid": valid,
                            "count": len(data) if isinstance(data, dict) else 0,
                        }

                    print(
                        f"✅ {component}: {'Valid' if validation_results[component]['valid'] else 'Invalid'}"
                    )

                except Exception as e:
                    validation_results[component] = {"valid": False, "error": str(e)}
                    print(f"❌ {component}: Validation error - {e}")

            # Calculate overall integrity score
            valid_components = sum(
                1
                for result in validation_results.values()
                if result.get("valid", False)
            )
            total_components = len(validation_results)
            integrity_score = (
                (valid_components / total_components) * 100
                if total_components > 0
                else 0
            )

            integrity_report = {
                "overall_score": integrity_score,
                "validation_results": validation_results,
                "timestamp": time.time(),
                "snapshot_version": snapshot.get("version", 0),
            }

            context.set_variable("integrity_report", integrity_report)
            context.set_output("data_integrity_score", integrity_score)

            print(f"📊 Overall integrity score: {integrity_score:.1f}%")
            return None

        # Add states to consistency agent
        consistency_agent.add_state("create_data_snapshot", create_data_snapshot)
        consistency_agent.add_state("validate_data_integrity", validate_data_integrity)

        # Run the data consistency workflow
        result = await consistency_agent.run()

        # Verify snapshot was created
        snapshot = result.get_variable("data_snapshot")
        assert snapshot["version"] == 1
        assert len(snapshot["data"]) == 4
        assert "checksum" in snapshot

        # Create a new agent instance for validation test
        validation_agent = Agent("data-consistency-validator")
        validation_agent.add_state("validate_data_integrity", validate_data_integrity)

        result = await validation_agent.run(initial_context={"data_snapshot": snapshot})

        integrity_score = result.get_output("data_integrity_score")
        assert integrity_score == 100.0  # All components should validate successfully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
