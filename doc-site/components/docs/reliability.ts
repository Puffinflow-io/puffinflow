export const reliabilityMarkdown = `# Reliability & Production Patterns

Puffinflow provides comprehensive reliability patterns to ensure your workflows operate consistently in production environments. This guide covers health monitoring, graceful degradation, system resilience, and operational best practices for building bulletproof AI workflows.

## Reliability Philosophy

**Reliability isn't just error handling** - it's designing systems that:
- Continue operating when components fail
- Provide consistent service quality
- Monitor their own health
- Gracefully handle resource exhaustion
- Maintain data consistency
- Recover automatically when possible

## System Health Monitoring

### Health Check Patterns

Implement comprehensive health checks to monitor system components:

\`\`\`python
import asyncio
import time
from puffinflow import Agent
from puffinflow import state
from puffinflow.core.agent.state import Priority

health_monitor = Agent("health-monitor")

@state(
    priority=Priority.HIGH,
    timeout=10.0,
    max_retries=1
)
async def database_health_check(context):
    """Monitor database connectivity and performance"""
    print("🔍 Database health check...")

    start_time = time.time()

    try:
        # Simulate database connection test
        await asyncio.sleep(0.5)  # Connection time

        # Test basic query performance
        query_start = time.time()
        await asyncio.sleep(0.1)  # Query execution time
        query_time = time.time() - query_start

        total_time = time.time() - start_time

        health_status = {
            "status": "healthy",
            "connection_time": total_time,
            "query_time": query_time,
            "timestamp": time.time()
        }

        # Set thresholds for performance alerts
        if query_time > 0.5:
            health_status["status"] = "degraded"
            health_status["warning"] = "Slow query performance"

        if total_time > 2.0:
            health_status["status"] = "unhealthy"
            health_status["error"] = "Connection timeout"

        context.set_variable("database_health", health_status)
        print(f"✅ Database: {health_status['status']} ({total_time:.2f}s)")

    except Exception as e:
        context.set_variable("database_health", {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        })
        print(f"❌ Database: unhealthy - {e}")

@state(
    priority=Priority.HIGH,
    timeout=5.0,
    rate_limit=2.0
)
async def external_api_health_check(context):
    """Monitor external API dependencies"""
    print("🌐 External API health check...")

    apis_to_check = [
        {"name": "OpenAI API", "endpoint": "api.openai.com"},
        {"name": "Vector Store", "endpoint": "vectorstore.service"},
        {"name": "Auth Service", "endpoint": "auth.service"}
    ]

    api_statuses = {}

    for api in apis_to_check:
        try:
            start_time = time.time()

            # Simulate API health check
            await asyncio.sleep(0.2)  # Network request

            response_time = time.time() - start_time

            status = {
                "status": "healthy",
                "response_time": response_time,
                "endpoint": api["endpoint"],
                "timestamp": time.time()
            }

            # Performance thresholds
            if response_time > 1.0:
                status["status"] = "degraded"
                status["warning"] = "High latency"

            api_statuses[api["name"]] = status
            print(f"✅ {api['name']}: {status['status']} ({response_time:.2f}s)")

        except Exception as e:
            api_statuses[api["name"]] = {
                "status": "unhealthy",
                "error": str(e),
                "endpoint": api["endpoint"],
                "timestamp": time.time()
            }
            print(f"❌ {api['name']}: unhealthy - {e}")

    context.set_variable("api_health", api_statuses)

@state(
    priority=Priority.NORMAL,
    timeout=15.0
)
async def system_resource_check(context):
    """Monitor system resource utilization"""
    print("📊 System resource check...")

    # Simulate resource monitoring
    resources = {
        "cpu_usage": 45.2,      # CPU percentage
        "memory_usage": 62.8,   # Memory percentage
        "disk_usage": 78.5,     # Disk percentage
        "network_latency": 12.3, # Network latency in ms
        "active_connections": 150
    }

    resource_health = {}
    overall_status = "healthy"

    # Define thresholds
    thresholds = {
        "cpu_usage": {"warning": 70, "critical": 90},
        "memory_usage": {"warning": 80, "critical": 95},
        "disk_usage": {"warning": 85, "critical": 95},
        "network_latency": {"warning": 100, "critical": 500},
        "active_connections": {"warning": 1000, "critical": 1500}
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
                "threshold_critical": threshold["critical"]
            }

            status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🔴"}[status]
            print(f"{status_emoji} {metric}: {value}")

    resource_health["overall_status"] = overall_status
    resource_health["timestamp"] = time.time()

    context.set_variable("resource_health", resource_health)
    print(f"📈 Overall resource status: {overall_status}")

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
    unhealthy_apis = [name for name, status in api_health.items()
                     if status.get("status") == "unhealthy"]
    degraded_apis = [name for name, status in api_health.items()
                    if status.get("status") == "degraded"]

    if unhealthy_apis:
        critical_issues.extend([f"{api} API unavailable" for api in unhealthy_apis])
    if degraded_apis:
        warnings.extend([f"{api} API degraded" for api in degraded_apis])

    healthy_apis = [name for name, status in api_health.items()
                   if status.get("status") == "healthy"]
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
        status_emoji = "🔴"
    elif warnings:
        overall_status = "degraded"
        status_emoji = "⚠️"
    else:
        overall_status = "healthy"
        status_emoji = "✅"

    health_summary = {
        "overall_status": overall_status,
        "critical_issues": critical_issues,
        "warnings": warnings,
        "healthy_components": healthy_components,
        "timestamp": time.time(),
        "detailed_health": {
            "database": db_health,
            "apis": api_health,
            "resources": resource_health
        }
    }

    context.set_output("system_health", health_summary)

    print(f"{status_emoji} Overall System Status: {overall_status.upper()}")

    if critical_issues:
        print("🔴 Critical Issues:")
        for issue in critical_issues:
            print(f"   - {issue}")

    if warnings:
        print("⚠️ Warnings:")
        for warning in warnings:
            print(f"   - {warning}")

    if healthy_components:
        print("✅ Healthy Components:")
        for component in healthy_components:
            print(f"   - {component}")

# Build health monitoring workflow
health_monitor.add_state("database_health_check", database_health_check)
health_monitor.add_state("external_api_health_check", external_api_health_check)
health_monitor.add_state("system_resource_check", system_resource_check)
health_monitor.add_state("aggregate_health_status", aggregate_health_status)
\`\`\`

---

## Graceful Degradation Strategies

### Service Degradation Levels

Implement multiple service levels to maintain functionality during partial failures:

\`\`\`python
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
    overall_status = system_health.get("overall_status", "unknown")

    critical_issues = system_health.get("critical_issues", [])
    warnings = system_health.get("warnings", [])

    # Service level decision logic
    if overall_status == "healthy":
        service_level = ServiceLevel.FULL
        print("✅ Service Level: FULL - All features available")

    elif overall_status == "degraded" and len(warnings) <= 2:
        service_level = ServiceLevel.REDUCED
        print("⚠️ Service Level: REDUCED - Some features may be slower")

    elif overall_status == "degraded" or (critical_issues and len(critical_issues) <= 1):
        service_level = ServiceLevel.MINIMAL
        print("🟡 Service Level: MINIMAL - Core features only")

    else:
        service_level = ServiceLevel.EMERGENCY
        print("🔴 Service Level: EMERGENCY - Emergency operations only")

    # Store service level configuration
    service_config = {
        "level": service_level.value,
        "features_enabled": get_enabled_features(service_level),
        "performance_settings": get_performance_settings(service_level),
        "user_message": get_user_message(service_level),
        "timestamp": time.time()
    }

    context.set_variable("service_level", service_config)
    context.set_output("current_service_level", service_level.value)

def get_enabled_features(service_level: ServiceLevel) -> dict:
    """Get enabled features for each service level"""
    features = {
        ServiceLevel.FULL: {
            "ai_chat": True,
            "document_search": True,
            "advanced_analytics": True,
            "real_time_updates": True,
            "file_uploads": True,
            "background_processing": True
        },
        ServiceLevel.REDUCED: {
            "ai_chat": True,
            "document_search": True,
            "advanced_analytics": False,
            "real_time_updates": False,
            "file_uploads": True,
            "background_processing": False
        },
        ServiceLevel.MINIMAL: {
            "ai_chat": True,
            "document_search": False,
            "advanced_analytics": False,
            "real_time_updates": False,
            "file_uploads": False,
            "background_processing": False
        },
        ServiceLevel.EMERGENCY: {
            "ai_chat": False,
            "document_search": False,
            "advanced_analytics": False,
            "real_time_updates": False,
            "file_uploads": False,
            "background_processing": False
        }
    }
    return features[service_level]

def get_performance_settings(service_level: ServiceLevel) -> dict:
    """Get performance settings for each service level"""
    settings = {
        ServiceLevel.FULL: {
            "max_concurrent_requests": 100,
            "response_timeout": 30,
            "cache_ttl": 300,
            "retry_attempts": 3
        },
        ServiceLevel.REDUCED: {
            "max_concurrent_requests": 50,
            "response_timeout": 20,
            "cache_ttl": 600,
            "retry_attempts": 2
        },
        ServiceLevel.MINIMAL: {
            "max_concurrent_requests": 20,
            "response_timeout": 15,
            "cache_ttl": 900,
            "retry_attempts": 1
        },
        ServiceLevel.EMERGENCY: {
            "max_concurrent_requests": 5,
            "response_timeout": 10,
            "cache_ttl": 1800,
            "retry_attempts": 0
        }
    }
    return settings[service_level]

def get_user_message(service_level: ServiceLevel) -> str:
    """Get user-facing message for each service level"""
    messages = {
        ServiceLevel.FULL: "All systems operational",
        ServiceLevel.REDUCED: "Some features may experience slower response times",
        ServiceLevel.MINIMAL: "Operating in limited mode - core features only",
        ServiceLevel.EMERGENCY: "Service temporarily unavailable - emergency maintenance mode"
    }
    return messages[service_level]

@state(timeout=5.0)
async def adaptive_feature_routing(context):
    """Route requests based on current service level"""
    print("🔀 Adaptive feature routing...")

    service_config = context.get_variable("service_level", {})
    enabled_features = service_config.get("features_enabled", {})

    # Simulate incoming feature requests
    requested_features = ["ai_chat", "document_search", "file_uploads", "advanced_analytics"]

    routing_plan = []

    for feature in requested_features:
        if enabled_features.get(feature, False):
            routing_plan.append({
                "feature": feature,
                "status": "enabled",
                "route": f"primary_{feature}_service"
            })
            print(f"✅ {feature}: Enabled -> primary service")
        else:
            # Find fallback option
            fallback = get_feature_fallback(feature, service_config["level"])
            if fallback:
                routing_plan.append({
                    "feature": feature,
                    "status": "fallback",
                    "route": fallback,
                    "message": f"Using {fallback} instead"
                })
                print(f"🔄 {feature}: Disabled -> {fallback}")
            else:
                routing_plan.append({
                    "feature": feature,
                    "status": "unavailable",
                    "route": None,
                    "message": "Feature temporarily unavailable"
                })
                print(f"❌ {feature}: Unavailable")

    context.set_variable("routing_plan", routing_plan)

def get_feature_fallback(feature: str, service_level: str) -> str:
    """Get fallback options for disabled features"""
    fallbacks = {
        "ai_chat": {
            "reduced": None,
            "minimal": None,
            "emergency": None
        },
        "document_search": {
            "reduced": "cached_search_service",
            "minimal": "basic_search_service",
            "emergency": None
        },
        "advanced_analytics": {
            "reduced": "basic_analytics_service",
            "minimal": "static_reports_service",
            "emergency": None
        },
        "file_uploads": {
            "reduced": None,
            "minimal": "email_submission_service",
            "emergency": None
        }
    }

    return fallbacks.get(feature, {}).get(service_level)

# Add states to degradation agent
degradation_agent.add_state("determine_service_level", determine_service_level)
degradation_agent.add_state("adaptive_feature_routing", adaptive_feature_routing)
\`\`\`

---

## Data Consistency & Integrity

### Distributed State Management

Ensure data consistency across distributed workflow components:

\`\`\`python
import json
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DataSnapshot:
    version: int
    data: Dict[str, Any]
    timestamp: float
    checksum: str

consistency_agent = Agent("data-consistency-manager")

@state(priority=Priority.HIGH, timeout=30.0)
async def create_data_snapshot(context):
    """Create consistent snapshot of critical data"""
    print("📸 Creating data snapshot...")

    # Gather critical data
    critical_data = {
        "user_sessions": context.get_variable("active_sessions", {}),
        "processing_queue": context.get_variable("work_queue", []),
        "cache_state": context.get_variable("cache_data", {}),
        "metrics": context.get_variable("system_metrics", {}),
        "configuration": {
            "service_level": context.get_variable("service_level", {}),
            "feature_flags": context.get_variable("feature_flags", {})
        }
    }

    # Create versioned snapshot
    current_version = context.get_variable("data_version", 0) + 1

    # Calculate checksum for integrity
    data_json = json.dumps(critical_data, sort_keys=True)
    checksum = str(hash(data_json))

    snapshot = DataSnapshot(
        version=current_version,
        data=critical_data,
        timestamp=time.time(),
        checksum=checksum
    )

    # Store snapshot
    context.set_variable("data_snapshot", snapshot.__dict__)
    context.set_variable("data_version", current_version)

    # Keep history of recent snapshots
    snapshot_history = context.get_variable("snapshot_history", [])
    snapshot_history.append({
        "version": current_version,
        "timestamp": snapshot.timestamp,
        "checksum": checksum
    })

    # Keep only last 10 snapshots
    if len(snapshot_history) > 10:
        snapshot_history = snapshot_history[-10:]

    context.set_variable("snapshot_history", snapshot_history)

    print(f"✅ Snapshot v{current_version} created (checksum: {checksum[:8]})")

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
                valid = validate_session_data(data)
                validation_results[component] = {
                    "valid": valid,
                    "count": len(data) if isinstance(data, dict) else 0
                }

            elif component == "processing_queue":
                # Validate queue data
                valid = validate_queue_data(data)
                validation_results[component] = {
                    "valid": valid,
                    "count": len(data) if isinstance(data, list) else 0
                }

            elif component == "cache_state":
                # Validate cache consistency
                valid = validate_cache_data(data)
                validation_results[component] = {
                    "valid": valid,
                    "count": len(data) if isinstance(data, dict) else 0
                }

            print(f"✅ {component}: {'Valid' if validation_results[component]['valid'] else 'Invalid'}")

        except Exception as e:
            validation_results[component] = {
                "valid": False,
                "error": str(e)
            }
            print(f"❌ {component}: Validation error - {e}")

    # Calculate overall integrity score
    valid_components = sum(1 for result in validation_results.values() if result.get("valid", False))
    total_components = len(validation_results)
    integrity_score = (valid_components / total_components) * 100 if total_components > 0 else 0

    integrity_report = {
        "overall_score": integrity_score,
        "validation_results": validation_results,
        "timestamp": time.time(),
        "snapshot_version": snapshot.get("version", 0)
    }

    context.set_variable("integrity_report", integrity_report)
    context.set_output("data_integrity_score", integrity_score)

    print(f"📊 Overall integrity score: {integrity_score:.1f}%")

    if integrity_score < 80:
        print("🔴 WARNING: Data integrity below acceptable threshold")
        context.set_variable("integrity_alert", True)

def validate_session_data(sessions: dict) -> bool:
    """Validate user session data structure"""
    if not isinstance(sessions, dict):
        return False

    for session_id, session_data in sessions.items():
        if not isinstance(session_data, dict):
            return False

        required_fields = ["user_id", "created_at", "last_activity"]
        if not all(field in session_data for field in required_fields):
            return False

    return True

def validate_queue_data(queue: list) -> bool:
    """Validate processing queue data structure"""
    if not isinstance(queue, list):
        return False

    for item in queue:
        if not isinstance(item, dict):
            return False

        required_fields = ["id", "type", "created_at", "status"]
        if not all(field in item for field in required_fields):
            return False

    return True

def validate_cache_data(cache: dict) -> bool:
    """Validate cache data structure"""
    if not isinstance(cache, dict):
        return False

    for key, value in cache.items():
        if not isinstance(value, dict):
            return False

        # Check for required cache metadata
        if "data" not in value or "expires_at" not in value:
            return False

    return True

@state(timeout=20.0)
async def recovery_coordination(context):
    """Coordinate recovery actions when integrity issues are detected"""
    print("🔧 Coordinating recovery actions...")

    integrity_alert = context.get_variable("integrity_alert", False)
    integrity_report = context.get_variable("integrity_report", {})

    if not integrity_alert:
        print("✅ No recovery actions needed")
        return

    recovery_actions = []
    validation_results = integrity_report.get("validation_results", {})

    for component, result in validation_results.items():
        if not result.get("valid", True):
            action = get_recovery_action(component, result)
            if action:
                recovery_actions.append(action)

    # Execute recovery actions
    for action in recovery_actions:
        try:
            print(f"🔧 Executing: {action['description']}")
            await execute_recovery_action(action, context)
            print(f"✅ Recovery action completed: {action['description']}")

        except Exception as e:
            print(f"❌ Recovery action failed: {action['description']} - {e}")

    context.set_output("recovery_actions_executed", len(recovery_actions))

def get_recovery_action(component: str, validation_result: dict) -> dict:
    """Determine appropriate recovery action for component"""
    recovery_actions = {
        "user_sessions": {
            "type": "rebuild",
            "description": "Rebuild user session cache from persistent storage"
        },
        "processing_queue": {
            "type": "requeue",
            "description": "Revalidate and requeue processing items"
        },
        "cache_state": {
            "type": "clear",
            "description": "Clear corrupted cache and rebuild"
        }
    }

    return recovery_actions.get(component)

async def execute_recovery_action(action: dict, context):
    """Execute specific recovery action"""
    if action["type"] == "rebuild":
        # Rebuild data from authoritative source
        await asyncio.sleep(1)  # Simulate rebuild time

    elif action["type"] == "requeue":
        # Revalidate and requeue items
        await asyncio.sleep(0.5)  # Simulate requeue time

    elif action["type"] == "clear":
        # Clear and rebuild cache
        await asyncio.sleep(0.3)  # Simulate cache clear time

# Add states to consistency agent
consistency_agent.add_state("create_data_snapshot", create_data_snapshot)
consistency_agent.add_state("validate_data_integrity", validate_data_integrity)
consistency_agent.add_state("recovery_coordination", recovery_coordination)
\`\`\`

---

## Disaster Recovery Patterns

### Backup and Restore Operations

\`\`\`python
import gzip
import base64

disaster_recovery_agent = Agent("disaster-recovery")

@state(priority=Priority.CRITICAL, timeout=120.0)
async def create_disaster_recovery_backup(context):
    """Create comprehensive backup for disaster recovery"""
    print("💾 Creating disaster recovery backup...")

    backup_components = {
        "workflow_state": context.get_all_variables(),
        "configuration": context.get_all_constants(),
        "metadata": {
            "backup_time": time.time(),
            "agent_name": context.agent_name,
            "workflow_id": context.workflow_id,
            "version": "1.0"
        }
    }

    # Compress backup data
    backup_json = json.dumps(backup_components, indent=2)
    compressed_backup = gzip.compress(backup_json.encode('utf-8'))
    backup_b64 = base64.b64encode(compressed_backup).decode('utf-8')

    backup_info = {
        "backup_id": f"backup_{int(time.time())}",
        "size_original": len(backup_json),
        "size_compressed": len(compressed_backup),
        "compression_ratio": len(compressed_backup) / len(backup_json),
        "data": backup_b64,
        "created_at": time.time()
    }

    # Store backup
    context.set_variable("disaster_backup", backup_info)

    print(f"✅ Backup created: {backup_info['backup_id']}")
    print(f"   Original size: {backup_info['size_original']} bytes")
    print(f"   Compressed: {backup_info['size_compressed']} bytes ({backup_info['compression_ratio']:.2f} ratio)")

@state(priority=Priority.CRITICAL, timeout=60.0)
async def test_disaster_recovery(context):
    """Test disaster recovery procedures"""
    print("🧪 Testing disaster recovery...")

    backup_info = context.get_variable("disaster_backup")
    if not backup_info:
        print("❌ No backup available for testing")
        return

    try:
        # Simulate disaster recovery process
        print("   💥 Simulating disaster scenario...")

        # Test backup restoration
        backup_data = backup_info["data"]
        compressed_data = base64.b64decode(backup_data.encode('utf-8'))
        restored_json = gzip.decompress(compressed_data).decode('utf-8')
        restored_data = json.loads(restored_json)

        # Validate restored data
        required_components = ["workflow_state", "configuration", "metadata"]
        for component in required_components:
            if component not in restored_data:
                raise Exception(f"Missing component in backup: {component}")

        # Test data integrity
        metadata = restored_data["metadata"]
        backup_age = time.time() - metadata["backup_time"]

        recovery_test = {
            "status": "success",
            "backup_age_seconds": backup_age,
            "components_restored": len(restored_data),
            "data_integrity": "valid",
            "recovery_time": 2.5  # Simulated recovery time
        }

        context.set_variable("recovery_test", recovery_test)
        context.set_output("disaster_recovery_viable", True)

        print(f"✅ Disaster recovery test passed")
        print(f"   Backup age: {backup_age:.1f} seconds")
        print(f"   Estimated recovery time: {recovery_test['recovery_time']} seconds")

    except Exception as e:
        recovery_test = {
            "status": "failed",
            "error": str(e),
            "recovery_time": None
        }

        context.set_variable("recovery_test", recovery_test)
        context.set_output("disaster_recovery_viable", False)

        print(f"❌ Disaster recovery test failed: {e}")

# Add states to disaster recovery agent
disaster_recovery_agent.add_state("create_disaster_recovery_backup", create_disaster_recovery_backup)
disaster_recovery_agent.add_state("test_disaster_recovery", test_disaster_recovery)
\`\`\`

---

## Reliability Metrics & SLA Monitoring

### Service Level Objectives (SLOs)

\`\`\`python
from dataclasses import dataclass
from typing import List

@dataclass
class SLO:
    name: str
    target_percentage: float
    measurement_window_hours: int
    current_performance: float = 0.0

reliability_metrics_agent = Agent("reliability-metrics")

@state(rate_limit=1.0, timeout=30.0)
async def calculate_reliability_metrics(context):
    """Calculate key reliability metrics and SLO compliance"""
    print("📈 Calculating reliability metrics...")

    # Define SLOs
    slos = [
        SLO("Availability", 99.9, 24),      # 99.9% uptime in 24 hours
        SLO("Response Time", 95.0, 1),      # 95% requests under SLA in 1 hour
        SLO("Error Rate", 99.0, 1),         # 99% success rate in 1 hour
        SLO("Recovery Time", 90.0, 24)      # 90% incidents resolved within SLA
    ]

    # Simulate metric collection
    metrics_data = {
        "uptime_percentage": 99.95,
        "avg_response_time": 245,  # milliseconds
        "error_rate_percentage": 0.5,
        "incidents_resolved_in_sla": 8,
        "total_incidents": 9
    }

    slo_compliance = {}

    for slo in slos:
        if slo.name == "Availability":
            slo.current_performance = metrics_data["uptime_percentage"]

        elif slo.name == "Response Time":
            # Calculate percentage of requests meeting response time SLA
            slo.current_performance = 96.2  # Simulated

        elif slo.name == "Error Rate":
            # Calculate success rate
            slo.current_performance = 100 - metrics_data["error_rate_percentage"]

        elif slo.name == "Recovery Time":
            # Calculate incident resolution SLA compliance
            if metrics_data["total_incidents"] > 0:
                slo.current_performance = (metrics_data["incidents_resolved_in_sla"] /
                                         metrics_data["total_incidents"]) * 100
            else:
                slo.current_performance = 100.0

        # Determine compliance status
        compliance_status = "compliant" if slo.current_performance >= slo.target_percentage else "non_compliant"
        compliance_margin = slo.current_performance - slo.target_percentage

        slo_compliance[slo.name] = {
            "target": slo.target_percentage,
            "current": slo.current_performance,
            "status": compliance_status,
            "margin": compliance_margin,
            "window_hours": slo.measurement_window_hours
        }

        status_emoji = "✅" if compliance_status == "compliant" else "❌"
        print(f"{status_emoji} {slo.name}: {slo.current_performance:.2f}% (target: {slo.target_percentage}%)")

    # Calculate overall reliability score
    overall_score = sum(min(slo["current"], slo["target"]) / slo["target"] * 100
                       for slo in slo_compliance.values()) / len(slo_compliance)

    reliability_report = {
        "overall_reliability_score": overall_score,
        "slo_compliance": slo_compliance,
        "raw_metrics": metrics_data,
        "timestamp": time.time()
    }

    context.set_variable("reliability_report", reliability_report)
    context.set_output("overall_reliability_score", overall_score)

    print(f"🎯 Overall Reliability Score: {overall_score:.1f}%")

@state
async def generate_reliability_dashboard(context):
    """Generate comprehensive reliability dashboard"""
    print("📊 Generating reliability dashboard...")

    # Gather all reliability data
    system_health = context.get_variable("system_health", {})
    service_level = context.get_variable("service_level", {})
    integrity_report = context.get_variable("integrity_report", {})
    reliability_report = context.get_variable("reliability_report", {})
    recovery_test = context.get_variable("recovery_test", {})

    dashboard = {
        "system_status": {
            "overall_health": system_health.get("overall_status", "unknown"),
            "service_level": service_level.get("level", "unknown"),
            "critical_issues": len(system_health.get("critical_issues", [])),
            "warnings": len(system_health.get("warnings", []))
        },
        "data_integrity": {
            "score": integrity_report.get("overall_score", 0),
            "snapshot_version": integrity_report.get("snapshot_version", 0)
        },
        "reliability_metrics": {
            "overall_score": reliability_report.get("overall_reliability_score", 0),
            "slo_compliance": reliability_report.get("slo_compliance", {})
        },
        "disaster_recovery": {
            "backup_available": "disaster_backup" in context.get_all_variables(),
            "recovery_tested": recovery_test.get("status") == "success" if recovery_test else False,
            "last_test": recovery_test.get("backup_age_seconds", 0) if recovery_test else None
        },
        "generated_at": time.time()
    }

    context.set_output("reliability_dashboard", dashboard)

    print("📋 Reliability Dashboard Summary:")
    print(f"   🏥 System Health: {dashboard['system_status']['overall_health']}")
    print(f"   🎚️ Service Level: {dashboard['system_status']['service_level']}")
    print(f"   📊 Data Integrity: {dashboard['data_integrity']['score']:.1f}%")
    print(f"   🎯 Reliability Score: {dashboard['reliability_metrics']['overall_score']:.1f}%")
    print(f"   💾 Disaster Recovery: {'✅ Ready' if dashboard['disaster_recovery']['recovery_tested'] else '⚠️ Needs Testing'}")

# Add states to reliability metrics agent
reliability_metrics_agent.add_state("calculate_reliability_metrics", calculate_reliability_metrics)
reliability_metrics_agent.add_state("generate_reliability_dashboard", generate_reliability_dashboard)
\`\`\`

---

## Best Practices Summary

### Reliability Design Principles

1. **Design for Failure**
   - Assume components will fail
   - Implement graceful degradation
   - Plan recovery procedures

2. **Monitor Everything**
   - Health checks for all components
   - Performance metrics
   - Data integrity validation

3. **Automate Recovery**
   - Automatic failover mechanisms
   - Self-healing capabilities
   - Minimal manual intervention

4. **Test Disaster Scenarios**
   - Regular disaster recovery drills
   - Backup validation
   - Recovery time measurement

5. **Maintain Service Levels**
   - Clear SLOs and SLAs
   - Performance monitoring
   - User experience focus

### Quick Reference

\`\`\`python
# Health monitoring
@state(priority=Priority.HIGH, timeout=10.0)
async def health_check(context): pass

# Service degradation
service_level = determine_service_level(system_health)

# Data consistency
@state(priority=Priority.HIGH)
async def validate_integrity(context): pass

# Disaster recovery
@state(priority=Priority.CRITICAL, timeout=120.0)
async def create_backup(context): pass

# Reliability metrics
@state(rate_limit=1.0)
async def calculate_slos(context): pass
\`\`\`

Reliability is about building systems that continue to serve users even when things go wrong. Puffinflow's reliability patterns help you create workflows that are resilient, observable, and maintainable in production environments.
`.trim();
