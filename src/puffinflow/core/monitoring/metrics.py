"""Metrics collection with high availability and security."""

import asyncio
import logging
import time
import threading
import weakref
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Flag, auto
from functools import wraps
from typing import (
    Any, Callable, Dict, List, Optional, Set, Union,
    TypeVar, Generic, Protocol, runtime_checkable
)
import statistics
import hashlib
import json
import ssl
import gzip
import pickle
from urllib.parse import urlparse

import structlog
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, CollectorRegistry,
    multiprocess, generate_latest, CONTENT_TYPE_LATEST
)
from prometheus_client.exposition import MetricsHandler
from prometheus_client.parser import text_string_to_metric_families

from src.puffinflow.core.config import get_settings


logger = structlog.get_logger(__name__)
T = TypeVar('T')


class MetricType(Flag):
    """Comprehensive metric types for monitoring"""
    NONE = 0
    TIMING = auto()
    RESOURCES = auto()
    DEPENDENCIES = auto()
    STATE_CHANGES = auto()
    ERRORS = auto()
    THROUGHPUT = auto()
    QUEUE_STATS = auto()
    CONCURRENCY = auto()
    MEMORY = auto()
    PERIODIC = auto()
    RETRIES = auto()
    LATENCY = auto()
    SECURITY = auto()
    BUSINESS = auto()
    INFRASTRUCTURE = auto()
    COMPLIANCE = auto()
    ALL = (TIMING | RESOURCES | DEPENDENCIES | STATE_CHANGES |
           ERRORS | THROUGHPUT | QUEUE_STATS | CONCURRENCY |
           MEMORY | PERIODIC | RETRIES | LATENCY | SECURITY |
           BUSINESS | INFRASTRUCTURE | COMPLIANCE)


@runtime_checkable
class MetricCollector(Protocol):
    """Protocol for metric collectors."""

    def collect(self) -> Dict[str, Any]:
        """Collect current metrics."""
        ...

    def reset(self) -> None:
        """Reset metrics to initial state."""
        ...


@dataclass
class MetricsConfiguration:
    """Configuration for metrics collection."""
    enabled: bool = True
    retention_period: timedelta = timedelta(days=7)
    max_memory_mb: float = 100.0
    compression_enabled: bool = True
    encryption_enabled: bool = False
    pii_filtering: bool = True
    batch_size: int = 1000
    flush_interval: float = 30.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    high_cardinality_limit: int = 10000
    sample_rate: float = 1.0
    backup_enabled: bool = True
    backup_interval: timedelta = timedelta(hours=1)


@dataclass
class TimestampedValue:
    """Timestamped metric value."""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricAggregation:
    """Thread-safe aggregated metrics container with memory management"""
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _last_cleanup: float = field(default_factory=time.time)

    def add(self, value: float, timestamp: Optional[float] = None,
            labels: Optional[Dict[str, str]] = None) -> None:
        """Thread-safe add operation with automatic cleanup."""
        with self._lock:
            self.count += 1
            self.sum += value
            self.min = min(self.min, value)
            self.max = max(self.max, value)
            ts = timestamp or time.time()
            self.values.append(TimestampedValue(value, ts, labels or {}))

            # Periodic cleanup to prevent memory leaks
            if ts - self._last_cleanup > 300:  # 5 minutes
                self._cleanup_old_values(ts - 3600)  # Keep 1 hour
                self._last_cleanup = ts

    @property
    def avg(self) -> float:
        with self._lock:
            return self.sum / self.count if self.count > 0 else 0

    @property
    def median(self) -> float:
        with self._lock:
            if not self.values:
                return 0
            sorted_values = sorted([v.value for v in self.values])
            return statistics.median(sorted_values)

    @property
    def percentile_95(self) -> float:
        with self._lock:
            if len(self.values) < 20:
                return self.max if self.max != float('-inf') else 0
            sorted_values = sorted([v.value for v in self.values])
            idx = int(0.95 * len(sorted_values))
            return sorted_values[idx]

    @property
    def percentile_99(self) -> float:
        with self._lock:
            if len(self.values) < 100:
                return self.max if self.max != float('-inf') else 0
            sorted_values = sorted([v.value for v in self.values])
            idx = int(0.99 * len(sorted_values))
            return sorted_values[idx]

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "count": self.count,
                "min": self.min if self.min != float('inf') else 0,
                "max": self.max if self.max != float('-inf') else 0,
                "avg": self.avg,
                "median": self.median,
                "p95": self.percentile_95,
                "p99": self.percentile_99,
                "timestamp": time.time(),
                "memory_usage_kb": len(self.values) * 64 / 1024  # Rough estimate
            }

    def _cleanup_old_values(self, cutoff_time: float) -> None:
        """Remove values older than cutoff time."""
        while self.values and self.values[0].timestamp < cutoff_time:
            self.values.popleft()


class CircuitBreaker:
    """Circuit breaker for external metric exports with exponential backoff."""

    def __init__(self, threshold: int = 5, timeout: float = 60.0, max_timeout: float = 300.0):
        self.threshold = threshold
        self.timeout = timeout
        self.max_timeout = max_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'closed'  # closed, open, half-open
        self._lock = threading.Lock()
        self._consecutive_failures = 0

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            with self._lock:
                current_time = time.time()

                if self.state == 'open':
                    # Exponential backoff
                    backoff_time = min(
                        self.timeout * (2 ** self._consecutive_failures),
                        self.max_timeout
                    )

                    if current_time - self.last_failure_time > backoff_time:
                        self.state = 'half-open'
                        logger.info("circuit_breaker_half_open", func=func.__name__)
                    else:
                        logger.warning("circuit_breaker_blocked", func=func.__name__)
                        raise Exception("Circuit breaker is open")

            try:
                result = await func(*args, **kwargs)
                with self._lock:
                    if self.state == 'half-open':
                        self.state = 'closed'
                        self.failure_count = 0
                        self._consecutive_failures = 0
                        logger.info("circuit_breaker_closed", func=func.__name__)
                return result
            except Exception as e:
                with self._lock:
                    self.failure_count += 1
                    self._consecutive_failures += 1
                    self.last_failure_time = current_time
                    if self.failure_count >= self.threshold:
                        self.state = 'open'
                        logger.error("circuit_breaker_opened", func=func.__name__,
                                   failures=self.failure_count)
                raise

        return wrapper

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state for monitoring."""
        with self._lock:
            return {
                'state': self.state,
                'failure_count': self.failure_count,
                'consecutive_failures': self._consecutive_failures,
                'last_failure_time': self.last_failure_time
            }


class DataSanitizer:
    """Sanitize sensitive data from metrics."""

    PII_PATTERNS = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b(?:\d{4}[-\s]?){3}\d{4}\b',  # Credit card
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP address
        r'\b[A-Za-z0-9]{32,}\b',  # API keys/tokens (32+ chars)
    ]

    SENSITIVE_KEYS = {
        'password', 'secret', 'key', 'token', 'auth', 'credential',
        'private', 'confidential', 'ssn', 'social_security'
    }

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        import re
        self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.PII_PATTERNS]

    def sanitize(self, data: Any) -> Any:
        """Sanitize data by removing PII and sensitive information."""
        if not self.enabled:
            return data

        if isinstance(data, str):
            sanitized = data
            for pattern in self.patterns:
                sanitized = pattern.sub('[REDACTED]', sanitized)
            return sanitized
        elif isinstance(data, dict):
            return {
                k: '[REDACTED]' if self._is_sensitive_key(k) else self.sanitize(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self.sanitize(item) for item in data]
        return data

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if key contains sensitive information."""
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in self.SENSITIVE_KEYS)


class MetricsCompressor:
    """Compress metrics data for storage and transmission."""

    @staticmethod
    def compress(data: bytes) -> bytes:
        """Compress data using gzip."""
        return gzip.compress(data)

    @staticmethod
    def decompress(data: bytes) -> bytes:
        """Decompress gzip data."""
        return gzip.decompress(data)

    @staticmethod
    def serialize_and_compress(obj: Any) -> bytes:
        """Serialize object to pickle and compress."""
        pickled = pickle.dumps(obj)
        return MetricsCompressor.compress(pickled)

    @staticmethod
    def decompress_and_deserialize(data: bytes) -> Any:
        """Decompress and deserialize object."""
        decompressed = MetricsCompressor.decompress(data)
        return pickle.loads(decompressed)


class HighCardinalityController:
    """Control high cardinality metrics to prevent memory issues."""

    def __init__(self, limit: int = 10000):
        self.limit = limit
        self.cardinality_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def should_record(self, metric_name: str, labels: Dict[str, str]) -> bool:
        """Check if metric should be recorded based on cardinality limits."""
        label_hash = self._hash_labels(labels)
        key = f"{metric_name}:{label_hash}"

        with self._lock:
            if key in self.cardinality_counts:
                return True

            total_cardinality = sum(self.cardinality_counts.values())
            if total_cardinality >= self.limit:
                logger.warning(
                    "high_cardinality_limit_reached",
                    metric=metric_name,
                    total_cardinality=total_cardinality,
                    limit=self.limit
                )
                return False

            self.cardinality_counts[key] = 1
            return True

    def _hash_labels(self, labels: Dict[str, str]) -> str:
        """Create deterministic hash of labels."""
        sorted_items = sorted(labels.items())
        label_str = ','.join(f"{k}={v}" for k, v in sorted_items)
        return hashlib.md5(label_str.encode()).hexdigest()[:16]

    def get_cardinality_stats(self) -> Dict[str, Any]:
        """Get cardinality statistics."""
        with self._lock:
            return {
                'total_series': len(self.cardinality_counts),
                'limit': self.limit,
                'utilization': len(self.cardinality_counts) / self.limit,
                'top_metrics': dict(list(self.cardinality_counts.items())[:10])
            }


class MetricsCollector:
    """Production-grade metrics collector with comprehensive monitoring."""

    def __init__(self, config: Optional[MetricsConfiguration] = None):
        self.config = config or MetricsConfiguration()
        self.registry = CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._aggregations: Dict[str, MetricAggregation] = defaultdict(MetricAggregation)
        self._lock = threading.RLock()
        self._running = False
        self._tasks: Set[asyncio.Task] = set()

        # Components
        self._sanitizer = DataSanitizer(self.config.pii_filtering)
        self._circuit_breaker = CircuitBreaker(
            self.config.circuit_breaker_threshold,
            self.config.circuit_breaker_timeout
        )
        self._cardinality_controller = HighCardinalityController(
            self.config.high_cardinality_limit
        )
        self._compressor = MetricsCompressor()

        # Health monitoring
        self._health_metrics = {
            'collection_errors': 0,
            'export_errors': 0,
            'memory_usage_mb': 0,
            'last_collection_time': 0,
            'collections_per_second': 0,
            'dropped_metrics': 0,
            'compressed_bytes_saved': 0
        }

        # Sampling
        self._sample_counter = 0

        self._setup_prometheus_metrics()
        self._setup_self_monitoring()

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics with comprehensive labels."""
        common_labels = ['service', 'version', 'environment', 'instance', 'datacenter']

        # Core workflow metrics
        self._metrics['state_duration'] = Histogram(
            'workflow_state_duration_seconds',
            'State execution duration in seconds',
            ['agent', 'state', 'status', 'priority', 'retry_attempt'] + common_labels,
            registry=self.registry,
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, float('inf'))
        )

        self._metrics['state_total'] = Counter(
            'workflow_state_executions_total',
            'Total number of state executions',
            ['agent', 'state', 'status'] + common_labels,
            registry=self.registry
        )

        # Resource metrics
        self._metrics['resource_usage'] = Gauge(
            'workflow_resource_usage_ratio',
            'Current resource usage ratio (0-1)',
            ['resource_type', 'agent'] + common_labels,
            registry=self.registry
        )

        self._metrics['resource_allocation'] = Histogram(
            'workflow_resource_allocation_seconds',
            'Time taken to allocate resources',
            ['resource_type', 'agent'] + common_labels,
            registry=self.registry
        )

        # Queue metrics
        self._metrics['queue_size'] = Gauge(
            'workflow_queue_size_total',
            'Current queue size',
            ['agent', 'priority'] + common_labels,
            registry=self.registry
        )

        self._metrics['queue_wait_time'] = Histogram(
            'workflow_queue_wait_seconds',
            'Time spent waiting in queue',
            ['agent', 'priority'] + common_labels,
            registry=self.registry
        )

        # Error metrics
        self._metrics['errors_total'] = Counter(
            'workflow_errors_total',
            'Total number of errors',
            ['agent', 'state', 'error_type', 'error_code'] + common_labels,
            registry=self.registry
        )

        # Performance metrics
        self._metrics['throughput'] = Gauge(
            'workflow_throughput_per_second',
            'Workflow throughput per second',
            ['agent'] + common_labels,
            registry=self.registry
        )

        # Security metrics
        self._metrics['security_events'] = Counter(
            'workflow_security_events_total',
            'Security-related events',
            ['event_type', 'severity'] + common_labels,
            registry=self.registry
        )

        # Business metrics
        self._metrics['business_value'] = Counter(
            'workflow_business_value_total',
            'Business value generated by workflows',
            ['value_type', 'currency'] + common_labels,
            registry=self.registry
        )

    def _setup_self_monitoring(self):
        """Setup self-monitoring metrics."""
        self._metrics['collector_operations'] = Counter(
            'metrics_collector_operations_total',
            'Metrics collector operations',
            ['operation', 'status'],
            registry=self.registry
        )

        self._metrics['collector_memory'] = Gauge(
            'metrics_collector_memory_usage_bytes',
            'Memory usage of metrics collector',
            registry=self.registry
        )

        self._metrics['collector_cardinality'] = Gauge(
            'metrics_collector_cardinality_total',
            'Total number of metric series',
            registry=self.registry
        )

    async def start(self) -> None:
        """Start the metrics collector."""
        self._running = True

        # Start background tasks
        tasks = [
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._memory_manager()),
            asyncio.create_task(self._backup_manager()),
        ]

        self._tasks.update(tasks)

        logger.info(
            "metrics_collector_started",
            config=self.config.__dict__,
            prometheus_metrics=len(self._metrics)
        )

    async def stop(self) -> None:
        """Stop the metrics collector."""
        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)

        # Final export
        await self._export_final_metrics()

        logger.info("metrics_collector_stopped", health=self._health_metrics)

    def record_state_execution(
        self,
        agent: str,
        state: str,
        duration: float,
        status: str,
        priority: str = "normal",
        retry_attempt: int = 0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record state execution metrics with sampling."""
        if not self._should_sample():
            return

        try:
            # Sanitize inputs
            sanitized_labels = self._sanitizer.sanitize(labels or {})

            # Check cardinality
            all_labels = {
                'agent': agent,
                'state': state,
                'status': status,
                'priority': priority,
                'retry_attempt': str(retry_attempt),
                **sanitized_labels
            }

            if not self._cardinality_controller.should_record('state_duration', all_labels):
                self._health_metrics['dropped_metrics'] += 1
                return

            # Record metrics
            self._metrics['state_duration'].labels(**all_labels).observe(duration)
            self._metrics['state_total'].labels(
                agent=agent,
                state=state,
                status=status,
                **self._get_common_labels()
            ).inc()

            # Update aggregations
            key = f"state_execution:{agent}:{state}"
            self._aggregations[key].add(duration, labels=sanitized_labels)

            self._metrics['collector_operations'].labels(
                operation='record_state',
                status='success'
            ).inc()

        except Exception as e:
            self._health_metrics['collection_errors'] += 1
            self._metrics['collector_operations'].labels(
                operation='record_state',
                status='error'
            ).inc()
            logger.error("metrics_collection_error", error=str(e))

    def record_resource_usage(
        self,
        resource_type: str,
        agent: str,
        usage_ratio: float,
        allocation_time: Optional[float] = None
    ) -> None:
        """Record resource usage metrics."""
        if not self._should_sample():
            return

        try:
            labels = {
                'resource_type': resource_type,
                'agent': agent,
                **self._get_common_labels()
            }

            self._metrics['resource_usage'].labels(**labels).set(usage_ratio)

            if allocation_time is not None:
                self._metrics['resource_allocation'].labels(**labels).observe(allocation_time)

        except Exception as e:
            self._health_metrics['collection_errors'] += 1
            logger.error("resource_metrics_error", error=str(e))

    def record_security_event(
        self,
        event_type: str,
        severity: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record security events."""
        try:
            # Always record security events regardless of sampling
            sanitized_details = self._sanitizer.sanitize(details or {})

            self._metrics['security_events'].labels(
                event_type=event_type,
                severity=severity,
                **self._get_common_labels()
            ).inc()

            logger.info(
                "security_event_recorded",
                event_type=event_type,
                severity=severity,
                details=sanitized_details
            )

        except Exception as e:
            logger.error("security_metrics_error", error=str(e))

    def record_business_metric(
        self,
        value_type: str,
        amount: float,
        currency: str = "USD"
    ) -> None:
        """Record business value metrics."""
        try:
            self._metrics['business_value'].labels(
                value_type=value_type,
                currency=currency,
                **self._get_common_labels()
            ).inc(amount)

        except Exception as e:
            logger.error("business_metrics_error", error=str(e))

    def _should_sample(self) -> bool:
        """Determine if metric should be sampled."""
        if self.config.sample_rate >= 1.0:
            return True

        self._sample_counter += 1
        return (self._sample_counter % int(1 / self.config.sample_rate)) == 0

    def _get_common_labels(self) -> Dict[str, str]:
        """Get common labels for all metrics."""
        settings = get_settings()
        return {
            'service': 'workflow-orchestrator',
            'version': '1.0.0',  # Should come from version management
            'environment': settings.environment,
            'instance': 'default',  # Should be unique instance ID
            'datacenter': 'default'  # Should come from deployment config
        }

    async def _health_monitor(self) -> None:
        """Monitor collector health."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Update health metrics
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self._health_metrics['memory_usage_mb'] = memory_mb
                self._health_metrics['last_collection_time'] = time.time()

                # Update Prometheus self-monitoring
                self._metrics['collector_memory'].set(memory_mb * 1024 * 1024)
                self._metrics['collector_cardinality'].set(
                    len(self._cardinality_controller.cardinality_counts)
                )

                # Check memory limits
                if memory_mb > self.config.max_memory_mb:
                    logger.warning(
                        "metrics_memory_limit_exceeded",
                        current_mb=memory_mb,
                        limit_mb=self.config.max_memory_mb
                    )
                    await self._cleanup_memory()

            except Exception as e:
                logger.error("health_monitor_error", error=str(e))

    async def _memory_manager(self) -> None:
        """Manage memory usage by cleaning up old data."""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_memory()

            except Exception as e:
                logger.error("memory_manager_error", error=str(e))

    async def _cleanup_memory(self) -> None:
        """Clean up old aggregation data."""
        cutoff_time = time.time() - self.config.retention_period.total_seconds()

        with self._lock:
            # Clean up aggregations
            for agg in self._aggregations.values():
                agg._cleanup_old_values(cutoff_time)

            # Remove empty aggregations
            empty_keys = [
                key for key, agg in self._aggregations.items()
                if not agg.values
            ]
            for key in empty_keys:
                del self._aggregations[key]

        logger.debug("memory_cleanup_completed", removed_keys=len(empty_keys))

    async def _backup_manager(self) -> None:
        """Periodically backup metrics data."""
        if not self.config.backup_enabled:
            return

        while self._running:
            try:
                await asyncio.sleep(self.config.backup_interval.total_seconds())
                await self._create_backup()

            except Exception as e:
                logger.error("backup_manager_error", error=str(e))

    async def _create_backup(self) -> None:
        """Create compressed backup of current metrics."""
        try:
            backup_data = {
                'timestamp': time.time(),
                'aggregations': dict(self._aggregations),
                'health_metrics': self._health_metrics.copy(),
                'cardinality_stats': self._cardinality_controller.get_cardinality_stats()
            }

            # Compress backup
            compressed_data = self._compressor.serialize_and_compress(backup_data)

            # Calculate savings
            original_size = len(str(backup_data).encode())
            compressed_size = len(compressed_data)
            savings = original_size - compressed_size
            self._health_metrics['compressed_bytes_saved'] += savings

            # In production, save to persistent storage
            # await self._save_backup_to_storage(compressed_data)

            logger.debug(
                "backup_created",
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compressed_size / original_size
            )

        except Exception as e:
            logger.error("backup_creation_error", error=str(e))

    @_circuit_breaker
    async def _export_final_metrics(self) -> None:
        """Export final metrics on shutdown."""
        try:
            # Generate Prometheus metrics
            metrics_output = generate_latest(self.registry)

            # In production, send to monitoring system
            logger.info(
                "final_metrics_exported",
                size_bytes=len(metrics_output),
                health=self._health_metrics
            )

        except Exception as e:
            logger.error("final_export_error", error=str(e))

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            'status': 'healthy' if self._running else 'stopped',
            'metrics': self._health_metrics.copy(),
            'cardinality': self._cardinality_controller.get_cardinality_stats(),
            'circuit_breaker': self._circuit_breaker.get_state(),
            'config': {
                'retention_hours': self.config.retention_period.total_seconds() / 3600,
                'max_memory_mb': self.config.max_memory_mb,
                'sample_rate': self.config.sample_rate,
                'compression_enabled': self.config.compression_enabled
            }
        }

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')


# Decorator for monitoring agent operations
def monitor_agent_operations(
    metrics: MetricType = MetricType.ALL,
    collector: Optional[MetricsCollector] = None,
    sample_rate: float = 1.0
):
    """
    Production-grade decorator for agent monitoring with comprehensive metrics.
    """
    def decorator(coro):
        @wraps(coro)
        async def wrapper(*args, **kwargs):
            agent = args[0] if len(args) > 0 else kwargs.get('agent')
            if not agent:
                return await coro(*args, **kwargs)

            metrics_collector = collector or MetricsCollector()
            start_time = time.time()

            try:
                # Record start
                if MetricType.STATE_CHANGES in metrics:
                    metrics_collector.record_security_event(
                        "agent_execution_started",
                        "info",
                        {"agent_name": agent.name}
                    )

                # Execute function
                result = await coro(*args, **kwargs)

                # Record success metrics
                execution_time = time.time() - start_time
                if MetricType.TIMING in metrics:
                    metrics_collector.record_state_execution(
                        agent.name,
                        "workflow_execution",
                        execution_time,
                        "success"
                    )

                return result

            except Exception as e:
                # Record error metrics
                execution_time = time.time() - start_time
                if MetricType.ERRORS in metrics:
                    metrics_collector.record_state_execution(
                        agent.name,
                        "workflow_execution",
                        execution_time,
                        "error"
                    )

                    metrics_collector.record_security_event(
                        "agent_execution_failed",
                        "error",
                        {
                            "agent_name": agent.name,
                            "error_type": type(e).__name__,
                            "error_message": str(e)
                        }
                    )

                raise

        return wrapper
    return decorator