import threading
from typing import Any, Optional

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client import Counter as PrometheusCounter
from prometheus_client import Gauge as PrometheusGauge
from prometheus_client import Histogram as PrometheusHistogram

from .config import MetricsConfig
from .interfaces import Metric, MetricsProvider, MetricType


class PrometheusMetric(Metric):
    """Prometheus metric wrapper"""

    def __init__(
        self, prometheus_metric: Any, metric_type: MetricType, cardinality_limit: int
    ) -> None:
        self._prometheus_metric = prometheus_metric
        self._metric_type = metric_type
        self._cardinality_limit = cardinality_limit
        self._series_count = 0
        self._lock = threading.Lock()

    def record(self, value: float, **labels: Any) -> None:
        """Record metric value"""
        # Basic cardinality protection
        with self._lock:
            if self._series_count >= self._cardinality_limit:
                return  # Skip to prevent memory issues

            # Convert label values to strings
            str_labels = {k: str(v) for k, v in labels.items() if v is not None}

            try:
                if str_labels:
                    if self._metric_type == MetricType.COUNTER:
                        self._prometheus_metric.labels(**str_labels).inc(value)
                    elif self._metric_type == MetricType.GAUGE:
                        self._prometheus_metric.labels(**str_labels).set(value)
                    elif self._metric_type == MetricType.HISTOGRAM:
                        self._prometheus_metric.labels(**str_labels).observe(value)
                else:
                    if self._metric_type == MetricType.COUNTER:
                        self._prometheus_metric.inc(value)
                    elif self._metric_type == MetricType.GAUGE:
                        self._prometheus_metric.set(value)
                    elif self._metric_type == MetricType.HISTOGRAM:
                        self._prometheus_metric.observe(value)

                self._series_count += 1

            except Exception as e:
                # Log error but don't fail the application
                print(f"Failed to record metric: {e}")


class PrometheusMetricsProvider(MetricsProvider):
    """Prometheus metrics provider"""

    def __init__(self, config: MetricsConfig):
        self.config = config
        self._registry = CollectorRegistry()
        self._metrics_cache: dict[str, Metric] = {}
        self._lock = threading.Lock()

    def counter(
        self, name: str, description: str = "", labels: Optional[list[str]] = None
    ) -> Metric:
        """Create counter metric"""
        return self._get_or_create_metric(
            name, MetricType.COUNTER, description, labels or []
        )

    def gauge(
        self, name: str, description: str = "", labels: Optional[list[str]] = None
    ) -> Metric:
        """Create gauge metric"""
        return self._get_or_create_metric(
            name, MetricType.GAUGE, description, labels or []
        )

    def histogram(
        self, name: str, description: str = "", labels: Optional[list[str]] = None
    ) -> Metric:
        """Create histogram metric"""
        return self._get_or_create_metric(
            name, MetricType.HISTOGRAM, description, labels or []
        )

    def _get_or_create_metric(
        self, name: str, metric_type: MetricType, description: str, labels: list[str]
    ) -> Metric:
        """Get or create metric"""
        metric_key = f"{self.config.namespace}_{name}"

        with self._lock:
            if metric_key in self._metrics_cache:
                return self._metrics_cache[metric_key]

            labelnames = labels or []

            prometheus_metric: Any
            if metric_type == MetricType.COUNTER:
                prometheus_metric = PrometheusCounter(
                    metric_key,
                    description,
                    labelnames=labelnames,
                    registry=self._registry,
                )
            elif metric_type == MetricType.GAUGE:
                prometheus_metric = PrometheusGauge(
                    metric_key,
                    description,
                    labelnames=labelnames,
                    registry=self._registry,
                )
            elif metric_type == MetricType.HISTOGRAM:
                prometheus_metric = PrometheusHistogram(
                    metric_key,
                    description,
                    labelnames=labelnames,
                    registry=self._registry,
                )
            else:
                raise ValueError(f"Unsupported metric type: {metric_type}")

            metric = PrometheusMetric(
                prometheus_metric, metric_type, self.config.cardinality_limit
            )
            self._metrics_cache[metric_key] = metric
            return metric

    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        result: bytes = generate_latest(self._registry)
        return result.decode("utf-8")
