import threading
from unittest.mock import Mock, patch

import pytest
from prometheus_client import CollectorRegistry

from puffinflow.core.observability.config import MetricsConfig
from puffinflow.core.observability.interfaces import MetricType
from puffinflow.core.observability.metrics import (
    PrometheusMetric,
    PrometheusMetricsProvider,
)


class TestPrometheusMetric:
    """Test PrometheusMetric class"""

    def test_init(self):
        """Test PrometheusMetric initialization"""
        mock_prometheus_metric = Mock()
        metric = PrometheusMetric(mock_prometheus_metric, MetricType.COUNTER, 1000)

        assert metric._prometheus_metric is mock_prometheus_metric
        assert metric._metric_type == MetricType.COUNTER
        assert metric._cardinality_limit == 1000
        assert metric._series_count == 0
        assert isinstance(metric._lock, type(threading.Lock()))

    def test_record_counter_with_labels(self):
        """Test recording counter metric with labels"""
        mock_prometheus_metric = Mock()
        mock_labels = Mock()
        mock_prometheus_metric.labels.return_value = mock_labels

        metric = PrometheusMetric(mock_prometheus_metric, MetricType.COUNTER, 1000)

        metric.record(5.0, label1="value1", label2="value2")

        mock_prometheus_metric.labels.assert_called_once_with(
            label1="value1", label2="value2"
        )
        mock_labels.inc.assert_called_once_with(5.0)
        assert metric._series_count == 1

    def test_record_counter_without_labels(self):
        """Test recording counter metric without labels"""
        mock_prometheus_metric = Mock()

        metric = PrometheusMetric(mock_prometheus_metric, MetricType.COUNTER, 1000)

        metric.record(3.0)

        mock_prometheus_metric.inc.assert_called_once_with(3.0)
        assert metric._series_count == 1

    def test_record_gauge_with_labels(self):
        """Test recording gauge metric with labels"""
        mock_prometheus_metric = Mock()
        mock_labels = Mock()
        mock_prometheus_metric.labels.return_value = mock_labels

        metric = PrometheusMetric(mock_prometheus_metric, MetricType.GAUGE, 1000)

        metric.record(10.5, status="active")

        mock_prometheus_metric.labels.assert_called_once_with(status="active")
        mock_labels.set.assert_called_once_with(10.5)
        assert metric._series_count == 1

    def test_record_gauge_without_labels(self):
        """Test recording gauge metric without labels"""
        mock_prometheus_metric = Mock()

        metric = PrometheusMetric(mock_prometheus_metric, MetricType.GAUGE, 1000)

        metric.record(7.5)

        mock_prometheus_metric.set.assert_called_once_with(7.5)
        assert metric._series_count == 1

    def test_record_histogram_with_labels(self):
        """Test recording histogram metric with labels"""
        mock_prometheus_metric = Mock()
        mock_labels = Mock()
        mock_prometheus_metric.labels.return_value = mock_labels

        metric = PrometheusMetric(mock_prometheus_metric, MetricType.HISTOGRAM, 1000)

        metric.record(0.25, endpoint="/api/test")

        mock_prometheus_metric.labels.assert_called_once_with(endpoint="/api/test")
        mock_labels.observe.assert_called_once_with(0.25)
        assert metric._series_count == 1

    def test_record_histogram_without_labels(self):
        """Test recording histogram metric without labels"""
        mock_prometheus_metric = Mock()

        metric = PrometheusMetric(mock_prometheus_metric, MetricType.HISTOGRAM, 1000)

        metric.record(0.15)

        mock_prometheus_metric.observe.assert_called_once_with(0.15)
        assert metric._series_count == 1

    def test_record_with_none_labels(self):
        """Test recording metric with None label values"""
        mock_prometheus_metric = Mock()
        mock_labels = Mock()
        mock_prometheus_metric.labels.return_value = mock_labels

        metric = PrometheusMetric(mock_prometheus_metric, MetricType.COUNTER, 1000)

        metric.record(1.0, valid_label="value", none_label=None)

        # None values should be filtered out
        mock_prometheus_metric.labels.assert_called_once_with(valid_label="value")
        mock_labels.inc.assert_called_once_with(1.0)

    def test_record_converts_labels_to_strings(self):
        """Test that label values are converted to strings"""
        mock_prometheus_metric = Mock()
        mock_labels = Mock()
        mock_prometheus_metric.labels.return_value = mock_labels

        metric = PrometheusMetric(mock_prometheus_metric, MetricType.COUNTER, 1000)

        metric.record(1.0, int_label=42, float_label=3.14, bool_label=True)

        mock_prometheus_metric.labels.assert_called_once_with(
            int_label="42", float_label="3.14", bool_label="True"
        )

    def test_record_cardinality_limit(self):
        """Test cardinality limit protection"""
        mock_prometheus_metric = Mock()

        metric = PrometheusMetric(mock_prometheus_metric, MetricType.COUNTER, 2)

        # First two records should work
        metric.record(1.0)
        metric.record(2.0)
        assert metric._series_count == 2

        # Third record should be skipped due to cardinality limit
        metric.record(3.0)
        assert metric._series_count == 2  # Should not increment

        # Verify only first two calls were made
        assert mock_prometheus_metric.inc.call_count == 2

    def test_record_exception_handling(self):
        """Test exception handling during metric recording"""
        mock_prometheus_metric = Mock()
        mock_prometheus_metric.inc.side_effect = Exception("Prometheus error")

        metric = PrometheusMetric(mock_prometheus_metric, MetricType.COUNTER, 1000)

        with patch("builtins.print") as mock_print:
            # Should not raise exception
            metric.record(1.0)

            # Should print error message
            mock_print.assert_called_once_with(
                "Failed to record metric: Prometheus error"
            )

    def test_record_thread_safety(self):
        """Test thread safety of record method"""
        mock_prometheus_metric = Mock()
        metric = PrometheusMetric(mock_prometheus_metric, MetricType.COUNTER, 1000)

        results = []

        def record_metric():
            for _i in range(10):
                metric.record(1.0)
                results.append(metric._series_count)

        # Create multiple threads
        threads = [threading.Thread(target=record_metric) for _ in range(5)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Final count should be 50 (5 threads * 10 records each)
        assert metric._series_count == 50


class TestPrometheusMetricsProvider:
    """Test PrometheusMetricsProvider class"""

    def test_init(self):
        """Test PrometheusMetricsProvider initialization"""
        config = MetricsConfig(namespace="test", cardinality_limit=5000)
        provider = PrometheusMetricsProvider(config)

        assert provider.config is config
        assert isinstance(provider._registry, CollectorRegistry)
        assert provider._metrics_cache == {}
        assert isinstance(provider._lock, type(threading.Lock()))

    @patch("puffinflow.core.observability.metrics.PrometheusCounter")
    def test_counter_creation(self, mock_counter_class):
        """Test counter metric creation"""
        config = MetricsConfig(namespace="test")
        provider = PrometheusMetricsProvider(config)

        mock_prometheus_counter = Mock()
        mock_counter_class.return_value = mock_prometheus_counter

        metric = provider.counter(
            "requests_total", "Total requests", ["method", "status"]
        )

        # Verify Prometheus counter was created with correct parameters
        mock_counter_class.assert_called_once_with(
            "test_requests_total",
            "Total requests",
            labelnames=["method", "status"],
            registry=provider._registry,
        )

        # Verify PrometheusMetric wrapper was created
        assert isinstance(metric, PrometheusMetric)
        assert metric._metric_type == MetricType.COUNTER

    @patch("puffinflow.core.observability.metrics.PrometheusGauge")
    def test_gauge_creation(self, mock_gauge_class):
        """Test gauge metric creation"""
        config = MetricsConfig(namespace="test")
        provider = PrometheusMetricsProvider(config)

        mock_prometheus_gauge = Mock()
        mock_gauge_class.return_value = mock_prometheus_gauge

        metric = provider.gauge("active_connections", "Active connections", ["service"])

        # Verify Prometheus gauge was created with correct parameters
        mock_gauge_class.assert_called_once_with(
            "test_active_connections",
            "Active connections",
            labelnames=["service"],
            registry=provider._registry,
        )

        # Verify PrometheusMetric wrapper was created
        assert isinstance(metric, PrometheusMetric)
        assert metric._metric_type == MetricType.GAUGE

    @patch("puffinflow.core.observability.metrics.PrometheusHistogram")
    def test_histogram_creation(self, mock_histogram_class):
        """Test histogram metric creation"""
        config = MetricsConfig(namespace="test")
        provider = PrometheusMetricsProvider(config)

        mock_prometheus_histogram = Mock()
        mock_histogram_class.return_value = mock_prometheus_histogram

        metric = provider.histogram(
            "request_duration", "Request duration", ["endpoint"]
        )

        # Verify Prometheus histogram was created with correct parameters
        mock_histogram_class.assert_called_once_with(
            "test_request_duration",
            "Request duration",
            labelnames=["endpoint"],
            registry=provider._registry,
        )

        # Verify PrometheusMetric wrapper was created
        assert isinstance(metric, PrometheusMetric)
        assert metric._metric_type == MetricType.HISTOGRAM

    def test_metric_caching(self):
        """Test that metrics are cached"""
        config = MetricsConfig(namespace="test")
        provider = PrometheusMetricsProvider(config)

        with patch(
            "puffinflow.core.observability.metrics.PrometheusCounter"
        ) as mock_counter:
            mock_counter.return_value = Mock()

            # Create metric twice
            metric1 = provider.counter("test_metric")
            metric2 = provider.counter("test_metric")

            # Should be the same instance (cached)
            assert metric1 is metric2

            # Prometheus counter should only be created once
            assert mock_counter.call_count == 1

    def test_metric_with_no_labels(self):
        """Test metric creation with no labels"""
        config = MetricsConfig(namespace="test")
        provider = PrometheusMetricsProvider(config)

        with patch(
            "puffinflow.core.observability.metrics.PrometheusCounter"
        ) as mock_counter:
            mock_counter.return_value = Mock()

            provider.counter("simple_counter")

            mock_counter.assert_called_once_with(
                "test_simple_counter", "", labelnames=[], registry=provider._registry
            )

    def test_unsupported_metric_type(self):
        """Test error handling for unsupported metric type"""
        config = MetricsConfig(namespace="test")
        provider = PrometheusMetricsProvider(config)

        # Mock an unsupported metric type
        with pytest.raises(ValueError, match="Unsupported metric type"):
            provider._get_or_create_metric("test", "unsupported_type", "", [])

    @patch("puffinflow.core.observability.metrics.generate_latest")
    def test_export_metrics(self, mock_generate_latest):
        """Test metrics export"""
        config = MetricsConfig(namespace="test")
        provider = PrometheusMetricsProvider(config)

        mock_generate_latest.return_value = b"# HELP test_metric Test metric\n# TYPE test_metric counter\ntest_metric 1.0\n"

        result = provider.export_metrics()

        mock_generate_latest.assert_called_once_with(provider._registry)
        assert (
            result
            == "# HELP test_metric Test metric\n# TYPE test_metric counter\ntest_metric 1.0\n"
        )

    def test_thread_safety(self):
        """Test thread safety of metric creation"""
        config = MetricsConfig(namespace="test")
        provider = PrometheusMetricsProvider(config)

        created_metrics = []

        def create_metrics():
            for i in range(10):
                metric = provider.counter(f"metric_{i}")
                created_metrics.append(metric)

        # Create multiple threads
        threads = [threading.Thread(target=create_metrics) for _ in range(3)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have created 30 metrics total (3 threads * 10 metrics each)
        assert len(created_metrics) == 30

        # Each metric should be unique (different names)
        [f"test_metric_{i}" for i in range(10)]
        assert len(provider._metrics_cache) == 10  # 10 unique metric names

    def test_cardinality_limit_passed_to_metric(self):
        """Test that cardinality limit is passed to PrometheusMetric"""
        config = MetricsConfig(namespace="test", cardinality_limit=5000)
        provider = PrometheusMetricsProvider(config)

        with patch(
            "puffinflow.core.observability.metrics.PrometheusCounter"
        ) as mock_counter:
            mock_counter.return_value = Mock()

            metric = provider.counter("test_metric")

            assert metric._cardinality_limit == 5000
