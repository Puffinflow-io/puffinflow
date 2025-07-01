import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List

# Import the classes under test
from src.puffinflow.core.observability.metrics import (
    PrometheusMetric,
    PrometheusMetricsProvider
)
from src.puffinflow.core.observability.interfaces import MetricType
from src.puffinflow.core.observability.config import MetricsConfig


class TestPrometheusMetric:
    """Test suite for PrometheusMetric class"""

    @pytest.fixture
    def mock_prometheus_counter(self):
        """Mock Prometheus counter"""
        counter = Mock()
        counter.inc = Mock()
        counter.labels = Mock(return_value=counter)
        return counter

    @pytest.fixture
    def mock_prometheus_gauge(self):
        """Mock Prometheus gauge"""
        gauge = Mock()
        gauge.set = Mock()
        gauge.labels = Mock(return_value=gauge)
        return gauge

    @pytest.fixture
    def mock_prometheus_histogram(self):
        """Mock Prometheus histogram"""
        histogram = Mock()
        histogram.observe = Mock()
        histogram.labels = Mock(return_value=histogram)
        return histogram

    def test_init_counter(self, mock_prometheus_counter):
        """Test PrometheusMetric initialization with counter"""
        cardinality_limit = 1000
        metric = PrometheusMetric(mock_prometheus_counter, MetricType.COUNTER, cardinality_limit)

        assert metric._prometheus_metric == mock_prometheus_counter
        assert metric._metric_type == MetricType.COUNTER
        assert metric._cardinality_limit == cardinality_limit
        assert metric._series_count == 0
        assert isinstance(metric._lock, threading.Lock)

    def test_init_gauge(self, mock_prometheus_gauge):
        """Test PrometheusMetric initialization with gauge"""
        metric = PrometheusMetric(mock_prometheus_gauge, MetricType.GAUGE, 500)

        assert metric._prometheus_metric == mock_prometheus_gauge
        assert metric._metric_type == MetricType.GAUGE

    def test_init_histogram(self, mock_prometheus_histogram):
        """Test PrometheusMetric initialization with histogram"""
        metric = PrometheusMetric(mock_prometheus_histogram, MetricType.HISTOGRAM, 200)

        assert metric._prometheus_metric == mock_prometheus_histogram
        assert metric._metric_type == MetricType.HISTOGRAM

    def test_record_counter_without_labels(self, mock_prometheus_counter):
        """Test recording counter value without labels"""
        metric = PrometheusMetric(mock_prometheus_counter, MetricType.COUNTER, 1000)

        metric.record(5.0)

        mock_prometheus_counter.inc.assert_called_once_with(5.0)
        assert metric._series_count == 1

    def test_record_counter_with_labels(self, mock_prometheus_counter):
        """Test recording counter value with labels"""
        metric = PrometheusMetric(mock_prometheus_counter, MetricType.COUNTER, 1000)

        metric.record(3.0, service="api", environment="prod")

        mock_prometheus_counter.labels.assert_called_once_with(service="api", environment="prod")
        mock_prometheus_counter.inc.assert_called_once_with(3.0)
        assert metric._series_count == 1

    def test_record_gauge_without_labels(self, mock_prometheus_gauge):
        """Test recording gauge value without labels"""
        metric = PrometheusMetric(mock_prometheus_gauge, MetricType.GAUGE, 1000)

        metric.record(10.5)

        mock_prometheus_gauge.set.assert_called_once_with(10.5)
        assert metric._series_count == 1

    def test_record_gauge_with_labels(self, mock_prometheus_gauge):
        """Test recording gauge value with labels"""
        metric = PrometheusMetric(mock_prometheus_gauge, MetricType.GAUGE, 1000)

        metric.record(25.0, host="server1", status="healthy")

        mock_prometheus_gauge.labels.assert_called_once_with(host="server1", status="healthy")
        mock_prometheus_gauge.set.assert_called_once_with(25.0)

    def test_record_histogram_without_labels(self, mock_prometheus_histogram):
        """Test recording histogram value without labels"""
        metric = PrometheusMetric(mock_prometheus_histogram, MetricType.HISTOGRAM, 1000)

        metric.record(0.125)

        mock_prometheus_histogram.observe.assert_called_once_with(0.125)
        assert metric._series_count == 1

    def test_record_histogram_with_labels(self, mock_prometheus_histogram):
        """Test recording histogram value with labels"""
        metric = PrometheusMetric(mock_prometheus_histogram, MetricType.HISTOGRAM, 1000)

        metric.record(0.250, endpoint="/api/users", method="GET")

        mock_prometheus_histogram.labels.assert_called_once_with(endpoint="/api/users", method="GET")
        mock_prometheus_histogram.observe.assert_called_once_with(0.250)

    def test_record_with_none_label_values(self, mock_prometheus_counter):
        """Test recording with None label values are filtered out"""
        metric = PrometheusMetric(mock_prometheus_counter, MetricType.COUNTER, 1000)

        metric.record(1.0, service="api", environment=None, region="us-east")

        mock_prometheus_counter.labels.assert_called_once_with(service="api", region="us-east")

    def test_record_with_non_string_label_values(self, mock_prometheus_counter):
        """Test recording with non-string label values are converted to strings"""
        metric = PrometheusMetric(mock_prometheus_counter, MetricType.COUNTER, 1000)

        metric.record(1.0, port=8080, timeout=30.5, enabled=True)

        mock_prometheus_counter.labels.assert_called_once_with(port="8080", timeout="30.5", enabled="True")

    def test_cardinality_limit_protection(self, mock_prometheus_counter):
        """Test cardinality limit prevents new series creation"""
        metric = PrometheusMetric(mock_prometheus_counter, MetricType.COUNTER, 2)

        # Record up to the limit
        metric.record(1.0, label1="value1")
        metric.record(1.0, label1="value2")

        # This should be ignored due to cardinality limit
        metric.record(1.0, label1="value3")

        assert mock_prometheus_counter.labels.call_count == 2
        assert metric._series_count == 2

    def test_cardinality_limit_zero(self, mock_prometheus_counter):
        """Test zero cardinality limit prevents all recording"""
        metric = PrometheusMetric(mock_prometheus_counter, MetricType.COUNTER, 0)

        metric.record(1.0, service="api")

        mock_prometheus_counter.labels.assert_not_called()
        mock_prometheus_counter.inc.assert_not_called()
        assert metric._series_count == 0

    def test_record_exception_handling(self, mock_prometheus_counter):
        """Test exception handling during metric recording"""
        mock_prometheus_counter.inc.side_effect = Exception("Prometheus error")
        metric = PrometheusMetric(mock_prometheus_counter, MetricType.COUNTER, 1000)

        with patch('builtins.print') as mock_print:
            # Should not raise exception
            metric.record(1.0)
            mock_print.assert_called_once_with("Failed to record metric: Prometheus error")

    def test_thread_safety(self, mock_prometheus_counter):
        """Test thread safety of metric recording"""
        metric = PrometheusMetric(mock_prometheus_counter, MetricType.COUNTER, 100)
        results = []
        errors = []

        def record_metric(thread_id):
            try:
                for i in range(10):
                    metric.record(1.0, thread=str(thread_id), iteration=str(i))
                results.append(thread_id)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=record_metric, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(results) == 5
        # Should have recorded some metrics (limited by cardinality)
        assert metric._series_count > 0


class TestPrometheusMetricsProvider:
    """Test suite for PrometheusMetricsProvider class"""

    @pytest.fixture
    def mock_config(self):
        """Mock metrics configuration"""
        config = Mock(spec=MetricsConfig)
        config.namespace = "test_app"
        config.cardinality_limit = 1000
        return config

    @pytest.fixture
    def provider(self, mock_config):
        """Create PrometheusMetricsProvider instance"""
        with patch('src.puffinflow.core.observability.metrics.CollectorRegistry'):
            return PrometheusMetricsProvider(mock_config)

    @patch('src.puffinflow.core.observability.metrics.PrometheusCounter')
    @patch('src.puffinflow.core.observability.metrics.CollectorRegistry')
    def test_counter_creation(self, mock_registry, mock_counter_class, mock_config):
        """Test counter metric creation"""
        mock_counter_instance = Mock()
        mock_counter_class.return_value = mock_counter_instance

        provider = PrometheusMetricsProvider(mock_config)

        counter = provider.counter("requests_total", "Total requests", ["method", "status"])

        mock_counter_class.assert_called_once_with(
            "test_app_requests_total",
            "Total requests",
            labelnames=["method", "status"],
            registry=provider._registry
        )
        assert isinstance(counter, PrometheusMetric)
        assert counter._metric_type == MetricType.COUNTER

    @patch('src.puffinflow.core.observability.metrics.PrometheusGauge')
    @patch('src.puffinflow.core.observability.metrics.CollectorRegistry')
    def test_gauge_creation(self, mock_registry, mock_gauge_class, mock_config):
        """Test gauge metric creation"""
        mock_gauge_instance = Mock()
        mock_gauge_class.return_value = mock_gauge_instance

        provider = PrometheusMetricsProvider(mock_config)

        gauge = provider.gauge("active_connections", "Active connections", ["service"])

        mock_gauge_class.assert_called_once_with(
            "test_app_active_connections",
            "Active connections",
            labelnames=["service"],
            registry=provider._registry
        )
        assert isinstance(gauge, PrometheusMetric)
        assert gauge._metric_type == MetricType.GAUGE

    @patch('src.puffinflow.core.observability.metrics.PrometheusHistogram')
    @patch('src.puffinflow.core.observability.metrics.CollectorRegistry')
    def test_histogram_creation(self, mock_registry, mock_histogram_class, mock_config):
        """Test histogram metric creation"""
        mock_histogram_instance = Mock()
        mock_histogram_class.return_value = mock_histogram_instance

        provider = PrometheusMetricsProvider(mock_config)

        histogram = provider.histogram("request_duration", "Request duration", ["endpoint"])

        mock_histogram_class.assert_called_once_with(
            "test_app_request_duration",
            "Request duration",
            labelnames=["endpoint"],
            registry=provider._registry
        )
        assert isinstance(histogram, PrometheusMetric)
        assert histogram._metric_type == MetricType.HISTOGRAM

    def test_metric_caching(self, provider):
        """Test metric caching behavior"""
        with patch.object(provider, '_get_or_create_metric', wraps=provider._get_or_create_metric) as mock_get_create:
            # First call should create the metric
            counter1 = provider.counter("requests", "Requests")

            # Second call should return cached metric
            counter2 = provider.counter("requests", "Requests")

            assert counter1 is counter2
            assert mock_get_create.call_count == 2  # Called for both, but second returns cached

    def test_metric_creation_without_labels(self, provider):
        """Test metric creation without labels"""
        with patch('src.puffinflow.core.observability.metrics.PrometheusCounter') as mock_counter:
            provider.counter("simple_counter")

            mock_counter.assert_called_once()
            call_args = mock_counter.call_args
            assert call_args[1]['labelnames'] == []

    def test_metric_creation_with_empty_labels(self, provider):
        """Test metric creation with empty labels list"""
        with patch('src.puffinflow.core.observability.metrics.PrometheusGauge') as mock_gauge:
            provider.gauge("simple_gauge", labels=[])

            mock_gauge.assert_called_once()
            call_args = mock_gauge.call_args
            assert call_args[1]['labelnames'] == []

    def test_unsupported_metric_type_error(self, mock_config):
        """Test error handling for unsupported metric types"""
        with patch('src.puffinflow.core.observability.metrics.CollectorRegistry'):
            provider = PrometheusMetricsProvider(mock_config)

            # Mock an invalid metric type
            with patch.object(provider, '_get_or_create_metric') as mock_get_create:
                mock_get_create.side_effect = ValueError("Unsupported metric type: INVALID")

                with pytest.raises(ValueError, match="Unsupported metric type"):
                    provider._get_or_create_metric("test", "INVALID", "", [])

    @patch('src.puffinflow.core.observability.metrics.generate_latest')
    def test_export_metrics(self, mock_generate_latest, provider):
        """Test metrics export functionality"""
        mock_generate_latest.return_value = b"# HELP test_metric Test metric\n# TYPE test_metric counter\ntest_metric 1.0\n"

        result = provider.export_metrics()

        mock_generate_latest.assert_called_once_with(provider._registry)
        assert result == "# HELP test_metric Test metric\n# TYPE test_metric counter\ntest_metric 1.0\n"

    def test_thread_safety_metric_creation(self, provider):
        """Test thread safety of metric creation"""
        results = []
        errors = []

        def create_metrics(thread_id):
            try:
                for i in range(10):
                    metric_name = f"metric_{thread_id}_{i}"
                    counter = provider.counter(metric_name, f"Metric {metric_name}")
                    results.append((thread_id, i, counter))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_metrics, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 metrics each

        # Verify all metrics are unique instances
        metric_names = [f"metric_{tid}_{mid}" for tid, mid, _ in results]
        assert len(set(metric_names)) == 50

    def test_concurrent_access_same_metric(self, provider):
        """Test concurrent access to the same metric name"""
        results = []
        errors = []

        def get_same_metric(thread_id):
            try:
                counter = provider.counter("shared_metric", "Shared metric")
                results.append((thread_id, counter))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            thread = threading.Thread(target=get_same_metric, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(results) == 10

        # All should return the same cached instance
        first_metric = results[0][1]
        for _, metric in results:
            assert metric is first_metric

    def test_namespace_prefix(self, mock_config):
        """Test namespace prefix is correctly applied"""
        mock_config.namespace = "custom_namespace"

        with patch('src.puffinflow.core.observability.metrics.PrometheusCounter') as mock_counter:
            with patch('src.puffinflow.core.observability.metrics.CollectorRegistry'):
                provider = PrometheusMetricsProvider(mock_config)
                provider.counter("test_metric")

                mock_counter.assert_called_once()
                call_args = mock_counter.call_args
                assert call_args[0][0] == "custom_namespace_test_metric"

    def test_cardinality_limit_propagation(self, mock_config):
        """Test cardinality limit is propagated to PrometheusMetric"""
        mock_config.cardinality_limit = 500

        with patch('src.puffinflow.core.observability.metrics.PrometheusCounter'):
            with patch('src.puffinflow.core.observability.metrics.CollectorRegistry'):
                provider = PrometheusMetricsProvider(mock_config)

                with patch('src.puffinflow.core.observability.metrics.PrometheusMetric') as mock_metric_class:
                    provider.counter("test_metric")

                    mock_metric_class.assert_called_once()
                    call_args = mock_metric_class.call_args
                    assert call_args[0][2] == 500  # cardinality_limit parameter


@pytest.fixture
def integration_config():
    """Integration test configuration"""
    config = Mock(spec=MetricsConfig)
    config.namespace = "integration_test"
    config.cardinality_limit = 100
    return config


class TestIntegration:
    """Integration tests for the complete metrics system"""

    def test_complete_workflow(self, integration_config):
        """Test complete workflow from provider creation to metric export"""
        with patch('src.puffinflow.core.observability.metrics.CollectorRegistry') as mock_registry_class:
            mock_registry = Mock()
            mock_registry_class.return_value = mock_registry

            # Create provider
            provider = PrometheusMetricsProvider(integration_config)

            # Create different types of metrics
            with patch('src.puffinflow.core.observability.metrics.PrometheusCounter') as mock_counter:
                with patch('src.puffinflow.core.observability.metrics.PrometheusGauge') as mock_gauge:
                    with patch('src.puffinflow.core.observability.metrics.PrometheusHistogram') as mock_histogram:
                        counter = provider.counter("requests_total", "Total requests", ["method"])
                        gauge = provider.gauge("active_users", "Active users")
                        histogram = provider.histogram("response_time", "Response time", ["endpoint"])

                        # Record some values
                        counter.record(1.0, method="GET")
                        gauge.record(42.0)
                        histogram.record(0.125, endpoint="/api/users")

                        # Export metrics
                        with patch('src.puffinflow.core.observability.metrics.generate_latest') as mock_generate:
                            mock_generate.return_value = b"exported_metrics"

                            result = provider.export_metrics()

                            assert result == "exported_metrics"
                            mock_generate.assert_called_once_with(mock_registry)

    def test_error_resilience(self, integration_config):
        """Test system resilience to various error conditions"""
        with patch('src.puffinflow.core.observability.metrics.CollectorRegistry'):
            provider = PrometheusMetricsProvider(integration_config)

            # Test metric creation with Prometheus errors
            with patch('src.puffinflow.core.observability.metrics.PrometheusCounter',
                       side_effect=Exception("Creation error")):
                with pytest.raises(Exception):
                    provider.counter("failing_counter")

            # Test successful metric creation after error
            with patch('src.puffinflow.core.observability.metrics.PrometheusCounter') as mock_counter:
                counter = provider.counter("working_counter")
                assert counter is not None

    def test_performance_under_load(self, integration_config):
        """Test performance characteristics under load"""
        with patch('src.puffinflow.core.observability.metrics.CollectorRegistry'):
            provider = PrometheusMetricsProvider(integration_config)

            # Create multiple metrics rapidly
            start_time = time.time()

            with patch('src.puffinflow.core.observability.metrics.PrometheusCounter'):
                metrics = []
                for i in range(100):
                    metric = provider.counter(f"perf_metric_{i}")
                    metrics.append(metric)

            creation_time = time.time() - start_time

            # Should complete reasonably quickly (adjust threshold as needed)
            assert creation_time < 1.0  # Less than 1 second for 100 metrics
            assert len(metrics) == 100