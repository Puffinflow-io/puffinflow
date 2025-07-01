import os
import pytest
from unittest.mock import patch
from typing import List

from src.puffinflow.core.observability.config import (
    TracingConfig,
    MetricsConfig,
    AlertingConfig,
    EventsConfig,
    ObservabilityConfig
)


class TestTracingConfig:
    """Test cases for TracingConfig"""

    def test_default_values(self):
        """Test default configuration values"""
        config = TracingConfig()

        assert config.enabled is True
        assert config.service_name == "puffinflow"
        assert config.service_version == "1.0.0"
        assert config.sample_rate == 1.0
        assert config.otlp_endpoint is None
        assert config.jaeger_endpoint is None
        assert config.console_enabled is False

    def test_custom_values(self):
        """Test custom configuration values"""
        config = TracingConfig(
            enabled=False,
            service_name="test-service",
            service_version="2.0.0",
            sample_rate=0.5,
            otlp_endpoint="http://localhost:4317",
            jaeger_endpoint="http://localhost:14268",
            console_enabled=True
        )

        assert config.enabled is False
        assert config.service_name == "test-service"
        assert config.service_version == "2.0.0"
        assert config.sample_rate == 0.5
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.jaeger_endpoint == "http://localhost:14268"
        assert config.console_enabled is True

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        """Test from_env with no environment variables set"""
        config = TracingConfig.from_env()

        assert config.enabled is True
        assert config.service_name == "puffinflow"
        assert config.service_version == "1.0.0"
        assert config.sample_rate == 1.0
        assert config.otlp_endpoint is None
        assert config.jaeger_endpoint is None
        assert config.console_enabled is False

    @patch.dict(os.environ, {
        'TRACING_ENABLED': 'false',
        'SERVICE_NAME': 'test-app',
        'SERVICE_VERSION': '3.0.0',
        'TRACE_SAMPLE_RATE': '0.25',
        'OTLP_ENDPOINT': 'http://otlp:4317',
        'JAEGER_ENDPOINT': 'http://jaeger:14268',
        'TRACE_CONSOLE': 'true'
    })
    def test_from_env_with_values(self):
        """Test from_env with environment variables set"""
        config = TracingConfig.from_env()

        assert config.enabled is False
        assert config.service_name == "test-app"
        assert config.service_version == "3.0.0"
        assert config.sample_rate == 0.25
        assert config.otlp_endpoint == "http://otlp:4317"
        assert config.jaeger_endpoint == "http://jaeger:14268"
        assert config.console_enabled is True

    @patch.dict(os.environ, {'TRACING_ENABLED': 'TRUE'})
    def test_from_env_case_insensitive_bool(self):
        """Test boolean environment variable parsing is case insensitive"""
        config = TracingConfig.from_env()
        assert config.enabled is True

    @patch.dict(os.environ, {'TRACE_SAMPLE_RATE': '0.0'})
    def test_from_env_zero_sample_rate(self):
        """Test zero sample rate"""
        config = TracingConfig.from_env()
        assert config.sample_rate == 0.0


class TestMetricsConfig:
    """Test cases for MetricsConfig"""

    def test_default_values(self):
        """Test default configuration values"""
        config = MetricsConfig()

        assert config.enabled is True
        assert config.namespace == "puffinflow"
        assert config.prometheus_port == 9090
        assert config.prometheus_path == "/metrics"
        assert config.collection_interval == 15.0
        assert config.cardinality_limit == 10000

    def test_custom_values(self):
        """Test custom configuration values"""
        config = MetricsConfig(
            enabled=False,
            namespace="test-metrics",
            prometheus_port=8080,
            prometheus_path="/custom-metrics",
            collection_interval=30.0,
            cardinality_limit=5000
        )

        assert config.enabled is False
        assert config.namespace == "test-metrics"
        assert config.prometheus_port == 8080
        assert config.prometheus_path == "/custom-metrics"
        assert config.collection_interval == 30.0
        assert config.cardinality_limit == 5000

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        """Test from_env with no environment variables set"""
        config = MetricsConfig.from_env()

        assert config.enabled is True
        assert config.namespace == "puffinflow"
        assert config.prometheus_port == 9090
        assert config.prometheus_path == "/metrics"
        assert config.collection_interval == 15.0
        assert config.cardinality_limit == 10000

    @patch.dict(os.environ, {
        'METRICS_ENABLED': 'false',
        'METRICS_NAMESPACE': 'production-metrics',
        'METRICS_PORT': '8080',
        'METRICS_PATH': '/custom',
        'METRICS_INTERVAL': '60.0',
        'METRICS_CARDINALITY_LIMIT': '50000'
    })
    def test_from_env_with_values(self):
        """Test from_env with environment variables set"""
        config = MetricsConfig.from_env()

        assert config.enabled is False
        assert config.namespace == "production-metrics"
        assert config.prometheus_port == 8080
        assert config.prometheus_path == "/custom"
        assert config.collection_interval == 60.0
        assert config.cardinality_limit == 50000


class TestAlertingConfig:
    """Test cases for AlertingConfig"""

    def test_default_values(self):
        """Test default configuration values"""
        config = AlertingConfig()

        assert config.enabled is True
        assert config.evaluation_interval == 30.0
        assert config.webhook_urls == []
        assert config.email_recipients == []
        assert config.slack_webhook_url is None

    def test_custom_values(self):
        """Test custom configuration values"""
        webhook_urls = ["http://webhook1.com", "http://webhook2.com"]
        email_recipients = ["admin@example.com", "dev@example.com"]

        config = AlertingConfig(
            enabled=False,
            evaluation_interval=60.0,
            webhook_urls=webhook_urls,
            email_recipients=email_recipients,
            slack_webhook_url="http://slack.webhook"
        )

        assert config.enabled is False
        assert config.evaluation_interval == 60.0
        assert config.webhook_urls == webhook_urls
        assert config.email_recipients == email_recipients
        assert config.slack_webhook_url == "http://slack.webhook"

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        """Test from_env with no environment variables set"""
        config = AlertingConfig.from_env()

        assert config.enabled is True
        assert config.evaluation_interval == 30.0
        assert config.webhook_urls == []
        assert config.email_recipients == []
        assert config.slack_webhook_url is None

    @patch.dict(os.environ, {
        'ALERTING_ENABLED': 'false',
        'ALERT_EVALUATION_INTERVAL': '120.0',
        'ALERT_WEBHOOK_URLS': 'http://webhook1.com,http://webhook2.com',
        'ALERT_EMAIL_RECIPIENTS': 'admin@test.com,dev@test.com,ops@test.com',
        'ALERT_SLACK_WEBHOOK': 'http://slack.webhook.url'
    })
    def test_from_env_with_values(self):
        """Test from_env with environment variables set"""
        config = AlertingConfig.from_env()

        assert config.enabled is False
        assert config.evaluation_interval == 120.0
        assert config.webhook_urls == ["http://webhook1.com", "http://webhook2.com"]
        assert config.email_recipients == ["admin@test.com", "dev@test.com", "ops@test.com"]
        assert config.slack_webhook_url == "http://slack.webhook.url"

    @patch.dict(os.environ, {
        'ALERT_WEBHOOK_URLS': 'http://webhook1.com, , http://webhook2.com,',
        'ALERT_EMAIL_RECIPIENTS': 'admin@test.com, ,dev@test.com, '
    })
    def test_from_env_with_empty_list_items(self):
        """Test from_env handles empty list items and whitespace"""
        config = AlertingConfig.from_env()

        assert config.webhook_urls == ["http://webhook1.com", "http://webhook2.com"]
        assert config.email_recipients == ["admin@test.com", "dev@test.com"]

    @patch.dict(os.environ, {
        'ALERT_WEBHOOK_URLS': '',
        'ALERT_EMAIL_RECIPIENTS': ''
    })
    def test_from_env_with_empty_strings(self):
        """Test from_env handles empty string environment variables"""
        config = AlertingConfig.from_env()

        assert config.webhook_urls == []
        assert config.email_recipients == []


class TestEventsConfig:
    """Test cases for EventsConfig"""

    def test_default_values(self):
        """Test default configuration values"""
        config = EventsConfig()

        assert config.enabled is True
        assert config.buffer_size == 1000
        assert config.batch_size == 100
        assert config.flush_interval == 5.0

    def test_custom_values(self):
        """Test custom configuration values"""
        config = EventsConfig(
            enabled=False,
            buffer_size=2000,
            batch_size=200,
            flush_interval=10.0
        )

        assert config.enabled is False
        assert config.buffer_size == 2000
        assert config.batch_size == 200
        assert config.flush_interval == 10.0

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        """Test from_env with no environment variables set"""
        config = EventsConfig.from_env()

        assert config.enabled is True
        assert config.buffer_size == 1000
        assert config.batch_size == 100
        assert config.flush_interval == 5.0

    @patch.dict(os.environ, {
        'EVENTS_ENABLED': 'false',
        'EVENT_BUFFER_SIZE': '5000',
        'EVENT_BATCH_SIZE': '500',
        'EVENT_FLUSH_INTERVAL': '15.0'
    })
    def test_from_env_with_values(self):
        """Test from_env with environment variables set"""
        config = EventsConfig.from_env()

        assert config.enabled is False
        assert config.buffer_size == 5000
        assert config.batch_size == 500
        assert config.flush_interval == 15.0


class TestObservabilityConfig:
    """Test cases for ObservabilityConfig"""

    def test_default_values(self):
        """Test default configuration values"""
        config = ObservabilityConfig()

        assert config.enabled is True
        assert config.environment == "development"
        assert isinstance(config.tracing, TracingConfig)
        assert isinstance(config.metrics, MetricsConfig)
        assert isinstance(config.alerting, AlertingConfig)
        assert isinstance(config.events, EventsConfig)

    def test_custom_values(self):
        """Test custom configuration values"""
        tracing = TracingConfig(enabled=False)
        metrics = MetricsConfig(prometheus_port=8080)
        alerting = AlertingConfig(evaluation_interval=60.0)
        events = EventsConfig(buffer_size=2000)

        config = ObservabilityConfig(
            enabled=False,
            environment="production",
            tracing=tracing,
            metrics=metrics,
            alerting=alerting,
            events=events
        )

        assert config.enabled is False
        assert config.environment == "production"
        assert config.tracing == tracing
        assert config.metrics == metrics
        assert config.alerting == alerting
        assert config.events == events

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        """Test from_env with no environment variables set"""
        config = ObservabilityConfig.from_env()

        assert config.enabled is True
        assert config.environment == "development"
        assert isinstance(config.tracing, TracingConfig)
        assert isinstance(config.metrics, MetricsConfig)
        assert isinstance(config.alerting, AlertingConfig)
        assert isinstance(config.events, EventsConfig)

    @patch.dict(os.environ, {
        'OBSERVABILITY_ENABLED': 'false',
        'ENVIRONMENT': 'production',
        'TRACING_ENABLED': 'false',
        'METRICS_PORT': '8080',
        'ALERT_EVALUATION_INTERVAL': '120.0',
        'EVENT_BUFFER_SIZE': '5000'
    })
    def test_from_env_with_values(self):
        """Test from_env with environment variables affecting all sub-configs"""
        config = ObservabilityConfig.from_env()

        assert config.enabled is False
        assert config.environment == "production"
        assert config.tracing.enabled is False
        assert config.metrics.prometheus_port == 8080
        assert config.alerting.evaluation_interval == 120.0
        assert config.events.buffer_size == 5000

    @patch.dict(os.environ, {
        'ENVIRONMENT': 'staging',
        'SERVICE_NAME': 'staging-puffinflow',
        'METRICS_NAMESPACE': 'staging-metrics',
        'ALERT_WEBHOOK_URLS': 'http://staging.webhook.com',
        'ALERT_EMAIL_RECIPIENTS': 'staging-team@example.com'
    })
    def test_from_env_staging_environment(self):
        """Test complete staging environment configuration"""
        config = ObservabilityConfig.from_env()

        assert config.environment == "staging"
        assert config.tracing.service_name == "staging-puffinflow"
        assert config.metrics.namespace == "staging-metrics"
        assert config.alerting.webhook_urls == ["http://staging.webhook.com"]
        assert config.alerting.email_recipients == ["staging-team@example.com"]


class TestConfigIntegration:
    """Integration tests for configuration classes"""

    def test_all_configs_can_be_instantiated(self):
        """Test that all configuration classes can be instantiated without errors"""
        tracing = TracingConfig()
        metrics = MetricsConfig()
        alerting = AlertingConfig()
        events = EventsConfig()
        observability = ObservabilityConfig()

        assert all([
            isinstance(tracing, TracingConfig),
            isinstance(metrics, MetricsConfig),
            isinstance(alerting, AlertingConfig),
            isinstance(events, EventsConfig),
            isinstance(observability, ObservabilityConfig)
        ])

    @patch.dict(os.environ, {
        'OBSERVABILITY_ENABLED': 'true',
        'ENVIRONMENT': 'test',
        'TRACING_ENABLED': 'true',
        'SERVICE_NAME': 'test-service',
        'METRICS_ENABLED': 'true',
        'METRICS_PORT': '9091',
        'ALERTING_ENABLED': 'true',
        'ALERT_WEBHOOK_URLS': 'http://test.webhook.com',
        'EVENTS_ENABLED': 'true',
        'EVENT_BUFFER_SIZE': '500'
    })
    def test_complete_environment_configuration(self):
        """Test complete configuration from environment variables"""
        config = ObservabilityConfig.from_env()

        # Main config
        assert config.enabled is True
        assert config.environment == "test"

        # Tracing config
        assert config.tracing.enabled is True
        assert config.tracing.service_name == "test-service"

        # Metrics config
        assert config.metrics.enabled is True
        assert config.metrics.prometheus_port == 9091

        # Alerting config
        assert config.alerting.enabled is True
        assert config.alerting.webhook_urls == ["http://test.webhook.com"]

        # Events config
        assert config.events.enabled is True
        assert config.events.buffer_size == 500

    def test_config_hierarchy_consistency(self):
        """Test that nested configs maintain proper relationships"""
        config = ObservabilityConfig.from_env()

        # All sub-configs should be properly instantiated
        assert hasattr(config, 'tracing')
        assert hasattr(config, 'metrics')
        assert hasattr(config, 'alerting')
        assert hasattr(config, 'events')

        # Sub-configs should be of correct types
        assert isinstance(config.tracing, TracingConfig)
        assert isinstance(config.metrics, MetricsConfig)
        assert isinstance(config.alerting, AlertingConfig)
        assert isinstance(config.events, EventsConfig)


# Test fixtures and parametrized tests for edge cases
class TestEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.parametrize("bool_value,expected", [
        ("true", True),
        ("TRUE", True),
        ("True", True),
        ("false", False),
        ("FALSE", False),
        ("False", False),
        ("1", False),  # Only 'true' should be True
        ("0", False),
        ("yes", False),
        ("no", False),
        ("", False)
    ])
    def test_boolean_parsing(self, bool_value, expected):
        """Test boolean environment variable parsing"""
        with patch.dict(os.environ, {'TRACING_ENABLED': bool_value}):
            config = TracingConfig.from_env()
            assert config.enabled is expected

    @pytest.mark.parametrize("numeric_value,config_class,attribute", [
        ("1.5", TracingConfig, "sample_rate"),
        ("0.0", TracingConfig, "sample_rate"),
        ("9090", MetricsConfig, "prometheus_port"),
        ("30.5", AlertingConfig, "evaluation_interval"),
        ("1000", EventsConfig, "buffer_size")
    ])
    def test_numeric_parsing(self, numeric_value, config_class, attribute):
        """Test numeric environment variable parsing"""
        env_var_map = {
            (TracingConfig, "sample_rate"): "TRACE_SAMPLE_RATE",
            (MetricsConfig, "prometheus_port"): "METRICS_PORT",
            (AlertingConfig, "evaluation_interval"): "ALERT_EVALUATION_INTERVAL",
            (EventsConfig, "buffer_size"): "EVENT_BUFFER_SIZE"
        }

        env_var = env_var_map[(config_class, attribute)]
        with patch.dict(os.environ, {env_var: numeric_value}):
            config = config_class.from_env()
            actual_value = getattr(config, attribute)
            expected_value = float(numeric_value) if '.' in numeric_value else int(numeric_value)
            assert actual_value == expected_value

    def test_list_parsing_with_various_separators(self):
        """Test list parsing handles various comma-separated formats"""
        test_cases = [
            ("a,b,c", ["a", "b", "c"]),
            ("a, b, c", ["a", "b", "c"]),
            ("a ,b ,c", ["a", "b", "c"]),
            ("a,,c", ["a", "c"]),
            (",a,b,", ["a", "b"]),
            ("", [])
        ]

        for input_str, expected in test_cases:
            with patch.dict(os.environ, {'ALERT_WEBHOOK_URLS': input_str}):
                config = AlertingConfig.from_env()
                assert config.webhook_urls == expected