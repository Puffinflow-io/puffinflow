import os
from unittest.mock import patch

from puffinflow.core.observability.config import (
    AlertingConfig,
    EventsConfig,
    MetricsConfig,
    ObservabilityConfig,
    TracingConfig,
)


class TestTracingConfig:
    """Test TracingConfig class"""

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
            console_enabled=True,
        )

        assert config.enabled is False
        assert config.service_name == "test-service"
        assert config.service_version == "2.0.0"
        assert config.sample_rate == 0.5
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.jaeger_endpoint == "http://localhost:14268"
        assert config.console_enabled is True

    def test_from_env_defaults(self):
        """Test from_env with default values"""
        with patch.dict(os.environ, {}, clear=True):
            config = TracingConfig.from_env()

            assert config.enabled is True
            assert config.service_name == "puffinflow"
            assert config.service_version == "1.0.0"
            assert config.sample_rate == 1.0
            assert config.otlp_endpoint is None
            assert config.jaeger_endpoint is None
            assert config.console_enabled is False

    def test_from_env_custom(self):
        """Test from_env with custom environment variables"""
        env_vars = {
            "TRACING_ENABLED": "false",
            "SERVICE_NAME": "custom-service",
            "SERVICE_VERSION": "3.0.0",
            "TRACE_SAMPLE_RATE": "0.25",
            "OTLP_ENDPOINT": "http://custom:4317",
            "JAEGER_ENDPOINT": "http://custom:14268",
            "TRACE_CONSOLE": "true",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = TracingConfig.from_env()

            assert config.enabled is False
            assert config.service_name == "custom-service"
            assert config.service_version == "3.0.0"
            assert config.sample_rate == 0.25
            assert config.otlp_endpoint == "http://custom:4317"
            assert config.jaeger_endpoint == "http://custom:14268"
            assert config.console_enabled is True


class TestMetricsConfig:
    """Test MetricsConfig class"""

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
            namespace="test-namespace",
            prometheus_port=8080,
            prometheus_path="/custom-metrics",
            collection_interval=30.0,
            cardinality_limit=5000,
        )

        assert config.enabled is False
        assert config.namespace == "test-namespace"
        assert config.prometheus_port == 8080
        assert config.prometheus_path == "/custom-metrics"
        assert config.collection_interval == 30.0
        assert config.cardinality_limit == 5000

    def test_from_env_defaults(self):
        """Test from_env with default values"""
        with patch.dict(os.environ, {}, clear=True):
            config = MetricsConfig.from_env()

            assert config.enabled is True
            assert config.namespace == "puffinflow"
            assert config.prometheus_port == 9090
            assert config.prometheus_path == "/metrics"
            assert config.collection_interval == 15.0
            assert config.cardinality_limit == 10000

    def test_from_env_custom(self):
        """Test from_env with custom environment variables"""
        env_vars = {
            "METRICS_ENABLED": "false",
            "METRICS_NAMESPACE": "custom-namespace",
            "METRICS_PORT": "8080",
            "METRICS_PATH": "/custom-metrics",
            "METRICS_INTERVAL": "30.0",
            "METRICS_CARDINALITY_LIMIT": "5000",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = MetricsConfig.from_env()

            assert config.enabled is False
            assert config.namespace == "custom-namespace"
            assert config.prometheus_port == 8080
            assert config.prometheus_path == "/custom-metrics"
            assert config.collection_interval == 30.0
            assert config.cardinality_limit == 5000


class TestAlertingConfig:
    """Test AlertingConfig class"""

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
        config = AlertingConfig(
            enabled=False,
            evaluation_interval=60.0,
            webhook_urls=["http://webhook1.com", "http://webhook2.com"],
            email_recipients=["user1@example.com", "user2@example.com"],
            slack_webhook_url="http://slack-webhook.com",
        )

        assert config.enabled is False
        assert config.evaluation_interval == 60.0
        assert config.webhook_urls == ["http://webhook1.com", "http://webhook2.com"]
        assert config.email_recipients == ["user1@example.com", "user2@example.com"]
        assert config.slack_webhook_url == "http://slack-webhook.com"

    def test_from_env_defaults(self):
        """Test from_env with default values"""
        with patch.dict(os.environ, {}, clear=True):
            config = AlertingConfig.from_env()

            assert config.enabled is True
            assert config.evaluation_interval == 30.0
            assert config.webhook_urls == []
            assert config.email_recipients == []
            assert config.slack_webhook_url is None

    def test_from_env_custom(self):
        """Test from_env with custom environment variables"""
        env_vars = {
            "ALERTING_ENABLED": "false",
            "ALERT_EVALUATION_INTERVAL": "60.0",
            "ALERT_WEBHOOK_URLS": "http://webhook1.com,http://webhook2.com",
            "ALERT_EMAIL_RECIPIENTS": "user1@example.com,user2@example.com",
            "ALERT_SLACK_WEBHOOK": "http://slack-webhook.com",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = AlertingConfig.from_env()

            assert config.enabled is False
            assert config.evaluation_interval == 60.0
            assert config.webhook_urls == ["http://webhook1.com", "http://webhook2.com"]
            assert config.email_recipients == ["user1@example.com", "user2@example.com"]
            assert config.slack_webhook_url == "http://slack-webhook.com"

    def test_from_env_empty_lists(self):
        """Test from_env with empty webhook and email lists"""
        env_vars = {"ALERT_WEBHOOK_URLS": "", "ALERT_EMAIL_RECIPIENTS": ""}

        with patch.dict(os.environ, env_vars, clear=True):
            config = AlertingConfig.from_env()

            assert config.webhook_urls == []
            assert config.email_recipients == []

    def test_from_env_whitespace_handling(self):
        """Test from_env handles whitespace in lists"""
        env_vars = {
            "ALERT_WEBHOOK_URLS": " http://webhook1.com , http://webhook2.com ",
            "ALERT_EMAIL_RECIPIENTS": " user1@example.com , user2@example.com ",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = AlertingConfig.from_env()

            assert config.webhook_urls == ["http://webhook1.com", "http://webhook2.com"]
            assert config.email_recipients == ["user1@example.com", "user2@example.com"]


class TestEventsConfig:
    """Test EventsConfig class"""

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
            enabled=False, buffer_size=500, batch_size=50, flush_interval=10.0
        )

        assert config.enabled is False
        assert config.buffer_size == 500
        assert config.batch_size == 50
        assert config.flush_interval == 10.0

    def test_from_env_defaults(self):
        """Test from_env with default values"""
        with patch.dict(os.environ, {}, clear=True):
            config = EventsConfig.from_env()

            assert config.enabled is True
            assert config.buffer_size == 1000
            assert config.batch_size == 100
            assert config.flush_interval == 5.0

    def test_from_env_custom(self):
        """Test from_env with custom environment variables"""
        env_vars = {
            "EVENTS_ENABLED": "false",
            "EVENT_BUFFER_SIZE": "500",
            "EVENT_BATCH_SIZE": "50",
            "EVENT_FLUSH_INTERVAL": "10.0",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = EventsConfig.from_env()

            assert config.enabled is False
            assert config.buffer_size == 500
            assert config.batch_size == 50
            assert config.flush_interval == 10.0


class TestObservabilityConfig:
    """Test ObservabilityConfig class"""

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
        metrics = MetricsConfig(enabled=False)
        alerting = AlertingConfig(enabled=False)
        events = EventsConfig(enabled=False)

        config = ObservabilityConfig(
            enabled=False,
            environment="production",
            tracing=tracing,
            metrics=metrics,
            alerting=alerting,
            events=events,
        )

        assert config.enabled is False
        assert config.environment == "production"
        assert config.tracing is tracing
        assert config.metrics is metrics
        assert config.alerting is alerting
        assert config.events is events

    def test_from_env_defaults(self):
        """Test from_env with default values"""
        with patch.dict(os.environ, {}, clear=True):
            config = ObservabilityConfig.from_env()

            assert config.enabled is True
            assert config.environment == "development"
            assert isinstance(config.tracing, TracingConfig)
            assert isinstance(config.metrics, MetricsConfig)
            assert isinstance(config.alerting, AlertingConfig)
            assert isinstance(config.events, EventsConfig)

    def test_from_env_custom(self):
        """Test from_env with custom environment variables"""
        env_vars = {
            "OBSERVABILITY_ENABLED": "false",
            "ENVIRONMENT": "production",
            "TRACING_ENABLED": "false",
            "METRICS_ENABLED": "false",
            "ALERTING_ENABLED": "false",
            "EVENTS_ENABLED": "false",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config = ObservabilityConfig.from_env()

            assert config.enabled is False
            assert config.environment == "production"
            assert config.tracing.enabled is False
            assert config.metrics.enabled is False
            assert config.alerting.enabled is False
            assert config.events.enabled is False
