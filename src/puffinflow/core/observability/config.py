import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TracingConfig:
    """Tracing configuration"""

    enabled: bool = True
    service_name: str = "puffinflow"
    service_version: str = "1.0.0"
    sample_rate: float = 1.0
    otlp_endpoint: Optional[str] = None
    jaeger_endpoint: Optional[str] = None
    console_enabled: bool = False

    @classmethod
    def from_env(cls) -> "TracingConfig":
        return cls(
            enabled=os.getenv("TRACING_ENABLED", "true").lower() == "true",
            service_name=os.getenv("SERVICE_NAME", "puffinflow"),
            service_version=os.getenv("SERVICE_VERSION", "1.0.0"),
            sample_rate=float(os.getenv("TRACE_SAMPLE_RATE", "1.0")),
            otlp_endpoint=os.getenv("OTLP_ENDPOINT"),
            jaeger_endpoint=os.getenv("JAEGER_ENDPOINT"),
            console_enabled=os.getenv("TRACE_CONSOLE", "false").lower() == "true",
        )


@dataclass
class MetricsConfig:
    """Metrics configuration"""

    enabled: bool = True
    namespace: str = "puffinflow"
    prometheus_port: int = 9090
    prometheus_path: str = "/metrics"
    collection_interval: float = 15.0
    cardinality_limit: int = 10000

    @classmethod
    def from_env(cls) -> "MetricsConfig":
        return cls(
            enabled=os.getenv("METRICS_ENABLED", "true").lower() == "true",
            namespace=os.getenv("METRICS_NAMESPACE", "puffinflow"),
            prometheus_port=int(os.getenv("METRICS_PORT", "9090")),
            prometheus_path=os.getenv("METRICS_PATH", "/metrics"),
            collection_interval=float(os.getenv("METRICS_INTERVAL", "15.0")),
            cardinality_limit=int(os.getenv("METRICS_CARDINALITY_LIMIT", "10000")),
        )


@dataclass
class AlertingConfig:
    """Alerting configuration"""

    enabled: bool = True
    evaluation_interval: float = 30.0
    webhook_urls: list[str] = field(default_factory=list)
    email_recipients: list[str] = field(default_factory=list)
    slack_webhook_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "AlertingConfig":
        webhook_urls = (
            os.getenv("ALERT_WEBHOOK_URLS", "").split(",")
            if os.getenv("ALERT_WEBHOOK_URLS")
            else []
        )
        email_recipients = (
            os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(",")
            if os.getenv("ALERT_EMAIL_RECIPIENTS")
            else []
        )

        return cls(
            enabled=os.getenv("ALERTING_ENABLED", "true").lower() == "true",
            evaluation_interval=float(os.getenv("ALERT_EVALUATION_INTERVAL", "30.0")),
            webhook_urls=[url.strip() for url in webhook_urls if url.strip()],
            email_recipients=[
                email.strip() for email in email_recipients if email.strip()
            ],
            slack_webhook_url=os.getenv("ALERT_SLACK_WEBHOOK"),
        )


@dataclass
class EventsConfig:
    """Events configuration"""

    enabled: bool = True
    buffer_size: int = 1000
    batch_size: int = 100
    flush_interval: float = 5.0

    @classmethod
    def from_env(cls) -> "EventsConfig":
        return cls(
            enabled=os.getenv("EVENTS_ENABLED", "true").lower() == "true",
            buffer_size=int(os.getenv("EVENT_BUFFER_SIZE", "1000")),
            batch_size=int(os.getenv("EVENT_BATCH_SIZE", "100")),
            flush_interval=float(os.getenv("EVENT_FLUSH_INTERVAL", "5.0")),
        )


@dataclass
class ObservabilityConfig:
    """Complete observability configuration"""

    enabled: bool = True
    environment: str = "development"
    tracing: TracingConfig = field(default_factory=TracingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    events: EventsConfig = field(default_factory=EventsConfig)

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        return cls(
            enabled=os.getenv("OBSERVABILITY_ENABLED", "true").lower() == "true",
            environment=os.getenv("ENVIRONMENT", "development"),
            tracing=TracingConfig.from_env(),
            metrics=MetricsConfig.from_env(),
            alerting=AlertingConfig.from_env(),
            events=EventsConfig.from_env(),
        )
