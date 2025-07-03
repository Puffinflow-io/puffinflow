import asyncio
import threading
from typing import Optional, Dict, Any
from contextlib import contextmanager

from .interfaces import TracingProvider, MetricsProvider, AlertingProvider, EventProcessor
from .config import ObservabilityConfig
from .tracing import OpenTelemetryTracingProvider
from .metrics import PrometheusMetricsProvider
from .alerting import WebhookAlerting
from .events import BufferedEventProcessor


class ObservabilityManager:
    """Main observability coordinator"""

    def __init__(self, config: Optional[ObservabilityConfig] = None):
        self.config = config or ObservabilityConfig()
        self._initialized = False
        self._lock = threading.Lock()

        # Initialize providers
        self._tracing: Optional[TracingProvider] = None
        self._metrics: Optional[MetricsProvider] = None
        self._alerting: Optional[AlertingProvider] = None
        self._events: Optional[EventProcessor] = None

    @property
    def tracing(self) -> Optional[TracingProvider]:
        return self._tracing

    @property
    def metrics(self) -> Optional[MetricsProvider]:
        return self._metrics

    @property
    def alerting(self) -> Optional[AlertingProvider]:
        return self._alerting

    @property
    def events(self) -> Optional[EventProcessor]:
        return self._events

    async def initialize(self) -> None:
        """Initialize all components"""
        with self._lock:
            if self._initialized:
                return

            # Initialize tracing
            if self.config.tracing.enabled:
                self._tracing = OpenTelemetryTracingProvider(self.config.tracing)

            # Initialize metrics
            if self.config.metrics.enabled:
                self._metrics = PrometheusMetricsProvider(self.config.metrics)

            # Initialize alerting
            if self.config.alerting.enabled:
                self._alerting = WebhookAlerting(self.config.alerting)

            # Initialize events
            if self.config.events.enabled:
                self._events = BufferedEventProcessor(self.config.events)
                await self._events.initialize()

            self._initialized = True

    async def shutdown(self) -> None:
        """Shutdown all components"""
        if self._events:
            await self._events.shutdown()
        self._initialized = False

    # Convenience methods
    @contextmanager
    def trace(self, name: str, **attributes):
        """Create trace span"""
        if self._tracing:
            with self._tracing.span(name, **attributes) as span:
                yield span
        else:
            yield None

    def counter(self, name: str, description: str = "", labels=None):
        """Create counter metric"""
        if self._metrics:
            return self._metrics.counter(name, description, labels)
        return None

    def gauge(self, name: str, description: str = "", labels=None):
        """Create gauge metric"""
        if self._metrics:
            return self._metrics.gauge(name, description, labels)
        return None

    def histogram(self, name: str, description: str = "", labels=None):
        """Create histogram metric"""
        if self._metrics:
            return self._metrics.histogram(name, description, labels)
        return None

    async def alert(self, message: str, severity: str = "warning", **attributes):
        """Send alert"""
        if self._alerting:
            from .interfaces import AlertSeverity
            sev = AlertSeverity(severity.lower())
            await self._alerting.send_alert(message, sev, attributes)


# Global instance management
_global_observability: Optional[ObservabilityManager] = None
_lock = threading.Lock()


def get_observability() -> ObservabilityManager:
    """Get global observability instance"""
    global _global_observability
    with _lock:
        if _global_observability is None:
            _global_observability = ObservabilityManager()
        return _global_observability


async def setup_observability(config: Optional[ObservabilityConfig] = None) -> ObservabilityManager:
    """Setup and initialize observability"""
    global _global_observability
    with _lock:
        _global_observability = ObservabilityManager(config)
    await _global_observability.initialize()
    return _global_observability