import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class SpanType(Enum):
    """Types of spans for categorization"""

    WORKFLOW = "workflow"
    STATE = "state"
    RESOURCE = "resource"
    BUSINESS = "business"
    SYSTEM = "system"


class MetricType(Enum):
    """Types of metrics"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SpanContext:
    """Correlation context for distributed tracing"""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    workflow_id: Optional[str] = None
    agent_name: Optional[str] = None
    state_name: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    def child_context(self) -> "SpanContext":
        """Create child span context"""
        return SpanContext(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=self.span_id,
            workflow_id=self.workflow_id,
            agent_name=self.agent_name,
            state_name=self.state_name,
            user_id=self.user_id,
            session_id=self.session_id,
        )


@dataclass
class ObservabilityEvent:
    """Structured observability event"""

    timestamp: datetime
    event_type: str
    source: str
    level: str
    message: str
    attributes: dict[str, Any] = field(default_factory=dict)
    span_context: Optional[SpanContext] = None


class Span(ABC):
    """Abstract span interface"""

    @abstractmethod
    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute"""

    @abstractmethod
    def set_status(self, status: str, description: Optional[str] = None) -> None:
        """Set span status"""

    @abstractmethod
    def add_event(self, name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """Add event to span"""

    @abstractmethod
    def record_exception(self, exception: Exception) -> None:
        """Record exception in span"""

    @abstractmethod
    def end(self) -> None:
        """End the span"""

    @property
    @abstractmethod
    def context(self) -> SpanContext:
        """Get span context"""


class TracingProvider(ABC):
    """Abstract tracing provider"""

    @abstractmethod
    def start_span(
        self,
        name: str,
        span_type: SpanType = SpanType.SYSTEM,
        parent: Optional[SpanContext] = None,
        **attributes: Any,
    ) -> Span:
        """Start a new span"""

    @abstractmethod
    def get_current_span(self) -> Optional[Span]:
        """Get current active span"""

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType = SpanType.SYSTEM,
        parent: Optional[SpanContext] = None,
        **attributes: Any,
    ) -> Iterator[Span]:
        """Context manager for spans"""
        span = self.start_span(name, span_type, parent, **attributes)
        try:
            yield span
            span.set_status("ok")
        except Exception as e:
            span.record_exception(e)
            raise
        finally:
            span.end()


class Metric(ABC):
    """Abstract metric interface"""

    @abstractmethod
    def record(self, value: float, **labels: Any) -> None:
        """Record metric value"""


class MetricsProvider(ABC):
    """Abstract metrics provider"""

    @abstractmethod
    def counter(
        self, name: str, description: str = "", labels: Optional[list[str]] = None
    ) -> Metric:
        """Create counter metric"""

    @abstractmethod
    def gauge(
        self, name: str, description: str = "", labels: Optional[list[str]] = None
    ) -> Metric:
        """Create gauge metric"""

    @abstractmethod
    def histogram(
        self, name: str, description: str = "", labels: Optional[list[str]] = None
    ) -> Metric:
        """Create histogram metric"""

    @abstractmethod
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""


class AlertingProvider(ABC):
    """Abstract alerting provider"""

    @abstractmethod
    async def send_alert(
        self,
        message: str,
        severity: AlertSeverity,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        """Send alert"""


class EventProcessor(ABC):
    """Abstract event processor"""

    @abstractmethod
    async def process_event(self, event: ObservabilityEvent) -> None:
        """Process observability event"""
