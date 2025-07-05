import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from ..agent.context import Context
from .core import ObservabilityManager
from .interfaces import ObservabilityEvent, SpanType


class ObservableContext(Context):
    """Context with observability integration"""

    def __init__(self, shared_state: Dict[str, Any],
                 observability: Optional[ObservabilityManager] = None):
        super().__init__(shared_state)
        self._observability = observability

    @contextmanager
    def trace(self, name: str, **attributes):
        """Create trace span with context"""
        if not self._observability or not self._observability.tracing:
            yield None
            return

        # Add context attributes
        context_attrs = {
            "workflow_id": self.get_variable("workflow_id"),
            "agent_name": self.get_variable("agent_name"),
            "state_name": self.get_variable("current_state"),
            **attributes
        }

        with self._observability.tracing.span(name, SpanType.BUSINESS, **context_attrs) as span:
            yield span

    def metric(self, name: str, value: float, **labels):
        """Record metric with context"""
        if not self._observability or not self._observability.metrics:
            return

        context_labels = {
            "workflow_id": self.get_variable("workflow_id", "unknown"),
            "agent_name": self.get_variable("agent_name", "unknown"),
            **labels
        }

        histogram = self._observability.histogram(name, labels=list(context_labels.keys()))
        if histogram:
            histogram.record(value, **context_labels)

    def log(self, level: str, message: str, **kwargs):
        """Log with observability"""
        if not self._observability or not self._observability.events:
            return

        event = ObservabilityEvent(
            timestamp=time.time(),
            event_type="log",
            source="context",
            level=level.upper(),
            message=message,
            attributes={
                "workflow_id": self.get_variable("workflow_id"),
                "agent_name": self.get_variable("agent_name"),
                "state_name": self.get_variable("current_state"),
                **kwargs
            }
        )

        import asyncio
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._observability.events.process_event(event))
        except RuntimeError:
            pass
