import time
from typing import Any, Optional

from ..agent.base import Agent, AgentResult
from .context import ObservableContext
from .core import ObservabilityManager
from .interfaces import SpanType


class ObservableAgent(Agent):
    """Agent with observability"""

    def __init__(
        self,
        name: str,
        observability: Optional[ObservabilityManager] = None,
        **kwargs: Any,
    ) -> None:
        # Extract workflow_id before passing kwargs to parent
        self.workflow_id = kwargs.pop("workflow_id", f"workflow_{int(time.time())}")

        # Set observability BEFORE calling parent __init__
        self._observability = observability

        # Initialize cleanup handlers before calling parent __init__
        self._cleanup_handlers = []

        super().__init__(name, **kwargs)

        # Store any additional attributes that weren't handled by the parent
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

        # Setup basic metrics
        if self._observability and self._observability.metrics:
            self.workflow_duration = self._observability.metrics.histogram(
                "workflow_duration_seconds",
                "Workflow execution duration",
                ["agent_name", "status"],
            )

            self.state_duration = self._observability.metrics.histogram(
                "state_execution_duration_seconds",
                "State execution duration",
                ["agent_name", "state_name", "status"],
            )

    def _create_context(self, shared_state: dict[str, Any]) -> ObservableContext:
        """Create observable context"""
        context = ObservableContext(shared_state, self._observability)
        context.set_variable("agent_name", self.name)
        context.set_variable("workflow_id", self.workflow_id)
        return context

    async def run(self, timeout: Optional[float] = None) -> AgentResult:
        """Run workflow with observability"""
        workflow_start = time.time()

        if self._observability and self._observability.tracing:
            with self._observability.tracing.span(
                f"workflow.{self.name}",
                SpanType.WORKFLOW,
                agent_name=self.name,
                workflow_id=self.workflow_id,
            ) as span:
                try:
                    result = await super().run(timeout)

                    duration = time.time() - workflow_start
                    if span:
                        span.set_attribute("workflow.duration", duration)
                        span.set_status("ok")

                    if self.workflow_duration:
                        self.workflow_duration.record(
                            duration, agent_name=self.name, status="success"
                        )

                    return result

                except Exception as e:
                    duration = time.time() - workflow_start
                    if span:
                        span.record_exception(e)

                    if self.workflow_duration:
                        self.workflow_duration.record(
                            duration, agent_name=self.name, status="error"
                        )
                    raise
        else:
            return await super().run(timeout)

    async def run_state(self, state_name: str) -> None:
        """Run state with observability"""
        state_start = time.time()

        if self._observability and self._observability.tracing:
            with self._observability.tracing.span(
                f"state.{state_name}",
                SpanType.STATE,
                agent_name=self.name,
                state_name=state_name,
            ) as span:
                try:
                    context = self._create_context(self.shared_state)
                    context.set_variable("current_state", state_name)

                    await self.states[state_name](context)

                    duration = time.time() - state_start
                    if span:
                        span.set_attribute("state.duration", duration)
                        span.set_status("ok")

                    if self.state_duration:
                        self.state_duration.record(
                            duration,
                            agent_name=self.name,
                            state_name=state_name,
                            status="success",
                        )

                except Exception as e:
                    duration = time.time() - state_start
                    if span:
                        span.record_exception(e)

                    if self.state_duration:
                        self.state_duration.record(
                            duration,
                            agent_name=self.name,
                            state_name=state_name,
                            status="error",
                        )
                    raise
        else:
            await super().run_state(state_name)
