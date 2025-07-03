import functools
import time
import asyncio
from typing import Callable, Optional, Dict, Any

from .core import get_observability
from .interfaces import SpanType


def observe(name: Optional[str] = None, span_type: SpanType = SpanType.BUSINESS, **span_attributes):
    """Observability decorator"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            observability = get_observability()
            operation_name = name or f"{func.__module__}.{func.__name__}"

            if observability.tracing:
                with observability.tracing.span(
                        operation_name,
                        span_type,
                        function=func.__name__,
                        **span_attributes
                ) as span:
                    start_time = time.time()
                    try:
                        result = await func(*args, **kwargs)
                        duration = time.time() - start_time

                        if span:
                            span.set_attribute("function.duration", duration)
                            span.set_status("ok")

                        return result
                    except Exception as e:
                        if span:
                            span.record_exception(e)
                        raise
            else:
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            observability = get_observability()
            operation_name = name or f"{func.__module__}.{func.__name__}"

            if observability.tracing:
                with observability.tracing.span(operation_name, span_type, **span_attributes) as span:
                    try:
                        result = func(*args, **kwargs)
                        if span:
                            span.set_status("ok")
                        return result
                    except Exception as e:
                        if span:
                            span.record_exception(e)
                        raise
            else:
                return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def trace_state(span_type: SpanType = SpanType.STATE, **span_attributes):
    """Decorator for tracing PuffinFlow states"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(context, *args, **kwargs):
            observability = get_observability()

            if observability.tracing:
                attrs = {
                    "state.name": func.__name__,
                    "workflow.id": context.get_variable("workflow_id"),
                    "agent.name": context.get_variable("agent_name"),
                    **span_attributes
                }

                with observability.tracing.span(f"state.{func.__name__}", span_type, **attrs) as span:
                    try:
                        result = await func(context, *args, **kwargs)
                        if span:
                            span.set_status("ok")
                        return result
                    except Exception as e:
                        if span:
                            span.record_exception(e)
                        raise
            else:
                return await func(context, *args, **kwargs)

        return wrapper

    return decorator