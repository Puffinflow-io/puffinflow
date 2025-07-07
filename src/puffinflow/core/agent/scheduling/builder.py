"""Fluent API builder for agent scheduling."""

from typing import TYPE_CHECKING, Any

from .inputs import ScheduledInput, parse_inputs

if TYPE_CHECKING:
    from ..base import Agent
    from .scheduler import ScheduledAgent


class ScheduleBuilder:
    """Fluent API builder for scheduling agents."""

    def __init__(self, agent: "Agent", schedule_string: str):
        self._agent = agent
        self._schedule_string = schedule_string
        self._inputs: dict[str, ScheduledInput] = {}

    def with_inputs(self, **inputs: Any) -> "ScheduleBuilder":
        """Add regular variable inputs.

        Args:
            **inputs: Input key-value pairs

        Returns:
            Self for chaining
        """
        parsed = parse_inputs(**inputs)
        self._inputs.update(parsed)
        return self

    def with_secrets(self, **secrets: Any) -> "ScheduleBuilder":
        """Add secret inputs.

        Args:
            **secrets: Secret key-value pairs

        Returns:
            Self for chaining
        """
        for key, value in secrets.items():
            prefixed_value = f"secret:{value}"
            parsed = parse_inputs(**{key: prefixed_value})
            self._inputs.update(parsed)
        return self

    def with_constants(self, **constants: Any) -> "ScheduleBuilder":
        """Add constant inputs.

        Args:
            **constants: Constant key-value pairs

        Returns:
            Self for chaining
        """
        for key, value in constants.items():
            prefixed_value = f"const:{value}"
            parsed = parse_inputs(**{key: prefixed_value})
            self._inputs.update(parsed)
        return self

    def with_cache(self, ttl: int, **cached_inputs: Any) -> "ScheduleBuilder":
        """Add cached inputs with TTL.

        Args:
            ttl: Time to live in seconds
            **cached_inputs: Cached input key-value pairs

        Returns:
            Self for chaining
        """
        for key, value in cached_inputs.items():
            # Convert value to string for cache prefix
            if isinstance(value, (dict, list)):
                import json

                value_str = json.dumps(value)
            else:
                value_str = str(value)
            prefixed_value = f"cache:{ttl}:{value_str}"
            parsed = parse_inputs(**{key: prefixed_value})
            self._inputs.update(parsed)
        return self

    def with_typed(self, **typed_inputs: Any) -> "ScheduleBuilder":
        """Add typed inputs.

        Args:
            **typed_inputs: Typed input key-value pairs

        Returns:
            Self for chaining
        """
        for key, value in typed_inputs.items():
            # Convert value to string for typed prefix
            if isinstance(value, (dict, list)):
                import json

                value_str = json.dumps(value)
            else:
                value_str = str(value)
            prefixed_value = f"typed:{value_str}"
            parsed = parse_inputs(**{key: prefixed_value})
            self._inputs.update(parsed)
        return self

    def with_outputs(self, **outputs: Any) -> "ScheduleBuilder":
        """Add pre-set outputs.

        Args:
            **outputs: Output key-value pairs

        Returns:
            Self for chaining
        """
        for key, value in outputs.items():
            prefixed_value = f"output:{value}"
            parsed = parse_inputs(**{key: prefixed_value})
            self._inputs.update(parsed)
        return self

    def run(self) -> "ScheduledAgent":
        """Execute the scheduling with configured inputs.

        Returns:
            ScheduledAgent instance
        """
        # Convert ScheduledInput objects back to input format for schedule method
        input_kwargs = {}
        for key, scheduled_input in self._inputs.items():
            if scheduled_input.input_type.value == "secret":
                input_kwargs[key] = f"secret:{scheduled_input.value}"
            elif scheduled_input.input_type.value == "const":
                input_kwargs[key] = f"const:{scheduled_input.value}"
            elif scheduled_input.input_type.value == "cache":
                input_kwargs[
                    key
                ] = f"cache:{scheduled_input.ttl}:{scheduled_input.value}"
            elif scheduled_input.input_type.value == "typed":
                input_kwargs[key] = f"typed:{scheduled_input.value}"
            elif scheduled_input.input_type.value == "output":
                input_kwargs[key] = f"output:{scheduled_input.value}"
            else:  # variable
                input_kwargs[key] = scheduled_input.value

        return self._agent.schedule(self._schedule_string, **input_kwargs)


def create_schedule_builder(agent: "Agent", schedule_string: str) -> ScheduleBuilder:
    """Create a schedule builder for fluent API.

    Args:
        agent: Agent to schedule
        schedule_string: Schedule string

    Returns:
        ScheduleBuilder instance
    """
    return ScheduleBuilder(agent, schedule_string)
