"""Input types and magic prefix parsing for scheduled agents."""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .exceptions import InvalidInputTypeError


class InputType(Enum):
    """Types of inputs for scheduled agents."""

    VARIABLE = "variable"  # Regular variables (no prefix)
    SECRET = "secret"  # secret:value
    CONSTANT = "const"  # const:value
    CACHED = "cache"  # cache:TTL:value
    TYPED = "typed"  # typed:value
    OUTPUT = "output"  # output:value


@dataclass
class ScheduledInput:
    """Configuration for a scheduled input."""

    key: str
    value: Any
    input_type: InputType
    ttl: Optional[int] = None  # For cached inputs

    def apply_to_context(self, context: Any) -> None:
        """Apply this input to a context."""
        if self.input_type == InputType.SECRET:
            context.set_secret(self.key, self.value)
        elif self.input_type == InputType.CONSTANT:
            context.set_constant(self.key, self.value)
        elif self.input_type == InputType.CACHED:
            context.set_cached(self.key, self.value, self.ttl)
        elif self.input_type == InputType.TYPED:
            context.set_typed_variable(self.key, self.value)
        elif self.input_type == InputType.OUTPUT:
            context.set_output(self.key, self.value)
        else:  # VARIABLE
            context.set_variable(self.key, self.value)


def parse_magic_prefix(key: str, value: Any) -> ScheduledInput:
    """Parse magic prefix from input value and return ScheduledInput.

    Args:
        key: The input key name
        value: The input value, potentially with magic prefix

    Returns:
        ScheduledInput with parsed type and value

    Raises:
        InvalidInputTypeError: If prefix is invalid
    """
    if not isinstance(value, str):
        # Non-string values are treated as regular variables
        return ScheduledInput(key, value, InputType.VARIABLE)

    # Check for magic prefixes
    if ":" not in value:
        # No prefix, regular variable
        return ScheduledInput(key, value, InputType.VARIABLE)

    parts = value.split(":", 1)
    prefix = parts[0].lower()

    if prefix == "secret":
        if len(parts) != 2 or not parts[1]:
            raise InvalidInputTypeError(prefix, "Secret format: secret:value")
        return ScheduledInput(key, parts[1], InputType.SECRET)

    elif prefix == "const":
        if len(parts) != 2 or not parts[1]:
            raise InvalidInputTypeError(prefix, "Constant format: const:value")
        return ScheduledInput(key, parts[1], InputType.CONSTANT)

    elif prefix == "cache":
        # Format: cache:TTL:value
        cache_parts = value.split(":", 2)
        if len(cache_parts) != 3:
            raise InvalidInputTypeError(prefix, "Cache format: cache:TTL:value")

        try:
            ttl = int(cache_parts[1])
        except ValueError as e:
            raise InvalidInputTypeError(prefix, "Cache TTL must be an integer") from e

        # Try to parse value as JSON, fall back to string
        raw_value = cache_parts[2]
        try:
            parsed_value = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed_value = raw_value

        return ScheduledInput(key, parsed_value, InputType.CACHED, ttl=ttl)

    elif prefix == "typed":
        if len(parts) != 2 or not parts[1]:
            raise InvalidInputTypeError(prefix, "Typed format: typed:value")

        # Try to parse value as JSON for complex types
        raw_value = parts[1]
        try:
            parsed_value = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed_value = raw_value

        return ScheduledInput(key, parsed_value, InputType.TYPED)

    elif prefix == "output":
        if len(parts) != 2 or not parts[1]:
            raise InvalidInputTypeError(prefix, "Output format: output:value")
        return ScheduledInput(key, parts[1], InputType.OUTPUT)

    else:
        # Unknown prefix, treat as regular variable
        return ScheduledInput(key, value, InputType.VARIABLE)


def parse_inputs(**inputs: Any) -> dict[str, ScheduledInput]:
    """Parse all inputs with magic prefixes.

    Args:
        **inputs: Input key-value pairs

    Returns:
        Dictionary mapping keys to ScheduledInput objects
    """
    parsed_inputs = {}
    for key, value in inputs.items():
        parsed_inputs[key] = parse_magic_prefix(key, value)
    return parsed_inputs
