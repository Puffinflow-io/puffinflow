"""Context with rich content management."""

import asyncio
import contextlib
import time
from typing import Any, Callable, Optional, TypeVar, Union

try:
    from pydantic import BaseModel as PydanticBaseModel

    _PYD_VER = 2
    _PBM: Any = PydanticBaseModel
except ImportError:
    try:
        from pydantic.v1 import BaseModel as PydanticBaseModel  # type: ignore

        _PYD_VER = 1
        _PBM = PydanticBaseModel
    except ImportError as _e:
        _PBM = type("BaseModel", (object,), {})
        _PYD_VER = 0
        _PYD_ERR = _e

from typing import Protocol, runtime_checkable

_PBM_T = TypeVar("_PBM_T", bound=_PBM)


@runtime_checkable
class TypedContextData(Protocol):
    """Protocol for typed context data."""

    pass


class StateType:
    """State type enumeration for context data."""

    ANY = "any"
    TYPED = "typed"
    UNTYPED = "untyped"


class Context:
    """Enhanced context for agent state management with rich content support."""

    _META_TYPED = "_meta_typed_"
    _META_VALIDATED = "_meta_validated_"
    _META_METADATA = "_meta_metadata_"
    _IMMUTABLE_PREFIXES = ("const_", "secret_")

    def __init__(self, shared_state: dict[str, Any], cache_ttl: int = 300) -> None:
        self.shared_state = shared_state if shared_state is not None else {}
        self.cache_ttl = cache_ttl
        self._typed_data: dict[str, Any] = {}
        self._typed_var_types: dict[str, type] = {}
        self._validated_types: dict[str, type] = {}
        self._cache: dict[str, tuple] = {}  # (value, expiry_time)
        self._outputs: dict[str, Any] = {}
        self._metadata: dict[str, Any] = {}
        self._metrics: dict[str, Union[int, float]] = {}
        self._restore_metadata()

    def _restore_metadata(self) -> None:
        """Restore metadata from shared state."""
        # Restore typed data metadata
        typed_keys = [k for k in self.shared_state if k.startswith(self._META_TYPED)]
        for k in typed_keys:
            orig = k[len(self._META_TYPED) :]
            self._typed_var_types[orig] = self.shared_state[k]

        # Restore validated data metadata
        validated_keys = [
            k for k in self.shared_state if k.startswith(self._META_VALIDATED)
        ]
        for k in validated_keys:
            orig = k[len(self._META_VALIDATED) :]
            self._validated_types[orig] = self.shared_state[k]

    @staticmethod
    def _now() -> float:
        """Get current timestamp."""
        return time.time()

    def _ensure_pydantic(self) -> None:
        """Ensure Pydantic is available."""
        if _PYD_VER == 0:
            raise ImportError(f"Pydantic is required for typed operations: {_PYD_ERR}")

    def _guard_reserved(self, key: str) -> None:
        """Guard against reserved key prefixes."""
        if any(key.startswith(prefix) for prefix in self._IMMUTABLE_PREFIXES):
            raise ValueError(f"Cannot modify reserved key: {key}")

    def _persist_meta(self, prefix: str, key: str, cls: type) -> None:
        """Persist metadata to shared state."""
        meta_key = f"{prefix}{key}"
        self.shared_state[meta_key] = cls

    # Basic state management
    def set_state(self, key: str, value: Any) -> None:
        """Set a state value."""
        self._guard_reserved(key)
        self.shared_state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self.shared_state.get(key, default)

    # Typed data management
    def set_typed(self, key: str, value: _PBM) -> None:
        """Set typed data with Pydantic model validation."""
        self._ensure_pydantic()
        if not isinstance(value, _PBM):
            raise TypeError(f"Value must be a Pydantic model, got {type(value)}")

        self._typed_data[key] = value
        self._typed_var_types[key] = type(value)
        self._persist_meta(self._META_TYPED, key, type(value))

    def get_typed(self, key: str, expected: type[_PBM_T]) -> Optional[_PBM_T]:
        """Get typed data with type checking."""
        self._ensure_pydantic()
        val = self._typed_data.get(key)
        if val is None:
            return None

        if not isinstance(val, expected):
            return None

        return val

    def update_typed(self, key: str, **updates: Any) -> None:
        """Update fields in typed data."""
        self._ensure_pydantic()
        current = self._typed_data.get(key)
        if current and isinstance(current, _PBM):
            # Create new instance with updated fields
            updated_data = current.dict()
            updated_data.update(updates)
            new_instance = type(current)(**updated_data)
            self._typed_data[key] = new_instance

    # Variable management (free variables)
    def set_variable(self, key: str, value: Any) -> None:
        """Set a variable in shared state."""
        if any(key.startswith(prefix) for prefix in self._IMMUTABLE_PREFIXES):
            raise ValueError(f"Cannot set variable with reserved prefix: {key}")
        self.shared_state[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a variable from shared state."""
        return self.shared_state.get(key, default)

    def get_variable_keys(self) -> set[str]:
        """Get all variable keys, excluding reserved prefixes."""
        return {
            k
            for k in self.shared_state
            if not any(k.startswith(prefix) for prefix in self._IMMUTABLE_PREFIXES)
            and not k.startswith(self._META_TYPED)
            and not k.startswith(self._META_VALIDATED)
            and not k.startswith(self._META_METADATA)
        }

    # Typed variables (with type consistency checking)
    def set_typed_variable(self, key: str, value: Any) -> None:
        """Set a typed variable with type consistency checking."""
        if any(key.startswith(prefix) for prefix in self._IMMUTABLE_PREFIXES):
            raise ValueError(f"Cannot set typed variable with reserved prefix: {key}")

        current_cls = self._typed_var_types.get(key)
        if current_cls and not isinstance(value, current_cls):
            raise TypeError(
                f"Type mismatch for {key}: expected {current_cls}, got {type(value)}"
            )

        self.shared_state[key] = value
        if key not in self._typed_var_types:
            cls = type(value)
            self._typed_var_types[key] = cls
            self._persist_meta(self._META_TYPED, key, cls)

    def get_typed_variable(self, key: str, expected: type[Any]) -> Optional[Any]:
        """Get a typed variable with type checking."""
        val = self.shared_state.get(key)
        if val is None:
            return None

        if not isinstance(val, expected):
            return None

        return val

    # Validated data (Pydantic models stored in shared state)
    def set_validated_data(self, key: str, value: _PBM) -> None:
        """Set validated Pydantic data in shared state."""
        self._ensure_pydantic()
        self._guard_reserved(key)
        if not isinstance(value, _PBM):
            raise TypeError(f"Value must be a Pydantic model, got {type(value)}")

        current_cls = self._validated_types.get(key)
        if current_cls and not isinstance(value, current_cls):
            raise TypeError(
                f"Type mismatch for {key}: expected {current_cls}, got {type(value)}"
            )

        self.shared_state[key] = value
        self._validated_types[key] = type(value)
        self._persist_meta(self._META_VALIDATED, key, type(value))

    def get_validated_data(self, key: str, expected: type[_PBM_T]) -> Optional[_PBM_T]:
        """Get validated data with type checking."""
        self._ensure_pydantic()
        val = self.shared_state.get(key)
        if val is None:
            return None

        if not isinstance(val, expected):
            return None

        return val

    # Immutable data (constants and secrets)
    def _set_immutable(self, prefix: str, key: str, value: Any) -> None:
        """Set immutable data with prefix."""
        full = f"{prefix}{key}"
        if full in self.shared_state:
            raise ValueError(f"Immutable key {key} already exists")
        self.shared_state[full] = value

    def set_constant(self, key: str, value: Any) -> None:
        """Set a constant value (immutable)."""
        self._set_immutable("const_", key, value)

    def get_constant(self, key: str, default: Any = None) -> Any:
        """Get a constant value."""
        return self.shared_state.get(f"const_{key}", default)

    def set_secret(self, key: str, value: str) -> None:
        """Set a secret value (immutable, string only)."""
        if not isinstance(value, str):
            raise TypeError("Secrets must be strings")
        self._set_immutable("secret_", key, value)

    def get_secret(self, key: str) -> Optional[str]:
        """Get a secret value."""
        return self.shared_state.get(f"secret_{key}")

    # Output management
    def set_output(self, key: str, value: Any) -> None:
        """Set an output value."""
        self._outputs[key] = value

    def get_output(self, key: str, default: Any = None) -> Any:
        """Get an output value."""
        return self._outputs.get(key, default)

    def get_output_keys(self) -> set[str]:
        """Get all output keys."""
        return set(self._outputs.keys())

    def get_all_outputs(self) -> dict[str, Any]:
        """Get all outputs."""
        return self._outputs.copy()

    # Metadata management
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value."""
        self._metadata[key] = value
        # Also persist to shared state for cross-agent access
        self.shared_state[f"{self._META_METADATA}{key}"] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        # Try local metadata first, then shared state
        value = self._metadata.get(key)
        if value is not None:
            return value
        return self.shared_state.get(f"{self._META_METADATA}{key}", default)

    def get_all_metadata(self) -> dict[str, Any]:
        """Get all metadata."""
        result = self._metadata.copy()
        # Add shared metadata
        for key, value in self.shared_state.items():
            if key.startswith(self._META_METADATA):
                orig_key = key[len(self._META_METADATA) :]
                if orig_key not in result:
                    result[orig_key] = value
        return result

    # Metrics management
    def set_metric(self, key: str, value: Union[int, float]) -> None:
        """Set a metric value."""
        if not isinstance(value, (int, float)):
            raise TypeError("Metrics must be numeric")
        self._metrics[key] = value

    def get_metric(self, key: str, default: Union[int, float] = 0) -> Union[int, float]:
        """Get a metric value."""
        return self._metrics.get(key, default)

    def increment_metric(self, key: str, amount: Union[int, float] = 1) -> None:
        """Increment a metric."""
        current = self._metrics.get(key, 0)
        self._metrics[key] = current + amount

    def get_all_metrics(self) -> dict[str, Union[int, float]]:
        """Get all metrics."""
        return self._metrics.copy()

    # Cache management with TTL
    def set_cached(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a cached value with TTL."""
        if ttl is None:
            ttl = self.cache_ttl

        expiry_time = self._now() + ttl if ttl > 0 else self._now() - 1
        self._cache[key] = (value, expiry_time)

    def get_cached(self, key: str, default: Any = None) -> Any:
        """Get a cached value, respecting TTL."""
        if key not in self._cache:
            return default

        value, expiry_time = self._cache[key]
        if self._now() > expiry_time:
            del self._cache[key]
            return default

        return value

    def clear_expired_cache(self) -> int:
        """Clear expired cache entries and return count cleared."""
        now = self._now()
        expired_keys = [
            key for key, (_, expiry_time) in self._cache.items() if now > expiry_time
        ]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    # State management and cleanup
    def remove_state(self, key: str, state_type: str = StateType.ANY) -> bool:
        """Remove state data by type."""
        removed = False

        if (
            state_type in (StateType.ANY, StateType.UNTYPED)
            and key in self.shared_state
        ):
            del self.shared_state[key]
            removed = True

        if state_type in (StateType.ANY, StateType.TYPED):
            if key in self._typed_data:
                del self._typed_data[key]
                removed = True
            if key in self._typed_var_types:
                del self._typed_var_types[key]
                removed = True

        return removed

    def clear_state(self, state_type: str = StateType.ANY) -> None:
        """Clear state data by type."""
        if state_type in (StateType.ANY, StateType.UNTYPED):
            # Clear non-reserved keys from shared state
            keys_to_remove = [
                k
                for k in self.shared_state
                if not any(k.startswith(prefix) for prefix in self._IMMUTABLE_PREFIXES)
                and not k.startswith(self._META_TYPED)
                and not k.startswith(self._META_VALIDATED)
            ]
            for key in keys_to_remove:
                del self.shared_state[key]

        if state_type in (StateType.ANY, StateType.TYPED):
            self._typed_data.clear()
            self._typed_var_types.clear()

    def get_keys(self, state_type: str = StateType.ANY) -> set[str]:
        """Get keys by state type."""
        keys = set()

        if state_type in (StateType.ANY, StateType.UNTYPED):
            keys.update(self.get_variable_keys())

        if state_type in (StateType.ANY, StateType.TYPED):
            keys.update(self._typed_data.keys())

        return keys

    # Human-in-the-loop functionality
    async def human_in_the_loop(
        self,
        prompt: str,
        timeout: Optional[float] = None,
        default: Optional[str] = None,
        validator: Optional[Callable[[str], bool]] = None,
    ) -> Optional[str]:
        """Get human input with optional timeout and validation."""
        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            try:
                if timeout:
                    # Async input with timeout
                    reply = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, input, prompt),
                        timeout=timeout,
                    )
                else:
                    # Synchronous input
                    reply = input(prompt)

                # Validate if validator provided
                if validator:
                    if validator(reply):
                        return reply
                    else:
                        print("Invalid input, please try again.")
                        attempt += 1
                        continue

                return reply

            except asyncio.TimeoutError:
                return default
            except Exception:
                attempt += 1
                if attempt >= max_attempts:
                    return default

        return default

    # Content inspection
    def get_content_summary(self) -> dict[str, Any]:
        """Get summary of all content in context."""
        return {
            "variables": len(self.get_variable_keys()),
            "outputs": len(self._outputs),
            "metadata": len(self._metadata),
            "metrics": len(self._metrics),
            "typed_data": len(self._typed_data),
            "cached_items": len(self._cache),
            "constants": len([k for k in self.shared_state if k.startswith("const_")]),
            "secrets": len([k for k in self.shared_state if k.startswith("secret_")]),
        }

    def export_content(self, include_secrets: bool = False) -> dict[str, Any]:
        """Export all context content."""
        content = {
            "variables": {k: self.shared_state[k] for k in self.get_variable_keys()},
            "outputs": self._outputs.copy(),
            "metadata": self.get_all_metadata(),
            "metrics": self._metrics.copy(),
            "typed_data": self._typed_data.copy(),
        }

        # Add constants
        constants = {
            k[6:]: v for k, v in self.shared_state.items() if k.startswith("const_")
        }
        if constants:
            content["constants"] = constants

        # Add secrets if requested
        if include_secrets:
            secrets = {
                k[7:]: v
                for k, v in self.shared_state.items()
                if k.startswith("secret_")
            }
            if secrets:
                content["secrets"] = secrets

        return content

    def import_content(self, content: dict[str, Any]) -> None:
        """Import content into context."""
        # Import variables
        if "variables" in content:
            for key, value in content["variables"].items():
                self.set_variable(key, value)

        # Import outputs
        if "outputs" in content:
            self._outputs.update(content["outputs"])

        # Import metadata
        if "metadata" in content:
            for key, value in content["metadata"].items():
                self.set_metadata(key, value)

        # Import metrics
        if "metrics" in content:
            for key, value in content["metrics"].items():
                self.set_metric(key, value)

        # Import typed data
        if "typed_data" in content:
            self._typed_data.update(content["typed_data"])

        # Import constants
        if "constants" in content:
            for key, value in content["constants"].items():
                with contextlib.suppress(ValueError):
                    self.set_constant(key, value)

        # Import secrets
        if "secrets" in content:
            for key, value in content["secrets"].items():
                with contextlib.suppress(ValueError):
                    self.set_secret(key, value)
