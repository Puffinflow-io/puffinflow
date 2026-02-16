"""State reducers for safe parallel state merging."""

from typing import Any, Callable


def add_reducer(existing: Any, new: Any) -> Any:
    """Merge values by addition: list concat, number add, dict merge."""
    if isinstance(existing, list) and isinstance(new, list):
        return existing + new
    if isinstance(existing, dict) and isinstance(new, dict):
        merged = existing.copy()
        merged.update(new)
        return merged
    if isinstance(existing, (int, float)) and isinstance(new, (int, float)):
        return existing + new
    # Fallback: replace
    return new


def append_reducer(existing: Any, new: Any) -> Any:
    """Always append new value to a list."""
    if not isinstance(existing, list):
        existing = [existing] if existing is not None else []
    if isinstance(new, list):
        return existing + new
    return [*existing, new]


def replace_reducer(existing: Any, new: Any) -> Any:
    """Last-write-wins (default behavior)."""
    return new


class ReducerRegistry:
    """Registry mapping state keys to reducer functions."""

    __slots__ = ("_reducers",)

    def __init__(self) -> None:
        self._reducers: dict[str, Callable[[Any, Any], Any]] = {}

    def register(self, key: str, reducer: Callable[[Any, Any], Any]) -> None:
        """Register a reducer for the given key."""
        self._reducers[key] = reducer

    def has(self, key: str) -> bool:
        """Check if a reducer is registered for the key."""
        return key in self._reducers

    def apply(self, key: str, existing: Any, new: Any) -> Any:
        """Apply the registered reducer for the key, or replace by default."""
        reducer = self._reducers.get(key)
        if reducer is not None:
            return reducer(existing, new)
        return new

    def keys(self) -> set[str]:
        """Return the set of keys with registered reducers."""
        return set(self._reducers.keys())
