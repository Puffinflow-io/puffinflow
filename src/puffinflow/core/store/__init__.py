"""Persistent key-value store for agent memory."""

from .base import BaseStore, Item, MemoryStore

__all__ = [
    "BaseStore",
    "Item",
    "MemoryStore",
]

# SqliteStore is optional — requires aiosqlite
try:
    from .sqlite import SqliteStore  # noqa: F401

    __all__.append("SqliteStore")
except ImportError:
    pass
