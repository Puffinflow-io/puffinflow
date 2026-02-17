"""Base store protocol and in-memory implementation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Protocol, runtime_checkable  # noqa: UP035

Namespace = tuple[str, ...]


@dataclass
class Item:
    """A stored item with metadata."""

    namespace: Namespace
    key: str
    value: Any
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class BaseStore(Protocol):
    """Protocol for persistent key-value stores."""

    async def put(
        self,
        namespace: Namespace,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    async def get(self, namespace: Namespace, key: str) -> Item | None: ...

    async def delete(self, namespace: Namespace, key: str) -> bool: ...

    async def list(
        self, namespace: Namespace, limit: int = 100, offset: int = 0
    ) -> list[Item]: ...

    async def search(
        self, namespace: Namespace, query: str = "", limit: int = 10
    ) -> List[Item]:  # noqa: UP006  # 'list' resolves to the method, not the builtin
        ...


class MemoryStore:
    """In-memory implementation of BaseStore."""

    __slots__ = ("_data", "_order", "_seq")

    def __init__(self) -> None:
        # Keyed by (namespace, key)
        self._data: dict[tuple[Namespace, str], Item] = {}
        self._seq: int = 0
        self._order: dict[tuple[Namespace, str], int] = {}

    async def put(
        self,
        namespace: Namespace,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = time.time()
        self._seq += 1
        self._order[(namespace, key)] = self._seq
        existing = self._data.get((namespace, key))
        if existing is not None:
            existing.value = value
            existing.updated_at = now
            if metadata is not None:
                existing.metadata.update(metadata)
        else:
            self._data[(namespace, key)] = Item(
                namespace=namespace,
                key=key,
                value=value,
                created_at=now,
                updated_at=now,
                metadata=metadata or {},
            )

    async def get(self, namespace: Namespace, key: str) -> Item | None:
        return self._data.get((namespace, key))

    async def delete(self, namespace: Namespace, key: str) -> bool:
        if (namespace, key) in self._data:
            del self._data[(namespace, key)]
            self._order.pop((namespace, key), None)
            return True
        return False

    async def list(
        self, namespace: Namespace, limit: int = 100, offset: int = 0
    ) -> list[Item]:
        items = [item for (ns, _), item in self._data.items() if ns == namespace]
        # Sort by updated_at descending, then by insertion order descending
        items.sort(
            key=lambda i: (i.updated_at, self._order.get((i.namespace, i.key), 0)),
            reverse=True,
        )
        return items[offset : offset + limit]

    async def search(
        self, namespace: Namespace, query: str = "", limit: int = 10
    ) -> List[Item]:  # noqa: UP006  # 'list' resolves to the method, not the builtin
        results: List[Item] = []  # noqa: UP006
        for (ns, k), item in self._data.items():
            if ns[: len(namespace)] != namespace:
                continue
            if query:
                # Simple text search across key and string values
                searchable = str(k) + str(item.value)
                if query.lower() not in searchable.lower():
                    continue
            results.append(item)
            if len(results) >= limit:
                break
        return results
