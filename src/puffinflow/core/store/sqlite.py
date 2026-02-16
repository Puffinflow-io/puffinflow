"""SQLite-backed persistent store using aiosqlite."""

from __future__ import annotations

import json
import time
from typing import Any, List

try:
    import aiosqlite  # type: ignore[import-not-found]
except ImportError as _e:
    raise ImportError(
        "SqliteStore requires 'aiosqlite'. Install with: pip install aiosqlite"
    ) from _e

from .base import Item, Namespace


class SqliteStore:
    """Persistent store backed by SQLite via aiosqlite."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS store (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
            """)
            await self._db.commit()
        return self._db

    @staticmethod
    def _ns_str(namespace: Namespace) -> str:
        return "/".join(namespace)

    async def put(
        self,
        namespace: Namespace,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        db = await self._ensure_db()
        now = time.time()
        ns_str = self._ns_str(namespace)
        value_json = json.dumps(value)
        meta_json = json.dumps(metadata or {})
        await db.execute(
            """INSERT INTO store (namespace, key, value, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(namespace, key) DO UPDATE SET
                 value=excluded.value,
                 metadata=excluded.metadata,
                 updated_at=excluded.updated_at""",
            (ns_str, key, value_json, meta_json, now, now),
        )
        await db.commit()

    async def get(self, namespace: Namespace, key: str) -> Item | None:
        db = await self._ensure_db()
        ns_str = self._ns_str(namespace)
        cursor = await db.execute(
            "SELECT value, metadata, created_at, updated_at FROM store WHERE namespace=? AND key=?",
            (ns_str, key),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return Item(
            namespace=namespace,
            key=key,
            value=json.loads(row[0]),
            created_at=row[2],
            updated_at=row[3],
            metadata=json.loads(row[1]),
        )

    async def delete(self, namespace: Namespace, key: str) -> bool:
        db = await self._ensure_db()
        ns_str = self._ns_str(namespace)
        cursor = await db.execute(
            "DELETE FROM store WHERE namespace=? AND key=?", (ns_str, key)
        )
        await db.commit()
        return bool(cursor.rowcount > 0)

    async def list(
        self, namespace: Namespace, limit: int = 100, offset: int = 0
    ) -> List[Item]:
        db = await self._ensure_db()
        ns_str = self._ns_str(namespace)
        cursor = await db.execute(
            "SELECT key, value, metadata, created_at, updated_at FROM store "
            "WHERE namespace=? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (ns_str, limit, offset),
        )
        rows = await cursor.fetchall()
        return [
            Item(
                namespace=namespace,
                key=row[0],
                value=json.loads(row[1]),
                created_at=row[3],
                updated_at=row[4],
                metadata=json.loads(row[2]),
            )
            for row in rows
        ]

    async def search(
        self, namespace: Namespace, query: str = "", limit: int = 10
    ) -> List[Item]:
        db = await self._ensure_db()
        ns_str = self._ns_str(namespace)
        if query:
            cursor = await db.execute(
                "SELECT key, value, metadata, created_at, updated_at FROM store "
                "WHERE namespace LIKE ? AND (key LIKE ? OR value LIKE ?) LIMIT ?",
                (ns_str + "%", f"%{query}%", f"%{query}%", limit),
            )
        else:
            cursor = await db.execute(
                "SELECT key, value, metadata, created_at, updated_at FROM store "
                "WHERE namespace LIKE ? LIMIT ?",
                (ns_str + "%", limit),
            )
        rows = await cursor.fetchall()
        # Parse namespace back from string
        return [
            Item(
                namespace=namespace,
                key=row[0],
                value=json.loads(row[1]),
                created_at=row[3],
                updated_at=row[4],
                metadata=json.loads(row[2]),
            )
            for row in rows
        ]

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
