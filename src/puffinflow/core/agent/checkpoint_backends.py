"""Pluggable checkpoint storage backends (Redis, PostgreSQL, S3)."""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from .checkpoint import AgentCheckpoint
from .checkpoint_serializer import CheckpointSerializer, JsonCheckpointSerializer

logger = logging.getLogger(__name__)


@dataclass
class RetentionPolicy:
    """Policy for automatic checkpoint cleanup."""

    max_count: Optional[int] = None
    """Keep only the last N checkpoints per agent."""

    max_age_seconds: Optional[float] = None
    """Delete checkpoints older than this many seconds."""


class RedisCheckpointStorage:
    """Redis-backed checkpoint storage — fast, good for short-lived checkpoints.

    Key format: ``puffinflow:checkpoints:{agent_name}:{checkpoint_id}``

    Requires the ``redis`` package (optional dependency).
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "puffinflow:checkpoints",
        serializer: Optional[CheckpointSerializer] = None,
        retention_policy: Optional[RetentionPolicy] = None,
        ttl: Optional[int] = None,
    ) -> None:
        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError(
                "redis is required for RedisCheckpointStorage. "
                "Install with: pip install redis"
            )

        self._redis = aioredis.from_url(url, decode_responses=False)
        self._prefix = prefix
        self._serializer = serializer or JsonCheckpointSerializer()
        self._retention_policy = retention_policy
        self._ttl = ttl

    def _key(self, agent_name: str, checkpoint_id: str) -> str:
        return f"{self._prefix}:{agent_name}:{checkpoint_id}"

    def _index_key(self, agent_name: str) -> str:
        return f"{self._prefix}:{agent_name}:__index__"

    async def save_checkpoint(
        self, agent_name: str, checkpoint: AgentCheckpoint
    ) -> str:
        """Save checkpoint to Redis."""
        checkpoint_id = f"checkpoint_{int(checkpoint.timestamp)}"
        key = self._key(agent_name, checkpoint_id)
        data = self._serializer.serialize(checkpoint)

        if self._ttl:
            await self._redis.setex(key, self._ttl, data)
        else:
            await self._redis.set(key, data)

        # Maintain an ordered index of checkpoint IDs
        await self._redis.zadd(
            self._index_key(agent_name),
            {checkpoint_id: checkpoint.timestamp},
        )

        # Enforce retention policy
        if self._retention_policy:
            await self.cleanup_checkpoints(agent_name)

        logger.info("Checkpoint saved to Redis: %s/%s", agent_name, checkpoint_id)
        return checkpoint_id

    async def load_checkpoint(
        self, agent_name: str, checkpoint_id: Optional[str] = None
    ) -> Optional[AgentCheckpoint]:
        """Load checkpoint from Redis."""
        if checkpoint_id is None:
            # Get latest from index
            ids = await self._redis.zrange(
                self._index_key(agent_name), -1, -1
            )
            if not ids:
                return None
            checkpoint_id = ids[0] if isinstance(ids[0], str) else ids[0].decode()

        key = self._key(agent_name, checkpoint_id)
        data = await self._redis.get(key)
        if data is None:
            return None

        return self._serializer.deserialize(data)  # type: ignore[return-value]

    async def list_checkpoints(self, agent_name: str) -> list[str]:
        """List checkpoint IDs from Redis index."""
        ids = await self._redis.zrange(self._index_key(agent_name), 0, -1)
        return [
            cid if isinstance(cid, str) else cid.decode() for cid in ids
        ]

    async def delete_checkpoint(self, agent_name: str, checkpoint_id: str) -> bool:
        """Delete a checkpoint from Redis."""
        key = self._key(agent_name, checkpoint_id)
        deleted = await self._redis.delete(key)
        await self._redis.zrem(self._index_key(agent_name), checkpoint_id)
        return deleted > 0

    async def cleanup_checkpoints(self, agent_name: str) -> int:
        """Remove checkpoints exceeding the retention policy."""
        if not self._retention_policy:
            return 0

        removed = 0
        index_key = self._index_key(agent_name)

        # Enforce max_count
        if self._retention_policy.max_count is not None:
            total = await self._redis.zcard(index_key)
            if total > self._retention_policy.max_count:
                excess = total - self._retention_policy.max_count
                old_ids = await self._redis.zrange(index_key, 0, excess - 1)
                for cid in old_ids:
                    cid_str = cid if isinstance(cid, str) else cid.decode()
                    await self.delete_checkpoint(agent_name, cid_str)
                    removed += 1

        # Enforce max_age_seconds
        if self._retention_policy.max_age_seconds is not None:
            cutoff = time.time() - self._retention_policy.max_age_seconds
            old_ids = await self._redis.zrangebyscore(index_key, "-inf", cutoff)
            for cid in old_ids:
                cid_str = cid if isinstance(cid, str) else cid.decode()
                await self.delete_checkpoint(agent_name, cid_str)
                removed += 1

        return removed


class PostgresCheckpointStorage:
    """PostgreSQL-backed checkpoint storage — durable, queryable.

    Table: ``puffinflow_checkpoints``
        - id SERIAL PRIMARY KEY
        - agent_name TEXT
        - checkpoint_id TEXT
        - schema_version INT
        - data JSONB
        - created_at TIMESTAMPTZ

    Requires the ``asyncpg`` package (optional dependency).
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost/puffinflow",
        table: str = "puffinflow_checkpoints",
        serializer: Optional[CheckpointSerializer] = None,
        retention_policy: Optional[RetentionPolicy] = None,
    ) -> None:
        try:
            import asyncpg  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            raise ImportError(
                "asyncpg is required for PostgresCheckpointStorage. "
                "Install with: pip install asyncpg"
            )

        self._dsn = dsn
        self._table = table
        self._serializer = serializer or JsonCheckpointSerializer()
        self._retention_policy = retention_policy
        self._pool: Any = None

    async def _get_pool(self) -> Any:
        if self._pool is None:
            import asyncpg  # type: ignore[import-not-found]

            self._pool = await asyncpg.create_pool(self._dsn)
            await self._ensure_table()
        return self._pool

    async def _ensure_table(self) -> None:
        """Create the checkpoints table if it doesn't exist."""
        pool = self._pool
        async with pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id SERIAL PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    schema_version INT NOT NULL DEFAULT 1,
                    data JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (agent_name, checkpoint_id)
                )
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table}_agent_name
                ON {self._table} (agent_name, created_at)
            """)

    async def save_checkpoint(
        self, agent_name: str, checkpoint: AgentCheckpoint
    ) -> str:
        """Save checkpoint to PostgreSQL."""
        import json

        checkpoint_id = f"checkpoint_{int(checkpoint.timestamp)}"
        serialized = self._serializer.serialize(checkpoint)
        data_json = json.loads(serialized.decode("utf-8"))

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._table} (agent_name, checkpoint_id, schema_version, data)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (agent_name, checkpoint_id)
                DO UPDATE SET data = $4::jsonb, schema_version = $3, created_at = NOW()
                """,
                agent_name,
                checkpoint_id,
                checkpoint.schema_version,
                json.dumps(data_json),
            )

        # Enforce retention policy
        if self._retention_policy:
            await self.cleanup_checkpoints(agent_name)

        logger.info("Checkpoint saved to PostgreSQL: %s/%s", agent_name, checkpoint_id)
        return checkpoint_id

    async def load_checkpoint(
        self, agent_name: str, checkpoint_id: Optional[str] = None
    ) -> Optional[AgentCheckpoint]:
        """Load checkpoint from PostgreSQL."""
        import json

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if checkpoint_id is None:
                row = await conn.fetchrow(
                    f"""
                    SELECT data FROM {self._table}
                    WHERE agent_name = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    agent_name,
                )
            else:
                row = await conn.fetchrow(
                    f"""
                    SELECT data FROM {self._table}
                    WHERE agent_name = $1 AND checkpoint_id = $2
                    """,
                    agent_name,
                    checkpoint_id,
                )

        if row is None:
            return None

        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)
        # data is the full envelope from the serializer
        serialized = json.dumps(data).encode("utf-8")
        return self._serializer.deserialize(serialized)  # type: ignore[return-value]

    async def list_checkpoints(self, agent_name: str) -> list[str]:
        """List checkpoint IDs from PostgreSQL."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT checkpoint_id FROM {self._table}
                WHERE agent_name = $1
                ORDER BY created_at ASC
                """,
                agent_name,
            )
        return [row["checkpoint_id"] for row in rows]

    async def delete_checkpoint(self, agent_name: str, checkpoint_id: str) -> bool:
        """Delete a checkpoint from PostgreSQL."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                f"""
                DELETE FROM {self._table}
                WHERE agent_name = $1 AND checkpoint_id = $2
                """,
                agent_name,
                checkpoint_id,
            )
        return result != "DELETE 0"  # type: ignore[return-value]

    async def cleanup_checkpoints(self, agent_name: str) -> int:
        """Remove checkpoints exceeding the retention policy."""
        if not self._retention_policy:
            return 0

        removed = 0
        pool = await self._get_pool()

        async with pool.acquire() as conn:
            # Enforce max_count
            if self._retention_policy.max_count is not None:
                result = await conn.execute(
                    f"""
                    DELETE FROM {self._table}
                    WHERE agent_name = $1
                    AND id NOT IN (
                        SELECT id FROM {self._table}
                        WHERE agent_name = $1
                        ORDER BY created_at DESC
                        LIMIT $2
                    )
                    """,
                    agent_name,
                    self._retention_policy.max_count,
                )
                # result is like "DELETE 3"
                count = int(result.split()[-1]) if result else 0
                removed += count

            # Enforce max_age_seconds
            if self._retention_policy.max_age_seconds is not None:
                import datetime

                cutoff = datetime.datetime.now(
                    tz=datetime.timezone.utc
                ) - datetime.timedelta(seconds=self._retention_policy.max_age_seconds)
                result = await conn.execute(
                    f"""
                    DELETE FROM {self._table}
                    WHERE agent_name = $1 AND created_at < $2
                    """,
                    agent_name,
                    cutoff,
                )
                count = int(result.split()[-1]) if result else 0
                removed += count

        return removed


class S3CheckpointStorage:
    """S3-compatible checkpoint storage — cheap long-term storage.

    Key format: ``{prefix}/{agent_name}/{checkpoint_id}.json``

    Requires the ``aiobotocore`` package (optional dependency).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "puffinflow/checkpoints",
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        serializer: Optional[CheckpointSerializer] = None,
        retention_policy: Optional[RetentionPolicy] = None,
    ) -> None:
        try:
            import aiobotocore  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            raise ImportError(
                "aiobotocore is required for S3CheckpointStorage. "
                "Install with: pip install aiobotocore"
            )

        self._bucket = bucket
        self._prefix = prefix
        self._region = region
        self._endpoint_url = endpoint_url
        self._serializer = serializer or JsonCheckpointSerializer()
        self._retention_policy = retention_policy

    def _s3_key(self, agent_name: str, checkpoint_id: str) -> str:
        return f"{self._prefix}/{agent_name}/{checkpoint_id}.json"

    def _create_session(self) -> Any:
        import aiobotocore.session  # type: ignore[import-not-found]

        return aiobotocore.session.get_session()

    async def save_checkpoint(
        self, agent_name: str, checkpoint: AgentCheckpoint
    ) -> str:
        """Save checkpoint to S3."""
        checkpoint_id = f"checkpoint_{int(checkpoint.timestamp)}"
        key = self._s3_key(agent_name, checkpoint_id)
        data = self._serializer.serialize(checkpoint)

        session = self._create_session()
        async with session.create_client(
            "s3",
            region_name=self._region,
            endpoint_url=self._endpoint_url,
        ) as client:
            await client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType="application/json",
            )

        # Enforce retention policy
        if self._retention_policy:
            await self.cleanup_checkpoints(agent_name)

        logger.info("Checkpoint saved to S3: s3://%s/%s", self._bucket, key)
        return checkpoint_id

    async def load_checkpoint(
        self, agent_name: str, checkpoint_id: Optional[str] = None
    ) -> Optional[AgentCheckpoint]:
        """Load checkpoint from S3."""
        if checkpoint_id is None:
            # List and pick latest
            checkpoints = await self.list_checkpoints(agent_name)
            if not checkpoints:
                return None
            checkpoint_id = checkpoints[-1]

        key = self._s3_key(agent_name, checkpoint_id)

        session = self._create_session()
        try:
            async with session.create_client(
                "s3",
                region_name=self._region,
                endpoint_url=self._endpoint_url,
            ) as client:
                response = await client.get_object(
                    Bucket=self._bucket, Key=key
                )
                async with response["Body"] as stream:
                    data = await stream.read()
        except Exception as exc:
            logger.warning("Failed to load checkpoint from S3: %s", exc)
            return None

        return self._serializer.deserialize(data)  # type: ignore[return-value]

    async def list_checkpoints(self, agent_name: str) -> list[str]:
        """List checkpoint IDs from S3."""
        prefix = f"{self._prefix}/{agent_name}/"

        session = self._create_session()
        checkpoint_ids = []

        async with session.create_client(
            "s3",
            region_name=self._region,
            endpoint_url=self._endpoint_url,
        ) as client:
            paginator = client.get_paginator("list_objects_v2")
            async for page in paginator.paginate(
                Bucket=self._bucket, Prefix=prefix
            ):
                for obj in page.get("Contents", []):
                    # Extract checkpoint_id from key
                    key = obj["Key"]
                    filename = key.rsplit("/", 1)[-1]
                    if filename.endswith(".json"):
                        checkpoint_ids.append(filename[:-5])  # Strip .json

        # Sort by timestamp
        checkpoint_ids.sort(
            key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0
        )
        return checkpoint_ids

    async def delete_checkpoint(self, agent_name: str, checkpoint_id: str) -> bool:
        """Delete a checkpoint from S3."""
        key = self._s3_key(agent_name, checkpoint_id)

        session = self._create_session()
        try:
            async with session.create_client(
                "s3",
                region_name=self._region,
                endpoint_url=self._endpoint_url,
            ) as client:
                await client.delete_object(Bucket=self._bucket, Key=key)
            return True
        except Exception as exc:
            logger.error("Failed to delete checkpoint from S3: %s", exc)
            return False

    async def cleanup_checkpoints(self, agent_name: str) -> int:
        """Remove checkpoints exceeding the retention policy."""
        if not self._retention_policy:
            return 0

        checkpoints = await self.list_checkpoints(agent_name)
        removed = 0

        # Enforce max_count
        if (
            self._retention_policy.max_count is not None
            and len(checkpoints) > self._retention_policy.max_count
        ):
            excess = checkpoints[: len(checkpoints) - self._retention_policy.max_count]
            for cid in excess:
                await self.delete_checkpoint(agent_name, cid)
                removed += 1

        # Enforce max_age_seconds
        if self._retention_policy.max_age_seconds is not None:
            cutoff = time.time() - self._retention_policy.max_age_seconds
            for cid in checkpoints:
                try:
                    ts = int(cid.split("_")[-1])
                    if ts < cutoff:
                        await self.delete_checkpoint(agent_name, cid)
                        removed += 1
                except (ValueError, IndexError):
                    pass

        return removed
