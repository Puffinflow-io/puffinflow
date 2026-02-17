"""Tests for pluggable checkpoint backends (Redis, PostgreSQL, S3)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from puffinflow.core.agent.checkpoint import AgentCheckpoint
from puffinflow.core.agent.checkpoint_backends import RetentionPolicy
from puffinflow.core.agent.state import (
    AgentStatus,
    Priority,
    StateMetadata,
    StateStatus,
)


def _make_checkpoint(ts: float = 1000.0) -> AgentCheckpoint:
    """Create a sample checkpoint for testing."""
    meta = StateMetadata(
        status=StateStatus.COMPLETED,
        attempts=1,
        max_retries=3,
        last_execution=100.0,
        last_success=100.0,
        state_id="test-id",
        priority=Priority.NORMAL,
    )
    return AgentCheckpoint(
        timestamp=ts,
        agent_name="test-agent",
        agent_status=AgentStatus.COMPLETED,
        priority_queue=[],
        state_metadata={"step_one": meta},
        running_states=set(),
        completed_states={"step_one"},
        completed_once={"step_one"},
        shared_state={"counter": 42},
        session_start=999.0,
        schema_version=1,
    )


class TestRetentionPolicy:
    """Test RetentionPolicy dataclass."""

    def test_defaults(self):
        policy = RetentionPolicy()
        assert policy.max_count is None
        assert policy.max_age_seconds is None

    def test_with_max_count(self):
        policy = RetentionPolicy(max_count=5)
        assert policy.max_count == 5

    def test_with_max_age(self):
        policy = RetentionPolicy(max_age_seconds=3600.0)
        assert policy.max_age_seconds == 3600.0

    def test_with_both(self):
        policy = RetentionPolicy(max_count=10, max_age_seconds=86400.0)
        assert policy.max_count == 10
        assert policy.max_age_seconds == 86400.0


class TestRedisCheckpointStorage:
    """Test Redis backend with mocked redis client."""

    @pytest.fixture
    def mock_redis(self):
        with patch(
            "puffinflow.core.agent.checkpoint_backends.RedisCheckpointStorage.__init__",
            return_value=None,
        ) as _:
            from puffinflow.core.agent.checkpoint_backends import RedisCheckpointStorage

            storage = RedisCheckpointStorage.__new__(RedisCheckpointStorage)
            storage._redis = AsyncMock()
            storage._prefix = "puffinflow:checkpoints"
            storage._serializer = MagicMock()
            storage._serializer.serialize.return_value = b'{"test": true}'
            storage._serializer.deserialize.return_value = _make_checkpoint()
            storage._retention_policy = None
            storage._ttl = None
            return storage

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, mock_redis):
        cp = _make_checkpoint()
        result = await mock_redis.save_checkpoint("test-agent", cp)
        assert result.startswith("checkpoint_")
        mock_redis._redis.set.assert_called_once()
        mock_redis._redis.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_checkpoint_with_ttl(self, mock_redis):
        mock_redis._ttl = 3600
        cp = _make_checkpoint()
        await mock_redis.save_checkpoint("test-agent", cp)
        mock_redis._redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_checkpoint_by_id(self, mock_redis):
        mock_redis._redis.get = AsyncMock(return_value=b'{"test": true}')
        result = await mock_redis.load_checkpoint("test-agent", "checkpoint_1000")
        assert result is not None

    @pytest.mark.asyncio
    async def test_load_checkpoint_latest(self, mock_redis):
        mock_redis._redis.zrange = AsyncMock(return_value=[b"checkpoint_1000"])
        mock_redis._redis.get = AsyncMock(return_value=b'{"test": true}')
        result = await mock_redis.load_checkpoint("test-agent")
        assert result is not None

    @pytest.mark.asyncio
    async def test_load_checkpoint_returns_none_when_empty(self, mock_redis):
        mock_redis._redis.zrange = AsyncMock(return_value=[])
        result = await mock_redis.load_checkpoint("test-agent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, mock_redis):
        mock_redis._redis.zrange = AsyncMock(
            return_value=[b"checkpoint_1000", b"checkpoint_2000"]
        )
        result = await mock_redis.list_checkpoints("test-agent")
        assert result == ["checkpoint_1000", "checkpoint_2000"]

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, mock_redis):
        mock_redis._redis.delete = AsyncMock(return_value=1)
        mock_redis._redis.zrem = AsyncMock()
        result = await mock_redis.delete_checkpoint("test-agent", "checkpoint_1000")
        assert result is True


class TestPostgresCheckpointStorage:
    """Test PostgreSQL backend with mocked asyncpg."""

    @pytest.fixture
    def mock_pg(self):
        with patch(
            "puffinflow.core.agent.checkpoint_backends."
            "PostgresCheckpointStorage.__init__",
            return_value=None,
        ):
            from puffinflow.core.agent.checkpoint_backends import (
                PostgresCheckpointStorage,
            )

            storage = PostgresCheckpointStorage.__new__(PostgresCheckpointStorage)
            storage._dsn = "postgresql://localhost/test"
            storage._table = "puffinflow_checkpoints"
            storage._serializer = MagicMock()
            storage._serializer.serialize.return_value = (
                b'{"schema_version": 1, "data": {}}'
            )
            storage._serializer.deserialize.return_value = _make_checkpoint()
            storage._retention_policy = None

            # Mock pool with proper async context manager
            mock_conn = AsyncMock()

            class _AcquireCM:
                async def __aenter__(self_inner):
                    return mock_conn

                async def __aexit__(self_inner, *args):
                    pass

            mock_pool = MagicMock()
            mock_pool.acquire.return_value = _AcquireCM()
            storage._pool = mock_pool
            storage._mock_conn = mock_conn
            return storage

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, mock_pg):
        cp = _make_checkpoint()
        result = await mock_pg.save_checkpoint("test-agent", cp)
        assert result.startswith("checkpoint_")

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, mock_pg):
        mock_pg._mock_conn.fetch = AsyncMock(
            return_value=[
                {"checkpoint_id": "checkpoint_1000"},
                {"checkpoint_id": "checkpoint_2000"},
            ]
        )
        result = await mock_pg.list_checkpoints("test-agent")
        assert result == ["checkpoint_1000", "checkpoint_2000"]

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, mock_pg):
        mock_pg._mock_conn.execute = AsyncMock(return_value="DELETE 1")
        result = await mock_pg.delete_checkpoint("test-agent", "checkpoint_1000")
        assert result is True


class TestS3CheckpointStorage:
    """Test S3 backend with mocked aiobotocore."""

    @pytest.fixture
    def mock_s3(self):
        with patch(
            "puffinflow.core.agent.checkpoint_backends.S3CheckpointStorage.__init__",
            return_value=None,
        ):
            from puffinflow.core.agent.checkpoint_backends import S3CheckpointStorage

            storage = S3CheckpointStorage.__new__(S3CheckpointStorage)
            storage._bucket = "test-bucket"
            storage._prefix = "puffinflow/checkpoints"
            storage._region = "us-east-1"
            storage._endpoint_url = None
            storage._serializer = MagicMock()
            storage._serializer.serialize.return_value = b'{"test": true}'
            storage._serializer.deserialize.return_value = _make_checkpoint()
            storage._retention_policy = None
            return storage

    @pytest.mark.asyncio
    async def test_s3_key_format(self, mock_s3):
        key = mock_s3._s3_key("test-agent", "checkpoint_1000")
        assert key == "puffinflow/checkpoints/test-agent/checkpoint_1000.json"

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, mock_s3):
        mock_client = AsyncMock()
        mock_session = MagicMock()
        mock_session.create_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_client
        )
        mock_session.create_client.return_value.__aexit__ = AsyncMock(
            return_value=False
        )
        mock_s3._create_session = MagicMock(return_value=mock_session)

        cp = _make_checkpoint()
        result = await mock_s3.save_checkpoint("test-agent", cp)
        assert result.startswith("checkpoint_")
        mock_client.put_object.assert_called_once()
