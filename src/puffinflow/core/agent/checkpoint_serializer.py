"""Pluggable serializers for checkpoint data with schema versioning."""

import json
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1

# Migration registry: {(from_version, to_version): migration_function}
_MIGRATIONS: dict[tuple[int, int], Callable[[dict], dict]] = {}


def register_migration(
    from_version: int, to_version: int
) -> Callable[[Callable[[dict], dict]], Callable[[dict], dict]]:
    """Decorator to register a checkpoint migration function."""

    def decorator(func: Callable[[dict], dict]) -> Callable[[dict], dict]:
        _MIGRATIONS[(from_version, to_version)] = func
        return func

    return decorator


def migrate_checkpoint(
    data: dict, from_version: int, to_version: int
) -> dict:
    """Apply sequential migrations from from_version to to_version.

    Args:
        data: The checkpoint data dict to migrate.
        from_version: The current schema version of the data.
        to_version: The target schema version.

    Returns:
        Migrated checkpoint data dict.

    Raises:
        ValueError: If no migration path exists.
    """
    if from_version == to_version:
        return data

    current = from_version
    while current < to_version:
        next_version = current + 1
        migration = _MIGRATIONS.get((current, next_version))
        if migration is None:
            raise ValueError(
                f"No migration path from schema version {current} to {next_version}"
            )
        data = migration(data)
        data["schema_version"] = next_version
        current = next_version
        logger.info(
            "Migrated checkpoint from schema v%d to v%d", current - 1, current
        )

    return data


class CheckpointSerializer:
    """Base class for checkpoint serializers."""

    def serialize(self, checkpoint: Any) -> bytes:
        """Serialize a checkpoint to bytes.

        Args:
            checkpoint: An AgentCheckpoint instance.

        Returns:
            Serialized bytes.
        """
        raise NotImplementedError

    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to an AgentCheckpoint.

        Args:
            data: Serialized checkpoint bytes.

        Returns:
            An AgentCheckpoint instance.
        """
        raise NotImplementedError


class JsonCheckpointSerializer(CheckpointSerializer):
    """JSON serializer — human-readable, portable, secure.

    Wraps checkpoint data in an envelope:
        {"schema_version": 1, "data": {...}}
    """

    def serialize(self, checkpoint: Any) -> bytes:
        """Serialize checkpoint to JSON bytes."""
        from .checkpoint import AgentCheckpoint

        if isinstance(checkpoint, AgentCheckpoint):
            checkpoint_dict = checkpoint.to_dict()
        else:
            checkpoint_dict = checkpoint

        envelope = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "data": checkpoint_dict,
        }
        return json.dumps(envelope, indent=2, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> Any:
        """Deserialize JSON bytes to AgentCheckpoint."""
        from .checkpoint import AgentCheckpoint

        envelope = json.loads(data.decode("utf-8"))

        schema_version = envelope.get("schema_version", 1)
        checkpoint_data = envelope.get("data", envelope)

        # Apply migrations if needed
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            checkpoint_data = migrate_checkpoint(
                checkpoint_data, schema_version, CHECKPOINT_SCHEMA_VERSION
            )

        return AgentCheckpoint.from_dict(checkpoint_data)


class MsgpackCheckpointSerializer(CheckpointSerializer):
    """MessagePack serializer — compact binary, faster than JSON.

    Requires the ``msgpack`` package (optional dependency).
    Same envelope format with schema_version.
    """

    def __init__(self) -> None:
        try:
            import msgpack  # noqa: F401

            self._msgpack = msgpack
        except ImportError:
            raise ImportError(
                "msgpack is required for MsgpackCheckpointSerializer. "
                "Install with: pip install msgpack"
            )

    def serialize(self, checkpoint: Any) -> bytes:
        """Serialize checkpoint to MessagePack bytes."""
        from .checkpoint import AgentCheckpoint

        if isinstance(checkpoint, AgentCheckpoint):
            checkpoint_dict = checkpoint.to_dict()
        else:
            checkpoint_dict = checkpoint

        envelope = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "data": checkpoint_dict,
        }
        return self._msgpack.packb(envelope, use_bin_type=True)

    def deserialize(self, data: bytes) -> Any:
        """Deserialize MessagePack bytes to AgentCheckpoint."""
        from .checkpoint import AgentCheckpoint

        envelope = self._msgpack.unpackb(data, raw=False)

        schema_version = envelope.get("schema_version", 1)
        checkpoint_data = envelope.get("data", envelope)

        # Apply migrations if needed
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            checkpoint_data = migrate_checkpoint(
                checkpoint_data, schema_version, CHECKPOINT_SCHEMA_VERSION
            )

        return AgentCheckpoint.from_dict(checkpoint_data)
