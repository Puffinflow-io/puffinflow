"""Tests for checkpoint serializer with JSON/MsgPack + schema versioning."""

import json

import pytest

from puffinflow.core.agent.checkpoint import AgentCheckpoint
from puffinflow.core.agent.checkpoint_serializer import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointSerializer,
    JsonCheckpointSerializer,
    migrate_checkpoint,
    register_migration,
)
from puffinflow.core.agent.state import (
    AgentStatus,
    PrioritizedState,
    Priority,
    StateMetadata,
    StateStatus,
)


def _make_checkpoint() -> AgentCheckpoint:
    """Create a sample checkpoint for testing."""
    meta = StateMetadata(
        status=StateStatus.COMPLETED,
        attempts=1,
        max_retries=3,
        last_execution=100.0,
        last_success=100.0,
        state_id="test-state-id",
        priority=Priority.NORMAL,
    )
    ps = PrioritizedState(
        priority=-1,
        timestamp=100.0,
        state_name="step_one",
        metadata=meta,
    )
    return AgentCheckpoint(
        timestamp=1000.0,
        agent_name="test-agent",
        agent_status=AgentStatus.COMPLETED,
        priority_queue=[ps],
        state_metadata={"step_one": meta},
        running_states=set(),
        completed_states={"step_one"},
        completed_once={"step_one"},
        shared_state={"counter": 42, "name": "test"},
        session_start=999.0,
        schema_version=1,
    )


class TestAgentCheckpointDictMethods:
    """Test to_dict() and from_dict() on AgentCheckpoint."""

    def test_to_dict_returns_serializable(self):
        cp = _make_checkpoint()
        d = cp.to_dict()
        # Must be JSON-serializable
        json_str = json.dumps(d, default=str)
        assert isinstance(json_str, str)

    def test_to_dict_includes_schema_version(self):
        cp = _make_checkpoint()
        d = cp.to_dict()
        assert d["schema_version"] == 1

    def test_from_dict_round_trip(self):
        cp = _make_checkpoint()
        d = cp.to_dict()
        restored = AgentCheckpoint.from_dict(d)
        assert restored.agent_name == cp.agent_name
        assert restored.timestamp == cp.timestamp
        assert restored.agent_status == cp.agent_status
        assert restored.shared_state == cp.shared_state
        assert restored.completed_states == cp.completed_states
        assert restored.completed_once == cp.completed_once
        assert restored.running_states == cp.running_states
        assert restored.session_start == cp.session_start
        assert restored.schema_version == 1

    def test_from_dict_restores_state_metadata(self):
        cp = _make_checkpoint()
        d = cp.to_dict()
        restored = AgentCheckpoint.from_dict(d)
        assert "step_one" in restored.state_metadata
        meta = restored.state_metadata["step_one"]
        assert meta.status == StateStatus.COMPLETED
        assert meta.attempts == 1
        assert meta.priority == Priority.NORMAL

    def test_from_dict_restores_priority_queue(self):
        cp = _make_checkpoint()
        d = cp.to_dict()
        restored = AgentCheckpoint.from_dict(d)
        assert len(restored.priority_queue) == 1
        ps = restored.priority_queue[0]
        assert ps.state_name == "step_one"
        assert ps.metadata.status == StateStatus.COMPLETED


class TestJsonCheckpointSerializer:
    """Test JSON serializer."""

    def test_serialize_returns_bytes(self):
        serializer = JsonCheckpointSerializer()
        cp = _make_checkpoint()
        data = serializer.serialize(cp)
        assert isinstance(data, bytes)

    def test_serialize_produces_valid_json(self):
        serializer = JsonCheckpointSerializer()
        cp = _make_checkpoint()
        data = serializer.serialize(cp)
        parsed = json.loads(data.decode("utf-8"))
        assert "schema_version" in parsed
        assert "data" in parsed

    def test_serialize_includes_schema_version(self):
        serializer = JsonCheckpointSerializer()
        cp = _make_checkpoint()
        data = serializer.serialize(cp)
        parsed = json.loads(data.decode("utf-8"))
        assert parsed["schema_version"] == CHECKPOINT_SCHEMA_VERSION

    def test_deserialize_restores_checkpoint(self):
        serializer = JsonCheckpointSerializer()
        cp = _make_checkpoint()
        data = serializer.serialize(cp)
        restored = serializer.deserialize(data)
        assert isinstance(restored, AgentCheckpoint)
        assert restored.agent_name == "test-agent"
        assert restored.shared_state == {"counter": 42, "name": "test"}

    def test_round_trip(self):
        serializer = JsonCheckpointSerializer()
        cp = _make_checkpoint()
        data = serializer.serialize(cp)
        restored = serializer.deserialize(data)
        assert restored.agent_name == cp.agent_name
        assert restored.agent_status == cp.agent_status
        assert restored.completed_states == cp.completed_states
        assert restored.completed_once == cp.completed_once


class TestMsgpackCheckpointSerializer:
    """Test MsgPack serializer (skipped if msgpack not installed)."""

    @pytest.fixture
    def serializer(self):
        try:
            from puffinflow.core.agent.checkpoint_serializer import (
                MsgpackCheckpointSerializer,
            )

            return MsgpackCheckpointSerializer()
        except ImportError:
            pytest.skip("msgpack not installed")

    def test_serialize_returns_bytes(self, serializer):
        cp = _make_checkpoint()
        data = serializer.serialize(cp)
        assert isinstance(data, bytes)

    def test_round_trip(self, serializer):
        cp = _make_checkpoint()
        data = serializer.serialize(cp)
        restored = serializer.deserialize(data)
        assert isinstance(restored, AgentCheckpoint)
        assert restored.agent_name == "test-agent"
        assert restored.shared_state == {"counter": 42, "name": "test"}


class TestMigration:
    """Test schema migration system."""

    def test_migrate_same_version_is_noop(self):
        data = {"schema_version": 1, "foo": "bar"}
        result = migrate_checkpoint(data, 1, 1)
        assert result is data

    def test_migrate_raises_on_missing_path(self):
        with pytest.raises(ValueError, match="No migration path"):
            migrate_checkpoint({"schema_version": 99}, 99, 100)

    def test_register_and_apply_migration(self):
        @register_migration(100, 101)
        def _migrate_100_to_101(data):
            data["new_field"] = "added"
            return data

        data = {"schema_version": 100}
        result = migrate_checkpoint(data, 100, 101)
        assert result["new_field"] == "added"
        assert result["schema_version"] == 101


class TestCheckpointSerializerBase:
    """Test base CheckpointSerializer raises NotImplementedError."""

    def test_serialize_raises(self):
        s = CheckpointSerializer()
        with pytest.raises(NotImplementedError):
            s.serialize(None)

    def test_deserialize_raises(self):
        s = CheckpointSerializer()
        with pytest.raises(NotImplementedError):
            s.deserialize(b"")
