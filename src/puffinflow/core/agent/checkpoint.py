"""Checkpoint management for agents."""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .base import Agent
    from .state import (
        AgentStatus,
        PrioritizedState,
        StateMetadata,
    )


@dataclass
class AgentCheckpoint:
    """Checkpoint data for agent state."""

    timestamp: float
    agent_name: str
    agent_status: "AgentStatus"
    priority_queue: list["PrioritizedState"]
    state_metadata: dict[str, "StateMetadata"]
    running_states: set[str]
    completed_states: set[str]
    completed_once: set[str]
    shared_state: dict[str, Any]
    session_start: Optional[float]
    schema_version: int = field(default=1)

    @classmethod
    def create_from_agent(cls, agent: "Agent") -> "AgentCheckpoint":
        """Create checkpoint from agent instance."""
        from copy import deepcopy

        # Handle missing session_start gracefully
        session_start = getattr(agent, "session_start", None)

        return cls(
            timestamp=time.time(),
            agent_name=agent.name,
            agent_status=agent.status,
            priority_queue=deepcopy(agent.priority_queue),
            state_metadata=deepcopy(agent.state_metadata),
            running_states=set(agent.running_states),
            completed_states=set(agent.completed_states),
            completed_once=set(agent.completed_once),
            shared_state=deepcopy(agent.shared_state),
            session_start=session_start,
            schema_version=1,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to a JSON-serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "agent_status": self.agent_status.value
            if hasattr(self.agent_status, "value")
            else str(self.agent_status),
            "priority_queue": [
                {
                    "priority": ps.priority,
                    "timestamp": ps.timestamp,
                    "state_name": ps.state_name,
                    "metadata": {
                        "status": ps.metadata.status.value,
                        "attempts": ps.metadata.attempts,
                        "max_retries": ps.metadata.max_retries,
                        "last_execution": ps.metadata.last_execution,
                        "last_success": ps.metadata.last_success,
                        "state_id": ps.metadata.state_id,
                        "priority": ps.metadata.priority.value,
                    },
                }
                for ps in self.priority_queue
            ],
            "state_metadata": {
                k: {
                    "status": v.status.value,
                    "attempts": v.attempts,
                    "max_retries": v.max_retries,
                    "last_execution": v.last_execution,
                    "last_success": v.last_success,
                    "state_id": v.state_id,
                    "priority": v.priority.value,
                }
                for k, v in self.state_metadata.items()
            },
            "running_states": list(self.running_states),
            "completed_states": list(self.completed_states),
            "completed_once": list(self.completed_once),
            "shared_state": self.shared_state,
            "session_start": self.session_start,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentCheckpoint":
        """Reconstruct an AgentCheckpoint from a dictionary.

        Handles schema migration if schema_version differs from current.
        """
        from .state import (
            AgentStatus,
            PrioritizedState,
            Priority,
            StateMetadata,
            StateStatus,
        )

        schema_version = data.get("schema_version", 1)

        # Apply migrations if schema version differs
        if schema_version != 1:
            from .checkpoint_serializer import (
                CHECKPOINT_SCHEMA_VERSION,
                migrate_checkpoint,
            )

            data = migrate_checkpoint(data, schema_version, CHECKPOINT_SCHEMA_VERSION)

        return cls(
            timestamp=data["timestamp"],
            agent_name=data["agent_name"],
            agent_status=AgentStatus(data["agent_status"]),
            priority_queue=[
                PrioritizedState(
                    priority=ps["priority"],
                    timestamp=ps["timestamp"],
                    state_name=ps["state_name"],
                    metadata=StateMetadata(
                        status=StateStatus(ps["metadata"]["status"]),
                        attempts=ps["metadata"]["attempts"],
                        max_retries=ps["metadata"]["max_retries"],
                        last_execution=ps["metadata"]["last_execution"],
                        last_success=ps["metadata"]["last_success"],
                        state_id=ps["metadata"]["state_id"],
                        priority=Priority(ps["metadata"]["priority"]),
                    ),
                )
                for ps in data["priority_queue"]
            ],
            state_metadata={
                k: StateMetadata(
                    status=StateStatus(v["status"]),
                    attempts=v["attempts"],
                    max_retries=v["max_retries"],
                    last_execution=v["last_execution"],
                    last_success=v["last_success"],
                    state_id=v["state_id"],
                    priority=Priority(v["priority"]),
                )
                for k, v in data["state_metadata"].items()
            },
            running_states=set(data["running_states"]),
            completed_states=set(data["completed_states"]),
            completed_once=set(data["completed_once"]),
            shared_state=data["shared_state"],
            session_start=data["session_start"],
            schema_version=data.get("schema_version", 1),
        )
