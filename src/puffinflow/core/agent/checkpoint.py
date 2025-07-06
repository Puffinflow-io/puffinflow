"""Checkpoint management for agents."""

import time
from dataclasses import dataclass
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
        )
