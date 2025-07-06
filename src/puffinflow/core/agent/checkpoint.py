"""Checkpoint management for agents."""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

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
    priority_queue: List["PrioritizedState"]
    state_metadata: Dict[str, "StateMetadata"]
    running_states: Set[str]
    completed_states: Set[str]
    completed_once: Set[str]
    shared_state: Dict[str, Any]
    session_start: Optional[float]

    @classmethod
    def create_from_agent(cls, agent: "Agent") -> "AgentCheckpoint":
        """Create checkpoint from agent instance."""
        from copy import deepcopy

        # Handle missing session_start gracefully
        session_start = getattr(agent, 'session_start', None)

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
            session_start=session_start
        )
