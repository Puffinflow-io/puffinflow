"""Graceful shutdown drain protocol for running agents."""

import asyncio
import logging
import platform
import time
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .base import Agent

logger = logging.getLogger(__name__)


@dataclass
class DrainResult:
    """Result of a drain operation."""

    agents_drained: list[str] = field(default_factory=list)
    agents_timed_out: list[str] = field(default_factory=list)
    checkpoints_saved: list[str] = field(default_factory=list)
    duration: float = 0.0


class DrainProtocol:
    """Manages graceful shutdown of running agents.

    When drain is initiated:
    1. Sets draining flag — agents stop accepting new work
    2. Waits for running states to complete (up to timeout)
    3. Checkpoints all agents
    4. Cancels remaining work
    5. Returns DrainResult with status per agent
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._draining = False
        self._drain_event = asyncio.Event()
        self._active_agents: weakref.WeakSet = weakref.WeakSet()
        self._timeout = timeout

    def register(self, agent: "Agent") -> None:
        """Register an agent for drain management."""
        self._active_agents.add(agent)

    def unregister(self, agent: "Agent") -> None:
        """Unregister an agent from drain management."""
        self._active_agents.discard(agent)

    @property
    def is_draining(self) -> bool:
        """Whether a drain is currently in progress."""
        return self._draining

    async def drain(self) -> DrainResult:
        """Initiate graceful shutdown of all registered agents.

        Returns:
            DrainResult with per-agent status information.
        """
        if self._draining:
            logger.warning("Drain already in progress")
            return DrainResult()

        self._draining = True
        self._drain_event.set()
        start_time = time.time()

        result = DrainResult()

        # Snapshot current agents (WeakSet may change during iteration)
        agents = list(self._active_agents)

        if not agents:
            result.duration = time.time() - start_time
            return result

        logger.info(
            "Draining %d agent(s) with timeout=%.1fs", len(agents), self._timeout
        )

        # Wait for agents to reach a safe checkpoint, up to timeout
        for agent in agents:
            agent_name = agent.name
            try:
                # Give agent time to finish current state
                deadline = start_time + self._timeout
                remaining = deadline - time.time()

                if remaining <= 0:
                    result.agents_timed_out.append(agent_name)
                    logger.warning("Agent %s: drain timed out immediately", agent_name)
                    continue

                # Wait for the agent to not be actively running a state
                waited = 0.0
                poll_interval = 0.1
                while agent.running_states and waited < remaining:
                    await asyncio.sleep(poll_interval)
                    waited += poll_interval
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break

                if agent.running_states:
                    result.agents_timed_out.append(agent_name)
                    logger.warning(
                        "Agent %s: timed out with running states: %s",
                        agent_name,
                        agent.running_states,
                    )
                else:
                    result.agents_drained.append(agent_name)

                # Checkpoint the agent regardless of timeout
                try:
                    if agent._checkpoint_storage is not None:
                        from .checkpoint import AgentCheckpoint

                        checkpoint = AgentCheckpoint.create_from_agent(agent)
                        checkpoint_id = await agent.checkpoint_storage.save_checkpoint(
                            agent_name, checkpoint
                        )
                        result.checkpoints_saved.append(checkpoint_id)
                        logger.info(
                            "Agent %s: checkpoint saved as %s",
                            agent_name,
                            checkpoint_id,
                        )
                except Exception as exc:
                    logger.error(
                        "Agent %s: failed to save checkpoint: %s", agent_name, exc
                    )

            except Exception as exc:
                logger.error("Agent %s: drain error: %s", agent_name, exc)
                result.agents_timed_out.append(agent_name)

        result.duration = time.time() - start_time
        logger.info(
            "Drain completed in %.2fs: %d drained, %d timed out, %d checkpoints",
            result.duration,
            len(result.agents_drained),
            len(result.agents_timed_out),
            len(result.checkpoints_saved),
        )

        return result

    def install_signal_handlers(self) -> None:
        """Register SIGTERM/SIGINT handlers that trigger drain.

        On Windows, only SIGINT (Ctrl+C) is supported via signal module.
        On Unix, both SIGTERM and SIGINT are registered via the event loop.
        """
        import signal

        if platform.system() == "Windows":
            # Windows: use signal module directly (loop.add_signal_handler not supported)
            original_handler = signal.getsignal(signal.SIGINT)

            def _win_handler(signum: int, frame: Any) -> None:
                logger.info("Received signal %d, initiating drain", signum)
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.drain())
                    else:
                        loop.run_until_complete(self.drain())
                except Exception:
                    pass
                # Restore original handler for double-ctrl-c force quit
                signal.signal(signal.SIGINT, original_handler)

            signal.signal(signal.SIGINT, _win_handler)
        else:
            # Unix: use event loop signal handlers
            try:
                loop = asyncio.get_event_loop()
                for sig in (signal.SIGTERM, signal.SIGINT):
                    loop.add_signal_handler(
                        sig,
                        lambda: asyncio.ensure_future(self.drain()),
                    )
            except (NotImplementedError, RuntimeError):
                # Fallback if loop doesn't support signal handlers
                logger.warning(
                    "Could not install signal handlers on current event loop"
                )

    def reset(self) -> None:
        """Reset drain state (for testing or re-use)."""
        self._draining = False
        self._drain_event.clear()
