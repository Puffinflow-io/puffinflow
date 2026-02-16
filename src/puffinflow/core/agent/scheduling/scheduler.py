"""Global scheduler and scheduled agent implementation."""

import asyncio
import contextlib
import logging
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional

from .exceptions import SchedulingError
from .inputs import ScheduledInput, parse_inputs
from .parser import ParsedSchedule, parse_schedule_string

if TYPE_CHECKING:
    from ..base import Agent, AgentResult

logger = logging.getLogger(__name__)


@dataclass
class ScheduledJob:
    """Represents a scheduled job."""

    job_id: str
    agent_ref: weakref.ReferenceType  # Weak reference to agent
    schedule: ParsedSchedule
    inputs: dict[str, ScheduledInput]
    next_run: float
    last_run: Optional[float] = None
    run_count: int = 0
    is_running: bool = False
    created_at: float = field(default_factory=time.time)

    @property
    def agent(self) -> Optional["Agent"]:
        """Get the agent if it still exists."""
        return self.agent_ref() if self.agent_ref else None

    def calculate_next_run(self) -> float:
        """Calculate the next run time based on schedule."""
        now = time.time()

        if self.schedule.schedule_type == "interval":
            if self.last_run is None:
                return now  # Run immediately for first time
            interval = self.schedule.interval_seconds or 60  # Default to 60 seconds
            return self.last_run + interval

        elif self.schedule.schedule_type == "cron":
            # For cron expressions, we'll use a simple approximation
            # In a production system, you'd use a proper cron library
            return self._calculate_next_cron_run(now)

        return now + 3600  # Default to 1 hour if unknown type

    def _calculate_next_cron_run(self, from_time: float) -> float:
        """Calculate next cron run time (simplified implementation)."""
        # This is a simplified cron calculator
        # For production, use a library like croniter

        cron = self.schedule.cron_expression
        if not cron:
            return from_time + 3600

        # Handle some common patterns
        if cron == "0 * * * *":  # hourly
            dt = datetime.fromtimestamp(from_time)
            next_dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            return next_dt.timestamp()

        elif cron == "0 0 * * *":  # daily at midnight
            dt = datetime.fromtimestamp(from_time)
            next_dt = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
                days=1
            )
            return next_dt.timestamp()

        elif cron.startswith("0 ") and " * * *" in cron:  # daily at specific hour
            parts = cron.split()
            if len(parts) >= 2:
                try:
                    hour = int(parts[1])
                    dt = datetime.fromtimestamp(from_time)
                    next_dt = dt.replace(hour=hour, minute=0, second=0, microsecond=0)
                    if next_dt <= dt:
                        next_dt += timedelta(days=1)
                    return next_dt.timestamp()
                except ValueError:
                    pass

        # Default fallback
        return from_time + 3600


class ScheduledAgent:
    """Represents a scheduled agent execution."""

    def __init__(
        self,
        job_id: str,
        agent: "Agent",
        schedule: ParsedSchedule,
        inputs: dict[str, ScheduledInput],
    ):
        self.job_id = job_id
        self.agent = agent
        self.schedule = schedule
        self.inputs = inputs
        self.created_at = time.time()

    def cancel(self) -> bool:
        """Cancel this scheduled execution.

        Returns:
            True if successfully cancelled
        """
        return GlobalScheduler.get_instance_sync().cancel_job(self.job_id)

    def get_next_run_time(self) -> Optional[datetime]:
        """Get the next scheduled run time.

        Returns:
            Next run time as datetime, or None if not found
        """
        scheduler = GlobalScheduler.get_instance_sync()
        job = scheduler._jobs.get(self.job_id)
        if job:
            return datetime.fromtimestamp(job.next_run)
        return None

    def get_run_count(self) -> int:
        """Get the number of times this job has run.

        Returns:
            Run count
        """
        scheduler = GlobalScheduler.get_instance_sync()
        job = scheduler._jobs.get(self.job_id)
        return job.run_count if job else 0


class GlobalScheduler:
    """Global scheduler for managing scheduled agent executions."""

    _instance: Optional["GlobalScheduler"] = None
    _lock = asyncio.Lock()

    def __init__(self) -> None:
        self._jobs: dict[str, ScheduledJob] = {}
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._check_interval = 10.0  # Check every 10 seconds
        self._job_counter = 0

    @classmethod
    async def get_instance(cls) -> "GlobalScheduler":
        """Get or create the global scheduler instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance.start()
        return cls._instance

    @classmethod
    def get_instance_sync(cls) -> "GlobalScheduler":
        """Get the global scheduler instance synchronously (for non-async contexts)."""
        if cls._instance is None:
            cls._instance = cls()
            # Try to start scheduler if event loop is available
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create task but don't need to store reference as it's just startup
                    task = asyncio.create_task(cls._instance.start())
                    task.add_done_callback(lambda t: None)  # Prevent warnings
            except RuntimeError:
                # No event loop running, scheduler will start when needed
                pass
        return cls._instance

    async def start(self) -> None:
        """Start the scheduler background task."""
        if not self._running:
            self._running = True
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            logger.info("Global scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler background task."""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._scheduler_task
            self._scheduler_task = None
        logger.info("Global scheduler stopped")

    def schedule_agent(
        self, agent: "Agent", schedule_string: str, **inputs: Any
    ) -> ScheduledAgent:
        """Schedule an agent for execution.

        Args:
            agent: Agent to schedule
            schedule_string: Schedule string (natural language or cron)
            **inputs: Input parameters with magic prefixes

        Returns:
            ScheduledAgent instance

        Raises:
            SchedulingError: If scheduling fails
        """
        try:
            # Parse schedule
            parsed_schedule = parse_schedule_string(schedule_string)

            # Parse inputs
            parsed_inputs = parse_inputs(**inputs)

            # Generate job ID
            self._job_counter += 1
            job_id = f"{agent.name}_{self._job_counter}_{int(time.time())}"

            # Create job
            job = ScheduledJob(
                job_id=job_id,
                agent_ref=weakref.ref(agent),
                schedule=parsed_schedule,
                inputs=parsed_inputs,
                next_run=time.time(),  # Will be recalculated
            )
            job.next_run = job.calculate_next_run()

            # Store job
            self._jobs[job_id] = job

            # Start scheduler if not running
            if not self._running:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create task but don't need to store reference as it's just startup
                        task = asyncio.create_task(self.start())
                        task.add_done_callback(lambda t: None)  # Prevent warnings
                except RuntimeError:
                    # No event loop, will start when one is available
                    pass

            logger.info(
                f"Scheduled agent {agent.name} with schedule '{schedule_string}' (job_id: {job_id})"
            )

            return ScheduledAgent(job_id, agent, parsed_schedule, parsed_inputs)

        except Exception as e:
            raise SchedulingError(f"Failed to schedule agent {agent.name}: {e}") from e

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a scheduled job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if job was cancelled
        """
        if job_id in self._jobs:
            del self._jobs[job_id]
            logger.info(f"Cancelled scheduled job {job_id}")
            return True
        return False

    def get_scheduled_jobs(
        self, agent_name: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Get information about scheduled jobs.

        Args:
            agent_name: Filter by agent name (optional)

        Returns:
            List of job information dictionaries
        """
        jobs = []
        for job_id, job in self._jobs.items():
            agent = job.agent
            if agent is None:
                continue  # Agent was garbage collected

            if agent_name and agent.name != agent_name:
                continue

            jobs.append(
                {
                    "job_id": job_id,
                    "agent_name": agent.name,
                    "schedule_description": job.schedule.description,
                    "next_run": datetime.fromtimestamp(job.next_run),
                    "last_run": (
                        datetime.fromtimestamp(job.last_run) if job.last_run else None
                    ),
                    "run_count": job.run_count,
                    "is_running": job.is_running,
                    "created_at": datetime.fromtimestamp(job.created_at),
                }
            )

        return jobs

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        logger.info("Scheduler loop started")

        while self._running:
            try:
                await self._check_and_run_jobs()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(self._check_interval)

        logger.info("Scheduler loop stopped")

    async def _check_and_run_jobs(self) -> None:
        """Check for jobs that need to run and execute them."""
        now = time.time()
        jobs_to_run = []

        # Find jobs that need to run
        for job_id, job in list(self._jobs.items()):
            agent = job.agent
            if agent is None:
                # Agent was garbage collected, remove job
                del self._jobs[job_id]
                continue

            if not job.is_running and job.next_run <= now:
                jobs_to_run.append(job)

        # Run jobs
        for job in jobs_to_run:
            # Create job execution task and store reference to prevent garbage collection
            task = asyncio.create_task(self._run_job(job))
            if not hasattr(self, "_background_tasks"):
                self._background_tasks = set()
            self._background_tasks.add(task)
            task.add_done_callback(lambda t: self._background_tasks.discard(t))

    async def _run_job(self, job: ScheduledJob) -> None:
        """Run a scheduled job.

        Args:
            job: Job to run
        """
        agent = job.agent
        if agent is None:
            return

        job.is_running = True
        job.last_run = time.time()
        job.run_count += 1

        try:
            logger.info(f"Running scheduled job {job.job_id} for agent {agent.name}")

            # Create context with scheduled inputs
            context = agent._create_context(agent.shared_state)

            # Apply scheduled inputs to context
            for scheduled_input in job.inputs.values():
                scheduled_input.apply_to_context(context)

            # Run the agent
            result: AgentResult = await agent.run()

            logger.info(
                f"Completed scheduled job {job.job_id} for agent {agent.name} (status: {result.status})"
            )

        except Exception as e:
            logger.error(
                f"Error running scheduled job {job.job_id} for agent {agent.name}: {e}"
            )

        finally:
            job.is_running = False
            # Calculate next run time
            job.next_run = job.calculate_next_run()

    def cleanup_dead_jobs(self) -> int:
        """Remove jobs for agents that have been garbage collected.

        Returns:
            Number of jobs cleaned up
        """
        dead_jobs = []
        for job_id, job in self._jobs.items():
            if job.agent is None:
                dead_jobs.append(job_id)

        for job_id in dead_jobs:
            del self._jobs[job_id]

        if dead_jobs:
            logger.info(f"Cleaned up {len(dead_jobs)} dead scheduled jobs")

        return len(dead_jobs)
