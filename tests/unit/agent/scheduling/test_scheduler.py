"""Tests for global scheduler and scheduled agent implementation."""

import asyncio
import time
import weakref
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from puffinflow.core.agent.scheduling.exceptions import SchedulingError
from puffinflow.core.agent.scheduling.inputs import InputType, ScheduledInput
from puffinflow.core.agent.scheduling.parser import ParsedSchedule
from puffinflow.core.agent.scheduling.scheduler import (
    GlobalScheduler,
    ScheduledAgent,
    ScheduledJob,
)


class TestScheduledJob:
    """Test ScheduledJob dataclass."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_agent = Mock()
        self.mock_agent.name = "test_agent"
        self.agent_ref = weakref.ref(self.mock_agent)
        self.schedule = ParsedSchedule("interval", interval_seconds=300)
        self.inputs = {"key": ScheduledInput("key", "value", InputType.VARIABLE)}
        self.next_run = time.time() + 300

        self.job = ScheduledJob(
            job_id="test_job",
            agent_ref=self.agent_ref,
            schedule=self.schedule,
            inputs=self.inputs,
            next_run=self.next_run,
        )

    def test_scheduled_job_creation(self):
        """Test ScheduledJob creation with required fields."""
        assert self.job.job_id == "test_job"
        assert self.job.agent_ref == self.agent_ref
        assert self.job.schedule == self.schedule
        assert self.job.inputs == self.inputs
        assert self.job.next_run == self.next_run
        assert self.job.last_run is None
        assert self.job.run_count == 0
        assert self.job.is_running is False
        assert isinstance(self.job.created_at, float)

    def test_agent_property_valid_reference(self):
        """Test agent property returns agent when reference is valid."""
        assert self.job.agent == self.mock_agent

    def test_agent_property_dead_reference(self):
        """Test agent property returns None when reference is dead."""
        # Delete the agent to make the weak reference dead
        del self.mock_agent
        assert self.job.agent is None

    def test_calculate_next_run_interval_first_time(self):
        """Test calculate_next_run for interval schedule on first run."""
        job = ScheduledJob(
            job_id="test",
            agent_ref=self.agent_ref,
            schedule=ParsedSchedule("interval", interval_seconds=300),
            inputs={},
            next_run=0,
            last_run=None,
        )

        now = time.time()
        next_run = job.calculate_next_run()

        # Should run immediately for first time
        assert abs(next_run - now) < 1.0

    def test_calculate_next_run_interval_subsequent(self):
        """Test calculate_next_run for interval schedule on subsequent runs."""
        last_run = time.time() - 100
        job = ScheduledJob(
            job_id="test",
            agent_ref=self.agent_ref,
            schedule=ParsedSchedule("interval", interval_seconds=300),
            inputs={},
            next_run=0,
            last_run=last_run,
        )

        next_run = job.calculate_next_run()
        expected = last_run + 300

        assert abs(next_run - expected) < 1.0

    def test_calculate_next_run_cron_hourly(self):
        """Test calculate_next_run for cron schedule (hourly)."""
        job = ScheduledJob(
            job_id="test",
            agent_ref=self.agent_ref,
            schedule=ParsedSchedule("cron", cron_expression="0 * * * *"),
            inputs={},
            next_run=0,
        )

        now = time.time()
        next_run = job.calculate_next_run()

        # Should be scheduled for next hour
        assert next_run > now

        # Convert to datetime to check it's at the top of an hour
        next_dt = datetime.fromtimestamp(next_run)
        assert next_dt.minute == 0
        assert next_dt.second == 0

    def test_calculate_next_run_cron_daily(self):
        """Test calculate_next_run for cron schedule (daily)."""
        job = ScheduledJob(
            job_id="test",
            agent_ref=self.agent_ref,
            schedule=ParsedSchedule("cron", cron_expression="0 0 * * *"),
            inputs={},
            next_run=0,
        )

        now = time.time()
        next_run = job.calculate_next_run()

        # Should be scheduled for next midnight
        assert next_run > now

        next_dt = datetime.fromtimestamp(next_run)
        assert next_dt.hour == 0
        assert next_dt.minute == 0
        assert next_dt.second == 0

    def test_calculate_next_run_cron_daily_specific_hour(self):
        """Test calculate_next_run for cron schedule (daily at specific hour)."""
        job = ScheduledJob(
            job_id="test",
            agent_ref=self.agent_ref,
            schedule=ParsedSchedule("cron", cron_expression="0 14 * * *"),
            inputs={},
            next_run=0,
        )

        now = time.time()
        next_run = job.calculate_next_run()

        # Should be scheduled for next 2 PM
        assert next_run > now

        next_dt = datetime.fromtimestamp(next_run)
        assert next_dt.hour == 14
        assert next_dt.minute == 0
        assert next_dt.second == 0

    def test_calculate_next_run_unknown_type(self):
        """Test calculate_next_run for unknown schedule type."""
        job = ScheduledJob(
            job_id="test",
            agent_ref=self.agent_ref,
            schedule=ParsedSchedule("unknown"),
            inputs={},
            next_run=0,
        )

        now = time.time()
        next_run = job.calculate_next_run()

        # Should default to 1 hour
        expected = now + 3600
        assert abs(next_run - expected) < 1.0

    def test_calculate_next_cron_run_invalid_expression(self):
        """Test _calculate_next_cron_run with invalid expression."""
        job = ScheduledJob(
            job_id="test",
            agent_ref=self.agent_ref,
            schedule=ParsedSchedule("cron", cron_expression="invalid"),
            inputs={},
            next_run=0,
        )

        now = time.time()
        next_run = job._calculate_next_cron_run(now)

        # Should default to 1 hour
        expected = now + 3600
        assert abs(next_run - expected) < 1.0


class TestScheduledAgent:
    """Test ScheduledAgent class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_agent = Mock()
        self.mock_agent.name = "test_agent"
        self.schedule = ParsedSchedule("interval", interval_seconds=300)
        self.inputs = {"key": ScheduledInput("key", "value", InputType.VARIABLE)}

        self.scheduled_agent = ScheduledAgent(
            "test_job", self.mock_agent, self.schedule, self.inputs
        )

    def test_scheduled_agent_creation(self):
        """Test ScheduledAgent creation."""
        assert self.scheduled_agent.job_id == "test_job"
        assert self.scheduled_agent.agent == self.mock_agent
        assert self.scheduled_agent.schedule == self.schedule
        assert self.scheduled_agent.inputs == self.inputs
        assert isinstance(self.scheduled_agent.created_at, float)

    @patch(
        "puffinflow.core.agent.scheduling.scheduler.GlobalScheduler.get_instance_sync"
    )
    def test_cancel(self, mock_get_instance):
        """Test cancel method."""
        mock_scheduler = Mock()
        mock_scheduler.cancel_job.return_value = True
        mock_get_instance.return_value = mock_scheduler

        result = self.scheduled_agent.cancel()

        assert result is True
        mock_scheduler.cancel_job.assert_called_once_with("test_job")

    @patch(
        "puffinflow.core.agent.scheduling.scheduler.GlobalScheduler.get_instance_sync"
    )
    def test_get_next_run_time(self, mock_get_instance):
        """Test get_next_run_time method."""
        mock_scheduler = Mock()
        mock_job = Mock()
        next_run_timestamp = time.time() + 300
        mock_job.next_run = next_run_timestamp
        mock_scheduler._jobs = {"test_job": mock_job}
        mock_get_instance.return_value = mock_scheduler

        result = self.scheduled_agent.get_next_run_time()

        assert isinstance(result, datetime)
        assert abs(result.timestamp() - next_run_timestamp) < 1.0

    @patch(
        "puffinflow.core.agent.scheduling.scheduler.GlobalScheduler.get_instance_sync"
    )
    def test_get_next_run_time_job_not_found(self, mock_get_instance):
        """Test get_next_run_time when job is not found."""
        mock_scheduler = Mock()
        mock_scheduler._jobs = {}
        mock_get_instance.return_value = mock_scheduler

        result = self.scheduled_agent.get_next_run_time()

        assert result is None

    @patch(
        "puffinflow.core.agent.scheduling.scheduler.GlobalScheduler.get_instance_sync"
    )
    def test_get_run_count(self, mock_get_instance):
        """Test get_run_count method."""
        mock_scheduler = Mock()
        mock_job = Mock()
        mock_job.run_count = 5
        mock_scheduler._jobs = {"test_job": mock_job}
        mock_get_instance.return_value = mock_scheduler

        result = self.scheduled_agent.get_run_count()

        assert result == 5

    @patch(
        "puffinflow.core.agent.scheduling.scheduler.GlobalScheduler.get_instance_sync"
    )
    def test_get_run_count_job_not_found(self, mock_get_instance):
        """Test get_run_count when job is not found."""
        mock_scheduler = Mock()
        mock_scheduler._jobs = {}
        mock_get_instance.return_value = mock_scheduler

        result = self.scheduled_agent.get_run_count()

        assert result == 0


class TestGlobalScheduler:
    """Test GlobalScheduler class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset singleton instance
        GlobalScheduler._instance = None
        self.scheduler = GlobalScheduler()

    def teardown_method(self):
        """Clean up after tests."""
        # Reset singleton instance
        GlobalScheduler._instance = None

    def test_global_scheduler_creation(self):
        """Test GlobalScheduler creation."""
        assert self.scheduler._jobs == {}
        assert self.scheduler._running is False
        assert self.scheduler._scheduler_task is None
        assert self.scheduler._check_interval == 10.0
        assert self.scheduler._job_counter == 0

    @pytest.mark.asyncio
    async def test_get_instance_singleton(self):
        """Test get_instance returns singleton."""
        instance1 = await GlobalScheduler.get_instance()
        instance2 = await GlobalScheduler.get_instance()

        assert instance1 is instance2
        assert GlobalScheduler._instance is instance1

    def test_get_instance_sync(self):
        """Test get_instance_sync returns singleton."""
        instance1 = GlobalScheduler.get_instance_sync()
        instance2 = GlobalScheduler.get_instance_sync()

        assert instance1 is instance2
        assert GlobalScheduler._instance is instance1

    @pytest.mark.asyncio
    async def test_start_scheduler(self):
        """Test starting the scheduler."""
        await self.scheduler.start()

        assert self.scheduler._running is True
        assert self.scheduler._scheduler_task is not None
        assert isinstance(self.scheduler._scheduler_task, asyncio.Task)

        # Clean up
        await self.scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_scheduler(self):
        """Test stopping the scheduler."""
        await self.scheduler.start()
        assert self.scheduler._running is True

        await self.scheduler.stop()

        assert self.scheduler._running is False
        assert self.scheduler._scheduler_task is None

    @patch("puffinflow.core.agent.scheduling.scheduler.parse_schedule_string")
    @patch("puffinflow.core.agent.scheduling.scheduler.parse_inputs")
    def test_schedule_agent(self, mock_parse_inputs, mock_parse_schedule):
        """Test schedule_agent method."""
        # Set up mocks
        mock_agent = Mock()
        mock_agent.name = "test_agent"

        mock_schedule = ParsedSchedule("interval", interval_seconds=300)
        mock_parse_schedule.return_value = mock_schedule

        mock_inputs = {"key": ScheduledInput("key", "value", InputType.VARIABLE)}
        mock_parse_inputs.return_value = mock_inputs

        # Schedule the agent
        result = self.scheduler.schedule_agent(
            mock_agent, "every 5 minutes", key="value"
        )

        # Verify result
        assert isinstance(result, ScheduledAgent)
        assert result.agent == mock_agent
        assert result.schedule == mock_schedule
        assert result.inputs == mock_inputs

        # Verify job was stored
        assert len(self.scheduler._jobs) == 1
        job_id = next(iter(self.scheduler._jobs.keys()))
        job = self.scheduler._jobs[job_id]

        assert job.agent == mock_agent
        assert job.schedule == mock_schedule
        assert job.inputs == mock_inputs

        # Verify parsing was called
        mock_parse_schedule.assert_called_once_with("every 5 minutes")
        mock_parse_inputs.assert_called_once_with(key="value")

    @patch("puffinflow.core.agent.scheduling.scheduler.parse_schedule_string")
    def test_schedule_agent_error(self, mock_parse_schedule):
        """Test schedule_agent with parsing error."""
        mock_agent = Mock()
        mock_agent.name = "test_agent"

        mock_parse_schedule.side_effect = Exception("Parse error")

        with pytest.raises(SchedulingError) as exc_info:
            self.scheduler.schedule_agent(mock_agent, "invalid schedule")

        assert "Failed to schedule agent test_agent" in str(exc_info.value)
        assert "Parse error" in str(exc_info.value)

    def test_cancel_job_existing(self):
        """Test cancel_job with existing job."""
        # Add a job
        mock_agent = Mock()
        job = ScheduledJob(
            job_id="test_job",
            agent_ref=weakref.ref(mock_agent),
            schedule=ParsedSchedule("interval", interval_seconds=300),
            inputs={},
            next_run=time.time() + 300,
        )
        self.scheduler._jobs["test_job"] = job

        result = self.scheduler.cancel_job("test_job")

        assert result is True
        assert "test_job" not in self.scheduler._jobs

    def test_cancel_job_nonexistent(self):
        """Test cancel_job with non-existent job."""
        result = self.scheduler.cancel_job("nonexistent_job")

        assert result is False

    def test_get_scheduled_jobs_empty(self):
        """Test get_scheduled_jobs with no jobs."""
        result = self.scheduler.get_scheduled_jobs()

        assert result == []

    def test_get_scheduled_jobs_with_jobs(self):
        """Test get_scheduled_jobs with jobs."""
        # Add jobs
        mock_agent1 = Mock()
        mock_agent1.name = "agent1"
        mock_agent2 = Mock()
        mock_agent2.name = "agent2"

        job1 = ScheduledJob(
            job_id="job1",
            agent_ref=weakref.ref(mock_agent1),
            schedule=ParsedSchedule(
                "interval", interval_seconds=300, description="Every 5 minutes"
            ),
            inputs={},
            next_run=time.time() + 300,
            last_run=time.time() - 100,
            run_count=3,
        )

        job2 = ScheduledJob(
            job_id="job2",
            agent_ref=weakref.ref(mock_agent2),
            schedule=ParsedSchedule(
                "cron", cron_expression="0 * * * *", description="Every hour"
            ),
            inputs={},
            next_run=time.time() + 600,
            run_count=1,
        )

        self.scheduler._jobs["job1"] = job1
        self.scheduler._jobs["job2"] = job2

        result = self.scheduler.get_scheduled_jobs()

        assert len(result) == 2

        # Check job1
        job1_info = next(j for j in result if j["job_id"] == "job1")
        assert job1_info["agent_name"] == "agent1"
        assert job1_info["schedule_description"] == "Every 5 minutes"
        assert job1_info["run_count"] == 3
        assert job1_info["last_run"] is not None

        # Check job2
        job2_info = next(j for j in result if j["job_id"] == "job2")
        assert job2_info["agent_name"] == "agent2"
        assert job2_info["schedule_description"] == "Every hour"
        assert job2_info["run_count"] == 1
        assert job2_info["last_run"] is None

    def test_get_scheduled_jobs_filtered_by_agent(self):
        """Test get_scheduled_jobs filtered by agent name."""
        # Add jobs for different agents
        mock_agent1 = Mock()
        mock_agent1.name = "agent1"
        mock_agent2 = Mock()
        mock_agent2.name = "agent2"

        job1 = ScheduledJob(
            job_id="job1",
            agent_ref=weakref.ref(mock_agent1),
            schedule=ParsedSchedule("interval", interval_seconds=300),
            inputs={},
            next_run=time.time() + 300,
        )

        job2 = ScheduledJob(
            job_id="job2",
            agent_ref=weakref.ref(mock_agent2),
            schedule=ParsedSchedule("interval", interval_seconds=600),
            inputs={},
            next_run=time.time() + 600,
        )

        self.scheduler._jobs["job1"] = job1
        self.scheduler._jobs["job2"] = job2

        result = self.scheduler.get_scheduled_jobs(agent_name="agent1")

        assert len(result) == 1
        assert result[0]["agent_name"] == "agent1"
        assert result[0]["job_id"] == "job1"

    def test_get_scheduled_jobs_dead_agent_reference(self):
        """Test get_scheduled_jobs skips jobs with dead agent references."""
        mock_agent = Mock()
        mock_agent.name = "agent1"

        job = ScheduledJob(
            job_id="job1",
            agent_ref=weakref.ref(mock_agent),
            schedule=ParsedSchedule("interval", interval_seconds=300),
            inputs={},
            next_run=time.time() + 300,
        )

        self.scheduler._jobs["job1"] = job

        # Delete the agent to make the reference dead
        del mock_agent

        result = self.scheduler.get_scheduled_jobs()

        assert result == []

    @pytest.mark.asyncio
    async def test_check_and_run_jobs_no_jobs(self):
        """Test _check_and_run_jobs with no jobs."""
        # Should not raise any errors
        await self.scheduler._check_and_run_jobs()

    @pytest.mark.asyncio
    async def test_check_and_run_jobs_not_ready(self):
        """Test _check_and_run_jobs with jobs not ready to run."""
        mock_agent = Mock()
        job = ScheduledJob(
            job_id="job1",
            agent_ref=weakref.ref(mock_agent),
            schedule=ParsedSchedule("interval", interval_seconds=300),
            inputs={},
            next_run=time.time() + 1000,  # Far in the future
        )

        self.scheduler._jobs["job1"] = job

        with patch.object(self.scheduler, "_run_job") as mock_run_job:
            await self.scheduler._check_and_run_jobs()
            mock_run_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_and_run_jobs_ready_to_run(self):
        """Test _check_and_run_jobs with jobs ready to run."""
        mock_agent = Mock()
        job = ScheduledJob(
            job_id="job1",
            agent_ref=weakref.ref(mock_agent),
            schedule=ParsedSchedule("interval", interval_seconds=300),
            inputs={},
            next_run=time.time() - 10,  # In the past
        )

        self.scheduler._jobs["job1"] = job

        with patch.object(self.scheduler, "_run_job") as mock_run_job:
            await self.scheduler._check_and_run_jobs()
            mock_run_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_and_run_jobs_dead_agent(self):
        """Test _check_and_run_jobs removes jobs with dead agents."""
        mock_agent = Mock()
        job = ScheduledJob(
            job_id="job1",
            agent_ref=weakref.ref(mock_agent),
            schedule=ParsedSchedule("interval", interval_seconds=300),
            inputs={},
            next_run=time.time() - 10,
        )

        self.scheduler._jobs["job1"] = job

        # Delete the agent
        del mock_agent

        await self.scheduler._check_and_run_jobs()

        # Job should be removed
        assert "job1" not in self.scheduler._jobs

    @pytest.mark.asyncio
    async def test_run_job_success(self):
        """Test _run_job with successful execution."""
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        mock_agent._create_context = Mock()
        mock_agent.run = AsyncMock()

        mock_context = Mock()
        mock_agent._create_context.return_value = mock_context

        mock_result = Mock()
        mock_result.status = "success"
        mock_agent.run.return_value = mock_result

        scheduled_input = ScheduledInput("key", "value", InputType.VARIABLE)
        scheduled_input.apply_to_context = Mock()

        job = ScheduledJob(
            job_id="job1",
            agent_ref=weakref.ref(mock_agent),
            schedule=ParsedSchedule("interval", interval_seconds=300),
            inputs={"key": scheduled_input},
            next_run=time.time(),
        )

        await self.scheduler._run_job(job)

        # Verify job state was updated
        assert job.is_running is False  # Should be reset after completion
        assert job.last_run is not None
        assert job.run_count == 1
        assert job.next_run > time.time()  # Should be recalculated

        # Verify agent was called
        mock_agent._create_context.assert_called_once()
        scheduled_input.apply_to_context.assert_called_once_with(mock_context)
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_job_dead_agent(self):
        """Test _run_job with dead agent reference."""
        mock_agent = Mock()
        job = ScheduledJob(
            job_id="job1",
            agent_ref=weakref.ref(mock_agent),
            schedule=ParsedSchedule("interval", interval_seconds=300),
            inputs={},
            next_run=time.time(),
        )

        # Delete the agent
        del mock_agent

        # Should not raise error
        await self.scheduler._run_job(job)

    @pytest.mark.asyncio
    async def test_run_job_exception(self):
        """Test _run_job with exception during execution."""
        mock_agent = Mock()
        mock_agent.name = "test_agent"
        mock_agent._create_context = Mock()
        mock_agent.run = AsyncMock()
        mock_agent.run.side_effect = Exception("Test error")

        job = ScheduledJob(
            job_id="job1",
            agent_ref=weakref.ref(mock_agent),
            schedule=ParsedSchedule("interval", interval_seconds=300),
            inputs={},
            next_run=time.time(),
        )

        # Should not raise error, but should handle it gracefully
        await self.scheduler._run_job(job)

        # Job state should still be updated
        assert job.is_running is False
        assert job.last_run is not None
        assert job.run_count == 1

    def test_cleanup_dead_jobs(self):
        """Test cleanup_dead_jobs method."""
        # Add jobs with live and dead agents
        live_agent = Mock()
        dead_agent = Mock()

        live_job = ScheduledJob(
            job_id="live_job",
            agent_ref=weakref.ref(live_agent),
            schedule=ParsedSchedule("interval", interval_seconds=300),
            inputs={},
            next_run=time.time() + 300,
        )

        dead_job = ScheduledJob(
            job_id="dead_job",
            agent_ref=weakref.ref(dead_agent),
            schedule=ParsedSchedule("interval", interval_seconds=300),
            inputs={},
            next_run=time.time() + 300,
        )

        self.scheduler._jobs["live_job"] = live_job
        self.scheduler._jobs["dead_job"] = dead_job

        # Delete the dead agent
        del dead_agent

        result = self.scheduler.cleanup_dead_jobs()

        assert result == 1  # One job cleaned up
        assert "live_job" in self.scheduler._jobs
        assert "dead_job" not in self.scheduler._jobs

    def test_cleanup_dead_jobs_no_dead_jobs(self):
        """Test cleanup_dead_jobs with no dead jobs."""
        live_agent = Mock()

        live_job = ScheduledJob(
            job_id="live_job",
            agent_ref=weakref.ref(live_agent),
            schedule=ParsedSchedule("interval", interval_seconds=300),
            inputs={},
            next_run=time.time() + 300,
        )

        self.scheduler._jobs["live_job"] = live_job

        result = self.scheduler.cleanup_dead_jobs()

        assert result == 0
        assert "live_job" in self.scheduler._jobs


class TestSchedulerIntegration:
    """Integration tests for scheduler components."""

    def teardown_method(self):
        """Clean up after tests."""
        GlobalScheduler._instance = None

    @pytest.mark.asyncio
    async def test_full_scheduling_workflow(self):
        """Test complete scheduling workflow."""
        # Create scheduler
        scheduler = await GlobalScheduler.get_instance()

        # Create mock agent
        mock_agent = Mock()
        mock_agent.name = "integration_test_agent"
        mock_agent._create_context = Mock()
        mock_agent.run = AsyncMock()
        mock_agent.shared_state = {}

        mock_context = Mock()
        mock_agent._create_context.return_value = mock_context

        mock_result = Mock()
        mock_result.status = "success"
        mock_agent.run.return_value = mock_result

        try:
            # Schedule the agent
            scheduled_agent = scheduler.schedule_agent(
                mock_agent, "every 1 second", test_input="test_value"
            )

            assert isinstance(scheduled_agent, ScheduledAgent)
            assert len(scheduler._jobs) == 1

            # Wait a bit for the job to potentially run
            await asyncio.sleep(1.5)

            # Check that the job ran
            jobs = scheduler.get_scheduled_jobs()
            assert len(jobs) == 1
            assert jobs[0]["run_count"] >= 1

            # Cancel the job
            result = scheduled_agent.cancel()
            assert result is True
            assert len(scheduler._jobs) == 0

        finally:
            # Clean up
            await scheduler.stop()
