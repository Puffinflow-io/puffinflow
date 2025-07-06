"""Tests for agent scheduling functionality."""

import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from puffinflow.core.agent.base import Agent
from puffinflow.core.agent.scheduling.builder import ScheduleBuilder
from puffinflow.core.agent.scheduling.exceptions import (
    InvalidInputTypeError,
    InvalidScheduleError,
)
from puffinflow.core.agent.scheduling.inputs import (
    InputType,
    parse_inputs,
    parse_magic_prefix,
)
from puffinflow.core.agent.scheduling.parser import parse_schedule_string
from puffinflow.core.agent.scheduling.scheduler import (
    GlobalScheduler,
    ScheduledAgent,
    ScheduledJob,
)


class TestScheduleParser:
    """Test schedule string parsing."""

    def test_parse_natural_language_basic(self):
        """Test parsing basic natural language schedules."""
        # Hourly
        result = parse_schedule_string("hourly")
        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 * * * *"
        assert "Every hour" in result.description

        # Daily
        result = parse_schedule_string("daily")
        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 0 * * *"
        assert "Every day" in result.description

        # Weekly
        result = parse_schedule_string("weekly")
        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 0 * * 0"
        assert "Every Sunday" in result.description

    def test_parse_natural_language_with_time(self):
        """Test parsing natural language with specific times."""
        # Daily at specific time
        result = parse_schedule_string("daily at 09:30")
        assert result.schedule_type == "cron"
        assert result.cron_expression == "30 9 * * *"
        assert "Every day at 09:30" in result.description

        # Daily at hour only
        result = parse_schedule_string("daily at 14")
        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 14 * * *"
        assert "Every day at 14:00" in result.description

    def test_parse_intervals(self):
        """Test parsing interval-based schedules."""
        # Every X minutes
        result = parse_schedule_string("every 5 minutes")
        assert result.schedule_type == "interval"
        assert result.interval_seconds == 300
        assert "Every 5 minutes" in result.description

        # Every X hours
        result = parse_schedule_string("every 2 hours")
        assert result.schedule_type == "interval"
        assert result.interval_seconds == 7200
        assert "Every 2 hours" in result.description

        # Every X seconds
        result = parse_schedule_string("every 30 seconds")
        assert result.schedule_type == "interval"
        assert result.interval_seconds == 30
        assert "Every 30 seconds" in result.description

    def test_parse_cron_expressions(self):
        """Test parsing cron expressions."""
        # Valid cron
        result = parse_schedule_string("0 9 * * 1-5")
        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 9 * * 1-5"
        assert "Cron:" in result.description

        # Another valid cron
        result = parse_schedule_string("*/15 * * * *")
        assert result.schedule_type == "cron"
        assert result.cron_expression == "*/15 * * * *"

    def test_parse_invalid_schedules(self):
        """Test parsing invalid schedule strings."""
        with pytest.raises(InvalidScheduleError):
            parse_schedule_string("")

        with pytest.raises(InvalidScheduleError):
            parse_schedule_string("invalid schedule")

        with pytest.raises(InvalidScheduleError):
            parse_schedule_string("every tuesday at maybe")


class TestMagicPrefixParsing:
    """Test magic prefix parsing for inputs."""

    def test_parse_regular_variable(self):
        """Test parsing regular variables (no prefix)."""
        result = parse_magic_prefix("source", "database")
        assert result.key == "source"
        assert result.value == "database"
        assert result.input_type == InputType.VARIABLE
        assert result.ttl is None

    def test_parse_secret_prefix(self):
        """Test parsing secret prefix."""
        result = parse_magic_prefix("api_key", "secret:sk-1234567890abcdef")
        assert result.key == "api_key"
        assert result.value == "sk-1234567890abcdef"
        assert result.input_type == InputType.SECRET

    def test_parse_constant_prefix(self):
        """Test parsing constant prefix."""
        result = parse_magic_prefix("max_size", "const:100")
        assert result.key == "max_size"
        assert result.value == "100"
        assert result.input_type == InputType.CONSTANT

    def test_parse_cache_prefix(self):
        """Test parsing cache prefix with TTL."""
        result = parse_magic_prefix("config", 'cache:3600:{"timeout": 30}')
        assert result.key == "config"
        assert result.value == {"timeout": 30}  # Should be parsed as JSON
        assert result.input_type == InputType.CACHED
        assert result.ttl == 3600

        # Test with string value
        result = parse_magic_prefix("setting", "cache:1800:production")
        assert result.key == "setting"
        assert result.value == "production"
        assert result.input_type == InputType.CACHED
        assert result.ttl == 1800

    def test_parse_typed_prefix(self):
        """Test parsing typed prefix."""
        result = parse_magic_prefix("user", 'typed:{"name": "john", "age": 30}')
        assert result.key == "user"
        assert result.value == {"name": "john", "age": 30}  # Should be parsed as JSON
        assert result.input_type == InputType.TYPED

        # Test with string value
        result = parse_magic_prefix("status", "typed:active")
        assert result.key == "status"
        assert result.value == "active"
        assert result.input_type == InputType.TYPED

    def test_parse_output_prefix(self):
        """Test parsing output prefix."""
        result = parse_magic_prefix("result", "output:processed")
        assert result.key == "result"
        assert result.value == "processed"
        assert result.input_type == InputType.OUTPUT

    def test_parse_invalid_prefixes(self):
        """Test parsing invalid prefixes."""
        # Invalid cache format
        with pytest.raises(InvalidInputTypeError):
            parse_magic_prefix("config", "cache:invalid_ttl:value")

        # Invalid secret format (empty value)
        with pytest.raises(InvalidInputTypeError):
            parse_magic_prefix("secret", "secret:")

    def test_parse_non_string_values(self):
        """Test parsing non-string values."""
        result = parse_magic_prefix("count", 42)
        assert result.key == "count"
        assert result.value == 42
        assert result.input_type == InputType.VARIABLE

        result = parse_magic_prefix("enabled", True)
        assert result.key == "enabled"
        assert result.value is True
        assert result.input_type == InputType.VARIABLE

    def test_parse_inputs_batch(self):
        """Test parsing multiple inputs at once."""
        inputs = {
            "source": "database",
            "api_key": "secret:sk-123",
            "max_size": "const:100",
            "config": 'cache:3600:{"x": 1}',
            "count": 42,
        }

        parsed = parse_inputs(**inputs)

        assert len(parsed) == 5
        assert parsed["source"].input_type == InputType.VARIABLE
        assert parsed["api_key"].input_type == InputType.SECRET
        assert parsed["max_size"].input_type == InputType.CONSTANT
        assert parsed["config"].input_type == InputType.CACHED
        assert parsed["count"].input_type == InputType.VARIABLE


class TestScheduleBuilder:
    """Test fluent API schedule builder."""

    def test_builder_with_inputs(self):
        """Test builder with regular inputs."""
        agent = Agent("test_agent")
        builder = ScheduleBuilder(agent, "daily")

        result = builder.with_inputs(source="database", count=100)
        assert isinstance(result, ScheduleBuilder)
        assert len(builder._inputs) == 2
        assert builder._inputs["source"].input_type == InputType.VARIABLE
        assert builder._inputs["count"].input_type == InputType.VARIABLE

    def test_builder_with_secrets(self):
        """Test builder with secrets."""
        agent = Agent("test_agent")
        builder = ScheduleBuilder(agent, "hourly")

        result = builder.with_secrets(api_key="sk-123", db_pass="secret123")
        assert isinstance(result, ScheduleBuilder)
        assert len(builder._inputs) == 2
        assert builder._inputs["api_key"].input_type == InputType.SECRET
        assert builder._inputs["db_pass"].input_type == InputType.SECRET

    def test_builder_with_constants(self):
        """Test builder with constants."""
        agent = Agent("test_agent")
        builder = ScheduleBuilder(agent, "daily")

        result = builder.with_constants(max_size=100, timeout=30)
        assert isinstance(result, ScheduleBuilder)
        assert len(builder._inputs) == 2
        assert builder._inputs["max_size"].input_type == InputType.CONSTANT
        assert builder._inputs["timeout"].input_type == InputType.CONSTANT

    def test_builder_with_cache(self):
        """Test builder with cached inputs."""
        agent = Agent("test_agent")
        builder = ScheduleBuilder(agent, "hourly")

        result = builder.with_cache(ttl=3600, config={"x": 1}, setting="prod")
        assert isinstance(result, ScheduleBuilder)
        assert len(builder._inputs) == 2
        assert builder._inputs["config"].input_type == InputType.CACHED
        assert builder._inputs["setting"].input_type == InputType.CACHED

    def test_builder_chaining(self):
        """Test chaining multiple builder methods."""
        agent = Agent("test_agent")

        builder = (
            ScheduleBuilder(agent, "daily")
            .with_inputs(source="database")
            .with_secrets(api_key="sk-123")
            .with_constants(max_size=100)
            .with_cache(ttl=1800, config={"timeout": 30})
        )

        assert len(builder._inputs) == 4
        assert builder._inputs["source"].input_type == InputType.VARIABLE
        assert builder._inputs["api_key"].input_type == InputType.SECRET
        assert builder._inputs["max_size"].input_type == InputType.CONSTANT
        assert builder._inputs["config"].input_type == InputType.CACHED


class TestAgentScheduling:
    """Test agent scheduling methods."""

    def test_agent_schedule_method(self):
        """Test basic agent.schedule() method."""
        agent = Agent("test_agent")

        with patch(
            "puffinflow.core.agent.scheduling.scheduler.GlobalScheduler.get_instance_sync"
        ) as mock_scheduler:
            mock_scheduler_instance = Mock()
            mock_scheduler.return_value = mock_scheduler_instance
            mock_scheduler_instance.schedule_agent.return_value = Mock(
                spec=ScheduledAgent
            )

            agent.schedule("daily", source="database", api_key="secret:sk-123")

            mock_scheduler_instance.schedule_agent.assert_called_once_with(
                agent, "daily", source="database", api_key="secret:sk-123"
            )

    def test_agent_every_method(self):
        """Test agent.every() fluent API method."""
        agent = Agent("test_agent")

        result = agent.every("5 minutes")
        assert isinstance(result, ScheduleBuilder)
        assert result._schedule_string == "every 5 minutes"

        # Test with already prefixed "every"
        result = agent.every("every 2 hours")
        assert (
            result._schedule_string == "every 2 hours"
        )  # Should handle this gracefully

    def test_agent_daily_method(self):
        """Test agent.daily() fluent API method."""
        agent = Agent("test_agent")

        # Without time
        result = agent.daily()
        assert isinstance(result, ScheduleBuilder)
        assert result._schedule_string == "daily"

        # With time
        result = agent.daily("09:00")
        assert result._schedule_string == "daily at 09:00"

    def test_agent_hourly_method(self):
        """Test agent.hourly() fluent API method."""
        agent = Agent("test_agent")

        # Without minute
        result = agent.hourly()
        assert isinstance(result, ScheduleBuilder)
        assert result._schedule_string == "hourly"

        # With minute
        result = agent.hourly(30)
        assert result._schedule_string == "every hour at 30"


class TestGlobalScheduler:
    """Test global scheduler functionality."""

    def setUp(self):
        """Reset scheduler instance before each test."""
        GlobalScheduler._instance = None

    def test_scheduler_singleton(self):
        """Test scheduler singleton pattern."""
        scheduler1 = GlobalScheduler.get_instance_sync()
        scheduler2 = GlobalScheduler.get_instance_sync()
        assert scheduler1 is scheduler2

    def test_schedule_agent(self):
        """Test scheduling an agent."""
        agent = Agent("test_agent")
        scheduler = GlobalScheduler()

        result = scheduler.schedule_agent(agent, "daily", source="database")

        assert isinstance(result, ScheduledAgent)
        assert len(scheduler._jobs) == 1

        job_id = next(iter(scheduler._jobs.keys()))
        job = scheduler._jobs[job_id]
        assert job.agent is agent
        assert job.schedule.schedule_type == "cron"
        assert len(job.inputs) == 1
        assert job.inputs["source"].value == "database"

    def test_cancel_job(self):
        """Test cancelling a scheduled job."""
        agent = Agent("test_agent")
        scheduler = GlobalScheduler()

        scheduled_agent = scheduler.schedule_agent(agent, "hourly", source="api")
        job_id = scheduled_agent.job_id

        assert len(scheduler._jobs) == 1

        result = scheduler.cancel_job(job_id)
        assert result is True
        assert len(scheduler._jobs) == 0

        # Try to cancel non-existent job
        result = scheduler.cancel_job("non_existent")
        assert result is False

    def test_get_scheduled_jobs(self):
        """Test getting scheduled jobs information."""
        agent1 = Agent("agent1")
        agent2 = Agent("agent2")
        scheduler = GlobalScheduler()

        scheduler.schedule_agent(agent1, "daily", source="db1")
        scheduler.schedule_agent(agent2, "hourly", source="db2")

        # Get all jobs
        jobs = scheduler.get_scheduled_jobs()
        assert len(jobs) == 2

        # Get jobs for specific agent
        jobs = scheduler.get_scheduled_jobs("agent1")
        assert len(jobs) == 1
        assert jobs[0]["agent_name"] == "agent1"

    def test_cleanup_dead_jobs(self):
        """Test cleaning up jobs for garbage collected agents."""
        scheduler = GlobalScheduler()

        # Create agent and schedule it
        agent = Agent("test_agent")
        scheduler.schedule_agent(agent, "daily", source="database")
        assert len(scheduler._jobs) == 1

        # Simulate agent being garbage collected
        job = next(iter(scheduler._jobs.values()))
        job.agent_ref = lambda: None  # Simulate dead weak reference

        cleaned = scheduler.cleanup_dead_jobs()
        assert cleaned == 1
        assert len(scheduler._jobs) == 0


class TestScheduledJob:
    """Test scheduled job functionality."""

    def test_calculate_next_run_interval(self):
        """Test calculating next run time for interval schedules."""
        agent = Agent("test_agent")
        from puffinflow.core.agent.scheduling.parser import ParsedSchedule

        schedule = ParsedSchedule("interval", interval_seconds=300)  # 5 minutes
        job = ScheduledJob(
            job_id="test_job",
            agent_ref=lambda: agent,
            schedule=schedule,
            inputs={},
            next_run=time.time(),
        )

        # First run should be immediate
        next_run = job.calculate_next_run()
        assert next_run <= time.time() + 1  # Allow small time difference

        # After first run
        job.last_run = time.time()
        next_run = job.calculate_next_run()
        expected = job.last_run + 300
        assert abs(next_run - expected) < 1

    def test_calculate_next_run_cron_hourly(self):
        """Test calculating next run time for hourly cron."""
        agent = Agent("test_agent")
        from puffinflow.core.agent.scheduling.parser import ParsedSchedule

        schedule = ParsedSchedule("cron", cron_expression="0 * * * *")
        job = ScheduledJob(
            job_id="test_job",
            agent_ref=lambda: agent,
            schedule=schedule,
            inputs={},
            next_run=time.time(),
        )

        now = time.time()
        next_run = job.calculate_next_run()

        # Should be within the next hour
        assert next_run > now
        assert next_run <= now + 3600


class TestScheduledAgent:
    """Test scheduled agent functionality."""

    def test_scheduled_agent_creation(self):
        """Test creating a scheduled agent."""
        agent = Agent("test_agent")
        from puffinflow.core.agent.scheduling.parser import ParsedSchedule

        schedule = ParsedSchedule("cron", "0 9 * * *", "Daily at 9 AM")
        inputs = {"source": parse_magic_prefix("source", "database")}

        scheduled_agent = ScheduledAgent("job_123", agent, schedule, inputs)

        assert scheduled_agent.job_id == "job_123"
        assert scheduled_agent.agent is agent
        assert scheduled_agent.schedule is schedule
        assert scheduled_agent.inputs is inputs

    def test_scheduled_agent_cancel(self):
        """Test cancelling a scheduled agent."""
        agent = Agent("test_agent")
        from puffinflow.core.agent.scheduling.parser import ParsedSchedule

        schedule = ParsedSchedule("cron", "0 9 * * *")
        scheduled_agent = ScheduledAgent("job_123", agent, schedule, {})

        with patch(
            "puffinflow.core.agent.scheduling.scheduler.GlobalScheduler.get_instance_sync"
        ) as mock_get_instance:
            mock_scheduler = Mock()
            mock_get_instance.return_value = mock_scheduler
            mock_scheduler.cancel_job.return_value = True

            result = scheduled_agent.cancel()
            assert result is True
            mock_scheduler.cancel_job.assert_called_once_with("job_123")


@pytest.mark.asyncio
class TestSchedulerExecution:
    """Test scheduler execution functionality."""

    async def test_scheduler_start_stop(self):
        """Test starting and stopping the scheduler."""
        scheduler = GlobalScheduler()

        await scheduler.start()
        assert scheduler._running is True
        assert scheduler._scheduler_task is not None

        await scheduler.stop()
        assert scheduler._running is False
        assert scheduler._scheduler_task is None

    async def test_run_job_execution(self):
        """Test job execution."""
        # Create a mock agent with a simple run method
        agent = Mock(spec=Agent)
        agent.name = "test_agent"
        agent._create_context = Mock()
        agent.shared_state = {}

        # Mock context
        mock_context = Mock()
        agent._create_context.return_value = mock_context

        # Mock agent result
        from puffinflow.core.agent.state import AgentStatus

        mock_result = Mock()
        mock_result.status = AgentStatus.COMPLETED
        agent.run = AsyncMock(return_value=mock_result)

        # Create scheduler and job
        scheduler = GlobalScheduler()
        from puffinflow.core.agent.scheduling.parser import ParsedSchedule

        schedule = ParsedSchedule("interval", interval_seconds=60)
        inputs = {"source": parse_magic_prefix("source", "database")}

        job = ScheduledJob(
            job_id="test_job",
            agent_ref=lambda: agent,
            schedule=schedule,
            inputs=inputs,
            next_run=time.time(),
        )

        # Run the job
        await scheduler._run_job(job)

        # Verify execution
        agent._create_context.assert_called_once()
        agent.run.assert_called_once()
        assert job.run_count == 1
        assert job.last_run is not None
        assert not job.is_running


if __name__ == "__main__":
    pytest.main([__file__])
