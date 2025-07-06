"""Tests for schedule string parsing."""

import pytest

from puffinflow.core.agent.scheduling.exceptions import InvalidScheduleError
from puffinflow.core.agent.scheduling.parser import (
    ParsedSchedule,
    ScheduleParser,
    parse_schedule_string,
)


class TestParsedSchedule:
    """Test ParsedSchedule dataclass."""

    def test_parsed_schedule_creation_minimal(self):
        """Test ParsedSchedule creation with minimal fields."""
        schedule = ParsedSchedule("cron")

        assert schedule.schedule_type == "cron"
        assert schedule.cron_expression is None
        assert schedule.interval_seconds is None
        assert schedule.description == ""

    def test_parsed_schedule_creation_full(self):
        """Test ParsedSchedule creation with all fields."""
        schedule = ParsedSchedule(
            schedule_type="interval",
            cron_expression="0 * * * *",
            interval_seconds=3600,
            description="Every hour",
        )

        assert schedule.schedule_type == "interval"
        assert schedule.cron_expression == "0 * * * *"
        assert schedule.interval_seconds == 3600
        assert schedule.description == "Every hour"


class TestScheduleParserBasicIntervals:
    """Test parsing basic interval expressions."""

    def test_parse_hourly(self):
        """Test parsing 'hourly' schedule."""
        result = ScheduleParser.parse("hourly")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 * * * *"
        assert result.description == "Every hour"

    def test_parse_daily(self):
        """Test parsing 'daily' schedule."""
        result = ScheduleParser.parse("daily")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 0 * * *"
        assert result.description == "Every day at midnight"

    def test_parse_weekly(self):
        """Test parsing 'weekly' schedule."""
        result = ScheduleParser.parse("weekly")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 0 * * 0"
        assert result.description == "Every Sunday at midnight"

    def test_parse_monthly(self):
        """Test parsing 'monthly' schedule."""
        result = ScheduleParser.parse("monthly")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 0 1 * *"
        assert result.description == "First day of every month"


class TestScheduleParserDailyWithTime:
    """Test parsing daily schedules with specific times."""

    def test_parse_daily_with_time_24h(self):
        """Test parsing 'daily at HH:MM' format."""
        result = ScheduleParser.parse("daily at 14:30")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "30 14 * * *"
        assert result.description == "Every day at 14:30"

    def test_parse_daily_with_time_single_digit_hour(self):
        """Test parsing daily with single digit hour."""
        result = ScheduleParser.parse("daily at 9:15")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "15 9 * * *"
        assert result.description == "Every day at 9:15"

    def test_parse_daily_at_hour_am_implicit(self):
        """Test parsing 'daily at H' format (implicit AM)."""
        result = ScheduleParser.parse("daily at 8")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 8 * * *"
        assert result.description == "Every day at 8:00"

    def test_parse_daily_at_hour_am_explicit(self):
        """Test parsing 'daily at H am' format."""
        result = ScheduleParser.parse("daily at 9 am")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 9 * * *"
        assert result.description == "Every day at 9:00"

    def test_parse_daily_at_hour_pm(self):
        """Test parsing 'daily at H pm' format."""
        result = ScheduleParser.parse("daily at 3 pm")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 15 * * *"
        assert result.description == "Every day at 15:00"

    def test_parse_daily_at_hour_with_colon_pm(self):
        """Test parsing 'daily at H:00 pm' format."""
        result = ScheduleParser.parse("daily at 7:00 pm")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 19 * * *"
        assert result.description == "Every day at 19:00"


class TestScheduleParserIntervals:
    """Test parsing interval-based schedules."""

    def test_parse_every_minutes(self):
        """Test parsing 'every X minutes' format."""
        result = ScheduleParser.parse("every 15 minutes")

        assert result.schedule_type == "interval"
        assert result.interval_seconds == 900  # 15 * 60
        assert result.description == "Every 15 minutes"

    def test_parse_every_minute_singular(self):
        """Test parsing 'every X minute' format (singular)."""
        result = ScheduleParser.parse("every 1 minute")

        assert result.schedule_type == "interval"
        assert result.interval_seconds == 60
        assert result.description == "Every 1 minute"

    def test_parse_every_hours(self):
        """Test parsing 'every X hours' format."""
        result = ScheduleParser.parse("every 2 hours")

        assert result.schedule_type == "interval"
        assert result.interval_seconds == 7200  # 2 * 3600
        assert result.description == "Every 2 hours"

    def test_parse_every_hour_singular(self):
        """Test parsing 'every X hour' format (singular)."""
        result = ScheduleParser.parse("every 1 hour")

        assert result.schedule_type == "interval"
        assert result.interval_seconds == 3600
        assert result.description == "Every 1 hour"

    def test_parse_every_seconds(self):
        """Test parsing 'every X seconds' format."""
        result = ScheduleParser.parse("every 30 seconds")

        assert result.schedule_type == "interval"
        assert result.interval_seconds == 30
        assert result.description == "Every 30 seconds"

    def test_parse_every_second_singular(self):
        """Test parsing 'every X second' format (singular)."""
        result = ScheduleParser.parse("every 1 second")

        assert result.schedule_type == "interval"
        assert result.interval_seconds == 1
        assert result.description == "Every 1 second"


class TestScheduleParserHourlyWithMinute:
    """Test parsing hourly schedules with specific minutes."""

    def test_parse_every_hour_at_minute(self):
        """Test parsing 'every hour at MM' format."""
        result = ScheduleParser.parse("every hour at 30")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "30 * * * *"
        assert result.description == "Every hour at minute 30"

    def test_parse_every_hour_at_minute_single_digit(self):
        """Test parsing 'every hour at M' format."""
        result = ScheduleParser.parse("every hour at 5")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "5 * * * *"
        assert result.description == "Every hour at minute 5"


class TestScheduleParserWeekdays:
    """Test parsing weekday schedules."""

    def test_parse_weekdays_default(self):
        """Test parsing 'weekdays' schedule."""
        result = ScheduleParser.parse("weekdays")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 9 * * 1-5"
        assert result.description == "Weekdays at 9 AM"

    def test_parse_weekdays_with_time(self):
        """Test parsing 'weekdays at HH:MM' format."""
        result = ScheduleParser.parse("weekdays at 8:30")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "30 8 * * 1-5"
        assert result.description == "Weekdays at 8:30"


class TestScheduleParserWeekends:
    """Test parsing weekend schedules."""

    def test_parse_weekends_default(self):
        """Test parsing 'weekends' schedule."""
        result = ScheduleParser.parse("weekends")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 10 * * 0,6"
        assert result.description == "Weekends at 10 AM"

    def test_parse_weekends_with_time(self):
        """Test parsing 'weekends at HH:MM' format."""
        result = ScheduleParser.parse("weekends at 11:45")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "45 11 * * 0,6"
        assert result.description == "Weekends at 11:45"


class TestScheduleParserCronExpressions:
    """Test parsing cron expressions."""

    def test_parse_valid_cron_hourly(self):
        """Test parsing valid cron expression for hourly."""
        result = ScheduleParser.parse("0 * * * *")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 * * * *"
        assert result.description == "Cron: 0 * * * *"

    def test_parse_valid_cron_daily(self):
        """Test parsing valid cron expression for daily."""
        result = ScheduleParser.parse("30 6 * * *")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "30 6 * * *"
        assert result.description == "Cron: 30 6 * * *"

    def test_parse_valid_cron_weekly(self):
        """Test parsing valid cron expression for weekly."""
        result = ScheduleParser.parse("0 0 * * 1")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 0 * * 1"
        assert result.description == "Cron: 0 0 * * 1"

    def test_parse_valid_cron_with_ranges(self):
        """Test parsing cron expression with ranges."""
        result = ScheduleParser.parse("0 9-17 * * 1-5")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 9-17 * * 1-5"
        assert result.description == "Cron: 0 9-17 * * 1-5"

    def test_parse_valid_cron_with_steps(self):
        """Test parsing cron expression with steps."""
        result = ScheduleParser.parse("*/15 * * * *")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "*/15 * * * *"
        assert result.description == "Cron: */15 * * * *"


class TestScheduleParserCaseInsensitive:
    """Test case insensitive parsing."""

    def test_parse_case_insensitive_basic(self):
        """Test basic schedules are case insensitive."""
        test_cases = [
            ("HOURLY", "0 * * * *"),
            ("Daily", "0 0 * * *"),
            ("WEEKLY", "0 0 * * 0"),
            ("monthly", "0 0 1 * *"),
        ]

        for input_str, expected_cron in test_cases:
            result = ScheduleParser.parse(input_str)
            assert result.schedule_type == "cron"
            assert result.cron_expression == expected_cron

    def test_parse_case_insensitive_complex(self):
        """Test complex schedules are case insensitive."""
        test_cases = [
            "DAILY AT 14:30",
            "Every 15 MINUTES",
            "WEEKDAYS at 9:00",
            "weekends AT 10:30",
        ]

        for input_str in test_cases:
            result = ScheduleParser.parse(input_str)
            assert result.schedule_type in ["cron", "interval"]


class TestScheduleParserValidation:
    """Test schedule validation and error handling."""

    def test_parse_empty_string(self):
        """Test parsing empty string raises error."""
        with pytest.raises(InvalidScheduleError) as exc_info:
            ScheduleParser.parse("")

        assert "Schedule string cannot be empty" in str(exc_info.value)

    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only string raises error."""
        with pytest.raises(InvalidScheduleError) as exc_info:
            ScheduleParser.parse("   ")

        assert "Schedule string cannot be empty" in str(exc_info.value)

    def test_parse_none(self):
        """Test parsing None raises error."""
        with pytest.raises(InvalidScheduleError):
            ScheduleParser.parse(None)

    def test_parse_invalid_natural_language(self):
        """Test parsing invalid natural language raises error."""
        with pytest.raises(InvalidScheduleError) as exc_info:
            ScheduleParser.parse("invalid schedule string")

        assert exc_info.value.schedule == "invalid schedule string"
        assert "Try: 'daily', 'hourly', 'every 5 minutes', or cron expression." in str(
            exc_info.value
        )

    def test_parse_invalid_cron_too_few_parts(self):
        """Test parsing cron with too few parts raises error."""
        with pytest.raises(InvalidScheduleError):
            ScheduleParser.parse("0 * *")

    def test_parse_invalid_cron_too_many_parts(self):
        """Test parsing cron with too many parts raises error."""
        with pytest.raises(InvalidScheduleError):
            ScheduleParser.parse("0 * * * * * *")

    def test_parse_invalid_cron_bad_minute(self):
        """Test parsing cron with invalid minute raises error."""
        with pytest.raises(InvalidScheduleError):
            ScheduleParser.parse("60 * * * *")  # minute 60 is invalid

    def test_parse_invalid_cron_bad_hour(self):
        """Test parsing cron with invalid hour raises error."""
        with pytest.raises(InvalidScheduleError):
            ScheduleParser.parse("0 24 * * *")  # hour 24 is invalid


class TestScheduleParserCronValidation:
    """Test cron expression validation."""

    def test_is_valid_cron_valid_expressions(self):
        """Test _is_valid_cron with valid expressions."""
        valid_expressions = [
            "0 * * * *",
            "30 6 * * *",
            "0 0 * * 0",
            "*/15 * * * *",
            "0 9-17 * * 1-5",
            "30 8 1 * *",
            "0 0 1 1 *",
        ]

        for expr in valid_expressions:
            assert ScheduleParser._is_valid_cron(expr), f"Should be valid: {expr}"

    def test_is_valid_cron_invalid_expressions(self):
        """Test _is_valid_cron with invalid expressions."""
        invalid_expressions = [
            "",
            "0",
            "0 *",
            "0 * *",
            "0 * * *",
            "0 * * * * *",  # too many parts
            "60 * * * *",  # invalid minute
            "0 24 * * *",  # invalid hour
            "0 0 32 * *",  # invalid day
            "0 0 * 13 *",  # invalid month
            "0 0 * * 7",  # invalid day of week
            "invalid * * * *",
        ]

        for expr in invalid_expressions:
            assert not ScheduleParser._is_valid_cron(expr), f"Should be invalid: {expr}"

    def test_is_valid_cron_edge_cases(self):
        """Test _is_valid_cron with edge cases."""
        edge_cases = [
            ("0 0 1 1 0", True),  # Valid: Jan 1st if it's Sunday
            ("59 23 31 12 6", True),  # Valid: Dec 31st if it's Saturday at 23:59
            (
                "0 0 29 2 *",
                True,
            ),  # Valid: Feb 29th (leap year handling not in basic validation)
        ]

        for expr, expected in edge_cases:
            assert (
                ScheduleParser._is_valid_cron(expr) == expected
            ), f"Expression: {expr}"


class TestParseScheduleStringFunction:
    """Test the convenience function parse_schedule_string."""

    def test_parse_schedule_string_function(self):
        """Test parse_schedule_string function works correctly."""
        result = parse_schedule_string("daily")

        assert isinstance(result, ParsedSchedule)
        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 0 * * *"
        assert result.description == "Every day at midnight"

    def test_parse_schedule_string_function_error(self):
        """Test parse_schedule_string function raises errors correctly."""
        with pytest.raises(InvalidScheduleError):
            parse_schedule_string("invalid")


class TestScheduleParserWhitespace:
    """Test handling of whitespace in schedule strings."""

    def test_parse_with_leading_trailing_whitespace(self):
        """Test parsing with leading/trailing whitespace."""
        result = ScheduleParser.parse("  daily  ")

        assert result.schedule_type == "cron"
        assert result.cron_expression == "0 0 * * *"

    def test_parse_with_extra_internal_whitespace(self):
        """Test parsing with extra internal whitespace."""
        result = ScheduleParser.parse("every   15   minutes")

        assert result.schedule_type == "interval"
        assert result.interval_seconds == 900

    def test_parse_cron_with_whitespace(self):
        """Test parsing cron expressions with extra whitespace."""
        result = ScheduleParser.parse("  0   *   *   *   *  ")

        assert result.schedule_type == "cron"
        assert (
            result.cron_expression == "0   *   *   *   *"
        )  # Preserves internal whitespace
