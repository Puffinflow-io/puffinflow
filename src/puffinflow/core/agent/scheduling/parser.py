"""Schedule string parsing for natural language and cron expressions."""

import re
from dataclasses import dataclass
from re import Match
from typing import Callable, Optional

from .exceptions import InvalidScheduleError


@dataclass
class ParsedSchedule:
    """Result of parsing a schedule string."""

    schedule_type: str  # "cron", "interval", or "natural"
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    description: str = ""


class ScheduleParser:
    """Parser for schedule strings supporting natural language and cron."""

    # Natural language patterns
    NATURAL_PATTERNS: dict[str, Callable[[Match[str]], ParsedSchedule]] = {
        # Basic intervals
        r"^hourly$": lambda m: ParsedSchedule(
            "cron", "0 * * * *", description="Every hour"
        ),
        r"^daily$": lambda m: ParsedSchedule(
            "cron", "0 0 * * *", description="Every day at midnight"
        ),
        r"^weekly$": lambda m: ParsedSchedule(
            "cron", "0 0 * * 0", description="Every Sunday at midnight"
        ),
        r"^monthly$": lambda m: ParsedSchedule(
            "cron", "0 0 1 * *", description="First day of every month"
        ),
        # Daily with time
        r"^daily\s+at\s+(\d{1,2}):(\d{2})$": lambda m: ParsedSchedule(
            "cron",
            f"{int(m.group(2))} {int(m.group(1))} * * *",
            description=f"Every day at {m.group(1)}:{m.group(2)}",
        ),
        r"^daily\s+at\s+(\d{1,2})(?::00)?(?:\s*am)?$": lambda m: ParsedSchedule(
            "cron",
            f"0 {int(m.group(1))} * * *",
            description=f"Every day at {m.group(1)}:00",
        ),
        r"^daily\s+at\s+(\d{1,2})(?::00)?\s*pm$": lambda m: ParsedSchedule(
            "cron",
            f"0 {int(m.group(1)) + 12} * * *",
            description=f"Every day at {int(m.group(1)) + 12}:00",
        ),
        # Every X minutes/hours
        r"^every\s+(\d+)\s+minutes?$": lambda m: ParsedSchedule(
            "interval",
            interval_seconds=int(m.group(1)) * 60,
            description=f"Every {m.group(1)} {'minute' if int(m.group(1)) == 1 else 'minutes'}",
        ),
        r"^every\s+(\d+)\s+hours?$": lambda m: ParsedSchedule(
            "interval",
            interval_seconds=int(m.group(1)) * 3600,
            description=f"Every {m.group(1)} {'hour' if int(m.group(1)) == 1 else 'hours'}",
        ),
        r"^every\s+(\d+)\s+seconds?$": lambda m: ParsedSchedule(
            "interval",
            interval_seconds=int(m.group(1)),
            description=f"Every {m.group(1)} {'second' if int(m.group(1)) == 1 else 'seconds'}",
        ),
        # Every hour at minute
        r"^every\s+hour\s+at\s+(\d{1,2})$": lambda m: ParsedSchedule(
            "cron",
            f"{int(m.group(1))} * * * *",
            description=f"Every hour at minute {m.group(1)}",
        ),
        # Weekdays
        r"^weekdays$": lambda m: ParsedSchedule(
            "cron", "0 9 * * 1-5", description="Weekdays at 9 AM"
        ),
        r"^weekdays\s+at\s+(\d{1,2}):(\d{2})$": lambda m: ParsedSchedule(
            "cron",
            f"{int(m.group(2))} {int(m.group(1))} * * 1-5",
            description=f"Weekdays at {m.group(1)}:{m.group(2)}",
        ),
        # Weekends
        r"^weekends$": lambda m: ParsedSchedule(
            "cron", "0 10 * * 0,6", description="Weekends at 10 AM"
        ),
        r"^weekends\s+at\s+(\d{1,2}):(\d{2})$": lambda m: ParsedSchedule(
            "cron",
            f"{int(m.group(2))} {int(m.group(1))} * * 0,6",
            description=f"Weekends at {m.group(1)}:{m.group(2)}",
        ),
    }

    @classmethod
    def parse(cls, schedule_string: str) -> ParsedSchedule:
        """Parse a schedule string.

        Args:
            schedule_string: Natural language or cron expression

        Returns:
            ParsedSchedule object

        Raises:
            InvalidScheduleError: If schedule string is invalid
        """
        if not schedule_string or not schedule_string.strip():
            raise InvalidScheduleError(
                schedule_string, "Schedule string cannot be empty"
            )

        schedule_string = schedule_string.strip().lower()

        # Try natural language patterns first
        for pattern, handler in cls.NATURAL_PATTERNS.items():
            match = re.match(pattern, schedule_string, re.IGNORECASE)
            if match:
                return handler(match)

        # Try as cron expression
        if cls._is_valid_cron(schedule_string):
            return ParsedSchedule(
                "cron", schedule_string, description=f"Cron: {schedule_string}"
            )

        # If nothing matches, raise error with suggestions
        raise InvalidScheduleError(schedule_string)

    @staticmethod
    def _is_valid_cron(expression: str) -> bool:
        """Validate cron expression format.

        Args:
            expression: Cron expression to validate

        Returns:
            True if valid cron format
        """
        if not expression:
            return False

        parts = expression.split()
        if len(parts) != 5:
            return False

        # Basic validation patterns for each field
        patterns = [
            r"^(\*|[0-5]?\d(-[0-5]?\d)?(/\d+)?|\*/\d+)$",  # minute (0-59)
            r"^(\*|\d{1,2}(-\d{1,2})?(/\d+)?|\*/\d+)$",  # hour (0-23)
            r"^(\*|[1-3]?\d(-[1-3]?\d)?(/\d+)?|\*/\d+)$",  # day (1-31)
            r"^(\*|1?\d(-1?\d)?(/\d+)?|\*/\d+)$",  # month (1-12)
            r"^(\*|[0-6](-[0-6])?(/\d+)?|\*/\d+)$",  # day of week (0-6)
        ]

        for i, part in enumerate(parts):
            if not re.match(patterns[i], part):
                return False

        # Additional range validation
        try:
            # Check minute (0-59)
            if parts[0] != "*" and not parts[0].startswith("*/"):
                minute_val = int(parts[0].split("-")[0].split("/")[0])
                if minute_val > 59:
                    return False

            # Check hour (0-23)
            if parts[1] != "*" and not parts[1].startswith("*/"):
                hour_val = int(parts[1].split("-")[0].split("/")[0])
                if hour_val > 23:
                    return False

            # Check day (1-31)
            if parts[2] != "*" and not parts[2].startswith("*/"):
                day_val = int(parts[2].split("-")[0].split("/")[0])
                if day_val < 1 or day_val > 31:
                    return False

            # Check month (1-12)
            if parts[3] != "*" and not parts[3].startswith("*/"):
                month_val = int(parts[3].split("-")[0].split("/")[0])
                if month_val < 1 or month_val > 12:
                    return False

            # Check day of week (0-6)
            if parts[4] != "*" and not parts[4].startswith("*/"):
                dow_val = int(parts[4].split("-")[0].split("/")[0])
                if dow_val > 6:
                    return False

        except (ValueError, IndexError):
            return False

        return True


def parse_schedule_string(schedule: str) -> ParsedSchedule:
    """Parse a schedule string (convenience function).

    Args:
        schedule: Schedule string to parse

    Returns:
        ParsedSchedule object
    """
    return ScheduleParser.parse(schedule)
