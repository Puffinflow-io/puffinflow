"""Scheduling exceptions."""

from typing import Optional


class SchedulingError(Exception):
    """Base exception for scheduling errors."""

    pass


class InvalidScheduleError(SchedulingError):
    """Raised when a schedule string is invalid."""

    def __init__(self, schedule: str, message: Optional[str] = None):
        self.schedule = schedule
        if message is None:
            message = (
                f"Invalid schedule '{schedule}'. "
                "Try: 'daily', 'hourly', 'every 5 minutes', or cron expression."
            )
        super().__init__(message)


class InvalidInputTypeError(SchedulingError):
    """Raised when an input type prefix is invalid."""

    def __init__(self, prefix: str, message: Optional[str] = None):
        self.prefix = prefix
        if message is None:
            message = (
                f"Unknown input type '{prefix}'. "
                "Supported: secret:, const:, cache:TTL:, typed:, output:"
            )
        super().__init__(message)
