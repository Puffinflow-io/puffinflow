"""Tests for scheduling exceptions."""

from puffinflow.core.agent.scheduling.exceptions import (
    InvalidInputTypeError,
    InvalidScheduleError,
    SchedulingError,
)


class TestSchedulingError:
    """Test SchedulingError base exception."""

    def test_scheduling_error_inheritance(self):
        """Test SchedulingError inherits from Exception."""
        assert issubclass(SchedulingError, Exception)

    def test_scheduling_error_creation(self):
        """Test SchedulingError can be created with message."""
        error = SchedulingError("Test error")
        assert str(error) == "Test error"

    def test_scheduling_error_empty(self):
        """Test SchedulingError can be created without message."""
        error = SchedulingError()
        assert isinstance(error, SchedulingError)


class TestInvalidScheduleError:
    """Test InvalidScheduleError exception."""

    def test_invalid_schedule_error_inheritance(self):
        """Test InvalidScheduleError inherits from SchedulingError."""
        assert issubclass(InvalidScheduleError, SchedulingError)
        assert issubclass(InvalidScheduleError, Exception)

    def test_invalid_schedule_error_with_schedule_only(self):
        """Test InvalidScheduleError with schedule string only."""
        schedule = "invalid schedule"
        error = InvalidScheduleError(schedule)

        assert error.schedule == schedule
        expected_message = (
            f"Invalid schedule '{schedule}'. "
            "Try: 'daily', 'hourly', 'every 5 minutes', or cron expression."
        )
        assert str(error) == expected_message

    def test_invalid_schedule_error_with_custom_message(self):
        """Test InvalidScheduleError with custom message."""
        schedule = "bad schedule"
        custom_message = "Custom error message"
        error = InvalidScheduleError(schedule, custom_message)

        assert error.schedule == schedule
        assert str(error) == custom_message

    def test_invalid_schedule_error_attributes(self):
        """Test InvalidScheduleError stores schedule attribute."""
        schedule = "test schedule"
        error = InvalidScheduleError(schedule)

        assert hasattr(error, "schedule")
        assert error.schedule == schedule

    def test_invalid_schedule_error_default_message_format(self):
        """Test default message format for various schedule strings."""
        test_cases = [
            "xyz",
            "every xyz minutes",
            "invalid cron",
            "",
            "123",
        ]

        for schedule in test_cases:
            error = InvalidScheduleError(schedule)
            assert schedule in str(error)
            assert (
                "Try: 'daily', 'hourly', 'every 5 minutes', or cron expression."
                in str(error)
            )


class TestInvalidInputTypeError:
    """Test InvalidInputTypeError exception."""

    def test_invalid_input_type_error_inheritance(self):
        """Test InvalidInputTypeError inherits from SchedulingError."""
        assert issubclass(InvalidInputTypeError, SchedulingError)
        assert issubclass(InvalidInputTypeError, Exception)

    def test_invalid_input_type_error_with_prefix_only(self):
        """Test InvalidInputTypeError with prefix only."""
        prefix = "unknown"
        error = InvalidInputTypeError(prefix)

        assert error.prefix == prefix
        expected_message = (
            f"Unknown input type '{prefix}'. "
            "Supported: secret:, const:, cache:TTL:, typed:, output:"
        )
        assert str(error) == expected_message

    def test_invalid_input_type_error_with_custom_message(self):
        """Test InvalidInputTypeError with custom message."""
        prefix = "bad"
        custom_message = "Custom input type error"
        error = InvalidInputTypeError(prefix, custom_message)

        assert error.prefix == prefix
        assert str(error) == custom_message

    def test_invalid_input_type_error_attributes(self):
        """Test InvalidInputTypeError stores prefix attribute."""
        prefix = "test"
        error = InvalidInputTypeError(prefix)

        assert hasattr(error, "prefix")
        assert error.prefix == prefix

    def test_invalid_input_type_error_default_message_format(self):
        """Test default message format for various prefixes."""
        test_cases = [
            "xyz",
            "invalid",
            "unknown_type",
            "",
            "123",
        ]

        for prefix in test_cases:
            error = InvalidInputTypeError(prefix)
            assert prefix in str(error)
            assert "Supported: secret:, const:, cache:TTL:, typed:, output:" in str(
                error
            )


class TestExceptionInteraction:
    """Test exception interactions and edge cases."""

    def test_exception_chaining(self):
        """Test exception chaining works properly."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise SchedulingError("Scheduling failed") from e
        except SchedulingError as se:
            assert se.__cause__ is not None
            assert isinstance(se.__cause__, ValueError)
            assert str(se.__cause__) == "Original error"

    def test_exception_repr(self):
        """Test exception repr methods."""
        schedule_error = SchedulingError("Test")
        assert "SchedulingError" in repr(schedule_error)

        invalid_schedule = InvalidScheduleError("bad")
        assert "InvalidScheduleError" in repr(invalid_schedule)

        invalid_input = InvalidInputTypeError("bad")
        assert "InvalidInputTypeError" in repr(invalid_input)

    def test_exception_equality(self):
        """Test exception equality comparisons."""
        error1 = InvalidScheduleError("test")
        error2 = InvalidScheduleError("test")
        error3 = InvalidScheduleError("different")

        # Exceptions are not equal even with same message
        assert error1 is not error2
        assert error1 is not error3

    def test_exception_with_none_values(self):
        """Test exceptions handle None values gracefully."""
        # Test with None schedule
        error1 = InvalidScheduleError(None)
        assert error1.schedule is None
        assert "None" in str(error1)

        # Test with None prefix
        error2 = InvalidInputTypeError(None)
        assert error2.prefix is None
        assert "None" in str(error2)
