"""Tests for agent scheduling module initialization."""

from puffinflow.core.agent.scheduling import (
    GlobalScheduler,
    InputType,
    InvalidInputTypeError,
    InvalidScheduleError,
    ScheduleBuilder,
    ScheduledAgent,
    ScheduledInput,
    ScheduleParser,
    SchedulingError,
    parse_magic_prefix,
    parse_schedule_string,
)


class TestSchedulingModuleImports:
    """Test that all expected classes and functions are importable."""

    def test_global_scheduler_import(self):
        """Test GlobalScheduler can be imported."""
        assert GlobalScheduler is not None
        assert hasattr(GlobalScheduler, "get_instance")
        assert hasattr(GlobalScheduler, "schedule_agent")

    def test_scheduled_agent_import(self):
        """Test ScheduledAgent can be imported."""
        assert ScheduledAgent is not None
        assert hasattr(ScheduledAgent, "cancel")
        assert hasattr(ScheduledAgent, "get_next_run_time")

    def test_schedule_builder_import(self):
        """Test ScheduleBuilder can be imported."""
        assert ScheduleBuilder is not None

    def test_scheduled_input_import(self):
        """Test ScheduledInput can be imported."""
        assert ScheduledInput is not None
        assert hasattr(ScheduledInput, "apply_to_context")

    def test_input_type_import(self):
        """Test InputType enum can be imported."""
        assert InputType is not None
        assert hasattr(InputType, "VARIABLE")
        assert hasattr(InputType, "SECRET")
        assert hasattr(InputType, "CONSTANT")
        assert hasattr(InputType, "CACHED")
        assert hasattr(InputType, "TYPED")
        assert hasattr(InputType, "OUTPUT")

    def test_parse_magic_prefix_import(self):
        """Test parse_magic_prefix function can be imported."""
        assert parse_magic_prefix is not None
        assert callable(parse_magic_prefix)

    def test_schedule_parser_import(self):
        """Test ScheduleParser can be imported."""
        assert ScheduleParser is not None
        assert hasattr(ScheduleParser, "parse")

    def test_parse_schedule_string_import(self):
        """Test parse_schedule_string function can be imported."""
        assert parse_schedule_string is not None
        assert callable(parse_schedule_string)

    def test_scheduling_error_import(self):
        """Test SchedulingError can be imported."""
        assert SchedulingError is not None
        assert issubclass(SchedulingError, Exception)

    def test_invalid_schedule_error_import(self):
        """Test InvalidScheduleError can be imported."""
        assert InvalidScheduleError is not None
        assert issubclass(InvalidScheduleError, SchedulingError)

    def test_invalid_input_type_error_import(self):
        """Test InvalidInputTypeError can be imported."""
        assert InvalidInputTypeError is not None
        assert issubclass(InvalidInputTypeError, SchedulingError)


class TestSchedulingModuleAll:
    """Test the __all__ list contains expected exports."""

    def test_all_list_completeness(self):
        """Test that __all__ contains all expected exports."""
        from puffinflow.core.agent.scheduling import __all__

        expected_exports = [
            "GlobalScheduler",
            "ScheduledAgent",
            "ScheduleBuilder",
            "ScheduledInput",
            "InputType",
            "parse_magic_prefix",
            "ScheduleParser",
            "parse_schedule_string",
            "SchedulingError",
            "InvalidScheduleError",
            "InvalidInputTypeError",
        ]

        assert set(__all__) == set(expected_exports)

    def test_all_exports_are_importable(self):
        """Test that all items in __all__ can be imported."""
        import puffinflow.core.agent.scheduling as scheduling_module
        from puffinflow.core.agent.scheduling import __all__

        for export_name in __all__:
            assert hasattr(
                scheduling_module, export_name
            ), f"{export_name} not found in module"
            exported_item = getattr(scheduling_module, export_name)
            assert exported_item is not None, f"{export_name} is None"
