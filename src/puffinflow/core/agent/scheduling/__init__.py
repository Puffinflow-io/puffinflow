"""Agent scheduling module for PuffinFlow."""

from typing import Any

_LAZY_IMPORTS = {
    "ScheduleBuilder": (".builder", "ScheduleBuilder"),
    "InvalidInputTypeError": (".exceptions", "InvalidInputTypeError"),
    "InvalidScheduleError": (".exceptions", "InvalidScheduleError"),
    "SchedulingError": (".exceptions", "SchedulingError"),
    "InputType": (".inputs", "InputType"),
    "ScheduledInput": (".inputs", "ScheduledInput"),
    "parse_magic_prefix": (".inputs", "parse_magic_prefix"),
    "ScheduleParser": (".parser", "ScheduleParser"),
    "parse_schedule_string": (".parser", "parse_schedule_string"),
    "GlobalScheduler": (".scheduler", "GlobalScheduler"),
    "ScheduledAgent": (".scheduler", "ScheduledAgent"),
}

__all__ = [
    "GlobalScheduler",
    "InputType",
    "InvalidInputTypeError",
    "InvalidScheduleError",
    "ScheduleBuilder",
    "ScheduleParser",
    "ScheduledAgent",
    "ScheduledInput",
    "SchedulingError",
    "parse_magic_prefix",
    "parse_schedule_string",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib

        module_path, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_path, __package__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
