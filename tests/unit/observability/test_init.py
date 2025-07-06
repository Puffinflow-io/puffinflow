"""Tests for observability module initialization and imports."""


class TestObservabilityModuleImports:
    """Test observability module imports and __all__ exports."""

    def test_module_imports_successfully(self):
        """Test that the observability module can be imported."""
        import puffinflow.core.observability

        assert puffinflow.core.observability is not None

    def test_all_exports_defined(self):
        """Test that __all__ is properly defined."""
        from puffinflow.core.observability import __all__

        expected_exports = [
            "ObservabilityManager",
            "get_observability",
            "setup_observability",
            "ObservabilityConfig",
            "observe",
            "trace_state",
            "ObservableContext",
            "ObservableAgent",
            "core",
            "config",
            "decorators",
            "context",
            "agent",
            "alerting",
            "events",
            "interfaces",
            "metrics",
            "tracing",
        ]

        assert isinstance(__all__, list)
        assert len(__all__) == len(expected_exports)

        for export in expected_exports:
            assert export in __all__, f"Missing export: {export}"

    def test_core_imports(self):
        """Test core observability imports."""
        from puffinflow.core.observability import (
            ObservabilityManager,
            get_observability,
            setup_observability,
        )

        assert ObservabilityManager is not None
        assert callable(get_observability)
        assert callable(setup_observability)

    def test_config_imports(self):
        """Test configuration imports."""
        from puffinflow.core.observability import ObservabilityConfig

        assert ObservabilityConfig is not None

    def test_decorator_imports(self):
        """Test decorator imports."""
        from puffinflow.core.observability import observe, trace_state

        assert callable(observe)
        assert callable(trace_state)

    def test_context_imports(self):
        """Test context imports."""
        from puffinflow.core.observability import ObservableContext

        assert ObservableContext is not None

    def test_agent_imports(self):
        """Test agent imports."""
        from puffinflow.core.observability import ObservableAgent

        assert ObservableAgent is not None

    def test_module_docstring(self):
        """Test that the module has proper documentation."""
        import puffinflow.core.observability

        assert puffinflow.core.observability.__doc__ is not None
        assert "observability" in puffinflow.core.observability.__doc__.lower()

    def test_import_star_behavior(self):
        """Test that import * works correctly with __all__."""
        import puffinflow.core.observability

        all_exports = puffinflow.core.observability.__all__

        # Verify that all items in __all__ are actually available in the module
        for export_name in all_exports:
            assert hasattr(
                puffinflow.core.observability, export_name
            ), f"Export '{export_name}' in __all__ but not available in module"

    def test_no_unexpected_exports(self):
        """Test that only expected items are exported."""
        import puffinflow.core.observability

        # Get all public attributes (not starting with _)
        public_attrs = [
            attr
            for attr in dir(puffinflow.core.observability)
            if not attr.startswith("_")
        ]

        # All public attributes should be in __all__
        for attr in public_attrs:
            if attr != "__all__":  # __all__ itself is not in __all__
                assert (
                    attr in puffinflow.core.observability.__all__
                ), f"Public attribute '{attr}' not in __all__"


class TestObservabilityModuleStructure:
    """Test observability module structure and organization."""

    def test_module_has_required_attributes(self):
        """Test that the module has all required attributes."""
        import puffinflow.core.observability

        required_attrs = ["__all__", "__doc__"]
        for attr in required_attrs:
            assert hasattr(
                puffinflow.core.observability, attr
            ), f"Module missing required attribute: {attr}"

    def test_all_is_list_of_strings(self):
        """Test that __all__ contains only strings."""
        from puffinflow.core.observability import __all__

        assert isinstance(__all__, list)
        for item in __all__:
            assert isinstance(item, str), f"__all__ contains non-string item: {item}"

    def test_no_duplicate_exports(self):
        """Test that __all__ contains no duplicates."""
        from puffinflow.core.observability import __all__

        assert len(__all__) == len(set(__all__)), "Duplicate items found in __all__"

    def test_exports_are_logically_grouped(self):
        """Test that exports are logically organized."""
        from puffinflow.core.observability import __all__

        # Verify that related items are present
        core_items = [
            item
            for item in __all__
            if any(keyword in item.lower() for keyword in ["manager", "get_", "setup_"])
        ]
        config_items = [item for item in __all__ if "config" in item.lower()]
        decorator_items = [
            item
            for item in __all__
            if any(keyword in item.lower() for keyword in ["observe", "trace"])
        ]
        context_items = [item for item in __all__ if "context" in item.lower()]
        agent_items = [item for item in __all__ if "agent" in item.lower()]

        # Verify we have items in each category
        assert len(core_items) > 0, "No core observability items found"
        assert len(config_items) > 0, "No config items found"
        assert len(decorator_items) > 0, "No decorator items found"
        assert len(context_items) > 0, "No context items found"
        assert len(agent_items) > 0, "No agent items found"


class TestObservabilityFunctionality:
    """Test basic observability functionality."""

    def test_observability_manager_available(self):
        """Test that ObservabilityManager is available and usable."""
        from puffinflow.core.observability import ObservabilityManager

        # Should be able to reference the class
        assert ObservabilityManager is not None
        assert hasattr(ObservabilityManager, "__name__")

    def test_get_observability_callable(self):
        """Test that get_observability function is callable."""
        from puffinflow.core.observability import get_observability

        assert callable(get_observability)

    def test_setup_observability_callable(self):
        """Test that setup_observability function is callable."""
        from puffinflow.core.observability import setup_observability

        assert callable(setup_observability)

    def test_observe_decorator_callable(self):
        """Test that observe decorator is callable."""
        from puffinflow.core.observability import observe

        assert callable(observe)

    def test_trace_state_decorator_callable(self):
        """Test that trace_state decorator is callable."""
        from puffinflow.core.observability import trace_state

        assert callable(trace_state)

    def test_config_class_available(self):
        """Test that ObservabilityConfig class is available."""
        from puffinflow.core.observability import ObservabilityConfig

        assert ObservabilityConfig is not None
        assert hasattr(ObservabilityConfig, "__name__")

    def test_context_class_available(self):
        """Test that ObservableContext class is available."""
        from puffinflow.core.observability import ObservableContext

        assert ObservableContext is not None
        assert hasattr(ObservableContext, "__name__")

    def test_agent_class_available(self):
        """Test that ObservableAgent class is available."""
        from puffinflow.core.observability import ObservableAgent

        assert ObservableAgent is not None
        assert hasattr(ObservableAgent, "__name__")


class TestObservabilityImportPaths:
    """Test that imports are from correct paths."""

    def test_core_import_path(self):
        """Test that core imports are from the correct module."""
        # This test ensures the imports are structured correctly
        import puffinflow.core.observability.core

        # Verify the core module exists
        assert hasattr(puffinflow.core.observability.core, "ObservabilityManager")
        assert hasattr(puffinflow.core.observability.core, "get_observability")
        assert hasattr(puffinflow.core.observability.core, "setup_observability")

    def test_config_import_path(self):
        """Test that config imports are from the correct module."""
        import puffinflow.core.observability.config

        assert hasattr(puffinflow.core.observability.config, "ObservabilityConfig")

    def test_decorators_import_path(self):
        """Test that decorator imports are from the correct module."""
        import puffinflow.core.observability.decorators

        assert hasattr(puffinflow.core.observability.decorators, "observe")
        assert hasattr(puffinflow.core.observability.decorators, "trace_state")

    def test_context_import_path(self):
        """Test that context imports are from the correct module."""
        import puffinflow.core.observability.context

        assert hasattr(puffinflow.core.observability.context, "ObservableContext")

    def test_agent_import_path(self):
        """Test that agent imports are from the correct module."""
        import puffinflow.core.observability.agent

        assert hasattr(puffinflow.core.observability.agent, "ObservableAgent")
