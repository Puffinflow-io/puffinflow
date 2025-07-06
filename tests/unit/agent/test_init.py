"""Tests for agent module initialization and convenience functions."""

from unittest.mock import Mock, patch

import pytest

from puffinflow.core.agent import (
    # Core classes
    Agent,
    AgentCheckpoint,
    AgentResult,
    AgentStatus,
    Context,
    DeadLetter,
    DependencyConfig,
    DependencyLifecycle,
    # Dependencies
    DependencyType,
    PrioritizedState,
    # State management
    Priority,
    ResourceTimeoutError,
    RetryPolicy,
    StateMetadata,
    StateResult,
    StateStatus,
    StateType,
    create_environment_decorator,
    create_external_team_decorator,
    create_reliable_team_decorator,
    create_service_decorator,
    # Team decorators
    create_team_decorator,
)


class TestModuleImports:
    """Test that all expected imports are available."""

    def test_core_classes_imported(self):
        """Test that core classes are properly imported."""
        assert Agent is not None
        assert AgentResult is not None
        assert Context is not None
        assert AgentCheckpoint is not None
        assert ResourceTimeoutError is not None

    def test_state_management_imported(self):
        """Test that state management classes are imported."""
        assert Priority is not None
        assert AgentStatus is not None
        assert StateStatus is not None
        assert StateMetadata is not None
        assert PrioritizedState is not None
        assert RetryPolicy is not None
        assert DeadLetter is not None
        assert StateResult is not None
        assert StateType is not None

    def test_dependency_classes_imported(self):
        """Test that dependency classes are imported."""
        assert DependencyType is not None
        assert DependencyLifecycle is not None
        assert DependencyConfig is not None


class TestTeamDecorators:
    """Test team decorator creation functions."""

    def test_create_team_decorator_with_decorators_available(self):
        """Test creating team decorator when decorators are available."""
        with patch(
            "puffinflow.core.agent.decorators.flexible.create_custom_decorator"
        ) as mock_create:
            mock_decorator = Mock()
            mock_create.return_value = mock_decorator

            decorator = create_team_decorator("backend", priority="high")

            assert decorator is mock_decorator
            mock_create.assert_called_once_with(
                tags={"team": "backend"},
                description="Agent for backend team",
                priority="high",
            )

    def test_create_team_decorator_without_decorators(self):
        """Test creating team decorator when decorators are not available."""
        with patch(
            "puffinflow.core.agent.decorators.flexible.create_custom_decorator",
            side_effect=ImportError,
        ):
            decorator = create_team_decorator("backend")

            # Should return a lambda that just returns the function unchanged
            def test_func(x):
                return x

            result = decorator(test_func)
            assert result is test_func

    def test_create_environment_decorator_with_decorators_available(self):
        """Test creating environment decorator when decorators are available."""
        with patch(
            "puffinflow.core.agent.decorators.flexible.create_custom_decorator"
        ) as mock_create:
            mock_decorator = Mock()
            mock_create.return_value = mock_decorator

            decorator = create_environment_decorator("production", timeout=60.0)

            assert decorator is mock_decorator
            mock_create.assert_called_once_with(
                tags={"environment": "production"},
                description="Agent for production environment",
                timeout=60.0,
            )

    def test_create_environment_decorator_without_decorators(self):
        """Test creating environment decorator when decorators are not available."""
        with patch(
            "puffinflow.core.agent.decorators.flexible.create_custom_decorator",
            side_effect=ImportError,
        ):
            decorator = create_environment_decorator("production")

            # Should return a lambda that just returns the function unchanged
            def test_func(x):
                return x

            result = decorator(test_func)
            assert result is test_func

    def test_create_service_decorator_with_decorators_available(self):
        """Test creating service decorator when decorators are available."""
        with patch(
            "puffinflow.core.agent.decorators.flexible.create_custom_decorator"
        ) as mock_create:
            mock_decorator = Mock()
            mock_create.return_value = mock_decorator

            decorator = create_service_decorator("auth-service", cpu=2.0)

            assert decorator is mock_decorator
            mock_create.assert_called_once_with(
                tags={"service": "auth-service"},
                description="Agent for auth-service service",
                cpu=2.0,
            )

    def test_create_service_decorator_without_decorators(self):
        """Test creating service decorator when decorators are not available."""
        with patch(
            "puffinflow.core.agent.decorators.flexible.create_custom_decorator",
            side_effect=ImportError,
        ):
            decorator = create_service_decorator("auth-service")

            # Should return a lambda that just returns the function unchanged
            def test_func(x):
                return x

            result = decorator(test_func)
            assert result is test_func

    def test_create_reliable_team_decorator_with_decorators_available(self):
        """Test creating reliable team decorator when decorators are available."""
        with patch(
            "puffinflow.core.agent.decorators.flexible.create_custom_decorator"
        ) as mock_create:
            mock_decorator = Mock()
            mock_create.return_value = mock_decorator

            decorator = create_reliable_team_decorator("ml-team", max_retries=10)

            assert decorator is mock_decorator
            mock_create.assert_called_once_with(
                tags={"team": "ml-team", "reliability": "high"},
                circuit_breaker=True,
                bulkhead=True,
                retries=5,
                max_retries=10,
            )

    def test_create_reliable_team_decorator_without_decorators(self):
        """Test creating reliable team decorator when decorators are not available."""
        with patch(
            "puffinflow.core.agent.decorators.flexible.create_custom_decorator",
            side_effect=ImportError,
        ):
            decorator = create_reliable_team_decorator("ml-team")

            # Should return a lambda that just returns the function unchanged
            def test_func(x):
                return x

            result = decorator(test_func)
            assert result is test_func

    def test_create_external_team_decorator_with_decorators_available(self):
        """Test creating external team decorator when decorators are available."""
        with patch(
            "puffinflow.core.agent.decorators.flexible.create_custom_decorator"
        ) as mock_create:
            mock_decorator = Mock()
            mock_create.return_value = mock_decorator

            decorator = create_external_team_decorator("api-team", timeout=45.0)

            assert decorator is mock_decorator
            mock_create.assert_called_once_with(
                tags={"team": "api-team", "type": "external"},
                circuit_breaker=True,
                timeout=45.0,
                retries=3,
            )

    def test_create_external_team_decorator_without_decorators(self):
        """Test creating external team decorator when decorators are not available."""
        with patch(
            "puffinflow.core.agent.decorators.flexible.create_custom_decorator",
            side_effect=ImportError,
        ):
            decorator = create_external_team_decorator("api-team")

            # Should return a lambda that just returns the function unchanged
            def test_func(x):
                return x

            result = decorator(test_func)
            assert result is test_func


class TestDecoratorImportHandling:
    """Test handling of decorator import failures."""

    def test_decorator_imports_with_import_error(self):
        """Test that decorator imports are handled gracefully when they fail."""
        # This test ensures that the try/except block in __init__.py works
        # We can't easily mock the import at module level, but we can test
        # that the functions handle ImportError correctly

        # The decorators should be available in normal circumstances
        try:
            from puffinflow.core.agent.decorators.flexible import state

            assert state is not None
        except ImportError:
            # If decorators aren't available, the module should still load
            pass

    def test_team_decorators_handle_missing_decorators(self):
        """Test that team decorators work even when flexible decorators are missing."""
        # Test all team decorator functions with mocked ImportError
        functions_to_test = [
            create_team_decorator,
            create_environment_decorator,
            create_service_decorator,
            create_reliable_team_decorator,
            create_external_team_decorator,
        ]

        for func in functions_to_test:
            with patch(
                "puffinflow.core.agent.decorators.flexible.create_custom_decorator",
                side_effect=ImportError,
            ):
                # Should not raise an exception
                decorator = func("test")
                assert callable(decorator)

                # Should return function unchanged
                def test_func(x):
                    return x

                result = decorator(test_func)
                assert result is test_func


class TestModuleDocstring:
    """Test module documentation and metadata."""

    def test_module_has_docstring(self):
        """Test that the module has proper documentation."""
        import puffinflow.core.agent as agent_module

        assert hasattr(agent_module, "__doc__")
        assert agent_module.__doc__ is not None
        assert len(agent_module.__doc__) > 100  # Should have substantial documentation

    def test_module_has_all_attribute(self):
        """Test that the module defines __all__ properly."""
        import puffinflow.core.agent as agent_module

        assert hasattr(agent_module, "__all__")
        assert isinstance(agent_module.__all__, list)
        assert len(agent_module.__all__) > 20  # Should export many items

    def test_all_exports_are_available(self):
        """Test that all items in __all__ are actually available."""
        import puffinflow.core.agent as agent_module

        for item_name in agent_module.__all__:
            assert hasattr(agent_module, item_name), f"Module should export {item_name}"
            item = getattr(agent_module, item_name)
            assert item is not None, f"Exported item {item_name} should not be None"


class TestConvenienceDecorators:
    """Test convenience decorator functions with various parameters."""

    def test_team_decorator_with_multiple_parameters(self):
        """Test team decorator with multiple custom parameters."""
        with patch(
            "puffinflow.core.agent.decorators.flexible.create_custom_decorator"
        ) as mock_create:
            mock_decorator = Mock()
            mock_create.return_value = mock_decorator

            create_team_decorator(
                "data-team",
                cpu=4.0,
                memory=2048.0,
                priority="high",
                timeout=120.0,
                custom_param="value",
            )

            expected_call_args = {
                "tags": {"team": "data-team"},
                "description": "Agent for data-team team",
                "cpu": 4.0,
                "memory": 2048.0,
                "priority": "high",
                "timeout": 120.0,
                "custom_param": "value",
            }

            mock_create.assert_called_once_with(**expected_call_args)

    def test_environment_decorator_with_custom_description(self):
        """Test environment decorator with custom parameters."""
        with patch(
            "puffinflow.core.agent.decorators.flexible.create_custom_decorator"
        ) as mock_create:
            mock_decorator = Mock()
            mock_create.return_value = mock_decorator

            create_environment_decorator(
                "staging", description="Custom staging environment agent", retries=3
            )

            expected_call_args = {
                "tags": {"environment": "staging"},
                "description": "Custom staging environment agent",  # Should override default
                "retries": 3,
            }

            mock_create.assert_called_once_with(**expected_call_args)

    def test_reliable_team_decorator_overrides_defaults(self):
        """Test that reliable team decorator allows overriding defaults."""
        with patch(
            "puffinflow.core.agent.decorators.flexible.create_custom_decorator"
        ) as mock_create:
            mock_decorator = Mock()
            mock_create.return_value = mock_decorator

            create_reliable_team_decorator(
                "critical-team",
                circuit_breaker=False,  # Override default
                retries=10,  # Override default
                custom_reliability_setting=True,
            )

            expected_call_args = {
                "tags": {"team": "critical-team", "reliability": "high"},
                "circuit_breaker": False,  # Should be overridden
                "bulkhead": True,
                "retries": 10,  # Should be overridden
                "custom_reliability_setting": True,
            }

            mock_create.assert_called_once_with(**expected_call_args)

    def test_external_team_decorator_timeout_override(self):
        """Test that external team decorator allows timeout override."""
        with patch(
            "puffinflow.core.agent.decorators.flexible.create_custom_decorator"
        ) as mock_create:
            mock_decorator = Mock()
            mock_create.return_value = mock_decorator

            create_external_team_decorator(
                "external-api",
                timeout=60.0,  # Override default 30.0
                retries=5,  # Override default 3
            )

            expected_call_args = {
                "tags": {"team": "external-api", "type": "external"},
                "circuit_breaker": True,
                "timeout": 60.0,  # Should override default
                "retries": 5,  # Should override default
            }

            mock_create.assert_called_once_with(**expected_call_args)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
