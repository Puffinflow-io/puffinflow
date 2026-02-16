"""Tests for main package __init__.py module."""

import pytest

import puffinflow


class TestPackageMetadata:
    """Test package metadata and constants."""

    def test_version_exists(self):
        """Test that version is defined."""
        assert hasattr(puffinflow, "__version__")
        assert isinstance(puffinflow.__version__, str)
        assert len(puffinflow.__version__) > 0

    def test_author_exists(self):
        """Test that author is defined."""
        assert hasattr(puffinflow, "__author__")
        assert isinstance(puffinflow.__author__, str)
        assert puffinflow.__author__ == "Mohamed Ahmed"

    def test_email_exists(self):
        """Test that email is defined."""
        assert hasattr(puffinflow, "__email__")
        assert isinstance(puffinflow.__email__, str)
        assert puffinflow.__email__ == "mohamed.ahmed.4894@gmail.com"


class TestCoreAgentImports:
    """Test core agent functionality imports."""

    def test_agent_imports(self):
        """Test that core agent classes are importable."""
        from puffinflow import Agent, AgentCheckpoint, AgentResult, Context

        assert Agent is not None
        assert AgentResult is not None
        assert Context is not None
        assert AgentCheckpoint is not None

    def test_status_enums_imports(self):
        """Test that status enums are importable."""
        from puffinflow import AgentStatus, Priority, StateResult, StateStatus

        assert Priority is not None
        assert AgentStatus is not None
        assert StateStatus is not None
        assert StateResult is not None

    def test_decorator_imports(self):
        """Test that decorators are importable."""
        from puffinflow import (
            cpu_intensive,
            critical_state,
            gpu_accelerated,
            io_intensive,
            memory_intensive,
            network_intensive,
            state,
        )

        assert state is not None
        assert cpu_intensive is not None
        assert memory_intensive is not None
        assert io_intensive is not None
        assert gpu_accelerated is not None
        assert network_intensive is not None
        assert critical_state is not None

    def test_builder_imports(self):
        """Test that builder classes are importable."""
        from puffinflow import StateBuilder, build_state

        assert build_state is not None
        assert StateBuilder is not None


class TestCoordinationImports:
    """Test coordination functionality imports."""

    def test_team_imports(self):
        """Test that team classes are importable."""
        from puffinflow import AgentGroup, AgentTeam, ParallelAgentGroup, TeamResult

        assert AgentTeam is not None
        assert TeamResult is not None
        assert AgentGroup is not None
        assert ParallelAgentGroup is not None

    def test_orchestrator_imports(self):
        """Test that orchestrator classes are importable."""
        from puffinflow import AgentOrchestrator, Agents

        assert AgentOrchestrator is not None
        assert Agents is not None

    def test_execution_function_imports(self):
        """Test that execution functions are importable."""
        from puffinflow import run_agents_parallel, run_agents_sequential

        assert run_agents_parallel is not None
        assert run_agents_sequential is not None

    def test_pool_imports(self):
        """Test that pool classes are importable."""
        from puffinflow import AgentPool, DynamicProcessingPool, WorkItem, WorkQueue

        assert AgentPool is not None
        assert WorkQueue is not None
        assert WorkItem is not None
        assert DynamicProcessingPool is not None

    def test_event_and_factory_imports(self):
        """Test that event and factory functions are importable."""
        from puffinflow import EventBus, create_pipeline, create_team

        assert EventBus is not None
        assert create_team is not None
        assert create_pipeline is not None


class TestResourceImports:
    """Test resource management imports."""

    def test_resource_imports(self):
        """Test that resource classes are importable."""
        from puffinflow import (
            AllocationStrategy,
            QuotaManager,
            ResourcePool,
            ResourceRequirements,
            ResourceType,
        )

        assert ResourceRequirements is not None
        assert ResourceType is not None
        assert ResourcePool is not None
        assert QuotaManager is not None
        assert AllocationStrategy is not None


class TestReliabilityImports:
    """Test reliability pattern imports."""

    def test_circuit_breaker_imports(self):
        """Test that circuit breaker classes are importable."""
        from puffinflow import CircuitBreaker, CircuitBreakerConfig

        assert CircuitBreaker is not None
        assert CircuitBreakerConfig is not None

    def test_bulkhead_imports(self):
        """Test that bulkhead classes are importable."""
        from puffinflow import Bulkhead, BulkheadConfig

        assert Bulkhead is not None
        assert BulkheadConfig is not None

    def test_leak_detector_imports(self):
        """Test that leak detector is importable."""
        from puffinflow import ResourceLeakDetector

        assert ResourceLeakDetector is not None


class TestConfigurationImports:
    """Test configuration imports."""

    def test_config_imports(self):
        """Test that configuration classes are importable."""
        from puffinflow import Features, Settings, get_features, get_settings

        assert Settings is not None
        assert get_settings is not None
        assert Features is not None
        assert get_features is not None


class TestAllExports:
    """Test __all__ exports."""

    def test_all_defined(self):
        """Test that __all__ is defined."""
        assert hasattr(puffinflow, "__all__")
        assert isinstance(puffinflow.__all__, list)
        assert len(puffinflow.__all__) > 0

    def test_all_exports_importable(self):
        """Test that all items in __all__ are importable."""
        for item in puffinflow.__all__:
            assert hasattr(puffinflow, item), f"Item '{item}' not found in module"
            assert getattr(puffinflow, item) is not None, f"Item '{item}' is None"

    def test_core_exports_in_all(self):
        """Test that core exports are in __all__."""
        core_exports = [
            "Agent",
            "AgentResult",
            "Context",
            "AgentCheckpoint",
            "Priority",
            "AgentStatus",
            "StateStatus",
            "StateResult",
        ]

        for export in core_exports:
            assert (
                export in puffinflow.__all__
            ), f"Core export '{export}' missing from __all__"

    def test_decorator_exports_in_all(self):
        """Test that decorator exports are in __all__."""
        decorator_exports = [
            "state",
            "cpu_intensive",
            "memory_intensive",
            "io_intensive",
            "gpu_accelerated",
            "network_intensive",
            "critical_state",
            "build_state",
            "StateBuilder",
        ]

        for export in decorator_exports:
            assert (
                export in puffinflow.__all__
            ), f"Decorator export '{export}' missing from __all__"

    def test_coordination_exports_in_all(self):
        """Test that coordination exports are in __all__."""
        coordination_exports = [
            "AgentTeam",
            "TeamResult",
            "AgentGroup",
            "ParallelAgentGroup",
            "AgentOrchestrator",
            "Agents",
            "run_agents_parallel",
            "run_agents_sequential",
            "AgentPool",
            "WorkQueue",
            "WorkItem",
            "DynamicProcessingPool",
            "EventBus",
            "create_team",
            "create_pipeline",
        ]

        for export in coordination_exports:
            assert (
                export in puffinflow.__all__
            ), f"Coordination export '{export}' missing from __all__"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_version_function(self):
        """Test get_version function."""
        assert hasattr(puffinflow, "get_version")
        version = puffinflow.get_version()
        assert isinstance(version, str)
        assert version == puffinflow.__version__

    def test_get_info_function(self):
        """Test get_info function."""
        assert hasattr(puffinflow, "get_info")
        info = puffinflow.get_info()

        assert isinstance(info, dict)
        assert "version" in info
        assert "author" in info
        assert "email" in info
        assert "description" in info

        assert info["version"] == puffinflow.__version__
        assert info["author"] == puffinflow.__author__
        assert info["email"] == puffinflow.__email__
        assert isinstance(info["description"], str)
        assert len(info["description"]) > 0


class TestModuleStructure:
    """Test module structure and organization."""

    def test_docstring_exists(self):
        """Test that module has a docstring."""
        assert puffinflow.__doc__ is not None
        assert isinstance(puffinflow.__doc__, str)
        assert "PuffinFlow" in puffinflow.__doc__
        assert "Workflow Orchestration Framework" in puffinflow.__doc__

    def test_no_private_exports(self):
        """Test that private items are not in __all__."""
        for item in puffinflow.__all__:
            assert not item.startswith(
                "_"
            ), f"Private item '{item}' should not be in __all__"

    def test_import_structure_consistency(self):
        """Test that imports are organized consistently."""
        # All items in __all__ should be importable from their respective modules
        from puffinflow.core.agent import Agent
        from puffinflow.core.config import Settings
        from puffinflow.core.coordination import AgentTeam
        from puffinflow.core.reliability import CircuitBreaker
        from puffinflow.core.resources import ResourceRequirements

        # These should be the same objects as imported in __init__.py
        assert puffinflow.Agent is Agent
        assert puffinflow.AgentTeam is AgentTeam
        assert puffinflow.ResourceRequirements is ResourceRequirements
        assert puffinflow.CircuitBreaker is CircuitBreaker
        assert puffinflow.Settings is Settings


class TestImportPerformance:
    """Test import performance and lazy loading."""

    def test_fast_import(self):
        """Test that importing puffinflow is reasonably fast."""
        import time

        start_time = time.time()
        import importlib

        importlib.reload(puffinflow)
        end_time = time.time()

        # Import should take less than 1 second
        import_time = end_time - start_time
        assert import_time < 1.0, f"Import took {import_time:.2f}s, which is too slow"

    def test_no_circular_imports(self):
        """Test that there are no circular import issues."""
        # If we can import all the main exports without error,
        # there are likely no circular import issues
        try:
            from puffinflow import (
                Agent,  # noqa: F401
                AgentTeam,  # noqa: F401
                CircuitBreaker,  # noqa: F401
                ResourceRequirements,  # noqa: F401
                Settings,  # noqa: F401
            )

            # If we reach here, no circular imports detected
            assert True
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")
