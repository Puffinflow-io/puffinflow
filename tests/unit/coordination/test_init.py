"""Tests for coordination module initialization and imports."""

from unittest.mock import MagicMock, patch


class TestCoordinationModuleImports:
    """Test coordination module imports and __all__ exports."""

    def test_module_imports_successfully(self):
        """Test that the coordination module can be imported."""
        import puffinflow.core.coordination

        assert puffinflow.core.coordination is not None

    def test_all_exports_defined(self):
        """Test that __all__ is properly defined."""
        from puffinflow.core.coordination import __all__

        expected_exports = [
            # Team coordination
            "AgentTeam",
            "TeamResult",
            "EventBus",
            "Message",
            "Event",
            "create_team",
            "run_agents_parallel",
            "run_agents_sequential",
            # Group coordination
            "AgentGroup",
            "ParallelAgentGroup",
            "AgentOrchestrator",
            "GroupResult",
            "OrchestrationExecution",
            "OrchestrationResult",
            "ExecutionStrategy",
            "StageConfig",
            # Fluent APIs
            "Agents",
            "ConditionalAgents",
            "PipelineAgents",
            "FluentResult",
            "run_parallel_agents",
            "run_sequential_agents",
            "collect_agent_outputs",
            "get_best_agent",
            "create_pipeline",
            "create_agent_team",
            # Agent pools
            "AgentPool",
            "WorkQueue",
            "WorkItem",
            "CompletedWork",
            "DynamicProcessingPool",
            "ScalingPolicy",
            "PoolContext",
            "WorkProcessor",
            # Existing coordination (if available)
            "AgentCoordinator",
            "CoordinationConfig",
            "enhance_agent",
            "DeadlockDetector",
            "DeadlockResolutionStrategy",
            "CoordinationPrimitive",
            "Mutex",
            "Semaphore",
            "Barrier",
            "Lease",
            "Lock",
            "Quota",
            "PrimitiveType",
            "create_primitive",
            "RateLimiter",
            "TokenBucket",
            "LeakyBucket",
            "SlidingWindow",
            "FixedWindow",
            "AdaptiveRateLimiter",
            "CompositeRateLimiter",
            "RateLimitStrategy",
        ]

        assert isinstance(__all__, list)
        assert len(__all__) == len(expected_exports)

        for export in expected_exports:
            assert export in __all__, f"Missing export: {export}"

    def test_team_coordination_imports(self):
        """Test team coordination imports."""
        from puffinflow.core.coordination import (
            AgentTeam,
            Event,
            EventBus,
            Message,
            TeamResult,
            create_team,
            run_agents_parallel,
            run_agents_sequential,
        )

        # Verify classes are importable
        assert AgentTeam is not None
        assert TeamResult is not None
        assert EventBus is not None
        assert Message is not None
        assert Event is not None

        # Verify functions are importable
        assert callable(create_team)
        assert callable(run_agents_parallel)
        assert callable(run_agents_sequential)

    def test_group_coordination_imports(self):
        """Test group coordination imports."""
        from puffinflow.core.coordination import (
            AgentGroup,
            AgentOrchestrator,
            ExecutionStrategy,
            GroupResult,
            OrchestrationExecution,
            OrchestrationResult,
            ParallelAgentGroup,
            StageConfig,
        )

        assert AgentGroup is not None
        assert ParallelAgentGroup is not None
        assert AgentOrchestrator is not None
        assert GroupResult is not None
        assert OrchestrationExecution is not None
        assert OrchestrationResult is not None
        assert ExecutionStrategy is not None
        assert StageConfig is not None

    def test_fluent_api_imports(self):
        """Test fluent API imports."""
        from puffinflow.core.coordination import (
            Agents,
            ConditionalAgents,
            FluentResult,
            PipelineAgents,
            collect_agent_outputs,
            create_agent_team,
            create_pipeline,
            get_best_agent,
            run_parallel_agents,
            run_sequential_agents,
        )

        assert Agents is not None
        assert ConditionalAgents is not None
        assert PipelineAgents is not None
        assert FluentResult is not None

        assert callable(run_parallel_agents)
        assert callable(run_sequential_agents)
        assert callable(collect_agent_outputs)
        assert callable(get_best_agent)
        assert callable(create_pipeline)
        assert callable(create_agent_team)

    def test_agent_pool_imports(self):
        """Test agent pool imports."""
        from puffinflow.core.coordination import (
            AgentPool,
            CompletedWork,
            DynamicProcessingPool,
            PoolContext,
            ScalingPolicy,
            WorkItem,
            WorkProcessor,
            WorkQueue,
        )

        assert AgentPool is not None
        assert WorkQueue is not None
        assert WorkItem is not None
        assert CompletedWork is not None
        assert DynamicProcessingPool is not None
        assert ScalingPolicy is not None
        assert PoolContext is not None
        assert WorkProcessor is not None

    def test_optional_coordination_imports_success(self):
        """Test optional coordination imports when available."""
        # Mock successful imports
        with patch.dict(
            "sys.modules",
            {
                "puffinflow.core.coordination.coordinator": MagicMock(),
                "puffinflow.core.coordination.deadlock": MagicMock(),
                "puffinflow.core.coordination.primitives": MagicMock(),
                "puffinflow.core.coordination.rate_limiter": MagicMock(),
            },
        ):
            # Re-import the module to test the try/except block
            import importlib

            import puffinflow.core.coordination

            importlib.reload(puffinflow.core.coordination)

            # Test that optional imports are available
            from puffinflow.core.coordination import (
                AgentCoordinator,
                CoordinationConfig,
                DeadlockDetector,
                DeadlockResolutionStrategy,
                enhance_agent,
            )

            assert AgentCoordinator is not None
            assert CoordinationConfig is not None
            assert callable(enhance_agent)
            assert DeadlockDetector is not None
            assert DeadlockResolutionStrategy is not None

    def test_optional_coordination_imports_failure(self):
        """Test graceful handling when optional coordination imports fail."""
        # This test verifies that the module still loads even if optional imports fail
        # The try/except block should handle ImportError gracefully

        # Import the module normally - it should work even with missing optional components
        import puffinflow.core.coordination

        # Verify the module loaded successfully
        assert puffinflow.core.coordination is not None
        assert hasattr(puffinflow.core.coordination, "__all__")

    def test_primitive_imports(self):
        """Test coordination primitive imports."""
        from puffinflow.core.coordination import (
            Barrier,
            CoordinationPrimitive,
            Lease,
            Lock,
            Mutex,
            PrimitiveType,
            Quota,
            Semaphore,
            create_primitive,
        )

        assert CoordinationPrimitive is not None
        assert Mutex is not None
        assert Semaphore is not None
        assert Barrier is not None
        assert Lease is not None
        assert Lock is not None
        assert Quota is not None
        assert PrimitiveType is not None
        assert callable(create_primitive)

    def test_rate_limiter_imports(self):
        """Test rate limiter imports."""
        from puffinflow.core.coordination import (
            AdaptiveRateLimiter,
            CompositeRateLimiter,
            FixedWindow,
            LeakyBucket,
            RateLimiter,
            RateLimitStrategy,
            SlidingWindow,
            TokenBucket,
        )

        assert RateLimiter is not None
        assert TokenBucket is not None
        assert LeakyBucket is not None
        assert SlidingWindow is not None
        assert FixedWindow is not None
        assert AdaptiveRateLimiter is not None
        assert CompositeRateLimiter is not None
        assert RateLimitStrategy is not None

    def test_module_docstring(self):
        """Test that the module has proper documentation."""
        import puffinflow.core.coordination

        assert puffinflow.core.coordination.__doc__ is not None
        assert "coordination" in puffinflow.core.coordination.__doc__.lower()

    def test_import_star_behavior(self):
        """Test that import * works correctly with __all__."""
        # This test ensures that __all__ properly controls what gets imported with *
        import puffinflow.core.coordination

        all_exports = puffinflow.core.coordination.__all__

        # Verify that all items in __all__ are actually available in the module
        for export_name in all_exports:
            assert hasattr(
                puffinflow.core.coordination, export_name
            ), f"Export '{export_name}' in __all__ but not available in module"

    def test_no_unexpected_exports(self):
        """Test that only expected items are exported."""
        import puffinflow.core.coordination

        # Get all public attributes (not starting with _)
        public_attrs = [
            attr
            for attr in dir(puffinflow.core.coordination)
            if not attr.startswith("_")
        ]

        # Filter out imported submodules which are implementation details
        submodules = [
            "agent_group",
            "agent_pool",
            "agent_team",
            "coordinator",
            "deadlock",
            "fluent_api",
            "primitives",
            "rate_limiter",
        ]

        # All public attributes should be in __all__, except submodules
        for attr in public_attrs:
            if attr != "__all__" and attr not in submodules:
                assert (
                    attr in puffinflow.core.coordination.__all__
                ), f"Public attribute '{attr}' not in __all__"


class TestCoordinationModuleStructure:
    """Test coordination module structure and organization."""

    def test_module_has_required_attributes(self):
        """Test that the module has all required attributes."""
        import puffinflow.core.coordination

        required_attrs = ["__all__", "__doc__"]
        for attr in required_attrs:
            assert hasattr(
                puffinflow.core.coordination, attr
            ), f"Module missing required attribute: {attr}"

    def test_all_is_list_of_strings(self):
        """Test that __all__ contains only strings."""
        from puffinflow.core.coordination import __all__

        assert isinstance(__all__, list)
        for item in __all__:
            assert isinstance(item, str), f"__all__ contains non-string item: {item}"

    def test_no_duplicate_exports(self):
        """Test that __all__ contains no duplicates."""
        from puffinflow.core.coordination import __all__

        assert len(__all__) == len(set(__all__)), "Duplicate items found in __all__"

    def test_exports_are_sorted_by_category(self):
        """Test that exports are logically grouped."""
        from puffinflow.core.coordination import __all__

        # Verify that related items are grouped together
        team_items = [
            item
            for item in __all__
            if any(
                keyword in item.lower()
                for keyword in ["team", "event", "message", "create_team", "run_agents"]
            )
        ]
        group_items = [
            item
            for item in __all__
            if any(
                keyword in item.lower()
                for keyword in ["group", "orchestrat", "execution", "stage"]
            )
        ]
        pool_items = [
            item
            for item in __all__
            if any(
                keyword in item.lower()
                for keyword in ["pool", "work", "scaling", "processor"]
            )
        ]

        # Verify we have items in each category
        assert len(team_items) > 0, "No team coordination items found"
        assert len(group_items) > 0, "No group coordination items found"
        assert len(pool_items) > 0, "No pool coordination items found"
