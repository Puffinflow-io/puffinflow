"""Tests for core module initialization."""

import pytest


class TestCoreInit:
    """Test core module initialization."""

    def test_core_module_imports(self):
        """Test that core module can be imported."""
        import src.puffinflow.core

        assert src.puffinflow.core is not None

    def test_core_module_attributes(self):
        """Test core module has expected attributes."""
        import src.puffinflow.core

        # The core module should be importable
        assert hasattr(src.puffinflow.core, "__name__")
        assert src.puffinflow.core.__name__ == "src.puffinflow.core"

    def test_core_submodules_importable(self):
        """Test that core submodules are importable."""
        # Test that main submodules can be imported
        from src.puffinflow.core import (
            agent,
            config,
            coordination,
            observability,
            reliability,
            resources,
        )

        assert config is not None
        assert agent is not None
        assert coordination is not None
        assert observability is not None
        assert reliability is not None
        assert resources is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
