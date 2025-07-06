"""Tests for core module initialization."""

import pytest


class TestCoreInit:
    """Test core module initialization."""

    def test_core_module_imports(self):
        """Test that core module can be imported."""
        import puffinflow.core

        assert puffinflow.core is not None

    def test_core_module_attributes(self):
        """Test core module has expected attributes."""
        import puffinflow.core

        # The core module should be importable
        assert hasattr(puffinflow.core, "__name__")
        assert puffinflow.core.__name__ == "puffinflow.core"

    def test_core_submodules_importable(self):
        """Test that core submodules are importable."""
        # Test that main submodules can be imported
        from puffinflow.core import (
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
