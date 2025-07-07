"""Tests for version module."""

import pytest

from puffinflow import version


class TestVersionModule:
    """Test version module functionality."""

    def test_version_attributes_exist(self):
        """Test that all version attributes exist."""
        assert hasattr(version, "__version__")
        assert hasattr(version, "__version_tuple__")
        assert hasattr(version, "version")
        assert hasattr(version, "version_tuple")

    def test_version_is_string(self):
        """Test that version is a string."""
        assert isinstance(version.__version__, str)
        assert isinstance(version.version, str)

    def test_version_tuple_is_tuple(self):
        """Test that version tuple is a tuple."""
        assert isinstance(version.__version_tuple__, tuple)
        assert isinstance(version.version_tuple, tuple)

    def test_version_consistency(self):
        """Test that version attributes are consistent."""
        assert version.__version__ == version.version
        assert version.__version_tuple__ == version.version_tuple

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        expected_exports = [
            "__version__",
            "__version_tuple__",
            "version",
            "version_tuple",
        ]
        assert version.__all__ == expected_exports

    def test_type_checking_constant(self):
        """Test TYPE_CHECKING constant."""
        assert hasattr(version, "TYPE_CHECKING")
        assert version.TYPE_CHECKING is False

    def test_version_tuple_type(self):
        """Test VERSION_TUPLE type definition."""
        assert hasattr(version, "VERSION_TUPLE")
        # When TYPE_CHECKING is False, VERSION_TUPLE should be object
        assert version.VERSION_TUPLE is object


class TestVersionValues:
    """Test specific version values."""


class TestVersionImports:
    """Test version module imports and accessibility."""

    def test_direct_import(self):
        """Test direct import of version module."""
        from puffinflow import version as v

        assert hasattr(v, "__version__")

    def test_version_accessible_from_main_package(self):
        """Test that version info is accessible from main package."""
        import puffinflow

        # The main package should have version info
        assert hasattr(puffinflow, "__version__")


class TestVersionEdgeCases:
    """Test edge cases and error conditions."""

    def test_version_not_none(self):
        """Test that version values are not None."""
        assert version.__version__ is not None
        assert version.__version_tuple__ is not None
        assert version.version is not None
        assert version.version_tuple is not None

    def test_version_not_empty(self):
        """Test that version strings are not empty."""
        assert len(version.__version__) > 0
        assert len(version.version) > 0
        assert len(version.__version_tuple__) > 0
        assert len(version.version_tuple) > 0

    def test_version_immutable(self):
        """Test that version tuple is immutable."""
        original_tuple = version.__version_tuple__

        # Attempting to modify should raise TypeError
        with pytest.raises(TypeError):
            version.__version_tuple__[0] = 999

        # Original should remain unchanged
        assert version.__version_tuple__ == original_tuple
