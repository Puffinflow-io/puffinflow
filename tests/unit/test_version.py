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

    def test_version_format(self):
        """Test that version follows expected format."""
        # Version should contain at least major.minor
        version_parts = version.__version__.split(".")
        assert len(version_parts) >= 2

        # First part should be numeric
        assert version_parts[0].isdigit()
        assert version_parts[1].split("+")[0].split("dev")[0].isdigit()

    def test_version_tuple_structure(self):
        """Test that version tuple has expected structure."""
        version_tuple = version.__version_tuple__
        assert len(version_tuple) >= 2

        # First element should be integer (major version)
        assert isinstance(version_tuple[0], int)
        # Second element should be integer (minor version)
        assert isinstance(version_tuple[1], int)

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

    def test_current_version_format(self):
        """Test current version format matches expected pattern."""
        current_version = version.__version__

        # Should start with major.minor (flexible for different version schemes)
        import re

        # Match versions like "0.1.x", "1.0.x", etc.
        version_pattern = r"^\d+\.\d+"
        assert re.match(
            version_pattern, current_version
        ), f"Version {current_version} doesn't match expected pattern"

        # May contain dev, git hash, and date info
        if "dev" in current_version:
            assert "+g" in current_version or "dev" in current_version

    def test_current_version_tuple_format(self):
        """Test current version tuple format."""
        current_tuple = version.__version_tuple__

        # Should have at least major, minor version numbers
        assert len(current_tuple) >= 2
        assert isinstance(current_tuple[0], int)  # Major version
        assert isinstance(current_tuple[1], int)  # Minor version

        # May contain additional version info
        if len(current_tuple) > 2:
            # Third element might be dev info or patch version
            assert isinstance(current_tuple[2], (int, str))


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
