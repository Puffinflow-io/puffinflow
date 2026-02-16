"""Code formatter using black."""

from __future__ import annotations


def format_code(code: str) -> str:
    """Format Python source code with black, falling back to raw code."""
    try:
        import black

        return black.format_str(code, mode=black.Mode())
    except ImportError:
        return code
    except Exception:
        return code
