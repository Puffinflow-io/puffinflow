"""Abstract base class for deployment generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ir import WorkflowIR


class DeployGenerator(ABC):
    """Base class for platform-specific deploy file generators."""

    @abstractmethod
    def generate(self, ir: WorkflowIR, python_code: str, output_dir: str) -> list[str]:
        """Generate deploy files.

        Args:
            ir: The workflow intermediate representation.
            python_code: The generated agent Python source code.
            output_dir: Directory to write generated files into.

        Returns:
            List of generated file paths.
        """
        ...
