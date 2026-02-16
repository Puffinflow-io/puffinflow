"""Codegen service — wraps the code generator for use by API routes."""
from __future__ import annotations

import yaml

from studio.codegen.generator import CodeGenerator
from studio.codegen.ir import WorkflowIR
from studio.codegen.reverse_parser import ReverseParser


class CodegenService:
    """Stateless helpers for code generation and reverse-parsing."""

    def generate_from_yaml(self, yaml_content: str) -> str:
        """Parse YAML into IR, then generate Python."""
        data = yaml.safe_load(yaml_content)
        ir = WorkflowIR(**data)
        gen = CodeGenerator(ir)
        return gen.generate()

    def generate_from_ir(self, ir: WorkflowIR) -> str:
        """Generate Python directly from an IR object."""
        gen = CodeGenerator(ir)
        return gen.generate()

    def reverse_parse(self, python_source: str) -> WorkflowIR:
        """Convert Python source back into a WorkflowIR."""
        parser = ReverseParser()
        return parser.parse(python_source)

    def validate_yaml(self, yaml_content: str) -> list[str]:
        """Validate workflow YAML and return a list of errors (empty = valid)."""
        errors: list[str] = []
        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as exc:
            return [f"YAML parse error: {exc}"]

        if not isinstance(data, dict):
            return ["Top-level YAML must be a mapping"]

        try:
            WorkflowIR(**data)
        except Exception as exc:
            errors.append(str(exc))

        return errors


codegen_service = CodegenService()
