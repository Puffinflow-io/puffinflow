"""Eval suite YAML parser and data models."""
from __future__ import annotations
from typing import Any
from pathlib import Path
from pydantic import BaseModel, Field
import yaml


class EvalCaseConfig(BaseModel):
    """A single eval test case."""
    name: str
    input: dict[str, Any] = Field(default_factory=dict)  # Input variables for the agent
    expected: dict[str, Any] = Field(default_factory=dict)  # Expected output
    tags: list[str] = Field(default_factory=list)
    timeout: float | None = None


class ScoringConfig(BaseModel):
    """Scoring configuration for a suite."""
    default_scorer: str = "contains"
    threshold: float = 0.7
    scorers: dict[str, dict[str, Any]] = Field(default_factory=dict)  # named scorer configs


class EvalSuiteConfig(BaseModel):
    """Full eval suite configuration."""
    name: str
    workflow: str  # Path to workflow YAML or Python
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    cases: list[EvalCaseConfig] = Field(default_factory=list)
    parallel: int = 1
    description: str = ""


def parse_suite(path: str) -> EvalSuiteConfig:
    """Parse an eval suite YAML file."""
    content = Path(path).read_text()
    data = yaml.safe_load(content)
    return EvalSuiteConfig(**data)
