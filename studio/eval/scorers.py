"""Evaluation scorers for comparing expected vs actual outputs."""
from __future__ import annotations
import importlib
from abc import ABC, abstractmethod
from typing import Any, Optional


class Scorer(ABC):
    """Base scorer protocol."""

    @abstractmethod
    def score(self, expected: dict[str, Any], actual: dict[str, Any]) -> float:
        """Score actual output against expected. Returns 0.0-1.0."""
        ...


class ExactMatchScorer(Scorer):
    """1.0 if all expected keys match exactly, 0.0 otherwise."""

    def score(self, expected: dict[str, Any], actual: dict[str, Any]) -> float:
        if not expected:
            return 1.0
        matches = 0
        total = len(expected)
        for key, value in expected.items():
            if key in actual and actual[key] == value:
                matches += 1
        return matches / total if total > 0 else 1.0


class ContainsScorer(Scorer):
    """Percentage of expected substrings found in actual output values.

    Expected format:
        {"key_contains": ["substring1", "substring2"]}

    Checks if each substring is found anywhere in the actual output values
    (converted to string). Score = num_found / total_expected.
    """

    def score(self, expected: dict[str, Any], actual: dict[str, Any]) -> float:
        total = 0
        found = 0
        actual_text = " ".join(str(v) for v in actual.values())

        for key, value in expected.items():
            if key.endswith("_contains") and isinstance(value, list):
                for substring in value:
                    total += 1
                    if str(substring).lower() in actual_text.lower():
                        found += 1
            else:
                # Treat as exact match for non-contains keys
                total += 1
                actual_key = key
                if actual_key in actual and str(actual[actual_key]) == str(value):
                    found += 1

        return found / total if total > 0 else 1.0


class LLMJudgeScorer(Scorer):
    """Uses an LLM to judge output quality against criteria.

    Returns 0.0-1.0 score based on LLM evaluation.
    """

    def __init__(self, criteria: str = "", model: str = "gpt-4"):
        self.criteria = criteria
        self.model = model

    def score(self, expected: dict[str, Any], actual: dict[str, Any]) -> float:
        # Stub implementation - in production this would call an LLM
        # For now, return 0.5 as a neutral score
        # TODO: Integrate with actual LLM provider
        return 0.5


class CustomScorer(Scorer):
    """Dynamically imports and calls a user-provided scoring function.

    The function must have signature: (expected: dict, actual: dict) -> float
    """

    def __init__(self, module_path: str):
        self.module_path = module_path
        self._func = self._load_function()

    def _load_function(self):
        parts = self.module_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid module path: {self.module_path}. Expected 'module.function'")
        module_name, func_name = parts
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        return func

    def score(self, expected: dict[str, Any], actual: dict[str, Any]) -> float:
        result = self._func(expected, actual)
        return float(result)


def get_scorer(name: str, **kwargs) -> Scorer:
    """Factory function to get a scorer by name."""
    scorers = {
        "exact_match": ExactMatchScorer,
        "contains": ContainsScorer,
        "llm_judge": LLMJudgeScorer,
        "custom": CustomScorer,
    }
    scorer_class = scorers.get(name)
    if scorer_class is None:
        raise ValueError(f"Unknown scorer: {name}. Available: {list(scorers.keys())}")
    return scorer_class(**kwargs)
