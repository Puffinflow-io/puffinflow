"""Eval engine for running evaluation suites against workflows."""
from __future__ import annotations
import asyncio
import importlib.util
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .scorers import Scorer, get_scorer

if TYPE_CHECKING:
    from .suite import EvalCaseConfig, EvalSuiteConfig


@dataclass
class EvalCaseResult:
    """Result of running a single eval case."""
    case_name: str
    scores: dict[str, float] = field(default_factory=dict)
    actual_output: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    passed: bool = False
    error: str | None = None


@dataclass
class EvalRunResult:
    """Result of running a full eval suite."""
    suite_name: str
    results: list[EvalCaseResult] = field(default_factory=list)
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    error_cases: int = 0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    total_duration_ms: float = 0.0

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / self.total_cases if self.total_cases > 0 else 0.0


class EvalEngine:
    """Engine for running evaluation suites."""

    def __init__(self, workflow_path: str, scorers: dict[str, Scorer] | None = None):
        self.workflow_path = workflow_path
        self.scorers = scorers or {}

    async def run_suite(
        self,
        suite: EvalSuiteConfig,
        parallel: int = 1,
    ) -> EvalRunResult:
        """Run all cases in the suite with optional parallelism."""
        start_time = time.time()

        # Set up default scorer if not provided
        if not self.scorers:
            self.scorers["default"] = get_scorer(suite.scoring.default_scorer)

        # Run cases with semaphore-controlled parallelism
        semaphore = asyncio.Semaphore(parallel)

        async def run_with_semaphore(case: EvalCaseConfig) -> EvalCaseResult:
            async with semaphore:
                return await self._run_case(case, suite.scoring.threshold)

        tasks = [run_with_semaphore(case) for case in suite.cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        case_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                case_results.append(EvalCaseResult(
                    case_name=suite.cases[i].name,
                    error=str(result),
                ))
            else:
                case_results.append(result)

        # Compute summary
        total = len(case_results)
        passed = sum(1 for r in case_results if r.passed)
        errors = sum(1 for r in case_results if r.error is not None)
        failed = total - passed - errors
        scores = [s for r in case_results for s in r.scores.values() if r.error is None]
        latencies = [r.latency_ms for r in case_results if r.error is None]

        return EvalRunResult(
            suite_name=suite.name,
            results=case_results,
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            error_cases=errors,
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
            total_duration_ms=(time.time() - start_time) * 1000,
        )

    async def _run_case(
        self,
        case: EvalCaseConfig,
        threshold: float = 0.7,
    ) -> EvalCaseResult:
        """Run a single eval case."""
        start_time = time.time()

        try:
            # Dynamic import of workflow module
            agent_class = self._load_agent()

            # Create agent instance and run
            agent = agent_class(agent_class.__name__.lower())
            result = await agent.run(initial_context={"variables": case.input})

            actual_output = result.outputs if hasattr(result, 'outputs') else {}

            # Score with all configured scorers
            scores = {}
            for scorer_name, scorer in self.scorers.items():
                scores[scorer_name] = scorer.score(case.expected, actual_output)

            # Determine pass/fail based on average score vs threshold
            avg_score = sum(scores.values()) / len(scores) if scores else 0.0

            latency_ms = (time.time() - start_time) * 1000

            return EvalCaseResult(
                case_name=case.name,
                scores=scores,
                actual_output=actual_output,
                latency_ms=latency_ms,
                passed=avg_score >= threshold,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return EvalCaseResult(
                case_name=case.name,
                latency_ms=latency_ms,
                error=str(e),
            )

    def _load_agent(self):
        """Dynamically load agent class from workflow path."""
        path = Path(self.workflow_path)

        if path.suffix == ".py":
            # Direct Python file
            spec = importlib.util.spec_from_file_location("workflow_module", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["workflow_module"] = module
            spec.loader.exec_module(module)

            # Find Agent subclass
            from puffinflow import Agent
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and issubclass(attr, Agent) and attr is not Agent):
                    return attr

            raise ValueError(f"No Agent subclass found in {path}")

        elif path.suffix in (".yaml", ".yml"):
            # YAML workflow - generate Python first then load
            from studio.codegen.generator import CodeGenerator
            from studio.codegen.ir import WorkflowIR
            yaml_content = path.read_text()
            import yaml as yaml_lib
            data = yaml_lib.safe_load(yaml_content)
            ir = WorkflowIR(**data)
            gen = CodeGenerator(ir)
            python_code = gen.generate()

            # Execute generated code
            namespace = {}
            exec(python_code, namespace)

            from puffinflow import Agent
            for value in namespace.values():
                if isinstance(value, type) and issubclass(value, Agent) and value is not Agent:
                    return value

            raise ValueError(f"No Agent subclass found in generated code from {path}")

        else:
            raise ValueError(f"Unsupported workflow file type: {path.suffix}")
