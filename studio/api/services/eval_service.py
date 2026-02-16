"""Eval service — runs evaluation suites and persists results."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import selectinload

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

from studio.eval.engine import EvalEngine
from studio.eval.scorers import get_scorer
from studio.eval.suite import EvalCaseConfig, EvalSuiteConfig, ScoringConfig
from ..models import EvalCase, EvalResult, EvalRun, EvalSuite, Workflow


class EvalService:
    """Orchestrates eval suite execution and result persistence."""

    async def run_suite(
        self,
        session: AsyncSession,
        suite: EvalSuite,
        workflow: Workflow,
        parallel: int = 1,
    ) -> EvalRun:
        """Execute every case in *suite* against *workflow* and store results."""
        # Create run record
        run = EvalRun(
            suite_id=suite.id,
            workflow_id=workflow.id,
            status="running",
        )
        session.add(run)
        await session.commit()
        await session.refresh(run)

        try:
            # Load cases from DB
            result = await session.execute(
                select(EvalCase).where(EvalCase.suite_id == suite.id)
            )
            db_cases = result.scalars().all()

            # Build eval config
            scoring_data = json.loads(suite.scoring_config) if suite.scoring_config else {}
            scoring = ScoringConfig(**scoring_data) if scoring_data else ScoringConfig()

            cases = [
                EvalCaseConfig(
                    name=c.name,
                    input=json.loads(c.input_data) if c.input_data else {},
                    expected=json.loads(c.expected_output) if c.expected_output else {},
                    tags=json.loads(c.tags_json) if c.tags_json else [],
                )
                for c in db_cases
            ]

            suite_config = EvalSuiteConfig(
                name=suite.name,
                workflow=workflow.generated_python or "",
                scoring=scoring,
                cases=cases,
                parallel=parallel,
            )

            # Set up scorers
            scorers = {"default": get_scorer(scoring.default_scorer)}

            # Run evaluation
            engine = EvalEngine(
                workflow_path=workflow.generated_python or "",
                scorers=scorers,
            )
            eval_result = await engine.run_suite(suite_config, parallel=parallel)

            # Persist results
            case_lookup = {c.name: c for c in db_cases}
            for cr in eval_result.results:
                db_case = case_lookup.get(cr.case_name)
                if db_case is None:
                    continue
                er = EvalResult(
                    run_id=run.id,
                    case_id=db_case.id,
                    actual_output=json.dumps(cr.actual_output),
                    scores_json=json.dumps(cr.scores),
                    latency_ms=cr.latency_ms,
                    passed=1 if cr.passed else 0,
                )
                session.add(er)

            run.status = "completed"
            run.summary_scores = json.dumps({
                "avg_score": eval_result.avg_score,
                "pass_rate": eval_result.pass_rate,
                "total": eval_result.total_cases,
                "passed": eval_result.passed_cases,
                "failed": eval_result.failed_cases,
                "errors": eval_result.error_cases,
            })
            await session.commit()
            await session.refresh(run)

        except Exception as exc:
            run.status = "failed"
            run.summary_scores = json.dumps({"error": str(exc)})
            await session.commit()
            await session.refresh(run)

        return run

    async def get_run_with_results(
        self,
        session: AsyncSession,
        run_id: str,
    ) -> EvalRun | None:
        result = await session.execute(
            select(EvalRun)
            .options(selectinload(EvalRun.results))
            .where(EvalRun.id == run_id)
        )
        return result.scalar_one_or_none()


eval_service = EvalService()
