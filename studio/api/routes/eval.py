"""Eval suite management and execution routes."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..db import get_session
from ..models import EvalCase, EvalRun, EvalSuite, Workflow
from ..services.eval_service import eval_service

router = APIRouter(prefix="/api/eval", tags=["eval"])


# --- Schemas ---


class SuiteCreate(BaseModel):
    project_id: str
    name: str
    scoring_config: dict = Field(default_factory=dict)


class SuiteUpdate(BaseModel):
    name: Optional[str] = None
    scoring_config: Optional[dict] = None


class SuiteOut(BaseModel):
    id: str
    project_id: str
    name: str
    scoring_config: dict
    created_at: str
    updated_at: str


class CaseCreate(BaseModel):
    name: str
    input_data: dict = Field(default_factory=dict)
    expected_output: dict = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class CaseOut(BaseModel):
    id: str
    suite_id: str
    name: str
    input_data: dict
    expected_output: dict
    tags: list[str]


class RunRequest(BaseModel):
    workflow_id: str
    parallel: int = 1


class ResultOut(BaseModel):
    id: str
    case_id: str
    actual_output: dict
    scores: dict
    latency_ms: float
    passed: bool


class RunOut(BaseModel):
    id: str
    suite_id: str
    workflow_id: str
    status: str
    summary_scores: dict
    results: list[ResultOut] = Field(default_factory=list)
    created_at: str


# --- Helpers ---


def _suite_out(s: EvalSuite) -> dict:
    return {
        "id": s.id,
        "project_id": s.project_id,
        "name": s.name,
        "scoring_config": json.loads(s.scoring_config) if s.scoring_config else {},
        "created_at": s.created_at.isoformat() if s.created_at else "",
        "updated_at": s.updated_at.isoformat() if s.updated_at else "",
    }


def _case_out(c: EvalCase) -> dict:
    return {
        "id": c.id,
        "suite_id": c.suite_id,
        "name": c.name,
        "input_data": json.loads(c.input_data) if c.input_data else {},
        "expected_output": json.loads(c.expected_output) if c.expected_output else {},
        "tags": json.loads(c.tags_json) if c.tags_json else [],
    }


def _run_out(r: EvalRun, include_results: bool = False) -> dict:
    out = {
        "id": r.id,
        "suite_id": r.suite_id,
        "workflow_id": r.workflow_id,
        "status": r.status or "",
        "summary_scores": json.loads(r.summary_scores) if r.summary_scores else {},
        "created_at": r.created_at.isoformat() if r.created_at else "",
        "results": [],
    }
    if include_results and hasattr(r, "results") and r.results:
        out["results"] = [
            {
                "id": res.id,
                "case_id": res.case_id,
                "actual_output": json.loads(res.actual_output) if res.actual_output else {},
                "scores": json.loads(res.scores_json) if res.scores_json else {},
                "latency_ms": res.latency_ms or 0.0,
                "passed": bool(res.passed),
            }
            for res in r.results
        ]
    return out


# --- Suite Routes ---


@router.get("/suites", response_model=list[SuiteOut])
async def list_suites(
    project_id: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    query = select(EvalSuite).order_by(EvalSuite.created_at.desc())
    if project_id:
        query = query.where(EvalSuite.project_id == project_id)
    result = await session.execute(query)
    return [_suite_out(s) for s in result.scalars().all()]


@router.post("/suites", response_model=SuiteOut, status_code=201)
async def create_suite(body: SuiteCreate, session: AsyncSession = Depends(get_session)):
    suite = EvalSuite(
        project_id=body.project_id,
        name=body.name,
        scoring_config=json.dumps(body.scoring_config),
    )
    session.add(suite)
    await session.commit()
    await session.refresh(suite)
    return _suite_out(suite)


@router.get("/suites/{suite_id}", response_model=SuiteOut)
async def get_suite(suite_id: str, session: AsyncSession = Depends(get_session)):
    suite = await session.get(EvalSuite, suite_id)
    if not suite:
        raise HTTPException(404, "Suite not found")
    return _suite_out(suite)


@router.put("/suites/{suite_id}", response_model=SuiteOut)
async def update_suite(
    suite_id: str,
    body: SuiteUpdate,
    session: AsyncSession = Depends(get_session),
):
    suite = await session.get(EvalSuite, suite_id)
    if not suite:
        raise HTTPException(404, "Suite not found")
    if body.name is not None:
        suite.name = body.name
    if body.scoring_config is not None:
        suite.scoring_config = json.dumps(body.scoring_config)
    await session.commit()
    await session.refresh(suite)
    return _suite_out(suite)


@router.delete("/suites/{suite_id}", status_code=204)
async def delete_suite(suite_id: str, session: AsyncSession = Depends(get_session)):
    suite = await session.get(EvalSuite, suite_id)
    if not suite:
        raise HTTPException(404, "Suite not found")
    await session.delete(suite)
    await session.commit()


# --- Case Routes ---


@router.get("/suites/{suite_id}/cases", response_model=list[CaseOut])
async def list_cases(suite_id: str, session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(EvalCase).where(EvalCase.suite_id == suite_id)
    )
    return [_case_out(c) for c in result.scalars().all()]


@router.post("/suites/{suite_id}/cases", response_model=CaseOut, status_code=201)
async def create_case(
    suite_id: str,
    body: CaseCreate,
    session: AsyncSession = Depends(get_session),
):
    suite = await session.get(EvalSuite, suite_id)
    if not suite:
        raise HTTPException(404, "Suite not found")
    case = EvalCase(
        suite_id=suite_id,
        name=body.name,
        input_data=json.dumps(body.input_data),
        expected_output=json.dumps(body.expected_output),
        tags_json=json.dumps(body.tags),
    )
    session.add(case)
    await session.commit()
    await session.refresh(case)
    return _case_out(case)


@router.delete("/cases/{case_id}", status_code=204)
async def delete_case(case_id: str, session: AsyncSession = Depends(get_session)):
    case = await session.get(EvalCase, case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    await session.delete(case)
    await session.commit()


# --- Run Routes ---


@router.post("/suites/{suite_id}/run", response_model=RunOut, status_code=201)
async def run_suite(
    suite_id: str,
    body: RunRequest,
    session: AsyncSession = Depends(get_session),
):
    suite = await session.get(EvalSuite, suite_id)
    if not suite:
        raise HTTPException(404, "Suite not found")
    workflow = await session.get(Workflow, body.workflow_id)
    if not workflow:
        raise HTTPException(404, "Workflow not found")
    run = await eval_service.run_suite(session, suite, workflow, parallel=body.parallel)
    return _run_out(run)


@router.get("/runs/{run_id}", response_model=RunOut)
async def get_run(run_id: str, session: AsyncSession = Depends(get_session)):
    run = await eval_service.get_run_with_results(session, run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return _run_out(run, include_results=True)
