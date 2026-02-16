"""Workflow CRUD routes with code generation integration."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import get_session
from ..models import Workflow
from ..services.codegen_service import codegen_service

router = APIRouter(prefix="/api/workflows", tags=["workflows"])


# --- Schemas ---


class WorkflowCreate(BaseModel):
    project_id: str
    name: str
    yaml_content: str = ""


class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    yaml_content: Optional[str] = None


class WorkflowOut(BaseModel):
    id: str
    project_id: str
    name: str
    yaml_content: str
    generated_python: str
    version: int
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


def _to_out(w: Workflow) -> dict:
    return {
        "id": w.id,
        "project_id": w.project_id,
        "name": w.name,
        "yaml_content": w.yaml_content or "",
        "generated_python": w.generated_python or "",
        "version": w.version or 1,
        "created_at": w.created_at.isoformat() if w.created_at else "",
        "updated_at": w.updated_at.isoformat() if w.updated_at else "",
    }


# --- Routes ---


@router.get("", response_model=list[WorkflowOut])
async def list_workflows(
    project_id: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
):
    query = select(Workflow).order_by(Workflow.created_at.desc())
    if project_id:
        query = query.where(Workflow.project_id == project_id)
    result = await session.execute(query)
    return [_to_out(w) for w in result.scalars().all()]


@router.post("", response_model=WorkflowOut, status_code=201)
async def create_workflow(body: WorkflowCreate, session: AsyncSession = Depends(get_session)):
    generated = ""
    if body.yaml_content:
        try:
            generated = codegen_service.generate_from_yaml(body.yaml_content)
        except Exception:
            pass  # Store YAML even if codegen fails

    workflow = Workflow(
        project_id=body.project_id,
        name=body.name,
        yaml_content=body.yaml_content,
        generated_python=generated,
    )
    session.add(workflow)
    await session.commit()
    await session.refresh(workflow)
    return _to_out(workflow)


@router.get("/{workflow_id}", response_model=WorkflowOut)
async def get_workflow(workflow_id: str, session: AsyncSession = Depends(get_session)):
    workflow = await session.get(Workflow, workflow_id)
    if not workflow:
        raise HTTPException(404, "Workflow not found")
    return _to_out(workflow)


@router.put("/{workflow_id}", response_model=WorkflowOut)
async def update_workflow(
    workflow_id: str,
    body: WorkflowUpdate,
    session: AsyncSession = Depends(get_session),
):
    workflow = await session.get(Workflow, workflow_id)
    if not workflow:
        raise HTTPException(404, "Workflow not found")
    if body.name is not None:
        workflow.name = body.name
    if body.yaml_content is not None:
        workflow.yaml_content = body.yaml_content
        try:
            workflow.generated_python = codegen_service.generate_from_yaml(body.yaml_content)
            workflow.version = (workflow.version or 1) + 1
        except Exception:
            pass
    await session.commit()
    await session.refresh(workflow)
    return _to_out(workflow)


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow(workflow_id: str, session: AsyncSession = Depends(get_session)):
    workflow = await session.get(Workflow, workflow_id)
    if not workflow:
        raise HTTPException(404, "Workflow not found")
    await session.delete(workflow)
    await session.commit()


@router.post("/{workflow_id}/generate", response_model=WorkflowOut)
async def regenerate_workflow(workflow_id: str, session: AsyncSession = Depends(get_session)):
    """Regenerate Python code from stored YAML."""
    workflow = await session.get(Workflow, workflow_id)
    if not workflow:
        raise HTTPException(404, "Workflow not found")
    if not workflow.yaml_content:
        raise HTTPException(400, "No YAML content to generate from")
    try:
        workflow.generated_python = codegen_service.generate_from_yaml(workflow.yaml_content)
        workflow.version = (workflow.version or 1) + 1
    except Exception as exc:
        raise HTTPException(422, f"Code generation failed: {exc}")
    await session.commit()
    await session.refresh(workflow)
    return _to_out(workflow)


class ReverseParsebody(BaseModel):
    python_source: str


@router.post("/{workflow_id}/reverse-parse", response_model=WorkflowOut)
async def reverse_parse_workflow(
    workflow_id: str,
    body: ReverseParsebody,
    session: AsyncSession = Depends(get_session),
):
    """Parse Python code back into YAML and update the workflow."""
    workflow = await session.get(Workflow, workflow_id)
    if not workflow:
        raise HTTPException(404, "Workflow not found")
    try:
        ir = codegen_service.reverse_parse(body.python_source)
        import yaml
        workflow.yaml_content = yaml.dump(ir.model_dump(), default_flow_style=False)
        workflow.generated_python = body.python_source
        workflow.version = (workflow.version or 1) + 1
    except Exception as exc:
        raise HTTPException(422, f"Reverse parse failed: {exc}")
    await session.commit()
    await session.refresh(workflow)
    return _to_out(workflow)
