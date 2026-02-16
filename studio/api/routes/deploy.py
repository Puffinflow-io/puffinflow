"""Deployment routes — trigger and check status."""
from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..db import get_session
from ..models import Workflow
from ..services.deploy_service import deploy_service

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/deploy", tags=["deploy"])


# --- Schemas ---


class DeployRequest(BaseModel):
    workflow_id: str
    target: str  # "modal" | "docker"
    dry_run: bool = False


class DeployOut(BaseModel):
    id: str
    workflow_id: str
    target: str
    status: str
    url: str
    logs: str
    created_at: str
    updated_at: str


def _to_out(d) -> dict:
    return {
        "id": d.id,
        "workflow_id": d.workflow_id,
        "target": d.target,
        "status": d.status or "",
        "url": d.url or "",
        "logs": d.logs or "",
        "created_at": d.created_at.isoformat() if d.created_at else "",
        "updated_at": d.updated_at.isoformat() if d.updated_at else "",
    }


# --- Routes ---


@router.post("", response_model=DeployOut, status_code=201)
async def trigger_deploy(
    body: DeployRequest, session: AsyncSession = Depends(get_session)
):
    workflow = await session.get(Workflow, body.workflow_id)
    if not workflow:
        raise HTTPException(404, "Workflow not found")
    if not workflow.yaml_content:
        raise HTTPException(400, "Workflow has no YAML content")
    try:
        deployment = await deploy_service.trigger_deploy(
            session=session,
            workflow=workflow,
            target=body.target,
            dry_run=body.dry_run,
        )
        return _to_out(deployment)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/{deployment_id}", response_model=DeployOut)
async def get_deployment(
    deployment_id: str, session: AsyncSession = Depends(get_session)
):
    deployment = await deploy_service.get_deployment(session, deployment_id)
    if not deployment:
        raise HTTPException(404, "Deployment not found")
    return _to_out(deployment)
