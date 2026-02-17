"""Stateless code generation routes and WebSocket live preview."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..services.codegen_service import codegen_service

router = APIRouter(prefix="/api/codegen", tags=["codegen"])


# --- Schemas ---


class GenerateRequest(BaseModel):
    yaml_content: str


class GenerateResponse(BaseModel):
    python: str
    errors: list[str] = []


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[str] = []


# --- REST Routes ---


@router.post("/generate", response_model=GenerateResponse)
async def generate_code(body: GenerateRequest):
    """Stateless YAML → Python code generation."""
    errors = codegen_service.validate_yaml(body.yaml_content)
    if errors:
        return GenerateResponse(python="", errors=errors)
    try:
        python_code = codegen_service.generate_from_yaml(body.yaml_content)
        return GenerateResponse(python=python_code)
    except Exception as exc:
        return GenerateResponse(python="", errors=[str(exc)])


@router.post("/validate", response_model=ValidateResponse)
async def validate_yaml(body: GenerateRequest):
    """Validate workflow YAML against the IR schema."""
    errors = codegen_service.validate_yaml(body.yaml_content)
    return ValidateResponse(valid=len(errors) == 0, errors=errors)


# --- WebSocket Live Preview ---


@router.websocket("/ws/live-preview")
async def live_preview(ws: WebSocket):
    """Real-time code generation preview.

    Client sends ``{"yaml": "..."}`` messages.  Server responds with
    ``{"python": "..."}`` or ``{"error": "..."}`` after a 300 ms debounce.
    """
    await ws.accept()
    debounce_task: asyncio.Task | None = None

    async def _process(yaml_text: str):
        """Generate code and send result back."""
        await asyncio.sleep(0.3)  # 300 ms debounce
        try:
            python_code = codegen_service.generate_from_yaml(yaml_text)
            await ws.send_json({"python": python_code})
        except Exception as exc:
            await ws.send_json({"error": str(exc)})

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
                yaml_text = data.get("yaml", "")
            except (json.JSONDecodeError, AttributeError):
                yaml_text = raw

            # Cancel previous pending generation
            if debounce_task and not debounce_task.done():
                debounce_task.cancel()

            debounce_task = asyncio.create_task(_process(yaml_text))
    except WebSocketDisconnect:
        pass
    finally:
        if debounce_task and not debounce_task.done():
            debounce_task.cancel()
