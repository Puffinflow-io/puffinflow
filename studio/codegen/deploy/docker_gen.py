"""Docker deployment generator."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .base import DeployGenerator

if TYPE_CHECKING:
    from ..ir import WorkflowIR


class DockerGenerator(DeployGenerator):
    """Generate Dockerfile, FastAPI serve wrapper, and requirements."""

    def generate(self, ir: WorkflowIR, python_code: str, output_dir: str) -> list[str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        generated: list[str] = []

        class_name = ir.agent.class_name
        agent_name = ir.agent.name

        # Write the agent module
        agent_path = out / "agent.py"
        agent_path.write_text(python_code, encoding="utf-8")
        generated.append(str(agent_path))

        # Generate FastAPI serve wrapper
        serve_code = f'''"""FastAPI server for {agent_name}."""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any

from agent import {class_name}

app = FastAPI(title="{agent_name}")


class RunRequest(BaseModel):
    input_data: dict[str, Any] = {{}}


class RunResponse(BaseModel):
    outputs: dict[str, Any] = {{}}
    status: str = "success"


@app.post("/run", response_model=RunResponse)
async def run_agent(request: RunRequest) -> RunResponse:
    """Run the agent with the given input data."""
    agent = {class_name}()
    result = await agent.run(initial_context={{"variables": request.input_data}})
    return RunResponse(outputs=result.outputs)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {{"status": "ok", "agent": "{agent_name}"}}
'''

        serve_path = out / "serve.py"
        serve_path.write_text(serve_code, encoding="utf-8")
        generated.append(str(serve_path))

        # Generate Dockerfile
        dockerfile = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
"""

        dockerfile_path = out / "Dockerfile"
        dockerfile_path.write_text(dockerfile, encoding="utf-8")
        generated.append(str(dockerfile_path))

        # Generate requirements.txt
        requirements = "puffinflow\nfastapi\nuvicorn[standard]\n"

        req_path = out / "requirements.txt"
        req_path.write_text(requirements, encoding="utf-8")
        generated.append(str(req_path))

        return generated
