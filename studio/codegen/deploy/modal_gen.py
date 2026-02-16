"""Modal deployment generator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from .base import DeployGenerator

if TYPE_CHECKING:
    from ..ir import WorkflowIR


class ModalGenerator(DeployGenerator):
    """Generate Modal deployment script for a PuffinFlow agent."""

    def generate(
        self, ir: WorkflowIR, python_code: str, output_dir: str
    ) -> list[str]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        generated: list[str] = []

        # Write the agent module
        agent_path = out / "agent.py"
        agent_path.write_text(python_code, encoding="utf-8")
        generated.append(str(agent_path))

        # Generate Modal deploy script
        class_name = ir.agent.class_name
        agent_name = ir.agent.name

        deploy_code = f'''"""Modal deployment for {agent_name}."""

import modal

from agent import {class_name}

app = modal.App("{agent_name}")

image = modal.Image.debian_slim(python_version="3.11").pip_install("puffinflow")


@app.function(image=image)
async def run_agent(input_data: dict) -> dict:
    """Run the agent with the given input data."""
    agent = {class_name}()
    result = await agent.run(initial_context={{"variables": input_data}})
    return result.outputs


@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    result = run_agent.remote({{}})
    print(result)
'''

        deploy_path = out / "deploy_modal.py"
        deploy_path.write_text(deploy_code, encoding="utf-8")
        generated.append(str(deploy_path))

        return generated
