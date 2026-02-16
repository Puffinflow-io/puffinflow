"""Deploy service — orchestrates deployment generation and tracking."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import yaml
from sqlalchemy.ext.asyncio import AsyncSession

from studio.codegen.deploy.docker_gen import DockerGenerator
from studio.codegen.deploy.modal_gen import ModalGenerator
from studio.codegen.generator import CodeGenerator
from studio.codegen.ir import WorkflowIR
from ..models import Deployment, Workflow


class DeployService:
    """Manages deployment generation and status tracking."""

    _generators = {
        "modal": ModalGenerator,
        "docker": DockerGenerator,
    }

    async def trigger_deploy(
        self,
        session: AsyncSession,
        workflow: Workflow,
        target: str,
        dry_run: bool = False,
    ) -> Deployment:
        """Generate deploy artefacts and create a Deployment record."""
        generator_cls = self._generators.get(target)
        if generator_cls is None:
            raise ValueError(f"Unknown deploy target: {target}. Available: {list(self._generators)}")

        # Parse workflow YAML → IR → Python
        data = yaml.safe_load(workflow.yaml_content)
        ir = WorkflowIR(**data)
        python_code = CodeGenerator(ir).generate()

        # Generate deploy files into a temp directory
        output_dir = tempfile.mkdtemp(prefix="puffinflow_deploy_")
        generator = generator_cls()
        generated_files = generator.generate(ir, python_code, output_dir)

        logs = f"Generated {len(generated_files)} file(s) in {output_dir}\n"
        for f in generated_files:
            logs += f"  - {f}\n"

        deployment = Deployment(
            workflow_id=workflow.id,
            target=target,
            status="generated" if dry_run else "deploying",
            logs=logs,
        )
        session.add(deployment)
        await session.commit()
        await session.refresh(deployment)

        if not dry_run:
            # In a real implementation this would kick off an async deploy job.
            deployment.status = "deployed"
            deployment.url = f"https://{ir.agent.name}.puffinflow.app"
            await session.commit()
            await session.refresh(deployment)

        return deployment

    async def get_deployment(
        self,
        session: AsyncSession,
        deployment_id: str,
    ) -> Optional[Deployment]:
        return await session.get(Deployment, deployment_id)


deploy_service = DeployService()
