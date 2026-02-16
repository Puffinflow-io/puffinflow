"""PuffinFlow Studio CLI — powered by Typer."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="puffinflow",
    help="PuffinFlow Studio — build, test, and deploy AI agent workflows.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------
@app.command()
def init(
    name: str = typer.Argument(..., help="Project name"),
    template: str = typer.Option(
        "basic", "--template", "-t", help="Template: basic, research, pipeline"
    ),
    output_dir: str = typer.Option(
        ".", "--output", "-o", help="Parent directory for the project"
    ),
):
    """Scaffold a new PuffinFlow project."""
    from .scaffold import scaffold_project

    try:
        root = scaffold_project(name, template=template, output_dir=output_dir)
        typer.echo(f"Created project at {root}")
        typer.echo("  workflows/main.yaml  — starter workflow")
        typer.echo("  evals/main_eval.yaml — starter eval suite")
        typer.echo()
        typer.echo("Next steps:")
        typer.echo(f"  cd {root}")
        typer.echo("  puffinflow codegen workflows/main.yaml")
        typer.echo("  puffinflow dev")
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# codegen
# ---------------------------------------------------------------------------
@app.command()
def codegen(
    file: Path = typer.Argument(..., help="Path to workflow YAML file"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output Python file path"
    ),
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Watch for changes and auto-regenerate"
    ),
):
    """Generate Python code from a workflow YAML file."""
    import yaml as yaml_lib

    from studio.codegen.generator import CodeGenerator
    from studio.codegen.ir import WorkflowIR

    def _generate_once(src: Path) -> str:
        content = src.read_text()
        data = yaml_lib.safe_load(content)
        ir = WorkflowIR(**data)
        gen = CodeGenerator(ir)
        return gen.generate()

    def _run(src: Path):
        python_code = _generate_once(src)
        if output:
            output.write_text(python_code)
            typer.echo(f"Generated {output}")
        else:
            dest = src.with_suffix(".py")
            dest.write_text(python_code)
            typer.echo(f"Generated {dest}")

    try:
        _run(file)
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    if watch:
        import time

        typer.echo("Watching for changes… (Ctrl+C to stop)")
        last_mtime = file.stat().st_mtime
        try:
            while True:
                time.sleep(1)
                mtime = file.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime
                    try:
                        _run(file)
                    except Exception as exc:
                        typer.echo(f"Regeneration error: {exc}", err=True)
        except KeyboardInterrupt:
            typer.echo("\nStopped.")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------
@app.command()
def validate(
    file: Path = typer.Argument(..., help="Path to workflow YAML file"),
):
    """Validate a workflow YAML file against the IR schema."""
    import yaml as yaml_lib

    from studio.codegen.ir import WorkflowIR

    try:
        content = file.read_text()
        data = yaml_lib.safe_load(content)
        WorkflowIR(**data)
        typer.echo(f"Valid: {file}")
    except Exception as exc:
        typer.echo(f"Invalid: {exc}", err=True)
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------
@app.command(name="test")
def run_tests(
    suite: Path = typer.Argument(..., help="Path to eval suite YAML"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    parallel: int = typer.Option(
        1, "--parallel", "-p", help="Number of parallel eval cases"
    ),
):
    """Run an evaluation suite."""
    from studio.eval.engine import EvalEngine
    from studio.eval.scorers import get_scorer
    from studio.eval.suite import parse_suite

    suite_config = parse_suite(str(suite))

    # Resolve workflow path relative to suite file
    workflow_path = suite.parent / suite_config.workflow
    scorers = {"default": get_scorer(suite_config.scoring.default_scorer)}

    engine = EvalEngine(str(workflow_path), scorers=scorers)
    result = asyncio.run(engine.run_suite(suite_config, parallel=parallel))

    # Print results
    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"Suite: {result.suite_name}")
    typer.echo(
        f"Total: {result.total_cases}  Passed: {result.passed_cases}  "
        f"Failed: {result.failed_cases}  Errors: {result.error_cases}"
    )
    typer.echo(
        f"Avg Score: {result.avg_score:.2f}  "
        f"Avg Latency: {result.avg_latency_ms:.0f}ms  "
        f"Pass Rate: {result.pass_rate:.0%}"
    )
    typer.echo(f"{'=' * 60}")

    if verbose:
        for cr in result.results:
            status = "PASS" if cr.passed else ("ERROR" if cr.error else "FAIL")
            typer.echo(f"\n  [{status}] {cr.case_name}")
            if cr.error:
                typer.echo(f"    Error: {cr.error}")
            else:
                typer.echo(f"    Scores: {cr.scores}")
                typer.echo(f"    Latency: {cr.latency_ms:.0f}ms")

    if result.failed_cases > 0 or result.error_cases > 0:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------
@app.command()
def deploy(
    workflow: Path = typer.Argument(..., help="Path to workflow YAML file"),
    target: str = typer.Option(
        "docker", "--target", "-t", help="Deploy target: modal, docker"
    ),
    output_dir: Path | None = typer.Option(
        None, "--output", "-o", help="Output directory for deploy files"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Generate files only, don't deploy"
    ),
):
    """Generate deployment artefacts for a workflow."""
    import yaml as yaml_lib

    from studio.codegen.deploy.docker_gen import DockerGenerator
    from studio.codegen.deploy.modal_gen import ModalGenerator
    from studio.codegen.generator import CodeGenerator
    from studio.codegen.ir import WorkflowIR

    generators = {"modal": ModalGenerator, "docker": DockerGenerator}
    gen_cls = generators.get(target)
    if gen_cls is None:
        typer.echo(
            f"Unknown target: {target}. Choose from: {list(generators)}", err=True
        )
        raise typer.Exit(1)

    try:
        content = workflow.read_text()
        data = yaml_lib.safe_load(content)
        ir = WorkflowIR(**data)
        python_code = CodeGenerator(ir).generate()

        dest = str(output_dir or workflow.parent / "deploy")
        Path(dest).mkdir(parents=True, exist_ok=True)

        generator = gen_cls()
        files = generator.generate(ir, python_code, dest)

        typer.echo(f"Generated {len(files)} deploy file(s):")
        for f in files:
            typer.echo(f"  {f}")

        if dry_run:
            typer.echo("\n(dry-run — files generated but not deployed)")
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


# ---------------------------------------------------------------------------
# dev
# ---------------------------------------------------------------------------
@app.command()
def dev(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
):
    """Start the Studio development servers (API + frontend)."""
    import subprocess

    typer.echo("Starting PuffinFlow Studio dev servers…")
    typer.echo(f"  API:      http://{host}:{port}")
    typer.echo("  Frontend: http://localhost:3000")
    typer.echo()

    # Start API server
    try:
        api_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "studio.api.main:app",
                "--host",
                host,
                "--port",
                str(port),
                "--reload",
            ],
        )
    except FileNotFoundError as exc:
        typer.echo(
            "Error: uvicorn not found. Install with: pip install uvicorn", err=True
        )
        raise typer.Exit(1) from exc

    # Start Next.js dev server
    web_dir = Path(__file__).resolve().parent.parent / "web"
    next_proc = None
    if (web_dir / "package.json").exists():
        try:
            next_proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(web_dir),
            )
        except FileNotFoundError:
            typer.echo(
                "Warning: npm not found — frontend server not started.", err=True
            )

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        typer.echo("\nShutting down…")
        api_proc.terminate()
        if next_proc:
            next_proc.terminate()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app()


if __name__ == "__main__":
    main()
