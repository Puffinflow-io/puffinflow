#!/usr/bin/env python3
"""
Test runner script for PuffinFlow project.

This script provides convenient commands to run different test suites:
- Unit tests
- Integration tests
- End-to-end tests
- All tests
- Coverage reports
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run PuffinFlow tests")
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "e2e", "all", "coverage"],
        help="Type of tests to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--fail-fast", "-x", action="store_true", help="Stop on first failure"
    )
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel workers")

    args = parser.parse_args()

    # Base pytest command
    base_cmd = ["python3", "-m", "pytest"]

    # Add common options
    if args.verbose:
        base_cmd.append("-v")
    if args.fail_fast:
        base_cmd.append("-x")
    if args.parallel:
        base_cmd.extend(["-n", str(args.parallel)])

    # Determine test paths and markers
    test_commands = []

    if args.test_type == "unit":
        cmd = [*base_cmd, "tests/unit/", "-m", "unit"]
        test_commands.append((cmd, "Unit Tests"))

    elif args.test_type == "integration":
        cmd = [*base_cmd, "tests/integration/", "-m", "integration"]
        test_commands.append((cmd, "Integration Tests"))

    elif args.test_type == "e2e":
        cmd = [*base_cmd, "tests/e2e/", "-m", "e2e"]
        test_commands.append((cmd, "End-to-End Tests"))

    elif args.test_type == "all":
        # Run all test types in sequence
        unit_cmd = [*base_cmd, "tests/unit/", "-m", "unit"]
        integration_cmd = [*base_cmd, "tests/integration/", "-m", "integration"]
        e2e_cmd = [*base_cmd, "tests/e2e/", "-m", "e2e"]

        test_commands.extend(
            [
                (unit_cmd, "Unit Tests"),
                (integration_cmd, "Integration Tests"),
                (e2e_cmd, "End-to-End Tests"),
            ]
        )

    elif args.test_type == "coverage":
        # Run tests with coverage
        coverage_cmd = [
            *base_cmd,
            "tests/",
            "--cov=src/puffinflow",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=80",
        ]
        test_commands.append((coverage_cmd, "Coverage Tests"))

    # Run the commands
    all_passed = True
    for cmd, description in test_commands:
        success = run_command(cmd, description)
        if not success:
            all_passed = False
            if args.fail_fast:
                break

    # Summary
    print(f"\n{'='*60}")
    if all_passed:
        print("🎉 All tests passed successfully!")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Add src to Python path for imports
    src_path = script_dir / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))

    main()
