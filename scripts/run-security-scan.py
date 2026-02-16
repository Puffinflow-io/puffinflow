#!/usr/bin/env python3
"""
Local security scanning script for puffinflow project.
This script runs the same security checks as the CI/CD pipeline locally.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_command(
    cmd: list[str], capture_output: bool = True
) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=capture_output, text=True, check=False
        )
        return result
    except FileNotFoundError:
        print(f"Error: Command '{cmd[0]}' not found. Please install it first.")
        sys.exit(1)


def check_git_history() -> bool:
    """Check if git history is sufficient for TruffleHog."""
    result = run_command(["git", "rev-list", "--count", "HEAD"])
    if result.returncode != 0:
        print("Error: Not a git repository or git not available")
        return False

    commit_count = int(result.stdout.strip())
    print(f"Git commit count: {commit_count}")

    if commit_count < 2:
        print("Warning: Less than 2 commits. TruffleHog may not work properly.")
        return False

    return True


def run_trufflehog() -> bool:
    """Run TruffleHog security scan."""
    print("\n=== Running TruffleHog Security Scan ===")

    if not check_git_history():
        print("Skipping TruffleHog due to insufficient git history")
        return True

    # Check if trufflehog is installed
    result = run_command(["trufflehog", "--version"])
    if result.returncode != 0:
        print("TruffleHog not found. Install it with:")
        print(
            "  curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh | sh -s -- -b /usr/local/bin"
        )
        return False

    # Run TruffleHog scan
    cmd = [
        "trufflehog",
        "filesystem",
        ".",
        "--config",
        ".trufflehog.yml",
        "--only-verified",
        "--json",
    ]

    result = run_command(cmd, capture_output=False)
    return result.returncode == 0


def run_bandit() -> bool:
    """Run Bandit security linter."""
    print("\n=== Running Bandit Security Linter ===")

    cmd = ["bandit", "-r", "src/", "-f", "json", "-o", "bandit-report.json"]
    result = run_command(cmd)

    if result.returncode == 0:
        print("✅ Bandit scan completed successfully")
        return True
    else:
        print("❌ Bandit found security issues")
        # Also run with text output for readability
        run_command(["bandit", "-r", "src/"], capture_output=False)
        return False


def run_safety() -> bool:
    """Run Safety check for known vulnerabilities."""
    print("\n=== Running Safety Vulnerability Check ===")

    cmd = ["safety", "check", "--json", "--output", "safety-report.json"]
    result = run_command(cmd)

    if result.returncode == 0:
        print("✅ Safety check completed successfully")
        return True
    else:
        print("❌ Safety found vulnerabilities")
        # Also run with text output for readability
        run_command(["safety", "check"], capture_output=False)
        return False


def run_semgrep() -> bool:
    """Run Semgrep security analysis."""
    print("\n=== Running Semgrep Security Analysis ===")

    cmd = ["semgrep", "--config=auto", "src/", "--json", "--output=semgrep-report.json"]
    result = run_command(cmd)

    if result.returncode == 0:
        print("✅ Semgrep scan completed successfully")
        return True
    else:
        print("❌ Semgrep found security issues")
        # Also run with text output for readability
        run_command(["semgrep", "--config=auto", "src/"], capture_output=False)
        return False


def check_coverage() -> Optional[float]:
    """Check test coverage."""
    print("\n=== Checking Test Coverage ===")

    # Run tests with coverage
    cmd = [
        "pytest",
        "tests/unit/",
        "--cov=src/puffinflow",
        "--cov-report=xml",
        "--cov-report=term",
    ]

    result = run_command(cmd, capture_output=False)

    if result.returncode != 0:
        print("❌ Tests failed")
        return None

    # Extract coverage from coverage.xml if it exists
    coverage_file = Path("coverage.xml")
    if coverage_file.exists():
        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(coverage_file)
            root = tree.getroot()
            coverage = float(root.attrib["line-rate"]) * 100
            print(f"Coverage: {coverage:.2f}%")
            return coverage
        except Exception as e:
            print(f"Error parsing coverage: {e}")

    return None


def main():
    """Main function to run all security checks."""
    print("🔍 Running local security and quality checks for puffinflow")
    print("=" * 60)

    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    results = {}

    # Run security scans
    results["trufflehog"] = run_trufflehog()
    results["bandit"] = run_bandit()
    results["safety"] = run_safety()
    results["semgrep"] = run_semgrep()

    # Check coverage
    coverage = check_coverage()
    results["coverage"] = coverage

    # Summary
    print("\n" + "=" * 60)
    print("📊 SECURITY SCAN SUMMARY")
    print("=" * 60)

    all_passed = True

    for check, passed in results.items():
        if check == "coverage":
            if passed is not None:
                threshold = 85.0
                meets_threshold = passed >= threshold
                status = "✅" if meets_threshold else "❌"
                print(f"{status} Coverage: {passed:.2f}% (threshold: {threshold}%)")
                if not meets_threshold:
                    all_passed = False
            else:
                print("❌ Coverage: Unable to determine")
                all_passed = False
        else:
            status = "✅" if passed else "❌"
            print(f"{status} {check.title()}: {'Passed' if passed else 'Failed'}")
            if not passed:
                all_passed = False

    print("=" * 60)

    if all_passed:
        print("🎉 All security checks passed! Ready for PyPI publication.")
        sys.exit(0)
    else:
        print("⚠️  Some checks failed. Please review and fix issues before publishing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
