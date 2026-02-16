#!/usr/bin/env python3
"""
Script to test TruffleHog configuration locally.
This helps debug TruffleHog issues before pushing to GitHub Actions.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        return result
    except subprocess.TimeoutExpired:
        print("❌ Command timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return None


def check_docker():
    """Check if Docker is available."""
    result = run_command("docker --version")
    if result and result.returncode == 0:
        print("✅ Docker is available")
        return True
    else:
        print("❌ Docker is not available or not running")
        return False


def check_git_repo():
    """Check if we're in a git repository."""
    result = run_command("git status")
    if result and result.returncode == 0:
        print("✅ Git repository detected")
        return True
    else:
        print("❌ Not in a git repository")
        return False


def check_trufflehog_config():
    """Check if TruffleHog config exists."""
    config_path = Path(".trufflehog.yml")
    if config_path.exists():
        print("✅ TruffleHog configuration found")
        return True
    else:
        print("❌ TruffleHog configuration not found")
        return False


def run_trufflehog_full_scan():
    """Run TruffleHog on the entire repository."""
    print("\n🔍 Running TruffleHog full repository scan...")

    cmd = """docker run --rm -v .:/tmp -w /tmp \
ghcr.io/trufflesecurity/trufflehog:latest \
git file:///tmp/ \
--only-verified \
--json"""

    result = run_command(cmd)
    if result:
        if result.returncode == 0:
            print("✅ TruffleHog scan completed successfully")
            if result.stdout.strip():
                print("⚠️  Secrets found:")
                print(result.stdout)
            else:
                print("✅ No secrets found")
        else:
            print("❌ TruffleHog scan failed")
            print(f"Exit code: {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")

    return result


def run_trufflehog_commit_range():
    """Run TruffleHog on a commit range (last 5 commits)."""
    print("\n🔍 Running TruffleHog on last 5 commits...")

    # Get the commit hash from 5 commits ago
    result = run_command("git rev-parse HEAD~5")
    if not result or result.returncode != 0:
        print("❌ Could not get commit range")
        return None

    base_commit = result.stdout.strip()

    cmd = f"""docker run --rm -v .:/tmp -w /tmp \
ghcr.io/trufflesecurity/trufflehog:latest \
git file:///tmp/ \
--since-commit {base_commit} \
--only-verified \
--json"""

    result = run_command(cmd)
    if result:
        if result.returncode == 0:
            print("✅ TruffleHog commit range scan completed successfully")
            if result.stdout.strip():
                print("⚠️  Secrets found:")
                print(result.stdout)
            else:
                print("✅ No secrets found")
        else:
            print("❌ TruffleHog commit range scan failed")
            print(f"Exit code: {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")

    return result


def main():
    """Main function."""
    print("🔧 TruffleHog Local Test Script")
    print("=" * 40)

    # Check prerequisites
    if not check_docker():
        print("\n💡 Install Docker to run TruffleHog locally")
        sys.exit(1)

    if not check_git_repo():
        print("\n💡 Run this script from the root of a git repository")
        sys.exit(1)

    if not check_trufflehog_config():
        print("\n💡 Create a .trufflehog.yml configuration file")
        sys.exit(1)

    print("\n" + "=" * 40)

    # Run scans
    full_scan_result = run_trufflehog_full_scan()
    commit_range_result = run_trufflehog_commit_range()

    print("\n" + "=" * 40)
    print("📊 Summary:")

    if full_scan_result and full_scan_result.returncode == 0:
        print("✅ Full repository scan: PASSED")
    else:
        print("❌ Full repository scan: FAILED")

    if commit_range_result and commit_range_result.returncode == 0:
        print("✅ Commit range scan: PASSED")
    else:
        print("❌ Commit range scan: FAILED")

    # Exit with error if any scan failed
    if (full_scan_result and full_scan_result.returncode != 0) or (
        commit_range_result and commit_range_result.returncode != 0
    ):
        sys.exit(1)

    print("\n🎉 All TruffleHog tests passed!")


if __name__ == "__main__":
    main()
