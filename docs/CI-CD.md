# CI/CD Pipeline Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline for the PuffinFlow project.

## Overview

The CI/CD pipeline consists of several automated workflows that ensure code quality, security, and reliable deployments:

1. **Security Scanning** - TruffleHog secret detection
2. **Code Quality** - Linting and formatting checks
3. **Testing** - Unit tests with coverage reporting
4. **Building** - Package building and validation
5. **Deployment** - Automated PyPI publishing

## Workflows

### Security Workflow (`.github/workflows/security.yml`)

Runs TruffleHog to scan for secrets and sensitive information in the codebase.

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main`
- Daily scheduled runs at 2 AM UTC
- Manual workflow dispatch

**Key Features:**
- Scans entire repository history
- Uses custom configuration (`.trufflehog.yml`)
- Only reports verified secrets to reduce false positives
- Uploads scan results as artifacts

### Main CI/CD Workflow (`.github/workflows/ci.yml`)

Comprehensive pipeline that runs linting, testing, security scanning, building, and deployment.

**Jobs:**

1. **Lint and Format Check**
   - Runs pre-commit hooks
   - Ensures code style consistency

2. **Test**
   - Runs on Python 3.9, 3.10, 3.11, and 3.12
   - Executes unit tests with coverage reporting
   - Requires minimum 85% code coverage
   - Uploads coverage reports to Codecov

3. **Security Scan**
   - Runs TruffleHog secret detection
   - Uploads scan results as artifacts

4. **Build**
   - Builds Python package
   - Validates package integrity
   - Uploads build artifacts

5. **Deploy**
   - Deploys to PyPI on tagged releases
   - Requires manual approval (production environment)

## TruffleHog Configuration

### Issue Resolution

The original TruffleHog error occurred because the BASE and HEAD commits were the same, meaning there was nothing to scan between commits. This has been resolved by:

1. **Proper Commit Range Handling**: The workflow now correctly handles different event types:
   - **Pull Requests**: Scans between base and head of the PR
   - **Push Events**: Scans between the previous commit and current commit
   - **Scheduled/Manual Runs**: Scans the entire repository

2. **Fetch Depth**: Uses `fetch-depth: 0` to ensure full git history is available

3. **Conditional Logic**: Uses GitHub Actions expressions to set appropriate base/head commits based on event type

### Configuration File (`.trufflehog.yml`)

Key settings:
- **Only Verified Secrets**: Reduces false positives by only reporting verified secrets
- **Custom Detectors**: Focuses on common cloud providers, databases, and API services
- **Exclusions**: Excludes test files, build artifacts, and common false positive patterns
- **Performance**: Optimized for CI/CD with appropriate timeouts and concurrency settings

## Troubleshooting

### TruffleHog "BASE and HEAD are the same" Error

This error occurs when TruffleHog tries to scan between identical commits. The updated workflow handles this by:

1. Checking the event type (`push`, `pull_request`, `schedule`, `workflow_dispatch`)
2. Setting appropriate base and head commits for each event type
3. Falling back to full repository scan for scheduled and manual runs

### Common Issues and Solutions

1. **Coverage Below Threshold**
   - Add more unit tests
   - Remove unused code
   - Check coverage report in artifacts

2. **Security Scan Failures**
   - Review TruffleHog results in artifacts
   - Add legitimate secrets to allowlist in `.trufflehog.yml`
   - Use environment variables for sensitive data

3. **Build Failures**
   - Check package metadata in `pyproject.toml`
   - Ensure all dependencies are properly specified
   - Validate package structure

## Environment Variables

Required secrets in GitHub repository settings:

- `PYPI_API_TOKEN`: PyPI API token for package publishing
- `CODECOV_TOKEN`: Codecov token for coverage reporting (optional)

## Local Development

To run the same checks locally:

```bash
# Install pre-commit hooks
pre-commit install

# Run all pre-commit hooks
pre-commit run --all-files

# Run tests with coverage
pytest tests/ --cov=src/puffinflow --cov-report=html

# Run TruffleHog locally (requires Docker)
docker run --rm -v .:/tmp -w /tmp \
  ghcr.io/trufflesecurity/trufflehog:latest \
  git file:///tmp/ --config=.trufflehog.yml --only-verified

# Build package
python -m build
```

## Best Practices

1. **Commit Messages**: Use conventional commit format
2. **Branch Protection**: Enable branch protection rules for `main`
3. **Secret Management**: Never commit secrets; use GitHub Secrets
4. **Testing**: Maintain high test coverage (>85%)
5. **Documentation**: Update docs with significant changes

## Monitoring

- **GitHub Actions**: Monitor workflow runs in the Actions tab
- **Codecov**: Track coverage trends at codecov.io
- **PyPI**: Monitor package downloads and versions
- **Security**: Review TruffleHog scan results regularly
