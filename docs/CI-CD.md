# CI/CD Pipeline Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline for the puffinflow project.

## Overview

The CI/CD pipeline is designed to ensure code quality, security, and automated publishing to PyPI when certain conditions are met. The pipeline consists of multiple workflows that handle different aspects of the development lifecycle.

## Workflows

### 1. Main CI/CD Pipeline (`.github/workflows/ci-cd.yml`)

This is the primary workflow that handles the complete CI/CD process.

#### Triggers
- Push to `main` or `develop` branches
- Pull requests to `main` branch
- Release publications

#### Jobs

##### Security Scan
- **Purpose**: Scan for secrets and security vulnerabilities
- **Tool**: TruffleHog OSS
- **Configuration**: Uses `.trufflehog.yml` for custom settings
- **Features**:
  - Full git history fetch for proper comparison
  - Only verified secrets to reduce false positives
  - Debug output for troubleshooting

##### Lint and Format Check
- **Purpose**: Ensure code quality and consistency
- **Tools**:
  - Ruff (linting and formatting)
  - Black (code formatting)
  - MyPy (type checking)
  - Bandit (security linting)
- **Artifacts**: Bandit security report

##### Test Suite
- **Purpose**: Run comprehensive tests across multiple environments
- **Matrix Strategy**:
  - OS: Ubuntu, Windows, macOS
  - Python: 3.9, 3.10, 3.11, 3.12, 3.13
- **Coverage**: Minimum 85% required
- **Artifacts**: Coverage reports (XML, HTML)
- **Integration**: Codecov for coverage tracking

##### Coverage Analysis
- **Purpose**: Extract and validate test coverage
- **Threshold**: 85% minimum
- **Output**: Coverage percentage and threshold validation
- **Reporting**: GitHub Step Summary with detailed results

##### Build Package
- **Purpose**: Build Python package for distribution
- **Tools**: `build` and `twine`
- **Validation**: Package integrity check
- **Artifacts**: Distribution files (`dist/`)

##### PyPI Publishing
Two separate jobs for different environments:

###### Test PyPI
- **Trigger**: Push to `main` branch + coverage ≥ 85%
- **Environment**: `test-pypi`
- **URL**: https://test.pypi.org/p/puffinflow
- **Purpose**: Staging environment for testing

###### Production PyPI
- **Trigger**: Release publication + coverage ≥ 85%
- **Environment**: `pypi`
- **URL**: https://pypi.org/p/puffinflow
- **Purpose**: Production package distribution

##### Notification
- **Purpose**: Provide summary of pipeline results
- **Features**:
  - Coverage status reporting
  - Publication status updates
  - Clear success/failure indicators

### 2. Security-Focused Workflow (`.github/workflows/security-scan.yml`)

Dedicated security scanning workflow with enhanced TruffleHog configuration.

#### Features
- **Git History Verification**: Ensures sufficient commit history for TruffleHog
- **Fallback Mechanisms**: Multiple scan strategies to handle edge cases
- **Comprehensive Security Tools**:
  - TruffleHog (secret scanning)
  - Bandit (Python security linting)
  - Safety (vulnerability database check)
  - Semgrep (static analysis)
- **Scheduled Scans**: Daily security scans at 2 AM UTC
- **Detailed Reporting**: Security summary with recommendations

## Configuration Files

### TruffleHog Configuration (`.trufflehog.yml`)

Customizes TruffleHog behavior for optimal security scanning:

```yaml
# Key settings
only_verified: true          # Reduce false positives
chunk_size: 10240           # Optimize performance
concurrency: 10             # Parallel processing
detector_timeout: 10s       # Prevent hanging
verify_timeout: 30s         # Allow time for verification
```

#### Detectors Included
- Cloud providers (AWS, Azure, GCP, DigitalOcean)
- Databases (MongoDB, MySQL, PostgreSQL, Redis)
- API services (GitHub, GitLab, Slack, Discord)
- CI/CD platforms (Docker Hub, Heroku, Netlify, Vercel)
- Payment services (Stripe, PayPal)
- Email services (SendGrid, Mailgun, Postmark)

#### Exclusions
- Build artifacts and cache directories
- Generated files (version.py)
- Test and documentation files
- Common false positive patterns

## PyPI Publishing Requirements

For a package to be published to PyPI, the following conditions must be met:

### Mandatory Requirements
1. **Test Coverage**: Must be ≥ 85%
2. **Security Scan**: TruffleHog must pass (no verified secrets found)
3. **Code Quality**: All linting and formatting checks must pass
4. **Tests**: All unit tests must pass across supported Python versions
5. **Build**: Package must build successfully and pass integrity checks

### Trigger Conditions

#### Test PyPI
- Push to `main` branch
- All mandatory requirements met

#### Production PyPI
- GitHub release published
- All mandatory requirements met

## Local Development

### Running Security Scans Locally

Use the provided script to run the same security checks locally:

```bash
# Make script executable (Unix/Linux/macOS)
chmod +x scripts/run-security-scan.py

# Run security checks
python scripts/run-security-scan.py
```

The script will:
1. Check git history for TruffleHog compatibility
2. Run TruffleHog secret scanning
3. Execute Bandit security linting
4. Perform Safety vulnerability checks
5. Run Semgrep static analysis
6. Validate test coverage
7. Provide a comprehensive summary

### Prerequisites

Install required security tools:

```bash
# Python tools (included in dev dependencies)
pip install -e ".[dev,security]"

# TruffleHog (external tool)
curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh | sh -s -- -b /usr/local/bin

# Semgrep (optional, for enhanced analysis)
pip install semgrep
```

## Environment Setup

### GitHub Secrets

The following secrets should be configured in the GitHub repository:

#### For PyPI Publishing (if not using trusted publishing)
- `PYPI_API_TOKEN`: PyPI API token for production publishing
- `TEST_PYPI_API_TOKEN`: Test PyPI API token for staging

#### For Enhanced Security (optional)
- `CODECOV_TOKEN`: Codecov integration token
- `SLACK_WEBHOOK`: Slack notifications webhook

### GitHub Environments

Configure the following environments in GitHub:

#### `test-pypi`
- **Protection Rules**: Require review for deployments
- **Secrets**: Test PyPI credentials
- **URL**: https://test.pypi.org/p/puffinflow

#### `pypi`
- **Protection Rules**: Require review for production deployments
- **Secrets**: Production PyPI credentials
- **URL**: https://pypi.org/p/puffinflow

## Trusted Publishing (Recommended)

For enhanced security, use PyPI's trusted publishing feature:

1. Configure trusted publisher in PyPI project settings
2. Use `id-token: write` permission in workflows
3. Remove API token secrets (handled automatically)

## Monitoring and Alerts

### Coverage Tracking
- **Codecov Integration**: Automatic coverage reporting
- **Threshold Enforcement**: 85% minimum coverage required
- **Trend Analysis**: Coverage history and trends

### Security Monitoring
- **Daily Scans**: Automated security scans via scheduled workflow
- **Vulnerability Alerts**: GitHub Dependabot integration
- **Security Reports**: Detailed artifacts for each scan

### Build Status
- **GitHub Status Checks**: Required checks for PR merging
- **Deployment Status**: Environment-specific deployment tracking
- **Artifact Management**: Automatic cleanup of old artifacts

## Troubleshooting

### Common Issues

#### TruffleHog "BASE and HEAD commits are the same"
- **Cause**: Insufficient git history or single commit
- **Solution**: Workflow automatically creates dummy commit if needed
- **Prevention**: Ensure proper git history in repository

#### Coverage Below Threshold
- **Cause**: Test coverage < 85%
- **Solution**: Add more tests or remove untested code
- **Check**: Run `pytest --cov=src/puffinflow --cov-report=term-missing`

#### Security Scan Failures
- **Cause**: Secrets detected or security vulnerabilities
- **Solution**: Review and remediate flagged issues
- **Tools**: Use local security scan script for debugging

#### Build Failures
- **Cause**: Package configuration or dependency issues
- **Solution**: Test locally with `python -m build`
- **Validation**: Run `twine check dist/*`

### Getting Help

1. **Check Workflow Logs**: Detailed logs available in GitHub Actions
2. **Run Local Checks**: Use `scripts/run-security-scan.py`
3. **Review Configuration**: Verify `.trufflehog.yml` and `pyproject.toml`
4. **Test Coverage**: Generate detailed coverage report locally

## Best Practices

### Development Workflow
1. Create feature branch from `develop`
2. Run local security checks before committing
3. Ensure tests pass and coverage is adequate
4. Create pull request to `main`
5. Address any CI/CD feedback
6. Merge after approval and successful checks

### Security Practices
1. Never commit real secrets or credentials
2. Use environment variables for sensitive data
3. Regularly update dependencies
4. Review security scan results promptly
5. Follow principle of least privilege

### Release Management
1. Use semantic versioning
2. Update CHANGELOG.md
3. Create GitHub release with proper tags
4. Monitor PyPI publication success
5. Verify package installation and functionality

This CI/CD pipeline ensures high code quality, security, and reliable package distribution while maintaining developer productivity and project integrity.