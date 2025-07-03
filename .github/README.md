# GitHub Workflows and Templates

This directory contains comprehensive GitHub Actions workflows and issue/PR templates for the PuffinFlow project.

## 📁 Directory Structure

```
.github/
├── workflows/
│   ├── ci.yml           # Continuous Integration
│   ├── release.yml      # Automated releases
│   ├── docs.yml         # Documentation building
│   └── security.yml     # Security scanning
├── ISSUE_TEMPLATE/
│   ├── bug_report.md    # Bug report template
│   ├── feature_request.md # Feature request template
│   └── question.md      # Question/support template
├── PULL_REQUEST_TEMPLATE.md # Pull request template
├── dependabot.yml       # Dependency updates
└── README.md           # This file
```

## 🔄 Workflows

### CI Workflow (`ci.yml`)
**Triggers:** Push to main/develop, PRs to main/develop, manual dispatch

**Jobs:**
- **Lint**: Code formatting and style checks (ruff, black, mypy)
- **Test**: Multi-platform testing (Ubuntu, Windows, macOS) across Python 3.9-3.13
- **Integration Tests**: Integration test suite
- **Benchmark**: Performance benchmarks (main branch only)
- **Build**: Package building and validation
- **Security**: Basic security scanning
- **All Checks**: Final validation gate

**Features:**
- Matrix testing across OS and Python versions
- Code coverage reporting with Codecov
- Artifact uploads for build results
- Concurrency control to cancel outdated runs

### Release Workflow (`release.yml`)
**Triggers:** Git tags starting with 'v', manual dispatch

**Jobs:**
- **Validate Tag**: Version validation and prerelease detection
- **Test**: Full test suite before release
- **Build**: Distribution building
- **Create Release**: GitHub release creation with changelog
- **Publish PyPI**: Production PyPI publishing
- **Publish Test PyPI**: Test PyPI for prereleases
- **Notify**: Post-release notifications

**Features:**
- Automatic changelog generation
- Prerelease detection and handling
- Dual PyPI publishing (test and production)
- Asset uploads to GitHub releases

### Documentation Workflow (`docs.yml`)
**Triggers:** Changes to docs/, src/, README.md, or workflow file

**Jobs:**
- **Build Docs**: Sphinx documentation building with auto-generated API docs
- **Deploy Docs**: GitHub Pages deployment (main branch only)
- **Check Coverage**: Documentation coverage analysis

**Features:**
- Automatic Sphinx configuration creation
- API documentation generation
- Link checking
- Documentation coverage badges
- GitHub Pages deployment

### Security Workflow (`security.yml`)
**Triggers:** Push, PRs, daily schedule (2 AM UTC), manual dispatch

**Jobs:**
- **Dependency Scan**: Safety and pip-audit vulnerability scanning
- **Code Scan**: Bandit and Semgrep static analysis
- **CodeQL Analysis**: GitHub's semantic code analysis
- **Secrets Scan**: TruffleHog secret detection
- **License Scan**: License compliance checking
- **Security Report**: Consolidated reporting
- **Notify**: Critical finding alerts

**Features:**
- Multiple security scanning tools
- Scheduled daily scans
- License compliance verification
- Consolidated security reporting
- PR comments with security summaries

## 📝 Issue Templates

### Bug Report (`bug_report.md`)
Comprehensive bug reporting template with:
- Detailed reproduction steps
- Environment information
- Error logs and stack traces
- Impact assessment
- Workaround documentation

### Feature Request (`feature_request.md`)
Structured feature request template with:
- Use case description
- API design sketches
- Implementation considerations
- Priority and impact assessment
- Community contribution options

### Question (`question.md`)
Support and question template with:
- Categorized question types
- Context and research documentation
- Environment details
- Community guidelines compliance

## 🔀 Pull Request Template

Comprehensive PR template covering:
- Change summary and type classification
- Testing and quality assurance
- Documentation updates
- Security considerations
- Deployment requirements
- Review checklists

## 🤖 Dependabot Configuration

Automated dependency management with:
- **Python dependencies**: Weekly updates with intelligent grouping
- **GitHub Actions**: Weekly action updates
- **Docker**: Weekly base image updates (when applicable)

**Features:**
- Grouped updates by category (core, dev tools, observability, etc.)
- Security update prioritization
- Version strategy configuration
- Maintainer assignment to @m-ahmed-elbeskeri

## 🚀 Getting Started

### Setting Up Secrets

For full functionality, configure these repository secrets:

1. **CODECOV_TOKEN**: For code coverage reporting
2. **PYPI_API_TOKEN**: For PyPI publishing
3. **TEST_PYPI_API_TOKEN**: For Test PyPI publishing

### Enabling GitHub Pages

1. Go to repository Settings → Pages
2. Set source to "GitHub Actions"
3. Documentation will be automatically deployed on main branch changes

### Branch Protection

Recommended branch protection rules for `main`:

1. Require pull request reviews
2. Require status checks to pass:
   - `All Checks Passed` (from CI workflow)
   - `Build Documentation` (from docs workflow)
3. Require branches to be up to date
4. Restrict pushes to matching branches

### Release Process

1. **Create a release tag**: `git tag v1.0.0 && git push origin v1.0.0`
2. **Automatic process**: Release workflow handles building, testing, and publishing
3. **Manual release**: Use workflow dispatch with version input

## 🔧 Customization

### Modifying Workflows

- **Python versions**: Update matrix in `ci.yml`
- **Test commands**: Modify test steps in workflows
- **Security tools**: Add/remove tools in `security.yml`
- **Documentation**: Customize Sphinx configuration in `docs.yml`

### Updating Templates

- **Issue templates**: Modify files in `ISSUE_TEMPLATE/`
- **PR template**: Edit `PULL_REQUEST_TEMPLATE.md`
- **Labels**: Update labels in template frontmatter

### Dependabot Configuration

- **Update frequency**: Modify `schedule.interval` in `dependabot.yml`
- **Dependency groups**: Add/remove groups in configuration
- **Ignore rules**: Update ignore patterns for specific dependencies

## 📊 Monitoring

### Workflow Status

Monitor workflow status in the Actions tab:
- Green checkmarks indicate successful runs
- Red X marks indicate failures requiring attention
- Yellow dots indicate in-progress runs

### Security Alerts

- Check Security tab for vulnerability alerts
- Review Dependabot PRs for dependency updates
- Monitor security workflow artifacts for detailed reports

### Performance

- Benchmark results are stored as workflow artifacts
- Performance regressions trigger alerts on main branch
- Historical performance data available in workflow runs

## 🆘 Troubleshooting

### Common Issues

1. **Test failures**: Check test logs in CI workflow
2. **Build failures**: Review build artifacts and logs
3. **Security alerts**: Address findings in security workflow reports
4. **Documentation build failures**: Check Sphinx configuration and dependencies

### Getting Help

- Create an issue using the question template
- Check workflow logs for detailed error information
- Review GitHub Actions documentation for workflow syntax

## 🤝 Contributing

When contributing to workflows:

1. Test changes in a fork first
2. Update this README for significant changes
3. Follow the established patterns and conventions
4. Ensure backwards compatibility when possible

---

**Maintainer**: @m-ahmed-elbeskeri
**Last Updated**: January 2025