# Pull Request

## Summary

**What does this PR do?**
<!-- Provide a brief description of the changes in this PR -->

**Related Issue(s)**
<!-- Link to related issues using "Fixes #123" or "Closes #123" -->
- Fixes #
- Related to #

## Type of Change

**What type of change is this?**
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring (no functional changes)
- [ ] Test improvements
- [ ] CI/CD improvements
- [ ] Other (please describe)

## Changes Made

**Detailed description of changes:**
<!-- Provide a more detailed description of what you changed and why -->

**Components affected:**
- [ ] Core agents (`src/puffinflow/core/agent/`)
- [ ] Coordination system (`src/puffinflow/core/coordination/`)
- [ ] Resource management (`src/puffinflow/core/resources/`)
- [ ] Observability (`src/puffinflow/core/observability/`)
- [ ] Reliability features (`src/puffinflow/core/reliability/`)
- [ ] CLI tools
- [ ] Documentation
- [ ] Tests
- [ ] CI/CD
- [ ] Other (please specify)

**Key files changed:**
<!-- List the most important files that were changed -->
- `path/to/file.py` - Description of changes
- `path/to/another/file.py` - Description of changes

## Testing

**How has this been tested?**
- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing
- [ ] Performance testing
- [ ] No testing required

**Test coverage:**
- [ ] All new code is covered by tests
- [ ] Existing tests still pass
- [ ] Test coverage percentage maintained or improved
- [ ] No tests needed for this change

**Manual testing steps:**
<!-- If manual testing was performed, describe the steps -->
1. Step 1
2. Step 2
3. Step 3

**Test results:**
```
# Paste relevant test output here if applicable
```

## Code Quality

**Code quality checks:**
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Code is properly documented
- [ ] No new linting errors introduced
- [ ] Type hints added where appropriate

**Performance considerations:**
- [ ] No performance impact
- [ ] Performance improved
- [ ] Performance impact acceptable
- [ ] Performance impact needs discussion

## Documentation

**Documentation updates:**
- [ ] Code comments added/updated
- [ ] API documentation updated
- [ ] README updated
- [ ] Examples updated/added
- [ ] Migration guide created (for breaking changes)
- [ ] No documentation changes needed

**Breaking changes documentation:**
<!-- If this introduces breaking changes, document them here -->

## Backwards Compatibility

**Is this change backwards compatible?**
- [ ] Yes, fully backwards compatible
- [ ] No, but provides migration path
- [ ] No, breaking change (requires major version bump)

**Migration required:**
<!-- If migration is required, provide instructions -->

## Security

**Security considerations:**
- [ ] No security implications
- [ ] Security improvement
- [ ] Potential security impact (please describe)
- [ ] Security review requested

**Security checklist:**
- [ ] No hardcoded secrets or credentials
- [ ] Input validation implemented where needed
- [ ] No new attack vectors introduced
- [ ] Dependencies are secure and up-to-date

## Deployment

**Deployment considerations:**
- [ ] No special deployment requirements
- [ ] Database migrations required
- [ ] Configuration changes required
- [ ] Environment variable changes required
- [ ] Special deployment instructions (see below)

**Deployment instructions:**
<!-- If special deployment steps are needed, document them here -->

## Checklist

**Before submitting:**
- [ ] I have read the contributing guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

**Code quality:**
- [ ] Code follows the established style guide
- [ ] All tests pass
- [ ] Linting passes
- [ ] Type checking passes
- [ ] No merge conflicts

## Screenshots/Examples

**Visual changes:**
<!-- If your changes affect the UI or output, include screenshots -->

**Code examples:**
```python
# If your changes affect the API, provide usage examples
from puffinflow import SomeFeature

# Example of how to use the new feature
feature = SomeFeature()
result = feature.do_something()
```

## Additional Notes

**Reviewer notes:**
<!-- Any specific areas you'd like reviewers to focus on -->

**Future work:**
<!-- Any follow-up work that should be done -->

**Dependencies:**
<!-- Any new dependencies added or version changes -->

**Known issues:**
<!-- Any known issues or limitations with this change -->

## Review Checklist (for reviewers)

- [ ] Code is well-structured and readable
- [ ] Logic is sound and efficient
- [ ] Error handling is appropriate
- [ ] Tests are comprehensive and meaningful
- [ ] Documentation is clear and complete
- [ ] No security vulnerabilities introduced
- [ ] Performance impact is acceptable
- [ ] Breaking changes are properly documented

---

**Thank you for contributing to PuffinFlow!** 🎉

Your contribution helps make PuffinFlow better for everyone. We appreciate the time and effort you've put into this pull request.
