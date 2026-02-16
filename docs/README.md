# PuffinFlow Documentation

This directory contains the complete Sphinx documentation for PuffinFlow, a Python workflow orchestration framework.

## Documentation Structure

```
docs/
├── source/                    # Sphinx source files
│   ├── conf.py               # Sphinx configuration
│   ├── index.rst             # Main documentation index
│   ├── changelog.rst         # Version history
│   ├── contributing.rst      # Contribution guidelines
│   ├── security.rst          # Security policy
│   ├── api/                  # API reference documentation
│   │   ├── index.rst         # API overview
│   │   ├── agent.rst         # Agent system
│   │   ├── coordination.rst  # Multi-agent coordination
│   │   ├── resources.rst     # Resource management
│   │   ├── observability.rst # Monitoring and tracing
│   │   └── reliability.rst   # Fault tolerance
│   ├── guides/               # User guides
│   │   ├── quickstart.rst    # Getting started
│   │   ├── advanced.rst      # Advanced usage
│   │   ├── examples.rst      # Real-world examples
│   │   └── migration.rst     # Migration guide
│   ├── _static/              # Static assets
│   │   └── custom.css        # Custom styling
│   └── _templates/           # Custom templates
├── Makefile                  # Build automation
├── requirements.txt          # Documentation dependencies
└── README.md                 # This file
```

## Building the Documentation

### Prerequisites

1. **Install documentation dependencies:**
   ```bash
   pip install -r docs/requirements.txt
   ```

2. **Install PuffinFlow in development mode:**
   ```bash
   pip install -e .
   ```

### Build Commands

**Build HTML documentation:**
```bash
cd docs
make html
```

**Build and serve with live reload:**
```bash
cd docs
make livehtml
```

**Clean build artifacts:**
```bash
cd docs
make clean
```

**Check for broken links:**
```bash
cd docs
make linkcheck
```

**Run doctests:**
```bash
cd docs
make doctest
```

**Build all formats:**
```bash
cd docs
make all
```

### Development Workflow

1. **Start live server for development:**
   ```bash
   cd docs
   make livehtml
   ```
   This will start a development server at `http://localhost:8000` with automatic rebuilding.

2. **Make changes to `.rst` files** in the `source/` directory

3. **View changes** automatically in your browser

4. **Run quality checks:**
   ```bash
   cd docs
   make check
   ```

## Documentation Features

### Sphinx Extensions

- **sphinx.ext.autodoc**: Automatic API documentation from docstrings
- **sphinx.ext.viewcode**: Source code links
- **sphinx.ext.napoleon**: Google/NumPy style docstrings
- **sphinx.ext.intersphinx**: Cross-project references
- **myst_parser**: Markdown support
- **sphinx_copybutton**: Copy code button
- **sphinxcontrib.mermaid**: Diagram support

### Theme and Styling

- **Theme**: Read the Docs theme with custom styling
- **Custom CSS**: Brand colors and enhanced styling in `_static/custom.css`
- **Responsive**: Mobile-friendly design
- **Dark/Light**: Theme switching support

### Content Types

- **API Reference**: Auto-generated from source code
- **User Guides**: Step-by-step tutorials and guides
- **Examples**: Real-world usage examples
- **Changelog**: Version history and release notes
- **Contributing**: Development and contribution guidelines
- **Security**: Security policy and best practices

## Writing Documentation

### reStructuredText Basics

**Headers:**
```rst
Main Title
==========

Section
-------

Subsection
~~~~~~~~~~
```

**Code blocks:**
```rst
.. code-block:: python

   from puffinflow import Agent, Context

   class MyAgent(Agent):
       async def run(self, ctx: Context) -> None:
           print("Hello, PuffinFlow!")
```

**Links:**
```rst
:doc:`quickstart`
:ref:`api-reference`
:class:`puffinflow.Agent`
:meth:`puffinflow.Agent.run`
```

**Admonitions:**
```rst
.. note::
   This is a note.

.. warning::
   This is a warning.

.. tip::
   This is a tip.
```

### API Documentation

API documentation is automatically generated from docstrings using Sphinx autodoc. Use Google-style docstrings:

```python
async def process_data(self, ctx: Context, batch_size: int = 100) -> None:
    """Process data in batches.

    Args:
        ctx: The execution context containing input data.
        batch_size: Number of items to process in each batch.

    Raises:
        ValueError: If batch_size is less than 1.
        ProcessingError: If data processing fails.

    Example:
        >>> agent = DataProcessor()
        >>> ctx = Context({'data': [1, 2, 3, 4, 5]})
        >>> await agent.process_data(ctx, batch_size=2)
    """
```

### Adding New Documentation

1. **Create new `.rst` file** in appropriate directory
2. **Add to toctree** in parent `index.rst` file
3. **Follow existing patterns** for consistency
4. **Test locally** before committing

## Deployment

### GitHub Pages

Documentation can be automatically deployed to GitHub Pages using GitHub Actions:

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r docs/requirements.txt
        pip install -e .
    - name: Build documentation
      run: |
        cd docs
        make html
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      if: github.ref == 'refs/heads/main'
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/build/html
```

### Read the Docs

1. Connect your repository to [Read the Docs](https://readthedocs.org/)
2. Configure build settings:
   - **Python version**: 3.9+
   - **Requirements file**: `docs/requirements.txt`
   - **Documentation type**: Sphinx
3. Enable automatic builds on commits

## Troubleshooting

### Common Issues

**Build fails with import errors:**
- Ensure PuffinFlow is installed: `pip install -e .`
- Check Python path in `conf.py`

**Missing references:**
- Run `make linkcheck` to find broken links
- Update intersphinx mappings in `conf.py`

**Styling issues:**
- Clear browser cache
- Rebuild with `make clean html`
- Check `_static/custom.css`

**Autodoc not finding modules:**
- Verify module imports work
- Check `sys.path` configuration in `conf.py`
- Ensure all `__init__.py` files exist

### Getting Help

- **Sphinx Documentation**: https://www.sphinx-doc.org/
- **reStructuredText Guide**: https://docutils.sourceforge.io/rst.html
- **Read the Docs**: https://docs.readthedocs.io/
- **PuffinFlow Issues**: https://github.com/yourusername/puffinflow/issues

## Contributing to Documentation

We welcome contributions to improve the documentation! Please see [`contributing.rst`](source/contributing.rst) for detailed guidelines.

### Quick Contribution Steps

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Test locally**: `make html`
5. **Submit a pull request**

Thank you for helping make PuffinFlow documentation better!
