.PHONY: help install install-dev test lint format docs clean build upload

help:
	@echo "Available commands:"
	@echo "  install     Install package"
	@echo "  install-dev Install package with development dependencies"
	@echo "  test        Run tests"
	@echo "  lint        Run linting"
	@echo "  format      Format code"
	@echo "  docs        Build documentation"
	@echo "  clean       Clean build artifacts"
	@echo "  build       Build package"
	@echo "  upload      Upload to PyPI"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

test-cov:
	pytest --cov=puffinflow --cov-report=html --cov-report=term-missing

lint:
	flake8 src tests
	mypy src/puffinflow
	black --check src tests
	isort --check-only src tests

format:
	black src tests
	isort src tests

docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	python -m twine upload dist/*

upload-test: build
	python -m twine upload --repository testpypi dist/*