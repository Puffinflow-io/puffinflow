# PuffinFlow Makefile

.PHONY: help install test lint format benchmark clean dev-install docs

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package and dependencies"
	@echo "  dev-install  - Install package in development mode with all extras"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  benchmark-save - Run benchmarks and save results"
	@echo "  benchmark-html - Run benchmarks and generate HTML report"
	@echo "  clean        - Clean build artifacts"
	@echo "  docs         - Build documentation"

# Installation
install:
	pip install -e .

dev-install:
	pip install -e .[dev,performance,observability,all]

# Testing
test:
	pytest tests/ -v

test-coverage:
	pytest tests/ --cov=src/puffinflow --cov-report=html --cov-report=term-missing

# Code quality
lint:
	ruff check src/ tests/ benchmarks/ examples/
	mypy src/puffinflow/

format:
	black src/ tests/ benchmarks/ examples/
	ruff check --fix src/ tests/ benchmarks/ examples/

# Benchmarking
benchmark:
	python benchmarks/run_all_benchmarks.py

benchmark-save:
	python benchmarks/run_all_benchmarks.py --save-results --format json

benchmark-html:
	python benchmarks/run_all_benchmarks.py --save-results --format html

benchmark-core:
	python benchmarks/benchmark_core_agent.py

benchmark-resources:
	python benchmarks/benchmark_resource_management.py

benchmark-coordination:
	python benchmarks/benchmark_coordination.py

benchmark-observability:
	python benchmarks/benchmark_observability.py

# Individual benchmark suites
benchmark-quick:
	python benchmarks/benchmark_core_agent.py --iterations 50
	python benchmarks/benchmark_resource_management.py --iterations 100

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs && python -m http.server 8000 --directory build/html

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf benchmark_results/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development workflow
dev-setup: dev-install
	pre-commit install

dev-test: format lint test

dev-benchmark: benchmark-save
	@echo "Benchmark results saved to benchmark_results/"

# CI/CD targets
ci-test:
	pytest tests/ --cov=src/puffinflow --cov-report=xml --cov-fail-under=85

ci-benchmark:
	python benchmarks/run_all_benchmarks.py --save-results --format json
	python benchmarks/run_all_benchmarks.py --save-results --format html

# Release targets
check-release:
	python -m build
	python -m twine check dist/*

build:
	python -m build

upload-test:
	python -m twine upload --repository testpypi dist/*

upload:
	python -m twine upload dist/*
