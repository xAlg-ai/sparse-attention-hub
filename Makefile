# Makefile for sparse-attention-hub development

.PHONY: help install install-dev test test-unit test-integration lint format clean docs build upload pre-commit

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package in development mode"
	@echo "  install-dev  - Install package with development dependencies"
	@echo "  test         - Run all tests"
	@echo "  test-unit    - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  lint         - Run all linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  format-check - Check code formatting without making changes"
	@echo "  pre-commit   - Install and run pre-commit hooks"
	@echo "  clean        - Clean build artifacts"
	@echo "  docs         - Build documentation"
	@echo "  build        - Build package"
	@echo "  upload       - Upload package to PyPI"

# Installation targets
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing targets
test:
	python scripts/run_tests.py --type all --verbose

test-unit:
	python scripts/run_tests.py --type unit --verbose

test-integration:
	python scripts/run_tests.py --type integration --verbose

test-coverage:
	pytest --cov=sparse_attention_hub --cov-report=html --cov-report=term

# Linting and formatting targets
lint:
	bash scripts/lint.sh

lint-flake8:
	bash scripts/lint.sh --flake8

lint-mypy:
	bash scripts/lint.sh --mypy

lint-pylint:
	bash scripts/lint.sh --pylint

lint-bandit:
	bash scripts/lint.sh --bandit

format:
	bash scripts/format.sh --no-lint

format-check:
	bash scripts/lint.sh --black
	bash scripts/lint.sh --isort

format-all:
	bash scripts/format.sh --all --no-lint

# Pre-commit targets
pre-commit:
	pre-commit install
	pre-commit run --all-files

pre-commit-update:
	pre-commit autoupdate

# Development workflow
dev-setup: install-dev pre-commit
	@echo "Development environment setup complete!"

dev-check: format-check lint test
	@echo "All development checks passed!"

# Cleaning targets
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf bandit-report.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Documentation targets
docs:
	@echo "Building documentation..."
	@echo "Documentation build not yet implemented"

# Build and upload targets
build: clean
	python -m build

upload: build
	python -m twine upload dist/*

upload-test: build
	python -m twine upload --repository testpypi dist/*

# Quick development commands
quick-test: test-unit

quick-check: format-check lint-flake8

# CI/CD simulation
ci: install-dev format-check lint test
	@echo "CI pipeline simulation completed successfully!"