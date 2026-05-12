
.PHONY: help install install-dev test lint format build clean clean-cache dry-run train

help:
	@echo "training-module Makefile commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install         Install in editable mode (development)"
	@echo "  make install-dev     Install with development dependencies (lint, test, build tools)"
	@echo "  make build           Build distribution wheel (dist/training_module-*.whl)"
	@echo ""
	@echo "Development:"
	@echo "  make lint            Run ruff linter"
	@echo "  make format          Format code with ruff"
	@echo "  make test            Run pytest test suite"
	@echo ""
	@echo "Training & Debugging:"
	@echo "  make train           Run training CLI with sample args"
	@echo "  make dry-run         Resolve config without training (validates setup)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           Remove Python cache files (__pycache__, *.pyc)"
	@echo "  make clean-cache     Remove tool caches (.pytest_cache, .mypy_cache, .ruff_cache)"
	@echo "  make clean-build     Remove build artifacts (dist/, build/, *.egg-info)"
	@echo "  make clean-all       Clean everything (combined clean targets)"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

build:
	python -m build

test:
	pytest tests/ -v

lint:
	ruff check training/ --fix

format:
	ruff format training/

clean:
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete

clean-cache:
	@find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	@find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -prune -exec rm -rf {} +

clean-build:
	@rm -rf dist/ build/ *.egg-info

clean-all: clean clean-cache clean-build
	@echo "All artifacts cleaned"

train:
	python train.py --task classification --dataset imagenet --experiment baseline

dry-run:
	python train.py --task classification --dataset imagenet --experiment baseline --dry-run