# Makefile for Open-Vocabulary Semantic Segmentation Pipeline

.PHONY: help install test clean docker-build docker-run docker-clean format lint

# Default target
help:
	@echo "Open-Vocabulary Semantic Segmentation Pipeline"
	@echo ""
	@echo "Available targets:"
	@echo "  install        - Install dependencies"
	@echo "  test           - Run tests"
	@echo "  format         - Format code with black"
	@echo "  lint           - Lint code with flake8"
	@echo "  clean          - Clean generated files"
	@echo ""
	@echo "Docker targets:"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
	@echo "  docker-shell   - Start interactive Docker shell"
	@echo "  docker-clean   - Clean Docker resources"
	@echo ""
	@echo "Quick start:"
	@echo "  make install   - Install dependencies"
	@echo "  make docker-build - Build Docker image"

# Installation
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install git+https://github.com/facebookresearch/segment-anything-2.git
	@echo "✓ Installation complete"

install-dev: install
	pip install pytest black flake8 mypy
	@echo "✓ Development dependencies installed"

# Testing
test:
	pytest tests/ -v

test-quick:
	pytest tests/ -v -k "not slow"

# Code quality
format:
	black --line-length 100 models/ *.py examples/*.py
	@echo "✓ Code formatted"

lint:
	flake8 models/ *.py examples/*.py --max-line-length 100
	@echo "✓ Linting complete"

typecheck:
	mypy models/ pipeline.py config.py utils.py
	@echo "✓ Type checking complete"

# Cleaning
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/
	@echo "✓ Cleaned generated files"

clean-output:
	rm -rf output/*
	@echo "✓ Cleaned output directory"

# Docker targets
docker-build:
	docker build -t openvocab-segmentation:latest .
	@echo "✓ Docker image built"

docker-run:
	./docker-run.sh interactive

docker-shell: docker-run

docker-clean:
	docker container prune -f
	docker image prune -f
	@echo "✓ Docker resources cleaned"

docker-clean-all: docker-clean
	docker volume rm openvocab-model-cache || true
	@echo "✓ All Docker resources and cache cleaned"

# Docker Compose targets
compose-up:
	docker-compose up -d

compose-down:
	docker-compose down

compose-logs:
	docker-compose logs -f

# Development
dev-setup: install-dev
	pre-commit install || true
	@echo "✓ Development environment ready"

# Examples
example-segment:
	@echo "Running segmentation example..."
	python main.py --image examples/test.jpg --prompt "car" --mode segment || \
	echo "Note: Needs example image at examples/test.jpg"

example-docker:
	@echo "Running Docker example..."
	./docker-run.sh segment input/test.jpg "car" || \
	echo "Note: Needs example image at input/test.jpg"

# Documentation
docs:
	@echo "Documentation files:"
	@echo "  - README.md        : Main documentation"
	@echo "  - QUICK_START.md   : Quick start guide"
	@echo "  - DOCKER.md        : Docker deployment guide"
	@echo "  - requirements.txt : Dependencies"

# CI/CD helpers
ci-install:
	pip install -r requirements.txt
	pip install pytest black flake8

ci-test: ci-install test

ci-lint: ci-install lint

# Version info
version:
	@echo "Python: $(shell python --version)"
	@echo "Docker: $(shell docker --version)"
	@echo "CUDA: $(shell nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'Not available')"
