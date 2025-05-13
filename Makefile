.PHONY: setup install test lint format clean help

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
POETRY := poetry
PYTEST := pytest
BLACK := black
MYPY := mypy

help: ## Display this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@awk -F ':|##' '/^[^\t].+?:.*?##/ { printf "  %-20s %s\n", $$1, $$NF }' $(MAKEFILE_LIST)

setup: ## Install Poetry and project dependencies
	@echo "Installing Poetry..."
	curl -sSL https://install.python-poetry.org | python3 -
	@echo "Installing project dependencies..."
	$(POETRY) install

install: ## Install project dependencies
	$(POETRY) install

test: ## Run tests
	$(POETRY) run $(PYTEST) tests/

test-coverage: ## Run tests with coverage report
	$(POETRY) run $(PYTEST) --cov=src tests/

lint: ## Run type checking
	$(POETRY) run $(MYPY) src/

format: ## Format code using black
	$(POETRY) run $(BLACK) src/ tests/

clean: ## Clean up Python cache files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

demo: ## Run live demo
	$(POETRY) run python scripts/demo_live.py

asl-data: ## Process ASL dataset
	$(MAKE) wipe-data
	$(MAKE) download-data
	$(MAKE) process-data

download-data: ## Download ASL dataset
	$(POETRY) run python scripts/download_asl_digits.py

process-data: ## Process ASL dataset (extract landmarks and split into train/val/test)
	$(POETRY) run python scripts/process_asl_data.py

wipe-data: ## Wipe ASL dataset
	rm -rf data/raw/asl_digits
	rm -rf data/processed/asl_digits

train: ## Train model
	$(POETRY) run python scripts/train.py

evaluate-baseline: ## Evaluate model against MediaPipe baseline
	$(POETRY) run python scripts/evaluate_baseline.py
