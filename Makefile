# BookRAG developer shortcuts.
# Windows:  make (via `make` from Git Bash / Chocolatey) or run the commands directly.

PY ?= py -3.11
PYTHON ?= python

.DEFAULT_GOAL := help

.PHONY: help install install-dev test test-cov lint format typecheck verify verify-model verify-e2e sample-pdf demo diagrams export-diagrams clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PY) -m pip install -r requirements.txt

install-dev: ## Install runtime + development dependencies
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e .

test: ## Run the test suite
	$(PY) -m pytest -q

test-cov: ## Run the test suite with a coverage report
	$(PY) -m pytest --cov=bookrag --cov-report=term-missing

lint: ## Check formatting and lint rules
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

format: ## Auto-format the code
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check src tests scripts --fix

typecheck: ## Run the static type checker
	$(PY) -m mypy src

verify: ## Verify against the real ChromaDB and PyMuPDF
	$(PY) scripts/verify_real_backends.py

verify-model: ## Verify against the real Jina CLIP v2 model (downloads ~1.8 GB)
	$(PY) scripts/verify_real_model.py

sample-pdf: ## Build the four-page sample book used by verify-e2e
	$(PY) scripts/make_sample_pdf.py

verify-e2e: ## Run the installed `bookrag` CLI end to end against the real stack
	$(PY) scripts/verify_e2e.py

demo: ## Run the built-in smoke test
	$(PY) -m bookrag demo

diagrams: ## Validate the draw.io diagram source
	$(PY) scripts/validate_drawio.py docs/bookrag-architecture.drawio

export-diagrams: ## Regenerate the PNG / SVG / PDF diagrams (needs draw.io)
	$(PY) scripts/export_drawio.py docs/bookrag-architecture.drawio --prefix bookrag

clean: ## Remove build artefacts and caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
